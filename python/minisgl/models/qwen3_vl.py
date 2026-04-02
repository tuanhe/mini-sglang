from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
    get_rope,
    silu_and_mul,
)
from minisgl.layers.rotary import _make_rope_scaling_hashable
from minisgl.utils import div_even, nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3MLP

if TYPE_CHECKING:
    from .config import ModelConfig, VisionConfig


# ======================== Vision Encoder Components ========================


class VisionLinear(BaseOP):
    """Simple linear layer with bias for vision encoder (not TP-sharded)."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool = True):
        self.weight = torch.empty(out_features, in_features)
        self.bias = torch.empty(out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class VisionLayerNorm(BaseOP):
    """LayerNorm for vision encoder."""

    def __init__(self, size: int, eps: float = 1e-6):
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.weight.shape[0],), self.weight, self.bias, self.eps)


class VisionConv3d(BaseOP):
    """Conv3d for patch embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
    ):
        t, h, w = kernel_size
        self.weight = torch.empty(out_channels, in_channels, t, h, w)
        self.bias = torch.empty(out_channels)
        self._kernel_size = kernel_size
        self._in_channels = in_channels
        self._out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            stride=self._kernel_size,
        )


class VisionEmbedding(BaseOP):
    """Embedding for position encoding in vision encoder."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = torch.empty(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)


class Qwen3VLVisionRotaryEmbedding:
    """Rotary position embedding for vision encoder attention."""

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta
        self._inv_freq: torch.Tensor | None = None

    def forward(self, seqlen: int, device: torch.device) -> torch.Tensor:
        if self._inv_freq is None or self._inv_freq.device != device:
            self._inv_freq = 1.0 / (
                self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device) / self.dim)
            )
        seq = torch.arange(seqlen, device=device, dtype=self._inv_freq.dtype)
        freqs = torch.outer(seq, self._inv_freq)
        return freqs


class Qwen3VLVisionMLP(BaseOP):
    def __init__(self, config: VisionConfig):
        self.linear_fc1 = VisionLinear(config.hidden_size, config.intermediate_size)
        self.linear_fc2 = VisionLinear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x), approximate="tanh"))


class Qwen3VLVisionAttention(BaseOP):
    def __init__(self, config: VisionConfig):
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = VisionLinear(self.dim, self.dim * 3)
        self.proj = VisionLinear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv.forward(hidden_states)
        qkv = qkv.reshape(seq_length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
        # q, k, v: (seq_length, num_heads, head_dim)

        # Apply rotary embedding
        q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Process each sequence segment separately (full attention, non-causal)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits = torch.split(q, lengths, dim=0)
        k_splits = torch.split(k, lengths, dim=0)
        v_splits = torch.split(v, lengths, dim=0)

        outputs = []
        for qs, ks, vs in zip(q_splits, k_splits, v_splits):
            # (seq, heads, dim) -> (1, heads, seq, dim)
            qs = qs.transpose(0, 1).unsqueeze(0)
            ks = ks.transpose(0, 1).unsqueeze(0)
            vs = vs.transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(qs, ks, vs, is_causal=False)
            out = out.squeeze(0).transpose(0, 1)  # (seq, heads, dim)
            outputs.append(out)

        attn_output = torch.cat(outputs, dim=0)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.proj.forward(attn_output)


class Qwen3VLVisionBlock(BaseOP):
    def __init__(self, config: VisionConfig):
        self.norm1 = VisionLayerNorm(config.hidden_size)
        self.norm2 = VisionLayerNorm(config.hidden_size)
        self.attn = Qwen3VLVisionAttention(config)
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn.forward(
            self.norm1.forward(hidden_states),
            cu_seqlens=cu_seqlens,
            cos=cos,
            sin=sin,
        )
        hidden_states = hidden_states + self.mlp.forward(self.norm2.forward(hidden_states))
        return hidden_states


class Qwen3VLVisionPatchEmbed(BaseOP):
    def __init__(self, config: VisionConfig):
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = VisionConv3d(self.in_channels, self.embed_dim, kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj.forward(hidden_states.to(dtype=target_dtype))
        return hidden_states.view(-1, self.embed_dim)


class Qwen3VLVisionPatchMerger(BaseOP):
    def __init__(self, config: VisionConfig, use_postshuffle_norm: bool = False):
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_size = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = VisionLayerNorm(norm_size)
        self.linear_fc1 = VisionLinear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = VisionLinear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm.forward(x.view(-1, self.hidden_size)).view(-1, self.hidden_size)
        else:
            x = self.norm.forward(x).view(-1, self.hidden_size)
        x = self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x)))
        return x


class Qwen3VLVisionModel(BaseOP):
    def __init__(self, config: VisionConfig):
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = VisionEmbedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = OPList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.deepstack_merger_list = OPList(
            [
                Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

    def _rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        device = grid_thw.device
        freq_table = self.rotary_pos_emb.forward(max_hw, device)

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        return embeddings.flatten(1)

    def _fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(h.item()))
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(w.item()))

            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_floor + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=dtype, device=device)
        pos_embeds = self.pos_embed.forward(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([int(h * w) for h, w in zip(grid_hs, grid_ws)])

        merge_size = self.spatial_merge_size
        result = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            t, h, w = int(t.item()), int(h.item()), int(w.item())
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            result.append(pos_embed)
        return torch.cat(result)

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden_states = self.patch_embed.forward(hidden_states)
        pos_embeds = self._fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self._rot_pos_emb(grid_thw)
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos, sin = emb.cos(), emb.sin()

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists: List[torch.Tensor] = []
        for layer_num, blk in enumerate(self.blocks.op_list):
            hidden_states = blk.forward(hidden_states, cu_seqlens=cu_seqlens, cos=cos, sin=sin)
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list.op_list[idx].forward(hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger.forward(hidden_states)
        return hidden_states, deepstack_feature_lists


# ======================== M-RoPE ========================


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to vision attention Q/K."""
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()  # (seq, 1, dim)
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class MRoPE:
    """Multimodal Rotary Position Embedding for Qwen3-VL.

    Applies interleaved 3D (temporal, height, width) rotary embeddings
    to query and key tensors based on mrope_section configuration.
    """

    def __init__(
        self,
        head_dim: int,
        base: float,
        mrope_section: Tuple[int, ...],
        mrope_interleaved: bool = True,
    ):
        self.head_dim = head_dim
        self.half_dim = head_dim // 2
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        self._head_dim = head_dim
        self._base = base
        self._inv_freq: torch.Tensor | None = None

    def forward(
        self,
        mrope_position_ids: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply M-RoPE to q and k.

        Args:
            mrope_position_ids: (3, total_tokens) - positions for T, H, W dims
            q: (total_tokens, qo_dim)
            k: (total_tokens, kv_dim)
        """
        device = q.device
        if self._inv_freq is None or self._inv_freq.device != device:
            self._inv_freq = 1.0 / (
                self._base ** (torch.arange(0, self._head_dim, 2, dtype=torch.float, device=device) / self._head_dim)
            )
        inv_freq = self._inv_freq

        # Compute freqs for each of 3 dims: (3, total_tokens, half_dim)
        freqs_3d = torch.einsum("dt,f->dtf", mrope_position_ids.float(), inv_freq)

        # Interleave
        freqs = freqs_3d[0].clone()
        if self.mrope_interleaved:
            for dim_idx in range(1, 3):
                length = self.mrope_section[dim_idx] * 3
                freqs[:, dim_idx:length:3] = freqs_3d[dim_idx, :, dim_idx:length:3]

        # cos/sin: (total_tokens, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(1)  # (total_tokens, 1, head_dim)
        sin = emb.sin().unsqueeze(1)

        # Apply to q: reshape to (total_tokens, num_heads, head_dim)
        orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
        q = q.float().view(-1, num_qo_heads, self.head_dim)
        k = k.float().view(-1, num_kv_heads, self.head_dim)

        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

        q = q.to(orig_q_dtype).view(-1, num_qo_heads * self.head_dim)
        k = k.to(orig_k_dtype).view(-1, num_kv_heads * self.head_dim)
        return q, k


# ======================== VL Attention with M-RoPE ========================


class Qwen3VLAttn(BaseOP):
    """Attention layer with M-RoPE support for Qwen3-VL language model."""

    def __init__(self, config: ModelConfig, layer_id: int, has_qk_norm: bool = True):
        head_dim = config.head_dim
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(config.num_qo_heads, tp_size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim

        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=False,
        )
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=_make_rope_scaling_hashable(config.rotary_config.scaling)
            if config.rotary_config.scaling
            else None,
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )

        # M-RoPE
        self.mrope = MRoPE(
            head_dim=head_dim,
            base=config.rotary_config.base,
            mrope_section=config.mrope_section,
            mrope_interleaved=config.mrope_interleaved,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        qkv = self.qkv_proj.forward(x)
        del x
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)

        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))

        mrope_ids = ctx.batch.mrope_position_ids
        if mrope_ids is not None:
            q, k = self.mrope.forward(mrope_ids, q, k, self.num_qo_heads, self.num_kv_heads)
        else:
            # During decode after VL prefill, add rope_delta to positions for correct RoPE
            positions = ctx.batch.positions
            if not ctx.batch.is_prefill:
                # Gather rope_delta from requests
                deltas = [getattr(r, "rope_delta", 0) for r in ctx.batch.reqs]
                if any(d != 0 for d in deltas):
                    delta_tensor = torch.tensor(deltas, dtype=torch.int32, device=positions.device)
                    positions = positions + delta_tensor
            q, k = self.rotary.forward(positions, q, k)

        q = q.view(-1, self.num_qo_heads, self.head_dim)
        k = k.contiguous()
        v = v.contiguous()
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return self.o_proj.forward(o.view(-1, self.qo_attn_dim))


# ======================== VL Decoder Layer ========================


class Qwen3VLDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3VLAttn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


# ======================== VL Language Model ========================


class Qwen3VLLanguageModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3VLDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        image_embeds: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        deepstack_visual_embeds: List[torch.Tensor] | None = None,
        visual_pos_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)

        # Replace image token embeddings with vision encoder output
        if image_embeds is not None and image_mask is not None:
            x[image_mask] = image_embeds.to(x.dtype)

        residual: torch.Tensor | None = None
        for layer_idx, layer in enumerate(self.layers.op_list):
            x, residual = layer.forward(x, residual)
            # DeepStack: add visual features after early decoder layers
            if (
                deepstack_visual_embeds is not None
                and visual_pos_mask is not None
                and layer_idx < len(deepstack_visual_embeds)
            ):
                vis_embed = deepstack_visual_embeds[layer_idx].to(x.device, x.dtype)
                # x is the pre-residual output; need to add to residual (which holds the stream)
                residual[visual_pos_mask] = residual[visual_pos_mask].clone() + vis_embed

        return self.norm.forward(x, residual)[0]


# ======================== Top-level VL Model ========================


class Qwen3VLModel(BaseOP):
    def __init__(self, config: ModelConfig):
        assert config.vision_config is not None
        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.language_model = Qwen3VLLanguageModel(config)
        self._image_token_id = config.image_token_id
        self._spatial_merge_size = config.vision_config.spatial_merge_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch

        image_embeds = None
        image_mask = None
        deepstack_visual_embeds = None
        visual_pos_mask = None

        if batch.is_prefill and batch.pixel_values is not None:
            # Build mask for image token positions in the extend range
            image_mask = input_ids == self._image_token_id
            num_image_tokens = image_mask.sum().item()

            if num_image_tokens > 0:
                pixel_values = batch.pixel_values.to(input_ids.device)
                image_grid_thw = batch.image_grid_thw.to(input_ids.device)

                # Run vision encoder
                merged_embeds, deepstack_embeds = self.visual.forward(
                    pixel_values, image_grid_thw
                )
                image_embeds = merged_embeds
                visual_pos_mask = image_mask

                # Build deepstack embeds (after spatial merge, same length as image_embeds)
                deepstack_visual_embeds = deepstack_embeds

        return self.language_model.forward(
            input_ids,
            image_embeds=image_embeds,
            image_mask=image_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
            visual_pos_mask=visual_pos_mask,
        )


class Qwen3VLForConditionalGeneration(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3VLModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.language_model.embed_tokens
            if config.tie_word_embeddings
            else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3VLForConditionalGeneration"]
