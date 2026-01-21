from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch
import torch.nn as nn
from minisgl.core import get_global_ctx
from minisgl.layers import ( BaseOP, OPList, ParallelLMHead, RMSNormFused, 
                    VocabParallelEmbedding, LinearColParallelMerged, LinearRowParallel,
                    QKVParallelLinear)

from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn
from .utils import VisionMLP as Qwen3_VisionMLP
from .utils import VisionAtten as Qwen3_VisionAtten

if TYPE_CHECKING:
    from .config import ModelConfig, VisionConfig

from einops import rearrange


class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
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


class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]

class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class Qwen3_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        intermediate_dim: int,
        hidden_act="silu",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        self.attn = Qwen3_VisionAtten(
            embed_dim=dim,
            num_heads=num_heads,
            # projection_size=dim,
            # use_qkv_parallel=True,
            # proj_bias=True,
            # flatten_batch=True,
        )
        self.mlp = Qwen3_VisionMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.norm1(x)
        hidden_states = rearrange(hidden_states, "s b ... -> b s ...")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        attn = rearrange(attn, "b s ... -> s b ...")
        x += attn
        norm2 = self.norm2(x)
        mlp = self.mlp(norm2)
        x += mlp
        return x

class Qwen3VLVisionPatchMerger(nn.Module):

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)

        self.use_postshuffle_norm = use_postshuffle_norm

        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else context_dim
        )
        self.linear_fc1 = LinearColParallelMerged(
            self.hidden_size,
            self.hidden_size,
            bias=True,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = LinearRowParallel(
            self.hidden_size,
            dim,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)

        x_parallel, _ = self.linear_fc1(x)
        x_parallel = self.act_fn(x_parallel)
        out, _ = self.linear_fc2(x_parallel)
        return out

class Qwen3VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3VLVisionModel(nn.Module):
    def __init__(
        self,
        vision_config: VisionConfig,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)
        self.num_grid = self.num_grid_per_side * self.num_grid_per_side

        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        # layer indexes of which layer's output should be deep-stacked
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self.out_hidden_size = vision_config.out_hidden_size * (
            1 + len(self.deepstack_visual_indexes)
        )
        
        self.patch_embed = Qwen3VLVisionPatchEmbed(config=vision_config)
        self.pos_embed = VocabParallelEmbedding(
                self.num_position_embeddings,
                self.hidden_size,
            )
        ## transformer
        ## self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        
        self.blocks = nn.ModuleList(
            [
                Qwen3_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    intermediate_dim=vision_config.intermediate_size,
                    hidden_act=vision_config.hidden_act,
                )
                for layer_idx in range(vision_config.depth)
            ]
        )
        
        self.merger = Qwen3VLVisionPatchMerger(
            dim=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    dim=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                )
                for layer_idx in range(len(self.deepstack_visual_indexes))
            ]
        )

    def fast_pos_embed_interpolate(self, grid_thw):
        num_grid_per_side = int(self.num_position_embeddings**0.5)

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        # TODO: use torch instand of np
        for t, h, w in grid_thw:
            h_idxs = np.linspace(0, num_grid_per_side - 1, h)
            w_idxs = np.linspace(0, num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.astype(int)
            w_idxs_floor = w_idxs.astype(int)
            h_idxs_ceil = (h_idxs.astype(int) + 1).clip(max=num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.astype(int) + 1).clip(max=num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            idx_list[0].extend(
                ((h_idxs_floor * num_grid_per_side)[None].T + w_idxs_floor[None])
                .flatten()
                .tolist()
                * t
            )
            idx_list[1].extend(
                ((h_idxs_floor * num_grid_per_side)[None].T + w_idxs_ceil[None])
                .flatten()
                .tolist()
                * t
            )
            idx_list[2].extend(
                ((h_idxs_ceil * num_grid_per_side)[None].T + w_idxs_floor[None])
                .flatten()
                .tolist()
                * t
            )
            idx_list[3].extend(
                ((h_idxs_ceil * num_grid_per_side)[None].T + w_idxs_ceil[None])
                .flatten()
                .tolist()
                * t
            )

            weight_list[0].extend(
                ((1 - dh)[None].T * (1 - dw)[None]).flatten().tolist() * t
            )
            weight_list[1].extend(((1 - dh)[None].T * dw[None]).flatten().tolist() * t)
            weight_list[2].extend((dh[None].T * (1 - dw)[None]).flatten().tolist() * t)
            weight_list[3].extend((dh[None].T * dw[None]).flatten().tolist() * t)

        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype

        p0 = (
            self.pos_embed(torch.tensor(idx_list[0], dtype=torch.long, device=device))
            * torch.tensor(weight_list[0], dtype=dtype, device=device)[:, None]
        )
        p1 = (
            self.pos_embed(torch.tensor(idx_list[1], dtype=torch.long, device=device))
            * torch.tensor(weight_list[1], dtype=dtype, device=device)[:, None]
        )
        p2 = (
            self.pos_embed(torch.tensor(idx_list[2], dtype=torch.long, device=device))
            * torch.tensor(weight_list[2], dtype=dtype, device=device)[:, None]
        )
        p3 = (
            self.pos_embed(torch.tensor(idx_list[3], dtype=torch.long, device=device))
            * torch.tensor(weight_list[3], dtype=dtype, device=device)[:, None]
        )

        patch_pos_embeds = p0 + p1 + p2 + p3
        patch_pos_embeds = patch_pos_embeds.split([t * h * w for t, h, w in grid_thw])
        patch_pos_embeds_permute = []
        m_size = self.spatial_merge_size
        for pos_embed, (t, h, w) in zip(patch_pos_embeds, grid_thw):
            pos_embed = (
                pos_embed.view(t, h // m_size, m_size, w // m_size, m_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        x += pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = x.size()
        rotary_pos_emb = rotary_pos_emb.to(x.device)

        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # compute cu_seqlens
        # cu_seqlens = compute_cu_seqlens_from_grid_numpy(grid_thw)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        x = x.unsqueeze(1)

        deepstack_feature_lists = []
        num_deepstack_captured = 0
        for layer_num, blk in enumerate(self.blocks):
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[num_deepstack_captured](
                    x
                )
                deepstack_feature_lists.append(deepstack_feature)
                num_deepstack_captured += 1
        x = self.merger(x)
        hidden_states = torch.cat(
            [x] + deepstack_feature_lists, dim=1
        )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
        return hidden_states



class Qwen3VLForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
       
        print(f"Qwen3vl model_config  info:\n {config}") 
        
        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()
        
        # deepstack
        self.deepstack_visual_indexes = config.vision_config.deepstack_visual_indexes
        self.num_deepstack_embeddings = len(self.deepstack_visual_indexes)
        self.use_deepstack = {Modality.IMAGE: True, Modality.VIDEO: True}

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3VLForCausalLM"]
