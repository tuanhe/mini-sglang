from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
from transformers.activations import ACT2FN
from einops import rearrange

from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearRowParallel,
    LinearColumnParallel,
    QKVParallelLinear,
    RMSNorm,
    silu_and_mul,
)
from minisgl.models import ModelConfig
from minisgl.utils import nvtx_annotate
from minisgl.core import get_global_ctx

if TYPE_CHECKING:
    import torch


class GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )

        match config.hidden_act:
            case "silu":
                self.act_fn = silu_and_mul
            case act_fn:
                raise ValueError(f"Unsupported activation function: {act_fn}")

        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


class RopeAttn(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
    ):
        head_dim = config.head_dim
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
        )
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x
        o = self.attn.forward(qkv)
        return self.o_proj.forward(o)


class VisionMLP(BaseOP):

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
        hidden_act="silu",
    ):
        super().__init__()
        self.linear_fc1 = LinearColumnParallel(
            in_features,
            hidden_features,
            bias=bias,
        )
        self.linear_fc2 = LinearRowParallel(
            hidden_features,
            in_features,
            bias=bias,
        )
        self.act = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor):
        x_fc1, _ = self.linear_fc1(x)
        mlp_output, _ = self.linear_fc2(self.act(x_fc1))
        return mlp_output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed

class VisionAtten(BaseOP):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        # projection_size: int,
    ):
        self.head_size = embed_dim // num_heads
        
        self.qkv_proj = QKVParallelLinear(
                hidden_size=embed_dim,
                head_size=self.head_size,
                total_num_heads= num_heads,
                total_num_kv_heads= num_heads,
                bias=True)
        
        self.proj = LinearRowParallel(
            input_size=self.dummy_dim,
            output_size=embed_dim,
            bias=True,
        )
        
    
    def forward(self, hidden_states: torch.Tensor,
                        cu_seqlens: Optional[torch.Tensor] = None,
                        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                        attention_mask: Optional[torch.Tensor] = None,
                        **kwargs,):
        ctx = get_global_ctx()
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attn_output = ctx.attn_backend.forward(query_states, key_states, value_states, self.layer_id, ctx.batch)
        
        attn_output = rearrange(attn_output, "(b s) ... h d -> b s ... (h d)", b=ctx.batch)
        
        attn_output, _ = self.proj(attn_output)
        
        return attn_output
    

__all__ = ["GatedMLP", "RopeAttn", "VisionMLP","VisionAtten"]
