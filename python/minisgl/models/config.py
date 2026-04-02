from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class VisionConfig:
    depth: int
    hidden_size: int
    hidden_act: str
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    out_hidden_size: int
    num_position_embeddings: int
    deepstack_visual_indexes: Tuple[int, ...]


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    # VL-specific fields
    vision_config: VisionConfig | None = None
    image_token_id: int | None = None
    video_token_id: int | None = None
    mrope_section: Tuple[int, ...] | None = None
    mrope_interleaved: bool = False

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type

    @property
    def is_vl(self) -> bool:
        return self.vision_config is not None

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        # Extract VL-specific fields from top-level config before descending into text_config
        vision_config = None
        image_token_id = None
        video_token_id = None
        mrope_section = None
        mrope_interleaved = False

        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

            # Parse vision config if present
            vc = getattr(top, "vision_config", None)
            if vc is not None:
                vc_dict = vc.to_dict() if hasattr(vc, "to_dict") else vc
                vision_config = VisionConfig(
                    depth=vc_dict["depth"],
                    hidden_size=vc_dict["hidden_size"],
                    hidden_act=vc_dict["hidden_act"],
                    intermediate_size=vc_dict["intermediate_size"],
                    num_heads=vc_dict["num_heads"],
                    in_channels=vc_dict.get("in_channels", 3),
                    patch_size=vc_dict["patch_size"],
                    spatial_merge_size=vc_dict["spatial_merge_size"],
                    temporal_patch_size=vc_dict["temporal_patch_size"],
                    out_hidden_size=vc_dict["out_hidden_size"],
                    num_position_embeddings=vc_dict["num_position_embeddings"],
                    deepstack_visual_indexes=tuple(vc_dict.get("deepstack_visual_indexes", [])),
                )
                image_token_id = getattr(top, "image_token_id", None)
                video_token_id = getattr(top, "video_token_id", None)

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]

        # Extract M-RoPE config from rope_scaling
        if rope_scaling is not None:
            mrope_section = rope_scaling.get("mrope_section", None)
            if mrope_section is not None:
                mrope_section = tuple(mrope_section)
            mrope_interleaved = rope_scaling.get("mrope_interleaved", False)

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            vision_config=vision_config,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
        )