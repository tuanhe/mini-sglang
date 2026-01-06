from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from transformers import LlamaConfig, Qwen3VLConfig

@dataclass(frozen=True)
class VisionConfig:
    # Qwen3VL视觉配置的核心字段（与HF配置完全对齐）
    depth: int  # 视觉Transformer层数（对应HF的depth）
    hidden_size: int  # 视觉隐藏层维度
    hidden_act: str  # 视觉激活函数（如gelu_pytorch_tanh）
    intermediate_size: int  # 视觉FFN中间层维度
    num_heads: int  # 视觉注意力头数
    patch_size: int  # 图像分块大小（如16）
    in_channels: int  # 输入图像通道数（固定为3）
    initializer_range: float  # 初始化范围
    num_position_embeddings: int  # 视觉位置编码长度
    out_hidden_size: int  # 视觉输出维度（与语言模型hidden_size对齐）
    spatial_merge_size: int  # 空间合并窗口大小（Qwen3VL的下采样策略）
    deepstack_visual_indexes: list[int]  # DeepStack视觉特征索引（Qwen3VL特有）
    # （删除原num_experts、rms_norm_eps等HF配置中不存在的字段）

    
@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, float] | None


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
    
    # 新增：视觉模态配置（多模态模型时非空，纯语言时为None）
    vision_config: Optional[VisionConfig] = field(default=None)

    @classmethod
    def from_hf(cls, config: LlamaConfig | Qwen3VLConfig) -> ModelConfig:
        
        # 多模态配置：语言参数存在于text_config中
        if isinstance(config, Qwen3VLConfig):
            text_config = config.text_config
        else:  # 纯语言配置：直接用config本身
            text_config = config
        
        num_kv_heads = getattr(text_config, "num_key_value_heads", text_config.num_attention_heads)
        head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
        tie_word_embeddings = getattr(text_config, "tie_word_embeddings", False)

        visual_config = None
        if isinstance(config, Qwen3VLConfig):
            hf_vision_config = config.vision_config
            visual_config = VisionConfig(
                depth=hf_vision_config.depth,
                hidden_size=hf_vision_config.hidden_size,
                hidden_act=hf_vision_config.hidden_act,
                intermediate_size=hf_vision_config.intermediate_size,
                num_heads=hf_vision_config.num_heads,
                patch_size=hf_vision_config.patch_size,
                in_channels=hf_vision_config.in_channels,
                initializer_range=hf_vision_config.initializer_range,
                num_position_embeddings=hf_vision_config.num_position_embeddings,
                out_hidden_size=hf_vision_config.out_hidden_size,
                spatial_merge_size=hf_vision_config.spatial_merge_size,
                deepstack_visual_indexes=hf_vision_config.deepstack_visual_indexes,
            )

        return cls(
            num_layers=text_config.num_hidden_layers,
            num_qo_heads=text_config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=text_config.hidden_size,
            vocab_size=text_config.vocab_size,
            intermediate_size=text_config.intermediate_size,
            hidden_act=text_config.hidden_act,
            rms_norm_eps=text_config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=text_config.max_position_embeddings,
                base=text_config.rope_theta,
                scaling=getattr(text_config, "rope_scaling", None),
            ),
            vision_config=visual_config
        )
