from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
    # VL-specific fields
    pixel_values: torch.Tensor | None = None  # CPU tensor for vision encoder
    image_grid_thw: torch.Tensor | None = None  # (num_images, 3)
    mrope_position_ids: torch.Tensor | None = None  # (3, input_len) for M-RoPE
    rope_delta: int = 0


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int
