from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg,
)
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.utils import cached_load_hf_config


class RequestAllFinished(Exception):
    pass


@dataclass
class RequestStatus:
    uid: int
    input_ids: List[int]
    output_ids: List[int]


@dataclass
class _VLRequestData:
    """Pre-processed VL request data."""

    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    mrope_position_ids: torch.Tensor
    rope_delta: int


class LLM(Scheduler):
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        self.pending_requests: List[Tuple[Any, SamplingParams]] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.counter = 0

        # VL support: lazy-init processor
        self._model_config = config.model_config
        self._vl_processor = None
        self._model_path = model_path

    def _get_vl_processor(self):
        if self._vl_processor is None:
            from transformers import AutoProcessor

            self._vl_processor = AutoProcessor.from_pretrained(
                self._model_path, trust_remote_code=True
            )
        return self._vl_processor

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def _process_vl_request(
        self, messages: List[Dict], images: List[Any]
    ) -> _VLRequestData:
        """Process a VL request: tokenize text+images, compute M-RoPE positions."""
        from PIL import Image

        processor = self._get_vl_processor()

        # Load images from file paths if needed
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img).convert("RGB"))
            else:
                loaded_images.append(img)

        # Use the HF processor to get input_ids and pixel_values
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=loaded_images if loaded_images else None,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0).to(torch.int32)
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        if pixel_values is None:
            # No images, return simple text request
            return _VLRequestData(
                input_ids=input_ids,
                pixel_values=None,
                image_grid_thw=None,
                mrope_position_ids=None,
                rope_delta=0,
            )

        # Compute M-RoPE position IDs
        mrope_position_ids, rope_delta = self._compute_mrope_positions(
            input_ids, image_grid_thw
        )

        return _VLRequestData(
            input_ids=input_ids,
            pixel_values=pixel_values.squeeze(0) if pixel_values.dim() > 2 else pixel_values,
            image_grid_thw=image_grid_thw,
            mrope_position_ids=mrope_position_ids,
            rope_delta=rope_delta,
        )

    def _compute_mrope_positions(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, int]:
        """Compute M-RoPE 3D position IDs for a single request."""
        if image_grid_thw is None or self._model_config.mrope_section is None:
            seq_len = len(input_ids)
            pos = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(3, -1)
            return pos, 0

        spatial_merge_size = self._model_config.vision_config.spatial_merge_size
        image_token_id = self._model_config.image_token_id
        video_token_id = self._model_config.video_token_id
        vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")

        input_tokens = input_ids.tolist()
        llm_pos_ids_list = []
        st = 0
        image_index = 0

        # Find contiguous groups of image tokens (each group = one image)
        image_groups = []  # list of (start, length)
        i = 0
        while i < len(input_tokens):
            if input_tokens[i] == image_token_id:
                group_start = i
                while i < len(input_tokens) and input_tokens[i] == image_token_id:
                    i += 1
                image_groups.append((group_start, i - group_start))
            else:
                i += 1

        for group_start, group_len in image_groups:
            t, h, w = image_grid_thw[image_index]
            image_index += 1

            llm_grid_t = t.item()
            llm_grid_h = h.item() // spatial_merge_size
            llm_grid_w = w.item() // spatial_merge_size

            # Text before image
            text_len = group_start - st
            if text_len > 0:
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            # Image tokens
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            t_index = (
                torch.arange(llm_grid_t)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + st_idx
            )
            st = group_start + group_len

        # Remaining text after last image
        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        if not llm_pos_ids_list:
            seq_len = len(input_ids)
            pos = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(3, -1)
            return pos, 0

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).to(torch.int32)
        rope_delta = int(llm_positions.max().item()) + 1 - len(input_ids)
        return llm_positions, rope_delta

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for request_data, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break

            # Check if this is a VL request (dict with messages + images)
            if isinstance(request_data, _VLRequestData):
                input_ids = request_data.input_ids
                sum_input_len += len(input_ids)
                uid, added = self.counter + added, added + 1
                results.append(
                    UserMsg(
                        uid=uid,
                        input_ids=input_ids,
                        sampling_params=sampling_params,
                        pixel_values=request_data.pixel_values,
                        image_grid_thw=request_data.image_grid_thw,
                        mrope_position_ids=request_data.mrope_position_ids,
                        rope_delta=request_data.rope_delta,
                    )
                )
                self.status_map[uid] = RequestStatus(
                    uid=uid, input_ids=input_ids.tolist(), output_ids=[]
                )
            else:
                input_ids = self._tokenize_one(request_data)
                sum_input_len += len(input_ids)
                uid, added = self.counter + added, added + 1
                results.append(
                    UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params)
                )
                self.status_map[uid] = RequestStatus(
                    uid=uid,
                    input_ids=(
                        input_ids.tolist()
                        if isinstance(request_data, str)
                        else request_data
                    ),
                    output_ids=[],
                )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                status.output_ids.append(msg.next_token)

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
        images: List[Any] | None = None,
    ) -> List[Dict[str, str | List[int]]]:
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)

        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            req_images = None
            if images is not None and i < len(images) and images[i] is not None:
                req_images = images[i]

            if req_images is not None and self._model_config.is_vl:
                # VL request: prompt should be a string or chat messages
                if isinstance(prompt, str):
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img}
                                for img in (
                                    req_images
                                    if isinstance(req_images, list)
                                    else [req_images]
                                )
                            ]
                            + [{"type": "text", "text": prompt}],
                        }
                    ]
                elif isinstance(prompt, list) and isinstance(prompt[0], dict):
                    messages = prompt
                else:
                    raise ValueError(
                        "For VL requests, prompt must be a string or list of message dicts"
                    )
                vl_data = self._process_vl_request(
                    messages,
                    req_images if isinstance(req_images, list) else [req_images],
                )
                self.pending_requests.append((vl_data, sp))
            else:
                self.pending_requests.append((prompt, sp))

        try:
            self.run_forever()
        except RequestAllFinished:
            pass
        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results
