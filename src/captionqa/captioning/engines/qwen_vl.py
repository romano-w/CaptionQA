"""Qwen-VL caption engine (optional heavy dependency).

This engine uses the panoramic frame sampler to select a small set of frames
and prompts a Qwen2.5-VL/Qwen3-VL model to produce a caption. When transformers
or the model are unavailable, it falls back to a deterministic message so the
CLI remains usable in lightweight environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional heavy dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - optional heavy dependency
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:  # pragma: no cover - optional heavy dependency
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
except Exception:  # pragma: no cover
    AutoProcessor = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoModelForVision2Seq = None  # type: ignore

from ..panorama import PanoramicFrameSampler, PanoramaSamplingConfig
from ..config import QwenVLConfig


@dataclass
class QwenVLEngine:
    sampler: PanoramicFrameSampler
    config: QwenVLConfig
    _processor: Optional[object] = field(init=False, default=None)
    _model: Optional[object] = field(init=False, default=None)

    @classmethod
    def from_configs(
        cls, *, sampler_cfg: PanoramaSamplingConfig, qwen_cfg: QwenVLConfig
    ) -> "QwenVLEngine":
        return cls(PanoramicFrameSampler(sampler_cfg), qwen_cfg)

    def _to_images(self, frames: List[np.ndarray]) -> List[Image.Image]:
        if Image is None:
            return []
        images: List[Image.Image] = []
        for arr in frames[: max(int(self.config.num_frames or 1), 1)]:
            if isinstance(arr, np.ndarray):
                images.append(Image.fromarray(arr.astype(np.uint8)))
        return images

    def __post_init__(self) -> None:
        self._processor = None
        self._model = None

    def _load_model(self) -> Tuple[Optional[object], Optional[object]]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        if AutoProcessor is None or torch is None:
            return None, None
        try:
            processor = AutoProcessor.from_pretrained(self.config.model_name, trust_remote_code=True)
        except Exception:
            return None, None

        model = None
        model_kwargs = {
            "torch_dtype": getattr(torch, "float16", None),
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True,
        }

        for loader in (AutoModelForVision2Seq, AutoModelForCausalLM):
            if loader is None:
                continue
            try:
                model = loader.from_pretrained(self.config.model_name, **model_kwargs)
                break
            except Exception:  # pragma: no cover - informative logging
                model = None
        if model is None:
            return None, None

        self._processor = processor
        self._model = model
        return self._processor, self._model

    def generate(self, video_path: str, *, prompt: Optional[str] = None, max_new_tokens: Optional[int] = None, start_sec: float | None = None, end_sec: float | None = None) -> str:
        frames = self.sampler.sample(video_path, start_sec=start_sec, end_sec=end_sec)
        images = self._to_images(frames)

        base_prompt = (
            prompt
            or f"{(self.config.caption_template or '').strip()}\n"
            f"Context: The input consists of ~{len(images)} perspective views sampled from a 360-degree panorama."
        )

        processor, model = self._load_model()
        if processor is None or model is None or not images:
            return (
                f"{base_prompt.strip()} [Qwen-VL unavailable or no frames; falling back to deterministic text. "
                f"frames={len(images)}]"
            )

        # Build inputs according to Qwen-VL processors; rely on trust_remote_code interface
        try:
            device = self.config.device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
            mm_inputs = processor(images=images, text=base_prompt, return_tensors="pt").to(device)
            generated = model.generate(
                **mm_inputs,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                do_sample=(self.config.temperature or 0.0) > 0,
                temperature=max(0.0, float(self.config.temperature)),
                top_p=float(self.config.top_p),
            )
            text = processor.batch_decode(generated, skip_special_tokens=True)[0]
            return text
        except Exception:
            return (
                f"{base_prompt.strip()} [Qwen-VL inference error; falling back to deterministic text.]"
            )
