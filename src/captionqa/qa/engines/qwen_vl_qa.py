"""Qwen-VL QA engine (optional dependency, mirrors caption engine style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:  # pragma: no cover
    from transformers import AutoProcessor, AutoModelForCausalLM
except Exception:  # pragma: no cover
    AutoProcessor = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore

from ...captioning.panorama import PanoramicFrameSampler, PanoramaSamplingConfig
from ...captioning.config import QwenVLConfig


@dataclass
class QwenVLVQAEngine:
    sampler: PanoramicFrameSampler
    config: QwenVLConfig

    @classmethod
    def from_configs(
        cls, *, sampler_cfg: PanoramaSamplingConfig, qwen_cfg: QwenVLConfig
    ) -> "QwenVLVQAEngine":
        return cls(PanoramicFrameSampler(sampler_cfg), qwen_cfg)

    def _to_images(self, frames: List[np.ndarray]) -> List[Image.Image]:
        if Image is None:
            return []
        images: List[Image.Image] = []
        for arr in frames[: max(int(self.config.num_frames or 1), 1)]:
            if isinstance(arr, np.ndarray):
                images.append(Image.fromarray(arr.astype(np.uint8)))
        return images

    def _load_model(self):
        if AutoProcessor is None or AutoModelForCausalLM is None or torch is None:
            return None, None
        try:
            processor = AutoProcessor.from_pretrained(self.config.model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=getattr(torch, "float16", None),
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            return processor, model
        except Exception:
            return None, None

    def answer(self, video_path: str, question: str, *, context: Optional[str] = None, start_sec: float | None = None, end_sec: float | None = None) -> str:
        frames = self.sampler.sample(video_path, start_sec=start_sec, end_sec=end_sec)
        images = self._to_images(frames)
        qtext = question.strip()
        if context:
            qtext = f"Context: {context.strip()}\nQuestion: {qtext}"

        processor, model = self._load_model()
        if processor is None or model is None or not images:
            return f"[Qwen-VL unavailable or no frames]"

        try:
            device = self.config.device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
            mm_inputs = processor(images=images, text=qtext, return_tensors="pt").to(device)
            generated = model.generate(
                **mm_inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=(self.config.temperature or 0.0) > 0,
                temperature=max(0.0, float(self.config.temperature)),
                top_p=float(self.config.top_p),
            )
            text = processor.batch_decode(generated, skip_special_tokens=True)[0]
            return text
        except Exception:
            return "[Qwen-VL inference error]"
