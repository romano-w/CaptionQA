"""Qwen-VL QA engine (optional dependency, mirrors caption engine style)."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import List, Optional, Tuple

import numpy as np

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover
    from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
except Exception:  # pragma: no cover
    AutoConfig = None  # type: ignore
    AutoProcessor = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoModelForVision2Seq = None  # type: ignore

from ...captioning.panorama import PanoramicFrameSampler, PanoramaSamplingConfig
from ...captioning.config import QwenVLConfig
from ...captioning._qwen_utils import prepare_video_payload, sanitize_qwen_config


logger = logging.getLogger(__name__)


@dataclass
class QwenVLVQAEngine:
    sampler: PanoramicFrameSampler
    config: QwenVLConfig
    _processor: Optional[object] = field(init=False, default=None)
    _model: Optional[object] = field(init=False, default=None)

    @classmethod
    def from_configs(
        cls, *, sampler_cfg: PanoramaSamplingConfig, qwen_cfg: QwenVLConfig
    ) -> "QwenVLVQAEngine":
        return cls(PanoramicFrameSampler(sampler_cfg), qwen_cfg)

    def __post_init__(self) -> None:
        self._processor = None
        self._model = None

    def _load_model(self) -> Tuple[Optional[object], Optional[object]]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        if AutoProcessor is None or torch is None:
            logger.warning("Qwen-VL QA dependencies unavailable (processor=%s, torch=%s)", AutoProcessor, torch)
            return None, None
        try:
            processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.exception("Failed to load Qwen processor %s: %s", self.config.model_name, exc)
            return None, None

        model = None
        config = None
        if AutoConfig is not None:
            try:
                config = AutoConfig.from_pretrained(self.config.model_name, trust_remote_code=True)
                config = sanitize_qwen_config(config)
            except Exception as exc:
                logger.warning("Unable to load Qwen config for %s: %s", self.config.model_name, exc)
                config = None
        model_kwargs = {
            "torch_dtype": getattr(torch, "float16", None),
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True,
        }
        if config is not None:
            model_kwargs["config"] = config
        for loader in (AutoModelForVision2Seq, AutoModelForCausalLM):
            if loader is None:
                continue
            try:
                model = loader.from_pretrained(self.config.model_name, **model_kwargs)
                break
            except Exception as exc:
                logger.warning("Qwen QA loader %s failed: %s", getattr(loader, "__name__", loader), exc)
                model = None
        if model is None:
            logger.error("Unable to initialize Qwen QA model %s", self.config.model_name)
            return None, None

        self._processor = processor
        self._model = model
        return self._processor, self._model

    def answer(self, video_path: str, question: str, *, context: Optional[str] = None, start_sec: float | None = None, end_sec: float | None = None) -> str:
        frames = self.sampler.sample(video_path, start_sec=start_sec, end_sec=end_sec)
        video_payload = prepare_video_payload(frames, self.config.num_frames)
        base = (self.config.qa_template or "").strip()
        qtext = f"{base}\nQuestion: {question.strip()}"
        if context:
            qtext = f"{qtext}\nContext: {context.strip()}"

        processor, model = self._load_model()
        if processor is None or model is None or video_payload is None:
            logger.warning(
                "Qwen-VL QA unavailable (processor=%s model=%s video=%s)",
                processor is not None,
                model is not None,
                video_payload is not None,
            )
            return f"[Qwen-VL unavailable or no frames]"

        try:
            device = self.config.device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_payload},
                        {"type": "text", "text": qtext.strip()},
                    ],
                }
            ]
            chat_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            mm_inputs = processor(videos=[video_payload], text=chat_prompt, return_tensors="pt").to(device)
            generated = model.generate(
                **mm_inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=(self.config.temperature or 0.0) > 0,
                temperature=max(0.0, float(self.config.temperature)),
                top_p=float(self.config.top_p),
            )
            input_length = mm_inputs["input_ids"].shape[-1]
            trimmed = generated[:, input_length:]
            if trimmed.shape[-1] == 0:
                trimmed = generated
            text = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            return text
        except Exception as exc:
            logger.exception("Qwen-VL QA inference error: %s", exc)
            return "[Qwen-VL inference error]"
