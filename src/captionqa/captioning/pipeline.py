"""High-level captioning pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .config import CaptioningConfig, build_pipeline
from .decoders import CaptionDecoder
from .encoders import AudioEncoder, VisualEncoder
from .panorama import PanoramicFrameSampler


@dataclass
class CaptioningPipeline:
    """Composable pipeline for panoramic caption generation."""

    sampler: PanoramicFrameSampler
    visual_encoder: VisualEncoder
    audio_encoder: AudioEncoder
    decoder: CaptionDecoder

    @classmethod
    def from_config(cls, config: Optional[CaptioningConfig] = None) -> "CaptioningPipeline":
        config = config or CaptioningConfig.from_defaults()
        sampler, visual, audio, decoder = build_pipeline(config)
        return cls(sampler=sampler, visual_encoder=visual, audio_encoder=audio, decoder=decoder)

    def generate(
        self,
        video_path: str,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate a caption for ``video_path``."""

        frames = self.sampler.sample(video_path)
        visual_features = self.visual_encoder.encode(frames)
        audio_features = self.audio_encoder.encode(video_path)

        fused_prompt = self._compose_prompt(prompt, visual_features, audio_features)

        if max_new_tokens is not None:
            self.decoder.config.max_new_tokens = max_new_tokens

        return self.decoder.generate(fused_prompt)

    def _compose_prompt(
        self,
        prompt: Optional[str],
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> str:
        base_prompt = (
            prompt
            or "Describe the salient events occurring in this 360-degree video."
        )

        visual_summary = self._tensor_summary(visual_features, label="visual")
        audio_summary = self._tensor_summary(audio_features, label="audio")

        return (
            f"{base_prompt.strip()}\n"
            f"Visual embedding stats: {visual_summary}.\n"
            f"Audio embedding stats: {audio_summary}."
        )

    @staticmethod
    def _tensor_summary(tensor: torch.Tensor, label: str) -> str:
        if tensor is None or tensor.numel() == 0:
            return f"No {label} features available"
        mean = tensor.float().mean().item()
        std = tensor.float().std(unbiased=False).item()
        return f"mean={mean:.4f}, std={std:.4f}, shape={tuple(tensor.shape)}"


def generate_captions(
    video_path: str,
    *,
    config: Optional[CaptioningConfig] = None,
    prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> str:
    """Convenience wrapper around :class:`CaptioningPipeline`."""

    pipeline = CaptioningPipeline.from_config(config)
    return pipeline.generate(video_path, prompt=prompt, max_new_tokens=max_new_tokens)

