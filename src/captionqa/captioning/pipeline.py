"""High-level captioning pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import hashlib
from pathlib import Path

import torch

from .config import CaptioningConfig, build_pipeline
from .decoders import CaptionDecoder
from .encoders import AudioEncoder, VisualEncoder
from .panorama import PanoramicFrameSampler
from .fusion import FusionHead
from .engines.qwen_vl import QwenVLEngine


@dataclass
class CaptioningPipeline:
    """Composable pipeline for panoramic caption generation."""

    sampler: PanoramicFrameSampler
    visual_encoder: VisualEncoder
    audio_encoder: AudioEncoder
    decoder: CaptionDecoder
    fusion: FusionHead

    @classmethod
    def from_config(cls, config: Optional[CaptioningConfig] = None) -> "CaptioningPipeline":
        config = config or CaptioningConfig.from_defaults()
        sampler, visual, audio, decoder, fusion = build_pipeline(config)
        return cls(sampler=sampler, visual_encoder=visual, audio_encoder=audio, decoder=decoder, fusion=fusion)

    def generate(
        self,
        video_path: str,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate a caption for ``video_path``."""

        frames = self.sampler.sample(video_path)

        # Build stable cache keys tied to video path, sampler & model configs
        key_material_visual = (
            f"{Path(video_path).resolve()}|vis:{self.visual_encoder.config.model_name}|"
            f"fps:{self.sampler.config.frame_rate}|proj:{self.sampler.config.enable_projection}|"
            f"views:{self.sampler.config.num_views}|res:{self.sampler.config.target_resolution}"
        )
        key_material_audio = (
            f"{Path(video_path).resolve()}|aud:{self.audio_encoder.config.model_name}|sr:{self.audio_encoder.config.sample_rate}"
        )
        v_key = hashlib.sha1(key_material_visual.encode("utf-8")).hexdigest()
        a_key = hashlib.sha1(key_material_audio.encode("utf-8")).hexdigest()

        visual_features = self.visual_encoder.encode(frames, cache_key=v_key)
        audio_features = self.audio_encoder.encode(video_path, cache_key=a_key)

        # Compose human-readable base prompt and fuse features for conditioning
        fused_prompt = self._compose_prompt(prompt, visual_features, audio_features)
        conditioning = self.fusion.fuse(visual_features, audio_features)

        if max_new_tokens is not None:
            self.decoder.config.max_new_tokens = max_new_tokens

        try:
            return self.decoder.generate(fused_prompt, conditioning=conditioning)
        except TypeError:
            # Back-compat: if decoder doesn't accept conditioning, fall back to prompt-only
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
    temporal_window: Optional[tuple[float, float]] = None,
) -> str:
    """Generate a caption using the configured engine.

    Engines:
    - fusion (default): use Visual/Audio encoders + Fusion + Decoder (existing path)
    - qwen_vl: sample frames and prompt a Qwen-VL model
    """

    cfg = config or CaptioningConfig.from_defaults()
    if getattr(cfg, "engine", "fusion") == "qwen_vl":
        engine = QwenVLEngine.from_configs(sampler_cfg=cfg.panorama, qwen_cfg=cfg.qwen_vl)
        start_end = temporal_window or (None, None)
        return engine.generate(
            video_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            start_sec=start_end[0],
            end_sec=start_end[1],
        )

    pipeline = CaptioningPipeline.from_config(cfg)
    return pipeline.generate(video_path, prompt=prompt, max_new_tokens=max_new_tokens)

