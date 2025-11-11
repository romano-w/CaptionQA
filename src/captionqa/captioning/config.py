"""Configuration objects and factories for the captioning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .decoders import CaptionDecoder, CaptionDecoderConfig
from .encoders import AudioEncoder, AudioEncoderConfig, VisualEncoder, VisualEncoderConfig
from .panorama import PanoramaSamplingConfig, PanoramicFrameSampler
from .fusion import FusionConfig, FusionHead


@dataclass
class CaptioningConfig:
    """Top level configuration describing the captioning pipeline."""

    panorama: PanoramaSamplingConfig
    visual_encoder: VisualEncoderConfig
    audio_encoder: AudioEncoderConfig
    decoder: CaptionDecoderConfig
    fusion: FusionConfig

    @classmethod
    def from_defaults(
        cls,
        panorama: Optional[PanoramaSamplingConfig] = None,
        visual_encoder: Optional[VisualEncoderConfig] = None,
        audio_encoder: Optional[AudioEncoderConfig] = None,
        decoder: Optional[CaptionDecoderConfig] = None,
        fusion: Optional[FusionConfig] = None,
    ) -> "CaptioningConfig":
        """Create a configuration populated with sensible defaults."""

        return cls(
            panorama=panorama or PanoramaSamplingConfig(),
            visual_encoder=visual_encoder or VisualEncoderConfig(),
            audio_encoder=audio_encoder or AudioEncoderConfig(),
            decoder=decoder or CaptionDecoderConfig(),
            fusion=fusion or FusionConfig(),
        )


def build_pipeline(config: CaptioningConfig):
    """Instantiate pipeline components described by ``config``."""

    sampler = PanoramicFrameSampler(config.panorama)
    visual = VisualEncoder(config.visual_encoder)
    audio = AudioEncoder(config.audio_encoder)
    decoder = CaptionDecoder(config.decoder)
    fusion = FusionHead(config.fusion)
    return sampler, visual, audio, decoder, fusion

