"""Configuration objects and factories for the captioning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .decoders import CaptionDecoder, CaptionDecoderConfig
from .encoders import AudioEncoder, AudioEncoderConfig, VisualEncoder, VisualEncoderConfig
from .panorama import PanoramaSamplingConfig, PanoramicFrameSampler
from .fusion import FusionConfig, FusionHead


@dataclass
class QwenVLConfig:
    """Configuration for Qwen-VL caption engine.

    This keeps the project lean by only storing the model identifier and a few
    generation knobs; actual model loading is optional at runtime.
    """

    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: Optional[str] = None
    num_frames: int = 8
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    caption_template: str = (
        "You are an expert assistant describing 360-degree panoramic video. "
        "Summarize the salient events concisely and accurately. Avoid speculation."
    )
    action_caption_template: str = (
        "You are an expert spotter describing 360-degree panoramic video. "
        "Prioritize concrete actions, verbs, and human-object interactions. "
        "Keep sentences short, precise, and free of speculation."
    )
    qa_template: str = (
        "You are answering a question about a 360-degree panoramic video. "
        "Use the provided views to answer succinctly."
    )


@dataclass
class CaptioningConfig:
    """Top level configuration describing the captioning pipeline."""

    panorama: PanoramaSamplingConfig
    visual_encoder: VisualEncoderConfig
    audio_encoder: AudioEncoderConfig
    decoder: CaptionDecoderConfig
    fusion: FusionConfig
    # Which engine to use: "fusion" (default) or "qwen_vl"
    engine: str = "fusion"
    # Engine-specific config for Qwen-VL
    qwen_vl: QwenVLConfig = field(default_factory=QwenVLConfig)

    @classmethod
    def from_defaults(
        cls,
        panorama: Optional[PanoramaSamplingConfig] = None,
        visual_encoder: Optional[VisualEncoderConfig] = None,
        audio_encoder: Optional[AudioEncoderConfig] = None,
        decoder: Optional[CaptionDecoderConfig] = None,
        fusion: Optional[FusionConfig] = None,
        engine: Optional[str] = None,
        qwen_vl: Optional[QwenVLConfig] = None,
    ) -> "CaptioningConfig":
        """Create a configuration populated with sensible defaults."""

        return cls(
            panorama=panorama or PanoramaSamplingConfig(),
            visual_encoder=visual_encoder or VisualEncoderConfig(),
            audio_encoder=audio_encoder or AudioEncoderConfig(),
            decoder=decoder or CaptionDecoderConfig(),
            fusion=fusion or FusionConfig(),
            engine=engine or "fusion",
            qwen_vl=qwen_vl or QwenVLConfig(),
        )


def build_pipeline(config: CaptioningConfig):
    """Instantiate pipeline components described by ``config``."""

    sampler = PanoramicFrameSampler(config.panorama)
    visual = VisualEncoder(config.visual_encoder)
    audio = AudioEncoder(config.audio_encoder)
    decoder = CaptionDecoder(config.decoder)
    fusion = FusionHead(config.fusion)
    return sampler, visual, audio, decoder, fusion

