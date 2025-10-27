"""Caption generation pipeline components.

This package exposes a light-weight 360° captioning pipeline composed of
specialised panoramic frame samplers, audio/visual encoder backbones and a
decoder head built on top of Hugging Face ``transformers``.  The modules are
designed to be easily swappable via the :mod:`captionqa.captioning.config`
factory helpers while keeping the public API as simple as calling
``generate_captions``.
"""

from .config import CaptioningConfig, build_pipeline
from .pipeline import CaptioningPipeline, generate_captions

__all__ = [
    "CaptioningConfig",
    "CaptioningPipeline",
    "build_pipeline",
    "generate_captions",
]
