"""Caption generation pipeline components with lazy imports.

This package exposes a light-weight 360° captioning pipeline composed of
specialised panoramic frame samplers, audio/visual encoder backbones and a
decoder head built on top of Hugging Face ``transformers``.  The modules are
designed to be easily swappable via the :mod:`captionqa.captioning.config`
factory helpers while keeping the public API as simple as calling
``generate_captions``.

Historically importing :mod:`captionqa.captioning` pulled in heavyweight
dependencies such as PyTorch even when users only needed utility helpers like
the panorama projector.  This module now exposes the public API via lazy
attribute access so that optional dependencies are only imported when
explicitly requested.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["CaptioningConfig", "CaptioningPipeline", "build_pipeline", "generate_captions"]

_EXPORTS = {
    "CaptioningConfig": ("captionqa.captioning.config", "CaptioningConfig"),
    "build_pipeline": ("captionqa.captioning.config", "build_pipeline"),
    "CaptioningPipeline": ("captionqa.captioning.pipeline", "CaptioningPipeline"),
    "generate_captions": ("captionqa.captioning.pipeline", "generate_captions"),
}


if TYPE_CHECKING:  # pragma: no cover - aids static analysis only
    from .config import CaptioningConfig, build_pipeline
    from .pipeline import CaptioningPipeline, generate_captions


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'captionqa.captioning' has no attribute '{name}'")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr
