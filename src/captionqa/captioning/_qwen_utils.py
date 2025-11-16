from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency guards
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

logger = logging.getLogger(__name__)


def _torch_version_tuple() -> tuple[int, int, int]:
    if torch is None or not getattr(torch, "__version__", None):
        return (0, 0, 0)
    raw = str(torch.__version__).split("+")[0]
    parts = raw.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    patch_str = parts[2] if len(parts) > 2 else "0"
    patch_digits = "".join(ch for ch in patch_str if ch.isdigit())
    patch = int(patch_digits) if patch_digits else 0
    return (major, minor, patch)


def sanitize_qwen_config(config: Any) -> Any:
    """Drop tensor-parallel metadata for older torch builds."""

    if config is None:
        return None
    if _torch_version_tuple() >= (2, 5, 0):
        return config

    def _strip(obj: Any) -> None:
        if obj is None:
            return
        if hasattr(obj, "tensor_parallel_config"):
            setattr(obj, "tensor_parallel_config", None)
        if hasattr(obj, "base_model_tp_plan"):
            setattr(obj, "base_model_tp_plan", None)
        if hasattr(obj, "quantization_config"):
            quant_config = getattr(obj, "quantization_config")
            if isinstance(quant_config, dict):
                quant_config.pop("tensor_parallel_config", None)
        for attr in (
            "text_config",
            "vision_config",
            "language_model",
            "language_config",
            "model",
        ):
            child = getattr(obj, attr, None)
            if child is not None and child is not obj:
                _strip(child)

    _strip(config)
    return config


def prepare_video_payload(frames: List[np.ndarray], limit: Optional[int]) -> Optional[np.ndarray]:
    """Convert sampled frames into a stacked video tensor acceptable by Qwen processors."""

    if not frames:
        return None
    max_frames = max(int(limit) if limit else len(frames), 1)
    selected: List[np.ndarray] = []
    for frame in frames:
        if len(selected) >= max_frames:
            break
        if frame is None:
            continue
        arr = np.asarray(frame)
        if arr.ndim < 3:
            continue
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        selected.append(arr)

    if not selected:
        return None

    min_h = min(f.shape[0] for f in selected)
    min_w = min(f.shape[1] for f in selected)
    min_c = min(f.shape[2] for f in selected)
    processed = [f[:min_h, :min_w, :min_c] for f in selected]
    try:
        return np.stack(processed, axis=0)
    except ValueError as exc:  # pragma: no cover - defensive
        logger.warning("Unable to stack %s frames for Qwen input: %s", len(processed), exc)
        return None


__all__ = ["prepare_video_payload", "sanitize_qwen_config"]
