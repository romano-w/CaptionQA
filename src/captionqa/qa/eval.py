"""Evaluation utilities for AVQA models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .datasets import AVQADataset
from .model import AVQAModel, decode_answers
from ..captioning.encoders import (
    VisualEncoder,
    VisualEncoderConfig,
    AudioEncoder,
    AudioEncoderConfig,
)
from ..captioning.panorama import PanoramicFrameSampler, PanoramaSamplingConfig


@dataclass
class EvaluationResult:
    question_id: str
    question: str
    prediction: str
    reference: Optional[str]
    confidence: Optional[float] = None
    temporal_window: Optional[Tuple[float, float]] = None


def load_avqa_subset(
    dataset_root: Path,
    *,
    split: str = "val",
    subset_size: int = 8,
) -> AVQADataset:
    """Load a small AVQA subset for quick evaluations."""

    dataset = AVQADataset(dataset_root, split=split)
    if subset_size < len(dataset):
        dataset.samples = dataset.samples[:subset_size]  # type: ignore[attr-defined]
    return dataset


def run_zero_shot(
    model: AVQAModel,
    dataset: AVQADataset,
    tokenizer,
    *,
    device: str = "cpu",
    max_length: int = 16,
) -> List[EvaluationResult]:
    """Run zero-shot inference on the provided dataset subset."""

    model = model.to(device)
    model.eval()

    # Build lightweight captioning feature pipeline (shared encoders)
    sampler = PanoramicFrameSampler(PanoramaSamplingConfig())
    venc = VisualEncoder(VisualEncoderConfig())
    aenc = AudioEncoder(AudioEncoderConfig())

    results: List[EvaluationResult] = []
    for sample in dataset:
        # Question tokens
        encoded = tokenizer.encode(sample["question"])  # type: ignore[assignment]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        question_tokens = torch.tensor([encoded], device=device, dtype=torch.long)
        # Visual features from panoramic sampler
        video_path = str(sample["video"])  # type: ignore[index]
        start_end = None
        tw = sample.get("temporal_window")  # type: ignore[index]
        if isinstance(tw, (list, tuple)) and len(tw) == 2:
            try:
                start_end = (float(tw[0]), float(tw[1]))
            except Exception:
                start_end = None

        frames = sampler.sample(video_path, start_sec=(start_end[0] if start_end else None), end_sec=(start_end[1] if start_end else None))
        # Stable cache keys
        v_key = hashlib.sha1(
            (
                f"{video_path}|vis:{venc.config.model_name}|fps:{sampler.config.frame_rate}|"
                f"proj:{sampler.config.enable_projection}|views:{sampler.config.num_views}|"
                f"res:{sampler.config.target_resolution}|pitch:{sampler.config.num_pitch}"
            ).encode("utf-8")
        ).hexdigest()
        vfeat = venc.encode(frames, cache_key=v_key).to(device)
        if vfeat.dim() == 1:
            vfeat = vfeat.unsqueeze(0)
        batch_video = vfeat.unsqueeze(0)  # (1, T, D)

        # Audio features from audio path (or video if audio missing)
        audio_path = str(sample.get("audio") or sample["video"])  # type: ignore[index]
        a_key = hashlib.sha1(
            (f"{audio_path}|aud:{aenc.config.model_name}|sr:{aenc.config.sample_rate}").encode("utf-8")
        ).hexdigest()
        afeat = aenc.encode(audio_path, cache_key=a_key, start_sec=(start_end[0] if start_end else None), end_sec=(start_end[1] if start_end else None)).to(device)
        if afeat.dim() == 1:
            afeat = afeat.unsqueeze(0)
        if afeat.dim() == 2:
            afeat = afeat.unsqueeze(1)  # (B, 1, D)
        batch_audio = afeat  # (1, S, D)

        # Generate answer; prefer confidence-bearing path if available
        confidence: Optional[float] = None
        try:
            generated, probs = model.generate_with_confidence(
                batch_video, batch_audio, question_tokens, max_length=max_length
            )
            if probs is not None and probs.numel() > 0:
                confidence = float(probs.mean().item())
        except AttributeError:
            generated = model.generate(batch_video, batch_audio, question_tokens, max_length=max_length)
        decoded = decode_answers(tokenizer, generated)[0]
        results.append(
            EvaluationResult(
                question_id=sample["question_id"],
                question=sample["question"],
                prediction=decoded.text,
                reference=sample.get("answer"),
                confidence=confidence,
                temporal_window=start_end,
            )
        )
    return results


def run_fine_tuned(
    checkpoint: Path,
    dataset: AVQADataset,
    tokenizer,
    *,
    device: str = "cpu",
    max_length: int = 16,
) -> List[EvaluationResult]:
    """Load a fine-tuned checkpoint and evaluate it on the subset."""

    state = torch.load(checkpoint, map_location=device)
    config = state.get("config", {})
    model = AVQAModel(**config)
    model.load_state_dict(state["model"])
    return run_zero_shot(model, dataset, tokenizer, device=device, max_length=max_length)


__all__ = [
    "EvaluationResult",
    "load_avqa_subset",
    "run_zero_shot",
    "run_fine_tuned",
]

