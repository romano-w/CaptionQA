"""Evaluation utilities for AVQA models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from .datasets import AVQADataset
from .model import AVQAModel, decode_answers


@dataclass
class EvaluationResult:
    question_id: str
    question: str
    prediction: str
    reference: Optional[str]


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

    results: List[EvaluationResult] = []
    for sample in dataset:
        batch_video = torch.randn(1, 4, model.fusion.video_proj.in_features, device=device)
        batch_audio = torch.randn(1, 8, model.fusion.audio_proj.in_features, device=device)
        encoded = tokenizer.encode(sample["question"])  # type: ignore[assignment]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        question_tokens = torch.tensor([encoded], device=device, dtype=torch.long)
        generated = model.generate(batch_video, batch_audio, question_tokens, max_length=max_length)
        decoded = decode_answers(tokenizer, generated)[0]
        results.append(
            EvaluationResult(
                question_id=sample["question_id"],
                question=sample["question"],
                prediction=decoded.text,
                reference=sample.get("answer"),
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

