"""Helpers to summarize AVQA evaluation results with accuracy/F1."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple, Dict

from .eval import EvaluationResult
from ..evaluation.metrics import compute_qa_metrics


def _to_maps(results: Iterable[EvaluationResult]) -> Tuple[Dict[str, str], Dict[str, Sequence[str]]]:
    preds: Dict[str, str] = {}
    refs: Dict[str, Sequence[str]] = {}
    for r in results:
        preds[r.question_id] = r.prediction
        if r.reference is not None:
            refs[r.question_id] = [r.reference]
        else:
            refs[r.question_id] = [""]
    return preds, refs


def summarize_results(results: Iterable[EvaluationResult]) -> Mapping[str, float]:
    """Compute QA accuracy and F1 from a list of EvaluationResult.

    Returns a mapping with keys ``accuracy`` and ``f1``.
    """

    preds, refs = _to_maps(results)
    metrics = compute_qa_metrics(preds, refs)
    return {name: metric.score for name, metric in metrics.items()}


__all__ = ["summarize_results"]

