"""Evaluation metrics and benchmarking utilities."""

from .datasets import CaptionDatasetConfig, QADatasetConfig, dataset_summary
from .metrics import MetricResult, compute_caption_metrics, compute_qa_metrics

__all__ = [
    "CaptionDatasetConfig",
    "QADatasetConfig",
    "dataset_summary",
    "MetricResult",
    "compute_caption_metrics",
    "compute_qa_metrics",
]
