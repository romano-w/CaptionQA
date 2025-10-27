"""Command line interface for batch evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .datasets import CaptionDatasetConfig, QADatasetConfig
from .metrics import MetricResult, compute_caption_metrics, compute_qa_metrics


def _read_json_or_jsonl(path: Path) -> Iterable[Mapping[str, object]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"Unsupported JSON structure in {path}")
    except json.JSONDecodeError:
        # Fallback to JSON Lines
        records: List[Mapping[str, object]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
        return records


def _load_predictions(path: Path) -> Mapping[str, str]:
    predictions: MutableMapping[str, str] = {}
    for entry in _read_json_or_jsonl(path):
        if "id" not in entry or "prediction" not in entry:
            raise KeyError("Prediction entries must contain 'id' and 'prediction'.")
        predictions[str(entry["id"])] = str(entry["prediction"])
    return predictions


def _load_references(path: Path, value_field: str) -> Mapping[str, Sequence[str]]:
    references: MutableMapping[str, List[str]] = {}
    for entry in _read_json_or_jsonl(path):
        if "id" not in entry or value_field not in entry:
            raise KeyError(f"Reference entries must contain 'id' and '{value_field}'.")
        values = entry[value_field]
        if isinstance(values, str):
            references[str(entry["id"])] = [values]
        elif isinstance(values, Iterable):
            references[str(entry["id"])] = [str(v) for v in values]
        else:
            references[str(entry["id"])] = [str(values)]
    return references


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CaptionQA predictions")
    parser.add_argument("--task", choices=["captioning", "qa"], required=True)
    parser.add_argument("--preds", type=Path, required=True, help="Path to predictions JSON/JSONL file")
    parser.add_argument("--refs", type=Path, help="Path to references JSON/JSONL file")
    parser.add_argument("--value-field", default=None, help="Key name that stores reference text")
    parser.add_argument("--dataset-name", help="Optional Hugging Face dataset name")
    parser.add_argument("--dataset-config", help="Optional dataset configuration name")
    parser.add_argument("--split", default="validation", help="Dataset split (default: validation)")
    parser.add_argument("--id-column", default="id", help="Dataset column for example ids")
    parser.add_argument("--reference-column", default="references", help="Column containing captions/answers")
    parser.add_argument("--output-json", type=Path, help="Where to store the evaluation summary")
    return parser


def _resolve_references(args: argparse.Namespace, task: str) -> Mapping[str, Sequence[str]]:
    if args.refs:
        value_field = args.value_field or ("references" if task == "captioning" else "answers")
        return _load_references(Path(args.refs), value_field)

    if not args.dataset_name:
        raise ValueError("Either --refs or --dataset-name must be provided")

    if task == "captioning":
        dataset_cfg = CaptionDatasetConfig(
            name=args.dataset_name,
            split=args.split,
            config=args.dataset_config,
            id_column=args.id_column,
            reference_column=args.reference_column,
        )
        return dataset_cfg.load()

    dataset_cfg = QADatasetConfig(
        name=args.dataset_name,
        split=args.split,
        config=args.dataset_config,
        id_column=args.id_column,
        answer_column=args.reference_column,
    )
    return dataset_cfg.load()


def _serialize_metrics(metrics: Mapping[str, MetricResult]) -> Dict[str, Dict[str, object]]:
    return {name: metric.as_dict() for name, metric in metrics.items()}


def main(argv: Sequence[str] | None = None) -> Dict[str, object]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    predictions = _load_predictions(Path(args.preds))
    references = _resolve_references(args, args.task)

    if args.task == "captioning":
        metrics = compute_caption_metrics(predictions, references)
    else:
        metrics = compute_qa_metrics(predictions, references)

    summary: Dict[str, object] = {
        "task": args.task,
        "num_examples": len(predictions),
        "metrics": _serialize_metrics(metrics),
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
