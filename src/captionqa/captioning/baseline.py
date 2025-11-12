"""Baseline captioning runner for dev-mini subsets (360x-friendly).

Generates predictions over a small manifest of local videos using the selected
engine (fusion or qwen_vl), writes preds.jsonl, and optionally evaluates against
references via the existing evaluator CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional

from .pipeline import generate_captions
from .config import CaptioningConfig


def _read_json_or_jsonl(path: Path) -> List[Mapping[str, object]]:
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
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_manifest_from_root(root: Path, pattern: str, limit: Optional[int]) -> List[Mapping[str, object]]:
    videos: List[Mapping[str, object]] = []
    count = 0
    for path in sorted(root.rglob(pattern)):
        if limit is not None and count >= limit:
            break
        if path.is_file():
            videos.append({"id": path.stem, "video": str(path)})
            count += 1
    return videos


def run(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run baseline captioning over a dev-mini manifest")
    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--manifest", type=Path, help="JSON/JSONL file with {id, video, [references]} entries")
    gsrc.add_argument("--root", type=Path, help="Root directory to scan for videos")
    p.add_argument("--glob", default="**/*.mp4", help="Glob under --root (default: **/*.mp4)")
    p.add_argument("--limit", type=int, default=32, help="Max examples to process when scanning a root")

    p.add_argument("--engine", choices=["fusion", "qwen_vl"], default="fusion")
    p.add_argument("--config", type=Path, default=None, help="Optional captioning JSON config (merged)")
    p.add_argument("--output-dir", type=Path, default=Path("data/eval/captioning/devmini"))

    # Evaluation options
    p.add_argument("--refs", type=Path, default=None, help="References JSON/JSONL file (id, references)")
    p.add_argument("--dataset-name", default=None, help="HF dataset name for evaluation (if --refs not provided)")
    p.add_argument("--split", default="validation")
    p.add_argument("--id-column", default="id")
    p.add_argument("--reference-column", default="references")
    args = p.parse_args(list(argv) if argv is not None else None)

    # Load or build manifest
    if args.manifest:
        manifest = _read_json_or_jsonl(args.manifest)
    else:
        manifest = _build_manifest_from_root(args.root, args.glob, args.limit)
        (args.output_dir / "manifest.jsonl").parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(args.output_dir / "manifest.jsonl", manifest)

    # Prepare captioning config
    cfg = CaptioningConfig.from_defaults()
    cfg.engine = args.engine
    if args.config is not None:
        try:
            from .cli import load_config  # reuse merging logic
        except Exception:
            load_config = None
        if load_config is not None:
            cfg = load_config(args.config)
            cfg.engine = args.engine or getattr(cfg, "engine", "fusion")

    preds_path = args.output_dir / "preds.jsonl"
    preds: List[Mapping[str, object]] = []
    for ex in manifest:
        video = str(ex.get("video"))
        ex_id = str(ex.get("id"))
        # Optional temporal window support in manifest
        start_end = None
        for key_pair in (("start", "end"), ("start_time", "end_time"), ("s", "e")):
            if key_pair[0] in ex and key_pair[1] in ex:
                try:
                    start_end = (float(ex[key_pair[0]]), float(ex[key_pair[1]]))
                except Exception:
                    start_end = None
                break
        try:
            caption = generate_captions(video, config=cfg, temporal_window=start_end)
        except Exception as exc:  # be resilient in baseline runs
            caption = f"[error generating caption: {exc}]"
        preds.append({"id": ex_id, "prediction": caption})
    _write_jsonl(preds_path, preds)

    # If we have references, evaluate
    summary = None
    if args.refs and args.refs.exists():
        from ..evaluation.run import main as eval_main

        summary = eval_main(
            [
                "--task",
                "captioning",
                "--preds",
                str(preds_path),
                "--refs",
                str(args.refs),
                "--output-json",
                str(args.output_dir / "summary.json"),
            ]
        )
    elif args.dataset_name:
        from ..evaluation.run import main as eval_main

        summary = eval_main(
            [
                "--task",
                "captioning",
                "--preds",
                str(preds_path),
                "--dataset-name",
                args.dataset_name,
                "--split",
                args.split,
                "--id-column",
                args.id_column,
                "--reference-column",
                args.reference_column,
                "--output-json",
                str(args.output_dir / "summary.json"),
            ]
        )
    else:
        print("No references provided; skipping evaluation.", file=sys.stderr)

    if summary is not None:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({"task": "captioning", "num_examples": len(preds)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
