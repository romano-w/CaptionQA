"""Baseline captioning runner for dev-mini subsets (360x-friendly).

Generates predictions over a small manifest of local videos using the selected
engine (fusion or qwen_vl), writes preds.jsonl, and optionally evaluates against
references via the existing evaluator CLI.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional

from .pipeline import generate_captions
from .config import CaptioningConfig
from ..utils.progress import ProgressDisplay


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


def _run_evaluator(argv: List[str]) -> Mapping[str, object]:
    from ..evaluation.run import main as eval_main

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        summary = eval_main(argv)
    return summary


def _print_metric_summary(summary: Mapping[str, object], output_dir: Path) -> None:
    metrics = summary.get("metrics", {})
    if not metrics:
        print("No metrics returned.")
        return
    print("\nEvaluation Summary")
    for name, metric in metrics.items():
        score = metric.get("score")
        if isinstance(score, (int, float)):
            print(f"  - {name.upper():<8}: {score:.4f}")
        else:
            print(f"  - {name.upper():<8}: {score}")
    print(f"Metrics saved to {output_dir / 'summary.json'}")


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

    # Qwen-VL decoding + prompt controls
    p.add_argument("--qwen-temperature", type=float, default=None, help="Sampling temperature for Qwen-VL runs")
    p.add_argument("--qwen-top-p", type=float, default=None, help="Top-p nucleus sampling for Qwen-VL runs")
    p.add_argument("--qwen-max-new-tokens", type=int, default=None, help="Max new tokens for Qwen-VL runs")
    p.add_argument(
        "--action-heavy-prompt",
        action="store_true",
        help="Switch to an action-focused captioning prompt for Qwen-VL runs",
    )

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

    total = len(manifest)
    manifest_source = args.manifest if args.manifest else args.root
    print(
        f"Loaded manifest with {total} example(s) "
        f"from {manifest_source} -> writing preds to {args.output_dir}",
        flush=True,
    )
    if total == 0:
        print("Manifest is empty; nothing to process.", file=sys.stderr)

    progress = ProgressDisplay(total or 1)
    progress.show_status(0, "Starting caption generation...")

    # Prepare captioning config
    cfg = CaptioningConfig.from_defaults()
    cfg.engine = args.engine
    if cfg.engine == "qwen_vl":
        combos = max(int(cfg.panorama.num_views) * max(int(cfg.panorama.num_pitch), 1), 1)
        base_frames_needed = math.ceil(max(int(cfg.qwen_vl.num_frames or 1), 1) / combos)
        if base_frames_needed <= 0:
            base_frames_needed = 1
        if getattr(cfg.panorama, "max_total_frames", None) is None:
            cfg.panorama.max_total_frames = base_frames_needed
        if args.qwen_temperature is not None:
            cfg.qwen_vl.temperature = float(args.qwen_temperature)
        if args.qwen_top_p is not None:
            cfg.qwen_vl.top_p = float(args.qwen_top_p)
        if args.qwen_max_new_tokens is not None:
            cfg.qwen_vl.max_new_tokens = int(args.qwen_max_new_tokens)
        if args.action_heavy_prompt:
            cfg.qwen_vl.caption_template = cfg.qwen_vl.action_caption_template
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
    for idx, ex in enumerate(manifest, start=1):
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
        window_msg = ""
        if start_end is not None:
            window_msg = f"window {start_end[0]:.2f}-{start_end[1]:.2f}s"
        progress.show_status(idx - 1, f"Processing {ex_id} {window_msg}".strip())
        start_time = time.perf_counter()
        try:
            caption = generate_captions(video, config=cfg, temporal_window=start_end)
        except Exception as exc:  # be resilient in baseline runs
            caption = f"[error generating caption: {exc}]"
            progress.finish_step(idx, f"{ex_id} ERROR: {exc}")
        else:
            elapsed = time.perf_counter() - start_time
            progress.finish_step(idx, f"{ex_id} done in {elapsed:.1f}s")
        preds.append({"id": ex_id, "prediction": caption})
    _write_jsonl(preds_path, preds)

    # If we have references, evaluate
    summary = None
    if args.refs and args.refs.exists():
        print(f"Evaluating against references in {args.refs}...", flush=True)
        summary = _run_evaluator(
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
        print(
            "Evaluating via HF dataset "
            f"{args.dataset_name} (split={args.split}, column={args.reference_column})...",
            flush=True,
        )
        summary = _run_evaluator(
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
        _print_metric_summary(summary, args.output_dir)
    else:
        print(f"Generated {len(preds)} prediction(s); results saved to {preds_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
