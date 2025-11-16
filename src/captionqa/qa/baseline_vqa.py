"""Baseline VQA runner using Qwen-VL engine over a dev-mini manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Iterable, List, Mapping, Optional

from .engines.qwen_vl_qa import QwenVLVQAEngine
from ..captioning.config import QwenVLConfig
from ..captioning.panorama import PanoramaSamplingConfig
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


def run(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run baseline VQA over a dev-mini manifest with Qwen-VL")
    p.add_argument("--manifest", type=Path, required=True, help="JSON/JSONL with {id, video, question}")
    p.add_argument("--refs", type=Path, required=True, help="JSON/JSONL with {id, answers: [...]}")
    p.add_argument("--output-dir", type=Path, default=Path("data/eval/qa/devmini"))
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args(list(argv) if argv is not None else None)

    manifest = _read_json_or_jsonl(args.manifest)
    total = len(manifest)
    print(
        f"Loaded QA manifest with {total} example(s) "
        f"from {args.manifest} -> writing preds to {args.output_dir}",
        flush=True,
    )
    if total == 0:
        print("Manifest is empty; nothing to process.", file=sys.stderr)
    qcfg = QwenVLConfig(model_name=args.model_name, num_frames=args.num_frames, max_new_tokens=args.max_new_tokens)
    engine = QwenVLVQAEngine.from_configs(sampler_cfg=PanoramaSamplingConfig(), qwen_cfg=qcfg)

    preds_path = args.output_dir / "preds.jsonl"
    preds: List[Mapping[str, object]] = []
    progress = ProgressDisplay(total or 1)
    progress.show_status(0, "Starting QA generation...")
    for idx, ex in enumerate(manifest, start=1):
        vid = str(ex.get("video"))
        q = str(ex.get("question", ""))
        ex_id = str(ex.get("id"))
        start_end = None
        for key_pair in (("start", "end"), ("start_time", "end_time"), ("s", "e")):
            if key_pair[0] in ex and key_pair[1] in ex:
                try:
                    start_end = (float(ex[key_pair[0]]), float(ex[key_pair[1]]))
                except Exception:
                    start_end = None
                break
        progress.show_status(idx - 1, f"Answering {ex_id}")
        start_time = time.perf_counter()
        try:
            ans = engine.answer(
                vid,
                q,
                start_sec=(start_end[0] if start_end else None),
                end_sec=(start_end[1] if start_end else None),
            )
        except Exception as exc:
            ans = f"[error: {exc}]"
            progress.finish_step(idx, f"{ex_id} ERROR: {exc}")
        else:
            elapsed = time.perf_counter() - start_time
            progress.finish_step(idx, f"{ex_id} done in {elapsed:.1f}s")
        preds.append({"id": ex_id, "prediction": ans})
    _write_jsonl(preds_path, preds)

    # Evaluate
    from ..evaluation.run import main as eval_main

    summary = eval_main(
        [
            "--task",
            "qa",
            "--preds",
            str(preds_path),
            "--refs",
            str(args.refs),
            "--value-field",
            "answers",
            "--output-json",
            str(args.output_dir / "summary.json"),
        ]
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
