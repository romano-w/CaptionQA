"""Generate QA manifests from 360+x TAL annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


def _read_jsonl(path: Path) -> List[Mapping[str, object]]:
    rows: List[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_metadata(cache: Dict[str, Mapping[str, object]], uuid: str, root: Path) -> Mapping[str, object]:
    if uuid in cache:
        return cache[uuid]
    ann_path = root / f"{uuid}.json"
    if not ann_path.exists():
        cache[uuid] = {}
        return {}
    with ann_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    metadata = data.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    cache[uuid] = metadata
    return metadata


def _iter_segments(metadata: Mapping[str, Mapping[str, object]]) -> Iterable[Tuple[str, Mapping[str, object]]]:
    pairs: List[Tuple[str, Mapping[str, object]]] = []
    for seg_id, meta in metadata.items():
        if not isinstance(meta, Mapping):
            continue
        pairs.append((seg_id, meta))
    def _start_time(meta: Mapping[str, object]) -> float:
        duration = meta.get("duration")
        if isinstance(duration, (list, tuple)) and len(duration) == 2:
            try:
                return float(duration[0])
            except (TypeError, ValueError):
                return float("inf")
        return float("inf")
    pairs.sort(key=lambda item: _start_time(item[1]))
    for item in pairs:
        yield item


def _format_question(start: Optional[float], end: Optional[float]) -> str:
    if start is None or end is None:
        return "What action is occurring in this interval of the video?"
    return f"What action occurs between {start:.1f} seconds and {end:.1f} seconds?"


def _extract_times(meta: Mapping[str, object]) -> Tuple[Optional[float], Optional[float]]:
    duration = meta.get("duration")
    if isinstance(duration, (list, tuple)) and len(duration) == 2:
        try:
            start = float(duration[0])
            end = float(duration[1])
            return start, end
        except (TypeError, ValueError):
            return None, None
    return None, None


def _extract_answer(meta: Mapping[str, object]) -> Optional[str]:
    action = meta.get("action")
    if isinstance(action, Mapping):
        for val in action.values():
            if isinstance(val, str) and val.strip():
                return val.strip()
    if isinstance(action, str) and action.strip():
        return action.strip()
    return None


def build_qa(
    manifest_path: Path,
    annotations_root: Path,
    qa_manifest_path: Path,
    qa_refs_path: Path,
    max_questions_per_video: int,
) -> None:
    manifest = _read_jsonl(manifest_path)
    ann_cache: Dict[str, Mapping[str, object]] = {}
    qa_rows: List[Mapping[str, object]] = []
    ref_rows: List[Mapping[str, object]] = []

    for entry in manifest:
        video = str(entry.get("video"))
        clip_id = str(entry.get("id"))
        uuid = Path(video).parent.name
        metadata = _load_metadata(ann_cache, uuid, annotations_root)
        if not metadata:
            continue
        count = 0
        for seg_id, meta in _iter_segments(metadata):
            answer = _extract_answer(meta)
            if not answer:
                continue
            start, end = _extract_times(meta)
            qa_id = f"{clip_id}_{seg_id}"
            qa_rows.append(
                {
                    "id": qa_id,
                    "video": video,
                    "question": _format_question(start, end),
                    "start": start,
                    "end": end,
                }
            )
            ref_rows.append({"id": qa_id, "answers": [answer]})
            count += 1
            if max_questions_per_video and count >= max_questions_per_video:
                break

    qa_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    qa_refs_path.parent.mkdir(parents=True, exist_ok=True)
    with qa_manifest_path.open("w", encoding="utf-8") as fh:
        for row in qa_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with qa_refs_path.open("w", encoding="utf-8") as fh:
        for row in ref_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(qa_rows)} QA entries -> {qa_manifest_path}")
    print(f"Wrote {len(ref_rows)} reference rows -> {qa_refs_path}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate QA manifest from 360+x TAL annotations")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Caption manifest referencing local video paths",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("data/raw/360x/360x_dataset_LR/TAL_annotations"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/eval/qa/360x_devmini/manifest.jsonl"),
    )
    parser.add_argument(
        "--output-refs",
        type=Path,
        default=Path("data/eval/qa/360x_devmini/refs.jsonl"),
    )
    parser.add_argument("--max-questions-per-video", type=int, default=3)
    args = parser.parse_args(list(argv) if argv is not None else None)

    build_qa(
        manifest_path=args.manifest,
        annotations_root=args.annotations_root,
        qa_manifest_path=args.output_manifest,
        qa_refs_path=args.output_refs,
        max_questions_per_video=max(0, args.max_questions_per_video),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
