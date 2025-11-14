"""Generate reference captions for 360+x clips from TAL annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


def _read_manifest(path: Path) -> List[Mapping[str, object]]:
    rows: List[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_annotation(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("metadata", {})


def _format_reference(metadata: Mapping[str, Mapping[str, object]]) -> str:
    if not metadata:
        return "No annotated actions provided in the TAL file."

    segments = sorted(
        metadata.items(),
        key=lambda item: float(item[1].get("duration", [0.0])[0]),
    )
    parts: List[str] = []
    for _, meta in segments:
        duration = meta.get("duration") or []
        start, end = (None, None)
        if isinstance(duration, (list, tuple)) and len(duration) == 2:
            try:
                start = float(duration[0])
                end = float(duration[1])
            except (TypeError, ValueError):
                start = end = None
        action = ""
        if isinstance(meta.get("action"), Mapping):
            values = [str(v) for v in meta["action"].values() if v]
            if values:
                action = values[0]
        action = action or "unknown action"
        if start is not None and end is not None:
            parts.append(f"{action} from {start:.1f}s to {end:.1f}s")
        else:
            parts.append(action)
    return " ; ".join(parts)


def build_references(
    manifest_path: Path,
    annotations_root: Path,
    output_path: Path,
) -> int:
    manifest = _read_manifest(manifest_path)
    annotations_cache: Dict[str, Mapping[str, object]] = {}
    missing = 0
    rows: List[Mapping[str, object]] = []

    for entry in manifest:
        video_path = Path(str(entry["video"]))
        clip_id = str(entry["id"])
        uuid = video_path.parent.name
        ann_path = annotations_root / f"{uuid}.json"
        if uuid not in annotations_cache:
            annotations_cache[uuid] = _load_annotation(ann_path)
        metadata = annotations_cache[uuid]
        if not metadata:
            missing += 1
            reference = "No annotations found for this clip."
        else:
            reference = _format_reference(metadata)
        rows.append({"id": clip_id, "references": [reference]})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(rows)} reference rows -> {output_path} "
        f"(missing annotations: {missing})"
    )
    return missing


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate refs.jsonl for 360+x clips using TAL annotations"
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Caption manifest JSONL")
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("data/raw/360x/360x_dataset_LR/TAL_annotations"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/captioning/360x_devmini/refs.jsonl"),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    build_references(args.manifest, args.annotations_root, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
