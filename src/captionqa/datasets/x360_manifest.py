"""Utilities for building and enriching 360-degree video manifests."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class ManifestRow:
    """Lightweight record representing a manifest entry."""

    id: str
    video: str

    def to_mapping(self) -> Dict[str, str]:
        return {"id": self.id, "video": self.video}


def _read_json_or_jsonl(path: Path) -> List[Mapping[str, object]]:
    """Load a JSON/JSONL file into a list of mappings."""

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # JSON lines fallback
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    if isinstance(data, list):
        return list(data)
    if isinstance(data, dict):
        return [data]

    raise ValueError(f"Unsupported JSON structure in {path}")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Persist a sequence of mappings to disk as JSON Lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_manifest(root: Path, pattern: str = "**/*.mp4", limit: Optional[int] = None) -> List[MutableMapping[str, object]]:
    """Scan ``root`` for media files and produce CaptionQA manifest rows."""

    rows: List[MutableMapping[str, object]] = []
    for candidate in sorted(root.rglob(pattern)):
        if not candidate.is_file():
            continue
        rows.append({"id": candidate.stem, "video": str(candidate)})
        if limit is not None and len(rows) >= limit:
            break
    return rows


def merge_references(
    rows: Iterable[MutableMapping[str, object]],
    refs_path: Path,
    *,
    id_field: str = "id",
    references_field: str = "references",
) -> List[MutableMapping[str, object]]:
    """Attach reference captions from ``refs_path`` onto manifest ``rows``."""

    references = _read_json_or_jsonl(refs_path)
    ref_index = {
        str(entry.get(id_field)): entry.get(references_field)
        for entry in references
        if id_field in entry and references_field in entry
    }

    merged: List[MutableMapping[str, object]] = []
    for row in rows:
        row_copy = dict(row)
        ref_value = ref_index.get(str(row_copy.get(id_field)))
        if ref_value is not None:
            row_copy[references_field] = ref_value
        merged.append(row_copy)
    return merged


__all__ = [
    "ManifestRow",
    "_read_json_or_jsonl",
    "_write_jsonl",
    "build_manifest",
    "merge_references",
]


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a manifest JSONL by scanning a video root.")
    parser.add_argument("--root", type=Path, required=True, help="Directory that contains media files.")
    parser.add_argument(
        "--glob",
        default="**/*.mp4",
        help="Glob pattern (relative to --root) for selecting files (default: **/*.mp4).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of files to emit.",
    )
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help="If set, rewrite each video path relative to this directory.",
    )
    parser.add_argument(
        "--relative-prefix",
        type=Path,
        default=None,
        help="Optional prefix prepended to paths rewritten via --relative-to (e.g., data/raw).",
    )
    parser.add_argument(
        "--id-template",
        default="{stem}",
        help=(
            "Template used to derive manifest IDs. "
            "Available keys: {stem}, {name}, {parent}, {parent_name}, {relative}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL for the manifest rows.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = build_manifest(args.root, pattern=args.glob, limit=args.limit)
    processed: List[MutableMapping[str, object]] = []
    relative_base = args.relative_to.resolve() if args.relative_to else None
    prefix = args.relative_prefix

    for row in rows:
        video_path = Path(str(row["video"])).resolve()
        relative_path: Optional[Path] = None
        if relative_base is not None:
            try:
                relative_path = video_path.relative_to(relative_base)
            except ValueError:
                relative_path = None
        if relative_path is not None:
            if prefix:
                relative_path = Path(prefix) / relative_path
            row["video"] = relative_path.as_posix()
            relative_value = row["video"]
        else:
            row["video"] = video_path.as_posix()
            relative_value = row["video"]

        context = {
            "stem": video_path.stem,
            "name": video_path.name,
            "parent": video_path.parent.as_posix(),
            "parent_name": video_path.parent.name,
            "relative": relative_value,
        }
        try:
            row["id"] = args.id_template.format(**context)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Failed to render id template '{args.id_template}': {exc}") from exc
        processed.append(row)

    _write_jsonl(args.output, processed)
    print(f"Wrote {len(processed)} rows -> {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
