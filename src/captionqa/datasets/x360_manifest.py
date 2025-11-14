"""Utilities for building and enriching 360-degree video manifests."""

from __future__ import annotations

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
