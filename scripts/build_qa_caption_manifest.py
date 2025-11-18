#!/usr/bin/env python3
"""Export QA manifest entries for per-window captioning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


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


def build_manifest(
    source: Path,
    output: Path,
    *,
    limit: int | None,
    ids: Sequence[str] | None,
) -> int:
    rows = _read_json_or_jsonl(source)
    keep: List[Mapping[str, object]] = []
    id_set = {id_.strip() for id_ in ids} if ids else None
    for row in rows:
        if id_set and str(row.get("id")) not in id_set:
            continue
        keep.append(row)
        if limit is not None and limit >= 0 and len(keep) >= limit:
            break
    _write_jsonl(output, keep)
    return len(keep)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter QA manifest rows for per-window captioning.")
    parser.add_argument("--qa-manifest", type=Path, required=True, help="Source QA manifest JSON/JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL for captioning runner.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows to export.")
    parser.add_argument("--ids", nargs="+", help="Optional list of QA ids to export (overrides --limit ordering).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    count = build_manifest(args.qa_manifest, args.output, limit=args.limit, ids=args.ids)
    print(f"Wrote {count} QA entries to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
