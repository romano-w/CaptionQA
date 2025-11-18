"""Slice clip-level caption summaries into QA-span snippets."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .baseline_vqa import _summary_lookup_keys

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


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


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> List[str]:
    normalized = _clean_text(text)
    if not normalized:
        return []
    sentences = [sent.strip() for sent in SENTENCE_SPLIT_RE.split(normalized) if sent.strip()]
    if sentences:
        return sentences
    return [normalized]


def _extract_window(row: Mapping[str, object]) -> Tuple[Optional[float], Optional[float]]:
    for start_key, end_key in (("start", "end"), ("start_time", "end_time"), ("s", "e")):
        if start_key in row and end_key in row:
            try:
                return float(row[start_key]), float(row[end_key])
            except (TypeError, ValueError):
                continue
    return None, None


def _center_ratio(start: Optional[float], end: Optional[float], clip_duration: Optional[float]) -> Optional[float]:
    if start is None or end is None or clip_duration is None or clip_duration <= 0:
        return None
    midpoint = (start + end) / 2.0
    ratio = midpoint / clip_duration
    if math.isnan(ratio) or math.isinf(ratio):
        return None
    return max(0.0, min(1.0, ratio))


def _select_snippet(
    sentences: Sequence[str], ratio: Optional[float], max_sentences: int
) -> Tuple[str, int, int]:
    if max_sentences <= 0:
        max_sentences = 1
    total = len(sentences)
    if total == 0:
        return "", 0, 0
    if total <= max_sentences:
        snippet = " ".join(sentences)
        return snippet.strip(), 0, total

    if ratio is None:
        start_idx = 0
    else:
        idx = int(round(ratio * (total - 1)))
        start_idx = max(0, min(idx, total - max_sentences))
    end_idx = start_idx + max_sentences
    snippet = " ".join(sentences[start_idx:end_idx]).strip()
    return snippet, start_idx, end_idx - start_idx


@dataclass
class QAExample:
    """Representation of a QA example and its candidate summary keys."""

    example_id: str
    video: str
    question: str
    start: Optional[float]
    end: Optional[float]
    candidates: List[str]


def _prepare_examples(rows: Sequence[Mapping[str, object]]) -> Tuple[List[QAExample], Dict[str, float]]:
    examples: List[QAExample] = []
    clip_durations: Dict[str, float] = {}
    for row in rows:
        ex_id = str(row.get("id", "")).strip()
        video = str(row.get("video", "")).strip()
        question = str(row.get("question", "")).strip()
        start, end = _extract_window(row)
        candidates = _summary_lookup_keys(video, ex_id)
        if end is not None:
            for key in candidates:
                if not key:
                    continue
                clip_durations[key] = max(clip_durations.get(key, 0.0), float(end))
        examples.append(
            QAExample(
                example_id=ex_id or str(len(examples)),
                video=video,
                question=question,
                start=start,
                end=end,
                candidates=[key for key in candidates if key],
            )
        )
    return examples, clip_durations


def _load_caption_sentences(rows: Sequence[Mapping[str, object]]) -> Dict[str, List[str]]:
    sentences: Dict[str, List[str]] = {}
    for row in rows:
        key = str(row.get("id", "")).strip()
        if not key:
            continue
        text = row.get("prediction") or row.get("summary") or row.get("text")
        if not text:
            continue
        parsed = _split_sentences(str(text))
        if parsed:
            sentences[key] = parsed
    return sentences


def slice_summaries(
    manifest_path: Path,
    captions_path: Path,
    output_path: Path,
    *,
    max_sentences: int = 2,
) -> Mapping[str, int]:
    manifest_rows = _read_json_or_jsonl(manifest_path)
    caption_rows = _read_json_or_jsonl(captions_path)
    examples, clip_durations = _prepare_examples(manifest_rows)
    caption_sentences = _load_caption_sentences(caption_rows)

    stats = {"total": len(examples), "written": 0, "missing_summary": 0, "empty_snippet": 0}
    outputs = []
    for ex in examples:
        summary_key = None
        for cand in ex.candidates:
            if cand in caption_sentences:
                summary_key = cand
                break
        if summary_key is None:
            stats["missing_summary"] += 1
            continue
        sentences = caption_sentences[summary_key]
        ratio = _center_ratio(ex.start, ex.end, clip_durations.get(summary_key))
        snippet, start_idx, sentence_count = _select_snippet(sentences, ratio, max_sentences)
        if not snippet:
            stats["empty_snippet"] += 1
            continue
        outputs.append(
            {
                "id": ex.example_id,
                "summary": snippet,
                "question": ex.question,
                "video": ex.video,
                "start": ex.start,
                "end": ex.end,
                "source_id": summary_key,
                "total_sentences": len(sentences),
                "selected_index": start_idx,
                "selected_count": sentence_count,
            }
        )
        stats["written"] += 1

    _write_jsonl(output_path, outputs)
    return stats


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice caption summaries to QA span snippets.")
    parser.add_argument("--manifest", type=Path, required=True, help="QA manifest JSON/JSONL with {id, video, start, end}.")
    parser.add_argument(
        "--captions",
        type=Path,
        required=True,
        help="Caption predictions JSON/JSONL with {id, prediction} entries (clip-level).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path where per-question summaries will be written.",
    )
    parser.add_argument("--max-sentences", type=int, default=2, help="Maximum sentences to keep per QA example.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    stats = slice_summaries(args.manifest, args.captions, args.output, max_sentences=args.max_sentences)
    print(
        f"Wrote {stats['written']} per-question summaries "
        f"(missing={stats['missing_summary']}, empty={stats['empty_snippet']}) to {args.output}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
