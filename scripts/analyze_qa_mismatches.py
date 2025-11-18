#!/usr/bin/env python3
"""Analyze QA prediction mismatches grouped by TAL label."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from captionqa.qa.normalization import TAL_LABELS  # noqa: E402

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "at",
    "between",
    "from",
    "second",
    "seconds",
    "person",
    "people",
    "video",
    "camera",
    "scene",
    "action",
    "what",
    "that",
    "this",
    "it",
    "during",
    "time",
    "frame",
    "frames",
}
MAX_REGEX_WORDS = 5


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
        raise ValueError(f"Unsupported JSON in {path}")
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _build_refs(path: Path) -> Dict[str, List[str]]:
    refs: Dict[str, List[str]] = {}
    for row in _read_json_or_jsonl(path):
        ex_id = str(row.get("id", "")).strip()
        if not ex_id:
            continue
        answers = [str(ans).strip().lower() for ans in row.get("answers", []) if str(ans).strip()]
        if answers:
            refs[ex_id] = answers
    return refs


def _normalize_label(text: str) -> str:
    return text.strip().lower()


def _collect_counts(
    preds_path: Path,
    refs: Mapping[str, List[str]],
    *,
    include_matches: bool = False,
) -> tuple[MutableMapping[str, Counter], Counter, List[str]]:
    counts: MutableMapping[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    missing_preds: List[str] = []
    for ex_id, answers in refs.items():
        if answers:
            totals[_normalize_label(answers[0])] += 1

    ref_ids = set(refs.keys())
    seen_ids = set()
    for row in _read_json_or_jsonl(preds_path):
        ex_id = str(row.get("id", "")).strip()
        if not ex_id or ex_id not in refs:
            continue
        seen_ids.add(ex_id)
        answers = refs.get(ex_id, [])
        if not answers:
            continue
        ref_label = _normalize_label(answers[0])
        predicted = str(row.get("prediction") or "").strip()
        raw = str(row.get("raw_prediction") or predicted).strip()
        if not raw:
            continue
        normalized_prediction = predicted.lower()
        if not include_matches and normalized_prediction == ref_label:
            continue
        counts[ref_label][raw] += 1

    missing_preds = sorted(ref_ids - seen_ids)
    return counts, totals, missing_preds


def _ordered_labels(totals: Mapping[str, int], requested: Sequence[str] | None) -> List[str]:
    if requested:
        ordered: List[str] = []
        seen: set[str] = set()
        for label in (_normalize_label(lbl) for lbl in requested):
            if label not in seen:
                ordered.append(label)
                seen.add(label)
        return ordered
    present = [label for label in TAL_LABELS if totals.get(label)]
    extras = sorted(label for label in totals if label not in TAL_LABELS and totals.get(label))
    return present + extras


def _regex_hint(phrase: str, min_words: int) -> str:
    tokens = re.findall(r"[a-z0-9]+", phrase.lower())
    if not tokens:
        return ""
    filtered = [tok for tok in tokens if tok not in STOPWORDS]
    if len(filtered) < min_words:
        filtered = tokens
    filtered = filtered[:MAX_REGEX_WORDS]
    if not filtered:
        return ""
    pattern = r"\b" + r"\s+".join(filtered) + r"\b"
    return pattern


def _export_csv(path: Path, counts: Mapping[str, Counter], top_k: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "prediction", "count"])
        for label in sorted(counts):
            bucket = counts[label]
            rows = bucket.most_common(top_k) if top_k and top_k > 0 else bucket.items()
            for phrase, freq in rows:
                writer.writerow([label, phrase, freq])


def _print_report(
    counts: Mapping[str, Counter],
    totals: Mapping[str, int],
    *,
    top_k: int,
    include_matches: bool,
    requested: Sequence[str] | None,
    suggest_regex: bool,
    regex_min_words: int,
) -> None:
    labels = _ordered_labels(totals, requested)
    if not labels:
        print("No predictions matched the requested filters.")
        return
    for label in labels:
        total = totals.get(label, 0)
        bucket = counts.get(label, Counter())
        bucket_total = sum(bucket.values())
        if total == 0 and bucket_total == 0:
            continue
        header = (
            f"\nLabel: {label or '<unknown>'} "
            f"(references={total}, {'examples' if include_matches else 'mismatches'}={bucket_total})"
        )
        print(header)
        if not bucket_total:
            print("  (no rows)")
            continue
        for idx, (phrase, freq) in enumerate(bucket.most_common(top_k), start=1):
            print(f"  {idx:>2}. {phrase} — {freq}x")
            if suggest_regex:
                hint = _regex_hint(phrase, regex_min_words)
                if hint:
                    print(f"       regex hint: r\"{hint}\"")


def analyze(
    preds: Path,
    refs: Path,
    *,
    top_k: int,
    include_matches: bool,
    labels: Sequence[str] | None,
    export_csv: Path | None,
    csv_top_k: int | None,
    suggest_regex: bool,
    regex_min_words: int,
) -> None:
    ref_map = _build_refs(refs)
    counts, totals, missing = _collect_counts(preds, ref_map, include_matches=include_matches)
    if labels:
        filtered_counts: Dict[str, Counter] = {}
        filtered_totals: Counter = Counter()
        for label in labels:
            key = _normalize_label(label)
            filtered_counts[key] = counts.get(key, Counter())
            filtered_totals[key] = totals.get(key, 0)
        counts = filtered_counts
        totals = filtered_totals
    if export_csv:
        _export_csv(export_csv, counts, csv_top_k)
        print(f"[info] CSV report written to {export_csv}")
    if missing:
        print(f"[info] {len(missing)} reference id(s) lacked predictions (examples: {', '.join(missing[:5])})")
    _print_report(
        counts,
        totals,
        top_k=top_k,
        include_matches=include_matches,
        requested=labels,
        suggest_regex=suggest_regex,
        regex_min_words=regex_min_words,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Surface mismatched QA predictions per TAL label.")
    parser.add_argument("--preds", type=Path, required=True, help="preds.jsonl from baseline_vqa")
    parser.add_argument("--refs", type=Path, required=True, help="refs.jsonl with {id, answers}")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of phrases to list per label (default: 10).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional subset of TAL labels to print (default: all present labels).",
    )
    parser.add_argument(
        "--include-matches",
        action="store_true",
        help="Include correctly-normalized predictions instead of filtering to mismatches only.",
    )
    parser.add_argument("--export-csv", type=Path, help="Optional CSV path for downstream regex editing.")
    parser.add_argument(
        "--csv-top-k",
        type=int,
        default=None,
        help="Limit rows per label when writing CSV (default: all mismatches).",
    )
    parser.add_argument(
        "--suggest-regex",
        action="store_true",
        help="Print naive regex hints (space-delimited tokens) for each phrase.",
    )
    parser.add_argument(
        "--regex-min-words",
        type=int,
        default=2,
        help="Minimum keywords to keep before falling back to all tokens (default: 2).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    analyze(
        args.preds,
        args.refs,
        top_k=max(1, args.top_k),
        include_matches=args.include_matches,
        labels=args.labels,
        export_csv=args.export_csv,
        csv_top_k=args.csv_top_k,
        suggest_regex=args.suggest_regex,
        regex_min_words=max(1, args.regex_min_words),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
