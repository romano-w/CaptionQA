"""Metric computation utilities for captioning and QA tasks."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


@dataclass
class MetricResult:
    """Normalized container for metric outputs."""

    name: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "score": self.score, "metadata": self.metadata}


def _align_predictions(
    predictions: Mapping[str, str],
    references: Mapping[str, Sequence[str]],
) -> Dict[str, Dict[str, Any]]:
    missing_refs = sorted(set(predictions) - set(references))
    if missing_refs:
        raise KeyError(
            "Missing references for ids: " + ", ".join(missing_refs[:5]) + ("..." if len(missing_refs) > 5 else "")
        )

    missing_preds = sorted(set(references) - set(predictions))
    if missing_preds:
        raise KeyError(
            "Missing predictions for ids: " + ", ".join(missing_preds[:5]) + ("..." if len(missing_preds) > 5 else "")
        )

    aligned: Dict[str, Dict[str, Any]] = {}
    for example_id, candidate in predictions.items():
        aligned[example_id] = {
            "prediction": candidate,
            "references": list(references[example_id]),
        }
    return aligned


def _tokenize(text: str) -> List[str]:
    return text.lower().strip().split()


def _extract_ngrams(tokens: Sequence[str], order: int) -> Counter:
    ngrams: Counter = Counter()
    if len(tokens) < order or order <= 0:
        return ngrams
    for i in range(len(tokens) - order + 1):
        ngram = tuple(tokens[i : i + order])
        ngrams[ngram] += 1
    return ngrams


def _modified_precision(candidate: Counter, references: List[Counter]) -> Tuple[int, int]:
    overlap = 0
    total = sum(candidate.values())
    if total == 0:
        return 0, 0
    max_counts: MutableMapping[Tuple[str, ...], int] = {}
    for ref in references:
        for ngram, count in ref.items():
            max_counts[ngram] = max(max_counts.get(ngram, 0), count)
    for ngram, count in candidate.items():
        overlap += min(count, max_counts.get(ngram, 0))
    return overlap, total


def _brevity_penalty(pred_len: int, ref_lens: List[int]) -> float:
    if pred_len == 0:
        return 0.0
    closest_ref_len = min(ref_lens, key=lambda rl: (abs(rl - pred_len), rl))
    if pred_len > closest_ref_len:
        return 1.0
    return math.exp(1 - closest_ref_len / pred_len) if pred_len else 0.0


def _compute_bleu_score_internal(pred: str, refs: Sequence[str], max_order: int = 4) -> Dict[str, Any]:
    tokenized_pred = _tokenize(pred)
    tokenized_refs = [_tokenize(ref) for ref in refs]
    precisions: List[float] = []
    eps = 1e-9
    for order in range(1, max_order + 1):
        pred_counts = _extract_ngrams(tokenized_pred, order)
        ref_counts = [_extract_ngrams(ref, order) for ref in tokenized_refs]
        overlap, total = _modified_precision(pred_counts, ref_counts)
        if total == 0:
            precisions.append(0.0)
            continue
        # Smoothed precision (method 1): add-one smoothing
        numer = overlap + 1 if overlap == 0 else overlap
        denom = total + 1 if overlap == 0 else total
        p = numer / denom
        # clamp to [0,1]
        p = max(0.0, min(1.0, p))
        precisions.append(p)
    # geometric mean with epsilon to avoid -inf
    log_precision = sum(math.log(max(p, eps)) for p in precisions)
    geo_mean = math.exp(log_precision / max_order)
    bp = _brevity_penalty(len(tokenized_pred), [len(ref) for ref in tokenized_refs])
    return {"bleu": bp * geo_mean, "precisions": precisions, "bp": bp}


def _compute_bleu_score(pred: str, refs: Sequence[str], max_order: int = 4) -> Dict[str, Any]:
    # Prefer sacrebleu if installed for robust BLEU; fall back to internal
    if sacrebleu is not None:
        try:
            sent = sacrebleu.sentence_bleu(pred, refs, smooth_method="exp")
            # We also compute internal precisions for metadata sanity
            internal = _compute_bleu_score_internal(pred, refs, max_order=max_order)
            return {
                "bleu": sent.score / 100.0,
                "precisions": internal["precisions"],
                "bp": internal["bp"],
            }
        except Exception:
            pass
    return _compute_bleu_score_internal(pred, refs, max_order=max_order)


def _build_document_frequency(references: Mapping[str, Sequence[str]], max_order: int = 4) -> Counter:
    df: Counter = Counter()
    for refs in references.values():
        for ref in refs:
            tokens = _tokenize(ref)
            for order in range(1, max_order + 1):
                ngrams = set(_extract_ngrams(tokens, order))
                df.update(ngrams)
    return df


def _tfidf_vector(counts: Counter, df: Counter, num_documents: int) -> Dict[Tuple[str, ...], float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    vector: Dict[Tuple[str, ...], float] = {}
    for ngram, count in counts.items():
        idf = math.log((num_documents + 1) / (df.get(ngram, 0) + 1)) + 1
        vector[ngram] = (count / total) * idf
    return vector


def _cosine_similarity(vec1: Mapping[Tuple[str, ...], float], vec2: Mapping[Tuple[str, ...], float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    intersection = set(vec1) & set(vec2)
    numerator = sum(vec1[k] * vec2[k] for k in intersection)
    denom1 = math.sqrt(sum(v * v for v in vec1.values()))
    denom2 = math.sqrt(sum(v * v for v in vec2.values()))
    if denom1 == 0 or denom2 == 0:
        return 0.0
    return numerator / (denom1 * denom2)


def _compute_cider_score(
    pred: str,
    refs: Sequence[str],
    df: Counter,
    num_documents: int,
    max_order: int = 4,
) -> Dict[str, Any]:
    tokenized_pred = _tokenize(pred)
    pred_counts = {order: _extract_ngrams(tokenized_pred, order) for order in range(1, max_order + 1)}
    ref_counts_list = []
    for ref in refs:
        tokens = _tokenize(ref)
        ref_counts_list.append({order: _extract_ngrams(tokens, order) for order in range(1, max_order + 1)})

    scores: List[float] = []
    for order in range(1, max_order + 1):
        pred_vec = _tfidf_vector(pred_counts[order], df, num_documents)
        if not pred_vec:
            scores.append(0.0)
            continue
        sim_total = 0.0
        for ref_counts in ref_counts_list:
            ref_vec = _tfidf_vector(ref_counts[order], df, num_documents)
            sim_total += _cosine_similarity(pred_vec, ref_vec)
        scores.append(sim_total / max(len(ref_counts_list), 1))
    cider = sum(scores) / max_order
    return {"score": cider, "order_scores": scores}


def _compute_spice_score(pred: str, refs: Sequence[str]) -> Dict[str, Any]:
    tokenized_pred = set(_tokenize(pred))
    if not tokenized_pred:
        return {"score": 0.0}
    best = 0.0
    for ref in refs:
        tokenized_ref = set(_tokenize(ref))
        if not tokenized_ref and not tokenized_pred:
            best = 1.0
            break
        intersection = tokenized_pred & tokenized_ref
        if not tokenized_ref:
            continue
        precision = len(intersection) / len(tokenized_pred)
        recall = len(intersection) / len(tokenized_ref) if tokenized_ref else 0.0
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return {"score": best}


def compute_caption_metrics(
    predictions: Mapping[str, str],
    references: Mapping[str, Sequence[str]],
) -> Dict[str, MetricResult]:
    aligned = _align_predictions(predictions, references)
    df = _build_document_frequency({k: v["references"] for k, v in aligned.items()})
    num_documents = sum(len(v["references"]) for v in aligned.values())

    bleu_scores: List[float] = []
    cider_scores: List[float] = []
    spice_scores: List[float] = []

    bleu_metadata: List[Dict[str, Any]] = []
    cider_metadata: List[Dict[str, Any]] = []
    spice_metadata: List[Dict[str, Any]] = []

    for item in aligned.values():
        bleu_result = _compute_bleu_score(item["prediction"], item["references"])
        cider_result = _compute_cider_score(item["prediction"], item["references"], df, num_documents)
        spice_result = _compute_spice_score(item["prediction"], item["references"])

        bleu_scores.append(bleu_result["bleu"])
        cider_scores.append(cider_result["score"])
        spice_scores.append(spice_result["score"])

        bleu_metadata.append(bleu_result)
        cider_metadata.append(cider_result)
        spice_metadata.append(spice_result)

    bleu_avg = sum(bleu_scores) / len(bleu_scores) if bleu_scores else float("nan")
    cider_avg = sum(cider_scores) / len(cider_scores) if cider_scores else float("nan")
    spice_avg = sum(spice_scores) / len(spice_scores) if spice_scores else float("nan")

    return {
        "bleu": MetricResult("BLEU", bleu_avg, metadata={"per_example": bleu_metadata}),
        "cider": MetricResult("CIDEr", cider_avg, metadata={"per_example": cider_metadata}),
        "spice": MetricResult("SPICE", spice_avg, metadata={"per_example": spice_metadata}),
    }


def compute_qa_metrics(
    predictions: Mapping[str, str],
    references: Mapping[str, Sequence[str]],
) -> Dict[str, MetricResult]:
    aligned = _align_predictions(predictions, references)
    preds = [item["prediction"] for item in aligned.values()]
    ref_strings = [item["references"] for item in aligned.values()]

    accuracy_targets: List[int] = []
    for pred, answers in zip(preds, ref_strings):
        normalized_pred = pred.strip().lower()
        normalized_refs = [ans.strip().lower() for ans in answers]
        accuracy_targets.append(int(normalized_pred in normalized_refs))

    accuracy_value = sum(accuracy_targets) / len(accuracy_targets) if accuracy_targets else float("nan")

    f1_scores: List[float] = []
    for pred, answers in zip(preds, ref_strings):
        tokenized_pred = _tokenize(pred)
        best = 0.0
        for answer in answers:
            tokenized_answer = _tokenize(answer)
            if not tokenized_answer and not tokenized_pred:
                best = 1.0
                break
            pred_counter = Counter(tokenized_pred)
            answer_counter = Counter(tokenized_answer)
            common = pred_counter & answer_counter
            overlap = sum(common.values())
            if overlap == 0:
                continue
            precision = overlap / sum(pred_counter.values()) if pred_counter else 0.0
            recall = overlap / sum(answer_counter.values()) if answer_counter else 0.0
            if precision + recall == 0:
                continue
            f1_score = 2 * precision * recall / (precision + recall)
            best = max(best, f1_score)
        f1_scores.append(best)

    f1_value = sum(f1_scores) / len(f1_scores) if f1_scores else float("nan")

    return {
        "accuracy": MetricResult("Accuracy", accuracy_value, metadata={"per_example": accuracy_targets}),
        "f1": MetricResult("F1", f1_value, metadata={"per_example": f1_scores}),
    }


def normalize_references(values: Iterable[Any]) -> List[str]:
    normalized: List[str] = []
    for value in values:
        if isinstance(value, str):
            normalized.append(value)
        elif isinstance(value, Iterable):
            normalized.extend(str(v) for v in value)
        else:
            normalized.append(str(value))
    return normalized
try:  # pragma: no cover - optional dependency
    import sacrebleu  # type: ignore
except Exception:  # pragma: no cover
    sacrebleu = None
