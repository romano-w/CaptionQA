"""Baseline VQA runner using Qwen-VL engine over a dev-mini manifest."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Mapping, Optional

from .engines.qwen_vl_qa import QwenVLVQAEngine
from ..captioning.config import QwenVLConfig
from ..captioning.panorama import PanoramaSamplingConfig
from ..utils.progress import ProgressDisplay
from .normalization import TAL_LABELS, normalize_prediction


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


def _build_refs_map(rows: List[Mapping[str, object]]) -> Dict[str, List[str]]:
    refs: Dict[str, List[str]] = {}
    for row in rows:
        ex_id = str(row.get("id"))
        answers = [str(ans).lower() for ans in row.get("answers", [])]
        refs[ex_id] = answers
    return refs


def _build_summary_map(rows: List[Mapping[str, object]]) -> Dict[str, str]:
    summaries: Dict[str, str] = {}
    for row in rows:
        key = str(row.get("id", "")).strip()
        if not key:
            continue
        text = row.get("prediction") or row.get("summary") or row.get("text")
        if not text:
            continue
        summaries[key] = str(text).strip()
    return summaries


def _summary_lookup_keys(video_path: str, example_id: str) -> List[str]:
    """Return a deterministic set of lookup keys for the summary map."""
    keys: List[str] = []
    if example_id:
        keys.append(example_id)
    try:
        path = Path(video_path)
        clip = path.stem
        parent = path.parent.name if path.parent else ""
        if parent and clip:
            keys.append(f"{parent}_{clip}")
        if clip:
            keys.append(clip)
    except Exception:
        pass
    return keys


def _write_confusion_matrix(
    preds: List[Mapping[str, object]],
    refs_map: Dict[str, List[str]],
    label_order: List[str],
    output_path: Path,
) -> None:
    other_label = "<other>"
    pred_labels = list(label_order) + [other_label]
    actual_labels: List[str] = []
    actual_set = {ans for answers in refs_map.values() for ans in answers}
    seen = set()
    for label in label_order:
        if label in actual_set:
            actual_labels.append(label)
            seen.add(label)
    for extra in sorted(actual_set - seen):
        actual_labels.append(extra)

    matrix: Dict[str, Dict[str, int]] = {}

    def ensure_actual(label: str) -> None:
        if label not in matrix:
            matrix[label] = {pred_label: 0 for pred_label in pred_labels}
            if label not in actual_labels:
                actual_labels.append(label)

    for actual in actual_labels:
        ensure_actual(actual)

    for row in preds:
        pred_label = row.get("prediction", "").lower()
        pred_bucket = pred_label if pred_label in pred_labels else other_label
        answers = refs_map.get(str(row.get("id")), [])
        if not answers:
            ensure_actual("<missing>")
            answers = ["<missing>"]
        for actual in answers:
            ensure_actual(actual)
            matrix[actual][pred_bucket] += 1

    data = {
        "actual_labels": actual_labels,
        "predicted_labels": pred_labels,
        "matrix": [[matrix[actual][pred] for pred in pred_labels] for actual in actual_labels],
        "support": {actual: sum(matrix[actual].values()) for actual in actual_labels},
    }
    output_path.write_text(json.dumps(data, indent=2))


def run(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run baseline VQA over a dev-mini manifest with Qwen-VL")
    p.add_argument("--manifest", type=Path, required=True, help="JSON/JSONL with {id, video, question}")
    p.add_argument("--refs", type=Path, required=True, help="JSON/JSONL with {id, answers: [...]}")
    p.add_argument("--output-dir", type=Path, default=Path("data/eval/qa/devmini"))
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="Process only the first N manifest entries (debugging).")
    p.add_argument("--debug", action="store_true", help="Enable verbose logging for Qwen engines.")
    p.add_argument(
        "--normalize-preds",
        dest="normalize_preds",
        action="store_true",
        default=True,
        help="Map free-form answers to the TAL action vocabulary (default: enabled).",
    )
    p.add_argument(
        "--no-normalize-preds",
        dest="normalize_preds",
        action="store_false",
        help="Disable answer normalization and emit raw model text.",
    )
    p.add_argument(
        "--force-label-prompt",
        dest="force_label_prompt",
        action="store_true",
        default=False,
        help="Instruct Qwen to reply with a single TAL action label (opt-in).",
    )
    p.add_argument(
        "--summary-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional captioning outputs (JSON/JSONL with {id, prediction}) used as textual context. "
            "IDs matching QA example IDs or <scene>_<clip> will be passed as working memory."
        ),
    )
    p.add_argument(
        "--summary-max-chars",
        type=int,
        default=320,
        help="Trim summaries to this many characters before passing them as context (0 disables trimming).",
    )
    p.add_argument(
        "--video-first-prompt",
        action="store_true",
        help="Use a stricter prompt that emphasizes the requested time window, TAL actions, and ignoring camera motion.",
    )
    p.add_argument(
        "--justify-predictions",
        action="store_true",
        help="After emitting a forced-label answer, run a second pass to generate a short textual justification.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.justify_predictions and not args.force_label_prompt:
        p.error("--justify-predictions requires --force-label-prompt to explain a discrete TAL label.")

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    manifest = _read_json_or_jsonl(args.manifest)
    if args.limit is not None and args.limit >= 0:
        manifest = manifest[: args.limit]
    total = len(manifest)
    refs_records = _read_json_or_jsonl(args.refs)
    refs_map = _build_refs_map(refs_records)
    summary_map: Dict[str, str] = {}
    if args.summary_jsonl:
        try:
            summary_records = _read_json_or_jsonl(args.summary_jsonl)
            summary_map = _build_summary_map(summary_records)
            logging.info("Loaded %d summaries from %s", len(summary_map), args.summary_jsonl)
        except FileNotFoundError:
            logging.warning("Summary file %s not found; continuing without context.", args.summary_jsonl)
        except Exception as exc:
            logging.warning("Failed to parse summary file %s: %s", args.summary_jsonl, exc)
    print(
        f"Loaded QA manifest with {total} example(s) "
        f"from {args.manifest} -> writing preds to {args.output_dir}",
        flush=True,
    )
    if total == 0:
        print("Manifest is empty; nothing to process.", file=sys.stderr)
    qcfg = QwenVLConfig(model_name=args.model_name, num_frames=args.num_frames, max_new_tokens=args.max_new_tokens)
    justification_template: Optional[str] = None
    if args.force_label_prompt:
        label_bullets = "\n".join(f"- {label}" for label in TAL_LABELS)
        examples = [
            ("A person walks down a hallway.", "walking"),
            ("Someone sits on a sofa adjusting a pillow.", "sitting"),
            ("A presenter speaks into a microphone.", "speaking"),
            ("A person texts on their phone.", "operating phone"),
            ("Liquid is poured from a kettle into a cup.", "pouring"),
            ("A person lifts a cup to drink.", "drinking"),
            ("Someone opens a refrigerator door.", "opening"),
            ("A person buttons a jacket in front of a mirror.", "dressing"),
            ("Someone wipes down a countertop with a rag.", "cleaning"),
            ("A person holds a phone outward to record a video.", "photographing"),
            ("A person strolls while scrolling on a phone.", "operating phone"),
            ("Someone zips a coat while pacing slowly.", "dressing"),
            ("A person sips from a mug while walking.", "drinking"),
            ("Two people sway rhythmically with music.", "dancing"),
            ("Someone covers their mouth while coughing.", "coughing"),
        ]
        example_lines = "\n".join(f"- Scene: {scene} -> Label: {label}" for scene, label in examples)
        video_first_lines = ""
        if args.video_first_prompt:
            video_first_lines = (
                "- Focus strictly on the requested time window; ignore frames outside it.\n"
                "- Trust the video over any summary text; use summaries only if they describe the same moment.\n"
                "- If uncertain, prefer the closest TAL label instead of describing camera motion or scenery.\n"
            )
        qcfg.qa_template = (
            "You are classifying the dominant action in a 360-degree video snippet. "
            "Respond with exactly one label from the taxonomy below (no punctuation, no explanations).\n"
            "Guidance:\n"
            f"{video_first_lines}"
            "- Use 'operating phone' whenever a person handles, looks at, or records with a phone/tablet.\n"
            "- Use 'speaking' for conversations, presentations, or anyone using a microphone.\n"
            "- Choose 'sitting' if the subject remains seated or mostly stationary on a chair/sofa.\n"
            "- Choose 'walking' for general locomotion, even indoors.\n"
            "- Use 'drinking' when a cup/bottle is raised to the mouth; use 'pouring' when liquid moves between containers.\n"
            "- Use 'opening' when doors, drawers, cabinets, fridges, or packages are opened.\n"
            "- Prefer 'dressing' when someone adjusts, buttons, zips, or changes clothing even if they move around.\n"
            "- Prefer 'operating phone' or 'speaking' over 'walking' when those cues are obvious (phone in hand, microphone, conversation).\n"
            "- Choose 'cleaning' or 'housekeeping' when wiping, tidying, or organizing dominates the scene.\n"
            "- Select 'coughing' when someone covers their mouth or audibly coughs, even if they continue to move.\n"
            "- Use 'dancing' for rhythmic swaying, coordinated steps, or repeating moves synchronized to music.\n"
            "- Choose 'drinking' whenever a cup reaches the mouth, regardless of whether the subject is seated, standing, or walking.\n"
            "- Reserve 'standing' for idle posture with no other object interaction; use the specific action label whenever hands work with an item.\n"
            "If unsure, pick the closest label instead of describing the scene.\n"
            f"Action labels:\n{label_bullets}\n"
            "Examples (scene -> label):\n"
            f"{example_lines}\n"
            "Answer with one label only."
        )
        qcfg.max_new_tokens = min(qcfg.max_new_tokens, 8)
        if args.justify_predictions:
            justification_template = (
                "You already selected a TAL action label for this clip. "
                "Provide a concise (≤25 words) justification that cites the subject, objects, and motion visible within the requested time window. "
                "Mention concrete cues (hands on phone, cup to mouth, zipping jacket, microphone use) instead of repeating the label."
            )
    else:
        if args.video_first_prompt:
            qcfg.qa_template = (
                "You are an expert annotator for the TAL action taxonomy. "
                "Follow these rules:\n"
                "1. Inspect the video frames within the requested time window before reasoning.\n"
                "2. Identify the dominant human action or interaction taking place.\n"
                "3. Answer with a concise action phrase (≤5 words) or the closest TAL label from the taxonomy.\n"
                "4. Ignore camera motion, scene setup, or lighting notes unless they explain the action.\n"
                "5. If the action is unclear, pick the closest TAL label instead of describing scenery.\n"
                "6. Cues:\n"
                "   - Handling, staring at, or tapping a phone/tablet → 'operating phone'.\n"
                "   - Sitting or reclining on furniture → 'sitting'.\n"
                "   - Speaking to others, microphones, or narration → 'speaking'.\n"
                "   - Adjusting, putting on, or removing clothing → 'dressing'.\n"
                "   - Standing still without notable motion → 'standing'."
            )
        else:
            qcfg.qa_template = (
                "You are answering a question about a 360-degree panoramic video. "
                "Inspect the video frames first, respond with the action or event that directly answers the question, "
                "and keep your reply to a short phrase."
            )
    summary_guidance = ""
    if summary_map:
        if args.video_first_prompt:
            summary_guidance = (
                "\nWhen a summary snippet is provided: use it only to double-check the video, "
                "ignore sentences about unrelated time spans or camera motion, and trust the video if there is any conflict."
            )
        else:
            summary_guidance = (
                "\nIf a text summary is provided: treat it as auxiliary context, favor the video evidence, "
                "and ignore summary sentences unrelated to the specific question."
            )
        qcfg.qa_template = (qcfg.qa_template or "").strip() + summary_guidance
        if justification_template:
            justification_template = justification_template.strip() + summary_guidance
    engine = QwenVLVQAEngine.from_configs(sampler_cfg=PanoramaSamplingConfig(), qwen_cfg=qcfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    refs_path = args.refs
    if args.limit is not None and args.limit >= 0 and total:
        keep_ids = {str(ex.get("id")) for ex in manifest}
        filtered_refs = [row for row in refs_records if str(row.get("id")) in keep_ids]
        refs_map = _build_refs_map(filtered_refs)
        refs_path = args.output_dir / "refs.filtered.jsonl"
        _write_jsonl(refs_path, filtered_refs)

    preds_path = args.output_dir / "preds.jsonl"
    justifications_path = args.output_dir / "justifications.jsonl"
    preds: List[Mapping[str, object]] = []
    justification_rows: List[Mapping[str, object]] = []
    overall_start = time.perf_counter()
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
        context = None
        time_hint = ""
        if start_end:
            try:
                time_hint = f"Focus on {start_end[0]:.1f}s to {start_end[1]:.1f}s. "
            except Exception:
                time_hint = ""
        if summary_map:
            for key in _summary_lookup_keys(vid, ex_id):
                if key in summary_map:
                    summary_text = summary_map[key]
                    if args.summary_max_chars and args.summary_max_chars > 0 and len(summary_text) > args.summary_max_chars:
                        snippet = summary_text[: args.summary_max_chars]
                        if " " in snippet:
                            snippet = snippet.rsplit(" ", 1)[0]
                        summary_text = snippet + "…"
                    question_hint = q.strip()
                    question_tag = f"The question is: {question_hint}. " if question_hint else ""
                    if args.video_first_prompt:
                        context = (
                            f"{question_tag}{time_hint}"
                            "Video-first rule: inspect the frames from this window before reading the snippet below. "
                            "Only use the snippet if it describes the same moment; ignore it otherwise.\n"
                            f"Summary snippet: {summary_text}"
                        )
                    else:
                        context = (
                            f"{question_tag}{time_hint}"
                            "Auxiliary summary (use only when the video lacks detail; ignore irrelevant sentences): "
                            f"{summary_text}"
                        )
                    break
        try:
            raw_ans = engine.answer(
                vid,
                q,
                context=context,
                start_sec=(start_end[0] if start_end else None),
                end_sec=(start_end[1] if start_end else None),
            )
        except Exception as exc:
            raw_ans = f"[error: {exc}]"
            progress.finish_step(idx, f"{ex_id} ERROR: {exc}")
        else:
            elapsed = time.perf_counter() - start_time
            progress.finish_step(idx, f"{ex_id} done in {elapsed:.1f}s")
        pred_text = raw_ans.strip()
        if args.normalize_preds:
            normalized = normalize_prediction(pred_text)
            record = {"id": ex_id, "prediction": normalized, "raw_prediction": pred_text}
        else:
            record = {"id": ex_id, "prediction": pred_text}
        preds.append(record)
        if args.justify_predictions:
            question_bits = [
                f"The predicted TAL action label is '{record['prediction']}'.",
                "Explain, in ≤25 words, the visual cues from this video window that support this label.",
            ]
            if time_hint:
                question_bits.append(time_hint.strip())
            if q.strip():
                question_bits.append(f"Original question: {q.strip()}")
            justification_question = " ".join(part for part in question_bits if part)
            prev_template = engine.config.qa_template
            prev_max_tokens = engine.config.max_new_tokens
            engine.config.qa_template = justification_template or prev_template
            engine.config.max_new_tokens = max(prev_max_tokens, 48)
            try:
                justification_answer = engine.answer(
                    vid,
                    justification_question,
                    context=context,
                    start_sec=(start_end[0] if start_end else None),
                    end_sec=(start_end[1] if start_end else None),
                )
            except Exception as exc:
                justification_answer = f"[justification-error: {exc}]"
            finally:
                engine.config.qa_template = prev_template
                engine.config.max_new_tokens = prev_max_tokens
            record["justification"] = justification_answer.strip()
            justification_rows.append({"id": ex_id, "justification": record["justification"]})
    _write_jsonl(preds_path, preds)
    justifications_written = False
    if args.justify_predictions:
        _write_jsonl(justifications_path, justification_rows)
        justifications_written = True

    # Evaluate
    from ..evaluation.run import main as eval_main

    summary = eval_main(
        [
            "--task",
            "qa",
            "--preds",
            str(preds_path),
            "--refs",
            str(refs_path),
            "--value-field",
            "answers",
            "--output-json",
            str(args.output_dir / "summary.json"),
            "--quiet",
        ]
    )
    metrics = summary.get("metrics", {})
    metric_pairs: List[str] = []
    for key, metric in metrics.items():
        score = None
        if isinstance(metric, Mapping):
            score = metric.get("score")
            label = metric.get("name") or key
        else:
            label = key
        if isinstance(score, (int, float)):
            metric_pairs.append(f"{label}={score:.4f}")
    metric_text = ", ".join(metric_pairs) if metric_pairs else "n/a"

    confusion_path = args.output_dir / "confusion.json"
    _write_confusion_matrix(preds, refs_map, TAL_LABELS, confusion_path)

    total_elapsed = time.perf_counter() - overall_start
    avg_example = total_elapsed / max(total or 1, 1)
    print("\n=== QA Run Summary ===")
    print(f"Processed examples : {total}")
    print(f"Elapsed time       : {total_elapsed/60:.2f} min ({avg_example:.2f}s / example)")
    print(f"Manifest           : {args.manifest}")
    print(f"Refs               : {refs_path}")
    if args.summary_jsonl:
        print(f"Summaries          : {args.summary_jsonl}")
    print(f"Predictions        : {preds_path}")
    if justifications_written:
        print(f"Justifications     : {justifications_path}")
    print(f"Summary JSON       : {args.output_dir / 'summary.json'}")
    print(f"Confusion matrix   : {confusion_path}")
    print(f"QA metrics         : {metric_text}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
