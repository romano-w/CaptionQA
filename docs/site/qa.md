# QA Pipeline

Visual question answering runs live in `captionqa.qa.baseline_vqa`. This page summarizes the main configurations, summary-context experiments, and diagnostics.

## Baseline Commands

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m captionqa.qa.baseline_vqa \
  --manifest data/eval/qa/360x_devmini/manifest.jsonl \
  --refs data/eval/qa/360x_devmini/refs.jsonl \
  --output-dir data/eval/qa/360x_devmini
```

- `--normalize-preds/--no-normalize-preds` toggles TAL label normalization (defaults to on).
- `--force-label-prompt` constrains Qwen to emit a single taxonomy label (Acc/F1 ≈ 0.1759 for dev-mini).

## Summary Context Variants

`--summary-jsonl` accepts any JSON/JSONL with `{id, prediction}` rows. Lookup order is:

1. QA example ID (exact match).
2. `<scene>_<clip>` derived from the manifest’s video path.
3. Clip stem alone.

When provided, the summary text is prepended to the question with instructions to “treat as auxiliary context, prefer the video.”

### Caption-Level Context

- `data/eval/captioning/360x_devmini/preds.jsonl` → `data/eval/qa/360x_devmini_summary` (Acc/F1 0.079).
- Forced-label variant at `data/eval/qa/360x_devmini_summary_forceprompt` (Acc/F1 0.155). Both collapse toward “photographing.”

### QA Window Captions

New per-window captions avoid mismatched sentences:

```bash
# 1. Create a 60-question manifest for windows
python3 scripts/build_qa_caption_manifest.py \
  --qa-manifest data/eval/qa/360x_devmini/manifest.jsonl \
  --output data/eval/qa/360x_devmini_perwindow60/manifest.jsonl \
  --limit 60

# 2. Caption those windows with Qwen-VL
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m captionqa.captioning.baseline \
  --manifest data/eval/qa/360x_devmini_perwindow60/manifest.jsonl \
  --engine qwen_vl \
  --output-dir data/eval/captioning/360x_devmini_perwindow60

# 3. Feed results into the QA runner
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m captionqa.qa.baseline_vqa \
  --manifest data/eval/qa/360x_devmini/manifest.jsonl \
  --refs data/eval/qa/360x_devmini/refs.jsonl \
  --output-dir data/eval/qa/360x_devmini_perwindow60 \
  --summary-jsonl data/eval/captioning/360x_devmini_perwindow60/preds.jsonl \
  --summary-max-chars 320 \
  --limit 60
```

Result: Accuracy/F1 rose slightly to **0.05** on the 60-question probe (`data/eval/qa/360x_devmini_perwindow60/summary.json`), confirming per-window captions reduce collapse but still lag the no-summary baseline (0.159).

## Diagnostics

- `scripts/analyze_qa_mismatches.py` surfaces the most common raw predictions per TAL label; see README for usage.
- Confusion matrices accompany every run under `data/eval/qa/.../confusion.json`.
- Use `--limit` for quick iterations; copy matching refs via `refs.filtered.jsonl` when limiting.
