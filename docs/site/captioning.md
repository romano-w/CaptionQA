# Captioning Pipeline

This page tracks how we generate panoramic captions for 360x dev-mini, including the per-window variant that now powers summary-aware QA experiments.

## Engines

- **Fusion (default)** — samples panoramic frames, encodes audio/visual streams, and decodes text via the custom fusion stack (`captionqa.captioning.pipeline`).
- **Qwen-VL (`--engine qwen_vl`)** — samples frames and prompts Qwen2.5‑VL‑7B directly. This engine honors `start`/`end` timestamps in the manifest, which lets us caption QA-sized windows.

## Standard dev-mini run

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m captionqa.captioning.baseline \
  --manifest data/eval/captioning/360x_devmini/manifest.jsonl \
  --engine qwen_vl \
  --output-dir data/eval/captioning/360x_devmini
```

Outputs live under the `--output-dir` (manifest/preds/summary).

## QA Window Captions

1. Slice the QA manifest down to the subset you want to caption. The helper below copies the first 60 QA rows into a new manifest:

```bash
python3 scripts/build_qa_caption_manifest.py \
  --qa-manifest data/eval/qa/360x_devmini/manifest.jsonl \
  --output data/eval/qa/360x_devmini_perwindow60/manifest.jsonl \
  --limit 60
```

2. Run the captioner over those per-window examples with Qwen-VL:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m captionqa.captioning.baseline \
  --manifest data/eval/qa/360x_devmini_perwindow60/manifest.jsonl \
  --engine qwen_vl \
  --output-dir data/eval/captioning/360x_devmini_perwindow60
```

The resulting `preds.jsonl` keeps QA IDs (e.g., `...clip1_1_1_1`), so downstream QA runs can load it via `--summary-jsonl`.

## Metrics

- Full 100-clip captioning runs: BLEU 0.0053 / CIDEr 0.0050 / SPICE 0.0536 (`data/eval/captioning/360x_devmini/summary.json`).
- QA window captions are diagnostics only; we do not evaluate them against TAL references but use them exclusively as working memory for summary-aware QA experiments (`data/eval/captioning/360x_devmini_perwindow60`).
