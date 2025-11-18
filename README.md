# CaptionQA

Panoramic captioning + question answering experiments over the 360x dataset using Qwen2.5‑VL 7B.

> Authors: Will Romano & Ethan Baird

---

## Quick Links
- **GPU bootstrap**: `scripts/setup_vast_gpu.sh`
- **Roadmap & latest status**: `docs/living_roadmap.md`
- **360x data helpers**: `python -m captionqa.datasets.x360_manifest|x360_tal_refs|x360_tal_qa`
- **Notebook playground**: `notebooks/eda_360x.ipynb`

---

## Local Quickstart
1. Install [uv](https://docs.astral.sh/uv/) and FFmpeg (e.g., `winget install Astral.Uv` + `winget install Gyan.FFmpeg` on Windows).
2. Create / activate a virtual environment:
   ```bash
   uv venv captionqa
   source captionqa/bin/activate  # .\captionqa\Scripts\Activate.ps1 on Windows
   ```
3. Install the project in editable mode:
   ```bash
   uv pip install --editable .
   ```
4. Authenticate with Hugging Face once: `huggingface-cli login --token <HF_TOKEN>`.

All CLI entry points live under `captionqa.*`. Example captioning invocation:

```bash
uv run python -m captionqa.captioning.baseline \
  --manifest data/eval/captioning/devmini/manifest.jsonl \
  --engine qwen_vl \
  --output-dir data/eval/captioning/devmini
```

---

## Data Downloads
- `python -m captionqa.data.download 360x --output <root> --360x-resolution lr|hr|both`
- New behavior: the helper now **skips existing dataset folders** (unless `--overwrite`) and shows a **single aggregate progress bar** instead of hundreds of per-file bars. This keeps Vast logs legible and prevents surprise re-downloads.
- `HF_HOME` defaults to `/workspace/hf_cache` inside the setup script; override via `export HF_HOME=/custom/path`.

Generated artifacts:
- Caption manifest: `data/eval/captioning/360x_devmini/manifest.jsonl`
- Caption refs (TAL): `data/eval/captioning/360x_devmini/refs.jsonl`
- QA manifest + refs: `data/eval/qa/360x_devmini/{manifest,refs}.jsonl`

### Manifest helper commands
Once the dataset lives under `${DATA_ROOT:-/workspace/data}/360x`, you can rebuild dev-mini manifests manually with the same invocations used inside `scripts/setup_vast_gpu.sh`:

```bash
# 1) Caption manifest (~100 clips)
uv run python -m captionqa.datasets.x360_manifest \
  --root "${DATA_ROOT:-/workspace/data}/360x/360x_dataset_LR/binocular" \
  --glob "*.mp4" \
  --limit 100 \
  --relative-to "${DATA_ROOT:-/workspace/data}" \
  --relative-prefix "data/raw" \
  --id-template '{parent_name}_{stem}' \
  --output data/eval/captioning/360x_devmini/manifest.jsonl

# 2) TAL caption references
uv run python -m captionqa.datasets.x360_tal_refs \
  --manifest data/eval/captioning/360x_devmini/manifest.jsonl \
  --annotations-root "${DATA_ROOT:-/workspace/data}/360x/360x_dataset_LR/TAL_annotations" \
  --output data/eval/captioning/360x_devmini/refs.jsonl

# 3) QA manifest + references (3 Qs per clip)
uv run python -m captionqa.datasets.x360_tal_qa \
  --manifest data/eval/captioning/360x_devmini/manifest.jsonl \
  --annotations-root "${DATA_ROOT:-/workspace/data}/360x/360x_dataset_LR/TAL_annotations" \
  --output-manifest data/eval/qa/360x_devmini/manifest.jsonl \
  --output-refs data/eval/qa/360x_devmini/refs.jsonl \
  --max-questions-per-video 3
```

If the manifest builder reports “No files matched the requested pattern,” double-check the dataset path or pass `--allow-empty` to intentionally emit an empty file (rarely necessary outside CI tests).

---

## Vast.ai A10 Workflow
The `scripts/setup_vast_gpu.sh` script is the one-stop bootstrap for a fresh Vast container:

1. Clone + `cd /workspace/CaptionQA`
2. (Optional) `export HF_TOKEN=hf_xxxxx`
3. `bash scripts/setup_vast_gpu.sh`

What the script does:
- Installs system deps (`ffmpeg`, `git`) when `apt-get` is present.
- Creates / activates `captionqa` venv, installs `uv`, `hf_transfer`, and the editable package.
- Sets `HF_HOME=/workspace/hf_cache` (configurable) and logs into HF if `HF_TOKEN` is exported.
- Downloads the **360x LR subset** into `${DATA_ROOT:-/workspace/data}/360x`, leveraging the new progress bar and skip behavior.
- Wires `data/raw -> ${DATA_ROOT}`.
- Builds manifest + TAL references for both captioning and QA devmini splits.
- Runs the captioning + QA devmini baselines (set `SKIP_BASELINES=1` to skip these long runs).

Re-running the baselines later:

```bash
# Captioning
uv run python -m captionqa.captioning.baseline \
  --manifest data/eval/captioning/360x_devmini/manifest.jsonl \
  --engine qwen_vl \
  --refs data/eval/captioning/360x_devmini/refs.jsonl \
  --output-dir data/eval/captioning/360x_devmini

# QA
uv run python -m captionqa.qa.baseline_vqa \
  --manifest data/eval/qa/360x_devmini/manifest.jsonl \
  --refs data/eval/qa/360x_devmini/refs.jsonl \
  --output-dir data/eval/qa/360x_devmini

# QA (force TAL label replies)
uv run python -m captionqa.qa.baseline_vqa \
  --manifest data/eval/qa/360x_devmini/manifest.jsonl \
  --refs data/eval/qa/360x_devmini/refs.jsonl \
  --output-dir data/eval/qa/360x_devmini_forceprompt \
  --force-label-prompt

# QA (force TAL label replies)
uv run python -m captionqa.qa.baseline_vqa \
  --manifest data/eval/qa/360x_devmini/manifest.jsonl \
  --refs data/eval/qa/360x_devmini/refs.jsonl \
  --output-dir data/eval/qa/360x_devmini_forceprompt \
  --force-label-prompt
```

If HF rate limits, re-run only the download step:

```bash
source captionqa/bin/activate
HF_HUB_ENABLE_HF_TRANSFER=1 python -m captionqa.data.download 360x \
  --output /workspace/data/360x \
  --overwrite \
  --360x-resolution lr
```

---

## Baseline Status (devmini)
| Task | Engine | Manifest | Metrics (summary.json) | Notes |
| --- | --- | --- | --- | --- |
| Captioning | Qwen2.5‑VL‑7B | `data/eval/captioning/360x_devmini/manifest.jsonl` | BLEU ≈ **0.0053** · CIDEr ≈ **0.0050** · SPICE ≈ **0.0536** (`data/eval/captioning/360x_devmini/summary.json`) | TAL references are action labels, so absolute scores remain tiny even when captions align. |
| QA | Qwen2.5‑VL‑7B | `data/eval/qa/360x_devmini/manifest.jsonl` | Accuracy = **0.159** · F1 = **0.159** (`data/eval/qa/360x_devmini/summary.json`) | Predictions are normalized to the TAL verb set; lingering confusion collapses dressing/operating phone/speaking into “walking” and accounts for most misses. Confusion matrix lives at `data/eval/qa/360x_devmini/confusion.json`. |

The label-forcing variant (Qwen instructed to emit exactly one TAL label) stores results under `data/eval/qa/360x_devmini_forceprompt` and currently reaches Accuracy/F1 ≈ **0.176**. Its confusion matrix lives alongside the summary for quick inspection.

### Summary-Augmented QA (caption context)
- Pass caption summaries into the QA engine as “working memory” via `--summary-jsonl`. The runner matches QA examples against caption IDs (`<scene>_<clip>`) and includes that text as context to Qwen.
- Slice clip-level captions down to the QA span with `python -m captionqa.qa.summary_slices`. Example:

```bash
PYTHONPATH=src python3 -m captionqa.qa.summary_slices \
  --manifest data/eval/qa/360x_devmini/manifest.jsonl \
  --captions data/eval/captioning/360x_devmini/preds.jsonl \
  --output data/eval/qa/360x_devmini_summaryslice/summaries.jsonl \
  --max-sentences 2
```

- Example debug sweep:

```bash
uv run python -m captionqa.qa.baseline_vqa \
  --manifest data/eval/qa/360x_devmini/manifest.jsonl \
  --refs data/eval/qa/360x_devmini/refs.jsonl \
  --output-dir data/eval/qa/360x_devmini_summarydebug \
  --limit 8 \
  --summary-jsonl data/eval/captioning/360x_devmini/preds.jsonl \
--debug
```

The summary file can be any JSON/JSONL rows with `{id, prediction}` and will be used whenever the ID matches the QA example ID or `<scene>_<clip>` derived from its video path.

- **Latest results**: Full dev-mini runs with summary context land at Accuracy/F1 = **0.079** (normalized) under `data/eval/qa/360x_devmini_summary` and **0.155** for the forced-label prompt at `data/eval/qa/360x_devmini_summary_forceprompt`. Most predictions collapse to “photographing,” suggesting the current summaries overpower the question cues; treat these as a diagnostic baseline before iterating on summary length or prompt conditioning. A 60-question probe that used the sliced two-sentence summaries (`data/eval/qa/360x_devmini_summaryslice/summaries.jsonl`) and writes outputs to `data/eval/qa/360x_devmini_summaryslice60` reached Accuracy/F1 = **0.033**, so trimming alone is insufficient and we still need better prompts or per-window captioning.
- **Per-window caption context**: For true QA-span summaries, copy the QA manifest to a new file and caption it directly with Qwen-VL before feeding it back into the QA runner:

```bash
python3 scripts/build_qa_caption_manifest.py \
  --qa-manifest data/eval/qa/360x_devmini/manifest.jsonl \
  --output data/eval/qa/360x_devmini_perwindow60/manifest.jsonl \
  --limit 60

PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m captionqa.captioning.baseline \
  --manifest data/eval/qa/360x_devmini_perwindow60/manifest.jsonl \
  --engine qwen_vl \
  --output-dir data/eval/captioning/360x_devmini_perwindow60

PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m captionqa.qa.baseline_vqa \
  --manifest data/eval/qa/360x_devmini/manifest.jsonl \
  --refs data/eval/qa/360x_devmini/refs.jsonl \
  --output-dir data/eval/qa/360x_devmini_perwindow60 \
  --summary-jsonl data/eval/captioning/360x_devmini_perwindow60/preds.jsonl \
  --summary-max-chars 320 \
  --limit 60
```

Result: Accuracy/F1 ticked up to **0.05** on that 60-question probe (`data/eval/qa/360x_devmini_perwindow60/summary.json`). Still below the no-context baseline, but the collapse is slightly less severe than using whole-clip captions.

### Known QA Issues (devmini)
- 54 prior `<engine-unavailable>` outputs are gone after retrying frame sampling, so remaining <other> predictions are genuine semantic errors.
- Dressing, operating phone, and speaking questions still map to “walking/standing” ~60% of the time even with expanded prompts; normalization only helps when the raw text contains an explicit action verb.
- Pouring/housekeeping occasionally drift into `<other>` because references mention multi-step activities; review `data/eval/qa/360x_devmini/preds.jsonl` when tuning regexes or prompts.
- Use `python3 scripts/analyze_qa_mismatches.py --preds data/eval/qa/360x_devmini/preds.jsonl --refs data/eval/qa/360x_devmini/refs.jsonl --top-k 5 --suggest-regex --export-csv data/eval/qa/360x_devmini/mismatch_report.csv` to dump the most common raw predictions per TAL label, emit naive regex hints, and capture everything in a CSV for editing TAL normalization rules.

Latest progress + action items live in `docs/living_roadmap.md`.

---

## Troubleshooting Tips
- **HF 429s**: wait ~10 minutes or seed `/workspace/data/360x/360x_dataset_LR` manually, then rerun manifest steps.
- **No space left**: Vast defaults to 32 GB disks; captioning + data needs ≥80 GB. Provision a larger root disk or attach a separate `/workspace/data` volume.
- **Qwen cache mismatch**: `uv run` warns when `VIRTUAL_ENV` doesn’t match `.venv`; safe to ignore because runs happen inside `captionqa`.

---

## Documentation Backlog
- README now focuses on daily tasks; the MkDocs staging area (see `docs/site/index.md`, `docs/site/captioning.md`, and `docs/site/qa.md`) hosts deeper architecture notes plus per-window QA instructions. Run `mkdocs serve -f docs/mkdocs.yml` to preview and expand the content.
