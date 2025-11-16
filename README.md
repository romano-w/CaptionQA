# CaptionQA

Panoramic captioning + question answering experiments over the 360x dataset using Qwen2.5‑VL 7B.

> Authors: Will Romano & Ethan Baird

---

## Quick Links
- **GPU bootstrap**: `scripts/setup_vast_a10.sh`
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

---

## Vast.ai A10 Workflow
The `scripts/setup_vast_a10.sh` script is the one-stop bootstrap for a fresh Vast container:

1. Clone + `cd /workspace/CaptionQA`
2. (Optional) `export HF_TOKEN=hf_xxxxx`
3. `bash scripts/setup_vast_a10.sh`

What the script does:
- Installs system deps (`ffmpeg`, `git`) when `apt-get` is present.
- Creates / activates `captionqa` venv, installs `uv`, `hf_transfer`, and the editable package.
- Sets `HF_HOME=/workspace/hf_cache` (configurable) and logs into HF if `HF_TOKEN` is exported.
- Downloads the **360x LR subset** into `${DATA_ROOT:-/workspace/data}/360x`, leveraging the new progress bar and skip behavior.
- Wires `data/raw -> ${DATA_ROOT}`.
- Builds manifest + TAL references for both captioning and QA devmini splits.

Run baselines immediately after setup:

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
| QA | Qwen2.5‑VL‑7B | `data/eval/qa/360x_devmini/manifest.jsonl` | Accuracy = **0.0** (`data/eval/qa/360x_devmini/summary.json`) | Free-form answers rarely match TAL action strings verbatim; need normalization or looser scoring. |

Latest progress + action items live in `docs/living_roadmap.md`.

---

## Troubleshooting Tips
- **HF 429s**: wait ~10 minutes or seed `/workspace/data/360x/360x_dataset_LR` manually, then rerun manifest steps.
- **No space left**: Vast defaults to 32 GB disks; captioning + data needs ≥80 GB. Provision a larger root disk or attach a separate `/workspace/data` volume.
- **Qwen cache mismatch**: `uv run` warns when `VIRTUAL_ENV` doesn’t match `.venv`; safe to ignore because runs happen inside `captionqa`.

---

## Documentation Backlog
- README now focuses on daily tasks; **consider spinning up a `docs/` site (GitHub Pages / MkDocs)** to hold deeper architecture notes, dataset deep dives, and troubleshooting logs.
