# **CaptionQA**

360° Panoramic Video Captioning + QA system for BLV accessibility.

> *Authors: Will Romano, Ethan Baird*

## Getting Started

### Prerequisite

Install [uv](https://docs.astral.sh/uv/) once on your machine.

- **Windows (PowerShell)**
  - `winget install --id Astral.Uv -e`
  - or `pipx install uv`
- **macOS / Linux**
  - `brew install uv`
  - or `curl -LsSf https://astral.sh/uv/install.sh | sh`

You will also need system FFmpeg binaries available on your `PATH` to process video/audio assets.

- Windows: `winget install --id Gyan.FFmpeg`
- macOS: `brew install ffmpeg`

### Environment Setup

After uv is available, open a terminal in the repository root and run:

```bash
uv venv captionqa
```

Activate the virtual environment for your platform:

- Windows PowerShell: `.\captionqa\Scripts\Activate.ps1`
- macOS / Linux (bash/zsh): `source captionqa/bin/activate`

Then install the project in editable mode:

```bash
uv pip install --editable .
```

Linux environments will pull in [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) automatically; macOS and Windows skip this GPU-only dependency so installs complete without errors.

Launch VS Code (with the Jupyter extension) and open `notebooks/quickstart.ipynb`; you can stay inside Notebook view without starting a separate Jupyter server.

### Generate captions from the CLI

Once the environment is activated you can invoke the panoramic captioning pipeline directly. The entry point loads off-the-shelf encoders/decoders defined in `pyproject.toml` and emits a descriptive caption:

```powershell
# Pinned uv run (Windows PowerShell)
./scripts/uv_run.ps1 python -m captionqa.captioning --print-config path/to/video.mp4
```

Override defaults with a JSON configuration file that mirrors the structure printed via `--print-config`:

```powershell
./scripts/uv_run.ps1 python -m captionqa.captioning path/to/video.mp4 --config configs/custom_captioning.json
```

#### Caching

- Encoders cache encoded features as `.pt` under `data/cache/{visual,audio}` by default.
- Control via CLI:
  - Disable caching: `--no-cache`
  - Custom cache directory root: `--cache-dir data/cache`
- Config fields (JSON): `visual_encoder.batch_size`, `visual_encoder.cache_dir`, `visual_encoder.use_cache`, `audio_encoder.cache_dir`, `audio_encoder.use_cache`.

The CLI is a thin wrapper around `captionqa.captioning.generate_captions`, so you can call the API from notebooks or other scripts with the same arguments used above.

### Hugging Face Access (required for 360x & Leader360V)

Several dataset mirrors (e.g., `360x`, `Leader360V`) are **gated** on Hugging Face. You must:

1. Sign in at <https://huggingface.co/> and request access on each dataset card.
2. Create a personal access token (Settings → Access Tokens → New token) with **Read** scope.
3. Authenticate the local environment once:

   ```bash
   huggingface-cli login --token <your_token>
   ```

   The CLI stores credentials under `~/.cache/huggingface/token` (Linux/macOS) or `%USERPROFILE%\.cache\huggingface\token` (Windows). Alternatively, set `HF_TOKEN` before running downloads:

   - Windows PowerShell: `$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxx"`
   - macOS / Linux: `export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"`

After these steps, `python -m captionqa.data.download 360x --output <path>` will authenticate automatically and download the low-resolution (`LR`) split by default. Use `--360x-resolution hr` to fetch only the high-resolution data or `--360x-resolution both` to mirror both splits. Repeat the access request step for future gated datasets as needed.

## Datasets

We're starting development using the following datasets:

- **360+x** ([x360dataset.github.io](https://x360dataset.github.io)): Panoramic video dataset with scene descriptions, action labels, and binaural spatial audio for spatially-aware captioning.
- **360DVD** ([GitHub](https://github.com/Akaneqwq/360DVD)): Dense video understanding dataset featuring 360° content with spatial-temporal annotations for video-language modeling.
- **Leader360V** ([Hugging Face](https://huggingface.co/datasets/Leader360V/Leader360V)): Large-scale 360° dataset for object tracking and viewpoint-aware understanding.
- **360SR** ([GitHub](https://github.com/360SR/360SR-Challenge)): Static panoramic scene classification dataset, useful for pretraining or augmenting spatial scene context models.
- **AVQA** ([GitHub](https://github.com/AlyssaYoung/AVQA)): Audio-visual question answering benchmark pairing multi-channel videos with spatio-temporal QA annotations.

These datasets offer rich multimodal supervision (video, audio, and text) for both captioning and interactive QA in immersive environments.

## Goals

- Spatially grounded event captions
- Interactive natural language questions

## Evaluation & Benchmarking

The `captionqa.evaluation` module ships reference metrics and a batch CLI for reproducible
benchmarking. Metric implementations are self-contained so no external services are
required; installing the project in editable mode pulls in the only additional dependency
(`datasets`).

### Directory layout

By convention, predictions are stored in JSON or JSONL files under `data/eval/<task>/`.
Each record must contain an `"id"` field matching the dataset identifier and a
`"prediction"` string. References can either be supplied via JSON/JSONL or sourced
from Hugging Face datasets.

```
data/
  eval/
    captioning/
      preds.jsonl           # {"id": "...", "prediction": "..."}
      refs.jsonl            # {"id": "...", "references": ["..."]}
    qa/
      preds.jsonl
      refs.jsonl            # {"id": "...", "answers": ["..."]}
```

### Running the evaluator

Use the CLI to aggregate BLEU/CIDEr/SPICE (captioning) or accuracy/F1 (QA).

```powershell
./scripts/uv_run.ps1 python -m captionqa.evaluation.run \
  --task captioning \
  --preds data/eval/captioning/preds.jsonl \
  --refs data/eval/captioning/refs.jsonl \
  --output-json data/eval/captioning/summary.json
```

To evaluate directly against a Hugging Face dataset split, omit `--refs` and supply the
dataset metadata instead:

```powershell
./scripts/uv_run.ps1 python -m captionqa.evaluation.run \
  --task qa \
  --preds data/eval/qa/preds.jsonl \
  --dataset-name Leader360V/Leader360V \
  --split validation \
  --reference-column answers
```

Both invocations print metrics to stdout and, when `--output-json` is provided, write a
machine-readable summary that can be archived alongside experiment checkpoints.

### CI integration

The repository exposes a `uv run python -m captionqa.evaluation.run ...` command that can
be dropped into GitHub Actions or other CI systems. Configure the workflow to download
model predictions and references, then invoke the evaluator with the same arguments used
locally to ensure consistent scoring.

## Windows Notes (Day 0 setup)

- GPU and drivers: Verify `nvidia-smi` works and CUDA is visible to PyTorch. You can force CPU with `--device cpu` via config if needed.
- FFmpeg: Confirm `ffmpeg -version` is on `PATH`.
- Symlinks: Hugging Face caches warn on Windows without Developer Mode. It’s safe to ignore or enable Developer Mode to allow symlinks.
- Storage planning: Full datasets can require 1–2 TB. Consider placing `datasets` on a large SSD and symlink from the repo (example: `datasets -> D:\CaptionQA\data`).
- Dev‑mini: A tiny local subset is scaffolded under `data/dev-mini/` for quick iteration:
  - `data/dev-mini/samples/dummy.mp4` – synthetic 360‑ish test clip
  - `configs/day0_deterministic.json` – disables HF models for deterministic fallback
  - Run: `./scripts/uv_run.ps1 python -m captionqa.captioning data/dev-mini/samples/dummy.mp4 --config configs/day0_deterministic.json`
  - Evaluate: `./scripts/uv_run.ps1 python -m captionqa.evaluation.run --task captioning --preds data/dev-mini/captioning/preds.jsonl --refs data/dev-mini/captioning/refs.jsonl --output-json data/dev-mini/captioning/summary.json`

## Known Good Versions

Pinned in `pyproject.toml` to ensure Windows-friendly wheels:

- torch 2.2.x, torchvision 0.17.x, torchaudio 2.2.x
- transformers 4.40–4.52, tokenizers 0.15–0.19
- opencv-python 4.8–4.9, huggingface-hub >= 0.23

These ranges avoid older Rust-build paths on Windows and keep CLIP vision models working with `AutoImageProcessor`.

## Multimodal Conditioning

The caption decoder accepts a fused audio/visual conditioning vector via a soft prompt prefix. A lightweight fusion MLP mean-pools each modality, concatenates, then projects to `soft_prompt_tokens * embed_dim` before generation. If models are unavailable, the CLI falls back to deterministic text-only output.

## Packaging Hygiene

Build artifacts like `*.egg-info/` are ignored. If you see stale metadata, run a clean build or delete any `*.egg-info` folders.

**Panorama Sampling**
- Keys (JSON `panorama` section):
  - `frame_rate`: frames per second to sample (0 = uniform ~32 frames).
  - `target_resolution`: `[height, width]` of projected frames.
  - `enable_projection`: if true, uses equirectangular→perspective projection.
  - `fov_degrees`: horizontal field of view for each perspective view.
  - `num_views`: number of yaw angles around the equator (0–360).
  - `num_pitch`: vertical bands to sample (1 = equator only).
  - `pitch_min_degrees`/`pitch_max_degrees`: pitch range (negative = down, positive = up).
  - `roll_degrees`: roll rotation (usually 0).

Example multi‑band config (three pitch bands, eight yaws):

```json
{
  "panorama": {
    "frame_rate": 1.0,
    "target_resolution": [256, 512],
    "enable_projection": true,
    "fov_degrees": 90.0,
    "num_views": 8,
    "num_pitch": 3,
    "pitch_min_degrees": -45.0,
    "pitch_max_degrees": 45.0,
    "roll_degrees": 0.0
  }
}
```

Run with the pinned uv wrapper:

```powershell
./scripts/uv_run.ps1 python -m captionqa.captioning path/to/video.mp4 --config configs/panorama_multiband.json
```
