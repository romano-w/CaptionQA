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

After these steps, `python -m captionqa.data.download 360x --output <path>` will authenticate automatically. Repeat the access request step for future gated datasets as needed.

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
