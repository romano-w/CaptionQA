# CaptionQA

360° Panoramic Video Captioning + QA system for BLV accessibility.

## Getting Started

### Prerequisite

Install [uv](https://docs.astral.sh/uv/) once on your machine. On Windows PowerShell you can run either:

```powershell
winget install --id Astral.Uv -e
```

or, if you prefer pipx:

```powershell
pipx install uv
```

### Environment Setup

After uv is available, open a PowerShell session in the repository root and run:

```powershell
uv venv captionqa
.\captionqa\Scripts\Activate.ps1
uv pip install --editable .
```

Launch VS Code with the Jupyter extension and open `notebooks/quickstart.ipynb`; you can stay inside Notebook view without starting a separate Jupyter server.

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

## Authors

- Will Romano
- Ethan Baird
