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

## Goals

- Spatially grounded event captions
- Interactive natural language questions

## Authors

- Will Romano
- Ethan Baird
