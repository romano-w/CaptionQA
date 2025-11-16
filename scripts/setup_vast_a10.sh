#!/usr/bin/env bash

# One-time setup script for running CaptionQA baselines on a Vast.ai A10 instance.
# Usage (inside the cloned repo on the VM/container):
#   bash scripts/setup_vast_a10.sh
#
# Optional env vars:
#   HF_TOKEN   - Hugging Face token (for gated datasets / faster non-interactive login)
#   DATA_ROOT  - Root directory for downloaded datasets (default: /workspace/data)
#   HF_HOME    - Hugging Face cache directory (default: /workspace/hf_cache)

set -euo pipefail

has_dataset_payload() {
  python - <<'PY' "$1"
from pathlib import Path
import sys

root = Path(sys.argv[1])
if not root.exists():
    sys.exit(1)
for child in root.iterdir():
    if child.name != ".cache":
        sys.exit(0)
sys.exit(1)
PY
}

# Determine repo root (script is expected to live inside the cloned repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
echo "[setup] Repo root: ${REPO_ROOT}"

if command -v apt-get >/dev/null 2>&1; then
  echo "[setup] Installing system dependencies via apt-get..."
  apt-get update -y
  apt-get install -y --no-install-recommends ffmpeg git
fi

# Optional: configure git identity for this repo
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if ! git config user.name >/dev/null 2>&1; then
    read -rp "[setup] GitHub username (leave blank to skip git config): " GH_USER || true
    if [ -n "${GH_USER:-}" ]; then
      read -rp "[setup] Git email (e.g., you@example.com): " GH_EMAIL || true
      git config user.name "${GH_USER}"
      if [ -n "${GH_EMAIL:-}" ]; then
        git config user.email "${GH_EMAIL}"
      fi
      echo "[setup] Configured git user.name/user.email for this repo."
    fi
  fi
fi

if [ ! -d "captionqa" ]; then
  echo "[setup] Creating Python virtual environment (captionqa)..."
  python -m venv captionqa
fi

echo "[setup] Activating virtual environment..."
source captionqa/bin/activate

echo "[setup] Installing uv and project dependencies..."
python -m pip install --upgrade pip
pip install uv hf_transfer
uv pip install --editable .

HF_HOME_DEFAULT="/workspace/hf_cache"
HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
mkdir -p "${HF_HOME}"
export HF_HOME
echo "[setup] HF_HOME set to ${HF_HOME}"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "[setup] Logging in to Hugging Face using HF_TOKEN..."
  huggingface-cli login --token "${HF_TOKEN}" || true
else
  echo "[setup] HF_TOKEN not set; ensure you have already run 'huggingface-cli login'."
fi

DATA_ROOT_DEFAULT="/workspace/data"
DATA_ROOT="${DATA_ROOT:-$DATA_ROOT_DEFAULT}"
DATA_360X="${DATA_ROOT}/360x"

# Detect prior runs that stored the dataset under ${DATA_ROOT}/360x/360x/...
LEGACY_DATA_360X="${DATA_360X}/360x"
if [ -d "${LEGACY_DATA_360X}" ] && \
   [ ! -d "${DATA_360X}/360x_dataset_LR" ] && \
   [ ! -d "${DATA_360X}/360x_dataset_HR" ]; then
  if [ -d "${LEGACY_DATA_360X}/360x_dataset_LR" ] || [ -d "${LEGACY_DATA_360X}/360x_dataset_HR" ]; then
    echo "[setup] Detected nested 360x dataset under ${LEGACY_DATA_360X}; reusing that path."
    DATA_360X="${LEGACY_DATA_360X}"
  fi
fi

DATASET_LR_PATH="${DATA_360X}/360x_dataset_LR"
if ! has_dataset_payload "${DATASET_LR_PATH}"; then
  if [ -d "${DATASET_LR_PATH}" ]; then
    echo "[setup] 360x LR dataset directory exists but looks empty/incomplete; removing it."
    rm -rf "${DATASET_LR_PATH}"
  fi
  echo "[setup] Downloading 360x LR dataset to ${DATA_360X}..."
  python -m captionqa.data.download 360x --output "${DATA_360X}" --360x-resolution lr
else
  echo "[setup] 360x LR dataset already present at ${DATA_360X}."
fi

echo "[setup] Wiring data/raw symlink to ${DATA_ROOT}..."
mkdir -p "${REPO_ROOT}/data"
if [ -L "${REPO_ROOT}/data/raw" ] || [ -e "${REPO_ROOT}/data/raw" ]; then
  rm -rf "${REPO_ROOT}/data/raw"
fi
ln -s "${DATA_ROOT}" "${REPO_ROOT}/data/raw"

echo "[setup] Generating 360x dev-mini caption manifest..."
uv run python -m captionqa.datasets.x360_manifest \
  --root "${DATA_360X}/360x_dataset_LR/binocular" \
  --glob "*.mp4" \
  --limit 100 \
  --relative-to "${DATA_ROOT}" \
  --relative-prefix "data/raw" \
  --id-template '{parent_name}_{stem}' \
  --output data/eval/captioning/360x_devmini/manifest.jsonl

echo "[setup] Generating TAL-derived caption references..."
uv run python -m captionqa.datasets.x360_tal_refs \
  --manifest data/eval/captioning/360x_devmini/manifest.jsonl \
  --annotations-root "${DATA_360X}/360x_dataset_LR/TAL_annotations" \
  --output data/eval/captioning/360x_devmini/refs.jsonl

echo "[setup] Generating TAL-derived QA manifest and references..."
uv run python -m captionqa.datasets.x360_tal_qa \
  --manifest data/eval/captioning/360x_devmini/manifest.jsonl \
  --annotations-root "${DATA_360X}/360x_dataset_LR/TAL_annotations" \
  --output-manifest data/eval/qa/360x_devmini/manifest.jsonl \
  --output-refs data/eval/qa/360x_devmini/refs.jsonl \
  --max-questions-per-video 3

echo "[setup] Setup complete. You can now run:"
echo "  # Captioning baseline"
echo "  uv run python -m captionqa.captioning.baseline \\"
echo "    --manifest data/eval/captioning/360x_devmini/manifest.jsonl \\"
echo "    --engine qwen_vl \\"
echo "    --output-dir data/eval/captioning/360x_devmini \\"
echo "    --refs data/eval/captioning/360x_devmini/refs.jsonl"
echo
echo "  # QA baseline"
echo "  uv run python -m captionqa.qa.baseline_vqa \\"
echo "    --manifest data/eval/qa/360x_devmini/manifest.jsonl \\"
echo "    --refs data/eval/qa/360x_devmini/refs.jsonl \\"
echo "    --output-dir data/eval/qa/360x_devmini"
