#!/usr/bin/env bash

# One-time setup script for running CaptionQA baselines on a Vast.ai A10 instance.
# Usage (inside the cloned repo on the VM/container):
#   bash scripts/setup_vast_gpu.sh
#
# Optional env vars:
#   HF_TOKEN   - Hugging Face token (for gated datasets / faster non-interactive login)
#   DATA_ROOT  - Root directory for downloaded datasets (default: /workspace/data)
#   HF_HOME    - Hugging Face cache directory (default: /workspace/hf_cache)
#   SKIP_BASELINES - Set to 1 to skip running captioning/QA baselines

set -euo pipefail

supports_color=false
if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
  if color_count=$(tput colors 2>/dev/null); then
    if [ -n "${color_count}" ] && [ "${color_count}" -ge 8 ]; then
      supports_color=true
    fi
  fi
fi

if [ "${supports_color}" = true ]; then
  STYLE_BOLD="$(tput bold)"
  STYLE_RESET="$(tput sgr0)"
  COLOR_CYAN="$(tput setaf 6)"
  COLOR_GREEN="$(tput setaf 2)"
  COLOR_YELLOW="$(tput setaf 3)"
else
  STYLE_BOLD=""
  STYLE_RESET=""
  COLOR_CYAN=""
  COLOR_GREEN=""
  COLOR_YELLOW=""
fi

SETUP_TAG_INFO="${STYLE_BOLD}${COLOR_CYAN}[setup]${STYLE_RESET}"
SETUP_TAG_WARN="${STYLE_BOLD}${COLOR_YELLOW}[setup]${STYLE_RESET}"
SETUP_TAG_SUCCESS="${STYLE_BOLD}${COLOR_GREEN}[setup]${STYLE_RESET}"
SETUP_PROMPT_PREFIX="${SETUP_TAG_INFO:-[setup]}"

log_step() {
  printf "%s %s\n" "${SETUP_TAG_INFO}" "$*"
}

log_warn() {
  printf "%s %s\n" "${SETUP_TAG_WARN}" "$*"
}

log_success() {
  printf "%s %s\n" "${SETUP_TAG_SUCCESS}" "$*"
}

normalize_data_root() {
  local candidate="$1"
  while [ -d "${candidate}/360x" ] && \
        [ ! -d "${candidate}/360x_dataset_LR" ] && \
        [ ! -d "${candidate}/360x_dataset_HR" ]; do
    if [ -d "${candidate}/360x/360x_dataset_LR" ] || \
       [ -d "${candidate}/360x/360x_dataset_HR" ] || \
       [ -d "${candidate}/360x/360x" ]; then
      candidate="${candidate}/360x"
    else
      break
    fi
  done
  printf "%s" "${candidate}"
}

has_dataset_payload() {
  python - <<'PY' "$1"
from pathlib import Path
import sys

root = Path(sys.argv[1])
if not root.exists():
    sys.exit(1)

binocular = root / "binocular"
annotations = root / "TAL_annotations"

if not binocular.is_dir() or not annotations.is_dir():
    sys.exit(1)

has_video = any(binocular.rglob("*.mp4"))
has_annotations = any(annotations.rglob("*.json"))

if has_video and has_annotations:
    sys.exit(0)

sys.exit(1)
PY
}

# Determine repo root (script is expected to live inside the cloned repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
log_step "Repo root: ${REPO_ROOT}"

if command -v apt-get >/dev/null 2>&1; then
  log_step "Installing system dependencies via apt-get..."
  apt-get update -y
  apt-get install -y --no-install-recommends ffmpeg git
fi

# Optional: configure git identity for this repo
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if ! git config user.name >/dev/null 2>&1; then
    read -rp "${SETUP_PROMPT_PREFIX} GitHub username (leave blank to skip git config): " GH_USER || true
    if [ -n "${GH_USER:-}" ]; then
      read -rp "${SETUP_PROMPT_PREFIX} Git email (e.g., you@example.com): " GH_EMAIL || true
      git config user.name "${GH_USER}"
      if [ -n "${GH_EMAIL:-}" ]; then
        git config user.email "${GH_EMAIL}"
      fi
      log_success "Configured git user.name/user.email for this repo."
    fi
  fi
fi

if [ ! -d "captionqa" ]; then
  log_step "Creating Python virtual environment (captionqa)..."
  python -m venv captionqa
fi

log_step "Activating virtual environment..."
source captionqa/bin/activate

log_step "Installing uv and project dependencies..."
python -m pip install --upgrade pip
pip install uv hf_transfer
uv pip install --editable .

HF_HOME_DEFAULT="/workspace/hf_cache"
HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
mkdir -p "${HF_HOME}"
export HF_HOME
log_step "HF_HOME set to ${HF_HOME}"

if [ -n "${HF_TOKEN:-}" ]; then
  log_step "Logging in to Hugging Face using HF_TOKEN..."
  huggingface-cli login --token "${HF_TOKEN}" || true
else
  log_warn "HF_TOKEN not set; ensure you have already run 'huggingface-cli login'."
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
    log_step "Detected nested 360x dataset under ${LEGACY_DATA_360X}; reusing that path."
    DATA_360X="${LEGACY_DATA_360X}"
  fi
fi

NORMALIZED_DATA_360X="$(normalize_data_root "${DATA_360X}")"
if [ "${NORMALIZED_DATA_360X}" != "${DATA_360X}" ]; then
  log_step "Normalized 360x root to ${NORMALIZED_DATA_360X}"
  DATA_360X="${NORMALIZED_DATA_360X}"
fi

DATASET_LR_PATH="${DATA_360X}/360x_dataset_LR"
if ! has_dataset_payload "${DATASET_LR_PATH}"; then
  if [ -d "${DATASET_LR_PATH}" ]; then
    log_warn "360x LR dataset directory exists but looks empty/incomplete; removing it."
    rm -rf "${DATASET_LR_PATH}"
  fi
  log_step "Downloading 360x LR dataset to ${DATA_360X}..."
  python -m captionqa.data.download 360x --output "${DATA_360X}" --360x-resolution lr
else
  log_step "360x LR dataset already present at ${DATA_360X}."
fi

log_step "Wiring data/raw symlink to ${DATA_ROOT}..."
mkdir -p "${REPO_ROOT}/data"
if [ -L "${REPO_ROOT}/data/raw" ] || [ -e "${REPO_ROOT}/data/raw" ]; then
  rm -rf "${REPO_ROOT}/data/raw"
fi
ln -s "${DATA_ROOT}" "${REPO_ROOT}/data/raw"

log_step "Generating 360x dev-mini caption manifest..."
uv run python -m captionqa.datasets.x360_manifest \
  --root "${DATA_360X}/360x_dataset_LR/binocular" \
  --glob "*.mp4" \
  --limit 100 \
  --relative-to "${DATA_ROOT}" \
  --relative-prefix "data/raw" \
  --id-template '{parent_name}_{stem}' \
  --output data/eval/captioning/360x_devmini/manifest.jsonl

log_step "Generating TAL-derived caption references..."
uv run python -m captionqa.datasets.x360_tal_refs \
  --manifest data/eval/captioning/360x_devmini/manifest.jsonl \
  --annotations-root "${DATA_360X}/360x_dataset_LR/TAL_annotations" \
  --output data/eval/captioning/360x_devmini/refs.jsonl

log_step "Generating TAL-derived QA manifest and references..."
uv run python -m captionqa.datasets.x360_tal_qa \
  --manifest data/eval/captioning/360x_devmini/manifest.jsonl \
  --annotations-root "${DATA_360X}/360x_dataset_LR/TAL_annotations" \
  --output-manifest data/eval/qa/360x_devmini/manifest.jsonl \
  --output-refs data/eval/qa/360x_devmini/refs.jsonl \
  --max-questions-per-video 3

run_captioning_baseline() {
  local manifest="data/eval/captioning/360x_devmini/manifest.jsonl"
  local refs="data/eval/captioning/360x_devmini/refs.jsonl"
  local output_dir="data/eval/captioning/360x_devmini"

  log_step "Running captioning baseline (output -> ${output_dir})..."
  uv run python -m captionqa.captioning.baseline \
    --manifest "${manifest}" \
    --engine qwen_vl \
    --refs "${refs}" \
    --output-dir "${output_dir}"
}

run_qa_baseline() {
  local manifest="data/eval/qa/360x_devmini/manifest.jsonl"
  local refs="data/eval/qa/360x_devmini/refs.jsonl"
  local output_dir="data/eval/qa/360x_devmini"

  log_step "Running QA baseline (output -> ${output_dir})..."
  uv run python -m captionqa.qa.baseline_vqa \
    --manifest "${manifest}" \
    --refs "${refs}" \
    --output-dir "${output_dir}"
}

if [ "${SKIP_BASELINES:-0}" = "1" ]; then
  log_warn "SKIP_BASELINES=1 detected; skipping captioning/QA runs."
else
  run_captioning_baseline
  run_qa_baseline
fi

log_success "Setup complete. Outputs live under data/eval/captioning/360x_devmini and data/eval/qa/360x_devmini."
