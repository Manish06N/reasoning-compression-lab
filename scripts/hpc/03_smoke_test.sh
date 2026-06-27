#!/usr/bin/env bash
set -euo pipefail

export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

export HF_HOME="${HF_HOME:-$QR/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

CONDA_ROOT="${CONDA_ROOT:-/home/apps/MSCC/miniconda3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate qreason

export QREASON_MODEL_QWEN7B="${QREASON_MODEL_QWEN7B:-$QR/models/DeepSeek-R1-Distill-Qwen-7B}"

SMOKE_LIMIT="${SMOKE_LIMIT:-3}"
SMOKE_OUTPUT="${SMOKE_OUTPUT:-runs/raw/smoke_test.jsonl}"
SMOKE_MIN_FREE_GPU_MB="${SMOKE_MIN_FREE_GPU_MB:-30000}"
MAX_TOKENS_ARGS=()
if [[ -n "${SMOKE_MAX_TOKENS:-}" ]]; then
  MAX_TOKENS_ARGS=(--max-tokens "$SMOKE_MAX_TOKENS")
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  FREE_GPU_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ -n "$FREE_GPU_MB" && "$FREE_GPU_MB" -lt "$SMOKE_MIN_FREE_GPU_MB" ]]; then
    echo "Only ${FREE_GPU_MB} MiB GPU memory free; require ${SMOKE_MIN_FREE_GPU_MB} MiB. Exiting before vLLM init."
    exit 75
  fi
fi

python -u scripts/smoke_test.py \
  --cell-config configs/cells/level_a_bf16_seed0.json \
  --decoding-config configs/decoding/smoke_debug.yaml \
  --limit "$SMOKE_LIMIT" \
  --output "$SMOKE_OUTPUT" \
  "${MAX_TOKENS_ARGS[@]}"

echo "Smoke test output: $SMOKE_OUTPUT"
