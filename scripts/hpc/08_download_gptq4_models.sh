#!/usr/bin/env bash
# Download pre-quantized GPTQ-4 / AWQ-4 weights from QRM HuggingFace collection (b04 prep).
# Reference: external_repos/01-core-baselines/Quantized-Reasoning-Models/README.md#modelzoo
set -euo pipefail

export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

QWEN7B_GPTQ4="${QREASON_MODEL_QWEN7B_GPTQ4:-$QR/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4}"
LLAMA8B_GPTQ4="${QREASON_MODEL_LLAMA8B_GPTQ4:-$QR/models/DeepSeek-R1-Distill-Llama-8B-GPTQ-4}"

# HF repo IDs from ruikangliu collection — verify on model card if download fails.
QWEN7B_GPTQ4_REPO="${QWEN7B_GPTQ4_REPO:-ruikangliu/DeepSeek-R1-Distill-Qwen-7B-GPTQ-W4G128}"
LLAMA8B_GPTQ4_REPO="${LLAMA8B_GPTQ4_REPO:-ruikangliu/DeepSeek-R1-Distill-Llama-8B-GPTQ-W4G128}"

usage() {
  echo "Usage: bash scripts/hpc/08_download_gptq4_models.sh [qwen|llama|both]"
  echo ""
  echo "Downloads GPTQ W4G128 checkpoints for b04_parallel_gptq4."
  echo "Option B (self-quantize): see docs/GPTQ4_PREP.md and QRM scripts/data/gen_calib.sh"
  exit "${1:-0}"
}

TARGET="${1:-both}"

download_one() {
  local repo="$1"
  local dest="$2"
  if [[ -f "$dest/config.json" ]]; then
    echo "SKIP: $dest already has config.json"
    return 0
  fi
  mkdir -p "$dest"
  echo "Downloading $repo -> $dest"
  huggingface-cli download "$repo" --local-dir "$dest"
}

case "$TARGET" in
  qwen)
    download_one "$QWEN7B_GPTQ4_REPO" "$QWEN7B_GPTQ4"
    ;;
  llama)
    download_one "$LLAMA8B_GPTQ4_REPO" "$LLAMA8B_GPTQ4"
    ;;
  both)
    download_one "$QWEN7B_GPTQ4_REPO" "$QWEN7B_GPTQ4"
    download_one "$LLAMA8B_GPTQ4_REPO" "$LLAMA8B_GPTQ4"
    ;;
  -h|--help|help)
    usage 0
    ;;
  *)
    echo "Unknown target: $TARGET"
    usage 1
    ;;
esac

echo ""
echo "Verify before b04:"
echo "  export QREASON_MODEL_QWEN7B_GPTQ4=$QWEN7B_GPTQ4"
echo "  bash scripts/hpc/06_verify_gptq4_model.sh"
