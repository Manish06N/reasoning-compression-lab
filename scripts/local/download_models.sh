#!/usr/bin/env bash
# Download canonical Paper 1 model checkpoints.
# Usage: bash scripts/local/download_models.sh [phase0|gptq4|levelb|all]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

TARGET="${1:-phase0}"

download_one() {
  local repo="$1"
  local dest="$2"
  mkdir -p "$dest"
  echo "=== Downloading $repo -> $dest ==="
  hf download "$repo" --local-dir "$dest"
}

case "$TARGET" in
  phase0)
    download_one "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
      "${QREASON_MODEL_QWEN15B:-$QR/models/DeepSeek-R1-Distill-Qwen-1.5B}"
    download_one "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
      "${QREASON_MODEL_QWEN7B:-$QR/models/DeepSeek-R1-Distill-Qwen-7B}"
    ;;
  gptq4)
    # ruikangliu repo removed from HF; RedHatAI is the canonical 5080/HPC GPTQ-4 source.
    download_one "RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16" \
      "${QREASON_MODEL_QWEN7B_GPTQ4:-$QR/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4}"
    ;;
  5080)
    # Full RTX 5080 roster — quantized + small BF16 only (no 7B/8B BF16 full runs).
    download_one "RedHatAI/DeepSeek-R1-Distill-Qwen-1.5B-FP8-dynamic" \
      "${QREASON_MODEL_QWEN15B_FP8:-$QR/models/DeepSeek-R1-Distill-Qwen-1.5B-FP8}"
    download_one "jakiAJK/DeepSeek-R1-Distill-Qwen-1.5B_AWQ" \
      "${QREASON_MODEL_QWEN15B_AWQ4:-$QR/models/DeepSeek-R1-Distill-Qwen-1.5B-AWQ-4}"
    download_one "RedHatAI/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w4a16" \
      "${QREASON_MODEL_QWEN15B_GPTQ4:-$QR/models/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ-4}"
    download_one "RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic" \
      "${QREASON_MODEL_QWEN7B_FP8:-$QR/models/DeepSeek-R1-Distill-Qwen-7B-FP8}"
    download_one "jakiAJK/DeepSeek-R1-Distill-Qwen-7B_AWQ" \
      "${QREASON_MODEL_QWEN7B_AWQ4:-$QR/models/DeepSeek-R1-Distill-Qwen-7B-AWQ-4}"
    download_one "RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16" \
      "${QREASON_MODEL_QWEN7B_GPTQ4:-$QR/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4}"
    download_one "irish-quant/deepseek-ai-DeepSeek-R1-Distill-Qwen-7B-3bit" \
      "${QREASON_MODEL_QWEN7B_GPTQ3:-$QR/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-3}"
    download_one "RedHatAI/DeepSeek-R1-Distill-Llama-8B-FP8-dynamic" \
      "${QREASON_MODEL_LLAMA8B_FP8:-$QR/models/DeepSeek-R1-Distill-Llama-8B-FP8}"
    download_one "jakiAJK/DeepSeek-R1-Distill-Llama-8B_AWQ" \
      "${QREASON_MODEL_LLAMA8B_AWQ4:-$QR/models/DeepSeek-R1-Distill-Llama-8B-AWQ-4}"
    download_one "RedHatAI/DeepSeek-R1-Distill-Llama-8B-quantized.w4a16" \
      "${QREASON_MODEL_LLAMA8B_GPTQ4:-$QR/models/DeepSeek-R1-Distill-Llama-8B-GPTQ-4}"
    ;;
  levelb)
    download_one "RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic" \
      "${QREASON_MODEL_QWEN7B_FP8:-$QR/models/DeepSeek-R1-Distill-Qwen-7B-FP8}"
    download_one "jakiAJK/DeepSeek-R1-Distill-Qwen-7B_AWQ" \
      "${QREASON_MODEL_QWEN7B_AWQ4:-$QR/models/DeepSeek-R1-Distill-Qwen-7B-AWQ-4}"
    download_one "irish-quant/deepseek-ai-DeepSeek-R1-Distill-Qwen-7B-3bit" \
      "${QREASON_MODEL_QWEN7B_GPTQ3:-$QR/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-3}"
    ;;
  levelc)
    download_one "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
      "${QREASON_MODEL_LLAMA8B:-$QR/models/DeepSeek-R1-Distill-Llama-8B}"
    download_one "RedHatAI/DeepSeek-R1-Distill-Llama-8B-FP8-dynamic" \
      "${QREASON_MODEL_LLAMA8B_FP8:-$QR/models/DeepSeek-R1-Distill-Llama-8B-FP8}"
    download_one "jakiAJK/DeepSeek-R1-Distill-Llama-8B_AWQ" \
      "${QREASON_MODEL_LLAMA8B_AWQ4:-$QR/models/DeepSeek-R1-Distill-Llama-8B-AWQ-4}"
    download_one "RedHatAI/DeepSeek-R1-Distill-Llama-8B-quantized.w4a16" \
      "${QREASON_MODEL_LLAMA8B_GPTQ4:-$QR/models/DeepSeek-R1-Distill-Llama-8B-GPTQ-4}"
    ;;
  all)
    bash "$SCRIPT_DIR/download_models.sh" phase0
    bash "$SCRIPT_DIR/download_models.sh" gptq4
    bash "$SCRIPT_DIR/download_models.sh" levelb
    bash "$SCRIPT_DIR/download_models.sh" levelc
    ;;
    *)
    echo "Unknown target: $TARGET (phase0|gptq4|levelb|levelc|5080|all)"
    exit 1
    ;;
esac

echo "Download complete: $TARGET"
