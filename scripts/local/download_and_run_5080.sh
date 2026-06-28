#!/usr/bin/env bash
# Download all RTX 5080-fit models, then start sequential experiments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

DATE_TAG="${QREASON_5080_DATE:-2026-06-28}"
export QREASON_OUTPUT_ROOT="${QREASON_OUTPUT_ROOT:-$QR/outputs-win5080-${DATE_TAG}}"
mkdir -p "$QREASON_OUTPUT_ROOT/logs"

DOWNLOAD_LOG="$QREASON_OUTPUT_ROOT/logs/download.log"
MASTER_LOG="$QREASON_OUTPUT_ROOT/logs/master.log"

echo "=== $(date -Is) Phase 1: Download all 5080 models ===" | tee "$DOWNLOAD_LOG"

# Verifier BF16 (already on disk from phase0 — hf download resumes if partial).
if [[ ! -f "$QREASON_MODEL_QWEN15B/config.json" ]]; then
  echo "=== Downloading Qwen-1.5B BF16 ===" | tee -a "$DOWNLOAD_LOG"
  hf download "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --local-dir "$QREASON_MODEL_QWEN15B" 2>&1 | tee -a "$DOWNLOAD_LOG"
fi

echo "=== Downloading 5080 quant roster ===" | tee -a "$DOWNLOAD_LOG"
bash "$SCRIPT_DIR/download_models.sh" 5080 2>&1 | tee -a "$DOWNLOAD_LOG"

echo "" | tee -a "$DOWNLOAD_LOG"
echo "=== Download inventory ===" | tee -a "$DOWNLOAD_LOG"
du -sh "$QR/models/"* 2>&1 | tee -a "$DOWNLOAD_LOG"

echo "" | tee -a "$DOWNLOAD_LOG"
echo "=== $(date -Is) Phase 2: Start experiments ===" | tee -a "$MASTER_LOG"
exec bash "$SCRIPT_DIR/run_all_5080_phases.sh" "$@" 2>&1 | tee -a "$MASTER_LOG"
