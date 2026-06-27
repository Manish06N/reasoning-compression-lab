#!/usr/bin/env bash
# Gate check: GPTQ-4 model must exist before running level_a_gptq4_seed0.
set -euo pipefail

export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

MODEL_PATH="${QREASON_MODEL_QWEN7B_GPTQ4:-$QR/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4}"

echo "Checking GPTQ-4 model path: $MODEL_PATH"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "FAIL: directory does not exist."
  echo "See docs/GPTQ4_PREP.md for download/quantize options."
  exit 1
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "FAIL: config.json not found in $MODEL_PATH"
  exit 1
fi

WEIGHTS=$(find "$MODEL_PATH" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' -o -name '*.gptq' \) | head -1)
if [[ -z "$WEIGHTS" ]]; then
  echo "FAIL: no weight files found in $MODEL_PATH"
  exit 1
fi

echo "OK: GPTQ-4 model path verified."
echo "  config.json: present"
echo "  weights: $(basename "$WEIGHTS") (+ possibly more)"
echo ""
echo "Next:"
echo "  python scripts/run_inference.py --cell-config configs/cells/level_a_gptq4_seed0.json"
