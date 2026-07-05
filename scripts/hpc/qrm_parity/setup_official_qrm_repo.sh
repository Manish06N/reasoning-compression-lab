#!/usr/bin/env bash
# Clone official QRM repo for stack cross-check (login node safe).
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
DEST="${QRM_REPO_DIR:-$QR/external/Quantized-Reasoning-Models}"
REF="${QRM_REPO_REF:-main}"

mkdir -p "$(dirname "$DEST")"
if [[ -d "$DEST/.git" ]]; then
  echo "QRM repo exists: $DEST"
  git -C "$DEST" fetch origin "$REF" --depth 1
  git -C "$DEST" checkout "$REF"
else
  git clone --depth 1 --branch "$REF" \
    https://github.com/ruikangliu/Quantized-Reasoning-Models.git "$DEST"
fi

echo ""
echo "Official QRM repo ready at: $DEST"
echo ""
echo "Paper §3.1 settings (inference.py):"
echo "  temperature=0.6 top_p=0.95 max_new_tokens=32768 max_model_length=32768"
echo "  seed=42 enforce_eager=True gpu_memory_utilization=0.9"
echo "  enable_prefix_caching=False enable_chunked_prefill=False"
echo "  repetition_penalty: NOT SET"
echo ""
echo "Side-by-side on GPU (requires separate venv — do NOT mix with qreason):"
echo "  export MODEL=$QR/models/DeepSeek-R1-Distill-Qwen-7B"
echo "  cd $DEST"
echo "  python inference.py --model \"\$MODEL\" --dataset MATH-500 --max_samples 10 --seed 42"
echo ""
echo "Compare traces with our harness:"
echo "  python $QR/scripts/hpc/qrm_parity/compare_side_by_side.py --limit 10"