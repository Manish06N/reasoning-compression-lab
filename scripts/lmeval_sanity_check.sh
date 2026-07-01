#!/usr/bin/env bash
# Optional one-cell sanity check: lm-evaluation-harness vs reasoning-compression-lab pass@1.
# Reference: external_repos/04-inference-and-eval-tools/lm-evaluation-harness/
#
# Install (HPC qreason env, optional):
#   pip install lm_eval[vllm]
#
# Usage:
#   bash scripts/lmeval_sanity_check.sh MODEL_PATH [LIMIT]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${1:?Usage: lmeval_sanity_check.sh MODEL_PATH [LIMIT]}"
LIMIT="${2:-10}"
OUT_DIR="${ROOT}/runs/lmeval_sanity"
mkdir -p "$OUT_DIR"

if ! command -v lm_eval >/dev/null 2>&1; then
  echo "lm_eval not installed. Run: pip install 'lm_eval[vllm]'"
  echo "See docs/reference_notes/LMEVAL_SANITY.md"
  exit 1
fi

echo "Running lm_eval gsm8k (limit=$LIMIT) on $MODEL ..."
lm_eval run \
  --model vllm \
  --model_args "pretrained=${MODEL},dtype=bfloat16,max_model_len=8192,gpu_memory_utilization=0.85,enforce_eager=True" \
  --tasks gsm8k \
  --batch_size 1 \
  --limit "$LIMIT" \
  --output_path "$OUT_DIR" \
  --log_samples

echo ""
echo "Compare lm_eval accuracy to our harness on the same model:"
echo "  python scripts/run_inference.py --cell-config configs/cells/level_b_qwen7b_bf16_gsm8k_seed0.json --limit $LIMIT"
echo "  python scripts/score_run.py --input runs/raw/<cell_id>.jsonl"
echo ""
echo "Or use: python scripts/lmeval_compare_summary.py --lmeval-dir $OUT_DIR --summary results/<cell>_summary.json"
