#!/usr/bin/env bash
# Submit Path C parity pilot (n=10 Qwen) with QRM serving-stack fixes.
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

DATE_TAG="${QREASON_HPC_DATE:-$(date +%Y-%m-%d)}"
export QREASON_OUTPUT_ROOT="${QREASON_OUTPUT_ROOT:-$QR/outputs-hpc-diag-pathc-parity-${DATE_TAG}}"
export QREASON_HPC_DATE="$DATE_TAG"
export QREASON_FRESH_RUN=1
export QREASON_INFERENCE_LIMIT=10
export QREASON_SLURM_EXCLUSIVE=0
export QREASON_CHECKPOINT_EVERY=5
export QREASON_DECODING=configs/decoding/repro_qrm_strict.yaml
export QREASON_SLURM_TIME=4:00:00

echo "Parity pilot archive: $QREASON_OUTPUT_ROOT"
echo "Verify config (no GPU): python scripts/hpc/qrm_parity/verify_stack_parity.py"
echo ""

bash scripts/hpc/submit_hpc_blocks.sh d03_pathc_parity_pilot

echo ""
echo "After job completes:"
echo "  python scripts/score_run.py --input $QREASON_OUTPUT_ROOT/raw/diag_qwen7b_bf16_math500_seed42_n10_parity.jsonl \\"
echo "    --summary $QREASON_OUTPUT_ROOT/results/diag_qwen7b_bf16_math500_seed42_n10_parity_summary.json --skip-calibration"
echo "  python scripts/hpc/qrm_parity/compare_side_by_side.py \\"
echo "    --parity-archive outputs-hpc-diag-pathc-parity-${DATE_TAG} --limit 10"