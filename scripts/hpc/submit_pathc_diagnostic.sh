#!/usr/bin/env bash
# Path C diagnostic sprint — 50 MATH-500 problems, strict QRM protocol.
#
# Wave 1: Qwen + Llama BF16 @ 32k (reproduction prompt, seed 42, no repetition_penalty)
# Wave 2: Qwen BF16 @ 64k output cap (budget diagnostic)
#
# Usage (PARAM Rudra):
#   cd /scratch/$USER/reasoning-compression-lab
#   bash scripts/hpc/submit_pathc_diagnostic.sh
#
# After jobs finish:
#   bash scripts/hpc/report_pathc_diagnostic.sh
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
export QR
cd "$QR"

DATE_TAG="${QREASON_HPC_DATE:-$(date +%Y-%m-%d)}"
export QREASON_OUTPUT_ROOT="${QREASON_OUTPUT_ROOT:-$QR/outputs-hpc-diag-pathc-${DATE_TAG}}"
export QREASON_HPC_DATE="$DATE_TAG"
export QREASON_FRESH_RUN=1
export QREASON_INFERENCE_LIMIT=50
export QREASON_SLURM_EXCLUSIVE=0
export QREASON_CHECKPOINT_EVERY=10

echo "Path C diagnostic archive: $QREASON_OUTPUT_ROOT"
echo "Limit: ${QREASON_INFERENCE_LIMIT} problems per cell"
echo ""

echo "=== Wave 1: strict QRM 32k — Qwen-7B + Llama-8B (parallel split jobs) ==="
export QREASON_DECODING=configs/decoding/repro_qrm_strict.yaml
export QREASON_SLURM_TIME=12:00:00
bash scripts/hpc/submit_hpc_blocks.sh d01_pathc_32k_diagnostic

echo ""
echo "=== Wave 2: 64k budget diagnostic — Qwen-7B only (may queue behind wave 1) ==="
export QREASON_DECODING=configs/decoding/repro_qrm_64k.yaml
export QREASON_SLURM_TIME=24:00:00
bash scripts/hpc/submit_hpc_blocks.sh d02_pathc_64k_qwen

echo ""
echo "Done. Monitor: squeue -u \$USER"
echo "Report when complete: bash scripts/hpc/report_pathc_diagnostic.sh"
echo "Archive: $QREASON_OUTPUT_ROOT"