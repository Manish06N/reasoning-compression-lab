#!/usr/bin/env bash
# Submit HPC 2×A100 publication blocks to SLURM (≤48 h each).
#
# Usage (on PARAM Rudra):
#   export QR=/scratch/$USER/reasoning-compression-lab
#   cd $QR
#   bash scripts/hpc/submit_hpc_blocks.sh
#   bash scripts/hpc/submit_hpc_blocks.sh b01        # one block only
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"
mkdir -p logs/slurm

submit_block() {
  local block="$1"
  local job_name="qreason-${block}"
  echo "Submitting $block ..."
  sbatch --job-name="$job_name" \
    --output="logs/slurm/${block}_%j.out" \
    --error="logs/slurm/${block}_%j.err" \
    --time=47:00:00 \
    --partition=gpu \
    --cpus-per-task=16 \
    --gres=gpu:2 \
    --wrap="bash scripts/hpc/run_hpc_2a100_publication.sh ${block}"
}

BLOCK="${1:-all}"
case "$BLOCK" in
  all)
    submit_block b01_parallel_bf16_anchors
    echo ""
    echo "b02_gpqa_fp8 not submitted — run manually after GPQA gate:"
    echo "  sbatch slurm/hpc_2a100_b02_gpqa.slurm"
    ;;
  b01|b01_parallel_bf16_anchors)
    submit_block b01_parallel_bf16_anchors
    ;;
  b02|b02_gpqa_fp8)
    sbatch slurm/hpc_2a100_b02_gpqa.slurm
    ;;
  *)
    echo "Unknown block: $BLOCK"
    exit 1
    ;;
esac

echo "Done. Monitor: squeue -u \$USER"
