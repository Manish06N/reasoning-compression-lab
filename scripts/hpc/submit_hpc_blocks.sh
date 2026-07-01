#!/usr/bin/env bash
# Submit HPC 2×A100 publication blocks to SLURM (≤48 h each).
#
# Usage (on PARAM Rudra):
#   export QR=/scratch/$USER/reasoning-compression-lab
#   cd $QR && git pull
#   bash scripts/hpc/submit_hpc_blocks.sh        # submit b01-b06 only
#   bash scripts/hpc/submit_hpc_blocks.sh b08    # optional future Qwen-1.5B block
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"
mkdir -p logs/slurm

submit_2gpu() {
  local block="$1"
  local block_file="$QR/configs/machine_split/hpc_blocks/${block}.sh"
  echo "Submitting $block as independent 1-GPU cell jobs ..."
  # shellcheck disable=SC1090
  source "$block_file"
  for entry in "${HPC_BLOCK_CELLS[@]}"; do
    local cfg="${entry#*:}"
    local cell_id
    cell_id="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['cell_id'])" "$cfg")"
    echo "Submitting $block / $cell_id (1xA100) ..."
    sbatch --job-name="qreason-${cell_id}" \
      --output="logs/slurm/${block}_${cell_id}_%j.out" \
      --error="logs/slurm/${block}_${cell_id}_%j.err" \
      --time=47:00:00 \
      --partition=gpu \
      --cpus-per-task=8 \
      --gres=gpu:1 \
      --wrap="bash scripts/hpc/run_hpc_2a100_publication.sh cell ${cfg} ${block}"
  done
}

submit_1gpu() {
  local block="$1"
  echo "Submitting $block (1×A100) ..."
  sbatch --job-name="qreason-${block}" \
    --output="logs/slurm/${block}_%j.out" \
    --error="logs/slurm/${block}_%j.err" \
    --time=47:00:00 \
    --partition=gpu \
    --cpus-per-task=8 \
    --gres=gpu:1 \
    --wrap="bash scripts/hpc/run_hpc_2a100_publication.sh ${block}"
}

BLOCK="${1:-all}"
case "$BLOCK" in
  all)
    submit_2gpu b01_parallel_bf16_anchors
    submit_2gpu b02_parallel_fp8
    submit_2gpu b03_parallel_awq4
    submit_2gpu b04_parallel_gptq4
    submit_1gpu b05_single_gptq3
    submit_1gpu b06_single_gsm8k
    echo ""
    echo "b07_gpqa_fp8 NOT submitted — run after GPQA gate:"
    echo "  sbatch slurm/hpc_2a100_b07_gpqa.slurm"
    echo "b08-b09 Qwen-1.5B NOT submitted by default — future HPC-only lower-bound jobs:"
    echo "  bash scripts/hpc/submit_hpc_blocks.sh b08"
    echo "  bash scripts/hpc/submit_hpc_blocks.sh b09"
    ;;
  b01|b01_parallel_bf16_anchors) submit_2gpu b01_parallel_bf16_anchors ;;
  b02|b02_parallel_fp8) submit_2gpu b02_parallel_fp8 ;;
  b03|b03_parallel_awq4) submit_2gpu b03_parallel_awq4 ;;
  b04|b04_parallel_gptq4) submit_2gpu b04_parallel_gptq4 ;;
  b05|b05_single_gptq3) submit_1gpu b05_single_gptq3 ;;
  b06|b06_single_gsm8k) submit_1gpu b06_single_gsm8k ;;
  b07|b07_gpqa_fp8) submit_1gpu b07_gpqa_fp8 ;;
  b08|b08_qwen15b_bf16_fp8) submit_2gpu b08_qwen15b_bf16_fp8 ;;
  b09|b09_qwen15b_awq4_gptq4) submit_2gpu b09_qwen15b_awq4_gptq4 ;;
  *)
    echo "Unknown block: $BLOCK"
    bash scripts/hpc/run_hpc_2a100_publication.sh list
    exit 1
    ;;
esac

echo "Done. Monitor: squeue -u \$USER"
