#!/usr/bin/env bash
# Submit HPC 2×A100 publication blocks to SLURM (≤48 h each).
#
# Usage (on PARAM Rudra):
#   export QR=/scratch/$USER/reasoning-compression-lab
#   cd $QR && git pull
#   bash scripts/hpc/submit_hpc_blocks.sh              # submit b01 only (default)
#   bash scripts/hpc/submit_hpc_blocks.sh b01 --fresh  # fresh archive for b01
#   bash scripts/hpc/submit_hpc_blocks.sh all_blocks   # b01-b06 soak (violates b01 gate)
#   bash scripts/hpc/submit_hpc_blocks.sh b08          # optional future Qwen-1.5B block
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
export QR
cd "$QR"
mkdir -p logs/slurm

FRESH_FLAG=""
BLOCK=""
for arg in "$@"; do
  case "$arg" in
    --fresh) FRESH_FLAG=1 ;;
    *)
      if [[ -z "$BLOCK" ]]; then
        BLOCK="$arg"
      fi
      ;;
  esac
done
BLOCK="${BLOCK:-all}"

export QREASON_HPC_DATE="${QREASON_HPC_DATE:-$(date +%Y-%m-%d)}"
export QREASON_OUTPUT_ROOT="${QREASON_OUTPUT_ROOT:-$QR/outputs-hpc-2a100-main-${QREASON_HPC_DATE}}"

if [[ -n "$FRESH_FLAG" ]]; then
  export QREASON_FRESH_RUN=1
else
  unset QREASON_FRESH_RUN || true
fi

SBATCH_EXPORT="ALL,QR=${QR},QREASON_OUTPUT_ROOT=${QREASON_OUTPUT_ROOT},QREASON_HPC_DATE=${QREASON_HPC_DATE}"
if [[ -n "${QREASON_FRESH_RUN:-}" ]]; then
  SBATCH_EXPORT="${SBATCH_EXPORT},QREASON_FRESH_RUN=${QREASON_FRESH_RUN}"
else
  SBATCH_EXPORT="${SBATCH_EXPORT},QREASON_FRESH_RUN="
fi

# PARAM Rudra QOS: MaxTRESPerUser = gres/gpu=2. ragpu nodes have 2× A100.
# NEVER use --exclusive on split 1-GPU cell jobs: SLURM counts exclusive as BOTH GPUs
# on the node toward your quota, so 1 running + 1 pending exclusive → 3 counted → QOSMaxGRESPerUser.
# Use dirty_nodes.txt + VRAM preflight instead. See docs/PARAM_RUDRA_SLURM.md.
export QREASON_SLURM_EXCLUSIVE="${QREASON_SLURM_EXCLUSIVE:-0}"

resolve_slurm_excludes() {
  local -a nodes=()
  local dirty_file="${QREASON_OUTPUT_ROOT}/metadata/dirty_nodes.txt"
  local merged=""

  if [[ -n "${QREASON_SLURM_EXCLUDE:-}" ]]; then
    IFS="," read -r -a nodes <<<"${QREASON_SLURM_EXCLUDE}"
  fi
  if [[ -f "$dirty_file" ]]; then
    while IFS= read -r node; do
      [[ -n "$node" ]] && nodes+=("$node")
    done <"$dirty_file"
  fi

  if [[ "${#nodes[@]}" -eq 0 ]]; then
    return 0
  fi

  merged="$(printf '%s\n' "${nodes[@]}" | awk 'NF && !seen[$0]++' | paste -sd, -)"
  if [[ -n "$merged" ]]; then
    SBATCH_EXCLUDE_ARGS=(--exclude="$merged")
    echo "SLURM exclude list: $merged"
  fi
}

SBATCH_EXCLUDE_ARGS=()
resolve_slurm_excludes

ensure_autopush() {
  if [[ "${QREASON_ENABLE_AUTOPUSH:-}" != "1" ]]; then
    echo "HPC output autopush disabled (set QREASON_ENABLE_AUTOPUSH=1 to enable)."
    return 0
  fi
  if ! command -v tmux >/dev/null 2>&1; then
    echo "WARN: tmux not found; HPC output autopush daemon not started." >&2
    return 0
  fi
  if tmux has-session -t hpc_git_autopush 2>/dev/null; then
    return 0
  fi
  echo "Starting HPC output autopush daemon (results + backups) ..."
  tmux new-session -d -s hpc_git_autopush \
    "cd '$QR' && QR='$QR' GIT_AUTOPUSH_INTERVAL=300 scripts/hpc/git_autopush_outputs.sh loop"
}

ensure_autopush

submit_split_2gpu() {
  local block="$1"
  local block_file="$QR/configs/machine_split/hpc_blocks/${block}.sh"
  echo "Submitting $block as independent 1-GPU cell jobs ..."
  echo "Archive: $QREASON_OUTPUT_ROOT"
  # shellcheck disable=SC1090
  source "$block_file"
  # Split mode: never --exclusive (QOS trap on 2-GPU ragpu nodes).
  if [[ "${QREASON_SLURM_EXCLUSIVE:-0}" == "1" ]]; then
    echo "WARN: QREASON_SLURM_EXCLUSIVE=1 ignored for split 1-GPU cells (see docs/PARAM_RUDRA_SLURM.md)." >&2
  fi
  for entry in "${HPC_BLOCK_CELLS[@]}"; do
    local cfg="${entry#*:}"
    local cell_id
    cell_id="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['cell_id'])" "$cfg")"
    echo "Submitting $block / $cell_id (1xA100) ..."
    sbatch --export="$SBATCH_EXPORT" \
      --job-name="qreason-${cell_id}" \
      --output="logs/slurm/${block}_${cell_id}_%j.out" \
      --error="logs/slurm/${block}_${cell_id}_%j.err" \
      --time=47:00:00 \
      "${SBATCH_EXCLUDE_ARGS[@]}" \
      --partition=gpu \
      --cpus-per-task=8 \
      --gres=gpu:1 \
      --wrap="bash scripts/hpc/run_hpc_2a100_publication.sh cell ${cfg} ${block}"
  done
}

submit_single_cell() {
  local cell_cfg="$1"
  local parent_block="${2:-single_cell}"
  local cell_id
  cell_id="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['cell_id'])" "$cell_cfg")"
  echo "Submitting single cell $cell_id (1×A100, non-exclusive) ..."
  echo "Archive: $QREASON_OUTPUT_ROOT"
  if [[ "${QREASON_SLURM_EXCLUSIVE:-0}" == "1" ]]; then
    echo "WARN: QREASON_SLURM_EXCLUSIVE=1 ignored for single 1-GPU cell (see docs/PARAM_RUDRA_SLURM.md)." >&2
  fi
  sbatch --export="$SBATCH_EXPORT" \
    --job-name="qreason-${cell_id}" \
    --output="logs/slurm/single_${cell_id}_%j.out" \
    --error="logs/slurm/single_${cell_id}_%j.err" \
    --time=47:00:00 \
    "${SBATCH_EXCLUDE_ARGS[@]}" \
    --partition=gpu \
    --cpus-per-task=8 \
    --gres=gpu:1 \
    --wrap="bash scripts/hpc/run_hpc_2a100_publication.sh cell ${cell_cfg} ${parent_block}"
}

submit_2gpu_block() {
  local block="$1"
  local block_file="$QR/configs/machine_split/hpc_blocks/${block}.sh"
  # shellcheck disable=SC1090
  source "$block_file"
  local cpus_per_task=$((8 * HPC_BLOCK_GPUS))
  local -a exclusive_args=()
  if [[ "${QREASON_SLURM_EXCLUSIVE:-0}" == "1" ]]; then
    exclusive_args+=(--exclusive)
  fi
  echo "Submitting $block as one ${HPC_BLOCK_GPUS}-GPU block job (exclusive=${QREASON_SLURM_EXCLUSIVE:-0}) ..."
  echo "Archive: $QREASON_OUTPUT_ROOT"
  sbatch --export="$SBATCH_EXPORT" \
    --job-name="qreason-${block}" \
    --output="logs/slurm/${block}_%j.out" \
    --error="logs/slurm/${block}_%j.err" \
    --time=47:00:00 \
    "${SBATCH_EXCLUDE_ARGS[@]}" \
    --partition=gpu \
    --cpus-per-task="$cpus_per_task" \
    --gres="gpu:${HPC_BLOCK_GPUS}" \
    "${exclusive_args[@]}" \
    --wrap="bash scripts/hpc/run_hpc_2a100_publication.sh ${block}"
}

submit_2gpu() {
  local block="$1"
  case "${QREASON_SUBMIT_2GPU_MODE:-split}" in
    split) submit_split_2gpu "$block" ;;
    exclusive_block|block) submit_2gpu_block "$block" ;;
    *)
      echo "ERROR: unknown QREASON_SUBMIT_2GPU_MODE=${QREASON_SUBMIT_2GPU_MODE}" >&2
      exit 2
      ;;
  esac
}

submit_1gpu() {
  local block="$1"
  echo "Submitting $block (1×A100) ..."
  echo "Archive: $QREASON_OUTPUT_ROOT"
  sbatch --export="$SBATCH_EXPORT" \
    --job-name="qreason-${block}" \
    --output="logs/slurm/${block}_%j.out" \
    --error="logs/slurm/${block}_%j.err" \
    --time=47:00:00 \
    "${SBATCH_EXCLUDE_ARGS[@]}" \
    --partition=gpu \
    --cpus-per-task=8 \
    --gres=gpu:1 \
    --wrap="bash scripts/hpc/run_hpc_2a100_publication.sh ${block}"
}

submit_all_blocks() {
  echo "WARN: submitting b01-b06 together violates the documented b01 hard gate." >&2
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
}

case "$BLOCK" in
  cell|single-cell|single_cell)
    if [[ $# -lt 2 ]]; then
      echo "Usage: $0 cell <cell-config.json> [parent-block-id]"
      exit 1
    fi
    submit_single_cell "$2" "${3:-single_cell}"
    ;;
  all|b01|b01_parallel_bf16_anchors)
    submit_2gpu b01_parallel_bf16_anchors
    ;;
  all_blocks)
    submit_all_blocks
    ;;
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
echo "Archive: $QREASON_OUTPUT_ROOT"
