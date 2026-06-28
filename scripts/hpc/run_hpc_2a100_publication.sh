#!/usr/bin/env bash
# HPC 2× A100 (80 GB) publication blocks — journal protocol, ≤48 h SLURM jobs.
#
# All 7B/8B cells, GSM8K, GPQA run here. 5080 = Qwen-1.5B only (≤24 h/cell).
#
# Usage:
#   bash scripts/hpc/run_hpc_2a100_publication.sh list
#   bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors
#   bash scripts/hpc/run_hpc_2a100_publication.sh b02_gpqa_fp8
#   bash scripts/hpc/submit_hpc_blocks.sh              # sbatch all ready blocks
#
# On PARAM Rudra after git pull:
#   export QR=/scratch/$USER/reasoning-compression-lab
#   cd $QR && bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/param_rudra_env.sh"
param_rudra_activate_conda

DATE_TAG="${QREASON_HPC_DATE:-$(date +%Y-%m-%d)}"
export QREASON_OUTPUT_ROOT="${QREASON_OUTPUT_ROOT:-$QR/outputs-hpc-2a100-main-${DATE_TAG}}"
RAW="$QREASON_OUTPUT_ROOT/raw"
SCORED="$QREASON_OUTPUT_ROOT/scored"
RESULTS="$QREASON_OUTPUT_ROOT/results"
LOGS="$QREASON_OUTPUT_ROOT/logs"
CHECKPOINTS="$QREASON_OUTPUT_ROOT/checkpoints"
mkdir -p "$RAW" "$SCORED" "$RESULTS" "$LOGS" "$CHECKPOINTS"

DECODING="${QREASON_DECODING:-configs/decoding/repro_qrm.yaml}"
BATCH_SIZE="${QREASON_BATCH_SIZE:-1}"
CHECKPOINT_EVERY="${QREASON_CHECKPOINT_EVERY:-10}"

cd "$QR"

cell_id_from_cfg() {
  python -c "import json; print(json.load(open('$1'))['cell_id'])"
}

run_one_cell() {
  local gpu_id="$1" cell_cfg="$2"
  local cell_id out log summary scored
  cell_id="$(cell_id_from_cfg "$cell_cfg")"
  out="$RAW/${cell_id}.jsonl"
  log="$LOGS/${cell_id}.log"
  scored="$SCORED/${cell_id}.jsonl"
  summary="$RESULTS/${cell_id}_summary.json"

  if [[ -f "$out" ]]; then
    local got want task
    got="$(wc -l < "$out" | tr -d ' ')"
    task="$(python -c "
import json
from pathlib import Path
cell = json.loads(Path('$cell_cfg').read_text())
task = json.loads(Path(cell['task_config']).read_text())
print(task['task_name'])
")"
    case "$task" in
      math500) want=500 ;;
      gsm8k) want=1319 ;;
      *) want=1 ;;
    esac
    if [[ "$got" -ge "$want" ]]; then
      echo "[gpu $gpu_id][skip] $cell_id complete ($got rows)"
      return 0
    fi
    echo "[gpu $gpu_id][resume] $cell_id — $got/$want rows"
  fi

  echo "[gpu $gpu_id] === inference: $cell_id ==="
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    python scripts/run_inference.py \
      --cell-config "$cell_cfg" \
      --decoding-config "$DECODING" \
      --batch-size "$BATCH_SIZE" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      --output "$out"
  ) 2>&1 | tee "$log"

  echo "[gpu $gpu_id] === score: $cell_id ==="
  rel_raw="${out#"$QR"/}"
  rel_scored="${scored#"$QR"/}"
  rel_summary="${summary#"$QR"/}"
  python scripts/score_run.py \
    --input "$rel_raw" \
    --output "$rel_scored" \
    --summary "$rel_summary" 2>&1 | tee -a "$log"
}

run_block() {
  local block_file="$1"
  # shellcheck disable=SC1090
  source "$block_file"
  echo "=== HPC block: $HPC_BLOCK_ID ==="
  echo "Archive: $QREASON_OUTPUT_ROOT"
  echo "GPUs: $HPC_BLOCK_GPUS | Est: ${HPC_BLOCK_EST_HOURS}h | Parallel: $HPC_PARALLEL"

  if [[ "$HPC_PARALLEL" == "true" && "$HPC_BLOCK_GPUS" -ge 2 ]]; then
    pids=()
    for entry in "${HPC_BLOCK_CELLS[@]}"; do
      gpu="${entry%%:*}"
      cfg="${entry#*:}"
      run_one_cell "$gpu" "$cfg" &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do
      wait "$pid" || return 1
    done
  else
    for entry in "${HPC_BLOCK_CELLS[@]}"; do
      gpu="${entry%%:*}"
      cfg="${entry#*:}"
      run_one_cell "$gpu" "$cfg"
    done
  fi
  echo "=== Block $HPC_BLOCK_ID finished ==="
}

list_blocks() {
  echo "HPC publication blocks (SLURM --time=47:00:00 max):"
  echo ""
  echo "  b01_parallel_bf16_anchors  2×A100  ~12–24h  BF16 Qwen-7B + Llama-8B MATH"
  echo "  b02_parallel_fp8           2×A100  ~12–24h  FP8 Qwen-7B + Llama-8B MATH"
  echo "  b03_parallel_awq4          2×A100  ~12–24h  AWQ Qwen-7B + Llama-8B MATH"
  echo "  b04_parallel_gptq4         2×A100  ~12–24h  GPTQ-4 Qwen-7B + Llama-8B MATH"
  echo "  b05_single_gptq3           1×A100  ~12–20h  GPTQ-3 Qwen-7B MATH"
  echo "  b06_single_gsm8k           1×A100  ~20–40h  FP8 Qwen-7B GSM8K (n=1319)"
  echo "  b07_gpqa_fp8               1×A100  ~8–20h   GPQA (after HF gate)"
  echo ""
  echo "5080 (~≤24h/cell): Qwen-1.5B only — bash scripts/local/run_5080_publication.sh"
}

BLOCK="${1:-list}"
BLOCK_DIR="$QR/configs/machine_split/hpc_blocks"

case "$BLOCK" in
  list|--list|-h|--help)
    list_blocks
    ;;
  b01|b01_parallel_bf16_anchors)
    run_block "$BLOCK_DIR/b01_parallel_bf16_anchors.sh"
    ;;
  b02|b02_parallel_fp8)
    run_block "$BLOCK_DIR/b02_parallel_fp8.sh"
    ;;
  b03|b03_parallel_awq4)
    run_block "$BLOCK_DIR/b03_parallel_awq4.sh"
    ;;
  b04|b04_parallel_gptq4)
    run_block "$BLOCK_DIR/b04_parallel_gptq4.sh"
    ;;
  b05|b05_single_gptq3)
    run_block "$BLOCK_DIR/b05_single_gptq3.sh"
    ;;
  b06|b06_single_gsm8k)
    run_block "$BLOCK_DIR/b06_single_gsm8k.sh"
    ;;
  b07|b07_gpqa_fp8|b02_gpqa_fp8)
    run_block "$BLOCK_DIR/b02_gpqa_fp8.sh"
    ;;
  *)
    echo "Unknown block: $BLOCK"
    list_blocks
    exit 1
    ;;
esac
