#!/usr/bin/env bash
# HPC 2× A100 (80 GB) publication blocks — journal protocol, ≤48 h SLURM jobs.
#
# All publication experiments run on HPC. Windows/5080 is retired for publication runs.
# Qwen-1.5B cells are future HPC-only work after model download/preflight.
#
# Usage:
#   bash scripts/hpc/run_hpc_2a100_publication.sh list
#   bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors
#   bash scripts/hpc/run_hpc_2a100_publication.sh cell configs/cells/level_a_bf16_seed0.json b01_parallel_bf16_anchors
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
param_rudra_assert_triton_cc

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

echo "=== DEBUG: after activate, python=$(which python 2>/dev/null || echo none) git=$(command -v git 2>/dev/null || echo none) ===" >&2

DATE_TAG="${QREASON_HPC_DATE:-$(date +%Y-%m-%d)}"
export QREASON_OUTPUT_ROOT="${QREASON_OUTPUT_ROOT:-$QR/outputs-hpc-2a100-main-${DATE_TAG}}"
RAW="$QREASON_OUTPUT_ROOT/raw"
SCORED="$QREASON_OUTPUT_ROOT/scored"
RESULTS="$QREASON_OUTPUT_ROOT/results"
LOGS="$QREASON_OUTPUT_ROOT/logs"
CHECKPOINTS="$QREASON_OUTPUT_ROOT/checkpoints"
METADATA="$QREASON_OUTPUT_ROOT/metadata"
BACKUP_ROOT="$QREASON_OUTPUT_ROOT/_backup"
mkdir -p "$RAW" "$SCORED" "$RESULTS" "$LOGS" "$CHECKPOINTS" "$METADATA" "$BACKUP_ROOT"

echo "=== DEBUG: dirs created for $QREASON_OUTPUT_ROOT ===" >&2

# Block accidental resume into the bad June-29 archive or stale pre-fix JSONL.
bash "$SCRIPT_DIR/09_assert_fresh_archive.sh"

echo "=== DEBUG: after 09_assert_fresh_archive ===" >&2

# Clean zero-byte stale lock files from prior crashed publication attempts.
# These can cause the early manifest/backup python steps (atomic_locked_json_update,
# backup_mirror) to block forever, preventing the job from ever reaching GPU preflight
# or run_inference (observed on 86579 etc.: wrapper stops after "Archive check passed",
# 0 GPU usage, job stays RUNNING for 1h+).
find "${QREASON_OUTPUT_ROOT:-}" -name '*.lock' -size 0 -delete 2>/dev/null || true

echo "=== DEBUG: stale locks cleaned ===" >&2

FRESH_FLAG=""
if [[ "${QREASON_FRESH_RUN:-}" == "1" ]]; then
  FRESH_FLAG="--fresh"
fi

DECODING="${QREASON_DECODING:-configs/decoding/repro_qrm.yaml}"
BATCH_SIZE="${QREASON_BATCH_SIZE:-1}"
CHECKPOINT_EVERY="${QREASON_CHECKPOINT_EVERY:-10}"
MIN_FREE_GPU_MB="${QREASON_MIN_FREE_GPU_MB:-55000}"
GPU_PREFLIGHT_REQUEUE="${QREASON_GPU_PREFLIGHT_REQUEUE:-1}"
GPU_PREFLIGHT_REQUEUE_MAX="${QREASON_GPU_PREFLIGHT_REQUEUE_MAX:-240}"

export QREASON_PUBLICATION_MODE=1
export VLLM_BATCH_INVARIANT=1
if [[ "${BATCH_SIZE:-1}" != "1" ]]; then
  echo "Publication mode requires BATCH_SIZE=1" >&2
  exit 1
fi

echo "=== DEBUG: about to run git clean assert ===" >&2

if ! python -c "import sys; sys.path.insert(0, '$QR'); from src.runners.publication_mode import assert_code_paths_clean; assert_code_paths_clean('$QR')"; then
  exit 1
fi

echo "=== DEBUG: git clean assert PASSED ===" >&2

export QR DATE_TAG BACKUP_ROOT DECODING BATCH_SIZE CHECKPOINT_EVERY QREASON_OUTPUT_ROOT

cd "$QR"

cuda_visible_for_gpu() {
  local gpu_id="$1"
  local visible="${CUDA_VISIBLE_DEVICES:-}"
  local -a devices=()

  if [[ -z "$visible" ]]; then
    echo "$gpu_id"
    return 0
  fi

  IFS="," read -r -a devices <<< "$visible"
  if [[ "${#devices[@]}" -le 1 ]]; then
    echo "$visible"
    return 0
  fi

  if [[ "$gpu_id" =~ ^[0-9]+$ && "$gpu_id" -lt "${#devices[@]}" ]]; then
    echo "${devices[$gpu_id]}"
    return 0
  fi

  echo "$gpu_id"
}

check_gpu_free_memory() {
  local gpu_id="$1" cuda_devices="$2" free_mb

  if [[ "${MIN_FREE_GPU_MB:-0}" == "0" ]]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARN: nvidia-smi not found; skipping GPU free-memory preflight" >&2
    return 0
  fi

  # Log who is using the GPU(s) for diagnostics on dirty nodes
  echo "[gpu $gpu_id] nvidia-smi processes on id=$cuda_devices:"
  nvidia-smi --id="$cuda_devices" --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "  (no compute apps or query failed)"
  nvidia-smi --id="$cuda_devices" --query-gpu=index,name,memory.free,memory.used,memory.total --format=csv,noheader 2>/dev/null || true

  # Local re-sample loop: transient holders may exit; avoid expensive full requeues
  local attempts=0
  local max_local_attempts=4
  local sleep_s=20
  while (( attempts < max_local_attempts )); do
    attempts=$((attempts + 1))
    free_mb="$(nvidia-smi --id="$cuda_devices" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
    if [[ -z "$free_mb" || ! "$free_mb" =~ ^[0-9]+$ ]]; then
      echo "WARN: could not read free GPU memory for CUDA_VISIBLE_DEVICES=$cuda_devices (attempt $attempts)" >&2
      if (( attempts < max_local_attempts )); then sleep "$sleep_s"; continue; fi
      free_mb=0
    fi

    echo "[gpu $gpu_id] free VRAM before vLLM (attempt $attempts): ${free_mb} MiB (required >= ${MIN_FREE_GPU_MB} MiB)"
    if (( free_mb >= MIN_FREE_GPU_MB )); then
      return 0
    fi

    if (( attempts < max_local_attempts )); then
      echo "  GPU still busy; sleeping ${sleep_s}s before recheck (local attempt)..."
      sleep "$sleep_s"
    fi
  done

  # After local attempts, still too low: decide on requeue or hard fail
  echo "ERROR: GPU $gpu_id (CUDA_VISIBLE_DEVICES=$cuda_devices) has only ${free_mb} MiB free; refusing to start vLLM on a busy GPU." >&2
  local restart_count="${SLURM_RESTART_COUNT:-0}"
  if [[ "$GPU_PREFLIGHT_REQUEUE" == "1" && -n "${SLURM_JOB_ID:-}" && "$restart_count" =~ ^[0-9]+$ && "$GPU_PREFLIGHT_REQUEUE_MAX" =~ ^[0-9]+$ && "$restart_count" -lt "$GPU_PREFLIGHT_REQUEUE_MAX" ]]; then
    echo "WARN: requeueing Slurm job ${SLURM_JOB_ID} after busy-GPU preflight failure (${restart_count}/${GPU_PREFLIGHT_REQUEUE_MAX})." >&2
    if scontrol requeue "$SLURM_JOB_ID"; then
      exit 0
    fi
    echo "WARN: scontrol requeue failed; leaving job failed with exit 75." >&2
  fi
  echo "ERROR: preflight failed after local retries and no more requeues allowed; aborting (exit 75)." >&2
  exit 75
}

cell_id_from_cfg() {
  python -c "import json; print(json.load(open('$1'))['cell_id'])"
}

backup_archive() {
  python - <<'PY_BACKUP' || echo "WARN: backup_archive failed" >&2
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["QR"])
from src.runners.checkpoint_utils import backup_mirror

backup_mirror(Path(os.environ["BACKUP_ROOT"]), Path(os.environ["QREASON_OUTPUT_ROOT"]))
PY_BACKUP
}

write_manifest_header() {
  local block_id="${1:-unknown}" block_file="${2:-}"
  python -m src.runners.archive_manifest header \
    --archive "$QREASON_OUTPUT_ROOT" \
    --block-id "$block_id" \
    --block-file "$block_file" \
    --decoding "$DECODING" \
    --batch-size "$BATCH_SIZE" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --date-tag "$DATE_TAG" \
    --repo "$QR" \
    || echo "WARN: write_manifest_header failed" >&2
}

write_cell_metadata() {
  local cell_id="$1" cell_cfg="$2" gpu_id="$3" status="$4" out="$5" summary="${6:-}"
  python -m src.runners.archive_manifest cell-metadata \
    --archive "$QREASON_OUTPUT_ROOT" \
    --cell-id "$cell_id" \
    --cell-config "$cell_cfg" \
    --gpu-id "$gpu_id" \
    --status "$status" \
    --raw "$out" \
    --summary "$summary" \
    --decoding "$DECODING" \
    --batch-size "$BATCH_SIZE" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --repo "$QR" \
    || echo "WARN: write_cell_metadata failed" >&2
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
    local got want
    got="$(wc -l < "$out" | tr -d ' ')"
    want="$(python scripts/expected_rows.py --cell-config "$cell_cfg")"
    if [[ "$got" -ge "$want" ]]; then
      if [[ ! -f "$summary" || ! -f "$scored" ]]; then
        echo "[gpu $gpu_id][score-only] $cell_id raw complete ($got/$want rows), scoring..."
        rel_raw="${out#"$QR"/}"
        rel_scored="${scored#"$QR"/}"
        rel_summary="${summary#"$QR"/}"
        python scripts/score_run.py \
          --publication \
          --skip-calibration \
          --input "$rel_raw" \
          --output "$rel_scored" \
          --summary "$rel_summary" 2>&1 | tee -a "$log"
        write_cell_metadata "$cell_id" "$cell_cfg" "$gpu_id" "scored" "$out" "$summary"
        backup_archive
        return 0
      fi
      echo "[gpu $gpu_id][skip] $cell_id complete ($got/$want rows)"
      write_cell_metadata "$cell_id" "$cell_cfg" "$gpu_id" "scored" "$out" "$summary"
      backup_archive
      return 0
    fi
    echo "[gpu $gpu_id][resume] $cell_id — $got/$want rows"
  fi

  write_cell_metadata "$cell_id" "$cell_cfg" "$gpu_id" "in_progress" "$out" ""
  backup_archive
  local cuda_devices
  cuda_devices="$(cuda_visible_for_gpu "$gpu_id")"
  echo "[gpu $gpu_id] === inference: $cell_id (CUDA_VISIBLE_DEVICES=$cuda_devices) ==="
  check_gpu_free_memory "$gpu_id" "$cuda_devices"
  (
    export CUDA_VISIBLE_DEVICES="$cuda_devices"
    python scripts/run_inference.py \
      --publication \
      --cell-config "$cell_cfg" \
      --decoding-config "$DECODING" \
      --batch-size "$BATCH_SIZE" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      $FRESH_FLAG \
      --output "$out"
  ) 2>&1 | tee "$log"
  write_cell_metadata "$cell_id" "$cell_cfg" "$gpu_id" "inference_completed" "$out" ""
  backup_archive

  echo "[gpu $gpu_id] === score: $cell_id ==="
  rel_raw="${out#"$QR"/}"
  rel_scored="${scored#"$QR"/}"
  rel_summary="${summary#"$QR"/}"
  python scripts/score_run.py \
    --publication \
    --skip-calibration \
    --input "$rel_raw" \
    --output "$rel_scored" \
    --summary "$rel_summary" 2>&1 | tee -a "$log"
  write_cell_metadata "$cell_id" "$cell_cfg" "$gpu_id" "scored" "$out" "$summary"
  backup_archive
}

run_block() {
  local block_file="$1"
  # shellcheck disable=SC1090
  source "$block_file"
  echo "=== HPC block: $HPC_BLOCK_ID ==="
  echo "Archive: $QREASON_OUTPUT_ROOT"
  echo "GPUs: $HPC_BLOCK_GPUS | Est: ${HPC_BLOCK_EST_HOURS}h | Parallel: $HPC_PARALLEL"
  write_manifest_header "$HPC_BLOCK_ID" "$block_file"
  backup_archive

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

run_single_cell() {
  local cell_cfg="$1"
  local parent_block="${2:-single_cell}"
  local cell_id
  cell_id="$(cell_id_from_cfg "$cell_cfg")"
  echo "=== HPC cell: $cell_id ==="
  echo "Archive: $QREASON_OUTPUT_ROOT"
  write_manifest_header "$parent_block" "$cell_cfg"
  backup_archive
  run_one_cell 0 "$cell_cfg"
  echo "=== Cell $cell_id finished ==="
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
  echo "  b08_qwen15b_bf16_fp8       2×A100  ~12–24h  Qwen-1.5B BF16 + FP8 MATH"
  echo "  b09_qwen15b_awq4_gptq4     2×A100  ~12–24h  Qwen-1.5B AWQ-4 + GPTQ-4 MATH"
  echo ""
  echo "b08-b09 are future HPC-only lower-bound jobs; do not submit until current queue strategy allows."
}

BLOCK="${1:-list}"
BLOCK_DIR="$QR/configs/machine_split/hpc_blocks"

case "$BLOCK" in
  list|--list|-h|--help)
    list_blocks
    ;;
  cell|single-cell|single_cell)
    if [[ $# -lt 2 ]]; then
      echo "Usage: $0 cell <cell-config.json> [parent-block-id]"
      exit 1
    fi
    run_single_cell "$2" "${3:-single_cell}"
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
  b08|b08_qwen15b_bf16_fp8)
    run_block "$BLOCK_DIR/b08_qwen15b_bf16_fp8.sh"
    ;;
  b09|b09_qwen15b_awq4_gptq4)
    run_block "$BLOCK_DIR/b09_qwen15b_awq4_gptq4.sh"
    ;;
  *)
    echo "Unknown block: $BLOCK"
    list_blocks
    exit 1
    ;;
esac
