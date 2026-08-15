#!/usr/bin/env bash
# Run official QRM inference.py on MATH-500.
set -eo pipefail

export PATH="/usr/bin:/bin:${PATH}"

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
QRM_DIR="${QRM_REPO_DIR:-$QR/external/Quantized-Reasoning-Models}"
MODEL="${QRM_MODEL_PATH:-$QR/models/DeepSeek-R1-Distill-Qwen-7B}"
SEED="${QRM_SEED:-42}"
MAX_SAMPLES="${QRM_MAX_SAMPLES:-10}"
OUTPUT_ROOT="${QRM_OUTPUT_ROOT:-$QR/outputs-hpc-qrm-official-$(date +%Y-%m-%d)}"
OFFICIAL_RUN_DIR="$OUTPUT_ROOT/inference/$(basename "$MODEL")-seed${SEED}"
MODEL_NAME="$(basename "$MODEL")"

CONDA_ROOT="${CONDA_ROOT:-/home/apps/MSCC/miniconda3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "${QRM_CONDA_ENV:-qrm-official}"
export PATH="$CONDA_PREFIX/bin:/usr/bin:/bin:${PATH}"
if [[ -x "$CONDA_PREFIX/bin/git" ]]; then
  export GIT_PYTHON_GIT_EXECUTABLE="$CONDA_PREFIX/bin/git"
elif command -v git >/dev/null 2>&1; then
  export GIT_PYTHON_GIT_EXECUTABLE="$(command -v git)"
fi
export CC="${CC:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"

export HF_HOME="${HF_HOME:-$QR/hf_cache}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

QRM_MIN_FREE_GPU_MB="${QRM_MIN_FREE_GPU_MB:-62000}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU memory preflight (require >= ${QRM_MIN_FREE_GPU_MB} MiB free) ==="
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true

  GPU_QUERY_IDS="${CUDA_VISIBLE_DEVICES:-0}"
  IFS=',' read -r -a QRM_VISIBLE_GPUS <<<"$GPU_QUERY_IDS"
  FREE_GPU_MB=""
  for gpu_id in "${QRM_VISIBLE_GPUS[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    [[ -z "$gpu_id" ]] && continue
    gpu_free="$(nvidia-smi -i "$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
    if [[ -n "$gpu_free" ]]; then
      echo "GPU ${gpu_id} free memory: ${gpu_free} MiB"
      if [[ -z "$FREE_GPU_MB" || "$gpu_free" -lt "$FREE_GPU_MB" ]]; then
        FREE_GPU_MB="$gpu_free"
      fi
    fi
  done
  if [[ -z "$FREE_GPU_MB" ]]; then
    FREE_GPU_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    echo "GPU free memory fallback: ${FREE_GPU_MB:-unknown} MiB"
  fi

  if [[ -n "$FREE_GPU_MB" && "$FREE_GPU_MB" -lt "$QRM_MIN_FREE_GPU_MB" ]]; then
    echo "ERROR: only ${FREE_GPU_MB} MiB GPU memory free; need ${QRM_MIN_FREE_GPU_MB} MiB before vLLM load." >&2
    if [[ -n "${SLURM_JOB_ID:-}" && "${QRM_REQUEUE_ON_DIRTY_GPU:-1}" == "1" ]]; then
      restarts="${SLURM_RESTART_COUNT:-0}"
      max_requeues="${QRM_MAX_DIRTY_GPU_REQUEUES:-3}"
      if [[ "$restarts" -lt "$max_requeues" ]]; then
        echo "Requeueing job ${SLURM_JOB_ID} instead of failing on a dirty GPU (restart ${restarts}/${max_requeues})." >&2
        if scontrol requeue "$SLURM_JOB_ID"; then
          if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
            current_exc="$(scontrol show job "${SLURM_JOB_ID}" | grep -o 'ExcNodeList=[^ ]*' | cut -d= -f2 || true)"
            if [[ -n "$current_exc" && "$current_exc" != "(null)" ]]; then
              new_exc="${current_exc},${SLURM_JOB_NODELIST}"
            else
              new_exc="${SLURM_JOB_NODELIST}"
            fi
            echo "Excluding dirty node(s) ${new_exc} from future scheduling of job ${SLURM_JOB_ID}." >&2
            scontrol update job "${SLURM_JOB_ID}" ExcNodeList="${new_exc}" || echo "WARN: failed to update ExcNodeList" >&2
          fi
          exit 0
        fi
        echo "WARN: scontrol requeue failed; exiting with retryable code 75." >&2
      fi
    fi
    exit 75
  fi
fi

bash "$QR/scripts/hpc/qrm_parity/prepare_qrm_datasets.sh"

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$OFFICIAL_RUN_DIR"
cd "$QRM_DIR"

echo "Model: $MODEL"
echo "Output: $OUTPUT_ROOT"
echo "max_samples=$MAX_SAMPLES seed=$SEED"

QRM_GPU_MEMORY_UTILIZATION="${QRM_GPU_MEMORY_UTILIZATION:-0.75}"
DTYPE_ARGS=()
if [[ -n "${QRM_DTYPE:-}" ]]; then
  DTYPE_ARGS=(--dtype "$QRM_DTYPE")
elif [[ "$MODEL" == *"AWQ"* || "$MODEL" == *"awq"* ]]; then
  DTYPE_ARGS=(--dtype "float16")
fi

DATASET="${QRM_DATASET:-MATH-500}"

python inference.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --max_samples "$MAX_SAMPLES" \
  --seed "$SEED" \
  --output_dir "$OFFICIAL_RUN_DIR" \
  --gpu_memory_utilization "$QRM_GPU_MEMORY_UTILIZATION" \
  "${DTYPE_ARGS[@]}" \
  --overwrite

RESULT_JSON="$OFFICIAL_RUN_DIR/${DATASET}.jsonl"
DATASET_CLEAN="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]' | tr -d '-')"
if [[ -f "$RESULT_JSON" ]]; then
  RESULT_COPY="$OUTPUT_ROOT/qrm_official_${MODEL_NAME}_${DATASET_CLEAN}_n${MAX_SAMPLES}_seed${SEED}.json"
  VALIDATION_REPORT="$OUTPUT_ROOT/validation/${MODEL_NAME}_${DATASET_CLEAN}_n${MAX_SAMPLES}_seed${SEED}.json"
  mkdir -p "$(dirname "$VALIDATION_REPORT")"
  python3 "$QR/scripts/hpc/qrm_parity/validate_official_results.py" \
    --result "$RESULT_JSON" \
    --model "$MODEL" \
    --expected-rows "$MAX_SAMPLES" \
    --min-accuracy "${QRM_MIN_ACCURACY:-0}" \
    --min-boxed-rate "${QRM_MIN_BOXED_RATE:-0}" \
    --max-new-tokens 32768 \
    --max-token-limit-hits "${QRM_MAX_TOKEN_LIMIT_HITS:-$MAX_SAMPLES}" \
    --max-repetition-rows "${QRM_MAX_REPETITION_ROWS:-$MAX_SAMPLES}" \
    --report "$VALIDATION_REPORT"
  cp "$RESULT_JSON" "$RESULT_COPY"
  echo "Copied to $RESULT_COPY"
else
  echo "ERROR: expected result file missing: $RESULT_JSON" >&2
  find "$OUTPUT_ROOT" -name '*.json' | head -5
  exit 1
fi

echo ""
echo "Compare with our harness:"
echo "  python $QR/scripts/hpc/qrm_parity/compare_side_by_side.py --limit $MAX_SAMPLES"
