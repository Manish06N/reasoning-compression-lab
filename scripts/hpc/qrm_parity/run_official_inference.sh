#!/usr/bin/env bash
# Run official QRM inference.py on MATH-500 (n=10 pilot).
set -eo pipefail

export PATH="/usr/bin:/bin:${PATH}"

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
QRM_DIR="${QRM_REPO_DIR:-$QR/external/Quantized-Reasoning-Models}"
MODEL="${QRM_MODEL_PATH:-$QR/models/DeepSeek-R1-Distill-Qwen-7B}"
SEED="${QRM_SEED:-42}"
MAX_SAMPLES="${QRM_MAX_SAMPLES:-10}"
OUTPUT_ROOT="${QRM_OUTPUT_ROOT:-$QR/outputs-hpc-qrm-official-$(date +%Y-%m-%d)}"

CONDA_ROOT="${CONDA_ROOT:-/home/apps/MSCC/miniconda3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "${QRM_CONDA_ENV:-qrm-official}"
export PATH="$CONDA_PREFIX/bin:/usr/bin:/bin:${PATH}"
export CC="${CC:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"

export HF_HOME="${HF_HOME:-$QR/hf_cache}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

bash "$QR/scripts/hpc/qrm_parity/prepare_qrm_datasets.sh"

mkdir -p "$OUTPUT_ROOT"
cd "$QRM_DIR"

echo "Model: $MODEL"
echo "Output: $OUTPUT_ROOT"
echo "max_samples=$MAX_SAMPLES seed=$SEED"

python inference.py \
  --model "$MODEL" \
  --dataset MATH-500 \
  --max_samples "$MAX_SAMPLES" \
  --seed "$SEED" \
  --output_dir "$OUTPUT_ROOT/inference" \
  --overwrite

RESULT_JSON="$OUTPUT_ROOT/inference/$(basename "$MODEL")-seed${SEED}/MATH-500.json"
if [[ -f "$RESULT_JSON" ]]; then
  python3 - "$RESULT_JSON" <<'PY'
import json, sys
path = sys.argv[1]
rows = json.load(open(path))
print(f"Official QRM results: {len(rows)} rows")
boxed = sum(1 for r in rows if "\\boxed" in (r.get("generated_text") or ""))
print(f"Rows with \\boxed: {boxed}/{len(rows)}")
if rows:
    m0 = rows[0].get("metrics", {})
    print(f"Sample metrics keys: {list(m0.keys())[:5]}")
PY
  cp "$RESULT_JSON" "$OUTPUT_ROOT/qrm_official_math500_n${MAX_SAMPLES}_seed${SEED}.json"
  echo "Copied to $OUTPUT_ROOT/qrm_official_math500_n${MAX_SAMPLES}_seed${SEED}.json"
else
  echo "WARN: expected result file missing: $RESULT_JSON" >&2
  find "$OUTPUT_ROOT" -name '*.json' | head -5
fi

echo ""
echo "Compare with our harness:"
echo "  python $QR/scripts/hpc/qrm_parity/compare_side_by_side.py --limit $MAX_SAMPLES"