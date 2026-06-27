#!/usr/bin/env bash
# Validate MATH-500 dataset access before long inference jobs.
set -euo pipefail

export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

export HF_HOME="${HF_HOME:-$QR/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

CONDA_ROOT="${CONDA_ROOT:-/home/apps/MSCC/miniconda3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate qreason

python - <<'PY'
from datasets import load_dataset

name = "HuggingFaceH4/MATH-500"
print(f"Loading {name} ...")
ds = load_dataset(name, split="test")
print("OK:", name)
print("  num_examples:", len(ds))
print("  columns:", ds.column_names)
print("  sample problem preview:", ds[0]["problem"][:120], "...")
PY

echo "Dataset validation passed."
