#!/usr/bin/env bash
# Prepare MATH-500 on disk where QRM lighteval expects it: ./datasets/MATH-500
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
QRM_DIR="${QRM_REPO_DIR:-$QR/external/Quantized-Reasoning-Models}"
DATASET_DIR="$QRM_DIR/datasets/MATH-500"

mkdir -p "$QRM_DIR/datasets"
if [[ -f "$DATASET_DIR/dataset_info.json" ]] || [[ -f "$DATASET_DIR/state.json" ]]; then
  echo "MATH-500 already prepared at $DATASET_DIR"
  exit 0
fi

CONDA_ROOT="${CONDA_ROOT:-/home/apps/MSCC/miniconda3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "${QRM_CONDA_ENV:-qreason}"

export HF_HOME="${HF_HOME:-$QR/hf_cache}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

python3 <<PY
from datasets import load_dataset
from pathlib import Path

out = Path("$DATASET_DIR")
out.parent.mkdir(parents=True, exist_ok=True)
ds = load_dataset(
    "HuggingFaceH4/MATH-500",
    split="test",
    revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    trust_remote_code=True,
)
ds.save_to_disk(str(out))
print(f"Saved {len(ds)} rows to {out}")
PY