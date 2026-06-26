#!/usr/bin/env bash
# Validate MATH-500 dataset access before long inference jobs.
set -euo pipefail

export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

source "$(conda info --base)/etc/profile.d/conda.sh"
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
