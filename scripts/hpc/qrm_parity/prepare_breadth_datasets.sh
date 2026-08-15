#!/usr/bin/env bash
# Prepare GSM8K and GPQA datasets on disk for breadth benchmark evaluations
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
QRM_DIR="${QRM_REPO_DIR:-$QR/external/Quantized-Reasoning-Models}"
mkdir -p "$QRM_DIR/datasets"

CONDA_ROOT="${CONDA_ROOT:-/home/apps/MSCC/miniconda3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "${QRM_CONDA_ENV:-qreason}"

export HF_HOME="${HF_HOME:-$QR/hf_cache}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

python3 <<'PY'
import os
from pathlib import Path
from datasets import load_dataset

qrm_dir = Path(os.environ.get("QRM_DIR", "external/Quantized-Reasoning-Models"))
datasets_dir = qrm_dir / "datasets"
datasets_dir.mkdir(parents=True, exist_ok=True)

# 1. GSM8K (openai/gsm8k, split='test')
gsm8k_out = datasets_dir / "GSM8K"
if not (gsm8k_out / "dataset_info.json").exists():
    print("Downloading GSM8K (test split, n=1319)...")
    try:
        ds_gsm = load_dataset("openai/gsm8k", "main", split="test")
        ds_gsm.save_to_disk(str(gsm8k_out))
        print(f"Successfully saved GSM8K ({len(ds_gsm)} rows) to {gsm8k_out}")
    except Exception as e:
        print(f"Warning: Failed to download GSM8K: {e}")
else:
    print(f"GSM8K already prepared at {gsm8k_out}")

# 2. GPQA-Diamond (Idavidrein/gpqa)
gpqa_out = datasets_dir / "GPQA-Diamond"
if not (gpqa_out / "dataset_info.json").exists():
    print("Checking GPQA-Diamond (gated dataset)...")
    try:
        ds_gpqa = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        ds_gpqa.save_to_disk(str(gpqa_out))
        print(f"Successfully saved GPQA-Diamond ({len(ds_gpqa)} rows) to {gpqa_out}")
    except Exception as e:
        print(f"Note: GPQA-Diamond requires gated approval on Hugging Face: {e}")
else:
    print(f"GPQA-Diamond already prepared at {gpqa_out}")

print("Breadth datasets preparation check complete.")
PY
