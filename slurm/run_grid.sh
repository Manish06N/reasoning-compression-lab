#!/bin/bash
# Multi-seed SLURM grid template.
# Adapted from sober-reasoning/run.sh (seed loop pattern only).
#
# EDIT BEFORE USE:
#   LOCAL_DIR, OUTPUT_DIR, PARTITION, ACCOUNT, CELL_CONFIG, SEEDS

set -euo pipefail

LOCAL_DIR="${LOCAL_DIR:-/scratch/$USER/reasoning-compression-lab}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOCAL_DIR/logs/grid}"
PARTITION="${PARTITION:-gpu}"
ACCOUNT="${ACCOUNT:-}"
CELL_CONFIG="${CELL_CONFIG:-configs/cells/level_a_bf16_seed0.json}"
SEEDS=(0 1 2 3 4)

mkdir -p "$OUTPUT_DIR"

for SEED in "${SEEDS[@]}"; do
  echo "Submitting seed=$SEED cell=$CELL_CONFIG"
  SBATCH_ACCOUNT=()
  if [[ -n "$ACCOUNT" ]]; then
    SBATCH_ACCOUNT=(--account="$ACCOUNT")
  fi
  sbatch "${SBATCH_ACCOUNT[@]}" <<EOF
#!/bin/bash
#SBATCH --job-name=qreason-seed${SEED}
#SBATCH --output=${OUTPUT_DIR}/seed_${SEED}_%j.out
#SBATCH --error=${OUTPUT_DIR}/seed_${SEED}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00

set -euo pipefail
export QR=${LOCAL_DIR}
cd \$QR
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate qreason

export QREASON_MODEL_QWEN7B=\${QREASON_MODEL_QWEN7B:-\$QR/models/DeepSeek-R1-Distill-Qwen-7B}

python scripts/run_inference.py --cell-config ${CELL_CONFIG}
CELL_ID=\$(python -c "import json; print(json.load(open('${CELL_CONFIG}'))['cell_id'])")
python scripts/score_run.py --input runs/raw/\${CELL_ID}.jsonl
EOF
done

echo "Submitted \${#SEEDS[@]} seed jobs."
