#!/usr/bin/env bash
# Check HPC Gate 3-4 status and print next commands (run from MacBook or after SSH).
set -euo pipefail

HPC_USER="${HPC_USER:-manishn_iitp}"
HPC_HOST="${HPC_HOST:-paramrudra.iitp.ac.in}"
HPC_PORT="${HPC_PORT:-4422}"
QR="/scratch/${HPC_USER}/reasoning-compression-lab"

echo "=== HPC gate status for ${HPC_USER}@${HPC_HOST} ==="
echo ""

ssh -p "$HPC_PORT" "${HPC_USER}@${HPC_HOST}" bash -s <<EOF
set -euo pipefail
export QR="$QR"
cd "\$QR" 2>/dev/null || { echo "Repo not at \$QR — git clone first"; exit 1; }

echo "--- squeue ---"
squeue -u $HPC_USER 2>/dev/null || true

echo ""
echo "--- smoke artifacts ---"
for f in runs/raw/smoke_test.jsonl runs/raw/smoke_test_quick.jsonl; do
  if [[ -f "\$f" ]]; then
    echo "FOUND: \$f (\$(wc -l < "\$f") lines)"
  else
    echo "MISSING: \$f"
  fi
done

echo ""
echo "--- Level A BF16 output ---"
if [[ -f runs/raw/level_a_qwen7b_bf16_math500_seed0.jsonl ]]; then
  echo "FOUND: level_a BF16 raw (\$(wc -l < runs/raw/level_a_qwen7b_bf16_math500_seed0.jsonl) lines)"
else
  echo "MISSING: level_a BF16 raw"
fi

if [[ -f results/level_a_qwen7b_bf16_math500_seed0_summary.json ]]; then
  echo "FOUND: level_a summary (first paper brick)"
else
  echo "MISSING: level_a summary"
fi

echo ""
echo "--- Next commands on HPC ---"
echo "  cd \$QR && conda activate qreason"
echo "  bash scripts/hpc/03_smoke_test.sh"
echo "  bash scripts/hpc/04_run_level_a_bf16.sh 10"
echo "  bash scripts/hpc/05_score_level_a.sh"
EOF
