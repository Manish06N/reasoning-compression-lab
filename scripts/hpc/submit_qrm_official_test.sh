#!/usr/bin/env bash
# Cancel Path C (if any) and submit official QRM inference.py cross-check.
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"
mkdir -p logs

echo "=== Ensuring QRM repo clone ==="
bash scripts/hpc/qrm_parity/setup_official_qrm_repo.sh

echo ""
echo "=== Submitting official QRM MATH-500 test (n=10, seed=42, 1 GPU, 8h) ==="
JOBID=$(sbatch --parsable slurm/qrm_official_math500_n10.slurm)
echo "Submitted job $JOBID"
echo "  Monitor: squeue -j $JOBID"
echo "  Logs:    tail -f logs/qrm_official_${JOBID}.out"
echo "  Output:  outputs-hpc-qrm-official-$(date +%Y-%m-%d)/"
echo ""
echo "After completion, compare traces:"
echo "  python scripts/hpc/qrm_parity/compare_side_by_side.py --limit 10"