#!/usr/bin/env bash
# Export pinned pip freeze from HPC qreason env for reproducibility bundles.
# Run on PARAM Rudra after: conda activate qreason
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="${1:-requirements-hpc.lock.txt}"

echo "Exporting pip freeze to $OUT"
python -m pip freeze > "$OUT"
echo "Python: $(python --version 2>&1)"
echo "Done. Commit $OUT after verifying on qreason."
