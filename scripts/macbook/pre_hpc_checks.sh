#!/usr/bin/env bash
# MacBook pre-HPC checks — run before first git push / HPC sync.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "=== 1. Git status ==="
git status --short
if git remote -v | grep -q origin; then
  echo "Remote configured."
else
  echo "WARN: No git remote. Add GitHub remote before HPC (see docs/GITHUB_PUSH.md)."
fi

echo ""
echo "=== 2. Script permissions ==="
chmod +x scripts/hpc/*.sh scripts/*.py slurm/*.sh 2>/dev/null || true
chmod +x slurm/*.slurm 2>/dev/null || true
echo "Permissions set."

echo ""
echo "=== 3. Shell syntax check ==="
for f in scripts/hpc/*.sh slurm/run_grid.sh; do
  bash -n "$f" && echo "OK: $f"
done

echo ""
echo "=== 4. Python syntax check ==="
python3 -m compileall -q src scripts tests && echo "OK: python compileall passed"

echo ""
echo "=== 5. Unit tests ==="
python3 -m pytest tests/ -q

echo ""
echo "=== 6. Decoding + cell validation ==="
python3 scripts/verify_decoding_params.py
python3 scripts/validate_cell_matrix.py

echo ""
echo "=== 7. Ruff (optional) ==="
if command -v ruff >/dev/null 2>&1; then
  ruff check src scripts tests
else
  echo "SKIP: ruff not installed (pip install -r requirements-dev.txt)"
fi

echo ""
echo "=== 8. .gitignore check ==="
for pattern in runs/ results/ models/ quant/ hf_cache/; do
  git check-ignore -q "$pattern" 2>/dev/null && echo "ignored: $pattern" || echo "check: $pattern (may need entry)"
done

echo ""
echo "All MacBook pre-HPC checks complete."
