#!/usr/bin/env bash
# Print Path C diagnostic summaries (pass@1, truncation, vs QRM bands).
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
DATE_TAG="${QREASON_HPC_DATE:-$(date +%Y-%m-%d)}"
ARCHIVE="${QREASON_OUTPUT_ROOT:-$QR/outputs-hpc-diag-pathc-${DATE_TAG}}"

PYTHON="${PYTHON:-$HOME/.conda/envs/qreason/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

CELLS=(
  diag_qwen7b_bf16_math500_seed42_n50
  diag_llama8b_bf16_math500_seed42_n50
  diag_qwen7b_bf16_math500_seed42_n50_64k
)

echo "Archive: $ARCHIVE"
echo ""
printf "%-45s %8s %10s %10s %6s\n" "cell" "pass@1" "trunc%" "parse_fail" "n"
printf "%-45s %8s %10s %10s %6s\n" "----" "------" "------" "----------" "---"

for cell in "${CELLS[@]}"; do
  summary="$ARCHIVE/results/${cell}_summary.json"
  raw="$ARCHIVE/raw/${cell}.jsonl"
  if [[ ! -f "$summary" ]]; then
    if [[ -f "$raw" ]]; then
      rows="$(wc -l <"$raw" | tr -d ' ')"
      printf "%-45s %8s %10s %10s %6s\n" "$cell" "—" "—" "—" "$rows/?"
    else
      printf "%-45s %8s %10s %10s %6s\n" "$cell" "pending" "—" "—" "0"
    fi
    continue
  fi
  "$PYTHON" -c "
import json, sys
s=json.load(open(sys.argv[1]))
print('{:<45} {:7.1%} {:9.1%} {:9.1%} {:>6}'.format(
  sys.argv[2],
  s.get('pass_at_1', 0),
  s.get('truncation_rate', 0),
  s.get('parse_failure_rate', 0),
  s.get('n', 0),
))" "$summary" "$cell"
done

echo ""
echo "QRM MATH-500 reference (BF16, 32k, reproduction): Qwen ~93.9%, Llama ~91.0%"
echo "Pass heuristic for 32k wave: pass@1 >= 80% AND truncation <= 25% on n=50"
echo "64k wave: if pass@1 >> 32k Qwen diag → truncation was main bottleneck"