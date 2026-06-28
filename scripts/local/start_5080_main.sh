#!/usr/bin/env bash
# Start (or resume) the publication-standard 5080 main grid in the background.
#
# Usage:
#   bash scripts/local/start_5080_main.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

DATE_TAG="${QREASON_5080_DATE:-2026-06-28}"
ARCHIVE="${QREASON_OUTPUT_ROOT:-$QR/outputs-win5080-main-${DATE_TAG}}"
LOGS="$ARCHIVE/logs"
mkdir -p "$LOGS"

if pgrep -f "scripts/local/run_all_5080_phases.sh" >/dev/null 2>&1; then
  echo "Pipeline already running. Monitor: $LOGS/orchestrator.log"
  pgrep -af "run_all_5080|run_inference" || true
  exit 0
fi

pkill -f 'vllm.v1.engine.core' 2>/dev/null || true
pkill -f 'scripts/run_inference.py' 2>/dev/null || true
sleep 3

echo "=== $(date -Is) Starting 5080 publication main grid ===" >> "$LOGS/orchestrator.log"
setsid bash -lc "source '$SCRIPT_DIR/env.sh' && export QREASON_PUBLICATION_MODE=1 && export VLLM_BATCH_INVARIANT=1 && export QREASON_CELL_QUEUE='$SCRIPT_DIR/../../configs/machine_split/5080_cells.sh' && exec bash '$SCRIPT_DIR/run_5080_publication.sh' --skip-download" \
  >> "$LOGS/orchestrator.log" 2>&1 < /dev/null &
BGPID=$!
echo "PID=$BGPID"
echo "Archive: $ARCHIVE"
echo "Windows: G:/ALL MY Projects/2026/03-paper1-experiments/$(basename "$ARCHIVE")"
echo "Monitor: tail -f $LOGS/orchestrator.log"
