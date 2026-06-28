#!/usr/bin/env bash
# Start (or resume) the 5080 pilot pipeline in the background.
#
# Usage:
#   bash scripts/local/start_5080_pilot.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

DATE_TAG="${QREASON_5080_DATE:-2026-06-28}"
ARCHIVE="${QREASON_OUTPUT_ROOT:-$QR/outputs-win5080-pilot-${DATE_TAG}}"
LOGS="$ARCHIVE/logs"
mkdir -p "$LOGS"

if pgrep -f "scripts/local/run_all_5080_phases.sh" >/dev/null 2>&1; then
  echo "Pipeline already running. Monitor: $LOGS/orchestrator.log"
  pgrep -af "run_all_5080|run_inference" || true
  exit 0
fi

# Clear stale vLLM workers from interrupted runs (orchestrator not running).
pkill -f 'vllm.v1.engine.core' 2>/dev/null || true
pkill -f 'scripts/run_inference.py' 2>/dev/null || true
sleep 3

echo "=== $(date -Is) Starting 5080 pilot pipeline ===" >> "$LOGS/orchestrator.log"
# setsid + login shell so conda/env.sh apply in detached WSL session.
setsid bash -lc "source '$SCRIPT_DIR/env.sh' && exec bash '$SCRIPT_DIR/resume_5080_pilot.sh'" \
  >> "$LOGS/orchestrator.log" 2>&1 < /dev/null &
BGPID=$!
echo "PID=$BGPID"
echo "Tip: from Windows, run via Cursor background task or keep WSL open."
echo "Or: wsl -d Ubuntu-22.04 --cd \"$QR\" -- bash -lc \"source scripts/local/env.sh && exec bash scripts/local/resume_5080_pilot.sh >> outputs-win5080-pilot-2026-06-28/logs/orchestrator.log 2>&1\""
echo "Archive: $ARCHIVE"
echo "Windows: G:/ALL MY Projects/2026/03-paper1-experiments/$(basename "$ARCHIVE")"
echo "Monitor: tail -f $LOGS/orchestrator.log"
