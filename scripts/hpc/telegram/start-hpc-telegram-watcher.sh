#!/usr/bin/env bash
# Start/restart Telegram progress watcher for active HPC campaign.
#
# Path C diagnostic (default 2026-07-05):
#   Jobs 87116 Qwen 32k, 87117 Llama 32k, 87118 Qwen 64k
#   Archive: outputs-hpc-diag-pathc-2026-07-05
#   Progress ping every 45 minutes while jobs run
#
# Override:
#   WATCH_JOB_IDS="87116 87117 87118" \
#   OUTPUT_ROOT=/scratch/$USER/reasoning-compression-lab/outputs-hpc-diag-pathc-2026-07-05 \
#   bash ~/start-hpc-telegram-watcher.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/$USER/reasoning-compression-lab}"
DATE_TAG="${QREASON_HPC_DATE:-2026-07-05}"
OUTPUT_ROOT="${QREASON_OUTPUT_ROOT:-$PROJECT_DIR/outputs-hpc-diag-pathc-${DATE_TAG}}"
SESSION="${TG_WATCH_SESSION:-hpc_progress}"

# Default: Path C diagnostic jobs (override with WATCH_JOB_IDS)
if [[ -z "${WATCH_JOB_IDS:-}" ]]; then
  WATCH_JOB_IDS="${PATHC_WATCH_JOB_IDS:-87116 87117 87118}"
fi

# Drop jobs that already finished (keeps watcher useful after partial completion)
alive_ids=()
for id in $WATCH_JOB_IDS; do
  [[ -z "$id" ]] && continue
  if squeue -j "$id" -h -o '%T' 2>/dev/null | grep -qE 'RUNNING|PENDING|CONFIGURING'; then
    alive_ids+=("$id")
  elif sacct -j "$id" -n -X -o State 2>/dev/null | grep -qE 'RUNNING|PENDING|CONFIGURING'; then
    alive_ids+=("$id")
  fi
done
if [[ "${#alive_ids[@]}" -gt 0 ]]; then
  WATCH_JOB_IDS="${alive_ids[*]}"
else
  # Fallback: any qreason jobs in queue
  WATCH_JOB_IDS=$(squeue -u "$USER" -h -o '%i %j' 2>/dev/null | awk '/qreason-/ {print $1}' | tr '\n' ' ' | sed 's/ $//')
fi

WATCH_LABEL="${WATCH_LABEL:-Path C diagnostic: Qwen+Llama 32k (n=50) + Qwen 64k}"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" "bash -lc '
  cd \"$HOME\" && \
  PROJECT_DIR=\"$PROJECT_DIR\" \
  OUTPUT_ROOT=\"$OUTPUT_ROOT\" \
  QREASON_INFERENCE_LIMIT=50 \
  WATCH_JOB_IDS=\"$WATCH_JOB_IDS\" \
  WATCH_LABEL=\"$WATCH_LABEL\" \
  RUNNING_PROGRESS_INTERVAL=2700 \
  PENDING_PROGRESS_INTERVAL=2700 \
  PROGRESS_INTERVAL=2700 \
  STATE_POLL_INTERVAL=300 \
  LOG_PROGRESS_INTERVAL=10 \
  \"$HOME/send-hpc-progress-telegram.sh\" loop
'"

echo "Telegram watcher started (tmux session: $SESSION)"
echo "  Archive: $OUTPUT_ROOT"
echo "  Jobs: ${WATCH_JOB_IDS:-none}"
echo "  Interval: 45 min (running/pending)"
echo "Preview: PROJECT_DIR=$PROJECT_DIR OUTPUT_ROOT=$OUTPUT_ROOT WATCH_JOB_IDS=\"$WATCH_JOB_IDS\" QREASON_INFERENCE_LIMIT=50 $HOME/send-hpc-progress-telegram.sh preview"
echo "Logs: tmux attach -t $SESSION"

# Immediate status ping
PROJECT_DIR="$PROJECT_DIR" OUTPUT_ROOT="$OUTPUT_ROOT" WATCH_JOB_IDS="$WATCH_JOB_IDS" \
  QREASON_INFERENCE_LIMIT=50 WATCH_LABEL="$WATCH_LABEL" \
  "$HOME/send-hpc-progress-telegram.sh" once || echo "WARN: Telegram send failed (DNS/network?)" >&2