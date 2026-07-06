#!/usr/bin/env bash
# Start/restart Telegram progress watcher for active HPC campaign.
#
# Experiment A — official QRM (default 2026-07-05):
#   Jobs: qreason-qrm_official_n10 (Qwen), qreason-qrm_official_llama_n10 (Llama)
#   Archive: outputs-hpc-qrm-official-2026-07-05
#   Progress ping every 45 minutes while jobs run
#
# Override:
#   WATCH_JOB_IDS="87130 87131" \
#   OUTPUT_ROOT=/scratch/$USER/reasoning-compression-lab/outputs-hpc-qrm-official-2026-07-05 \
#   bash ~/start-hpc-telegram-watcher.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/$USER/reasoning-compression-lab}"
DATE_TAG="${QREASON_HPC_DATE:-2026-07-05}"
OUTPUT_ROOT="${QREASON_OUTPUT_ROOT:-$PROJECT_DIR/outputs-hpc-qrm-official-${DATE_TAG}}"
OUTPUT_ROOT_LLAMA="${QREASON_OUTPUT_ROOT_LLAMA:-$PROJECT_DIR/outputs-hpc-qrm-official-llama-${DATE_TAG}}"
SESSION="${TG_WATCH_SESSION:-hpc_progress}"

# Default: Experiment A official QRM jobs (auto-discover from queue)
if [[ -z "${WATCH_JOB_IDS:-}" ]]; then
  WATCH_JOB_IDS=$(squeue -u "$USER" -h -o '%i %j' 2>/dev/null | awk '/qreason-qrm_official/ {print $1}' | tr '\n' ' ' | sed 's/ $//')
  if [[ -f "$PROJECT_DIR/.qrm_official_llama_job_submitted" ]]; then
    llama_id=$(tr -d '[:space:]' < "$PROJECT_DIR/.qrm_official_llama_job_submitted")
    if [[ -n "$llama_id" ]] && [[ " $WATCH_JOB_IDS " != *" $llama_id "* ]]; then
      WATCH_JOB_IDS="${WATCH_JOB_IDS:+$WATCH_JOB_IDS }$llama_id"
    fi
  fi
  if [[ -z "$WATCH_JOB_IDS" ]]; then
    WATCH_JOB_IDS="${QRM_OFFICIAL_WATCH_JOB_IDS:-87130}"
  fi
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

WATCH_LABEL="${WATCH_LABEL:-Experiment A: official QRM inference.py (n=10, seed=42)}"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" "bash -lc '
  cd \"$HOME\" && \
  PROJECT_DIR=\"$PROJECT_DIR\" \
  OUTPUT_ROOT=\"$OUTPUT_ROOT\" \
  OUTPUT_ROOT_LLAMA=\"$OUTPUT_ROOT_LLAMA\" \
  QREASON_INFERENCE_LIMIT=10 \
  QRM_OFFICIAL_MODE=1 \
  QRM_OFFICIAL_MAX_SAMPLES=10 \
  WATCH_JOB_IDS=\"$WATCH_JOB_IDS\" \
  WATCH_LABEL=\"$WATCH_LABEL\" \
  RUNNING_PROGRESS_INTERVAL=2700 \
  PENDING_PROGRESS_INTERVAL=2700 \
  PROGRESS_INTERVAL=2700 \
  STATE_POLL_INTERVAL=300 \
  LOG_PROGRESS_INTERVAL=1 \
  \"$HOME/send-hpc-progress-telegram.sh\" loop
'"

echo "Telegram watcher started (tmux session: $SESSION)"
echo "  Archive (Qwen): $OUTPUT_ROOT"
echo "  Archive (Llama): $OUTPUT_ROOT_LLAMA"
echo "  Jobs: ${WATCH_JOB_IDS:-none}"
echo "  Interval: 45 min (running/pending); phase/result changes ping sooner"
echo "Preview: PROJECT_DIR=$PROJECT_DIR OUTPUT_ROOT=$OUTPUT_ROOT QRM_OFFICIAL_MODE=1 WATCH_JOB_IDS=\"$WATCH_JOB_IDS\" $HOME/send-hpc-progress-telegram.sh preview"
echo "Logs: tmux attach -t $SESSION"

# Immediate status ping
PROJECT_DIR="$PROJECT_DIR" OUTPUT_ROOT="$OUTPUT_ROOT" OUTPUT_ROOT_LLAMA="$OUTPUT_ROOT_LLAMA" \
  WATCH_JOB_IDS="$WATCH_JOB_IDS" QREASON_INFERENCE_LIMIT=10 QRM_OFFICIAL_MODE=1 \
  QRM_OFFICIAL_MAX_SAMPLES=10 WATCH_LABEL="$WATCH_LABEL" \
  "$HOME/send-hpc-progress-telegram.sh" once || echo "WARN: Telegram send failed (DNS/network?)" >&2