#!/usr/bin/env bash
# Periodically commit and push HPC output archives, including backup mirrors.
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
INTERVAL="${GIT_AUTOPUSH_INTERVAL:-600}"
MESSAGE_PREFIX="${GIT_AUTOPUSH_MESSAGE_PREFIX:-Update HPC outputs}"
LOCK_DIR="$QR/.git/hpc-autopush.lock"
LOG_FILE="$QR/logs/hpc_git_autopush.log"

cd "$QR"
mkdir -p logs

log() {
  printf '[%s] %s
' "$(date)" "$*" | tee -a "$LOG_FILE"
}

with_lock() {
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    log "another autopush is active; skipping this cycle"
    return 0
  fi
  trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' RETURN
  "$@"
}

push_once() {
  git add scripts/hpc/run_hpc_2a100_publication.sh scripts/hpc/submit_hpc_blocks.sh scripts/hpc/git_autopush_outputs.sh 2>/dev/null || true
  git add -f outputs-hpc-*/manifest.json outputs-hpc-*/state.json outputs-hpc-*/reproducibility_bundle.json 2>/dev/null || true
  git add -f outputs-hpc-*/checkpoints outputs-hpc-*/metadata outputs-hpc-*/results outputs-hpc-*/paper_tables outputs-hpc-*/logs 2>/dev/null || true
  git add -f outputs-hpc-*/raw outputs-hpc-*/scored outputs-hpc-*/_backup 2>/dev/null || true

  if git diff --cached --quiet; then
    log "no staged output changes"
    return 0
  fi

  local stamp
  stamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  git commit -m "$MESSAGE_PREFIX ($stamp)"
  git push origin HEAD
  log "pushed output changes"
}

case "${1:-loop}" in
  once)
    with_lock push_once
    ;;
  loop)
    log "starting HPC output autopush loop; interval=${INTERVAL}s"
    while true; do
      with_lock push_once || log "autopush cycle failed"
      sleep "$INTERVAL"
    done
    ;;
  *)
    echo "Usage: $0 [once|loop]" >&2
    exit 2
    ;;
esac
