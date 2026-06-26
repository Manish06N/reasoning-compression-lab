#!/usr/bin/env bash
# Pull code (and optionally results) from PARAM Rudra HPC to MacBook.
# Run interactively — SSH requires captcha + password on PARAM Rudra.

set -euo pipefail

HPC_USER="${HPC_USER:-manishn_iitp}"
HPC_HOST="${HPC_HOST:-paramrudra.iitp.ac.in}"
HPC_PORT="${HPC_PORT:-4422}"
QR="/scratch/${HPC_USER}/reasoning-compression-lab"
LOCAL="/Users/manish/Projects/2026/paper 1/reasoning-compression-lab"

SSH_OPTS=(-p "$HPC_PORT" -o ServerAliveInterval=60)

echo "Pulling repo code from ${HPC_USER}@${HPC_HOST}:${QR}"
echo "  (excludes runs/, results/, models/, hf_cache/, .git/)"

rsync -avz -e "ssh ${SSH_OPTS[*]}" \
  --exclude '.git/' \
  --exclude 'runs/' \
  --exclude 'results/' \
  --exclude 'models/' \
  --exclude 'hf_cache/' \
  --exclude 'logs/' \
  --exclude '__pycache__/' \
  --exclude '.env' \
  "${HPC_USER}@${HPC_HOST}:${QR}/" "${LOCAL}/"

if [[ "${SYNC_RESULTS:-0}" == "1" ]]; then
  echo "Also syncing runs/ and results/ ..."
  mkdir -p "${LOCAL}/runs" "${LOCAL}/results"
  rsync -avz -e "ssh ${SSH_OPTS[*]}" \
    "${HPC_USER}@${HPC_HOST}:${QR}/runs/" "${LOCAL}/runs/"
  rsync -avz -e "ssh ${SSH_OPTS[*]}" \
    "${HPC_USER}@${HPC_HOST}:${QR}/results/" "${LOCAL}/results/"
fi

echo "Done. Review with: cd \"${LOCAL}\" && git status"
