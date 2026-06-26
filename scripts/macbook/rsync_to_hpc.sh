#!/usr/bin/env bash
# Run from MacBook to copy repo to PARAM Rudra HPC.
set -euo pipefail

HPC_USER="${HPC_USER:-manishn_iitp}"
HPC_HOST="${HPC_HOST:-paramrudra.iitp.ac.in}"
HPC_PORT="${HPC_PORT:-4422}"
QR="/scratch/${HPC_USER}/reasoning-compression-lab"
LOCAL="/Users/manish/Projects/2026/paper 1/reasoning-compression-lab"

SSH_OPTS=(-p "$HPC_PORT" -o ServerAliveInterval=60)

echo "Syncing local repo to ${HPC_USER}@${HPC_HOST}:${QR}"

ssh "${SSH_OPTS[@]}" "${HPC_USER}@${HPC_HOST}" "mkdir -p ${QR}"

rsync -avz -e "ssh ${SSH_OPTS[*]}" --delete \
  --exclude '.git' \
  --exclude 'runs/' \
  --exclude 'results/' \
  --exclude 'models/' \
  --exclude 'hf_cache/' \
  --exclude 'logs/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.env' \
  "${LOCAL}/" "${HPC_USER}@${HPC_HOST}:${QR}/"

echo "Done. SSH in and run: cd ${QR} && bash scripts/hpc/00_setup_env.sh"
