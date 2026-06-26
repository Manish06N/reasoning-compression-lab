#!/usr/bin/env bash
# Run from MacBook to copy repo to HPC without GitHub.
# Edit HPC_USER and HPC_HOST before running.

set -euo pipefail

HPC_USER="${HPC_USER:-your_username}"
HPC_HOST="${HPC_HOST:-your_hpc_address}"
QR="/scratch/${HPC_USER}/reasoning-compression-lab"
LOCAL="/Users/manish/Projects/2026/paper 1/reasoning-compression-lab"

echo "Syncing local repo to ${HPC_USER}@${HPC_HOST}:${QR}"

ssh "${HPC_USER}@${HPC_HOST}" "mkdir -p ${QR}"

rsync -avz --delete \
  --exclude '.git' \
  --exclude 'runs/' \
  --exclude 'results/' \
  --exclude 'models/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  "${LOCAL}/" "${HPC_USER}@${HPC_HOST}:${QR}/"

echo "Done. SSH in and run: cd ${QR} && bash scripts/hpc/00_setup_env.sh"
