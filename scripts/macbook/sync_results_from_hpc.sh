#!/usr/bin/env bash
# Pull experiment outputs from PARAM Rudra to MacBook for analysis/plotting.

set -euo pipefail

HPC_USER="${HPC_USER:-manishn_iitp}"
HPC_HOST="${HPC_HOST:-paramrudra.iitp.ac.in}"
HPC_PORT="${HPC_PORT:-4422}"
QR="/scratch/${HPC_USER}/reasoning-compression-lab"
LOCAL="/Users/manish/Projects/2026/paper 1/reasoning-compression-lab"

SSH_OPTS=(-p "$HPC_PORT" -o ServerAliveInterval=60)

mkdir -p "${LOCAL}/runs" "${LOCAL}/results" "${LOCAL}/logs"

echo "Syncing runs/, results/, logs/ from HPC ..."
rsync -avz -e "ssh ${SSH_OPTS[*]}" \
  "${HPC_USER}@${HPC_HOST}:${QR}/runs/" "${LOCAL}/runs/"
rsync -avz -e "ssh ${SSH_OPTS[*]}" \
  "${HPC_USER}@${HPC_HOST}:${QR}/results/" "${LOCAL}/results/"
rsync -avz -e "ssh ${SSH_OPTS[*]}" \
  "${HPC_USER}@${HPC_HOST}:${QR}/logs/" "${LOCAL}/logs/" 2>/dev/null || true

echo "Done. Check ${LOCAL}/results/"
