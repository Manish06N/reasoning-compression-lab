#!/usr/bin/env bash
# Wait for QRM official env install (job 87130) then submit Llama n=10 on a second GPU.
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
MARKER="$QR/.qrm_official_env_ready"
PARENT_JOB="${QRM_PARENT_JOB:-87130}"
LOG="$QR/logs/qrm_official_${PARENT_JOB}.err"
SUBMIT_LOG="$QR/logs/qrm_official_llama_watch.log"
DONE_FLAG="$QR/.qrm_official_llama_job_submitted"

exec >>"$SUBMIT_LOG" 2>&1
echo "=== Llama watcher started $(date) ==="
echo "Waiting for marker: $MARKER (parent job $PARENT_JOB)"

if [[ -f "$DONE_FLAG" ]]; then
  echo "Already submitted ($(cat "$DONE_FLAG")); exit."
  exit 0
fi

for i in $(seq 1 120); do
  if [[ -f "$MARKER" ]] || grep -q "QRM official env ready" "$LOG" 2>/dev/null; then
    echo "Install ready (poll $i) at $(date); sleeping 30s buffer..."
    sleep 30
    cd "$QR"
    OUT="$QR/outputs-hpc-qrm-official-llama-$(date +%Y-%m-%d)"
    JOBID=$(QRM_MODEL_PATH="$QR/models/DeepSeek-R1-Distill-Llama-8B" \
      QRM_OUTPUT_ROOT="$OUT" \
      sbatch --parsable --job-name=qreason-qrm_official_llama_n10 \
      slurm/qrm_official_math500_n10.slurm)
    echo "$JOBID" >"$DONE_FLAG"
    echo "Submitted Llama job $JOBID at $(date)"
    echo "  Model: $QR/models/DeepSeek-R1-Distill-Llama-8B"
    echo "  Output: $OUT"
    squeue -j "$JOBID" || true
    exit 0
  fi

  if ! squeue -j "$PARENT_JOB" -h 2>/dev/null | grep -q .; then
    echo "Parent job $PARENT_JOB left queue at $(date)"
    if [[ -f "$MARKER" ]] || grep -q "QRM official env ready" "$LOG" 2>/dev/null; then
      echo "Marker present after parent exit; submitting..."
      sleep 10
      cd "$QR"
      OUT="$QR/outputs-hpc-qrm-official-llama-$(date +%Y-%m-%d)"
      JOBID=$(QRM_MODEL_PATH="$QR/models/DeepSeek-R1-Distill-Llama-8B" \
        QRM_OUTPUT_ROOT="$OUT" \
        sbatch --parsable --job-name=qreason-qrm_official_llama_n10 \
        slurm/qrm_official_math500_n10.slurm)
      echo "$JOBID" >"$DONE_FLAG"
      echo "Submitted Llama job $JOBID"
      exit 0
    fi
    echo "ERROR: parent ended without install marker — check logs/qrm_official_${PARENT_JOB}.out"
    tail -40 "$QR/logs/qrm_official_${PARENT_JOB}.out" || true
    exit 1
  fi

  du -sh "$QR/external/Quantized-Reasoning-Models/third-party/lighteval" \
    "$QR/external/Quantized-Reasoning-Models/third-party/vllm" 2>/dev/null \
    | tr '\n' ' '
  echo "| poll $i/120 $(date +%H:%M:%S)"
  sleep 60
done

echo "ERROR: timed out waiting for QRM official env (120 min)"
exit 1