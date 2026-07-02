# HPC post-merge checklist (manual steps)

Run these on PARAM Rudra **after** syncing this code pass to `main`. These steps complete the review's scientific-readiness loop and cannot be done from MacBook alone.

## 1. Export locked dependencies

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR && conda activate qreason
bash scripts/hpc/export_requirements_lock.sh
# Review requirements-hpc.lock.txt, commit from MacBook after rsync pull
```

## 2. Fresh publication archive (b01)

```bash
rm -rf outputs-hpc-2a100-main-2026-06-29   # diagnostic only — do not cite
export QREASON_OUTPUT_ROOT=$QR/outputs-hpc-2a100-main-$(date +%Y-%m-%d)-rerun
export QREASON_FRESH_RUN=1
cd $QR && git fetch origin && git reset --hard origin/main
python scripts/hpc/07_preflight_publication.py
bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors
```

## 3. Score and reproduction gate

After b01 completes and you sync MacBook → GitHub → HPC:

```bash
python scripts/score_run.py \
  --input $QREASON_OUTPUT_ROOT/raw/level_a_qwen7b_bf16_math500_seed0.jsonl \
  --summary $QREASON_OUTPUT_ROOT/results/level_a_qwen7b_bf16_math500_seed0_summary.json \
  --skip-calibration
python scripts/compare_qrm_baseline.py \
  --summary $QREASON_OUTPUT_ROOT/results/level_a_qwen7b_bf16_math500_seed0_summary.json
```

## 4. Multi-seed subset (if b01–b09 trends are clear)

Per [J1_VALIDATION_RUNBOOK.md](J1_VALIDATION_RUNBOOK.md): add seed1/seed2 only for key Qwen-7B/Llama-8B MATH-500 cells after seed0 looks sane.

## 5. Trace audit

Manually audit **20–50 traces per task/model family** before manuscript calibration or pass@1 claims. Log findings in `CHANGELOG.md`.

## 6. Update ops log

Record in [CHANGELOG.md](../CHANGELOG.md) and [progress.md](../progress.md):

- SLURM job IDs
- `QREASON_OUTPUT_ROOT` path
- pass@1 vs QRM bands
- Any resume/config_hash blocks (expected after this code pass)
