# HPC post-merge checklist (manual steps)

> **Historical checklist — superseded 2026-08-14.** Do not execute the old b01 seed-0 flow for new paper results. The current order is [plans/2026-08-14-publication-recovery.md](plans/2026-08-14-publication-recovery.md): reproducibility/observability Phase 0, tiny smoke, matched BF16/FP8, then a reviewed three-seed pilot.

The commands below are retained for incident provenance and may contain useful mechanics only.

## 0. Cancel stale b01 job (if still queued)

```bash
scancel 86212 2>/dev/null || true
squeue -u "$USER" | grep -E '86212|b01' || echo "86212 not running"
```

Do **not** use archives from job 86212 or earlier pre-P0 runs for paper numbers.

## 1. Export locked dependencies

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR && conda activate qreason
bash scripts/hpc/export_requirements_lock.sh
# Review requirements-hpc.lock.txt, commit from MacBook after rsync pull
```

## 2. Fresh publication archive (b01)

```bash
export QREASON_OUTPUT_ROOT=$QR/outputs-hpc-2a100-main-2026-07-02-p0fix
export QREASON_FRESH_RUN=1
cd $QR && git fetch origin && git reset --hard origin/main
python scripts/hpc/07_preflight_publication.py
bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors
```

## 3. Score and reproduction gate

After b01 completes and you sync MacBook → GitHub → HPC:

```bash
python scripts/score_run.py \
  --publication \
  --input outputs-hpc-.../raw/level_a_qwen7b_bf16_math500_seed0.jsonl \
  --output outputs-hpc-.../scored/level_a_qwen7b_bf16_math500_seed0.jsonl \
  --summary outputs-hpc-.../results/level_a_qwen7b_bf16_math500_seed0_summary.json

python scripts/compare_qrm_baseline.py --summary outputs-hpc-.../results/level_a_qwen7b_bf16_math500_seed0_summary.json
python scripts/build_repro_bundle.py --archive outputs-hpc-2a100-main-2026-07-02-p0fix
```

## 4. Manual trace audit (required before citing numbers)

Sample 200 raw completions per cell; verify extraction, scoring, and truncation handling. Record audit notes in `progress.md`.
