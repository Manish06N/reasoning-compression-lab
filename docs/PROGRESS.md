# Progress — Paper 1 Experiments

**Last updated:** 2026-06-28 (evening — 5080 stopped, HPC-only)  
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Local path:** `G:\ALL MY Projects\2026\03-paper1-experiments`

---

## Summary (2026-06-28)

| Area | Status |
|------|--------|
| **Policy** | **HPC-only** for all publication experiments |
| HPC block scripts (b01–b07) | Ready in repo |
| GitHub | Pushed after 5080 stop + doc updates |
| 5080 | **Stopped** — partial test only (`10/500` rows, do not cite) |
| HPC runs | **Not started** — your next step |

5080 was tried for Qwen-1.5B only; at ~15 min/question the 4-cell grid would take **~3 weeks**. Run stopped; everything moves to PARAM Rudra.

---

## Q1 publication analysis utilities

Post-run analysis now includes bootstrap 95% confidence intervals, explicit failure-rate summaries, task-aware GSM8K/GPQA scoring, paper-table CSV generation, and a reproducibility-bundle script. After each archive is scored, run `scripts/build_paper_tables.py --archive <outputs-hpc-...>` and `scripts/build_repro_bundle.py --archive <outputs-hpc-...>`.

## Archive metadata

HPC publication archives now include `manifest.json` plus `metadata/<cell_id>.json` per-cell snapshots. These capture the exact cell, model, task, decoding, batch/checkpoint, git, SLURM, raw-output, summary, and row-count metadata needed to reconstruct each run. The backup mirror includes `metadata/` and is locked during full mirror updates.

## Publication sufficiency strategy

The b01-b09 HPC plan is the current minimum publishable core. It is enough to start the paper if the completed results show clean trends. Do not add broad new jobs before scoring b01-b09.

After b01-b09 complete:

1. Score all raw outputs and build the main tables.
2. Check whether trends are stable and interpretable.
3. If needed, add seed1/seed2 for a small key subset: Qwen-7B and Llama-8B on MATH-500 for BF16, FP8, AWQ-4, and GPTQ-4.

## Live run — RTX 5080

**Status: STOPPED (2026-06-28)** — user moved all experiments to HPC. Partial run archived; not for paper tables.

| Field | Value |
|-------|--------|
| Archive | `outputs-win5080-main-2026-06-28/` (partial — do not cite) |
| Stopped at | ~Q12/500 on cell 1; **10 rows saved** on disk |
| Reason | ~15 min/question → ~3 weeks for 4 cells; PC cannot stay on that long |

### Cell status (seed 0)

| # | Cell | Model | Task | Status | Progress |
|---|------|-------|------|--------|----------|
| 0 | smoke_qwen15b_bf16 | Qwen-1.5B BF16 | smoke | completed | 1/1 |
| 1 | level_c_qwen15b_bf16_math500_seed0 | Qwen-1.5B BF16 | MATH-500 | **stopped** | 10/500 saved (discard) |
| 2 | level_c_qwen15b_fp8_math500_seed0 | Qwen-1.5B FP8 | MATH-500 | **→ HPC** | not started |
| 3 | level_c_qwen15b_awq4_math500_seed0 | Qwen-1.5B AWQ-4 | MATH-500 | **→ HPC** | not started |
| 4 | level_c_qwen15b_gptq4_math500_seed0 | Qwen-1.5B GPTQ-4 | MATH-500 | **→ HPC** | not started |

### Why 5080 was stopped (timing sample, cell 1 BF16)

| Question | Wall time |
|----------|-----------|
| Q1 | ~50 s |
| Q2–Q4 | ~21 min each |
| Q5–Q11 | ~15 min each |

At ~15 min/question → **~5 days/cell**, **~3 weeks for 4 cells** — not viable on desktop.

---

## Pending — HPC (PARAM Rudra)

| Block | GPUs | Content | Status |
|-------|------|---------|--------|
| b01 | 2× A100 | BF16 Qwen-7B + BF16 Llama-8B MATH-500 | not submitted |
| b02 | 2× A100 | FP8 Qwen-7B + FP8 Llama-8B MATH-500 | not submitted |
| b03 | 2× A100 | AWQ-4 Qwen-7B + AWQ-4 Llama-8B MATH-500 | not submitted |
| b04 | 2× A100 | GPTQ-4 Qwen-7B + GPTQ-4 Llama-8B MATH-500 | not submitted |
| b05 | 1× A100 | GPTQ-3 Qwen-7B MATH-500 | not submitted |
| b06 | 1× A100 | FP8 Qwen-7B GSM8K (n=1319) | not submitted |
| b07 | 1× A100 | GPQA-Diamond | blocked — HF gate |

**Next commands on HPC:**

```bash
ssh manishn_iitp@paramrudra.iitp.ac.in -p 4422
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR && git pull origin main
source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh && conda activate qreason
# download 7B/8B models if missing (see HPC_2A100_PLAN.md)
bash scripts/hpc/submit_hpc_blocks.sh
squeue -u $USER
```

**HPC archive (when started):** `outputs-hpc-2a100-main-YYYY-MM-DD/`

---

## What we built today

### Policy

- **5080:** only jobs expected to finish in ≤ ~24 h **per cell** → Qwen-1.5B × 4 quants only
- **HPC:** all 7B/8B quants, BF16 anchors, GSM8K, GPQA (SLURM ≤48 h per block)
- Rejected running 13-cell grid on 5080 (would take weeks at batch_size=1)

### Scripts & config

| Path | Purpose |
|------|---------|
| `scripts/local/run_5080_publication.sh` | Canonical 5080 entry (4-cell queue) |
| `scripts/local/start_5080_main.sh` | Background launcher |
| `scripts/hpc/run_hpc_2a100_publication.sh` | HPC block runner b01–b07 |
| `scripts/hpc/submit_hpc_blocks.sh` | SLURM submit helper |
| `configs/machine_split/5080_cells.sh` | 5080 cell queue |
| `configs/machine_split/hpc_blocks/b01`–`b06` | HPC block definitions |
| `slurm/hpc_2a100_b01_parallel.slurm` | BF16 parallel template |
| `slurm/hpc_2a100_b07_gpqa.slurm` | GPQA block |

### Documentation

| Doc | Content |
|-----|---------|
| `docs/HPC_2A100_PLAN.md` | Full split + block table |
| `docs/GIT_CREDENTIALS.md` | Safe PAT storage (Credential Manager) |
| `docs/RTX5080_EXECUTION_PLAN.md` | 5080 = 1.5B only |
| `docs/MODEL_ROSTER.md` | Machine assignment |
| `README.md` | Push guide, block grid, quick commands |

### Git / GitHub

| Commit | Message |
|--------|---------|
| `30c8c08` | Add 5080/HPC split with publication run scripts and 48h SLURM blocks |
| `03c3766` | Revise split: 5080 ≤24h (1.5B only); full 7B/8B grid on HPC b01-b07 |

Pushed to https://github.com/Manish06N/reasoning-compression-lab on 2026-06-28. Token stored in Windows Credential Manager (revoke any token shared in chat and rotate).

---

## Next actions

1. **HPC:** `git pull` → download models → `bash scripts/hpc/submit_hpc_blocks.sh` (b01–b06)
2. **HPC:** add/run 1.5B cells there if still in paper grid (not in b01–b06 yet — extend blocks or run manually)
3. ~~5080~~ — do not restart long runs locally
4. **GPQA:** HF gate → submit b07
5. Merge `outputs-hpc-2a100-main-*` summaries when complete

---

## Related docs

- [**progress.md**](../progress.md) — master dated progress log (full MacBook + HPC + 5080 history)
- [HPC_2A100_PLAN.md](HPC_2A100_PLAN.md)
- [RTX5080_EXECUTION_PLAN.md](RTX5080_EXECUTION_PLAN.md)
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
- [CHANGELOG.md](../CHANGELOG.md)
