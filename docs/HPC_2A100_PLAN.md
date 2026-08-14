# HPC 2× A100 (80 GB) — Publication Plan

**SLURM limit:** 48 hours max per job  
**Rule:** Paper 1/J1 publication numbers run on PARAM Rudra HPC. RTX 5080 outputs are historical or J3 local-transfer only.
**Protocol:** historical b01–b09 used `repro_qrm.yaml`/seed 0. That grid is **not authorized for publication expansion** after the 2026-08-14 audit. Current protocol and seeds are defined in [the recovery plan](plans/2026-08-14-publication-recovery.md).

GitHub: [reasoning-compression-lab](https://github.com/Manish06N/reasoning-compression-lab)

**Current status (2026-08-14):** modern-stack b02 jobs **96086/96087** were canceled for unhealthy output. Exact-stack full jobs **96100/96101 completed** at 94.4% Qwen and 89.0% Llama. The [publication audit](PUBLICATION_READINESS.md) classifies them as replication/control evidence. Do not submit b03/b04 or the broad block grid; recovery Phase 0 is next.

> The block tables below are retained as historical resource planning. They are not the current scientific execution order.

First b02 attempt jobs **96084/96085** failed before raw rows with the vLLM FP8 checkpoint plus FP8 KV-cache incompatibility; commit `542f622` fixes FP8 configs to `kv_cache_dtype: auto`.

---

## Machine split

| Machine | VRAM | Role | Entry script |
|---------|------|------|--------------|
| **RTX 5080** | 16 GB | Historical/J3 local-transfer only; not J1 publication numbers | archived local scripts |
| **HPC 2× A100** | 160 GB total | All 7B/8B cells, GSM8K, GPQA | `scripts/hpc/run_hpc_2a100_publication.sh` |

**Do NOT run on 5080:** Qwen-7B, Llama-8B (any quant), GSM8K, BF16 anchors — they OOM or take weeks at batch_size=1.

---

## RTX 5080 cells (historical/J3 only)

Queue: `configs/machine_split/5080_cells.sh`

| # | Cell | Model × Quant | Task | n | Est. |
|---|------|---------------|------|---|------|
| 0 | smoke | Qwen-1.5B BF16 | smoke | 1 | minutes |
| 1 | level_c_qwen15b_bf16 | 1.5B BF16 | MATH-500 | 500 | ~≤24 h |
| 2 | level_c_qwen15b_fp8 | 1.5B FP8 | MATH-500 | 500 | ~≤24 h |
| 3 | level_c_qwen15b_awq4 | 1.5B AWQ-4 | MATH-500 | 500 | ~≤24 h |
| 4 | level_c_qwen15b_gptq4 | 1.5B GPTQ-4 | MATH-500 | 500 | ~≤24 h |

**Archive:** `outputs-win5080-main-YYYY-MM-DD/` (historical only for J1)

```bash
# Historical/J3 only - do not use for current J1 paper numbers
bash scripts/local/run_5080_publication.sh --skip-download
bash scripts/local/start_5080_main.sh
```

---

## HPC blocks (each <= 47 h)

Historical monitor commands (completed jobs; logs remain useful):

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
squeue -u $USER
tail -f logs/qrm_official_96100.err
tail -f logs/qrm_official_96101.err
```

Do not submit later blocks through this historical path:

```bash
# BLOCKED: b03/b04 require recovery P0, matched P1, and an approved pilot plan
```


### b01 — 2× A100 parallel (~12–24 h)

| GPU | Cell | Model | Task | n |
|-----|------|-------|------|---|
| 0 | `level_a_bf16_seed0` | Qwen-7B **BF16** | MATH-500 | 500 |
| 1 | `level_c_llama8b_bf16` | Llama-8B **BF16** | MATH-500 | 500 |

> **`level_b_qwen7b_bf16`** uses the **sober** prompt profile (Level B grid), not Level A **reproduction** — it is **not** a duplicate of `level_a_bf16_seed0`. Do not compare pass@1 across profiles directly.

### b02 — 2× A100 parallel (~12–24 h)

| GPU | Cell | Model | Task | n |
|-----|------|-------|------|---|
| 0 | `level_b_qwen7b_fp8` | Qwen-7B FP8 | MATH-500 | 500 |
| 1 | `level_c_llama8b_fp8` | Llama-8B FP8 | MATH-500 | 500 |

### b03 — 2× A100 parallel (~12–24 h)

| GPU | Cell | Model | Task | n |
|-----|------|-------|------|---|
| 0 | `level_b_qwen7b_awq4` | Qwen-7B AWQ-4 | MATH-500 | 500 |
| 1 | `level_c_llama8b_awq4` | Llama-8B AWQ-4 | MATH-500 | 500 |

### b04 — 2× A100 parallel (~12–24 h)

| GPU | Cell | Model | Task | n |
|-----|------|-------|------|---|
| 0 | `level_a_gptq4` | Qwen-7B GPTQ-4 | MATH-500 | 500 |
| 1 | `level_c_llama8b_gptq4` | Llama-8B GPTQ-4 | MATH-500 | 500 |

### b05 — 1× A100 (~12–20 h)

| GPU | Cell | Model | Task | n |
|-----|------|-------|------|---|
| 0 | `level_b_qwen7b_gptq3` | Qwen-7B GPTQ-3 | MATH-500 | 500 |

### b06 — 1× A100 (~20–40 h)

| GPU | Cell | Model | Task | n |
|-----|------|-------|------|---|
| 0 | `level_b_gsm8k` | Qwen-7B FP8 | GSM8K | 1319 |

### b07 — 1× A100 (~8–20 h, after GPQA gate)

| GPU | Cell | Model | Task |
|-----|------|-------|------|
| 0 | `level_c_qwen7b_fp8_gpqa` | Qwen-7B FP8 | GPQA-Diamond |

Submit only after Hugging Face gated access (see `docs/GPQA_ACCESS.md`):

```bash
sbatch slurm/hpc_2a100_b07_gpqa.slurm
```

**Archive:** `outputs-hpc-2a100-main-YYYY-MM-DD/`

---

## Publication sufficiency and expansion rule

The b01–b09 seed-0 grid is a historical wiring/resource map, **not** the current publishable core. Do not submit it as a wave.

Current order:

1. Recovery Phase 0 and tiny smoke.
2. Four matched BF16/FP8 seed-42 cells.
3. Review, then 24 MATH-500 pilot cells across seeds 42–44.
4. Contribution gate.
5. Extend selected headline cells to seeds 42–46 and authorize breadth only if justified.

See [plans/2026-08-14-publication-recovery.md](plans/2026-08-14-publication-recovery.md).

---

## SLURM templates

| File | GPUs | Time | Block |
|------|------|------|-------|
| `slurm/hpc_2a100_b01_parallel.slurm` | 2× A100 | 47:00:00 | b01 |
| `slurm/hpc_2a100_b07_gpqa.slurm` | 1× A100 | 47:00:00 | b07 |

Blocks b02–b06 use `submit_hpc_blocks.sh` (inline `--wrap` sbatch).

---

## HPC setup (after git push + pull)

```bash
ssh manishn_iitp@paramrudra.iitp.ac.in -p 4422
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
git fetch origin && git reset --hard origin/main

source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh
conda activate qreason

bash scripts/hpc/01_gpu_check.sh
bash scripts/hpc/02_download_model.sh   # Qwen-7B + quants
# Llama-8B BF16 if missing:
# huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B ...

# Current order: recovery Phase 0, tiny smoke, then matched P1 only
```

---

## Merging results for the paper

| Archive | Cells |
|---------|-------|
| `outputs-win5080-main-*` | Qwen-1.5B × 4 quants |
| `outputs-hpc-2a100-main-*` | All 7B/8B, GSM8K, GPQA |

Combine `results/*_summary.json` from both archives into `results/` for tables.

---

## Future: seed sweeps (Level B/C)

Each seed multiplies runtime. Not in current blocks — add after seed-0 grid completes.

- **5080:** seeds 1–4 for 1.5B cells only
- **HPC:** one seed per 48 h block, or split datasets with `--limit` + resume
