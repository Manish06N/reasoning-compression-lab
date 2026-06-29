# HPC 2× A100 (80 GB) — Publication Plan

**SLURM limit:** 48 hours max per job  
**Rule:** RTX 5080 runs **only** cells expected to finish in **≤24 h each** (Qwen-1.5B). Everything else → HPC.  
**Protocol (both machines):** `repro_qrm.yaml`, batch_size=1, full datasets, seed 0

GitHub: [reasoning-compression-lab](https://github.com/Manish06N/reasoning-compression-lab)

---

## Machine split

| Machine | VRAM | Role | Entry script |
|---------|------|------|--------------|
| **RTX 5080** | 16 GB | Qwen-1.5B × 4 quants × MATH-500 (~1 day/cell, ~4 days total) | `scripts/local/run_5080_publication.sh` |
| **HPC 2× A100** | 160 GB total | All 7B/8B cells, GSM8K, GPQA | `scripts/hpc/run_hpc_2a100_publication.sh` |

**Do NOT run on 5080:** Qwen-7B, Llama-8B (any quant), GSM8K, BF16 anchors — they OOM or take weeks at batch_size=1.

---

## RTX 5080 cells (4 inference + smoke)

Queue: `configs/machine_split/5080_cells.sh`

| # | Cell | Model × Quant | Task | n | Est. |
|---|------|---------------|------|---|------|
| 0 | smoke | Qwen-1.5B BF16 | smoke | 1 | minutes |
| 1 | level_c_qwen15b_bf16 | 1.5B BF16 | MATH-500 | 500 | ~≤24 h |
| 2 | level_c_qwen15b_fp8 | 1.5B FP8 | MATH-500 | 500 | ~≤24 h |
| 3 | level_c_qwen15b_awq4 | 1.5B AWQ-4 | MATH-500 | 500 | ~≤24 h |
| 4 | level_c_qwen15b_gptq4 | 1.5B GPTQ-4 | MATH-500 | 500 | ~≤24 h |

**Archive:** `outputs-win5080-main-YYYY-MM-DD/`

```bash
# Windows / WSL
bash scripts/local/run_5080_publication.sh --skip-download
bash scripts/local/start_5080_main.sh
```

---

## HPC blocks (each ≤ 47 h)

Submit all ready blocks:

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR && git pull
bash scripts/hpc/submit_hpc_blocks.sh        # b01–b06
bash scripts/hpc/submit_hpc_blocks.sh b01    # one block
```

### b01 — 2× A100 parallel (~12–24 h)

| GPU | Cell | Model | Task | n |
|-----|------|-------|------|---|
| 0 | `level_a_bf16_seed0` | Qwen-7B **BF16** | MATH-500 | 500 |
| 1 | `level_c_llama8b_bf16` | Llama-8B **BF16** | MATH-500 | 500 |

> `level_b_qwen7b_bf16` duplicates Level A — **do not run** separately.

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

The b01-b09 seed0 grid is the current first publishable core result set. It covers:

- Qwen-1.5B, Qwen-7B, and Llama-8B.
- BF16, FP8, AWQ-4, GPTQ-4, and GPTQ-3.
- MATH-500, GSM8K, and GPQA-Diamond.

Do not add broad new experiments before b01-b09 are scored. If the paper needs robustness, add a small multi-seed subset rather than repeating the full grid: seed1/seed2 for Qwen-7B and Llama-8B on MATH-500 with BF16, FP8, AWQ-4, and GPTQ-4.

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
git pull origin main

source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh
conda activate qreason

bash scripts/hpc/01_gpu_check.sh
bash scripts/hpc/02_download_model.sh   # Qwen-7B + quants
# Llama-8B BF16 if missing:
# huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B ...

bash scripts/hpc/submit_hpc_blocks.sh
squeue -u $USER
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
