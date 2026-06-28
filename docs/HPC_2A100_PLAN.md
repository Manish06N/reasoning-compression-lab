# HPC 2× A100 (80 GB) — Publication Plan

**SLURM limit:** 48 hours max per job  
**Strategy:** Small/quants on RTX 5080; BF16 anchors + GPQA on HPC in ≤47 h blocks  
**Protocol (both machines):** `repro_qrm.yaml`, batch_size=1, full datasets, seed 0

---

## Machine split

| Machine | VRAM | Role | Entry script |
|---------|------|------|--------------|
| **RTX 5080** | 16 GB | All quants + Qwen-1.5B BF16 | `scripts/local/run_5080_publication.sh` |
| **HPC 2× A100** | 160 GB total | BF16 7B/8B anchors, GPQA | `scripts/hpc/run_hpc_2a100_publication.sh` |

---

## RTX 5080 cells (13 inference + smoke)

Runs locally — no SLURM time limit. Expect **~4–5 weeks** total.

| # | Cell | Model × Quant | Task | n |
|---|------|---------------|------|---|
| 0 | smoke | Qwen-1.5B BF16 | smoke | 1 |
| 1–4 | level_c_qwen15b_* | 1.5B BF16/FP8/AWQ/GPTQ | MATH-500 | 500 |
| 5 | level_a_gptq4 | Qwen-7B GPTQ-4 | MATH-500 | 500 |
| 6–9 | level_b_qwen7b_* | 7B FP8/AWQ/GPTQ-4/GPTQ-3 | MATH-500 | 500 |
| 10 | level_b_gsm8k | Qwen-7B FP8 | GSM8K | 1319 |
| 11–13 | level_c_llama8b_* | 8B FP8/AWQ/GPTQ | MATH-500 | 500 |

**Archive:** `outputs-win5080-main-YYYY-MM-DD/`

```bash
# Windows / WSL
bash scripts/local/run_5080_publication.sh --skip-download
bash scripts/local/start_5080_main.sh
```

---

## HPC blocks (each ≤ 47 h)

### Block b01 — 2× A100 parallel (~12–24 h wall)

| GPU | Cell | Model | Task | n | Est. |
|-----|------|-------|------|---|------|
| 0 | `level_a_bf16_seed0` | Qwen-7B **BF16** | MATH-500 | 500 | 12–24 h |
| 1 | `level_c_llama8b_bf16` | Llama-8B **BF16** | MATH-500 | 500 | 12–24 h |

Both GPUs run **simultaneously** → wall time ≈ slower of the two, well under 48 h.

> `level_b_qwen7b_bf16` duplicates Level A — **do not run** on HPC.

**Archive:** `outputs-hpc-2a100-main-YYYY-MM-DD/`

```bash
# On PARAM Rudra
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
git pull
bash scripts/hpc/submit_hpc_blocks.sh b01
# or interactively:
bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors
```

### Block b02 — 1× A100 (~8–20 h, after GPQA gate)

| GPU | Cell | Model | Task |
|-----|------|-------|------|
| 0 | `level_c_qwen7b_fp8_gpqa` | Qwen-7B FP8 | GPQA-Diamond |

Submit only after Hugging Face gated access (see `docs/GPQA_ACCESS.md`):

```bash
sbatch slurm/hpc_2a100_b02_gpqa.slurm
```

---

## SLURM templates

| File | GPUs | Time | Block |
|------|------|------|-------|
| `slurm/hpc_2a100_b01_parallel.slurm` | 2× A100 | 47:00:00 | b01 |
| `slurm/hpc_2a100_b02_gpqa.slurm` | 1× A100 | 47:00:00 | b02 |

---

## HPC setup (after git pull)

```bash
ssh manishn_iitp@paramrudra.iitp.ac.in -p 4422
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
git pull origin main

source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh
conda activate qreason

# Download BF16 weights if missing
bash scripts/hpc/02_download_model.sh   # Qwen-7B
# Llama-8B BF16: huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B ...

bash scripts/hpc/submit_hpc_blocks.sh b01
squeue -u $USER
```

---

## Merging results for the paper

| Archive | Cells |
|---------|-------|
| `outputs-win5080-main-*` | All quants + 1.5B |
| `outputs-hpc-2a100-main-*` | BF16 7B + BF16 8B (+ GPQA) |

Combine `results/*_summary.json` from both archives into `results/` for tables.

---

## Future: seed sweeps (Level B/C)

Each seed multiplies runtime. Options:

- **5080:** run seeds 1–4 for quant cells (slow but no SLURM cap)
- **HPC:** one seed per 48 h block, or split MATH-500 into halves with `--limit` + resume

Not in current blocks — add after seed-0 grid completes.
