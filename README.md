# reasoning-compression-lab

Deployment-science evaluation harness for compressed reasoning LLMs.

**Paper 1:** *Beyond Accuracy: Reliability, Calibration, Seed Variance, and Cost-per-Correct of Quantized Reasoning LLMs*

## Publication goal (journal)

> **Split execution:** quants + 1.5B on **RTX 5080**; BF16 7B/8B + GPQA on **HPC 2× A100** (≤48 h SLURM blocks).  
> Full plan: [HPC_2A100_PLAN.md](docs/HPC_2A100_PLAN.md)

| Machine | Entry script | What runs there |
|---------|--------------|-----------------|
| **RTX 5080** | `scripts/local/run_5080_publication.sh` | 13 quant cells + 1.5B BF16 (fits 16 GB) |
| **HPC 2× A100** | `scripts/hpc/run_hpc_2a100_publication.sh` | BF16 Qwen-7B + Llama-8B anchors; GPQA |

**Protocol (both):** `repro_qrm.yaml`, batch_size=1, full n, seed 0.

## Research question

When reasoning LLMs are compressed, do they remain **accurate, calibrated, stable, fast, memory-efficient, and economically useful** under real serving conditions?

## Division of labor

| Machine | Role |
|---------|------|
| **MacBook** | Design docs, scripts, configs, writing, plotting from CSVs, log review |
| **Windows 5080 (WSL2)** | **Primary:** quant grid + 1.5B (`run_5080_publication.sh`) |
| **HPC 2× A100** | **BF16 anchors** + GPQA; ≤48 h SLURM blocks (`run_hpc_2a100_publication.sh`) |

## Repository layout

```
configs/          Experiment cell, task, model, and quantization configs
prompts/          Task prompts and templates
scripts/          CLI entrypoints and batch helpers
src/              Runners, metrics, extraction, stats, profiling, quantization
runs/             Raw → extracted → scored pipeline outputs
results/          Aggregated CSVs and tables for paper
paper_figures/    Final publication figures
paper/            Manuscript draft
docs/             Design, runbook, experiment log
slurm/            HPC job scripts
notebooks/        Analysis dashboards
```

## Execution gates

1. **Level A — reproduction:** Qwen-7B × {BF16, GPTQ-4} × MATH-500 × seed 0
2. **Level B — pilot:** Qwen-7B × 5 configs × {MATH-500, GPQA-Diamond} × 5 seeds
3. **Level C — main grid:** 3 models × 5 configs × 4 tasks × 5 seeds

No full grid before reproduction works. No 14B before Qwen-7B pilot works.

## First experiment

```
Model:  DeepSeek-R1-Distill-Qwen-7B
Task:   MATH-500
Config: BF16
Seed:   0
Hardware: A100
```

## Reference repos

Cloned under `../external_repos/` for reading only — do not develop inside them. See `../external_repos/README.md` and `../external_repos/EXTERNAL_REPOS_REFERENCE.md`.

**Core paper baselines:** Quantized-Reasoning-Models, sober-reasoning, Calibrating-LLMs-with-Consistency, Cost-of-Pass

**Method/tool references:** gptq, smoothquant, vllm, lm-evaluation-harness, AbstentionBench

## Docs

- [HPC_2A100_PLAN.md](docs/HPC_2A100_PLAN.md) — **5080 vs HPC split + 48 h SLURM blocks**
- [RTX5080_EXECUTION_PLAN.md](docs/RTX5080_EXECUTION_PLAN.md) — 5080 local runs
- [MODEL_ROSTER.md](docs/MODEL_ROSTER.md) — canonical HF IDs and machine assignment
- [HPC_STEP_BY_STEP.md](docs/HPC_STEP_BY_STEP.md) — HPC gate-by-gate guide
- [PAPER1_DESIGN.md](docs/PAPER1_DESIGN.md) — scope, models, tasks, metrics, claim
- [RUNBOOK.md](docs/RUNBOOK.md) — MacBook ↔ HPC workflow
- [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) — dated experiment record

## HPC quick commands (legacy gates)

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR && conda activate qreason
bash scripts/hpc/01_gpu_check.sh          # Gate 1
bash scripts/hpc/02_download_model.sh     # Gate 2
bash scripts/hpc/03_smoke_test.sh         # Gate 3
bash scripts/hpc/04_run_level_a_bf16.sh 10  # Gate 4 debug
bash scripts/hpc/05_score_level_a.sh
sbatch slurm/run_level_a_bf16.slurm       # Gate 4 full
```

## Windows RTX 5080

```bash
bash scripts/local/run_5080_publication.sh --skip-download
bash scripts/local/start_5080_main.sh
```

Archive: `outputs-win5080-main-YYYY-MM-DD/`

## HPC 2× A100 (PARAM Rudra)

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR && git pull
bash scripts/hpc/submit_hpc_blocks.sh b01
```

Archive: `outputs-hpc-2a100-main-YYYY-MM-DD/`  
See [HPC_2A100_PLAN.md](docs/HPC_2A100_PLAN.md).
