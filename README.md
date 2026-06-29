# reasoning-compression-lab

Deployment-science evaluation harness for compressed reasoning LLMs.

**GitHub:** https://github.com/Manish06N/reasoning-compression-lab  
**Paper 1:** *Beyond Accuracy: Reliability, Calibration, Seed Variance, and Cost-per-Correct of Quantized Reasoning LLMs*

## Current status (2026-06-29)

| Machine | Status | Details |
|---------|--------|---------|
| **5080** | **Stopped** | Partial test only (10/500 rows) — **HPC-only** for publication |
| **HPC** | **Next** | GPU smoke gate → `submit_hpc_blocks.sh` (b01–b06) |
| **GitHub** | Synced | See [progress.md](progress.md) for full timeline |

**Live tracker:** [docs/PROGRESS.md](docs/PROGRESS.md) · **Full history:** [progress.md](progress.md)

## Publication goal (journal)

> **Split execution:** 5080 = **≤24 h/cell only** (Qwen-1.5B); **all 7B/8B + GSM8K + GPQA** on **HPC 2× A100** (≤48 h SLURM blocks).  
> Full plan: [HPC_2A100_PLAN.md](docs/HPC_2A100_PLAN.md)

| Machine | Entry script | What runs there |
|---------|--------------|-----------------|
| **RTX 5080** | `scripts/local/run_5080_publication.sh` | Qwen-1.5B × 4 quants × MATH-500 (~4–7 days est.; see PROGRESS.md) |
| **HPC 2× A100** | `scripts/hpc/run_hpc_2a100_publication.sh` | 7B/8B all quants, BF16 anchors, GSM8K, GPQA |

**Protocol (both):** `repro_qrm.yaml`, batch_size=1, full n, seed 0.

### HPC block grid (seed 0)

| Block | GPUs | Est. time | Cells |
|-------|------|-----------|-------|
| b01 | 2× A100 | ~12–24 h | BF16 Qwen-7B + BF16 Llama-8B MATH-500 |
| b02 | 2× A100 | ~12–24 h | FP8 Qwen-7B + FP8 Llama-8B MATH-500 |
| b03 | 2× A100 | ~12–24 h | AWQ-4 Qwen-7B + AWQ-4 Llama-8B MATH-500 |
| b04 | 2× A100 | ~12–24 h | GPTQ-4 Qwen-7B + GPTQ-4 Llama-8B MATH-500 |
| b05 | 1× A100 | ~12–20 h | GPTQ-3 Qwen-7B MATH-500 |
| b06 | 1× A100 | ~20–40 h | FP8 Qwen-7B GSM8K (n=1319) |
| b07 | 1× A100 | ~8–20 h | GPQA-Diamond (after Hugging Face gate) |

**Do not run on 5080:** any 7B/8B cell, GSM8K, or BF16 anchors — they OOM or take weeks at batch_size=1.

## Publication sufficiency strategy

The current HPC plan is sufficient for a first publishable core result set if b01-b09 complete cleanly and produce clear trends. It covers three model families/scales, five compression settings, and MATH-500/GSM8K/GPQA-Diamond.

Do not expand the queue before seeing seed0 results. First finish b01-b09, score outputs, and build pass@1, trace length, latency, VRAM, cost-per-correct, and calibration/reliability tables. If robustness is needed, add seed1/seed2 only for the key Qwen-7B/Llama-8B MATH-500 cells: BF16, FP8, AWQ-4, and GPTQ-4.

## Push code to GitHub (Windows)

Your local repo: `G:\ALL MY Projects\2026\03-paper1-experiments`  
Remote: `https://github.com/Manish06N/reasoning-compression-lab.git`

**Status (2026-06-28):** Pushed successfully. Future updates: `git add -A && git commit -m "..." && git push origin main`.  
See [docs/GIT_CREDENTIALS.md](docs/GIT_CREDENTIALS.md) for PAT / Credential Manager setup.

### Step 1 — Open PowerShell in the project folder

```powershell
cd "G:\ALL MY Projects\2026\03-paper1-experiments"
git status
```

You should see a clean tree or local changes to commit.

### Step 2 — Stage and commit (if you have uncommitted changes)

```powershell
git add -A
git status
git commit -m "Revise split: 5080 ≤24h (1.5B only); full 7B/8B grid on HPC b01-b07."
```

Skip this step if `git status` shows a clean working tree and only says `ahead N`.

### Step 3 — Authenticate (pick one method)

#### Option A — GitHub CLI (recommended)

1. Install: https://cli.github.com/ (or `winget install GitHub.cli`)
2. Log in once:

```powershell
gh auth login
```

Choose: **GitHub.com** → **HTTPS** → **Login with a web browser** → follow the browser prompt.

3. Push:

```powershell
git push origin main
```

#### Option B — Personal Access Token (HTTPS)

GitHub no longer accepts account passwords for `git push`. Use a **Personal Access Token (PAT)** instead.

1. Open: https://github.com/settings/tokens
2. **Generate new token (classic)** → name it e.g. `reasoning-compression-lab`
3. Scopes: check **`repo`** (full control of private repositories)
4. Copy the token (you only see it once)

5. Push:

```powershell
git push origin main
```

When prompted:
- **Username:** `Manish06N`
- **Password:** paste the **token** (not your GitHub password)

To avoid re-entering the token, use Git Credential Manager (usually installed with Git for Windows):

```powershell
git config --global credential.helper manager
git push origin main
```

#### Option C — SSH key

1. Generate a key (if you don't have one):

```powershell
ssh-keygen -t ed25519 -C "your-email@example.com"
```

Press Enter to accept the default path (`C:\Users\manis\.ssh\id_ed25519`).

2. Copy the public key:

```powershell
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

3. Add it at: https://github.com/settings/keys → **New SSH key**

4. Switch remote to SSH and push:

```powershell
git remote set-url origin git@github.com:Manish06N/reasoning-compression-lab.git
git push origin main
```

### Step 4 — Verify on GitHub

Open https://github.com/Manish06N/reasoning-compression-lab and confirm you see:

- `scripts/hpc/submit_hpc_blocks.sh`
- `configs/machine_split/5080_cells.sh` (4 cells)
- `configs/machine_split/hpc_blocks/b01`–`b06`
- Updated `README.md` with the split table above

### Step 5 — Pull on HPC (after push succeeds)

```bash
ssh manishn_iitp@paramrudra.iitp.ac.in -p 4422
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
git pull origin main
bash scripts/hpc/submit_hpc_blocks.sh
squeue -u $USER
```

If the HPC folder does not exist yet, clone once:

```bash
cd /scratch/$USER
git clone https://github.com/Manish06N/reasoning-compression-lab.git
```

### Troubleshooting

| Error | Fix |
|-------|-----|
| `Authentication failed` | Use PAT or `gh auth login`, not your GitHub password |
| `rejected (fetch first)` | Someone else pushed first: `git pull origin main` then `git push origin main` |
| `Permission denied (publickey)` | SSH key not added to GitHub, or use HTTPS + PAT instead |
| Large files rejected | Outputs are gitignored; never commit `outputs-*` or model weights |

## Research question

When reasoning LLMs are compressed, do they remain **accurate, calibrated, stable, fast, memory-efficient, and economically useful** under real serving conditions?

## Division of labor

| Machine | Role |
|---------|------|
| **MacBook** | Design docs, scripts, configs, writing, plotting from CSVs, log review |
| **Windows 5080 (WSL2)** | **Short jobs only:** Qwen-1.5B grid (`run_5080_publication.sh`) |
| **HPC 2× A100** | **Main grid:** 7B/8B, GSM8K, GPQA; ≤48 h SLURM blocks (`run_hpc_2a100_publication.sh`) |

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

- [**progress.md**](progress.md) — **master dated progress log** (MacBook + HPC + 5080 full timeline)
- [PROGRESS.md](docs/PROGRESS.md) — **live status**, ETAs, next actions
- [BEGINNER_HPC_GUIDE.md](docs/BEGINNER_HPC_GUIDE.md) — **start here:** project overview + PARAM Rudra step-by-step
- [HPC_2A100_PLAN.md](docs/HPC_2A100_PLAN.md) — **5080 vs HPC split + 48 h SLURM blocks**
- [GIT_CREDENTIALS.md](docs/GIT_CREDENTIALS.md) — **store GitHub PAT safely (Credential Manager)**
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

**Local long runs disabled (2026-06-28).** Publication grid runs on **HPC only**. Partial archive `outputs-win5080-main-2026-06-28/` (10/500 rows) — not for paper.

For debug/smoke only:
```bash
bash scripts/local/run_5080_publication.sh --skip-download  # do not use for main grid
```

## HPC 2× A100 (PARAM Rudra)

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR && git pull
bash scripts/hpc/submit_hpc_blocks.sh        # b01–b06
# GPQA after HF gate: sbatch slurm/hpc_2a100_b07_gpqa.slurm
```

Archive: `outputs-hpc-2a100-main-YYYY-MM-DD/`  
See [HPC_2A100_PLAN.md](docs/HPC_2A100_PLAN.md).
