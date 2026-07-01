# Paper 1 — Beginner's Guide to `reasoning-compression-lab` on PARAM Rudra

**Audience:** Someone new to the project who will run all experiments on **HPC only** (no personal GPU).  
**Cluster:** PARAM Rudra, IIT Patna (`paramrudra.iitp.ac.in`)  
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Support:** `rudrasupport@iitp.ac.in` · ticket portal: https://paramrudra.iitp.ac.in/support  
**Last verified:** 2026-06-28 (cross-checked against repo scripts/configs and PARAM Rudra User Manual, IIT Patna, April 2025)

---

## 1. What this project is (and why it exists)

### Research question

> When reasoning LLMs are compressed (quantized), do they remain **accurate, calibrated, stable, fast, memory-efficient, and economically useful** under real serving conditions?

Most papers report a single accuracy number. This project treats deployment as a **systems problem**: a quantized model that is 2% less accurate but 4× faster and uses half the VRAM may be the better production choice — or it may be worse if calibration collapses and wrong answers are over-confident.

### Paper title

**Beyond Accuracy: Reliability, Calibration, Seed Variance, and Cost-per-Correct of Quantized Reasoning LLMs**

### What we compare

| Dimension | Details |
|-----------|---------|
| **Models** | DeepSeek-R1-Distill: Qwen-7B, Llama-8B, Qwen-1.5B |
| **Quantization** | BF16 (baseline), FP8, AWQ-4, GPTQ-4, GPTQ-3 |
| **Tasks** | MATH-500, GSM8K, GPQA-Diamond (gated); LiveCodeBench planned later |
| **Inference engine** | vLLM (production-style batch serving) |
| **Metrics** | pass@1, calibration (Brier, ECE), seed variance, latency, peak VRAM, cost-per-correct |

### Why this repo exists

`reasoning-compression-lab` is a **reproducible experiment harness**, not a training codebase. It:

1. Loads a model + quant config via vLLM  
2. Runs a benchmark (e.g. 500 MATH problems) with a fixed decoding protocol  
3. Saves every generation to JSONL with latency and VRAM  
4. Scores answers and writes a summary JSON for paper tables  

Everything is driven by **JSON configs** in `configs/` and **shell scripts** in `scripts/hpc/`.

---

## 2. Big-picture architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  configs/cells/*.json     <- one "experiment cell" (model x task x seed) │
│  configs/models/*.json    <- vLLM model paths and settings               │
│  configs/tasks/*.json     <- dataset name and split                      │
│  configs/decoding/*.yaml  <- temperature, top_p, max_tokens                │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                v
                    scripts/run_inference.py  (vLLM)
                                │
                                v
              outputs-.../raw/{cell_id}.jsonl   <- raw generations
                                │
                                v
                    scripts/score_run.py
                                │
                                v
              outputs-.../scored/{cell_id}.jsonl
              outputs-.../results/{cell_id}_summary.json  <- paper numbers
```

### Execution gates (strict order)

Do **not** jump to the full grid until early gates pass:

| Gate | What | Pass condition |
|------|------|----------------|
| **1** | GPU + software | `01_gpu_check.sh` — CUDA, torch, vLLM import |
| **2** | Model weights | Qwen-7B downloaded to `$QR/models/` |
| **2b** | Dataset | MATH-500 loads from Hugging Face |
| **3** | Smoke test | vLLM loads model, writes ≥1 JSONL line |
| **4** | Level A BF16 | 10-question debug, then full MATH-500 scored |
| **5** | Level A GPTQ-4 | Same after GPTQ weights verified |
| **6+** | Full grid | All blocks b01–b07 (+ 1.5B cells on HPC) |

---

## 3. What you will run on HPC

**All compute is on PARAM Rudra.** There is no personal GPU in this workflow.

### Publication grid (seed 0)

| Block | GPUs | What runs | Est. time |
|-------|------|-----------|-----------|
| **b01** | 2× A100 | Qwen-7B BF16 + Llama-8B BF16 (MATH-500) | ~12–24 h |
| **b02** | 2× A100 | Qwen-7B FP8 + Llama-8B FP8 | ~12–24 h |
| **b03** | 2× A100 | Qwen-7B AWQ-4 + Llama-8B AWQ-4 | ~12–24 h |
| **b04** | 2× A100 | Qwen-7B GPTQ-4 + Llama-8B GPTQ-4 | ~12–24 h |
| **b05** | 1× A100 | Qwen-7B GPTQ-3 | ~12–20 h |
| **b06** | 1× A100 | Qwen-7B FP8 × GSM8K (1319 Q) | ~20–40 h |
| **b07** | 1× A100 | Qwen-7B FP8 × GPQA-Diamond | ~8–20 h (after HF gate) |

### Qwen-1.5B cells (also on HPC)

These are small but still part of the paper grid. Run each as a **single-GPU job** (~≤24 h each):

| Cell config | Model × quant |
|-------------|---------------|
| `configs/cells/level_c_qwen15b_bf16_math500_seed0.json` | 1.5B BF16 |
| `configs/cells/level_c_qwen15b_fp8_math500_seed0.json` | 1.5B FP8 |
| `configs/cells/level_c_qwen15b_awq4_math500_seed0.json` | 1.5B AWQ-4 |
| `configs/cells/level_c_qwen15b_gptq4_math500_seed0.json` | 1.5B GPTQ-4 |

You can submit all four in parallel on separate GPUs once models are downloaded.

### Decoding protocol (paper standard)

From `configs/decoding/repro_qrm.yaml`:

| Setting | Value |
|---------|-------|
| temperature | 0.6 |
| top_p | 0.95 |
| max_tokens | 32768 |
| seed | 0 |
| batch_size | 1 |

Long reasoning traces mean **each cell can take many hours**. That is expected.

---

## 4. Hardware and time you need

### PARAM Rudra GPU nodes (from official manual)

| Resource | Spec |
|----------|------|
| GPU nodes | **8 nodes** |
| GPUs per node | **2× NVIDIA A100 80 GB** |
| CPU per node | 2× Intel Xeon Gold 6240R (48 cores/node) |
| RAM per node | 192 GB |
| Scratch | `/scratch/<username>/` — use this for models, cache, outputs |
| Max job walltime | **4 days** default (always set `--time` explicitly) |
| GPU partition | `--partition=gpu` |
| GPU request | `--gres=gpu:1` or `--gres=gpu:2` |

### Total compute (seed 0, HPC-only)

| Metric | Estimate |
|--------|----------|
| Peak concurrency | **2× A100** per publication block |
| Total GPU-hours | **~250–350** (7B/8B grid + 1.5B cells) |
| SLURM jobs | 7 blocks + 4× 1.5B cells + gate/smoke jobs |
| Wall-clock (if queued well) | ~**1–2 weeks** including queue wait |

---

## 5. Repository layout

```text
reasoning-compression-lab/
├── configs/
│   ├── cells/           # One JSON per experiment (model × task × seed)
│   ├── models/          # vLLM settings + HF paths per checkpoint
│   ├── tasks/           # MATH-500, GSM8K, GPQA
│   ├── decoding/        # repro_qrm.yaml (paper), smoke_debug.yaml (gates)
│   └── machine_split/   # HPC block definitions (b01–b07)
├── scripts/
│   ├── hpc/             # ← everything you run on PARAM Rudra
│   ├── run_inference.py # Main inference entrypoint
│   └── score_run.py     # Scoring entrypoint
├── slurm/               # Example SLURM scripts
├── src/runners/         # vLLM wrapper, checkpoints, config helpers
├── docs/                # Design docs and this guide
└── requirements-hpc.txt # Pinned stack for PARAM Rudra (vLLM 0.8.5)
```

**Not in git (you create on scratch):**

```text
models/          # Downloaded HF weights (~50–100+ GB total for full grid)
hf_cache/        # Hugging Face hub + datasets cache
outputs-hpc-*/   # Publication archives (raw, scored, results, logs)
runs/            # Legacy path used by gate scripts
logs/            # SLURM stdout/stderr
```

---

## 6. Prerequisites checklist

Before starting, confirm you have:

| Item | Where to get it |
|------|-----------------|
| PARAM Rudra account | IIT Patna / NSM registration - `rudrasupport@iitp.ac.in` |
| SSH client | Terminal (Mac/Linux) or PuTTY/MobaXterm (Windows) |
| GitHub access to repo | https://github.com/Manish06N/reasoning-compression-lab |
| GitHub PAT (optional) | https://github.com/settings/tokens — scope `repo`, for private clone |
| Hugging Face account | https://huggingface.co |
| HF token | https://huggingface.co/settings/tokens — for model download |
| GPQA access (later) | Request at https://huggingface.co/datasets/Idavidrein/gpqa |

---

## 7. Step 1 — Get a PARAM Rudra account and log in

### Request an account (NSM portal — PARAM Rudra User Manual §First Things First)

1. Visit https://nsmindia.in  
2. Open **"How to access NSM HPC System"** - User Creation portal  
3. Complete the registration form  
4. Select **IITP, Indian Institute of Technology Patna** as the institute  
5. Upload required documents and submit  
6. After NSM committee approval, you receive **username + temporary password** by email  

### First login

1. You receive credentials by email from PARAM Rudra / NSM support.  
2. On first login you **must change your password**.  
3. Connect to the cluster:

**IIT Patna campus (port 22):**

```bash
ssh YOUR_USERNAME@paramrudra.iitp.ac.in
```

**External / off-campus (port 4422):**

```bash
ssh YOUR_USERNAME@paramrudra.iitp.ac.in -p 4422
```

4. Complete the **captcha**, then enter your password.  
5. You land on a **login node** — shared for editing scripts and submitting jobs. **Never run vLLM inference on the login node.**

### Use tmux so work survives disconnects

```bash
tmux new -s qreason
# If disconnected later:
tmux attach -t qreason
```

### Set your workspace variable (every session)

```bash
export QR=/scratch/$USER/reasoning-compression-lab
mkdir -p "$QR"
```

> **Storage policy:** `/scratch` is for job data. Files not accessed for **3 months** may be deleted. Back up important results to your laptop.

---

## 8. Step 2 — Clone the repository on scratch

Work in **`/scratch`**, not `$HOME`, for models and outputs (fast Lustre, large quota).

### Option A — HTTPS clone (simplest)

```bash
cd /scratch/$USER
git clone https://github.com/Manish06N/reasoning-compression-lab.git
cd reasoning-compression-lab
export QR=/scratch/$USER/reasoning-compression-lab
```

If the repo is **private**, Git will prompt for credentials:

- **Username:** your GitHub username  
- **Password:** a **Personal Access Token** (not your GitHub password)

Create a token at https://github.com/settings/tokens - **Generate new token (classic)** - check **`repo`**.

To store credentials for future `git pull` on the cluster (one-time):

```bash
git config --global credential.helper store
git pull   # enter username + PAT once; saved in ~/.git-credentials
```

> **Security:** Do not commit tokens to the repo. Do not put PATs in SLURM scripts. See `docs/GIT_CREDENTIALS.md`.

### Option B — SSH clone (if you added an SSH key to GitHub)

```bash
cd /scratch/$USER
git clone git@github.com:Manish06N/reasoning-compression-lab.git
```

### Keep the repo updated

After the project owner pushes changes:

```bash
cd $QR
git pull origin main
```

### Verify clone

```bash
cd $QR
ls
# Expect: configs/  scripts/  slurm/  src/  docs/  requirements-hpc.txt
```

---

## 9. Step 3 — Create the Python environment

PARAM Rudra provides conda at `/home/apps/MSCC/miniconda3` (not via `module load`).

### One-time setup

```bash
cd $QR
bash scripts/hpc/00_setup_env.sh
```

This creates conda env **`qreason`** (Python 3.11) and installs `requirements-hpc.txt`:

| Package | Version note |
|---------|--------------|
| vLLM | **0.8.5** (pinned — do not upgrade on Rudra) |
| PyTorch | Installed as vLLM dependency (CUDA 12.x) |
| transformers, datasets, huggingface_hub | Latest compatible |
| math-verify | Answer grading for MATH |

### Activate environment (every session)

```bash
CONDA_ROOT=/home/apps/MSCC/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate qreason
cd $QR
```

Or rely on `scripts/hpc/param_rudra_env.sh`, which sets paths and activates conda automatically.

### If `pip install vllm` fails

1. Run inside a **GPU interactive session** (not login node).  
2. Check CUDA: `nvidia-smi`  
3. Email `rudrasupport@iitp.ac.in` with the full pip error — do not guess random torch versions.  
4. The project pins **vLLM 0.8.5** intentionally; a tokenizer shim in `src/runners/vllm_runner.py` depends on it.

---

## 10. Step 4 — Hugging Face token and model downloads

### Store HF token on scratch

```bash
mkdir -p $QR/hf_cache
chmod 700 $QR/hf_cache

# Paste your token (starts with hf_...)
read -s HF_TOKEN
echo "$HF_TOKEN" > $QR/hf_cache/token
chmod 600 $QR/hf_cache/token
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
```

Or copy the template and edit:

```bash
cp .env.example .env
nano .env   # set HF_TOKEN= and QR=/scratch/$USER/reasoning-compression-lab
```

`param_rudra_env.sh` sets `HF_HOME=$QR/hf_cache` so caches stay on scratch.

### Download models in stages (do not download everything day one)

**Stage 1 — Gate 2 (required first):**

```bash
conda activate qreason
cd $QR
bash scripts/hpc/02_download_model.sh
```

Downloads **DeepSeek-R1-Distill-Qwen-7B** (~15 GB) to:

```text
$QR/models/DeepSeek-R1-Distill-Qwen-7B/
```

**Alternative — download via SLURM** (if login-node downloads are discouraged):

```bash
sbatch slurm/download_model.slurm
tail -f logs/download_*.out
```

Authenticate first: `hf auth login` or `export HF_TOKEN=hf_...`

**Stage 2 — After Level A BF16 works, add GPTQ-4:**

```bash
hf download RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16 \
  --local-dir $QR/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4
```

**Stage 3 — Before full grid (all quants + Llama-8B + 1.5B):**

See canonical HF IDs in `docs/MODEL_ROSTER.md`. Example batch:

```bash
# Llama-8B BF16 anchor
hf download deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --local-dir $QR/models/DeepSeek-R1-Distill-Llama-8B

# Qwen-7B FP8
hf download RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic \
  --local-dir $QR/models/DeepSeek-R1-Distill-Qwen-7B-FP8

# Qwen-1.5B BF16 (small, fast)
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --local-dir $QR/models/DeepSeek-R1-Distill-Qwen-1.5B
```

Repeat for AWQ/GPTQ variants listed in `docs/MODEL_ROSTER.md`.

### Validate MATH-500 dataset

```bash
bash scripts/hpc/02b_validate_dataset.sh
```

---

## 11. Step 5 — Execution gates (do not skip)

### Gate 1 — GPU and software check

Request an **interactive GPU session** (PARAM Rudra manual: use `srun` for debugging):

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --time=01:00:00 --pty bash
```

Inside the GPU shell:

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh
conda activate qreason
bash scripts/hpc/01_gpu_check.sh
```

**Success looks like:**

```text
cuda available: True
device: NVIDIA A100 80GB PCIe
vllm: 0.8.5
GPU check passed.
```

(Exact device string may vary; Rudra GPU nodes have **2× A100 80 GB HBM2e** per node.)

Type `exit` to leave the interactive session.

---

### Gate 2 — Validate dataset (after model download)

```bash
bash scripts/hpc/02b_validate_dataset.sh
```

Confirms `HuggingFaceH4/MATH-500` loads (500 rows) before long GPU jobs.

---

### Gate 3 — Smoke test (model loads and generates)

Use the **exclusive quick smoke** SLURM script to avoid shared-GPU OOM:

```bash
cd $QR
mkdir -p logs runs/raw
sbatch slurm/smoke_test_quick_exclusive.slurm
squeue -u $USER
tail -f logs/smoke_quick_*.out
```

This runs **1 question**, **64 max tokens**, output:

```text
runs/raw/smoke_test_quick.jsonl
```

**Alternative (3 questions, interactive GPU):**

```bash
bash scripts/hpc/03_smoke_test.sh
# writes runs/raw/smoke_test.jsonl
```

If smoke fails with exit code **75**, the GPU had insufficient free memory — wait and retry, or use `--exclusive`.

---

### Gate 4 — Level A BF16 debug (10 questions)

Inside an interactive GPU session:

```bash
export QR=/scratch/$USER/reasoning-compression-lab
export QREASON_MODEL_QWEN7B=$QR/models/DeepSeek-R1-Distill-Qwen-7B
cd $QR
conda activate qreason

bash scripts/hpc/04_run_level_a_bf16.sh 10
bash scripts/hpc/05_score_level_a.sh
cat results/level_a_qwen7b_bf16_math500_seed0_summary.json
```

Check that `pass_at_1`, `latency_sec_mean`, and `peak_vram_gb_max` are present.

---

### Gate 4b — Full Level A BF16 (500 questions)

Submit as batch job:

```bash
cd $QR
sbatch slurm/run_level_a_bf16.slurm
```

Monitor:

```bash
squeue -u $USER
tail -f logs/level_a_bf16_*.out
```

Outputs:

| File | Purpose |
|------|---------|
| `runs/raw/level_a_qwen7b_bf16_math500_seed0.jsonl` | Raw generations |
| `runs/scored/level_a_qwen7b_bf16_math500_seed0.jsonl` | + correctness |
| `results/level_a_qwen7b_bf16_math500_seed0_summary.json` | Paper summary |

**Only after this passes:** download GPTQ-4 and run Gate 5.

---

### Gate 5 — Level A GPTQ-4 (after BF16 works)

**Stage 2 download** (canonical HF ID — see `docs/MODEL_ROSTER.md`):

```bash
hf download RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16 \
  --local-dir $QR/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4
export QREASON_MODEL_QWEN7B_GPTQ4=$QR/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4
bash scripts/hpc/06_verify_gptq4_model.sh
```

Inside a GPU session (or SLURM job):

```bash
python scripts/run_inference.py --cell-config configs/cells/level_a_gptq4_seed0.json
python scripts/score_run.py --input runs/raw/level_a_qwen7b_gptq4_math500_seed0.jsonl
cat results/level_a_qwen7b_gptq4_math500_seed0_summary.json
```

See `docs/GPTQ4_PREP.md` if download fails (alternate quantize path).

---

## 12. Step 6 — Submit SLURM jobs on PARAM Rudra

### PARAM Rudra SLURM essentials (from User Manual)

| Rule | Value for this project |
|------|------------------------|
| GPU partition | `--partition=gpu` |
| Request GPUs | `--gres=gpu:1` or `--gres=gpu:2` |
| **Do not** use | `--gres=gpu:a100:1` (wrong on Rudra) |
| **Do not** use | `#SBATCH --mem=...` (omit memory flag) |
| Default walltime | **2 hours** if you forget — always set `--time` |
| Max walltime | **4 days** — our scripts use **47:00:00** |
| Login node | Submit jobs only — no vLLM here |
| Scratch | Put code, models, outputs under `/scratch/$USER/` |

### Example minimal SLURM script

```bash
#!/bin/bash
#SBATCH --job-name=qreason-test
#SBATCH --output=logs/myjob_%j.out
#SBATCH --error=logs/myjob_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail
export QR=/scratch/$USER/reasoning-compression-lab
cd "$QR"
source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh
conda activate qreason

bash scripts/hpc/03_smoke_test.sh
```

Submit:

```bash
sbatch myjob.slurm
```

### Monitor jobs (from User Manual)

```bash
squeue -u $USER              # your jobs in queue
squeue -j JOBID              # one job
scancel JOBID                # cancel a job
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -f logs/myjob_*.out     # live log
```

Job states: `PD` = pending (waiting for GPU), `R` = running, `CG` = completing.

---

## 13. Step 7 — Full publication grid (all models, HPC-only)

After Gates 1–5 pass and **all model weights for blocks b01–b06** are on scratch:

### Preflight (recommended before b01–b06)

```bash
cd $QR
conda activate qreason
python scripts/hpc/07_preflight_publication.py
```

This is a **CPU-only** check: configs, prompts, dataset access, and that **every model folder** referenced in b01–b06 exists with weights. It does **not** load vLLM on GPU. If models are missing, it exits with an error — download Stage 3 weights first.

> **Note:** `07_preflight_publication.py` validates HPC blocks b01–b06 only. Qwen-1.5B cells are run separately (below).

### Submit blocks b01–b06

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
git pull origin main
bash scripts/hpc/submit_hpc_blocks.sh
```

This submits:

| Job | GPUs | Block |
|-----|------|-------|
| qreason-b01 | 2 | BF16 anchors |
| qreason-b02 | 2 | FP8 pair |
| qreason-b03 | 2 | AWQ-4 pair |
| qreason-b04 | 2 | GPTQ-4 pair |
| qreason-b05 | 1 | GPTQ-3 |
| qreason-b06 | 1 | GSM8K |

Submit **one block at a time** if the queue is busy:

```bash
bash scripts/hpc/submit_hpc_blocks.sh b01
```

### Block b07 — GPQA (after Hugging Face gate approved)

```bash
sbatch slurm/hpc_2a100_b07_gpqa.slurm
```

See `docs/GPQA_ACCESS.md`.

### Qwen-1.5B cells (single GPU each)

Gate scripts write to `runs/` and `results/`. Publication blocks write to `outputs-hpc-2a100-main-YYYY-MM-DD/`. For 1.5B cells, use the **same publication archive** as b01–b07:

```bash
export QREASON_OUTPUT_ROOT=$QR/outputs-hpc-2a100-main-$(date +%Y-%m-%d)
mkdir -p "$QREASON_OUTPUT_ROOT"/{raw,scored,results,logs}

CELL=level_c_qwen15b_bf16_math500_seed0
sbatch --job-name=qreason-15b-bf16 \
  --partition=gpu --gres=gpu:1 --cpus-per-task=8 --time=24:00:00 \
  --output=logs/15b_bf16_%j.out --error=logs/15b_bf16_%j.err \
  --wrap="cd $QR && source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh && conda activate qreason && export QREASON_OUTPUT_ROOT=$QREASON_OUTPUT_ROOT && python scripts/run_inference.py --cell-config configs/cells/level_c_qwen15b_bf16_math500_seed0.json --output $QREASON_OUTPUT_ROOT/raw/${CELL}.jsonl && python scripts/score_run.py --input $QREASON_OUTPUT_ROOT/raw/${CELL}.jsonl --output $QREASON_OUTPUT_ROOT/scored/${CELL}.jsonl --summary $QREASON_OUTPUT_ROOT/results/${CELL}_summary.json"
```

Repeat for `level_c_qwen15b_fp8`, `_awq4`, `_gptq4` configs. These four can run **in parallel** on four GPUs.

### Where publication outputs go

```text
$QR/outputs-hpc-2a100-main-YYYY-MM-DD/
├── raw/           # JSONL per cell
├── scored/        # JSONL + correctness
├── results/       # *_summary.json for tables
├── logs/          # per-cell inference logs
└── checkpoints/   # resume state
```

Jobs **checkpoint every 10 rows** — safe to re-submit; completed cells are skipped.

---

## 14. Understanding outputs

### One experiment cell

Defined by `configs/cells/*.json`. Example:

```json
{
  "cell_id": "level_a_qwen7b_bf16_math500_seed0",
  "model_config": "configs/models/deepseek_r1_qwen_7b.json",
  "task_config": "configs/tasks/math500.json",
  "quant_config": "bf16",
  "seed": 0
}
```

### Two output locations (important)

| Phase | Output root | Used by |
|-------|-------------|---------|
| **Gates / Level A** | `runs/raw/`, `runs/scored/`, `results/` | `04_run_level_a_bf16.sh`, smoke tests |
| **Publication grid** | `outputs-hpc-2a100-main-YYYY-MM-DD/` | `submit_hpc_blocks.sh`, 1.5B HPC cells |

### Pipeline stages

```text
raw/*.jsonl      -> model output, tokens, latency, peak VRAM per question
scored/*.jsonl   -> + extracted answer + correct/incorrect
results/*_summary.json -> pass@1, mean latency, mean tokens, max VRAM
```

### Summary JSON fields (typical)

```json
{
  "cell_id": "level_a_qwen7b_bf16_math500_seed0",
  "n": 500,
  "pass_at_1": 0.72,
  "latency_sec_mean": 45.2,
  "peak_vram_gb_max": 38.1,
  "completion_tokens_mean": 8192
}
```

Record each gate in `docs/EXPERIMENT_LOG.md`.

---

## 15. Monitoring jobs and troubleshooting

### Common issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Job `PD` forever | Queue busy (8 GPU nodes shared) | Wait; check `squeue`; avoid duplicate smoke jobs |
| `CUDA OOM` | Shared GPU nearly full | Use `smoke_test_quick_exclusive.slurm` or `--exclusive` |
| Smoke exit **75** | <30 GB free VRAM | Retry when cluster quieter |
| vLLM import error | Wrong env / wrong vLLM version | `conda activate qreason`; must be **0.8.5** |
| Model download fails | HF token missing | Set `HF_TOKEN`; `hf auth login` |
| Empty / garbage answers | Wrong prompt template | Check `prompts/math500.txt`; inspect one JSONL row |
| Job killed at 2 h | Forgot `--time` | Resubmit with `--time=24:00:00` or higher |
| Scoring wrong | Missing `\boxed{}` in output | Inspect raw JSONL; ensure `math-verify` installed |

### Investigating failed jobs (PARAM Rudra manual)

Always check:

```bash
cat logs/YOURJOB_*.err
cat logs/YOURJOB_*.out
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
```

Non-zero exit code means job state `FAILED`.

### Do not run on login node

The manual states: compile/edit on login nodes; **execution** goes through SLURM. vLLM on a login node will be killed or harm other users.

---

## 16. Copying results off the cluster

From your **laptop** (replace `USER` and use port 4422 off-campus):

```bash
rsync -avz -e "ssh -p 4422" \
  USER@paramrudra.iitp.ac.in:/scratch/USER/reasoning-compression-lab/outputs-hpc-2a100-main-*/ \
  ./paper1-results/

rsync -avz -e "ssh -p 4422" \
  USER@paramrudra.iitp.ac.in:/scratch/USER/reasoning-compression-lab/results/ \
  ./paper1-results/legacy-results/
```

Or SCP (from manual):

```bash
scp -r -P 4422 USER@paramrudra.iitp.ac.in:/scratch/USER/reasoning-compression-lab/outputs-hpc-2a100-main-* ./
```

---

## 17. PARAM Rudra rules that matter for this project

Summarized from the **PARAM Rudra User Manual (IIT Patna, April 2025)**:

| Topic | Rule |
|-------|------|
| Hostname | `paramrudra.iitp.ac.in` |
| SSH port | **22** on campus, **4422** external |
| Filesystems | `/home` (small) + `/scratch` (job data — **use this**) |
| GPU partition | `--partition=gpu` (8 nodes, 2× A100 80GB each) |
| GPU flag | `--gres=gpu:1` or `--gres=gpu:2` |
| Walltime | Default 2 h — **always specify**; max 4 days |
| Concurrent jobs | Up to **10 simultaneous jobs** per user |
| Login node | Shared — no heavy compute |
| Support email | `rudrasupport@iitp.ac.in` |
| Support portal | https://paramrudra.iitp.ac.in/support |
| Acknowledgement | Required in publications (text below) |

### NSM acknowledgement (required in papers — from User Manual)

Include in thesis, preprints, and journal papers:

> We acknowledge National Supercomputing Mission (NSM) for providing computing resources of **PARAM Rudra** at **Indian Institute of Technology Patna**, implemented by C-DAC and supported by the Ministry of Electronics and Information Technology (MeitY) and Department of Science and Technology (DST), Government of India.

Send copies of publications that cite NSM to HPC Technologies, C-DAC Pune (see User Manual §Acknowledging NSM for postal address).

### This project's PARAM-specific choices

| Setting | Why |
|---------|-----|
| vLLM 0.8.5 | Validated on Rudra A100; tokenizer shim in repo |
| `enforce_eager: true` | Stability on shared GPUs |
| No `#SBATCH --mem` | Rudra policy — omit memory directive |
| `$QR/hf_cache` on scratch | Avoid home quota; keep token mode 600 |
| Exclusive smoke | Avoid OOM on partially used GPUs |

---

## 18. Quick command cheat sheet

```bash
# ── Every SSH session ──
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh
conda activate qreason

# ── Update code ──
git pull origin main

# ── Gate 1 (interactive GPU) ──
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --time=01:00:00 --pty bash
bash scripts/hpc/01_gpu_check.sh

# ── Gate 3 smoke (batch) ──
sbatch slurm/smoke_test_quick_exclusive.slurm

# ── Gate 4 debug ──
bash scripts/hpc/04_run_level_a_bf16.sh 10
bash scripts/hpc/05_score_level_a.sh

# ── Gate 4 full ──
sbatch slurm/run_level_a_bf16.slurm

# ── Full publication grid ──
bash scripts/hpc/submit_hpc_blocks.sh
sbatch slurm/hpc_2a100_b07_gpqa.slurm   # after GPQA gate

# ── Monitor ──
squeue -u $USER
tail -f logs/*.out
bash scripts/hpc/check_hpc_gate_status.sh
```

---

## 19. Further reading inside the repo

| Document | Contents |
|----------|----------|
| [docs/README.md](README.md) | **Doc index** — what to read / what was archived |
| [PAPER1_DESIGN.md](PAPER1_DESIGN.md) | Research scope, metrics, claim |
| [MODEL_ROSTER.md](MODEL_ROSTER.md) | Canonical Hugging Face model IDs |
| [HPC_2A100_PLAN.md](HPC_2A100_PLAN.md) | Block grid b01–b07 details |
| [HPC_PARAM_RUDRA.md](HPC_PARAM_RUDRA.md) | Rudra-specific env and sync |
| [GPQA_ACCESS.md](GPQA_ACCESS.md) | Gated dataset steps |
| [progress.md](../progress.md) | Master execution log |
| [CHANGELOG.md](../CHANGELOG.md) | Ops log of HPC runs and fixes |

---

## Workflow diagram (start to paper numbers)

```text
Account -> SSH -> clone repo -> conda env -> HF token
    -> download Qwen-7B -> Gate 1 GPU check -> Gate 3 smoke
    -> Gate 4 BF16 (10 Q -> 500 Q) -> Gate 5 GPTQ-4
    -> download all models -> submit b01-b07 + 1.5B cells
    -> rsync outputs-hpc-* to laptop -> tables / paper
```

**You are ready.** Start at **Step 1 (Section 7)** on PARAM Rudra.
