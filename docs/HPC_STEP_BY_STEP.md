# HPC Step-by-Step Guide — Paper 1 First Experiments

This is your **complete HPC playbook**. Follow it in order. Do not skip gates.

**Goal:** Prove the pipeline works, then run Level A:

```text
DeepSeek-R1-Distill-Qwen-7B × BF16 × MATH-500 × seed 0
```

Then (only after BF16 works):

```text
DeepSeek-R1-Distill-Qwen-7B × GPTQ-4 × MATH-500 × seed 0
```

---

## Before you SSH — checklist

| Item | You need |
|------|----------|
| HPC username | e.g. `mn123` |
| HPC hostname | e.g. `login.hpc.university.edu` |
| SLURM access | GPU partition name, account/project if required |
| Hugging Face token | https://huggingface.co/settings/tokens |
| Repo on HPC | GitHub clone **or** rsync from MacBook (Section 1) |

---

## Section 0 — MacBook: get repo onto HPC

You have **two options**. Pick one.

### Option A — GitHub (recommended long-term)

On MacBook:

```bash
cd "/Users/manish/Projects/2026/paper 1/reasoning-compression-lab"

# Create empty repo on github.com named reasoning-compression-lab, then:
git remote add origin https://github.com/YOUR_USERNAME/reasoning-compression-lab.git
git push -u origin main
```

On HPC:

```bash
export QR=/scratch/$USER/reasoning-compression-lab
git clone https://github.com/YOUR_USERNAME/reasoning-compression-lab.git $QR
```

### Option B — rsync (fastest if no GitHub yet)

On MacBook, edit `scripts/macbook/rsync_to_hpc.sh`:

```bash
HPC_USER=your_username
HPC_HOST=your_hpc_address
```

Then run:

```bash
bash "/Users/manish/Projects/2026/paper 1/reasoning-compression-lab/scripts/macbook/rsync_to_hpc.sh"
```

This copies code to `/scratch/$USER/reasoning-compression-lab` on HPC.

---

## Section 0 — MacBook: push to GitHub first

Before deep HPC work, protect your repo. See [GITHUB_PUSH.md](GITHUB_PUSH.md).

```bash
cd "/Users/manish/Projects/2026/paper 1/reasoning-compression-lab"
git remote add origin https://github.com/YOUR_USERNAME/reasoning-compression-lab.git
git push -u origin main
```

---

## Section 1 — SSH and persistent session

From MacBook terminal:

```bash
ssh your_username@your_hpc_address
```

Start `tmux` so jobs survive disconnects:

```bash
tmux new -s qreason
```

If disconnected later, reattach with:

```bash
ssh your_username@your_hpc_address
tmux attach -t qreason
```

Set workspace variable (use this in every session):

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
pwd
ls
```

You should see: `configs/`, `scripts/`, `src/`, `slurm/`, `docs/`, etc.

---

## Section 2 — Load cluster modules

Every HPC is different. Discover what yours provides:

```bash
module avail cuda
module avail anaconda
```

Load the versions your cluster docs recommend. Example:

```bash
module load cuda/12.1
module load anaconda
```

If `module` command doesn't exist, skip this section and use system Python/conda directly.

---

## Section 3 — Create Python environment

```bash
cd $QR
bash scripts/hpc/00_setup_env.sh
```

This creates conda env `qreason` and installs `requirements-hpc.txt`.

Activate manually anytime:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qreason
```

### If `pip install vllm` fails

Common fixes (try in order):

1. **Check CUDA module matches GPU driver**
   ```bash
   nvidia-smi
   module list
   ```

2. **Install torch first with CUDA index** (example for CUDA 12.1):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install vllm
   ```

3. **Ask your HPC support** for recommended vLLM install command — many clusters have a tested recipe.

4. **Paste the full error** in your experiment log — do not switch topics; fix install first.

---

## Section 4 — Hugging Face login

Required for model download:

```bash
conda activate qreason
huggingface-cli login
```

Or set token in environment:

```bash
export HF_TOKEN=hf_your_token_here
huggingface-cli login --token $HF_TOKEN
```

Optional: copy env template on HPC:

```bash
cp .env.example .env
nano .env   # fill HF_TOKEN and model paths
```

---

## Section 5 — GPU smoke test (Gate 1)

Request one interactive A100 (adjust flags for your cluster):

```bash
srun --gres=gpu:a100:1 --cpus-per-task=8 --mem=80G --time=01:00:00 --pty bash
```

Inside the GPU session:

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qreason

bash scripts/hpc/01_gpu_check.sh
```

**Success looks like:**

```text
cuda available: True
device: NVIDIA A100-SXM4-80GB
vllm: 0.x.x
GPU check passed.
```

If this fails, stop here and fix GPU/CUDA/vLLM before downloading models.

Exit interactive session when done: `exit`

---

## Section 6 — Download Qwen-7B only (Gate 2)

Do **not** download all models. One model first.

### Option A — login node (if allowed)

```bash
cd $QR
conda activate qreason
bash scripts/hpc/02_download_model.sh
```

### Option B — SLURM job (safer on strict clusters)

Edit `slurm/download_model.slurm` if needed, then:

```bash
cd $QR
sbatch slurm/download_model.slurm
squeue -u $USER
tail -f logs/download_*.out
```

Download target:

```text
/scratch/$USER/reasoning-compression-lab/models/DeepSeek-R1-Distill-Qwen-7B
```

Set path:

```bash
export QREASON_MODEL_QWEN7B=$QR/models/DeepSeek-R1-Distill-Qwen-7B
```

Verify:

```bash
ls $QREASON_MODEL_QWEN7B | head
```

You should see `config.json`, tokenizer files, and `.safetensors` shards.

---

## Section 7 — Tiny generation smoke test (Gate 3)

This runs **3 simple math questions**, not full MATH-500.

Request GPU again:

```bash
srun --gres=gpu:a100:1 --cpus-per-task=8 --mem=80G --time=01:00:00 --pty bash
```

Inside GPU session:

```bash
export QR=/scratch/$USER/reasoning-compression-lab
export QREASON_MODEL_QWEN7B=$QR/models/DeepSeek-R1-Distill-Qwen-7B
cd $QR
conda activate qreason

bash scripts/hpc/03_smoke_test.sh
```

Or submit batch job:

```bash
sbatch slurm/smoke_test.slurm
```

**Success looks like:**

- Script prints 3 completions with latency and VRAM
- File created: `runs/raw/smoke_test.jsonl`

Inspect:

```bash
head -n 1 runs/raw/smoke_test.jsonl | python -m json.tool | head -30
```

If model loads but outputs are garbage, check chat template / prompt — but loading + generation is the gate.

---

## Section 8 — Level A BF16 on MATH-500 (Gate 4)

### 8a. Debug run first (10 questions)

Inside GPU session:

```bash
export QR=/scratch/$USER/reasoning-compression-lab
export QREASON_MODEL_QWEN7B=$QR/models/DeepSeek-R1-Distill-Qwen-7B
cd $QR
conda activate qreason

bash scripts/hpc/04_run_level_a_bf16.sh 10
bash scripts/hpc/05_score_level_a.sh
```

Check summary:

```bash
cat results/level_a_qwen7b_bf16_math500_seed0_summary.json
```

Expected fields:

```json
{
  "n": 10,
  "pass_at_1": 0.x,
  "latency_sec_mean": ...,
  "peak_vram_gb_max": ...,
  "completion_tokens_mean": ...
}
```

### 8b. Full MATH-500 run (500 questions)

This can take many hours (long reasoning traces). Use SLURM:

```bash
cd $QR
sbatch slurm/run_level_a_bf16.slurm
```

Monitor:

```bash
squeue -u $USER
tail -f logs/level_a_bf16_*.out
```

The SLURM script auto-scores at the end.

Outputs:

| File | Purpose |
|------|---------|
| `runs/raw/level_a_qwen7b_bf16_math500_seed0.jsonl` | Raw generations + latency/VRAM |
| `runs/scored/level_a_qwen7b_bf16_math500_seed0.jsonl` | + correctness per item |
| `results/level_a_qwen7b_bf16_math500_seed0_summary.json` | pass@1 + systems stats |

### Experiment settings (for your paper methods section)

From `configs/cells/level_a_bf16_seed0.json`:

| Setting | Value |
|---------|-------|
| Model | DeepSeek-R1-Distill-Qwen-7B |
| Precision | BF16 |
| Task | MATH-500 test split |
| Seed | 0 |
| temperature | 0.6 |
| top_p | 0.95 |
| max_tokens | 32768 |
| Prompt | `\boxed{}` format (sober-reasoning style) |

---

## Section 9 — Sync results back to MacBook

On MacBook:

```bash
HPC_USER=your_username
HPC_HOST=your_hpc_address
QR="/scratch/${HPC_USER}/reasoning-compression-lab"
LOCAL="/Users/manish/Projects/2026/paper 1/reasoning-compression-lab"

rsync -avz "${HPC_USER}@${HPC_HOST}:${QR}/runs/" "${LOCAL}/runs/"
rsync -avz "${HPC_USER}@${HPC_HOST}:${QR}/results/" "${LOCAL}/results/"
```

Update `docs/EXPERIMENT_LOG.md` on MacBook with pass@1, latency, VRAM.

---

## Section 10 — GPTQ-4 (Gate 6 — only after BF16 + model prep)

**Do not run inference until weights exist.** Full guide: [GPTQ4_PREP.md](GPTQ4_PREP.md).

True order:

```text
BF16 smoke → BF16 MATH-500 seed 0 → prepare GPTQ-4 weights → verify → GPTQ-4 inference
```

### 10a. Obtain GPTQ-4 weights (pick one)

| Option | Action |
|--------|--------|
| A — Download | Pre-quantized from HF model zoo (see GPTQ4_PREP.md) |
| B — Quantize | Reference repo `real_quantization/gptq.sh` or `auto-gptq` |
| C — AWQ first | If GPTQ install blocked, try AWQ checkpoint |

### 10b. Verify path before inference

```bash
export QREASON_MODEL_QWEN7B_GPTQ4=$QR/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4
bash scripts/hpc/06_verify_gptq4_model.sh
```

Must print `OK: GPTQ-4 model path verified.`

### 10c. Run GPTQ-4 cell

```bash
python scripts/run_inference.py \
  --cell-config configs/cells/level_a_gptq4_seed0.json

python scripts/score_run.py \
  --input runs/raw/level_a_qwen7b_gptq4_math500_seed0.jsonl
```

Compare BF16 vs GPTQ summaries in `results/`.

---

## Section 11 — Troubleshooting

### `CUDA out of memory`

- Confirm A100 80GB allocated (`--mem=80G`, `--gres=gpu:a100:1`)
- Reduce `max_model_len` in `configs/models/deepseek_r1_qwen_7b.json` temporarily (e.g. 16384) for debugging only

### Model download slow / fails

- Use `huggingface-cli login`
- Retry with `huggingface-cli download ... --resume-download`

### vLLM import error

- Reinstall torch + vllm with cluster-specific instructions
- Check `python -c "import torch; print(torch.version.cuda)"`

### Scoring looks wrong

- Inspect a few rows in `runs/scored/*.jsonl`
- Check `\boxed{}` appears in completions
- `math-verify` improves matching when installed

### Job killed / timeout

- Increase `#SBATCH --time` in `slurm/run_level_a_bf16.slurm`
- Full MATH-500 with 32k max tokens can take 12–24+ hours

---

## Section 12 — Command cheat sheet

```bash
# Every new SSH session
export QR=/scratch/$USER/reasoning-compression-lab
export QREASON_MODEL_QWEN7B=$QR/models/DeepSeek-R1-Distill-Qwen-7B
cd $QR
conda activate qreason

# GPU check
bash scripts/hpc/01_gpu_check.sh

# Smoke test
bash scripts/hpc/03_smoke_test.sh

# Debug 10-question run
bash scripts/hpc/04_run_level_a_bf16.sh 10
bash scripts/hpc/05_score_level_a.sh

# Full run via SLURM
sbatch slurm/run_level_a_bf16.slurm

# Watch job
squeue -u $USER
tail -f logs/level_a_bf16_*.out
```

---

## Section 13 — What to record in EXperiment_LOG.md

After each gate, append:

```text
Date:
Gate: 1|2|3|4|5
Command:
pass@1:
latency_sec_mean:
peak_vram_gb_max:
completion_tokens_mean:
Notes:
```

Gate meanings:

| Gate | What passed |
|------|-------------|
| 1 | GPU + torch + vLLM import |
| 2 | Qwen-7B downloaded |
| 3 | 3-question smoke test |
| 4 | MATH-500 BF16 scored |
| 5 | MATH-500 GPTQ-4 scored |

---

## Section 14 — After Level A

Only after Gates 1–5:

1. Level B pilot: 5 seeds, 5 quant configs, add GPQA
2. Calibration scripts (Brier, ECE from Calibrating-LLMs reference)
3. Cost-per-correct (Cost-of-Pass formula + your latency/token logs)

Do not start Level B until Level A BF16 + GPTQ-4 rows exist in `results/`.

---

**You are now in execution mode. Start at Section 1 on HPC.**
