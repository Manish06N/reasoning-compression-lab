# Runbook — MacBook ↔ HPC

## MacBook (control room)

Use for: reading papers, writing, repo structure, Python scripts, configs, VS Code Remote SSH, logs, CSV analysis, plotting, README/docs.

**Project root (MacBook):**

```
/Users/manish/Projects/2026/paper 1/reasoning-compression-lab
```

## HPC (experiment factory)

Use for: model download, vLLM, BF16/FP8/GPTQ/AWQ inference, quantization, benchmark generation, latency/VRAM profiling, final paper numbers.

## Windows RTX 5080 (WSL2 pilot lab)

Use for: quant grid pilots, pipeline smoke, 1.5B verifier, 7B/8B **quantized** runs that fit 16 GB VRAM.

**Project root (WSL):**

```bash
cd "/mnt/g/ALL MY Projects/2026/03-paper1-experiments"
source scripts/local/env.sh
```

**Recommended first pass:**

```bash
bash scripts/local/start_5080_pilot.sh      # background
bash scripts/local/resume_5080_pilot.sh     # foreground / after reboot
bash scripts/local/backup_5080_archive.sh --snapshot
```

Full repro (days) or HPC for BF16 anchors — 5080 retired; see [archive/RTX5080_EXECUTION_PLAN.md](archive/RTX5080_EXECUTION_PLAN.md).

---

**Scratch workspace (HPC):**

```bash
export QR=/scratch/$USER/reasoning-compression-lab
mkdir -p $QR && cd $QR
```

Clone this repo on HPC when pushed to GitHub, or rsync from MacBook:

```bash
rsync -avz --exclude '.git' --exclude 'runs/' \
  "/Users/manish/Projects/2026/paper 1/reasoning-compression-lab/" \
  your_username@your_hpc:$QR/
```

## HPC environment setup

```bash
tmux new -s qreason
module avail cuda
module load cuda/12.1   # adjust to your cluster
module load anaconda

conda create -n qreason python=3.11 -y
conda activate qreason

pip install --upgrade pip
pip install torch transformers accelerate datasets huggingface_hub vllm
pip install pandas numpy scipy scikit-learn statsmodels pyarrow tqdm pynvml
pip install math-verify
pip install auto-gptq autoawq   # may need version tuning
```

## GPU smoke test

```bash
srun --gres=gpu:a100:1 --cpus-per-task=8 --mem=80G --time=01:00:00 --pty bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## First model download (HPC only)

```bash
mkdir -p $QR/models
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir $QR/models/DeepSeek-R1-Distill-Qwen-7B
```

## First real experiment (Level A gate)

| Field | Value |
|-------|-------|
| Model | DeepSeek-R1-Distill-Qwen-7B |
| Task | MATH-500 |
| Configs | BF16, then GPTQ-4 |
| Seed | 0 |
| Hardware | A100 |
| Outputs | accuracy, tokens, latency, peak VRAM |

Save raw outputs to `runs/raw/`. Score into `runs/scored/`. Aggregate to `results/`.

## Result pipeline

```
runs/raw/        → model generations (JSONL)
runs/extracted/  → parsed answers + confidence
runs/scored/     → per-item correctness + metrics
results/         → aggregated CSVs for paper tables/figures
```

## Sync results back to MacBook

```bash
rsync -avz your_username@your_hpc:$QR/runs/ \
  "/Users/manish/Projects/2026/paper 1/reasoning-compression-lab/runs/"

rsync -avz your_username@your_hpc:$QR/results/ \
  "/Users/manish/Projects/2026/paper 1/reasoning-compression-lab/results/"
```

Plot on MacBook from `results/*.csv` → `paper_figures/`.

## Gate rules

- No full grid before reproduction works.
- No 14B before Qwen-7B pilot works.
- No energy claims before latency/VRAM/cost-per-correct works.
- No Paper 2 before Paper 1 pilot exists.
