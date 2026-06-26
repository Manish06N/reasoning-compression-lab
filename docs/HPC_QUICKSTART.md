# HPC Quick Start

Run these commands on your HPC after pushing this repo to GitHub.

```bash
ssh your_username@your_hpc_address
tmux new -s qreason

export QR=/scratch/$USER/reasoning-compression-lab
mkdir -p $QR && cd $QR

git clone <your-github-repo-url> .

conda create -n qreason python=3.11 -y
conda activate qreason
pip install --upgrade pip
pip install -r requirements-hpc.txt

# GPU smoke test (adjust SLURM flags for your cluster)
srun --gres=gpu:a100:1 --cpus-per-task=8 --mem=80G --time=01:00:00 --pty bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# First model (one only — do not download all models on day one)
mkdir -p $QR/models
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir $QR/models/DeepSeek-R1-Distill-Qwen-7B
```

## Execution gates

1. Tiny BF16 smoke test (1–3 math questions)
2. Qwen-7B × BF16 × MATH-500 × seed 0
3. Qwen-7B × GPTQ-4 × MATH-500 × seed 0

Full workflow: [docs/RUNBOOK.md](docs/RUNBOOK.md)
