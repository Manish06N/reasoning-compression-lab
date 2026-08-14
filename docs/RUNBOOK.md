# Runbook - MacBook <-> HPC

## Current project state (2026-08-14)

- Canonical scientific verdict: **Needs revision** — [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md).
- Canonical next actions: [plans/2026-08-14-publication-recovery.md](plans/2026-08-14-publication-recovery.md). No b03/b04 or broad-grid launch before recovery Phase 0 passes.

- GitHub/HPC baseline is `4796614`; current audit/plan/docs are uncommitted on HPC and must be preserved for sync; `.qrm_official_env_ready` should remain untracked.
- Modern-stack b02 jobs **96086/96087** were canceled after unhealthy generated output; retain `outputs-hpc-2a100-main-2026-08-13` as evidence.
- First b02 attempt jobs **96084/96085** failed before raw rows with the vLLM FP8 checkpoint plus FP8 KV-cache incompatibility; commit `542f622` fixes FP8 configs to `kv_cache_dtype: auto`.
- Official QRM parity job **87302** completed 10/10 correct with 0 truncation under `qrm-official`; this confirmed prompt/protocol and isolated a `qreason` stack behavior gap.
- Exact-stack FP8 validation jobs **96093/96094** both passed 10/10 correct, 10/10 boxed, with no cap hits or repetition flags.
- Completed full correctness jobs: **96100** Qwen 472/500 (94.4%) and **96101** Llama 445/500 (89.0%), n=500/seed42, archive `outputs-hpc-qrm-official-fp8-full-2026-08-13`.
- Interpretation: replication/control only; no same-stack BF16, multi-seed, calibration, or controlled performance evidence.
- Calibration claims still require valid confidence rows; b02 uses `--skip-calibration`.


## MacBook (control room)

Use for: reading papers, writing, repo structure, Python scripts, configs, VS Code Remote SSH, logs, CSV analysis, plotting, README/docs.

**Project root (MacBook):**

```
/Users/manish/Projects/2026/paper 1/reasoning-compression-lab
```

## HPC (experiment factory)

Use for: model download, vLLM, BF16/FP8/GPTQ/AWQ inference, quantization, benchmark generation, latency/VRAM profiling, final paper numbers.

## Windows RTX 5080 (WSL2 pilot lab)

Use for: historical pilots and J3 local transfer only. Paper 1/J1 publication numbers run on PARAM Rudra HPC.

**Project root (WSL):**

```bash
cd "/mnt/g/ALL MY Projects/2026/03-paper1-experiments"
source scripts/local/env.sh
```

**Historical 5080 commands:**

```bash
bash scripts/local/start_5080_pilot.sh      # background
bash scripts/local/resume_5080_pilot.sh     # foreground / after reboot
bash scripts/local/backup_5080_archive.sh --snapshot
```

5080 is retired for J1 publication numbers; see [archive/RTX5080_EXECUTION_PLAN.md](archive/RTX5080_EXECUTION_PLAN.md).

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

- Do not submit b03/b04 or a broad grid; completed 96100/96101 were reviewed and triggered recovery Phase 0.
- Do not cite Brier/AURC/ECE until rows have valid confidence from logprobs or maj@5.
- Do not use June/July diagnostic archives as QRM reproduction claims; use them as deployment-stack evidence only.
- No 14B, Paper 2, or breadth expansion before the matched three-seed contribution gate.
