# PARAM Rudra (IIT Patna) — Cluster Notes

**Login:** `manishn_iitp@paramrudra.iitp.ac.in`  
**Port:** `4422` (external) or `22` (on-campus IITP)  
**Scratch workspace:** `/scratch/manishn_iitp/reasoning-compression-lab`  
**GitHub:** https://github.com/Manish06N/reasoning-compression-lab

---

## Every SSH session

```bash
export QR=/scratch/$USER/reasoning-compression-lab
export QREASON_MODEL_QWEN7B=$QR/models/DeepSeek-R1-Distill-Qwen-7B
cd $QR
module load mldl/Miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qreason
```

HF cache lives on scratch (set automatically by `scripts/hpc/param_rudra_env.sh`):

```text
/scratch/$USER/reasoning-compression-lab/hf_cache/
```

---

## SLURM (PARAM Rudra)

| Setting | Value |
|---------|-------|
| GPU partition | `--partition=gpu` |
| GPU request | `--gres=gpu:1` (not `gpu:a100:1`) |
| Typical CPU | `--cpus-per-task=8` |
| Typical RAM | `--mem=80G` for GPU inference jobs |

Interactive GPU session:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=04:00:00 --pty bash
```

Submit smoke test:

```bash
sbatch slurm/smoke_test.slurm
# logs: logs/smoke_test_<jobid>.out
```

---

## MacBook ↔ HPC sync

**Preferred:** push from MacBook, pull on HPC:

```bash
# MacBook (after commits)
git push

# HPC
cd $QR && git pull origin main
```

**Alternative:** rsync from MacBook:

```bash
bash scripts/macbook/rsync_from_hpc.sh          # pull code from HPC
SYNC_RESULTS=1 bash scripts/macbook/rsync_from_hpc.sh  # code + runs/results
bash scripts/macbook/sync_results_from_hpc.sh   # runs/results only
```

---

## Gate checklist (your progress)

| Gate | Command | Status |
|------|---------|--------|
| 1 GPU | `bash scripts/hpc/01_gpu_check.sh` | Passed |
| 2 Model | `bash scripts/hpc/02_download_model.sh` | Done |
| 2b Dataset | `bash scripts/hpc/02b_validate_dataset.sh` | Done |
| 3 Smoke | `sbatch slurm/smoke_test.slurm` | Check job logs |
| 4 Debug | `bash scripts/hpc/04_run_level_a_bf16.sh 10` | After Gate 3 |
| 4 Full | `sbatch slurm/run_level_a_bf16.slurm` | After debug |

---

## Security

- Rotate HF token if it was pasted in chat.
- Keep `hf_cache/token` at mode `600`; it is gitignored.
- Do not add cluster SSH keys to personal GitHub unless they are personal deploy keys.
