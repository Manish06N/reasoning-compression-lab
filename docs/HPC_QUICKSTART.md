# HPC Quick Start

**Full guide:** [HPC_STEP_BY_STEP.md](HPC_STEP_BY_STEP.md) — read that document on HPC and follow gates 1–5 in order.

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
bash scripts/hpc/00_setup_env.sh
bash scripts/hpc/01_gpu_check.sh
bash scripts/hpc/02_download_model.sh
bash scripts/hpc/03_smoke_test.sh
sbatch slurm/run_level_a_bf16.slurm
```
