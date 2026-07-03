# Environment variables

Central reference for environment variables used by `reasoning-compression-lab`. Copy secrets and paths into `.env` (never commit `.env`); see [`.env.example`](../.env.example).

**Set on:** MacBook via `.env` + `source scripts/local/env.sh`; HPC via `source scripts/hpc/param_rudra_env.sh` and block launchers.

---

## Workspace and Hugging Face cache

| Variable | Default | Set in | Effect | Publication |
|----------|---------|--------|--------|-------------|
| `QR` | HPC: `/scratch/$USER/reasoning-compression-lab`; local: see `.env.example` | `.env`, `param_rudra_env.sh`, `env.sh` | Repo/scratch root; models and outputs live under here | Required |
| `HF_TOKEN` | (none) | `.env` | Hugging Face Hub auth for gated models | Required for GPQA download |
| `HF_HOME` | `$QR/hf_cache` | `param_rudra_env.sh`, `env.sh` | Hugging Face cache root on scratch | Recommended on HPC |
| `HF_HUB_CACHE` | `$HF_HOME/hub` | same | Model hub cache | Recommended on HPC |
| `TRANSFORMERS_CACHE` | `$HF_HOME/transformers` | same | Transformers cache | Recommended on HPC |
| `HF_DATASETS_CACHE` | `$HF_HOME/datasets` | same | Datasets cache | Recommended on HPC |

---

## Model local paths (`QREASON_MODEL_*`)

Each maps to a on-disk model directory. Defaults are under `$QR/models/`. See [`scripts/local/env.sh`](../scripts/local/env.sh) and [`scripts/hpc/param_rudra_env.sh`](../scripts/hpc/param_rudra_env.sh).

| Variable | Default subdirectory |
|----------|---------------------|
| `QREASON_MODEL_QWEN15B` | `DeepSeek-R1-Distill-Qwen-1.5B` |
| `QREASON_MODEL_QWEN15B_FP8` | `DeepSeek-R1-Distill-Qwen-1.5B-FP8` |
| `QREASON_MODEL_QWEN15B_AWQ4` | `DeepSeek-R1-Distill-Qwen-1.5B-AWQ-4` |
| `QREASON_MODEL_QWEN15B_GPTQ4` | `DeepSeek-R1-Distill-Qwen-1.5B-GPTQ-4` |
| `QREASON_MODEL_QWEN7B` | `DeepSeek-R1-Distill-Qwen-7B` |
| `QREASON_MODEL_QWEN7B_FP8` | `DeepSeek-R1-Distill-Qwen-7B-FP8` |
| `QREASON_MODEL_QWEN7B_AWQ4` | `DeepSeek-R1-Distill-Qwen-7B-AWQ-4` |
| `QREASON_MODEL_QWEN7B_GPTQ4` | `DeepSeek-R1-Distill-Qwen-7B-GPTQ-4` |
| `QREASON_MODEL_QWEN7B_GPTQ3` | `DeepSeek-R1-Distill-Qwen-7B-GPTQ-3` |
| `QREASON_MODEL_LLAMA8B` | `DeepSeek-R1-Distill-Llama-8B` |
| `QREASON_MODEL_LLAMA8B_FP8` | `DeepSeek-R1-Distill-Llama-8B-FP8` |
| `QREASON_MODEL_LLAMA8B_AWQ4` | `DeepSeek-R1-Distill-Llama-8B-AWQ-4` |
| `QREASON_MODEL_LLAMA8B_GPTQ4` | `DeepSeek-R1-Distill-Llama-8B-GPTQ-4` |

Cell configs reference these via `local_path_env` in `configs/models/*.json`.

---

## Run control and archives

| Variable | Default | Set in | Effect | Publication |
|----------|---------|--------|--------|-------------|
| `QREASON_OUTPUT_ROOT` | `run_hpc_2a100_publication.sh`: `$QR/outputs-hpc-2a100-main-${QREASON_HPC_DATE}` | Block launchers, manual export | Root for `raw/`, `scored/`, `results/`, `logs/` | **Required** — use a fresh dated path per rerun |
| `QREASON_HPC_DATE` | `$(date +%Y-%m-%d)` | `run_hpc_2a100_publication.sh` | Date tag in default archive name | Optional |
| `QREASON_FRESH_RUN` | unset | Block launchers | `1` → pass `--fresh` to inference (wipe per-cell output on start) | Use for intentional restarts |
| `QREASON_ALLOW_RESUME` | unset | Python (`resume_guard`), shell assert | `1` / `true` / `yes` → bypass resume safety blocks (not recommended) | Avoid for publication |
| `QREASON_ALLOW_BAD_ARCHIVE` | unset | Shell assert (legacy) | `1` → allow resume into a forbidden archive path | **Never** for paper numbers |
| `QREASON_FORBIDDEN_ARCHIVE_PATTERNS` | (legacy built-ins) | Manual export | Comma-separated substrings; path matching any substring is blocked | Optional extra blocklist |
| `QREASON_PUBLICATION_MODE` | unset (launchers set `1`) | HPC/5080 launchers, Python | `1` → fail-closed validation, clean git, `batch_size=1`, math-verify required | **Set to `1`** on HPC publication |
| `QREASON_DECODING` | `configs/decoding/repro_qrm.yaml` | `run_hpc_2a100_publication.sh` | Decoding config path for block runs | Keep default for J1 |
| `QREASON_BATCH_SIZE` | `1` | Block launchers | vLLM batch size; publication forces `1` | Must be `1` |
| `QREASON_CHECKPOINT_EVERY` | `10` | Block launchers | Write checkpoint validation every N rows | Default OK |
| `QREASON_CELL_QUEUE` | unset | 5080 scripts | Path to shell snippet defining `CELL_QUEUE` array | 5080 only |
| `QREASON_ENABLE_AUTOPUSH` | unset (off) | Manual export or submit script | `1` -> start tmux `git_autopush_outputs.sh` loop on submit | **Off by default** — prefer MacBook rsync after runs |
| `QREASON_SUBMIT_2GPU_MODE` | `split` | `submit_hpc_blocks.sh` | `split` -> two independent `--gres=gpu:1` jobs; `exclusive_block` -> one `--gres=gpu:2 --exclusive` block job | Use default for b01 when 2-GPU allocations are scarce |
| `QREASON_SLURM_EXCLUSIVE` | `1` | `submit_hpc_blocks.sh` | Adds `--exclusive` for 2-GPU block submits; set `0` only for debugging | Keep `1` for publication |
| `QREASON_SLURM_EXCLUDE` | unset | `submit_hpc_blocks.sh` | Optional comma-separated node exclude list passed to `sbatch --exclude` | Use to avoid nodes that just failed free-VRAM preflight |
| `QREASON_MIN_FREE_GPU_MB` | `70000` | `run_hpc_2a100_publication.sh` | Refuse to start vLLM when the selected GPU has less free VRAM; set `0` to disable | Keep default for A100 BF16 anchors |
| `QREASON_GPU_PREFLIGHT_REQUEUE` | `1` | `run_hpc_2a100_publication.sh` | Requeue the Slurm job when free-VRAM preflight finds a busy assigned GPU | Keep enabled for split b01 retries |
| `QREASON_GPU_PREFLIGHT_REQUEUE_MAX` | `240` | `run_hpc_2a100_publication.sh` | Maximum self-requeue attempts based on `SLURM_RESTART_COUNT` | Lower only if you want busy-GPU retries to fail sooner |

**Forbidden archive marker:** Place `INVALID_FOR_PUBLICATION.txt` in an archive root to block resume/scoring into that tree (see `src/runners/resume_guard.py`).

**`QREASON_ALLOW_BAD_ARCHIVE` vs `QREASON_ALLOW_RESUME`:** The former only bypasses the forbidden-archive path check (shell legacy). The latter bypasses all Python resume guards (stale decoding, config hash, git commit). Use neither for publication runs.

**Submit script (`scripts/hpc/submit_hpc_blocks.sh`):** Resolves `QR`, `QREASON_OUTPUT_ROOT`, and `QREASON_HPC_DATE` once per submit batch and passes them to every `sbatch` job. Default target `all` submits **b01 only** (gate-safe). Two-GPU blocks default to two independent `--gres=gpu:1` jobs; set `QREASON_SUBMIT_2GPU_MODE=exclusive_block` only when a 2-GPU allocation is acceptable. Use `all_blocks` for b01–b06 soak tests (stderr warning). Pass `--fresh` to set `QREASON_FRESH_RUN=1` for that batch only; split cell jobs clear `QREASON_FRESH_RUN` so later cells do not wipe progress.

---

## Cluster and GPU

| Variable | Default | Set in | Effect | Publication |
|----------|---------|--------|--------|-------------|
| `CONDA_ROOT` | `/home/apps/MSCC/miniconda3` | `param_rudra_env.sh` | Conda install path on PARAM Rudra | Override on other clusters |
| `CUDA_VISIBLE_DEVICES` | Slurm/job | Slurm | GPU index visible to process; affects vLLM placement and NVML mapping | Preserve scheduler allocation; launcher only narrows multi-GPU block jobs to one selected visible device |
| `VLLM_BATCH_INVARIANT` | unset | 5080 publication | `1` → enforce batch-size invariance checks | HPC uses `QREASON_PUBLICATION_MODE` instead |

---

## CLI equivalents

Several Python flags mirror env vars:

| Env | CLI flag |
|-----|----------|
| `QREASON_PUBLICATION_MODE=1` | `--publication` on `run_inference.py`, `run_inference_multisample.py`, `score_run.py` |
| `QREASON_ALLOW_RESUME=1` | `--allow-resume` on inference scripts |
| Fresh run | `--fresh` on inference scripts |

Publication mode is active if **either** the env var or the CLI flag is set (`src/runners/publication_mode.py`).
