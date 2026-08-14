# Environment variables

Central reference for environment variables used by `reasoning-compression-lab`. Copy secrets and paths into `.env` (never commit `.env`); see [`.env.example`](../.env.example).

**Set on:** MacBook via `.env` + `source scripts/local/env.sh`; HPC via `source scripts/hpc/param_rudra_env.sh` and block launchers.

**Publication warning (2026-08-14):** environment variables do not define scientific comparability by themselves. New runs require a frozen Protocol P1-2026-08 config hash and the complete provenance/finish/timing/telemetry schema in [the recovery plan](plans/2026-08-14-publication-recovery.md). Historical seed-0 defaults must not silently control new pilot cells.

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
| `QREASON_SLURM_EXCLUSIVE` | `0` | `submit_hpc_blocks.sh` | `1` adds `--exclusive` for **exclusive_block** mode only; **ignored for split/single 1-GPU cells** (QOS trap on ragpu) | **Keep `0`** for parallel b01; see `docs/PARAM_RUDRA_SLURM.md` |
| `QREASON_SLURM_EXCLUDE` | unset | `submit_hpc_blocks.sh` | Optional comma-separated node exclude list passed to `sbatch --exclude` | Use to avoid nodes that just failed free-VRAM preflight |
| `QREASON_MIN_FREE_GPU_MB` | `70000` | `run_hpc_2a100_publication.sh` | Refuse to start vLLM when the selected GPU has less free VRAM; set `0` to disable | Keep default for A100 BF16 anchors |
| `QREASON_GPU_PREFLIGHT_REQUEUE` | `1` | `run_hpc_2a100_publication.sh` | Requeue the Slurm job when free-VRAM preflight finds a busy assigned GPU | Keep enabled for split b01 retries |
| `QREASON_GPU_PREFLIGHT_REQUEUE_MAX` | `240` | `run_hpc_2a100_publication.sh` | Maximum self-requeue attempts based on `SLURM_RESTART_COUNT` | Lower only if you want busy-GPU retries to fail sooner |

**Forbidden archive marker:** Place `INVALID_FOR_PUBLICATION.txt` in an archive root to block resume/scoring into that tree (see `src/runners/resume_guard.py`).

**`QREASON_ALLOW_BAD_ARCHIVE` vs `QREASON_ALLOW_RESUME`:** The former only bypasses the forbidden-archive path check (shell legacy). The latter bypasses all Python resume guards (stale decoding, config hash, git commit). Use neither for publication runs.

**Submit script (`scripts/hpc/submit_hpc_blocks.sh`):** Resolves `QR`, `QREASON_OUTPUT_ROOT`, and `QREASON_HPC_DATE` once per submit batch and passes them to every `sbatch` job. Default target `all` submits **b01 only** (gate-safe). Two-GPU blocks default to two independent `--gres=gpu:1` jobs; set `QREASON_SUBMIT_2GPU_MODE=exclusive_block` only when a 2-GPU allocation is acceptable. Use `all_blocks` for b01–b06 soak tests (stderr warning). Pass `--fresh` to set `QREASON_FRESH_RUN=1` for that batch only; split cell jobs clear `QREASON_FRESH_RUN` so later cells do not wipe progress.

---

## Official QRM parity and FP8 gate (`QRM_*`)

These variables apply only to `slurm/qrm_official_math500_n10.slurm`, `scripts/hpc/qrm_parity/run_official_inference.sh`, and `scripts/hpc/submit_qrm_fp8_full.sh`. They use the isolated `qrm-official` environment, not the main `qreason` harness.

| Variable | Default | Effect |
|----------|---------|--------|
| `QRM_CONDA_ENV` | `qrm-official` | Conda environment containing the pinned authors' stack |
| `QRM_REPO_DIR` | `$QR/external/Quantized-Reasoning-Models` | Authors' repository/entrypoint path |
| `QRM_MODEL_PATH` | Qwen-7B BF16 path | Model loaded by official `inference.py`; submitters override per model |
| `QRM_OUTPUT_ROOT` | dated `outputs-hpc-qrm-official-*` | Shared archive root; per-model inference subdirectories prevent collisions |
| `QRM_MAX_SAMPLES` | `10` | Number of MATH-500 examples; guarded full submitter sets `500` |
| `QRM_SEED` | `42` | Lighteval/vLLM generation seed |
| `QRM_GPU_MEMORY_UTILIZATION` | `0.75` | vLLM memory fraction on the assigned A100 |
| `QRM_MIN_FREE_GPU_MB` | `62000` | Dirty-GPU preflight threshold |
| `QRM_REQUEUE_ON_DIRTY_GPU` | `1` | Requeue instead of running on an insufficiently free GPU |
| `QRM_MAX_DIRTY_GPU_REQUEUES` | `3` | Maximum dirty-GPU requeues |
| `QRM_MIN_ACCURACY` | `0` | Optional post-run minimum extractive-match accuracy |
| `QRM_MIN_BOXED_RATE` | `0` | Optional post-run minimum fraction containing `\\boxed` |
| `QRM_MAX_TOKEN_LIMIT_HITS` | `QRM_MAX_SAMPLES` | Optional maximum number of 32768-token cap hits |
| `QRM_MAX_REPETITION_ROWS` | `QRM_MAX_SAMPLES` | Optional maximum rows flagged for long repeated-word runs |
| `QRM_VALIDATION_ROOT` | `outputs-hpc-qrm-official-fp8-validation-2026-08-13` | Pilot archive consumed by the guarded full submitter |
| `QRM_PYTHON` | `/home/manishn_iitp/.conda/envs/qrm-official/bin/python3` | Interpreter used for tokenizer-level pilot validation |

The general runner always checks result structure, row count, and numeric metrics. Its default quality thresholds are non-blocking because full-run accuracy and truncation are scientific outcomes. `submit_qrm_fp8_full.sh` is deliberately strict before submission: both n=10 pilots must have 100% accuracy/boxed rate, zero token-cap hits, and zero repetition flags.

Top-level copies are model-qualified:

```text
qrm_official_<MODEL>_math500_n<SAMPLES>_seed<SEED>.json
validation/<MODEL>_math500_n<SAMPLES>_seed<SEED>.json
```

The canonical per-model result remains `inference/<MODEL>-seed<SEED>/MATH-500.jsonl`. Despite the suffix, the official file contains one JSON array and appears only after the complete batch finishes.

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
