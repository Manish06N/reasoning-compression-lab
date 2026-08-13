# reasoning-compression-lab

Deployment-science evaluation harness for compressed reasoning LLMs.

**GitHub:** https://github.com/Manish06N/reasoning-compression-lab  
**Paper 1:** *Beyond Accuracy: Reliability, Calibration, Seed Variance, and Cost-per-Correct of Quantized Reasoning LLMs*

**What is QRM?** Shorthand for Liu et al., COLM 2025 — [*Quantization Hurts Reasoning?*](https://arxiv.org/abs/2504.04823) ([code](https://github.com/ruikangliu/Quantized-Reasoning-Models)). That paper is our **accuracy baseline**; Paper 1 goes **beyond** accuracy (calibration, cost, truncation). Level A / b01 checks we can run their protocol before the quant grid. Full beginner explanation: [docs/BEGINNER_HPC_GUIDE.md §1.1–1.2](docs/BEGINNER_HPC_GUIDE.md).

**Roadmap:** PhD plan V8.2 (1 Jul 2026) — see [docs/plans/2026-07-01-v82-reengineering.md](docs/plans/2026-07-01-v82-reengineering.md) and [papers/j1/protocol.yaml](papers/j1/protocol.yaml).

## Current status (2026-08-13)

The modern-stack b02 FP8 run was stopped after unhealthy output. Both FP8 checkpoints then passed an n=10 health gate on the exact successful `qrm-official` stack.

| Item | Status |
|------|--------|
| **Git sync** | GitHub and HPC include the FP8 KV-cache fix (`542f622`); MacBook should pull latest `origin/main`; `.qrm_official_env_ready` remains untracked on HPC |
| **Stopped modern b02** | Jobs **96086/96087** canceled; Qwen's first 10 rows were 2/10 correct with 8/10 truncations and obvious repetition |
| **V0 probe** | Jobs **96091/96092** showed that `VLLM_USE_V1=0` alone did not fix malformed/looping output |
| **b02 first attempt** | Jobs **96084/96085** failed before raw rows with `fp8_e5m2 kv-cache is not supported with fp8 checkpoints`; fixed in `542f622` by setting FP8 checkpoint KV cache to `auto` |
| **Official QRM parity** | Job **87302** completed: Qwen-7B BF16, n=10 MATH-500, seed 42, **10/10 correct**, **0 truncation** |
| **FP8 exact-stack gate** | Jobs **96093/96094** completed: Qwen and Llama each **10/10 correct**, 10/10 boxed, 0 token-cap hits, 0 repetition flags |
| **Path C / our stack** | Canceled at n=20 after Qwen **10% pass@1 / 90% trunc** and Llama **15% pass@1 / 75% trunc** |
| **b01 July archive** | Deployment-stack BF16 evidence: Llama 500/500 **19.6% pass@1 / 58% trunc**; Qwen 410/500 about 94% trunc |
| **Next gate** | `submit_qrm_fp8_full.sh` revalidates both pilots before submitting full MATH-500 correctness runs; b03/b04 remain blocked |
| **Calibration boundary** | b02 launcher scores with `--skip-calibration`; do not use b02 for Brier/AURC/ECE claims until valid confidence exists |

```bash
bash scripts/hpc/submit_qrm_fp8_full.sh
```

**Experiments A-D explained:** [notes.md sections 31-34](notes.md) . full audit: [docs/QRM_STACK_PARITY_AUDIT.md](docs/QRM_STACK_PARITY_AUDIT.md) . env debug log: [docs/QRM_OFFICIAL_HPC_TROUBLESHOOTING.md](docs/QRM_OFFICIAL_HPC_TROUBLESHOOTING.md)

---

## One repo, two conda envs (do not mix)

Everything lives in **this** repo (`reasoning-compression-lab`). Experiment A is **not** a separate project — it is a diagnostic track inside the same tree, using a **second conda env** so vLLM versions do not collide.

| Track | Conda env | vLLM | Entry point | Outputs |
|-------|-----------|------|-------------|---------|
| **Main harness** (b01–b09, smoke, Path C) | `qreason` | 0.8.5 | `scripts/run_inference.py` | `outputs-hpc-2a100-main-*`, `outputs-hpc-diag-*` |
| **Experiment A** (official QRM cross-check) | `qrm-official` | 0.7.0 fork | `external/.../inference.py` | `outputs-hpc-qrm-official-*` |

| Path | Purpose |
|------|---------|
| `src/`, `scripts/run_inference.py` | **Our** evaluation harness (`qreason`) |
| `external/Quantized-Reasoning-Models/` | **Authors'** cloned repo (lighteval + vLLM 0.7.0 submodules) |
| `scripts/hpc/qrm_parity/` | Install, run, and compare official QRM stack |
| `models/DeepSeek-R1-Distill-Qwen-7B/` | Shared model weights (both tracks) |

```bash
# Main grid — always qreason
conda activate qreason
bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors

# Experiment A only — separate env, never pip-install into qreason
bash scripts/hpc/submit_qrm_official_test.sh   # uses qrm-official inside the job
```

**Policy:** HPC-only for paper numbers — [HARDWARE_POLICY.md](docs/HARDWARE_POLICY.md).

**Read first:** [progress.md](progress.md) · [notes.md](notes.md) §30–31 · [CHANGELOG.md](CHANGELOG.md)

**b01 QRM gate:** Targets Qwen **93.9%**, Llama **91%** ±5 pp, truncation ≤15% — **not met** on July BF16. Gate fail does **not** block Paper 1; see [qrm_literature_targets.yaml](configs/baselines/qrm_literature_targets.yaml).

**Docs:** [docs/J1_VALIDATION_RUNBOOK.md](docs/J1_VALIDATION_RUNBOOK.md) · [docs/CODEBASE_OVERVIEW.md](docs/CODEBASE_OVERVIEW.md) · [docs/ENV_VARS.md](docs/ENV_VARS.md) · [docs/MODEL_SCOPE_DECISION.md](docs/MODEL_SCOPE_DECISION.md) · [docs/REPO_MAP.md](docs/REPO_MAP.md) · [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) · [docs/PROGRESS.md](docs/PROGRESS.md) · [progress.md](progress.md)

**Environment variables:** [docs/ENV_VARS.md](docs/ENV_VARS.md) (see also [`.env.example`](.env.example))

**Live tracker:** [docs/PROGRESS.md](docs/PROGRESS.md) · **Full history:** [progress.md](progress.md) · **Ops log:** [CHANGELOG.md](CHANGELOG.md)

### Pre-push / pre-rerun checklist (MacBook → GitHub → HPC)

```bash
# MacBook (before git push)
python -m pytest tests/ -q
python scripts/verify_decoding_params.py          # must print VERIFY OK

# After push — on HPC (before inference only, OR at score time after job finishes)
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR && git fetch origin && git reset --hard origin/main
python scripts/verify_decoding_params.py
python scripts/hpc/07_preflight_publication.py
# GPU smoke via Slurm — not on login node (no CUDA)

# Fresh archive — see docs/J1_VALIDATION_RUNBOOK.md
export QREASON_FRESH_RUN=1
bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors

# After b01 completes — sync again, then score (needs 286f5e4+ for baseline yaml)
python scripts/score_run.py --input $QREASON_OUTPUT_ROOT/raw/level_a_qwen7b_bf16_math500_seed0.jsonl \
  --summary $QREASON_OUTPUT_ROOT/results/level_a_qwen7b_bf16_math500_seed0_summary.json --skip-calibration
python scripts/compare_qrm_baseline.py --summary $QREASON_OUTPUT_ROOT/results/level_a_qwen7b_bf16_math500_seed0_summary.json
```

Do **not** cite archive `outputs-hpc-2a100-main-2026-06-29` pass@1 in the paper — rerun with `repetition_penalty: 1.05` first.

## Publication goal (journal)

> **HPC-only:** All publication experiments on **PARAM Rudra 2× A100** (`run_hpc_2a100_publication.sh`).  
> Full plan: [HPC_2A100_PLAN.md](docs/HPC_2A100_PLAN.md)

| Machine | Entry script | What runs there |
|---------|--------------|-----------------|
| **HPC 2× A100** | `scripts/hpc/run_hpc_2a100_publication.sh` | 7B/8B all quants, BF16 anchors, GSM8K, GPQA, 1.5B (b08–b09) |
| **5080 (retired)** | — | Historical only — [archive](docs/archive/RTX5080_EXECUTION_PLAN.md) |

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

## Push to GitHub

Preferred: MacBook commits code/docs and runs `git push origin main`; HPC then runs `git fetch && git reset --hard origin/main`. If credentials are intentionally configured on HPC, HPC may push small project-doc/code commits directly after checking `git status` and excluding runtime markers such as `.qrm_official_env_ready`.

Credentials: [docs/GIT_CREDENTIALS.md](docs/GIT_CREDENTIALS.md). Windows setup notes: [docs/archive/GITHUB_PUSH.md](docs/archive/GITHUB_PUSH.md).

### HPC after push

```bash
ssh manishn_iitp@paramrudra.iitp.ac.in -p 4422
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR && git fetch origin && git reset --hard origin/main
python scripts/verify_decoding_params.py
bash scripts/hpc/submit_hpc_blocks.sh        # default: b01 only
# bash scripts/hpc/submit_hpc_blocks.sh --fresh b01   # new archive + wipe cells
squeue -u $USER
```

## Research question

When reasoning LLMs are compressed, do they remain **accurate, calibrated, stable, fast, memory-efficient, and economically useful** under real serving conditions?

## Division of labor

| Machine | Role |
|---------|------|
| **MacBook** | Design docs, scripts, configs, writing, plotting, pre-push tests |
| **HPC 2× A100** | **All publication runs** — main grid (`run_hpc_2a100_publication.sh`) |
| **5080** | **Retired** — historical partial archive only |

## Repository layout (V8.2)

See **[docs/REPO_MAP.md](docs/REPO_MAP.md)** for the full map.

```
configs/          cells, models, tasks, decoding, quantization, serving
papers/           j1, j2, j3 protocols (V8.2 thesis alignment)
schemas/          JSON Schema for raw rows and summaries
src/
  generation/     vLLM (active), SGLang/llama.cpp (J2/J3 stubs)
  evaluation/     correctness, calibration, selective risk, statistics (canonical)
  runners/        config, vLLM, checkpoints, raw rows, inference session
  metrics/        deprecated shims — prefer src/evaluation/
prompts/          sober + QRM reproduction templates
scripts/          run_inference, score_run, j1/j2/j3, hpc/
tests/            unit tests (see CI badge / pytest)
docs/             All documentation — start at docs/README.md
dashboards/       Generated HTML dashboards
outputs-hpc-*/    Publication archives (git-tracked when autopushed)
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

**Index:** [docs/README.md](docs/README.md) — what to read and what was archived.

| Read first | Purpose |
|------------|---------|
| [CODEBASE_OVERVIEW.md](docs/CODEBASE_OVERVIEW.md) | **High-level overview** — architecture, papers, modules, gates |
| [MODEL_SCOPE_DECISION.md](docs/MODEL_SCOPE_DECISION.md) | **Frozen J1 model scope** — in / out / gated |
| [BEGINNER_HPC_GUIDE.md](docs/BEGINNER_HPC_GUIDE.md) | HPC workflow (start here for runs) |
| [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) | **Critical bugs and traps** |
| [docs/REPO_MAP.md](docs/REPO_MAP.md) | Directory map and pipeline |
| [docs/PROGRESS.md](docs/PROGRESS.md) | Live status + pre-rerun checklist |
| [docs/V8_2_ARCHITECTURE.md](docs/V8_2_ARCHITECTURE.md) | V8.2 module layout |
| [progress.md](progress.md) | Full execution history |
| [CHANGELOG.md](CHANGELOG.md) | Ops log (job IDs, fixes) |

## Tooling (2026-07-01)

| Script | Purpose |
|--------|---------|
| `scripts/verify_decoding_params.py` | Preflight: decoding reaches vLLM |
| `scripts/compare_qrm_baseline.py` | pass@1 sanity vs literature |
| `scripts/j1/compare_configs.py` | McNemar + Holm paired stats |
| `scripts/j1/sample_audit.py` | Extraction audit sample |
| `scripts/run_inference_multisample.py` | maj@5 calibration pilot |
| `scripts/build_pareto_frontier.py` | Cost-per-correct Pareto |
| `scripts/build_dashboard.py` | HTML archive dashboard |
| `scripts/export_parquet.py` | Parquet export for analysis |
| `scripts/j2/run_method_pilot.py` | Paper 2 method gate manifest |
| `scripts/j3/preflight_indic.py` | Paper 3 Indic preflight |
| `scripts/hpc/08_download_gptq4_models.sh` | GPTQ-4 weights for b04 |

See [docs/reference_notes/COPY_ADAPT_CHECKLIST.md](docs/reference_notes/COPY_ADAPT_CHECKLIST.md) and `../external_repos/EXTERNAL_REPOS_REFERENCE.md`.

## HPC quick commands

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

**Retired for publication (2026-06-28).** Partial archive `outputs-win5080-main-2026-06-28/` (10/500 rows) — not for paper. All publication work on HPC.

## HPC 2× A100 (PARAM Rudra)

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR && git pull
bash scripts/hpc/submit_hpc_blocks.sh        # default: b01 only (gate-safe)
# bash scripts/hpc/submit_hpc_blocks.sh all_blocks  # b01–b06 soak only
# GPQA after HF gate: sbatch slurm/hpc_2a100_b07_gpqa.slurm
```

Archive: `outputs-hpc-2a100-main-YYYY-MM-DD/`  
See [HPC_2A100_PLAN.md](docs/HPC_2A100_PLAN.md).
