# reasoning-compression-lab

Deployment-science evaluation harness for compressed reasoning LLMs.

**GitHub:** https://github.com/Manish06N/reasoning-compression-lab  
**Paper 1:** *Beyond Accuracy: Reliability, Calibration, Seed Variance, and Cost-per-Correct of Quantized Reasoning LLMs*

**What is QRM?** Shorthand for Liu et al., COLM 2025 — [*Quantization Hurts Reasoning?*](https://arxiv.org/abs/2504.04823) ([code](https://github.com/ruikangliu/Quantized-Reasoning-Models)). That paper is our **accuracy baseline**; Paper 1 goes **beyond** accuracy (calibration, cost, truncation). Level A / b01 checks we can run their protocol before the quant grid. Full beginner explanation: [docs/BEGINNER_HPC_GUIDE.md §1.1–1.2](docs/BEGINNER_HPC_GUIDE.md).

**Roadmap:** PhD plan V8.2 (1 Jul 2026) — see [docs/plans/2026-07-01-v82-reengineering.md](docs/plans/2026-07-01-v82-reengineering.md) and [papers/j1/protocol.yaml](papers/j1/protocol.yaml).

> **Publication decision (2026-08-14): Needs revision.** The completed FP8 runs are valid replication/control evidence, not a standalone publishable result. Read the [publication-readiness audit](docs/PUBLICATION_READINESS.md) and follow the [publication-recovery plan](docs/plans/2026-08-14-publication-recovery.md). No broad b03/b04 or b01–b09 launch is allowed before recovery Phase 0 passes.

## Current status (2026-08-14)

The modern-stack b02 FP8 run was stopped after unhealthy output. Both checkpoints then passed the exact-stack pilot, and full jobs 96100/96101 completed successfully. Independent review shows that these runs reproduce known FP8 accuracy but do not isolate quantization or support the planned calibration/cost claims.

| Item | Status |
|------|--------|
| **Repository state** | Baseline code is commit `4796614`; the 2026-08-14 audit/plan and earlier run-status doc edits are currently uncommitted on HPC; `.qrm_official_env_ready` remains untracked |
| **Stopped modern b02** | Jobs **96086/96087** canceled; Qwen's first 10 rows were 2/10 correct with 8/10 truncations and obvious repetition |
| **V0 probe** | Jobs **96091/96092** showed that `VLLM_USE_V1=0` alone did not fix malformed/looping output |
| **b02 first attempt** | Jobs **96084/96085** failed before raw rows with `fp8_e5m2 kv-cache is not supported with fp8 checkpoints`; fixed in `542f622` by setting FP8 checkpoint KV cache to `auto` |
| **Official QRM parity** | Job **87302** completed: Qwen-7B BF16, n=10 MATH-500, seed 42, **10/10 correct**, **0 truncation** |
| **FP8 exact-stack gate** | Jobs **96093/96094** completed: Qwen and Llama each **10/10 correct**, 10/10 boxed, 0 token-cap hits, 0 repetition flags |
| **Full exact-stack run** | **COMPLETED** — 96100 Qwen: 472/500 (**94.4%**); 96101 Llama: 445/500 (**89.0%**); MATH-500, seed 42 |
| **Interpretation** | Both values are compatible with existing FP8 model-card results. They are replication evidence, not a matched BF16-vs-FP8 quantization comparison |
| **Runtime qualification** | A100 used vLLM weight-only Marlin fallback for the FP8 checkpoints; do not describe the result as native FP8/W8A8 compute |
| **Trace audit** | Six likely near-cap endings and phrase-level degeneration were found; saved output lacks `finish_reason`, so instrumentation must be repaired before causal trace claims |
| **Path C / our stack** | Canceled at n=20 after Qwen **10% pass@1 / 90% trunc** and Llama **15% pass@1 / 75% trunc** |
| **b01 July archive** | Deployment-stack BF16 evidence: Llama 500/500 **19.6% pass@1 / 58% trunc**; Qwen 410/500 about 94% trunc |
| **Publication verdict** | **Needs revision** — suitable for an appendix/control table only |
| **Next gate** | Recovery **Phase 0**: clean reproducibility, tracked QRM patches, finish/token/timing/telemetry schema, stronger pathology validation, tests, then tiny smoke runs |
| **Calibration boundary** | b02 launcher scores with `--skip-calibration`; do not use b02 for Brier/AURC/ECE claims until valid confidence exists |

**Decision and execution:** [publication audit](docs/PUBLICATION_READINESS.md) · [recovery plan](docs/plans/2026-08-14-publication-recovery.md) · [notes.md sections 31-37](notes.md) · [stack audit](docs/QRM_STACK_PARITY_AUDIT.md) · [environment debug log](docs/QRM_OFFICIAL_HPC_TROUBLESHOOTING.md)

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

### Historical pre-rerun mechanics (do not submit a grid under the current gate)

Use these commands only as references for sync/preflight mechanics. The authorized order is recovery Phase 0 → tiny smoke → matched P1.

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

# Broad b01 launch is blocked by the 2026-08-14 publication-recovery gate.

# After b01 completes — sync again, then score (needs 286f5e4+ for baseline yaml)
python scripts/score_run.py --input $QREASON_OUTPUT_ROOT/raw/level_a_qwen7b_bf16_math500_seed0.jsonl \
  --summary $QREASON_OUTPUT_ROOT/results/level_a_qwen7b_bf16_math500_seed0_summary.json --skip-calibration
python scripts/compare_qrm_baseline.py --summary $QREASON_OUTPUT_ROOT/results/level_a_qwen7b_bf16_math500_seed0_summary.json
```

Do **not** cite archive `outputs-hpc-2a100-main-2026-06-29` as a quantization result. It remains diagnostic stack-failure evidence; new comparisons follow Protocol P1-2026-08.

## Publication execution strategy

> **HPC-only for paper numbers; gated by evidence, not queue availability.** The historical b01–b09 scripts remain operational tools but are not the current scientific order.

| Stage | Authorized scope |
|-------|------------------|
| Recovery P0 | Track dependency patches; add finish/token/timing/telemetry fields; strengthen validators; test clean recreation |
| Matched P1 | Qwen-7B + Llama-8B × {BF16, FP8} × MATH-500 × seed 42 |
| Pilot P2 | Same models × {BF16, FP8, AWQ4, GPTQ4} × MATH-500 × seeds 42/43/44 |
| Contribution P3 | Select quantization reliability–cost, controlled stack transfer, or negative-results artifact |
| Confirmation | Extend approved headline cells to seeds 42–46; add breadth only after approval |

All compared cells must use one frozen Protocol P1-2026-08 stack and complete provenance. The old `repro_qrm.yaml` seed-0 block grid is historical engineering evidence. RTX 5080 remains outside J1 paper numbers.

## Push to GitHub

Preferred: MacBook commits code/docs and runs `git push origin main`; HPC then runs `git fetch && git reset --hard origin/main`. If credentials are intentionally configured on HPC, HPC may push small project-doc/code commits directly after checking `git status` and excluding runtime markers such as `.qrm_official_env_ready`.

Credentials: [docs/GIT_CREDENTIALS.md](docs/GIT_CREDENTIALS.md). Windows setup notes: [docs/archive/GITHUB_PUSH.md](docs/archive/GITHUB_PUSH.md).

### HPC after push

```bash
ssh manishn_iitp@paramrudra.iitp.ac.in -p 4422
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR && git fetch origin && git reset --hard origin/main
python scripts/verify_decoding_params.py
# Do not call submit_hpc_blocks.sh until the recovery plan authorizes a cell.
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

1. **Recovery P0:** clean reproducibility and observability.
2. **Matched P1:** four BF16/FP8 seed-42 cells.
3. **Pilot P2:** 24 MATH-500 cells across three seeds.
4. **Contribution P3:** supervisor-approved primary RQs/endpoints.
5. **Confirmation:** five-seed headline evidence and gated breadth.

## First experiment

The next work is Phase 0 code/instrumentation and clean-recreation testing. The next GPU action is a tiny instrumented smoke; the next scientific comparison is matched BF16/FP8 at seed 42.

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
