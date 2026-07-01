# Documentation index

**Understand the system:** [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md) — **canonical high-level map** of the entire repository (architecture, papers, modules, gates, status).

**Run experiments:** [BEGINNER_HPC_GUIDE.md](BEGINNER_HPC_GUIDE.md) — full HPC workflow on PARAM Rudra.

**Live status:** [PROGRESS.md](PROGRESS.md) (short) · [../progress.md](../progress.md) (full history) · [../CHANGELOG.md](../CHANGELOG.md) (ops detail)

---

## Essential (use these)

| Doc | Purpose |
|-----|---------|
| [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md) | **High-level overview** — thesis alignment, architecture, modules, paper status |
| [J1_VALIDATION_RUNBOOK.md](J1_VALIDATION_RUNBOOK.md) | **Next steps** — fresh b01 rerun on HPC |
| [BEGINNER_HPC_GUIDE.md](BEGINNER_HPC_GUIDE.md) | HPC workflow step-by-step |
| [HARDWARE_POLICY.md](HARDWARE_POLICY.md) | J1 HPC-only; RTX for J3 transfer |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | **Critical** — resume trap, bad archive, limitations |
| [REPO_MAP.md](REPO_MAP.md) | Directory map and pipeline flow |
| [PROGRESS.md](PROGRESS.md) | Live status + pre-rerun checklist |
| [V8_2_ARCHITECTURE.md](V8_2_ARCHITECTURE.md) | V8.2 module layout (J1–J3) |
| [plans/2026-07-01-v82-reengineering.md](plans/2026-07-01-v82-reengineering.md) | V8.2 implementation checklist |
| [HPC_2A100_PLAN.md](HPC_2A100_PLAN.md) | Block grid b01–b09, SLURM timing |
| [HPC_PARAM_RUDRA.md](HPC_PARAM_RUDRA.md) | Cluster cheat sheet (SSH, conda, SLURM) |
| [RUNBOOK.md](RUNBOOK.md) | MacBook ↔ GitHub ↔ HPC sync |
| [GPTQ4_PREP.md](GPTQ4_PREP.md) | b04 GPTQ-4 weight gate |
| [GPQA_ACCESS.md](GPQA_ACCESS.md) | Hugging Face GPQA gate |
| [MODEL_SCOPE_DECISION.md](MODEL_SCOPE_DECISION.md) | **Frozen** — in / out / gated model scope for J1 |
| [MODEL_ROSTER.md](MODEL_ROSTER.md) | Canonical model paths / HF IDs |
| [PAPER1_DESIGN.md](PAPER1_DESIGN.md) | Scope, metrics, claim |
| [GIT_CREDENTIALS.md](GIT_CREDENTIALS.md) | GitHub PAT / credentials |

## Reference (read when needed)

| Doc | Purpose |
|-----|---------|
| [V8_2_ARCHITECTURE.md](V8_2_ARCHITECTURE.md) | **V8.2** complete codebase map (J1–J3) |
| [plans/2026-07-01-v82-reengineering.md](plans/2026-07-01-v82-reengineering.md) | V8.2 implementation checklist (complete) |
| [PHD_ROADMAP.md](PHD_ROADMAP.md) | Thesis spine (long; not daily ops) |
| [literature/PAPER1_READING_MAP.md](literature/PAPER1_READING_MAP.md) | Paper reading list |
| [reference_notes/COPY_ADAPT_CHECKLIST.md](reference_notes/COPY_ADAPT_CHECKLIST.md) | What we ported from external repos |
| [reference_notes/LMEVAL_SANITY.md](reference_notes/LMEVAL_SANITY.md) | Optional lm-eval cross-check |
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | Per-cell run template (optional; master log is `progress.md`) |

## External repos (read-only, MacBook)

Not in this repo — see `../external_repos/README.md` and `../external_repos/EXTERNAL_REPOS_REFERENCE.md`.

## Archived

Superseded docs: [archive/README.md](archive/README.md)
