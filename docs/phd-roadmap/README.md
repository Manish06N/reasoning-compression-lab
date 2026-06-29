# PhD Roadmap Documentation

**Prepared for:** Manish Nandish  
**Thesis spine:** Reliable and Cost-Efficient Deployment of Reasoning LLMs under Compression, Evaluation, and Multilingual Constraints  
**Repo:** [reasoning-compression-lab](https://github.com/Manish06N/reasoning-compression-lab)

This folder is the **canonical PhD planning archive** for Paper 1 and the wider thesis. It complements operational logs (`progress.md`, `CHANGELOG.md`) and experiment design (`docs/PAPER1_DESIGN.md`).

---

## How to read these documents

| If you are… | Start here |
|-------------|------------|
| New to the topic | [01_BEGINNER_AND_THESIS_SPINE.md](01_BEGINNER_AND_THESIS_SPINE.md) |
| Choosing or defending the track | [02_TRACK_DECISION.md](02_TRACK_DECISION.md) |
| Planning journals / supervisor meeting | [03_PUBLICATION_STRATEGY.md](03_PUBLICATION_STRATEGY.md) |
| Running Paper 1 experiments | [04_PAPER1_FULL_SPEC.md](04_PAPER1_FULL_SPEC.md) + [docs/PAPER1_DESIGN.md](../PAPER1_DESIGN.md) |
| Paper 2 / Paper 3 direction | [05_PAPER2_AND_PAPER3.md](05_PAPER2_AND_PAPER3.md) |
| Executing on HPC / cluster | [06_EXECUTION_PLAN.md](06_EXECUTION_PLAN.md) + [docs/HPC_STEP_BY_STEP.md](../HPC_STEP_BY_STEP.md) |
| Job market / artifacts | [07_V7_JOB_FIRST_STRATEGY.md](07_V7_JOB_FIRST_STRATEGY.md) |
| Literature / scoop checks | [08_EXECUTION_VALIDATION_V6.md](08_EXECUTION_VALIDATION_V6.md) |
| After vLLM grid — GGUF / edge | [09_STACK_TRANSFER_GGUF.md](09_STACK_TRANSFER_GGUF.md) |
| Quick lookup | [10_APPENDICES_REFERENCE.md](10_APPENDICES_REFERENCE.md) |

---

## Version history

| Version | File(s) | Role |
|---------|---------|------|
| **V5** | Parts 1–15 + Appendices B–L | Beginner-to-execution master report |
| **V6** | [08_EXECUTION_VALIDATION_V6.md](08_EXECUTION_VALIDATION_V6.md) | Literature saturation, exact specs, supervisor tracker, artifact architecture |
| **V7** | [07_V7_JOB_FIRST_STRATEGY.md](07_V7_JOB_FIRST_STRATEGY.md) | Job-first reweighting; V6 remains evidence base |

**Control rule (V7):** Do not rewrite the roadmap again. The next intellectual document after preflight passes should be **Paper 1 Design Doc v1 with reproduction numbers**.

---

## Relationship to this repository

| Roadmap concept | Repo artifact |
|-----------------|---------------|
| Paper 1 harness | `scripts/run_inference.py`, `scripts/score_run.py`, `src/metrics/` |
| HPC publication grid | `configs/machine_split/hpc_blocks/`, `scripts/hpc/submit_hpc_blocks.sh` |
| Live experiment status | [progress.md](../../progress.md), [docs/PROGRESS.md](../PROGRESS.md) |
| Paper tables / repro bundle | `scripts/build_paper_tables.py`, `scripts/build_repro_bundle.py` |

---

## Evidence boundary

Literature saturation tables and journal quartile notes in these documents are **planning worksheets**. Before submission, verify every paper title, venue, year, URL, and quartile from primary sources and your institution’s ranking rules.
