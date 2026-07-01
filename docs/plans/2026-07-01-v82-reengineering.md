# V8.2 Codebase Re-engineering Plan

> **Status: COMPLETE** (codebase architecture — 1 July 2026)

**Goal:** Align `reasoning-compression-lab` with PhD Roadmap V8.2 while keeping HPC entrypoints stable.

See also: [V8_2_ARCHITECTURE.md](../V8_2_ARCHITECTURE.md)

---

## Phase 0 — Reference repos

- [x] Clone script: `external_repos/clone_v82_repos.sh`
- [x] J1–J3 repo folders documented in `external_repos/README.md`
- [x] Pin recorder: `scripts/record_external_repo_pins.sh`
- [ ] **Manual:** `lighteval` if git-lfs missing — run clone script on MacBook

---

## Phase 1 — J1 foundation

- [x] `papers/j1/protocol.yaml`
- [x] `schemas/*.v1.json`
- [x] `src/schemas/` provenance + validation
- [x] `src/evaluation/statistics/` McNemar, cluster bootstrap, Holm
- [x] QRM prompts + `prompt_profile` on all cells
- [x] Provenance in `run_inference.py`
- [x] `scripts/j1/compare_configs.py`
- [x] Calibration + selective risk in `score_run.py`
- [x] Adaptive ECE, NLL, reliability bins
- [x] Parquet export `scripts/export_parquet.py`
- [x] Repro seeds 42/43/44 cells
- [x] Audit sampler `papers/j1/audit/` + `scripts/j1/sample_audit.py`
- [x] Seed aggregation + C1 cache export

---

## Phase 2 — Reproduction gate (HPC ops)

Operational steps (run on HPC, not in repo):

1. `git reset --hard origin/main`
2. Delete bad archive; fresh `QREASON_OUTPUT_ROOT`
3. Level A with `prompt_profile: reproduction`
4. `compare_qrm_baseline.py` + `scripts/j1/compare_configs.py`

---

## Phase 3 — Core grid + C1

- [x] Cell grid configs (Level A/B/C)
- [x] `scripts/j1/aggregate_seeds.py`
- [x] `scripts/j1/export_conference_cache.py`
- [x] Manual audit tooling

---

## Phase 4 — Paper 2 (J2)

- [x] `papers/j2/protocol.yaml`
- [x] `src/generation/sglang/` pilot stub
- [x] `configs/serving/sglang.yaml`
- [x] `scripts/j2/run_method_pilot.py`

---

## Phase 5 — Paper 3 (J3)

- [x] `papers/j3/protocol.yaml` + `language_matrix.yaml`
- [x] `src/generation/llamacpp/` pilot stub
- [x] `configs/serving/llamacpp.yaml`
- [x] `scripts/j3/preflight_indic.py`
- [x] `scripts/j3/run_local_transfer.py`

---

## Shared infrastructure

- [x] `configs/quantization/registry.yaml`
- [x] `configs/serving/vllm.yaml`
- [x] `scripts/build_dashboard.py` → `dashboards/`
- [x] `src/generation/vllm/` canonical backend wrapper
- [x] `src/evaluation/correctness|calibration|selective_risk/`
- [x] Tests: `tests/test_v82_*.py`

---

## Backward compatibility

1. `scripts/run_inference.py` and `scripts/score_run.py` unchanged CLI
2. Archive layout `outputs-hpc-2a100-main-*/` unchanged
3. `src/metrics/` and `src/runners/` retained

---

## What remains **experiment execution** (not code)

These are runtime PhD milestones, not missing codebase pieces:

- HPC b01 rerun with fixed decoding
- J2 4–6 week method-selection pilot **runs**
- J3 human reviewer recruitment and Indic **runs**
- Journal submissions
