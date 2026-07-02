# Progress — Paper 1 Experiments

**Last updated:** 2026-07-02  
**GitHub `main`:** `c32a423`  
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical log:** [progress.md](../progress.md) · **Ops detail:** [CHANGELOG.md](../CHANGELOG.md)

---

## Summary

| Area | Status |
|------|--------|
| **J1 engineering** | **MVP complete** — RunSpec, revision pins, publication mode, CI, env docs, manifest locking, logprob pipeline |
| **J1 scientific validation** | **In flight** — b01 job **86229** on `outputs-hpc-2a100-main-2026-07-02-p0fix` |
| **QRM baseline gates** | **Fixed** — task bands + quant/profile SKIP logic |
| **15 core validation cells** | Wired seed 0 (b01–b09) |
| **Logprobs** | **Code wired**; GPU smoke required before removing `--skip-calibration` in launcher |
| **Policy** | HPC-only for J1 numbers |
| **Tests** | 32 targeted pass (operational fixes); 99/100 full suite (1 pre-existing scoring test fail) |

**Status label:** *J1 engineering MVP complete; b01 reproduction gate pending job 86229.*

**Read first:** [KNOWN_ISSUES.md](KNOWN_ISSUES.md) · [ENV_VARS.md](ENV_VARS.md) · **Runbook:** [J1_VALIDATION_RUNBOOK.md](J1_VALIDATION_RUNBOOK.md)

---

## What changed (2026-07-02)

### 1. Deep re-audit P0–P2 (`af4b8c2`, pushed earlier)

- Frozen `RunSpec`, immutable HF revision pins, publication mode end-to-end
- McNemar/Holm fixes, CI workflow, 81 tests at time of commit
- Fresh archive: `outputs-hpc-2a100-main-2026-07-02-p0fix`; job **86229** submitted

### 2. Review hardening (this push)

- **`docs/ENV_VARS.md`** — central env reference; `.env.example` expanded
- **Archive blocking** — `INVALID_FOR_PUBLICATION.txt` + `QREASON_FORBIDDEN_ARCHIVE_PATTERNS`
- **`model_id`** on raw rows/summaries; QRM comparator + paper tables prefer it
- Publication git: clear error when Git missing

### 3. HPC operational fixes (this push)

| Fix | Effect |
|-----|--------|
| Code-path git only | Output manifest commits no longer block `score_run.py --publication` |
| Autopush opt-in | `QREASON_ENABLE_AUTOPUSH=1`; default off at submit |
| Locked manifest | `archive_manifest.py`; split cell jobs no longer race on `manifest.json` |
| Resume guard | Allow resume when HEAD moved but code unchanged |
| Submit script | Single `QREASON_OUTPUT_ROOT`; `--fresh`; default `all` → b01 only |
| QRM comparator | FP8 vs BF16 → SKIP; gptq3/qwen15b key fixes |
| Logprobs | `capture_logprobs` + `normalized_sequence_logprob` on raw rows |
| Pins | `requirements-hpc.txt` aligned to lock (transformers 5.12.1) |

---

## Sync rules (critical)

| Phase | HPC git action |
|-------|----------------|
| Job **86229 running** | **Do NOT** reset mid-run |
| **After inference completes** | `git fetch origin && git reset --hard origin/main` |
| **Before scoring** | Kill autopush: `tmux kill-session -t hpc_git_autopush 2>/dev/null || true` |
| **Then score** | Operational-fix commit required if job hit git/manifest gate failures |

MacBook push is inert for running Slurm jobs.

---

## b01 pass criteria (MATH-500 BF16, hard gate)

| Model | QRM ref | ±5 pp band |
|-------|---------|------------|
| Qwen-7B | 93.9% | 88.9–98.9% |
| Llama-8B | 91.0% | 86.0–96.0% |

Also check: `truncation_rate`, `completion_tokens_mean`, `decoding_repetition_penalty`, manual audit.

```bash
python scripts/score_run.py --publication --skip-calibration \
  --input "$ROOT/raw/level_a_bf16_seed0.jsonl" \
  --summary "$ROOT/results/level_a_bf16_seed0_summary.json"
python scripts/compare_qrm_baseline.py --summary "$ROOT/results/level_a_bf16_seed0_summary.json"
```

---

## Block status

| Block | Status |
|-------|--------|
| b01 | Job **86229** submitted |
| b02–b07 | Hold until b01 QRM gate + logprob GPU smoke |
| b08–b09 | Wired; optional |

Submit: `bash scripts/hpc/submit_hpc_blocks.sh b01` (not `all_blocks`).

---

## Hard gates before expanding

1. **b01 reproduction** — QRM MATH-500 bands + truncation/token checks
2. **Logprob GPU smoke** — 3-question run; confirm `confidence_source=normalized_sequence_logprob`
3. **3-seed pilot** — before full breadth grid
4. **LiveCodeBench** — wire or descope via amendment

---

## Pre-push verification (MacBook)

```bash
python -m pytest tests/test_publication_mode.py tests/test_archive_manifest.py \
  tests/test_resume_guard.py tests/test_checkpoint_utils.py \
  tests/test_logprob_confidence.py tests/test_sampling_params.py \
  tests/test_compare_qrm_baseline.py -q
python scripts/pin_hf_revisions.py --verify
python -m compileall -q src scripts tests
```

---

## Documentation index

| Doc | Purpose |
|-----|---------|
| [ENV_VARS.md](ENV_VARS.md) | All `QREASON_*` variables + submit behavior |
| [J1_VALIDATION_RUNBOOK.md](J1_VALIDATION_RUNBOOK.md) | Phases 0–7 |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Traps + 2026-07-02 fixes |
| [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md) | Architecture map |
