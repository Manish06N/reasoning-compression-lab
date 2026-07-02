# Progress — Paper 1 Experiments

**Last updated:** 2026-07-01 (late evening)  
**GitHub `main`:** `286f5e4`  
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical log:** [progress.md](../progress.md) · **Ops detail:** [CHANGELOG.md](../CHANGELOG.md)

---

## Summary

| Area | Status |
|------|--------|
| **J1 engineering** | **MVP complete** — pipeline, stats, provenance, fail-closed calibration, matrix validator |
| **J1 scientific validation** | **In flight** — Slurm 86015 (smoke) → 86016 (b01 BF16) |
| **QRM baseline gates** | **Fixed** (`286f5e4`) — task-specific bands; old MATH-500 45–65% was wrong |
| **15 core validation cells** | Wired seed 0 (b01–b09) — `papers/j1/publication_matrix.yaml` |
| **Full Level C (300 cells)** | Not generated — gated after pilot signal |
| **Policy** | HPC-only for J1 numbers; RTX for J3 transfer only |
| **June-29 archive** | Diagnostic only (7% / 21%) — do not cite |
| **Tests** | 43 pass on MacBook (`pytest tests/`) |

**Status label:** *J1 engineering MVP complete; scientific validation pending.*

**Read first:** [KNOWN_ISSUES.md](KNOWN_ISSUES.md) · **Runbook:** [J1_VALIDATION_RUNBOOK.md](J1_VALIDATION_RUNBOOK.md)

---

## What changed (2026-07-01)

### 1. Fail-closed calibration (`8fb0fb0`)

- `score_run.py` no longer uses parse success as publication confidence.
- Flags: `--skip-calibration` (b01 repro), `--require-calibration` (analysis gate).
- New: `src/evaluation/calibration/confidence.py`, `scripts/validate_cell_matrix.py`.

### 2. QRM baseline band fix (`286f5e4`)

- **Bug:** MATH-500 gates used ~45–65% (AIME/GPQA scale) → false-pass at ~60%, false-fail at ~93%.
- **Fix:** Full audit of `configs/baselines/qrm_literature_targets.yaml` for MATH-500, GSM8K, GPQA-D, 1.5B.
- **Comparator:** `compare_qrm_baseline.py` prints yaml sha256, git commit, ref, band, source.
- **Protocol:** `papers/j1/amendments.yaml` amd-002.

### 3. HPC queue

- Cancelled stale job 86010.
- GPU smoke 86015 → b01 86016 (`afterok`).
- Login-node smoke fails (no CUDA) — Slurm path is correct.

---

## Sync rules (critical)

| Phase | HPC git action |
|-------|----------------|
| Jobs 86015/86016 **running** | **Do NOT** reset — job uses launch-time tree |
| **After inference completes** | `git fetch origin && git reset --hard origin/main` |
| **Then score** | Must be on `286f5e4+` for correct baseline yaml |

MacBook push is inert for running Slurm jobs.

---

## b01 pass criteria (MATH-500 BF16, hard gate)

| Model | QRM ref | ±5 pp band | Source |
|-------|---------|------------|--------|
| Qwen-7B | 93.9% | 88.9–98.9% | QRM Table 1 p.119 (BF16) |
| Llama-8B | 91.0% | 86.0–96.0% | QRM Appendix B Table 4 |

**Not sufficient alone:** also check `truncation_rate`, `completion_tokens_mean` (thousands), `decoding_repetition_penalty` in rows, manual audit.

**Wrong gates to ignore:** ~45–65% (AIME-scale), ~7% (June-29 decode bug).

---

## Pre-push (MacBook)

```bash
python -m pytest tests/ -q
python scripts/validate_cell_matrix.py
python scripts/verify_decoding_params.py
```

## Score-time (HPC — after 86016)

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh
conda activate qreason
git fetch origin && git reset --hard origin/main

ROOT=$(ls -dt "$QR"/outputs-hpc-2a100-main-* 2>/dev/null | grep -v DIAGNOSTIC | head -1)
echo "Scoring: $ROOT"
if [[ -z "$ROOT" || "$ROOT" == *"2026-06-29"* || "$ROOT" == *DIAGNOSTIC* ]]; then
  echo "ERROR: no valid rerun archive — abort." >&2; exit 1
fi
```

python scripts/score_run.py \
  --input "$ROOT/raw/level_a_bf16_seed0.jsonl" \
  --summary "$ROOT/results/level_a_bf16_seed0_summary.json" \
  --skip-calibration

python scripts/score_run.py \
  --input "$ROOT/raw/level_c_llama8b_bf16_math500_seed0.jsonl" \
  --summary "$ROOT/results/level_c_llama8b_bf16_math500_seed0_summary.json" \
  --skip-calibration

python scripts/compare_qrm_baseline.py \
  --summary "$ROOT/results/level_a_bf16_seed0_summary.json" 2>&1

python scripts/compare_qrm_baseline.py \
  --summary "$ROOT/results/level_c_llama8b_bf16_math500_seed0_summary.json" 2>&1
```

Verify stderr provenance banner shows yaml sha256 and git commit.

---

## Block status

| Block | Status |
|-------|--------|
| b01 | Queued (86016) |
| b02–b07 | Hold until b01 + logprob gate |
| b08–b09 | Wired; optional |

Old archive `outputs-hpc-2a100-main-2026-06-29`: Qwen 7% / Llama 21% — **invalid**.

---

## Hard gates before expanding

1. **b01 reproduction** — pass QRM MATH-500 bands + truncation/token checks.
2. **Logprobs in raw JSONL** — before b02 (avoid 5× maj@5 everywhere).
3. **3-seed pilot** — Qwen-7B × {BF16, GPTQ-4, GPTQ-3} × {MATH-500, GPQA-D} before full breadth.
4. **LiveCodeBench** — wire or descope via protocol amendment.

---

## Documentation index

| Doc | Purpose |
|-----|---------|
| [J1_VALIDATION_RUNBOOK.md](J1_VALIDATION_RUNBOOK.md) | Phases 0–7 |
| [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md) | Full architecture |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Traps (§8 baseline bands) |
| [HARDWARE_POLICY.md](HARDWARE_POLICY.md) | HPC vs RTX roles |
| [MODEL_SCOPE_DECISION.md](MODEL_SCOPE_DECISION.md) | Frozen model scope |
| [configs/baselines/qrm_literature_targets.yaml](../configs/baselines/qrm_literature_targets.yaml) | QRM sanity bands |

---

## Tooling reference

| Script | When |
|--------|------|
| `verify_decoding_params.py` | Before every HPC run |
| `validate_cell_matrix.py` | After cell config changes |
| `compare_qrm_baseline.py` | After scoring — check provenance banner |
| `score_run.py --skip-calibration` | b01 accuracy repro |
| `scripts/j1/sample_audit.py` | 50-sample extraction gate |
| `run_inference_multisample.py` | maj@5 calibration (after repro) |

---

## HPC (PARAM Rudra)

Guide: [BEGINNER_HPC_GUIDE.md](BEGINNER_HPC_GUIDE.md)  
Submit: `bash scripts/hpc/submit_hpc_blocks.sh`
