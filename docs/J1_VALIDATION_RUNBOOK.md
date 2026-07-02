# J1 validation runbook — prove the pipeline before expanding

**Status target:** Engineering MVP → **scientific validation pending** → pilot signal → publication draft.

**GitHub:** `286f5e4` or later for scoring gates.  
**Active queue (2026-07-01):** smoke job 86015 → b01 job 86016 (`afterok`).

Use this on HPC (login + captcha required — agent cannot SSH for you).

---

## Phase 0 — MacBook (before push)

```bash
cd "/Users/manish/Projects/2026/paper 1/reasoning-compression-lab"
python -m pytest tests/ -q
python scripts/verify_decoding_params.py
python scripts/validate_cell_matrix.py
git status
```

Push when tests pass. Push is **inert** for already-running Slurm jobs.

---

## Phase 1 — HPC sync and preflight (before inference)

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR

CONDA_ROOT=/home/apps/MSCC/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate qreason

git fetch origin && git reset --hard origin/main

python scripts/verify_decoding_params.py
python scripts/hpc/07_preflight_publication.py
```

**Smoke test:** must run on a **GPU node** (Slurm), not the login node — login shells lack CUDA (`libcuda.so.1 missing` is expected).

```bash
# Example: submit GPU smoke (do not run 03_smoke_test.sh on login node)
sbatch slurm/smoke_test.slurm
# Or chain: b01 with dependency afterok on smoke job id
```

---

## Phase 2 — Fresh archive (mandatory)

**Do not resume** `outputs-hpc-2a100-main-2026-06-29` — diagnostic only (decoding bug, ~7% pass@1).

Prefer **rename** over delete for diagnostic evidence:

```bash
if [ -d outputs-hpc-2a100-main-2026-06-29 ]; then
  mv outputs-hpc-2a100-main-2026-06-29 \
     outputs-hpc-2a100-main-2026-06-29-DIAGNOSTIC-INVALID
  echo "INVALID FOR PUBLICATION: decoding bug ~7% pass@1." \
    > outputs-hpc-2a100-main-2026-06-29-DIAGNOSTIC-INVALID/INVALID_FOR_PUBLICATION.txt
fi

RUN_TS=$(date +%Y%m%d-%H%M%S)
GIT_SHA=$(git rev-parse --short HEAD)

export QREASON_OUTPUT_ROOT="$QR/outputs-hpc-2a100-main-${RUN_TS}-${GIT_SHA}"
export QREASON_FRESH_RUN=1
mkdir -p "$QREASON_OUTPUT_ROOT"
```

---

## Phase 3 — b01 BF16 reproduction (Gate 1)

```bash
bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors
```

Cells:

- GPU 0: `level_a_bf16_seed0` (Qwen-7B BF16 MATH-500)
- GPU 1: `level_c_llama8b_bf16_math500_seed0` (Llama-8B BF16 MATH-500)

### Pre-score archive checks

```bash
ROOT="$QREASON_OUTPUT_ROOT"
find "$ROOT/raw" -name "*.jsonl" -ls
wc -l "$ROOT/raw/"*.jsonl
grep -c decoding_repetition_penalty "$ROOT/raw/level_a_bf16_seed0.jsonl"   # expect 500
head -n 1 "$ROOT/raw/level_a_bf16_seed0.jsonl" | python -m json.tool | head -30
```

### Sync at score time (NOT while job running)

```bash
cd $QR
source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh
conda activate qreason
git fetch origin && git reset --hard origin/main   # get amd-003+ baseline fix

# QREASON_OUTPUT_ROOT is empty in a fresh shell — discover archive explicitly:
ROOT=$(ls -dt "$QR"/outputs-hpc-2a100-main-* 2>/dev/null | head -1)
echo "Scoring: $ROOT"
# Eyeball: must be July rerun archive, NOT June-29 DIAGNOSTIC-INVALID
```

### Score both cells (pass@1 only — no calibration yet)

```bash
ROOT="$QREASON_OUTPUT_ROOT"

python scripts/score_run.py \
  --input "$ROOT/raw/level_a_bf16_seed0.jsonl" \
  --summary "$ROOT/results/level_a_bf16_seed0_summary.json" \
  --skip-calibration

python scripts/score_run.py \
  --input "$ROOT/raw/level_c_llama8b_bf16_math500_seed0.jsonl" \
  --summary "$ROOT/results/level_c_llama8b_bf16_math500_seed0_summary.json" \
  --skip-calibration
```

### QRM baseline gate

```bash
python scripts/compare_qrm_baseline.py \
  --summary "$ROOT/results/level_a_bf16_seed0_summary.json" 2>&1

python scripts/compare_qrm_baseline.py \
  --summary "$ROOT/results/level_c_llama8b_bf16_math500_seed0_summary.json" 2>&1
```

**Confirm stderr banner** shows:

- yaml path + **sha256**
- git commit (`286f5e4` or later)
- ref + band + source citation

**Pass criteria (MATH-500 BF16, `gate: hard`, reproduction profile):**

| Model | QRM ref | ±5 pp band | Source |
|-------|---------|------------|--------|
| Qwen-7B | 94.0% (DeepSeek 92.8) | **89.0–99.0%** | QRM Table 1 (Qwen-only) |
| Llama-8B | 91.0% (DeepSeek 89.1) | **86.0–96.0%** | QRM Appendix B Table 4 |

**Do NOT use ~45–65%** — that is AIME/GPQA scale, not MATH-500.

**GPQA (b07):** `gate: sanity` only — ±8 pp band, sober profile; comparator warns but does not exit 1 on pass@1 alone.

**Also required (not pass@1 alone):**

| Metric | Gate |
|--------|------|
| `truncation_rate` | ≤ 0.15 |
| `completion_tokens_mean` | ≥ 1000 (thousands — truncation smell if low) |
| `parse_failure_rate` | ≤ 0.10 |
| Raw rows | `decoding_repetition_penalty: 1.05` on every row |
| Manual audit | 20–50 traces reviewed |

b01 is **single-seed pass@1** vs QRM's **seeds 42–44 average** — expect ~1–2 pp sampling noise; band is the gate.

---

## Phase 4 — GPTQ-W4 reproduction

After BF16 passes all checks:

```bash
bash scripts/hpc/run_hpc_2a100_publication.sh b04_parallel_gptq4
```

Score with `--skip-calibration` until valid confidence exists.

```bash
python scripts/j1/compare_configs.py \
  --baseline "$ROOT/scored/level_a_bf16_seed0.jsonl" \
  --variant "$ROOT/scored/level_b_qwen7b_gptq4_math500_seed0.jsonl"
```

---

## Phase 5 — Manual extraction audit

```bash
python scripts/j1/sample_audit.py \
  --scored "$ROOT/scored/level_a_bf16_seed0.jsonl" \
  --output "$ROOT/metadata/audit_level_a_bf16_seed0.json" \
  --n 50
```

Fix extractor if needed; rescore without re-inference:

```bash
python scripts/rescore_archive.py --archive "$ROOT"
```

---

## Phase 6 — Valid calibration (before Brier/AURC claims)

**Hard gate before b02:** logprobs must be stored in raw JSONL (patch + smoke test).

Until then:

1. **maj@5 subset** (recommended):

```bash
python scripts/run_inference_multisample.py \
  --cell-config configs/cells/level_b_qwen7b_bf16_math500_seed0.json \
  --samples 5 --limit 100

python scripts/compute_calibration.py --input runs/raw/<multisample>.jsonl
```

2. Score with `--require-calibration` once rows carry valid `confidence_source`.

---

## Phase 7 — Three-seed pilot (after Phase 3–5 pass)

Recommended shape (includes GPTQ-3 at failure boundary):

- Qwen-7B × **{BF16, GPTQ-4, GPTQ-3}** × **{MATH-500, GPQA-Diamond}** × 3 seeds
- Use `scripts/j1/aggregate_seeds.py`

Do **not** launch full 300-cell grid until pilot shows a reliability signal.

---

## Task-specific baseline bands (reference)

From `configs/baselines/qrm_literature_targets.yaml`:

| Task | Typical BF16 scale | Example hard gate (Qwen-7B) |
|------|-------------------|----------------------------|
| MATH-500 | ~85–95% | 89.0–99.0% (QRM T1 ref 94.0 ±5pp) |
| GSM8K (b06) | ~85–92% | 86.0–96.0% (QRM T1 ref 91.0 ±5pp) |
| GPQA-D (b07) | ~45–55% | sanity ±8pp only (QRM T1 ref 51.0; sober profile) |

Never copy GPQA/AIME bands onto MATH-500.

---

## Status language

| Stage | Label |
|-------|--------|
| Now | **J1 engineering MVP complete; scientific validation pending** |
| After Phase 3 pass | **Reproduction gate passed (BF16)** |
| After Phase 4–5 | **Extraction gate passed; GPTQ repro validated** |
| After Phase 6–7 | **Pilot signal; calibration endpoints valid** |

---

## Quick checks

```bash
squeue -u $USER
sacct -j 86015,86016 --format=JobID,JobName,State,ExitCode,Elapsed
ls -la "$QREASON_OUTPUT_ROOT/raw/"
python scripts/validate_cell_matrix.py
```

See also: [KNOWN_ISSUES.md](KNOWN_ISSUES.md), [HPC_2A100_PLAN.md](HPC_2A100_PLAN.md), [PROGRESS.md](PROGRESS.md).
