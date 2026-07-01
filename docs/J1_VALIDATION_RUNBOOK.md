# J1 validation runbook — prove the pipeline before expanding

**Status target:** Engineering MVP → **scientific validation pending** → pilot signal → publication draft.

Use this after MacBook push; run commands on HPC (login + captcha required — agent cannot SSH for you).

---

## Phase 0 — MacBook (before push)

```bash
cd "/Users/manish/Projects/2026/paper 1/reasoning-compression-lab"
python -m pytest tests/ -q
python scripts/verify_decoding_params.py
python scripts/validate_cell_matrix.py
git status
```

Push when tests pass and you are ready for HPC to pull.

---

## Phase 1 — HPC sync and preflight

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR

CONDA_ROOT=/home/apps/MSCC/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate qreason

git fetch origin && git reset --hard origin/main

python scripts/verify_decoding_params.py
python scripts/hpc/07_preflight_publication.py
bash scripts/hpc/03_smoke_test.sh
```

Smoke must produce valid JSONL under `runs/raw/` (or block output with exit 75 if GPU busy).

---

## Phase 2 — Fresh archive (mandatory)

**Do not resume** `outputs-hpc-2a100-main-2026-06-29` — diagnostic only (decoding bug).

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR

rm -rf outputs-hpc-2a100-main-2026-06-29

export QREASON_OUTPUT_ROOT=$QR/outputs-hpc-2a100-main-$(date +%Y-%m-%d)-rerun
export QREASON_FRESH_RUN=1
mkdir -p "$QREASON_OUTPUT_ROOT"
```

---

## Phase 3 — b01 BF16 reproduction (Gate 1 numbers)

```bash
bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors
```

Cells:

- GPU 0: `level_a_bf16_seed0` (Qwen-7B BF16 MATH-500)
- GPU 1: `level_c_llama8b_bf16_math500_seed0` (Llama-8B BF16 MATH-500)

### Score (pass@1 only — calibration skipped until maj@5)

```bash
export QR=/scratch/$USER/reasoning-compression-lab
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

### Sanity vs QRM literature

```bash
python scripts/compare_qrm_baseline.py \
  --summary "$ROOT/results/level_a_bf16_seed0_summary.json"
```

**Pass criterion:** pass@1 within ±0.05 of QRM band (see `configs/baselines/qrm_literature_targets.yaml`).

---

## Phase 4 — GPTQ-W4 reproduction

After BF16 is explainably close, run GPTQ block (when weights downloaded):

```bash
bash scripts/hpc/run_hpc_2a100_publication.sh b04_parallel_gptq4
```

Score with `--skip-calibration` until valid confidence exists.

Compare BF16 vs GPTQ:

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

Review failures/truncations; fix extractor if needed; **rescore** without re-inference:

```bash
python scripts/rescore_archive.py --archive "$ROOT"
```

---

## Phase 6 — Valid calibration (before Brier/AURC claims)

Default `score_run.py` **refuses** to emit publication calibration without a valid confidence source.

Options:

1. **maj@5 subset** (recommended first):

```bash
python scripts/run_inference_multisample.py \
  --cell-config configs/cells/level_b_qwen7b_bf16_math500_seed0.json \
  --samples 5 --limit 100

python scripts/compute_calibration.py --input runs/raw/<multisample>.jsonl
```

2. Then score with `--require-calibration` once rows carry `confidence` + valid `confidence_source`.

---

## Phase 7 — Three-seed pilot (after Phase 3–5 pass)

Only if BF16/GPTQ repro and audit are acceptable:

- Qwen-7B × {BF16, FP8, GPTQ-4} × MATH-500 × seeds {0, 1, 2}
- Use `scripts/j1/aggregate_seeds.py`

Do **not** launch full 300-cell grid until pilot shows a reliability signal.

---

## Status language (use in reports)

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
ls -la "$QREASON_OUTPUT_ROOT/raw/"
ls -la "$QREASON_OUTPUT_ROOT/results/"
python scripts/validate_cell_matrix.py
```

See also: [KNOWN_ISSUES.md](KNOWN_ISSUES.md), [HPC_2A100_PLAN.md](HPC_2A100_PLAN.md), [PROGRESS.md](PROGRESS.md).
