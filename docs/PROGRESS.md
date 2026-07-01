# Progress — Paper 1 Experiments

**Last updated:** 2026-07-01 (evening)  
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical log:** [progress.md](../progress.md) · **Ops detail:** [CHANGELOG.md](../CHANGELOG.md)

---

## Summary

| Area | Status |
|------|--------|
| **J1 engineering** | **MVP complete** — pipeline, stats, provenance, fail-closed calibration |
| **J1 scientific validation** | **Pending** — fresh b01 BF16 + GPTQ repro on HPC |
| **Minimum publishable grid** | **15 cells wired** (seed 0, b01–b09) — see `papers/j1/publication_matrix.yaml` |
| **Full Level C (300 cells)** | **Not generated** — gated after pilot signal |
| **Policy** | **HPC-only** for J1 paper numbers; RTX for J3 transfer only |
| **First HPC scores** | b01 archive **diagnostic only** (7% / 21% — decode bug) |
| **Blocker** | Fresh b01 rerun — see [J1_VALIDATION_RUNBOOK.md](J1_VALIDATION_RUNBOOK.md) |

**Read first if anything looks wrong:** [KNOWN_ISSUES.md](KNOWN_ISSUES.md)

---

## Pre-push (MacBook)

```bash
python -m pytest tests/ -q
python scripts/verify_decoding_params.py
python scripts/hpc/07_preflight_publication.py   # optional on Mac if models absent
```

## Pre-rerun (HPC) — required

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR && git fetch origin && git reset --hard origin/main

python scripts/verify_decoding_params.py
python scripts/hpc/07_preflight_publication.py
bash scripts/hpc/03_smoke_test.sh

# CRITICAL: do not resume bad archive
rm -rf outputs-hpc-2a100-main-2026-06-29
export QREASON_OUTPUT_ROOT=$QR/outputs-hpc-2a100-main-$(date +%Y-%m-%d)-rerun
mkdir -p "$QREASON_OUTPUT_ROOT"
export QREASON_FRESH_RUN=1

bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors
```

After first clean cell:

```bash
python scripts/compare_qrm_baseline.py --summary $QREASON_OUTPUT_ROOT/results/<cell>_summary.json
python scripts/j1/compare_configs.py --baseline ... --variant ...
python scripts/build_paper_tables.py --archive $QREASON_OUTPUT_ROOT
python scripts/build_dashboard.py --archive $QREASON_OUTPUT_ROOT
```

---

## Block status (old archive — do not use for paper)

Archive `outputs-hpc-2a100-main-2026-06-29`:

| Block | Status | Notes |
|-------|--------|-------|
| b01 | Scored | Qwen 7% / Llama 21% — **invalid for publication** |
| b02 | Partial | Qwen FP8 50/500 |
| b03–b07 | Not in archive | |
| b08–b09 | Wired | Qwen-1.5B when capacity allows |

---

## Documentation index

| Doc | Purpose |
|-----|---------|
| [REPO_MAP.md](REPO_MAP.md) | Where everything lives |
| [V8_2_ARCHITECTURE.md](V8_2_ARCHITECTURE.md) | V8.2 module layout |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Critical bugs and traps |
| [BEGINNER_HPC_GUIDE.md](BEGINNER_HPC_GUIDE.md) | Full HPC walkthrough |
| [plans/2026-07-01-v82-reengineering.md](plans/2026-07-01-v82-reengineering.md) | V8.2 checklist |

---

## New tooling (2026-07-01)

| Script | When |
|--------|------|
| `verify_decoding_params.py` | Before every HPC rerun |
| `compare_qrm_baseline.py` | After scoring — pass@1 sanity |
| `scripts/j1/compare_configs.py` | Paired McNemar vs BF16 |
| `scripts/j1/sample_audit.py` | Extraction audit gate |
| `scripts/j1/aggregate_seeds.py` | QRM-style mean±std |
| `run_inference_multisample.py` | maj@5 calibration |
| `build_pareto_frontier.py` | Cost frontier figures |
| `build_dashboard.py` | HTML summary dashboard |
| `export_parquet.py` | Parquet analysis export |
| `j2/run_method_pilot.py` | Paper 2 pilot manifest |
| `j3/preflight_indic.py` | Paper 3 preflight |

---

## 5080 (historical — stopped 2026-06-28)

Partial archive only — not for paper. See [progress.md](../progress.md).

---

## HPC (PARAM Rudra)

Main grid: `bash scripts/hpc/submit_hpc_blocks.sh`  
Guide: [BEGINNER_HPC_GUIDE.md](BEGINNER_HPC_GUIDE.md)
