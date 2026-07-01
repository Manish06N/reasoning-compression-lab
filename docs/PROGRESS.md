# Progress — Paper 1 Experiments

**Last updated:** 2026-07-01  
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical log:** [progress.md](../progress.md)

---

## Summary (2026-07-01)

| Area | Status |
|------|--------|
| **Policy** | **HPC-only** for all publication experiments |
| **First HPC scores** | b01 BF16 anchors scored — **not publication-ready** (decode loops; 7% / 21% pass@1) |
| **MacBook** | Fixes + new tooling ready to **push** |
| **HPC next** | Sync → verify decoding → **rerun b01 fresh** (do not resume bad archive) |
| **5080** | **Retired** — partial archive only (`10/500` rows) |
| **Tests** | 17 pass on MacBook (`pytest tests/`) |

### Block status (archive `outputs-hpc-2a100-main-2026-06-29`)

| Block | Status | Notes |
|-------|--------|-------|
| b01 | Scored | Qwen 7% / Llama 21% — **rerun after push** |
| b02 | Partial | Qwen FP8 50/500 — discard or restart |
| b03–b07 | Not in archive | Queued / not started |
| b08–b09 | Wired | Qwen-1.5B on HPC when capacity allows |

---

## Pre-push checklist (MacBook)

```bash
python -m pytest tests/ -q
python scripts/verify_decoding_params.py    # VERIFY OK
git status && git diff --stat
# commit + push when ready
```

## Pre-rerun checklist (HPC)

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR && git fetch origin && git reset --hard origin/main
python scripts/verify_decoding_params.py
python scripts/hpc/07_preflight_publication.py
bash scripts/hpc/03_smoke_test.sh
# Fresh archive — do not resume outputs-hpc-2a100-main-2026-06-29 raw JSONL
export QREASON_OUTPUT_ROOT=$QR/outputs-hpc-2a100-main-$(date +%Y-%m-%d)-rerun
bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors
```

After first clean cell:

```bash
python scripts/compare_qrm_baseline.py --summary results/<cell>_summary.json
python scripts/build_paper_tables.py --archive $QREASON_OUTPUT_ROOT
```

---

## New tooling (2026-07-01)

| Script | When |
|--------|------|
| `verify_decoding_params.py` | Before every HPC rerun |
| `compare_qrm_baseline.py` | After scoring — pass@1 sanity band |
| `run_inference_multisample.py` | Level B maj@5 pilot |
| `build_pareto_frontier.py` | After multiple quants scored |
| `hpc/08_download_gptq4_models.sh` | Before b04 |

---

## Q1 publication analysis utilities

Post-run: `rescore_archive.py` → `build_paper_tables.py` → `build_repro_bundle.py`.  
See [progress.md](../progress.md) for full pipeline history.

---

## 5080 (historical — stopped 2026-06-28)

Partial run `outputs-win5080-main-2026-06-28/` — **10 rows**, not for paper.  
Reason: ~15 min/question → weeks for 4 cells. All work moved to HPC.

---

## Pending — HPC (PARAM Rudra)

Main grid via `bash scripts/hpc/submit_hpc_blocks.sh` or `run_hpc_2a100_publication.sh`.  
Guide: [BEGINNER_HPC_GUIDE.md](BEGINNER_HPC_GUIDE.md)
