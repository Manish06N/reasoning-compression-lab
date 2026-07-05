# Progress — Paper 1 Experiments

**Last updated:** 2026-07-05 (Path C launch)  
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical log:** [progress.md](../progress.md) · **Session notes:** [notes.md](../notes.md) · **Ops:** [CHANGELOG.md](../CHANGELOG.md)

---

## Summary (2026-07-05)

| Area | Status |
|------|--------|
| **Active experiment** | **Path C diagnostic** (strict QRM repro, n=50) |
| **Archive** | `outputs-hpc-diag-pathc-2026-07-05` |
| **Jobs** | **87116** Qwen 32k RUNNING · **87117** Llama 32k RUNNING · **87118** Qwen 64k PENDING |
| **b01 QRM gate** | **FAILED** (July archive — see below) |
| **Quant grid b02–b06** | **On hold** until Path C results |

**Strategic label:** *b01 gate failed → Path C diagnostic sprint to test strict QRM protocol before further GPU spend.*

---

## Path C diagnostic (active)

| Wave | Job | Cell | Settings |
|------|-----|------|----------|
| d01 | 87116 | `diag_qwen7b_bf16_math500_seed42_n50` | 32k, seed 42, reproduction, no rep_pen, eager |
| d01 | 87117 | `diag_llama8b_bf16_math500_seed42_n50` | same |
| d02 | 87118 | `diag_qwen7b_bf16_math500_seed42_n50_64k` | 64k output cap, n=50 |

```bash
# Submit (already done)
bash scripts/hpc/submit_pathc_diagnostic.sh

# Monitor
squeue -u $USER

# Report when complete
bash scripts/hpc/report_pathc_diagnostic.sh
```

**Pass heuristic (n=50):** 32k pass@1 ≥ ~80% and truncation ≤ ~25% → stack repro OK.

---

## b01 July archive (complete / failed gate)

| Cell | Result |
|------|--------|
| Llama BF16 | 500/500 — pass@1 **19.6%**, trunc **58%**, `sober` prompt |
| Qwen BF16 | 410/500 — trunc **~94%** (90 rows not run) |

Archive: `outputs-hpc-2a100-main-2026-07-03`

---

## Historical entries

Earlier snapshots remain in [progress.md](../progress.md) and [CHANGELOG.md](../CHANGELOG.md).