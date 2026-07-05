# Progress — Paper 1 Experiments

**Last updated:** 2026-07-05 (Path C + stack parity audit)  
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical log:** [progress.md](../progress.md) · **Audit:** [QRM_STACK_PARITY_AUDIT.md](QRM_STACK_PARITY_AUDIT.md) · **Ops:** [CHANGELOG.md](../CHANGELOG.md)

---

## Summary (2026-07-05)

| Area | Status |
|------|--------|
| **Active experiment** | **Path C diagnostic** (strict QRM repro, n=50) |
| **Early signal (n=20)** | Qwen **10%** / **90%** trunc; Llama **15%** / **75%** trunc — **not QRM reproduction** |
| **Audit** | Protocol verified in raw JSONL; **stack gap** (vLLM 0.8.5 loops) — see [QRM_STACK_PARITY_AUDIT.md](QRM_STACK_PARITY_AUDIT.md) |
| **Parity fixes** | `vllm_serving.py`, d03 parity pilot ready |
| **Jobs** | **87116–87117** RUNNING · **87118** PENDING |
| **Quant grid** | **On hold** until Path C + parity pilot |

**Strategic label:** *Protocol correct, stack not equivalent — document gap honestly; Paper 1 leads with truncation + cost.*

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
**Current trajectory:** failing heuristic → run **d03 parity pilot** + official QRM `inference.py` cross-check.

```bash
python scripts/hpc/qrm_parity/verify_stack_parity.py
bash scripts/hpc/submit_pathc_parity_pilot.sh
```

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