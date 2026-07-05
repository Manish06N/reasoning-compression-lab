# Progress — Paper 1 Experiments

**Last updated:** 2026-07-05 (evening)  
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical log:** [progress.md](../progress.md) · **Session notes:** [notes.md](../notes.md) · **Ops:** [CHANGELOG.md](../CHANGELOG.md)

---

## Summary (2026-07-05)

| Area | Status |
|------|--------|
| **b01 QRM gate** | **FAILED** — pivot to Paper 1 deployment narrative |
| **Llama BF16** | **500/500 scored** — 19.6% pass@1, 58% truncation, sober prompt |
| **Qwen BF16** | **410/500** — 94% truncation; **90-row finish optional, not required** |
| **Quant grid b02–b06** | **Not started** — recommended next step |
| **Queue** | Empty |

**Strategic label:** *Gate failed; first trustworthy BF16 deployment metrics obtained; proceed to quantization grid.*

**Read:** [notes.md §28](../notes.md) (pivot + literature map) · [J1_VALIDATION_RUNBOOK.md](J1_VALIDATION_RUNBOOK.md)

---

## b01 gate result

| Cell | pass@1 | truncation | Gate |
|------|--------|------------|------|
| Llama BF16 (`sober`) | 19.6% | 58% | FAIL (profile + metrics) |
| Qwen BF16 (`reproduction`, n=410) | TBD | ~94% | FAIL (truncation) |

Checker: `compare_qrm_baseline.py` — Llama returns `SKIP` (prompt_profile mismatch).

---

## Next steps (approved)

1. Reframe Paper 1 — truncation + cost-per-correct as main findings.
2. Submit **b02 FP8** one block (same 32k protocol).
3. Skip Qwen 90 rows unless table symmetry needed.
4. Fix Llama b01 cell prompt profile for future repro attempts.

---

## Historical entries

Earlier snapshots (2026-07-02 split b01, P0 audit, etc.) remain in [progress.md](../progress.md) below the 2026-07-05 section.