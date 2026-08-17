# Progress — Paper 1 Experiments

**Last updated:** 2026-08-17 (major revision frozen at `d707e44` on `paper-major-revision`)
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical log:** [progress.md](../progress.md) · **publication decision:** [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md) · **canonical paper:** [paper/main.tex](../paper/main.tex) / 20-page [paper/main.pdf](../paper/main.pdf) · **ops:** [CHANGELOG.md](../CHANGELOG.md)

> **Freeze (2026-08-17):** Science is frozen on `paper-major-revision` (`d707e44`, pushed). Next: visual PDF QA, then independent referee review. GPU work is closed. Do not merge to `main`. The 2026-08-16 table below is historical chronology, not current execution authority.

---

## Summary (2026-08-16 — historical)

| Area | Status |
|------|--------|
| **Publication verdict** | **Historical:** then labeled ArXiv-ready pending PDF review. **Now:** freeze `d707e44`; visual PDF QA remaining. |
| **MATH-500 Confirmatory** | Qwen-7B: BF16 94.00% ± 0.55%, FP8 94.40% ± 1.05%, AWQ-4 93.12% ± 0.67%, GPTQ-4 93.48% ± 0.77%; Llama-8B: BF16 89.24% ± 0.74%, FP8 89.52% ± 1.01%, AWQ-4 86.48% ± 1.96%, GPTQ-4 88.92% ± 1.55% |
| **Phase 5 Statistical Analysis** | *(historical 2026-08-16 row — do not cite)* maj@5 McNemar vs BF16 non-significant; gold-hit ECE and $65$ tok/s $C_{\mathrm{pass}}$ are retracted. Use `paper/main.tex`. |
| **GSM8K Breadth Grid** | 24/24 complete. Qwen-7B: BF16 91.26%, FP8 91.33%, AWQ-4 91.05%, GPTQ-4 91.13%; Llama-8B: BF16 88.68%, FP8 88.80%, AWQ-4 87.11%, GPTQ-4 88.96% |
| **GPQA-Diamond Breadth Grid** | 24/24 complete. Qwen: BF16 50.34%, FP8 49.49%, AWQ-4 44.78%, GPTQ-4 47.98%; Llama: BF16 46.13%, FP8 47.81%, AWQ-4 46.97%, GPTQ-4 44.95% |
| **Hardware Compliance** | Campaign finished. Do not submit extra GPU jobs unless asked. |
| **Campaign archives** | `outputs-hpc-campaign-2026-08-14/` (MATH-500), GSM8K/GPQA archives, and 88 JSON files in `results/` |

**Strategic label:** Matched BF16/FP8/AWQ-4/GPTQ-4 grid on pinned vLLM 0.7.0 eager is complete (56,408 completions). See [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md) (2026-08-17 freeze) and [paper/main.tex](../paper/main.tex).

## Experiments A-D (diagnostic matrix)

| ID | Question | What ran | Status |
|----|----------|----------|--------|
| **A** | Does official QRM code score well on the same 10 problems? | `external/Quantized-Reasoning-Models/inference.py` (`qrm-official`) | **COMPLETED** - job 87302, 10/10 correct, 0 truncation |
| **B** | Did logprobs break our stack? | Our harness, `capture_logprobs: false` | Code fixed; not rerun |
| **C** | Does `repetition_penalty` explain failure? | Our harness, with vs without | **Answered** - both fail |
| **D** | Is 32k budget too tight? | Qwen 64k max_tokens | **Canceled** - not needed after A |

Plain English: [archive/notes_2026-07-03.md](archive/notes_2026-07-03.md)

## Next gated action

Visual QA of the 20-page `paper/main.pdf`, then independent referee review. Upload `paper/arxiv_source.zip` to arXiv only when asked. Do not launch new GPU jobs. Do not merge to `main`. The 2026-08-14 recovery Phase 0 freeze is historical.

---

## Path C (canceled — historical)

| Wave | Job | Result |
|------|-----|--------|
| d01 Qwen 32k | 87116 | **CANCELED** ~20/50 rows |
| d01 Llama 32k | 87117 | **CANCELED** ~20/50 rows |
| d02 Qwen 64k | 87118 | **CANCELED** (never started) |

Partial scored: Qwen 10% pass@1, 90% trunc; Llama 15%, 75% trunc (n=20).

---

## b01 July archive (gate failed)

| Cell | Result |
|------|--------|
| Llama BF16 | 500/500 — pass@1 **19.6%**, trunc **58%**, `sober` prompt |
| Qwen BF16 | 410/500 — trunc **~94%** (90 rows skipped) |

Archive: `outputs-hpc-2a100-main-2026-07-03`
