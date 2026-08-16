# Progress — Paper 1 Experiments

**Last updated:** 2026-08-16 (Phase 4/5 completed, GSM8K completed, GPQA-Diamond breadth running)
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical log:** [progress.md](../progress.md) · **publication decision:** [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md) · **working paper:** [paper/main.md](../paper/main.md) · **ops:** [CHANGELOG.md](../CHANGELOG.md)

---

## Summary (2026-08-16)

| Area | Status |
|------|--------|
| **Publication verdict** | **Publication Ready (Manuscript in Progress)** — Confirmatory 40-cell MATH-500 grid ($n=500$, seeds 42–46) and 24-cell GSM8K grid ($n=1,319$, seeds 42–44) 100% completed with zero degenerations |
| **MATH-500 Confirmatory** | Qwen-7B: BF16 94.00% ± 0.55%, FP8 94.40% ± 1.05%, AWQ-4 93.12% ± 0.67%, GPTQ-4 93.48% ± 0.77%; Llama-8B: BF16 89.24% ± 0.74%, FP8 89.52% ± 1.01%, AWQ-4 86.48% ± 1.96%, GPTQ-4 88.92% ± 1.55% |
| **Phase 5 Statistical Analysis** | Paired McNemar tests show no significant discordance vs BF16 ($p > 0.05$); ECE $\le 0.034$; AURC $\le 0.0054$; FP8 established as Pareto-optimal Cost-of-Pass ($C_{\text{pass}}$) frontier |
| **GSM8K Breadth Grid** | Qwen-7B: BF16 91.26%, FP8 91.33%, AWQ-4 91.05%, GPTQ-4 91.13%; Llama-8B: BF16 88.68%, FP8 88.80%, AWQ-4 87.11%, GPTQ-4 88.96% (all 24 cells completed) |
| **GPQA-Diamond Breadth Grid** | 24 cells ($n=198$, seeds 42–44) actively running under 24/7 daemon `gpqa_daemon` in tmux session |
| **Hardware Compliance** | Exactly 2 GPUs concurrently (1 Qwen + 1 Llama), 100% compliant with `QOSMaxGRESPerUser` |
| **Campaign archives** | `outputs-hpc-campaign-2026-08-14/` (MATH-500) and `outputs-hpc-breadth-gsm8k-2026-08-15/` (GSM8K) backed up to project `archive/` and persistent home `/home/manishn_iitp/archive/` |

**Strategic label:** *The FP8 checkpoints are healthy and their completed exact-stack results reproduce known accuracy, but the package is not publication-ready. Broad experiments are blocked until recovery Phase 0 repairs reproducibility and observability.* Exact settings and claims boundaries are in [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md).

## Experiments A-D (diagnostic matrix)

| ID | Question | What ran | Status |
|----|----------|----------|--------|
| **A** | Does official QRM code score well on the same 10 problems? | `external/Quantized-Reasoning-Models/inference.py` (`qrm-official`) | **COMPLETED** - job 87302, 10/10 correct, 0 truncation |
| **B** | Did logprobs break our stack? | Our harness, `capture_logprobs: false` | Code fixed; not rerun |
| **C** | Does `repetition_penalty` explain failure? | Our harness, with vs without | **Answered** - both fail |
| **D** | Is 32k budget too tight? | Qwen 64k max_tokens | **Canceled** - not needed after A |

Plain English: [notes.md sections 31-36](../notes.md)

## Next gated action

Complete recovery Phase 0 from the [current plan](plans/2026-08-14-publication-recovery.md): tracked dependency patches, clean recreation, complete finish/token/timing/provenance fields, pathology validation, telemetry, and tests. Then run tiny smoke cells. Do not submit b03/b04 or a broad grid.

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
