# Progress — Paper 1 Experiments

**Last updated:** 2026-08-14 (full FP8 result audited; publication recovery required)
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical log:** [progress.md](../progress.md) · **publication decision:** [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md) · **execution:** [plans/2026-08-14-publication-recovery.md](plans/2026-08-14-publication-recovery.md) · **ops:** [CHANGELOG.md](../CHANGELOG.md)

---

## Summary (2026-08-14)

| Area | Status |
|------|--------|
| **Publication verdict** | **Active Execution** — Matched Phase 1/2 publication campaign actively running across 2 A100 channels |
| **Latest gate** | Phase 1 Headline matched controls and Phase 2 quantization pilots active in `outputs-hpc-campaign-2026-08-14` |
| **Completed matched jobs** | **96237** Qwen BF16 (472/500, **94.4%**); **96238** Qwen FP8 (472/500, **94.4%**); **96240** Qwen GPTQ-4 (469/500, **93.8%**) |
| **Active jobs** | **96289** Qwen AWQ-4 (running, `float16` fix); **96247** Llama FP8 (running, ~60% complete) |
| **Campaign archive** | `outputs-hpc-campaign-2026-08-14` |
| **Pipeline Daemon** | Upgraded `queue_manager_daemon.py` running in tmux session `campaign_daemon` managing continuous 24/7 SLURM chaining |
| **Validation jobs** | **96093** Qwen FP8: 10/10 correct; **96094** Llama FP8: 10/10 correct |
| **Official QRM parity** | Job **87302** completed under `qrm-official`: **10/10 correct**, **0 truncation** on Qwen-7B BF16 n=10 |

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
