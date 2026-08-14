# Progress — Paper 1 Experiments

**Last updated:** 2026-08-14 (full FP8 result audited; publication recovery required)
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical log:** [progress.md](../progress.md) · **publication decision:** [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md) · **execution:** [plans/2026-08-14-publication-recovery.md](plans/2026-08-14-publication-recovery.md) · **ops:** [CHANGELOG.md](../CHANGELOG.md)

---

## Summary (2026-08-14)

| Area | Status |
|------|--------|
| **Publication verdict** | **Needs revision** — current result is appendix/control evidence only |
| **Latest gate** | Full exact-stack FP8 replication completed and audited; matched quantization comparison absent |
| **Validation jobs** | **96093** Qwen FP8: 10/10 correct, 0 cap hits; **96094** Llama FP8: 10/10 correct, 0 cap hits |
| **Validation archive** | `outputs-hpc-qrm-official-fp8-validation-2026-08-13` |
| **Completed full jobs** | **96100** Qwen: 472/500 (**94.4%**); **96101** Llama: 445/500 (**89.0%**); both seed 42 |
| **Full archive** | `outputs-hpc-qrm-official-fp8-full-2026-08-13` |
| **Interpretation** | Compatible with public FP8 references; no same-stack BF16, multi-seed, calibration, or controlled systems evidence |
| **Trace/provenance** | Six likely near-cap traces; phrase loops underdetected; output lacks `finish_reason`/token IDs; external QRM requires uncommitted patches |
| **Stopped b02** | Jobs **96086/96087** canceled after Qwen's first 10 rows showed 2/10 correct, 8/10 truncation, and repetition loops |
| **b02 first attempt** | Jobs **96084/96085** failed before raw rows with `fp8_e5m2 kv-cache is not supported with fp8 checkpoints`; fixed in `542f622` by setting FP8 checkpoint KV cache to `auto` |
| **V0 probe** | Jobs **96091/96092** showed that `VLLM_USE_V1=0` alone is insufficient: malformed answers and a 32768-token repetition loop remained |
| **qreason stack** | vLLM **0.8.5** + transformers **5.12.1**; both V1 and V0 probes failed output-quality checks |
| **Official QRM parity** | Job **87302** completed under `qrm-official`: **10/10 correct**, **0 truncation** on Qwen-7B BF16 n=10 |
| **Path C archive** | `outputs-hpc-diag-pathc-2026-07-05` (~20 rows; kept for side-by-side stack-gap evidence) |
| **Repository state** | Baseline `4796614` is pushed; current audit/plan/docs are uncommitted on HPC; keep `.qrm_official_env_ready` untracked |
| **Calibration/systems** | `--skip-calibration` supports diagnostic correctness/trace scoring only; no Brier/AURC/ECE or controlled cost/performance claim |

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
