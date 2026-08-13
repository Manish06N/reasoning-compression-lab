# Progress — Paper 1 Experiments

**Last updated:** 2026-08-13 (bad b02 stopped; exact official-stack FP8 validation passed)
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical log:** [progress.md](../progress.md) . **Audit:** [QRM_STACK_PARITY_AUDIT.md](QRM_STACK_PARITY_AUDIT.md) . **Ops:** [CHANGELOG.md](../CHANGELOG.md)

---

## Summary (2026-08-13)

| Area | Status |
|------|--------|
| **Latest gate** | n=10 FP8 validation on the exact successful `qrm-official` stack: **PASSED** |
| **Validation jobs** | **96093** Qwen FP8: 10/10 correct, 0 cap hits; **96094** Llama FP8: 10/10 correct, 0 cap hits |
| **Validation archive** | `outputs-hpc-qrm-official-fp8-validation-2026-08-13` |
| **Stopped b02** | Jobs **96086/96087** canceled after Qwen's first 10 rows showed 2/10 correct, 8/10 truncation, and repetition loops |
| **b02 first attempt** | Jobs **96084/96085** failed before raw rows with `fp8_e5m2 kv-cache is not supported with fp8 checkpoints`; fixed in `542f622` by setting FP8 checkpoint KV cache to `auto` |
| **V0 probe** | Jobs **96091/96092** showed that `VLLM_USE_V1=0` alone is insufficient: malformed answers and a 32768-token repetition loop remained |
| **qreason stack** | vLLM **0.8.5** + transformers **5.12.1**; both V1 and V0 probes failed output-quality checks |
| **Official QRM parity** | Job **87302** completed under `qrm-official`: **10/10 correct**, **0 truncation** on Qwen-7B BF16 n=10 |
| **Path C archive** | `outputs-hpc-diag-pathc-2026-07-05` (~20 rows; kept for side-by-side stack-gap evidence) |
| **Git sync** | GitHub/HPC include the FP8 KV-cache fix (`542f622`); MacBook should pull latest `origin/main`; keep `.qrm_official_env_ready` untracked |
| **Calibration** | b02 auto-scoring uses `--skip-calibration`; valid for pass@1/truncation/cost, not Brier/AURC/ECE |

**Strategic label:** *The modern-stack b02 run is stopped. Jobs 96093/96094 prove both FP8 checkpoints generate healthy answers on the known-good official stack. A full run may now be submitted only through the strict pilot gate.* Exact settings and failure reasons are recorded in [the canonical progress log](../progress.md#2026-08-13-run-diagnosis-what-worked-what-did-not-and-why).

## Experiments A-D (diagnostic matrix)

| ID | Question | What ran | Status |
|----|----------|----------|--------|
| **A** | Does official QRM code score well on the same 10 problems? | `external/Quantized-Reasoning-Models/inference.py` (`qrm-official`) | **COMPLETED** - job 87302, 10/10 correct, 0 truncation |
| **B** | Did logprobs break our stack? | Our harness, `capture_logprobs: false` | Code fixed; not rerun |
| **C** | Does `repetition_penalty` explain failure? | Our harness, with vs without | **Answered** - both fail |
| **D** | Is 32k budget too tight? | Qwen 64k max_tokens | **Canceled** - not needed after A |

Plain English: [notes.md sections 31-34](../notes.md)

## Next gated action

```bash
bash scripts/hpc/submit_qrm_fp8_full.sh
```

The submitter revalidates both n=10 outputs before calling `sbatch`. Do not submit b03/b04 until the full FP8 correctness results are reviewed. The official path does not provide all main-harness deployment telemetry.

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
