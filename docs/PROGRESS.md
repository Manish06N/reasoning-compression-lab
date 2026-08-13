# Progress — Paper 1 Experiments

**Last updated:** 2026-08-13 (b02 FP8 submitted; official QRM parity completed)
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical log:** [progress.md](../progress.md) . **Audit:** [QRM_STACK_PARITY_AUDIT.md](QRM_STACK_PARITY_AUDIT.md) . **Ops:** [CHANGELOG.md](../CHANGELOG.md)

---

## Summary (2026-08-13)

| Area | Status |
|------|--------|
| **Active experiment** | **b02 FP8 deployment block** in fresh archive `outputs-hpc-2a100-main-2026-08-13` |
| **Qwen FP8** | Job **96086** - `level_b_qwen7b_fp8_math500_seed0`, running on `ragpu004`, 1x A100; passed FP8 model load with `kv_cache_dtype=auto` and started generation |
| **Llama FP8** | Job **96087** - `level_c_llama8b_fp8_math500_seed0`, pending/resources, 1x A100 |
| **b02 first attempt** | Jobs **96084/96085** failed before raw rows with `fp8_e5m2 kv-cache is not supported with fp8 checkpoints`; fixed in `542f622` by setting FP8 checkpoint KV cache to `auto` |
| **qreason stack** | vLLM **0.8.5**; this is the stack that looped/truncated on BF16 |
| **Official QRM parity** | Job **87302** completed under `qrm-official`: **10/10 correct**, **0 truncation** on Qwen-7B BF16 n=10 |
| **Path C archive** | `outputs-hpc-diag-pathc-2026-07-05` (~20 rows; kept for side-by-side stack-gap evidence) |
| **Git sync** | GitHub/HPC include the FP8 KV-cache fix (`542f622`); MacBook should pull latest `origin/main`; keep `.qrm_official_env_ready` untracked |
| **Calibration** | b02 auto-scoring uses `--skip-calibration`; valid for pass@1/truncation/cost, not Brier/AURC/ECE |

**Strategic label:** *b02 asks whether FP8 changes the modern-stack loop/truncation behavior. It is not a calibration run and not a QRM Table 1 reproduction claim.*

## Experiments A-D (diagnostic matrix)

| ID | Question | What ran | Status |
|----|----------|----------|--------|
| **A** | Does official QRM code score well on the same 10 problems? | `external/Quantized-Reasoning-Models/inference.py` (`qrm-official`) | **COMPLETED** - job 87302, 10/10 correct, 0 truncation |
| **B** | Did logprobs break our stack? | Our harness, `capture_logprobs: false` | Code fixed; not rerun |
| **C** | Does `repetition_penalty` explain failure? | Our harness, with vs without | **Answered** - both fail |
| **D** | Is 32k budget too tight? | Qwen 64k max_tokens | **Canceled** - not needed after A |

Plain English: [notes.md sections 31-34](../notes.md)

## Monitor b02

```bash
squeue -u $USER
tail -f logs/slurm/b02_parallel_fp8_level_b_qwen7b_fp8_math500_seed0_96086.out
tail -f logs/slurm/b02_parallel_fp8_level_c_llama8b_fp8_math500_seed0_96087.out
```

Do **not** submit b03/b04 until both b02 cells finish and pass@1, truncation, latency/VRAM, and cost-per-correct are reviewed against BF16 Path C/July numbers.

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