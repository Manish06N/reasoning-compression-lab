# Progress — Paper 1 Experiments

**Last updated:** 2026-07-06 (Experiment A — env ready, job queued)
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical log:** [progress.md](../progress.md) · **Audit:** [QRM_STACK_PARITY_AUDIT.md](QRM_STACK_PARITY_AUDIT.md) · **Ops:** [CHANGELOG.md](../CHANGELOG.md)

---

## Summary (2026-07-05)

| Area | Status |
|------|--------|
| **Active experiment** | **Experiment A** — official QRM `inference.py` (job **87216**, `PENDING`) |
| **qrm-official env** | **Ready** — marker `2026-07-06-conda-gcc12-nvcc124-vllm070wheel` |
| **Path C** | **CANCELED** (87116–87118) — n=20 sufficient |
| **Path C archive** | `outputs-hpc-diag-pathc-2026-07-05` (~20 rows; kept for side-by-side) |
| **Our harness signal** | Qwen **10%** / **90%** trunc — protocol OK, **stack gap** (`qreason` env) |
| **Official test** | n=10, seed=42, Qwen-7B, env **`qrm-official`** (not `qreason`), output `outputs-hpc-qrm-official-2026-07-06/` |
| **Quant grid** | **On hold** until Experiment A result |

**Strategic label:** *Run authors' code (A) to decide stack vs config — then document honestly for Paper 1.*

---

## Experiments A–D (diagnostic matrix)

| ID | Question | What runs | Status |
|----|----------|-----------|--------|
| **A** | Does **official QRM code** score well on same 10 problems? | `external/Quantized-Reasoning-Models/inference.py` (`qrm-official`) | **QUEUED** — 87216 (exclusive GPU) |
| **B** | Did **logprobs** break our stack? | Our harness, `capture_logprobs: false` | Code fixed; **not rerun** |
| **C** | Does **repetition_penalty** explain failure? | Our harness, with vs without | **Answered** — both fail |
| **D** | Is **32k budget** too tight? | Qwen 64k max_tokens | **Canceled** (87118) |

Plain English: [notes.md §31](../notes.md)

---

## One repo, two envs

| Env | Use for |
|-----|---------|
| **`qreason`** | Main harness — `run_inference.py`, b01–b09, smoke, Path C |
| **`qrm-official`** | Experiment A only — authors' `inference.py`, vLLM 0.7.0 fork |

Same repo: `reasoning-compression-lab`. Authors' code: `external/Quantized-Reasoning-Models/`.  
**Do not** install QRM packages into `qreason` or our vLLM 0.8.5 into `qrm-official`.

Troubleshooting: [QRM_OFFICIAL_HPC_TROUBLESHOOTING.md](QRM_OFFICIAL_HPC_TROUBLESHOOTING.md)

---

## Monitor Experiment A

```bash
squeue -j 87216
tail -f logs/qrm_official_87216.out
# after finish:
python scripts/hpc/qrm_parity/compare_side_by_side.py --limit 10
```

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