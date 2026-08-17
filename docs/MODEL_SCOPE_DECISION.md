# Model Scope Decision — Paper 1 (J1)

**Status:** Model-family scope remains frozen; experimental protocol revised 2026-08-14; GPU closed 2026-08-17
**Roadmap:** PhD V8.2 §6, scale gate §6.9  
**Companion:** [MODEL_ROSTER.md](MODEL_ROSTER.md) (HF IDs, paths, env vars)

**Current authority:** [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md) (2026-08-17 freeze). The [2026-08-14 recovery plan](plans/2026-08-14-publication-recovery.md) is historical. The old seed-0 expansion order is superseded.

This document records **what models are in scope, out of scope, and gated for later** for Paper 1. It exists to prevent scope creep from model-inventory research (6×5 quant matrices, Qwen3, base Llama, 70B BF16 grids, etc.).

---

## Paper 1 question (why the scope is narrow)

> When **the same** distilled reasoning models are compressed, do calibration, selective-risk ordering, seed stability, and cost-per-correct change even when pass@1 looks stable?

Paper 1 measures **compression under deployment**, not a reasoning-model leaderboard. Model families are controls, not the novelty headline.

---

## In scope (frozen)

### Model families — DeepSeek-R1-Distill only

| Tier | Model | Role | Publication machine |
|------|-------|------|-------------------|
| **Core** | DeepSeek-R1-Distill-Qwen-7B | Primary anchor; QRM reproduction | HPC 2× A100 |
| **Core** | DeepSeek-R1-Distill-Llama-8B | Cross-architecture control (Qwen vs Llama) | HPC 2× A100 |
| **Scale** | DeepSeek-R1-Distill-Qwen-1.5B | Size interaction; pipeline verifier | HPC (b08–b09); 5080 debug only |

No other model families in the main J1 grid.

### Quantization formats — five configs

| Config ID | Method | Canonical HF source (see MODEL_ROSTER) |
|-----------|--------|----------------------------------------|
| `bf16` | Full precision baseline | `deepseek-ai/DeepSeek-R1-Distill-*` |
| `fp8` | FP8 checkpoint; A100 executes weight-only Marlin fallback | `RedHatAI/*-FP8-dynamic` |
| `awq4` | AWQ W4G128 | `jakiAJK/*_AWQ` or `casperhansen/*-awq` per roster |
| `gptq4` | GPTQ W4G128 (QRM-aligned) | `ruikangliu/*-GPTQ-W4G128` |
| `gptq3` | Bit-width stress | `irish-quant/*-3bit` (Qwen-7B); roster for others |

**Not adding** W8A8 INT8, NVFP4, or a sixth quant axis before b01–b09 seed-0 completes.

### Tasks

| Task | Dataset | Priority |
|------|---------|----------|
| MATH-500 | `HuggingFaceH4/MATH-500` | **First** — reproduction + core grid |
| GPQA-Diamond | `Idavidrein/gpqa` (gated) | After HF access gate |
| GSM8K | `openai/gsm8k` (`validation` split) | Control / breadth |
| LiveCodeBench | — | **Deferred** — not wired in harness yet |

### Execution levels

| Level | Scope | Gate |
|-------|-------|------|
| **A** | Qwen-7B × {BF16, GPTQ-4} × MATH-500 × repro seeds | Reproduction gate |
| **B** | Qwen-7B + Llama-8B × {BF16, FP8, AWQ4, GPTQ4} × MATH-500 × seeds 42–44 | Discriminating pilot after recovery P0/P1 |
| **C** | Five-seed headline cells plus gated breadth | Only after contribution selection |

### Hardware policy

| Machine | J1 role |
|---------|---------|
| **HPC 2× A100 80 GB** | **All publication numbers** — blocks b01–b09 |
| **RTX 5080 16 GB** | **Retired for publication** — smoke/debug only if used |
| **MacBook** | Code, configs, docs — no GPU paper runs |

### Tensor parallelism (2× A100)

| Model size | BF16 TP | Quantized TP |
|------------|---------|--------------|
| 1.5B, 7B, 8B, 14B (if gated) | 1 | 1 |
| 32B (if gated) | 1 (fits) or 2 | 1 |
| 70B (if gated) | **2 only** — tight at 32k context | 1 |

---

## Out of scope (do not add to J1 grid)

| Item | Reason |
|------|--------|
| **Qwen2.5-Instruct / Qwen3 / base Llama 3.1** | Different training, templates, thinking format — confounds compression |
| **Full DeepSeek-R1 / R1-Zero (671B MoE)** | Not runnable on 2× A100; wrong paper question |
| **6-size × 5-quant master matrix** (1.5B–70B all formats) | Compute explosion; does not sharpen J1 claim |
| **Llama-70B BF16 @ 32k context** | ~79 GB/GPU weights + KV — failure mode, not baseline |
| **GGUF / llama.cpp checkpoints** | Wrong stack for J1 vLLM grid (J3 local transfer only) |
| **Uncensored / random community forks** | Breaks reproducibility and QRM comparability |
| **Mixing two GPTQ-4 HF repos** under one `gptq4` label | Invalid paired comparison |
| **5080 full MATH-500 for 7B/8B BF16** | OOM; not for manuscript |
| **W8A8 as parallel sixth format** | Optional systems note only — not before core grid done |

---

## Gated for later (explicit stop rules)

Add nothing below until **V8.2 scale gate** passes: core 7B/8B results stable and publication story visible.

| Extension | Trigger | Maximum form |
|-----------|---------|--------------|
| **Qwen-14B** | 7B/8B seed-0 scored; clear trends | 1 model × {BF16, GPTQ-4} × MATH-500 × 3 seeds |
| **Qwen-32B** | 14B validation informative + compute budget | BF16 TP=1 or 2; **not** full 5-quant grid |
| **Llama-70B** | Explicit extension paragraph only | **GPTQ-4 or AWQ-4 only** on 1× A100; no BF16 32k |
| **One Qwen3-14B cell** | Novelty re-check + supervisor sign-off | Single “current-model” validation run |
| **Five-seed confirmation** | Three-seed pilot selects a contribution | **Key cells only:** Qwen-7B + Llama-8B × {BF16, FP8, AWQ-4, GPTQ-4} × MATH-500 × seeds 42–46 |
| **LiveCodeBench** | Extraction gate passed on math/code | Fixed repo date; subset only |
| **W8A8 INT8** | FP8/A100 behavior documented | Optional appendix column — not headline |

**Hard stops (V8.2):** No 32B/70B scale expansion before core story stable. No new model family before J1 submission.

---

## Compute priority (when queue is limited)

Run in this order; stop expanding if lower tiers are not scored.

1. Recovery Phase 0: reproducibility, schema, validators, telemetry, tests.
2. Qwen-7B + Llama-8B × {BF16, FP8} × MATH-500 × seed 42.
3. Same models × {BF16, FP8, AWQ-4, GPTQ-4} × MATH-500 × seeds 42–44.
4. Contribution-selection gate and supervisor approval.
5. Extend headline cells to seeds 42–46.
6. Gated GPQA/GSM8K breadth.
7. GPTQ-3/scale/extensions only if the selected story requires them.

---

## FP8 on A100 (resolved policy)

External notes disagree on Ampere FP8 support. **This project uses:**

- Checkpoints: `RedHatAI/*-FP8-dynamic`
- vLLM: `quantization: compressed-tensors` (W8A16-style on Ampere)
- Config: see `configs/models/deepseek_r1_qwen_7b_fp8.json`

**Policy:** Keep FP8 in the grid (QRM-aligned). If A100 FP8 cells behave anomalously, report as a **stack/format finding** — do not swap in five new model families.

---

## Change control

To add a model or quant format to J1:

1. Update this file with rationale and gate trigger.  
2. Update [MODEL_ROSTER.md](MODEL_ROSTER.md) with single canonical HF ID.  
3. Add `configs/models/*.json` + cell config(s).  
4. Supervisor sign-off if beyond “gated for later” table.  
5. Record in [CHANGELOG.md](../CHANGELOG.md).

**Default answer to “should we add model X?”** → No, unless it fits a row in **Gated for later** and the scale gate has passed.

---

## Quick reference

| Question | Answer |
|----------|--------|
| Keep DeepSeek-R1-Distill only? | **Yes** |
| Add Qwen3 / Llama 3.1 base? | **No** (J1) |
| Run 70B BF16 on 2× A100? | **No** |
| Minimum pilot set? | Qwen-7B + Llama-8B × {BF16, GPTQ-4, AWQ-4, FP8} × MATH-500 × seeds 42–44 |
| Minimum publishable evidence? | Pilot plus selected five-seed headline cells, valid reliability/calibration/systems fields, paired analysis, and clean artifacts |
| Where are HF IDs? | [MODEL_ROSTER.md](MODEL_ROSTER.md) |
| Where is full codebase map? | [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md) |
