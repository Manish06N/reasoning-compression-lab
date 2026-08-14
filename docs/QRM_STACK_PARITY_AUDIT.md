# QRM Stack Parity Audit (updated 2026-08-14)

**Purpose:** Explain why Path C strict QRM protocol reproduces the *failure mode* but not QRM Table 1 numbers, what we verified, what we fixed, and what experiments come next.

**Paper 1 is not a QRM reproduction paper.** QRM (Liu et al., COLM 2025) is a **sanity baseline**. This audit documents an honest reproduction attempt and isolates the stack gap.

**Final full-run decision:** jobs 96100/96101 completed at 94.4% Qwen and 89.0% Llama. These values validate the checkpoints/official path and reproduce existing FP8 references. They do not establish a quantization effect or a causal stack effect. See [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md).

---

## 1. The story in one paragraph

We launched Path C because July b01 failed the QRM hard gate (Llama 19.6% pass@1, Qwen ~94% truncation) even after fixing YAML passthrough and `repetition_penalty`. Path C pinned **every protocol knob** QRM documents: `reproduction` prompt (`qrm_math500.txt`), temp 0.6, top_p 0.95, max_tokens 32768, seed 42, **no** `repetition_penalty`, `enforce_eager=true`. Early n=20 results still show **~10–15% pass@1** and **75–90% truncation** with obvious degeneration loops (`yeah yeah`, `the the the`). Raw JSONL proves the protocol is correct; the model burns the full 32k budget before emitting `\boxed{}`. When generation stops cleanly, pass@1 on the tiny non-truncated subset is much higher (Qwen 2/2, Llama 3/5). **Conclusion:** config parity is achieved; **inference-stack parity is not** — our vLLM 0.8.5 V1 + transformers 5.12.1 path behaves differently from QRM's Lighteval + older stack.

---

## 2. What QRM paper §3.1 says

From COLM 2025 (extract in `docs/literature/paper1/ALL_PAPERS_MERGED.md`):

- Harness: **Lighteval** with **vLLM** backend
- Sampling: temperature **0.6**, top-p **0.95**
- Max generation tokens: **32,768**
- Seeds: **three** (42, 43, 44) averaged to reduce variance
- MATH-500 prompt: `{problem}\n\nPlease reason step by step, and put your final answer within \boxed{}.`

Official repo: [Quantized-Reasoning-Models/inference.py](https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/main/inference.py)

| Parameter | QRM `inference.py` |
|-----------|-------------------|
| temperature | 0.6 |
| top_p | 0.95 |
| max_new_tokens | 32768 |
| max_model_length | 32768 |
| seed | 42 (default) |
| repetition_penalty | **not set** |
| enforce_eager | True |
| gpu_memory_utilization | 0.9 |
| enable_prefix_caching | False |
| enable_chunked_prefill | False |
| use_chat_template | True |
| Scorer | Lighteval `latex_gold_metric` |

QRM targets (`configs/baselines/qrm_literature_targets.yaml`):

| Model | MATH-500 pass@1 | truncation max |
|-------|-----------------|----------------|
| Qwen-7B BF16 | 93.9% ± 0.7% | ≤ 15% |
| Llama-8B BF16 | 91.0% ± 1.1% | ≤ 15% |

---

## 3. What we verified in Path C (pre-parity fix)

Archive: `outputs-hpc-diag-pathc-2026-07-05`  
Jobs: 87116 (Qwen 32k), 87117 (Llama 32k), 87118 (Qwen 64k, pending)

### Protocol fields (raw JSONL row 1 — all correct)

| Field | Observed |
|-------|----------|
| prompt_profile | `reproduction` |
| prompt_template_file | `prompts/qrm_math500.txt` |
| decoding_config | `configs/decoding/repro_qrm_strict.yaml` |
| temperature / top_p / max_tokens | 0.6 / 0.95 / 32768 |
| repetition_penalty | null |
| seed | 42 |
| max_model_len | 32768 |
| enforce_eager | true |

### Early scored results (n=20)

| Cell | pass@1 | trunc | parse_fail | mean completion tok |
|------|--------|-------|------------|---------------------|
| Qwen 32k | 10% | 90% | 90% | ~29,531 |
| Llama 32k | 15% | 75% | 80% | ~27,454 |

### Degeneration patterns

- Truncated rows: **zero** `\boxed{}` on Qwen — parse failure = never answered
- Loop tails: `yeah yeah`, `0 Wait`, `Which which`, `the the the`
- Clean finishes: Qwen **2/2** correct; Llama **3/5** correct

### First 10 problem IDs (side-by-side baseline)

```bash
python scripts/hpc/qrm_parity/compare_side_by_side.py --limit 10
```

Typical row: `test/precalculus/807.json` → trunc=True, 32,699 tokens, `yeah-loop`.

---

## 4. Root-cause ranking

### Tier 1 — Very likely

1. **vLLM 0.8.5 V1 + transformers 5.12.1 vs QRM stack**  
   Same nominal decoding; different loop/stop behavior. QRM pins `transformers==4.47.1`; we pin `5.12.1`.

2. **R1 degeneration loops exhausting 32k**  
   Primary failure mode. QRM also omits `repetition_penalty` but their stack apparently avoids catastrophic looping.

3. **Tight shared budget** (`max_model_len == max_tokens == 32768`)  
   Prompt + completion share one ceiling; loops consume entire budget.

### Tier 2 — Plausible

4. **String prompts vs QRM `prompt_token_ids`** (Lighteval tokenizes first)  
5. **Always-on `logprobs=1`** in our harness (QRM only when `returns_logits`) — **fixed 2026-07-05**  
6. **LLM engine `seed=None`** in logs while SamplingParams had seed=42 — **fixed 2026-07-05**  
7. **Serving flags not wired** (`gpu_memory_utilization`, prefix caching, chunked prefill) — **fixed 2026-07-05**

### Tier 3 — Unlikely primary cause

8. Scorer (`math_verify` vs `latex_gold_metric`) — most failures lack `\boxed{}`  
9. Single seed / n=50 pilot — explains variance, not 80pp gap  
10. July Llama `sober` prompt — ruled out for Path C (uses `reproduction`)

---

## 5. Fixes applied (2026-07-05 evening)

### Code

| File | Change |
|------|--------|
| `src/runners/vllm_serving.py` | **NEW** — `build_llm_init_kwargs()` merges QRM serving defaults |
| `src/runners/vllm_runner.py` | `build_llm(..., seed=cell_seed)` passes engine seed + serving flags |
| `scripts/run_inference.py` | Forwards `cell["seed"]` to `build_llm` |
| `configs/decoding/repro_qrm_strict.yaml` | `capture_logprobs: false` |
| `configs/models/*_qrm_strict.json` | `gpu_memory_utilization`, `enable_prefix_caching`, `enable_chunked_prefill` |

### Tooling

| Script | Role |
|--------|------|
| `scripts/hpc/qrm_parity/verify_stack_parity.py` | No-GPU checklist (must print `Overall: PASS`) |
| `scripts/hpc/qrm_parity/compare_side_by_side.py` | Compare traces on first N MATH-500 IDs |
| `scripts/hpc/qrm_parity/setup_official_qrm_repo.sh` | Clone QRM repo to `external/Quantized-Reasoning-Models` |
| `scripts/hpc/submit_pathc_parity_pilot.sh` | Submit n=10 Qwen parity rerun (Experiment B — optional) |
| `scripts/hpc/submit_qrm_official_test.sh` | Submit **Experiment A** (job 87130) |
| `slurm/qrm_official_math500_n10.slurm` | SLURM wrapper for official `inference.py` |
| `configs/cells/diag_qwen7b_bf16_math500_seed42_n10_parity.json` | Parity pilot cell |
| `configs/machine_split/hpc_blocks/d03_pathc_parity_pilot.sh` | SLURM block |

### Verify (login node)

```bash
python scripts/hpc/qrm_parity/verify_stack_parity.py
python -m pytest tests/test_vllm_serving.py -q
```

### Parity pilot (GPU)

```bash
bash scripts/hpc/submit_pathc_parity_pilot.sh
# after job:
python scripts/hpc/qrm_parity/compare_side_by_side.py \
  --parity-archive outputs-hpc-diag-pathc-parity-$(date +%Y-%m-%d) --limit 10
```

### Experiment A - official QRM cross-check (**COMPLETED**, job 87302)

```bash
tail -n 120 logs/qrm_official_87302.out
python scripts/hpc/qrm_parity/compare_side_by_side.py --limit 10
```

Output: `outputs-hpc-qrm-official-2026-07-06/`

**Result:** official QRM got 10/10 correct with 0 truncation on the same first 10 MATH-500 items. Our Path C comparison on the modern `qreason` stack was 1/10 with 90% truncation. Later modern-stack FP8 jobs 96086/96087 were also unhealthy and canceled; exact-official-stack FP8 pilots 96093/96094 then passed 10/10 for both models.

---

## 6. Experiments A-D and full FP8 follow-up (status 2026-08-14)

| ID | Tests | Stack | Status |
|----|-------|-------|--------|
| **A** | Official `inference.py` on 10 MATH-500 IDs | QRM Lighteval + QRM vLLM | **COMPLETED** - 87302, 10/10 correct, 0 truncation |
| **B** | `capture_logprobs: false` on our harness | Our vLLM 0.8.5 | Code fixed; **not rerun** |
| **C** | `repetition_penalty` none vs 1.05 | Our harness | **Answered** — both fail (~90% trunc) |
| **D** | Qwen 64k max_tokens | Our harness | **Canceled** (87118) |
| **FP8 full** | Exact-stack Qwen/Llama FP8 n=500, seed 42 | QRM stack | **COMPLETED** — 94.4% / 89.0%; replication only |

Path C (our strict QRM protocol, n=50) was **canceled** at n=20 — sufficient to justify A.

Plain English: [notes.md sections 31-35](../notes.md)

---

## 7. Decision tree (current)

```
Path C canceled at n=20
└─ Experiment A COMPLETED successfully (Job 87302)
    └─ QRM got 10/10 (100% correct, 0% loops) vs. our stack 1/10 (10% correct, 90% loops)
        └─ Modern-stack FP8 96086/96087 failed output health and was stopped
            └─ Exact-stack FP8 96093/96094 passed
                └─ Gated full correctness jobs 96100/96101 completed
                    └─ Accuracy replicated; publication audit found unmatched design and observability gaps
                        └─ Recovery Phase 0 before any broad grid
```

---

## 8. Paper 1 claims boundary after full audit

- Supported: the pinned QRM path produces healthy FP8 output and reproduces known MATH-500 accuracy.
- Supported: the modern and pinned paths exhibit a stack-sensitive behavior gap worth controlled study.
- Not supported: FP8-vs-BF16 effects, native FP8 performance, calibration, cost, or causal stack attribution.
- Candidate contribution: matched reliability–cost effects under quantization and controlled stack shift, selected only after the three-seed pilot.

The old broad “beyond accuracy” statement is a motivation, not a completed contribution.

---

## 9. References

- `configs/baselines/qrm_literature_targets.yaml`
- `notes.md` §18–19, §29–31
- `CHANGELOG.md` 2026-07-05 parity entry
- `progress.md` current snapshot
- External: `external/Quantized-Reasoning-Models/` (cloned via setup script)
