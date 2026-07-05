# Session Notes — 2026-07-03

Personal backup of findings, decisions, and learnings from the July 3 working session, cross-checked against `CHANGELOG.md`, `progress.md`, live HPC state, and `docs/literature/`.

**Repo:** `/scratch/manishn_iitp/reasoning-compression-lab`  
**GitHub:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical ops log:** `CHANGELOG.md` (detailed job history)  
**Dated progress:** `progress.md` (update after material changes)

---

## Executive summary (read this first)

### Match QRM 100%?

| Question | Answer |
|----------|--------|
| Match QRM everywhere? | **No** — Paper 1 is not a QRM reproduction paper |
| Match QRM for b01 gate? | **Yes** — on params that change generations (32k, prompt, temp, top_p, seeds 42–44) |
| Match QRM for quant grid (b02–b06)? | **No** — use fixed **32k** + labeled **deployment protocol** (repetition_penalty OK) |

**Bottom line:** QRM is a **sanity baseline** before opening the quant grid. The thesis contribution is pass@1 **plus** truncation, calibration, and cost — not hitting 93.9% in every table.

### Two protocols at a glance

| | **Protocol A — `qrm_repro`** | **Protocol B — `our_hpc_deployment`** |
|---|---|---|
| **When** | b01 hard gate; claim “reproduced Table 1” | Main grid b02–b06; Paper 1 tables |
| **32k `max_tokens`** | Yes | Yes (non-negotiable across quants) |
| **Prompt** | `qrm_math500.txt` (= QRM GitHub) | Same file, `reproduction` profile |
| **Seeds** | **42, 43, 44** (mean ± std) | **0** for grid; more only for variance section |
| **repetition_penalty** | **None** (match QRM) | **1.05** — label as vLLM anti-loop default |
| **enforce_eager** | true preferred | false on A100 (document in methods) |
| **Truncation** | Gate metric (≤15%) | **First-class paper metric** beside pass@1 |

**Never** mix Protocol A and B rows in one table without a `protocol` column.

### Current run label

Jobs **86757** / **86758** = **Protocol B pilot, seed 0**. Let finish; do **not** cancel for strict QRM mid-run. If gate fails after score → see decision tree in **§19**.

### Verified good news (QRM GitHub audit)

MATH-500 **prompt text matches** QRM `reasoning.py` exactly. Decoding (0.6 / 0.95 / 32768) matches `inference.py`. Real gaps: **seed 0 vs 42–44**, **repetition_penalty 1.05**, **enforce_eager false**, possible **scorer** diff (Lighteval vs math_verify).

---

## 1. What this project is (one paragraph)

**Paper 1** (*Beyond Accuracy: Reliability, Calibration, Seed Variance, and Cost-per-Correct of Quantized Reasoning LLMs*) studies whether compressed reasoning models stay **trustworthy and economical** under real serving — not just whether a single accuracy number drops.

The harness runs DeepSeek-R1 distill models (Qwen-7B, Llama-8B, later 1.5B) through vLLM on PARAM Rudra, compares BF16 / FP8 / AWQ / GPTQ on MATH-500 (and later GSM8K, GPQA), and records pass@1, **truncation_rate**, latency, VRAM, and cost-per-correct.

**Thesis spine (PhD roadmap):** Reliable and cost-efficient deployment of reasoning LLMs under compression, evaluation, and multilingual constraints. See `docs/PHD_ROADMAP.md`, `docs/PAPER1_DESIGN.md`.

---

## 2. What is QRM? (baseline paper — not our paper)

| Field | Value |
|-------|--------|
| **Shorthand** | QRM = Quantized Reasoning Models |
| **Title** | *Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models* |
| **Authors** | Ruikang Liu et al. |
| **Venue** | COLM 2025 |
| **arXiv** | [2504.04823](https://arxiv.org/abs/2504.04823) |
| **Code** | [github.com/ruikangliu/Quantized-Reasoning-Models](https://github.com/ruikangliu/Quantized-Reasoning-Models) |
| **In literature bundle** | `docs/literature/paper1/ALL_PAPERS_MERGED.md` (~p. 11438+) |

**QRM’s question:** Does quantization hurt **reasoning accuracy**?  
**Our Paper 1 question:** Is **accuracy alone** enough for deployment (calibration, cost, truncation, seeds)?

**QRM evaluation setup (§3.1 — from their paper text):**

- **Lighteval** + **vLLM** backend  
- temperature **0.6**, top-p **0.95**  
- max generation **32,768** tokens  
- **3 seeds** averaged (42–44)  
- Benchmarks: AIME-120, MATH-500, GSM8K, GPQA-Diamond, LiveCodeBench  

**Key reference numbers (our gates):**

| Model | Task | QRM reference | Our tolerance |
|-------|------|---------------|---------------|
| Qwen-7B BF16 | MATH-500 | **93.9±0.7%** (Table 1) | ±5 pp |
| Llama-8B BF16 | MATH-500 | **91.0±1.1%** (Appendix Table 4) | ±5 pp |
| Qwen-7B BF16 | truncation | ≤ **15%** (our gate) | `qrm_literature_targets.yaml` |

**Repo mapping:**

| File | Role |
|------|------|
| `configs/baselines/qrm_literature_targets.yaml` | Reference bands |
| `configs/decoding/repro_qrm.yaml` | Decoding protocol |
| `prompts/qrm_math500.txt` | Reproduction prompt |
| `scripts/compare_qrm_baseline.py` | Hard gate checker |
| `docs/BEGINNER_HPC_GUIDE.md` §1.1–1.2 | Beginner QRM + repro gap explainer |

**Level A / b01:** Sanity check that our stack can run QRM’s protocol before opening the quant grid (b02–b06). **Not** the thesis contribution itself.

---

## 3. Literature archive (pulled 2026-07-03, commit `9912d7d`)

| Path | Contents |
|------|----------|
| `docs/literature/paper1/ALL_PAPERS_MERGED.md` | ~1049 pp text extract — Paper 1 bundle (calibration, QRM, sober, cost, abstention, …) |
| `docs/literature/paper1/ALL_PAPERS_MERGED.pdf` | **Gitignored** (116 MB > GitHub limit) — keep on MacBook / `~/Downloads/merged for paper 1/` |
| `docs/literature/paper2/ALL_PAPERS_MERGED.pdf` | Paper 2 bundle (~165 pp, GPTQ, speculative decoding refs) |
| `docs/literature/PAPER1_READING_MAP.md` | Reading order and paper groups |
| `docs/literature/README.md` | PDF vs `.md` usage |

**Paper 1 bundle groups:**

1. **QRM** — baseline accuracy / quantization  
2. **GPTQ, AWQ, SmoothQuant** — methods we run (b02–b05)  
3. **Calibration papers** — our novelty (beyond accuracy)  
4. **A Sober Look** — seed variance / reproducibility  
5. **Cost-of-Pass, OckBench** — cost-per-correct  
6. **AbstentionBench** — selective prediction  

**Week 1 reading order:** QRM → Sober Look → Calibrating LLMs with Sample Consistency → Cost-of-Pass → GPTQ.

---

## 4. Campaign timeline (June 26 – July 3)

### Act 1 — June 29: pipeline OK, science weak

| Item | Detail |
|------|--------|
| Smoke | Job 85306 PASSED |
| b01 | Job 85394 on `ragpu008`, 2-GPU block, ~309/500 Qwen + ~381/500 Llama at Codex check |
| Speed | ~**7 min/problem** (~420 s) |
| Scored archive | `outputs-hpc-2a100-main-2026-06-29` |

| Cell | pass@1 | Truncation | Parse fail |
|------|--------|------------|------------|
| Qwen-7B BF16 | 7.0% | ~90% | ~86% |
| Llama-8B BF16 | 21.4% | ~59% | ~60% |

**Diagnosis:** Models hit **32k token cap** before `\boxed{}` → scored wrong.  
**Bug:** `repetition_penalty: 1.05` in YAML **never reached vLLM** (fixed 2026-07-01).  
**Verdict:** Infrastructure worked; **do not cite** this archive in the manuscript.

### Act 2 — July 1–3: over-correction

| Attempt | Result |
|---------|--------|
| **1M context** | BF16 KV OOM at init (job 86703, ~56 GiB KV needed) |
| **131k context** (`a3414a4`) | Loaded but ~36+ min on Q1, 0 rows (job 86743) |
| **9-cell parallel queue** | Violated b01 gate; 0/4500 rows for days |
| **`--exclusive` on 1-GPU split cells** | QOS trap — counts as 2 GPUs on ragpu nodes (`7448164`) |
| Strict git gate | Jobs died mid-iteration |
| Many output roots (`-queued`, `-attempt`, …) | Resume / manifest confusion |

**Discard all 131k and 1M inference rows** — wrong `config_hash`, not protocol-compliant.

### Act 3 — July 3 afternoon: protocol reset (`729d773`)

| Setting | Active b01 value |
|---------|----------------|
| `max_tokens` | **32768** (QRM protocol) |
| `max_model_len` | **40960** (KV headroom, not 131k) |
| `enforce_eager` | **false** (BF16 on A100) |
| `repetition_penalty` | **1.05** (must reach vLLM) |
| Submit | Split 1-GPU cells, **`QREASON_SLURM_EXCLUSIVE=0`** |
| Archive | `outputs-hpc-2a100-main-2026-07-03` |
| Jobs | **86757** (Qwen), **86758** (Llama) |

**Also fixed:** logprob double-count, lock-file delete breaking flock, `max_tokens` in verify_decoding.

---

## 5. Why ~7% pass@1 vs QRM ~94%? (reproduction gap)

**June 7% is NOT valid QRM reproduction** — truncation + config bugs.

| Factor | Effect |
|--------|--------|
| **~90% truncated** | No `\boxed{}` → `correct=0` for most of 500 |
| **repetition_penalty bug** | More looping → more cap hits |
| **Same nominal 32k as QRM** | Budget matches on paper; behavior does not |

**After `729d773`:** First fair retry. Early signal (session evening): Qwen still **~7 min/question** ≈ **~32k tokens** at ~78 tok/s → truncation may repeat.

**Plausible stack differences vs QRM (same 32k in their §3.1):**

| | QRM | Us |
|---|-----|-----|
| Harness | Lighteval tasks | `run_inference.py` + `qrm_math500.txt` |
| Prompt text | Same MATH-500 instruction (verified vs `reasoning.py`) | **Matches** |
| Seeds | 3 averaged (42–44) | seed 0 first |
| repetition_penalty | Not set | 1.05 (deployment default) |
| Extraction | `latex_gold_metric` | `\boxed{}` + math_verify |

**Publishability:**

- **Cannot claim:** “We reproduced QRM at 93.9%” with June or high-truncation runs.  
- **Can publish:** pass@1 **+ truncation_rate** under fixed 32k — deployment / budget-limited science.  
- **Thesis does not require** matching 93.9% — only the **b01 gate** does before opening b02.

---

## 6. Truncation methodology (official policy)

### Score truncated = wrong; n stays 500

- Hit `max_tokens` → often no `\boxed{}` → wrong.  
- **Never drop** truncated rows from denominator (selection bias).  
- QRM’s 93.9% **includes** truncations-as-wrong at 32k.

### Report truncation_rate as first-class metric

- Paper tables: **column next to pass@1**.  
- Quantized cells: higher truncation is often a **finding** (compression → longer traces).

### Do NOT selectively re-run truncated items at 64k

- Biases cross-cell comparison.  
- **Allowed:** separate sweep — **all 500** at 8k / 16k / 32k / 64k, new protocol label.

### Calibration rule (for manuscript)

- **Primary:** pass@1 all rows, truncated = wrong.  
- **Calibration:** rows with valid confidence only.  
- **Selective risk:** include truncated as confidently wrong; appendix without them.  
- **Trap:** looping traces have high token logprobs → “confidently wrong”; fix logprob bug before calibration claims.

### Gate check

```bash
python3 scripts/compare_qrm_baseline.py --summary results/<cell>_summary.json
```

| Metric | Qwen MATH-500 hard gate |
|--------|-------------------------|
| pass@1 | 93.9% ± 5 pp |
| truncation_rate | ≤ 0.15 |
| parse_failure_rate | ≤ 0.10 |

Gate failure ≠ discard run. Report honestly.

### Decision tree (if truncation high again)

```
Finish b01 at 32k → score all 500
  → truncation ≤15% AND pass@1 ~93% → PASS gate → b02 one block at a time
  → truncation HIGH, pass@1 LOW → report as-is; paper = budget-limited deployment
  → optional: full 500× budget sweep (labeled new protocol)
  → truncation LOW, pass@1 LOW → debug prompts/extraction, not budget
```

---

## 7. Why stay at 32k (and when to increase)

**32k is the QRM protocol** — fair comparison across BF16/FP8/AWQ/GPTQ and vs QRM Table 1.

**Two limits (don’t confuse):**

| Setting | Meaning |
|---------|---------|
| `max_tokens` = 32768 | **Output cap per answer** (protocol) |
| `max_model_len` = 40960 | **GPU KV reservation** (technical headroom) |

**Why not bump mid-campaign:**

- Breaks cross-cell and QRM comparability.  
- 1M → OOM; 131k → extreme slowness.  
- Per-item rescue = biased science.

**When increase is OK:** New labeled experiment — **all 500** at e.g. 64k (appendix / budget sensitivity).

---

## 8. PARAM Rudra / SLURM rules (must follow)

| Rule | Detail |
|------|--------|
| **QOS** | Max **2 GPUs/user** (`QOSMaxGRESPerUser`) |
| **Walltime** | ~48 h per job |
| **Never `--exclusive`** on split 1-GPU b01 cells | Exclusive counts as **both** GPUs on ragpu node |
| **Submit b01** | `bash scripts/hpc/submit_hpc_blocks.sh b01` (or `b01 --fresh`) |
| **Docs** | `docs/PARAM_RUDRA_SLURM.md` |
| **Triton** | Conda gcc required (`4da8913`) — compute nodes lack `stdlib.h` |
| **Login node** | No GPU compute — always `sbatch` |

### Publication blocks (b01–b09)

| Block | Content | Default submit |
|-------|---------|----------------|
| b01 | Qwen + Llama BF16 MATH-500 | Yes (default) |
| b02–b04 | FP8, AWQ, GPTQ pairs | After b01 gate |
| b05 | GPTQ-3 single GPU | Queued with b02–b06 |
| b06 | GSM8K | Queued |
| b07 | GPQA | Manual after HF gate |
| b08–b09 | Qwen-1.5B future | Not default |

**Do not** `submit_hpc_blocks.sh all_blocks` before b01 passes.

---

## 9. Infrastructure fixes to keep

| Commit | Fix | Keep? |
|--------|-----|-------|
| `4da8913` | Conda gcc + Triton preflight | **Yes** |
| `8ec36f8` / `1e53e10` | AWQ/GPTQ `dtype: float16` | **Yes** |
| `60111a8` | `fp8_e5m2` KV on **quant** cells only | **Yes** |
| `7448164` | No `--exclusive` on split cells | **Yes** |
| `729d773` | 32k/40k BF16, `enforce_eager: false` | **Yes** (active protocol) |
| `a3414a4` | Soft git gate (`QREASON_STRICT_GIT=1` for final runs) | **Yes** |

**Do not reintroduce:** 1M runtime clamp, BF16 `kv_cache_dtype`, 131k as publication protocol, 9-cell parallel waves.

---

## 10. Archives — what to use / discard

| Archive | Status |
|---------|--------|
| `outputs-hpc-2a100-main-2026-06-29` | **Invalid** — no repetition_penalty, high truncation; delete / never resume |
| `outputs-hpc-2a100-main-2026-07-03` (131k / 1M waves) | **Discard** partial rows from wrong configs |
| `outputs-hpc-2a100-main-2026-07-03` (post-`729d773`, jobs 86757/86758) | **Active publication campaign** |

---

## 11. Recommended action plan (agreed session direction)

### Now

1. **Let 86757/86758 run** — do not cancel; do not change `max_tokens`.  
2. **At 10 rows** — check `raw/*.jsonl`, spot-check `truncated` / `finish_reason`.  
3. **At ~50 rows** — estimate truncation_rate; decides paper narrative.  
4. **Plan resume** — Qwen ~7 min/q → ~58 h for 500 > 48 h walltime; resubmit **same archive** without `--fresh`.

### After b01 scored

```bash
python3 scripts/compare_qrm_baseline.py --summary results/level_a_qwen7b_bf16_math500_seed0_summary.json
```

### Paper strategy (lead with contribution, not repro)

1. **Main table:** all quants at **fixed 32k** — pass@1, **truncation_rate**, cost-per-correct, VRAM.  
2. **QRM paragraph:** honest repro attempt + gap if any.  
3. **Optional appendix:** all-500 budget sweep (8k/16k/32k/64k).  
4. **Do not cite** June 7% or 131k rows.

### Execution order (2–4 weeks)

```
Week 1: Finish b01 (resume if needed) → score → gate check
Week 2–3: b02 → b03 → b04 (one block at a time, same archive)
Week 4+: b06 GSM8K, b07 GPQA; seeds 42–44 only if needed for robustness
```

### If truncation stays ~90% at clean 32k

- **Still finish 500/500** and report.  
- Frame as **budget-limited deployment** finding (Protocol B).  
- Prompt already matches QRM — investigate stack (vLLM version, scorer, seed) not prompt text.  
- Optional: controlled 64k sweep for **all 500** — separate table.  
- See **§19** for strict Protocol A rerun if gate miss with low truncation.

---

## 12. Live snapshot (2026-07-03 ~19:10 IST, session end)

| Item | Value |
|------|-------|
| Git | `9912d7d` (literature archive) + earlier `ab58206`, `e733b7a`, `729d773` |
| Jobs | **86757** Qwen RUNNING `ragpu006` (~45 min); **86758** Llama RUNNING `racn116` (~34 min) |
| Progress | Qwen **~6/500** in logs (~7 min/question) |
| Raw on disk | Checkpoint every 10 rows — check `wc -l raw/*.jsonl` |
| Config | `729d773`, `repro_qrm.yaml`, git commit in cell metadata |
| Telegram | `~/start-hpc-telegram-watcher.sh` |

**ETA warning:** 500 × 7 min ≈ 58 h > 48 h SLURM limit → expect **resume** on same archive.

---

## 13. Useful commands

```bash
# Queue
squeue -u $USER

# Progress
tail -30 outputs-hpc-2a100-main-2026-07-03/logs/level_a_qwen7b_bf16_math500_seed0.log
wc -l outputs-hpc-2a100-main-2026-07-03/raw/*.jsonl

# Gate
python3 scripts/compare_qrm_baseline.py --summary results/<cell>_summary.json

# Submit b01 (fresh campaign)
cd /scratch/manishn_iitp/reasoning-compression-lab
bash scripts/hpc/submit_hpc_blocks.sh b01 --fresh

# Resume (same archive — NO --fresh)
bash scripts/hpc/submit_hpc_blocks.sh b01

# Sync git on HPC
git fetch origin && git reset --hard origin/main
```

---

## 14. What NOT to do

- Cite `outputs-hpc-2a100-main-2026-06-29` pass@1 in the paper.  
- Mix 131k / 32k / 1M rows in one table.  
- Drop truncated rows from pass@1 denominator.  
- Re-run only truncated problems at higher budget.  
- Submit b02–b06 before b01 is 500/500 and scored.  
- Use `sbatch --exclusive` for parallel 1-GPU split cells.  
- Submit 9-cell waves that violate b01 gate.  
- Treat SLURM FAILED jobs from July iteration as paper results.  
- Claim QRM reproduction without `compare_qrm_baseline.py` hard_passed.

---

## 15. Key doc index

| Doc | Purpose |
|-----|---------|
| `notes.md` | **This file** — session backup |
| `CHANGELOG.md` | Canonical campaign narrative + truncation policy |
| `progress.md` | Dated execution log (may lag; update after changes) |
| `docs/BEGINNER_HPC_GUIDE.md` | Beginner + QRM §1.1–1.2 |
| `docs/KNOWN_ISSUES.md` | Invalid archives, resume traps |
| `docs/PARAM_RUDRA_SLURM.md` | Exclusive / QOS trap |
| `docs/PHD_ROADMAP.md` | Full PhD plan |
| `docs/literature/` | Reference PDFs / text extracts |
| `AGENTS.md` / `~/CLAUDE.md` | Agent + cluster memory |

---

## 16. Git commits referenced today

```text
7f1b32c notes: QRM protocol decision, queue audit, infra checklist (§17–22)
9912d7d Add literature archive (paper1 .md, paper2 pdf+md)
ab58206 Docs: QRM baseline and pass@1 reproduction gap
e733b7a CHANGELOG: campaign narrative + truncation methodology
729d773 Fix b01: QRM max_tokens 32768, max_model_len 40960, enforce_eager false
7448164 Fix QOS trap: no --exclusive on split 1-GPU cells
```

---

## 17. Queue audit (2026-07-03 ~19:30 IST)

| Question | Answer |
|----------|--------|
| Are all required jobs queued? | **Only b01** — intentional. b02–b06 **not** submitted (b01 gate). |
| Jobs running | **86757** Qwen BF16 `ragpu006`; **86758** Llama BF16 `racn116` |
| Progress | Qwen **8/500** in logs (~7 min/question, ~78 tok/s output) |
| Raw JSONL on disk | **0 rows** — checkpoint every **10** rows; first file at row 10 |
| b07–b09 | Not queued (GPQA needs HF gate; 1.5B cells future) |

**Verdict:** Queue state is **correct** for current strategy. Do **not** submit b02–b06 until b01 is 500/500 and scored.

---

## 18. QRM reproduction audit (vs GitHub `inference.py` + `reasoning.py`)

Fetched 2026-07-03 from [ruikangliu/Quantized-Reasoning-Models](https://github.com/ruikangliu/Quantized-Reasoning-Models).

### What matches (good)

| Item | QRM | Us | Status |
|------|-----|-----|--------|
| Prompt (MATH-500) | `{problem}\n\nPlease reason step by step…\boxed{}` | `prompts/qrm_math500.txt` — **same text** | **MATCH** |
| Chat template | `use_chat_template=True` | `render_prompt()` applies HF chat template | **MATCH** |
| temperature | 0.6 | 0.6 | **MATCH** |
| top_p | 0.95 | 0.95 | **MATCH** |
| max generation | `max_new_tokens=32768` | `max_tokens=32768` | **MATCH** |
| Models / task | Qwen-7B, Llama-8B BF16 MATH-500 | b01 cells | **MATCH** |
| `prompt_profile` | reproduction task | `reproduction` in cell configs | **MATCH** |
| `verify_decoding_params.py` | QRM ref in `sampling_utils.py` | VERIFY OK on active yaml | **MATCH** |

QRM `reasoning.py` prompt_fn (MATH-500):

```python
query=f"{line['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
```

Our `prompts/qrm_math500.txt` is the same instruction (with `{question}` placeholder). **No long 5-rule system prompt** — earlier concern was wrong.

### Gaps vs strict QRM (material)

| Item | QRM | Us (active `729d773`) | Impact |
|------|-----|----------------------|--------|
| **Seed** | default **42**; paper **3 seeds** (42–44) | **seed 0** | Medium — variance; gate expects QRM bands |
| **repetition_penalty** | **Not set** in `GenerationParameters` | **1.05** (June loop fix) | Medium — may shorten traces vs QRM |
| **enforce_eager** | **true** | **false** (BF16 speed) | Low — documented deviation |
| **max_model_len** | **32768** | **40960** (KV headroom) | Low — output still capped at 32k |
| **Harness** | Lighteval + `latex_gold_metric` | `run_inference.py` + `\boxed{}` / math_verify | Medium — scorer may disagree on edge cases |
| **top_k** | 30 for QwQ only; **None** for Qwen/Llama | not set | **MATCH** for our models |

### Reproduction status

- **Decoding + prompt aligned** with QRM `inference.py` defaults.  
- **Results not yet proven** — June 7% invalid; current run too early (8/500, no checkpoint file).  
- **Cannot claim** “reproduced QRM Table 1” until `compare_qrm_baseline.py` **hard_passed** on scored 500/500.

---

## 19. Should we match QRM 100%? (protocol decision)

**Short answer: No — not everywhere. Yes — for the b01 gate claim only, on parameters that change generations.**

Paper 1 is **not** the QRM paper. Our contribution is pass@1 **plus** truncation, calibration, cost, and seed variance under a **fixed deployment budget**. QRM is a **sanity baseline**, not the thesis.

Use **two labeled protocols** (already sketched in `configs/baselines/qrm_literature_targets.yaml`):

### Protocol A — `qrm_repro` (b01 hard gate)

**Purpose:** “Our stack can hit QRM Table 1 bands before we open the quant grid.”

| Parameter | Target |
|-----------|--------|
| Prompt | `prompts/qrm_math500.txt` (verified = QRM) |
| temp / top_p / max_tokens | 0.6 / 0.95 / 32768 |
| Seeds | **42, 43, 44** (report mean ± std like QRM) |
| repetition_penalty | **None** (match QRM `inference.py`) |
| enforce_eager | **true** preferred; if false, **document** in methods |
| max_model_len | 32768 or 40960 OK if `max_tokens` stays 32768 |
| Scorer | If gate fails with low truncation, diff Lighteval `latex_gold_metric` vs our math_verify |

**When to run:** After current pilot finishes **or** if pilot shows gate miss with low truncation.

### Protocol B — `our_hpc_deployment` (Paper 1 main grid b02–b06)

**Purpose:** Fair **internal** comparison across BF16 / FP8 / AWQ / GPTQ at fixed budget.

| Parameter | Target |
|-----------|--------|
| Fixed budget | **32k** `max_tokens` all cells (non-negotiable for cross-quant tables) |
| repetition_penalty | **1.05** allowed — label as *our vLLM anti-loop deployment default* |
| enforce_eager | **false** on BF16 A100 (speed; same across all quants in a table) |
| Seeds | seed 0 for grid; add 42–44 only if Sober Look / variance section needs it |
| Truncation | **First-class metric** beside pass@1 |

**Do not** mix Protocol A and B rows in one table without a protocol column.

### Side-by-side: what changes between protocols

| Parameter | Protocol A (`qrm_repro`) | Protocol B (`our_hpc_deployment`) | Why different |
|-----------|------------------------|-----------------------------------|---------------|
| Claim in paper | “Reproduced QRM baseline” | “Deployment under fixed budget” | Different scientific claims |
| Seeds | 42, 43, 44 | 0 (grid) | QRM reports mean ± std over 3 seeds |
| repetition_penalty | unset | 1.05 | June proved R1 loops without it on vLLM 0.8.x |
| enforce_eager | true | false | Speed on A100; document if A uses false |
| max_model_len | 32768 or 40960 | 40960 | Headroom only; output capped at 32k |
| Harness | Ideally Lighteval parity if gate fails | Custom harness OK | Only matters if trunc low but pass@1 low |
| Opens b02? | Only if `compare_qrm_baseline.py` hard_passed | Grid can proceed with honest truncation story | Gate vs contribution |

### What to do with jobs 86757/86758 (running now)

| Choice | Rationale |
|--------|-----------|
| **Let them finish** | Sunk cost; first clean post-`729d773` signal at 32k |
| **Label as** `our_hpc_repro` seed-0 **pilot** | Not strict QRM repro until seeds 42–44 + no repetition_penalty rerun |
| **Do not cancel** for 100% QRM mid-run | Would waste ~8h; learn truncation_rate first |
| **If gate fails after score** | (1) truncation high → budget/stack issue, not prompt; (2) truncation low, pass@1 low → scorer or seed rerun under Protocol A; (3) then decide b02 |

### Decision tree

```
86757/86758 finish 500/500 (Protocol B pilot, seed 0)
  → score → compare_qrm_baseline.py
    → PASS (pass@1 ~94% ±5pp, trunc ≤15%) → submit b02 ONE block, same 32k protocol
    → FAIL trunc high, pass@1 low → finish Paper 1 as budget-limited deployment; optional Protocol A rerun
    → FAIL trunc low, pass@1 low → Protocol A strict rerun (seeds 42–44, no repetition_penalty) before b02
```

**Bottom line:** Match QRM **where it gates credibility** (32k, prompt, temp, top_p, seeds for the repro table). **Do not** slow the whole thesis to Lighteval parity or drop `repetition_penalty` from the main quant grid without labeling — that is a **feature** of our deployment study, not a bug.

---

## 20. Telegram watcher, 48h walltime, checkpoints, backups

### Telegram watcher

| Item | Status |
|------|--------|
| Script | `~/start-hpc-telegram-watcher.sh` |
| Watches | Jobs **86757**, **86758** |
| Archive | `outputs-hpc-2a100-main-2026-07-03` |
| Caveat | Compute nodes may not resolve `api.telegram.org` — watcher can fail DNS-side; use `squeue` + logs as ground truth |

### 48h SLURM walltime

| Item | Detail |
|------|--------|
| Cluster limit | ~**47–48 h** per job |
| Qwen ETA | 500 × ~7 min ≈ **58 h** → **will hit wall** before 500/500 |
| Auto-pause | **None** — job ends at walltime |
| Resume | Resubmit **same archive**, **no** `--fresh`: `bash scripts/hpc/submit_hpc_blocks.sh b01` |
| Checkpoint | Every **10** rows to `raw/*.jsonl` + `state.json` |
| SIGTERM handler | **No** graceful flush — rely on last checkpoint ≤10 rows behind log |

### Backups

| Path | Contents |
|------|----------|
| `_backup/latest/` | Mirrors logs, metadata, manifest |
| `_backup/latest/raw/` | Empty until first 10-row checkpoint |
| Git autopush | **Off** unless `QREASON_ENABLE_AUTOPUSH=1` |

---

## 21. Publishability recap (simple language)

| Claim | OK now? |
|-------|---------|
| “We reproduced QRM 93.9%” | **No** — June invalid; July run incomplete |
| “Under fixed 32k, pass@1 = X, truncation = Y%” | **Yes** — after 500/500 scored (Paper 1 core) |
| “Quantization increases truncation at same budget” | **Yes** — after b02–b05 (novelty) |
| “Calibration / cost-per-correct under compression” | **Yes** — Paper 1 contribution (after scoring pipeline) |

~7% pass@1 with ~90% truncation is **not** publishable as QRM reproduction. It **is** publishable as evidence that **budget-capped deployment** fails silently (high truncation, low scored accuracy) — if reported honestly with `truncation_rate`.

---

## 22. Live snapshot (refresh)

| Item | Value |
|------|-------|
| Time | 2026-07-03 ~19:35 IST |
| Jobs | **86757** RUNNING ~1h05m `ragpu006`; **86758** RUNNING ~54m `racn116` |
| Qwen log | **9/500** (~7 min/q, ~78 tok/s) |
| Llama log | ~8/500 (similar pace) |
| Raw JSONL | 0 lines (first checkpoint at row **10**) |
| Protocol label | **Protocol B** pilot, seed 0 (`our_hpc_repro`) |
| Next milestone | Row 10 → `wc -l raw/*.jsonl`; spot-check `truncated` / `finish_reason` |
| ETA risk | 500 × 7 min ≈ 58 h > 48 h wall → **resume** same archive, no `--fresh` |

---

## 23. Practical commands by protocol

### Protocol B — current pilot (86757/86758)

```bash
# Monitor
squeue -u $USER
tail -30 outputs-hpc-2a100-main-2026-07-03/logs/level_a_qwen7b_bf16_math500_seed0.log
wc -l outputs-hpc-2a100-main-2026-07-03/raw/*.jsonl

# Resume after 48h wall (pin archive date — NO --fresh)
cd /scratch/manishn_iitp/reasoning-compression-lab
QREASON_OUTPUT_ROOT=$PWD/outputs-hpc-2a100-main-2026-07-03 \
QREASON_HPC_DATE=2026-07-03 \
bash scripts/hpc/submit_hpc_blocks.sh b01

# Score after 500/500
python3 scripts/compare_qrm_baseline.py \
  --summary results/level_a_qwen7b_bf16_math500_seed0_summary.json
```

### Protocol A — strict QRM gate rerun (only if needed)

Trigger when: gate fails with **low truncation** but **low pass@1** (scorer/seed issue, not budget).

1. Edit `configs/decoding/repro_qrm.yaml`: remove or comment `repetition_penalty`; set `seed: 42` (repeat 43, 44).
2. Optionally set `enforce_eager: true` in serving config for BF16 cells.
3. Fresh archive or new protocol label in cell metadata — do **not** mix with Protocol B rows.
4. Submit one seed at a time:

```bash
# Example: seed 42 only, fresh archive for strict repro
QREASON_OUTPUT_ROOT=outputs-hpc-2a100-qrm-repro-seed42 \
  bash scripts/hpc/submit_hpc_blocks.sh b01 --fresh
```

5. Gate must pass on **each** of seeds 42–44 before claiming QRM reproduction (report mean ± std).

### After b01 gate (either protocol path)

```bash
# ONE block at a time — never all_blocks before gate
bash scripts/hpc/submit_hpc_blocks.sh b02   # only after b01 scored + decision in §19
```

---

## 24. Manuscript wording cheat sheet

| Situation | Write this | Do NOT write this |
|-----------|------------|-------------------|
| July pilot (seed 0, rep_penalty 1.05) | “BF16 baseline under our deployment protocol at 32k” | “We reproduced QRM at 93.9%” |
| Protocol A passes 42–44 | “BF16 matches QRM Table 1 within ±5 pp (seeds 42–44)” | “Identical to QRM” (harness still differs) |
| High truncation at 32k | “X% of completions hit the budget cap before `\boxed{}`” | “Model accuracy is X%” without truncation column |
| Quant grid (b02–b05) | “Fixed 32k across quants; truncation_rate in Table N” | Mix June 7% archive with July rows |

---

## 25. The b01 gate — what it is and what you should know

**Full detail:** §6, §19, `configs/baselines/qrm_literature_targets.yaml`, `scripts/compare_qrm_baseline.py`.

### What the gate is

The **b01 hard gate** asks: *“Can our HPC stack reproduce QRM Table 1 BF16 numbers before we spend GPU weeks on the quant grid?”*

| Metric | Qwen-7B MATH-500 | Llama-8B MATH-500 |
|--------|------------------|-------------------|
| pass@1 reference | **93.9%** ± 0.7 (QRM Table 1) | **91.0%** ± 1.1 (QRM Appendix B Table 4) |
| Tolerance | **±5 pp** absolute | **±5 pp** absolute |
| truncation_rate max | **≤ 15%** | (same policy) |
| parse_failure_rate max | **≤ 10%** | (same policy) |
| Required `prompt_profile` | **`reproduction`** | **`reproduction`** |

Run checker (use conda `qreason` python):

```bash
/home/manishn_iitp/.conda/envs/qreason/bin/python3 scripts/compare_qrm_baseline.py \
  --summary outputs-hpc-2a100-main-2026-07-03/results/<cell>_summary.json
```

**Gate outcomes:**

| Result | Meaning | Action |
|--------|---------|--------|
| `hard_passed: true` | Stack matches QRM bands | Submit **b02** one block |
| `hard_passed: false`, high truncation | Budget exhaustion dominates | Paper 1 “deployment budget” story; optional Protocol A rerun |
| `hard_passed: null`, profile mismatch | Wrong prompt profile for gate | Re-run with `reproduction` cell, or don’t use for QRM claim |
| Qwen incomplete | Can’t gate yet | Resume until 500/500 scored |

**Gate failure does not mean trash the run** — it means don’t open b02 claiming reproduction.

### Resume trap (memorise)

`submit_hpc_blocks.sh` defaults `QREASON_OUTPUT_ROOT` to **today’s date**. Resuming July 3 data on July 5 **without** pinning the archive creates an empty new folder.

**Correct resume:**

```bash
cd /scratch/manishn_iitp/reasoning-compression-lab
QREASON_OUTPUT_ROOT=$PWD/outputs-hpc-2a100-main-2026-07-03 \
QREASON_HPC_DATE=2026-07-03 \
bash scripts/hpc/submit_hpc_blocks.sh b01
```

Also fix `metadata/dirty_nodes.txt` if nodes are concatenated on one line (causes `Invalid node name` sbatch error).

---

## 26. Llama BF16 results — verification vs June & QRM (2026-07-05)

### Job outcomes (86757/86758 wave)

| Cell | Job | Outcome | Rows |
|------|-----|---------|------|
| Qwen BF16 `reproduction` | 86757 | **TIMEOUT** @ 47h | **410/500** checkpointed |
| Llama BF16 `sober` | 86758 | **COMPLETED** | **500/500** scored |

**Resume submitted:** jobs **87111** (Qwen), **87112** (Llama) on `racn116`, archive `outputs-hpc-2a100-main-2026-07-03`.

### Llama July 2026 scored summary (genuine, complete run)

| Metric | July 2026 (`729d773` protocol) | June 2026 (invalid archive) | QRM paper (Llama BF16) |
|--------|-------------------------------|------------------------------|------------------------|
| pass@1 | **19.6%** (98/500) | 21.4% | **91.0%** ± 1.1% |
| truncation_rate | **58.0%** (290/500) | ~59% | (included in their pass@1 at 32k) |
| parse_failure_rate | **60.4%** (302 no `pred_answer`) | ~60% | — |
| completion_tokens p50 | **32768** (hits cap) | similar | 32k protocol |
| finish_reason | stop=210, length=290 | — | — |
| prompt_profile | **`sober`** (not QRM repro) | unknown / broken rep_pen | `reproduction` (QRM) |

### Is July Llama genuine vs June?

**Yes — more trustworthy than June, scientifically similar.**

| Evidence | Interpretation |
|----------|----------------|
| Truncation **58% vs 59%** | Same budget-exhaustion phenomenon; not a one-off bug |
| pass@1 **19.6% vs 21.4%** | Within June noise; **not** a dramatic fix from `repetition_penalty` |
| 500/500 completed + scored | July run is **complete**; June had broken `repetition_penalty` wiring |
| Median output = **32k tokens** | Models still fill the cap — ~7 min/question, ~78 tok/s |
| Non-truncated subset pass@1 = **45.2%** (95/210) | Even **without** budget limit, still **far below QRM 91%** |

**Conclusion:** July confirms June’s *shape* (high truncation, low pass@1) under fixed 32k. The `729d773` fixes made the run **valid to cite** for Paper 1 deployment metrics, but **do not** close the QRM reproduction gap.

### Why Llama ≠ QRM 91% (theoretical)

1. **Prompt profile:** b01 Llama uses **`sober`** (`prompts/math500.txt` — long “Therefore, the final answer is…” format), not QRM **`reproduction`** (`qrm_math500.txt`). Gate checker returns `SKIP: prompt_profile mismatch`.
2. **Truncation → wrong:** 290 rows hit length cap → no extractable answer → scored wrong (by design, §6).
3. **Reasoning without answer:** 100 rows have `\boxed{}` but wrong math — model reasons but errs.
4. **Stack gap:** QRM uses Lighteval `latex_gold_metric`; we use `math_verify`. Edge-case disagreements possible.
5. **Seed:** seed 0 vs QRM seeds 42–44.

**Paper 1 framing:** This is exactly the “beyond accuracy” story — pass@1 alone hides that **58% of serves never returned a scorable answer** under deployment budget.

### Qwen partial (410 rows) — early signal

| Metric | Value |
|--------|-------|
| truncation | **94.1%** (386/410) |
| `\boxed{}` in completion | **7.6%** |
| vs June Qwen | 90% trunc, 7% pass@1 — **same failure mode** |

Expect QRM gate **fail** on Qwen unless remaining 90 rows differ radically (unlikely).

---

## 27. Live snapshot (2026-07-05)

| Item | Value |
|------|-------|
| Jobs | **87111** Qwen resume RUNNING; **87112** Llama (may skip if scored) |
| Archive | `outputs-hpc-2a100-main-2026-07-03` |
| Qwen progress | **410/500** durable; resume rows 411–500 (~10.5 h ETA) |
| Llama | **500/500 scored** — `results/level_c_llama8b_bf16_math500_seed0_summary.json` |
| b01 gate | **Not passed** (Llama profile mismatch + metrics; Qwen incomplete) |
| b02–b06 | **Not submitted** (correct) |
| Fixed | `dirty_nodes.txt` split across lines; resume uses pinned `QREASON_OUTPUT_ROOT` |

---

## 28. Post–gate-fail pivot — future of the experiment (2026-07-05)

**Gate failed. Project is NOT dead.** Paper 1 was never “reproduce QRM.” QRM was a **sanity check**. The July BF16 numbers are the **first real scientific result** for the thesis.

### Finish Qwen 90 rows (~10 h)?

| | Verdict |
|--|---------|
| **Necessary?** | **No** |
| **Why** | 410/500 at **94.1%** truncation — gate needs ≤15%; 90 more rows cannot pass QRM gate or change Paper 1 story |
| **Resume status** | Job **87111 FAILED** (55s, GPU busy on shared node) |
| **Optional** | Complete 500/500 only for table symmetry (footnote: “Qwen n=410 partial”) |

### What the gate failure teaches us

1. **Fixed 32k is a deployment constraint, not a free hyperparameter.** Median completion = 32,768 tokens for both models — they fill the budget.
2. **pass@1 alone is misleading.** 58% (Llama) / ~94% (Qwen) of serves produce no scorable answer under cap.
3. **Stack ≠ QRM Lighteval at same nominal budget.** Even Llama **non-truncated** pass@1 = **45%** vs QRM **91%** — prompt profile, scorer, seeds, vLLM stack all matter.
4. **repetition_penalty fix did not rescue Llama.** June 21.4% → July 19.6%, truncation ~59% → 58%.
5. **Cost-of-Pass (literature)** — Llama `cost_per_correct_seconds` ≈ **1614 s** — quantization/truncation paper should foreground this.

### Literature map — what to do with these results

From `docs/literature/PAPER1_READING_MAP.md` and merged bundle:

| Paper group | How July results fit |
|-------------|---------------------|
| **QRM** | Honest negative repro in §4.1; pivot main claim to deployment metrics |
| **GPTQ / AWQ / SmoothQuant** | Proceed b02–b05 — **same 32k** — test if quants **worsen** truncation |
| **Cost-of-Pass** | Table column: cost-per-correct vs truncation_rate; frontier shifts under budget |
| **Calibrating LLMs / Sample Consistency** | Score calibration on truncated-as-wrong; watch “confidently wrong” loops |
| **A Sober Look** | Report seed 0; add 42–44 only if variance section needs it |
| **AbstentionBench** | Truncation ≈ forced abstention without `\boxed{}` — link to selective prediction |

### Future of the project (3 paths — pick 1+2)

**Path A — Main Paper 1 grid (recommended)**  
Submit **b02** (FP8 pair) → **b03** (AWQ) → **b04** (GPTQ4) → **b05** (GPTQ3) one block at a time.  
**Hypothesis:** BF16 truncation 58% → FP8/AWQ/GPTQ **equal or worse** at same 32k.  
**Main figure:** pass@1 **and** truncation_rate **and** cost-per-correct across quants.

**Path B — Budget sensitivity appendix**  
All 500 MATH-500 at 8k / 16k / 32k / 64k (one model first). Shows **where** accuracy lives vs deployment cap.

**Path C — Strict QRM debug (lower priority)**  
Protocol A: reproduction prompt, seeds 42–44, no repetition_penalty, Lighteval scorer diff. Only if reviewer demands repro.

### Immediate action list

```
[ ] Accept gate fail — update supervisor narrative (Beyond Accuracy, not QRM clone)
[ ] SKIP Qwen 90 rows (or low-priority resume when GPU exclusive)
[ ] Fix b01 Llama → reproduction profile OR document sober in every table
[ ] Submit b02 FP8 when ready (same archive or fresh — label protocol)
[ ] Draft Table 1 skeleton: Model × Quant × pass@1 × trunc_rate × cost/correct × VRAM
[ ] Enable calibration scoring (remove --skip-calibration) on next wave
[ ] MacBook sync HPC doc commits
```

### One-paragraph supervisor pitch

> We validated the HPC harness on BF16 anchors. Under a fixed 32k deployment budget, Llama-8B achieves 19.6% pass@1 with 58% truncation; Qwen-7B shows ~94% truncation on 410 problems. We do not reproduce QRM Table 1 on our vLLM stack, but that strengthens Paper 1’s thesis: **accuracy alone hides deployment failure**. Next we run the quantization grid at the same budget to measure how compression affects truncation and cost-per-correct.

---

## 29. Live snapshot (2026-07-05 evening)

| Item | Value |
|------|-------|
| Queue | **Empty** |
| Qwen b01 | **410/500** — skipped (not required) |
| Llama b01 | **500/500 scored** — gate failed |
| **Path C diagnostic** | **SUBMITTED** jobs **87116** Qwen 32k, **87117** Llama 32k, **87118** Qwen 64k |
| Archive | `outputs-hpc-diag-pathc-2026-07-05` |
| Report | `bash scripts/hpc/report_pathc_diagnostic.sh` when jobs finish |

### Path C protocol (commit `7d46c3f`)

| Wave | Cell | Settings |
|------|------|----------|
| d01 Qwen | `diag_qwen7b_bf16_math500_seed42_n50` | reproduction prompt, seed 42, 32k, no rep_pen, eager=true, **n=50** |
| d01 Llama | `diag_llama8b_bf16_math500_seed42_n50` | same |
| d02 Qwen | `diag_qwen7b_bf16_math500_seed42_n50_64k` | same + **max_tokens=65536** |

**Pass heuristic (n=50):** pass@1 ≥ **80%** and truncation ≤ **25%** on 32k → stack repro OK, consider full 500.  
**64k:** if pass@1 jumps vs 32k Qwen → truncation was bottleneck; if still low → harness/scorer gap.

---

*Last updated: 2026-07-05 (evening). Gate failed → see §28 for pivot. §25 = gate definition.*