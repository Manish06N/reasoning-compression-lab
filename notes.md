# Session Notes — 2026-07-03

Personal backup of findings, decisions, and learnings from the July 3 working session, cross-checked against `CHANGELOG.md`, `progress.md`, live HPC state, and `docs/literature/`.

**Repo:** `/scratch/manishn_iitp/reasoning-compression-lab`  
**GitHub:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical ops log:** `CHANGELOG.md` (detailed job history)  
**Dated progress:** `progress.md` (update after material changes)

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
| Seeds | 3 averaged | seed 0 first |
| Extraction | Their pipeline | `\boxed{}` + MATH scorer |

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
- Frame as **budget-limited deployment** finding.  
- Diff prompts vs QRM GitHub / Lighteval tasks.  
- Optional: controlled 64k sweep for **all 500** — separate table.

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
9912d7d Add literature archive (paper1 .md, paper2 pdf+md)
ab58206 Docs: QRM baseline and pass@1 reproduction gap
e733b7a CHANGELOG: campaign narrative + truncation methodology
729d773 Fix b01: QRM max_tokens 32768, max_model_len 40960, enforce_eager false
7448164 Fix QOS trap: no --exclusive on split 1-GPU cells
```

---

*Last updated: 2026-07-03 (evening IST). Refresh §12 after major job milestones.*