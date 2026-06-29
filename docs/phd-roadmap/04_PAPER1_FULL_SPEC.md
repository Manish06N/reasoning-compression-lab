# Part 4 — Paper 1 Design (Full Specification)

[← Index](README.md) · Repo design: [docs/PAPER1_DESIGN.md](../PAPER1_DESIGN.md)

---

## Working title

**Beyond Accuracy: Calibration and Deployment Cost of Quantized Reasoning Models**

(Extended title in repo: *…Reliability, Calibration, Seed Variance, and Cost-per-Correct of Quantized Reasoning LLMs*.)

---

## Beginner-level explanation

Two models both get 80% accuracy. Many say the compressed one is “good enough.” This paper asks: Is it **overconfident when wrong**? **Unstable across seeds**? **Slower** because of longer traces? **Cheaper per token but not per correct answer**?

---

## Core research gap

Quantization-accuracy has been studied. The gap is **after accuracy**: calibration, selective risk, cost-per-correct, seed variance, latency, VRAM, reproducibility.

---

## Research questions

| RQ | Question | Why it matters |
|----|----------|----------------|
| RQ1 | Does calibration degrade at bit-widths where accuracy looks lossless? | Confident wrong answers |
| RQ2 | What is the cost-per-correct Pareto frontier across quant configs? | Industry economics |
| RQ3 | Is seed variance large enough to make single-run conclusions unsafe? | Statistical rigor |
| RQ4 (optional) | Does reasoning trace behavior change under compression? | Latency/cost illusion |
| RQ5 (optional) | Do edge/llama.cpp conclusions match datacenter/vLLM? | Paper 3 / stack transfer |

---

## Hypotheses

- **H1:** Calibration degrades before accuracy.  
- **H2:** Cheapest per correct ≠ most accurate config.  
- **H3:** At 4-bit or lower, seed variance may rival method-level differences.  
- **H4 (exploratory):** Compression may change trace length/termination (frame carefully — repetition confounds “more reasoning”).

---

## Minimum publishable grid (J1)

| Dimension | Minimum choice | Notes |
|-----------|----------------|-------|
| Models | R1-Distill Qwen-1.5B, Qwen-7B, Llama-8B | Size + architecture variation |
| Quant configs | BF16, FP8, AWQ-4, GPTQ-4, GPTQ-3 | Reference → aggressive low-bit |
| Tasks | MATH-500, GPQA-Diamond, LiveCodeBench subset, GSM8K control | Math, science MCQ, code, easy control |
| Seeds | **5** | maj@5, vote-share confidence, variance |
| Hardware | A100 datacenter (vLLM) for J1 minimum | Edge/energy → ambitious or J3 |
| Serving | vLLM (pin version + git SHA) | SGLang comparison → ambitious |

**Current repo execution:** HPC blocks b01–b09 (seed 0) — see [progress.md](../../progress.md).

---

## Metrics

| Family | Metrics | Role |
|--------|---------|------|
| Accuracy | pass@1, maj@5 | Correctness |
| Calibration | Brier, ECE (adaptive/equal-width) | Trustworthiness |
| Selective prediction | Risk-coverage, AURC | Safe abstention |
| Systems | Latency, throughput, peak VRAM | Deployment |
| Economics | $/1k correct, cost-per-correct | Industry |
| Stability | Seed variance, bootstrap 95% CIs | Reliability claim |
| Trace (exploratory) | Output/think tokens, truncation, answer flips | Secondary |

Repo implementations: `src/metrics/scoring.py`, `scripts/score_run.py`, `scripts/build_paper_tables.py`.

---

## Statistical plan

- **5 seeds** per cell for main claims  
- **Cluster bootstrap** on problems (not independent generations)  
- **Paired McNemar** vs BF16 on same problems  
- **Holm correction** on pre-registered primary endpoints  
- Pre-register: W4 accuracy Δ, ECE/Brier Δ, cost-per-correct ratio  
- **200-trace manual extraction audit** — target &lt;2% extraction error  

---

## Minimum vs ambitious (J1)

| Version | Includes | Rule |
|---------|----------|------|
| **Minimum J1** | 3 models × 5 configs × 4 tasks × 5 seeds, A100, calibration + cost + variance | Execute first; submit fast |
| **Ambitious J1** | +14B/Qwen3, edge RTX/GGUF, energy/J-per-correct, trace dynamics, stack transfer | Do not delay J1 — see [09_STACK_TRANSFER_GGUF.md](09_STACK_TRANSFER_GGUF.md) |

---

## Expected figures and tables (Paper 1)

| Figure/Table | Question | Paper role |
|--------------|----------|------------|
| Table 1: Experiment grid | What was run? | Methods / reproducibility |
| Fig 1: Accuracy vs quantization | Compression vs correctness | Baseline |
| Fig 2: Calibration vs quantization | Confidently wrong? | **Main novelty** |
| Fig 3: Cost-per-correct Pareto | Economic best config? | Industry relevance |
| Fig 4: Risk-coverage curves | Safe abstention? | Selective prediction |
| Fig 5: Seed variance | Single-run unsafe? | Methodological contribution |
| Fig 6: Latency/VRAM/energy | Deployment trade-off | Systems |
| Table 2: Failure taxonomy | Failure modes | Reviewer interpretation |
| Appendix: Extraction audit | Parser trustworthy? | Credibility gate |

---

## Paper 1 headline (V7 tightened)

> Deployment decisions are **unsafe** if based only on one-run accuracy; **calibration, selective risk, seed variance, and cost-per-correct** change the conclusion.

Not: “Quantization hurts reasoning.”

---

## Failure taxonomy (detection)

| Failure type | Meaning | Detection |
|--------------|---------|-----------|
| Accuracy collapse | Wrong under compression | Scored correctness |
| Calibration collapse | Confidently wrong | Brier/ECE/AURC vs accuracy |
| Trace inflation | More tokens / loops | Token counts, truncation rate |
| Answer drift | Different seeds → different answers | Cross-seed disagreement |
| Extraction failure | Parser misses answer | Manual audit |
| Format/refusal failure | No parseable answer | Unparsed rate |
| Degeneration | Repetition / incoherent loops | Trace audit |
| Cost illusion | Cheap/token, expensive/correct | cost-per-correct |

Full table: [08_EXECUTION_VALIDATION_V6.md](08_EXECUTION_VALIDATION_V6.md#appendix-s-failure-taxonomy).

---

## Prompt and logging specification

| Item | Requirement |
|------|-------------|
| Prompts | Frozen under `prompts/` or task configs; version on change |
| Datasets | HF id, split, row count, revision |
| Models | HF id, revision, local path, quant artifact path |
| Decoding | temperature, top_p, max_tokens, max_model_len, seed, n |
| Quantization | method, bits, group size, calib data, library version |
| Serving | vLLM/SGLang/llama.cpp version, CUDA, torch, GPU, TP size |
| Row schema | run_id, model, quant, task, problem_id, seed, output, tokens, latency, VRAM, gold, pred, correct, confidence |

---

## Reviewer objections (summary)

| Objection | Response |
|-----------|----------|
| “Only empirical” | Rigorous deployment science + released traces + multi-seed + cost/calibration endpoints |
| “Quantization already studied” | Pivot beyond accuracy; cite Liu et al. |
| “Too many metrics” | Pre-register primaries; label rest exploratory |
| “Only small models” | Deployment relevance of open 7B/8B class |
| “One seed” | **Part of the claim** — use 5 seeds |
| “Extraction wrong” | math-verify + 200-trace audit |

Full table: [10_APPENDICES_REFERENCE.md](10_APPENDICES_REFERENCE.md#appendix-i-reviewer-objections).
