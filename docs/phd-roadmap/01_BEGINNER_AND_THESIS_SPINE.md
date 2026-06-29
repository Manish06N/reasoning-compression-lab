# Part 1 — Beginner Explanation and Thesis Spine

[← Index](README.md)

---

## 1. What problem are we solving?

This PhD plan is about making **reasoning language models** cheaper, more reliable, and more deployable.

A **reasoning LLM** does not only produce a short answer; it often generates a **long reasoning trace** before the final answer (math, science MCQ, code, multi-step tasks).

**Central problem:** Strong reasoning models are expensive — GPU memory, many tokens, slow inference. Organizations compress them via **quantization** (fewer bits per weight: 16-bit → 8-bit → 4-bit). That usually saves memory and cost, but can damage reliability in ways **invisible if you only check final accuracy**.

### Plain-English version

The question is not only: *“Does the compressed model still get the answer right?”*

The better question is: *“Does the compressed model stay trustworthy, calibrated, fast, cheap, and stable across repeated runs?”*

That is the research opening. Earlier work studied whether quantization hurts reasoning **accuracy**. This PhD moves **beyond accuracy** into calibration, selective prediction, seed variance, latency, VRAM, and **cost-per-correct-answer**.

---

## Key terms (quick reference)

| Term | Simple meaning | Why it matters |
|------|----------------|----------------|
| Reasoning LLM | Multi-step reasoning before answering | Long traces → cost |
| Quantization | Fewer bits (BF16, FP8, GPTQ, AWQ, GGUF) | Memory/cost vs reliability tradeoff |
| Calibration | Confidence matches correctness | Overconfident wrong answers are dangerous |
| Selective prediction | Answer when confident; abstain/escalate otherwise | Safe deployment |
| Cost-per-correct | Money per **correct** answer, not per token | Industry economics |
| Seed variance | Results change across random seeds | Single-run papers can mislead |
| A100 / RTX 5080 | Datacenter vs local GPU | Deployment regimes differ |

Full glossary: [10_APPENDICES_REFERENCE.md](10_APPENDICES_REFERENCE.md#appendix-b-beginner-glossary).

---

## 2. Your constraints and why they shape the topic

| Constraint | Implication |
|------------|-------------|
| 3+ first-author Q1/Q2 journals + 2 conferences | Topic must be **journal-native**, not purely frontier conference method work |
| ~2 years left | Paper 1 must submit fast; avoid overbuilding |
| 2× A100 80GB, 48h SLURM | Inference + moderate training OK; pretraining / multi-week RL not |
| Distance mode, flexible supervisor | Strict operating system, public artifacts, self-driven execution |
| Career: GenAI labs / research engineering | Build inference, eval, deployment, quantization, profiling skills |
| India + abroad options | Globally legible spine; Paper 3 adds Indic/multilingual deployment |

> **Load-bearing warning:** Before final institutional commitment, verify in writing:
> - Accepted vs submitted/under-review papers count?
> - Q1/Q2 judged by Scopus/SJR, JCR, CORE, or other?
> - Does TMLR count as a journal?

---

## 5. How the thesis spine works

**Thesis spine (frozen):**

> Reliable and Cost-Efficient Deployment of Reasoning LLMs under Compression, Evaluation, and Multilingual Constraints.

Four load-bearing words: **reliable**, **cost-efficient**, **deployment**, **reasoning**.

| Spine component | Meaning | Which paper |
|-----------------|---------|-------------|
| Reliable | Correct + trustworthy + seed-stable | Paper 1 + C1 |
| Cost-efficient | Lower latency/memory/$ per correct answer | Paper 1 + Paper 2 |
| Deployment | Real stacks: vLLM, llama.cpp, A100, edge | All papers |
| Reasoning LLMs | Long-trace models (math, science, code) | All papers |
| Compression | BF16, FP8, AWQ, GPTQ, GGUF, KV-cache | Paper 1 + Paper 3 |
| Evaluation | Calibration, selective risk, seed variance | Paper 1 + C1 |
| Multilingual | Token-cost inequity, Indic deployment | Paper 3 |

### One-paragraph story

**Paper 1** asks whether compressed reasoning models remain reliable and economical beyond raw accuracy. **Paper 2** asks whether reasoning generation can be accelerated (speculative decoding / draft models) without destroying reliability. **Paper 3** asks how deployment lessons change in Indic/multilingual settings where tokenization and cost inequity matter. Together: strong reasoning models are not useful if too expensive, unreliable, or inaccessible — this work builds evidence and tools for responsible deployment.

---

## Correct vs incorrect thesis identity (V7)

| Correct | Incorrect |
|---------|-----------|
| Reliable and **cost-efficient deployment** of reasoning LLMs | “Quantization of reasoning models” only |
| Quantization is **one compression tool** in a deployment study | Quantization is the whole PhD |

See also: [07_V7_JOB_FIRST_STRATEGY.md](07_V7_JOB_FIRST_STRATEGY.md).
