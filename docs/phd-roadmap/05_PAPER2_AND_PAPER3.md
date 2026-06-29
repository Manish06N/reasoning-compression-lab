# Part 5 — Paper 2 and Paper 3 Strategy

[← Index](README.md)

---

## Paper 2 — Reasoning-aware speculative decoding

**Concept:** A small **draft model** proposes tokens; the **target model** verifies. Fast when draft acceptance is high. Reasoning traces are long, branchy, and high-entropy — generic drafts may fail.

**Publishable question:** Why does speculative decoding break on reasoning traces, and can a **reasoning-aware draft** improve the latency–quality frontier?

| Version | Contribution | Fallback |
|---------|--------------|----------|
| **Minimum** | Failure analysis: acceptance rates on reasoning traces | Still publishable systems study |
| **Ambitious** | Train 0.5B–1.5B draft aligned to 7B/8B targets | If weak gains → failure taxonomy + design rules |

### Paper 2 must be industry-heavy (V7)

| Element | Requirement |
|---------|-------------|
| Core direction | Speculative decoding / draft-model acceleration / serving optimization |
| Post-training closure | Small SFT → DPO (or preference-style) artifact for draft/verifier |
| Systems closure | vLLM/SGLang, profiling, latency/throughput, KV-cache, optional plugin |
| Fallback | Acceptance-rate failure analysis if method gains weak |

**Timing target:** J2 submit month 18–22; C2 month 14–20.

---

## Paper 3 — Indic / multilingual deployment economics

**Focus:** Token-cost **inequity**, cost–quality–latency–energy frontiers, practical serving recipes for Indian/local constraints.

| Component | Explanation |
|-----------|-------------|
| Token-cost inequity | Same meaning → more tokens in some languages |
| Cost-quality-energy frontier | Quality vs latency vs memory vs cost vs energy |
| Serving recipes | Practical guidance for local hardware |
| Fallback | Tokenization/cost audit + curated benchmark subset if datasets messy |

**Rule:** Indic/multilingual is the **application layer**, not the whole thesis spine.

If Paper 1 energy measurement is unstable → move energy-heavy claims to Paper 3 rather than delaying J1.

**Timing target:** J3 submit month 24–30.

---

## Paper-by-paper thesis logic map

| Stage | Question | Output |
|-------|----------|--------|
| J1 / Paper 1 | What happens when reasoning models are compressed? | Harness, traces, calibration/cost tables |
| C1 | Quick visibility on eval protocol? | Workshop paper, arXiv, protocol demo |
| J2 / Paper 2 | Can inference be faster without hurting quality? | Draft method or failure study + Pareto |
| C2 | Systems demo for ML systems community? | Workshop + GitHub plugin |
| J3 / Paper 3 | Do lessons transfer to Indic/multilingual/edge? | Benchmark, recipes, HF dataset |

Plain language: **Paper 1** — what breaks under compression? **Paper 2** — can we speed up intelligently? **Paper 3** — does it help real multilingual users?

---

## Minimum vs ambitious (all papers)

| Paper | Minimum | Ambitious | Scope rule |
|-------|---------|-----------|------------|
| J1 | 3×5×4×5 grid; A100; calibration+cost+variance | Edge/GGUF, 14B, energy, stack transfer | No RLVR/agents in J1 |
| C1 | Protocol / seed variance short paper | Leaderboard adoption | Don’t wait for journal acceptance for visibility |
| J2 | Spec-decode failure analysis OR small draft win | vLLM plugin + trained draft beating Pareto | No huge RL until failure analysis clear |
| C2 | Workshop demo | Code + demo video | Not a new thesis |
| J3 | Token-cost + cost/quality/latency audit | Full Indic benchmark + deployment guide | Indic not whole spine unless supervisor redirects |

Full table: [10_APPENDICES_REFERENCE.md](10_APPENDICES_REFERENCE.md#appendix-e-minimum-vs-ambitious).
