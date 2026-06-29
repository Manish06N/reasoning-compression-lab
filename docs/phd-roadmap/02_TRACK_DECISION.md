# Part 2 — All Tracks Considered and Why Track A Was Chosen

[← Index](README.md)

---

## 3. All tracks considered

| Track | What it means | Why attractive | Why not spine | Final decision |
|-------|---------------|----------------|---------------|----------------|
| Citation faithfulness / RAG | Answers supported by cited sources | Low compute, standard metrics | Crowded; weak hardware edge | Fallback only |
| Adaptive test-time compute | Control reasoning length / overthinking | Fresh; cost-relevant | RL/method risk; journal path indirect | Absorbed into Paper 2 |
| Mechanistic interpretability | Internal features, steering | High prestige | Conference-native; too theoretical | Not spine |
| Agentic reliability | Long-horizon agent failures | Industry-visible | API cost, unstable benchmarks | Avoid as spine |
| Calibrated uncertainty / selective prediction | When to answer vs abstain | Journal-native, rigorous | Fewer hard systems skills alone | **Core methodology in Paper 1** |
| Semantic KV-cache management | Memory for long context | Strong systems angle | Low-level; risky as first paper | Pivot rung if scooped |
| Latent/compressed reasoning | Compress reasoning itself | High ceiling | High risk; conference-leaning | Do not choose now |
| Reasoning-aware speculative decoding | Draft model speeds target | Industry demand | Must beat baselines or fail-analysis | **Paper 2** |
| Continual knowledge editing | Update facts without retrain | Journal-friendly | Less aligned to deployment spine | Safe alternative |
| LLM evaluation metrology | Judge/benchmark reliability | Eval hiring signal | Fewer deployment skills as full spine | Methodology + C1 |
| Query routing / cascades | Easy→cheap, hard→strong model | Strong economics | Different thesis | Future extension |
| Cross-lingual token-cost inequity | Language token/cost differences | India-relevant | Weaker alone abroad | **Paper 3** |
| Fine-tuning data selection | Which examples matter | Cheap, useful | Less deployment-aligned | Not chosen |
| Post-training / RLVR | GRPO, verifiers, alignment | Frontier prestige | Conference-native, high compute | Paper 2 tool layer only |
| Multimodal / Indic / on-device | Local/multilingual deployment | India market fit | Weak as global spine alone | Paper 3 layer |
| **Efficient inference & deployment of reasoning models** | Compression + serving + reliability | Best constraint fit | Scope discipline needed | **Chosen spine** |

---

## 4. Why Track A / Track 1 was chosen

**Final track:** Efficient Inference and Deployment of Reasoning Models.

| Criterion | Why Track A wins |
|-----------|------------------|
| Market demand | Inference cost is real; hiring for serving, quantization, eval, systems |
| Journal fit | Empirical deployment studies with rigorous methodology |
| Hardware fit | 2× A100 + local GPU enough for inference, quant, profiling, small draft training |
| Artifact fit | Harness, HF traces, runbooks, leaderboards |
| Supervisor fit | Clear empirical plan with tables and reproducible outputs |
| Career fit | Research Engineer, Applied Scientist, ML Systems, Eval Scientist |
| Risk control | Pivot to calibration, seed variance, cost, KV-cache, Indic without restart |

> **Key decision:** The spine is **not** “quantization only.” It is **reliability and cost-efficient deployment**. Quantization is Paper 1’s first experimental object.

### How other tracks stay inside the thesis

- **Evaluation science** → statistics: calibration, selective risk, bootstrap CIs, extraction audits
- **Post-training** → Paper 2 draft-model training / distillation for speculative decoding
- **Indic/multilingual** → Paper 3 token-cost and deployment economics
- **Agents** → avoided for now (benchmark instability, API cost)
- **Interpretability / latent reasoning** → left out (high-risk, conference-native)

Full decision log table: [10_APPENDICES_REFERENCE.md](10_APPENDICES_REFERENCE.md#appendix-c-decision-log).
