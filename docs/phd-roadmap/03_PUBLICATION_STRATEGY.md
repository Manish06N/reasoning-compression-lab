# Part 3 — Publication Architecture and Q1/Q2 Strategy

[← Index](README.md)

---

## 6. The 3-journal + 2-conference strategy

Dual-track model: **journals for the degree**, **artifacts + workshops for visibility**.

| Output | Purpose | Likely source | Timing |
|--------|---------|---------------|--------|
| **J1** | Main journal; proves harness + core novelty | Paper 1: calibration/cost/variance of quantized reasoning | Month 6–12 |
| **C1** | Conference/workshop visibility | Single-run unreliability / multi-seed eval protocol | After J1 pilot |
| **J2** | Method or systems paper | Reasoning-aware speculative decoding or failure analysis | Year 2 |
| **C2** | Workshop derivative from J2 | Spec-decode failure analysis or deployment benchmark | Year 2 |
| **J3** | Application / deployment economics | Indic/multilingual serving + token-cost inequity | Year 2 → thesis end |

### How the papers connect

1. **J1** — measurement harness; compressed reasoning must be evaluated beyond accuracy  
2. **C1** — methodological insight: single-run evaluation is statistically unsafe  
3. **J2** — acceleration using J1 metrics (speculative decoding / draft models)  
4. **C2** — focused systems demo or failure taxonomy  
5. **J3** — token-cost / energy / recipes in multilingual settings  

### Q1/Q2 in plain English

Choose **journal-native** questions: measurement, systems evaluation, cost, reliability, reproducibility, deployment. Post strong preprints on arXiv; formal targets should accept rigorous empirical systems work. **Verify quartiles at submission time** — rankings change.

| Paper | Journal type to target | Why it fits |
|-------|------------------------|-------------|
| J1 | FGCS / JSS / Neurocomputing-type | Empirical systems + deployment economics + harness |
| J2 | JSS / EAAI / FGCS-type | Speculative decoding or failure-analysis systems study |
| J3 | FGCS / Sustainable Computing / LRE-type | Deployment economics, energy, Indic/multilingual |

Journal worksheet (verify before submit): [10_APPENDICES_REFERENCE.md](10_APPENDICES_REFERENCE.md#appendix-h-journal-strategy-worksheet).

---

## 9. Supervisor and institution strategy

**Do not delay these questions** — journal review timelines dominate the PhD clock.

| Question | Why it matters |
|----------|----------------|
| Accepted vs submitted/under-review for thesis? | Timeline risk |
| IIT Patna vs Lincoln University Malaysia rules? | Which institution binds |
| Q1/Q2: Scopus/SJR, JCR, CORE/ERA? | Target selection |
| Which year’s ranking: submission or acceptance? | Quartile claims |
| Does TMLR count as journal? | Pressure valve |
| Conference papers: separate or extras? | 3J + 2C plan |
| First-author / co-author rules? | Collaboration |
| arXiv / GitHub / HF before acceptance? | Artifact strategy |
| Ethics/IRB for benchmark-only work? | Admin blockers |
| APC / open-access funding? | Venue choice |

**Supervisor pack:** One-page proposal with spine, three-paper plan, hardware feasibility, publication targets, and the two blocking rule questions. Ask for **written** confirmation.

Meeting script: [10_APPENDICES_REFERENCE.md](10_APPENDICES_REFERENCE.md#appendix-g-supervisor-meeting-script).

Four-page supervisor version: [08_EXECUTION_VALIDATION_V6.md](08_EXECUTION_VALIDATION_V6.md#appendix-t-four-page-supervisor-version).

---

## Saturation and pivot rules

**Differentiator axes:**

1. Calibration / selective risk beyond accuracy  
2. Multi-seed variance (single-run unsafe)  
3. Cost-per-correct economics  
4. Deployment measurement (latency, VRAM, throughput, energy)  
5. Trace / stack transfer (vLLM vs llama.cpp)  

| Situation | Action |
|-----------|--------|
| No paper claims 3+ axes | **Proceed** with Paper 1 |
| Paper claims 1–2 axes | Cite and sharpen contribution |
| Paper claims 3+ axes | **Pivot within 2 weeks** (KV-cache × long trace, stack reproducibility, Indic pull-forward) |

Do **not** restart topic selection from zero; reuse the harness.

Pivot decision tree: [08_EXECUTION_VALIDATION_V6.md](08_EXECUTION_VALIDATION_V6.md#appendix-q-scoop-pivot-decision-tree).
