# V6 — Execution Validation Pack

[← Index](README.md)

**Evidence boundary:** Literature saturation tables here are **starting points**. Live-verify titles, venues, years, URLs, and quartiles before submission.

---

## Appendix K — Literature saturation matrix

| Bucket | ~Papers in scan | What it proves | Risk to P1 | How P1 differentiates |
|--------|-----------------|----------------|------------|------------------------|
| Quantization of reasoning | 8+ | Accuracy/low-bit failure crowded; Liu/COLM 2025 baseline | High if accuracy-only | Beyond accuracy: calibration, selective risk, seed variance, cost-per-correct |
| Calibration of reasoning LLMs | 13+ | Calibration growing; mostly general methods | Medium | Compression × calibration × **deployment** |
| Selective prediction / abstention | 10+ | Reasoning-native abstention emerging | Medium for P2; low if metric-only in P1 | AURC as eval layer, not new method |
| Speculative decoding for reasoning | 12+ | Paper 2 area hot | High for pure method | Pre-commit failure analysis fallback |
| Multilingual token-cost inequity | 3+ core | Indic angle attractive | Low–medium | Paper 3 layer |
| Energy/cost in LLM serving | 14+ | J/token metrics expected | Medium if energy-only novelty | Energy as deployment metric for reasoning/compression |

### Read-first literature path

| Order | Read | Action after |
|-------|------|--------------|
| 1 | Quantization Hurts Reasoning? / Liu baseline | Reproduce qwen7b GPTQ-W4 MATH-500 |
| 2 | Low-bit degradation diagnosis papers | Design extraction audit + failure labels |
| 3 | Reasoning calibration papers | Frame novelty as compression × calibration × deployment |
| 4 | Selective prediction papers | Use AURC/risk-coverage as metrics |
| 5 | Spec-decode for reasoning | Prepare P2 baselines + failure backup |
| 6 | Multilingual token-cost papers | Paper 3 rationale |
| 7 | Energy/cost serving papers | NVML caveats, repeated measurements |

### Novelty decision rule

**Proceed** if no single paper covers **3+** of:

1. Multi-seed statistical protocol  
2. Calibration/selective risk on sampled chains  
3. Cost/latency/energy economics  
4. Datacenter-vs-edge / stack sensitivity  
5. Released traces for reproducibility  

---

## Appendix L — Exact Paper 1 experiment specification

| Component | Minimum J1 | Ambitious | Freeze rule |
|-----------|------------|-----------|-------------|
| Models | qwen1_5b, qwen7b, llama8b | +14B/Qwen3 | Freeze at sign-off; max one addition |
| Tasks | MATH-500, GPQA, LCB subset, GSM8K | +MMLU-Redux, AIME exploratory | No headline AIME |
| Quant | bf16, fp8, awq, gptq4, gptq3 | +GGUF Q8/Q4/Q3 edge | No SmoothQuant unless pivot |
| Seeds | 5 | 10–20 for calibration subset only | Never headline one seed |
| Stack | vLLM on 2×A100 | SGLang; llama.cpp edge | Pin version + git SHA |
| Metrics | pass@1, maj@5, Brier, ECE-adaptive, AURC, latency, VRAM, cost-per-correct | J/correct, trace-flip, truncation | Pre-register primaries |
| Artifacts | GitHub + scored outputs | HF traces + leaderboard | Release after extraction audit |

Full detail: [04_PAPER1_FULL_SPEC.md](04_PAPER1_FULL_SPEC.md).

---

## Appendix M — Expected Paper 1 figures and tables

See [04_PAPER1_FULL_SPEC.md](04_PAPER1_FULL_SPEC.md#expected-figures-and-tables-paper-1).

---

## Appendix N — Journal target matrix (worksheet)

| Journal | Likely fit | Must verify | Desk-reject risk |
|---------|------------|-------------|------------------|
| Future Generation Computer Systems | Systems/deployment/cost | Quartile, APC, scope | Needs strong systems framing |
| Journal of Systems and Software | Reproducible software/system study | Quartile, artifact policy | Needs engineering contribution |
| Neurocomputing | Broad empirical ML | Quartile, novelty bar | May reject if too engineering-only |
| Engineering Applications of AI | Method + deployment utility | Quartile, review speed | Needs application framing |
| Sustainable Computing | Energy, J/correct | Quartile | Energy must be central |
| Information Processing & Management | Eval/reliability | Quartile | Measurement-science framing |
| TMLR | Journal-like review venue | **Institution acceptance** | Ask supervisor |

**Do not assert quartiles from memory.**

---

## Appendix O — Compute budget and execution gating

| Block | Models | Configs | Tasks | Seeds | Purpose |
|-------|--------|---------|-------|-------|---------|
| Reproduction | 1 | 2 | MATH-500 | 1 | Validate harness |
| Pilot | 1 | 3 | 1–2 | 3 | Debug pipeline |
| J1 minimum | 3 | 5 | 4 | 5 | Submit-worthy study |
| J1 ambitious | 3–5 | 8 | 5–6 | 5 | Edge/energy/stack |
| J2 pilot | 1 target + 1 draft | — | 2 | 3 | Spec-decode test |
| J3 pilot | 2–3 multilingual | 2–3 | Indic tasks | 3 | Token-cost story |

**Budget rule:** If pilot cost/cell &gt;50% over estimate → shrink grid. **Do not cut seeds first.**

---

## Appendix P — Supervisor approval tracker

| Question | Status |
|----------|--------|
| Accepted vs submitted/under-review required? | **Pending written reply** |
| Ranking system: Scopus/SJR, JCR, CORE? | Pending |
| Does TMLR count as journal? | Pending |
| Conference papers: separate or extras? | Pending |
| First-author / authorship rule? | Pending |
| arXiv/GitHub/HF release allowed? | Pending |
| Ethics/IRB for benchmark-only? | Pending |
| APC / open-access funding? | Pending |
| Compute/storage beyond current cluster? | Pending |

---

## Appendix Q — Scoop / pivot decision tree

| Trigger | Decision |
|---------|----------|
| Accuracy-only quant study scooped | Shift to calibration + selective risk + cost-per-correct (**current plan**) |
| Calibration under quant crowded | Shift to seed variance + reproducibility crisis |
| Energy noisy/unavailable | Energy → Paper 3; P1 = A100 cost/calibration minimum |
| RTX 5080 unstable | Datacenter minimum first; edge = extension |
| GPTQ-3 fails | Report as low-bit failure or replace config |
| Spec-decode gains weak | Publish acceptance-rate failure taxonomy |
| Supervisor requires accepted papers | Submit J1 earlier + fast Q2 target |

---

## Appendix R — Artifact architecture

```
reasoning-compression-lab/
  configs/          models, tasks, cells, machine_split
  prompts/
  scripts/          run, score, analyze, build_paper_tables, build_repro_bundle
  src/
    runners/        vLLM / future SGLang / llama.cpp
    metrics/        accuracy, calibration, selective risk, cost
    extraction/
    stats/          bootstrap, McNemar, seed variance
  runs/             raw → scored pipeline
  results/          summaries, paper_tables
  docs/             runbook, phd-roadmap, design
```

| Release | Artifact | When |
|---------|----------|------|
| v0.1 | Repo skeleton + selftest + runbook | Before first GPU cell |
| v0.2 | Reproduction results | After Liu cell |
| v0.3 | Pilot traces + calibration table | After pilot |
| v1.0 | Full J1 harness + data card | At arXiv/journal submission |

---

## Appendix S — Failure taxonomy

See [04_PAPER1_FULL_SPEC.md](04_PAPER1_FULL_SPEC.md#failure-taxonomy-detection).

---

## Appendix T — Four-page supervisor version

**Page 1 — Spine:** Reliable cost-efficient deployment of reasoning LLMs; compression helps but accuracy alone insufficient; measure calibration, reliability, cost, reproducibility.

**Page 2 — Paper 1:** Beyond accuracy under quantization; 3×5×4×5 on 2×A100 vLLM; pilot/repro gates before full grid.

**Page 3 — Three-paper plan:** J1 reliability/cost; C1 protocol; J2 spec-decode; C2 demo; J3 Indic economics.

**Page 4 — Feasibility:** P1 inference-heavy on 2×A100; P2 small draft or failure analysis; P3 measurement-led. Risks: publication rules, review latency, scoop, extraction. **Requests:** written rules, TMLR, arXiv policy, ethics, review cadence.

---

## Appendix U — Stop-expanding rule

After V6, roadmap is complete enough. Next document = **Paper 1 Design Doc v1 with reproduction numbers**, not V7+.

(V7 adds the job-first control layer — see [07_V7_JOB_FIRST_STRATEGY.md](07_V7_JOB_FIRST_STRATEGY.md).)
