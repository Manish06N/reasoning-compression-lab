# PhD Roadmap Master Report

**From Beginner-Level Understanding to A100 Execution**

*Reliable and Cost-Efficient Deployment of Reasoning LLMs under Compression, Evaluation, and Multilingual Constraints*

**Prepared for:** Manish Nandish  
**Version:** 5 + V6 execution validation + V7 job-first control layer + stack-transfer extension + 2026-08-14 publication-recovery amendment
**Repo:** [reasoning-compression-lab](https://github.com/Manish06N/reasoning-compression-lab)  
**Live execution log:** [progress.md](../progress.md) (what actually ran on HPC — read alongside this doc)

> **Controlling amendment (2026-08-14):** Paper 1 is **Needs revision**. Completed FP8 jobs 96100/96101 are valid replication/control evidence, not a publishable core. [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md) controls scientific interpretation and [plans/2026-08-14-publication-recovery.md](plans/2026-08-14-publication-recovery.md) controls execution. Any older instruction to expand the seed-0 b01–b09 grid is superseded.

---

## How to use this document

1. **New to the topic** → read [§1 Beginner explanation](#1-beginner-explanation-what-problem-are-we-solving)
2. **Why this track** → [§3–4 Tracks and decision](#3-all-tracks-considered)
3. **Publication plan** → [§6–9 Three papers and journals](#6-the-3-journal--2-conference-strategy)
4. **Paper 1 experiments** → [§7 Paper 1 design](#7-paper-1-design)
5. **Running on cluster** → [§10 Execution plan](#10-90-day-execution-plan) + [progress.md](../progress.md)
6. **After vLLM grid — GGUF/edge** → [Stack-transfer extension](#stack-transfer-extension-gguf-kv-cache-backends)

**Control rule:** preserve the thesis spine, but update factual gates when evidence changes. The next work is recovery Phase 0 and the matched pilot—not another broad roadmap or topic search.

---

## Table of contents

1. [Beginner explanation](#1-beginner-explanation-what-problem-are-we-solving)
2. [Your constraints](#2-your-constraints-and-why-they-shape-the-topic)
3. [All tracks considered](#3-all-tracks-considered)
4. [Why Track A was chosen](#4-why-track-a--track-1-was-chosen)
5. [Thesis spine](#5-how-the-thesis-spine-works)
6. [3-journal + 2-conference strategy](#6-the-3-journal--2-conference-strategy)
7. [Paper 1 design](#7-paper-1-design)
8. [Paper 2 and Paper 3](#8-paper-2-and-paper-3-strategy)
9. [Supervisor and Q1/Q2 strategy](#9-supervisor-and-institution-strategy)
10. [90-day execution plan](#10-90-day-execution-plan)
11. [Skills roadmap](#11-skills-roadmap)
12. [Job-market mapping](#12-job-market-mapping)
13. [Risk register and pivot rules](#13-risk-register-and-pivot-rules)
14. [Immediate next actions](#14-immediate-next-actions)
15. [Appendix B — Glossary](#appendix-b-beginner-glossary)
16. [Appendix C — Decision log](#appendix-c-decision-log)
17. [Appendix D — Thesis logic map](#appendix-d-paper-by-paper-thesis-logic-map)
18. [Appendix E — Minimum vs ambitious](#appendix-e-minimum-vs-ambitious-versions)
19. [Appendix F — First 30-day checklist](#appendix-f-first-30-day-exact-checklist)
20. [Appendix G — Supervisor script](#appendix-g-supervisor-meeting-script)
21. [Appendix H — Journal worksheet](#appendix-h-journal-strategy-worksheet)
22. [Appendix I — Reviewer objections](#appendix-i-reviewer-objections)
23. [Appendix J — Public artifacts](#appendix-j-public-artifact-release-plan)
24. [Appendix K — What not to do](#appendix-k-what-not-to-do)
25. [Appendix L — Command sheet](#appendix-l-one-page-command-sheet)
26. [V6 — Literature saturation](#v6-literature-saturation-matrix)
27. [V6 — Exact Paper 1 spec](#v6-exact-paper-1-experiment-specification)
28. [V6 — Figures, journals, compute budget](#v6-figures-journals-compute-budget)
29. [V6 — Supervisor tracker and pivot tree](#v6-supervisor-tracker-and-pivot-tree)
30. [V6 — Artifact architecture and failure taxonomy](#v6-artifact-architecture)
31. [V6 — Four-page supervisor version](#v6-four-page-supervisor-version)
32. [V7 — Job-first control layer](#v7-job-first-re-engineered-control-layer)
33. [Stack-transfer extension](#stack-transfer-extension-gguf-kv-cache-backends)

---

## 1. Beginner explanation: what problem are we solving?

This PhD plan is about making **reasoning language models** cheaper, more reliable, and more deployable. A reasoning LLM does not only produce a short answer; it often generates a **long reasoning trace** before the final answer (math, science MCQ, code, multi-step tasks).

**Central problem:** Strong reasoning models are expensive — GPU memory, many tokens, slow inference. Organizations compress them via **quantization** (fewer bits: 16-bit → 8-bit → 4-bit). That usually saves memory and cost, but can damage reliability in ways **invisible if you only check final accuracy**.

**Plain-English version:** The question is not only *“Does the compressed model still get the answer right?”* The better question is: *“Does the compressed model stay trustworthy, calibrated, fast, cheap, and stable across repeated runs?”*

That is the research opening. Earlier work studied quantization vs accuracy. This PhD moves **beyond accuracy** into calibration, selective prediction, seed variance, latency, VRAM, and **cost-per-correct-answer**.

### Key terms (quick)

| Term | Simple meaning |
|------|----------------|
| Reasoning LLM | Multi-step reasoning before answering |
| Quantization | Fewer bits (BF16, FP8, GPTQ, AWQ, GGUF) |
| Calibration | Confidence matches correctness |
| Selective prediction | Answer when confident; abstain/escalate otherwise |
| Cost-per-correct | Money per **correct** answer, not per token |
| Seed variance | Results change across random seeds |
| A100 / RTX 5080 | Datacenter vs local GPU |

Full glossary: [Appendix B](#appendix-b-beginner-glossary).

---

## 2. Your constraints and why they shape the topic

| Constraint | Implication |
|------------|-------------|
| 3+ first-author Q1/Q2 journals + 2 conferences | Topic must be **journal-native** |
| ~2 years left | Paper 1 must submit fast |
| 2× A100 80GB, 48h SLURM | Inference + moderate training OK; pretraining / multi-week RL not |
| Distance mode, flexible supervisor | Strict operating system, public artifacts, self-driven execution |
| Career: GenAI labs / research engineering | Build inference, eval, deployment, quantization, profiling skills |
| India + abroad options | Globally legible spine; Paper 3 adds Indic/multilingual |

> **Verify in writing:** Accepted vs submitted papers? Q1/Q2 by Scopus/SJR, JCR, or CORE? Does TMLR count?

---

## 3. All tracks considered

| Track | Why attractive | Why not spine | Final decision |
|-------|----------------|---------------|----------------|
| Citation faithfulness / RAG | Low compute, standard metrics | Crowded | Fallback only |
| Adaptive test-time compute | Fresh; cost-relevant | RL/method risk | Absorbed into Paper 2 |
| Mechanistic interpretability | High prestige | Conference-native | Not spine |
| Agentic reliability | Industry-visible | API cost, unstable benchmarks | Avoid as spine |
| Calibrated uncertainty / selective prediction | Journal-native, rigorous | Fewer systems skills alone | **Core of Paper 1** |
| Semantic KV-cache management | Hardware-native | Risky as first paper | Pivot if scooped |
| Latent/compressed reasoning | High ceiling | High risk | Do not choose now |
| Reasoning-aware speculative decoding | Industry demand | Must beat baselines | **Paper 2** |
| Continual knowledge editing | Journal-friendly | Less deployment-aligned | Not chosen |
| LLM evaluation metrology | Eval hiring signal | Fewer deployment skills | Methodology + C1 |
| Query routing / cascades | Strong economics | Different thesis | Future extension |
| Cross-lingual token-cost inequity | India-relevant | Weak alone abroad | **Paper 3** |
| Fine-tuning data selection | Cheap, useful | Less deployment-aligned | Not chosen |
| Post-training / RLVR | Frontier prestige | Conference-native, high compute | Paper 2 tool layer only |
| Multimodal / Indic / on-device | India market fit | Weak as global spine | Paper 3 layer |
| **Efficient inference & deployment of reasoning models** | Best constraint fit | Needs scope discipline | **Chosen spine** |

---

## 4. Why Track A / Track 1 was chosen

**Final track:** Efficient Inference and Deployment of Reasoning Models.

| Criterion | Why Track A wins |
|-----------|------------------|
| Market demand | Inference cost is real; hiring for serving, quantization, eval, systems |
| Journal fit | Empirical deployment studies with rigorous methodology |
| Hardware fit | 2× A100 + local GPU for inference, quant, profiling, small draft training |
| Artifact fit | Harness, HF traces, runbooks, leaderboards |
| Career fit | Research Engineer, Applied Scientist, ML Systems, Eval Scientist |
| Risk control | Pivot to calibration, seed variance, cost, KV-cache, Indic without restart |

> **Key decision:** The spine is **not** “quantization only.” It is **reliability and cost-efficient deployment**. Quantization is Paper 1’s first experimental object.

**How other tracks stay inside the thesis:** Evaluation science → statistics; post-training → Paper 2 drafts; Indic → Paper 3; agents/interpretability/latent reasoning → avoided or deferred.

---

## 5. How the thesis spine works

**Thesis spine (frozen):**

> Reliable and Cost-Efficient Deployment of Reasoning LLMs under Compression, Evaluation, and Multilingual Constraints.

| Component | Which paper |
|-----------|-------------|
| Reliable (calibration, seed stability) | Paper 1 + C1 |
| Cost-efficient | Paper 1 + Paper 2 |
| Deployment (vLLM, llama.cpp, A100, edge) | All |
| Compression (BF16, FP8, AWQ, GPTQ, GGUF) | Paper 1 + Paper 3 |
| Evaluation (calibration, selective risk, seed variance) | Paper 1 + C1 |
| Multilingual (token-cost inequity) | Paper 3 |

**One paragraph:** Paper 1 asks whether compressed reasoning models remain reliable and economical beyond accuracy. Paper 2 asks whether reasoning generation can be accelerated without destroying reliability. Paper 3 asks how deployment lessons change in Indic/multilingual settings. Together: strong reasoning models are not useful if too expensive, unreliable, or inaccessible.

---

## 6. The 3-journal + 2-conference strategy

Dual-track: **journals for the degree**, **artifacts + workshops for visibility**.

| Output | Purpose | Timing |
|--------|---------|--------|
| **J1** | Main journal; harness + core novelty | Month 6–12 |
| **C1** | Workshop visibility from J1 methods | After pilot |
| **J2** | Speculative decoding or failure analysis | Year 2 |
| **C2** | Workshop from J2 | Year 2 |
| **J3** | Indic/multilingual deployment economics | Year 2 → thesis end |

**Connection:** J1 measures → C1 packages protocol → J2 accelerates → C2 demos → J3 applies to multilingual/Indic.

**Journal targets (verify quartile at submission):** FGCS, JSS, Neurocomputing-type (J1); JSS/EAAI/FGCS (J2); FGCS/Sustainable Computing/LRE (J3).

---

## 7. Paper 1 design

**Provisional title:** *Beyond Pass@1: Reliability–Cost Frontiers of Quantized Reasoning Models under Controlled Serving-Stack Shift*

**Core gap under validation:** quantized-reasoning accuracy, calibration, and cost have all received substantial study. The candidate contribution is their reasoning-specific interaction with termination/degeneration and controlled serving-stack shift, supported by matched trace-level evidence.

### Research questions

| RQ | Question |
|----|----------|
| RQ1 | Under a matched stack, do quantized formats change correctness, termination, or degeneration relative to BF16? |
| RQ2 | Do calibrated selective-risk and cost-per-correct rankings differ from pass@1 rankings? |
| RQ3 | Are those rankings stable across seeds 42–46? |
| RQ4 | Are serving-stack effects larger than quantization effects when factors are isolated? |

### Staged evidence plan (J1)

| Stage | Scope |
|-------|-------|
| Matched reconstruction | Qwen-7B + Llama-8B × {BF16, FP8} × MATH-500 × seed 42 |
| Discriminating pilot | Same models × {BF16, FP8, AWQ4, GPTQ4} × MATH-500 × seeds 42–44 |
| Contribution gate | Select quantization reliability–cost, controlled stack transfer, or negative-results artifact |
| Headline confirmation | Selected MATH-500 cells extended to seeds 42–46 |
| Gated breadth | GPQA-Diamond/GSM8K only after contribution approval |

Qwen-1.5B, GPTQ3, LiveCodeBench, GGUF/KV-cache, and larger models are extensions only. Historical HPC blocks b01–b09/seed 0 are engineering evidence and must not be mixed into the new protocol.

### Metrics

pass@1, predeclared maj@5 subset, Brier, ECE, AURC, finish/cap/loop/parse rates, latency distribution, throughput, peak VRAM, cost-per-correct, seed variance, and paired/bootstrap 95% CIs. Energy is reported only when measured, never as zero by default.

### Statistics

Three pilot seeds and five headline seeds; paired bootstrap on problems; paired McNemar vs BF16; seed-level variability; Holm correction; at least 200 stratified traces plus every flagged pathology.

### Paper 1 headline (V7)

> Candidate hypothesis: matched reliability, calibrated selective risk, and cost evidence can change deployment choices that appear equivalent under pass@1. This remains a hypothesis until the contribution gate passes.

Short repo summary: [PAPER1_DESIGN.md](PAPER1_DESIGN.md).

---

## 8. Paper 2 and Paper 3 strategy

### Paper 2 — Reasoning-aware speculative decoding

Draft model proposes tokens; target verifies. Reasoning traces are long and high-entropy — generic drafts may fail.

| Version | Contribution |
|---------|--------------|
| Minimum | Failure analysis: acceptance rates on reasoning traces |
| Ambitious | Train 0.5B–1.5B draft; vLLM/SGLang plugin |

Must be **industry-heavy:** profiling, small SFT→DPO artifact, fallback = failure taxonomy.

### Paper 3 — Indic / multilingual deployment economics

Token-cost inequity, cost-quality-latency-energy frontiers, serving recipes. **Application layer**, not whole thesis spine.

---

## 9. Supervisor and institution strategy

**Ask in writing before full grid:**

- Accepted vs submitted/under-review for thesis?
- IIT Patna vs Lincoln University Malaysia rules?
- Q1/Q2: Scopus/SJR, JCR, CORE?
- TMLR counts as journal?
- Conference papers: separate or extras?
- First-author rules; arXiv/GitHub/HF before acceptance?
- Ethics/IRB for benchmark-only work?
- APC funding?

See [Appendix G — Supervisor script](#appendix-g-supervisor-meeting-script).

---

## 10. 90-day execution plan

| Week | Focus | Gate |
|------|-------|------|
| 1–2 | Recovery P0: pin/track dependencies; extend schema and validators | Clean recreation + tests |
| 3 | Controlled telemetry and tiny instrumented smoke | Complete valid smoke rows |
| 4 | Four matched BF16/FP8 seed-42 cells | P1 audit |
| 5–6 | 24-cell, three-seed MATH-500 pilot | Contribution-selection report |
| 7 | Literature refresh + supervisor contribution gate | Track/RQs approved |
| 8–10 | Five-seed headline confirmation | Complete primary cells |
| 10–11 | Gated breadth and maj@5/calibration subset | Endpoints valid |
| 12 | Frozen analysis + 200-trace audit | Reproducible figures/tables |
| 13 | Draft and artifact review | Candidate manuscript |

**V7 artifact-first variant:** Monthly public ship (repo → blog → HF pilot → dashboard → arXiv → journal).

---

## 11. Skills roadmap

| Phase | Skills |
|-------|--------|
| Now | Linux, SLURM, Conda, Git, HF, PyTorch, vLLM |
| Paper 1 | GPTQ/AWQ/FP8, extraction, calibration, bootstrap/McNemar, NVML |
| Paper 2 | Speculative decoding, SGLang, TRL SFT/DPO |
| Paper 3 | Indic benchmarks, tokenization/fertility, energy |
| Job premium | CUDA/Triton, PRs to vLLM/lm-eval |

---

## 12. Job-market mapping

| Role | Your evidence |
|------|---------------|
| Research Engineer | qreason harness, reproducible runs, HF dataset |
| Applied Scientist | cost-per-correct, Pareto guidance |
| ML Systems Engineer | vLLM profiling, latency/VRAM, quantization |
| LLM Evaluation Scientist | calibration protocol, seed variance, extraction audit |
| AI Infrastructure | Runbook, SLURM, cost model |
| India roles | J3 Indic deployment economics |

**Rule:** For jobs, **artifacts** are the product; for the degree, **papers** are the product. Same experiments must produce both (V7).

---

## 13. Risk register and pivot rules

| Risk | Mitigation |
|------|------------|
| Accepted vs submitted rule unknown | Written answer now |
| Engineering quicksand | Smoke tests, minimum grid first |
| Second scoop | Weekly saturation scan; pivot rungs |
| A100 instability | Pilot first; resume-safe SLURM |
| Extraction errors | 200-trace audit |
| Scope creep | Minimum J1; edge/energy → J3 |
| Journal delay | Q1 target + Q2 fallback |

**Saturation decision (2026-08-14):** current work now occupies several of the former novelty axes, so the broad “3+” rule has triggered. Do not abandon the thesis spine; narrow Paper 1 through the contribution gate. Preferred differentiators are trace-level termination/degeneration under matched quantization and controlled serving-stack shift. If the pilot shows no defensible effect, release a replication/negative-results artifact and re-scope with the supervisor.

---

## 14. Immediate next actions

1. Preserve and sync the 2026-08-14 audit/plan documentation.
2. Complete recovery Phase 0 without broad GPU use.
3. Run only the four matched BF16/FP8 seed-42 cells after P0.
4. Review them before authorizing the 24-cell pilot.
5. Use the pilot and current literature to select one contribution with the supervisor.
6. Expand to five seeds and breadth only after approval.

**Stop-planning rule:** Next document = **Paper 1 Design Doc v1 with reproduction numbers**, not another roadmap version.

---

# Appendices

## Appendix B — Beginner glossary

| Term | Meaning |
|------|---------|
| Reasoning LLM | Long trace before final answer |
| Quantization | Fewer bits (BF16, FP8, GPTQ, AWQ, GGUF) |
| Calibration | Confidence matches correctness |
| Selective prediction | Answer when confident; abstain otherwise |
| Brier / ECE / AURC | Calibration and selective-risk metrics |
| pass@1 / maj@5 | Single-sample / majority-vote accuracy |
| Seed variance | Results change across seeds |
| vLLM / SGLang / llama.cpp | Serving backends |
| Cost-per-correct | Cost per correct answer |
| KV-cache | Attention memory during generation |

---

## Appendix C — Decision log

Final spine: **reliable cost-efficient deployment of reasoning LLMs**. Quantization = Paper 1 object. Spec-decode = Paper 2. Indic economics = Paper 3. Full track table: [§3](#3-all-tracks-considered).

---

## Appendix D — Paper-by-paper thesis logic map

| Stage | Question | Output |
|-------|----------|--------|
| J1 | What happens when reasoning models are compressed? | Harness, calibration/cost tables |
| C1 | Quick visibility on eval protocol? | Workshop, arXiv |
| J2 | Can inference be faster without hurting quality? | Draft method or failure study |
| C2 | Systems demo? | Workshop + plugin |
| J3 | Lessons for Indic/multilingual/edge? | Benchmark, recipes, HF dataset |

---

## Appendix E — Minimum vs ambitious versions

| Paper | Minimum | Ambitious |
|-------|---------|-----------|
| J1 | 3×5×4×5; A100; calibration+cost+variance | Edge/GGUF, 14B, energy, stack transfer |
| C1 | Protocol short paper | Leaderboard |
| J2 | Spec-decode failure analysis | Trained draft + plugin |
| J3 | Token-cost audit | Full Indic benchmark |

---

## Appendix F — First 30-day exact checklist

| Days | Task | Pass |
|------|------|------|
| 1–2 | Conda env | Imports OK |
| 3 | CPU selftest | PASS |
| 4 | validate_tasks | MATH-500 OK |
| 5–6 | make/validate cells | PASS |
| 7 | Mock smoke | PASS |
| 8–10 | Download Qwen-7B | Weights exist |
| 11–13 | Tiny BF16 smoke (5 Q) | 5 rows |
| 14–18 | Quantize GPTQ-W4 | Artifact exists |
| 19–22 | BF16 + GPTQ repro cells | Two JSONLs |
| 23–25 | extract → score → calibrate | CSV rows |
| 26–28 | Compare to Liu et al. | Tolerance or diagnosis |
| 29–30 | Supervisor packet | Approval for pilot |

---

## Appendix G — Supervisor meeting script

**Opening:** Freeze thesis around cost-efficient reliable deployment of reasoning LLMs — journal-feasible, hardware-feasible, industry-relevant.

**Paper 1:** Not accuracy-only; beyond accuracy — calibration, selective risk, seed variance, cost-per-correct.

**Three papers:** P1 compression/reliability; P2 acceleration; P3 Indic economics.

**Need written:** accepted vs submitted, ranking source, TMLR, arXiv policy, authorship.

**Ask:** Proceed with 30-day reproduction/pilot; return with numbers before full grid.

---

## Appendix H — Journal strategy worksheet

Verify quartile at submission from institution’s database. Candidates: FGCS, JSS, Neurocomputing, EAAI, Sustainable Computing, IP&M, TMLR (confirm counting rule). **Submit J1 when minimum is complete — do not wait for ambitious extensions.**

---

## Appendix I — Reviewer objections

| Objection | Response |
|-----------|----------|
| Only empirical | Rigorous deployment science + traces + multi-seed + cost/calibration |
| Quantization studied | Cite; pivot beyond accuracy |
| Too many metrics | Pre-register primaries |
| One seed | Part of claim — use 5 seeds |
| Extraction wrong | math-verify + 200-trace audit |
| Negative results | Publishable if rigorous and actionable |

---

## Appendix J — Public artifact release plan

| Month | Artifact |
|-------|----------|
| 1 | GitHub skeleton, runbook |
| 2 | Replication blog |
| 3 | Harness v0.1 |
| 4 | Pilot report |
| 5–6 | HF trace dataset |
| 6–8 | Dashboard / leaderboard |
| 8–10 | Workshop / arXiv |
| 10–12 | Journal package |

**Rule:** One public artifact per month — small but real.

---

## Appendix K — What not to do

1. No new topics before BF16/GPTQ reproduction completes.  
2. No full grid before pilot passes.  
3. No RLVR/agents in Paper 1.  
4. No chasing every new model — freeze list at sign-off.  
5. Indic is not the whole spine.  
6. GPTQ-3 failure = document as result.  
7. No energy claims without clean measurement.  
8. No single-seed conclusions.  
9. No custom framework — use vLLM, llama.cpp, lm-eval.  
10. No months of repo polish before results.  
11. Confirm supervisor rules in writing early.  
12. Verify journal quartile before submit.  
13. No 5080 publication long runs — HPC-only for paper numbers.  
14. No GGUF extension before scoring current vLLM grid.

---

## Appendix L — One-page command sheet

```bash
# PARAM Rudra (current repo)
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh && conda activate qreason

python scripts/hpc/07_preflight_publication.py
sbatch slurm/smoke_test_quick_exclusive.slurm
bash scripts/hpc/submit_hpc_blocks.sh b01

python scripts/score_run.py --input <archive>/raw/<cell>.jsonl
python scripts/build_paper_tables.py --archive outputs-hpc-2a100-main-YYYY-MM-DD
python scripts/build_repro_bundle.py --archive outputs-hpc-2a100-main-YYYY-MM-DD
```

---

# V6 — Execution validation pack

**Evidence boundary:** Literature counts and quartiles are planning worksheets — live-verify before submission.

## V6 — Literature saturation matrix

| Bucket | Risk to P1 | Differentiate by |
|--------|--------------|------------------|
| Quantization of reasoning (8+) | High if accuracy-only | Matched termination/degeneration and controlled stack shift |
| Calibration of reasoning (13+) | High after ACL 2025 quantized-calibration work | Reasoning-specific confidence + selective deployment only if pilot differentiates |
| Selective prediction (10+) | Medium for P2 | AURC as eval layer |
| Speculative decoding (12+) | High for pure method | Failure analysis fallback |
| Multilingual token-cost (3+) | Low–medium | Paper 3 |
| Energy/cost serving (14+) | Medium if energy-only | Energy as deployment metric |

**Novelty amendment:** the former “3+” threshold is met by the current landscape. The project is now in contribution-selection mode; [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md) records the current sources and claims boundary.

## V6 — Exact Paper 1 experiment specification

See [§7 Paper 1 design](#7-paper-1-design). Freeze models/tasks/configs at supervisor sign-off. Pin vLLM version + git SHA.

## V6 — Figures, journals, compute budget

**Figures:** grid table, accuracy vs quant, calibration vs quant, cost-per-correct Pareto, risk-coverage, seed variance, latency/VRAM, failure taxonomy, extraction audit.

**Compute gating:**

| Block | Purpose |
|-------|---------|
| Recovery P0 | Clean reproducibility and observability before GPU expansion |
| Matched P1 | 2 models × 2 formats × 1 seed |
| Pilot P2 | 2 models × 4 formats × 3 seeds = 24 cells |
| Headline confirmation | Extend selected MATH cells to 5 seeds |
| Breadth | Only after contribution gate |

If pilot cost/cell &gt;50% over estimate → shrink grid. **Do not cut seeds first.**

## V6 — Supervisor tracker and pivot tree

| Question | Status |
|----------|--------|
| Accepted vs submitted? | Pending |
| Ranking source? | Pending |
| TMLR counts? | Pending |
| arXiv/HF allowed? | Pending |

**Pivot triggers:** accuracy-only scooped → current plan; calibration crowded → seed variance; energy noisy → Paper 3; GPTQ-3 fails → report failure; spec-decode weak → failure taxonomy.

## V6 — Artifact architecture

```
reasoning-compression-lab/
  configs/  prompts/  scripts/  src/  runs/  results/  docs/
```

Releases: v0.1 skeleton → v0.2 reproduction → v0.3 pilot → v1.0 journal package.

## V6 — Four-page supervisor version

1. **Spine:** Reliable cost-efficient deployment; measure beyond accuracy.  
2. **Paper 1:** 3×5×4×5 on A100; repro/pilot gates first.  
3. **Three papers:** J1 reliability; J2 acceleration; J3 Indic economics.  
4. **Feasibility + requests:** Written publication rules, TMLR, arXiv, ethics.

---

# V7 — Job-first re-engineered control layer

V6 = evidence base. **V7 = controlling priority layer** — same topic, reweighted execution.

| V6 | V7 change |
|----|-----------|
| PhD roadmap | Degree strategy **+** job strategy |
| Artifacts support papers | **Artifacts = main hiring product** |
| Paper 1 compression study | Tighter: single-run unreliability + calibration + cost |
| Paper 2 spec-decode | Industry-heavy: vLLM/SGLang, draft, SFT/DPO |
| Skills optional CUDA | CUDA/Triton + OSS PRs = strategic |

**Thesis identity:** Reliable **cost-efficient deployment** — not “quantization only.”

**V7 dashboard:**

| Decision | Status |
|----------|--------|
| Spine | Frozen |
| Paper 1 | Needs revision; matched reliability/stack contribution gate |
| Paper 2 | Industry-heavy acceleration |
| Paper 3 | Indic economics |
| Job strategy | Artifact-first |
| Next | Recovery Phase 0, then four matched BF16/FP8 cells |

**Do not change:** spine (unless supervisor vetoes), no RLVR/agents as spine, no Paper 4 before P1 submit, no custom framework before repro cell.

---

# Stack-transfer extension (GGUF, KV-cache, backends)

**When:** Only if the three-seed contribution gate selects controlled stack transfer or after the Track A core is complete. Do not execute the historical broad vLLM grid first by default.

| Layer | Current (minimum) | Ambitious |
|-------|-------------------|-----------|
| Question | vLLM × BF16/FP8/AWQ/GPTQ effects | Same conclusions on llama.cpp/GGUF? |
| Weights | BF16, FP8, AWQ, GPTQ | + GGUF Q8, Q6, Q4, Q3 |
| Runtime | vLLM (A100) | + llama.cpp + SGLang |
| Memory | Weights | Weights + **KV-cache** quant |

### Priority 1 — GGUF (llama.cpp)

| Format | Role |
|--------|------|
| Q8_0 | Near-reference local |
| Q6_K / Q6_K_M | Quality/size balance |
| Q4_K_M | Main 4-bit local baseline |
| Q3_K_M | vs GPTQ-3 |

**First grid:**

| Model | GGUF formats |
|-------|--------------|
| Qwen-7B | Q8_0, Q6_K, Q4_K_M, Q3_K_M |
| Llama-8B | Q8_0, Q6_K, Q4_K_M |
| Qwen-1.5B | Q8_0, Q4_K_M |

Tasks: MATH-500 + GSM8K (7B/8B); MATH-500 only (1.5B). Seeds: 3 then 5.

### Priority 2 — KV-cache quantization

| KV mode | Context |
|---------|---------|
| FP16/BF16 KV | Reference |
| FP8 KV | vLLM / A100 |
| q8_0 / q4_0 KV | llama.cpp (if stable) |

Long reasoning traces → KV memory matters.

### Priority 3 — Backend comparison

Same model + task, different stack: vLLM GPTQ-4 vs AWQ-4 vs llama.cpp GGUF Q4_K_M vs SGLang FP8.

**Publishable claim:** Deployment decisions depend on **both** compression format **and** serving stack.

### Priority 4 — Qwen-14B (after core stable)

BF16, FP8, AWQ/GPTQ-4 on A100; GGUF Q4 exploratory on edge. No 32B/70B unless supervisor requests.

### Priority 5 — Energy (Joules-per-correct)

Secondary unless clean measurement. Strong for Paper 3 or ambitious appendix — **do not block J1**.

### What NOT to add immediately

SmoothQuant, NF4, ONNX, TensorRT-LLM (optional later), EXL2, every GGUF variant.

### Recommended order after vLLM grid

1. GGUF Q8_0  
2. GGUF Q4_K_M  
3. GGUF Q6_K  
4. GGUF Q3_K_M  
5. KV-cache quant  
6. SGLang compare  
7. Qwen-14B if budget allows  

**First extension:** Qwen-7B GGUF Q8 + Q4 + Q3 — cleanest next step.

**Research question:** If GPTQ-4 wins in vLLM, does GGUF Q4_K_M win on llama.cpp — or does the ranking **change**?

---

*End of PhD Roadmap Master Report.*
