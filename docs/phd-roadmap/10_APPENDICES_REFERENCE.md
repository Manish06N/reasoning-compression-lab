# Appendices — Glossary, Checklists, Scripts, and Rules

[← Index](README.md)

---

## Appendix B — Beginner glossary

| Term | Simple meaning | Why it matters |
|------|----------------|----------------|
| Reasoning LLM | Long reasoning trace before final answer | Costly generation |
| Reasoning trace / CoT | Intermediate solving text | Latency, tokens, energy |
| Quantization | Fewer bits per weight (BF16→FP8→4-bit) | Memory/cost vs reliability |
| BF16 | 16-bit reference baseline | Comparison point |
| FP8 | 8-bit float serving mode | Datacenter speed |
| GPTQ | Post-training weight quant (W4, W3) | Common deployment format |
| AWQ | Activation-aware 4-bit quant | Strong inference option |
| GGUF / llama.cpp | Local/edge format + runtime | Stack transfer studies |
| Calibration | Confidence matches correctness | Beyond-accuracy wedge |
| Selective prediction | Answer when confident; abstain otherwise | Safe deployment |
| Brier score | Calibration metric; penalizes confident errors | Primary calibration metric |
| ECE | Expected calibration error across bins | Confidence gap |
| AURC | Area under risk-coverage curve | Selective prediction quality |
| pass@1 | Single-sample accuracy | Standard metric |
| maj@5 | Majority vote over 5 samples | Self-consistency |
| Seed variance | Results change across seeds | Single-run unreliability |
| vLLM | Datacenter LLM serving engine | Paper 1 main backend |
| SGLang | Alternative serving framework | Backend comparison |
| lm-eval-harness | Standard eval framework | Benchmark discipline |
| NVML / pynvml | GPU memory/power APIs | Profiling / energy |
| Cost-per-correct | Cost per **correct** answer | Industry economics |
| A100 | Datacenter GPU (80GB) | Main PhD compute |
| RTX 5080 | Local consumer GPU (16GB) | Edge/debug only (retired for publication) |
| KV-cache | Attention memory during generation | Long reasoning → memory pressure |

---

## Appendix C — Decision log

See full track table: [02_TRACK_DECISION.md](02_TRACK_DECISION.md).

**Final spine:** Reliable and cost-efficient deployment of reasoning LLMs.  
**Paper 1 object:** Quantization under deployment constraints.  
**Paper 2:** Reasoning-aware acceleration.  
**Paper 3:** Indic/multilingual deployment economics.

---

## Appendix D — Paper-by-paper thesis logic map

See [05_PAPER2_AND_PAPER3.md](05_PAPER2_AND_PAPER3.md#paper-by-paper-thesis-logic-map).

---

## Appendix E — Minimum vs ambitious versions

| Paper | Minimum | Ambitious | Scope rule |
|-------|---------|-----------|------------|
| J1 | 3×5×4×5; A100; calibration+cost+variance | Edge/GGUF, 14B, energy, stack transfer | No RLVR/agents in J1 |
| C1 | Protocol / seed variance short paper | Leaderboard | Don’t wait for journal for visibility |
| J2 | Spec-decode failure OR small draft win | vLLM plugin + trained draft | Failure analysis before huge RL |
| C2 | Workshop demo | Code + video | Not a new thesis |
| J3 | Token-cost + audit | Full Indic benchmark + recipes | Indic not whole spine |

---

## Appendix F — First 30-day checklist

See [06_EXECUTION_PLAN.md](06_EXECUTION_PLAN.md#appendix-f--first-30-day-exact-checklist).

---

## Appendix G — Supervisor meeting script

**Opening:**  
“Sir/Ma’am, I want to freeze my thesis spine around **cost-efficient and reliable deployment of reasoning LLMs**. It satisfies: Q1/Q2 journal feasibility, 2×A100 + local GPU feasibility, and industry demand in inference, deployment, evaluation, and AI infrastructure.”

**Why this track:**  
“I considered RAG faithfulness, evaluation science, agents, RLVR, interpretability, routing, multilingual AI, and data selection. I choose **efficient inference/deployment** for the strongest combination of journal feasibility, job-market skills, and public artifacts.”

**Paper 1 pitch:**  
“Paper 1 will not ask only whether quantization hurts accuracy — that is partially studied. It asks what happens **beyond accuracy**: calibration, selective risk, seed variance, latency, VRAM, and cost-per-correct.”

**Three-paper plan:**  
“Paper 1 measures compression and reliability; Paper 2 studies reasoning-aware acceleration; Paper 3 applies the framework to Indic/multilingual deployment economics.”

**What I need (written):**  
Accepted vs submitted papers; ranking source (Scopus/JCR/CORE); TMLR status; conference counting; authorship rules; arXiv/GitHub/HF policy.

**Decision request:**  
“If you agree, I proceed with the 30-day reproduction and pilot plan and return with numbers before full-grid approval.”

---

## Appendix H — Journal strategy worksheet

**Rule:** Verify quartile in the database your institution accepts **at submission time**.

| Journal | Why it might fit | Must verify |
|---------|------------------|-------------|
| Future Generation Computer Systems | Deployment, cost, systems, energy | Q rank, APC, scope, preprint policy |
| Journal of Systems and Software | Reproducible harness, software/system eval | Q rank, artifact policy |
| Neurocomputing | Empirical ML / quantized reasoning | Q rank, scope |
| Engineering Applications of AI | Applied AI systems | Q rank, special issues |
| Sustainable Computing | Energy-per-query central | Q rank, energy scope |
| Information Processing & Management | Evaluation / reliability | Q rank, fit |
| TMLR | Journal-like review | **Degree counting rule** |

**Strategy rule:** Submit Paper 1 when **minimum publishable version** is complete. Do not wait for ambitious extensions.

---

## Appendix I — Reviewer objections

| Objection | Response |
|-----------|----------|
| “Only empirical” | Rigorous deployment science + public traces + multi-seed + cost/calibration endpoints |
| “Quantization already studied” | Cite prior work; pivot to calibration, selective risk, cost-per-correct, seed variance |
| “Too many metrics” | Pre-register primaries; secondary/exploratory labeled |
| “Only small/open models” | Deployment relevance of servable 7B/8B class |
| “Energy noisy” | Repeated measurements; NVML caveats; wall-power validation |
| “Extraction wrong” | math-verify, official evaluators, 200-trace audit &lt;2% |
| “One seed not enough” | **Part of claim** — 5 seeds + variance reporting |
| “Too broad” | Fixed minimum grid; extras secondary |
| “Negative/small results” | Rigorous negative deployment findings publishable if actionable |

---

## Appendix J — Public artifact release plan

| Time | Artifact | Why |
|------|----------|-----|
| Month 1 | GitHub skeleton, runbook, selftest | Reproducibility signal |
| Month 2 | Replication blog (BF16/GPTQ) | Shows prior-art reproduction |
| Month 3 | Harness v0.1 | Portfolio for Research Engineer roles |
| Month 4 | Pilot report (3 configs × 3 seeds) | Early evidence |
| Month 5–6 | HF trace dataset | Reusable research artifact |
| Month 6–8 | Static leaderboard / dashboard | Industry legibility |
| Month 8–10 | Workshop / arXiv | Visibility during journal review |
| Month 10–12 | Journal submission package | Degree output |

**Public shipping rule:** One public artifact per month — small but real. For distance-mode PhD, GitHub + Hugging Face = your lab hallway.

---

## Appendix K — What not to do

1. Do not ask for more topics before qwen7b BF16/GPTQ reproduction completes.  
2. Do not run full 300-cell grid before pilot passes.  
3. Do not add RLVR, agents, or interpretability to **Paper 1**.  
4. Do not chase every new model release — freeze list at design sign-off.  
5. Do not make Indic/multilingual the **whole** spine unless supervisor vetoes Track A.  
6. Do not depend on GPTQ-3 working — failure is a result if documented.  
7. Do not claim energy without clean repeated measurement.  
8. Do not treat **one seed** as evidence.  
9. Do not build a new framework — use vLLM, llama.cpp, lm-eval, math-verify.  
10. Do not polish repo for months before generating results.  
11. Do not ignore supervisor-rule uncertainty — confirm in writing early.  
12. Do not submit without verifying current journal quartile rules.  
13. Do not restart 5080 publication long runs — **HPC-only** for paper numbers.  
14. Do not `git pull` on HPC mid active publication job without a plan.  
15. Do not expand to GGUF/stack transfer before scoring current vLLM grid.

---

## Appendix L — Command sheet (current repo)

```bash
# PARAM Rudra — publication batch
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh && conda activate qreason

python scripts/hpc/07_preflight_publication.py
sbatch slurm/smoke_test_quick_exclusive.slurm
bash scripts/hpc/submit_hpc_blocks.sh b01    # or b01-b06 when ready

# After cells complete
python scripts/score_run.py --input <archive>/raw/<cell>.jsonl
python scripts/build_paper_tables.py --archive outputs-hpc-2a100-main-YYYY-MM-DD
python scripts/build_repro_bundle.py --archive outputs-hpc-2a100-main-YYYY-MM-DD
```

MacBook sync: see [docs/RUNBOOK.md](../RUNBOOK.md) and [progress.md](../../progress.md).

---

## Final stop-planning rule

Once preflight and smoke pass, **stop editing the roadmap**. Execute reproduction and pilot. The next decision happens **after qwen7b BF16/GPTQ numbers exist**, not before.
