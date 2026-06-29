# V7 — Job-First Re-Engineered Control Layer

[← Index](README.md)

**Purpose:** Restructure V6 so the roadmap separates **degree requirements** from **employability**, front-loads public artifacts, and preserves the 3-paper Q1/Q2 spine.

**Recommendation:** V6 remains the comprehensive evidence base. **V7 is the controlling decision layer** — do not rewrite V6 from zero.

---

## 1. Executive decision: addendum vs rewrite

| Option | Verdict |
|--------|---------|
| Small appendix to V6 | Not enough — operating model changed |
| Complete rewrite | Wasteful — creates drift |
| **V7 as control layer** | **Best** — same topic, new execution priority |

**Final choice:** Topic **frozen**; execution priority **reweighted**.

---

## 2. What V7 changes vs V6

| Area | V6 | V7 correction |
|------|-----|---------------|
| Identity | PhD roadmap + validation pack | Dual-track: degree + job strategy |
| Artifacts | Support papers | **Main hiring product**; papers = formal evidence |
| Paper 1 | Compression/eval/cost study | Tighter: single-run unreliability + calibration + cost-per-correct |
| Paper 2 | Speculative decoding | **Industry-heavy:** vLLM/SGLang, draft models, small post-training |
| Paper 3 | Indic economics | India-market layer, not main spine |
| Skills | Broad roadmap | CUDA/Triton + OSS PRs = strategic signals |

---

## 3. Final thesis identity

| Layer | Meaning | Papers |
|-------|---------|--------|
| Compression | BF16, FP8, GPTQ, AWQ, GGUF, KV-cache | P1, P3 |
| Evaluation | Calibration, seed variance, selective risk | P1, C1 |
| Acceleration | Speculative decoding, draft models, serving | P2, C2 |
| Multilingual/Indic | Token-cost inequity, edge recipes | P3 |
| Artifacts | Harness, datasets, dashboards, plugins | Every phase |

---

## 4. Two strategies in parallel

| Strategy | Goal | Primary outputs | Risk if ignored |
|----------|------|-----------------|-----------------|
| **Degree** | PhD + Q1/Q2 journals | 3 journals, 2 conferences, thesis chapters | Good portfolio, failed milestones |
| **Job** | Legible to GenAI / infra / applied teams | GitHub, HF, harness, blogs, PRs, demos | Degree done, weak hiring signal |

**Operating principle:** For the degree, **papers** are the product. For jobs, **artifacts** are the product. Same experiments must produce both.

---

## 5. Re-engineered 3J + 2C (with timing)

> **Warning:** Safe only after supervisor confirms accepted vs submitted rules.

| Output | Focus | Degree | Job artifact | Target month |
|--------|-------|--------|--------------|--------------|
| J1 | Calibration, selective risk, seed variance, cost-per-correct under compression | Anchor | Harness + trace dataset | 9–12 |
| C1 | Short eval/benchmark version of J1 | Conf 1 | arXiv, workshop, repo traffic | 6–9 |
| J2 | Reasoning-aware acceleration / spec-decode | Journal 2 | vLLM/SGLang integration, draft model | 18–22 |
| C2 | System demo from J2 | Conf 2 | Industry-facing demo | 14–20 |
| J3 | Indic/multilingual deployment economics | Journal 3 | Indic dataset, cost dashboard | 24–30 |

---

## 8. Paper 2 must be industry-heavy

| Element | Required |
|---------|----------|
| Core | Reasoning-aware speculative decoding / draft acceleration |
| Post-training | Small SFT → DPO (or preference) artifact |
| Systems | vLLM/SGLang, profiling, KV-cache, optional plugin |
| Fallback | Acceptance-rate failure taxonomy |

---

## 10. Artifact-first operating model

| Month | Artifact | Paper link |
|-------|----------|------------|
| 1 | Public repo, runbook, reproduction log | P1 reproducibility |
| 2 | Calibration + seed-variance harness v0.1; blog | P1 methods |
| 3 | Pilot traces on HF; figures v0 | P1 pilot |
| 4–6 | Full P1 dataset + dashboard | J1 + C1 |
| 7–12 | Spec-decode / draft sandbox | P2 base |
| 13–18 | vLLM/SGLang demo or failure package | J2 + C2 |
| 19–24 | Indic token-cost dataset | P3 base |

**Hiring rule:** Recruiters read repo, dashboard, dataset, blog, PRs — **build early**.

Public release plan: [10_APPENDICES_REFERENCE.md](10_APPENDICES_REFERENCE.md#appendix-j-public-artifact-release-plan).

---

## 11. Revised skill priority stack

| Priority | Skills | When |
|----------|--------|------|
| Core now | Python, PyTorch, HF, SLURM, Git, configs, parquet | Weeks 1–4 |
| Paper 1 core | vLLM, GPTQ/AWQ/GGUF, extraction, calibration, bootstrap/McNemar | Weeks 1–12 |
| Systems credibility | NVML, Nsight basics, latency/VRAM profiling | Months 2–6 |
| Paper 2 core | SGLang, spec-decode, TRL SFT/DPO, KV-cache | Months 6–15 |
| Job premium | CUDA/Triton, PRs to vLLM/lm-eval | Months 6–18 |
| Side demo | Tool-using agent eval (weekend) | After P1 pilot |

---

## 12. Two small portfolio projects

| Project | Scope cap | Output |
|---------|-----------|--------|
| Mini post-training | One small model; SFT → DPO; not a full paper unless useful | Notebook + model card |
| Mini agent-eval demo | Simple tool agent; log cost/latency/failures | Small repo + blog |

---

## 14. Supervisor script (updated)

> “I want to freeze the thesis around **reliable and cost-efficient deployment of reasoning LLMs**. Academic goal: 3 Q1/Q2 journals and 2 conferences. Career goal: public artifacts showing systems, evaluation, and deployment competence. Paper 1 is inference-only and should be first submission. I need written confirmation on: accepted vs submitted, ranking source, TMLR, conference expectations, authorship.”

---

## 15. Updated risk register

| Risk | Mitigation |
|------|------------|
| Institution requires **accepted** Q1/Q2 | Submit J1 by month 9–12; Q2 fallback |
| Paper 1 axis scooped | Pivot to seed variance / cost-per-correct / stack transfer |
| Artifacts delayed | Monthly public ship; repo first |
| Post-training gap | Small SFT/DPO in Paper 2 |
| No agent on CV | Weekend tool-agent demo after P1 pilot |
| CUDA/Triton ignored | Profiling + kernel/PR in Paper 2 |

---

## 16. What should NOT change

- Do not reopen thesis spine unless supervisor vetoes or P1 gap collapses after live verification  
- Do not make RLVR or agents the **main** spine  
- Do not make Indic AI the **whole** thesis  
- Do not add Paper 4 before Paper 1 submitted  
- Do not build custom framework before reproduction cell passes  

---

## 17. V7 control dashboard

| Decision | Status |
|----------|--------|
| Main thesis spine | **Frozen:** reliable cost-efficient deployment of reasoning LLMs |
| Paper 1 | Minimum: reliability beyond accuracy under compression |
| Paper 2 | Industry-heavy acceleration + small post-training artifact |
| Paper 3 | Indic/multilingual deployment economics |
| Job strategy | Artifact-first |
| Degree strategy | Confirm rules immediately; submit J1 early |
| Next action | Preflight + reproduction numbers + supervisor email |

---

## Stop-expanding rule (V7)

After V7, the master roadmap is **complete enough**. The next document should **not** be V8. It should be **Paper 1 Design Doc v1 with reproduction numbers**.
