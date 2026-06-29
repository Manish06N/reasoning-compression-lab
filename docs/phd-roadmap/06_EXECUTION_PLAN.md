# Part 6 — Execution Plan (90 Days, 30 Days, Commands)

[← Index](README.md) · HPC playbook: [docs/HPC_STEP_BY_STEP.md](../HPC_STEP_BY_STEP.md) · Live status: [progress.md](../../progress.md)

---

## 10. First 90-day plan

**Goal:** Prove pipeline, reproduce one known result, run pilot, produce supervisor packet — **not** explore new topics.

| Week | Read | Build / run | Output | Gate |
|------|------|-------------|--------|------|
| 1 | Liu et al. quant reasoning; vLLM basics | Repo/env; selftest; supervisor emails | Repo + emails | Selftest; emails sent |
| 2 | GPTQ/AWQ; dataset cards | validate_tasks; download qwen7b; cells | Validation logs | MATH-500 validates |
| 3 | MATH extraction; math-verify | Quantize GPTQ-W4; BF16 + GPTQ smoke | First rows | Generation works |
| 4 | Calibration / selective prediction | Reproduce BF16/GPTQ MATH-500 cell | Calibration CSV | ±2 pts or explain gap |
| 5 | Bootstrap / McNemar | Pilot: 3 configs × MATH-500 × 3 seeds | Pilot raw data | End-to-end pilot |
| 6 | Cost / profiling | Profiling; cost table | Latency/VRAM/cost | cost-per-correct computed |
| 7 | Extraction audit | Audit 200 traces | Audit CSV | Extraction &lt;2% error |
| 8 | Journal author guides | Figures; pilot summary | Reliability/Pareto figs | Supervisor packet |
| 9–10 | Saturation scan | Expand to minimum grid if approved | Full-run logs | No saturation blocker |
| 11–12 | Writing | Results analysis + outline | Draft sections | Paper 1 design locked |
| 13 | Target journal scope | 90-day review | Go/no-go packet | Full J1 execution |

### V7 re-engineered first 90 days (artifact-first)

| Week | Task | Public output | Gate |
|------|------|---------------|------|
| 1 | Supervisor email; rules; selftest | README/runbook | Rules sent; selftest OK |
| 2 | validate_tasks, cells, smoke | Preflight log | Preflight passes |
| 3 | qwen7b BF16/GPTQ reproduction | Reproduction blog/log | Gap acceptable |
| 4 | extract/score/calibrate | Blog #1 | Calibration rows exist |
| 5–6 | Harness v0.1 | Harness release | End-to-end pilot |
| 7–8 | Pilot 1×3×2×3 seeds | HF pilot v0.1 | CIs computable |
| 9 | Supervisor packet | Design doc v1 | Sign-off |
| 10–11 | Minimum grid prep | Dashboard draft | 50%+ grid validated |
| 12–13 | Paper skeleton; C1 abstract | Figures v0 | Story visible |

---

## Appendix F — First 30-day exact checklist

| Time | Task | Pass condition |
|------|------|----------------|
| Days 1–2 | Conda env; imports | Environment activates |
| Day 3 | CPU selftest | ALL SELFTESTS PASSED |
| Day 4 | validate_tasks | MATH-500 OK |
| Day 5 | make_cells (seed 1) | Cell YAMLs generated |
| Day 6 | validate_cells | PASS |
| Day 7 | Mock smoke | SMOKE TEST PASSED |
| Days 8–10 | Download Qwen-7B | Weights + config exist |
| Days 11–13 | Tiny BF16 smoke (5 Q) | 5 rows + marker |
| Days 14–18 | Quantize GPTQ-W4 | Quant artifact exists |
| Days 19–22 | Full repro cells BF16 + GPTQ | Two raw JSONLs |
| Days 23–25 | extract → score → calibrate | calibration.csv rows |
| Days 26–28 | Compare to Liu et al. | Within tolerance or diagnosis |
| Days 29–30 | Supervisor packet | Approval for pilot |

**Repo-aligned commands (PARAM Rudra):**

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
bash scripts/hpc/00_setup_env.sh          # once
bash scripts/hpc/02b_validate_dataset.sh  # MATH-500
bash scripts/hpc/03a_preflight_cpu.sh     # CPU gate
sbatch slurm/smoke_test_quick_exclusive.slurm
LIMIT=10 bash scripts/hpc/04_run_level_a_bf16.sh
bash scripts/hpc/05_score_level_a.sh
python scripts/hpc/07_preflight_publication.py
```

---

## Appendix L — One-page command sheet (legacy qreason path)

For the **current** repo layout, prefer:

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh && conda activate qreason
python scripts/hpc/07_preflight_publication.py
sbatch slurm/smoke_test_quick_exclusive.slurm
bash scripts/hpc/submit_hpc_blocks.sh b01
python scripts/score_run.py --input runs/raw/<cell>.jsonl
python scripts/build_paper_tables.py --archive outputs-hpc-2a100-main-YYYY-MM-DD
python scripts/build_repro_bundle.py --archive outputs-hpc-2a100-main-YYYY-MM-DD
```

---

## 11. Skills roadmap (summary)

| Phase | Skills |
|-------|--------|
| Immediately | Linux, SLURM, Conda, Git, HF, PyTorch, vLLM, JSONL |
| Paper 1 | GPTQ/AWQ/FP8, llama.cpp basics, lm-eval patterns, math-verify, NVML, bootstrap/McNemar/ECE/Brier/AURC |
| Paper 2 | Speculative decoding, SGLang, TRL SFT/DPO, FSDP basics |
| Paper 3 | Indic benchmarks, tokenization/fertility, energy, edge deployment |
| Job premium | CUDA/Triton reading, Docker/K8s, PRs to vLLM/lm-eval |

V7 priority stack: [07_V7_JOB_FIRST_STRATEGY.md](07_V7_JOB_FIRST_STRATEGY.md#11-revised-skill-priority-stack).

---

## 12. Job-market mapping (summary)

| Role | Your evidence |
|------|---------------|
| Research Engineer | qreason harness, reproducible runs, public dataset |
| Applied Scientist | cost-per-correct, Pareto guidance |
| ML Systems Engineer | vLLM profiling, latency/VRAM, quantization |
| LLM Evaluation Scientist | calibration protocol, seed variance, extraction audit |
| AI Infrastructure | Runbook, SLURM, resume-safe cells, cost model |
| India roles | J3 Indic deployment economics |

---

## 13. Risk register (summary)

| Risk | Mitigation |
|------|------------|
| Accepted vs submitted rule unknown | Written supervisor/institution answer now |
| Engineering quicksand | Smoke tests, validators, minimum grid first |
| Second scoop | Weekly saturation scan; pivot rungs |
| A100 access instability | Pilot first; resume-safe SLURM |
| Extraction errors | 200-trace audit; math-verify |
| Scope creep | Minimum J1 first; edge/energy → J3 |
| Journal delay | Barbell: Q1 target + Q2 fallback |

Full register: [07_V7_JOB_FIRST_STRATEGY.md](07_V7_JOB_FIRST_STRATEGY.md#15-updated-risk-register).

---

## 15. Immediate next actions

1. Send supervisor/institution publication-rule questions (written).  
2. Run selftest, validate_tasks, validate_cells, smoke.  
3. Reproduce qwen7b BF16 + GPTQ-W4 MATH-500 (or monitor active HPC grid).  
4. If reproduction passes → pilot (3 configs × 3 seeds).  
5. Prepare 30-day supervisor packet: reproduction, pilot table, reliability figure, cost table, design doc v1.  
6. **Stop asking for new topics** unless supervisor vetoes or saturation gate fails twice.

### Stop-planning rule

The next document that changes the PhD is **not another roadmap**. It is **Paper 1 Design Doc v1 with actual reproduction numbers** and supervisor rule answers.

---

## Compute budget gating

| Block | Purpose | Risk control |
|-------|---------|--------------|
| Reproduction | 1 model, 2 configs, MATH-500, 1 seed | Must pass before pilot |
| Pilot | 1 model, 3 configs, 1–2 tasks, 3 seeds | No full grid until pilot OK |
| J1 minimum | 3×5×4×5 | Freeze features |
| J1 ambitious | +edge/energy/stack | Only if no J1 delay |

**Rule:** If pilot cost/cell &gt;50% over estimate, shrink grid before launch. **Do not cut seeds first.**

Full table: [08_EXECUTION_VALIDATION_V6.md](08_EXECUTION_VALIDATION_V6.md#appendix-o-compute-budget).
