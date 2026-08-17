# AGENTS.md — PhD Master Operating System & Memory
**Scholar:** Manish Nandish, IIT Patna  
**Cluster:** PARAM Rudra HPC (C-DAC / NSM), NVIDIA A100 80GB GPUs  
**Repository:** `/scratch/manishn_iitp/reasoning-compression-lab`  
**GitHub:** [https://github.com/Manish06N/reasoning-compression-lab](https://github.com/Manish06N/reasoning-compression-lab)  
**Last Updated:** 2026-08-17 (P0 closure on `paper-major-revision`; GPU frozen)

**Superseding scientific claims (use these, not the 2026-08-15 blocks below):** Title is *Beyond Pass@1: Accuracy, Agreement, and Serving-Cost Effects of Public R1-Distill Quantization Checkpoints under a Pinned Stack*. Distinctive result: **estimand disagreement** (pass@1 vs maj@5, length estimators, token-proxy vs sequential vs batched GPU-second cost). Pathology: **25 loops / 0 exact cap hits / 209 near-cap completions**. Checkpoint language only (tested `jakiAJK` Llama AWQ-4; tested Qwen AWQ-4 on GPQA). FP8–BF16 clustered CIs include 0; TOST $\pm 1$ pp fails. Length: clustered mismatch excess $D$ excludes 0 in all six MATH contrasts. Cost: **hybrid scenario** $C_{\mathrm{pass}}$ with Monte Carlo intervals; rankings disagree across proxy / Cond A / Cond B; Qwen FP8 Cond B is five-rep bimodal, not a lone $-36.0\%$. Do not cite architecture-dependent, statistically tied, unique cheapest, “true Pareto,” or first-run $+18.7\%$ / $-19.8\%$. Frozen tables: `results/reports/major_revision_tables.md`. Campaign evaluator: LightEval **0.8.0**. **Experimental GPU work is closed. No further GPU jobs are required for the current manuscript.** Do not merge to `main`. Do not commit/push until asked.

---

## 1. Executive Summary & Thesis Spine

### PhD Thesis Title (Frozen)
> **"Reliable and Cost-Efficient Deployment of Reasoning LLMs under Compression, Evaluation, and Multilingual Constraints"**

### Degree Constraints & Mandate
- **Target Requirements:** 3+ first-author Q1/Q2 journal articles (Scopus/SJR / JCR) + 2 peer-reviewed conference/workshop papers.
- **Timeline:** ~2 years remaining (Degree completion target: 2027/2028).
- **Execution Mode:** Distance mode, flexible supervisor; highly disciplined self-driven execution, public reproducible artifacts, and rigorous statistical controls.

### 3-Journal + 2-Conference Publication Strategy

```mermaid
graph TD
    J1["Paper 1 (J1): Quantization Reliability & Cost Frontier\n(MATH-500, GSM8K, GPQA; BF16/FP8/AWQ4/GPTQ4; 5 seeds)"] --> C1["Conference 1 (C1): Evaluation Metrology Workshop\n(Packaging calibration & trace pathology protocol)"]
    J1 --> J2["Paper 2 (J2): Reasoning Speculative Decoding\n(0.5B-1.5B draft models, trace acceptance rates, acceleration)"]
    J2 --> C2["Conference 2 (C2): Speculative Serving Demo\n(vLLM / SGLang integration & systems benchmark)"]
    J1 --> J3["Paper 3 (J3): Indic & Multilingual Deployment Economics\n(Token-cost inequity, Indic reasoning, A100 vs RTX 5080 edge transfer)"]
```

| Output | Type | Title / Focus | Target Venues (Verify Q1) | Hardware / Stack | Status / Target Date |
|---|---|---|---|---|---|
| **J1** | Main Journal | *Beyond Pass@1: Accuracy, Agreement, and Serving-Cost Effects of Public R1-Distill Quantization Checkpoints under a Pinned Stack* | *Future Generation Computer Systems (FGCS)*, *Journal of Systems and Software (JSS)*, *Neurocomputing* | HPC 2× A100, `qrm-official` (vLLM 0.7.0 eager) | **88/88 cells + serving confirmation + major-revision CPU analysis**; do not merge to `main` until asked |
| **C1** | Conference / Workshop | *Trace-Level Evaluation Metrology for Compressed Reasoning Models* | NeurIPS/ICLR/ACL Workshops (Eval4NLP, Efficient Natural Language, MLPerf) | HPC A100 | Submission Month 6–12 (Post-J1 pilot packaging) |
| **J2** | Journal 2 | *Reasoning-Aware Speculative Decoding: Acceptance Dynamics and Serving Acceleration* | *JSS*, *Engineering Applications of AI (EAAI)*, *FGCS* | HPC 2× A100 | Year 2 (Methods & draft model training) |
| **C2** | Conference / Workshop | *High-Throughput Speculative Serving of Compressed Reasoning LLMs* | MLSys / EuroSys / ACL Demo Track | HPC A100 | Year 2 |
| **J3** | Journal 3 | *Deployment Economics and Token-Cost Inequity of Reasoning Models in Multilingual/Indic Settings* | *Sustainable Computing*, *Computer Speech & Language*, *Language Resources & Evaluation (LRE)* | HPC A100 (Primary) + RTX 5080 / llama.cpp (Edge Transfer) | Year 2 → Thesis Conclusion |

---

## 2. Cluster Architecture, Hardware Policy & SSH Tunneling

### PARAM Rudra HPC Specifications
- **Cluster:** PARAM Rudra HPC, IIT Patna (NSM / C-DAC).
- **Compute Partition:** `gpu` partition with NVIDIA A100-PCIE-80GB GPUs (`ragpu003`–`ragpu008`, `racn115`–`racn116`).
- **User Limits:** Max **2 GPUs** concurrently per user (`QOSMaxGRESPerUser`), max **48 hours** walltime per SLURM job.
- **FairShare:** Dynamic priority score decay. Monitor regularly via `sshare -u $USER -l`.

### Hardware Allocation Policy Across Papers

| Paper | Primary Hardware | Secondary / Transfer Layer | Publication Hardware Policy |
|---|---|---|---|
| **Paper 1 (J1)** | HPC 2× A100 80GB | None for cited numbers | **100% HPC A100**. RTX 5080 is **strictly retired** for J1 publication numbers (used for short local smoke/debug only). |
| **Paper 2 (J2)** | HPC 2× A100 80GB | Local workstation (small draft model prototyping) | All benchmarked acceleration numbers from HPC A100. |
| **Paper 3 (J3)** | HPC 2× A100 80GB | **RTX 5080 + llama.cpp / GGUF** | Bounded "Datacenter A100 $\rightarrow$ Local Edge RTX 5080" transfer evaluation layer. |

### Critical SLURM Rules for PARAM Rudra
1. **Never `--exclusive` on 1-GPU cells:** ragpu nodes have 2× A100 GPUs. Requesting `--exclusive` with `--gres=gpu:1` allocates the whole node and consumes the user's full 2-GPU quota, causing the second job to stall in `QOSMaxGRESPerUser`.
2. **Resource syntax:** Always use `--gres=gpu:1` or `--gres=gpu:2` with `--cpus-per-task=16`. Never specify `--mem` or GPU subtypes (e.g. no `--gres=gpu:a100:2`).
3. **Always `--enforce-eager` in vLLM:** Triton JIT compilation fails on compute nodes due to missing toolchains/permissions.
4. **GPU Memory Utilization:** Set `--gpu-memory-utilization 0.75` (reserves 60GB on 80GB A100) to prevent OOM errors on shared nodes.
5. **AWQ Quantization Requirement:** Always set `--dtype float16` for AWQ models (`torch.bfloat16` is unsupported by the AWQ kernel).
6. **Walltime Policy:** Always specify `--time=47:00:00` (47 hours) for all jobs going forward (48 hours is the cluster hard limit).

### SSH Tunnel for Local Interactive vLLM Inference
Run on your **local Windows PC** (PowerShell) to forward the active vLLM port:
```powershell
ssh -L 8080:<NODE>:8080 -N manishn_iitp@paramrudra.iitp.ac.in -p 4422
```
*(Replace `<NODE>` with the assigned compute node, e.g. `ragpu004` or `ragpu005`).*  
**Verify tunnel locally:** `curl http://localhost:8080/health` $\rightarrow$ Expected: `200 OK`.

### Standalone Interactive LLM Serving Scripts
| Model | Disk Path | Script | GPUs Required |
|---|---|---|---|
| **DeepSeek-R1-70B** | `/scratch/manishn_iitp/models/DeepSeek-R1-Distill-Llama-70B` | `~/start-llm-deepseek.sh` | 2 |
| **MiniMax-M2.7-XL** | `/scratch/manishn_iitp/models/MiniMax-M2.7-UD-Q4_K_XL/UD-Q4_K_XL` | `~/serve-minimax.sh` | 2 |
| **GLM-4.7-XL** | `/scratch/manishn_iitp/models/GLM-4.7-UD-Q2_K_XL/UD-Q2_K_XL` | `~/serve-glm.sh` | 2 |

---

## 3. Paper 1 (J1): Scientific Positioning & Breakthrough Results

### Provisional Title
> **"Beyond Pass@1: Accuracy, Agreement, and Serving-Cost Effects of Public R1-Distill Quantization Checkpoints under a Pinned Stack"**

### Novelty Positioning Against Prior Literature
* **The Literature Gap:** Prior works (QRM 2025, A Sober Look 2025, Quantized LLMs Can Still Be Calibrated 2025, Cost-of-Pass 2025, Quantization Inflates Reasoning 2026, Reliability Scaling Laws 2026) studied accuracy, seed variance, or token count in isolation.
* **Our Core Contribution:** A multi-seed, trace-level empirical study isolating the **joint reliability–calibration–cost frontier** across weight quantization formats (BF16, FP8, AWQ-4, GPTQ-4), proving where compression changes failure modes, selective prediction risk, and dollar-cost-per-correct answer even when raw accuracy appears preserved.

### Headline Confirmatory Results (2026-08-15)
**Dataset:** `HuggingFaceH4/MATH-500` ($n=500$) | **Seeds:** 42, 43, 44, 45, 46 (5 seeds) | **Total Evaluated Completions:** 20,000  
**Stack:** `qrm-official` (vLLM 0.7.0 eager, $T=0.6, p=0.95, \text{max\_tokens}=32,768$) | **Archive:** `outputs-hpc-campaign-2026-08-14/validation/`

```
====================================================================================================
MODEL MATRIX SUMMARY (MATH-500, n=500, 5 Seeds: 42, 43, 44, 45, 46)
====================================================================================================
Model & Format          Seed 42   Seed 43   Seed 44   Seed 45   Seed 46      Mean ± Std   Trunc   Loops
----------------------------------------------------------------------------------------------------
Qwen-7B BF16             94.4%     94.0%     93.8%     94.6%     93.2%    94.00% ± 0.53%      0       0
Qwen-7B FP8              94.4%     95.2%     94.8%     92.6%     95.0%    94.40% ± 1.05%      0       0
Qwen-7B AWQ-4            92.4%     92.8%     93.2%     93.0%     94.2%    93.12% ± 0.68%      0       0
Qwen-7B GPTQ-4           93.8%     92.6%     93.4%     94.6%     93.0%    93.48% ± 0.77%      0       0
----------------------------------------------------------------------------------------------------
Llama-8B BF16            89.0%     88.4%     90.2%     89.8%     88.8%    89.24% ± 0.73%      0       0
Llama-8B FP8             89.0%     89.6%     88.6%     89.2%     91.2%    89.52% ± 1.02%      0       0
Llama-8B AWQ-4           84.4%     84.8%     89.2%     87.4%     86.6%    86.48% ± 1.95%      0       0
Llama-8B GPTQ-4          88.0%     89.6%     86.8%     89.4%     90.8%    88.92% ± 1.54%      0       0
====================================================================================================
```

### Key Empirical Findings
**Superseded 2026-08-15 bullets (do not cite):** “FP8 parity,” “0 truncations / 0 loops,” “Pareto-optimal FP8.” Current findings are in `README.md` and `paper/main.tex`: clustered pass@1 (Llama AWQ-4 MATH $-2.76$ pp); 25 loops / 0 cap hits / 209 near-cap; Qwen 4-bit MATH token inflation $+6.3$–$6.9\%$; gold-free modal agreement; confirmation batched Qwen GPTQ-4 $-45.9\%$ scenario $C_{\mathrm{pass}}$ vs BF16.

---

## 4. Master PhD & Paper 1 Execution Roadmap

```mermaid
gantt
    title Paper 1 Execution Timeline
    dateFormat  YYYY-MM-DD
    section Completed
    Phase 0 Reproducibility & Schema Fixes       :done, 2026-08-13, 2026-08-14
    Phase 1 Matched BF16 vs FP8 Baseline        :done, 2026-08-14, 2026-08-14
    Phase 2 3-Seed 4-Format Pilot Grid          :done, 2026-08-14, 2026-08-15
    Phase 4 5-Seed Confirmatory Grid (MATH-500) :done, 2026-08-14, 2026-08-15
    section Active
    Phase 5 Statistical Analysis & Calibration  :active, 2026-08-15, 2026-08-20
    Breadth Tasks (GPQA-Diamond, GSM8K)         :2026-08-18, 2026-08-25
    Phase 6 Manuscript Draft (paper/main.md)    :2026-08-20, 2026-08-30
```

### Phase Details & Action Items

#### [x] Phase 0–4: Completed Foundation & Confirmatory Grid
- All 40 MATH-500 cells (2 models $\times$ 4 formats $\times$ 5 seeds) executed, verified, and backed up in `outputs-hpc-campaign-2026-08-14/`.

#### [x] Phase 5: Frozen Statistical Analysis & Calibration (COMPLETED)
1. **Paired Statistical Testing:** Paired McNemar tests confirm exact statistical parity between BF16 and quantized formats under Holm-Bonferroni correction ($p > 0.05$).
2. **Sample-Consistency Calibration:** `maj@5` evaluated; ECE $\le 0.034$ (Qwen) / $\le 0.072$ (Llama); Brier score $< 0.022$; AURC $\le 0.0054$.
3. **Systems & Economics:** Primary Cost-of-Pass uses confirmation GPU-seconds (balanced MATH-500 subsets, `max_num_seqs=8`) at a $\$1.50$/A100-h scenario. The old shared-$65$ tok/s proxy is historical. The first unconstrained serving run is provenance only. Do not cite a unique Pareto optimum.
4. **Structured Trace Audit:** 200-sample stratified audit documented in `results/trace_audit_report.json` with 0 degenerations.

#### [x] Phase 4 Extension: Breadth Benchmark Evaluation (COMPLETED)
- **GSM8K ($n=1,319$):** ✅ **100% COMPLETED** (24 cells, seeds 42–44). Qwen: BF16 91.26%, FP8 91.33%, AWQ-4 91.05%, GPTQ-4 91.13%; Llama: BF16 88.68%, FP8 88.80%, AWQ-4 87.11%, GPTQ-4 88.96%.
- **GPQA-Diamond ($n=198$):** ✅ **100% COMPLETED** (24 cells, seeds 42–44). Qwen: BF16 50.34%, FP8 49.49%, AWQ-4 44.78%, GPTQ-4 47.98%; Llama: BF16 46.13%, FP8 47.81%, AWQ-4 46.97%, GPTQ-4 44.95%.

#### [x] Phase 6: Manuscript Completion & Submission Packaging (COMPLETED)
- Full manuscript in [`paper/main.md`](paper/main.md) and [`paper/main.tex`](paper/main.tex) with all 8 tables and formal formulas.
- Camera-ready PDF compiled cleanly into [`paper/main.pdf`](paper/main.pdf) (7 pages).
- Target venue: *Future Generation Computer Systems (FGCS)* / *Journal of Systems and Software (JSS)* (Q1).
- All 88 validation JSON files across MATH-500, GSM8K, and GPQA backed up into `results/` and synced with GitHub.

---

## 5. Standard Operating Procedures & Workflows

### Autonomous 24/7 SLURM Campaign Daemon
The autonomous queue daemon (`scripts/hpc/queue_manager_daemon.py`) manages chained pipeline execution across 2 channels (`Qwen-7B` and `Llama-8B`), maintaining 100% utilization of the 2-GPU quota with automatic completion detection and self-healing retries:
```bash
# Launch campaign daemon in background
nohup python3 scripts/hpc/queue_manager_daemon.py > logs/queue_manager.log 2>&1 &
```

### HPC Queue Repair Workflow
If a job is corrupted or requires immediate cancellation:
1. Inspect queue: `squeue -u $USER`.
2. Hold downstream dependencies: `scontrol hold <downstream_jobids>`.
3. Cancel failing job: `scancel <jobid>`.
4. Resubmit corrected job with `sbatch ...`.
5. Release downstream jobs once the new job is safely active: `scontrol release <downstream_jobids>`.
6. Log changes in `CHANGELOG.md` and `progress.md`.

### Coordinated 3-Part Git Sync Workflow
**Rule:** When syncing changes between HPC, MacBook, and GitHub, strictly follow:

#### Part 1: Stage and Commit on HPC
```bash
cd /scratch/manishn_iitp/reasoning-compression-lab
git status -sb
git add CHANGELOG.md AGENTS.md docs/ TODO_LIST.md <changed_code_files>
git commit -m "docs/ops: update project state and campaign progress"
```

#### Part 2: Pull and Push on MacBook
```bash
bash "/Users/manish/Projects/2026/paper 1/reasoning-compression-lab/scripts/macbook/rsync_from_hpc.sh"
cd "/Users/manish/Projects/2026/paper 1/reasoning-compression-lab"
git status
git add -A
git commit -m "sync: integrate latest HPC commits and logs"
git push origin main
```

#### Part 3: Align HPC with GitHub
```bash
cd /scratch/manishn_iitp/reasoning-compression-lab
git fetch origin
git reset --hard origin/main
git status -sb
```

---

## 6. Supervisor Meeting Protocol & Institutional Governance

### Key Principles for Supervisor Alignment
- **Monthly Progress Structure:** Deliver a concise 2-page briefing highlighting:
  1. Completed empirical milestones with concrete pass@1, calibration, and cost numbers.
  2. Active publication timeline against J1/J2/J3 targets.
  3. Compute budget utilization on PARAM Rudra.
- **Journal Quartile Verification:** Before submitting any paper, verify that the journal is indexed in Scopus/SJR as **Q1** and recognized by IIT Patna doctoral guidelines.
- **Zero Hallucinated Claims:** All numbers in papers must trace back to verifiable raw JSONL logs with pinned seeds, prompt templates, and configuration hashes.

---

## 7. Canonical Key Files Reference

| File | Location | Purpose |
|---|---|---|
| **Master Memory** | `/scratch/manishn_iitp/reasoning-compression-lab/AGENTS.md` | Single source of truth for all AI agents & PhD ops |
| **PhD Roadmap** | `docs/PHD_ROADMAP.md` | Long-term 3-year PhD thesis strategy, tracks, and career mapping |
| **Publication Audit** | `docs/PUBLICATION_READINESS.md` | Controlling scientific interpretation & claims boundary |
| **Experimental Plan** | `docs/plans/2026-08-14-publication-recovery.md` | Step-by-step phased execution runbook |
| **Working Manuscript** | `paper/main.md` | Live working draft of Paper 1 (J1) |
| **Detailed Log** | `CHANGELOG.md` | Chronological job execution and code modification log |
| **Model Scope** | `docs/MODEL_ROSTER.md` | Checkpoint paths, HF revisions, and model scope decisions |
| **Hardware Policy** | `docs/HARDWARE_POLICY.md` | HPC A100 vs RTX 5080 role assignment |
| **Literature Map** | `docs/literature/PAPER1_READING_MAP.md` | Reading groups, paper references, and novelty positioning |
| **TODO Tracker** | `TODO_LIST.md` | Granular checklist of active, pending, and completed tasks |
