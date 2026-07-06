# Experiment Log

Optional per-cell notes. **Master log:** [progress.md](../progress.md) and [CHANGELOG.md](../CHANGELOG.md).

Chronological record of runs. One section per experiment cell.

---

## Template

```text
Date:
Level:        A | B | C
Model:
Quant config:
Task:
Seed(s):
Hardware:
Status:       planned | running | done | failed
Raw path:
Scored path:
Notes:
Key numbers:
```

---

## 2026-06-26 — Project setup (MacBook)

```text
Date:         2026-06-26
Level:        setup
Status:       done
Notes:        Full Level A execution pipeline built on MacBook (not just docs):
              smoke_test.py, run_inference.py, score_run.py, HPC gates 00–06,
              SLURM templates, cell configs, metrics/extraction from reference repos.
              11 commits pushed to GitHub (5cad28f → 7a287d1).
              External repos organized under paper 1/external_repos/ (read-only).
              See progress.md § 2026-06-26 for full component list.
```

---

## 2026-06-26 — HPC bootstrap (PARAM Rudra)

```text
Date:         2026-06-26
Level:        setup / gates 1–2b
Hardware:     PARAM Rudra A100 80GB (node ragpu006 for Gate 1)
Status:       done (gates 1–2b); Gate 3 submitted, not passed
Notes:        Repo cloned to /scratch/manishn_iitp/reasoning-compression-lab.
              PARAM Rudra adaptations: --partition=gpu, --gres=gpu:1, no --mem,
              conda /home/apps/MSCC/miniconda3, vllm==0.8.5, enforce_eager=true,
              HF cache on scratch ($QR/hf_cache/).
              Gate 1 PASSED (job 85013): CUDA + A100 + vLLM OK.
              Gate 2 PASSED: Qwen-7B ~15 GB downloaded (2 safetensors shards).
              Gate 2b PASSED: MATH-500 validated (500 examples).
              Gate 3 SUBMITTED: smoke job 85028; 10-q debug 85030 (afterok).
              HPC local commit 6d58d9b not pushed; MacBook pushed 7a287d1.
Key numbers:  Python 3.11.15, torch 2.6.0+cu124, vllm 0.8.5
```

---

## 2026-06-27 — HPC smoke failures + fixes

```text
Date:         2026-06-27
Level:        setup / Gate 3
Hardware:     PARAM Rudra A100
Status:       failed (Gate 3 not passed)
Notes:        Job 85028 FAILED: tokenizer all_special_tokens_extended missing.
              Job 85092 FAILED: shared-GPU OOM (~24 MiB free on A100).
              Job 85094 FAILED: KeyError ANSWER in prompt template.
              Fixes synced at dff36c1: tokenizer shim, memory preflight,
              quick exclusive smoke SLURM, {{ANSWER}} prompt escape.
              Job 85030 cancelled (dependency on failed smoke).
Raw path:     runs/raw/smoke_test.jsonl — missing
              runs/raw/smoke_test_quick.jsonl — missing
```

---

## 2026-06-28 — HPC publication preflight + smoke resubmit

```text
Date:         2026-06-28
Level:        setup / Gate 3 + publication prep
Hardware:     PARAM Rudra 2× A100 blocks (b01–b06)
Status:       running (smoke pending)
Notes:        Repo fast-forwarded to 03c3766 (5080/HPC machine split).
              All b01–b06 model folders downloaded (~9 models on scratch).
              07_preflight_publication.py passed (config/dataset/model wiring).
              Exclusive quick smoke job 85306 submitted; b01–b06 held until pass.
              See progress.md for model inventory and block wiring tables.
```

---

## 2026-06-28 — Windows 5080 publication main grid (running)

```text
Date:         2026-06-28
Level:        B/C main (5080 primary)
Hardware:     RTX 5080 16GB (WSL2), vLLM 0.23, torch 2.11+cu128
Status:       running (publication main grid)
Archive:      outputs-win5080-main-2026-06-28/
Protocol:     repro_qrm.yaml — temp 0.6, top_p 0.95, max_tokens 32768, seed 0, batch_size=1
Sample sizes: MATH-500 n=500, GSM8K n=1319
Policy:       5080 primary; HPC only for >16GB VRAM (BF16 7B/8B)
Skipped:      BF16 7B/8B (VRAM), GPQA (gated)
Notes:        Supersedes pilot archive (outputs-win5080-pilot-*). Pilot not for paper.
Command:      bash scripts/local/start_5080_main.sh
```

---

## 2026-06-28 — Windows 5080 pilot pipeline (superseded)

```text
Date:         2026-06-28
Level:        pilot (5080 debug)
Hardware:     RTX 5080 16GB (WSL2), vLLM 0.23, torch 2.11+cu128
Status:       superseded — do not cite in paper
Archive:      outputs-win5080-pilot-2026-06-28/
Protocol:     n=50, pilot_5080.yaml (max_tokens 8192), batched inference
Cells:        14 (smoke + 13 inference); BF16 7B/8B + GPQA skipped
Backup:       _backup/latest/ + snapshots every 3 cells
Resume:       bash scripts/local/resume_5080_pilot.sh
Notes:        Full grid aborted; pilot supersedes. 12 models on disk.
```

---

## 2026-06-28 — Windows 5080 setup + pilot protocol

```text
Date:         2026-06-28
Level:        setup + pilot infra
Hardware:     RTX 5080 16GB (WSL2), vLLM 0.23, torch 2.11+cu128
Status:       done (infra); pilot grid ready to run
Notes:        All 12 model checkpoints downloaded (~62 GB).
              Phase 0 smoke: Qwen-1.5B BF16 OK; Qwen-7B BF16 OOM (HPC only).
              Full grid started then superseded by pilot mode (batched n=50).
              Pilot archive: outputs-win5080-pilot-2026-06-28/
              Full archive:  outputs-win5080-2026-06-28/
Key numbers:  ~1-4k completion tokens/q on 1.5B; ~15 tok/s; batch 4/2/1 by model size
```

---

## 2026-07-05 — Path C diagnostic (RUNNING)

```text
Date:         2026-07-05
Level:        diagnostic (Path C)
Archive:      outputs-hpc-diag-pathc-2026-07-05
Hardware:     A100 80GB (racn116)
Status:       running
Jobs:         87116 Qwen 32k, 87117 Llama 32k, 87118 Qwen 64k (pending)
Cells:
  - diag_qwen7b_bf16_math500_seed42_n50     (strict QRM 32k, n=50, seed 42)
  - diag_llama8b_bf16_math500_seed42_n50    (same)
  - diag_qwen7b_bf16_math500_seed42_n50_64k (64k budget test, n=50)
Protocol:     repro_qrm_strict.yaml / repro_qrm_64k.yaml
              reproduction prompt, no repetition_penalty, enforce_eager=true
Submit:       bash scripts/hpc/submit_pathc_diagnostic.sh
Report:       bash scripts/hpc/report_pathc_diagnostic.sh
Notes:        Launched after b01 gate fail. Supersedes quant grid until results known.
```

---

## 2026-07-05 — b01 BF16 Llama (COMPLETE, gate FAIL)

```text
Date:         2026-07-03 → 2026-07-05
Level:        C (b01 block; sober prompt)
Model:        DeepSeek-R1-Distill-Llama-8B
Quant config: BF16
Task:         MATH-500
Seed(s):      0
Hardware:     A100 80GB (job 86758, racn116)
Status:       done / scored
Raw path:     outputs-hpc-2a100-main-2026-07-03/raw/level_c_llama8b_bf16_math500_seed0.jsonl
Scored path:  outputs-hpc-2a100-main-2026-07-03/scored/level_c_llama8b_bf16_math500_seed0.jsonl
Notes:        Protocol 729d773 (32k, rep_penalty 1.05). prompt_profile=sober (NOT QRM reproduction).
              compare_qrm_baseline.py SKIP on profile. Valid for Paper 1 deployment metrics.
Key numbers:  pass@1 19.6%, truncation 58%, parse_fail 60.4%, cost/correct ~1614s,
              non-truncated pass@1 45.2% (210 stop finishes)
```

---

## 2026-07-05 — b01 BF16 Qwen (PARTIAL, gate FAIL)

```text
Date:         2026-07-03 → 2026-07-05
Level:        A (reproduction prompt)
Model:        DeepSeek-R1-Distill-Qwen-7B
Quant config: BF16
Task:         MATH-500
Seed(s):      0
Hardware:     A100 80GB (job 86757 TIMEOUT; resume 87111 FAILED GPU busy)
Status:       partial — 410/500 checkpointed; 90 rows optional
Raw path:     outputs-hpc-2a100-main-2026-07-03/raw/level_a_qwen7b_bf16_math500_seed0.jsonl
Notes:        94.1% truncation on 410 rows. Finishing 90 rows NOT required for decisions.
Key numbers:  truncation ~94%, median completion 32k tokens, ~7 min/problem
```

---

## Level A — Reproduction gate (FAILED 2026-07-05)

```text
Date:         2026-07-05
Level:        A
Model:        Qwen-7B + Llama-8B BF16
Quant config: BF16
Task:         MATH-500
Seed(s):      0
Hardware:     A100
Status:       gate failed — pivot to Paper 1 deployment grid (see notes.md §28)
Notes:        QRM repro not achieved. Quant grid (b02–b05) on hold until Path C.
```

---

## Path C — Strict QRM diagnostic (CANCELED 2026-07-05)

```text
Date:         2026-07-05
Level:        diagnostic (pre-quant)
Model:        Qwen-7B + Llama-8B BF16 (+ Qwen 64k wave planned)
Jobs:         87116, 87117, 87118 — all CANCELED
Archive:      outputs-hpc-diag-pathc-2026-07-05 (partial ~20 rows)
Protocol:     reproduction prompt, repro_qrm_strict.yaml, no rep_pen, enforce_eager=true
Key numbers:  Qwen 10% pass@1, 90% trunc; Llama 15% pass@1, 75% trunc (n=20)
Notes:        Protocol verified in raw JSONL. Canceled → Experiment A (official QRM repo).
```

---

## Path C — CANCELED (2026-07-05)

```text
Date:         2026-07-05
Jobs:         87116, 87117, 87118
Status:       CANCELED by user after n=20 (~3h GPU each on 87116/87117)
Reason:       Sufficient signal; pivot to Experiment A
Partial data: outputs-hpc-diag-pathc-2026-07-05/raw/ (~20 rows per 32k cell)
```

---

## Experiment A — Official QRM inference.py (QUEUED 2026-07-06)

```text
Date:         2026-07-06
Level:        diagnostic (stack cross-check)
Model:        DeepSeek-R1-Distill-Qwen-7B BF16
Task:         MATH-500 (n=10 pilot)
Seed(s):      42
Env:          qrm-official (NOT qreason — vLLM 0.7.0 fork)
Hardware:     A100 job 87216 — PENDING (Resources), --exclusive
Status:       Env install VERIFIED on login node; waiting for exclusive GPU
Submit:       bash scripts/hpc/submit_qrm_official_test.sh
Output:       outputs-hpc-qrm-official-2026-07-06/
Compare:      python scripts/hpc/qrm_parity/compare_side_by_side.py --limit 10
Debug log:    docs/QRM_OFFICIAL_HPC_TROUBLESHOOTING.md (jobs 87130–87216)
Notes:        Same repo as main harness; separate conda env. Decisive test vs our stack.
```

---

## Experiment B — Parity pilot d03 (READY, not scheduled)

```text
Date:         2026-07-05
Status:       ready — bash scripts/hpc/submit_pathc_parity_pilot.sh
Notes:        Our stack with serving fixes (Experiment B). Skipped in favor of A.
```

---

## Level A — GPTQ-4 reproduction (blocked until prep)

```text
Date:
Level:        A
Model:        DeepSeek-R1-Distill-Qwen-7B
Quant config: GPTQ-4
Task:         MATH-500
Seed(s):      0
Hardware:     A100
Status:       blocked
Notes:        Requires BF16 Level A done first.
              Then: download/quantize GPTQ-4 → 06_verify_gptq4_model.sh → inference.
              See docs/GPTQ4_PREP.md.
```
