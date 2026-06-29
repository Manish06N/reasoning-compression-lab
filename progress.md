# Progress Log — Master Reference

Canonical dated record for **Paper 1: Beyond Accuracy** (`reasoning-compression-lab`).

**Purpose:** Future sessions can resume without guessing what was built, where it runs, which gates passed, and what failed. Update this file after every material change on MacBook, Windows/WSL, or HPC.

**GitHub:** https://github.com/Manish06N/reasoning-compression-lab  
**Related logs:** `CHANGELOG.md` (ops detail), `docs/EXPERIMENT_LOG.md` (experiment cells), `paper 1/AGENTS.md` (AI assistant memory)

---

## Current Status Snapshot (2026-06-29)

| Area | Status |
|------|--------|
| MacBook pipeline + docs | **Complete** |
| GitHub latest commit | Synced after latest HPC push; see `git log -1` |
| HPC env (qreason, vLLM 0.8.5, A100) | **Complete** — Gate 1 passed (job 85013) |
| HPC model + datasets | Qwen-7B BF16 (~15 GB) + all b01–b06 quant models downloaded |
| HPC CPU preflight | **Passed** — `scripts/hpc/07_preflight_publication.py` |
| HPC GPU smoke (Gate 3) | **PASSED** — job `85306`, exit `0:0` |
| First scored paper result | **None** — no `results/*_summary.json` from HPC yet |
| **Windows 5080 (WSL2)** | **Retired for experiments** — too slow; do not schedule publication work there |
| Publication strategy | **HPC-only** (5080 stopped 2026-06-28 — ~15 min/q too slow for 1.5B grid) |
| b01 BF16 anchors | **Corrected job running** — old `85342` canceled; new `85394` resumed Qwen from 20/500 and restarted Llama |
| b02–b06 full grid | **Queued** — pending with `QOSMaxGRESPerUser` until GPUs free |

**Current blocker for clean HPC paper numbers:** b01 must now run to completion or checkpoint enough rows before the 47h walltime. The state-file race has been fixed and the corrected b01 job is running.

**Current live progress:** corrected job `85394` is running on `ragpu008`. Qwen-7B resumed from `20/500` durable rows; Llama-8B restarted from `0/500`. b02-b06 are released and pending behind the user GPU quota. Future Qwen-1.5B model assets are staged on HPC; no extra experiment jobs were submitted.

---

## Current Experiment Coverage

The running/queued jobs are the main HPC publication batch b01-b06, not the complete final paper package. They cover:

| Block | Coverage | Status |
|------|----------|--------|
| b01 | Qwen-7B BF16 MATH-500 + Llama-8B BF16 MATH-500 | Corrected job `85394` running |
| b02 | Qwen-7B FP8 MATH-500 + Llama-8B FP8 MATH-500 | Queued as `85343` |
| b03 | Qwen-7B AWQ-4 MATH-500 + Llama-8B AWQ-4 MATH-500 | Queued as `85344` |
| b04 | Qwen-7B GPTQ-4 MATH-500 + Llama-8B GPTQ-4 MATH-500 | Queued as `85345` |
| b05 | Qwen-7B GPTQ-3 MATH-500 | Queued as `85346` |
| b06 | Qwen-7B FP8 GSM8K | Queued as `85347` |
| b07 | Qwen-7B FP8 GPQA-Diamond | Ready after HF gate approval; not queued |
| b08 | Qwen-1.5B BF16 + FP8 MATH-500 | Wired for future HPC-only submission; not queued |
| b09 | Qwen-1.5B AWQ-4 + GPTQ-4 MATH-500 | Wired for future HPC-only submission; not queued |

Work still left after b01-b06 finish:

- Score all raw JSONL outputs into scored JSONL files and summary JSON results.
- Build paper metrics and tables: pass@1, latency, token counts, VRAM, cost-per-correct, and compression tradeoffs.
- GPQA gated access is now approved for the saved HPC Hugging Face token; authenticated Hub check returned HTTP 200 for `gpqa_diamond.csv`. b07 can be queued after current queue strategy allows.
- Qwen-1.5B cells are now HPC-only future work. BF16, FP8, AWQ-4, and GPTQ-4 model directories were downloaded on HPC on 2026-06-29; queue these only after deciding they are needed and after adding/using an HPC block for them.
- Run multi-seed stability later; current publication jobs are seed0 only.
- Rerun or resume any cell that times out before completing all rows, especially b01 if BF16 remains slow.

---

## Future Asset Preparation

Prepared on HPC without using GPUs while b01-b06 continued running:

| Asset | Status | Path |
|------|--------|------|
| Qwen-1.5B BF16 | Downloaded | `models/DeepSeek-R1-Distill-Qwen-1.5B` |
| Qwen-1.5B FP8 | Downloaded | `models/DeepSeek-R1-Distill-Qwen-1.5B-FP8` |
| Qwen-1.5B AWQ-4 | Downloaded | `models/DeepSeek-R1-Distill-Qwen-1.5B-AWQ-4` |
| Qwen-1.5B GPTQ-4 | Downloaded | `models/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ-4` |
| MATH-500 | Available | Hugging Face cache/dataset load works |
| GSM8K | Available | Hugging Face cache/dataset load works |
| GPQA-Diamond | Access approved | Authenticated HF request returns HTTP 200 for `gpqa_diamond.csv` |

CPU preflight after staging assets: `scripts/hpc/07_preflight_publication.py` passed for the active b01-b06 publication blocks.

No additional jobs were submitted yet. Current recommendation is to let b01-b06 continue. Future jobs are now wired as b07 for GPQA and b08-b09 for Qwen-1.5B lower-bound runs; submit them only after deciding queue strategy.

---

## Archive Metadata and Backup Status

HPC publication runs now write a durable archive manifest at `outputs-hpc-2a100-main-YYYY-MM-DD/manifest.json` and per-cell metadata snapshots under `metadata/<cell_id>.json`. Each metadata file records the cell/model/task/decoding configs, batch and checkpoint settings, git commit, SLURM job info, raw output path, summary path, and saved row count.

For the active 2026-06-29 b01 run, the manifest and metadata files were created manually because job `85394` started before this runner change. Future queued jobs will write and update these automatically. `_backup/latest/` now includes `metadata/` and `manifest.json`; full archive mirrors are protected by a lock for parallel GPU branches.

---

## Publication Sufficiency Strategy

Current judgement: b01-b09 are enough for the first publishable core result set if they complete cleanly and show interpretable trends. The seed0 grid covers three model families/scales, five compression settings, and three reasoning benchmarks.

Do not expand the queue immediately. First finish b01-b09, score all outputs, and build the core tables/figures. After seeing the result quality, decide whether reviewers will need robustness checks.

Recommended expansion only if needed:

- Add seed1/seed2 for the most important subset, not the entire grid.
- Prioritize Qwen-7B and Llama-8B on MATH-500 for BF16, FP8, AWQ-4, and GPTQ-4.
- Add more dataset coverage only if MATH-500 trends are unclear or a reviewer-facing claim needs it.

This keeps the paper focused: seed0 b01-b09 for the main claim, then a small stability subset if the result variance or reviewer risk demands it.

---

## Machine Roles

| Machine | Path | Role |
|---------|------|------|
| **MacBook** | `/Users/manish/Projects/2026/paper 1/reasoning-compression-lab` | Design, docs, git push, rsync hub, writing |
| **GitHub** | `Manish06N/reasoning-compression-lab` | Code backup; HPC pulls from here |
| **HPC (PARAM Rudra)** | `/scratch/manishn_iitp/reasoning-compression-lab` | A100 inference, model downloads, paper numbers |
| **Windows 5080 (WSL2)** | `G:\ALL MY Projects\2026\03-paper1-experiments` (WSL: `/mnt/g/ALL MY Projects/2026/03-paper1-experiments`) | Retired for publication experiments; archive/proof only |

**Not in git (any machine):** `models/`, `runs/`, `results/`, `outputs-*`, `hf_cache/`, `logs/`

---

## Roadmap Position

```text
Phase 0  Literature + design          ✅ Complete
Phase 1  Reproducible harness         ✅ Mostly complete (GPU end-to-end unproven on HPC)
Phase 2  BF16 baseline                ⏳ Blocked at HPC GPU smoke
Phase 3  Quantized variants           ❌ Not started (HPC b01–b06 queued after smoke)
Phase 4  Reliability metrics          📦 Code ready, no real outputs yet
Phase 5  Multi-seed stability         ❌ Not started
Phase 6  Paper tables / figures       ❌ Not started
```

**First target artifact (Level A):** `results/level_a_qwen7b_bf16_math500_seed0_summary.json` (n=10 debug, then n=500 full).

---

## Timeline by Date

### 2026-06-26 — MacBook pipeline build + PARAM Rudra HPC bootstrap

#### MacBook (control room)

Built the full Level A execution pipeline — not just docs:

| Component | Purpose |
|-----------|---------|
| `scripts/smoke_test.py` | 3-question vLLM smoke test |
| `scripts/run_inference.py` | Full MATH-500 inference for one cell |
| `scripts/score_run.py` | Score raw run → pass@1 + latency/VRAM summary |
| `scripts/extract_answers.py` | Answer extraction step |
| `scripts/compute_calibration.py` | Calibration metrics hook |
| `scripts/hpc/00–06_*.sh` | Ordered HPC shell gates |
| `scripts/macbook/rsync_to_hpc.sh` | Copy repo to HPC without GitHub |
| `slurm/*.slurm` | Batch jobs (download, smoke, full BF16) |
| `configs/cells/level_a_*.json` | BF16 and GPTQ-4 experiment cells |

Adapted patterns (not vendored) from reference repos under `paper 1/external_repos/`:

- **sober-reasoning** — prompts, math/GPQA extractors, seed variance
- **Quantized-Reasoning-Models** — decoding protocol, trace length
- **Calibrating-LLMs-with-Consistency** — consistency, calibration (Brier, ECE, AURC)
- **Cost-of-Pass** — cost-per-correct, local cost model

**Git commits pushed to GitHub (11 commits on `main`):**

| Commit | Summary |
|--------|---------|
| `5cad28f` | Initial Paper 1 repo structure |
| `8e5ca4d` | HPC quick start guide |
| `0d794ab` | HPC execution pipeline for Level A |
| `a85096c` | Qwen-7B model config + gitignore fix |
| `516498a` | Literature map and external repos index |
| `6f3d2b0` | GitHub push step + GPTQ-4 prep gate |
| `c7cfe44` | Core prompts, extraction, metrics, SLURM patterns |
| `38ee34f` | External repos index link in README |
| `937543e` | Pre-HPC checks, dataset validation, smoke debug decoding |
| `d1d221a` | Dataset validation + smoke token limit in HPC guide |
| `7a287d1` | Adapt HPC scripts for PARAM Rudra (IIT Patna) |

#### HPC (PARAM Rudra) — first deployment

**User:** `manishn_iitp` · **Scratch:** `/scratch/manishn_iitp/reasoning-compression-lab`

1. Cloned from GitHub (`Manish06N/reasoning-compression-lab`).
2. Adapted generic SLURM/HPC scripts for PARAM Rudra:

   | Generic (broken on cluster) | PARAM Rudra fix |
   |-----------------------------|-----------------|
   | `#SBATCH --gres=gpu:a100:1` | `#SBATCH --partition=gpu` + `--gres=gpu:1` |
   | `#SBATCH --mem=80G` | Removed (cluster rule: no `--mem`) |
   | `$(conda info --base)` | `/home/apps/MSCC/miniconda3` |
   | Unpinned `vllm` → 0.23.0 + Rust build fail | **`vllm==0.8.5`** |
   | No eager mode | **`enforce_eager: true`** in model config + vLLM runner |
   | HF cache in home | **`$QR/hf_cache/`** on scratch |

3. Created conda env **`qreason`**: Python 3.11.15, torch 2.6.0+cu124, vLLM 0.8.5.
4. **Gate 1 passed** — job **85013** on node `ragpu006`: A100 80GB, CUDA OK, vLLM OK.
5. **HF auth** — account Manish99; token stored at `$QR/hf_cache/token` (gitignored).
6. **Gate 2 passed** — Qwen-7B downloaded (~15 GB, 2 safetensors shards).
7. **Gate 2b passed** — MATH-500 validated (500 examples).
8. **Gate 3 submitted** — job **85028** (smoke, 3 questions); job **85030** (10-q BF16 debug, `afterok:85028`).
9. Telegram watchers configured for 85028/85030 (compute nodes cannot reach Telegram — optional only).
10. Local HPC commit **`6d58d9b`** created but **not pushed** (cluster SSH key not on GitHub). MacBook later pushed equivalent fixes as **`7a287d1`** / **`be49fb5`**.

**End-of-day state (2026-06-26):** Setup complete through Gate 2b; smoke job 85028 pending GPU.

---

### 2026-06-27 — Smoke failures, fixes, MacBook/HPC sync

#### HPC job history

| Job | Purpose | Result | Root cause |
|-----|---------|--------|------------|
| 85028 | First smoke (3 Q, max_tokens=1024) | **FAILED** | Tokenizer: `all_special_tokens_extended` missing |
| 85030 | 10-q BF16 debug (`afterok:85028`) | **CANCELLED** | Dependency never satisfied |
| 85031/85032 | Telegram watchers | **FAILED** | Compute nodes can't reach api.telegram.org |
| 85092 | Smoke after tokenizer shim | **FAILED** | Shared-GPU OOM — only ~24 MiB free on A100 |
| 85094 | Exclusive quick smoke (1 Q, 64 tokens) | **FAILED** (later) | Prompt `KeyError: 'ANSWER'` — fixed in repo |

#### Fixes applied (synced to GitHub at `dff36c1`)

- **Tokenizer shim** in `src/runners/vllm_runner.py` for vLLM 0.8.5 + Transformers 5.12.1.
- **GPU memory preflight** — `SMOKE_MIN_FREE_GPU_MB=30000` → exit 75 if GPU too full.
- **`slurm/smoke_test_quick_exclusive.slurm`** — 1 question, 64 tokens, exclusive node.
- **`scripts/hpc/03a_preflight_cpu.sh`** — CPU gate (passed on HPC; synced later).
- **Prompt fix** — `prompts/math500.txt`: `{{ANSWER}}` so `.format()` preserves literal `{ANSWER}`.

#### Sync status

MacBook, GitHub, and HPC aligned at **`dff36c1`**: "Sync HPC smoke fixes: tokenizer shim, memory preflight, quick smoke SLURM."

**Gate 3 still not passed** — no `smoke_test_quick.jsonl` yet.

---

### 2026-06-28 — Windows RTX 5080 + publication machine split + HPC block grid

#### Windows 5080 (WSL2 Ubuntu 22.04)

**Hardware:** RTX 5080 16 GB (Blackwell sm_120)

| Step | Result |
|------|--------|
| Clone repo | Done — `G:\ALL MY Projects\2026\03-paper1-experiments` |
| Conda env `qreason` | Python 3.11 |
| CUDA stack | `torch 2.11.0+cu128` (Blackwell requires cu128; HPC stays on cu124) |
| vLLM | **0.23.0** (0.8.5 incompatible with torch 2.11 on sm_120) |
| Phase 0 smoke — Qwen-1.5B BF16 | **PASSED** — pipeline verified end-to-end |
| Phase 0 smoke — Qwen-7B BF16 | **OOM** (expected — ~14 GB weights, no KV cache room) |
| Model downloads | 12 checkpoints (~62 GB) for 5080-feasible quants |
| Pilot mode | 14-cell grid, n=50, batched inference — then superseded |
| Publication main grid | Started, then **superseded by policy change** |

**Blackwell workarounds in `scripts/local/env.sh`:**

- `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer JIT fails sm_120 check)
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` (WSL)
- `LD_LIBRARY_PATH` for pip-shipped CUDA 13 libs

**Added:** checkpoint/resume (`checkpoint_utils.py`), backup scripts, pilot + publication orchestrators, model roster docs, 5080/HPC machine split configs.

#### Policy change — machine split (commits `30c8c08`, `03c3766`)

| Machine | Scope | Rationale |
|---------|-------|-----------|
| **5080** | Qwen-1.5B × 4 quants × MATH-500 only | ≤24 h/cell; 7B/8B at batch_size=1 takes weeks |
| **HPC 2× A100** | b01–b06: 7B/8B all quants, BF16 anchors, GSM8K | 160 GB VRAM, 48 h SLURM max |
| **HPC b07** | GPQA-Diamond | After Hugging Face gate approval |

**5080 cells:** `level_c_qwen15b_{bf16,fp8,awq4,gptq4}` × MATH-500  
**HPC blocks:** b01 BF16 anchors → b02 FP8 → b03 AWQ-4 → b04 GPTQ-4 → b05 GPTQ-3 → b06 GSM8K

#### HPC session (2026-06-28 afternoon)

- Fast-forwarded scratch repo to `03c3766`; stashed local HPC-only changes.
- Downloaded all b01–b06 model folders (Llama-8B BF16 + all Qwen/Llama quants).
- Validated block → cell → model wiring via `load_cell_config()`.
- Validated MATH-500 (500 rows) and GSM8K (1319 rows) through repo task configs.
- Added **`scripts/hpc/07_preflight_publication.py`** — repeatable CPU preflight (passed).
- Submitted exclusive quick smoke job **`85306`** — pending at end of session.
- Created `/home/manishn_iitp/.codex/CODEX.md` for future Codex sessions.

**Git commits (2026-06-28):** `558d004` (progress log), `62ff8ad` (preflight), `0d5b9ce` (Codex notes), `b280a88` (5080 stopped, HPC-only policy).

#### Evening — 5080 publication run stopped (HPC-only policy)

- Started 5080 publication run: `outputs-win5080-main-2026-06-28/`
- Cell 1 (`level_c_qwen15b_bf16_math500_seed0`) stopped at ~Q12/500; **10 rows saved** (not for paper)
- Timing: Q1 ~50 s; Q2–Q4 ~21 min each; Q5–Q11 ~15 min each → **~3 weeks for 4-cell grid**
- **Decision:** stop 5080; all publication experiments on HPC only
- Partial archive preserved; `clean_5080_run.sh` killed background jobs

---


### 2026-06-29 — Smoke passed, b01 submitted, parallel state race fixed and committed

#### HPC job state checked at 2026-06-29 12:40 IST

| Job | Purpose | State | Notes |
|-----|---------|-------|-------|
| `85306` | Exclusive quick smoke | **COMPLETED** | Gate 3 passed, exit `0:0` |
| `85342` | b01 BF16 anchors | **RUNNING** | Qwen-7B branch running on `ragpu008`; Llama-8B branch failed early |
| `85343` | b02 FP8 | **PENDING** | `QOSMaxGRESPerUser` |
| `85344` | b03 AWQ-4 | **PENDING** | `QOSMaxGRESPerUser` |
| `85345` | b04 GPTQ-4 | **PENDING** | `QOSMaxGRESPerUser` |
| `85346` | b05 GPTQ-3 | **PENDING** | `QOSMaxGRESPerUser` |
| `85347` | b06 GSM8K | **PENDING** | `QOSMaxGRESPerUser` |

Simple meaning: 1 job is running, 5 jobs are waiting, 1 recent smoke succeeded, and 1 older smoke failed.

#### b01 failure diagnosis

The b01 SLURM block launches two inference processes in parallel against the same archive root:

- `level_a_qwen7b_bf16_math500_seed0`
- `level_c_llama8b_bf16_math500_seed0`

The Llama-8B process failed immediately while updating shared state:

```text
FileNotFoundError: state.json.tmp -> state.json
```

Root cause: `src/runners/checkpoint_utils.py:update_state()` used one shared temporary file name, `state.json.tmp`. In a parallel block, process A can replace/remove that temp file while process B is still trying to replace it.

Fix applied: `update_state()` now uses a `state.json.lock` file plus a unique temporary file from `tempfile.mkstemp()`. This prevents both the missing-temp crash and lost concurrent state writes for future job starts.

Validation: an 8-process local concurrency check repeatedly updated one shared `state.json` and passed without stale keys or temp-file failures.

Local commit: `6dc8ed3 Fix concurrent HPC state updates`.

The already-running Qwen-7B process loaded the old code before this fix, but it is now the only surviving process in b01, so the specific two-process state race is no longer active inside job `85342`. Queued jobs b02-b06 should load the fixed code when SLURM starts them.

#### Durable output observed

- Archive: `outputs-hpc-2a100-main-2026-06-29/`
- Durable Qwen-7B raw rows: `10/500`
- Current log had reached generation at row `20/500`; checkpoint interval is 10 rows, so rows after 10 are not durable until the next checkpoint lands.

### 2026-06-29 — Corrected b01 resubmitted after queue check

After checking the broader GPU queue, the pending competition was either blocked by dependencies, group run-minute limits, or user GPU quotas. This made it a reasonable window to repair b01 ordering.

Actions taken:

- Held queued jobs `85343`-`85347`.
- Canceled old b01 job `85342` after Qwen had checkpointed `20/500` rows and Llama had already failed.
- Submitted corrected b01 as `85394`; it started on `ragpu008`.
- Released b02-b06 after `85394` was running.

Current b01 state:

- Qwen-7B BF16 resumed from `20/500` durable rows.
- Llama-8B BF16 restarted from `0/500` and passed the previous immediate `state.json.tmp` crash point.
- b02-b06 are pending on `QOSMaxGRESPerUser` behind the running corrected b01.

## HPC Gate Checklist (PARAM Rudra)

| Gate | Command / artifact | Status |
|------|-------------------|--------|
| 1 GPU + vLLM | `01_gpu_check.sh` — job 85013 | **PASSED** |
| 2 Model | Qwen-7B + all b01–b06 models on scratch | **DONE** |
| 2b Dataset | MATH-500 + GSM8K via task configs | **DONE** |
| 2c CPU preflight | `07_preflight_publication.py` | **PASSED** |
| 3 GPU smoke | `smoke_test_quick.jsonl` | **PASSED** — job 85306, exit `0:0` |
| 4 Debug n=10 | `level_a_qwen7b_bf16_math500_seed0_summary.json` (n=10) | Not started |
| 4b Full n=500 | Same summary (n=500) | Not started |
| Publication b01–b06 | `submit_hpc_blocks.sh` | b01 running; b02–b06 queued behind GPU QoS |

---

## Known Failures and Fixes (reference)

| Issue | Symptom | Fix |
|-------|---------|-----|
| Unpinned vLLM on HPC | pip installs 0.23.0, Rust build fails | Pin `vllm==0.8.5` |
| Tokenizer compat | `all_special_tokens_extended` missing (85028) | Shim in `vllm_runner.py` |
| Shared GPU OOM | Only ~24 MiB free (85092) | Exclusive smoke + memory preflight |
| Prompt format | `KeyError: 'ANSWER'` (85094) | `{{ANSWER}}` in `math500.txt` |
| Blackwell CUDA | no kernel for sm_120 with torch 2.6 | torch 2.11+cu128 on 5080 only |
| vLLM 0.8.5 + torch 2.11 | ABI mismatch on 5080 | vLLM 0.23.0 on 5080; 0.8.5 on HPC |
| 7B BF16 on 5080 | KV cache OOM | Defer to HPC A100 (by design) |
| HPC git push | Permission denied (publickey) | Sync HPC → MacBook → push |
| Telegram on compute nodes | curl can't reach api.telegram.org | Ignore; use `squeue` / logs |
| Parallel state update race | `FileNotFoundError: state.json.tmp -> state.json` in b01 Llama-8B branch | `update_state()` now locks `state.json` and uses unique temp files |

---

## Sync Workflow (MacBook ↔ GitHub ↔ HPC)

HPC **cannot push** to GitHub. Standard 3-step sync when user says **"sync"**:

1. **MacBook:** `bash scripts/macbook/rsync_from_hpc.sh` (pull HPC-only changes)
2. **MacBook:** review → `git commit` → `git push origin main`
3. **HPC:** `cd $QR && git fetch origin && git reset --hard origin/main`

**Results only (separate):** `bash scripts/macbook/sync_results_from_hpc.sh`

**Do not `git pull` on HPC mid-run** while smoke or publication jobs are active.

---

## Immediate Next Actions

### HPC (paper numbers)

```bash
ssh -p 4422 manishn_iitp@paramrudra.iitp.ac.in
export QR=/scratch/manishn_iitp/reasoning-compression-lab && cd $QR
squeue -u manishn_iitp
sacct -j 85306 --format=JobID,State,ExitCode,Elapsed -P
ls -l runs/raw/smoke_test_quick.jsonl
cat logs/smoke_quick_85306.out
```

1. If smoke **85306** passes → `bash scripts/hpc/submit_hpc_blocks.sh b01` (or full b01–b06).
2. If smoke **fails** → read `.out`/`.err`, fix, resubmit `sbatch slurm/smoke_test_quick_exclusive.slurm`.
3. Do **not** start GPTQ-4 or full grid until BF16 anchor (b01) completes cleanly.

### Windows 5080 (1.5B publication)

```bash
wsl -d Ubuntu-22.04
cd "/mnt/g/ALL MY Projects/2026/03-paper1-experiments"
source scripts/local/env.sh
bash scripts/local/run_5080_publication.sh --skip-download
```

### MacBook

- Keep docs and `progress.md` updated after each gate.
- Push code changes; pull results via rsync when HPC has outputs.

---

## Detailed HPC Session Log — 2026-06-28

### Token Handling

- A GitHub personal access token was pasted in chat earlier.
- It was not saved to disk by Codex.
- Treat that token as compromised and revoke it in GitHub.
- Repo operations in this session used existing local Git/HPC authentication only.

### Scheduler State Before Work

- `squeue -u $USER` showed no running or pending jobs.
- Recent `sacct` history from the prior run showed failures for earlier smoke/publication attempts, including `85028`, `85092`, `85093`, and `85094`.
- Current live queue was empty before starting this setup pass.

### Repo State and Update

- Existing scratch repo found at `/scratch/manishn_iitp/reasoning-compression-lab`.
- Initial state before update:
  - Branch: `main`.
  - Commit: `dff36c1`.
  - Local uncommitted files existed:
    - Modified: `CHANGELOG.md`.
    - Untracked: `AGENTS.md`.
    - Untracked: `scripts/hpc/03a_preflight_cpu.sh`.
  - New publication block directory was missing: `configs/machine_split/hpc_blocks/`.
- Fetched `origin/main`; remote had two newer commits:
  - `30c8c08` add 5080/HPC split with publication run scripts and 48h SLURM blocks.
  - `03c3766` revise split so 5080 runs 1.5B only and HPC runs full 7B/8B grid.
- Local scratch-only changes were preserved with:
  - `git stash push -u -m hpc-local-before-publication-blocks`
- Repo was fast-forwarded to `origin/main`.
- Confirmed new HPC block files exist:
  - `configs/machine_split/hpc_blocks/b01_parallel_bf16_anchors.sh`
  - `configs/machine_split/hpc_blocks/b02_gpqa_fp8.sh`
  - `configs/machine_split/hpc_blocks/b02_parallel_fp8.sh`
  - `configs/machine_split/hpc_blocks/b03_parallel_awq4.sh`
  - `configs/machine_split/hpc_blocks/b04_parallel_gptq4.sh`
  - `configs/machine_split/hpc_blocks/b05_single_gptq3.sh`
  - `configs/machine_split/hpc_blocks/b06_single_gsm8k.sh`
- Confirmed submit script exists:
  - `scripts/hpc/submit_hpc_blocks.sh`

### Environment Checks

- Conda environment `qreason` exists and activates.
- Python version in `qreason`: `3.11.15`.
- Hugging Face CLI auth is configured:
  - `hf auth whoami` returned `user=Manish99`.
  - `hf --version` returned `1.21.0`.

### Static Code Checks

Passed:

- Shell syntax:
  - `scripts/hpc/submit_hpc_blocks.sh`
  - `scripts/hpc/run_hpc_2a100_publication.sh`
  - `slurm/hpc_2a100_b01_parallel.slurm`
  - `slurm/hpc_2a100_b07_gpqa.slurm`
- Python compile check:
  - `python -m compileall -q scripts src`

Incomplete / needs follow-up:

- Combined import check for `torch`, `transformers`, `vllm`, and `datasets` hung during deep `vllm`/`transformers` import on the login environment, inside SciPy import machinery.
- The check was interrupted manually. This is not yet evidence of a runtime failure, but it means package import validation needs a more targeted retry, preferably avoiding heavyweight login-node import behavior or running inside an allocated GPU job.

### Model Inventory Before Downloads

Model root: `/scratch/manishn_iitp/reasoning-compression-lab/models`

Present before this session's downloads:

- `DeepSeek-R1-Distill-Qwen-7B`: 24 files, about 14.19 GiB.

Missing before this session's downloads:

- `DeepSeek-R1-Distill-Llama-8B`
- `DeepSeek-R1-Distill-Qwen-7B-FP8`
- `DeepSeek-R1-Distill-Qwen-7B-AWQ-4`
- `DeepSeek-R1-Distill-Qwen-7B-GPTQ-4`
- `DeepSeek-R1-Distill-Qwen-7B-GPTQ-3`
- `DeepSeek-R1-Distill-Llama-8B-FP8`
- `DeepSeek-R1-Distill-Llama-8B-AWQ-4`
- `DeepSeek-R1-Distill-Llama-8B-GPTQ-4`

### Model Downloads Started

A detached tmux session was started:

- Session: `hpc_model_downloads`
- Log: `logs/hpc_model_downloads_20260628_174440.log`

Download command sequence in that session:

1. `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` -> `models/DeepSeek-R1-Distill-Llama-8B`
2. `RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic` -> `models/DeepSeek-R1-Distill-Qwen-7B-FP8`
3. `jakiAJK/DeepSeek-R1-Distill-Qwen-7B_AWQ` -> `models/DeepSeek-R1-Distill-Qwen-7B-AWQ-4`
4. `RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16` -> `models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4`
5. `irish-quant/deepseek-ai-DeepSeek-R1-Distill-Qwen-7B-3bit` -> `models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-3`
6. `RedHatAI/DeepSeek-R1-Distill-Llama-8B-FP8-dynamic` -> `models/DeepSeek-R1-Distill-Llama-8B-FP8`
7. `jakiAJK/DeepSeek-R1-Distill-Llama-8B_AWQ` -> `models/DeepSeek-R1-Distill-Llama-8B-AWQ-4`
8. `RedHatAI/DeepSeek-R1-Distill-Llama-8B-quantized.w4a16` -> `models/DeepSeek-R1-Distill-Llama-8B-GPTQ-4`

Observed progress:

- Llama-8B BF16 completed enough to move to the next model.
- Disk usage for `models/DeepSeek-R1-Distill-Llama-8B` was about 15 GiB after the first download.
- Downloader then moved to `DeepSeek-R1-Distill-Qwen-7B-FP8`.
- The log includes Hugging Face lock wait messages. The locks were associated with active `hf download` processes, not confirmed stale locks.

### Jobs Submitted

- None yet in this resumed pass.
- Submission is intentionally held until:
  - all required model folders for b01-b06 are present,
  - model/config wiring is checked,
  - dataset access is checked,
  - targeted environment checks are complete,
  - current `squeue` is reviewed.

## HPC Preflight Checklist

- [x] Confirm live SLURM queue is empty before setup.
- [x] Update scratch repo to latest `origin/main`.
- [x] Preserve older scratch-only changes in a stash.
- [x] Confirm HPC block scripts exist.
- [x] Confirm `qreason` environment activates.
- [x] Confirm Hugging Face auth.
- [x] Run shell syntax checks.
- [x] Run Python compile checks.
- [x] Finish model downloads for all b01-b06 model folders.
- [x] Verify every cell config points to an existing task config, model config, and local model path.
- [x] Verify MATH-500 and GSM8K dataset access/cache.
- [x] Run targeted package checks that do not hang indefinitely on the login node.
- [x] Add repeatable preflight script (`07_preflight_publication.py`).
- [ ] GPU smoke passes (`smoke_test_quick.jsonl` exists).
- [ ] Submit selected SLURM publication jobs (b01–b06).
- [ ] Record job IDs and output archive path.
- [ ] Monitor initial SLURM logs for early failures.

## 5080 Rig Notes

**Status (2026-06-28):** Environment ready; 12 models on disk (~62 GB); 1.5B smoke passed; publication script ready.

Expected 5080 scope from the runbook:

- Qwen-1.5B BF16 on MATH-500.
- Qwen-1.5B FP8 on MATH-500.
- Qwen-1.5B AWQ-4 on MATH-500.
- Qwen-1.5B GPTQ-4 on MATH-500.
- Do not run 7B/8B cells on the 5080.

**Entry point:** `bash scripts/local/run_5080_publication.sh --skip-download`

## Commands for Resuming HPC Work

Check downloader:

```bash
cd /scratch/$USER/reasoning-compression-lab
tmux ls
tail -120 logs/hpc_model_downloads_20260628_174440.log
ps -fu $USER | grep -E 'hpc_model_downloads|hf download' | grep -v grep
```

Check queue:

```bash
squeue -u $USER
```

After smoke passes, verify models and then submit:

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh
conda activate qreason
sbatch slurm/smoke_test_quick_exclusive.slurm   # if smoke not yet passed
bash scripts/hpc/submit_hpc_blocks.sh b01       # after smoke passes
# or, after confidence is high:
# bash scripts/hpc/submit_hpc_blocks.sh
```

## 2026-06-28 Update 2 — Preflight Results After Downloads

### Download Session Completed

- `tmux` session `hpc_model_downloads` completed at `Sun Jun 28 17:56:11 IST 2026`.
- No active `hf download` process remains for this repo.
- A separate older process unrelated to this repo was visible for `Qwen3-235B-GPTQ-Int4` under `/scratch/manishn_iitp/models`; it was not touched.

### Model Folder Verification

All required b01-b06 model folders now exist and passed basic completeness checks: `config.json` present, tokenizer present, and weight files present.

| Model folder | Size | Weight files |
| --- | ---: | ---: |
| `DeepSeek-R1-Distill-Qwen-7B` | 14.19 GiB | 2 |
| `DeepSeek-R1-Distill-Llama-8B` | 14.97 GiB | 2 |
| `DeepSeek-R1-Distill-Qwen-7B-FP8` | 8.12 GiB | 2 |
| `DeepSeek-R1-Distill-Qwen-7B-AWQ-4` | 5.20 GiB | 2 |
| `DeepSeek-R1-Distill-Qwen-7B-GPTQ-4` | 5.17 GiB | 2 |
| `DeepSeek-R1-Distill-Qwen-7B-GPTQ-3` | 4.44 GiB | 2 |
| `DeepSeek-R1-Distill-Llama-8B-FP8` | 8.47 GiB | 2 |
| `DeepSeek-R1-Distill-Llama-8B-AWQ-4` | 5.35 GiB | 2 |
| `DeepSeek-R1-Distill-Llama-8B-GPTQ-4` | 5.32 GiB | 2 |

### Block/Config Wiring Verification

A first custom checker incorrectly looked for `model_path` / `local_path` keys in model configs. That was a checker bug, not a repo bug.

The actual repo resolver in `src/runners/config_utils.py` uses:

- `local_path_env`
- `local_path_default`

Rerunning the check through `load_cell_config()` succeeded for every HPC block config:

- `b01_parallel_bf16_anchors.sh`
  - `level_a_qwen7b_bf16_math500_seed0` -> `models/DeepSeek-R1-Distill-Qwen-7B`
  - `level_c_llama8b_bf16_math500_seed0` -> `models/DeepSeek-R1-Distill-Llama-8B`
- `b02_gpqa_fp8.sh`
  - `level_c_qwen7b_fp8_gpqa_seed0` -> `models/DeepSeek-R1-Distill-Qwen-7B-FP8`
- `b02_parallel_fp8.sh`
  - `level_b_qwen7b_fp8_math500_seed0` -> `models/DeepSeek-R1-Distill-Qwen-7B-FP8`
  - `level_c_llama8b_fp8_math500_seed0` -> `models/DeepSeek-R1-Distill-Llama-8B-FP8`
- `b03_parallel_awq4.sh`
  - `level_b_qwen7b_awq4_math500_seed0` -> `models/DeepSeek-R1-Distill-Qwen-7B-AWQ-4`
  - `level_c_llama8b_awq4_math500_seed0` -> `models/DeepSeek-R1-Distill-Llama-8B-AWQ-4`
- `b04_parallel_gptq4.sh`
  - `level_a_qwen7b_gptq4_math500_seed0` -> `models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4`
  - `level_c_llama8b_gptq4_math500_seed0` -> `models/DeepSeek-R1-Distill-Llama-8B-GPTQ-4`
- `b05_single_gptq3.sh`
  - `level_b_qwen7b_gptq3_math500_seed0` -> `models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-3`
- `b06_single_gsm8k.sh`
  - `level_b_qwen7b_fp8_gsm8k_seed0` -> `models/DeepSeek-R1-Distill-Qwen-7B-FP8`

Conclusion: block -> cell -> task/model config -> local model path wiring is valid for b01-b06 and also resolves for b07, although b07 remains gated.

### Dataset Validation

A first manual test used the obsolete short dataset id `gsm8k`, which failed with an HF URI error. The repo config is correct and uses `openai/gsm8k`.

Validation through the actual repo task configs passed:

- `configs/tasks/math500.json`
  - Dataset: `HuggingFaceH4/MATH-500`
  - Split: `test`
  - Rows: `500`
  - Columns: `problem`, `solution`, `answer`, `subject`, `level`, `unique_id`
- `configs/tasks/gsm8k.json`
  - Dataset: `openai/gsm8k`
  - Config: `main`
  - Split: `test`
  - Rows: `1319`
  - Columns: `question`, `answer`

### Queue State

- `squeue -u $USER` was still empty after downloads and validation.

## 2026-06-28 Update 3 — Runtime Risk Review Before Submission

### Package Checks

Targeted checks completed:

- `torch` import passed.
  - Version: `2.6.0+cu124`
  - CUDA build: `12.4`
  - `torch.cuda.is_available()` on login node: `False` expected because this was not inside a GPU allocation.
- `datasets` import passed.
  - Version: `5.0.0`
- `transformers` import passed.
  - Version: `5.12.1`
- `vllm` import did not complete within a 180 second login-node timeout.
  - This is documented as a login-node/package import risk.
  - Previous GPU smoke logs show vLLM can initialize on a GPU node, so this is not being treated as a hard blocker by itself.

Installed package metadata:

- `vllm==0.8.5`
- `torch==2.6.0`
- `transformers==5.12.1`
- `datasets==5.0.0`
- `scipy==1.17.1`

### Prior Failure Review

Old logs were reviewed:

- `85028` failed with tokenizer compatibility: `Qwen2Tokenizer has no attribute all_special_tokens_extended`.
- `85092` got past tokenizer initialization but failed with CUDA OOM because the allocated/shared GPU had almost no free memory.
- `85094` exclusive quick smoke loaded the Qwen-7B BF16 model successfully, then failed at prompt formatting with `KeyError: 'ANSWER'`.

Current repo state after `git pull` fixes the prompt issue:

- `prompts/math500.txt` now escapes the example braces as `{{ANSWER}}`.
- Direct test of `build_prompt('prompts/math500.txt', question='2+2?')` passed and produced a literal `{ANSWER}` in the prompt.
- There are no local diffs in `prompts/math500.txt`, `src/runners/vllm_runner.py`, `scripts/smoke_test.py`, or `scripts/run_inference.py` besides the new `progress.md`.

### Submission Decision

Because previous failures were caused by issues that are either fixed or allocation-dependent, the next gate is an exclusive quick smoke SLURM job before submitting b01-b06.

Planned command:

```bash
sbatch slurm/smoke_test_quick_exclusive.slurm
```

If that smoke passes, submit publication jobs. If it fails, inspect `logs/smoke_quick_JOBID.out` and `.err` before submitting any publication block.

## 2026-06-28 Update 4 — Smoke Job Submitted

Submitted exclusive quick smoke test:

- Job ID: `85306`
- Command: `sbatch slurm/smoke_test_quick_exclusive.slurm`
- Initial queue state: `PD` pending with reason `(Priority)`.
- Expected logs after start:
  - `logs/smoke_quick_85306.out`
  - `logs/smoke_quick_85306.err`

Publication blocks are still not submitted. They are waiting on smoke result.

## 2026-06-28 Update 5 — README/CHANGELOG Cross-Check

The pulled repo documentation was explicitly reviewed after the user reminder.

Files read:

- `README.md`
- `CHANGELOG.md`

Confirmed current machine split:

- RTX 5080 runs only Qwen-1.5B x 4 quant cells on MATH-500.
- HPC runs b01-b06 for 7B/8B work and GSM8K:
  - b01: BF16 Qwen-7B + BF16 Llama-8B MATH-500
  - b02: FP8 Qwen-7B + FP8 Llama-8B MATH-500
  - b03: AWQ-4 Qwen-7B + AWQ-4 Llama-8B MATH-500
  - b04: GPTQ-4 Qwen-7B + GPTQ-4 Llama-8B MATH-500
  - b05: GPTQ-3 Qwen-7B MATH-500
  - b06: FP8 Qwen-7B GSM8K
- b07 GPQA is not part of the immediate run and must wait for Hugging Face gate approval.
- Do not run any 5080-scope Qwen-1.5B publication cells on HPC unless explicitly redirected later.

Current smoke gate state:

- Smoke job `85306` remains pending.
- `sacct` state: `PENDING`, start `Unknown`, end `Unknown`.
- Queue reason from `squeue`: `(Priority)`.
- No publication block jobs have been submitted.

Operational rule from this point:

- Wait for smoke job `85306` to pass before submitting b01-b06.
- If smoke fails, inspect `logs/smoke_quick_85306.out` and `logs/smoke_quick_85306.err`, update this file, and fix the root cause before publication submission.

## 2026-06-28 Update 6 — Repeatable Preflight Added

Added `scripts/hpc/07_preflight_publication.py` so the HPC preflight is no longer just terminal history.

The script checks:

- HPC shell scripts and SLURM wrapper syntax.
- Python compile for `scripts` and `src`.
- Math prompt formatting, including literal `{ANSWER}` preservation.
- b01-b06 block presence.
- No Qwen-1.5B 5080 cells in b01-b06 HPC blocks.
- Cell config resolution through `load_cell_config()`.
- Local model folder existence plus `config.json`, tokenizer, and weights.
- MATH-500 row count equals `500`.
- GSM8K test row count equals `1319`.

Result on HPC: passed.

Current answer to "why smoke instead of the whole experiment":

- The full jobs are expensive 47-hour SLURM allocations.
- Prior failures occurred before useful experiment work completed: tokenizer compatibility, GPU OOM on a shared node, and prompt formatting.
- CPU preflight now covers config/dataset/model/prompt/compile failures.
- The remaining untested part is the real GPU/vLLM engine path, which requires a GPU allocation.
- The exclusive quick smoke job tests that path with one question before b01-b06 are submitted.

Current smoke state remains pending under scheduler priority; no b01-b06 publication jobs have been submitted.

## 2026-06-28 Update 7 — Codex Notes and Credential Boundary

Created `/home/manishn_iitp/.codex/CODEX.md` for future Codex sessions. It records:

- HPC vs 5080 machine split.
- Required CPU preflight command.
- Required GPU smoke gate before b01-b06 submission.
- Current local-ahead commit state.
- Credential rule: never store GitHub tokens in repo files or persistent config.

Push status remains blocked by missing safe GitHub credentials on HPC:

- HTTPS push prompts cannot read username/password in this non-interactive environment.
- `gh` is not configured on HPC.
- SSH to GitHub on port 22 timed out.
- Codex will not write the pasted PAT to repo files or persistent credential storage.

Safe push options:

1. Run `git push origin main` from a terminal where Git can prompt for credentials.
2. Configure `gh auth login` on HPC or another machine, then push.
3. Add an SSH key with GitHub access and switch the remote to SSH.
4. Provide a one-time token through a secure credential prompt or environment mechanism, not committed files.

---

## How to Maintain This File

- Update **Current Status Snapshot** when a gate passes or a new blocker appears.
- Add a dated section under **Timeline by Date** for each major session.
- Keep detailed HPC session logs at the bottom for operational replay.
- Cross-update `CHANGELOG.md` and `docs/EXPERIMENT_LOG.md` for experiment cells and job IDs.
