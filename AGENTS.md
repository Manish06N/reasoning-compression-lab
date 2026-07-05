# CLAUDE.md — manishn_iitp @ PARAM Rudra HPC

This file is auto-loaded by Claude Code at the start of every session.

## System Overview
- **User:** manishn_iitp, IIT Patna
- **Cluster:** PARAM Rudra HPC (C-DAC / NSM), NVIDIA A100 80GB GPUs
- **Shell:** bash, home = `/home/manishn_iitp`, scratch = `/scratch/manishn_iitp/`
- **Python:** always use `python3` (not `python`)
- **GPU jobs:** always submit via SLURM — login node kills any compute process

---

## SSH Tunnel (copy-paste this every session)

Run this on your **home Windows PC** (PowerShell) to forward the vLLM port:

```powershell
ssh -L 8080:<NODE>:8080 -N manishn_iitp@paramrudra.iitp.ac.in -p 4422
```

> **Replace `<NODE>`** with the current compute node from `squeue` output (e.g. `ragpu004` or `ragpu005`).

**Full example (if node is ragpu004):**
```powershell
ssh -L 8080:ragpu004:8080 -N manishn_iitp@paramrudra.iitp.ac.in -p 4422
```

**Auth flow:** CAPTCHA string → Google Authenticator code → password

**Verify tunnel works (run locally):**
```bash
curl http://localhost:8080/health
# Expected: 200 OK
```

---

## Active vLLM Servers & Models

| Model | Shards Path | Serving Script | GPUs |
|---|---|---|---|
| **DeepSeek-R1-70B** | `/scratch/manishn_iitp/models/DeepSeek-R1-Distill-Llama-70B` | `~/start-llm-deepseek.sh` | 2 |
| **MiniMax-M2.7-XL** | `/scratch/manishn_iitp/models/MiniMax-M2.7-UD-Q4_K_XL/UD-Q4_K_XL` | `~/serve-minimax.sh` | 2 |
| **GLM-4.7-XL** | `/scratch/manishn_iitp/models/GLM-4.7-UD-Q2_K_XL/UD-Q2_K_XL` | `~/serve-glm.sh` | 2 |

---

## How to Switch Models (tell Claude: "Start the vLLM server")
1. **User Preference:** The user will verbally tell the agent which model to run (e.g., "Start MiniMax").
2. **Agent Action:** The agent MUST check running jobs (`squeue`), stop any active job (`scancel`), and then start the requested job using the appropriate script (e.g., `sbatch serve-minimax.sh`).
3. **Verification:** Wait for node assignment, then verify with `curl http://<NODE>:8080/health`.
4. **Communication:** Provide the user with the exact SSH tunnel command for their local machine.

---

## SLURM Rules for This Cluster
- `--gres=gpu:1` or `--gres=gpu:2` — NO `--mem`, NO GPU type tag (e.g. no `--gres=gpu:a100:2`)
- Always `--enforce-eager` for vLLM (triton JIT fails on compute nodes)
- Run heavy jobs from `/scratch/$USER/`, back up results to `$HOME`
- conda env: `/home/apps/MSCC/miniconda3/` (do NOT use the module system for Python/conda)

### Never `--exclusive` on parallel 1-GPU publication cells (2026-07-03)

- User quota: **2 GPUs** (`QOSMaxGRESPerUser`).
- ragpu nodes have **2× A100**; `--exclusive` with `--gres=gpu:1` counts as **2 GPUs** toward quota.
- Two split b01 cells need **non-exclusive** `gres/gpu:1` each (1+1=2). Exclusive second cell → `QOSMaxGRESPerUser`.
- **Read:** `docs/PARAM_RUDRA_SLURM.md`. **Use:** `bash scripts/hpc/submit_hpc_blocks.sh b01` or `… cell <config.json>` — never manual `sbatch --exclusive` for 1-GPU inference.
- Dirty GPUs: `metadata/dirty_nodes.txt` + `QREASON_MIN_FREE_GPU_MB`, not exclusive.

---

## Cluster Policies & FairShare (Important)
- **Max GPUs:** 2 per user at any time (`QOSMaxGRESPerUser`).
- **Max Walltime:** 48 hours (2 days) per job. Jobs are automatically killed after this. This is **not** a monthly limit; jobs can be resubmitted immediately.
- **FairShare:** A dynamic priority system. High usage leads to lower FairShare scores, which may increase wait times in the queue, but does not block jobs. Score recovers over time (decay).
- **Agent Mandate:** Periodically check `squeue`, `sacct`, and `sshare -u $USER -l` to update the user on job status, cumulative usage, and priority score.
- **Experiment Policy:** Publication experiments are HPC-only. Do not plan or schedule Windows/5080 experiment runs; that machine is retired for publication work because projected runtime is weeks. If Qwen-1.5B cells are still needed, plan them as HPC jobs after downloading and preflighting the required model variants.

---

## HPC Queue Repair Workflow

Use this when an experiment job is partially failed, wasting GPUs, or needs to be resubmitted with corrected code while follow-up jobs are already queued.

1. Inspect the user queue and the wider GPU partition before changing anything:
   ```bash
   squeue -u $USER
   squeue -p gpu -o "%.18i %.9P %.30j %.12u %.2t %.12M %.5D %.6C %.8b %.20R"
   squeue -p gpu --sort=-p,i -o "%.18i %.10Q %.30j %.12u %.2t %.12M %.5D %.6C %.8b %.25R"
   sinfo -p gpu -o "%P %.8a %.10l %.6D %.6t %N"
   ```
2. Read the failing or running job logs and durable checkpoint counts. Distinguish log progress from checkpointed rows.
3. If a priority job must be resubmitted, protect ordering:
   - hold downstream queued jobs first with `scontrol hold <jobids>`,
   - cancel the broken running job with `scancel <jobid>`,
   - submit the corrected priority job with the appropriate `sbatch ...`,
   - wait until the corrected job is running or safely queued ahead,
   - release downstream jobs with `scontrol release <jobids>`.
4. Verify the corrected job actually passed the old failure point:
   - check `squeue -u $USER`,
   - tail the new SLURM `.out` and `.err`,
   - confirm resume messages and durable row counts with `wc -l`.
5. Document the operational change immediately in the project `CHANGELOG.md` and `progress.md`, including old job ID, new job ID, node, checkpoint resume point, downstream holds/releases, and GitHub sync status.
6. Commit and push documentation or code changes when possible:
   ```bash
   git status -sb
   git add CHANGELOG.md progress.md <changed-code-files>
   git commit -m "<clear operational message>"
   git push origin main
   ```

Example from 2026-06-29: b01 job `85342` had one failed branch and one live branch, so downstream jobs `85343`-`85347` were held, `85342` was canceled, corrected b01 `85394` was submitted and verified running on `ragpu008`, then b02-b06 were released back to `QOSMaxGRESPerUser`.

---

## Key Files
| File | Purpose |
|---|---|
| `~/start-llm-deepseek.sh` | SLURM script for DeepSeek-R1-70B |
| `~/serve-minimax.sh` | SLURM script for MiniMax-M2.7-XL (GGUF) |
| `~/serve-glm.sh` | SLURM script for GLM-4.7-XL (GGUF) |
| `~/progress.md` | Full session log (all errors, fixes, history) |
| `~/watch-job-52772.sh` | Job watcher script |
| `~/llm_setup/` | Setup scripts (build, download, serve, tunnel, test) |

---

## Model Compatibility Note
- vLLM version: **0.8.5**
- **Models Ready on Disk:**
    - `DeepSeek-R1-70B`
    - `MiniMax-M2.7-XL`
    - `GLM-4.7-XL`
- All models require **2 GPUs** and have corresponding `serve-*.sh` scripts in `~/`.
- Qwen and Gemma models have been deleted.

---

## Active Project: reasoning-compression-lab

Use this section when working in:

```bash
/scratch/manishn_iitp/reasoning-compression-lab
```

### Current Sync State

- GitHub, MacBook, and HPC were synced on **2026-06-27**.
- Verified synced commit across all three places:

```text
dff36c1 Sync HPC smoke fixes: tokenizer shim, memory preflight, quick smoke SLURM.
```

- Verified locations:
  - GitHub `origin/main`: `dff36c1`.
  - MacBook repo: `dff36c1`, clean, `## main...origin/main`.
  - HPC repo: `dff36c1`, clean, `## main...origin/main`.
- Future sessions should read project memory before major actions:
  - `/scratch/manishn_iitp/reasoning-compression-lab/AGENTS.md`
  - `/scratch/manishn_iitp/reasoning-compression-lab/CHANGELOG.md`
  - This `/home/manishn_iitp/CODEX.md` section.

- Verify with:

```bash
cd /scratch/manishn_iitp/reasoning-compression-lab
git status -sb
git log --oneline -3
```

- Expected status: branch `main` is up to date with `origin/main`, working tree clean.

### Project-Specific Environment

- Conda env: `qreason`.
- Conda root: `/home/apps/MSCC/miniconda3`.
- Repo root: `/scratch/manishn_iitp/reasoning-compression-lab`.
- Main model for current experiment:

```text
/scratch/manishn_iitp/reasoning-compression-lab/models/DeepSeek-R1-Distill-Qwen-7B
```

- Do not re-download the model, redo HF login, or rerun earlier setup gates unless logs show they broke.
- Keep `models/`, `hf_cache/`, `runs/`, `results/`, and `logs/` out of GitHub.

### Current Experiment Gate (2026-07-05)

- **Active:** Path C diagnostic (jobs **87116–87118**, archive `outputs-hpc-diag-pathc-2026-07-05`).
- **b01 QRM gate:** **FAILED** (July archive `outputs-hpc-2a100-main-2026-07-03`).
- **Path C early signal (n=20):** Qwen **10%** pass@1 / **90%** trunc; Llama **15%** / **75%** trunc — protocol correct, **stack not QRM-equivalent**.
- **Stack parity fixes landed:** `src/runners/vllm_serving.py`, d03 parity pilot — see `docs/QRM_STACK_PARITY_AUDIT.md`.
- **Quant grid b02–b06:** **On hold** until Path C + parity pilot + optional official QRM cross-check.
- **Do not** claim QRM Table 1 reproduction without `compare_qrm_baseline.py` hard_passed on strict protocol runs.

### Important Jobs From 2026-06-26 and 2026-06-27

- `85013` (`qreason-gpu-check`): completed successfully.
  - Proved CUDA, PyTorch, and vLLM on A100.
  - PyTorch: `2.6.0+cu124`.
  - vLLM: `0.8.5`.
- `85028` (`qreason-smoke`): failed.
  - Root cause: tokenizer compatibility bug.
  - Error: `Qwen2Tokenizer has no attribute all_special_tokens_extended`.
  - Artifact missing: `runs/raw/smoke_test.jsonl`.
- `85092` (`qreason-smoke`): failed after code fix.
  - Tokenizer bug did **not** recur.
  - New root cause: shared GPU out of memory.
  - The assigned A100 had only `23.62 MiB` free because other processes were using GPU memory.
- `85030` (`qreason-level-a-bf16`): canceled.
  - It was a dependent 10-question debug job and could never run after smoke failed.
- `85094` (`qreason-smoke-quick`): quick exclusive smoke job.
  - Last known state: `PENDING (Resources)`.
  - Do not submit another smoke job while `85094` is queued or running.

Check current state with:

```bash
squeue -u $USER
sacct -j 85094 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End -P
```

### Code Fixes Already Applied

- `src/runners/vllm_runner.py`
  - Adds `_ensure_tokenizer_compatibility()`.
  - Bridges `vllm==0.8.5` with `transformers==5.12.1`.
  - Adds `all_special_tokens_extended` to `PreTrainedTokenizerBase` only when missing.
- `scripts/hpc/03_smoke_test.sh`
  - Uses `python -u` for unbuffered logs.
  - Supports:
    - `SMOKE_LIMIT`
    - `SMOKE_OUTPUT`
    - `SMOKE_MAX_TOKENS`
    - `SMOKE_MIN_FREE_GPU_MB`
  - Performs a free-GPU-memory preflight using `nvidia-smi`.
- `slurm/smoke_test_quick_exclusive.slurm`
  - Quick one-question validation job.
  - Uses `SMOKE_MAX_TOKENS=64`.
  - Requests `--exclusive` with `--gres=gpu:1`.
  - Writes `runs/raw/smoke_test_quick.jsonl`.
- `CHANGELOG.md`
  - Root-level project changelog with detailed job history and fix notes.

Verify fixes with:

```bash
grep -n "all_special_tokens_extended" src/runners/vllm_runner.py
grep -n "SMOKE_MIN_FREE_GPU_MB" scripts/hpc/03_smoke_test.sh
ls -l slurm/smoke_test_quick_exclusive.slurm
```

### Smoke-Test Workflow

Preferred quick smoke command:

```bash
sbatch slurm/smoke_test_quick_exclusive.slurm
```

When it finishes, check:

```bash
ls -l runs/raw/smoke_test_quick.jsonl
cat logs/smoke_quick_<JOBID>.out
cat logs/smoke_quick_<JOBID>.err
```

If quick smoke passes, run the 10-question debug:

```bash
LIMIT=10 sbatch slurm/run_level_a_bf16.slurm
```

Only after the 10-question debug passes should the full Level A BF16 MATH-500 run be submitted.

### Shared HPC Caveat

- A scheduled GPU allocation may still land on a GPU with other users' memory already present.
- If vLLM fails with CUDA OOM during model loading and logs show very little free GPU memory, treat it as resource contention rather than a project-code failure.
- Prefer the quick exclusive smoke job for validation on busy days.

### Sync Workflow

- HPC cannot push to GitHub directly. The canonical sync path is:

```text
HPC local changes -> MacBook rsync -> MacBook git commit/push -> HPC fetch/reset
```

- Trigger rule: if the user says "sync", "sync this", "sync to MacBook",
  "sync to GitHub", or similar while working on this project, treat it as a
  request to run the full coordinated sync workflow autonomously up to the
  MacBook boundary:
  1. On HPC, inspect `git status`, `git diff --stat`, and current log.
  2. If there are meaningful local changes, stage and commit them locally on
     HPC with a clear message.
  3. Tell the user exactly what to run on MacBook for Part 2.
  4. Wait for the user to confirm the MacBook push succeeded and provide the
     new GitHub commit hash if possible.
  5. Then run Part 3 on HPC: `git fetch origin`, compare logs, reset to
     `origin/main`, and verify status/fix checks.
  6. Report whether GitHub, MacBook, and HPC are synced.
- Agent rule: after any big project change on HPC, explicitly tell the user:
  - What changed.
  - Whether it is committed locally on HPC.
  - That MacBook/GitHub sync is needed.
  - The exact Part 2 MacBook commands to run.
  - That HPC must not be reset/pulled until the user confirms MacBook push
    succeeded.
- When an agent makes a meaningful project change on HPC, it must tell the user
  that MacBook/GitHub sync is needed. Meaningful changes include:
  - Code changes under `src/`.
  - Script changes under `scripts/` or `slurm/`.
  - Config changes under `configs/`.
  - Project documentation changes such as `CHANGELOG.md`, `AGENTS.md`, or docs
    under `docs/`.
  - Any fix needed to reproduce an HPC run later.
- Small runtime outputs do **not** need GitHub sync:
  - `logs/`
  - `runs/`
  - `results/`
  - `models/`
  - `hf_cache/`

#### Part 1: Save Work On HPC

Run from the project root. This is a local backup only; it does not update
GitHub.

```bash
cd /scratch/manishn_iitp/reasoning-compression-lab
git status
git diff --stat
git add <changed-code-doc-script-files>
git commit -m "<clear commit message>"
```

Do not run `git pull`, `git reset --hard`, or any history-rewriting command on
HPC before MacBook/GitHub has the HPC changes, unless the user explicitly says
the local HPC changes can be discarded.

#### Part 2: User Syncs MacBook And GitHub

The user runs this on MacBook:

```bash
bash "/Users/manish/Projects/2026/paper 1/reasoning-compression-lab/scripts/macbook/rsync_from_hpc.sh"
cd "/Users/manish/Projects/2026/paper 1/reasoning-compression-lab"
git status
git add -A
git commit -m "<sync commit message>"
git push origin main
```

After this step, GitHub and MacBook should share the same latest commit.

#### Part 3: Align HPC With GitHub

Only after the user confirms MacBook push succeeded, run on HPC:

```bash
cd /scratch/manishn_iitp/reasoning-compression-lab
git fetch origin
git status -sb
git log --oneline -3 HEAD
git log --oneline -3 origin/main
git reset --hard origin/main
git status -sb
git log --oneline -3
```

- `git reset --hard origin/main` only changes tracked files. It does not delete
  ignored `models/`, `hf_cache/`, `runs/`, `results/`, or `logs/`.
- After reset, success means:
  - `git status -sb` shows `## main...origin/main`.
  - No ahead/behind marker.
  - Latest HPC commit matches latest GitHub commit.
  - Any expected fix checks still pass, for example:

```bash
grep -n "all_special_tokens_extended" src/runners/vllm_runner.py | head -2
grep -n "SMOKE_MIN_FREE_GPU_MB" scripts/hpc/03_smoke_test.sh | head -1
ls -l slurm/smoke_test_quick_exclusive.slurm
```

#### Current Known Good Sync

- Last verified synced commit across GitHub, MacBook, and HPC:

```text
dff36c1 Sync HPC smoke fixes: tokenizer shim, memory preflight, quick smoke SLURM.
```

- If future agents see local HPC commits ahead of `origin/main`, preserve them
  by following Part 1 -> Part 2 -> Part 3. Do not discard local HPC work unless
  the user explicitly says it has already been copied and pushed.
- Repo-local `AGENTS.md` may be untracked on HPC if created after the last sync.
  If the user wants it in GitHub, include it in the next HPC -> MacBook ->
  GitHub sync.

### Notification Caveat

- Slurm watcher jobs `85031` and `85032` failed to send Telegram notifications because the node could not resolve `api.telegram.org`.
- Treat Telegram watcher failures as network/DNS issues unless the job logs show a separate compute failure.
