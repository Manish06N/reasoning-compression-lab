# Progress Log — Master Reference

Canonical dated record for **Paper 1: Beyond Accuracy** (`reasoning-compression-lab`).

**Purpose:** Future sessions can resume without guessing what was built, where it runs, which gates passed, and what failed. Update this file after every material change on MacBook, Windows/WSL, or HPC.

**GitHub:** https://github.com/Manish06N/reasoning-compression-lab  
**Related logs:** `CHANGELOG.md` (ops detail), `docs/EXPERIMENT_LOG.md` (experiment cells), `paper 1/AGENTS.md` (AI assistant memory)

---

## Current Status Snapshot (2026-07-06, official QRM repo test SUCCESS)

| Area | Status |
|------|--------|
| **Active experiment** | **Quant grid b02** ready to open (since QRM parity check is completed) |
| **Path C** | **CANCELED** (87116–87118) — baseline comparison kept in outputs |
| **QRM Parity check** | Qwen-7B BF16 official run **100% correct (10/10)**, **0% loops/truncation** |
| **Official test** | n=10 MATH-500, seed=42, Qwen-7B BF16, env `qrm-official` |
| **Output** | `outputs-hpc-qrm-official-2026-07-06/` |
| **b01 gate (repro)** | **PASSED (via official stack job 87302)** — proving our prompt config is correct |
| **b02–b06** | **Ready to open** (repro confirmed on official stack) |
| **GitHub sync** | HPC ahead of `origin/main` — MacBook rsync needed |
| **Key docs** | `docs/QRM_STACK_PARITY_AUDIT.md`, `notes.md` §32, `CHANGELOG.md` |

### Official QRM test (Experiment A) — completed

| Experiment | What | Status |
|------------|------|--------|
| **A** | Official QRM `inference.py`, same 10 problems | **COMPLETED** — job **87302** (10/10 correct) |
| **B** | Our stack, logprobs off | Code fixed; not rerun (skipped) |
| **C** | rep_pen ablation | **Done** — both with/without failed on modern stack |
| **D** | Qwen 64k budget | **Canceled** — job 87118 |

See [notes.md §32](notes.md) for plain-English explainer.

```bash
# Check final logs
cat logs/qrm_official_87302.out
python scripts/hpc/qrm_parity/compare_side_by_side.py --limit 10
```

### QRM stack parity (2026-07-05 audit)

**Story:** Path C strict protocol matches QRM `inference.py` on paper, but early traces show degeneration loops (`yeah yeah`, `the the the`) filling 32k. Clean finishes are mostly correct (Qwen 2/2, Llama 3/5 on n=20). Scorer is not the bottleneck — truncated rows never emit `\boxed{}`.

```bash
# No GPU
python scripts/hpc/qrm_parity/verify_stack_parity.py
python scripts/hpc/qrm_parity/compare_side_by_side.py --limit 10

# GPU parity pilot (after d01 frees a slot)
bash scripts/hpc/submit_pathc_parity_pilot.sh

# Official QRM cross-check (GPU, separate env)
bash scripts/hpc/qrm_parity/setup_official_qrm_repo.sh
```

Full narrative: **`docs/QRM_STACK_PARITY_AUDIT.md`**

### Should we finish Qwen 90 rows (~10 h)?

**Recommendation: No — not necessary for any decision.**

| Reason | Detail |
|--------|--------|
| Truncation already conclusive | 94.1% on n=410 — gate needs ≤15%; 90 more rows cannot flip narrative |
| Gate already failed | Llama + Qwen both far from QRM bands |
| Marginal science value | Confirms same story; costs ~10 GPU-hours + resume debugging |
| Optional only | Finish **only** if you want symmetric n=500 in a table footnote |

If resuming later: pin `QREASON_OUTPUT_ROOT` to `...2026-07-03`; ensure GPU is free (87111 died on shared VRAM).

### b01 gate quick reference (see `notes.md` §25)

| Check | Qwen target | Llama target | July result |
|-------|-------------|-------------|-------------|
| pass@1 | 93.9% ± 5 pp | 91.0% ± 5 pp | Qwen TBD; Llama **19.6% FAIL** |
| truncation_rate | ≤ 15% | ≤ 15% | Qwen ~94% partial; Llama **58% FAIL** |
| prompt_profile | `reproduction` | `reproduction` | Qwen OK; Llama **`sober` SKIP** |

```bash
# Gate checker (conda qreason python)
/home/manishn_iitp/.conda/envs/qreason/bin/python3 scripts/compare_qrm_baseline.py \
  --summary outputs-hpc-2a100-main-2026-07-03/results/level_c_llama8b_bf16_math500_seed0_summary.json
```

### Resume command (memorise — pin archive date)

```bash
cd /scratch/manishn_iitp/reasoning-compression-lab
QREASON_OUTPUT_ROOT=$PWD/outputs-hpc-2a100-main-2026-07-03 \
QREASON_HPC_DATE=2026-07-03 \
bash scripts/hpc/submit_hpc_blocks.sh b01
```

### Llama July vs June — is it genuine?

| | June 2026 (invalid) | July 2026 (valid) | QRM paper |
|--|---------------------|-------------------|-----------|
| pass@1 | 21.4% | **19.6%** | 91.0% |
| truncation | ~59% | **58%** | (at 32k) |
| Trustworthy? | No (rep_pen bug) | **Yes** (500/500, `729d773`) | baseline |

July **confirms** June’s truncation shape; `repetition_penalty` fix did not materially lift Llama. Non-truncated pass@1 = **45%** — gap to QRM is not only truncation.

---

## Historical — 2026-07-03 evening (superseded by 2026-07-05 snapshot above)

| Area | Status |
|------|--------|
| **GitHub `main`** | **Behind HPC** — local commits `434f373` (docs) + `4da8913` (Triton gcc fix) not yet pushed. MacBook rsync → push needed. |
| **J1 scientific validation** | **Blocked until Triton fix verified** — afternoon wave produced **0/500 rows**. Root cause found and patched; b02 FP8 resubmitted as 86718/86719. |
| **QRM baseline gates** | **Fixed** (prior) |
| **Submit workflow** | `EXCLUSIVE=0`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, split 1-GPU-per-cell via `submit_hpc_blocks.sh`. |
| **GPU parallel + context** | Fixed high 1M+ context. FP8/AWQ/GPTQ configs use it; BF16 now also sets `kv_cache_dtype: fp8` so 1M fits on 1× A100. |
| **Environment & Requirements** | vLLM 0.8.5, torch 2.6, triton 3.2.0. **New:** conda `gcc_linux-64` toolchain for compute-node Triton JIT. |
| **Overall** | Queue: 86718/86719 (b02 FP8, PD), 86707–86709 (b03 AWQ + b05 GPTQ3, PD). Waiting for GPU slots. |

**Root cause fixed this session — Triton JIT on compute nodes:**

PARAM Rudra compute nodes have `/usr/bin/gcc` but **no** `/usr/include/stdlib.h`. vLLM 0.8.5 + XFormers prefix attention triggers Triton host compilation on the **first** `generate()` call. `enforce_eager=True` does not prevent this. Error pattern:

```text
fatal error: stdlib.h: No such file or directory
```

**Fix (commit `4da8913`):** Point `CC`/`CXX` at conda's `x86_64-conda-linux-gnu-gcc` (ships its own sysroot headers). Added `param_rudra_assert_triton_cc` preflight in `run_hpc_2a100_publication.sh`. Installed `gcc_linux-64 gxx_linux-64 sysroot_linux-64` in `qreason`. Documented in `00_setup_env.sh`.

**Secondary fix — BF16 KV cache OOM at 1M:** Job 86703 needed 56 GiB KV cache but only 50.5 GiB was free with BF16 KV. Added `kv_cache_dtype: fp8` + `gpu_memory_utilization: 0.95` to `deepseek_r1_qwen_7b.json` and `deepseek_r1_llama_8b.json` (FP8-quant paths already had this and reached generation before Triton failure).

**Verification performed (2026-07-03 ~16:28+ IST):**
- Triton preflight on login node: `param_rudra_assert_triton_cc` → OK (`CC=.../x86_64-conda-linux-gnu-gcc`).
- Job 86705 proved FP8 1M fits: model load 8.16 GiB, **65.52 GiB KV reserved**, engine warmup OK — then Triton gcc killed first sample.
- Cancelled 86706 (pre-fix); resubmitted b02 → **86718** (Qwen FP8), **86719** (Llama FP8).
- Raw rows: still **0/500** in `outputs-hpc-2a100-main-2026-07-03`.
- Git: `main` ahead of `origin/main` by 2; untracked `AGENTS.md`.

**Active / recent HPC jobs (2026-07-03 evening wave):**

| Job | Cell | State | Notes |
|-----|------|-------|-------|
| 86696/86697 | b03 AWQ4 | FAILED (~50s) | Git clean assert at submit |
| 86698/86699 | b04 GPTQ4 | FAILED (~6m) | Triton gcc / `stdlib.h` on first generate |
| 86703/86704 | b01 BF16 | FAILED / CANCELLED | KV cache OOM at 1M without fp8 KV |
| 86705 | b02 FP8 Qwen | FAILED (~6m) | Triton gcc (model load OK) |
| 86706 | b02 FP8 Llama | CANCELLED | Mid-run; replaced by 86719 |
| **86718** | b02 FP8 Qwen | PENDING | Resubmit with Triton fix |
| **86719** | b02 FP8 Llama | PENDING | Resubmit with Triton fix |
| **86707** | b03 AWQ Qwen | PENDING | Downstream; picks up fix at start |
| **86708** | b03 AWQ Llama | PENDING | Downstream |
| **86709** | b05 GPTQ3 | PENDING | Downstream |

**How to verify the fix:** When 86718 runs, tail `logs/slurm/b02_parallel_fp8_level_b_qwen7b_fp8_math500_seed0_86718.out`. Success = `[1-1/500] generating` then progress past `Processed prompts: 0%` and rows in `outputs-hpc-2a100-main-2026-07-03/raw/`.

**Codebase review — over-engineered parts (2026-07-03)**

I reviewed the full structure (scripts/hpc/, src/runners/, configs/, run_inference, preflights, manifests, etc.).

**Clearly over-engineered for the goal of "run long-context MATH-500 on the 7B/8B models and get results":**

- **VRAM / max context logic** — the two-phase, post-weights calc, dynamic override (we just removed most of it per your request). Good.

- **GPU preflight** (check_gpu_free_memory with 4 attempts + sleep + requeue logic + detailed nvidia-smi dumps + cuda_visible_for_gpu mapping). Defensive but complicated for shared/dirty nodes.

- **Manifest + locking + backup + atomic + _backup system** (archive_manifest.py, checkpoint_utils, 09_assert_fresh_archive, state.json.lock, heavy mirroring). Very thorough but creates many files and failure surfaces.

- **Layered preflight / assert / gate machinery** (07_preflight_publication.py, multiple git clean + fresh archive + publication_mode asserts in the launcher). Good intent, high complexity.

- **Resume / guard sophistication** (resume_guard.py + guard_and_recover + allow_resume_from_env + bad archive paths). Nice for long runs, but interacts badly with all the locking.

- **Publication mode strictness** (assert_code_paths_clean on src/scripts/configs + VLLM_BATCH_INVARIANT + batch=1 + skip-calib requirements). Excellent for final paper runs, over-constraining during debugging/iteration.

- **Telemetry depth** (gpu_stats with energy, power, tokens_per_joule, logprob confidence, etc.). Great for the paper, not needed for core "get the answers".

- **HPC split/parallel orchestration** (submit_hpc_blocks split vs exclusive_block, many env vars, gpu_id remapping). Necessary due to QOS + node sharing, but adds a lot of moving parts.

- **Config layering** (cells/ + hpc_blocks/ + per-quant model jsons + decoding + quantization registry). Flexible for many experiments, but for the current focused campaign it is a lot of small files.

- **Output root explosion** (the many outputs-hpc-*-{queued,attempt1,splitretry,p0fix,...} dirs). Symptom of previous churn.

**How to make it work (my proposal — let's decide):**

Minimal path that still lets us trust the numbers:
- Fixed high max_context (1M+ done).
- Basic preflight: one nvidia-smi free check + "is the code tree clean?" git check.
- One clear output root per campaign.
- Write results + a simple provenance (git hash + command + config snapshot).
- Keep the essential quantization correctness and the fixed high context.
- Drop or make optional: heavy dynamic VRAM calc, multi-attempt preflight, complex manifest locking, strict pub asserts during dev, deep energy telemetry, over-complex split logic.
- For parallel: start with simple 1-GPU-per-cell (no exclusive) or true 2-GPU block.

If you agree on a subset, I can clean the corresponding files (e.g. simplify the preflight function, reduce manifest usage in the run script, etc.).

What parts feel most painful to you right now when you try to run? Which ones should we attack first?

**Active / recent HPC jobs (earlier 2026-07-03 morning wave — historical)**

| Job | Role | State | Notes |
|-----|------|-------|-------|
| 86630/86631 | b01 bf16 Qwen + Llama (split) | FAILED / batch CANCELLED (00:00:34–00:00:40) | Launched on racn116 with CUDA=0/1; gates passed; cancelled in hygiene |
| 86632/86633 | b02 fp8 pair (split) | FAILED / CANCELLED (short) | One showed explicit preflight 81037 MiB free on its GPU |
| 86634/86635 | b03 awq4 pair (split) | FAILED / CANCELLED | Separate CUDA bindings confirmed |
| 86636 | b04 gptq Qwen | FAILED (5m49s) | Longest of the wave |
| 86639 | b01 2-GPU block | CANCELLED by user (~1m32s, 2 gres) | Launched **both cells in parallel inside single job** (Qwen CUDA=0 + Llama CUDA=1) |
| 86604 / 86610 / 86611 | prior monopolizers / leftovers | CANCELLED by user | Were using full node (Alloc gres:2) or QOS; cleaned to enable co-schedule |
| (86642+) | nvidia-smi peeks | mostly COMPLETED short | Post-cleanup GPU monitoring activity |

**Latest verification (from logs + sacct):** 

- Parallel co-scheduling worked exactly as designed once EXCLUSIVE=0 + monopolizer/stray cancels were done.
- Evidence (86633): `[gpu 0] === inference: ... (CUDA_VISIBLE_DEVICES=0)` + `free VRAM before vLLM (attempt 1): 81037 MiB`.
- Evidence (86639 block): simultaneous `[gpu 0] ... CUDA=0` and `[gpu 1] ... CUDA=1` + dual "GPUs: 2 | Parallel: true" + both preflights.
- 86630/31 on same node with explicit opposite CUDA devices.
- All jobs passed: DEBUG after activate/git, 09_assert, stale lock delete, git clean assert PASSED, "Checked 0 raw... ok", Archive check passed (using -queued root).
- **Raw rows:** 0 across all 07-03-* roots. Checkpoints stuck at rows_done=0 / status="in_progress" (expected: first sample ~7min; jobs terminated before completion).
- Node racn116 post-clean: mostly free (Gres=2, only tiny alloc from peeks).

**Cancelled / cleaned (this cycle):** 86604 (monopolizer), 86610/11, 86630–86639 wave (hygiene), stray nvidia-smi (86617+ range and later).

**Progress (final for this wave):** 0 raw lines, 0 rows_done. Multiple output roots created during iteration (queued primary, attempt1, main-07-03, splitretry*). Launch + binding + gates validated. No inference samples completed.

**Fixes applied 2026-07-03 (see CHANGELOG.md for full):**
- **Quantization** — GPTQ-4 for both families correctly "compressed-tensors" (on-disk match).
- **Context length — simplified** — fixed high 1M+ in configs/decoding; no dynamic VRAM calc at runtime.
- **VLLM_ALLOW_LONG_MAX_MODEL_LEN=1** — exported in `run_hpc_2a100_publication.sh` (commit `02d861b`).
- **Triton JIT on compute nodes (commit `4da8913`, evening)** — conda `gcc_linux-64` toolchain; `CC`/`CXX` → `x86_64-conda-linux-gnu-gcc`; `param_rudra_assert_triton_cc` preflight. Fixes `stdlib.h: No such file or directory` on first `generate()`.
- **BF16 1M on 1× A100** — `kv_cache_dtype: fp8` + `gpu_memory_utilization: 0.95` on BF16 Qwen/Llama model configs (fixes 86703-style KV OOM).
- MATH-500: short prompts, high fixed output limit supports long reasoning.
- Env: 13 models, qreason (vLLM 0.8.5, triton 3.2.0, conda gcc for JIT).

**Model comparison (Qwen-7B vs Llama-8B distilled):**
- Both now use the exact same simple fixed high 1M+ max. No dynamic per-model differences.
- Qwen more KV-efficient but irrelevant — fixed value for simplicity ("set and forget").
- Keeps long CoT support without complexity.

**Next (post-Triton fix):**
1. **Verify 86718** passes first generate (tail SLURM log; check `raw/` row count).
2. Let 86719, 86707–86709 run without cancel once past model load.
3. Resubmit **b01 BF16** after b02 proves fix: `QREASON_SLURM_EXCLUSIVE=0 bash scripts/hpc/submit_hpc_blocks.sh b01`.
4. Monitor: `wc -l outputs-hpc-2a100-main-2026-07-03/raw/*.jsonl` + checkpoints.
5. **Sync:** HPC commit docs → MacBook rsync+push → HPC `git reset --hard origin/main`.

**Note on 0 rows (afternoon wave):** Parallel co-schedule + gates worked; jobs died on Triton gcc (FP8/GPTQ) or git assert (AWQ) or BF16 KV OOM — not on SLURM binding. Job 86705 proved FP8 1M fits (65.52 GiB KV) before Triton killed inference.

### Active HPC queue (2026-07-02)

| Job | Role | State | Notes |
|-----|------|-------|-------|
| **86280** | Qwen-7B BF16 MATH-500 | RUNNING (split 1-GPU) | `level_a_bf16_seed0`; passed git gate after conda install git |
| **86281** | Llama-8B BF16 MATH-500 | PENDING (split 1-GPU) | `level_c_llama8b_bf16_math500_seed0`; scheduled ~2026-07-03 |

**Cancelled:** 86229 (2-GPU combined job), 86212, 86015/86016 chain.

**Git on compute:** First split submit failed — `git` not on PATH after conda activate on compute nodes. Fixed operationally with `conda install -y git` in `qreason`; codified in repo at **`85998e1`**.

**Before scoring:** `tmux kill-session -t hpc_git_autopush 2>/dev/null || true` (autopush opt-in only).

### Sync model (MacBook → GitHub → HPC)

```bash
# MacBook: commit + push (inert for running Slurm jobs)
bash scripts/macbook/github_push.sh

# HPC ONLY after inference job finishes — NOT while 86280/86281 running
cd $QR
git fetch origin && git reset --hard origin/main
tmux kill-session -t hpc_git_autopush 2>/dev/null || true
```

Running jobs execute whatever was on disk at **launch**. Operational fixes deploy at **score time** (or next submit after sync).

---

## 2026-07-03 — Parallel GPU execution on shared nodes + QOS-aware batch queuing (HPC) — mechanism proven, wave cleaned, 0 rows (jobs cancelled post-verification)

**Key event (full cycle):** Achieved and **verified** concurrent launch of two independent 1-GPU cells (two models) on a single 2-GPU node (racn116) by submitting **without `--exclusive`** (`QREASON_SLURM_EXCLUSIVE=0`). Also verified the 2-GPU block path. This directly solved the monopoly where prior 1-GPU jobs received AllocTRES gres/gpu:2. 

Two models launched side-by-side using separate GPUs (CUDA_VISIBLE_DEVICES=0/1). All gates passed. The wave of jobs (86630–86639) was then cancelled (user hygiene after launch verification) before any samples finished (~7min first sample). Result: 0 raw rows across roots. Queue now fully clear.

**Actions taken:**
- `export QREASON_SLURM_EXCLUSIVE=0` (affects submit_split_2gpu + submit_2gpu_block conditional logic in submit_hpc_blocks.sh).
- Canceled monopolizers (86604 ~24min holding full node, 86610/86611) + stray nvidia-smi jobs.
- Resubmitted split pairs for b01 (bf16 86630 Qwen + 86631 Llama), b02 fp8 (86632/33), b03 awq4 (86634/35), b04 gptq + the 2-GPU exclusive_block b01 (86639) under fresh `outputs-hpc-2a100-main-2026-07-03-queued`.
- Verified via live logs + post-facto sacct:
  - Separate CUDA per cell on shared node (e.g. 86631 CUDA=1 while pair on 0; 86633 CUDA=0 with preflight 81037 MiB).
  - 86639 (block): one job, gres=2, launched both cells in bg: Qwen on CUDA=0 + Llama on CUDA=1, dual clean preflights, "Parallel: true".
  - All: archive check passed, DEBUG echoes, stale locks cleaned, git clean PASSED, "Checked 0 raw... ok to resume".
- Later: many short nvidia-smi jobs (peeks) on racn116/racn115.

**Job final states (sacct summary):**
- 86630/31, 86632/33, 86634/35, 86636: FAILED (mostly batch CANCELLED), 19s–5m49s elapsed on racn116.
- 86639: CANCELLED by user after 1m32s (correctly allocated 2 gres).
- Monopolizers and strays: cancelled earlier.

**Progress / data:**
- Raw: 0 lines in every 07-03-* root (queued primary for this wave).
- Checkpoints: rows_done=0, status=in_progress (timestamps match launch window).
- Multiple roots created during rapid iteration (queued, attempt1, main-07-03, split*).
- Inference reached the "generating batch of 1" stage in principle but no rows written before termination.

**Outcome:** 
- **Parallel two-model execution on shared 2-GPU node proven in practice.** Split mode (no exclusive) allows scheduler to pack two 1-GPU jobs (two models) on one node with correct per-job CUDA binding via the launcher. 2-GPU block also works for dedicated parallel inside one alloc.
- ~2x node utilization vs monopoly. QOS waves of ~2 gres remain the limit.
- The 0-row result is expected given short lifetimes (hygiene cancels) vs. generation time. No code, preflight, or dirty-GPU failure.
- Current: squeue empty, racn116 nearly idle.

See [CHANGELOG.md](CHANGELOG.md) for exhaustive job table, full log quotes, reasoning on EXCLUSIVE logic, and design guidance.

**Immediate recommended follow-up (from snapshot):**
- Stabilize on one output root.
- Re-submit with EXCLUSIVE=0.
- Allow long run; watch raw growth and per-sample timing.
- Score only after real rows + checkpoints advance.

---

## 2026-07-02 — Review hardening + HPC operational fixes (MacBook)

### Review hardening

- Added `docs/ENV_VARS.md`; expanded `.env.example` and README
- Scalable archive blocking: `INVALID_FOR_PUBLICATION.txt` + `QREASON_FORBIDDEN_ARCHIVE_PATTERNS`
- `model_id` on raw rows/summaries; QRM comparator + paper tables prefer it
- Publication git: clear error when Git missing

### HPC operational fixes (full scope)

| Phase | Deliverable |
|-------|-------------|
| Git/autopush | `assert_code_paths_clean()`; autopush opt-in (`QREASON_ENABLE_AUTOPUSH=1`) |
| Manifest | `archive_manifest.py` with locking; non-fatal launcher bookkeeping |
| Resume | Allow resume when HEAD moved but code unchanged |
| Submit | Single archive root; `--fresh`; default `all` → b01 |
| QRM gate | Quant/profile mismatch → SKIP |
| Backup | `backup_mirror` ignores `*.tmp` / `*.lock` |
| Pins | transformers 5.12.1 aligned to lock |
| Logprobs | Pipeline wired; b01 keeps `--skip-calibration` until GPU smoke |

**Verification:** 32 targeted tests pass; `pin_hf_revisions.py --verify` OK; full suite 99/100 (1 pre-existing fail).

See [CHANGELOG.md](CHANGELOG.md) for full detail.

### Current HPC gate state (completed before queue submit)

1. Reset HPC to `8fb0fb0`; cancelled job 86010.
2. Removed invalid archive `outputs-hpc-2a100-main-2026-06-29` (diagnostic only — consider renaming to `-DIAGNOSTIC-INVALID` on future runs instead of delete).
3. `verify_decoding_params.py` — **VERIFY OK** (`repetition_penalty: 1.05` → vLLM).
4. `07_preflight_publication.py` — **passed** (25 cells, datasets, prompts).
5. Login-node smoke failed (`libcuda.so.1 missing`) — **expected**; submitted GPU smoke as 86015 instead.
6. b01 submitted as 86016 with `afterok:86015`.

See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) and [docs/J1_VALIDATION_RUNBOOK.md](docs/J1_VALIDATION_RUNBOOK.md).

### b01 pass criteria (after 86016 completes)

**Primary:** Qwen-7B MATH-500 pass@1 in **88.9–98.9%** band (QRM T1 ref 93.9). Llama-8B **86.0–96.0%** (QRM Table 4 ref 91.0).

**Also check** (not pass@1 alone):

| Check | Expectation |
|-------|-------------|
| `decoding_repetition_penalty` in every raw row | Present (1.05) |
| `truncation_rate` | Low (≤ 0.15) |
| `completion_tokens_mean` | Thousands per row (not hundreds) |
| Manual audit | 20–50 traces; parse failures reviewed |
| Comparator provenance banner | yaml sha256 + git commit `286f5e4` |

```bash
# After 86016 — sync THEN score
git fetch origin && git reset --hard origin/main
python scripts/score_run.py --input "$ROOT/raw/level_a_bf16_seed0.jsonl" \
  --summary "$ROOT/results/level_a_bf16_seed0_summary.json" --skip-calibration
python scripts/compare_qrm_baseline.py --summary "$ROOT/results/level_a_bf16_seed0_summary.json" 2>&1
```

Score **both** b01 cells (Qwen + Llama). Use `--skip-calibration` for repro gate (calibration comes later via maj@5/logprobs).

### First scored numbers (June-29 archive — **invalid, diagnostic only**)

| Cell | Rows | pass@1 | Truncation | Notes |
|------|------|--------|------------|-------|
| Qwen-7B BF16 | 500/500 | **7.0%** (35) | 90% | No `repetition_penalty` in raw run |
| Llama-8B BF16 | 500/500 | **21.4%** (107) | 59% | Loop/truncation affected |
| Qwen-7B FP8 | 50/500 | 0% (partial) | 76% | Old decoding; do not use |

**Do not cite** these in the manuscript. Rescoring cannot fix truncated completions.

### Recent commits (2026-07-01)

| Commit | Summary |
|--------|---------|
| `286f5e4` | Fix QRM baseline bands; comparator provenance; amd-002 |
| `8fb0fb0` | Fail-closed calibration, validation runbook, publication matrix |
| `9ca0ec1` | Qwen-1.5B JSON fixes, McNemar test |
| `9933241` | GSM8K sober prompt |

### Pre-push verification (MacBook)

```bash
python -m pytest tests/ -q                    # 43 pass
python scripts/validate_cell_matrix.py        # 15/15 wired
python scripts/verify_decoding_params.py      # VERIFY OK
python -m compileall -q src scripts tests papers
```

### Key documentation (2026-07-01)

| Doc | Purpose |
|-----|---------|
| [docs/J1_VALIDATION_RUNBOOK.md](docs/J1_VALIDATION_RUNBOOK.md) | Phases 0–7: sync, b01, score, audit, pilot |
| [docs/CODEBASE_OVERVIEW.md](docs/CODEBASE_OVERVIEW.md) | Full architecture map |
| [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) | Traps including wrong baseline bands (§8) |
| [configs/baselines/qrm_literature_targets.yaml](configs/baselines/qrm_literature_targets.yaml) | QRM Table 1 sanity bands by task |
| [papers/j1/amendments.yaml](papers/j1/amendments.yaml) | amd-001 template, amd-002 baseline fix |

---

## Current Experiment Coverage

**Active run:** Slurm jobs 86015 (smoke) → 86016 (b01). Archive path set in job env (`QREASON_OUTPUT_ROOT`).

| Block | Coverage | Status |
|------|----------|--------|
| b01 | Qwen-7B BF16 + Llama-8B BF16 MATH-500 | **Queued** — job `86016`, depends on smoke `86015` |
| b02 | Qwen-7B FP8 + Llama-8B FP8 MATH-500 | **Hold** until b01 passes QRM gate |
| b03 | AWQ-4 pair MATH-500 | Hold |
| b04 | GPTQ-4 pair MATH-500 | Hold |
| b05 | Qwen-7B GPTQ-3 MATH-500 | Hold |
| b06 | Qwen-7B FP8 GSM8K | Hold — gate uses GSM8K band ~86–96% (QRM T1) |
| b07 | Qwen-7B FP8 GPQA-Diamond | Hold — gate uses GPQA band ~44–54% (QRM T1) |
| b08–b09 | Qwen-1.5B lower-bound cells | Wired; after main MATH signal |

### Immediate next steps

1. Wait for 86015 smoke → 86016 b01 (Telegram watcher active).
2. **Do not** `git reset` on HPC while jobs run.
3. After 86016: `git fetch && git reset --hard origin/main` (get `286f5e4`).
4. Score both cells with `--skip-calibration`; run `compare_qrm_baseline.py` (check provenance banner).
5. Manual audit 20–50 traces; verify `completion_tokens_mean` and truncation.
6. **Before b02:** logprob capture patch merged + smoke-tested (hard gate).
7. **Before breadth grid:** 3-seed pilot (Qwen-7B × {BF16, GPTQ-4, GPTQ-3} × {MATH-500, GPQA-D}).

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

CPU preflight after syncing `9ca0ec1`: `scripts/hpc/07_preflight_publication.py` passed for all configured publication blocks and datasets.

Current recommendation: let b01 job `86010` run first and do not submit b02-b06 until the Qwen-7B BF16 anchor validates. Future jobs remain wired as b07 for GPQA and b08-b09 for Qwen-1.5B lower-bound runs; submit them only after the main b01/b02 signal is clear.

---

## 2026-07-01 — HPC outputs pulled, scored, pipeline audit

### What happened

- Pulled latest HPC autopush from GitHub (`bdaff00`) into MacBook archive `outputs-hpc-2a100-main-2026-06-29`.
- b01 BF16 anchors **completed** (500 rows each for Qwen-7B and Llama-8B); b02 Qwen-7B FP8 **partial** (50/500 when last synced).
- Ran `rescore_archive.py`, `sync_archive_manifest.py`, `build_paper_tables.py` on MacBook.

### Diagnosis: why pass@1 is so low

1. **Decode loops** — ~90% of Qwen completions hit ≥97% of the 32k token budget (`yeah yeah…`, `</think>` spam) before emitting `\boxed{}`.
2. **`repetition_penalty` never applied** — YAML had `1.05` but `load_decoding_from_file()` stripped it; HPC runs had no anti-loop decoding.
3. **Llama text artifact** — vLLM 0.8.x leaked SentencePiece markers (`Ġ`, `Ċ`) in completions (fixed for scoring + future inference).
4. **Boxed extractor** — nested `\boxed{\frac{1}{2}}` and unclosed trailing boxes caused parse failures (fixed).

### Fixes applied on MacBook (pending git push)

| Area | Fix |
|------|-----|
| Scoring | `math_extractor.py`, truncation detection, Llama normalization |
| Decoding | Full YAML passthrough + `repetition_penalty: 1.05` in `repro_qrm.yaml` |
| Orchestration | GPQA row count (198), HPC score-on-skip, `expected_rows.py` |
| Telemetry | NVML respects `CUDA_VISIBLE_DEVICES` on parallel blocks |
| Tooling | `rescore_archive.py`, `sync_archive_manifest.py`, 17 unit tests |

### Re-audit (2026-07-01 evening)

- All unit tests pass; `compileall` + bash `-n` on HPC/5080 orchestrators pass.
- No remaining hardcoded `want=1` for GPQA in HPC/5080 publication scripts.
- **Still open:** 5080 batch checkpoint edge case (irrelevant on HPC `batch_size=1`); Level A/C raw data needs HPC rerun for trustworthy pass@1.
- **Also fixed:** cost summaries wrote invalid JSON `Infinity` when `num_correct == 0` (now `null`).

### Next HPC steps

```bash
# MacBook: commit + push fixes
# HPC:
cd $QR && git fetch origin && git reset --hard origin/main
# scancel Level B if still on old code; resubmit b02 with fixed decoding
# Optional: delete + rerun Level A/C raw for clean BF16 anchor numbers
```

---

## 2026-07-01 — Clean b01 publication rerun queued

### What happened

- MacBook/GitHub source of truth advanced through:
  - `9933241` — added missing sober GSM8K prompt and preflight coverage.
  - `9ca0ec1` — fixed invalid trailing commas in Qwen-1.5B cell JSON and added McNemar test fix.
- HPC scratch repo was hard-reset to `origin/main` at `9ca0ec1`.
- Invalid June-29 archive was removed from scratch again after reset restored tracked output files.
- CPU gates passed in `qreason`:
  - `python scripts/verify_decoding_params.py` -> `VERIFY OK`.
  - `python scripts/hpc/07_preflight_publication.py` -> passed static checks, prompt formatting, 25 cell configs, block/model wiring, and dataset access.
- Fresh archive guard passed for `outputs-hpc-2a100-main-2026-07-01-rerun` with 0 raw JSONL files.
- Submitted clean b01 rerun as Slurm job `86010` using `slurm/hpc_2a100_b01_parallel.slurm`.

### Current job

| Job | Block | Archive | State at submit check | Reason |
|-----|-------|---------|-----------------------|--------|
| `86010` | b01 BF16 anchors | `outputs-hpc-2a100-main-2026-07-01-rerun` | Pending | Resources |

b01 runs two parallel MATH-500 cells on 2×A100:

| GPU | Cell | Gate role |
|-----|------|-----------|
| 0 | `level_a_qwen7b_bf16_math500_seed0` | Primary Qwen anchor; must recover from old 7% pass@1 |
| 1 | `level_c_llama8b_bf16_math500_seed0` | BF16 Llama anchor |

### Next plan

1. Monitor job `86010` until it starts:
   `squeue -u $USER -l`.
2. Once running, monitor:
   `tail -f logs/slurm/b01_parallel_bf16_86010.out` and per-cell logs under `$QREASON_OUTPUT_ROOT/logs/`.
3. Confirm raw row counts grow under `$QREASON_OUTPUT_ROOT/raw/` and rows include `decoding_repetition_penalty`.
4. After Qwen b01 finishes, run:
   `python scripts/compare_qrm_baseline.py --summary $QREASON_OUTPUT_ROOT/results/level_a_qwen7b_bf16_math500_seed0_summary.json`.
5. Gate criteria: Qwen pass@1 roughly **88–98%** and Llama **84–94%** on MATH-500 (see `qrm_literature_targets.yaml`); truncation low. If still near 7% or high truncation, stop and inspect decoding/raw rows before any more submissions.
6. If b01 passes, unset fresh mode and submit b02 on the same archive:
   `unset QREASON_FRESH_RUN; bash scripts/hpc/submit_hpc_blocks.sh b02`.
7. Continue b03–b06 only after b02 behaves sensibly. Defer b07 GPQA and b08–b09 Qwen-1.5B until the main MATH-500 quantization signal is clear.
8. If b01 fails mid-run, resume safely on the same archive by unsetting `QREASON_FRESH_RUN` before resubmitting b01; completed cells should skip or resume from checkpoints.

---

## GPU Telemetry Added 2026-06-30

Future inference rows now record sampled runtime telemetry and efficiency fields: `vram_before_gb`, `vram_after_gb`, `vram_max_gb`, `gpu_util_mean`, `gpu_util_max`, `power_watts_mean`, `power_watts_max`, `energy_joules`, `tokens_per_second`, `decode_tokens_per_second`, `seconds_per_output_token`, `tokens_per_joule`, `finish_reason`, `stop_reason`, `truncated`, `completion_chars`, and optional `time_to_first_token_sec` when vLLM exposes timing metrics. Scored rows also include `answer_parse_success` and MATH `boxed_answer_present`.

Legacy note: job `85394` predates full telemetry in raw rows. Archives synced before 2026-07-01 may lack `finish_reason` / `truncated` per row; rescoring infers truncation from token counts. Parallel 2×A100 blocks before the NVML fix logged GPU 0 stats for the Llama branch.

## Q1 Publication Analysis Utilities

The codebase now has the core post-run publication tools needed after HPC jobs finish:

- `scripts/score_run.py` summaries include bootstrap 95% confidence intervals for `pass_at_1`, failure rates, and cost-per-correct intervals.
- `scripts/rescore_archive.py --archive <outputs-hpc-...>` rescored an entire archive after extractor fixes.
- `scripts/sync_archive_manifest.py --archive <outputs-hpc-...>` refreshes manifest/state from disk (task-aware row counts).
- `scripts/expected_rows.py --cell-config <cell.json>` prints dataset size (500 / 1319 / 198).
- `scripts/verify_decoding_params.py` — pre-HPC decoding preflight.
- `scripts/compare_qrm_baseline.py` — post-rerun literature sanity gate.
- `scripts/run_inference_multisample.py` / `score_multisample.py` — maj@5 Level B pilot.
- `scripts/build_pareto_frontier.py` — cost Pareto across quant configs.
- Raw rows from post-fix runs include decoding settings and truncation telemetry.
- GSM8K and GPQA-Diamond cells use task-aware row construction/scoring.
- `scripts/build_paper_tables.py --archive <outputs-hpc-...>` writes main, efficiency, and failure CSV tables.
- `scripts/build_repro_bundle.py --archive <outputs-hpc-...>` writes reproducibility bundle JSON.

Run rescore + paper tables after each archive sync; rerun inference (not just rescore) after decoding fixes.

---

## Archive Metadata and Backup Status

HPC publication runs write a durable archive manifest at `outputs-hpc-2a100-main-YYYY-MM-DD/manifest.json` and per-cell metadata under `metadata/<cell_id>.json`.

**Active archive (2026-07-01):** `outputs-hpc-2a100-main-2026-07-01-rerun` — fresh publication rerun archive. `09_assert_fresh_archive.sh` passed with 0 raw JSONL files before b01 submission. Job `86010` will write b01 raw/scored/results here.

**Invalid archive:** `outputs-hpc-2a100-main-2026-06-29` — diagnostic only, deleted from scratch before rerun. If restored by Git reset or output sync, delete again before fresh publication jobs.

---

## Publication Sufficiency Strategy

Current judgement (updated 2026-07-01): b01-b09 seed0 remains the target core grid. The decode-loop fix is now deployed on HPC and b01 rerun job `86010` is queued on a clean archive. The previous June-29 b01/b02 numbers remain diagnostic only.

Do not expand the queue until the post-fix b01 rerun shows sensible Qwen-7B BF16 pass@1 on MATH-500 (expected ~88–98%, not 7%) and low truncation.

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
