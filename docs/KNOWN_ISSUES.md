# Known issues and limitations

Last updated: 2026-07-05

Operational issues that can break paper results if ignored, plus known software limitations.

---

## Current campaign status (2026-07-05)

| Item | Status |
|------|--------|
| **Active** | **Path C diagnostic** — jobs **87116–87118**, archive `outputs-hpc-diag-pathc-2026-07-05` |
| b01 QRM gate | **FAILED** — July archive `outputs-hpc-2a100-main-2026-07-03` |
| Qwen b01 90 rows | **Skipped** (not required) |
| Quant grid | **On hold** until `report_pathc_diagnostic.sh` |

See [notes.md §29](../notes.md) · [CHANGELOG.md](../CHANGELOG.md) Path C entry.

---

## Critical — must fix before trusting numbers

### 1. Resume from a bad archive — **fixed in code (2026-07-01)**

**Symptom:** pass@1 stays ~7% after “rerun”  
**Cause:** `run_inference.py` used to resume from existing raw JSONL automatically.

**Automatic protection now:**
- `run_inference.py` **refuses** to resume into rows missing `decoding_repetition_penalty` or with wrong git/config hash
- `scripts/hpc/09_assert_fresh_archive.sh` **blocks** the forbidden `outputs-hpc-2a100-main-2026-06-29` path
- Use `--fresh` or `QREASON_FRESH_RUN=1` for intentional restarts

**Still required on HPC:**

```bash
rm -rf outputs-hpc-2a100-main-2026-06-29
export QREASON_OUTPUT_ROOT=$QR/outputs-hpc-2a100-main-$(date +%Y-%m-%d)-rerun
export QREASON_FRESH_RUN=1   # optional: wipe per-cell outputs on first inference
bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors
```

Override only if you mean it: `QREASON_ALLOW_RESUME=1`

### 2. Archive `outputs-hpc-2a100-main-2026-06-29` is diagnostic only

Generated **without** `repetition_penalty` reaching vLLM (YAML passthrough bug, fixed 2026-07-01).  
**Do not cite** 7% / 21% pass@1 in the manuscript. Rescoring cannot fix truncated raw text.

### 3. HPC must hard-reset git, not merge

HPC autopush may leave local output commits. Always:

```bash
git fetch origin && git reset --hard origin/main
```

### 3b. Git missing on compute after `conda activate qreason` — **fixed 2026-07-02**

**Symptom:** Split b01 Slurm jobs exit before inference with `Publication run requires Git installed and a git checkout.`  
**Cause:** `assert_code_paths_clean()` runs `git diff` at job start; compute nodes may not have `git` on PATH after `conda activate qreason` (login-node preflight did not simulate job env).

**Fix (repo + HPC):**
- `scripts/hpc/00_setup_env.sh` now runs `conda install -y git` in `qreason`
- `param_rudra_activate_conda()` prepends `/usr/bin:/bin` and fails fast if git is still missing
- `07_preflight_publication.py` (full mode) verifies git after conda activate

**One-time on HPC if jobs already running:** `conda activate qreason && conda install -y git` (already done for split jobs 86280/86281).

---

### 3c. `QOSMaxGRESPerUser` with parallel 1-GPU cells — **documented 2026-07-03**

**Symptom:** Second b01 cell `PENDING (QOSMaxGRESPerUser)` while only one inference job runs.  
**Cause:** `--exclusive` on ragpu (2-GPU node) counts as **2 GPUs** toward the 2-GPU user limit; 1 running + 1 pending exclusive = 3 counted.  
**Fix:** `submit_hpc_blocks.sh` split/single-cell submits **never** pass `--exclusive`. Use `docs/PARAM_RUDRA_SLURM.md` + dirty-node exclude.

---

## Important — affects interpretation

### 4. Single-sample calibration requires valid confidence

`score_run.py` **does not** use `answer_parse_success` as a publication confidence score.

- Default: pass@1/cost always computed; calibration **skipped** with `skipped: true` if no valid confidence.
- `--skip-calibration`: explicit skip (use for b01 reproduction scoring).
- `--require-calibration`: exit 1 if valid confidence unavailable (use before calibration analysis).
- `--allow-parse-confidence-proxy`: debug only — marks `confidence_valid_for_calibration: false`.

For manuscript Brier/AURC/ECE claims, use **maj@5** (`run_inference_multisample.py` + `compute_calibration.py`) or logprob-based confidence with a valid `confidence_source`.

**2026-07-02 update:** Raw rows now capture `confidence` / `confidence_source=normalized_sequence_logprob` when vLLM returns token logprobs. HPC launcher still passes `--skip-calibration` for b01 until a 3-question GPU smoke confirms logprobs on A100.

### 4b. Publication git gate vs output bookkeeping — **fixed 2026-07-02**

Publication mode now checks **code paths only** (`src`, `scripts`, `configs`, …). Tracked `outputs-hpc-*/manifest.json` updates no longer block `score_run.py --publication`.

Autopush tmux is **opt-in** (`QREASON_ENABLE_AUTOPUSH=1`). Default workflow: MacBook rsync after runs, not autopush during SLURM jobs.

### 5. Mixed provenance on resumed inference

Rows written before V8.2 provenance fields lack `run_id`, `git_commit`, etc. New rows in the same JSONL have them.  
Analysis should filter by `schema_version` or rerun fresh archives for publication.

**2026-07-02:** `config_hash` is now content-based (no absolute `model_path`). Resume into pre-fix archives will fail with hash mismatch — use `--fresh` or a new `QREASON_OUTPUT_ROOT`.

### 6. QRM reproduction vs main grid prompts

| Profile | Used for | Prompt style |
|---------|----------|--------------|
| `reproduction` | Level A repro gate | Short QRM `\boxed{}` |
| `sober` | Level B/C main grid | Long sober-reasoning template |

Comparing Level A to Level B pass@1 directly confounds prompt + quant — compare within profile.

### 7. GPQA answer shuffle

Deterministic per `(seed, row_index)` in this harness vs QRM’s fixed pipeline RNG. Document when comparing to QRM Table 1.

### 8. Wrong QRM baseline bands (fixed 2026-07-01, commit `286f5e4`)

**Symptom:** `compare_qrm_baseline.py` passes at ~60% pass@1 or fails at ~93% on MATH-500.  
**Cause:** Pre-fix `qrm_literature_targets.yaml` used **45–65% bands for MATH-500** — those are **AIME / GPQA-Diamond** scale, not MATH-500 (~88–98% for R1-distills).

| Task | Correct scale (BF16) | Wrong scale if mis-copied |
|------|----------------------|---------------------------|
| MATH-500 | ~85–95% | ~40–65% (AIME) |
| GSM8K | ~85–92% | — |
| GPQA-Diamond | ~44–54% | — |

**Fix:** Full yaml audit in `286f5e4`; protocol note amd-002 in `papers/j1/amendments.yaml`.

**Deploy rule:**
- Running Slurm jobs are **not** affected by MacBook push.
- HPC must `git fetch && git reset --hard origin/main` at **score time** (after inference completes).
- Always check comparator **stderr provenance banner** (yaml sha256 + git commit) before trusting PASS/FAIL.

Archives scored with pre-fix yaml are **invalid for gate comparison**. After amd-003, re-check provenance banner for Table 1 vs Table 4 sources and `gate_type`.

---

## Minor / environment

### 8. `lighteval` clone may fail without git-lfs

```bash
brew install git-lfs   # or GIT_LFS_SKIP_SMUDGE=1
bash ../external_repos/clone_v82_repos.sh
```

### 9. 5080 batch checkpoint (historical)

5080 scripts could lose up to `batch_size−1` rows on crash. HPC uses `batch_size=1` — not an issue for publication runs.

### 10. J2/J3 backends are pilot stubs

SGLang and llama.cpp modules produce manifests only until Paper 2/3 pilot gates run.

---

## Pre-flight commands (catch issues early)

```bash
python -m pytest tests/ -q
python scripts/verify_decoding_params.py
python scripts/hpc/07_preflight_publication.py   # HPC CPU gate
```

---

## Where fixes are logged

| Log | Purpose |
|-----|---------|
| [CHANGELOG.md](../CHANGELOG.md) | Dated fixes and HPC ops |
| [progress.md](../progress.md) | Full execution timeline |
| [docs/PROGRESS.md](PROGRESS.md) | Short live status |
