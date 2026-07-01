# Changelog

## 2026-07-01 — External repos plan: tooling wired (MacBook)

Implemented actionable items from external-repos analysis (HPC-only publication path):

- **Decoding verify:** `scripts/verify_decoding_params.py`, `src/runners/sampling_utils.py` — confirms `repetition_penalty` + seed reach vLLM SamplingParams (sober-reasoning pattern).
- **QRM baseline gate:** `configs/baselines/qrm_literature_targets.yaml`, `scripts/compare_qrm_baseline.py` — sanity-check pass@1 / truncation after HPC rerun.
- **GPTQ-4 download:** `scripts/hpc/08_download_gptq4_models.sh` (QRM HF collection); `docs/GPTQ4_PREP.md` updated.
- **maj@5 pilot:** `scripts/run_inference_multisample.py`, `scripts/score_multisample.py`, `configs/decoding/pilot_maj5.yaml`.
- **Pareto frontier:** `src/metrics/pareto_frontier.py`, `scripts/build_pareto_frontier.py` (Cost-of-Pass pattern).
- **lm-eval sanity (optional):** `scripts/lmeval_sanity_check.sh`, `scripts/lmeval_compare_summary.py`, `docs/reference_notes/LMEVAL_SANITY.md`.
- **Level C clones:** `external_repos/06-later/{livecodebench,OckBench,reasoning-models-confidence}`.
- Tests: `tests/test_sampling_params.py`, `tests/test_external_repos_integration.py` (17 total pass).

**Pre-HPC rerun:** `python scripts/verify_decoding_params.py`  
**Post-rerun:** `python scripts/compare_qrm_baseline.py --summary results/<cell>_summary.json`

### Pre-push audit (2026-07-01 evening)

- **17/17 tests pass**; `verify_decoding_params.py` → VERIFY OK.
- Refactored `prepare_example_row` → `src/runners/dataset_rows.py`.
- **Docs cleanup:** `docs/README.md` index; redundant guides moved to `docs/archive/`; README shortened.
- **Push status:** MacBook changes **not pushed yet** (local behind `origin/main` by HPC autopush commits — pull before push).

---

## 2026-07-01 — First HPC scores + pipeline audit (MacBook)

### Results (`outputs-hpc-2a100-main-2026-06-29`, rescored on MacBook)

| Cell | Status | pass@1 | Truncation | Parse fail |
|------|--------|--------|------------|------------|
| Qwen-7B BF16 MATH-500 | **500/500 scored** | 7.0% (35/500) | 90% | 86% |
| Llama-8B BF16 MATH-500 | **500/500 scored** | 21.4% (107/500) | 59% | 60% |
| Qwen-7B FP8 MATH-500 | **50/500 in progress** | 0% (partial) | 76% | 98% |

Paper tables populated under `outputs-hpc-2a100-main-2026-06-29/paper_tables/`. Absolute pass@1 is depressed by R1 decode loops hitting the 32k token cap (~90% near-max on Qwen). **Existing raw JSONL was generated without `repetition_penalty`** — rerun required for clean numbers, not just rescoring.

### Scoring fixes

- MATH `\boxed{}` extraction: nested braces, skip unclosed trailing boxes (lm-eval style).
- Llama vLLM 0.8.x SentencePiece artifacts (`Ġ`, `Ċ`) normalized before scoring and at generation time.
- Truncation rate inferred from `finish_reason == "length"` or completion tokens ≥ 97% of max when legacy rows lack telemetry.
- New scripts: `scripts/rescore_archive.py`, `scripts/sync_archive_manifest.py`, `scripts/expected_rows.py`.
- Tests: `tests/test_math_extractor.py`, `tests/test_config_and_tasks.py`, `tests/test_gpu_stats.py` (17 tests total after external-repos additions).

### Critical orchestration / config fixes

- **`load_decoding_from_file()`** now forwards all YAML keys (including `repetition_penalty`) — was silently dropped before every HPC run.
- Added `repetition_penalty: 1.05` to `configs/decoding/repro_qrm.yaml` for future anti-loop decoding.
- **GPQA row count:** shared `src/runners/task_utils.py` — MATH-500 (500), GSM8K (1319), GPQA (198); HPC/5080 no longer mark GPQA complete after 1 row.
- **NVML telemetry:** `CUDA_VISIBLE_DEVICES` mapped to correct physical GPU on parallel 2×A100 blocks (was always GPU 0).
- HPC skip path runs `score_run.py` when raw is complete but scored/summary missing.
- `sync_archive_manifest.py` uses task-aware row counts (not hardcoded 500).
- Preflight adds `b02_gpqa_fp8.sh` block + GPQA-Diamond dataset validation.
- `extract_answers.py`, `compute_calibration.py` route through `score_item()` / majority vote.
- `run_inference.py` persists finish/truncation/telemetry fields from `vllm_runner`.
- `vllm_runner` marks `truncated` when `finish_reason == "length"`.
- Cost summaries use `null` instead of invalid JSON `Infinity` when `num_correct == 0`.

### HPC action required

1. Push MacBook fixes → `git push origin main`
2. HPC: `cd $QR && git fetch origin && git reset --hard origin/main`
3. Stop/restart Level B FP8 (and optionally rerun Level A/C) so cells pick up `repetition_penalty: 1.05`
4. Do **not** treat current Level A/C pass@1 as publication-ready until rerun completes

### Known remaining gaps (not fixed this session)

- **5080 batch checkpoint** can lose up to `batch_size−1` rows on crash (low risk on HPC where `batch_size=1`).
- Level B partial (50/500) was generated with old decoding — discard or resume after HPC sync.

---

## 2026-06-30 (GPU telemetry and efficiency metrics)
- Raw inference rows now include throughput and completion-health fields: total/decode tokens per second, seconds per output token, tokens per joule, finish/stop reasons, truncation flag, completion character count, VRAM before/after/max, and optional time-to-first-token when vLLM exposes timing metrics.
- Scored rows now record explicit answer parse success and MATH boxed-answer presence.
- Summary JSON now aggregates throughput, VRAM, utilization, power, energy, tokens-per-joule, finish-reason counts, and the new sampled telemetry fields while staying compatible with old raw rows.
- Queue judgement at implementation time: `85394` b01 is still running on `ragpu008`; `85343`-`85347` b02-b06 remain pending behind `QOSMaxGRESPerUser`. Do not submit broader GSM8K/GPQA grids until b01 walltime behavior is clear. Pending jobs will pick up this instrumentation when they start.

---

## 2026-06-29 (PhD roadmap — single file)

- Replaced split `docs/phd-roadmap/` with one document: **`docs/PHD_ROADMAP.md`** (V5 + V6 + V7 + stack-transfer extension, appendices, execution plan).
- Updated **`README.md`** and **`docs/PAPER1_DESIGN.md`** links.

---

## 2026-06-29 (Q1 publication analysis utilities)

- Added bootstrap confidence intervals for `pass_at_1` and cost-per-correct summaries.
- Added explicit failure-rate summaries: parse failures, empty completions, truncation, and invalid answers.
- Future raw rows now include decoding temperature/top-p/max-token metadata and `max_model_len`.
- Added `scripts/build_paper_tables.py` for main, efficiency, and failure CSV tables.
- Added `scripts/build_repro_bundle.py` for archive-level reproducibility bundles with manifest, metadata, package versions, git info, CUDA probe, and file hashes.
- Added task-aware scoring/row support for future GSM8K and GPQA-Diamond runs.

---

## 2026-06-29 (HPC archive metadata manifest)

- Added HPC archive manifest generation for publication runs.
- Each HPC cell now records a `metadata/<cell_id>.json` snapshot with cell config, model config, task config, decoding config, batch/checkpoint settings, git commit, SLURM job info, raw path, summary path, and saved row count.
- `_backup/latest/` now mirrors `metadata/` and uses a lock for full archive mirrors so parallel GPU branches do not race while backing up.

---

## 2026-06-29 (Publication sufficiency strategy)

- Recorded the current publication-readiness judgement across planning docs.
- b01-b09 seed0 is treated as the first publishable core result set if it completes cleanly and produces interpretable trends.
- Expansion rule: do not add broad new jobs before scoring b01-b09; if robustness is needed, add seed1/seed2 only for the key Qwen-7B and Llama-8B MATH-500 BF16/FP8/AWQ-4/GPTQ-4 subset.

---

## 2026-06-29 (Future HPC blocks wired)

- Added future HPC-only Qwen-1.5B blocks:
  - `b08_qwen15b_bf16_fp8` for BF16 + FP8 MATH-500.
  - `b09_qwen15b_awq4_gptq4` for AWQ-4 + GPTQ-4 MATH-500.
- Removed the old preflight restriction that treated `qwen15b` cells as 5080-only.
- Confirmed GPQA-Diamond gated access is now available through the saved HPC Hugging Face token: authenticated request for `gpqa_diamond.csv` returned HTTP 200.
- CPU preflight passed after adding b08-b09: 14 HPC cell entries checked; MATH-500 and GSM8K dataset checks passed.
- No new SLURM jobs were submitted; current recommendation remains to let b01-b06 continue before queueing b07-b09.

---

## 2026-06-29 (Corrected b01 resubmitted)

- Held queued jobs `85343`-`85347`, canceled half-broken b01 job `85342`, and submitted corrected b01 job `85394`.
- `85394` started on `ragpu008` and uses the fixed `update_state()` code from GitHub.
- Qwen-7B BF16 resumed from `20/500` durable rows in `outputs-hpc-2a100-main-2026-06-29/raw/level_a_qwen7b_bf16_math500_seed0.jsonl`.
- Llama-8B BF16 restarted from `0/500` under the corrected state-locking code; it reached dataset/model loading without the old `state.json.tmp` crash.
- Released b02-b06 after `85394` started; they are pending on `QOSMaxGRESPerUser` behind the running corrected b01 job.

---

## 2026-06-29 (HPC b01 running — state race fix committed)

### HPC status

- Pulled latest GitHub changes on PARAM Rudra scratch repo: `b280a88` -> `e149159`.
- Smoke job `85306` completed successfully with exit code `0:0`; Gate 3 is now passed.
- Submitted publication blocks b01-b06:
  - `85342` / b01 BF16 anchors is running on `ragpu008`.
  - `85343`-`85347` are pending with `QOSMaxGRESPerUser`, meaning the current running job is using the allowed GPU quota.
- b01 durable progress at last check: `level_a_qwen7b_bf16_math500_seed0` has `10/500` saved rows; the log had reached row `20/500`.
- Local HPC commit `6dc8ed3` records the state-race fix; this changelog/progress update records the latest operational status before the GitHub push attempt.

### Failure found

- The b01 Llama-8B branch (`level_c_llama8b_bf16_math500_seed0`) failed early while updating archive state:

```text
FileNotFoundError: state.json.tmp -> state.json
```

- Root cause: parallel inference processes shared one temp file name in `update_state()`. One process could replace/remove `state.json.tmp` while another process still expected it to exist.

### Fixed

- Updated `src/runners/checkpoint_utils.py:update_state()` to:
  - create the archive root if needed,
  - serialize state updates with `state.json.lock` on Unix via `fcntl.flock`,
  - write through a unique temp file from `tempfile.mkstemp()`,
  - fsync before replacing `state.json`,
  - clean up any leftover unique temp file on error.
- Validated the fix with an 8-process local concurrency check that repeatedly updated one shared `state.json`; no stale keys or temp-file failures occurred.

### Operational note

- The fix protects future job starts, including queued b02-b06 jobs once SLURM launches them.
- The already-running Qwen-7B process in job `85342` loaded the old code before this edit, but the competing Llama-8B process has already exited, so the two-process state race is no longer active in that job.
- To recover the missing Llama-8B BF16 b01 result, resubmit that cell or the corrected b01 block after deciding whether to let the current Qwen-7B branch finish first.
- If GitHub push from HPC fails due to SSH credentials, sync the unpushed HPC commits through the MacBook rsync workflow before resetting or pulling the HPC worktree.

---

Detailed running log for project setup, HPC runs, code fixes, and operational decisions.

## 2026-06-29 (Master progress documentation)

### Added / updated

- **`progress.md`** — rewritten as the **canonical master reference** with:
  - Current status snapshot (2026-06-29)
  - Full dated timeline: 2026-06-26 (MacBook pipeline + HPC bootstrap), 2026-06-27 (smoke failures + fixes), 2026-06-28 (5080 + publication split + HPC preflight)
  - Machine roles, roadmap position, gate checklist, known failures, sync workflow
  - Preserved detailed 2026-06-28 HPC session logs at bottom
- **`docs/EXPERIMENT_LOG.md`** — added dated entries for 2026-06-26 HPC bootstrap, 2026-06-27 smoke failures, 2026-06-28 publication preflight
- **`docs/BEGINNER_HPC_GUIDE.md`** — added to repo (PARAM Rudra beginner guide)
- **`paper 1/AGENTS.md`** — refreshed snapshot to 2026-06-29 (MacBook workspace, not in git)

### Why

Consolidates MacBook, HPC, and Windows progress from multiple session reports into one dated file for future reference. **`docs/PROGRESS.md`** remains the short live-status tracker; **`progress.md`** is the full historical log.

---

## 2026-06-28 (5080 run stopped — HPC-only policy)

### Decision

- User stopped 5080 publication run at ~Q12/500 on cell 1 (~10 rows checkpointed)
- **Reason:** ~15 min/question → ~3 weeks for 4-cell 1.5B grid; PC cannot run continuously
- **New policy:** all publication experiments on **HPC only** (5080 for smoke/debug if needed)

### Actions

- Ran `clean_5080_run.sh` — killed `run_all_5080_phases`, `run_inference`, vLLM; GPU back to idle
- Background WSL task exited (code 15) — expected after manual stop
- Partial archive preserved: `outputs-win5080-main-2026-06-28/` (10/500 rows — not for paper tables)
- Updated **`docs/PROGRESS.md`**, **`README.md`**, **`CHANGELOG.md`**
- Pushed to GitHub: HPC-only policy + progress tracker

### Next (HPC)

```bash
ssh manishn_iitp@paramrudra.iitp.ac.in -p 4422
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR && git pull origin main
bash scripts/hpc/submit_hpc_blocks.sh   # b01–b06
```

- **1.5B cells:** not in b01–b06 yet — extend HPC blocks or run ad-hoc on A100
- **GPQA:** b07 after Hugging Face gate

---

## 2026-06-28 (end of day — GitHub push + 5080 run started)

### GitHub

- Pushed commits `30c8c08` and `03c3766` to https://github.com/Manish06N/reasoning-compression-lab (`main` synced)
- Added **`docs/GIT_CREDENTIALS.md`** — PAT via Windows Credential Manager (never commit tokens)
- Updated **`.gitignore`** — `.env.local`, `.github-token`

### 5080 publication run started

- Archive: **`outputs-win5080-main-2026-06-28/`**
- Launcher: `start_5080_main.sh` → `run_5080_publication.sh` with 4-cell queue (`5080_cells.sh`)
- Smoke: `smoke_qwen15b_bf16` completed
- **Cell 1 running:** `level_c_qwen15b_bf16_math500_seed0` (MATH-500 n=500)
- Observed timing: Q1 ~50 s; Q2 ~21 min (long reasoning at max_tokens=32768)
- Revised ETA: **~4–7 days** for all 4 cells (variable per question); monitor after Q10 checkpoint
- Lesson: background runs must stay in a **persistent WSL session** — short `wsl bash -lc` invocations kill detached jobs

### HPC (not started yet)

- User to run: `git pull` → download 7B/8B models → `bash scripts/hpc/submit_hpc_blocks.sh`
- Blocks b01–b06 ready in repo; b07 GPQA after HF gate

### Docs added/updated today

- **`docs/PROGRESS.md`** — live status tracker (new)
- **`README.md`** — current status banner, push complete, PROGRESS link
- **`docs/HPC_2A100_PLAN.md`**, **`RTX5080_EXECUTION_PLAN.md`**, **`MODEL_ROSTER.md`**

---

---

## 2026-06-28 (5080 ≤24h rule + full HPC block grid)

### HPC preflight follow-up

- Added `/home/manishn_iitp/.codex/CODEX.md` for future Codex sessions.
  - Records the HPC vs 5080 split.
  - Records the CPU preflight and GPU smoke gate before b01-b06 submission.
  - Records that GitHub tokens must not be stored in repo files or persistent config.
- Push to GitHub remains pending because HPC lacks safe GitHub auth in the current non-interactive environment.

- Pulled `origin/main` on PARAM Rudra scratch repo and fast-forwarded to
  `03c3766`.
- Added `progress.md` as the detailed operational record for HPC and 5080
  handoff notes.
- Added `scripts/hpc/07_preflight_publication.py`.
  - Runs CPU-side checks before expensive SLURM publication jobs.
  - Checks shell syntax for HPC submit/run scripts and SLURM wrappers.
  - Runs `python -m compileall -q scripts src`.
  - Verifies `prompts/math500.txt` formatting keeps literal `{ANSWER}`.
  - Verifies b01-b06 only contain HPC-scope 7B/8B/GSM8K cells, not Qwen-1.5B
    5080 cells.
  - Resolves every b01-b06 cell through `load_cell_config()`.
  - Checks every resolved model folder has `config.json`, tokenizer files, and
    weights.
  - Loads MATH-500 and GSM8K through the repo task configs and verifies row
    counts (`500` and `1319`).
- Ran the new preflight successfully on HPC after model downloads completed.
- Submitted only the exclusive quick smoke gate, job `85306`.
  - Purpose: verify the actual GPU/vLLM path before launching 47-hour
    publication jobs.
  - State when recorded: `PENDING`, reason `(Priority)`.
  - Estimated start from SLURM: `2026-06-29T02:21:03`.
- No b01-b06 publication jobs were submitted yet. They should wait for the
  smoke result unless explicitly overridden.

### Policy (revised)

- **5080:** **only** Qwen-1.5B × 4 quants × MATH-500 (~≤24 h/cell, ~4 days total) — `configs/machine_split/5080_cells.sh`
- **HPC 2× A100:** **all** 7B/8B quants, BF16 anchors, GSM8K (b01–b06); GPQA in b07
- **Rule:** if a cell exceeds ~1 day on 5080, it must run on HPC
- **Journal protocol** on both: `repro_qrm.yaml`, batch_size=1, full datasets, seed 0
- **GitHub:** https://github.com/Manish06N/reasoning-compression-lab

### Why the split changed

Earlier plan ran 13 cells on 5080 (7B/8B quants + GSM8K). At publication settings (`batch_size=1`, full MATH-500), that would take **weeks** on a 5080. User policy: **5080 only for jobs ≤ ~1 day per cell**; everything else on 2× A100 (160 GB VRAM, 48 h SLURM max).

### HPC blocks (seed 0)

| Block | GPUs | Est. | Content |
|-------|------|------|---------|
| b01 | 2× A100 | 12–24 h | BF16 Qwen-7B + BF16 Llama-8B MATH-500 (parallel) |
| b02 | 2× A100 | 12–24 h | FP8 Qwen-7B + FP8 Llama-8B MATH-500 |
| b03 | 2× A100 | 12–24 h | AWQ-4 Qwen-7B + AWQ-4 Llama-8B MATH-500 |
| b04 | 2× A100 | 12–24 h | GPTQ-4 Qwen-7B + GPTQ-4 Llama-8B MATH-500 |
| b05 | 1× A100 | 12–20 h | GPTQ-3 Qwen-7B MATH-500 |
| b06 | 1× A100 | 20–40 h | FP8 Qwen-7B GSM8K (n=1319) |
| b07 | 1× A100 | 8–20 h | GPQA-Diamond (after HF gate) |

Submit on HPC: `bash scripts/hpc/submit_hpc_blocks.sh` (b01–b06); GPQA: `sbatch slurm/hpc_2a100_b07_gpqa.slurm`

### 5080 cells (seed 0)

| # | Cell | Model | Task | n |
|---|------|-------|------|---|
| 1 | level_c_qwen15b_bf16 | Qwen-1.5B BF16 | MATH-500 | 500 |
| 2 | level_c_qwen15b_fp8 | Qwen-1.5B FP8 | MATH-500 | 500 |
| 3 | level_c_qwen15b_awq4 | Qwen-1.5B AWQ-4 | MATH-500 | 500 |
| 4 | level_c_qwen15b_gptq4 | Qwen-1.5B GPTQ-4 | MATH-500 | 500 |

Run on 5080: `bash scripts/local/run_5080_publication.sh --skip-download`

### Added

- **`scripts/local/run_5080_publication.sh`** — canonical 5080 entry point (4-cell queue)
- **`scripts/hpc/run_hpc_2a100_publication.sh`** — HPC block runner (b01–b07)
- **`scripts/hpc/submit_hpc_blocks.sh`** — SLURM submit b01–b06 (2-GPU / 1-GPU auto)
- **`configs/machine_split/5080_cells.sh`** — 4-cell 5080 queue (1.5B only)
- **`configs/machine_split/hpc_blocks/`** — b01–b06 block definitions + b07 GPQA
- **`slurm/hpc_2a100_b01_parallel.slurm`**, **`slurm/hpc_2a100_b07_gpqa.slurm`**
- **`docs/HPC_2A100_PLAN.md`** — full split table + HPC pull/run instructions
- **README.md** — GitHub push guide (PAT / `gh` / SSH) + block grid table

### Changed

- **`run_5080_main.sh`** → delegates to `run_5080_publication.sh`
- **`run_all_5080_phases.sh`** — loads `QREASON_CELL_QUEUE` from machine_split config
- **`param_rudra_env.sh`** — Llama-8B path exports for HPC BF16 block
- **`docs/RTX5080_EXECUTION_PLAN.md`**, **`docs/MODEL_ROSTER.md`** — 5080 = 1.5B only

### Git / deploy status (superseded by end-of-day entry above)

- Initial push failed from agent; later pushed successfully on 2026-06-28

### Operational next steps (partially done)

1. ~~Push to GitHub~~ — done
2. HPC: `git pull` → `submit_hpc_blocks.sh` (b01–b06) — **pending**
3. ~~5080: restart with 4-cell queue~~ — **running**
4. Merge `outputs-win5080-main-*` + `outputs-hpc-2a100-main-*` summaries — **after runs complete**

### Supersedes (same day, earlier entry below)

The entry *"5080 as primary machine"* (13-cell grid on 5080) is **obsolete** — replaced by this ≤24h rule.

---

## 2026-06-28 (5080 — publication main grid; 5080 as primary machine)

### Policy change (journal)

- **RTX 5080 is the primary experiment machine** — main grid runs at publication standard locally.
- **HPC only for overflow** — BF16 7B/8B, 14B+, or other models that exceed 16 GB VRAM.
- **Pilot mode demoted** to optional debug (`--pilot`); not for paper tables.

### Publication protocol (main grid)

| Setting | Value |
|---------|--------|
| Archive | `outputs-win5080-main-2026-06-28/` |
| Decoding | `configs/decoding/repro_qrm.yaml` |
| batch_size | **1** (sequential, QRM-compatible) |
| Sample sizes | MATH-500 n=500, GSM8K n=1319 |
| Reproducibility | `VLLM_BATCH_INVARIANT=1` |
| Checkpoints | every 10 rows |

### Added

- **`scripts/local/run_5080_main.sh`**, **`resume_5080_main.sh`**, **`start_5080_main.sh`**
- **`scripts/local/clean_5080_run.sh`** — generalized clean (main or pilot)
- **`outputs-win5080-main-2026-06-28/README.md`**

### Changed

- **`scripts/local/run_all_5080_phases.sh`** — non-pilot defaults: `repro_qrm.yaml`, batch=1, `outputs-win5080-main-*`, `publication_mode` in manifest
- **`README.md`**, **`docs/RTX5080_EXECUTION_PLAN.md`**, **`docs/MODEL_ROSTER.md`**, **`2026/CLAUDE.md`**

### Start

```bash
bash scripts/local/clean_5080_run.sh pilot   # stop old pilot
bash scripts/local/start_5080_main.sh --skip-download
```

Monitor: `outputs-win5080-main-2026-06-28/logs/orchestrator.log`

---

## 2026-06-28 (Windows 5080 — pilot pipeline started)

> **Session reference — keep this updated.** Mirrors `README.md` → "Current session — Windows RTX 5080 pilot".

### Operational status (as of 2026-06-28)

| Item | Status |
|------|--------|
| **Repo (Windows)** | `G:\ALL MY Projects\2026\03-paper1-experiments` |
| **Repo (WSL)** | `/mnt/g/ALL MY Projects/2026/03-paper1-experiments` |
| **Models on disk** | 12 checkpoints (~62 GB) — all 5080 quants + BF16 1.5B/7B |
| **Missing model** | Llama-8B BF16 (HPC only — run `download_models.sh levelc` later) |
| **Pilot archive** | `outputs-win5080-pilot-2026-06-28/` |
| **Aborted full run** | `outputs-win5080-2026-06-28/` — superseded, do not merge |
| **Stack** | `torch 2.11.0+cu128`, `vllm 0.23.0`, conda `qreason` |
| **Mode** | Pilot — n=50, `pilot_5080.yaml`, max_tokens 8192, batch 4/2/1 |
| **Pipeline** | `run_all_5080_phases.sh --pilot --skip-download` |
| **Smoke** | `smoke_qwen15b_bf16.jsonl` (1 row) — skipped on resume |

### What was superseded

| Old approach | Outcome |
|--------------|---------|
| `download_and_run_5080.sh` (full grid) | Killed (exit 9) — downloads done; full MATH-500 too slow on 5080 |
| `outputs-win5080-2026-06-28/` partial run | Abandoned — pilot archive is canonical for 5080 work |

### Start / monitor / resume

```bash
wsl -d Ubuntu-22.04
cd "/mnt/g/ALL MY Projects/2026/03-paper1-experiments"
source scripts/local/env.sh
bash scripts/local/start_5080_pilot.sh          # background start
bash scripts/local/resume_5080_pilot.sh         # foreground / after power cut
bash scripts/local/backup_5080_archive.sh --snapshot
```

Monitor: `outputs-win5080-pilot-2026-06-28/logs/orchestrator.log`

PowerShell: `Get-Content "G:\ALL MY Projects\2026\03-paper1-experiments\outputs-win5080-pilot-2026-06-28\logs\orchestrator.log" -Tail 15 -Wait`

### 14-cell pilot queue

1. `smoke_qwen15b_bf16`  
2–5. Qwen-1.5B BF16 / FP8 / AWQ-4 / GPTQ-4 × MATH-500  
6. `level_a_gptq4_seed0` (Qwen-7B GPTQ-4)  
7–10. Qwen-7B FP8 / AWQ-4 / GPTQ-4 / GPTQ-3 × MATH-500  
11. Qwen-7B FP8 × GSM8K  
12–14. Llama-8B FP8 / AWQ-4 / GPTQ-4 × MATH-500  

Skipped on 5080: BF16 Qwen-7B/8B full MATH-500, GPQA-Diamond (gated).

### Backup / resume mechanics

- Atomic JSONL every 10 rows → `_backup/latest/raw/` on each checkpoint  
- Full mirror after each cell; snapshot every 3 cells → `_backup/snapshots/`  
- Partial cells resume from row count in existing JSONL  
- Manifest **merged** on restart (cells[] not wiped)  
- Corrupt JSONL → auto-restore from `_backup/latest/raw/`

### Added this session

- **`scripts/local/start_5080_pilot.sh`** — idempotent background launcher (setsid, stale vLLM cleanup).
- **Smoke skip fix** — smoke skips on ≥1 row (not pilot `limit=50`).

---

## 2026-06-28 (Windows 5080 — backup + resume / power-cut recovery)

- **`src/runners/checkpoint_utils.py`** — atomic JSONL writes, progress sidecars, `_backup/latest` mirror, timestamped snapshots, corrupt-file recovery.
- **`scripts/local/backup_5080_archive.sh`** — manual full-archive backup (`--snapshot` for timestamped copy).
- **`scripts/local/resume_5080_pilot.sh`** — resume pilot after reboot (preserves manifest, skips done cells).

### Changed

- **`scripts/run_inference.py`** — atomic checkpoints every 10 rows; auto-backup to `_backup/latest/raw/`; corrupt JSONL → restore from backup; writes `checkpoints/` + `state.json`.
- **`scripts/local/run_all_5080_phases.sh`** — manifest **merge on restart** (no longer wipes `cells[]`); auto backup after each cell; snapshot every 3 cells; skip re-scoring if scored file is current; `[resume]` log for partial cells; fixed Windows path in manifest Python block (SyntaxError on startup).

### Recovery paths

| Artifact | Location |
|----------|----------|
| Live output | `{archive}/raw/{cell_id}.jsonl` |
| Backup mirror | `{archive}/_backup/latest/` |
| Snapshots | `{archive}/_backup/snapshots/YYYYMMDD_HHMMSS/` |
| Progress | `{archive}/checkpoints/{cell_id}.json`, `state.json` |

After power cut: `bash scripts/local/resume_5080_pilot.sh`

---

## 2026-06-28 (Windows 5080 — pilot mode + batched inference)

### Added

- **`configs/decoding/pilot_5080.yaml`** — 5080 pilot protocol: temp 0.6, top_p 0.95, `max_tokens` / `max_model_len` 8192 (same sampling as repro, capped length).
- **`scripts/local/run_5080_pilot.sh`** — one-command pilot grid (`--pilot`, n=50 default, separate archive).
- **`outputs-win5080-pilot-2026-06-28/`** — pilot output archive + `README.md` (do not mix with full repro results).

### Changed

- **`scripts/run_inference.py`** — vLLM-native batching (`--batch-size`), decoding override (`--decoding-config`), `max_model_len` override, resume from partial JSONL checkpoints.
- **`src/runners/vllm_runner.py`** — `render_prompt`, `generate_chunk` (true multi-prompt `llm.generate`), `generate_batch` uses chunked vLLM calls (no longer sequential one-by-one).
- **`src/runners/config_utils.py`** — `load_decoding_from_file()`; YAML may include `max_model_len`.
- **`scripts/local/run_all_5080_phases.sh`** — flags: `--pilot`, `--skip-download`, `--decoding-config`, `--max-model-len`, `--batch-size`; auto batch sizes (4/2/1 by model size); pilot writes to `outputs-win5080-pilot-*`; smoke always `--limit 1`.
- **`docs/RTX5080_EXECUTION_PLAN.md`**, **`docs/MODEL_ROSTER.md`**, archive READMEs — pilot vs full repro documented.

### Why

Live full-grid run on 5080 showed ~1–4k tokens/question but `max_tokens=32768` and batch-size-1 inference → **days per cell**. Pilot mode targets **~hours for all 14 cells** while preserving quant comparison validity (label `n=50`, separate archive).

### How to run pilot (stop any in-progress full run first)

```bash
wsl -d Ubuntu-22.04
cd "/mnt/g/ALL MY Projects/2026/03-paper1-experiments"
source scripts/local/env.sh
# Optional: pkill -f run_inference if a full MATH-500 cell is still running
bash scripts/local/run_5080_pilot.sh --skip-download
```

Monitor: `outputs-win5080-pilot-2026-06-28/logs/master.log`

---

## 2026-06-28 (Windows 5080 — local-only full phase run)

### Added

- **`outputs-win5080-2026-06-28/`** — dedicated Windows archive for all RTX 5080 outputs (`raw/`, `scored/`, `results/`, `logs/`, `manifest.json`). Windows path: `G:\ALL MY Projects\2026\03-paper1-experiments\outputs-win5080-2026-06-28\`.
- **`scripts/local/run_all_5080_phases.sh`** — runs every 5080-feasible cell (no HPC); skips BF16 7B/8B and gated GPQA; resumes completed cells.
- **`scripts/local/download_and_run_5080.sh`** — download all 5080 quants then chain into phase runner.
- **`download_models.sh` target `5080`** — 10 quant checkpoints (1.5B/7B/Llama-8B FP8/AWQ/GPTQ + 7B GPTQ-3); phase0 BF16 1.5B+7B separate.
- **Qwen-1.5B quant configs** — `deepseek_r1_qwen_15b_fp8.json`, `_awq4.json`, `_gptq4.json` + Level C cells.
- **`requirements-local-5080.txt`**, **`scripts/local/check_cuda.py`**.

### Changed

- GPTQ-4 canonical HF ID → `RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16` (ruikangliu repo gone); same for 1.5B/Llama GPTQ-4.
- `scripts/run_inference.py` — GSM8K `question`/`answer` fields, `config_name`, absolute `--output` paths.
- `configs/tasks/gsm8k.json` — `problem_field` / `solution_field`.
- `prompts/math500.txt` — escaped `{{ANSWER}}` for `.format()`.
- vLLM stack on 5080: `torch 2.11.0+cu128`, `vllm 0.23.0` (Blackwell sm_120).

### 5080 cells in scope

| Phase | Cells |
|-------|-------|
| Phase 0 | `smoke_qwen15b_bf16` |
| Level A | `level_a_gptq4_seed0` |
| Level B | FP8/AWQ-4/GPTQ-4/GPTQ-3 MATH-500 + FP8 GSM8K |
| Level C | Qwen-1.5B BF16/FP8/AWQ/GPTQ-4 + Llama-8B FP8/AWQ/GPTQ-4 MATH-500 |

### Skipped on 5080

- All Qwen-7B / Llama-8B **BF16** full runs (VRAM).
- GPQA-Diamond (gated).
- Llama-8B BF16 download deferred to HPC (`download_models.sh levelc`).

### Download status (2026-06-28)

All 12 local model folders present (~62 GB total). Experiments started on full grid before pilot mode was added; use pilot archive for new runs.

---


All work below was done on the **home Windows machine** with **NVIDIA GeForce RTX 5080 (16 GB VRAM, Blackwell sm_120)** via **WSL2**. HPC (PARAM Rudra A100) was **not** reached from this session — SSH from WSL failed with `Permission denied (publickey)`.

Repo path (Windows): `G:\ALL MY Projects\2026\03-paper1-experiments`  
Repo path (WSL): `/mnt/g/ALL MY Projects/2026/03-paper1-experiments`  
Conda env: `qreason` (Python 3.11)

---

### What worked

| Item | Result |
|------|--------|
| **CUDA / PyTorch on Blackwell** | After force-reinstall: `torch 2.11.0+cu128`, CUDA available, sm_120 tensor OK (`scripts/local/check_cuda.py`). |
| **vLLM import & inference (1.5B)** | Upgraded to `vllm 0.23.0` (0.8.5 incompatible with torch 2.11). Qwen-1.5B BF16 loads and generates on 5080. |
| **Phase 0 model downloads** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` and `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` downloaded to `models/` (~22 min for phase0 batch). |
| **Phase 0 smoke — Qwen-1.5B BF16** | `bash scripts/local/run_phase0_smoke.sh` → `runs/raw/smoke_qwen15b_local.jsonl` (1 row, limit=1, max_tokens=64). Pipeline verified: download → vLLM → JSONL. |
| **WSL `.env` loading** | CRLF from Windows editing broke `source .env`; fixed in `scripts/local/env.sh` via `tr -d '\r'` and CR-stripping on path vars. |
| **Blackwell vLLM env vars** | `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer JIT fails sm_120 check), `VLLM_WORKER_MULTIPROC_METHOD=spawn` (WSL), `LD_LIBRARY_PATH` for pip-shipped CUDA 13 libs (`nvidia/cu13/lib`). |
| **Prompt template fix** | `prompts/math500.txt`: escaped `{{ANSWER}}` so Python `.format()` no longer raises `KeyError: 'ANSWER'`. |

**Final local stack (5080):**

```
torch==2.11.0+cu128
vllm==0.23.0
GPU: NVIDIA GeForce RTX 5080, capability (12, 0)
```

**Sample 1.5B smoke output** (`runs/raw/smoke_qwen15b_local.jsonl`):

- Problem: `What is 17 + 28?`
- peak_vram_gb: ~15.9 (high due to max_model_len=32768 in 1.5B config)
- latency_sec: ~64 s (first run; WSL + 9P filesystem overhead on model load)

---

### What did not work

| Item | Symptom | Root cause / next step |
|------|---------|------------------------|
| **Original stack (torch 2.6 + vLLM 0.8.5)** | `CUDA error: no kernel image is available for execution on the device` | PyTorch cu124 only supports up to sm_90; RTX 5080 is sm_120. |
| **`upgrade_pytorch_blackwell.sh` (first run)** | No upgrade; torch stayed 2.6.0 | Plain `pip install` saw torch as satisfied. **Fix:** use `--force-reinstall` (now in script). |
| **vLLM 0.8.5 + torch 2.11** | `undefined symbol: _ZN5torch3jit17parseSchemaOrName...` in `vllm/_C.abi3.so` | vLLM 0.8.5 wheel built against torch 2.6. Upgraded to vLLM 0.23.0. |
| **vLLM 0.23.0 (first import)** | `libcudart.so.13: cannot open shared object file` | vLLM 0.23 links CUDA 13; added `LD_LIBRARY_PATH` in `env.sh`. |
| **FlashInfer sampler on 5080** | `RuntimeError: FlashInfer requires GPUs with sm75 or higher` during engine init | sm_120 not recognized by FlashInfer JIT (`SM 12.x requires CUDA >= 12.9`). **Workaround:** `VLLM_USE_FLASHINFER_SAMPLER=0`. |
| **Phase 0 smoke — Qwen-7B BF16 on 5080** | `ValueError: No available memory for the cache blocks` / KV cache `-2.03 GiB` | BF16 weights alone ~14.32 GiB on 16 GB card; no room for KV cache even with `max_model_len=512`, `kv_cache_dtype=fp8`, `gpu_memory_utilization=0.85`. **Expected per plan:** full 7B BF16 runs on HPC A100 only. |
| **GPTQ-4 download** | `Error: Model 'ruikangliu/DeepSeek-R1-Distill-Qwen-7B-GPTQ-W4G128' not found` | HF repo missing or renamed. **Next:** try `RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16` (canonical alternate in `docs/MODEL_ROSTER.md`) and update `download_models.sh`. |
| **HPC SSH from WSL** | `Permission denied (publickey)` after host key added | No SSH private key in WSL `~/.ssh/`. Phase 1 Gate 3–4 must be run manually via PuTTY or after copying Windows SSH key into WSL. |
| **PC forced restart** | Interrupted long-running 7B smoke attempt mid-session | Re-ran Phase 0 after reboot; 1.5B smoke passed again; 7B still OOM-deferred. |

**7B BF16 deferral artifact:** `runs/raw/smoke_qwen7b_local_status.json` — status `deferred_to_hpc`, points to `scripts/hpc/03_smoke_test.sh`.

---

### Files added or changed (5080 session)

| File | Change |
|------|--------|
| `scripts/local/env.sh` | CUDA 13 `LD_LIBRARY_PATH`, Blackwell vLLM env vars, CRLF-safe `.env` sourcing |
| `scripts/local/check_cuda.py` | Quick CUDA sanity check for WSL/5080 |
| `scripts/local/upgrade_pytorch_blackwell.sh` | `--force-reinstall` torch cu128 + upgrade vLLM |
| `scripts/local/run_phase0_smoke.sh` | 1.5B required; 7B BF16 try with graceful HPC deferral on OOM |
| `requirements-local-5080.txt` | Local-only deps (torch 2.11+, vllm 0.23+); HPC stays on `requirements-hpc.txt` (vLLM 0.8.5) |
| `prompts/math500.txt` | Escaped `{{ANSWER}}` for `.format()` |
| `configs/models/deepseek_r1_qwen_7b_smoke_5080.json` | `max_model_len=512`, `kv_cache_dtype=fp8`, `gpu_memory_utilization=0.85` (still OOM on 5080) |

*(Earlier 2026-06-28 entries below cover repo scaffolding, model roster docs, and HPC-side work.)*

---

### Current next steps (5080 vs HPC)

**On Windows 5080 (WSL):**

```powershell
wsl -d Ubuntu-22.04
cd "/mnt/g/ALL MY Projects/2026/03-paper1-experiments"
source scripts/local/env.sh
bash scripts/local/run_phase0_smoke.sh   # 1.5B smoke; 7B defers if OOM
# After GPTQ repo ID fixed:
bash scripts/local/download_models.sh gptq4
bash scripts/local/run_gptq4_smoke.sh    # quantized 7B should fit 5080
```

**On HPC (manual — SSH from WSL blocked):**

```bash
cd /scratch/manishn_iitp/reasoning-compression-lab   # or synced clone path
bash scripts/hpc/03_smoke_test.sh                    # Gate 3: 7B BF16 smoke
bash scripts/hpc/run_level_a_sequence.sh 10          # Gate 4 + score
# Target: results/level_a_qwen7b_bf16_math500_seed0_summary.json
```

---

## 2026-06-28

### Added

- Canonical model roster: `docs/MODEL_ROSTER.md`, `docs/GPQA_ACCESS.md`.
- Model JSON configs for Qwen-1.5B, Qwen-7B (BF16/FP8/AWQ-4/GPTQ-4/GPTQ-3), Llama-8B variants.
- Level B cell templates (`configs/cells/level_b_*`) and Level C cells (`configs/cells/level_c_*`).
- Task configs: `gsm8k.json`, `gpqa_diamond.json`.
- Local scripts: `download_models.sh`, `run_phase0_smoke.sh`, `run_gptq4_smoke.sh`, `upgrade_pytorch_blackwell.sh`.
- HPC helper: `scripts/hpc/check_hpc_gate_status.sh`, `scripts/hpc/run_level_a_sequence.sh`.

### Changed

- `src/runners/vllm_runner.py`: optional `quantization`, `kv_cache_dtype`, `gpu_memory_utilization`.
- `level_a_gptq4_seed0.json`: uses `deepseek_r1_qwen_7b_gptq4.json` (vLLM quant flags).
- `.env.example`: full `QREASON_MODEL_*` path map for WSL and HPC.

## 2026-06-28 (workspace move)

### Changed

- **Moved** from `2026/reasoning-compression-lab` to `2026/03-paper1-experiments`.
- **Updated** path references in docs and scripts.
- **Why:** Numbered PhD workspace layout; see `2026/README.md`.

## 2026-06-27

### Fixed

- Added a compatibility shim in `src/runners/vllm_runner.py` for the installed
  `vllm==0.8.5` and `transformers==5.12.1` combination.
  - Problem: `vLLM` calls `tokenizer.all_special_tokens_extended`, but the
    installed Transformers tokenizer base class does not expose that property.
  - Symptom from job `85028`: `AttributeError: Qwen2Tokenizer has no attribute
    all_special_tokens_extended`.
  - Fix: before constructing a vLLM `LLM`, add `all_special_tokens_extended` to
    `PreTrainedTokenizerBase` when the property is missing, returning
    `all_special_tokens`.
  - Scope: applies to both `build_llm()` and `generate_one()`, so smoke tests
    and normal inference share the same compatibility path.

### Changed

- Updated `scripts/hpc/03_smoke_test.sh` to make smoke testing configurable and
  more useful on shared HPC nodes.
  - Added `SMOKE_LIMIT`, default `3`.
  - Added `SMOKE_OUTPUT`, default `runs/raw/smoke_test.jsonl`.
  - Added `SMOKE_MAX_TOKENS`, optional override passed through to
    `scripts/smoke_test.py --max-tokens`.
  - Switched smoke execution to `python -u` so logs flush while the Slurm job is
    running.
  - Added a GPU free-memory preflight using `nvidia-smi`.
  - Added `SMOKE_MIN_FREE_GPU_MB`, default `30000`, to fail early when the
    allocated GPU is already too full for vLLM initialization.

### Added

- Added `slurm/smoke_test_quick_exclusive.slurm`.
  - Purpose: a smaller validation job for shared HPC conditions.
  - Requests one generic GPU and an exclusive node allocation.
  - Runs one smoke question instead of three.
  - Uses `SMOKE_MAX_TOKENS=64` instead of the default smoke setting of `1024`.
  - Writes to `runs/raw/smoke_test_quick.jsonl`.

### HPC Runs

- Job `85092` (`qreason-smoke`) was submitted after the tokenizer shim.
  - Start: `2026-06-27T06:47:23`.
  - End: `2026-06-27T06:52:30`.
  - State: `FAILED`.
  - Exit code: `1:0`.
  - Important result: the previous tokenizer error did not recur.
  - New failure: CUDA out of memory during model loading.
  - Root cause from logs: the assigned A100 had only `23.62 MiB` free. Other
    processes were using about `63.72 GiB` and `13.72 GiB`, so vLLM could not
    allocate another `26.00 MiB` while initializing the model.
  - Artifact status: `runs/raw/smoke_test.jsonl` was not created.

- Job `85094` (`qreason-smoke-quick`) was submitted as the exclusive quick smoke
  workaround.
  - Current state when last checked: `PENDING`.
  - Queue reason: `Resources`.
  - Artifact status: `runs/raw/smoke_test_quick.jsonl` does not exist yet
    because the job has not started.

- Job `85030` (`qreason-level-a-bf16`) was canceled.
  - Original purpose: dependent 10-question BF16 debug job.
  - Reason for cancellation: dependency could never be satisfied after the
    first smoke test failed.
  - Final state: `CANCELLED by 65865`.
  - End: `2026-06-27T07:03:09`.

### Current Next Step

- Wait for job `85094` to start and finish:

```bash
squeue -j 85094
sacct -j 85094 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End -P
```

- When it finishes, check:

```bash
ls -l runs/raw/smoke_test_quick.jsonl
cat logs/smoke_quick_85094.out
cat logs/smoke_quick_85094.err
```

- If `runs/raw/smoke_test_quick.jsonl` exists, smoke passed and the next step is
  a limited BF16 debug run before the full MATH-500 run.

## 2026-06-26

### Project State

- Project structure and experiment scaffolding were already present on the HPC
  filesystem at `/scratch/manishn_iitp/reasoning-compression-lab`.
- Existing experiment plan in `docs/EXPERIMENT_LOG.md` recorded:
  - Level A BF16 reproduction gate planned.
  - Model: `DeepSeek-R1-Distill-Qwen-7B`.
  - Task: `MATH-500`.
  - Seed: `0`.
  - Target hardware: A100.
  - GPTQ-4 reproduction blocked until BF16 Level A is complete.

### Validated

- Job `85013` (`qreason-gpu-check`) completed successfully.
  - Start: `2026-06-26T19:16:35`.
  - End: `2026-06-26T19:19:17`.
  - State: `COMPLETED`.
  - Exit code: `0:0`.
  - Node: `ragpu006`.
  - GPU: NVIDIA A100 80GB PCIe.
  - CUDA available: `True`.
  - PyTorch version: `2.6.0+cu124`.
  - vLLM version: `0.8.5`.
  - Result: GPU and vLLM import path were validated.

### Failed

- Job `85028` (`qreason-smoke`) ran the initial smoke test.
  - Start: `2026-06-26T20:58:20`.
  - End: `2026-06-26T21:03:52`.
  - State: `FAILED`.
  - Exit code: `1:0`.
  - Node: `ragpu004`.
  - Smoke config: `max_tokens=1024`.
  - Model path:
    `/scratch/manishn_iitp/reasoning-compression-lab/models/DeepSeek-R1-Distill-Qwen-7B`.
  - Failure point: vLLM tokenizer initialization.
  - Root cause:
    `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`.
  - Artifact status: `runs/raw/smoke_test.jsonl` was missing.

### Operational Notes

- Watcher logs for `85028` recorded the job moving from `PENDING` to `RUNNING`,
  then final state `FAILED 1:0`.
- Watcher logs for `85030` recorded that the dependent debug job remained
  pending because smoke had not passed.
- Slurm watcher jobs `85031` and `85032` failed to send Telegram notifications.
  - Error: `curl: (6) Could not resolve host: api.telegram.org`.
  - Interpretation: notification failure was a network/DNS issue and separate
    from the model smoke-test failure.

## How To Maintain This File

- Add a new dated section for every material project change or HPC run.
- For code changes, include:
  - File changed.
  - Reason.
  - Behavior before and after.
- For Slurm jobs, include:
  - Job ID and name.
  - Start/end time.
  - State and exit code.
  - Log files inspected.
  - Artifact path created or missing.
  - Root cause if failed.
