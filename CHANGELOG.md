# Changelog

Detailed running log for project setup, HPC runs, code fixes, and operational decisions.

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
