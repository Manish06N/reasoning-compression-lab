# Progress Log

This file is the shared operational record for the publication runs. Keep it updated with what was changed, checked, submitted, failed, and recovered so future sessions can resume without guessing.

## Current Scope

- Repo: `/scratch/manishn_iitp/reasoning-compression-lab` on PARAM Rudra HPC.
- Branch: `main`.
- Current commit on HPC: `03c3766` (`Revise 5080/HPC split: 1.5B only on 5080; full 7B/8B grid on HPC b01-b07.`).
- Protocol target: `configs/decoding/repro_qrm.yaml`, `batch_size=1`, full datasets, seed `0`.
- HPC target: blocks `b01` through `b06` only for now. `b07` GPQA is not to be submitted until the Hugging Face gate is approved.
- 5080 target: Qwen-1.5B only. User will run and update the 5080 section separately.

## 2026-06-28 HPC Session

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
- [ ] Finish model downloads for all b01-b06 model folders.
- [ ] Verify every cell config points to an existing task config, model config, and local model path.
- [ ] Verify MATH-500 and GSM8K dataset access/cache.
- [ ] Run targeted package checks that do not hang indefinitely on the login node.
- [ ] Decide whether to submit b01 first or all b01-b06.
- [ ] Submit selected SLURM jobs.
- [ ] Record job IDs and output archive path.
- [ ] Monitor initial SLURM logs for early failures.

## 5080 Rig Notes

User will update this section while working on the RTX 5080/WSL2 side.

Expected 5080 scope from the runbook:

- Qwen-1.5B BF16 on MATH-500.
- Qwen-1.5B FP8 on MATH-500.
- Qwen-1.5B AWQ-4 on MATH-500.
- Qwen-1.5B GPTQ-4 on MATH-500.
- Do not run 7B/8B cells on the 5080.

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

After downloads finish, verify models and then submit:

```bash
export QR=/scratch/$USER/reasoning-compression-lab
cd $QR
source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh
conda activate qreason
bash scripts/hpc/submit_hpc_blocks.sh b01
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

### Checklist Update

- [x] Finish model downloads for all b01-b06 model folders.
- [x] Verify every cell config points to an existing task config, model config, and local model path.
- [x] Verify MATH-500 and GSM8K dataset access/cache.
- [ ] Finish targeted package/runtime checks.
- [ ] Submit selected SLURM jobs.

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
