# Known issues and limitations

Last updated: 2026-07-01

Operational issues that can break paper results if ignored, plus known software limitations.

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

---

## Important — affects interpretation

### 4. Single-sample calibration is a proxy

`score_run.py` adds calibration/selective-risk using `answer_parse_success` as confidence when logprobs are absent.  
For real calibration claims, use **maj@5** (`run_inference_multisample.py` + `compute_calibration.py`).

### 5. Mixed provenance on resumed inference

Rows written before V8.2 provenance fields lack `run_id`, `git_commit`, etc. New rows in the same JSONL have them.  
Analysis should filter by `schema_version` or rerun fresh archives for publication.

### 6. QRM reproduction vs main grid prompts

| Profile | Used for | Prompt style |
|---------|----------|--------------|
| `reproduction` | Level A repro gate | Short QRM `\boxed{}` |
| `sober` | Level B/C main grid | Long sober-reasoning template |

Comparing Level A to Level B pass@1 directly confounds prompt + quant — compare within profile.

### 7. GPQA answer shuffle

Deterministic per `(seed, row_index)` in this harness vs QRM’s fixed pipeline RNG. Document when comparing to QRM Table 1.

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
