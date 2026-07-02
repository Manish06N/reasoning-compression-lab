# Progress — Paper 1 Experiments

**Last updated:** 2026-07-02 (evening)  
**GitHub `main`:** `85998e1`  
**Repo:** https://github.com/Manish06N/reasoning-compression-lab  
**Canonical log:** [progress.md](../progress.md) · **Ops detail:** [CHANGELOG.md](../CHANGELOG.md)

---

## Summary

| Area | Status |
|------|--------|
| **J1 engineering** | **MVP complete** — RunSpec, revision pins, publication mode, CI, env docs, manifest locking, logprob pipeline |
| **J1 scientific validation** | **In flight** — split b01: **86280** Qwen RUNNING, **86281** Llama PENDING |
| **QRM baseline gates** | **Fixed** — task bands + quant/profile SKIP logic |
| **15 core validation cells** | Wired seed 0 (b01–b09) |
| **Logprobs** | **Code wired**; GPU smoke required before removing `--skip-calibration` in launcher |
| **Submit workflow** | Split 1-GPU per cell: `submit_hpc_blocks.sh b01` |
| **Policy** | HPC-only for J1 numbers |
| **Tests** | 32 targeted pass (operational fixes); 99/100 full suite (1 pre-existing scoring test fail) |

**Status label:** *J1 engineering MVP complete; b01 reproduction gate pending split jobs 86280/86281.*

**Read first:** [KNOWN_ISSUES.md](KNOWN_ISSUES.md) · [ENV_VARS.md](ENV_VARS.md) · **Runbook:** [J1_VALIDATION_RUNBOOK.md](J1_VALIDATION_RUNBOOK.md)

---

## What changed (2026-07-02 evening)

### Split b01 + git-on-compute fix

- Cancelled 2-GPU job **86229**; resubmitted as split 1-GPU jobs **86280** (Qwen) + **86281** (Llama)
- First split attempts failed: `git` not on PATH after `conda activate qreason` on compute nodes
- Fixed operationally: `conda install -y git` in `qreason`; 86280 entered inference
- Repo fix: **`85998e1`** — `00_setup_env.sh`, `param_rudra_env.sh`, preflight git check

Archive: `outputs-hpc-2a100-main-2026-07-02-p0fix`

---

## What changed (2026-07-02)

### 1. Deep re-audit P0–P2 (`af4b8c2`, pushed earlier)

- Frozen `RunSpec`, immutable HF revision pins, publication mode end-to-end
- Fresh archive: `outputs-hpc-2a100-main-2026-07-02-p0fix`

### 2. Review hardening + operational fixes (`c32a423` / `59c84dd`)

- Code-path git gates, locked manifest, split submit env, QRM quant/profile SKIP, logprob pipeline
- See [CHANGELOG.md](../CHANGELOG.md)

---

## Sync rules (critical)

| Phase | HPC git action |
|-------|----------------|
| Jobs **86280/86281 running** | **Do NOT** reset mid-run |
| **After both cells complete** | `git fetch origin && git reset --hard origin/main` |
| **Before scoring** | Kill autopush: `tmux kill-session -t hpc_git_autopush 2>/dev/null || true` |

MacBook push is inert for running Slurm jobs.

---

## b01 pass criteria (MATH-500 BF16, hard gate)

| Model | QRM ref | ±5 pp band |
|-------|---------|------------|
| Qwen-7B | 93.9% | 88.9–98.9% |
| Llama-8B | 91.0% | 86.0–96.0% |

**Both cells required** — one job finishing does not pass b01.

```bash
export QREASON_OUTPUT_ROOT="$QR/outputs-hpc-2a100-main-2026-07-02-p0fix"
python scripts/compare_qrm_baseline.py --summary "$QREASON_OUTPUT_ROOT/results/level_a_bf16_seed0_summary.json"
python scripts/compare_qrm_baseline.py --summary "$QREASON_OUTPUT_ROOT/results/level_c_llama8b_bf16_math500_seed0_summary.json"
```

---

## Block status

| Block | Status |
|-------|--------|
| b01 | Split jobs **86280** + **86281** |
| b02–b07 | Hold until b01 QRM gate + logprob GPU smoke |
| b08–b09 | Wired; optional |

Submit: `bash scripts/hpc/submit_hpc_blocks.sh b01`

---

## Hard gates before expanding

1. **b01 reproduction** — both cells pass QRM MATH-500 bands
2. **Logprob GPU smoke** — confirm `confidence_source=normalized_sequence_logprob`
3. **3-seed pilot** — before full breadth grid

---

## Documentation index

| Doc | Purpose |
|-----|---------|
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Traps including git-on-compute (§3b) |
| [ENV_VARS.md](ENV_VARS.md) | All `QREASON_*` variables |
| [J1_VALIDATION_RUNBOOK.md](J1_VALIDATION_RUNBOOK.md) | Phases 0–7 |
