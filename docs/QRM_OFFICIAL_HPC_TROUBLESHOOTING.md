# QRM official stack — HPC debugging & troubleshooting log

Last updated: **2026-08-13**

Chronological record of every failed attempt, diagnosis, and fix while bringing up **Experiment A** (official QRM `inference.py`, n=10 MATH-500, seed=42) on PARAM Rudra. Use this before re-running or extending the `qrm-official` conda env.

**Same repo, not a separate project:** all paths are under `reasoning-compression-lab/`. The main paper harness uses conda env **`qreason`** (`scripts/run_inference.py`, vLLM 0.8.5). Experiment A uses conda env **`qrm-official`** only — do not mix packages between them.

**Related docs:** [QRM_STACK_PARITY_AUDIT.md](QRM_STACK_PARITY_AUDIT.md) · [KNOWN_ISSUES.md](KNOWN_ISSUES.md) · [PARAM_RUDRA_SLURM.md](PARAM_RUDRA_SLURM.md) · [README.md](../README.md) § “One repo, two conda envs”

---

## Goal

Run the authors' stack in an isolated conda env (`qrm-official`), separate from `qreason`:

| Piece | Source |
|-------|--------|
| Repo | `external/Quantized-Reasoning-Models` |
| lighteval | QRM fork submodule |
| vLLM | QRM fork **v0.7.0** submodule |
| fast-hadamard-transform | QRM submodule (CUDA extension) |
| Entrypoint | `inference.py` via `scripts/hpc/qrm_parity/run_official_inference.sh` |

Submit:

```bash
bash scripts/hpc/submit_qrm_official_test.sh
```

---

## Job timeline (chronological)

| Job | Date | Node | Runtime | State | Layer |
|-----|------|------|---------|-------|-------|
| **87130** | 2026-07-05 | ragpu004 | — | FAILED | Env install (first attempt) |
| **87111** | 2026-07-05 | racn116 | — | FAILED | Unrelated Qwen resume — shared GPU OOM |
| **86757** | 2026-07-04 | — | ~47h | TIMEOUT | Qwen 410/500 resume (separate track) |
| **87179–87182** | 2026-07-06 | various | short | FAILED | Env install retries during script fixes |
| **87187** | 2026-07-06 | ragpu006 | ~2 min | FAILED | `fast-hadamard` compile — missing C++ |
| **87192** | 2026-07-06 | ragpu006 | ~15 sec | FAILED | `set -u` + conda gcc activate |
| **87193** | 2026-07-06 | ragpu006 | ~2 min | FAILED | Wrong vLLM precompiled wheel |
| **87196** | 2026-07-06 | ragpu006 | ~2 min | FAILED | GitPython / missing git |
| **87213** | 2026-07-06 | racn115 | ~4 min | FAILED | Shared GPU CUDA OOM |
| **87216** | 2026-07-06 | - | - | SUPERSEDED | Exclusive GPU attempt waited in queue |
| **87302** | 2026-07-06 | ragpu006 | completed | **COMPLETED** | Final non-exclusive 1-GPU official QRM run: 10/10 correct, 0 truncation |
| **96084/96085** | 2026-08-13 | ragpu004 | short | FAILED | Modern-stack FP8 jobs rejected incompatible `fp8_e5m2` KV cache before output |
| **96086/96087** | 2026-08-13 | ragpu004/ragpu008 | 1h06m/39m | CANCELED | FP8 model load fixed, but generated output looped/truncated; stopped before full run |
| **96091/96092** | 2026-08-13 | ragpu004/ragpu008 | 11m/11m | CANCELED/FAILED | vLLM 0.8.5 V0 probe; malformed answer(s), one full-length loop, then partial-n scoring schema failure |
| **96093/96094** | 2026-08-13 | ragpu008/ragpu004 | 5m29s/4m52s | **COMPLETED** | Exact-stack FP8 validation: both 10/10 correct, 10/10 boxed, no cap hits or repetition flags |

Logs: `logs/qrm_official_<JOBID>.out` / `.err`

---

## 2026-08-13 FP8 follow-up: do not confuse model loading with healthy generation

Changing `kv_cache_dtype` from `fp8_e5m2` to `auto` fixed the immediate model-initialization error in the modern `qreason` environment, but it did **not** establish a valid inference run. Jobs 96086/96087 used vLLM 0.8.5, transformers 5.12.1, seed 0, max model length 40960, and repetition penalty 1.05. Qwen's first 10 saved rows were 2/10 correct with 8/10 truncations and obvious repetition. The jobs were canceled rather than allowing the 500-row run to continue.

Setting `VLLM_USE_V1=0` in jobs 96091/96092 was also insufficient. That changes the vLLM engine implementation but leaves the rest of the modern environment and project harness in place. The output still included malformed boxed answers and a Llama completion that exhausted all 32768 tokens in a repetition loop.

The known-good control is job 87302, not merely "vLLM V0":

| Component | Job 87302 value |
|-----------|-----------------|
| Model | Qwen-7B BF16 |
| Harness | QRM authors' `inference.py` + Lighteval custom MATH-500 task |
| Env | `qrm-official` |
| Versions | torch 2.5.1+cu124; transformers 4.47.1; vLLM 0.7.0+precompiled; Lighteval 0.8.0; datasets 3.3.1; compressed-tensors 0.9.0 |
| Sampling | seed 42; temperature 0.6; top-p 0.95; max new tokens 32768; no repetition penalty |
| Engine | BF16; max model length 32768; eager mode; prefix caching off; chunked prefill off; GPU memory utilization 0.75 |
| Result | 10/10 extractive match; 10/10 contained `\\boxed`; no observed truncation/loop |

Jobs 96093/96094 used that entire execution path with n=10 and replaced only the model path with the Qwen-7B FP8 or Llama-8B FP8 checkpoint. Both completed 10/10 correct and boxed, with no token-cap hits or repetition flags. Qwen token counts were 1,111-12,729 (mean 4,393.1); Llama counts were 1,163-8,638 (mean 3,580.8). Their archive is `outputs-hpc-qrm-official-fp8-validation-2026-08-13`.

One output-handling bug was exposed: both concurrent jobs copied their model-specific result to the same top-level filename. The model-specific files under `inference/` remained intact, but the convenience alias was overwritten by the later Qwen copy. `run_official_inference.sh` now includes the model basename in copied filenames, and `validate_official_results.py` records/gates output health. Use `submit_qrm_fp8_full.sh`; it refuses to call `sbatch` until both pilots pass strict validation.

---

## Failure 1 — Job 87130: broken env marker, missing submodules

### Symptom

```
ModuleNotFoundError: No module named 'fast_hadamard_transform'
```

Install script wrote `.qrm_official_env_ready` even though imports were broken.

### Root cause

`scripts/hpc/qrm_parity/install_official_qrm_env.sh` was incomplete:

1. Did not `git submodule update` for `fast-hadamard-transform`, `lighteval`, `vllm`
2. Editable vLLM install failed → fell back to pip `vllm==0.8.5` (wrong; QRM needs **v0.7.0** fork)
3. Marker written without import verification
4. Env polluted with torch 2.6.0 / transformers 4.57.6 from bad fallback

### Fix

- Submodule init in `setup_official_qrm_repo.sh` (login node) and `install_official_qrm_env.sh`
- Versioned install marker `INSTALL_REV` — reinstall when rev changes
- Import verification before writing marker
- `pip uninstall vllm torchvision torchaudio xformers` after requirements install
- Delete stale marker: `rm -f .qrm_official_env_ready`

### Verify

```bash
grep INSTALL_REV scripts/hpc/qrm_parity/install_official_qrm_env.sh
test -f external/Quantized-Reasoning-Models/third-party/fast-hadamard-transform/pyproject.toml
```

---

## Failure 2 — Jobs 87179–87182: compute-node toolchain gaps

### Symptoms

| Job | Error |
|-----|-------|
| 87179/87180 | `git: command not found` on compute nodes |
| 87181 | Git check failed after conda activate |
| 87182 | `nvcc not found` on `ragpu006` |

### Root cause

PARAM Rudra GPU nodes have **NVIDIA drivers only** — no system CUDA toolkit, stripped PATH, and often no working system `g++`/`cc1plus`. Login-node installs did not account for compute-node compile environment.

### Fixes applied (incremental)

1. `export PATH="/usr/bin:/bin:${PATH}"` in install + inference scripts
2. Skip `git submodule` on compute if submodules already present on shared scratch
3. Conda `cuda-nvcc=12.4` + `cuda-cudart-dev=12.4` when `nvcc` missing

---

## Failure 3 — Job 87187: C++ compiler missing during CUDA build

### Symptom

```
c++: command not found
gcc: error trying to exec 'cc1plus': execvp: No such file or directory
nvcc fatal: Failed to preprocess host compiler properties.
```

Conda `nvcc` install **succeeded** (`Using nvcc: .../qrm-official/bin/nvcc`), but host C++ compile failed.

### Root cause

Compute nodes lack a complete system GCC toolchain. PyTorch extension build needs both `nvcc` and a working `g++`.

### Fix

Same pattern as `qreason` Triton fix (`scripts/hpc/param_rudra_env.sh`):

```bash
conda install -y -c conda-forge gcc_linux-64=12 gxx_linux-64=12 sysroot_linux-64
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
```

**Important:** pin **gcc 12**, not latest (15.x) — see Failure 4.

---

## Failure 4 — Login-node compile: gcc 15 + CUDA header mismatch

### Symptoms (during manual install on login node)

```
#error -- unsupported GNU version! gcc versions later than 13 are not supported!
fatal error: nv/target: No such file or directory
fatal error: cusparse.h: No such file or directory
```

### Root cause

1. Default `conda-forge` pulled **gcc 15.2** — incompatible with **nvcc 12.4** (supports gcc ≤ 13)
2. Mixed CUDA versions: `cuda-cccl` 12.9 vs `cuda-nvcc` 12.4
3. PyTorch headers reference `cusparse.h` in pip `nvidia/*/include`, not in conda `$CONDA_PREFIX/include` alone

### Fix

```bash
conda install -y -c conda-forge -c nvidia \
  gcc_linux-64=12 gxx_linux-64=12 sysroot_linux-64 \
  cuda-nvcc=12.4 cuda-cudart-dev=12.4 cuda-cudart=12.4 cuda-cccl=12.4
```

Add PyTorch/nvidia pip include paths before building `fast-hadamard-transform`:

```bash
# Function qrm_export_cuda_build_env() in install_official_qrm_env.sh
# Sets CPATH / CPLUS_INCLUDE_PATH to $CONDA_PREFIX/include + site-packages/nvidia/*/include
```

### Verify (login node)

```bash
rm -f .qrm_official_env_ready
bash scripts/hpc/qrm_parity/install_official_qrm_env.sh
# Expect: Successfully installed fast_hadamard_transform
```

---

## Failure 5 — `set -u` breaks conda compiler activation

### Symptom

```
activate-gcc_linux-64.sh: line 114: SYS_SYSROOT: unbound variable
deactivate-gxx_linux-64.sh: line 68: CONDA_BACKUP_CXX: unbound variable
```

Job exits immediately after `conda activate qrm-official`.

### Root cause

`set -euo pipefail` in install/inference scripts conflicts with conda gcc package activate/deactivate hooks.

### Fix

Use `set -eo pipefail` (drop `-u`) in:

- `scripts/hpc/qrm_parity/install_official_qrm_env.sh`
- `scripts/hpc/qrm_parity/run_official_inference.sh`
- `slurm/qrm_official_math500_n10.slurm`

---

## Failure 6 — Jobs 87193 / 87196: wrong vLLM precompiled binary

### Symptom (GPU node)

```
ImportError: .../vllm/_C.abi3.so: undefined symbol: _ZN3c106ivalue14ConstantString6createE...
```

Login-node import check passed with `No platform detected` — **misleading** because `vllm._C` is only loaded when CUDA platform is detected.

### Root cause

`VLLM_USE_PRECOMPILED=1` without `VLLM_PRECOMPILED_WHEEL_LOCATION` defaults to:

```
https://wheels.vllm.ai/nightly/vllm-1.0.0.dev-...whl
```

That nightly `_C.abi3.so` (~407 MB) is ABI-incompatible with torch 2.5.1. Correct vLLM 0.7.0 wheel is ~264 MB.

### Fix

```bash
export VLLM_PRECOMPILED_WHEEL_LOCATION="https://files.pythonhosted.org/packages/51/70/6fc00dca2e9f53a76b7792d788cb2efbb9d2587ed0ca9a71d5ccf7fc7543/vllm-0.7.0-cp38-abi3-manylinux1_x86_64.whl"
rm -f external/Quantized-Reasoning-Models/third-party/vllm/vllm/_C.abi3.so  # before reinstall
```

Install script now verifies `vllm._C` and fails on `undefined symbol`; defers only if `libcuda.so.1` missing (login node).

### Verify

```bash
ls -la external/Quantized-Reasoning-Models/third-party/vllm/vllm/_C.abi3.so
# Good: ~215 MB, not ~407 MB
conda activate qrm-official
python -c "import vllm; print(vllm.__version__)"  # 0.7.0
```

On GPU job log, expect: `Automatically detected platform cuda.` with **no** ImportError.

---

## Failure 7 — Job 87213: GitPython needs git on compute nodes

### Symptom

```
ImportError: Failed to initialize: Bad git executable.
```

Occurs when lighteval imports `git` during inference (after vLLM CUDA load succeeded).

### Root cause

`GitPython` requires a real `git` binary. Compute nodes do not have system git on PATH after conda activate, even with `PATH=/usr/bin:/bin`.

### Fix

```bash
conda install -y -c conda-forge git   # in qrm-official env
```

In `run_official_inference.sh`:

```bash
export GIT_PYTHON_GIT_EXECUTABLE="$CONDA_PREFIX/bin/git"
```

Also added `git` to the conda install line in `install_official_qrm_env.sh`.

---

## Failure 8 — Job 87213: shared GPU CUDA OOM (not an env bug)

### Symptom

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB.
GPU 0 has a total capacity of 79.14 GiB of which 23.19 MiB is free.
Process 2235457 has 77.68 GiB memory in use.
```

Env install and vLLM import **succeeded**. Model weight allocation failed because another user's process held ~78 GB on the same A100.

### Root cause

SLURM `--gres=gpu:1` without `--exclusive` can land on a **shared** GPU (same physical card as another job).

### Fix

Final working approach:

1. Keep the job non-exclusive with `--gres=gpu:1` and `--cpus-per-task=16` so it does not consume both GPUs under the user quota.
2. Lower `gpu_memory_utilization` to `0.75` and require about 62 GB free VRAM before vLLM starts.
3. If a GPU is dirty, update `ExcNodeList` and requeue the job so SLURM can place it on a cleaner GPU.

This replaced the earlier exclusive-GPU idea, which avoided dirty GPUs but could sit behind QOS/resource limits on 2-GPU nodes.


---

## Install marker revisions (history)

| `INSTALL_REV` | What changed |
|---------------|--------------|
| *(none / stale)* | Job 87130 — marker written without verification |
| `2026-07-06-conda-nvcc-vllm07` | Conda nvcc only |
| `2026-07-06-conda-gcc-nvcc-vllm07` | Added conda gcc/g++ (pulled gcc 15 — still broken) |
| `2026-07-06-conda-gcc12-nvcc124-vllm07` | Pinned gcc 12, aligned CUDA 12.4, CPATH fix |
| **`2026-07-06-conda-gcc12-nvcc124-vllm070wheel`** | **Current** — correct PyPI vLLM 0.7.0 wheel URL |

Force reinstall:

```bash
rm -f /scratch/$USER/reasoning-compression-lab/.qrm_official_env_ready
bash scripts/hpc/qrm_parity/install_official_qrm_env.sh
```

---

## Final working configuration (2026-07-06)

### Conda env

- **Name:** `qrm-official` (python 3.11)
- **torch:** 2.5.1 (from QRM `requirements.txt`)
- **vLLM:** 0.7.0+precompiled (editable fork + PyPI wheel binaries)
- **Compilers:** gcc/g++ 12 via conda, nvcc 12.4 via conda
- **git:** conda-forge (for lighteval GitPython)

### Key files changed

| File | Purpose |
|------|---------|
| `scripts/hpc/qrm_parity/install_official_qrm_env.sh` | Full stack install + verification |
| `scripts/hpc/qrm_parity/setup_official_qrm_repo.sh` | Submodule init on login node |
| `scripts/hpc/qrm_parity/run_official_inference.sh` | git, compilers, GPU preflight |
| `slurm/qrm_official_math500_n10.slurm` | Non-exclusive `--gres=gpu:1`, `--cpus-per-task=16`, `set -eo pipefail` |

### Git sync status

The QRM official fixes and follow-up docs are synced to GitHub. As of 2026-08-13, GitHub/HPC include the FP8 KV-cache fix (`542f622`); MacBook should pull latest `origin/main`. Leave `.qrm_official_env_ready` untracked.

---

## Quick diagnostic checklist

Run on **login node** after any env change:

```bash
cd /scratch/$USER/reasoning-compression-lab
cat .qrm_official_env_ready   # must match latest INSTALL_REV
source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh
conda activate qrm-official

python -c "import fast_hadamard_transform; print('hadamard ok')"
python -c "import vllm; print(vllm.__version__)"
x86_64-conda-linux-gnu-gcc --version   # expect 12.x
command -v nvcc git
ls -la external/Quantized-Reasoning-Models/third-party/vllm/vllm/_C.abi3.so
```

On **GPU job** (from logs), confirm this sequence:

1. `QRM official env already installed` (or full install completes without compile errors)
2. `Automatically detected platform cuda.`
3. No `undefined symbol` on `vllm._C`
4. No `Bad git executable`
5. GPU preflight shows ≥ 40000 MiB free (or exclusive node is clean)
6. Model loads and inference progresses

---

## Common errors → action table

| Error snippet | Likely cause | Action |
|---------------|--------------|--------|
| `No module named 'fast_hadamard_transform'` | Submodule / compile not done | `setup_official_qrm_repo.sh`; delete marker; reinstall |
| `c++: command not found` / `cc1plus` | No host compiler on compute | Ensure gcc_linux-64=12 in env; check CC/CXX exports |
| `unsupported GNU version! gcc ... later than 13` | gcc 14/15 installed | `conda install gcc_linux-64=12 gxx_linux-64=12` |
| `nv/target: No such file` | CUDA header version mismatch | Pin `cuda-cccl=12.4` with `cuda-nvcc=12.4` |
| `cusparse.h: No such file` | Missing nvidia pip includes | Run `qrm_export_cuda_build_env` before pip build |
| `SYS_SYSROOT: unbound variable` | `set -u` + conda gcc | Use `set -eo pipefail` only |
| `undefined symbol` on `vllm._C` | Wrong precompiled wheel | Set `VLLM_PRECOMPILED_WHEEL_LOCATION` to PyPI 0.7.0 URL; rm stale `_C.abi3.so` |
| `Bad git executable` | No git on compute | `conda install git`; set `GIT_PYTHON_GIT_EXECUTABLE` |
| `CUDA out of memory` + other PID using ~78GB | Shared GPU | Use `--exclusive`; preflight; resubmit |
| `PENDING (Resources)` with `--exclusive` | Queue busy | Wait; normal for exclusive 1-GPU jobs |

---

## Operational commands

```bash
# Submit official test
bash scripts/hpc/submit_qrm_official_test.sh

# Monitor
squeue -u $USER
tail -f logs/qrm_official_<JOBID>.out

# After success
python scripts/hpc/qrm_parity/compare_side_by_side.py --limit 10
```

Output directory: `outputs-hpc-qrm-official-<date>/`

---

## Lessons for future sessions

1. **Never trust the env marker without import verification** — job 87130 wrote the marker with a broken stack.
2. **Login-node success ≠ GPU-node success** — vLLM defers `_C` load without CUDA driver; always confirm on a GPU job.
3. **PARAM Rudra compute nodes are not build workstations** — ship compilers (conda gcc 12 + nvcc 12.4) in the env, do not rely on `/usr/bin/gcc`.
4. **Pin versions explicitly** — unpinned conda-forge gcc and nvidia cuda-meta packages drift to incompatible combos.
5. **VLLM_USE_PRECOMPILED default wheel is v1.0 nightly** — always set `VLLM_PRECOMPILED_WHEEL_LOCATION` for QRM v0.7.0.
6. **Shared GPU OOM looks like a model bug** — check `nvidia-smi` for foreign processes before debugging vLLM memory flags.
7. **Separate envs:** never mix `qreason` (vLLM 0.8.5) with `qrm-official` (vLLM 0.7.0 fork).

---

## Current status

| Item | Value |
|------|-------|
| Env install | **Verified** on login node (marker `2026-07-06-conda-gcc12-nvcc124-vllm070wheel`) |
| GPU inference | **Validated end-to-end** by job **87302** |
| Official result | Qwen-7B BF16, MATH-500 n=10, seed 42: **10/10 correct**, **0 truncation** |
| Science conclusion | Prompt/protocol are correct; modern `qreason` stack behavior differs from QRM official stack |
| Follow-up | Modern-stack b02 jobs **96086/96087** stopped; exact-stack FP8 n=10 jobs **96093/96094** passed 10/10; gated full run is next |
| GitHub sync | Complete at `319cc56`; `.qrm_official_env_ready` remains untracked |
| Calibration | b02 is not a calibration run; use it for pass@1/truncation/cost only |
