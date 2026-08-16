# Runtime Evidence & Environment Provenance Report

**Generated:** 2026-08-16  
**Cluster:** PARAM Rudra HPC (IIT Patna / NSM / C-DAC)  
**Host (Current Validation Session):** `login01`  
**Target Repository:** `/scratch/manishn_iitp/reasoning-compression-lab`  

---

## 1. Runtime Variable Classification

Each runtime setting for the August campaign is classified below as:
* **`RUNTIME-OBSERVED`**: Directly verified from historical stdout/stderr logs (`logs/qrm_official_*.out`, `logs/qrm_official_*.err`) or SLURM/node telemetry.
* **`CODE/LAUNCHER-CONFIRMED`**: Present in frozen launcher scripts (`scripts/hpc/qrm_parity/run_official_inference.sh`) or upstream QRM `inference.py`. These are the values the 56k jobs were submitted with, but they are not per-job runtime dumps.
* **`DEFAULT/RECONSTRUCTED`**: Inherited from a library default, or inferred when no explicit log line exists.
* **`UNKNOWN`**: Cannot currently be established from existing artifacts.

| Runtime Variable | Classification | Value / Detail | Evidence Source |
|---|---|---|---|
| **vLLM Version** | `RUNTIME-OBSERVED` | `0.7.0` | `logs/qrm_official_96408.out` preflight check line 58 (`[PASS] vllm 0.7.0`) |
| **PyTorch Version** | `RUNTIME-OBSERVED` | `2.5.1+cu124` | `logs/qrm_official_96408.out` preflight check line 52 (`[PASS] torch 2.5.1+cu124`) |
| **Transformers Version** | `RUNTIME-OBSERVED` | `4.47.1` | `logs/qrm_official_96408.out` preflight check line 53 (`[PASS] transformers 4.47.1`) |
| **lighteval Version** | `RUNTIME-OBSERVED` | `0.8.0` | `logs/qrm_official_96408.out` preflight check line 55 (`[PASS] lighteval 0.8.0`) |
| **CUDA Toolchain** | `RUNTIME-OBSERVED` | `12.4` / GCC 12 | `logs/qrm_official_96408.out` preflight check lines 47, 49 (`nvcc 12.4`, `gcc 12`) |
| **GPU Model** | `RUNTIME-OBSERVED` | NVIDIA A100-PCIE-80GB | SLURM nodelist assignment (`ragpu003`–`ragpu008`, `racn115`–`racn116`) |
| **NVIDIA Driver** | `RUNTIME-OBSERVED` | `535.183.01` | Cluster compute node driver version |
| **Checkpoint IDs** | `RUNTIME-OBSERVED` | `DeepSeek-R1-Distill-Qwen-7B`, `DeepSeek-R1-Distill-Llama-8B` | Preflight log line 41 & validation JSON headers |
| **max_model_len** | `CODE/LAUNCHER-CONFIRMED` | `32768` | `VLLMModelConfig` argument in `inference.py` line 128 |
| **max_tokens (max_new_tokens)** | `CODE/LAUNCHER-CONFIRMED` | `32768` | `GenerationParameters` in `inference.py` line 122 |
| **gpu_memory_utilization** | `CODE/LAUNCHER-CONFIRMED` | `0.75` | `--gpu_memory_utilization 0.75` in `run_official_inference.sh` line 98 |
| **enforce_eager** | `RUNTIME-OBSERVED` | `True` | Log stderr message: `Since, enforce-eager is enabled, async output processor cannot be used` |
| **kv_cache_dtype** | `DEFAULT/RECONSTRUCTED` | `auto` (vLLM default) | `inference.py` does not pass `kv_cache_dtype`; no per-cell log proves the resolved dtype |
| **dtype (BF16 / FP8)** | `CODE/LAUNCHER-CONFIRMED` | `bfloat16` | Passed to vLLM engine for uncompressed and FP8 models |
| **dtype (AWQ-4 / GPTQ-4)** | `CODE/LAUNCHER-CONFIRMED` | `float16` | Required for Marlin & AWQ CUDA kernels in `inference.py` lines 49–50 |
| **temperature** | `CODE/LAUNCHER-CONFIRMED` | `0.6` | `GenerationParameters` in `inference.py` line 119 |
| **top_p** | `CODE/LAUNCHER-CONFIRMED` | `0.95` | `GenerationParameters` in `inference.py` line 120 |
| **repetition_penalty** | `DEFAULT/RECONSTRUCTED` | `1.0` | Default vLLM parameter (no repetition penalty applied; not an explicit logged override) |
| **tensor_parallel_size** | `CODE/LAUNCHER-CONFIRMED` | `1` | Single-GPU execution (`--gres=gpu:1`) per task cell |

---

## 2. Current Validation Login Node Environment

* **Date:** 2026-08-16T21:53:32+05:30
* **Hostname:** `login01`
* **Python Executable (Conda):** Python 3.11.15 (`/home/manishn_iitp/.conda/envs/qreason/bin/python3`)
* **Note:** The login node does not host active GPU devices (compute processes run strictly on assigned SLURM GPU compute nodes).
