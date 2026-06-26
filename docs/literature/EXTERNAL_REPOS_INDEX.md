# External Reference Repositories — Index

Read-only reference code for Paper 1. **Do not develop experiments inside these folders.**

Your experiment harness: `../reasoning-compression-lab/`

Comprehensive guide: [EXTERNAL_REPOS_REFERENCE.md](EXTERNAL_REPOS_REFERENCE.md)

---

## Folder layout

```
external_repos/
├── 01-core-baselines/              Paper-matched evaluation baselines
│   ├── Quantized-Reasoning-Models/ Quantized reasoning models (COLM 2025)
│   └── sober-reasoning/            Multi-seed reproducibility (COLM 2025)
│
├── 02-calibration-and-cost/        Confidence + economic metrics
│   ├── Calibrating-LLMs-with-Consistency/
│   └── Cost-of-Pass/
│
├── 03-quantization-implementations/ Primary method source code
│   ├── gptq/                       IST-DASLab/gptq (GPTQ official)
│   └── smoothquant/                mit-han-lab/smoothquant (W8A8 INT8)
│
├── 04-inference-and-eval-tools/    Serving + benchmark harnesses
│   ├── vllm/                       vLLM inference engine
│   └── lm-evaluation-harness/      Standardized LM benchmarks
│
└── 05-selective-prediction/        Abstention (Phase 2+)
    └── AbstentionBench/
```

---

## When to use which folder

| Stage | Folder | Action |
|-------|--------|--------|
| Level A reproduction | `01-core-baselines/` | Match accuracy protocol |
| Seed variance | `01-core-baselines/sober-reasoning/` | Copy prompt + seed grid pattern |
| Calibration pilot | `02-calibration-and-cost/` | Port Brier/ECE/consistency math |
| Cost-per-correct | `02-calibration-and-cost/Cost-of-Pass/` | Port CoP formula |
| GPTQ debugging | `03-quantization-implementations/gptq/` | Read only if quant fails |
| FP8/W8A8 background | `03-quantization-implementations/smoothquant/` | Related work + method details |
| vLLM issues | `04-inference-and-eval-tools/vllm/` | Read docs/examples only |
| Benchmark standardization | `04-inference-and-eval-tools/lm-evaluation-harness/` | Optional later |
| Selective risk extension | `05-selective-prediction/` | After Level A passes |

---

## Install vs clone

On HPC, **install** these (do not run experiments from cloned tool repos):

```bash
pip install vllm transformers datasets math-verify pynvml
pip install auto-gptq autoawq   # if needed for GPTQ/AWQ
```

Clone folders in `03-*` and `04-*` are for **reading reference implementations**, not for daily execution.

---

## Clone later (not present yet)

```bash
cd "/Users/manish/Projects/2026/paper 1/external_repos"
# mkdir -p 06-later && cd 06-later
# git clone https://github.com/livecodebench/livecodebench.git
# git clone https://github.com/OckBench/OckBench.git
```

Add to this index when cloned. Do not recreate the folder structure.
