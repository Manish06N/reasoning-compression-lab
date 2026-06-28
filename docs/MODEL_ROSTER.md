# Model Roster — Paper 1 Canonical Checkpoints

Single source of truth for Hugging Face IDs, local paths, env vars, and machine assignment.

**Stack:** vLLM 0.8.5, `enforce_eager=true`, chat template enabled.

## Machine roles

| Machine | Role |
|---------|------|
| **RTX 5080 (16 GB)** | **Short jobs only** — Qwen-1.5B × 4 quants × MATH-500 (~≤24 h/cell) |
| **HPC 2× A100 (80 GB)** | **Main grid** — all 7B/8B quants, BF16 anchors, GSM8K, GPQA (≤48 h SLURM blocks) |

**Decoding (paper runs):**

| Config | Use | max_tokens | batch |
|--------|-----|------------|-------|
| `configs/decoding/repro_qrm.yaml` | **5080 + HPC publication** | 32768 | 1 |
| `configs/decoding/pilot_5080.yaml` | Optional 5080 debug pilot | 8192 | 4/2/1 |

---

## Tier 1 — Must have (Level A)

| Role | HF ID | Local folder | Env var | vLLM config | 5080 | HPC |
|------|-------|--------------|---------|-------------|------|-----|
| Anchor BF16 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `models/DeepSeek-R1-Distill-Qwen-7B` | `QREASON_MODEL_QWEN7B` | `deepseek_r1_qwen_7b.json` | Smoke only | **Full** |
| Anchor GPTQ-4 | `ruikangliu/DeepSeek-R1-Distill-Qwen-7B-GPTQ-W4G128` | `models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4` | `QREASON_MODEL_QWEN7B_GPTQ4` | `deepseek_r1_qwen_7b_gptq4.json` | Yes | Yes |
| Pipeline verifier | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `models/DeepSeek-R1-Distill-Qwen-1.5B` | `QREASON_MODEL_QWEN15B` | `deepseek_r1_qwen_15b.json` | Yes | Yes |

**5080 smoke cell for 7B BF16:** `configs/cells/smoke_qwen7b_bf16_5080.json` — uses `max_model_len=2048`, `--limit 1`, `--max-tokens 64`.

---

## Tier 2 — Level B (Qwen-7B × 5 quants)

| Quant | HF ID | Local folder | Env var | Model JSON | 5080 | HPC |
|-------|-------|--------------|---------|------------|------|-----|
| BF16 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `models/DeepSeek-R1-Distill-Qwen-7B` | `QREASON_MODEL_QWEN7B` | `deepseek_r1_qwen_7b.json` | Smoke | Full |
| FP8 | `RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic` | `models/DeepSeek-R1-Distill-Qwen-7B-FP8` | `QREASON_MODEL_QWEN7B_FP8` | `deepseek_r1_qwen_7b_fp8.json` | Yes | Yes |
| AWQ-4 | `jakiAJK/DeepSeek-R1-Distill-Qwen-7B_AWQ` | `models/DeepSeek-R1-Distill-Qwen-7B-AWQ-4` | `QREASON_MODEL_QWEN7B_AWQ4` | `deepseek_r1_qwen_7b_awq4.json` | Yes | Yes |
| GPTQ-4 | `ruikangliu/DeepSeek-R1-Distill-Qwen-7B-GPTQ-W4G128` | `models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4` | `QREASON_MODEL_QWEN7B_GPTQ4` | `deepseek_r1_qwen_7b_gptq4.json` | Yes | Yes |
| GPTQ-3 | `irish-quant/deepseek-ai-DeepSeek-R1-Distill-Qwen-7B-3bit` | `models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-3` | `QREASON_MODEL_QWEN7B_GPTQ3` | `deepseek_r1_qwen_7b_gptq3.json` | Optional | Optional |

Cell templates: `configs/cells/level_b_qwen7b_*_math500_seed0.json`

---

## Tier 3 — Level C (3 models)

| Model | BF16 (HPC) | Best 5080 quant | Model JSONs |
|-------|------------|-----------------|-------------|
| Qwen-1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Same (always fits) | `deepseek_r1_qwen_15b.json` |
| Qwen-7B | See Tier 2 | FP8 or GPTQ-4 | See Tier 2 |
| Llama-8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | FP8 / AWQ-4 / GPTQ-4 only | `deepseek_r1_llama_8b*.json` |

**Do not run Llama-8B BF16 on 5080** — weights alone exceed 16 GB.

---

## Tasks

| Task | Dataset | Gated? | Config |
|------|---------|--------|--------|
| MATH-500 | `HuggingFaceH4/MATH-500` | No | `configs/tasks/math500.json` |
| GSM8K | `openai/gsm8k` | No | `configs/tasks/gsm8k.json` |
| GPQA-Diamond | `Idavidrein/gpqa` | **Yes** | `configs/tasks/gpqa_diamond.json` |

Request GPQA access: https://huggingface.co/datasets/Idavidrein/gpqa — see `docs/GPQA_ACCESS.md`.

---

## Download commands

```bash
source scripts/local/env.sh   # or HPC param_rudra env

# Phase 0
bash scripts/local/download_models.sh phase0

# Phase 2 (after Level A BF16 on HPC)
bash scripts/local/download_models.sh gptq4
```

---

## Reject for paper runs

- GGUF / llama.cpp checkpoints (wrong stack)
- Uncensored community forks
- BF16 full MATH-500 on 5080 for 7B/8B
- Mixing two GPTQ-4 HF repos within one quant label
