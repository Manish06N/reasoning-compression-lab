# GPTQ-4 Preparation Gate

**Do not run** `level_a_gptq4_seed0.json` until a GPTQ-4 model path exists and is verified.

## Correct order

```text
1. BF16 smoke test          (scripts/hpc/03_smoke_test.sh)
2. BF16 MATH-500 seed 0     (scripts/hpc/04_run_level_a_bf16.sh)
3. Prepare GPTQ-4 weights   (this document)
4. Verify model path        (scripts/hpc/06_verify_gptq4_model.sh)
5. GPTQ-4 MATH-500 seed 0   (level_a_gptq4_seed0.json)
```

---

## Option A — Download pre-quantized (recommended first try)

Quantized-Reasoning-Models releases real GPTQ W4G128 models on HuggingFace:

https://huggingface.co/collections/ruikangliu/deepseek-r1-distill-quantized-68357b2a87b1a76137ad20d0

On HPC (automated):

```bash
export QR=/scratch/$USER/reasoning-compression-lab
bash scripts/hpc/08_download_gptq4_models.sh both
```

Manual download:

(Use the exact HF repo ID from the model card — override with `QWEN7B_GPTQ4_REPO` if name varies.)

```bash
export QREASON_MODEL_QWEN7B_GPTQ4=$QR/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4
bash scripts/hpc/06_verify_gptq4_model.sh
```

---

## Option B — Quantize yourself on HPC

Use reference repo (read-only):

`external_repos/01-core-baselines/Quantized-Reasoning-Models/scripts/real_quantization/gptq.sh`

Or install tooling:

```bash
pip install auto-gptq
# or GPTQModel — version-sensitive, test on one model first
```

Requires BF16 base model + reasoning calibration data (`gen_calib.sh` in reference repo).

Output path must match:

```bash
export QREASON_MODEL_QWEN7B_GPTQ4=$QR/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4
```

---

## Option C — Try AWQ first (if GPTQ blocked)

Sometimes AWQ checkpoints are easier to obtain/run via vLLM:

1. Download AWQ W4 from same HF collection
2. Create a new cell config (copy `level_a_gptq4_seed0.json` → `level_a_awq4_seed0.json`)
3. Run BF16 vs AWQ-4 before debugging GPTQ install issues

---

## Config wiring

`configs/cells/level_a_gptq4_seed0.json` reads model path from:

| Source | Variable |
|--------|----------|
| Environment | `QREASON_MODEL_QWEN7B_GPTQ4` |
| Default | `models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4` |

Inference command (only after verify passes):

```bash
python scripts/run_inference.py --cell-config configs/cells/level_a_gptq4_seed0.json
python scripts/score_run.py --input runs/raw/level_a_qwen7b_gptq4_math500_seed0.jsonl
```
