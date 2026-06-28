# GPQA-Diamond — Gated Dataset Access

GPQA is required for Level B/C runs on `configs/tasks/gpqa_diamond.json` but is **gated** on Hugging Face.

## Steps

1. Log in at https://huggingface.co
2. Open https://huggingface.co/datasets/Idavidrein/gpqa
3. Accept the dataset terms (request access if prompted)
4. Ensure `HF_TOKEN` in `.env` belongs to the approved account
5. Test:

```bash
source scripts/local/env.sh
python -c "from datasets import load_dataset; ds=load_dataset('Idavidrein/gpqa','gpqa_diamond',split='train'); print(len(ds))"
```

## Cell config

After access is granted, run:

```bash
python scripts/run_inference.py --cell-config configs/cells/level_c_qwen7b_fp8_gpqa_seed0.json
```

Until access is approved, skip GPQA cells and use MATH-500 + GSM8K only.
