# lm-evaluation-harness sanity cross-check (optional)

One-time validation that our harness pass@1 is in the same ballpark as EleutherAI lm-eval.

**Reference clone:** `../external_repos/04-inference-and-eval-tools/lm-evaluation-harness/`

## When to run

After the first clean Qwen-7B BF16 cell on HPC — not required for publication pipeline.

## HPC

```bash
pip install 'lm_eval[vllm]'   # optional; do not pin over vLLM 0.8.5 without testing

export MODEL=$QR/models/DeepSeek-R1-Distill-Qwen-7B
bash scripts/lmeval_sanity_check.sh "$MODEL" 10
```

## Compare

```bash
python scripts/lmeval_compare_summary.py \
  --lmeval-dir runs/lmeval_sanity \
  --summary results/level_a_qwen7b_bf16_math500_seed0_summary.json \
  --task gsm8k
```

**Default stance:** Do not rebuild Paper 1 inside lm-eval. Use only for sanity checks.
