# lm-evaluation-harness sanity cross-check (optional)

One-time validation that our harness pass@1 is in the same ballpark as EleutherAI lm-eval.

**Reference clone:** `../external_repos/04-inference-and-eval-tools/lm-evaluation-harness/`

## When to run

After the first matched Qwen-7B BF16 cell on HPC *(that cell exists; this cross-check was optional and is not required for the frozen paper)*. It remains a sanity cross-check, not a replacement for the frozen publication harness. Do not install it into or mutate the controlled environment. Paper 1 GPU work is closed (2026-08-17).

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
