# Item-level descriptive analysis (CPU, frozen JSON)

No GPU. No causal claims. Counts are associations under the pinned A100 / vLLM 0.7.0 evaluation.

## Error flips (BF16 correct, quantized wrong)

### math500

| Cell | item×seed BF16✓ quant✗ | any-item flips | all-seed flips | both✓ | quant-only✓ | both✗ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen-7B_FP8 | 53 | 38 | 0 | 2297 | 63 | 87 |
| Qwen-7B_AWQ-4 | 78 | 53 | 0 | 2272 | 56 | 94 |
| Qwen-7B_GPTQ-4 | 66 | 49 | 0 | 2284 | 53 | 97 |
| Llama-8B_FP8 | 110 | 87 | 0 | 2121 | 117 | 152 |
| Llama-8B_AWQ-4 | 179 | 121 | 0 | 2052 | 110 | 159 |
| Llama-8B_GPTQ-4 | 134 | 99 | 0 | 2097 | 126 | 143 |

### gsm8k

| Cell | item×seed BF16✓ quant✗ | any-item flips | all-seed flips | both✓ | quant-only✓ | both✗ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen-7B_FP8 | 83 | 72 | 2 | 3528 | 86 | 260 |
| Qwen-7B_AWQ-4 | 105 | 90 | 1 | 3506 | 97 | 249 |
| Qwen-7B_GPTQ-4 | 104 | 87 | 2 | 3507 | 99 | 247 |
| Llama-8B_FP8 | 127 | 113 | 0 | 3382 | 132 | 316 |
| Llama-8B_AWQ-4 | 215 | 166 | 5 | 3294 | 153 | 295 |
| Llama-8B_GPTQ-4 | 136 | 119 | 0 | 3373 | 147 | 301 |

### gpqa_diamond

| Cell | item×seed BF16✓ quant✗ | any-item flips | all-seed flips | both✓ | quant-only✓ | both✗ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen-7B_FP8 | 76 | 61 | 0 | 223 | 71 | 224 |
| Qwen-7B_AWQ-4 | 98 | 75 | 0 | 201 | 65 | 230 |
| Qwen-7B_GPTQ-4 | 87 | 68 | 2 | 212 | 73 | 222 |
| Llama-8B_FP8 | 73 | 65 | 1 | 201 | 83 | 237 |
| Llama-8B_AWQ-4 | 87 | 74 | 0 | 187 | 92 | 228 |
| Llama-8B_GPTQ-4 | 91 | 72 | 1 | 183 | 84 | 236 |

## Length versus correctness (MATH-500)

| Cell | n correct | mean tokens correct | n incorrect | mean tokens incorrect |
|---|---:|---:|---:|---:|
| Qwen-7B_BF16 | 2350 | 3256.2 | 150 | 15842.7 |
| Qwen-7B_FP8 | 2360 | 3327.8 | 140 | 15468.5 |
| Qwen-7B_AWQ-4 | 2328 | 3368.1 | 172 | 16409.4 |
| Qwen-7B_GPTQ-4 | 2337 | 3392.6 | 163 | 17114.2 |
| Llama-8B_BF16 | 2231 | 3538.6 | 269 | 13928.5 |
| Llama-8B_FP8 | 2238 | 3558.6 | 262 | 13026.3 |
| Llama-8B_AWQ-4 | 2162 | 3564.9 | 338 | 12230.3 |
| Llama-8B_GPTQ-4 | 2223 | 3568.3 | 277 | 15050.5 |

## GPQA-Diamond item-level (row index only)

Row indices are campaign order, not published GPQA prompts. The Qwen AWQ GPQA result is significant within the primary Holm-6 family, but not under the Holm-18 joint sensitivity analysis.

Qwen AWQ flip-seed histogram: `{'0': 123, '1': 52, '2': 23}`

| Row | BF16✓ AWQ✗ seeds (of 3) | Qwen BF16 correct seeds | Qwen AWQ correct seeds |
|---:|---:|---:|---:|
| 7 | 2 | 2 | 1 |
| 10 | 2 | 2 | 0 |
| 17 | 2 | 3 | 1 |
| 41 | 2 | 2 | 1 |
| 49 | 2 | 2 | 0 |
| 61 | 2 | 2 | 1 |
| 73 | 2 | 3 | 1 |
| 79 | 2 | 2 | 0 |
| 92 | 2 | 2 | 0 |
| 98 | 2 | 3 | 1 |
| 105 | 2 | 2 | 0 |
| 111 | 2 | 2 | 0 |
| 114 | 2 | 2 | 0 |
| 120 | 2 | 3 | 1 |
| 138 | 2 | 2 | 0 |

