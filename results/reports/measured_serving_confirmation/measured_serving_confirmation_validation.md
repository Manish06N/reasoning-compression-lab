# Measured serving confirmation — validation notes

Independent of the campaign 56k accuracy grid. Preferred serving evidence for the manuscript.

## Dispersion conventions

- Report $\pm$ and `std_tokens_per_second` use **sample SD** (`statistics.stdev`, divisor $n-1$).
- The HPC runner expands $R=3\to 5$ when **population** CV of the first three tok/s values exceeds 3% (`np.std` without `ddof`, divisor $n$).
- Llama-8B FP8 Condition A has sample CV $3.67\%$ but population CV $3.00\%$, which is **not** $> 3.0$, so $R=3$ is correct under the frozen rule.
- Do not drop slow repeats after expansion. All technically valid repeats enter the reported mean/SD.

## Hardware

- All Qwen-7B confirmation files: host `ragpu003`, `torch` `gpu_count=1`.
- All Llama-8B confirmation files: host `ragpu004`, `torch` `gpu_count=1`.
- `nvidia_smi` lists both node UUIDs; the used-device UUID is **not** isolated.
- Valid wording: **within-architecture same-node (one visible A100) controlled serving comparisons.** Cross-architecture absolute tok/s is cautious.
- Do **not** claim a specific UUID or “all eight on one GPU.”

## Memory fields

`model_weights_memory_gb` is `torch.cuda.memory_allocated()` after `LLM()` init. Values $\sim 54$–$56$ GB track the engine pool at `gpu_memory_utilization=0.75`, not an isolated weight footprint. Do **not** publish a format-to-format weight table from these fields.

## Truncation / generation cap

Compact confirmation JSON has no `finish_reason`. Max mean output tokens per request is $\ll 32768$. Allowed wording: no serving-confirmation **mean** output reached the 32,768-token generation cap. The 56k campaign still has 25 loops / 209 near-cap rows.

## Cost-of-Pass

Scenario: hybrid $C_{\mathrm{pass}} = (\mathrm{GPU\text{-}sec}/q \times 1.50/3600) / $ campaign MATH-500 pass@1. Label **hybrid scenario Cost-of-Pass**. Intervals combine timing-rep resampling with problem-clustered pass@1 bootstrap. Do not score the 100 serving prompts unless extractive match is in those JSON files (it is not).

## Replicate tok/s (task-realistic)

| Cell | Cond | R | mean tok/s | sample SD | sample CV% | pop SD | pop CV% (first 3) | trigger? | expanded? | host(s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen-7B BF16 | A | 3 | 43.92 | 0.04 | 0.09 | 0.03 | 0.07 | N | N | ragpu003 |
| Qwen-7B BF16 | B | 3 | 252.72 | 0.56 | 0.22 | 0.45 | 0.18 | N | N | ragpu003 |
| Qwen-7B FP8 | A | 3 | 62.50 | 0.08 | 0.13 | 0.07 | 0.11 | N | N | ragpu003 |
| Qwen-7B FP8 | B | 5 | 449.79 | 97.53 | 21.68 | 87.23 | 12.84 | Y | Y | ragpu003 |
| Qwen-7B AWQ-4 | A | 3 | 76.89 | 0.70 | 0.91 | 0.57 | 0.75 | N | N | ragpu003 |
| Qwen-7B AWQ-4 | B | 3 | 418.15 | 2.03 | 0.49 | 1.66 | 0.40 | N | N | ragpu003 |
| Qwen-7B GPTQ-4 | A | 3 | 82.67 | 0.69 | 0.84 | 0.57 | 0.68 | N | N | ragpu003 |
| Qwen-7B GPTQ-4 | B | 3 | 488.26 | 0.64 | 0.13 | 0.52 | 0.11 | N | N | ragpu003 |
| Llama-8B BF16 | A | 3 | 72.34 | 0.27 | 0.38 | 0.22 | 0.31 | N | N | ragpu004 |
| Llama-8B BF16 | B | 3 | 366.98 | 1.90 | 0.52 | 1.55 | 0.42 | N | N | ragpu004 |
| Llama-8B FP8 | A | 3 | 78.75 | 2.89 | 3.67 | 2.36 | 3.00 | N | N | ragpu004 |
| Llama-8B FP8 | B | 3 | 481.42 | 1.36 | 0.28 | 1.11 | 0.23 | N | N | ragpu004 |
| Llama-8B AWQ-4 | A | 5 | 70.20 | 2.71 | 3.86 | 2.42 | 4.13 | Y | Y | ragpu004 |
| Llama-8B AWQ-4 | B | 3 | 391.11 | 0.48 | 0.12 | 0.40 | 0.10 | N | N | ragpu004 |
| Llama-8B GPTQ-4 | A | 3 | 63.49 | 0.41 | 0.65 | 0.34 | 0.53 | N | N | ragpu004 |
| Llama-8B GPTQ-4 | B | 3 | 366.16 | 0.79 | 0.22 | 0.65 | 0.18 | N | N | ragpu004 |

## Qwen-7B FP8 Condition B (retain all five)

Not a formatting bug. Generated token counts are identical; wall-clock elapsed time is not.

| Rep | host | elapsed s | output tokens | tok/s | gpu-sec/q |
|---|---|---|---|---|---|
| 1 | ragpu003 | 1065.221 | 373794 | 350.9076 | 10.6522 |
| 2 | ragpu003 | 1065.914 | 373794 | 350.6792 | 10.6591 |
| 3 | ragpu003 | 819.900 | 373794 | 455.9020 | 8.1990 |
| 4 | ragpu003 | 685.633 | 373794 | 545.1811 | 6.8563 |
| 5 | ragpu003 | 684.274 | 373794 | 546.2634 | 6.8427 |

First-three population CV exceeds 3%, so $R=5$ triggered. Averaging all five makes Qwen FP8 Condition-B Cost-of-Pass more conservative than dropping the slow repeats. Report $449.8 \pm 97.5$ tok/s (sample SD) and do not claim a tight FP8 throughput.

## Overturned first-run claim

The earlier unconstrained serving run (`results/measured_serving/`) is **not** averaged with this confirmation. Confirmation is preferred evidence. The first-run claim that all four 4-bit configurations were slower than matched BF16 under single-stream is **not supported** here: Qwen AWQ/GPTQ exceed Qwen BF16 Condition-A tok/s; Llama AWQ/GPTQ do not.

