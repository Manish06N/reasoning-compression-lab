# Beyond Pass@1: Reliability–Cost Frontiers of Quantized Reasoning Models under Controlled Serving-Stack Shift

**Manish Nandish**  
*Department of Computer Science & Engineering, Indian Institute of Technology Patna*  
*PARAM Rudra HPC (NSM / C-DAC)*  
Email: manishn_iitp@iitp.ac.in

**Canonical source for arXiv:** [`paper/main.tex`](main.tex) compiled to [`paper/main.pdf`](main.pdf). This markdown file tracks the same text.

**Keywords:** Reasoning language models, post-training quantization, sample-consistency calibration, Cost-of-Pass, selective prediction, serving-stack control.

---

## Abstract

Post-training quantization is standard practice for reducing the memory and latency footprints of large language models. Most compression studies report single-seed pass@1 accuracy on short-form tasks. Reasoning models generate long deliberation traces, so compression can also change token length, termination behavior, sample consistency, and serving cost. We evaluate the joint reliability–calibration–cost frontier of quantized reasoning models under one pinned serving stack (`vLLM 0.7.0`, eager execution, NVIDIA A100-80GB). The study comprises **56,408** full-length completions across **88** experimental cells: MATH-500 ($n=500$, seeds 42–46; 20,000 completions), GSM8K ($n=1{,}319$, seeds 42–44; 31,656 completions), and GPQA-Diamond ($n=198$, seeds 42–44; 4,752 completions), on DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B in BF16, FP8 (A100 Marlin W8A16 fallback), AWQ-4, and GPTQ-4.

Findings: (1) On MATH-500, FP8 matches BF16 within seed noise (Qwen $94.40\% \pm 1.05\%$ vs $94.00\% \pm 0.55\%$; Llama $89.52\% \pm 1.01\%$ vs $89.24\% \pm 0.74\%$). Paired McNemar tests on *maj@5* consensus, not on pass@1 means, find no significant discordance after Holm–Bonferroni correction ($p > 0.29$). Llama AWQ-4 still drops $2.76$ percentage points on mean pass@1. (2) Mean output-token inflation on MATH-500 is $+6.3\%$ to $+6.8\%$ for Qwen 4-bit and $+1.7\%$ to $+3.9\%$ for Llama 4-bit versus BF16. On a 200-item mixed-correctness audit subset the same 4-bit deltas are $+19.8\%$ to $+29.8\%$. (3) Heuristic detectors (token count at the $32{,}768$ cap; consecutive identical-word runs) flag 0 truncations and 0 loops across all 88 cells. Official QRM rows do not store vLLM `finish_reason`. (4) Sample-consistency ECE is the fraction of gold-correct seeds versus maj@5 correctness, not a per-token model confidence. (5) Cost-of-Pass is modeled at $\$1.50$ per A100-hour and $65$ tok/s, not measured wall-clock. Under that model, FP8 is Pareto-optimal. (6) Unanimous ($5/5$) filtering raises Qwen-7B FP8 selective accuracy to $98.23\%$ at $90.4\%$ coverage. Artifacts are released with this preprint.

---

## 1. Introduction

Reasoning language models such as DeepSeek-R1 produce long intermediate traces before a final answer. That generation pattern raises GPU memory and latency, so post-training quantization (FP8, AWQ-4, GPTQ-4) is a common serving default. Isolated pass@1 numbers hide whether compression also changes sample consistency, token length, or expected cost per correct answer. Serving-stack differences (engine version, CUDA graphs, decoding flags) can dominate the quantization effect.

We isolate weight format under one pinned stack and ask:

- **RQ1.** Does quantization change multi-seed pass@1 relative to matched BF16?
- **RQ2.** Do heuristic truncation and repetition detectors fire more often under compression?
- **RQ3.** How does quantization change sample-consistency ECE, Brier score, AURC, and selective risk?
- **RQ4.** Does token inflation change modeled Cost-of-Pass ($C_{\mathrm{pass}}$) at a fixed cloud price and assumed throughput?

---

## 2. Related Work

**Post-training quantization.** GPTQ, AWQ, and SmoothQuant are standard PTQ methods, largely validated on short-form benchmarks.

**Quantized reasoning.** Liu et al. (arXiv:2504.04823, *Quantization Hurts Reasoning?*) evaluate DeepSeek-R1 distillations and report that W8A8/W4A16 can be nearly lossless, while lower bit-widths risk accuracy. They also report that quantized models do *not* systematically lengthen outputs. Lian et al. (arXiv:2606.25519) argue the opposite: low-bit PTQ can inflate chain-of-thought length even when the final answer is correct. Alimaskina et al. (arXiv:2606.02011) study process-level failures under extreme 2-bit inference. Our contribution is a matched BF16/FP8/AWQ-4/GPTQ-4 grid on one pinned vLLM 0.7.0 eager stack. On this stack we *do* observe 4-bit token inflation (Section 4.4).

**Evaluation sensitivity.** Hochlehnert et al. (arXiv:2504.07086) show that reasoning scores move with seeds, prompts, decoding, and software. We freeze engine, prompt, and decoding, and vary only the checkpoint format.

**Calibration and cost.** Guo et al. define ECE for predicted probabilities. Self-consistency uses agreement across samples as confidence. Erol et al. (arXiv:2504.13359) define Cost-of-Pass as expected monetary cost per correct solution. We apply those ideas with the method caveats in Section 3.4.

---

## 3. Experimental Methodology

### 3.1 Models and precision formats

DeepSeek-R1-Distill-Qwen-7B (Qwen2.5, 151k vocab) and DeepSeek-R1-Distill-Llama-8B (Llama-3.1, 128k vocab). Formats: uncompressed **BF16**; **FP8** checkpoints executed on A100 via vLLM's Marlin **W8A16** weight-only fallback (not native W8A8 tensor-core FP8); **AWQ-4** (group size 128, `torch.float16`); **GPTQ-4** (group size 128, Marlin).

### 3.2 Benchmarks

- **MATH-500** ($n=500$, 5 seeds 42–46): 20,000 completions. Headline grid.
- **GSM8K** ($n=1{,}319$, 3 seeds 42–44): 31,656 completions. Breadth.
- **GPQA-Diamond** ($n=198$, 3 seeds 42–44): 4,752 completions. Breadth.
- Decoding: zero-shot `<think>` prefix, $T=0.6$, top-$p=0.95$, repetition penalty $1.0$, max new tokens $32{,}768$.

Total: $2 \times 4 \times (5\times 500 + 3\times 1319 + 3\times 198) = 56{,}408$ completions in 88 cells.

### 3.3 Hardware and serving stack

PARAM Rudra HPC, NVIDIA A100-PCIE-80GB. Engine pinned to `vLLM==0.7.0`, `--enforce-eager`, `--gpu-memory-utilization 0.75`. Official QRM inference path plus HPC patches in the artifact repo.

### 3.4 Metrics

**Pass@1 and maj@5.** Pass@1 is mean single-completion extractive accuracy. Majority vote is correct when at least 3 of 5 MATH-500 seeds match gold.

**Pathology detectors.** Length hit if encoded completion length $\ge 32{,}768$. Loop if the longest consecutive identical-word run exceeds a threshold. Heuristics. Official-QRM rows do not include vLLM `finish_reason`.

**Sample-consistency ECE.** Let $c_i$ be the number of MATH-500 seeds whose extracted answer matches gold, $K=5$. Confidence $\hat{c}_i = c_i/K$. Label $z_i = \mathbb{I}(c_i \ge 3)$ (maj@5). This is *not* agreement with the modal answer and not a softmax confidence. Because $\hat{c}_i$ and $z_i$ are both functions of $c_i$, ECE is expected to be small.

**McNemar.** Exact tests on maj@5 correctness, Holm–Bonferroni $\alpha=0.05$. They do not test equality of pass@1 means.

**Modeled Cost-of-Pass.** $C_{\mathrm{pass}} = (\text{cost per query}) / \text{pass@1}$ with $\tau = 65$ tok/s and $R=\$1.50$ per A100-hour. Latency columns are $\bar{T}/65$, not measured wall-clock.

---

## 4. Results

### 4.1 MATH-500 accuracy (Table 1)

FP8 is within seed noise of BF16. Qwen 4-bit stays above $93.1\%$. Llama AWQ-4 is the weak cell ($86.48\%$, $-2.76$ pp vs BF16). Heuristic detectors flag 0/0 on all 40 MATH-500 cells.

| Model & Format | 42 | 43 | 44 | 45 | 46 | Mean ± Std | 95% Wilson | Mean tok. | T/L |
|---|---|---|---|---|---|---|---|---|---|
| Qwen-7B BF16 | 94.4% | 94.0% | 93.8% | 94.6% | 93.2% | **94.00% ± 0.55%** | [93.0, 94.9] | 4,011.4 | 0/0 |
| Qwen-7B FP8 | 94.4% | 95.2% | 94.8% | 92.6% | 95.0% | **94.40% ± 1.05%** | [93.4, 95.2] | 4,007.7 | 0/0 |
| Qwen-7B AWQ-4 | 92.4% | 92.8% | 93.2% | 93.0% | 94.2% | **93.12% ± 0.67%** | [92.1, 94.0] | 4,265.3 | 0/0 |
| Qwen-7B GPTQ-4 | 93.8% | 92.6% | 93.4% | 94.6% | 93.0% | **93.48% ± 0.77%** | [92.4, 94.4] | 4,287.2 | 0/0 |
| Llama-8B BF16 | 89.0% | 88.4% | 90.2% | 89.8% | 88.8% | **89.24% ± 0.74%** | [88.0, 90.4] | 4,656.6 | 0/0 |
| Llama-8B FP8 | 89.0% | 89.6% | 88.6% | 89.2% | 91.2% | **89.52% ± 1.01%** | [88.3, 90.7] | 4,550.8 | 0/0 |
| Llama-8B AWQ-4 | 84.4% | 84.8% | 89.2% | 87.4% | 86.6% | **86.48% ± 1.96%** | [85.1, 87.8] | 4,736.5 | 0/0 |
| Llama-8B GPTQ-4 | 88.0% | 89.6% | 86.8% | 89.4% | 90.8% | **88.92% ± 1.55%** | [87.6, 90.1] | 4,840.5 | 0/0 |

### 4.2 maj@5 McNemar (Table 2)

After Holm–Bonferroni, no quantized format differs from BF16 on maj@5 ($p>0.29$). That is narrower than “quantization does not change accuracy.” Llama AWQ-4 remains non-significant on maj@5 despite the $2.76$ pp pass@1 drop.

| Contrast | n11 | n10 | n01 | n00 | Exact p | Holm |
|---|---|---|---|---|---|---|
| Llama AWQ-4 | 436 | 20 | 13 | 31 | 0.2962 | n.s. (α=0.0083) |
| Qwen FP8 | 467 | 5 | 8 | 20 | 0.5811 | n.s. (α=0.0100) |
| Qwen GPTQ-4 | 466 | 6 | 4 | 24 | 0.7539 | n.s. (α=0.0125) |
| Qwen AWQ-4 | 463 | 9 | 9 | 19 | 1.0000 | n.s. (α=0.0167) |
| Llama FP8 | 445 | 11 | 10 | 34 | 1.0000 | n.s. (α=0.0250) |
| Llama GPTQ-4 | 444 | 12 | 12 | 32 | 1.0000 | n.s. (α=0.0500) |

### 4.3 Sample-consistency ECE (Table 3)

Confidence $= c_i/5$ (gold-correct seed fraction); label = maj@5. Not Guo-style model calibration. Qwen ECE $\le 0.034$; Llama ECE $\le 0.072$. FP8 tracks BF16.

| Model & Format | maj@5 Acc. | ECE | Brier | AURC |
|---|---|---|---|---|
| Qwen-7B BF16 | 94.40% | 0.0264 | 0.0082 | 0.0016 |
| Qwen-7B FP8 | 95.00% | 0.0284 | 0.0094 | 0.0013 |
| Qwen-7B AWQ-4 | 94.40% | 0.0344 | 0.0112 | 0.0016 |
| Qwen-7B GPTQ-4 | 94.00% | 0.0300 | 0.0090 | 0.0018 |
| Llama-8B BF16 | 91.20% | 0.0572 | 0.0172 | 0.0040 |
| Llama-8B FP8 | 91.00% | 0.0492 | 0.0150 | 0.0042 |
| Llama-8B AWQ-4 | 89.80% | 0.0724 | 0.0220 | 0.0054 |
| Llama-8B GPTQ-4 | 91.20% | 0.0612 | 0.0191 | 0.0040 |

### 4.4 Token inflation and modeled $C_{\mathrm{pass}}$ (Table 4)

Est. latency is $\bar{T}/65$, not measured wall-clock. Qwen AWQ-4/GPTQ-4: $+6.3\%$/$+6.8\%$ tokens vs BF16; Llama GPTQ-4 $+3.9\%$, Llama AWQ-4 $+1.7\%$, Llama FP8 $-2.3\%$. Under the modeled $\$1.50$/h and $65$ tok/s assumption, FP8 is the lowest $C_{\mathrm{pass}}$.

| Model & Format | Mean tok. | Est. lat. (s) | Cost / Q | $C_{\mathrm{pass}}$ |
|---|---|---|---|---|
| Qwen-7B BF16 | 4,011.4 | 61.71 | $0.02571 | $0.02736 |
| Qwen-7B FP8 | 4,007.7 | 61.66 | $0.02569 | **$0.02721** |
| Qwen-7B AWQ-4 | 4,265.3 | 65.62 | $0.02734 | $0.02936 |
| Qwen-7B GPTQ-4 | 4,287.2 | 65.96 | $0.02748 | $0.02940 |
| Llama-8B BF16 | 4,656.6 | 71.64 | $0.02985 | $0.03345 |
| Llama-8B FP8 | 4,550.8 | 70.01 | $0.02917 | **$0.03259** |
| Llama-8B AWQ-4 | 4,736.5 | 72.87 | $0.03036 | $0.03511 |
| Llama-8B GPTQ-4 | 4,840.5 | 74.47 | $0.03103 | $0.03490 |

### 4.5 Trace audit (Table 5)

$n=200$ stratified MATH-500 items. Token deltas are versus BF16 on this subset, not the full 500-item mean. Subset inflation ($+10\%$ to $+30\%$) is larger than the full-grid mean ($+1.7\%$ to $+6.8\%$).

| Metric | Qwen-7B | Llama-8B |
|---|---|---|
| All 4 formats correct | 180 (90.0%) | 157 (78.5%) |
| All 4 formats fail | 6 (3.0%) | 9 (4.5%) |
| Mixed correctness | 14 (7.0%) | 34 (17.0%) |
| FP8 token Δ vs BF16 | +11.41% | +10.21% |
| AWQ-4 token Δ vs BF16 | +19.81% | +29.77% |
| GPTQ-4 token Δ vs BF16 | +21.14% | +28.45% |

### 4.6 GSM8K and GPQA-Diamond (Table 6)

MATH-500 uses 5 seeds; GSM8K and GPQA-Diamond use 3 seeds. FP8 stays within about one point of BF16. 4-bit degradation is small on GSM8K and larger on GPQA-Diamond. All 88 cells remain 0/0 on the heuristic detectors.

| Model & Format | MATH-500 | GSM8K | GPQA-D | T/L |
|---|---|---|---|---|
| Qwen-7B BF16 | 94.00±0.55 | 91.26±0.29 | 50.34±2.96 | 0/0 |
| Qwen-7B FP8 | 94.40±1.05 | 91.33±0.16 | 49.49±1.52 | 0/0 |
| Qwen-7B AWQ-4 | 93.12±0.67 | 91.05±1.14 | 44.78±3.04 | 0/0 |
| Qwen-7B GPTQ-4 | 93.48±0.77 | 91.13±0.27 | 47.98±1.75 | 0/0 |
| Llama-8B BF16 | 89.24±0.74 | 88.68±0.46 | 46.13±1.91 | 0/0 |
| Llama-8B FP8 | 89.52±1.01 | 88.80±0.62 | 47.81±0.29 | 0/0 |
| Llama-8B AWQ-4 | 86.48±1.96 | 87.11±0.23 | 46.97±2.02 | 0/0 |
| Llama-8B GPTQ-4 | 88.92±1.55 | 88.96±0.70 | 44.95±4.32 | 0/0 |

### 4.7 Difficulty (Table 7)

Level 1 is essentially immune. Level 5 Qwen FP8 is $93.13\%$ vs BF16 $92.69\%$. Llama AWQ-4 is weakest on Levels 2–4.

| Level | Qwen BF16 | Qwen FP8 | Qwen AWQ-4 | Qwen GPTQ-4 |
|---|---|---|---|---|
| 1 ($n=43$) | $98.60\pm1.27$ | $98.60\pm2.08$ | $98.14\pm1.95$ | $96.28\pm2.08$ |
| 2 ($n=90$) | $89.11\pm0.93$ | $90.44\pm4.20$ | $90.67\pm1.69$ | $88.67\pm2.14$ |
| 3 ($n=105$) | $95.81\pm1.09$ | $95.24\pm1.51$ | $94.10\pm0.80$ | $94.29\pm1.90$ |
| 4 ($n=128$) | $95.78\pm0.89$ | $96.41\pm1.42$ | $94.53\pm1.24$ | $96.56\pm1.18$ |
| 5 ($n=134$) | $92.69\pm1.11$ | $93.13\pm1.23$ | $91.04\pm1.67$ | $92.24\pm1.95$ |

| Level | Llama BF16 | Llama FP8 | Llama AWQ-4 | Llama GPTQ-4 |
|---|---|---|---|---|
| 1 ($n=43$) | $93.95\pm3.12$ | $94.88\pm3.03$ | $95.35\pm1.64$ | $96.28\pm2.65$ |
| 2 ($n=90$) | $84.67\pm2.14$ | $85.11\pm2.02$ | $81.11\pm3.04$ | $83.33\pm3.42$ |
| 3 ($n=105$) | $89.90\pm2.48$ | $90.10\pm1.73$ | $85.71\pm1.17$ | $89.71\pm1.83$ |
| 4 ($n=128$) | $91.56\pm1.69$ | $91.09\pm0.43$ | $87.03\pm2.25$ | $89.84\pm1.75$ |
| 5 ($n=134$) | $88.06\pm1.75$ | $88.81\pm2.84$ | $87.31\pm3.46$ | $88.81\pm1.18$ |

### 4.8 Selective prediction (Table 8)

Requiring $5/5$ agreement raises Qwen-7B FP8 to $98.23\%$ selective accuracy at $90.4\%$ coverage (risk $1.77\%$). Llama coverage at $5/5$ is $73$–$82\%$.

| Model | $k/5$ | Coverage | Sel. acc. | Sel. risk |
|---|---|---|---|---|
| Qwen BF16 | 3/5 | 100.0% | 94.40% | 5.60% |
| | 4/5 | 96.4% | 96.47% | 3.53% |
| | 5/5 | 90.4% | 97.57% | 2.43% |
| Qwen FP8 | 3/5 | 100.0% | 95.00% | 5.00% |
| | 4/5 | 95.4% | 97.06% | 2.94% |
| | 5/5 | 90.4% | **98.23%** | **1.77%** |
| Qwen AWQ-4 | 3/5 | 100.0% | 94.40% | 5.60% |
| | 4/5 | 94.6% | 95.98% | 4.02% |
| | 5/5 | 88.2% | 97.73% | 2.27% |
| Qwen GPTQ-4 | 3/5 | 100.0% | 94.00% | 6.00% |
| | 4/5 | 96.2% | 96.05% | 3.95% |
| | 5/5 | 88.8% | 97.75% | 2.25% |
| Llama BF16 | 3/5 | 100.0% | 91.20% | 8.80% |
| | 4/5 | 92.8% | 93.97% | 6.03% |
| | 5/5 | 78.6% | 96.69% | 3.31% |
| Llama FP8 | 3/5 | 100.0% | 91.00% | 9.00% |
| | 4/5 | 93.6% | 93.80% | 6.20% |
| | 5/5 | 81.8% | 95.60% | 4.40% |
| Llama AWQ-4 | 3/5 | 100.0% | 89.80% | 10.20% |
| | 4/5 | 90.6% | 91.61% | 8.39% |
| | 5/5 | 73.2% | 95.90% | 4.10% |
| Llama GPTQ-4 | 3/5 | 100.0% | 91.20% | 8.80% |
| | 4/5 | 91.4% | 94.31% | 5.69% |
| | 5/5 | 78.0% | 96.41% | 3.59% |

### 4.9 Figures

- Figure 1: `paper_figures/figure1_pareto_frontier.pdf` — modeled reliability–cost frontier.
- Figure 2: `paper_figures/figure2_token_inflation.pdf` — full-grid mean tokens.
- Figure 3: `paper_figures/figure3_calibration_reliability.pdf` — sample-consistency ECE/Brier.
- Figure 4: `paper_figures/figure4_seed_variance.pdf` — pass@1 by seed.

---

## 5. Discussion

FP8 on this A100 W8A16 stack is a practical default: it matches BF16 accuracy, does not inflate mean tokens, and wins the modeled $C_{\mathrm{pass}}$ ranking. 4-bit is architecture-dependent. Qwen retains MATH-500 accuracy with modest mean-token growth; Llama AWQ-4 is the cell that moves. That pattern is consistent with Liu et al. on accuracy risk at 4-bit, and with Lian et al. on hidden token cost, with the scope split between full-grid means and the hard-subset audit.

The serving stack is part of the result. Earlier vLLM 0.8.5 runs on the same checkpoints produced truncation and loops; pinning official QRM 0.7.0 eager execution removed those failures under our detectors. We claim a *stack-controlled* quantization comparison, not a universal property of FP8 or AWQ.

---

## 6. Limitations

1. **A100 W8A16 fallback.** FP8 is not native W8A8 compute.
2. **Modeled $C_{\mathrm{pass}}$.** Throughput is assumed $65$ tok/s. No per-request wall-clock/VRAM/energy on every cell.
3. **Pathology heuristics.** Zero truncations/loops means the token-cap and identical-word detectors did not fire. `finish_reason` is not in the saved schema.
4. **Sample-consistency ECE** is a function of seed-correct counts, not model-internal confidence.
5. **Breadth seeds.** GSM8K and GPQA-Diamond use three seeds, not five.
6. **Stack specificity.** vLLM 0.7.0 eager mode only.
7. **McNemar vs means.** maj@5 parity does not imply equal pass@1 (Llama AWQ-4).

---

## 7. Conclusion

Under a pinned vLLM 0.7.0 eager stack on A100, FP8 matches BF16 on 56,408 completions across 88 cells, 4-bit token inflation is real but format- and subset-dependent, and modeled Cost-of-Pass favors FP8 at a fixed $\$1.50$/h and $65$ tok/s assumption. Sample-consistency filtering is an operational safety gate. The honest reading is a stack-controlled reliability–cost map, not a claim of native FP8 speed or Guo-style calibration.

---

## Artifacts

https://github.com/Manish06N/reasoning-compression-lab

Per-cell records: `results/math500/`, `results/gsm8k/`, `results/gpqa/`. Summaries: `results/README.md`, `results/reports/`. Packaging: [`paper/ARTIFACT.md`](ARTIFACT.md). ArXiv zip: [`paper/arxiv_source.zip`](arxiv_source.zip).

## Acknowledgment

PARAM Rudra HPC at IIT Patna (National Supercomputing Mission, C-DAC) provided compute.
