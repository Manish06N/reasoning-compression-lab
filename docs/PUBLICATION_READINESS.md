# Publication Readiness

**Superseding decision (2026-08-18):** Canonical submission manuscript is on `paper-major-revision`. Science is frozen at `d707e44`. The matched 88-run grid is complete (**56,408** completions). Canonical manuscript: [`paper/main.tex`](../paper/main.tex) compiled to [`paper/main.pdf`](../paper/main.pdf). Scoreboard: [`results/README.md`](../results/README.md). Frozen tables: [`results/reports/major_revision_tables.md`](../results/reports/major_revision_tables.md). ArXiv zip: [`paper/arxiv_source.zip`](../paper/arxiv_source.zip) (current `main.tex` + `references.bib` + `main.bbl`). **Experimental GPU work is closed.** Immutable snapshot tag: `paper-v1.0-submission`.

The 2026-08-17, 2026-08-16 and 2026-08-14 banners below are **historical**. They do **not** describe the current evidence.

The 2026-08-16 and 2026-08-14 banners below are **historical**. They do **not** describe the current evidence.

**Current label:** P0-corrected preprint plus gold-free MATH-500 modal agreement plus controlled serving confirmation plus major-revision CPU analyses (mismatch excess $D$, bootstrap tail-area $p$, Holm-18 sensitivity, hybrid $C_{\mathrm{pass}}$). Canonical numbers: `results/reports/revision_reanalysis_report.json`, `results/reports/modal_agreement_report.json`, `results/reports/measured_serving_confirmation/measured_serving_confirmation_report.json`. Do not cite 0/0 pathologies, the 98.23% gold-hit gate, “true Pareto optimum,” FP8 “parity,” or unqualified “Qwen AWQ-4 is Holm-significant on GPQA.”

---

# Publication Readiness (2026-08-16 banner — historical)

**Historical superseding decision (2026-08-16):** The matched 88-cell grid is complete (**56,408** completions: MATH-500 40 + GSM8K 24 + GPQA-Diamond 24). Then-current PDF was ~14 pages. That banner is kept for chronology.

---

# Publication Readiness Audit — 2026-08-14 (historical)

**Historical decision:** **Needs revision — do not submit the (then) two-result package as a paper.**

**Then-current evidence label:** validated single-seed FP8 replication/control evidence.

**Canonical execution plan (historical):** [plans/2026-08-14-publication-recovery.md](plans/2026-08-14-publication-recovery.md)

This 2026-08-14 audit was the controlling scientific interpretation *until the matched 88-cell grid finished*. Keep it as chronological evidence. Any statement that jobs 96100/96101 are running, that seed 0 alone is a publishable core, or that the two-job FP8 result establishes a quantization effect remains superseded — first by this audit, then by the 2026-08-16 banner, and now by the 2026-08-17 freeze at the top of this file.

---

## 1. Decision in one paragraph

Jobs **96100** and **96101** completed successfully and produced credible MATH-500 results. They validate the pinned QRM execution path and reproduce existing FP8 model-card accuracy. They do **not** isolate a quantization effect because there is no matched BF16 run on the same stack, they use one seed and one task, and they contain no valid calibration, selective-risk, controlled systems, power, or cost telemetry. The results belong in an appendix, replication table, or later matched analysis—not as a standalone paper result.

---

## 2. Audit scope

The 2026-08-14 audit covered:

- All **190 Markdown files** visible across the project and pinned local dependencies: 58 project/model documents and 132 vendored upstream documents.
- Job state, Slurm accounting, stdout/stderr, configuration, model metadata, environment versions, and result hashes.
- All 1,000 generated rows, prompt/gold alignment, scoring fields, answer extraction, length distribution, boxing, and repetition/termination behavior.
- Reproducibility from a clean checkout and the distinction between tracked and uncommitted external-QRM patches.
- Current literature through 2026-08-14 for quantized reasoning, calibration, reliability, cost-per-correct, token inflation, and degeneration.

Generated literature text extracts and archived historical guides were reviewed but are not manually rewritten; their immutability is part of provenance.

---

## 3. Exact completed experiment

| Dimension | Audited setting |
|-----------|-----------------|
| Jobs | 96100 Qwen; 96101 Llama; both `COMPLETED`, exit code 0 |
| Models | Qwen-7B FP8 artifact commit `ceb2fcd178d477cd21b92ffb43164c74165a212c`; Llama-8B FP8 artifact commit `5d548d9169c53b7bb7ef0b3bc261509d5da6e3dd` |
| Runtime quantization | FP8-compressed checkpoints on A100 using vLLM's **weight-only Marlin fallback**; describe as FP8-checkpoint/W8A16-style execution, not native FP8 W8A8 compute |
| Dataset | `HuggingFaceH4/MATH-500`, revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, 500 rows, fingerprint `f3124375297911ba` |
| Prompt | DeepSeek chat template, no system prompt, zero-shot, step-by-step instruction, answer requested in `\\boxed{}`, assistant opened with `<think>` |
| Sampling | One completion per item; temperature 0.6; top-p 0.95; max new tokens 32,768; sampling seed 42; repetition penalty 1.0; other penalties/default filters unchanged |
| Engine | `qrm-official`; vLLM 0.7.0; eager mode; tensor/pipeline/data parallel 1; prefix caching off; chunked prefill off; GPU memory utilization 0.75 |
| Software | Python 3.11; PyTorch 2.5.1+cu124; Transformers 4.47.1; LightEval 0.8.0; Datasets 3.3.1; compressed-tensors 0.9.0 |
| Hardware | One A100 80 GB and 16 CPUs per nonexclusive job; Qwen on `ragpu008`, Llama on `ragpu004` |
| Elapsed | Qwen 22m12s; Llama 40m28s |
| Result archive | `outputs-hpc-qrm-official-fp8-full-2026-08-13` |

The Llama job logged more than 900 KV-cache recomputations/preemptions at 0.75 GPU-memory utilization. The two elapsed times therefore are not a controlled latency or throughput comparison.

---

## 4. Independently verified results

| Model | Correct | pass@1 | 95% Wilson interval | Existing FP8 model-card value | Boxed |
|-------|--------:|-------:|--------------------:|------------------------------:|------:|
| Qwen-7B | 472/500 | **94.4%** | 92.03–96.10% | 93.62% | 498/500 |
| Llama-8B | 445/500 | **89.0%** | 85.95–91.45% | 90.24% | 496/500 |

Both observed accuracies are compatible with the existing FP8 model-card values. The paired Qwen/Llama comparison is 440 both correct, 32 Qwen-only correct, 5 Llama-only correct, and 23 both wrong. The +5.4 percentage-point Qwen advantage is statistically clear (exact McNemar `p = 7.43e-6`) but is an architecture/model comparison, **not** a quantization comparison.

### Trace-quality findings

- Six traces are likely near-cap/length terminations based on abrupt endings and 32,568–32,725 generated tokens: Qwen rows 418 and 453; Llama rows 52, 200, 226, and 443.
- The saved schema lacks `finish_reason` and output token IDs, so those six cases cannot be proven as length termination after the fact.
- Llama row 443 contains a severe same-token loop; row 52 repeats a sentence many times. Phrase/sentence loops are underdetected by the current consecutive-identical-word heuristic.
- Wrong answers are strongly associated with long generations, but the current design cannot attribute that association to quantization without matched BF16 controls.
- Re-scoring final boxed answers found no substantive metric inflation; the LightEval extraction is acceptable for this result.

### Validator interpretation

The full-run validator defaults allow minimum accuracy 0, minimum boxed rate 0, and up to all 500 rows to hit the cap or repetition detector. Therefore `passed: true` means structural completion, not scientific quality. Future reports must separate:

- `integrity_pass`: schema, row count, provenance, prompt/gold alignment, numeric metrics.
- `quality_warnings`: cap endings, phrase loops, parse failures, pathological length, and unexpected accuracy.
- `publication_gate`: matched design, required seeds/metrics, clean reproducibility, and completed audit.

---

## 5. Claims boundary

### Supported now

- The two FP8 checkpoints are healthy on the pinned QRM stack.
- The run reproduces existing FP8 MATH-500 accuracy within sampling uncertainty.
- The modern `qreason` execution path and pinned QRM path can produce materially different trace behavior.
- Long/degenerate traces are an important measurement target for the next experiment.

### Not supported now

- FP8 preserves or harms accuracy relative to BF16 under a matched protocol.
- Native FP8 speed, energy, or W8A8 benefits on A100.
- FP8 improves latency, throughput, VRAM efficiency, energy, or cost-per-correct.
- Calibration, Brier score, ECE, AURC, selective risk, or majority-vote claims.
- Seed stability or generalization beyond MATH-500.
- A causal serving-stack conclusion: the old/new paths currently differ in several factors simultaneously.
- Submission readiness or a complete “Beyond Accuracy” contribution.

---

## 6. Reproducibility blockers

1. The external QRM checkout is pinned at `bf947e29f52e3f666e3263efac149dae0ac18d00`, but contains required uncommitted patches.
2. Those patches add GPU-memory control, Python 3.11-compatible syntax, and local dataset loading; the tracked setup scripts do not recreate all of them.
3. The output schema omits finish reason, output token IDs, per-request timing, confidence source, and power/VRAM telemetry.
4. The two protocol families are confounded: QRM reproduction uses seed 42 and the old stack, while the former main grid uses seed 0, a different prompt/profile, repetition penalty, and newer stack.
5. Several live documents previously described completed jobs as running and disagreed about whether one, three, or five seeds were sufficient.

No future paper number may be marked reproducible until a clean checkout recreates the environment and all required patches deterministically.

---

## 7. Novelty decision

The broad combination “quantized reasoning + calibration + cost” is no longer sufficient by itself:

- [Quantization Hurts Reasoning?](https://arxiv.org/abs/2504.04823) already provides a systematic quantized-reasoning baseline across models, tasks, formats, and seeds.
- [A Sober Look at Progress in Language Model Reasoning](https://arxiv.org/abs/2504.07086) establishes the importance of prompt, decoding, seed, hardware, and software sensitivity.
- [Quantized LLMs Can Still Be Calibrated](https://aclanthology.org/2025.acl-long.1473/) covers calibration degradation and recovery under quantization.
- [Cost-of-Pass](https://arxiv.org/abs/2504.13359) formalizes expected cost per correct solution.
- [Reliability Scaling Laws for Quantized LLMs](https://arxiv.org/abs/2607.10855) expands the reliability/efficiency landscape across bit widths and methods.
- [Quantization Inflates Reasoning](https://arxiv.org/abs/2606.25519) directly targets token inflation as a hidden quantization cost.

**Recommended differentiator:** a paired, reasoning-specific **reliability–cost frontier under quantization and controlled serving-stack shift**, with termination/degeneration evidence and released trace-level provenance.

The project must run the discriminating pilot in the recovery plan before choosing between:

- **Track A — Quantization reliability–cost:** proceed only if matched quantized cells show reproducible reliability/calibration/cost effects beyond pass@1.
- **Track B — Serving-stack transfer:** pivot if stack effects dominate; isolate one stack factor at a time rather than comparing two bundled environments.

---

## 8. Documentation propagation

The repository now contains 47 Markdown files:

- **38 live first-party files updated or created**: root status/history, manuscript, scientific design/scope/roadmap, model/hardware policy, runbooks, HPC instructions, diagnostics, indexes, dashboard guidance, literature map, and reference notes.
- **9 intentionally unchanged files**: six documents under `docs/archive/`, `docs/GIT_CREDENTIALS.md`, and the two generated `ALL_PAPERS_MERGED.md` literature extracts.

Archived documents remain chronological artifacts, the credential file contains no scientific claims, and generated extracts must be regenerated from their source PDFs rather than hand-edited. Provider/vendored Markdown outside the 47-file repository inventory was read for the audit but remains upstream-owned.

## 9. Publication decision rule

Paper 1 may move from **Needs revision** to **Candidate manuscript** only when all are true:

- Clean, pinned, reproducible environment and model/dataset/prompt hashes.
- Matched BF16 and quantized controls on the same protocol.
- At least three pilot seeds; five seeds for headline MATH-500 cells.
- Valid termination, degeneration, confidence/calibration, and controlled systems telemetry.
- Predeclared paired tests, confidence intervals, multiple-comparison correction, and trace audit.
- A literature-backed contribution that is narrower than the existing broad “beyond accuracy” framing.
- A populated manuscript with results, limitations, and an artifact manifest.

Until then, the current result remains **appendix/control evidence only**.
