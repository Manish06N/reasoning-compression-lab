# Scientific audit (2026-09-02)

Stack-pinned measurement study. **GPU campaign not reopened.** Frozen numeric tables were not edited.

The manuscript should read as: a careful empirical measurement showing that quantized reasoning checkpoint rankings are unstable across evaluation targets under a pinned serving stack.

It should **not** read as: a new quantization algorithm, a proof that AWQ fails, a universal 4-bit law, a native-FP8 result, or a production-deployment recommendation.

Title: *One Stack, Many Rankings: Measuring Evaluation-Target Instability in Quantized Reasoning Checkpoints*.

---

## Claim table

| Claim | Evidence | Location | Supported? |
|---|---|---|---|
| Rankings disagree under one pinned stack | Same eight public checkpoints; pass@1 vs maj@5 vs Both-OK length vs token proxy vs Condition A vs Condition B | Abstract; C2; Tables `tab:math-contrasts`, `tab:tokens`, `tab:serving-main`, `tab:economics`; Figure `fig:condB-scatter` | **Yes** (descriptive ranking disagreement) |
| MATH FP8–BF16 differences are small and uncertain | Qwen $+0.40$ pp, Llama $+0.28$ pp; clustered 95% CIs include 0; $\pm 1$ pp TOST not passed | Abstract; Table `tab:math-contrasts` | **Yes** |
| Tested community AWQ artifacts showed task-specific degradation | Llama AWQ MATH $-2.76$ pp and GSM8K $-1.57$ pp (Holm-significant); Qwen AWQ MATH/GSM near BF16; Qwen AWQ GPQA $-5.56$ pp Holm-6 only | C3; Tables `tab:math-contrasts`, `tab:gsm-contrasts`, `tab:gpqa-contrasts` | **Yes** as **artifact** claim |
| Qwen AWQ GPQA $5.56$ pp is a headline discovery | Holm-6 significant; Holm-18 joint sensitivity $p=0.109$ n.s.; 3 seeds; $n=198$ | Results GSM8K/GPQA; Appendix `tab:holm18` | **No** as headline. **Yes** as a caveat-bound contrast. Manuscript now says it is not a headline. |
| GPQA degradation is a few hard items | 75 of 198 items flip on $\ge 1$ seed; 0 items BF16✓/AWQ✗ on all 3 seeds | Results; `item_level_descriptive_report.md` | **Supported as distributed**, not concentrated |
| Tested Qwen 4-bit checkpoints showed longer MATH completions | RoM $+6.3$–$6.9\%$; Both-OK CIs exclude 0; mismatch $D$ larger because failures are long | Table `tab:tokens`; length figures | **Yes** as **tested-checkpoint** claim. **Not** “4-bit increases length” as a law |
| Incorrect traces are longer than correct traces | Every MATH cell: incorrect mean tokens $\gg$ correct (e.g. Qwen BF16 $15{,}843$ vs $3{,}256$) | Item-level subsection; 2×2 table | **Yes** (association) |
| MATH degradation is all-seed item flips | 0 MATH items BF16✓/quant✗ on all 5 seeds for any format | Item-level subsection | **Yes** (absence of all-seed MATH flips) |
| Cost rankings disagree across serving measurements | Token proxy vs sequential GPU-s vs batched GPU-s; Qwen GPTQ-4 Condition B $-45.9\%$ vs BF16; Condition A ranks AWQ first | Table `tab:serving-main`; appendix token proxy | **Yes** |
| FP8 on A100 is native W8A8 | Marlin W8A16 fallback documented | Methods; limitations; runtime manifest | **Not claimed.** Explicitly denied. |
| AWQ as a method fails | C3 + limitations: community uploads, unknown calibration | C3; limitations; appendix checkpoints | **Not claimed.** |
| Hybrid $\widetilde{C}_{\mathrm{pass}}^{\mathrm{hyb}}$ is Erol Cost-of-Pass | Defined as confirmation GPU-sec / campaign pass@1 at $\$1.50$/A100-h scenario | Methods; related work; limitations | **Not claimed.** Named as aggregate hybrid proxy. |
| Figure 3 is a Pareto frontier | Caption: Condition B cost-accuracy scatter; panels are separate serving conditions; whiskers are 95% intervals | `fig:condB-scatter` | **Not claimed.** |
| Results transfer to Hopper / 32B / AIME / production batching | Limitations list unevaluated settings | Limitations | **Not claimed.** |
| Causal compression lengthening | $D$ described as diagnostic of mismatch asymmetry | Token table caption; discussion | **Not claimed.** |
| Red Hat production serving | Public RedHatAI artifacts on this A100 W8A16 stack | Related work; appendix | **Not claimed.** |

---

## Required wording checks

| Ban | Status in `paper/main.tex` |
|---|---|
| Causal “quantization causes” | Absent as a result claim |
| Universal “AWQ fails” / “AWQ is” | Absent. “tested community AWQ artifacts” |
| Universal “4-bit is” | Absent. “tested Qwen 4-bit checkpoints” |
| Native FP8 / W8A8 as measured | Denied; W8A16 fallback stated |
| Production deployment recommendation | Denied; “not that one format is universally preferable” |
| “Pareto frontier” as this paper’s figure | Absent. Lotfi row in appendix related-work table describes *their* paper |
| “Beyond Accuracy” in the title | Removed |

---

## Frozen numbers (unchanged)

Needles in `scripts/analysis/check_manuscript_numbers.py` must remain: 88 runs, 56,408 completions, vLLM 0.7.0, Marlin W8A16, $-2.76$, $-5.56$, $-1.57$, $6.3$–$6.9$, Holm-6 family, Holm-18 joint sensitivity, $-45.9$, tested community AWQ, not a property of bit-width alone, 75 of 198, 0.109.

TeX tables remain transcribed from `results/reports/major_revision_tables.md` and are guarded by `scripts/check_tex_tables.py`.

---

## Reproducibility surface

| Reproducible on CPU | Requires A100 + weights + HPC |
|---|---|
| Tables, analysis scripts, statistical checks, serving-confirmation analysis | Full 88-run GPU campaign |
| Compact JSON + `--check` scripts | Confirmation serving wall-clock |

Manifest (`results/reports/runtime_manifest.json`) records checkpoint IDs and revisions, dataset SHAs, LightEval 0.8.0, CUDA 12.4, PyTorch 2.5.1, vLLM 0.7.0. NVIDIA driver is **UNRECORDED** because the driver version was unavailable in the archived environment (explicit `nvidia_driver_note`).

The released artifact enables verification of reported analyses; reproducing the complete GPU campaign requires equivalent hardware and checkpoint availability.

---

## Remaining reviewer risks

1. TeX tables remain manually formatted; consistency is CI-checked against frozen values (stated in the artifact section).
2. Full traces and `finish_reason` are not released.
3. NVIDIA driver version was unavailable in the archived environment (explained, not silently blank).
4. Qwen FP8 Condition B five-rep wall-clock mixture, cause unknown.
5. Two 7B/8B families; A100 only; vLLM 0.7.0; no AIME / LiveCodeBench / 32B / energy / Hopper W8A8.
6. CRediT is in the PDF (author-3: Supervision + review only). Advisors should still confirm the split before Elsevier submission.
7. Incremental-vs-QRM remains a possible review; related work now states the ranking-stability question explicitly.
8. Holm-6 vs Holm-18 split will be probed; the paper treats GPQA as non-headline.
9. This 22-page PDF is for **JSS** initial submission (Your Paper Your Way). Do not send it unchanged to TMLR or FGCS.
