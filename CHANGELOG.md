# Changelog

## 2026-10-03 — Revision after a fourth referee report (no GPU, no frozen result changed)

- **FP8 serving discrepancy.** FP8 has the highest measured-to-campaign length ratio in all four family-by-condition groups (Qwen 1.35/1.07, Llama 1.49/1.09). The Llama FP8 example is removed from the abstract and "tail draw is the likeliest reason" is withdrawn: model files, backend, prefix caching, chunked prefill and KV dtype match; scheduling (single-stream vs batched), seed and the extra BOS token remain untested. A single-BOS multi-seed GPU rerun was not run.
- **Accuracy-weighted change.** The plug-in estimate (biased upward) is replaced by a cross-fitted one (`followup_checks.py`, `estimand_check`); every estimate shrinks and Llama AWQ-4's interval now includes zero.
- **Abstract.** "Finished wrong answers, not truncations" is scoped to MATH-500 (no GSM8K decomposition was run).
- **Smaller fixes.** Agreement symbol renamed `g_i` (was `a_i`, which collided with BF16 accuracy); aggregate tok/s columns added to the campaign-length cost table; Condition B rank-1 shares given as 97.55%/2.45%; reproducibility pointer, engine-log and BOS wording, finished-only reflection caveat.
- Not done: GPQA letter-position analysis, winsorized subset simulation, AI-assistance disclosure (venue decision).

## 2026-10-03 — Revision after a third referee report (no GPU, no frozen result changed)

Every point of the report was checked against the data before editing (see `results/reports/followup_checks.json`, `audit_checks.json`).

- **Serving (C4).** The Llama FP8 Condition A subset (7,282 vs 4,889 tokens for the same prompts) is the largest of sixteen measured-vs-campaign comparisons (ratio 1.49; Qwen FP8 Condition A 1.35, range 0.88-1.49). The latency percentiles imply about three requests of roughly 18,000 tokens or more; a count model puts three or more such traces among 20 prompts at no more than about 4% (10% pooled), far above the 0.1% that resampling five stored completions suggests (resampling truncates the tail). C4 now rests on the subset simulation (random subsets reproduce the campaign-length order in 21-40% of draws). Newly disclosed: the timer's prompts carry one extra input token per request in every cell (a second BOS), and AWQ-4 timer prompts include the `<think>` suffix.
- **tok/s aggregation.** Campaign-length seconds now use total tokens over total time (matches the reported GPU-s/q and hybrid dollars). Qwen FP8 Condition B: 9.44 -> 9.82 s; Llama FP8 A 64.61, AWQ-4 A 78.12, BF16 A 72.14; ranks unchanged. The Qwen FP8 / GPTQ-4 "tie" is gone (GPTQ-4 first in 97.6% of resamples). Tables, scatter figure and text updated.
- **C1.** "Truncation-driven" replaced: Qwen 4-bit truncation increases (+0.52, +0.48 pp) are unadjusted (p=0.013 and 0.050; Holm-6 0.078 and 0.25), a small rise that accounts for most of non-significant point gaps.
- **RQ2/abstract.** Qualified as "under sampled decoding"; the placebo cannot be run under greedy decoding. What the excess estimates is now stated: conditional minus placebo estimates the unconditional change reweighted by per-item BF16 accuracy; weighted changes and CIs added (explains Qwen GPTQ-4: +223 vs +276).
- **Methods.** AutoRound-style provenance of the AWQ uploads and the AWQ backend explanation moved to Methods (one home); two-way bootstrap and detectable-effect results moved out of Methods into Results; MDE also given at the Holm-6 first-step multiplier (1.6-2.4 pp MATH-500, 6.8-7.6 pp GPQA-Diamond).
- **GPQA.** Appendix A: generation seed reaches vLLM only as a per-request sampling seed; prompts identical across checkpoints and across the three seeds (198/198), so every GPQA number is conditional on one answer-letter permutation (new limitation); the vLLM re-seed is code inspection, not reproduced from the letters; QRM caveat softened.
- **Reproducibility.** The statement that all additional analyses use only released JSON is corrected: reflection markers, the frozen-tokenizer recount, the exact-cap test and the `<think>` audit need completion text (not released). Script inventory replaced by a pointer to `docs/PAPER_TO_CODE.md`.
- **Shape.** Seed-level t-test dropped (2-4 df); reflection markers (now finished-completions-only, which removes the pooled Qwen FP8 artifact) and the graded-equivalence paragraph moved to the appendix; abstract reframed around estimand sensitivity and now states the failure-class finding; new appendix table of measured vs campaign subset lengths.
- **Smaller fixes.** k-curve and microbenchmark captions; revision-history sentences removed; duplicated AWQ text removed; subset description unified; Limitations items 2, 4, 7, 12 corrected (cap-band tolerance rationale, logs vs launcher record); related-work table Lian et al. row; difficulty trend added (post hoc, linear; Llama AWQ-4 -1.25 [-2.36, -0.15] pp/level) and the 118/11,382 statistic explained; summary-table 4,736; GPQA graded-equivalence maximum 0.351.
- Not changed: frozen results and raw outputs; releasing derived per-completion fields (an option for a later revision); greedy-decoding placebo (needs new runs).

## 2026-10-03 — New title

Title changed to *Estimand Sensitivity in the Evaluation of Quantized Reasoning Checkpoints: Selection Bias, Seed Placebos, and Subset-Cost Instability* (was *Pitfalls of Single-Metric Evaluation for Quantized Reasoning Checkpoints*). Updated `paper/main.tex`, `paper/main.md`, `README.md` (text and BibTeX), `AGENTS.md`; PDF and arXiv zip rebuilt. No scientific content changed. The separate venue packages (`one-stack-many-rankings-*`) were not touched and still use the old title; historical notes (`SCIENTIFIC_AUDIT.md`, `SUBMISSION_REVIEW.md`, `docs/archive/planning/VENUE.md`) keep it as a record.

## 2026-10-03 — Final corrections after an independent audit and referee review (no GPU, no frozen result changed)

Audit of commit `a5a1a16` (full recomputation of 1,366 reported numbers; 1,321 exact, 32 rounding/bootstrap-stream, 13 mismatches from nine root causes). New repo-traceable checks: `scripts/analysis/audit_checks.py` -> `results/reports/audit_checks.json` (frozen-tokenizer env).

- **Llama prompt range corrected to 27-776** (AWQ-4 prompts 25-774; Qwen 29-811, AWQ-4 27-809), effective cap 31,957-32,743. The Q1 revision had replaced the correct range by 28-747, computed with transformers 5.x, whose LlamaTokenizer mis-tokenizes Llama text (completion counts ~220 tokens off). With the frozen tokenizers (4.47.1) every stored `completion_tokens` value is reproduced (56,408/56,408). Cap-band counts (155/139/16) are identical under both tokenizers. Exact-cap test: 147 of 155 MATH-500 cap-band rows (and all 57 near-cap GSM8K rows) have exactly `32768 - prompt` tokens.
- `scripts/analysis/q1_revision_analyses.py` now refuses to run outside transformers 4.47.x; `REPRODUCE.md`: numpy pin needs Python >= 3.12, `check_tex_tables.py` lives in `scripts/`.
- AWQ backend: AWQ-Marlin needs a runtime zero point; both `jakiAJK` configs set `zero_point=false`, so they are ineligible whatever the flag; resolved method is logged as `awq`.
- Serving: Qwen FP8 vs GPTQ-4 Condition B depends on tok/s aggregation (9.82 s vs 9.39 s with total tokens / total time; GPTQ-4 first in ~97% of resamples); Llama FP8 Condition A subset length is longer than 99.9% of 20,000 resampled draws; abstract "decides" -> "can reorder"; CV rule disclosure (3.67% with sample SD).
- Length: paired excess-over-placebo intervals added (Llama AWQ-4 +97 [-66,+267] is not distinguishable from zero); "close to the unconditional delta" qualified (Qwen GPTQ-4 +188 vs +276).
- Wording/number fixes: maj@5 change-in-gap interval (+1.36 [-0.24,+3.00]); Lian et al. estimand (CoT-only, greedy); Kurtic et al. also evaluate R1-distill; "22 of 24" (one cause); paired-SE statement (Llama 0.60-0.70 pp); bootstrap-stream note and tolerance wording; GPQA answer-order mechanism (vLLM re-seeds Python's RNG at model init; order is constant across checkpoints and seeds); Table 5,444 and level L1 94.0; multitask/strata captions; ECE sentence and reviewer-facing sentence removed; AWQ provenance (`_name_or_path ./tmp_autoround_awq`, undocumented calibration) added; subset-file note; abstract qualifier "On MATH-500 and GSM8K".
- Known and left: Llama FP8 >=3/5 string reconstruction differs by one item from the LightEval-parsed table values (94.0/3.19 vs 93.8/2.99); declarations (funding, competing interests, AI-use) still to be supplied by the authors.

## 2026-10-02 — Wording and number corrections after the second independent review (no GPU, no frozen result changed)

Source: `~/paper1-review-2026-09-27/PEER_REVIEW.md` (findings 1-8, 10), checked against the repo before editing.

- AWQ backend: the launcher passes no `quantization` override; engine logs record the resolved `quantization=awq`; both AWQ configs set `zero_point=false`, which fails the pinned AWQ-Marlin compatibility check. Table 2, Section 4.9, C1, Conclusion reworded (inference, not profiled). Serving prompts (with `<think>`) differ from AWQ accuracy prompts: disclosed.
- Llama FP8 Condition A is 145,647/20 = 7,282.35 tokens per query (was 7,288, a product of rounded means); abstract, Section 4.9, CHANGELOG-era `main.md` corrected. Identity tok/s x s/query = tokens now stated per repeat only.
- "Identical token sequences" replaced by identical aggregate counts (sequences were not retained).
- Qwen serving savings: "upper bounds" replaced by "potentially confounded, bias direction unknown"; microbenchmark no longer called pure kernel speed.
- maj@5: "halves/recovers" now a point-estimate statement; the paired interval for the change in the Llama gap includes zero.
- Placebo: FP8 "fully reproduced" replaced by "no excess detected"; subtraction called a diagnostic contrast, not an unbiased correction.
- Lian et al. CTIR no longer called the same estimand (CoT tokens, temperature zero vs whole completion, T=0.6). QRM v2 uses 3 seeds.
- Near-cap vs cap-band: reported as heuristic agreement (139/140) and 16 further suspected truncations, not validated truncations.
- Llama `<think>\n` is 2 tokens in the frozen runtime (1 in newer tokenizers). "Within about one point" -> "within 1.7 pp". Figure 3 Llama AWQ bar 4,736; Figure 1 caption corrected. Manuscript date updated.
- Added `scripts/analysis/review_additions.py` -> `results/reports/review_additions.json` and Appendix "Additional sensitivities" (Tables 25-27): two-way items x seeds bootstrap and seed-level t-test for all 18 pass@1 contrasts (Llama AWQ-4 MATH-500 stays away from zero, [-4.96, -0.68]; the GSM8K seed-level test is marginal, p=0.051); minimum detectable effects (1.3-2.0 pp MATH, 0.9-1.4 GSM8K, 5.5-6.2 GPQA); median and trimmed token deltas (Qwen 4-bit mean difference comes from a minority of long traces); stratified subset-draw simulation (campaign-length order reproduced in 40% / 21% of random 20-prompt draws). Text added to Sections 3.5, 4.4 and 4.9 and Limitation 9.
- Added `scripts/analysis/trace_failure_analysis.py` -> `results/reports/trace_failure_analysis.json`, new Section 4.7 "Failure modes and the length signal" and Appendix Tables 28-29. Llama AWQ-4's extra errors are finished wrong answers (+3.04 pp [1.68, 4.40]), not truncations (-0.24); the Qwen 4-bit gaps are truncations (+0.52 / +0.48 pp) with no detectable wrong-answer change. Length predicts failure strongly when pooled (AUROC 0.83-0.94) but weakly within an item (0.62-0.76), and no quantized cell differs detectably from BF16. C1 and the Conclusion mention the mechanism.
- Added `scripts/analysis/adaptive_reflection_rope.py` -> `results/reports/adaptive_reflection_rope.json` and Appendix Tables 30-32 (text in Sections 4.1, 4.7, 4.8): graded equivalence (posterior P(|delta|<1 pp) under a flat Dirichlet Bayesian bootstrap: 0.84-0.91 for MATH FP8/GPTQ, <= 0.35 on every GPQA contrast); reflection-marker keyword counts (higher for Qwen AWQ-4/GPTQ-4 and Llama GPTQ-4, unchanged for Llama AWQ-4); adaptive agreement sampling (draw 2, escalate to 5 only on disagreement: about the fixed >=3/5 operating point at 37-46% fewer tokens). Point estimates, exploratory.
- Not changed: frozen results, raw outputs, analysis scripts, the mislabeled `selective_risk_clopper_pearson_ci_95` field (documented in the review), declarations (to be supplied by the authors).

## 2026-09-24 — Q1 revision after independent peer review (no GPU)

Addresses every point in `paper/REVISION_RESPONSE_Q1.md`. No frozen campaign result changed.

- Both AWQ-4 chat templates omit the `<think>\n` generation suffix; all 14,102 AWQ-4 completions emit it themselves. Disclosed in Methods, Table 2, Appendix A, Limitations. The earlier Appendix A sentence "saved completions also begin with <think>" was wrong for six of eight cells.
- **Supersedes the 2026-09-23 GPQA note:** stored prompts are byte-identical across the four checkpoints of each family and seed, so GPQA contrasts are paired.
- New difficulty-level table (exact question-text join, 500/500): the Llama AWQ-4 drop grows at levels 3–5. Per-item cap band: 155 MATH-500 truncations, 139 of 140 near-cap rows inside. Llama prompt range corrected to 28–747 tokens.
- Table 1 now shows QRM arXiv v1 and COLM v2 values; the old values were v1 while the bibliography cites COLM.
- New script `scripts/analysis/q1_revision_analyses.py` → `results/reports/q1_revision_analyses.json`; `think_prefix_audit` committed.
- Round-3 checks: CV rule stated with its population SD; five references moved to published versions (Nature, PPoPP, ICLR x2, COLM); Table 24 rebuilt readable; Tables 8, 12, 15 set sideways at natural size (were 6.5-7.5 pt); Table 17 no longer upscaled; floats kept out of the bibliography.
- `paper/main.pdf` is 28 pages. `paper/arxiv_source.zip` SHA256 `bbcce8b4561f84a2b7802ee41af018a393d8ee48b838bf3012a0c1b1ed6eca9c`.

## 2026-09-23 — Length table stacked cells and k-curve correction (no GPU)

`paper/main.tex` stacks each length interval under its point estimate. The k=5 Qwen GPTQ-4 coverage in the unanimity curve is 86.6%, one item below the modal table, because equivalence is recomputed on stored strings. `paper/arxiv_source.zip` SHA256 `49a9b4339c8784d4f97ab0954a55091b16addff9f0927e13c9a8998baa680087`. PDF is 24 pages.

## 2026-09-23 — Framing realigned to the evidence (no GPU)

Title is now *Pitfalls of Single-Metric Evaluation for Quantized Reasoning Checkpoints*. Research questions and contributions follow the four results the tables support: Llama AWQ-4, the BF16-correct length selection effect, subset cost as one length draw, and five-sample agreement versus a length rule. Frozen pass@1, token, and serving means are unchanged.

- Quant-correct length intervals are in the length table. Serving brackets are repeat min--max. The Condition B figure uses campaign-length seconds.
- MATH-500 level join, prompt-cap recount, and unboxed-by-near-cap counts are reported. GPQA letter order is the QRM `random.randint` prompt, not a certified paired contrast.
- The shared $65 tok/s dollar proxy is removed. The sensitivity script is `scripts/analysis/manuscript_sensitivities.py`.
- `paper/arxiv_source.zip` rebuilt from this manuscript. SHA256 `b35f4724e2effa5073c6fa6ae8d9a3df23b51de6f50e83767303e15b3630c540`.

## 2026-09-23 — Manuscript revision from two reviews (no GPU)

Frozen pass@1, token, and serving tables are unchanged. `paper/main.tex` now states what those tables do not support.

- Cost intervals cover wall-clock repeats of one sampling seed. A campaign-length sensitivity (full-grid tokens / measured tok/s / pass@1) is in the manuscript. Llama FP8 Condition A is about 7,288 tokens/query on the 20-prompt draw and 4,550.8 on the full grid.
- maj@5 accuracies, quant-correct length deltas, one-sample length abstention, dtype/kernel, GPQA prompt limit, cap definition, TOST margins, and rank-1 frequencies are reported. Script: `scripts/analysis/review_response_sensitivity.py`.
- No new GPU jobs. Qwen was not rerun on the other host. Cap hits were not reclassified, because `finish_reason` and prompt lengths are not in the compact JSON.
- Local venue packages were regenerated from this manuscript. Those folders are not in this repository.
- `paper/arxiv_source.zip` rebuilt 2026-09-23 from current `main.tex` + `references.bib` + `main.bbl`. SHA256 `26b6067bc25b1b17cd92908b42a620b06844ca3763dd17aef65fe59058fd1919`. Not pushed yet.

---

Earlier entries (2026-06 to 2026-09-21: venue handling, earlier manuscript versions, campaign setup) are in [docs/archive/CHANGELOG_to_2026-09-21.md](docs/archive/CHANGELOG_to_2026-09-21.md).
