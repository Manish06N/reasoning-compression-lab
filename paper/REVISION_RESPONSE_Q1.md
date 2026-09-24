# Q1 revision — point-by-point response

Branch `paper-q1-revision` (from `main@9204d62`). Source review: PhD repo
`docs/reviews/PAPER1_PEER_REVIEW_2026-09-24.md`. No frozen result, raw output or
existing analysis file was changed. New evidence comes from one new CPU script
(`scripts/analysis/q1_revision_analyses.py` → `results/reports/q1_revision_analyses.json`)
plus the previously untracked `think_prefix_audit` files. Every new number in the
manuscript traces to those files.

Build check: `main.tex` compiles to 28 pages with 0 errors, 0 undefined references,
0 overfull boxes and 0 BibTeX warnings. `scripts/analysis/check_manuscript_numbers.py
--check` still passes (20/20). `main.pdf` and `arxiv_source.zip` were rebuilt in
commit `8d384b5` (TeX Live 2018 on HPC, Type 1 fonts only; the zip compiles standalone).
Venue packages are not rebuilt; see "Venue status" below.

## Major comments

| # | Comment | Change | Where |
|---|---|---|---|
| M1 | AWQ chat template omits `<think>\n`; Appendix A said all completions begin with `<think>` | Appendix A rewritten with exact counts: all 14,102 AWQ-4 completions self-emit the marker; 0 in other cells; prompts otherwise byte-identical. Argument that the expected accuracy contrast is unaffected; token bias quantified (2 / 1 tokens, <0.05% of a trace). Template added to Methods §3.1, the collinear list in Limitations, a new "Prompt suffix" column in Table 2, contribution C1 and the Conclusion. Audit files committed. | §3.1, Table 2, App. A, Limitations 7, C1, Conclusion |
| M2 | Title's single-metric thesis vs C2/C3 | Abstract reframed as "one summary number misleads — a single pass@1, a single correctness-conditioned length estimate, and a single serving subset". New opening paragraph of the Discussion ties all findings to that thesis. Title kept. | Abstract, Discussion |
| M3 | Serving section mostly shows its own limits | §4.9 now opens with "This section does not claim a deployment cost ranking"; the length-draw mechanism and campaign-length estimator (Table 16) lead, and Table 15 is labelled supporting. | §4.9 |
| M4 | Kernel identity inferred | Wording changed to "configured"; cites the vLLM 0.7.0 path (`quantization=awq` disables the AWQ-Marlin conversion and selects `AWQLinearMethod`). Verified in the QRM vLLM fork, commit `5204ff5c3`, `awq_marlin.py::override_quantization_method`. States that the kernel was not logged at run time. | Table 2 caption, §4.9 |
| M5 | Deferred limitations resolvable with existing data | (a) New Table `tab:levels`: per-level pass@1 via exact question-text join (500/500 matched in both families). **New result:** the Llama AWQ-4 drop is not detectable at levels 1–2 and grows at levels 3–5 (−2.48, −3.75, −4.63 pp; all CIs exclude 0). (b) Per-item cap band from stored prompts: 155 truncations; 139/140 near-cap rows inside the band; 129/155 at levels 4–5. Prompt-length range corrected (Llama 28–747, not 27–776). Limitations 4 and 5 updated. | §3.5, §4.3, §4.5, Limitations |

## Minor comments

| # | Comment | Change |
|---|---|---|
| m1 | "Every Qwen saving" applies only to Condition B | Now "Every Qwen Condition B saving". |
| m2 | `tab:summary` never referenced | Referenced in the new Discussion opening. |
| m3 | Unused bib entry | `alimaskina2026extremelowbit` removed. |
| m4 | Bootstrap Monte Carlo error | Sentence added: an independent reimplementation reproduced every contrast-interval endpoint within 0.05 pp (MATH-500, GSM8K) and 0.2 pp (GPQA). |
| m5 | QRM values lack a source; **found an error** | The old values (94.60 / 91.00 / 50.00) came from QRM arXiv **v1**, while the bibliography cites the COLM version (**v2**: 93.9±0.7 / 92.5±1.4 / 48.1±2.8). Table 1 now shows v1, v2 and this grid with SDs, adds the BF16 GPQA row, and notes that QRM's own numbers moved between versions by more than the BF16–FP8 gaps. |
| m6 | 2026 arXiv references unverified | All four cited 2026 preprints checked on arXiv. Titles, full author lists and IDs match: Lian et al. 2606.25519, Lotfi et al. 2606.00206, Helcig et al. 2605.02404, Ayadi et al. 2607.10855. (The fifth, 2606.02011, was the unused entry removed in m3.) |
| m7 | Table 11 vs Table 12 one-item difference | Kept as disclosed; both tables state it. Regenerating Table 11 would change frozen published values. |
| m8 | Length rule is a weak baseline | Limitations now state that token-level confidence (log-probability, self-certainty) was not stored, so the stronger comparison is not made. |
| m9 | §3.5 is long | Not restructured, to keep the frozen estimand definitions in one place for a systems-journal audience. Move to an appendix only if the target journal's page limit requires it. |
| m10 | GPQA in the headline narrative | Already absent from the abstract. **Strengthened instead:** stored prompts prove GPQA answer order was identical across all four checkpoints for every family and seed (594/594), so GPQA contrasts are paired. "Not certified" caveats replaced in Table 1, Table 6, §4.2, App. A, Limitations and Conclusion. The Qwen AWQ-4 contrast stays borderline and non-headline. |

## Third verification round (added after the first push)

| Item | Change |
|---|---|
| CV rule | The R=3 → R=5 rule uses the population SD (as implemented; Llama FP8 Condition A = 2.995%), while reported tok/s spreads are sample SDs. Now stated in §3.4. |
| Published versions | DeepSeek-R1 → *Nature* 645:633–638 (2025); Marlin → PPoPP 2025; GPTQ → ICLR 2023; Let's Verify → ICLR 2024; GPQA → COLM 2024. DOIs checked on Crossref where available. |
| Table legibility | Table 24 (related studies) was shrunk to unreadable size; rebuilt as a wrapped five-column table. Tables 8, 12, 15 were scaled to ~6.5–7.5 pt; now set sideways at natural 10 pt. Table 17 was enlarged to ~11.6 pt; now natural size. |
| Float placement | Appendix tables no longer split the reference list (`\FloatBarrier` before the bibliography). |
| Verified, no change needed | Every numeric sentence in the prose; QRM `inference.py` seeding order and `use_chat_template=True`; Lian et al. CTIR and Appendix E, Kurtic et al., Lotfi et al., Helcig et al. characterizations; modal-report internals (116 ties, seeds, SHA prefix). |

## Author questions

| # | Question | Answer in the manuscript |
|---|---|---|
| Q1 | Other template differences? | None. FP8 and GPTQ-4 prompts are byte-identical to BF16, and AWQ-4 differs only by the suffix, on all three benchmarks (App. A). |
| Q2 | Unconditional placebo | Exactly zero by construction when pooled over ordered seed pairs; stated in §3.5 as the reason it is the primary length estimand. |
| Q3 | Equivalence-relation sensitivity of the k-curve | Exact-string equality moves coverage ≤0.4 pp and risk ≤0.11 pp; all comparisons with the length rule unchanged (§4.8). |

## Venue status (from `docs/VENUE.md`)

Supervisors require **Clarivate JCR Q1**. **Closed:** JSS (desk reject 2026-09-09,
out of scope) and FGCS (desk reject). **Parked:** TMLR (Scopus Q2). **Current:**
Neurocomputing (package built 2026-09-22). **Backups:** JMLR, ACM TIST. Only one journal
may be under review at a time.

The venue packages (`one-stack-many-rankings-{neurocomputing,aiopen,jmlr,tist}/`) are
separate MacBook-only repositories with a frozen `source_snapshot/`. **This revision does
not update them.** If Neurocomputing is already under review, keep the submitted version
and use this revision for the revision round. Otherwise refresh the chosen package from
`main@8d384b5` on the MacBook and rebuild it there.

Note for the packages' "no scientific number changes" rule: this revision adds new
analyses (difficulty levels, cap band, template and GPQA audits). It corrects one
descriptive number (Llama prompt range 27–776 → 28–747, from the stored prompts) and
the externally sourced QRM values in Table 1 (arXiv v1 → both v1 and the COLM v2). No
frozen campaign result changed.
