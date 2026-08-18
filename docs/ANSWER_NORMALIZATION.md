# Answer normalization policy (frozen before agreement analysis)

**Status:** frozen 2026-08-16, *before* any modal-agreement or selective-prediction rerun. HPC modal analysis (`scripts/analysis/modal_agreement_analysis.py`) used this policy; do not retune it. Modal agreement is complete (compact artifact SHA256 `23e9ead0...`).

Compact `results/*.json` records store `extractive_match` but **not** extracted answer strings. HPC JSONLs were recovered; agreement analysis used this frozen policy. Do not retune it after seeing risk–coverage curves.

## Chosen definition (conservative)

Use the **same extractive match / mathematical equivalence that scored the 56,408-completion campaign**.

That path is the official QRM LightEval task (`lighteval_custom/tasks/reasoning.py`) plus the LightEval `[math]` extra (math-verify). Campaign extraction used **LightEval 0.8.0**. A later throwaway MacBook venv installed LightEval 0.8.1 and was **not** used for canonical extraction or paper numbers. A pair of completions *agree* when the campaign evaluator would have given both the same extractive-match decision against a shared reference string — that is, when their extracted answers are equivalent under that judge.

Consequences:

- `"42"`, `"42.0"`, and `"042"` agree if math-verify treats them as the same number (they should).
- `\frac{84}{2}` vs `42` is decided by the evaluator, not by a new string-normalize function in this repo.
- We are not inventing a second judge for the paper's primary agreement metric.

## What we will not do for the primary curve

- New regex-only boxing
- LLM-as-judge equivalence
- Manual recoding of MATH expressions after looking at coverage

## Optional sensitivity (report only if both are computed)

If recovered JSONLs contain both the raw generation and the evaluator's extracted field:

1. **Evaluator agreement** (primary): math-verify / LightEval extractive equivalence.
2. **Exact normalized-string agreement** (appendix): strip, collapse whitespace, and compare the evaluator's extracted strings.

If operating points barely move, that strengthens the result. If they move, the paper reports both and keeps (1) as primary.

## Selective prediction, if answers are recovered

Abstention is **modal-answer agreement among 5 samples**, not gold-hit counts.

- Confidence signal: size of the modal extracted-answer cluster (3/5, 4/5, 5/5).
- Metric: risk–coverage, plus the **generation-cost penalty of 5 completions per query**.
- Do not call this an “operational safety gate.”
- Gold labels are used only to score selected predictions, never as the abstention feature.

## Implementation note

`scripts/hpc/qrm_parity/export_extracted_answers.py` dumps whatever answer-like fields exist in the JSONL. After export, agreement code must call the same math-verify path used at evaluation time, not a fresh normalizer written against the recovered strings.
