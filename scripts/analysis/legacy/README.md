# Deprecated analysis scripts

These files are kept for provenance. They are **not** the Paper 1 pipeline.

| Script | Why it is unsafe |
|---|---|
| `phase5_statistical_analysis.py` | Wrong pathology keys; pooled Wilson; circular gold-hit ECE |
| `selective_prediction_analysis.py` | Gold-hit counts, not modal-answer agreement |
| `audit_reasoning_traces.py` | 200-item even-index / seed-42 subset |
| `consolidate_multitask_results.py` | Pre-correction aggregator |
| `stratified_difficulty_analysis.py` | Needs an untracked local MATH-500 dataset |

Canonical entry points: `../revision_reanalysis.py` and `../emit_major_revision_tables.py`.
