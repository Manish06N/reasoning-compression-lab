# Final publication audit (2026-09-02)

Stack-pinned measurement study. GPU campaign **not** reopened.

The repository should communicate: *we provide a stack-pinned measurement protocol showing that public quantized reasoning checkpoints can change rank depending on what is measured.*

It should not communicate: *we discovered the best quantization method.*

---

## 1. Files inspected

Publication surfaces:

- `paper/main.tex`, `paper/main.md`, `paper/ARTIFACT.md`, `paper/references.bib`
- `README.md`, `REPRODUCE.md`, `REVISION_SUMMARY.md`, `AGENTS.md`
- `results/README.md`, `results/reports/runtime_manifest.json`
- `results/reports/major_revision_tables.md`, `results/reports/modal_agreement_report.md`
- `results/reports/item_level_descriptive_report.json` / `.md`
- `configs/models/README.md`, `configs/legacy_models/README.md`
- `.github/workflows/ci.yml`
- `docs/PUBLICATION_READINESS.md`
- Analysis scripts: `revision_reanalysis.py`, `emit_major_revision_tables.py`, `measured_serving_confirmation_analysis.py`, `measured_serving_analysis.py`, `modal_agreement_analysis.py`, `item_level_descriptive_analysis.py`, `benchmark_serving_confirmation.py`

Historical logs left as honesty notes (not rewritten as live claims): `CHANGELOG.md` older entries, `docs/supervisor/`, `docs/plans/`, `progress.md`.

---

## 2. Files modified in this audit

Wording and hygiene only. No frozen numeric tables were recalculated.

| File | Reason |
|------|--------|
| `paper/main.tex` | RQ1 checkpoint language; required Holm-6 / Holm-18 GPQA sentence; serving-condition (not causal batching) language; lower aggregate hybrid cost proxy; Figure token caption notes missing CIs; reproducibility-surface limitation; JSS threats-to-validity mapping |
| `paper/main.md` | Match tex RQ1 and GPQA wording |
| `paper/ARTIFACT.md` | `--check` commands; traces not released; tables vs GPU replay |
| `README.md` | Measurement-study identity; tested-artifact wording; traces not released |
| `REPRODUCE.md` | Compact-JSON reproducibility vs unreplicable full traces; expected PASS lines |
| `AGENTS.md` (repo) | Remove live “cost frontier / proving” contribution; Holm-6 / Holm-18 sentence |
| `../AGENTS.md` (Paper 1 memory) | Framing is measurement study, not bake-off |
| `docs/PUBLICATION_READINESS.md` | Current label; historical frontier differentiator marked superseded |
| `results/README.md` | Tested-artifact + required GPQA sentence |
| `results/reports/major_revision_tables.md` | “Rank order (1 = best)” → lowest aggregate cost proxy (wording; numbers unchanged) |
| `scripts/analysis/emit_major_revision_tables.py` | Same rank-order wording so `--check` still matches |
| `scripts/analysis/modal_agreement_analysis.py` + `modal_agreement_report.md` | Heading no longer “Frontier” (JSON numbers untouched) |
| `scripts/analysis/item_level_descriptive_analysis.py` + frozen JSON/MD | GPQA note uses required Holm sentence |
| `scripts/analysis/measured_serving_analysis.py` | Docstring: first-run provenance, not Pareto / native FP8 |
| `scripts/analysis/README.md` | Expected confirmation `--check` line |
| `.github/workflows/ci.yml` | Require `REPRODUCE.md` and this audit file |
| `CHANGELOG.md` | This audit |

---

## 3. Checks performed (fresh, this session)

| Command | Result |
|---------|--------|
| `python3 scripts/analysis/revision_reanalysis.py --check` | `OK: recomputed report matches .../revision_reanalysis_report.json` |
| `python3 scripts/hpc/qrm_parity/benchmark_serving_confirmation.py --check` | `OK: confirmation raw artifacts present (60 required files)` |
| `python3 scripts/analysis/emit_major_revision_tables.py --check` | `OK: generated tables match .../major_revision_tables.md and .../major_revision_validation.md` |
| `python3 scripts/analysis/measured_serving_confirmation_analysis.py --check` | `OK: recomputed confirmation report matches ... (8 configs)` |
| `python3 scripts/analysis/modal_agreement_analysis.py --check-artifact` | `OK: compact artifact SHA256, 20,000/4,000 structure, T5 accounting, and report internals match.` |
| `python3 scripts/analysis/item_level_descriptive_analysis.py --check` | `OK: item-level descriptive report matches ...` |
| Required-file presence (CI list + `REPRODUCE.md`) | `OK: 11 required artifacts present` |
| `.venv/bin/python -m pytest tests/ -q` | pass (100%) |
| `.venv/bin/python scripts/hpc/07_preflight_publication.py --ci` | `HPC publication CPU preflight passed.` |
| `ruff check` (CI publication scope) | All checks passed |

MacBook `lighteval` import warnings on emit/modal `--check` are expected outside `qrm-official`; both checks still PASS.

Remote GitHub Actions was not executed (no push). Local CI-equivalent jobs passed.

---

## 4. Scientific risks found

| Risk | Severity | Where |
|------|----------|--------|
| RQ1 “Does quantization change pass@1” reads as a method law | Medium | `paper/main.tex` |
| Discussion implied sequential vs batched *workloads* reorder ranks | Medium | Discussion |
| GPQA Holm-6 stated without the exact Holm-18 joint sentence in some captions | Medium | GPQA table caption / appendix |
| “Rank order (1 = best)” in frozen table markdown | Low | `major_revision_tables.md` |
| Modal report heading “Risk–Coverage–Cost Frontier” | Low | generated report MD |
| Live AGENTS “cost frontier / proving” contribution | Medium | `AGENTS.md` §3 |
| `REPRODUCE.md` implied full traces were released | Medium | GPU replay section |
| Dollar “cheaper” without scenario wording | Low | serving results |
| Mean-token bar chart had no uncertainty note | Low | Figure tokens |
| No JSS threats-to-validity mapping | Low | Limitations |

Unsupported causal claims of the form “AWQ hurts reasoning,” “FP8 is cheapest,” “batching caused ranking changes,” or “quantization increases reasoning length” were **not** present in `paper/main.tex` after the prior revision. Residual risks were phrasing, not invented results.

---

## 5. Scientific risks resolved

- RQ1 now asks whether **evaluated quantized checkpoints differ** from matched BF16.
- Serving language: **different serving conditions produced different rankings**; Conditions A and B are joint subset+regime contrasts, not a pure batching ablation.
- GPQA uses: *The Qwen AWQ GPQA result is significant within the primary Holm-6 family, but not under the Holm-18 joint sensitivity analysis.*
- AWQ findings remain **tested community artifact** claims.
- FP8 remains **Marlin W8A16 on A100**, not native W8A8 / Hopper.
- Hybrid metric remains **aggregate hybrid Cost-of-Pass proxy**; GPU-seconds/query primary; dollars = pricing scenario. Not “Erol Cost-of-Pass.”
- Figure 3 remains **Condition B cost-accuracy scatter** with separate serving-condition panels; not a Pareto plot.
- Limitations state traces are not public; tables are reproducible; GPU replay is not expected of reviewers.
- `configs/models/` contains only a warning README. Historical JSON is in `configs/legacy_models/`.
- `CANONICAL_PASS1` is loaded from the frozen reanalysis report (`load_canonical_pass1()`), not duplicated dicts.

Kept as scientific honesty (not live claims): retracted 0/0 pathology, 98.23% “safety gate,” first-run $+18.7\%$ / $-19.8\%$, and “true Pareto” **do-not-cite** notes.

---

## 6. Confirmation

- **No GPU experiments added.**
- **Frozen numeric campaign results unchanged** (`revision_reanalysis_report.json` `--check` PASS; confirmation serving `--check` PASS; emit `--check` PASS after a rank-order *label* wording change only).
- **No unsupported causal claims remain** on publication surfaces (`paper/main.tex`, README, REPRODUCE, live AGENTS superseding block).
- **FP8 W8A16 wording preserved.**
- **AWQ artifact distinction preserved.**
- **GPQA Holm-6 / Holm-18 wording preserved** (now the required sentence).
- **Cost proxy terminology correct.**
- **README matches the frozen-check workflow.**
- **CI-equivalent local checks passed.** CI will fail on table drift, modal SHA mismatch, missing files, and emit mismatch.

Runtime manifest records:

- Model HF IDs and revisions
- MATH-500 SHA `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`
- GSM8K SHA `740312add88f781978c0658806c59bc2815b9866`
- GPQA SHA `633f5ee89ab8ad4522a9f850766b73f62147ffdd`
- vLLM 0.7.0 eager, LightEval **0.8.0**, A100 serving flags

---

## 7. Remaining human decisions

1. **CRediT roles.** Current `paper/main.tex` statement (conservative; not invented):
   - Manish Nandish: Conceptualization, Methodology, Software, Investigation, Data curation, Formal analysis, Visualization, Writing — original draft.
   - Rajiv Misra: Supervision, Resources, Writing — review & editing.
   - Midhunchakkaravarthy Janarthanan: Supervision, Writing — review & editing.
   Confirm before submission if advisors should also be listed for Validation, methodology guidance, or Project administration.
2. **Venue.** Primary alignment in this pass is **Journal of Systems and Software** (artifact, pinned protocol, threats to validity). Related-work Table 1 is retained. For **TMLR**, consider moving Table 1 to an appendix. Do not optimize for FGCS unless requested.
3. **PDF / arXiv zip.** Rebuilt 2026-09-02: `paper/main.pdf` (22 pages) and `paper/arxiv_source.zip` SHA256 `420d62c599c03bd7c313c1b2ebb89b2953c7d3c3a98cd2f7f07842b21643d346`. Recompile again only if `main.tex` changes after this zip.
4. **Remote CI.** Local checks passed; GitHub Actions runs only after push to `paper-major-revision`.
5. Do **not** merge to `main` until the submission snapshot is tagged.

---

## Venue note (JSS vs TMLR)

JSS reviewers can evaluate: pinned serving stack, artifact table, threats-to-validity paragraph, and compact-JSON `--check` reproducibility.

TMLR reviewers can evaluate the same three contributions with less systems detail; Table 1 is optional there.

Neither venue should be sold a quantization bake-off.
