# Venue decision

**JSS desk-rejected 2026-09-09 (out of scope). Exclusive-submission lock is released.**

- **Manuscript ID:** `JSSOFTWARE-D-26-02208`
- **Submitted:** 2026-09-03
- **Decision:** Reject (pre-screen). Area Editor: Alexander Chatzigeorgiou.
- **Reason (verbatim gist):** not a Software Engineering contribution; better suited to an AI/ML venue.

Do **not** appeal. Do **not** resubmit to JSS. Do **not** reopen GPU.

**FGCS:** user reports a desk reject as out of scope (draft built as `FGCS-S-26-06835`, 2026-09-21). Do **not** resubmit to FGCS.

**AI Open (2026-09-22):** package built (`AIOPEN-S-26-01004` approval PDF reviewed). Do not submit it if Neurocomputing is the venue being uploaded.

**Neurocomputing (2026-09-22):** package `one-stack-many-rankings-neurocomputing/`. Upload `submission/manuscript.pdf` as Manuscript and `submission/abstract.docx` as Abstract. Declaration can be the on-screen “no competing interests” confirmation. Do not leave AI Open and Neurocomputing both under review.

## Constraint (2026-09-09)

Supervisors require **SCI / Web of Science (JCR) Q1**. “Q1” here means Clarivate JCR, not Scopus/SJR.

**TMLR is parked.** It is Scopus **Q2**, not a safe SCIE/JCR Q1. Do **not** click Submit on OpenReview. Cancel the draft if it is still open. Third-author OpenReview signup is no longer needed for the next venue.

## Next venue

Packages use title *Pitfalls of Single-Metric Evaluation for Quantized Reasoning Checkpoints* (third-author email `midhun@lincoln.edu.my`).

| Rank | Venue | Why | Package | Watch-out |
|------|--------|-----|---------|-----------|
| **1. Neurocomputing — current upload (2026-09-22)** | Neurocomputing (Elsevier) | Neural / machine-learning journal. | `one-stack-many-rankings-neurocomputing/submission/` | Manuscript PDF plus separate abstract.docx. Do not also submit AI Open. |
| 2. JMLR | Journal of Machine Learning Research | SCIE Q1 ML journal if FGCS is declined. | `one-stack-many-rankings-jmlr/submission/` | “Not a new algorithm” risk. |
| 3. ACM TIST | Trans. Intelligent Systems and Technology | SCIE Q1 backup. | `one-stack-many-rankings-tist/submission/` | ACM Open APC. |

**Parked:** TMLR (Scopus Q2). **Closed:** JSS, FGCS.

Only **one** journal may be under review. Upload `one-stack-many-rankings-aiopen/submission/manuscript.pdf`, not the blinded FGCS file and not the JSS PDF.

## Why JSS failed (do not repeat)

JSS wants a **direct Software Engineering** contribution. This paper pins a serving stack and measures ranking instability of **public quantized reasoning checkpoints**. That is ML evaluation / serving measurement.

## Package rules

- Independent sibling repos. Frozen science: commit `9f03909` numbers; live `main` may differ only in author metadata (third-author email `midhun@lincoln.edu.my`).
- No scientific number changes. Frozen source: `source_snapshot/`.
- JMLR and TIST: named PDFs (not double-blind).
- HPC never pushes. MacBook-only git for these packages.
