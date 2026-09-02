# Submission review (2026-09-02)

**Title update (same day, later pass):** the manuscript title is now *One Stack, Many Rankings: Measuring Evaluation-Target Instability in Quantized Reasoning Checkpoints*. “Beyond Accuracy” was dropped because it reads like a new evaluation framework. Frozen numbers were not changed. See `SCIENTIFIC_AUDIT.md`.

Final technical-editor / reproducibility review of the stack-pinned measurement study.

**GPU campaign was not reopened.** Frozen numeric tables were not rewritten.

**Score: 86 / 100** (was 78 before the TeX checker and config pointers).

**Recommendation: B — Needs minor fixes** is now **close to A for JSS** if Actions is green. Remaining: traces not released, CRediT still a human confirmation, no energy/AIME.

This is not a major-revision science problem. It is not artifact-eval ready either. The remaining blockers are human (venue, CRediT, visual PDF QA) plus one engineering gap (TeX tables are transcribed, not generated).

---

## Phase 1 — Current state map

| Item | State |
|------|--------|
| Git | Branch **`main`** at `6bdba78` (origin/main). Working tree has this audit’s uncommitted files. |
| `paper-major-revision` | Behind `main` (`13cc475`). Docs that still say “do not merge to main” are stale relative to GitHub. |
| Manuscript | `paper/main.tex`, 22 pages (xelatex). |
| Frozen tables | `results/reports/major_revision_tables.md` + `revision_reanalysis_report.json` |
| Completions | **88 JSON files, 56,408 rows** (`20,000 + 31,656 + 4,752`) |

### A. Already implemented correctly

- Central claim is a **pinned-stack measurement study**, not a quantizer, not an AWQ-failure law, not a bake-off.
- FP8 = Marlin **W8A16** on A100, not native W8A8.
- AWQ findings bound to **tested community artifacts**.
- GPQA Holm-6 vs Holm-18 joint sensitivity is stated in abstract, results, discussion, conclusion, and appendix.
- Cost language is **aggregate hybrid Cost-of-Pass proxy**; Conditions A/B are joint subset+regime contrasts; Figure 3 is a scatter, not Pareto.
- CPU `--check` path exists for reanalysis, emit markdown, confirmation, modal SHA, item-level, confirmation raw files.
- `configs/models/` is a warning README; historical JSON is in `configs/legacy_models/`.
- Limitations list AIME, LiveCodeBench, 32B, Hopper W8A8, energy, production batching, traces-not-released.

### B. What changed after the previous audit (`FINAL_PUBLICATION_AUDIT.md`)

This pass (uncommitted until you ask):

- C2 reworded from “Ranking instability” to **empirical characterization of ranking instability across deployment estimands**.
- GPQA drop described as **distributed** (75/198 any-seed flips; 0 all-seed flips) from existing CPU item-level JSON.
- `REPRODUCE.md` split: tables vs GPU campaign; env table; expected PASS lines.
- `validate_runtime_manifest.py --check` and `check_manuscript_numbers.py --check`.
- Manifest: NVIDIA driver **UNRECORDED**; lockfile; torch note.
- Config READMEs: `configs/`, `configs/serving/`, `configs/cells/`; YAML header on `vllm.yaml` (still pins 0.8.5 as history).
- CI steps for the new validators; `SUBMISSION_REVIEW.md` required-file gate.
- README no longer claims `paper-major-revision` / `d707e44` as the live canonical branch.

### C. Recommendations still missing (human / optional engineering)

1. **Generate TeX tables from markdown** (or freeze a cell-by-cell tex-vs-md checker). Needle check is a subset.
2. **Venue decision.** JSS vs TMLR vs FGCS. Table 1 placement depends on this.
3. **CRediT confirmation** with advisors. Current paragraph is a conservative draft.
4. **Cut `paper-v1.0-submission` after the commit that includes this zip.**
5. **Remote GitHub Actions** after push (local `--check` is not CI).
6. NVIDIA driver version if it can be recovered from an HPC log. Do not invent it.
7. Align or archive the stale `paper-major-revision` remote so reviewers cloning that branch do not get an old paper.

### D. What could still cause rejection

| Risk | Why a reviewer might stop |
|------|---------------------------|
| “Incremental / no method” | Measurement study with two 7B/8B families and no AIME/LiveCodeBench. |
| Artifact confusion | `configs/serving/vllm.yaml` still says 0.8.5 (now labeled). `campaign_cells.json` still lists scratch paths. |
| Hand-copied tables | A single TeX typo would be a reproducibility fail. |
| Traces not released | Cannot audit loops, `finish_reason`, or CoT. |
| Unexplained FP8 Cond B timing | Five-rep mixture; cause unknown. Honest, but systems reviewers may call the −36% mean unstable. |
| Holm-18 vs Holm-6 | A stats reviewer can demand the joint family as primary and then Qwen AWQ GPQA is n.s. |
| Scope | A100-only, vLLM 0.7.0, W8A16, no energy, no 32B. |

None of these require new GPU runs. Several require a human decision, not more prose.

---

## Phase 2 — Paper-to-code consistency

Checked against compact JSON, `major_revision_tables.md`, `revision_reanalysis_report.json`, confirmation report, item-level JSON, `runtime_manifest.json`.

| Claim | Source | Match? |
|-------|--------|--------|
| 56,408 completions | Sum of 88 compact JSON `n` | **Yes** (20,000+31,656+4,752) |
| 88 runs | File counts 40+24+24 | **Yes** |
| MATH-500 seed grid | TeX Table `tab:headline_results` vs frozen md | **Yes** (hand-transcribed) |
| Llama AWQ MATH −2.76 | Frozen md `0.0000` / Holm yes | **Yes**; TeX prints \(p<0.001\) |
| Llama AWQ GSM −1.57, \(p=0.002\) | md `0.0018` | **Yes** (rounded) |
| Qwen AWQ GPQA −5.56, \(p=0.007\) | md `0.0068` | **Yes** (rounded) |
| Holm-18 adj. \(p=0.109\) | md `0.1088` | **Yes** (rounded) |
| TOST ±1 pp MATH all fail | Frozen TOST flags | **Yes** |
| Qwen 4-bit RoM +6.33 / +6.88 | Frozen length table | **Yes**; abstract uses 6.3–6.9% |
| Both-OK CIs exclude 0 for Qwen AWQ/GPTQ | Frozen 2×2 | **Yes** |
| \(D\) diagnostic, not causal | Methods + mismatch table | **Yes** |
| 25 loops / 0 cap / 209 near-cap | `grid_totals` | **Yes** |
| Modal SHA `23e9ead0...` | `--check-artifact` | **Yes** |
| Cond A/B rankings + −45.9% | Confirmation `--check` | **Yes** |
| Hybrid \(C_{\mathrm{pass}}\), not Erol per-problem | Methods eq. + caption | **Yes** |
| FP8 Marlin W8A16 | TeX + manifest cells | **Yes** |
| FP8 Cond B timing mixture | Appendix reps table | **Yes**; cause unrecorded |
| GPQA 75/198 distributed | item-level JSON Qwen AWQ-4 | **Yes**; 0 all-seed flips |

**Known transcription policy (not a number error):** TeX rounds bootstrap \(p\) (0.0068→0.007, 0.1088→0.109). `emit --check` proves **markdown** regeneration, not **TeX = markdown**.

**Hidden assumptions (stated, but easy to miss):**

- Decoding seeds are **fixed**, not a random effect.
- Conditions A and B **confound** subset and serving regime.
- \$1.50/A100-h is a **scenario**.
- Compact JSON has **no** traces / `finish_reason` / token IDs.
- Manifest is **launcher-reconstructed**, not per-job `nvidia-smi` telemetry. Driver is UNRECORDED.

No frozen-table mismatch requiring a number change.

---

## Phase 3 — Priority-0 implemented this pass

1. **REPRODUCE.md** — CPU table checks vs GPU replay; versions; expected PASS lines; no overclaim.
2. **Artifact cleanup** — obsolete configs **kept** and labeled (`legacy_models/`, `serving/`, `cells/`, `campaign_cells*.json`). Nothing scientifically historical was deleted.
3. **Validation** — `validate_runtime_manifest.py --check`, `check_manuscript_numbers.py --check` (15 needles). CI runs both. README no longer points at a missing check.
4. **runtime_manifest.json** — model IDs/revisions, dataset SHAs, LightEval 0.8.0, vLLM 0.7.0, PyTorch 2.5.1, CUDA 12.4, A100, decoding flags, W8A16 notes, driver UNRECORDED.

---

## Phase 4 — Manuscript

- **C2** is a measurement contribution, not a slogan.
- **Title updated (later same-day pass).** “Beyond Accuracy” dropped; current title is *Measuring Evaluation-Target Instability in Quantized Reasoning Checkpoints*. See `SCIENTIFIC_AUDIT.md`.
  - Optional later: *One Stack, Many Rankings: Deployment Rankings of Quantized Reasoning Checkpoints Depend on the Estimand*.
- **Abstract** does not claim AWQ fails, FP8 wins, cost is guaranteed, or causal token inflation.
- **Limitations** already list the requested scope cuts.

---

## Phase 5 — CPU-only analyses (existing artifacts)

| Question | Possible? | Added to paper? |
|----------|-----------|-----------------|
| MATH flips BF16 vs quantized | Yes (`item_level_descriptive_report`) | **No extra section.** Counts duplicate the 2×2 \(n\) already in Table `tab:tokens`. MATH has **0** all-seed BF16✓/quant✗ items for every format — the interesting fact is already that flips are seed-unstable. |
| Length vs correctness | Yes. Incorrect traces are ~4–5× longer (Qwen BF16 3,256 vs 15,843). | **Already in the paper** (BF16 failure control + mismatch \(D\)). Repeating it would be filler. |
| Difficulty | Yes. Table `tab:difficulty` already shows Level 2 > Level 5 for BF16; Llama AWQ lowest on Levels 2–4. | **Already in the paper.** No compact item text to do a finer error taxonomy. |
| GPQA concentrated vs distributed | Yes. Qwen AWQ: **75/198** any-seed flips, **0** all-three-seed flips. | **One sentence added** in GSM8K/GPQA results. That is the only scientifically new wording. |

No further CPU figures were added.

---

## Phase 6 — Three-reviewer simulation

### Reviewer 1 — ML researcher

- **Likely outcome:** Weak accept / borderline reject for *incremental empirical study*.
- **Strongest criticism:** Two public 7B/8B families, MATH/GSM/GPQA only, no AIME or LiveCodeBench, no new method. Ranking instability is expected once you measure several estimands. Holm-18 knocks out the headline GPQA finding.
- **Exact fix (no GPU):** Keep Holm-6 as primary (already justified). Do **not** promote GPQA as a method failure. Optionally move Table 1 to appendix if the venue is TMLR. Do not add AIME without a new campaign.

### Reviewer 2 — Systems researcher

- **Likely outcome:** Weak accept if they value the pinned stack; reject if they wanted a serving paper.
- **Strongest criticism:** vLLM 0.7.0 eager on A100 W8A16 is not production Hopper W8A8 + CUDA graphs + continuous batching. Conditions A/B confound subset and concurrency. Qwen FP8 Cond B CV is huge; −36% mean is not a reliable systems number. VRAM is the 0.75 pool.
- **Exact fix:** Already in limitations. Do not average the five FP8 reps into a cleaner story. Do not call Figure 3 Pareto. Optional: put Cond B FP8 mean in the table with a footnote “mixture of three wall-clock regimes” (already in text/appendix).

### Reviewer 3 — Reproducibility / artifact

- **Likely outcome:** **Revise artifact** even if science is accepted.
- **Strongest criticism:** (1) TeX tables are not generated from code. (2) Full traces absent. (3) `configs/serving/vllm.yaml` still says 0.8.5. (4) Manifest is reconstructed, driver missing. (5) Branch docs disagreed with GitHub `main`.
- **Exact fix this pass:** REPRODUCE split; validators; YAML/README labels; README branch correction. **Remaining:** generate TeX from markdown, or a cell-by-cell checker; tag a snapshot; run Actions after push.

---

## Phase 7 — Final output

### 1. Completed this pass

- C2 wording; GPQA distributed-flip sentence; `\path` for manifest (overfull hbox).
- REPRODUCE.md, manifest fields, two `--check` scripts, CI hooks.
- Config labels (serving YAML comment, cells README, configs map).
- README live-branch correction.
- This report.

### 2. Files changed (this working tree)

- `paper/main.tex`, `paper/main.md`, `paper/main.pdf`, `paper/arxiv_source.zip`, `paper/ARTIFACT.md`
- `REPRODUCE.md`, `README.md`, `CHANGELOG.md`, `AGENTS.md`, `../AGENTS.md`
- `results/reports/runtime_manifest.json`
- `scripts/analysis/validate_runtime_manifest.py`, `scripts/analysis/check_manuscript_numbers.py`
- `.github/workflows/ci.yml`, `.gitignore`
- `configs/README.md`, `configs/serving/README.md`, `configs/serving/vllm.yaml`, `configs/cells/README.md`
- `SUBMISSION_REVIEW.md`

Frozen campaign JSON and `major_revision_tables.md` **numbers** were not edited.

### 3. Remaining risks

1. Hand-transcribed TeX (highest remaining engineering risk).
2. Traces / `finish_reason` not released.
3. CRediT and venue unset.
4. `paper-major-revision` remote stale.
5. Unexplained FP8 Cond B wall-clock regimes.
6. Scope (7B/8B, A100, vLLM 0.7.0, no AIME/LCB/energy).
7. This zip/PDF is **not on GitHub** until you commit and push. Current zip SHA256 `9ef0185da05e73e44f000b1963ec0bc0590e1ac12168362213dddca285af8097`.

### 4. Score: **78 / 100**

| Axis | Score | Note |
|------|------:|------|
| Claim discipline | 18/20 | Hedged; C2 still a finding, now labeled as characterization. |
| Number integrity | 18/20 | Frozen checks pass; TeX is transcribed. |
| Reproducibility of tables | 16/20 | Stdlib `--check` is real; GPU replay is not. |
| Artifact hygiene | 12/20 | Labeled, not removed; 0.8.5 YAML still present. |
| Scope / generality | 8/20 | Honest limitations; thin model/task coverage. |
| Systems evidence | 6/10 | Confirmation protocol is careful; FP8 timing is a hole. |

### 5. Recommendation

**B. Needs minor fixes.**

Do **not** submit tonight as-is if the venue has an artifact evaluation. Do **not** reopen the GPU campaign to chase AIME or Hopper.

Submit after:

1. You confirm CRediT and venue.
2. You commit this tree, rebuild/upload the zip SHA in `paper/ARTIFACT.md`, push `main`, confirm Actions green.
3. Optional but high-leverage: a TeX-vs-`major_revision_tables.md` cell checker (still no GPU).

If those three are done, the paper is a legitimate measurement study with internally consistent numbers. Acceptance is not guaranteed: the contribution is protocol + ranking disagreement on eight public checkpoints, which some ML reviewers will call incremental.
