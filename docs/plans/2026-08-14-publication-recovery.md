# Paper 1 Publication-Recovery Plan — 2026-08-14

**Status:** **Historical / completed.** The recovery campaign finished. Major revision is frozen on `paper-major-revision` (`d707e44`, 2026-08-17). GPU work is closed. Do not execute this plan.

**Scientific decision:** [Publication Readiness Audit](../PUBLICATION_READINESS.md) (read the 2026-08-17 freeze banner first)

**Goal (then):** turn the valid FP8 replication into a matched, reproducible, statistically defensible contribution without spending GPU time on a confounded broad grid.

---

## 1. Definition of done

Paper 1 is ready for supervisor/submission review only when it has:

1. A clean checkout that deterministically recreates the evaluation stack.
2. Same-stack BF16/FP8/AWQ4/GPTQ4 comparisons with pinned model, data, prompt, and decoding provenance.
3. Three-seed pilot evidence and five-seed headline evidence.
4. Valid correctness, termination/degeneration, calibration/selective-risk, latency/VRAM, and cost-per-correct measurements.
5. Paired statistical analysis and a manual extraction/trace audit.
6. A contribution selected by an explicit novelty gate.
7. A complete draft and release manifest.

The current jobs 96100/96101 satisfy none of items 2–7 alone; they are input evidence for Phase 0/1.

---

## 2. Non-negotiable rules

- **No broad b03/b04/b01–b09 launch before Phase 0 passes.**
- Do not compare QRM Protocol R numbers directly with old `qreason` Protocol P numbers.
- Do not call A100 Marlin fallback “native FP8” or “W8A8 FP8 compute.”
- Do not infer truncation without capturing `finish_reason`; label old cases “likely near-cap.”
- Do not compute Brier/ECE/AURC from parse success or another invalid confidence proxy.
- Do not use wall-clock Slurm duration as latency evidence.
- Do not use accuracy as a quality gate that could suppress a real negative result; use integrity and pathology warnings separately.
- No new model family, 14B/32B/70B, GGUF, KV-cache quantization, or LiveCodeBench expansion until the contribution gate selects a track.

---

## 3. Protocols

### Protocol R — replication only

Purpose: reproduce QRM/model-card correctness and provide an external sanity anchor.

- Stack: pinned `qrm-official` environment.
- Prompt: exact QRM MATH-500 prompt/chat template.
- Decoding: temperature 0.6, top-p 0.95, max new tokens 32,768, no repetition penalty.
- Seed: 42 for historical comparability; 42/43/44 only when used in a matched pilot.
- Claims: correctness/trace health only unless instrumentation is added and validated.

### Protocol P1-2026-08 — publication protocol

Purpose: all causal quantization, reliability, calibration, and systems comparisons.

- One frozen prompt profile per task.
- Same engine version, engine mode, model length, scheduler settings, GPU-memory target, and scoring code across compared cells.
- Models: Qwen-7B and Llama-8B headline anchors.
- Formats: BF16, FP8-checkpoint/Marlin, AWQ4, GPTQ4.
- Pilot seeds: 42, 43, 44.
- Headline seeds: 42, 43, 44, 45, 46.
- MATH-500 first. GPQA-Diamond and GSM8K only after the contribution gate.
- One-completion pass@1 plus a predeclared maj@5/calibration subset.

Seed 0 archives remain historical engineering evidence; they are not mixed into Protocol P1-2026-08 aggregates.

---

## 4. Phase plan and gates

### Phase 0 — repair reproducibility and observability

**GPU use:** none except a final 3-question smoke test.

- [ ] Pin the external QRM repository and every submodule in a tracked manifest.
- [ ] Convert the three required uncommitted QRM changes into tracked patch files or maintained project code.
- [ ] Make setup recreate the environment and patches from a clean clone; test in a temporary directory.
- [ ] Add output fields: output token IDs/count, `finish_reason`, stop reason, prompt/completion token counts, request start/end/latency, model/data/prompt revisions, stack ID, and config hash.
- [ ] Add phrase/sentence-loop, repeated n-gram, near-cap, parse-failure, and pathological-length diagnostics.
- [ ] Split validation into `integrity_pass`, `quality_warnings`, and `publication_gate`.
- [ ] Add controlled peak-VRAM and GPU power/energy collection, with explicit unavailable/error states instead of zero-filled telemetry.
- [ ] Validate confidence provenance; support sample-consistency confidence for maj@5 and logprob confidence only when semantically valid.
- [ ] Add unit/integration tests for all new schema and validation behavior.
- [ ] Run a clean-checkout CPU preflight and one 3-question GPU smoke per model/format family.

**Gate P0:** clean checkout reproduces the environment; tests pass; smoke rows contain complete provenance and finish/telemetry fields; no uncommitted dependency patch is required.

### Phase 1 — matched BF16/FP8 reconstruction

**Question:** Does FP8 differ from BF16 when every other experimental factor is held fixed?

Run Protocol P1-2026-08 on MATH-500:

| Model | Formats | Seeds | Cells |
|-------|---------|-------|------:|
| Qwen-7B | BF16, FP8 | 42 | 2 |
| Llama-8B | BF16, FP8 | 42 | 2 |

Rerun FP8 after instrumentation; do not reuse 96100/96101 as the matched half of the causal comparison.

**Required report:** pass@1, paired discordance table, Wilson and paired-bootstrap intervals, McNemar test, completion-length distribution, cap/loop/parse rates, latency distribution, peak VRAM, energy availability, and cost-per-correct assumptions.

**Gate P1:** all four cells pass integrity; no protocol mismatch; manual review of every flagged trace; differences are reported regardless of direction.

### Phase 2 — discriminating three-seed pilot

Run MATH-500 under the same protocol:

| Models | Formats | Seeds | Cells |
|--------|---------|-------|------:|
| Qwen-7B, Llama-8B | BF16, FP8, AWQ4, GPTQ4 | 42, 43, 44 | 24 |

Add maj@5 for a predeclared stratified subset sufficient to estimate sample-consistency confidence. Do not launch breadth tasks yet.

**Gate P2:** produce a blinded configuration table and answer:

1. Is any quantization effect reproducible across seeds beyond pass@1?
2. Does format ordering change for reliability, calibration, or cost?
3. Are stack/pathology effects larger than quantization effects?
4. Is the observed effect large and novel enough to justify the final grid?

### Phase 3 — contribution-selection gate

Choose exactly one primary paper track.

| Evidence from Phase 2 | Decision |
|-----------------------|----------|
| Stable quantization reliability/calibration/cost effect | **Track A:** quantization reliability–cost frontier |
| Small quantization effect but large reproducible stack effect | **Track B:** controlled serving-stack transfer study |
| Neither effect survives seeds/audit | Stop expansion; release a replication/negative-results artifact and re-scope with supervisor |

For Track B, build an incremental stack ladder that changes one factor at a time—engine version/mode, Transformers version, prompt wrapper, scheduler/memory setting, and decoding penalty. A bundled old-stack/new-stack comparison is not causal evidence.

**Gate P3:** supervisor approves the selected RQs, primary endpoints, cell matrix, and target venue before more GPU allocation.

### Phase 4 — confirmatory execution

For Track A, headline MATH-500 cells become:

| Models | Formats | Seeds | Cells |
|--------|---------|-------|------:|
| Qwen-7B, Llama-8B | BF16, FP8, AWQ4, GPTQ4 | 42–46 | 40 total, including Phase 2 |

Then add breadth only if P3 approves it:

- GPQA-Diamond and GSM8K: Qwen-7B × four formats × seeds 42–44.
- Llama breadth only if the MATH interaction is architecture-dependent.
- GPTQ3 only as a predeclared failure-boundary appendix.
- LiveCodeBench only after an extraction/version gate; otherwise defer.

For Track B, replace this matrix with the supervisor-approved controlled stack ladder; retain BF16/FP8 and paired prompts.

**Gate P4:** every primary cell has complete rows, hashes, telemetry, audit output, and five headline seeds; no silent resume/provenance mixture.

### Phase 5 — frozen analysis

- [ ] Predeclare primary comparisons before unblinding the final aggregate.
- [ ] Use paired McNemar tests for pass@1 and paired/bootstrap intervals over problems.
- [ ] Report seed-level variation separately from problem-level uncertainty.
- [ ] Use Holm correction for the finite primary comparison family.
- [ ] Report Brier/ECE/AURC only with a validated confidence source.
- [ ] Report latency p50/p95/p99, throughput, peak VRAM, energy/Joules-per-correct when measurable, and cost-per-correct with explicit pricing assumptions.
- [ ] Audit at least 200 stratified traces plus every flagged cap/loop/parse case.
- [ ] Preserve negative and null results; do not change gates after seeing model labels.

**Gate P5:** one immutable analysis bundle recreates every table and figure from raw manifests.

### Phase 6 — manuscript and artifact

- [ ] Complete `paper/main.md` with claim-evidence mapping.
- [ ] Add a limitations table covering A100 FP8 fallback, benchmark scope, confidence construction, shared-cluster measurements, and stack specificity.
- [ ] Release code, configs, patch series, hashes, environment lock, aggregate tables, and permitted traces/manifests.
- [ ] Run a final literature update and venue-policy check.
- [ ] Obtain internal/supervisor review before arXiv or journal submission.

**Gate P6:** no placeholder sections, no unsupported claim, all headline numbers trace to a clean artifact.

---

## 5. Primary endpoints

| Family | Primary endpoint | Required supporting fields |
|--------|------------------|----------------------------|
| Correctness | pass@1; maj@5 subset | exact prompt/gold, extraction result, paired problem ID |
| Reliability | cap/stop/loop/parse rates | `finish_reason`, token IDs/count, n-gram/phrase diagnostics |
| Calibration | Brier, ECE, AURC | validated confidence value and source |
| Stability | seed variance and rank reversals | seeds 42–46, identical protocol hash |
| Systems | latency distribution, throughput, peak VRAM | controlled benchmark mode, warm-up, request timing |
| Economics | cost-per-correct; optional Joules-per-correct | measured runtime/energy plus explicit hardware price model |

Accuracy is not replaced by these endpoints; it remains the correctness anchor.

---

## 6. Immediate work queue

Execute in this order:

1. **P0.1:** create tracked QRM pin/patch manifest and prove clean recreation.
2. **P0.2:** extend the result schema with finish reason, token IDs/counts, timing, and exact provenance.
3. **P0.3:** strengthen loop/termination validation and separate integrity from quality.
4. **P0.4:** implement controlled VRAM/power/latency capture and tests.
5. **P0.5:** run tiny smoke cells; review their rows manually.
6. **P1:** launch only the four matched BF16/FP8 seed-42 cells.
7. Review P1 before scheduling the 24-cell pilot.

Do not submit another large job merely because the queue is empty. The next allowed GPU action is the P0 smoke after the instrumentation and clean-reproduction gate are complete.

---

## 7. Progress ledger

| Phase | Status | Evidence |
|-------|--------|----------|
| Audit | **Complete** | `docs/PUBLICATION_READINESS.md` |
| P0 reproducibility/observability | Not started | — |
| P1 matched BF16/FP8 | Blocked by P0 | — |
| P2 three-seed pilot | Blocked by P1 | — |
| P3 contribution gate | Blocked by P2 | — |
| P4 confirmatory grid | Blocked by P3 | — |
| P5 frozen analysis | Blocked by P4 | — |
| P6 manuscript/artifact | Outline only | `paper/main.md` |
