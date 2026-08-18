# Hardware policy — J1 / J2 / J3

**Frozen machine roles:** 2026-07-01 · **measurement amendment:** 2026-08-14 · **GPU freeze:** 2026-08-17

One page so docs do not contradict each other.

> **Freeze (2026-08-17):** Paper 1 publication numbers are complete on HPC A100. Do not submit new J1 GPU jobs.

---

## Summary

| Paper | Primary hardware | Secondary / transfer |
|-------|------------------|----------------------|
| **J1** | HPC 2× A100 80 GB | None for paper numbers |
| **J2** | HPC 2× A100 | Same stack; server-mode profiling later |
| **J3** | HPC 2× A100 (controlled primary) | **RTX 5080 + llama.cpp** (local transfer layer only) |

---

## J1 — compression reliability (now)

- **All manuscript numbers** for 7B/8B/1.5B come from **PARAM Rudra HPC**.
- **RTX 5080 is not used** for J1 publication runs.
- 5080 may be used for **local smoke/debug** only (pipeline proof, not cited).

Reason: 7B BF16 and 8B BF16 do not fit 16 GB for full MATH-500 at paper decoding settings.

### A100 FP8 and systems-measurement boundary

- A100 does not provide native FP8 tensor-core execution. The audited FP8 checkpoints ran through vLLM's weight-only Marlin fallback; report the checkpoint format and runtime kernel separately.
- Do not infer speed, throughput, energy, or cost from Slurm elapsed time. Jobs 96100/96101 ran on different nodes and Llama logged 900+ KV-cache recomputations.
- Controlled systems runs require warm-up, per-request latency, scheduler/preemption logs, peak VRAM, and explicit power/energy availability.
- Zero or missing Slurm energy accounting means “unavailable,” not zero Joules.

See [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md) and [the recovery plan](plans/2026-08-14-publication-recovery.md).

---

## J2 — acceleration (later)

- HPC only until method pilot gate passes.
- Requires server-mode metrics (concurrency, p50/p95/p99) — not the same as J1 offline batch latency.

---

## J3 — Indic deployment (later)

**Option A (chosen):** A100 primary + RTX transfer section.

| Layer | Hardware | Role |
|-------|----------|------|
| Primary | HPC A100 + vLLM | Quality, compression, latency, cost — main claims |
| Transfer | RTX 5080 + llama.cpp/GGUF | Bounded “datacenter → local edge” comparison |

J3 may cite RTX results **only** in the transfer section, with:

- Pinned llama.cpp commit
- Warm-up runs
- Repeated trials
- Exact GPU/driver reporting
- Separate conclusions from A100 primary results

**5080 is not retired for the whole thesis** — only for J1/J2 paper numbers.

---

## Archive storage

- **Git:** code, configs, summaries, manifests, checksums, small fixtures.
- **Not in ordinary git long-term:** raw JSONL, full scored archives, large Parquet, model weights.
- Use HPC scratch + optional Zenodo/HF Dataset release with manifest.

HPC autopush commits **manifests, metadata, summaries, and logs only** — not raw/scored JSONL or `_backup/` mirrors (size, gated benchmark content, git history bloat).

---

## Doc alignment

If any doc says “5080 retired” without qualification, read it as: **retired for J1 publication**, not retired for J3 local transfer.

See also: [MODEL_SCOPE_DECISION.md](MODEL_SCOPE_DECISION.md), [HPC_2A100_PLAN.md](HPC_2A100_PLAN.md).
