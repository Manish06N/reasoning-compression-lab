# Hardware policy — J1 / J2 / J3

**Frozen:** 2026-07-01

One page so docs do not contradict each other.

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

HPC autopush of small summary commits may exist historically; do not treat full raw archives as permanent git assets.

---

## Doc alignment

If any doc says “5080 retired” without qualification, read it as: **retired for J1 publication**, not retired for J3 local transfer.

See also: [MODEL_SCOPE_DECISION.md](MODEL_SCOPE_DECISION.md), [HPC_2A100_PLAN.md](HPC_2A100_PLAN.md).
