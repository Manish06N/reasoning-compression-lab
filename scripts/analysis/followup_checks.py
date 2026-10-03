"""CPU-only follow-up checks after the second referee report (no GPU, no frozen result changed).

  A. Serving tails: for every (family, format, condition) the measured subset mean length versus the campaign completions of the
     same prompts; for Llama FP8 Condition A, how many very long requests the latency percentiles imply and how often the campaign
     produces that many (count model), alongside the resampling percentile (which truncates the tail: five completions per prompt).
  B. tok/s aggregation: campaign-length seconds and rank-1 frequencies with tok/s = total tokens / total time (the aggregate that
     matches the reported GPU-s/q and hybrid dollars) versus the mean of per-repeat ratios.
  C. Truncation-class differences (Qwen 4-bit): item-bootstrap p-values and Holm adjustment over the six contrasts.
  D. BF16-correct conditional minus placebo versus the unconditional change reweighted by per-item BF16 accuracy.
  E. Minimum detectable effects without and with the Holm-6 first-step multiplier.
  F. Difficulty trend of the Llama AWQ-4 drop (post hoc, linear in level) with an item-level bootstrap interval.

Needs numpy and the cached MATH-500 dataset for F.   python scripts/analysis/followup_checks.py
Writes results/reports/followup_checks.json.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FAM = {"Qwen": "DeepSeek-R1-Distill-Qwen-7B", "Llama": "DeepSeek-R1-Distill-Llama-8B"}
MS = {"Qwen": "Qwen-7B", "Llama": "Llama-8B"}
FMT = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]
B = 10_000
RAW = REPO / "results/measured_serving_confirmation/raw"


def cell(f, m):
    return FAM[f] if m == "BF16" else f"{FAM[f]}-{m}"


def det(bench, f, m, s):
    tag = {"math500": "math500_n500", "gsm8k": "gsm8k_n1319", "gpqa": "gpqadiamond_n198"}[bench]
    return json.loads((REPO / "results" / bench / f"{cell(f, m)}_{tag}_seed{s}.json").read_text())[
        "details"
    ]


def acc(bench, f, m, seeds):
    return np.array(
        [[float(x["extractive_match"] == 1.0) for x in det(bench, f, m, s)] for s in seeds]
    )


def tok(f, m):
    return np.array(
        [[float(x["completion_tokens"]) for x in det("math500", f, m, s)] for s in range(42, 47)]
    )


def weights(n, seed=0):
    rng = random.Random(seed)
    W = np.zeros((B, n))
    for b in range(B):
        for _ in range(n):
            W[b, rng.randrange(n)] += 1
    return W


def ci95(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def reps(f, m, c):
    return [json.loads(p.read_text()) for p in sorted(RAW.glob(f"{MS[f]}_{m}_rep*_cond{c}.json"))]


def main() -> int:
    out: dict = {}
    W = weights(500)
    T = {(f, m): tok(f, m) for f in FAM for m in FMT}
    A = {(f, m): acc("math500", f, m, range(42, 47)) for f in FAM for m in FMT}
    subs = {
        c: np.array(
            [
                int(x["campaign_index"])
                for x in json.loads(
                    (
                        REPO
                        / f"results/measured_serving_confirmation/condition_{c.lower()}_subset.json"
                    ).read_text()
                )
            ]
        )
        for c in "AB"
    }

    # ---- A. serving tails ------------------------------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    A_out = {}
    for f in FAM:
        for m in FMT:
            for c in "AB":
                rows = subs[c]
                rr = reps(f, m, c)
                measured = float(np.mean([r["mean_output_tokens_per_req"] for r in rr]))
                draws = np.array(
                    [T[(f, m)][rng.integers(0, 5, len(rows)), rows].mean() for _ in range(5000)]
                )
                A_out[f"{f}|{m}|{c}"] = {
                    "measured_mean_tokens": measured,
                    "campaign_mean_on_subset_items": float(T[(f, m)][:, rows].mean()),
                    "ratio_measured_to_campaign_same_prompts": measured
                    / float(T[(f, m)][:, rows].mean()),
                    "resample_percentile": 100 * float((draws < measured).mean()),
                }
    out["subset_length_vs_campaign"] = A_out
    # Llama FP8 Condition A: very long requests implied by latency percentiles, and how often the campaign produces them
    rows = subs["A"]
    Tl = T[("Llama", "FP8")]
    rr = reps("Llama", "FP8", "A")
    implied = [
        {
            "tok_s": r["output_tokens_per_second"],
            "p90_tokens": r["latency_p90_sec"] * r["output_tokens_per_second"],
            "p95_tokens": r["latency_p95_sec"] * r["output_tokens_per_second"],
        }
        for r in rr
    ]
    thr = 15000
    kk = (Tl >= thr).sum(0)
    m_ = kk.mean() / 5
    v_ = kk.var()
    n = 5
    ab = max((m_ * (1 - m_) * n / max(v_, 1e-9) - 1) / (n - 1), 0.5)
    a_, b_ = m_ * ab, (1 - m_) * ab
    p_i = ((Tl[:, rows] >= thr).sum(0) + a_) / (n + a_ + b_)
    dist = np.zeros(21)
    dist[0] = 1.0
    for p in p_i:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    pool = float((Tl >= thr).mean())
    binom = [
        sum(math.comb(20, j) * pool**j * (1 - pool) ** (20 - j) for j in range(k, 21))
        for k in (2, 3, 4)
    ]
    out["llama_fp8_condA_tail"] = {
        "latency_implied_tokens": implied,
        "threshold_tokens": thr,
        "campaign_rate_ge_threshold": pool,
        "per_seed_count_on_subset": [int((Tl[s][rows] >= thr).sum()) for s in range(5)],
        "p_at_least_k_long_beta_binomial": {str(k): float(dist[k:].sum()) for k in (2, 3, 4)},
        "p_at_least_k_long_pooled_binomial": {str(k): float(x) for k, x in zip((2, 3, 4), binom)},
        "note": "latency percentiles of 20 sequential requests; tokens = latency x tok/s; counts of >=15k-token completions in the campaign",
    }

    # ---- B. tok/s aggregation ----------------------------------------------------------------------------------------------
    idx = np.random.default_rng(1).integers(0, 500, (B, 500))
    out["campaign_length_seconds"] = {}
    for f in FAM:
        for c in "AB":
            rr_ = {m: reps(f, m, c) for m in FMT}
            for var in ("mean_of_ratios", "total_tokens_over_total_time"):
                ts = {
                    m: (
                        float(np.mean([r["output_tokens_per_second"] for r in rr_[m]]))
                        if var == "mean_of_ratios"
                        else sum(r["total_output_tokens"] for r in rr_[m])
                        / sum(r["elapsed_seconds"] for r in rr_[m])
                    )
                    for m in FMT
                }
                sec = {m: float(T[(f, m)].mean() / ts[m] / A[(f, m)].mean()) for m in FMT}
                cost = np.stack(
                    [
                        T[(f, m)].mean(0)[idx].mean(1) / ts[m] / A[(f, m)].mean(0)[idx].mean(1)
                        for m in FMT
                    ],
                    1,
                )
                p1 = np.bincount(cost.argmin(1), minlength=4) / B
                rank = {m: 1 + sum(sec[g] < sec[m] for g in FMT) for m in FMT}
                out["campaign_length_seconds"][f"{f}|{c}|{var}"] = {
                    "tok_s": ts,
                    "seconds_per_correct": sec,
                    "rank": rank,
                    "p_rank1": dict(zip(FMT, [float(x) for x in p1])),
                }

    # ---- C. truncation-class differences (cap band = completion within 16 tokens of its per-item cap, frozen tokenizers) -------
    rows_ = json.loads((REPO / "results/reports/audit_checks.json").read_text())["tokenizers"][
        "cap_band_rows"
    ]
    trunc = {(f, m): np.zeros((5, 500)) for f in FAM for m in FMT}
    for fam, fmt, seed, item in rows_:
        trunc[(fam, fmt)][seed - 42, item] = 1.0
    ps = {}
    for f in FAM:
        for m in FMT[1:]:
            # the paper's class counts INCORRECT completions that hit the cap band (a truncated completion can still be scored correct)
            d = (trunc[(f, m)] * (1 - A[(f, m)])).mean(0) - (
                trunc[(f, "BF16")] * (1 - A[(f, "BF16")])
            ).mean(0)
            bs = (W @ d) / 500
            p = float(min(1.0, 2 * min((bs >= 0).mean(), (bs <= 0).mean())))
            ps[f"{f}|{m}"] = {
                "diff_pp": float(100 * d.mean()),
                "ci95_pp": [100 * x for x in ci95(bs)],
                "p": p,
            }
    keys = list(ps)
    adj = holm([ps[k]["p"] for k in keys])
    for k, a in zip(keys, adj):
        ps[k]["holm6"] = a
    out["truncation_difference_tests"] = {
        "definition": "incorrect completions in the cap band (the truncated class of Table failure; item-level paired bootstrap vs BF16, stdlib Random(0) stream); Holm over the six contrasts",
        "contrasts": ps,
    }

    # ---- D. conditional minus placebo vs accuracy-weighted unconditional change ----------------------------------------------
    out["estimand_check"] = {}
    for f in FAM:
        Tb = T[(f, "BF16")]
        Cb = A[(f, "BF16")].astype(bool)
        a = Cb.mean(0)
        for m in FMT[1:]:
            dI = T[(f, m)].mean(0) - Tb.mean(0)
            w = (W @ (a * dI)) / (W @ a)
            out["estimand_check"][f"{f}|{m}"] = {
                "unconditional_delta": float(dI.mean()),
                "accuracy_weighted_delta": float((a * dI).sum() / a.sum()),
                "accuracy_weighted_ci95": ci95(w),
            }

    # ---- E. minimum detectable effects --------------------------------------------------------------------------------------
    z_unadj = 1.959964 + 0.841621
    z_holm6 = 2.637 + 0.841621  # alpha/6 two-sided = 0.008333 -> z = 2.638
    out["mde"] = {}
    for bench, seeds, n in (
        ("math500", range(42, 47), 500),
        ("gsm8k", range(42, 45), 1319),
        ("gpqa", range(42, 45), 198),
    ):
        ses = []
        for f in FAM:
            b = acc(bench, f, "BF16", seeds).mean(0)
            for m in FMT[1:]:
                ses.append(
                    float((acc(bench, f, m, seeds).mean(0) - b).std(ddof=1) / np.sqrt(n) * 100)
                )
        out["mde"][bench] = {
            "paired_se_pp_range": [min(ses), max(ses)],
            "mde_unadjusted_pp_range": [z_unadj * min(ses), z_unadj * max(ses)],
            "mde_holm6_first_step_pp_range": [z_holm6 * min(ses), z_holm6 * max(ses)],
            "multipliers": {"unadjusted": z_unadj, "holm6_first_step": z_holm6},
        }

    # ---- F. difficulty trend (post hoc) -------------------------------------------------------------------------------------
    os.environ.setdefault("HF_HOME", str(REPO / "hf_cache"))
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceH4/MATH-500", split="test", revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    )
    by_q = {r["problem"].strip(): int(r["level"]) for r in ds}
    USER, ASSIST = "<｜User｜>", "<｜Assistant｜>"
    SUF = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    raw = json.loads(
        (
            REPO
            / "outputs-hpc-campaign-2026-08-14/inference"
            / f"{cell('Qwen', 'BF16')}-seed42"
            / "MATH-500.jsonl"
        ).read_text()
    )

    def q_of(fp):
        s = fp.split(USER, 1)[1].split(ASSIST, 1)[0]
        return (s[: -len(SUF)] if s.endswith(SUF) else s).strip()

    lv = np.array([by_q[q_of(r["full_prompt"])] for r in raw], float)
    out["difficulty_trend"] = {}
    g = np.random.default_rng(2)
    for f in FAM:
        for m in FMT[1:]:
            d = (A[(f, m)].mean(0) - A[(f, "BF16")].mean(0)) * 100
            slope = float(np.polyfit(lv, d, 1)[0])
            bs = []
            for _ in range(5000):
                ii = g.integers(0, 500, 500)
                bs.append(np.polyfit(lv[ii], d[ii], 1)[0])
            out["difficulty_trend"][f"{f}|{m}"] = {
                "slope_pp_per_level": slope,
                "ci95": ci95(np.array(bs)),
            }
    (REPO / "results/reports/followup_checks.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote results/reports/followup_checks.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
