"""CPU-only sensitivities added after the second independent review (no GPU, no frozen result changed).

  1. Two-way (items x seeds) bootstrap and a seed-level paired t-test for all 18 pass@1 contrasts.
     The primary intervals resample items only and are conditional on the seeds that were run.
  2. Minimum detectable effect (80% power, two-sided alpha = 0.05) from the paired item-level SE.
  3. MATH-500 paired token deltas: mean, median and 10%-trimmed mean of the seed-averaged per-item
     delta, with item-level 95% CIs. The mean is dominated by a few very long traces.
  4. Subset-draw simulation for the Condition A serving claim: random 20-prompt subsets stratified
     4 per MATH level (as in the confirmation protocol), one random campaign seed per format,
     measured Condition A tok/s held fixed. Reports how often the subset cost order matches the
     campaign-length order of Table 16.

Writes results/reports/review_additions.json. Needs the cached MATH-500 dataset (same as
q1_revision_analyses.py). Usage:  python scripts/analysis/review_additions.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FAMILIES = {"Qwen-7B": "DeepSeek-R1-Distill-Qwen-7B", "Llama-8B": "DeepSeek-R1-Distill-Llama-8B"}
FORMATS = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]
QUANT = FORMATS[1:]
BENCH = {
    "math500": (500, range(42, 47)),
    "gsm8k": (1319, range(42, 45)),
    "gpqa": (198, range(42, 45)),
}
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
RAW_MATH = ("outputs-hpc-campaign-2026-08-14", "MATH-500.jsonl")
USER, ASSIST = "<｜User｜>", "<｜Assistant｜>"
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
B, CHUNK, RNG_SEED, N_SIM = 10_000, 500, 0, 20_000


def cell(family: str, fmt: str) -> str:
    return FAMILIES[family] if fmt == "BF16" else f"{FAMILIES[family]}-{fmt}"


def load(bench: str, family: str, fmt: str):
    n, seeds = BENCH[bench]
    tag = {"gpqa": "gpqadiamond"}.get(bench, bench)
    A, T = [], []
    for s in seeds:
        d = json.loads(
            (REPO / "results" / bench / f"{cell(family, fmt)}_{tag}_n{n}_seed{s}.json").read_text()
        )["details"]
        assert len(d) == n
        A.append([float(x["extractive_match"]) for x in d])
        T.append([float(x["completion_tokens"]) for x in d])
    return np.array(A), np.array(T)  # seeds x items


def ci(v, q=(2.5, 97.5)):
    return [float(x) for x in np.percentile(v, q)]


def boot_items(x: np.ndarray, rng, stat=np.mean):
    n = len(x)
    out = []
    for _ in range(B // CHUNK):
        idx = rng.integers(0, n, (CHUNK, n))
        out.append(stat(x[idx], axis=1))
    return np.concatenate(out)


def trimmed(x: np.ndarray, axis=1, frac=0.10):
    x = np.sort(x, axis=axis)
    k = int(frac * x.shape[axis])
    return x[:, k : x.shape[axis] - k].mean(axis=axis)


def two_way(D: np.ndarray, rng) -> list[float]:
    s, n = D.shape
    out = np.empty(B)
    for b in range(B):
        out[b] = D[np.ix_(rng.integers(0, s, s), rng.integers(0, n, n))].mean()
    return [round(v * 100, 2) for v in ci(out)]


def paired_t_p(per_seed: np.ndarray) -> float:
    from math import sqrt

    from scipy import stats

    k = len(per_seed)
    t = per_seed.mean() / (per_seed.std(ddof=1) / sqrt(k))
    return float(2 * stats.t.sf(abs(t), k - 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(REPO / "hf_cache")))
    ap.add_argument("--out", type=Path, default=REPO / "results/reports/review_additions.json")
    args = ap.parse_args()
    os.environ["HF_HOME"] = args.hf_home
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    rng = np.random.default_rng(RNG_SEED)
    rep: dict = {
        "inputs": {
            "B": B,
            "rng_seed": RNG_SEED,
            "n_subset_draws": N_SIM,
            "note": "exploratory sensitivities; conditional on the archived generation seeds",
        }
    }

    # 1 + 2: pass@1 contrasts, two-way bootstrap, seed-level t, MDE
    two, mde = {}, {}
    for bench, (n, seeds) in BENCH.items():
        for fam in FAMILIES:
            base, _ = load(bench, fam, "BF16")
            for fmt in QUANT:
                A, _ = load(bench, fam, fmt)
                D = A - base
                per_seed = D.mean(1)
                item = D.mean(0)
                se = item.std(ddof=1) / np.sqrt(n)
                key = f"{bench}|{fam}|{fmt}"
                two[key] = {
                    "delta_pp": round(float(D.mean()) * 100, 2),
                    "two_way_ci95_pp": two_way(D, rng),
                    "per_seed_delta_pp": [round(float(v) * 100, 2) for v in per_seed],
                    "seed_level_paired_t_p": round(paired_t_p(per_seed), 4),
                    "n_seeds": len(seeds),
                }
                mde[key] = {
                    "paired_se_pp": round(float(se) * 100, 2),
                    "mde_80pct_pp": round(float(2.8016 * se) * 100, 2),
                }
    rep["two_way_bootstrap"] = two
    rep["minimum_detectable_effect"] = mde

    # 3: token delta location estimators (MATH-500, seed-averaged per item)
    tokens = {}
    levels_of = {}
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", revision=MATH500_REVISION)
    by_q = {r["problem"].strip(): int(r["level"]) for r in ds}
    root, fname = RAW_MATH
    for fam in FAMILIES:
        raw0 = json.loads(
            (REPO / root / "inference" / f"{cell(fam, 'BF16')}-seed42" / fname).read_text()
        )

        def q(p):
            text = p.split(USER, 1)[1].split(ASSIST, 1)[0]
            return (text[: -len(SUFFIX)] if text.endswith(SUFFIX) else text).strip()

        levels_of[fam] = np.array([by_q[q(r["full_prompt"])] for r in raw0])
        assert len(levels_of[fam]) == 500
    Tm = {fam: {fmt: load("math500", fam, fmt) for fmt in FORMATS} for fam in FAMILIES}
    for fam in FAMILIES:
        base = Tm[fam]["BF16"][1]
        for fmt in QUANT:
            T = Tm[fam][fmt][1]
            d = (T - base).mean(0)  # seed-averaged per-item delta
            tokens[f"{fam}|{fmt}"] = {
                "mean": [round(float(d.mean()), 1), [round(v, 1) for v in ci(boot_items(d, rng))]],
                "median": [
                    round(float(np.median(d)), 1),
                    [round(v, 1) for v in ci(boot_items(d, rng, np.median))],
                ],
                "trimmed10": [
                    round(float(trimmed(d[None, :])[0]), 1),
                    [round(v, 1) for v in ci(boot_items(d, rng, trimmed))],
                ],
            }
    rep["token_delta_location"] = tokens

    # 4: stratified subset-draw simulation
    meas = REPO / "results/measured_serving_confirmation/raw"
    toks = {
        fam: {
            fmt: float(
                np.mean(
                    [
                        json.loads(f.read_text())["output_tokens_per_second"]
                        for f in sorted(meas.glob(f"{fam}_{fmt}_rep*_condA.json"))
                    ]
                )
            )
            for fmt in FORMATS
        }
        for fam in FAMILIES
    }
    sim = {}
    for fam in FAMILIES:
        T = {fmt: Tm[fam][fmt][1] for fmt in FORMATS}
        p1 = {fmt: Tm[fam][fmt][0].mean() for fmt in FORMATS}

        def cost(tokmean, fmt):
            return tokmean / toks[fam][fmt] / p1[fmt]

        camp = {fmt: cost(T[fmt].mean(), fmt) for fmt in FORMATS}
        camp_order = sorted(FORMATS, key=camp.get)
        lv = levels_of[fam]
        pools = [np.flatnonzero(lv == k) for k in range(1, 6)]
        first = dict.fromkeys(FORMATS, 0)
        same = 0
        for _ in range(N_SIM):
            idx = np.concatenate([rng.choice(p, 4, replace=False) for p in pools])
            c = {
                fmt: cost(T[fmt][rng.integers(0, T[fmt].shape[0]), idx].mean(), fmt)
                for fmt in FORMATS
            }
            order = sorted(FORMATS, key=c.get)
            first[order[0]] += 1
            same += order == camp_order
        sim[fam] = {
            "campaign_length_order": camp_order,
            "campaign_length_cost_s_per_correct": {k: round(v, 2) for k, v in camp.items()},
            "p_cheapest": {k: round(v / N_SIM, 3) for k, v in first.items()},
            "p_full_order_equals_campaign_order": round(same / N_SIM, 3),
            "measured_condA_tok_s": {k: round(v, 2) for k, v in toks[fam].items()},
        }
    rep["subset_draw_simulation"] = sim

    args.out.write_text(json.dumps(rep, indent=1) + "\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
