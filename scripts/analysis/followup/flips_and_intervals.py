#!/usr/bin/env python3
"""Two literature-motivated CPU analyses on the released per-cell records.

1. Flip placebo (motivated by Does Acceleration Cause Hidden Instability in VLMs, EMNLP 2025):
   NDR = share of (item, seed-pair) cases where the reference run is correct and the other run is wrong;
   PDR = the reverse.  For BF16-vs-quantized (cross-seed pairs) and for BF16-vs-BF16 (the placebo, same
   estimator, true effect zero).  Excess = quantized minus placebo, item-level bootstrap.
2. Interval comparison (motivated by Bowyer et al., ICML 2025: bootstrap/CLT intervals can be
   miscalibrated for small benchmarks): percentile bootstrap vs BCa vs t vs Bayesian bootstrap for
   all 18 pass@1 contrasts.
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "followup" / "reports"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import revision_sensitivities as rs

B = 10_000
FAM = ("Qwen-7B", "Llama-8B"); FMT = ("FP8", "AWQ-4", "GPTQ-4")
res = {"flips": {}, "intervals": {}}

def pair_rates(Aref, Aoth, pairs):
    ndr = np.zeros(Aref.shape[1]); pdr = np.zeros(Aref.shape[1])
    for r, o in pairs:
        ndr += (Aref[r] == 1) & (Aoth[o] == 0); pdr += (Aref[r] == 0) & (Aoth[o] == 1)
    return ndr / len(pairs), pdr / len(pairs)

print("1. FLIP PLACEBO (percentage points of items; item-level 95% bootstrap CI)")
for bench in ("math500", "gsm8k", "gpqa"):
    for fam in FAM:
        Ab, _ = rs.load_cell(bench, fam, "BF16"); S, N = Ab.shape
        idx = np.random.default_rng(0).integers(0, N, (B, N))
        n_pl, p_pl = pair_rates(Ab, Ab, list(itertools.permutations(range(S), 2)))
        print(f"  {bench:7s} {fam:8s} placebo flips (BF16 s correct -> BF16 t wrong): {100*n_pl.mean():.2f}pp")
        for fmt in FMT:
            Aq, _ = rs.load_cell(bench, fam, fmt)
            n_q, p_q = pair_rates(Ab, Aq, [(i, j) for i in range(S) for j in range(S) if i != j])
            ex_n = n_q - n_pl; ex_p = p_q - p_pl
            ci_n = np.percentile(ex_n[idx].mean(1), [2.5, 97.5]) * 100; ci_p = np.percentile(ex_p[idx].mean(1), [2.5, 97.5]) * 100
            res["flips"][f"{bench}|{fam}|{fmt}"] = {"NDR": 100 * n_q.mean(), "PDR": 100 * p_q.mean(), "placebo": 100 * n_pl.mean(),
                                                     "excess_NDR": 100 * ex_n.mean(), "excess_NDR_ci": ci_n.tolist(),
                                                     "excess_PDR": 100 * ex_p.mean(), "excess_PDR_ci": ci_p.tolist()}
            print(f"     {fmt:6s} NDR={100*n_q.mean():.2f}  PDR={100*p_q.mean():.2f}  | excess over placebo: NDR {100*ex_n.mean():+.2f} [{ci_n[0]:+.2f},{ci_n[1]:+.2f}]  PDR {100*ex_p.mean():+.2f} [{ci_p[0]:+.2f},{ci_p[1]:+.2f}]")

print("\n2. INTERVAL COMPARISON for the 18 pass@1 contrasts (pp); flags where a method's zero-exclusion differs")
rng = np.random.default_rng(0)
for bench in ("math500", "gsm8k", "gpqa"):
    for fam in FAM:
        Ab, _ = rs.load_cell(bench, fam, "BF16")
        for fmt in FMT:
            Aq, _ = rs.load_cell(bench, fam, fmt)
            d = (Aq.mean(0) - Ab.mean(0)) * 100; n = len(d)
            perc = np.percentile(d[rng.integers(0, n, (B, n))].mean(1), [2.5, 97.5])
            bca = stats.bootstrap((d,), np.mean, n_resamples=B, method="BCa", random_state=0).confidence_interval
            se = d.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1); tci = (d.mean() - t * se, d.mean() + t * se)
            w = rng.dirichlet(np.ones(n), B); bb = np.percentile(w @ d, [2.5, 97.5])
            ex = lambda lo, hi: not (lo <= 0 <= hi)
            decisions = {"perc": ex(*perc), "bca": ex(bca.low, bca.high), "t": ex(*tci), "bayes": ex(*bb)}
            width = {"perc": perc[1] - perc[0], "bca": bca.high - bca.low, "t": tci[1] - tci[0], "bayes": bb[1] - bb[0]}
            res["intervals"][f"{bench}|{fam}|{fmt}"] = {"delta_pp": float(d.mean()), "perc": perc.tolist(), "bca": [float(bca.low), float(bca.high)],
                                                         "t": list(map(float, tci)), "bayes": bb.tolist(), "excludes_zero": decisions}
            flag = "  <-- decisions differ" if len(set(decisions.values())) > 1 else ""
            print(f"  {bench:7s} {fam:8s} {fmt:6s} d={d.mean():+6.2f}  perc[{perc[0]:+.2f},{perc[1]:+.2f}] bca[{bca.low:+.2f},{bca.high:+.2f}] t[{tci[0]:+.2f},{tci[1]:+.2f}] width ratio bca/perc={width['bca']/width['perc']:.2f}{flag}")

(OUT / "flips_and_intervals.json").write_text(json.dumps(res, indent=1, default=float))
print("\nsaved")
