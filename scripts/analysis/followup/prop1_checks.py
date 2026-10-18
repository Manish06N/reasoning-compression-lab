#!/usr/bin/env python3
"""CPU checks of Proposition 1 on the released per-cell records, plus a simulation.

Notation (per item i, S seeds): a_i = P(BF16 run correct); mu1_i / mu0_i = mean BF16
length when the run is correct / wrong.  Prop. 1 (conditional independence of runs
given the item):  E[Pi_b] = S_b = sum a(1-a)(mu0-mu1) / sum a  and
E[Lambda_b] = Delta_a + S_b,  so Lambda_b - Pi_b estimates Delta_a = sum a(mu^q - mu^b)/sum a.

Parts
  A  reproduce the published placebo through the repo's own code
  B  exact identity: pooled placebo == S/(S-1) * plug-in S_b ; decomposition by k correct;
     cross-benchmark comparison (MATH-500, GSM8K, GPQA)
  C  coupling check: seed-matched vs cross-seed Lambda for each quantized arm
  D  simulation with known truth (null, uniform, hard-item inflation, accuracy drop,
     and a coupled-arms violation)
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "followup" / "reports"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import revision_sensitivities as rs  # repo loader + published placebo

B = 10_000
FAMILIES = ("Qwen-7B", "Llama-8B")
res: dict = {}


def item_sums(a_ref, t_ref, a_oth, t_oth, pairs):
    n_items = a_ref.shape[1]
    s = np.zeros(n_items); n = np.zeros(n_items)
    for r, o in pairs:
        s += a_ref[r] * (t_oth[o] - t_ref[r]); n += a_ref[r]
    return s, n


def point(s, n): return float(s.sum() / n.sum())


def boot(s, n, rng_seed=0, reps=B):
    N = len(s); idx = np.random.default_rng(rng_seed).integers(0, N, (reps, N))
    return s[idx].sum(1) / n[idx].sum(1)


def ci(x): return [float(v) for v in np.percentile(x, [2.5, 97.5])]


# ---------------------------------------------------------------- A: reproduce
pub = rs.seed_placebo()
res["A_published_reproduction"] = {k: {"ref_correct": round(v["reference_correct"], 1), "ci": [round(c, 1) for c in v["reference_correct_ci95"]]}
                                   for k, v in pub.items() if "placebo" in k or "FP8" in k}
print("A. placebo / FP8 reference-correct (tokens) from repo code:")
for k, v in res["A_published_reproduction"].items(): print(f"   {k:28s} {v['ref_correct']:+8.1f}  {v['ci']}")

# ---------------------------------------------------------------- B: identity + decomposition
print("\nB. identity pooled placebo == S/(S-1)*S_hat, and decomposition by k correct")
resB = {}
for bench in ("math500", "gsm8k", "gpqa"):
    for fam in FAMILIES:
        A, L = rs.load_cell(bench, fam, "BF16")
        S, N = A.shape
        k = A.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu1 = (A * L).sum(0) / k
            mu0 = ((1 - A) * L).sum(0) / (S - k)
        a = k / S
        c = np.where((k > 0) & (k < S), a * (1 - a) * (mu0 - mu1), 0.0)
        S_hat = c.sum() / a.sum()
        w = np.where((k > 0) & (k < S), a * (1 - a), 0.0)                  # within-item wrong/correct length ratio on intermediate items
        within_ratio = float((w * np.nan_to_num(mu0)).sum() / (w * np.nan_to_num(mu1)).sum())
        pairs = list(itertools.permutations(range(S), 2))
        s, n = item_sums(A, L, A, L, pairs)
        pi = point(s, n)
        ident = S / (S - 1) * S_hat
        ci95 = ci(boot(s, n))
        inter = (k > 0) & (k < S)
        by_k = {int(j): {"items": int((k == j).sum()), "share_of_S": float(c[k == j].sum() / c.sum())} for j in range(1, S)}
        mean_len = float(L.mean())
        resB[f"{bench}|{fam}"] = {
            "mean_acc": float(A.mean()), "frac_items_intermediate_k": float(inter.mean()),
            "placebo_tokens": pi, "placebo_ci95": ci95, "placebo_pct_of_mean_len": 100 * pi / mean_len,
            "identity_gap_tokens": float(abs(pi - ident)),
            "within_item_ratio": within_ratio,
            "mean_len_correct": float((A * L).sum() / A.sum()), "mean_len_wrong": float(((1 - A) * L).sum() / (1 - A).sum()),
            "share_by_k": by_k}
        r = resB[f"{bench}|{fam}"]
        print(f"   {bench:7s} {fam:8s} acc={r['mean_acc']:.3f} interm.items={100*r['frac_items_intermediate_k']:.1f}%  placebo={pi:+7.1f} {ci95[0]:+.0f}..{ci95[1]:+.0f} ({r['placebo_pct_of_mean_len']:+.1f}% of mean len)  identity gap={r['identity_gap_tokens']:.2e}  len correct/wrong={r['mean_len_correct']:.0f}/{r['mean_len_wrong']:.0f}")
        print("             share of S by k correct:", {j: f"{v['share_of_S']*100:.0f}% ({v['items']} items)" for j, v in by_k.items()})
res["B"] = resB

# ---------------------------------------------------------------- C: coupling check
print("\nC. seed-matched vs cross-seed Lambda (BF16-correct conditional) per quantized arm, MATH-500")
resC = {}
for fam in FAMILIES:
    Ab, Lb = rs.load_cell("math500", fam, "BF16")
    s_pl, n_pl_ = item_sums(Ab, Lb, Ab, Lb, list(itertools.permutations(range(5), 2)))
    for fmt in ("FP8", "AWQ-4", "GPTQ-4"):
        Aq, Lq = rs.load_cell("math500", fam, fmt)
        s_same, n_same = item_sums(Ab, Lb, Aq, Lq, [(i, i) for i in range(5)])
        s_cross, n_cross = item_sums(Ab, Lb, Aq, Lq, [(i, j) for i in range(5) for j in range(5) if i != j])
        idx = np.random.default_rng(0).integers(0, 500, (B, 500))
        d = s_same[idx].sum(1) / n_same[idx].sum(1) - s_cross[idx].sum(1) / n_cross[idx].sum(1)
        ex = s_cross[idx].sum(1) / n_cross[idx].sum(1) - s_pl[idx].sum(1) / n_pl_[idx].sum(1)
        resC[f"{fam}|{fmt}"] = {"lambda_same": point(s_same, n_same), "lambda_cross": point(s_cross, n_cross),
                                 "diff": point(s_same, n_same) - point(s_cross, n_cross), "diff_ci95": ci(d),
                                 "excess_cross": point(s_cross, n_cross) - point(s_pl, n_pl_), "excess_cross_ci95": ci(ex)}
        r = resC[f"{fam}|{fmt}"]
        print(f"   {fam:8s} {fmt:6s} same={r['lambda_same']:+7.1f} cross={r['lambda_cross']:+7.1f} diff={r['diff']:+6.1f} CI [{r['diff_ci95'][0]:+.0f},{r['diff_ci95'][1]:+.0f}]")
res["C"] = resC


# ---------------------------------------------------------------- D: simulation
def calibrate_conc(m, target_frac, S=5, rng=np.random.default_rng(1)):
    """Beta(m c, (1-m) c): pick c so P(0<k<S) matches the observed fraction of intermediate items."""
    lo, hi = 0.02, 20.0
    for _ in range(40):
        c = np.sqrt(lo * hi); a = rng.beta(m * c, (1 - m) * c, 200_000)
        frac = 1 - (a ** S + (1 - a) ** S).mean()
        lo, hi = (c, hi) if frac < target_frac else (lo, c)
    return c


def simulate(m, conc, ratio_wrong, scenario, target_len_correct, R=300, N=500, S=5, Bb=400, rho=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("lam", "pi", "diff", "truth", "cover_diff", "cover_lam", "diff_same")}
    for _ in range(R):
        ab = np.clip(rng.beta(m * conc, (1 - m) * conc, N), 1e-6, 1 - 1e-6)
        base = np.exp(rng.normal(0, 0.9, N)) * np.exp(0.8 * (1 - ab))      # harder items are longer
        sd1, sd0 = 0.5, 0.6
        base = base * target_len_correct / ((ab * base * np.exp(sd1 ** 2 / 2)).sum() / ab.sum())   # tokens: mean correct length matches the data
        mu1 = base * np.exp(sd1 ** 2 / 2); mu0b = base * ratio_wrong * np.exp(sd0 ** 2 / 2)
        aq, fq = ab.copy(), np.ones(N)                                      # quantized arm: accuracy and length multiplier
        if scenario == "uniform5": fq = np.full(N, 1.05)
        elif scenario == "hard10": fq = np.where(ab < 0.9, 1.10, 1.0)
        elif scenario == "accdrop": aq = np.clip(np.where((ab > 0.1) & (ab < 0.9), ab - 0.15, ab), 1e-6, 1 - 1e-6)
        mu_b = ab * mu1 + (1 - ab) * mu0b
        mu_q = fq * (aq * mu1 + (1 - aq) * mu0b)
        truth = (ab * (mu_q - mu_b)).sum() / ab.sum()
        cb = rng.random((S, N)) < ab; cq = rng.random((S, N)) < aq
        lb = np.where(cb, base * np.exp(rng.normal(0, sd1, (S, N))), base * ratio_wrong * np.exp(rng.normal(0, sd0, (S, N))))
        lq = fq * np.where(cq, base * np.exp(rng.normal(0, sd1, (S, N))), base * ratio_wrong * np.exp(rng.normal(0, sd0, (S, N))))
        if rho > 0:                                                           # coupled arms: same-seed quantized run copies the BF16 run
            cp = rng.random((S, N)) < rho; cq = np.where(cp, cb, cq); lq = np.where(cp, lb, lq)
        Ab_, Aq_ = cb.astype(float), cq.astype(float)
        s_l, n_l = item_sums(Ab_, lb, Aq_, lq, [(i, j) for i in range(S) for j in range(S) if i != j])   # cross-seed Lambda
        s_p, n_p = item_sums(Ab_, lb, Ab_, lb, list(itertools.permutations(range(S), 2)))
        s_s, n_s = item_sums(Ab_, lb, Aq_, lq, [(i, i) for i in range(S)])                             # seed-matched Lambda
        idx = rng.integers(0, N, (Bb, N))
        lam = s_l.sum() / n_l.sum(); pi = s_p.sum() / n_p.sum()
        bd = s_l[idx].sum(1) / n_l[idx].sum(1) - s_p[idx].sum(1) / n_p[idx].sum(1)
        bl = s_l[idx].sum(1) / n_l[idx].sum(1)
        lo, hi = np.percentile(bd, [2.5, 97.5]); lo2, hi2 = np.percentile(bl, [2.5, 97.5])
        out["lam"].append(lam); out["pi"].append(pi); out["diff"].append(lam - pi); out["truth"].append(truth)
        out["cover_diff"].append(lo <= truth <= hi); out["cover_lam"].append(lo2 <= truth <= hi2)
        out["diff_same"].append(s_s.sum() / n_s.sum() - pi)
    return {k: float(np.mean(v)) for k, v in out.items()} | {"bias_diff": float(np.mean(np.array(out["diff"]) - np.array(out["truth"]))),
                                                              "bias_lam": float(np.mean(np.array(out["lam"]) - np.array(out["truth"]))),
                                                              "bias_diff_same_seed": float(np.mean(np.array(out["diff_same"]) - np.array(out["truth"])))}


print("\nD. simulation (R=300 replications, N=500 items, 5 seeds; truth = accuracy-weighted unconditional change)")
resD = {}
for fam in FAMILIES:
    r = resB[f"math500|{fam}"]
    conc = calibrate_conc(r["mean_acc"], r["frac_items_intermediate_k"]); ratio = r["within_item_ratio"]
    print(f"   {fam}: Beta concentration={conc:.2f} (from accuracy + share of intermediate items), within-item wrong/correct length ratio={ratio:.2f} (global ratio {r['mean_len_wrong']/r['mean_len_correct']:.2f}); observed placebo {r['placebo_tokens']:+.0f}")
    for scen, rho in (("null", 0.0), ("uniform5", 0.0), ("hard10", 0.0), ("accdrop", 0.0), ("null", 0.5)):
        o = simulate(r["mean_acc"], conc, ratio, scen, r["mean_len_correct"], rho=rho)
        tag = f"{fam}|{scen}" + (f"|coupled{rho}" if rho else "")
        resD[tag] = o
        print(f"     {scen:9s}{'' if not rho else f' coupled rho={rho}':20s} truth={o['truth']:+7.1f}  Lambda={o['lam']:+7.1f} (bias {o['bias_lam']:+6.1f})  Pi={o['pi']:+6.1f}  Lambda-Pi bias={o['bias_diff']:+6.1f}  "
              f"cover: Lambda-Pi={o['cover_diff']:.2f}, Lambda alone={o['cover_lam']:.2f}  seed-matched Lambda-Pi bias={o['bias_diff_same_seed']:+6.1f}")
res["D"] = resD

(OUT / "prop1_checks.json").write_text(json.dumps(res, indent=1, default=float))
print("\nsaved", OUT / "prop1_checks.json")
