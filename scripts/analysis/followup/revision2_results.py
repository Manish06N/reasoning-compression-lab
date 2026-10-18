#!/usr/bin/env python3
"""Recompute E1, E2, E3 and X1 from the HPC per-item records (independent of the HPC agent's reports).

Estimators as in analysis/prop1_checks.py: cross-seed Lambda_b (all ordered pairs s != t, BF16 run
as reference), pooled placebo Pi_b, excess = Lambda_b - Pi_b, all with ONE shared item bootstrap.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "results" / "followup"
OUT = EXP / "reports"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import revision_sensitivities as rs

B = 10_000
FAM = {"Qwen-7B": "DeepSeek-R1-Distill-Qwen-7B", "Llama-8B": "DeepSeek-R1-Distill-Llama-8B"}


def load(path: Path, n: int):
    d = json.loads(path.read_text())
    rows = sorted(d["details"], key=lambda r: r["row"])
    assert len(rows) == n, (path, len(rows))
    return (np.array([r["extractive_match"] for r in rows], float),
            np.array([r["completion_tokens"] for r in rows], float))


def load_set(folder: str, stem: str, seeds, n=500):
    A, T = [], []
    for s in seeds:
        a, t = load(EXP / folder / f"{stem}_seed{s}.json", n)
        A.append(a); T.append(t)
    return np.array(A), np.array(T)


def pairs_cross(m):  # ordered (ref, other), ref != other
    return [(r, o) for r in range(m) for o in range(m) if r != o]


def item_sums(Aref, Tref, Toth, pairs):
    s = np.zeros(Aref.shape[1]); n = np.zeros(Aref.shape[1])
    for r, o in pairs:
        s += Aref[r] * (Toth[o] - Tref[r]); n += Aref[r]
    return s, n


def pct(x): return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def boot_ratio(parts, reps=B, seed=0):
    N = len(parts[0][0]); idx = np.random.default_rng(seed).integers(0, N, (reps, N))
    return [p[0][idx].sum(1) / p[1][idx].sum(1) for p in parts]


def analyse_length(Ab, Tb, Aq, Tq, label):
    ms, mq = Ab.shape[0], Tq.shape[0]
    sl, nl = item_sums(Ab, Tb, Tq, [(r, o) for r in range(ms) for o in range(mq) if r != o or ms != mq])
    sm, nm = item_sums(Ab, Tb, Tq, [(r, r) for r in range(ms)])      # seed-matched (the paper's headline convention)
    sp, npi = item_sums(Ab, Tb, Tb, pairs_cross(ms))
    lam, lamm, pi = sl.sum() / nl.sum(), sm.sum() / nm.sum(), sp.sum() / npi.sum()
    bl, bm, bp = boot_ratio([(sl, nl), (sm, nm), (sp, npi)])
    N = Ab.shape[1]; idx = np.random.default_rng(1).integers(0, N, (B, N))
    dT = Tq.mean(0) - Tb.mean(0); dA = (Aq.mean(0) - Ab.mean(0)) * 100
    return dict(label=label, lam_b=lam, lam_ci=pct(bl), lam_m=lamm, lam_m_ci=pct(bm), pi_b=pi, pi_ci=pct(bp),
                excess=lam - pi, excess_ci=pct(bl - bp), excess_m=lamm - pi, excess_m_ci=pct(bm - bp),
                uncond_tokens=float(dT.mean()), uncond_ci=pct(dT[idx].mean(1)),
                acc_ref=float(Ab.mean() * 100), acc_arm=float(Aq.mean() * 100),
                acc_diff_pp=float(dA.mean()), acc_diff_ci=pct(dA[idx].mean(1)),
                tok_ref=float(Tb.mean()), tok_arm=float(Tq.mean()))


def show(r):
    print(f"  {r['label']:34s} acc {r['acc_ref']:.2f}->{r['acc_arm']:.2f} (d {r['acc_diff_pp']:+.2f}pp {[round(c,2) for c in r['acc_diff_ci']]})  "
          f"tok {r['tok_ref']:.0f}->{r['tok_arm']:.0f} (d {r['uncond_tokens']:+.0f} {[round(c) for c in r['uncond_ci']]})\n"
          f"      matched Lambda_b {r['lam_m']:+.0f} {[round(c) for c in r['lam_m_ci']]} excess {r['excess_m']:+.0f} {[round(c) for c in r['excess_m_ci']]} | cross-seed Lambda_b {r['lam_b']:+.0f} {[round(c) for c in r['lam_ci']]}  Pi_b {r['pi_b']:+.0f} {[round(c) for c in r['pi_ci']]}  "
          f"excess {r['excess']:+.0f} {[round(c) for c in r['excess_ci']]}")


res = {}
base = {f: rs.load_cell("math500", f, "BF16") for f in FAM}

print("E2: fp16 vs BF16 (MATH-500, 5 seeds) -- cross-seed Lambda_b, shared item bootstrap")
res["E2"] = {}
for f, stem in FAM.items():
    Aq, Tq = load_set("revision2/compact/E2_fp16", f"{stem}_fp16_math500_n500", range(42, 47))
    r = analyse_length(*base[f], Aq, Tq, f"{f} fp16"); show(r); res["E2"][f] = r

print("\nE3: BF16 T=0.8 vs T=0.6 (MATH-500, 5 seeds)")
res["E3"] = {}
for f, stem in FAM.items():
    Aq, Tq = load_set("revision2/compact/E3_T08", f"{stem}_T0.8_math500_n500", range(42, 47))
    r = analyse_length(*base[f], Aq, Tq, f"{f} T=0.8"); show(r); res["E3"][f] = r

print("\nE1: self-AWQ Llama-8B vs BF16 campaign")
res["E1"] = {}
stem = "DeepSeek-R1-Distill-Llama-8B-selfawq"
Aq, Tq = load_set("revision2/compact/E1_awq", f"{stem}_math500_n500", range(42, 47))
r = analyse_length(*base["Llama-8B"], Aq, Tq, "Llama self-AWQ MATH-500"); show(r); res["E1"]["math500"] = r
Ac, Tc = rs.load_cell("math500", "Llama-8B", "AWQ-4")
comm = analyse_length(*base["Llama-8B"], Ac, Tc, "Llama community AWQ-4 MATH-500"); show(comm); res["E1"]["math500_community"] = comm
d = (Aq.mean(0) - Ac.mean(0)) * 100; idx = np.random.default_rng(2).integers(0, 500, (B, 500))
res["E1"]["selfawq_minus_community_pp"] = dict(diff=float(d.mean()), ci=pct(d[idx].mean(1)))
print(f"  self-AWQ minus community AWQ-4 (MATH-500): {d.mean():+.2f}pp {[round(c,2) for c in pct(d[idx].mean(1))]}")
bg = rs.load_cell("gsm8k", "Llama-8B", "BF16"); cg = rs.load_cell("gsm8k", "Llama-8B", "AWQ-4")
print(f"  (GSM8K campaign seeds: BF16 {bg[0].shape[0]}, community AWQ {cg[0].shape[0]})")
Ag, Tg = load_set("revision2/compact/E1_awq", f"{stem}_gsm8k_n1319", range(42, 45), n=1319)
r = analyse_length(bg[0][:3], bg[1][:3], Ag, Tg, "Llama self-AWQ GSM8K (42-44)"); show(r); res["E1"]["gsm8k"] = r
r2 = analyse_length(bg[0][:3], bg[1][:3], cg[0][:3], cg[1][:3], "Llama community AWQ GSM8K (42-44)"); show(r2); res["E1"]["gsm8k_community"] = r2

print("\nX1: held-out BF16 seeds 47-51 vs campaign 42-46 -- Pi_new - Pi_old, paired item bootstrap")
res["X1"] = {}


def kshare(A, T):
    m = A.shape[0]; k = A.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu1 = (A * T).sum(0) / k; mu0 = ((1 - A) * T).sum(0) / (m - k)
    num = np.where((k > 0) & (k < m), k * (m - k) * (mu0 - mu1), 0.0)
    return {int(j): float(num[k == j].sum() / num.sum()) for j in range(1, m)}, float(((k > 0) & (k < m)).mean())


for f, stem in FAM.items():
    Ao, To = base[f]
    An, Tn = load_set("revision3/compact/X1_heldout", f"{stem}_heldout_math500_n500", range(47, 52))
    so, no = item_sums(Ao, To, To, pairs_cross(5)); sn, nn = item_sums(An, Tn, Tn, pairs_cross(5))
    po, pn = so.sum() / no.sum(), sn.sum() / nn.sum()
    bo, bn = boot_ratio([(so, no), (sn, nn)], seed=3)
    diff_ci = pct(bn - bo)
    Aall, Tall = np.vstack([Ao, An]), np.vstack([To, Tn])
    sa, na = item_sums(Aall, Tall, Tall, pairs_cross(10)); pall = sa.sum() / na.sum()
    ball = boot_ratio([(sa, na)], seed=4)[0]
    ks_o, int_o = kshare(Ao, To); ks_n, int_n = kshare(An, Tn)
    res["X1"][f] = dict(pi_old=po, pi_old_ci=pct(bo), pi_new=pn, pi_new_ci=pct(bn), diff=pn - po, diff_ci=diff_ci,
                       supported=bool(diff_ci[0] <= 0 <= diff_ci[1]), pi_pooled10=pall, pi_pooled10_ci=pct(ball),
                       k_share_old=ks_o, k_share_new=ks_n, intermediate_old=int_o, intermediate_new=int_n,
                       acc_old=float(Ao.mean() * 100), acc_new=float(An.mean() * 100))
    x = res["X1"][f]
    print(f"  {f}: Pi_old {po:+.1f} {[round(c,1) for c in x['pi_old_ci']]}  Pi_new {pn:+.1f} {[round(c,1) for c in x['pi_new_ci']]}  "
          f"diff {pn-po:+.1f} {[round(c,1) for c in diff_ci]} -> {'SUPPORTED' if x['supported'] else 'NOT SUPPORTED'}")
    print(f"      pooled(10 seeds) {pall:+.1f} {[round(c,1) for c in x['pi_pooled10_ci']]}  acc old/new {x['acc_old']:.2f}/{x['acc_new']:.2f}  "
          f"intermediate items old/new {int_o:.3f}/{int_n:.3f}")
    print(f"      S_b share by k, old {ks_o}  new {ks_n}")


# ---- excess of every campaign arm over the 10-seed pooled placebo (original 5 + held-out 5 BF16 seeds)
print("\nExcess over the 10-seed placebo (seed-matched Lambda_b vs pooled 10-seed Pi_b; shared item bootstrap)")
res["excess10"] = {}
for f, stem in FAM.items():
    Ao, To = base[f]
    An, Tn = load_set("revision3/compact/X1_heldout", f"{stem}_heldout_math500_n500", range(47, 52))
    Aall, Tall = np.vstack([Ao, An]), np.vstack([To, Tn])
    sa, na = item_sums(Aall, Tall, Tall, pairs_cross(10)); pall = sa.sum() / na.sum()
    for fmt in ("FP8", "AWQ-4", "GPTQ-4"):
        Aq_, Tq_ = rs.load_cell("math500", f, fmt)
        sm, nm = item_sums(Ao, To, Tq_, [(r, r) for r in range(5)])
        bm, bp = boot_ratio([(sm, nm), (sa, na)], seed=7)
        ex = sm.sum() / nm.sum() - pall
        res["excess10"][f"{f} {fmt}"] = dict(lam_m=float(sm.sum() / nm.sum()), pi10=float(pall), excess=float(ex), ci=pct(bm - bp))
        print(f"  {f} {fmt:7s} matched Lambda_b {sm.sum()/nm.sum():+.0f}  Pi10 {pall:+.0f}  excess {ex:+.0f} {[round(c) for c in pct(bm-bp)]}")

# ---- per-level accuracy change for self-AWQ and community AWQ-4 (Llama, MATH-500), levels from the HPC item table
import csv, glob
tab = sorted(csv.DictReader(open(EXP / "provenance" / "MATH500_item_table.tsv"), delimiter="	"), key=lambda r: int(r["row"]))
lev = np.array([int(r["level"]) for r in tab])
Asf = Aq  # self-AWQ MATH-500 correctness (5 seeds), loaded above for E1
res["levels"] = {}
rng = np.random.default_rng(5)
groups = [(f"L{l}", lev == l) for l in range(1, 6)] + [("L1-2", lev <= 2), ("L3-5", lev >= 3)]
for name, arm in (("community", Ac), ("self", Asf)):
    res["levels"][name] = {}
    for g, m in groups:
        d = (arm[:, m].mean(0) - base["Llama-8B"][0][:, m].mean(0)) * 100
        idx = rng.integers(0, m.sum(), (B, m.sum()))
        res["levels"][name][g] = dict(n=int(m.sum()), diff=float(d.mean()), ci=pct(d[idx].mean(1)))
    print(name, {g: (round(v["diff"], 2), [round(c, 2) for c in v["ci"]]) for g, v in res["levels"][name].items()})

(OUT / "revision2_results.json").write_text(json.dumps(res, indent=1))
print("\nwrote", OUT / "revision2_results.json")
