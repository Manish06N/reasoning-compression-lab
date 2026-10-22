#!/usr/bin/env python3
"""E1c (self-AWQ, forced GEMM kernel) and X3 (Qwen BF16 at max_num_seqs=64), applying the rules of PREREG_E1c_X3_X5.md literally."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
V3 = EXP = ROOT / "results" / "followup"
sys.path.insert(0, str(ROOT / "scripts" / "analysis")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_sensitivities as rs, seed_placebo as sp

B = 10_000
def ld(p): return sp.load_compact(sorted(p))
def pct(x): return [float(v) for v in np.percentile(x, [2.5, 97.5])]
def fmt(v, ci, d=0): return f"{v:+.{d}f} [{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]"

res = {}
Ab, Tb = rs.load_cell("math500", "Llama-8B", "BF16")
a1, t1 = ld((EXP / "revision2/compact/E1_awq").glob("DeepSeek-R1-Distill-Llama-8B-selfawq_math500_n500_seed4[2-6].json"))   # E1a, awq_marlin
a2, t2 = ld((V3 / "revision2/compact/E1c_awq_gemm").glob("*seed4[2-6].json"))                                                  # E1c, forced GEMM
N = Ab.shape[1]; idx = np.random.default_rng(0).integers(0, N, (B, N))
pairs_m = [(s, s) for s in range(5)]
sp_, np_ = sp.placebo_sums(Ab, Tb)
s1, n1 = sp.item_sums(Ab, Tb, t1, pairs_m); s2, n2 = sp.item_sums(Ab, Tb, t2, pairs_m)
bp = sp_[idx].sum(1) / np_[idx].sum(1); b1 = s1[idx].sum(1) / n1[idx].sum(1); b2 = s2[idx].sum(1) / n2[idx].sum(1)
pi = sp_.sum() / np_.sum(); l1, l2 = s1.sum() / n1.sum(), s2.sum() / n2.sum()
ex1, ex2 = b1 - bp, b2 - bp
dA = lambda A: (A.mean(0) - Ab.mean(0)) * 100
acc2 = dA(a2); acc1 = dA(a1); acc21 = (a2.mean(0) - a1.mean(0)) * 100
dT2 = t2.mean(0) - Tb.mean(0)
excess_diff = ex1 - ex2                                  # excess(E1a) - excess(E1c), shared item resamples (seed-matched, descriptive)
# The pre-registered estimators are CROSS-SEED pairs (PREREG_E1c_X3_X5.md, header): the rule is applied to those.
pairs_c = [(r, o) for r in range(5) for o in range(5) if r != o]
s1c, n1c = sp.item_sums(Ab, Tb, t1, pairs_c); s2c, n2c = sp.item_sums(Ab, Tb, t2, pairs_c)
b1c = s1c[idx].sum(1) / n1c[idx].sum(1); b2c = s2c[idx].sum(1) / n2c[idx].sum(1)
l1c, l2c = s1c.sum() / n1c.sum(), s2c.sum() / n2c.sum(); ex1c, ex2c = b1c - bp, b2c - bp; excess_diff_c = ex1c - ex2c
rule_acc = "kernel-robust (loss detected under GEMM)" if pct(acc2[idx].mean(1))[1] < 0 else "not detected under GEMM"
if pct(ex2c)[0] > 0: rule_len = "kernel-robust (cross-seed excess excludes 0, positive)"
elif pct(ex2c)[0] <= 0 <= pct(ex2c)[1] and not (pct(excess_diff_c)[0] <= 0 <= pct(excess_diff_c)[1]): rule_len = "kernel-dependent"
else: rule_len = "inconclusive"
cross = dict(lam=float(l2c), lam_ci=pct(b2c), excess=float(l2c - pi), excess_ci=pct(ex2c), E1a_lam=float(l1c), E1a_excess=float(l1c - pi), E1a_excess_ci=pct(ex1c),
             excess_E1a_minus_E1c=float(l1c - l2c), excess_diff_ci=pct(excess_diff_c))
res["E1c"] = dict(acc_diff_pp=float(acc2.mean()), acc_diff_ci=pct(acc2[idx].mean(1)), acc_E1a_pp=float(acc1.mean()), acc_E1c_minus_E1a_pp=float(acc21.mean()),
                  acc_E1c_minus_E1a_ci=pct(acc21[idx].mean(1)), lam_m=float(l2), lam_m_ci=pct(b2), placebo=float(pi), placebo_ci=pct(bp), excess=float(l2 - pi), excess_ci=pct(ex2),
                  E1a_excess=float(l1 - pi), E1a_excess_ci=pct(ex1), excess_E1a_minus_E1c=float(l1 - l2), excess_diff_ci=pct(excess_diff),
                  uncond_tokens=float(dT2.mean()), uncond_ci=pct(dT2[idx].mean(1)), rule_accuracy=rule_acc, rule_length=rule_len, cross_seed=cross,
                  acc_by_seed=[float(a.mean() * 100) for a in a2], tokens_by_seed=[float(t.mean()) for t in t2])
print("E1c vs BF16 (Llama, MATH-500, 5 seeds, seed-matched)")
print("  accuracy diff pp", fmt(acc2.mean(), pct(acc2[idx].mean(1)), 2), "| E1a", f"{acc1.mean():+.2f}", "| E1c - E1a", fmt(acc21.mean(), pct(acc21[idx].mean(1)), 2))
print("  Lambda_b", fmt(l2, pct(b2)), "placebo", fmt(pi, pct(bp)), "excess", fmt(l2 - pi, pct(ex2)))
print("  E1a excess", fmt(l1 - pi, pct(ex1)), "| excess(E1a)-excess(E1c)", fmt(l1 - l2, pct(excess_diff)))
print("  uncond tokens", fmt(dT2.mean(), pct(dT2[idx].mean(1))))
print("  PRE-REGISTERED (cross-seed) Lambda_b", fmt(l2c, pct(b2c)), "excess", fmt(l2c - pi, pct(ex2c)), "| E1a excess", fmt(l1c - pi, pct(ex1c)), "| E1a-E1c", fmt(l1c - l2c, pct(excess_diff_c)))
print("  RULE accuracy:", rule_acc, "| RULE length (cross-seed, as pre-registered):", rule_len)

# X3: Qwen BF16 at max_num_seqs=64 against the campaign BF16 reference
Aq, Tq = rs.load_cell("math500", "Qwen-7B", "BF16")
ax, tx = ld((V3 / "revision3/compact/X3_batch").glob("*seed4[2-6].json"))
spq, npq = sp.placebo_sums(Aq, Tq); piq = spq.sum() / npq.sum()
sx, nx = sp.item_sums(Aq, Tq, tx, [(r, o) for r in range(5) for o in range(5) if r != o]); sm, nm = sp.item_sums(Aq, Tq, tx, pairs_m)
bpq = spq[idx].sum(1) / npq[idx].sum(1); bx = sx[idx].sum(1) / nx[idx].sum(1); bm = sm[idx].sum(1) / nm[idx].sum(1)
accx = (ax.mean(0) - Aq.mean(0)) * 100; dTx = tx.mean(0) - Tq.mean(0)
def ndr(A, O, pairs):
    n = np.zeros(A.shape[1]); d = np.zeros(A.shape[1])
    for r, o in pairs: n += A[r] * (1 - O[o]); d += A[r]
    return n, d
nc, dc = ndr(Aq, Aq, [(r, o) for r in range(5) for o in range(5) if r != o]); ns, ds = ndr(Aq, ax, pairs_m)
ndr_pl = nc.sum() / dc.sum() * 100; ndr_x = ns.sum() / ds.sum() * 100
bn = (nc[idx].sum(1) / dc[idx].sum(1)) * 100; bs = (ns[idx].sum(1) / ds[idx].sum(1)) * 100
exx = bx - bpq
if pct(exx)[0] > 0: rule_x = "batch limit induces inflation beyond the placebo"
else: rule_x = "consistent with the placebo (interval includes 0 or is negative)"
res["X3"] = dict(acc_diff_pp=float(accx.mean()), acc_diff_ci=pct(accx[idx].mean(1)), uncond_tokens=float(dTx.mean()), uncond_ci=pct(dTx[idx].mean(1)),
                 lam_cross=float(sx.sum() / nx.sum()), lam_cross_ci=pct(bx), lam_matched=float(sm.sum() / nm.sum()), lam_matched_ci=pct(bm), placebo=float(piq), placebo_ci=pct(bpq),
                 excess_cross=float(sx.sum() / nx.sum() - piq), excess_cross_ci=pct(exx), excess_matched=float(sm.sum() / nm.sum() - piq), excess_matched_ci=pct(bm - bpq),
                 ndr_placebo=float(ndr_pl), ndr_same_seed=float(ndr_x), ndr_excess=float(ndr_x - ndr_pl), ndr_excess_ci=pct(bs - bn), rule=rule_x,
                 acc_by_seed=[float(a.mean() * 100) for a in ax], tokens_by_seed=[float(t.mean()) for t in tx])
print("\nX3: Qwen BF16 at max_num_seqs=64 vs campaign (256)")
print("  accuracy diff pp", fmt(accx.mean(), pct(accx[idx].mean(1)), 2), "| uncond tokens", fmt(dTx.mean(), pct(dTx[idx].mean(1))))
print("  Lambda_b cross-seed", fmt(sx.sum() / nx.sum(), pct(bx)), "| matched", fmt(sm.sum() / nm.sum(), pct(bm)), "| placebo", fmt(piq, pct(bpq)))
print("  excess cross-seed", fmt(sx.sum() / nx.sum() - piq, pct(exx)), "| matched", fmt(sm.sum() / nm.sum() - piq, pct(bm - bpq)))
print(f"  correct-to-wrong flips: BF16-vs-BF16 baseline {ndr_pl:.2f}% | same-seed X3 {ndr_x:.2f}% | excess {ndr_x - ndr_pl:+.2f}pp {[round(c,2) for c in pct(bs-bn)]}")
print("  RULE:", rule_x)
out = ROOT / "results" / "followup" / "reports"; out.mkdir(exist_ok=True)
(out / "e1c_x3_results.json").write_text(json.dumps(res, indent=1)); print("\nwrote e1c_x3_results.json")
