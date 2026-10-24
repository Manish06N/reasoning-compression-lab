#!/usr/bin/env python3
"""E5a: Qwen-7B BF16, single instead of doubled beginning-of-sequence token, MATH-500.
Applies the rules of results/followup/PREREG_E5a.md literally.

    python scripts/analysis/followup/e5a_results.py --dir <folder with the E5a compact JSON files, one per seed>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "analysis")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_sensitivities as rs, seed_placebo as sp

ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--pattern", default="*seed4*.json")
a = ap.parse_args()
files = sorted(Path(a.dir).glob(a.pattern))
seeds = [int(f.stem.split("seed")[-1]) for f in files]
assert seeds and all(42 <= s <= 46 for s in seeds), (seeds, "expected seeds 42-46")
Ae, Te = sp.load_compact(files)                                   # E5a cells, in seed order
Ab5, Tb5 = rs.load_cell("math500", "Qwen-7B", "BF16")             # campaign BF16 (seeds 42-46)
rows = [s - 42 for s in seeds]; Ab, Tb = Ab5[rows], Tb5[rows]
B = 10_000; N = Ab.shape[1]; idx = np.random.default_rng(0).integers(0, N, (B, N))
def pct(x): return [float(v) for v in np.percentile(x, [2.5, 97.5])]
def fmt(v, ci, d=0): return f"{v:+.{d}f} [{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]"

m = len(seeds)
spl, npl = sp.placebo_sums(Ab5, Tb5); pi = spl.sum() / npl.sum(); bp = spl[idx].sum(1) / npl[idx].sum(1)         # five-seed placebo
pc = [(r, o) for r in range(m) for o in range(m) if r != o]; pm = [(s, s) for s in range(m)]
sx, nx = sp.item_sums(Ab, Tb, Te, pc); sm, nm = sp.item_sums(Ab, Tb, Te, pm)
bx = sx[idx].sum(1) / nx[idx].sum(1); bm = sm[idx].sum(1) / nm[idx].sum(1)
dA = (Ae.mean(0) - Ab.mean(0)) * 100; dT = Te.mean(0) - Tb.mean(0)
acc_ci, len_ci, ex_ci = pct(dA[idx].mean(1)), pct(dT[idx].mean(1)), pct(bx - bp)
ex_cross = sx.sum() / nx.sum() - pi
def ndr(A, O, pairs):
    n = np.zeros(A.shape[1]); d = np.zeros(A.shape[1])
    for r, o in pairs: n += A[r] * (1 - O[o]); d += A[r]
    return n, d
nb, db = ndr(Ab5, Ab5, [(r, o) for r in range(5) for o in range(5) if r != o]); ne, de = ndr(Ab, Ae, pm)
flip_base = nb.sum() / db.sum() * 100; flip_e5a = ne.sum() / de.sum() * 100
flip_ci = pct((ne[idx].sum(1) / de[idx].sum(1) - nb[idx].sum(1) / db[idx].sum(1)) * 100)
a_hit, l_hit, e_hit = not (acc_ci[0] <= 0 <= acc_ci[1]), not (len_ci[0] <= 0 <= len_ci[1]), not (ex_ci[0] <= 0 <= ex_ci[1])
verdict = "A BOS effect is detected" if (a_hit or l_hit or e_hit) else "not detected at this sample size"
print(f"E5a seeds {seeds}; campaign BF16 pass@1 {Ab.mean()*100:.2f}, E5a pass@1 {Ae.mean()*100:.2f}; mean tokens {Tb.mean():.0f} -> {Te.mean():.0f}")
print("accuracy diff (pp)         ", fmt(dA.mean(), acc_ci, 2), "  excludes 0:", a_hit)
print("unconditional length diff  ", fmt(dT.mean(), len_ci), "  excludes 0:", l_hit)
print("Lambda_b cross-seed        ", fmt(sx.sum() / nx.sum(), pct(bx)), "| matched", fmt(sm.sum() / nm.sum(), pct(bm)), "| placebo", fmt(pi, pct(bp)))
print("excess cross-seed          ", fmt(ex_cross, ex_ci), "  excludes 0:", e_hit, "| matched", fmt(sm.sum() / nm.sum() - pi, pct(bm - bp)))
print(f"flips (descriptive): baseline {flip_base:.2f}%  E5a same-seed {flip_e5a:.2f}%  excess {flip_e5a - flip_base:+.2f}pp {[round(c, 2) for c in flip_ci]}")
print("VERDICT (rule of the E5a message):", verdict)
out = ROOT / "results" / "followup" / "reports"; out.mkdir(exist_ok=True)
(out / "e5a_results.json").write_text(json.dumps(dict(seeds=seeds, acc_pp=float(dA.mean()), acc_ci=acc_ci, uncond=float(dT.mean()), uncond_ci=len_ci,
    lam_cross=float(sx.sum() / nx.sum()), lam_cross_ci=pct(bx), lam_matched=float(sm.sum() / nm.sum()), lam_matched_ci=pct(bm), placebo=float(pi), placebo_ci=pct(bp),
    excess_cross=float(ex_cross), excess_cross_ci=ex_ci, excess_matched=float(sm.sum() / nm.sum() - pi), excess_matched_ci=pct(bm - bp),
    flip_baseline=float(flip_base), flip_e5a=float(flip_e5a), flip_excess=float(flip_e5a - flip_base), flip_excess_ci=flip_ci, verdict=verdict), indent=1))
print("wrote e5a_results.json")
