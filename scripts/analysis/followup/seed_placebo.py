#!/usr/bin/env python3
"""seed_placebo: a seed placebo for correctness-conditioned length comparisons.

Given per-item correctness and completion length for several seeds of a BF16 reference and of a modified
model (quantized, different dtype, different temperature, ...), compute

  Lambda_b  BF16-correct conditional length change:  sum_i sum_pairs a_ref * (len_other - len_ref) / sum a_ref
  Pi_b      the same estimator applied to two seeds of the REFERENCE (true effect zero): the selection term
  excess    Lambda_b - Pi_b, on shared item resamples (an accuracy-weighted unconditional change, not a causal effect)

Requires only numpy.  Arrays have shape (seeds, items); correctness is 0/1.

    python seed_placebo.py --ref "ref/*_seed4*.json" --other "mod/*_seed4*.json"

reads compact JSON records (details[].row, extractive_match, completion_tokens) and prints the comparison.
Assumptions (Proposition 1 of the paper): runs of one model are i.i.d. across seeds given the item, and runs of the
reference are independent of runs of the modified model. Use pairing="cross" if you doubt the second one.
"""
from __future__ import annotations
import argparse, glob, json
import numpy as np


def load_compact(paths):
    """Return (correct, length) arrays of shape (seeds, items) from compact result JSON files, rows in 'row' order."""
    A, T = [], []
    for p in sorted(paths):
        rows = sorted(json.load(open(p, encoding="utf-8"))["details"], key=lambda r: r["row"])
        A.append([r["extractive_match"] for r in rows]); T.append([r["completion_tokens"] for r in rows])
    return np.array(A, float), np.array(T, float)


def _pairs(m_ref, m_oth, pairing):
    if pairing == "matched":
        assert m_ref == m_oth, "matched pairing needs the same number of seeds"
        return [(s, s) for s in range(m_ref)]
    return [(r, o) for r in range(m_ref) for o in range(m_oth) if r != o or m_ref != m_oth]


def item_sums(a_ref, t_ref, t_oth, pairs):
    """Per-item numerator and denominator of the BF16-correct estimator over the given (ref seed, other seed) pairs."""
    num = np.zeros(a_ref.shape[1]); den = np.zeros(a_ref.shape[1])
    for r, o in pairs:
        num += a_ref[r] * (t_oth[o] - t_ref[r]); den += a_ref[r]
    return num, den


def placebo_sums(a_ref, t_ref):
    """The placebo: reference seed against another reference seed, all ordered pairs s != t."""
    m = a_ref.shape[0]
    return item_sums(a_ref, t_ref, t_ref, [(r, o) for r in range(m) for o in range(m) if r != o])


def selection_identity_gap(a_ref, t_ref):
    """Exact identity Pi_b = m/(m-1) * plug-in S_b. Returns |difference| in tokens (should be ~1e-10)."""
    m, n = a_ref.shape; k = a_ref.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu1 = (a_ref * t_ref).sum(0) / k; mu0 = ((1 - a_ref) * t_ref).sum(0) / (m - k)
    mid = (k > 0) & (k < m); a = k / m
    s_hat = (a * (1 - a) * np.where(mid, mu0 - mu1, 0.0)).sum() / a.sum()
    num, den = placebo_sums(a_ref, t_ref)
    return abs(num.sum() / den.sum() - m / (m - 1) * s_hat)


def _ci(x, level=95.0):
    lo = (100 - level) / 2
    return [float(v) for v in np.percentile(x, [lo, 100 - lo])]


def compare(a_ref, t_ref, a_oth, t_oth, pairing="matched", reps=10_000, seed=0):
    """Lambda_b, placebo Pi_b and excess with one shared item bootstrap, plus the unconditional and accuracy contrasts."""
    pairs = _pairs(a_ref.shape[0], t_oth.shape[0], pairing)
    sl, nl = item_sums(a_ref, t_ref, t_oth, pairs)
    sp, npl = placebo_sums(a_ref, t_ref)
    n_items = a_ref.shape[1]
    idx = np.random.default_rng(seed).integers(0, n_items, (reps, n_items))
    bl = sl[idx].sum(1) / nl[idx].sum(1); bp = sp[idx].sum(1) / npl[idx].sum(1)
    d_len = t_oth.mean(0) - t_ref.mean(0); d_acc = (a_oth.mean(0) - a_ref.mean(0)) * 100
    lam, pi = sl.sum() / nl.sum(), sp.sum() / npl.sum()
    return dict(pairing=pairing, lambda_b=float(lam), lambda_b_ci=_ci(bl), placebo=float(pi), placebo_ci=_ci(bp),
                excess=float(lam - pi), excess_ci=_ci(bl - bp),
                unconditional=float(d_len.mean()), unconditional_ci=_ci(d_len[idx].mean(1)),
                accuracy_diff_pp=float(d_acc.mean()), accuracy_diff_ci=_ci(d_acc[idx].mean(1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="glob of compact JSON files, reference model, one per seed")
    ap.add_argument("--other", required=True, help="glob of compact JSON files, modified model, one per seed")
    ap.add_argument("--pairing", choices=["matched", "cross"], default="matched")
    ap.add_argument("--reps", type=int, default=10_000)
    a = ap.parse_args()
    ar, tr = load_compact(glob.glob(a.ref)); ao, to = load_compact(glob.glob(a.other))
    r = compare(ar, tr, ao, to, a.pairing, a.reps)
    fmt = lambda v, ci: f"{v:+.1f}  [{ci[0]:+.1f}, {ci[1]:+.1f}]"
    print(f"seeds: reference {ar.shape[0]}, other {ao.shape[0]}; items {ar.shape[1]}; pairing {r['pairing']}")
    print("accuracy change (pp)         ", fmt(r["accuracy_diff_pp"], r["accuracy_diff_ci"]))
    print("unconditional length change  ", fmt(r["unconditional"], r["unconditional_ci"]))
    print("Lambda_b (BF16-correct)      ", fmt(r["lambda_b"], r["lambda_b_ci"]))
    print("placebo Pi_b (ref vs ref)    ", fmt(r["placebo"], r["placebo_ci"]))
    print("excess Lambda_b - Pi_b       ", fmt(r["excess"], r["excess_ci"]))
    print("(excess is an accuracy-weighted unconditional change, not a causal effect)")


if __name__ == "__main__":
    main()
