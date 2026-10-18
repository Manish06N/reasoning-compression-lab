"""Tests for seed_placebo.py. Run: python analysis/test_seed_placebo.py  (or pytest)."""
import glob
from pathlib import Path
import numpy as np
import seed_placebo as sp

REPO_RESULTS = Path(__file__).resolve().parents[3] / "results" / "math500"


def population(rng, n, m):
    """Items with a per-item solve probability; wrong traces are 4x longer than correct ones (so a placebo exists)."""
    return np.clip(rng.beta(0.6, 0.25, n), 0.02, 0.99), rng.lognormal(7.8, 0.6, n)


def draw(p, base, seed, m=5):
    r = np.random.default_rng(seed); a = (r.random((m, len(p))) < p).astype(float)
    return a, base * np.where(a == 1, 1.0, 4.0) * r.lognormal(0, 0.2, (m, len(p)))


def test_identity_is_exact():
    p, base = population(np.random.default_rng(0), 400, 5)
    for s in range(5):
        A, T = draw(p, base, s)
        assert sp.selection_identity_gap(A, T) < 1e-8


def test_unconditional_placebo_is_zero_and_placebo_positive():
    p, base = population(np.random.default_rng(1), 400, 5)
    A, T = draw(p, base, 11)
    num, den = sp.placebo_sums(A, T)
    assert num.sum() / den.sum() > 0                      # selection term is positive when wrong traces are longer
    m = A.shape[0]
    total = sum((T[o] - T[r]).sum() for r in range(m) for o in range(m) if r != o)
    assert abs(total) < 1e-6                              # unconditional placebo is exactly zero


def test_null_comparison_covers_zero_excess():
    covered = 0
    for k in range(20):
        p, base = population(np.random.default_rng(7 + k), 300, 5)
        (a_r, t_r), (a_o, t_o) = draw(p, base, 1000 + k), draw(p, base, 2000 + k)   # two arms from the same population
        r = sp.compare(a_r, t_r, a_o, t_o, "cross", reps=400)
        covered += r["excess_ci"][0] <= 0 <= r["excess_ci"][1]
    assert covered >= 16                                   # nominal 95%; allow Monte Carlo slack with 20 replicates


def test_known_shift_is_recovered():
    p, base = population(np.random.default_rng(3), 400, 5)
    A, T = draw(p, base, 5)
    r = sp.compare(A, T, A, T + 300.0, "cross", reps=200)  # every length +300 tokens, same correctness
    assert abs(r["excess"] - 300.0) < 1e-6
    assert abs(r["unconditional"] - 300.0) < 1e-6


def test_reproduces_published_qwen_placebo():
    files = sorted(glob.glob(str(REPO_RESULTS / "DeepSeek-R1-Distill-Qwen-7B_math500_n500_seed4[2-6].json")))
    if len(files) != 5:
        print("  (skipped: campaign records not found)"); return
    A, T = sp.load_compact(files)
    num, den = sp.placebo_sums(A, T)
    assert abs(num.sum() / den.sum() - 138) < 1.0          # paper: +138 tokens for Qwen-7B on MATH-500


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
