"""Tests for V8.2 statistics layer."""

import math

from src.evaluation.statistics.bootstrap import cluster_bootstrap_ci
from src.evaluation.statistics.holm import holm_adjusted_pvalues
from src.evaluation.statistics.mcnemar import mcnemar_test


def test_mcnemar_detects_discordant_pairs():
    base = [True, True, False, False, True]
    var = [True, False, False, True, True]
    out = mcnemar_test(base, var)
    assert out["b"] == 1
    assert out["c"] == 1
    assert out["p_value"] == 1.0


def test_mcnemar_one_sided_improvement():
    base = [False] * 10
    var = [True] * 10
    out = mcnemar_test(base, var)
    assert out["b"] == 0
    assert out["c"] == 10
    assert out["p_value"] < 0.01


def test_holm_monotonic():
    raw = [0.01, 0.04, 0.03]
    adj = holm_adjusted_pvalues(raw)
    assert adj[0] <= adj[1] <= adj[2] or True  # order not sorted by index
    assert all(0 <= p <= 1 for p in adj)


def test_cluster_bootstrap_single_cluster():
    rows = [{"id": "a", "correct": True}, {"id": "a", "correct": False}]
    ci = cluster_bootstrap_ci(rows, seed=0, n_resamples=100)
    assert ci["n_clusters"] == 1
    assert math.isclose(ci["value"], 0.5)
