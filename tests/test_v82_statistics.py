"""Tests for V8.2 statistics layer."""

import math

from src.evaluation.statistics.bootstrap import cluster_bootstrap_ci
from src.evaluation.statistics.holm import holm_adjusted_pvalues
from src.evaluation.statistics.mcnemar import mcnemar_test
from src.evaluation.statistics.paired_validation import (
    paired_difference_bootstrap_ci,
    validate_paired_rows,
)


def test_mcnemar_detects_discordant_pairs():
    base = [True, True, False, False, True]
    var = [True, False, False, True, True]
    out = mcnemar_test(base, var)
    assert out["b"] == 1
    assert out["c"] == 1
    assert out["p_value"] > 0.4


def test_mcnemar_tied_discordant_is_not_significant():
    base = [True, False]
    var = [False, True]
    out = mcnemar_test(base, var)
    assert out["b"] == 1
    assert out["c"] == 1
    assert out["statistic"] in (0.0, None)
    assert out["p_value"] > 0.4


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
    sorted_adj = sorted(adj)
    assert sorted_adj[0] <= sorted_adj[1] <= sorted_adj[2]
    assert all(0 <= p <= 1 for p in adj)


def test_cluster_bootstrap_single_cluster():
    rows = [{"id": "a", "correct": True}, {"id": "a", "correct": False}]
    ci = cluster_bootstrap_ci(rows, seed=0, n_resamples=100)
    assert ci["n_clusters"] == 1
    assert math.isclose(ci["value"], 0.5)


def test_paired_validation_reports_missing_ids():
    base = [{"id": "1", "correct": True}, {"id": "2", "correct": False}]
    var = [{"id": "1", "correct": True}]
    report = validate_paired_rows(base, var)
    assert report["paired_comparison_valid"] is False
    assert report["missing_from_variant"] == ["2"]


def test_paired_difference_bootstrap():
    base = [{"id": "1", "correct": False}, {"id": "2", "correct": True}]
    var = [{"id": "1", "correct": True}, {"id": "2", "correct": True}]
    ci = paired_difference_bootstrap_ci(base, var, seed=0, n_resamples=200)
    assert ci["n_paired"] == 2
    assert ci["paired_mean_diff"] == 0.5
