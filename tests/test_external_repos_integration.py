"""Tests for Pareto frontier and QRM baseline comparison."""

import json
from pathlib import Path

from scripts.compare_qrm_baseline import compare_summary
from src.metrics.pareto_frontier import build_frontier_points, pareto_frontier
from src.runners.config_utils import load_yaml


def test_pareto_frontier_selects_non_dominated():
    summaries = [
        {"cell_id": "a", "quant_config": "bf16", "pass_at_1": 0.5, "cost_per_correct_seconds": 100},
        {"cell_id": "b", "quant_config": "fp8", "pass_at_1": 0.48, "cost_per_correct_seconds": 80},
        {"cell_id": "c", "quant_config": "gptq4", "pass_at_1": 0.45, "cost_per_correct_seconds": 90},
    ]
    points = build_frontier_points(summaries)
    frontier = pareto_frontier(points)
    assert len(frontier) >= 1
    assert any(p.quant_config == "bf16" for p in frontier)


def test_compare_qrm_baseline_fails_low_pass_at_1():
    targets = load_yaml("configs/baselines/qrm_literature_targets.yaml")
    summary = {
        "cell_id": "level_a_qwen7b_bf16_math500_seed0",
        "task": "math500",
        "pass_at_1": 0.07,
        "truncation_rate": 0.9,
        "parse_failure_rate": 0.86,
    }
    report = compare_summary(summary, targets)
    assert report["passed"] is False
    assert any(c["status"] == "FAIL" for c in report["checks"])
