"""Tests for QRM baseline comparison gate."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_compare_summary_includes_provenance_and_gate():
    from scripts.compare_qrm_baseline import compare_summary
    from src.runners.config_utils import load_yaml

    targets_path = ROOT / "configs/baselines/qrm_literature_targets.yaml"
    targets = load_yaml(targets_path)
    summary = {
        "cell_id": "level_a_qwen7b_bf16_math500_seed0",
        "task": "math500",
        "pass_at_1": 0.93,
        "truncation_rate": 0.05,
        "parse_failure_rate": 0.02,
        "completion_tokens_mean": 5000,
    }
    report = compare_summary(summary, targets, targets_path=targets_path)

    assert report["targets_provenance"]["yaml_sha256"]
    assert report["gate"]["reference_pct"] == 92.8
    assert report["gate"]["sanity_band_pct"] == [87.8, 97.8]
    assert report["passed"] is True


def test_compare_summary_fails_broken_pipeline_band():
    from scripts.compare_qrm_baseline import compare_summary
    from src.runners.config_utils import load_yaml

    targets_path = ROOT / "configs/baselines/qrm_literature_targets.yaml"
    targets = load_yaml(targets_path)
    summary = {
        "cell_id": "level_a_qwen7b_bf16_math500_seed0",
        "task": "math500",
        "pass_at_1": 0.07,
    }
    report = compare_summary(summary, targets, targets_path=targets_path)
    assert report["passed"] is False


def test_gpqa_band_is_mid_forties_not_math500():
    from scripts.compare_qrm_baseline import compare_summary
    from src.runners.config_utils import load_yaml

    targets_path = ROOT / "configs/baselines/qrm_literature_targets.yaml"
    targets = load_yaml(targets_path)
    task_targets = targets["models"]["DeepSeek-R1-Distill-Qwen-7B"]["GPQA-Diamond"]
    band = task_targets["pass_at_1_pct"]
    assert band["sanity_min"] >= 40
    assert band["sanity_max"] <= 60
    assert band["reference"] == 49.1

    # 60% would false-pass old MATH-500-style band but must fail GPQA gate if ref is ~49
    summary = {
        "cell_id": "level_c_qwen7b_fp8_gpqa_seed0",
        "task": "gpqa_diamond",
        "pass_at_1": 0.60,
    }
    report = compare_summary(summary, targets, targets_path=targets_path)
    assert report["passed"] is False
