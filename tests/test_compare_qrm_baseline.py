"""Tests for QRM baseline comparison gate."""

from __future__ import annotations

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
    assert report["gate"]["reference_pct"] == 94.0
    assert report["gate"]["gate_type"] == "hard"
    assert report["gate"]["sanity_band_pct"] == [89.0, 99.0]
    assert report["hard_passed"] is True


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
    assert report["hard_passed"] is False


def test_gpqa_sanity_gate_warns_not_hard_fails():
    from scripts.compare_qrm_baseline import compare_summary
    from src.runners.config_utils import load_yaml

    targets_path = ROOT / "configs/baselines/qrm_literature_targets.yaml"
    targets = load_yaml(targets_path)
    task_targets = targets["models"]["DeepSeek-R1-Distill-Qwen-7B"]["GPQA-Diamond"]
    band_cfg = task_targets["pass_at_1_pct"]
    assert band_cfg["reference"] == 51.0
    assert band_cfg["tolerance_pp"] == 8.0
    assert task_targets["gate"] == "sanity"

    summary = {
        "cell_id": "level_c_qwen7b_fp8_gpqa_seed0",
        "task": "gpqa_diamond",
        "pass_at_1": 0.60,
    }
    report = compare_summary(summary, targets, targets_path=targets_path)
    assert report["hard_passed"] is True
    pass_check = next(c for c in report["checks"] if c["metric"] == "pass_at_1_pct")
    assert pass_check["status"] == "SANITY_WARN"


def test_llama_math500_uses_table4_not_table1():
    from src.runners.config_utils import load_yaml

    targets_path = ROOT / "configs/baselines/qrm_literature_targets.yaml"
    targets = load_yaml(targets_path)
    cfg = targets["models"]["DeepSeek-R1-Distill-Llama-8B"]["MATH-500"]["pass_at_1_pct"]
    assert "Table 4" in cfg["source"]
    assert cfg["reference"] == 91.0


def test_llama_gsm8k_marked_unused():
    from scripts.compare_qrm_baseline import compare_summary
    from src.runners.config_utils import load_yaml

    targets_path = ROOT / "configs/baselines/qrm_literature_targets.yaml"
    targets = load_yaml(targets_path)
    assert targets["models"]["DeepSeek-R1-Distill-Llama-8B"]["GSM8K"]["status"] == "unused"

    summary = {
        "cell_id": "level_c_llama8b_bf16_gsm8k_seed0",
        "task": "gsm8k",
        "pass_at_1": 0.88,
    }
    report = compare_summary(summary, targets, targets_path=targets_path)
    assert report["passed"] is None
    assert report["checks"][0]["status"] == "SKIP"
