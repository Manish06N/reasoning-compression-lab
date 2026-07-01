"""Integration tests for V8.2 architecture completeness."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v82_directory_layout():
    required = [
        "schemas/raw_response.v1.json",
        "papers/j1/protocol.yaml",
        "papers/j2/protocol.yaml",
        "papers/j3/protocol.yaml",
        "papers/j3/language_matrix.yaml",
        "configs/quantization/registry.yaml",
        "configs/serving/vllm.yaml",
        "configs/serving/sglang.yaml",
        "configs/serving/llamacpp.yaml",
        "src/generation/vllm/runner.py",
        "src/generation/sglang/runner.py",
        "src/generation/llamacpp/runner.py",
        "src/evaluation/correctness/scoring.py",
        "src/evaluation/calibration/metrics.py",
        "src/evaluation/selective_risk/curves.py",
        "src/evaluation/statistics/mcnemar.py",
        "scripts/j1/compare_configs.py",
        "scripts/j2/run_method_pilot.py",
        "scripts/j3/preflight_indic.py",
        "scripts/export_parquet.py",
        "scripts/build_dashboard.py",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    assert not missing, f"Missing V8.2 paths: {missing}"


def test_all_cells_have_prompt_profile():
    cells = ROOT / "configs/cells"
    for path in cells.glob("*.json"):
        import json

        data = json.loads(path.read_text())
        assert "prompt_profile" in data, f"{path.name} missing prompt_profile"


def test_all_prompt_profile_templates_exist():
    from src.runners.config_utils import PROMPT_PROFILES

    missing = [
        rel
        for templates in PROMPT_PROFILES.values()
        for rel in templates.values()
        if not (ROOT / rel).exists()
    ]
    assert not missing, f"Missing prompt templates: {missing}"


def test_calibration_and_selective_risk_from_rows():
    from src.evaluation.calibration.metrics import calibration_summary_from_rows
    from src.evaluation.selective_risk.curves import selective_risk_from_rows

    rows = [
        {"id": "1", "correct": True, "confidence": 0.9, "confidence_source": "explicit"},
        {"id": "2", "correct": False, "confidence": 0.8, "confidence_source": "explicit"},
        {"id": "3", "correct": True, "confidence": 0.3, "confidence_source": "explicit"},
    ]
    cal = calibration_summary_from_rows(rows)
    assert cal.get("skipped") is False
    assert "brier" in cal
    risk = selective_risk_from_rows(rows)
    assert risk.get("skipped") is False
    assert "aurc" in risk
    assert "coverage_at_risk_10pct" in risk


def test_adaptive_ece_and_nll():
    from src.evaluation.calibration.reliability import adaptive_ece, negative_log_likelihood

    conf = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    assert 0 <= adaptive_ece(conf, labels) <= 1
    assert negative_log_likelihood(conf, labels) >= 0
