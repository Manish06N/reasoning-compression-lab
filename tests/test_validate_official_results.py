"""Tests for the official-QRM output gate."""

from scripts.hpc.qrm_parity.validate_official_results import analyze_rows, gate_errors


def _row(*, correct: float = 1.0, text: str = "Reasoning. \\boxed{3}") -> dict:
    return {"generated_text": text, "metrics": {"extractive_match": correct}}


def test_healthy_official_result_passes_strict_gate():
    rows = [_row(), _row(text="Another solution. \\boxed{4}")]
    report = analyze_rows(rows, encode=lambda text: list(range(len(text.split()))))

    errors = gate_errors(
        report,
        expected_rows=2,
        min_accuracy=1.0,
        min_boxed_rate=1.0,
        max_token_limit_hits=0,
        max_repetition_rows=0,
    )

    assert errors == []
    assert report["completion_tokens_max"] == 3


def test_gate_rejects_bad_accuracy_format_cap_and_repetition():
    rows = [
        _row(correct=0.0, text="the " * 25),
        _row(text="long answer \\boxed{3}"),
    ]
    report = analyze_rows(
        rows,
        encode=lambda text: list(range(32768 if text.startswith("long") else 25)),
    )

    errors = gate_errors(
        report,
        expected_rows=2,
        min_accuracy=1.0,
        min_boxed_rate=1.0,
        max_token_limit_hits=0,
        max_repetition_rows=0,
    )

    assert any(error.startswith("accuracy=") for error in errors)
    assert any(error.startswith("boxed_rate=") for error in errors)
    assert any(error.startswith("token_limit_hits=") for error in errors)
    assert any(error.startswith("repetition_rows=") for error in errors)


def test_gate_rejects_missing_rows_and_metrics():
    report = analyze_rows([{"generated_text": "\\boxed{1}", "metrics": {}}])

    errors = gate_errors(
        report,
        expected_rows=2,
        min_accuracy=0.0,
        min_boxed_rate=0.0,
        max_token_limit_hits=1,
        max_repetition_rows=1,
    )

    assert "rows=1 (expected 2)" in errors
    assert "numeric extractive_match metrics=0/1" in errors
