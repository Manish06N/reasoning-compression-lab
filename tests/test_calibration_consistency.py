"""Tests for calibration consistency agreement metrics."""

from src.evaluation.calibration.consistency import agreement_rates


def test_agreement_rates_identical_answers():
    rates = agreement_rates(["0.5", "0.5", "0.5"])
    assert rates["raw_string_agreement"] == 1.0
    assert rates["confidence_method"] == "self_consistency_5"
