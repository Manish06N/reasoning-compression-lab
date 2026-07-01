"""Tests for calibration confidence resolution (fail-closed)."""

from src.evaluation.calibration.confidence import (
    calibration_availability,
    enrich_scored_row,
    resolve_row_confidence,
)
from src.evaluation.calibration.metrics import calibration_summary_from_rows
from src.evaluation.selective_risk.curves import selective_risk_from_rows


def test_parse_success_not_valid_by_default():
    row = {"correct": True, "answer_parse_success": True}
    value, source, valid = resolve_row_confidence(row, allow_parse_proxy=False)
    assert value is None
    assert valid is False


def test_explicit_confidence_is_valid():
    row = {"correct": True, "confidence": 0.85, "confidence_source": "self_consistency_5"}
    value, source, valid = resolve_row_confidence(row)
    assert value == 0.85
    assert source == "self_consistency_5"
    assert valid is True


def test_calibration_skipped_without_valid_confidence():
    rows = [
        {"id": "1", "correct": True, "answer_parse_success": True},
        {"id": "2", "correct": False, "answer_parse_success": True},
    ]
    avail = calibration_availability(rows)
    assert avail["available"] is False
    assert avail["valid_for_publication"] is False

    cal = calibration_summary_from_rows(rows)
    assert cal.get("skipped") is True

    risk = selective_risk_from_rows(rows)
    assert risk.get("skipped") is True


def test_calibration_with_valid_confidence():
    rows = [
        {"id": "1", "correct": True, "confidence": 0.9, "confidence_source": "explicit"},
        {"id": "2", "correct": False, "confidence": 0.2, "confidence_source": "explicit"},
        {"id": "3", "correct": True, "confidence": 0.7, "confidence_source": "explicit"},
    ]
    avail = calibration_availability(rows)
    assert avail["available"] is True
    assert avail["valid_for_publication"] is True

    cal = calibration_summary_from_rows(rows)
    assert cal.get("skipped") is False
    assert "brier" in cal
    assert cal["confidence_valid_for_calibration"] is True

    risk = selective_risk_from_rows(rows)
    assert risk.get("skipped") is False
    assert "aurc" in risk


def test_parse_proxy_allowed_but_not_publication_valid():
    rows = [{"id": "1", "correct": True, "answer_parse_success": True}]
    avail = calibration_availability(rows, allow_parse_proxy=True)
    assert avail["available"] is True
    assert avail["valid_for_publication"] is False

    cal = calibration_summary_from_rows(rows, allow_parse_proxy=True)
    assert cal.get("skipped") is False
    assert cal["confidence_valid_for_calibration"] is False


def test_enrich_scored_row_fields():
    row = enrich_scored_row({"correct": True, "confidence": 0.5, "confidence_source": "explicit"})
    assert row["confidence_value"] == 0.5
    assert row["confidence_valid_for_calibration"] is True
