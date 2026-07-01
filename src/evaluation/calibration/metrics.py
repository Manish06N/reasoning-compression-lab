"""Calibration metrics — canonical module (V8.2 evaluation/calibration)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from src.evaluation.calibration.reliability import (
    adaptive_ece,
    negative_log_likelihood,
    reliability_diagram_bins,
)
from src.metrics.calibration import (  # noqa: F401 — shared implementation
    aurc_score,
    auroc_score,
    brier_score,
    compute_calibration_metrics,
    expected_calibration_error,
)


def compute_full_calibration_metrics(
    confidences: Sequence[float],
    labels: Sequence[int],
    *,
    n_bins: int = 10,
) -> Dict[str, Optional[float]]:
    base = compute_calibration_metrics(confidences, labels, n_bins=n_bins)
    base["adaptive_ece"] = adaptive_ece(confidences, labels, n_bins=max(n_bins, 10))
    base["nll"] = negative_log_likelihood(confidences, labels)
    return base


def calibration_summary_from_rows(rows: Sequence[dict]) -> Dict[str, object]:
    """Build calibration metrics from scored rows with optional confidence field."""
    confidences: List[float] = []
    labels: List[int] = []
    for row in rows:
        if row.get("correct") is None:
            continue
        conf = row.get("confidence")
        if conf is None:
            conf = 1.0 if row.get("answer_parse_success") else 0.0
        confidences.append(float(conf))
        labels.append(1 if row.get("correct") else 0)
    if not confidences:
        return {}
    metrics = compute_full_calibration_metrics(confidences, labels)
    metrics["reliability_bins"] = reliability_diagram_bins(confidences, labels)
    metrics["confidence_source"] = "row.confidence or answer_parse_success proxy"
    return metrics
