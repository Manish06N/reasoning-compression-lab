"""Calibration metrics — canonical module (V8.2 evaluation/calibration)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from src.evaluation.calibration.confidence import (
    calibration_availability,
    collect_calibration_pairs,
)
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


def calibration_summary_from_rows(
    rows: Sequence[dict],
    *,
    allow_parse_proxy: bool = False,
) -> Dict[str, object]:
    """Build calibration metrics when valid confidence is available."""
    availability = calibration_availability(rows, allow_parse_proxy=allow_parse_proxy)
    if not availability["available"]:
        return {"skipped": True, "availability": availability}

    confidences, labels, meta = collect_calibration_pairs(
        rows, allow_parse_proxy=allow_parse_proxy
    )
    if not confidences:
        return {"skipped": True, "availability": availability}

    metrics = compute_full_calibration_metrics(confidences, labels)
    metrics["reliability_bins"] = reliability_diagram_bins(confidences, labels)
    metrics["confidence_valid_for_calibration"] = availability["valid_for_publication"]
    metrics["confidence_sources_seen"] = meta.get("confidence_sources_seen", [])
    metrics["skipped"] = False
    metrics["availability"] = availability
    return metrics
