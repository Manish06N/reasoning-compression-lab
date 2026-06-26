"""Selective prediction / risk-coverage helpers."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from src.metrics.calibration import aurc_score


def risk_coverage_curve(
    confidences: Sequence[float],
    labels: Sequence[int],
) -> List[Tuple[float, float]]:
    """Return list of (coverage, risk) points sorted by descending confidence."""
    order = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    points = []
    for k in range(1, len(order) + 1):
        chosen = [labels[i] for i in order[:k]]
        coverage = k / len(order)
        risk = 1.0 - (sum(chosen) / len(chosen))
        points.append((coverage, risk))
    return points


def summarize_selective_risk(confidences: Sequence[float], labels: Sequence[int]) -> Dict[str, float]:
    return {"aurc": aurc_score(confidences, labels)}
