"""Risk-coverage curves and coverage at fixed risk (V8.2 §6.5)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from src.metrics.calibration import aurc_score
from src.metrics.selective_risk import risk_coverage_curve


def coverage_at_fixed_risk(
    confidences: Sequence[float],
    labels: Sequence[int],
    *,
    target_risk: float = 0.1,
) -> Dict[str, float]:
    curve = risk_coverage_curve(confidences, labels)
    best = {"coverage": 0.0, "risk": 1.0}
    for coverage, risk in curve:
        if risk <= target_risk and coverage >= best["coverage"]:
            best = {"coverage": coverage, "risk": risk}
    return {
        "target_risk": target_risk,
        "coverage_at_risk": best["coverage"],
        "achieved_risk": best["risk"],
    }


def summarize_selective_risk_full(
    confidences: Sequence[float],
    labels: Sequence[int],
) -> Dict[str, object]:
    curve: List[Tuple[float, float]] = risk_coverage_curve(confidences, labels)
    out: Dict[str, object] = {
        "aurc": aurc_score(confidences, labels),
        "risk_coverage_curve": [{"coverage": c, "risk": r} for c, r in curve],
    }
    for target in (0.05, 0.10, 0.20):
        out[f"coverage_at_risk_{int(target * 100)}pct"] = coverage_at_fixed_risk(
            confidences, labels, target_risk=target
        )
    return out


def selective_risk_from_rows(rows: Sequence[dict]) -> Dict[str, object]:
    confidences = []
    labels = []
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
    return summarize_selective_risk_full(confidences, labels)
