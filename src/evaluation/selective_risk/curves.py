"""Risk-coverage curves and coverage at fixed risk (V8.2 §6.5)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from src.evaluation.calibration.confidence import (
    calibration_availability,
    collect_calibration_pairs,
)
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


def selective_risk_from_rows(
    rows: Sequence[dict],
    *,
    allow_parse_proxy: bool = False,
) -> Dict[str, object]:
    availability = calibration_availability(rows, allow_parse_proxy=allow_parse_proxy)
    if not availability["available"]:
        return {"skipped": True, "availability": availability}

    confidences, labels, meta = collect_calibration_pairs(
        rows, allow_parse_proxy=allow_parse_proxy
    )
    if not confidences:
        return {"skipped": True, "availability": availability}

    out = summarize_selective_risk_full(confidences, labels)
    out["confidence_valid_for_calibration"] = availability["valid_for_publication"]
    out["confidence_sources_seen"] = meta.get("confidence_sources_seen", [])
    out["skipped"] = False
    out["availability"] = availability
    return out
