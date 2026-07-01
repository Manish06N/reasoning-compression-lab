"""Confidence resolution for calibration and selective-risk metrics (J1)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

# Sources acceptable for manuscript calibration / selective-risk claims.
VALID_CALIBRATION_SOURCES = frozenset(
    {
        "answer_logprob",
        "normalized_sequence_logprob",
        "verbalized_confidence",
        "self_consistency_5",
        "trained_confidence_predictor",
        "explicit",  # row.confidence set without a finer label (e.g. upstream logprob)
    }
)

PARSE_PROXY_SOURCE = "answer_parse_success"


def enrich_scored_row(row: dict, *, allow_parse_proxy: bool = False) -> dict:
    """Add confidence_value, confidence_source, confidence_valid_for_calibration."""
    value, source, valid = resolve_row_confidence(row, allow_parse_proxy=allow_parse_proxy)
    out = dict(row)
    out["confidence_value"] = value
    out["confidence_source"] = source
    out["confidence_valid_for_calibration"] = valid
    if "answer_parse_success" not in out:
        out["answer_parse_success"] = bool(out.get("extracted_answer"))
    return out


def resolve_row_confidence(
    row: dict,
    *,
    allow_parse_proxy: bool = False,
) -> Tuple[float | None, str | None, bool]:
    """Return (confidence_value, confidence_source, valid_for_calibration)."""
    raw_conf = row.get("confidence")
    declared_source = row.get("confidence_source")

    if raw_conf is not None:
        source = str(declared_source or "explicit")
        if source in VALID_CALIBRATION_SOURCES:
            return float(raw_conf), source, True
        if source == PARSE_PROXY_SOURCE and allow_parse_proxy:
            return float(raw_conf), source, False
        # Unknown source with explicit numeric confidence — treat as explicit.
        if declared_source is None:
            return float(raw_conf), "explicit", True
        return float(raw_conf), source, False

    if allow_parse_proxy:
        parse_ok = bool(row.get("answer_parse_success"))
        return (1.0 if parse_ok else 0.0), PARSE_PROXY_SOURCE, False

    return None, None, False


def collect_calibration_pairs(
    rows: Sequence[dict],
    *,
    allow_parse_proxy: bool = False,
) -> Tuple[List[float], List[int], Dict[str, Any]]:
    """Collect (confidence, label) pairs for rows with known correctness."""
    confidences: List[float] = []
    labels: List[int] = []
    sources: List[str] = []
    valid_flags: List[bool] = []

    for row in rows:
        if row.get("correct") is None:
            continue
        value, source, valid = resolve_row_confidence(row, allow_parse_proxy=allow_parse_proxy)
        if value is None:
            continue
        confidences.append(float(value))
        labels.append(1 if row.get("correct") else 0)
        sources.append(str(source))
        valid_flags.append(valid)

    meta: Dict[str, Any] = {
        "n_pairs": len(confidences),
        "confidence_sources_seen": sorted(set(sources)),
        "all_valid_for_calibration": bool(valid_flags) and all(valid_flags),
        "n_valid_for_calibration": sum(1 for v in valid_flags if v),
    }
    return confidences, labels, meta


def calibration_availability(
    rows: Sequence[dict],
    *,
    allow_parse_proxy: bool = False,
) -> Dict[str, Any]:
    """Summarise whether calibration metrics can be computed safely."""
    _, _, meta = collect_calibration_pairs(rows, allow_parse_proxy=allow_parse_proxy)
    if meta["n_pairs"] == 0:
        return {
            "available": False,
            "valid_for_publication": False,
            "reason": "no_confidence_values",
            "message": (
                "Calibration cannot be calculated: no confidence values are available. "
                "Use maj@5 (run_inference_multisample.py + compute_calibration.py) or "
                "logprob-based confidence."
            ),
            **meta,
        }
    if meta["all_valid_for_calibration"]:
        return {
            "available": True,
            "valid_for_publication": True,
            "reason": None,
            "message": None,
            **meta,
        }
    if allow_parse_proxy:
        return {
            "available": True,
            "valid_for_publication": False,
            "reason": "parse_success_proxy",
            "message": (
                "Calibration computed from answer_parse_success proxy only — "
                "NOT valid for publication claims."
            ),
            **meta,
        }
    return {
        "available": False,
        "valid_for_publication": False,
        "reason": "no_valid_confidence_source",
        "message": (
            "Calibration cannot be calculated: no valid confidence source is available. "
            "Parse success is not a confidence score."
        ),
        **meta,
    }
