"""Scoring pipeline helpers extracted from score_run.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.calibration.confidence import (
    calibration_availability,
    enrich_scored_row,
)
from src.evaluation.calibration.metrics import calibration_summary_from_rows
from src.evaluation.correctness.scoring import score_item, summarize_scored_rows
from src.evaluation.selective_risk.curves import selective_risk_from_rows
from src.evaluation.statistics.bootstrap import cluster_bootstrap_ci
from src.schemas.validate import validate_jsonl_sample


def load_raw_rows(in_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_all_rows(
    rows: list[dict[str, Any]],
    *,
    allow_parse_proxy: bool,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        score = score_item(row)
        merged = {**row, **score}
        scored.append(enrich_scored_row(merged, allow_parse_proxy=allow_parse_proxy))
    return scored


def build_summary(
    scored: list[dict[str, Any]],
    *,
    in_path: Path,
    out_path: Path,
    display_path,
) -> dict[str, Any]:
    summary = summarize_scored_rows(scored)
    summary["schema_version"] = "summary.v1"
    summary["input"] = display_path(in_path)
    summary["scored_output"] = display_path(out_path)
    if scored:
        summary["cell_id"] = scored[0].get("cell_id")
        summary["quant_config"] = scored[0].get("quant_config")
        summary["seed"] = scored[0].get("seed")
        summary["task"] = scored[0].get("task")
        if all(row.get("id") is not None for row in scored):
            cluster_ci = cluster_bootstrap_ci(scored, cluster_key="id", value_key="correct")
            summary["pass_at_1_cluster_ci95_low"] = cluster_ci["ci_low"]
            summary["pass_at_1_cluster_ci95_high"] = cluster_ci["ci_high"]
            summary["pass_at_1_n_clusters"] = cluster_ci["n_clusters"]
    return summary


def attach_calibration(
    summary: dict[str, Any],
    scored: list[dict[str, Any]],
    *,
    skip_calibration: bool,
    require_calibration: bool,
    allow_parse_proxy: bool,
) -> None:
    """Mutate summary with calibration / selective-risk blocks."""
    if skip_calibration:
        summary["calibration"] = {"skipped": True, "reason": "cli_skip_calibration"}
        summary["selective_risk"] = {"skipped": True, "reason": "cli_skip_calibration"}
        return

    availability = calibration_availability(scored, allow_parse_proxy=allow_parse_proxy)
    summary["calibration_availability"] = availability

    if require_calibration and not availability.get("valid_for_publication"):
        msg = availability.get("message") or "Calibration unavailable."
        raise SystemExit(f"ERROR: {msg}")

    if availability["available"]:
        cal = calibration_summary_from_rows(scored, allow_parse_proxy=allow_parse_proxy)
        if cal and not cal.get("skipped"):
            summary["calibration"] = cal
        risk = selective_risk_from_rows(scored, allow_parse_proxy=allow_parse_proxy)
        if risk and not risk.get("skipped"):
            summary["selective_risk"] = risk
    elif require_calibration:
        msg = availability.get("message") or "Calibration unavailable."
        raise SystemExit(f"ERROR: {msg}")
    else:
        summary["calibration"] = {"skipped": True, "availability": availability}
        summary["selective_risk"] = {"skipped": True, "availability": availability}


def validate_raw_input(in_path: Path) -> dict[str, Any]:
    """Validate a sample of raw rows before scoring."""
    return validate_jsonl_sample(in_path)
