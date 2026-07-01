"""Pareto frontier for cost-per-correct across a compression family.

Adapted from Cost-of-Pass/cost_of_pass/evaluation/estimate.py (FrontierCostofPass).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


@dataclass
class FrontierPoint:
    cell_id: str
    quant_config: str
    pass_at_1: float
    cost_per_correct_seconds: float
    latency_sec_mean: float
    peak_vram_gb_max: Optional[float]
    on_frontier: bool = False


def _finite_cost(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def build_frontier_points(summaries: Sequence[dict]) -> List[FrontierPoint]:
    """One point per cell summary JSON (must include pass_at_1 + cost metrics)."""
    points: List[FrontierPoint] = []
    for summary in summaries:
        cost = _finite_cost(summary.get("cost_per_correct_seconds"))
        if cost is None:
            continue
        points.append(
            FrontierPoint(
                cell_id=str(summary.get("cell_id", "")),
                quant_config=str(summary.get("quant_config", "")),
                pass_at_1=float(summary.get("pass_at_1", 0.0)),
                cost_per_correct_seconds=cost,
                latency_sec_mean=float(summary.get("latency_sec_mean", 0.0)),
                peak_vram_gb_max=summary.get("peak_vram_gb_max"),
            )
        )
    return points


def pareto_frontier(points: Sequence[FrontierPoint]) -> List[FrontierPoint]:
    """Non-dominated points: maximize pass_at_1, minimize cost_per_correct_seconds."""
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: (-p.pass_at_1, p.cost_per_correct_seconds))
    frontier: List[FrontierPoint] = []
    best_cost = float("inf")
    for pt in sorted_pts:
        if pt.cost_per_correct_seconds < best_cost:
            frontier.append(pt)
            best_cost = pt.cost_per_correct_seconds
    for pt in frontier:
        pt.on_frontier = True
    frontier_ids = {id(p) for p in frontier}
    for pt in points:
        if id(pt) in frontier_ids:
            pt.on_frontier = True
    return frontier


def frontier_table_rows(points: Sequence[FrontierPoint]) -> List[dict]:
    return [
        {
            "cell_id": p.cell_id,
            "quant_config": p.quant_config,
            "pass_at_1": round(p.pass_at_1, 4),
            "cost_per_correct_seconds": round(p.cost_per_correct_seconds, 2),
            "latency_sec_mean": round(p.latency_sec_mean, 2),
            "peak_vram_gb_max": p.peak_vram_gb_max,
            "on_pareto_frontier": p.on_frontier,
        }
        for p in points
    ]


def filter_by_model_family(summaries: Sequence[dict], family_hint: str) -> List[dict]:
    """Filter summaries where cell_id or model_path matches family (e.g. qwen7b)."""
    hint = family_hint.lower()
    out = []
    for s in summaries:
        blob = f"{s.get('cell_id', '')} {s.get('model_path', '')}".lower()
        if hint in blob:
            out.append(s)
    return out
