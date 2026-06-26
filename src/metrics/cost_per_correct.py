"""Cost-per-correct metrics.

Adapted from Cost-of-Pass/cost_of_pass/evaluation/estimate.py (CoP = E[cost] / E[performance]).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence


def query_cost_seconds(latency_sec: float, gpu_hour_rate: Optional[float] = None) -> float:
    """Cost of one query in GPU-seconds or USD if gpu_hour_rate is set."""
    if gpu_hour_rate is None:
        return latency_sec
    return latency_sec * (gpu_hour_rate / 3600.0)


def cost_of_pass(records: Sequence[dict], gpu_hour_rate: Optional[float] = None) -> float:
    """Expected cost divided by expected performance (pass rate proxy)."""
    if not records:
        return float("inf")
    costs = []
    perf = []
    for row in records:
        latency = row.get("latency_sec", 0.0)
        costs.append(query_cost_seconds(latency, gpu_hour_rate))
        perf.append(1.0 if row.get("correct") else 0.0)
    e_cost = sum(costs) / len(costs)
    e_perf = sum(perf) / len(perf)
    if e_perf == 0:
        return float("inf")
    return e_cost / e_perf


def cost_per_correct(
    records: Sequence[dict],
    gpu_hour_rate: Optional[float] = None,
) -> float:
    """Total inference cost divided by number of correct answers."""
    correct_rows = [r for r in records if r.get("correct")]
    if not correct_rows:
        return float("inf")
    total_cost = sum(
        query_cost_seconds(r.get("latency_sec", 0.0), gpu_hour_rate) for r in records
    )
    return total_cost / len(correct_rows)


def summarize_cost_metrics(
    records: Sequence[dict],
    gpu_hour_rate: Optional[float] = None,
) -> Dict[str, float]:
    total_latency = sum(r.get("latency_sec", 0.0) for r in records)
    num_correct = sum(1 for r in records if r.get("correct"))
    return {
        "total_gpu_seconds": total_latency,
        "cost_per_correct_seconds": cost_per_correct(records, gpu_hour_rate=None),
        "cost_of_pass_seconds": cost_of_pass(records, gpu_hour_rate=None),
        "num_correct": num_correct,
    }
