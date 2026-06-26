"""Seed variance reporting (mean ± std).

Adapted from sober-reasoning leaderboard style (data.json mean±std format).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    mu = sum(vals) / len(vals)
    if len(vals) == 1:
        return {"mean": mu, "std": 0.0, "n": 1}
    var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)
    return {"mean": mu, "std": math.sqrt(var), "n": len(vals)}


def format_mean_std(values: Sequence[float], digits: int = 1) -> str:
    stats = mean_std(values)
    return f"{stats['mean']:.{digits}f}±{stats['std']:.{digits}f}"


def aggregate_by_seed(summaries: List[dict], metric_key: str = "pass_at_1") -> Dict[str, float]:
    values = [s[metric_key] for s in summaries if metric_key in s]
    return mean_std(values)
