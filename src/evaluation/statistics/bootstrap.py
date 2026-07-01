"""Cluster bootstrap by item id (V8.2 §13)."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Hashable, Sequence


def _percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def cluster_bootstrap_ci(
    rows: Sequence[dict],
    *,
    cluster_key: str = "id",
    value_key: str = "correct",
    statistic: Callable[[Sequence[float]], float] | None = None,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """
    Bootstrap over item clusters (all rows sharing cluster_key stay together).

    For single-sample rows, each id is one cluster. For multi-sample rows (maj@k),
    resample clusters and flatten samples within each selected cluster.
    """
    stat = statistic or (lambda xs: sum(xs) / len(xs) if xs else 0.0)
    clusters: dict[Hashable, list[float]] = defaultdict(list)
    for row in rows:
        key = row.get(cluster_key)
        if key is None:
            continue
        val = row.get(value_key)
        if val is None:
            continue
        clusters[key].append(1.0 if val else 0.0)

    cluster_ids = list(clusters.keys())
    if not cluster_ids:
        return {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_clusters": 0}

    def cluster_mean(values: list[float]) -> float:
        return sum(values) / len(values)

    observed_vals = [cluster_mean(clusters[cid]) for cid in cluster_ids]
    observed = float(stat(observed_vals))

    if len(cluster_ids) == 1:
        return {"value": observed, "ci_low": observed, "ci_high": observed, "n_clusters": 1}

    rng = random.Random(seed)
    n_c = len(cluster_ids)
    samples: list[float] = []
    for _ in range(n_resamples):
        draw = [cluster_ids[rng.randrange(n_c)] for _ in range(n_c)]
        vals = [cluster_mean(clusters[cid]) for cid in draw]
        samples.append(float(stat(vals)))

    alpha = 1.0 - confidence
    return {
        "value": observed,
        "ci_low": _percentile(samples, alpha / 2.0),
        "ci_high": _percentile(samples, 1.0 - alpha / 2.0),
        "n_clusters": n_c,
    }
