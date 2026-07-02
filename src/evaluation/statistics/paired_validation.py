"""Paired comparison validation for model-vs-model reports."""

from __future__ import annotations

import random
from typing import Any, Sequence


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


def validate_paired_rows(
    baseline_rows: Sequence[dict[str, Any]],
    variant_rows: Sequence[dict[str, Any]],
    *,
    id_key: str = "id",
) -> dict[str, Any]:
    """Compare row sets and report whether a paired comparison is valid."""
    base_ids = [row.get(id_key) for row in baseline_rows if row.get(id_key) is not None]
    var_ids = [row.get(id_key) for row in variant_rows if row.get(id_key) is not None]
    base_set = set(base_ids)
    var_set = set(var_ids)
    common = sorted(base_set & var_set)
    missing_from_variant = sorted(base_set - var_set)
    missing_from_baseline = sorted(var_set - base_set)
    return {
        "baseline_n": len(base_ids),
        "variant_n": len(var_ids),
        "common_n": len(common),
        "missing_from_variant": missing_from_variant,
        "missing_from_baseline": missing_from_baseline,
        "paired_comparison_valid": (
            len(missing_from_variant) == 0
            and len(missing_from_baseline) == 0
            and len(common) > 0
            and len(base_ids) == len(var_ids) == len(common)
        ),
    }


def paired_difference_bootstrap_ci(
    baseline_rows: Sequence[dict[str, Any]],
    variant_rows: Sequence[dict[str, Any]],
    *,
    id_key: str = "id",
    seed: int = 0,
    n_resamples: int = 2000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap CI on mean paired accuracy difference (variant - baseline)."""
    base_by_id = {row[id_key]: bool(row.get("correct")) for row in baseline_rows if row.get(id_key)}
    var_by_id = {row[id_key]: bool(row.get("correct")) for row in variant_rows if row.get(id_key)}
    common_ids = sorted(set(base_by_id) & set(var_by_id))
    diffs = [float(var_by_id[i]) - float(base_by_id[i]) for i in common_ids]
    if not diffs:
        return {
            "paired_mean_diff": 0.0,
            "paired_diff_ci95_low": 0.0,
            "paired_diff_ci95_high": 0.0,
            "n_paired": 0,
        }
    observed = sum(diffs) / len(diffs)
    if len(diffs) == 1:
        return {
            "paired_mean_diff": observed,
            "paired_diff_ci95_low": observed,
            "paired_diff_ci95_high": observed,
            "n_paired": 1,
        }
    rng = random.Random(seed)
    n = len(diffs)
    samples: list[float] = []
    for _ in range(n_resamples):
        draw = [diffs[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    alpha = 1.0 - confidence
    return {
        "paired_mean_diff": observed,
        "paired_diff_ci95_low": _percentile(samples, alpha / 2.0),
        "paired_diff_ci95_high": _percentile(samples, 1.0 - alpha / 2.0),
        "n_paired": n,
    }
