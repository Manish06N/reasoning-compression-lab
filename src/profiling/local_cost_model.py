"""Local HPC cost model (GPU-seconds or USD)."""

from __future__ import annotations

from typing import Optional


def gpu_seconds_cost(latency_sec: float) -> float:
    return latency_sec


def usd_cost(latency_sec: float, gpu_hour_rate: float) -> float:
    return latency_sec * (gpu_hour_rate / 3600.0)


def cost_per_correct_from_totals(
    total_gpu_seconds: float,
    num_correct: int,
    gpu_hour_rate: Optional[float] = None,
) -> float:
    if num_correct == 0:
        return float("inf")
    if gpu_hour_rate is None:
        return total_gpu_seconds / num_correct
    return usd_cost(total_gpu_seconds, gpu_hour_rate) / num_correct
