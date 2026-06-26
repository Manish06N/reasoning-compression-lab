"""GPU profiling helpers for inference runs."""

from __future__ import annotations

from typing import Any, Dict, List

from src.profiling.gpu_stats import GpuStats, snapshot_vram_bytes, track_gpu


def profile_generation(fn, *args, **kwargs) -> tuple[Any, Dict[str, float]]:
    vram_before = snapshot_vram_bytes()
    with track_gpu() as stats:
        result = fn(*args, **kwargs)
    vram_after = snapshot_vram_bytes()
    profile = {
        "latency_sec": stats.latency_sec,
        "peak_vram_gb": max(vram_before, vram_after, stats.peak_vram_bytes) / (1024**3),
    }
    return result, profile


def summarize_profiles(rows: List[dict]) -> dict:
    latencies = [r["latency_sec"] for r in rows if "latency_sec" in r]
    vrams = [r["peak_vram_gb"] for r in rows if "peak_vram_gb" in r]
    out = {}
    if latencies:
        out["latency_sec_mean"] = sum(latencies) / len(latencies)
        out["latency_sec_p50"] = sorted(latencies)[len(latencies) // 2]
    if vrams:
        out["peak_vram_gb_max"] = max(vrams)
        out["peak_vram_gb_mean"] = sum(vrams) / len(vrams)
    return out
