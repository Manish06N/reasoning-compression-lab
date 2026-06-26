"""GPU memory and timing helpers for inference runs."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class GpuStats:
    latency_sec: float = 0.0
    peak_vram_bytes: int = 0

    @property
    def peak_vram_gb(self) -> float:
        return self.peak_vram_bytes / (1024**3)


@contextmanager
def track_gpu() -> Iterator[GpuStats]:
    stats = GpuStats()
    pynvml = _load_pynvml()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    start = time.perf_counter()
    peak = 0
    try:
        yield stats
    finally:
        stats.latency_sec = time.perf_counter() - start
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        peak = mem.used
        stats.peak_vram_bytes = peak


def snapshot_vram_bytes(device_index: int = 0) -> int:
    pynvml = _load_pynvml()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used


def _load_pynvml():
    try:
        import pynvml
    except ImportError as exc:
        raise ImportError("pynvml is required on HPC. pip install pynvml") from exc
    pynvml.nvmlInit()
    return pynvml
