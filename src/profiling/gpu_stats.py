"""GPU memory, utilization, power, and timing helpers for inference runs."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class GpuStats:
    latency_sec: float = 0.0
    peak_vram_bytes: int = 0
    samples: int = 0
    gpu_util_max: Optional[float] = None
    gpu_util_sum: float = 0.0
    power_watts_max: Optional[float] = None
    power_watts_sum: float = 0.0
    energy_joules: Optional[float] = None

    @property
    def peak_vram_gb(self) -> float:
        return self.peak_vram_bytes / (1024**3)

    @property
    def gpu_util_mean(self) -> Optional[float]:
        if not self.samples:
            return None
        return self.gpu_util_sum / self.samples

    @property
    def power_watts_mean(self) -> Optional[float]:
        if not self.samples:
            return None
        return self.power_watts_sum / self.samples


def nvml_device_index(default: int = 0) -> int:
    """Map CUDA_VISIBLE_DEVICES to the physical NVML index for this process."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip():
        parts = [part.strip() for part in visible.split(",") if part.strip()]
        if len(parts) == 1 and parts[0].isdigit():
            return int(parts[0])
    return default


@contextmanager
def track_gpu(sample_interval_sec: float = 0.5) -> Iterator[GpuStats]:
    stats = GpuStats()
    pynvml = _load_pynvml()
    handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_device_index())
    stop = threading.Event()
    lock = threading.Lock()

    def sample_loop() -> None:
        last_t = time.perf_counter()
        while not stop.is_set():
            now = time.perf_counter()
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                power_watts = _read_power_watts(pynvml, handle)
            except Exception:
                time.sleep(sample_interval_sec)
                continue

            dt = max(0.0, now - last_t)
            last_t = now
            with lock:
                stats.samples += 1
                stats.peak_vram_bytes = max(stats.peak_vram_bytes, mem.used)
                stats.gpu_util_sum += float(util.gpu)
                stats.gpu_util_max = (
                    float(util.gpu)
                    if stats.gpu_util_max is None
                    else max(stats.gpu_util_max, float(util.gpu))
                )
                if power_watts is not None:
                    stats.power_watts_sum += power_watts
                    stats.power_watts_max = (
                        power_watts
                        if stats.power_watts_max is None
                        else max(stats.power_watts_max, power_watts)
                    )
                    stats.energy_joules = (stats.energy_joules or 0.0) + power_watts * dt
            time.sleep(sample_interval_sec)

    start = time.perf_counter()
    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()
    try:
        yield stats
    finally:
        stop.set()
        sampler.join(timeout=max(1.0, sample_interval_sec * 2))
        stats.latency_sec = time.perf_counter() - start
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        with lock:
            stats.peak_vram_bytes = max(stats.peak_vram_bytes, mem.used)


def snapshot_vram_bytes(device_index: int | None = None) -> int:
    pynvml = _load_pynvml()
    index = nvml_device_index() if device_index is None else device_index
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used


def _read_power_watts(pynvml, handle) -> Optional[float]:
    try:
        return float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
    except Exception:
        return None


def _load_pynvml():
    try:
        import pynvml
    except ImportError as exc:
        raise ImportError("pynvml is required on HPC. pip install pynvml") from exc
    pynvml.nvmlInit()
    return pynvml
