"""llama.cpp / GGUF local serving stub for Paper 3 (V8.2 §10.4).

Reference: external_repos/09-local-edge/llama.cpp/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class LlamaCppPilotConfig:
    model_gguf_path: str
    n_gpu_layers: int = -1
    ctx_size: int = 8192
    threads: int = 8
    quant_levels: List[str] = field(default_factory=lambda: ["Q4_K_M", "Q8_0", "F16"])
    reference_repo: str = "external_repos/09-local-edge/llama.cpp"


def pilot_manifest(cfg: LlamaCppPilotConfig) -> Dict[str, Any]:
    return {
        "backend": "llama.cpp",
        "status": "stub_ready",
        "config": cfg.__dict__,
        "next_step": "scripts/j3/run_local_transfer.py",
    }
