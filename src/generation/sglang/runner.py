"""SGLang serving backend stub for Paper 2 method-selection pilot (V8.2 §8).

Reference: external_repos/07-j2-serving-acceleration/sglang/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SGLangPilotConfig:
    """Configuration for J2 cross-stack pilot trials."""

    model_path: str
    host: str = "127.0.0.1"
    port: int = 30000
    tensor_parallel: int = 1
    max_model_len: int = 32768
    concurrency_levels: List[int] = field(default_factory=lambda: [1, 4, 8])
    tasks: List[str] = field(default_factory=lambda: ["math500_subset", "gpqa_subset"])
    reference_repo: str = "external_repos/07-j2-serving-acceleration/sglang"


def pilot_manifest(cfg: SGLangPilotConfig) -> Dict[str, Any]:
    return {
        "backend": "sglang",
        "status": "stub_ready",
        "config": cfg.__dict__,
        "next_step": "scripts/j2/run_method_pilot.py --backend sglang",
    }


def check_sglang_available() -> bool:
    try:
        import sglang  # noqa: F401
        return True
    except ImportError:
        return False
