"""Tests for publication preflight CI subset."""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_preflight_ci_subset_passes():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/hpc/07_preflight_publication.py"), "--ci"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_preflight_module_loads():
    spec = importlib.util.spec_from_file_location(
        "preflight",
        ROOT / "scripts/hpc/07_preflight_publication.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "check_prompt")
