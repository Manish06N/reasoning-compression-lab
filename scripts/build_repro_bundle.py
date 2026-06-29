#!/usr/bin/env python3
"""Build a reproducibility bundle JSON for a publication archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

HASH_PATTERNS = [
    "configs/**/*.json",
    "configs/**/*.yaml",
    "prompts/**/*.txt",
    "scripts/run_inference.py",
    "scripts/score_run.py",
    "scripts/build_paper_tables.py",
    "scripts/build_repro_bundle.py",
    "src/**/*.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def package_versions() -> dict[str, str | None]:
    versions = {"python": sys.version.split()[0]}
    for name in ["torch", "vllm", "transformers", "datasets", "numpy", "scipy", "sklearn"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def file_hashes() -> dict[str, str]:
    hashes = {}
    for pattern in HASH_PATTERNS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and ".git" not in path.parts:
                hashes[str(path.relative_to(ROOT))] = sha256(path)
    return hashes


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Output archive, e.g. outputs-hpc-...")
    parser.add_argument("--output", default=None, help="Default: <archive>/reproducibility_bundle.json")
    args = parser.parse_args()

    archive = Path(args.archive)
    if not archive.is_absolute():
        archive = ROOT / archive
    output = Path(args.output) if args.output else archive / "reproducibility_bundle.json"

    manifest = load_json(archive / "manifest.json")
    metadata = []
    for path in sorted((archive / "metadata").glob("*.json")):
        data = load_json(path)
        if data:
            metadata.append(data)

    bundle = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive),
        "platform": {
            "hostname": platform.node(),
            "system": platform.platform(),
            "machine": platform.machine(),
        },
        "git": {
            "commit": cmd(["git", "rev-parse", "HEAD"]),
            "branch": cmd(["git", "branch", "--show-current"]),
            "status_short": cmd(["git", "status", "--short"]),
        },
        "cuda": {
            "nvidia_smi": cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        },
        "packages": package_versions(),
        "archive_manifest": manifest,
        "cell_metadata": metadata,
        "file_hashes_sha256": file_hashes(),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Wrote reproducibility bundle: {output}")


if __name__ == "__main__":
    main()
