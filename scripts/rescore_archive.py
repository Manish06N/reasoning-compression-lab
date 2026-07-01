#!/usr/bin/env python3
"""Rescore all raw JSONL files in an output archive."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore every raw/*.jsonl file in an archive.")
    parser.add_argument("archive", help="Archive root containing raw/, scored/, and results/.")
    args = parser.parse_args()

    archive = Path(args.archive)
    if not archive.is_absolute():
        archive = ROOT / archive
    raw_dir = archive / "raw"
    if not raw_dir.is_dir():
        raise SystemExit(f"Missing raw directory: {raw_dir}")

    for raw in sorted(raw_dir.glob("*.jsonl")):
        scored = archive / "scored" / raw.name
        summary = archive / "results" / f"{raw.stem}_summary.json"
        scored.parent.mkdir(parents=True, exist_ok=True)
        summary.parent.mkdir(parents=True, exist_ok=True)
        rel_raw = raw.relative_to(ROOT)
        rel_scored = scored.relative_to(ROOT)
        rel_summary = summary.relative_to(ROOT)
        print(f"rescoring {rel_raw}")
        subprocess.run([sys.executable, "scripts/score_run.py", "--input", str(rel_raw), "--output", str(rel_scored), "--summary", str(rel_summary)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
