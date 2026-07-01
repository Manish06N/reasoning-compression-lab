#!/usr/bin/env python3
"""Validate wired J1 cells against papers/j1/publication_matrix.yaml."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required (pip install pyyaml)", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    matrix_path = ROOT / "papers/j1/publication_matrix.yaml"
    cells_dir = ROOT / "configs/cells"
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))

    expected = set(matrix["minimum_publishable"]["expected_cells"])
    wired = {p.stem for p in cells_dir.glob("*.json") if not p.name.startswith("smoke_")}

    missing = sorted(expected - wired)
    extra = sorted(wired - expected - _repro_cells(matrix))

    print("J1 publication matrix validation")
    print(f"  Matrix file: {matrix_path.relative_to(ROOT)}")
    print(f"  Minimum publishable expected: {len(expected)}")
    print(f"  Wired cell configs (non-smoke): {len(wired)}")
    print(f"  Missing: {len(missing)}")
    print(f"  Extra (non-repro, not in min grid): {len(extra)}")

    if missing:
        print("\nMissing cells:")
        for cell_id in missing:
            print(f"  - {cell_id}")
    if extra:
        print("\nExtra cells (informational):")
        for cell_id in extra[:20]:
            print(f"  - {cell_id}")
        if len(extra) > 20:
            print(f"  ... and {len(extra) - 20} more")

    full = matrix.get("full_level_c_aspirational", {})
    theoretical = full.get("theoretical_cell_count")
    print(f"\nFull Level C (aspirational): {theoretical} cells — NOT required before minimum grid runs.")

    if missing:
        sys.exit(1)
    print("\nOK: minimum publishable cells are wired.")
    sys.exit(0)


def _repro_cells(matrix: dict) -> set[str]:
    repro = matrix.get("reproduction_gate", {}).get("cells", [])
    return set(repro)


if __name__ == "__main__":
    main()
