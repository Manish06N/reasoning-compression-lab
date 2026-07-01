#!/usr/bin/env python3
"""Print expected dataset row count for a cell config (used by shell orchestrators)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runners.task_utils import expected_rows_for_cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(expected_rows_for_cell(args.cell_config, limit=args.limit))


if __name__ == "__main__":
    main()
