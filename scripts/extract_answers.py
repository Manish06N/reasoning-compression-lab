#!/usr/bin/env python3
"""Extract parsed answers from raw JSONL into runs/extracted/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extraction.math_extractor import normalize_completion_text
from src.metrics.scoring import score_item


def extract_row(row: dict) -> dict:
    row = dict(row)
    row["completion"] = normalize_completion_text(row.get("completion", ""))
    score = score_item(row)
    out = {**row, **score}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract answers from raw JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    in_path = ROOT / args.input
    out_path = ROOT / (args.output or f"runs/extracted/{in_path.stem}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(extract_row(json.loads(line)))

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Extracted {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
