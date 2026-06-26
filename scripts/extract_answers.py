#!/usr/bin/env python3
"""Extract parsed answers from raw JSONL into runs/extracted/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extraction.gpqa_extractor import extract_gpqa_letter
from src.extraction.math_extractor import extract_boxed, extract_gold_answer


def extract_row(row: dict) -> dict:
    task = row.get("task", "math500")
    completion = row.get("completion", "")
    out = dict(row)
    if task.startswith("gpqa"):
        out["pred_answer"] = extract_gpqa_letter(completion)
        out["gold_answer"] = row.get("gold_letter")
    else:
        out["pred_answer"] = extract_boxed(completion)
        out["gold_answer"] = extract_gold_answer(row.get("gold_solution", ""))
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
