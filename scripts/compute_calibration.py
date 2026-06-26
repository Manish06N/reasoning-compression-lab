#!/usr/bin/env python3
"""Compute calibration metrics from multi-sample extracted JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.calibration import compute_calibration_metrics
from src.metrics.consistency import calculate_consistency
from src.metrics.scoring import score_math_item


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute calibration from multi-sample runs.")
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL grouped by item id with multiple samples in 'samples' list",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--method", default="agree_percent")
    args = parser.parse_args()

    in_path = ROOT / args.input
    out_path = ROOT / (args.output or f"results/{in_path.stem}_calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list] = defaultdict(list)
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            groups[row["id"]].append(row)

    confidences = []
    labels = []
    for item_id, samples in groups.items():
        answers = [s.get("pred_answer") for s in samples if s.get("pred_answer")]
        confidences.append(calculate_consistency([str(a) for a in answers], method=args.method))
        gold = samples[0].get("gold_solution", "")
        majority_completion = samples[0].get("completion", "")
        labels.append(1 if score_math_item(gold, majority_completion)["correct"] else 0)

    metrics = compute_calibration_metrics(confidences, labels)
    metrics["n_items"] = len(labels)
    metrics["consistency_method"] = args.method

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
