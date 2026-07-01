#!/usr/bin/env python3
"""Compare lm_eval results JSON to our harness summary pass@1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _find_lmeval_results(lmeval_dir: Path) -> dict | None:
    for path in sorted(lmeval_dir.rglob("results*.json"), reverse=True):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "results" in data:
            return data
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare lm_eval vs harness summary.")
    parser.add_argument("--lmeval-dir", required=True)
    parser.add_argument("--summary", required=True, help="Our results/*_summary.json")
    parser.add_argument("--task", default="gsm8k")
    args = parser.parse_args()

    lmeval_dir = ROOT / args.lmeval_dir
    summary = json.loads((ROOT / args.summary).read_text(encoding="utf-8"))
    lmeval = _find_lmeval_results(lmeval_dir)
    if not lmeval:
        print(f"No lm_eval results under {lmeval_dir}")
        sys.exit(1)

    task_key = args.task
    lmeval_acc = None
    results = lmeval.get("results", {})
    for key, val in results.items():
        if task_key in key.lower():
            lmeval_acc = val.get("acc,none") or val.get("exact_match,strict-match")
            break

    ours = float(summary.get("pass_at_1", 0.0))
    report = {
        "harness_pass_at_1": ours,
        "lmeval_accuracy": lmeval_acc,
        "delta": (ours - float(lmeval_acc)) if lmeval_acc is not None else None,
        "cell_id": summary.get("cell_id"),
        "note": "Large delta may indicate prompt/extraction protocol differences",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
