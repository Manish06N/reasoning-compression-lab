#!/usr/bin/env python3
"""Score multi-sample runs: maj@k, per-sample pass@1, calibration-ready extracted rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.consistency import calculate_consistency
from src.metrics.scoring import maj_at_k, majority_vote_answer, score_item, summarize_scored_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score maj@k multi-sample JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-scored", default=None)
    parser.add_argument("--output-summary", default=None)
    parser.add_argument("--k", type=int, default=None, help="Majority vote k (default: infer from n_samples).")
    parser.add_argument("--consistency-method", default="agree_percent")
    args = parser.parse_args()

    in_path = ROOT / args.input
    stem = in_path.stem
    scored_path = ROOT / (args.output_scored or f"runs/scored/{stem}.jsonl")
    summary_path = ROOT / (args.output_summary or f"results/{stem}_summary.json")
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_rows.append(json.loads(line))

    scored_rows = []
    for row in raw_rows:
        scored_rows.append({**row, **score_item(row)})

    with scored_path.open("w", encoding="utf-8") as f:
        for row in scored_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    groups: dict[str, list] = defaultdict(list)
    for row in scored_rows:
        groups[str(row["id"])].append(row)

    n_samples = args.k or (scored_rows[0].get("n_samples") if scored_rows else 5)
    maj_correct = 0
    any_correct = 0
    consistencies = []
    item_rows = []

    for item_id, samples in groups.items():
        samples = sorted(samples, key=lambda r: r.get("sample_index", 0))
        flags = [bool(s.get("correct")) for s in samples]
        answers = [str(s.get("pred_answer") or "") for s in samples if s.get("pred_answer")]
        majority = majority_vote_answer(answers)
        maj_ok = False
        if majority:
            base = dict(samples[0])
            task = str(base.get("task", "")).lower()
            if task.startswith("gpqa"):
                base["completion"] = f"Answer: {majority}"
            else:
                base["completion"] = f"\\boxed{{{majority}}}"
            maj_ok = score_item(base)["correct"]
        else:
            maj_ok = maj_at_k(flags)

        maj_correct += int(maj_ok)
        any_correct += int(any(flags))
        if answers:
            consistencies.append(calculate_consistency(answers, method=args.consistency_method))
        item_rows.append({"id": item_id, "maj_correct": maj_ok, "n_samples": len(samples)})

    n_items = len(groups)
    summary = {
        "input": str(in_path.relative_to(ROOT)) if str(in_path).startswith(str(ROOT)) else str(in_path),
        "n_items": n_items,
        "n_samples_per_item": n_samples,
        "n_rows": len(scored_rows),
        f"maj_at_{n_samples}": maj_correct / n_items if n_items else 0.0,
        "pass_at_1_any_sample": any_correct / n_items if n_items else 0.0,
        "pass_at_1_per_sample_mean": sum(r["correct"] for r in scored_rows) / len(scored_rows)
        if scored_rows
        else 0.0,
        "consistency_agree_percent_mean": sum(consistencies) / len(consistencies) if consistencies else None,
        "consistency_method": args.consistency_method,
    }
    if scored_rows:
        summary["cell_id"] = scored_rows[0].get("cell_id")
        summary["task"] = scored_rows[0].get("task")
        summary["seed"] = scored_rows[0].get("seed")

    per_sample_summary = summarize_scored_rows(scored_rows)
    summary["per_sample_metrics"] = {
        k: v for k, v in per_sample_summary.items() if k not in summary
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Scored: {scored_path}")
    print(f"Summary: {summary_path}")
    print(f"\nNext: python scripts/compute_calibration.py --input {scored_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
