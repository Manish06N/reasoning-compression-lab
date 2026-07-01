#!/usr/bin/env python3
"""Aggregate multi-seed summaries (QRM-style mean ± std)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.statistics.bootstrap import cluster_bootstrap_ci
from src.stats.seed_variance import aggregate_by_seed, format_mean_std


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate seed summaries for one cell family.")
    parser.add_argument("--summaries", nargs="+", required=True, help="Summary JSON paths")
    parser.add_argument("--metric", default="pass_at_1")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    summaries = []
    for p in args.summaries:
        path = Path(p)
        if not path.is_absolute():
            path = ROOT / path
        summaries.append(json.loads(path.read_text(encoding="utf-8")))

    stats = aggregate_by_seed(summaries, metric_key=args.metric)
    report = {
        "metric": args.metric,
        "n_seeds": stats["n"],
        "mean": stats["mean"],
        "std": stats["std"],
        "formatted": format_mean_std([s[args.metric] for s in summaries if args.metric in s]),
        "summaries": [s.get("cell_id") for s in summaries],
    }
    if args.metric == "pass_at_1":
        report["note"] = "For item-level CI use cluster_bootstrap on merged scored JSONL"

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
