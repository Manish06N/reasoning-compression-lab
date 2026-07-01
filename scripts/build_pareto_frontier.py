#!/usr/bin/env python3
"""Build Pareto cost-per-correct frontier CSV + optional plot from summary JSONs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.pareto_frontier import (
    build_frontier_points,
    filter_by_model_family,
    frontier_table_rows,
    pareto_frontier,
)


def _load_summaries(paths: list[Path]) -> list[dict]:
    summaries = []
    for path in paths:
        if path.is_dir():
            for f in sorted(path.glob("*_summary.json")):
                summaries.append(json.loads(f.read_text(encoding="utf-8")))
        else:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto frontier from results/*_summary.json")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Summary JSON files or results/ directory",
    )
    parser.add_argument(
        "--family",
        default=None,
        help="Filter cell_ids by substring (e.g. qwen7b, llama8b)",
    )
    parser.add_argument(
        "--output-csv",
        default="results/pareto_frontier.csv",
    )
    parser.add_argument(
        "--output-json",
        default="results/pareto_frontier.json",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional PNG path (requires matplotlib)",
    )
    args = parser.parse_args()

    paths = [ROOT / p if not Path(p).is_absolute() else Path(p) for p in args.inputs]
    summaries = _load_summaries(paths)
    if args.family:
        summaries = filter_by_model_family(summaries, args.family)

    points = build_frontier_points(summaries)
    if not points:
        print("No summaries with finite cost_per_correct_seconds — run score_run first.")
        sys.exit(1)

    frontier = pareto_frontier(points)
    rows = frontier_table_rows(points)

    csv_path = ROOT / args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = ROOT / args.output_json
    payload = {
        "n_points": len(points),
        "n_frontier": len(frontier),
        "frontier_cell_ids": [p.cell_id for p in frontier],
        "points": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skip --plot")
            return
        plot_path = ROOT / args.plot
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        for p in points:
            color = "C1" if p.on_frontier else "C0"
            marker = "*" if p.on_frontier else "o"
            ax.scatter(
                p.cost_per_correct_seconds,
                p.pass_at_1 * 100,
                c=color,
                marker=marker,
                s=80 if p.on_frontier else 40,
                label=p.quant_config if p.on_frontier else None,
            )
            ax.annotate(p.quant_config, (p.cost_per_correct_seconds, p.pass_at_1 * 100), fontsize=8)
        ax.set_xlabel("Cost per correct (GPU-seconds)")
        ax.set_ylabel("pass@1 (%)")
        ax.set_title("Compression Pareto frontier")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
