#!/usr/bin/env python3
"""Score a raw JSONL run and write summary metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.correctness.scoring import score_item, summarize_scored_rows
from src.evaluation.calibration.metrics import calibration_summary_from_rows
from src.evaluation.selective_risk.curves import selective_risk_from_rows
from src.evaluation.statistics.bootstrap import cluster_bootstrap_ci


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a raw inference JSONL file.")
    parser.add_argument("--input", required=True, help="Input raw JSONL under runs/raw/")
    parser.add_argument(
        "--output",
        default=None,
        help="Scored JSONL output. Default: runs/scored/<input_stem>.jsonl",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Summary JSON output. Default: results/<input_stem>_summary.json",
    )
    parser.add_argument(
        "--parquet",
        default=None,
        help="Optional Parquet export path for scored rows",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = ROOT / in_path
    stem = in_path.stem
    out_path = ROOT / (args.output or f"runs/scored/{stem}.jsonl")
    summary_path = ROOT / (args.summary or f"results/{stem}_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    scored = []
    for row in rows:
        score = score_item(row)
        scored.append({**row, **score})

    with out_path.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_scored_rows(scored)
    summary["schema_version"] = "summary.v1"
    summary["input"] = _display_path(in_path)
    summary["scored_output"] = _display_path(out_path)
    if scored:
        summary["cell_id"] = scored[0].get("cell_id")
        summary["quant_config"] = scored[0].get("quant_config")
        summary["seed"] = scored[0].get("seed")
        summary["task"] = scored[0].get("task")
        if all(row.get("id") is not None for row in scored):
            cluster_ci = cluster_bootstrap_ci(scored, cluster_key="id", value_key="correct")
            summary["pass_at_1_cluster_ci95_low"] = cluster_ci["ci_low"]
            summary["pass_at_1_cluster_ci95_high"] = cluster_ci["ci_high"]
            summary["pass_at_1_n_clusters"] = cluster_ci["n_clusters"]
        cal = calibration_summary_from_rows(scored)
        if cal:
            summary["calibration"] = cal
        risk = selective_risk_from_rows(scored)
        if risk:
            summary["selective_risk"] = risk

    if args.parquet:
        pq = Path(args.parquet)
        if not pq.is_absolute():
            pq = ROOT / pq
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_parquet.py"),
                "--input",
                str(out_path),
                "--output",
                str(pq),
            ],
            check=True,
        )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Scored rows: {out_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
