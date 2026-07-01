#!/usr/bin/env python3
"""Rescore all raw JSONL cells in an output archive and rebuild summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extraction.math_extractor import normalize_completion_text
from src.metrics.scoring import score_item, summarize_scored_rows


def rescore_cell(raw_path: Path, archive: Path) -> dict:
    scored_path = archive / "scored" / raw_path.name
    summary_path = archive / "results" / f"{raw_path.stem}_summary.json"
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    scored = []
    for row in rows:
        row = dict(row)
        row["completion"] = normalize_completion_text(row.get("completion", ""))
        scored.append({**row, **score_item(row)})

    with scored_path.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_scored_rows(scored)
    summary["input"] = str(raw_path.relative_to(ROOT)) if raw_path.is_relative_to(ROOT) else str(raw_path)
    summary["scored_output"] = (
        str(scored_path.relative_to(ROOT)) if scored_path.is_relative_to(ROOT) else str(scored_path)
    )
    if scored:
        summary["cell_id"] = scored[0].get("cell_id")
        summary["quant_config"] = scored[0].get("quant_config")
        summary["seed"] = scored[0].get("seed")
        summary["task"] = scored[0].get("task")

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Output archive directory")
    args = parser.parse_args()

    archive = Path(args.archive)
    if not archive.is_absolute():
        archive = ROOT / archive
    raw_dir = archive / "raw"
    if not raw_dir.exists():
        raise SystemExit(f"No raw/ directory in {archive}")

    summaries = []
    for raw_path in sorted(raw_dir.glob("*.jsonl")):
        summary = rescore_cell(raw_path, archive)
        summaries.append(summary)
        print(
            f"{summary.get('cell_id', raw_path.stem)}: "
            f"pass@1={summary.get('pass_at_1', 0):.3f} "
            f"parse_fail={summary.get('parse_failure_rate', 0):.3f} "
            f"n={summary.get('n', 0)}"
        )

    print(f"Rescored {len(summaries)} cell(s) under {archive}")


if __name__ == "__main__":
    main()
