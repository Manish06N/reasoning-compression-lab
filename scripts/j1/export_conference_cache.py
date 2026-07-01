#!/usr/bin/env python3
"""Export C1 response cache subset from scored archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.runners.checkpoint_utils import load_jsonl
from src.schemas.provenance import git_commit_short


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Scored JSONL")
    parser.add_argument("--output", required=True, help="Cache JSONL output")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = ROOT / in_path
    rows = load_jsonl(in_path)
    if args.limit:
        rows = rows[: args.limit]

    cache = []
    for row in rows:
        cache.append(
            {
                "id": row.get("id"),
                "task": row.get("task"),
                "prompt": row.get("prompt"),
                "completion": row.get("completion"),
                "pred_answer": row.get("pred_answer"),
                "gold_answer": row.get("gold_answer") or row.get("gold_letter"),
                "correct": row.get("correct"),
                "cell_id": row.get("cell_id"),
                "quant_config": row.get("quant_config"),
                "seed": row.get("seed"),
            }
        )

    manifest = {
        "paper": "c1",
        "source": str(in_path),
        "git_commit": git_commit_short(),
        "n_rows": len(cache),
    }
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, indent=2) + "\n")
        for row in cache:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cache)} cache rows → {out_path}")


if __name__ == "__main__":
    main()
