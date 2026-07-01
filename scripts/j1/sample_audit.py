#!/usr/bin/env python3
"""Sample rows for manual extraction audit (J1 gate)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from papers.j1.audit.sampler import sample_audit_rows
from src.runners.checkpoint_utils import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Scored JSONL")
    parser.add_argument("--output", required=True, help="Audit sample JSONL")
    parser.add_argument("-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = ROOT / in_path
    rows = load_jsonl(in_path)
    sample = sample_audit_rows(rows, n=args.n, seed=args.seed)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in sample:
            audit = {
                "id": row.get("id"),
                "task": row.get("task"),
                "completion": row.get("completion"),
                "pred_answer": row.get("pred_answer"),
                "gold_answer": row.get("gold_answer") or row.get("gold_letter"),
                "correct": row.get("correct"),
                "answer_parse_success": row.get("answer_parse_success"),
                "reviewer_decision": None,
                "adjudication_notes": None,
            }
            f.write(json.dumps(audit, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sample)} audit rows → {out_path}")


if __name__ == "__main__":
    main()
