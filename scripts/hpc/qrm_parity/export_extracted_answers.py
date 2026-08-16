#!/usr/bin/env python3
"""Export extracted answers from official-QRM JSONLs on HPC scratch.

The public compact validation JSON has extractive_match but no answer strings,
so a deployable modal-agreement gate cannot be recomputed on a laptop.

Run this on PARAM Rudra if the campaign JSONLs still exist, then rsync the
sidecar files to the MacBook and re-run revision_reanalysis.py.

Example:
  export QR=/scratch/$USER/reasoning-compression-lab
  python3 scripts/hpc/qrm_parity/export_extracted_answers.py \\
      --jsonl-root $QR/outputs-hpc-campaign-2026-08-14/inference \\
      --out $QR/results/extracted_answers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def prediction_from_row(row: dict[str, Any]) -> str | None:
    metrics = row.get("metrics") or {}
    for key in (
        "extracted_answer",
        "pred_answer",
        "predict",
        "prediction",
        "extractive_match_pred",
    ):
        value = row.get(key)
        if value in (None, ""):
            value = metrics.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n_files = n_rows = n_pred = 0
    for path in sorted(args.jsonl_root.rglob("*.jsonl")):
        n_files += 1
        rows = []
        with path.open() as fp:
            for line in fp:
                if not line.strip():
                    continue
                row = json.loads(line)
                pred = prediction_from_row(row)
                n_rows += 1
                n_pred += int(pred is not None)
                rows.append(
                    {
                        "row": len(rows) + 1,
                        "extracted_answer": pred,
                        "extractive_match": (row.get("metrics") or {}).get("extractive_match"),
                    }
                )
        sidecar = args.out / (path.parent.name + "_" + path.stem + "_answers.json")
        sidecar.write_text(json.dumps({"source": str(path), "rows": rows}, indent=2) + "\n")
        print(f"wrote {sidecar} ({len(rows)} rows)")
    print(f"files={n_files} rows={n_rows} with_pred={n_pred}")
    if n_pred == 0:
        print("No answer strings found. The JSONL may only store generated_text; parse \\\\boxed{} next.")


if __name__ == "__main__":
    main()
