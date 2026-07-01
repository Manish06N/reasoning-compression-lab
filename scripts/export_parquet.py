#!/usr/bin/env python3
"""Export scored rows or summaries to Parquet (V8.2 §11.1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL or summary JSON")
    parser.add_argument("--output", required=True, help=".parquet output path")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas + pyarrow required: pip install pandas pyarrow") from exc

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = ROOT / in_path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if in_path.suffix == ".json":
        data = json.loads(in_path.read_text(encoding="utf-8"))
        df = pd.json_normalize(data) if isinstance(data, dict) else pd.DataFrame([data])
    else:
        rows = []
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)

    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows → {out_path}")


if __name__ == "__main__":
    main()
