#!/usr/bin/env python3
"""How each campaign completion opens, from the HPC campaign traces. CPU only.

Writes results/reports/think_prefix_audit.json. Run from the repo root:
    python scripts/hpc/qrm_parity/think_prefix_audit.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
OUT = RESULTS / "reports" / "think_prefix_audit.json"
BENCHES = {"math500": "math500", "gsm8k": "gsm8k", "gpqa": "gpqadiamond"}
PREFIX = "<think>\n"


def cell_name(path: Path) -> str:
    stem = path.name.split("_")[0].replace("DeepSeek-R1-Distill-", "")
    family = "Qwen-7B" if stem.startswith("Qwen-7B") else "Llama-8B"
    fmt = stem[len(family) + 1:] or "BF16"
    return f"{family}_{fmt}"


def main() -> int:
    report: dict = {"prefix": PREFIX, "cells": {}}
    missing = []
    for bench, tag in BENCHES.items():
        agg: dict = defaultdict(lambda: {"rows": 0, "exact": 0, "any": 0,
                                         "exact_correct": 0, "other_rows": 0, "other_correct": 0})
        for compact_path in sorted((RESULTS / bench).glob(f"*_{tag}_*.json")):
            compact = json.loads(compact_path.read_text())
            trace_path = Path(compact["result"])
            if not trace_path.exists():
                missing.append(str(trace_path))
                continue
            rows = sorted(compact["details"], key=lambda r: r["row"])
            # Campaign ".jsonl" files are one pretty-printed JSON array; accept JSON Lines too.
            raw = trace_path.read_text(encoding="utf-8")
            if raw.lstrip().startswith("["):
                traces = json.loads(raw)
            else:
                traces = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if len(traces) != len(rows):
                print(f"ERROR: {trace_path} has {len(traces)} rows, compact has {len(rows)}", file=sys.stderr)
                return 1
            for k, (row, trace) in enumerate(zip(rows, traces)):
                if float(trace["metrics"]["extractive_match"]) != float(row["extractive_match"]):
                    print(f"ERROR: row {k} of {trace_path} disagrees with {compact_path.name}", file=sys.stderr)
                    return 1
                text = trace["generated_text"]
                cell = agg[cell_name(compact_path)]
                cell["rows"] += 1
                exact = text.startswith(PREFIX)
                cell["exact"] += exact
                cell["any"] += text.lstrip().startswith("<think>")
                if exact:
                    cell["exact_correct"] += row["extractive_match"] == 1.0
                else:
                    cell["other_rows"] += 1
                    cell["other_correct"] += row["extractive_match"] == 1.0
        for name, c in sorted(agg.items()):
            report["cells"][f"{bench}/{name}"] = {
                **c,
                "opens_think_exact_pct": 100 * c["exact"] / c["rows"],
                "opens_think_any_pct": 100 * c["any"] / c["rows"],
                "pass1_exact_rows_pct": 100 * c["exact_correct"] / c["exact"] if c["exact"] else None,
                "pass1_other_rows_pct": 100 * c["other_correct"] / c["other_rows"] if c["other_rows"] else None,
            }
    if missing:
        print(f"ERROR: {len(missing)} campaign trace files not found, e.g. {missing[0]}", file=sys.stderr)
        return 1
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for key, c in report["cells"].items():
        if "AWQ" in key:
            print(f"  {key}: opens with <think>\\n in {c['exact']}/{c['rows']} rows "
                  f"({c['opens_think_exact_pct']:.2f}%); other rows {c['other_rows']}, "
                  f"pass@1 on other rows {c['pass1_other_rows_pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
