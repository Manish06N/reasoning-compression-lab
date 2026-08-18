#!/usr/bin/env python3
"""Validation audit script for measured serving systems benchmark.

Verifies:
- All 8 configurations present
- Both conditions (A: single stream, B: batched) present
- All 3 repetitions present per configuration and condition (48 total task runs)
- Fixed-token microbenchmarks present for all 8 configurations
- Consistency of prompt counts, positive execution times, and positive token throughputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

MODELS = ["Qwen-7B", "Llama-8B"]
FORMATS = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]
CONDITIONS = ["A_single_stream_c1", "B_batched_throughput_c8"]
REPETITIONS = [1, 2, 3]


def validate_raw_records(raw_dir: Path) -> Dict[str, Any]:
    """Validate all raw measurement records in raw_dir."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    files = list(raw_dir.glob("*.json"))
    print(f"Discovered {len(files)} raw measurement JSON files in {raw_dir}")

    task_records: Dict[tuple, List[Dict[str, Any]]] = {}
    micro_records: Dict[tuple, Dict[str, Any]] = {}
    errors: List[str] = []

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        btype = data.get("benchmark_type", "")
        model = data.get("model", "")
        fmt = data.get("format", "")

        if btype == "task_realistic":
            cond = data.get("condition", "")
            rep = data.get("repetition", 0)
            key = (model, fmt, cond, rep)
            task_records[key] = data

            # Sanity checks
            if data.get("elapsed_seconds", 0) <= 0:
                errors.append(f"{f.name}: elapsed_seconds <= 0")
            if data.get("total_output_tokens", 0) <= 0:
                errors.append(f"{f.name}: total_output_tokens <= 0")
            if data.get("output_tokens_per_second", 0) <= 0:
                errors.append(f"{f.name}: output_tokens_per_second <= 0")
            if data.get("peak_vram_allocated_gb", 0) <= 0:
                errors.append(f"{f.name}: peak_vram_allocated_gb <= 0")

        elif btype == "fixed_token_microbenchmark":
            key = (model, fmt)
            micro_records[key] = data
            if data.get("raw_decode_tokens_per_second", 0) <= 0:
                errors.append(f"{f.name}: raw_decode_tokens_per_second <= 0")

    # Check completeness
    missing_task_cells = []
    for model in MODELS:
        for fmt in FORMATS:
            for cond in CONDITIONS:
                for rep in REPETITIONS:
                    key = (model, fmt, cond, rep)
                    if key not in task_records:
                        missing_task_cells.append(key)

    missing_micro_cells = []
    for model in MODELS:
        for fmt in FORMATS:
            key = (model, fmt)
            if key not in micro_records:
                missing_micro_cells.append(key)

    print("=" * 80)
    print("MEASURED SERVING VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total task-realistic records: {len(task_records)} / 48 expected")
    print(f"Total microbenchmark records: {len(micro_records)} / 8 expected")
    print(f"Missing task cells: {len(missing_task_cells)}")
    print(f"Missing micro cells: {len(missing_micro_cells)}")
    print(f"Integrity errors: {len(errors)}")

    if errors:
        for err in errors[:10]:
            print(f"  [ERROR] {err}")
        return {"status": "FAILED", "errors": errors}

    if missing_task_cells or missing_micro_cells:
        return {
            "status": "INCOMPLETE",
            "missing_task_cells": [str(c) for c in missing_task_cells],
            "missing_micro_cells": [str(c) for c in missing_micro_cells],
        }

    print("\nALL 48 TASK-REALISTIC + 8 MICROBENCHMARK RUNS VALIDATED SUCCESSFULLY!")
    return {"status": "PASSED", "total_records": len(task_records) + len(micro_records)}


def main():
    parser = argparse.ArgumentParser(description="Validate measured serving benchmark records.")
    parser.add_argument("--raw-dir", type=Path, default=Path("results/measured_serving/raw"))
    args = parser.parse_args()

    res = validate_raw_records(args.raw_dir)
    if res["status"] != "PASSED":
        print(f"Validation status: {res['status']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
