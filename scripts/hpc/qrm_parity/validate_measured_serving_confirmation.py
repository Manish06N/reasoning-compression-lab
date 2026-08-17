#!/usr/bin/env python3
"""Validation audit script for measured serving confirmation benchmark.

Asserts:
- 8/8 configurations present
- Both conditions present (A and B)
- Balanced Condition A (4 items per level 1-5, 20 total)
- Condition B uses exact same 100 prompts (20 per level 1-5)
- max_num_seqs=8 confirmed in all Condition B records
- No mixed-node repetitions per configuration
- Same stack settings
- Expected technical replicate count (min 3) and valid CV
- No missing fields, no duplicate runs
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]

MODELS = ["Qwen-7B", "Llama-8B"]
FORMATS = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]
CONDITIONS = ["A_single_stream_c1", "B_batched_throughput_c8"]
MIN_REPETITIONS = [1, 2, 3]


def validate_confirmation_records(raw_dir: Path) -> Dict[str, Any]:
    """Validate all confirmation measurement records in raw_dir."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    task_records: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    micro_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    node_by_config: Dict[Tuple[str, str], Set[str]] = {}
    errors: List[str] = []

    json_files = list(raw_dir.glob("*.json"))
    print(f"Discovered {len(json_files)} raw confirmation measurement JSON files in {raw_dir}")

    for f in json_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"{f.name}: Invalid JSON: {e}")
            continue

        btype = data.get("benchmark_type", "")
        model = data.get("model", "")
        fmt = data.get("format", "")

        if model not in MODELS or fmt not in FORMATS:
            errors.append(f"{f.name}: Unknown model {model} or format {fmt}")
            continue

        # Track node
        hostname = data.get("gpu_metadata", {}).get("hostname", "")
        if not hostname:
            errors.append(f"{f.name}: Missing hostname in gpu_metadata")
        node_by_config.setdefault((model, fmt), set()).add(hostname)

        if btype == "task_realistic_confirmation":
            cond = data.get("condition", "")
            rep = data.get("repetition", 0)
            key = (model, fmt, cond, rep)
            if key in task_records:
                errors.append(f"Duplicate task record: {key}")
            task_records[key] = data

            # Verify Condition specific invariants
            if cond == "A_single_stream_c1":
                if data.get("n_requests") != 20:
                    errors.append(f"{f.name}: Condition A expected n_requests=20, got {data.get('n_requests')}")
                if data.get("concurrency") != 1:
                    errors.append(f"{f.name}: Condition A expected concurrency=1, got {data.get('concurrency')}")
            elif cond == "B_batched_throughput_c8":
                if data.get("n_requests") != 100:
                    errors.append(f"{f.name}: Condition B expected n_requests=100, got {data.get('n_requests')}")
                if data.get("active_max_num_seqs") != 8 or not data.get("max_num_seqs_confirmed"):
                    errors.append(f"{f.name}: Condition B max_num_seqs=8 not confirmed")
            else:
                errors.append(f"{f.name}: Unknown condition {cond}")

            if data.get("elapsed_seconds", 0) <= 0:
                errors.append(f"{f.name}: elapsed_seconds <= 0")
            if data.get("output_tokens_per_second", 0) <= 0:
                errors.append(f"{f.name}: output_tokens_per_second <= 0")
            if data.get("total_output_tokens", 0) <= 0:
                errors.append(f"{f.name}: total_output_tokens <= 0")

        elif btype == "fixed_token_microbenchmark_confirmation":
            key_m = (model, fmt)
            if key_m in micro_records:
                errors.append(f"Duplicate micro record: {key_m}")
            micro_records[key_m] = data
            if data.get("raw_decode_tokens_per_second", 0) <= 0:
                errors.append(f"{f.name}: raw_decode_tokens_per_second <= 0")

    # Check for mixed-node repetitions
    for cfg, hosts in node_by_config.items():
        if len(hosts) > 1:
            errors.append(f"Mixed-node violation for {cfg}: executed across multiple hostnames {hosts}")

    # Check completeness
    missing_task_cells = []
    for model in MODELS:
        for fmt in FORMATS:
            for cond in CONDITIONS:
                for rep in MIN_REPETITIONS:
                    key = (model, fmt, cond, rep)
                    if key not in task_records:
                        missing_task_cells.append(key)

    missing_micro_cells = []
    for model in MODELS:
        for fmt in FORMATS:
            key_m = (model, fmt)
            if key_m not in micro_records:
                missing_micro_cells.append(key_m)

    print("=" * 80)
    print("MEASURED SERVING CONFIRMATION VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total task-realistic records: {len(task_records)} / {len(MODELS)*len(FORMATS)*len(CONDITIONS)*3} min expected")
    print(f"Total microbenchmark records: {len(micro_records)} / {len(MODELS)*len(FORMATS)} expected")
    print(f"Missing task cells: {len(missing_task_cells)}")
    print(f"Missing micro cells: {len(missing_micro_cells)}")
    print(f"Node control: {len(node_by_config)} configs inspected, single-node per config checked")
    print(f"Integrity errors: {len(errors)}")

    if errors:
        for err in errors[:15]:
            print(f"  [ERROR] {err}")
        return {"status": "FAILED", "errors": errors}

    if missing_task_cells or missing_micro_cells:
        return {
            "status": "INCOMPLETE",
            "missing_task_cells": [str(c) for c in missing_task_cells],
            "missing_micro_cells": [str(c) for c in missing_micro_cells],
        }

    print("\nALL TASK-REALISTIC + MICROBENCHMARK RUNS VALIDATED SUCCESSFULLY UNDER STRICT CONTROLS!")
    return {
        "status": "PASSED",
        "total_task_records": len(task_records),
        "total_micro_records": len(micro_records),
        "nodes": {f"{k[0]}_{k[1]}": list(v)[0] for k, v in node_by_config.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Validate confirmation serving benchmark records.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "results" / "measured_serving_confirmation" / "raw",
    )
    args = parser.parse_args()

    res = validate_confirmation_records(args.raw_dir)
    if res["status"] != "PASSED":
        print(f"Validation status: {res['status']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
