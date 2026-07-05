#!/usr/bin/env python3
"""Compare degeneration / pass@1 on shared MATH-500 problem IDs across archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def load_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["id"]] = row
    return out


def loop_heuristic(text: str) -> str:
    tail = (text or "")[-400:]
    if "yeah, yeah" in tail or "yeah yeah" in tail:
        return "yeah-loop"
    if "0 Wait" in tail:
        return "wait-loop"
    if "Which which" in tail:
        return "which-loop"
    if "the the" in tail:
        return "the-loop"
    if len(tail) > 200 and len(set(tail.split())) < 12:
        return "low-entropy"
    return "none"


def summarize(
    label: str,
    rows: Dict[str, Dict[str, Any]],
    ids: List[str],
    scored: Dict[str, Dict[str, Any]],
) -> None:
    print(f"\n=== {label} ===")
    if not rows:
        print("  (no raw rows)")
        return
    trunc = 0
    boxed = 0
    correct = 0
    loops: Dict[str, int] = {}
    present = 0
    for pid in ids:
        row = rows.get(pid)
        if not row:
            continue
        present += 1
        is_trunc = bool(row.get("truncated") or row.get("finish_reason") == "length")
        trunc += int(is_trunc)
        completion = row.get("completion") or row.get("model_output") or ""
        if "\\boxed" in completion:
            boxed += 1
        sc = scored.get(pid, {})
        if sc.get("correct"):
            correct += 1
        loop = loop_heuristic(completion)
        loops[loop] = loops.get(loop, 0) + 1
        print(
            f"  {pid}: trunc={is_trunc} tok={row.get('completion_tokens')} "
            f"correct={sc.get('correct', '?')} loop={loop}"
        )
    if present:
        print(
            f"  aggregate: n={present} pass@1={correct}/{present} ({100*correct/present:.1f}%) "
            f"trunc={trunc}/{present} ({100*trunc/present:.1f}%) boxed={boxed}/{present} loops={loops}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side trace comparison for MATH-500 IDs.")
    parser.add_argument(
        "--baseline-archive",
        default="outputs-hpc-diag-pathc-2026-07-05",
    )
    parser.add_argument("--parity-archive", default=None)
    parser.add_argument("--baseline-cell", default="diag_qwen7b_bf16_math500_seed42_n50")
    parser.add_argument("--parity-cell", default="diag_qwen7b_bf16_math500_seed42_n10_parity")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    from src.runners.config_utils import load_cell_config
    from src.runners.dataset_rows import prepare_example_row
    from src.runners.inference_session import load_task_dataset

    baseline_cell_path = ROOT / "configs" / "cells" / f"{args.baseline_cell}.json"
    cell = load_cell_config(str(baseline_cell_path.relative_to(ROOT)))
    dataset = load_task_dataset(cell, args.limit)
    ids = []
    for i in range(len(dataset)):
        _, row_base = prepare_example_row(dataset[i], cell["task"], cell, i)
        ids.append(row_base["id"])

    baseline_root = ROOT / args.baseline_archive
    baseline_raw = load_rows(baseline_root / "raw" / f"{args.baseline_cell}.jsonl")
    baseline_scored = load_rows(ROOT / "runs" / "scored" / f"{args.baseline_cell}.jsonl")
    summarize("Baseline (pre-parity Path C)", baseline_raw, ids, baseline_scored)

    if args.parity_archive:
        parity_root = ROOT / args.parity_archive
        parity_raw = load_rows(parity_root / "raw" / f"{args.parity_cell}.jsonl")
        parity_scored = load_rows(ROOT / "runs" / "scored" / f"{args.parity_cell}.jsonl")
        summarize("Parity pilot (post-fix)", parity_raw, ids, parity_scored)

    print("\nOfficial QRM repo cross-check:")
    print("  bash scripts/hpc/qrm_parity/setup_official_qrm_repo.sh")
    print("  # then on a GPU node with QRM env:")
    print("  python inference.py --model $MODEL --dataset MATH-500 --max_samples 10 --seed 42")


if __name__ == "__main__":
    main()