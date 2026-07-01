#!/usr/bin/env python3
"""Paired config comparison for J1 (McNemar + Holm + cluster bootstrap)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.statistics.bootstrap import cluster_bootstrap_ci
from src.evaluation.statistics.holm import holm_adjusted_pvalues
from src.evaluation.statistics.mcnemar import mcnemar_test
from src.runners.checkpoint_utils import load_jsonl


def _correct_by_id(rows: list[dict]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for row in rows:
        rid = str(row.get("id", ""))
        if not rid:
            continue
        out[rid] = bool(row.get("correct"))
    return out


def compare_scored(baseline_path: Path, variant_path: Path) -> dict:
    base_rows = load_jsonl(baseline_path)
    var_rows = load_jsonl(variant_path)
    base = _correct_by_id(base_rows)
    var = _correct_by_id(var_rows)
    common = sorted(set(base) & set(var))
    if not common:
        raise ValueError("No overlapping item ids between baseline and variant")

    base_flags = [base[i] for i in common]
    var_flags = [var[i] for i in common]
    mcn = mcnemar_test(base_flags, var_flags)

    merged_rows = [{"id": i, "correct": var[i]} for i in common]
    cluster_ci = cluster_bootstrap_ci(merged_rows, cluster_key="id", value_key="correct")

    return {
        "baseline": str(baseline_path),
        "variant": str(variant_path),
        "n_items": len(common),
        "baseline_pass_at_1": sum(base_flags) / len(common),
        "variant_pass_at_1": sum(var_flags) / len(common),
        "mcnemar": mcn,
        "variant_pass_at_1_cluster_ci95": cluster_ci,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two scored JSONL files (J1 stats).")
    parser.add_argument("--baseline", required=True, help="Baseline scored JSONL")
    parser.add_argument("--variant", required=True, help="Variant scored JSONL")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Additional variant paths; emits Holm-adjusted p-values vs baseline",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    baseline = Path(args.baseline)
    if not baseline.is_absolute():
        baseline = ROOT / baseline

    reports = [compare_scored(baseline, Path(v) if Path(v).is_absolute() else ROOT / v) for v in ([args.variant] + (args.variants or []))]

    if len(reports) > 1:
        pvals = [r["mcnemar"]["p_value"] for r in reports]
        adjusted = holm_adjusted_pvalues(pvals)
        for r, adj in zip(reports, adjusted):
            r["mcnemar"]["p_value_holm"] = adj

    out = {"comparisons": reports}
    text = json.dumps(out, indent=2)
    print(text)
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
