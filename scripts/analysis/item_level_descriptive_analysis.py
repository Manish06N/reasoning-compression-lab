#!/usr/bin/env python3
"""CPU-only descriptive item-level summaries from frozen per-cell JSON.

No GPU. No new models, seeds, or benchmarks. Does not claim causality.

Reports:
1. BF16-correct → quantized-wrong flips (item×seed and item-mean)
2. Completion length versus extractive correctness
3. GPQA-Diamond item-level means and flip counts (row index only; no item text)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

import revision_reanalysis as rev  # noqa: E402

OUT_JSON = os.path.join(REPO_ROOT, "results", "reports", "item_level_descriptive_report.json")
OUT_MD = os.path.join(REPO_ROOT, "results", "reports", "item_level_descriptive_report.md")

TASKS = (
    ("math500", rev.MATH_DIR, rev.MATH_SEEDS, 500),
    ("gsm8k", rev.GSM_DIR, rev.BREADTH_SEEDS, 1319),
    ("gpqa_diamond", rev.GPQA_DIR, rev.BREADTH_SEEDS, 198),
)


def _match(row: dict[str, Any]) -> bool:
    return row.get("extractive_match", 0.0) == 1.0


def flip_tables(data: dict[str, dict[str, dict[int, dict[str, Any]]]], seeds: list[int], n_items: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model in rev.MODELS:
        for fmt in ["FP8", "AWQ-4", "GPTQ-4"]:
            bf16 = data[model]["BF16"]
            other = data[model][fmt]
            pair_counts = {
                "both_correct": 0,
                "bf16_correct_quant_wrong": 0,
                "bf16_wrong_quant_correct": 0,
                "both_wrong": 0,
            }
            item_any_bf16_ok_quant_wrong = 0
            item_all_seeds_bf16_ok_quant_wrong = 0
            for i in range(n_items):
                item_flip_all = True
                item_flip_any = False
                for s in seeds:
                    bc = _match(bf16[s]["details"][i])
                    oc = _match(other[s]["details"][i])
                    if bc and oc:
                        pair_counts["both_correct"] += 1
                    elif bc and not oc:
                        pair_counts["bf16_correct_quant_wrong"] += 1
                        item_flip_any = True
                    elif (not bc) and oc:
                        pair_counts["bf16_wrong_quant_correct"] += 1
                    else:
                        pair_counts["both_wrong"] += 1
                    if not (bc and not oc):
                        item_flip_all = False
                if item_flip_any:
                    item_any_bf16_ok_quant_wrong += 1
                if item_flip_all:
                    item_all_seeds_bf16_ok_quant_wrong += 1
            n_pairs = n_items * len(seeds)
            out[f"{model}_{fmt}"] = {
                "n_items": n_items,
                "n_seeds": len(seeds),
                "n_item_seed_pairs": n_pairs,
                "item_seed": pair_counts,
                "items_with_any_bf16_correct_quant_wrong": item_any_bf16_ok_quant_wrong,
                "items_with_all_seeds_bf16_correct_quant_wrong": item_all_seeds_bf16_ok_quant_wrong,
                "note": (
                    "Counts are descriptive associations under this pinned evaluation. "
                    "They are not a causal effect of a quantization method."
                ),
            }
    return out


def length_by_correctness(data: dict[str, dict[str, dict[int, dict[str, Any]]]], seeds: list[int], n_items: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model in rev.MODELS:
        for fmt in rev.FORMATS:
            toks_ok: list[float] = []
            toks_bad: list[float] = []
            for s in seeds:
                for i in range(n_items):
                    row = data[model][fmt][s]["details"][i]
                    tok = float(row.get("completion_tokens") or 0)
                    if _match(row):
                        toks_ok.append(tok)
                    else:
                        toks_bad.append(tok)
            out[f"{model}_{fmt}"] = {
                "n_correct": len(toks_ok),
                "n_incorrect": len(toks_bad),
                "mean_tokens_correct": (sum(toks_ok) / len(toks_ok) if toks_ok else 0.0),
                "mean_tokens_incorrect": (sum(toks_bad) / len(toks_bad) if toks_bad else 0.0),
                "note": "Incorrect traces are longer on average in this grid; this is a descriptive association.",
            }
    return out


def gpqa_item_breakdown() -> dict[str, Any]:
    data = rev.load_dir(rev.GPQA_DIR)
    n_items = 198
    seeds = rev.BREADTH_SEEDS
    items: list[dict[str, Any]] = []
    for i in range(n_items):
        rec: dict[str, Any] = {"row": i + 1, "cells": {}}
        for model in rev.MODELS:
            for fmt in rev.FORMATS:
                matches = [_match(data[model][fmt][s]["details"][i]) for s in seeds]
                rec["cells"][f"{model}_{fmt}"] = {
                    "n_correct_seeds": int(sum(matches)),
                    "mean_correct": sum(matches) / len(matches),
                }
        rec["qwen_awq_bf16_correct_quant_wrong_seeds"] = int(
            sum(
                _match(data["Qwen-7B"]["BF16"][s]["details"][i])
                and not _match(data["Qwen-7B"]["AWQ-4"][s]["details"][i])
                for s in seeds
            )
        )
        items.append(rec)
    ranked = sorted(items, key=lambda r: r["qwen_awq_bf16_correct_quant_wrong_seeds"], reverse=True)
    top = [
        {
            "row": r["row"],
            "qwen_awq_bf16_correct_quant_wrong_seeds": r["qwen_awq_bf16_correct_quant_wrong_seeds"],
            "qwen_bf16_n_correct_seeds": r["cells"]["Qwen-7B_BF16"]["n_correct_seeds"],
            "qwen_awq_n_correct_seeds": r["cells"]["Qwen-7B_AWQ-4"]["n_correct_seeds"],
        }
        for r in ranked[:15]
        if r["qwen_awq_bf16_correct_quant_wrong_seeds"] > 0
    ]
    hist: dict[str, int] = defaultdict(int)
    for r in items:
        hist[str(r["qwen_awq_bf16_correct_quant_wrong_seeds"])] += 1
    return {
        "n_items": n_items,
        "n_seeds": len(seeds),
        "gated_benchmark": True,
        "item_text_omitted": True,
        "qwen_awq_flip_seed_histogram": dict(hist),
        "top_qwen_awq_flip_rows": top,
        "note": (
            "Row indices are campaign order, not published GPQA prompts. "
            "The Qwen AWQ GPQA result is significant within the primary Holm-6 family, "
            "but not under the Holm-18 joint sensitivity analysis."
        ),
    }


def compute_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "metadata": {
            "cpu_only": True,
            "gpu_campaign_reopened": False,
            "causal_claims": False,
            "description": (
                "Descriptive item-level associations from frozen compact JSON. "
                "Not a new experiment."
            ),
        },
        "error_flips": {},
        "length_versus_correctness": {},
        "gpqa_item_level": gpqa_item_breakdown(),
    }
    for name, path, seeds, n_items in TASKS:
        data = rev.load_dir(path)
        report["error_flips"][name] = flip_tables(data, seeds, n_items)
        report["length_versus_correctness"][name] = length_by_correctness(data, seeds, n_items)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Item-level descriptive analysis (CPU, frozen JSON)",
        "",
        "No GPU. No causal claims. Counts are associations under the pinned A100 / vLLM 0.7.0 evaluation.",
        "",
        "## Error flips (BF16 correct, quantized wrong)",
        "",
    ]
    for task, cells in report["error_flips"].items():
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| Cell | item×seed BF16✓ quant✗ | any-item flips | all-seed flips | both✓ | quant-only✓ | both✗ |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for key, rec in cells.items():
            p = rec["item_seed"]
            lines.append(
                f"| {key} | {p['bf16_correct_quant_wrong']} | "
                f"{rec['items_with_any_bf16_correct_quant_wrong']} | "
                f"{rec['items_with_all_seeds_bf16_correct_quant_wrong']} | "
                f"{p['both_correct']} | {p['bf16_wrong_quant_correct']} | {p['both_wrong']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Length versus correctness (MATH-500)",
            "",
            "| Cell | n correct | mean tokens correct | n incorrect | mean tokens incorrect |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, rec in report["length_versus_correctness"]["math500"].items():
        lines.append(
            f"| {key} | {rec['n_correct']} | {rec['mean_tokens_correct']:.1f} | "
            f"{rec['n_incorrect']} | {rec['mean_tokens_incorrect']:.1f} |"
        )
    gpqa = report["gpqa_item_level"]
    lines.extend(
        [
            "",
            "## GPQA-Diamond item-level (row index only)",
            "",
            gpqa["note"],
            "",
            f"Qwen AWQ flip-seed histogram: `{gpqa['qwen_awq_flip_seed_histogram']}`",
            "",
            "| Row | BF16✓ AWQ✗ seeds (of 3) | Qwen BF16 correct seeds | Qwen AWQ correct seeds |",
            "|---:|---:|---:|---:|",
        ]
    )
    for rec in gpqa["top_qwen_awq_flip_rows"]:
        lines.append(
            f"| {rec['row']} | {rec['qwen_awq_bf16_correct_quant_wrong_seeds']} | "
            f"{rec['qwen_bf16_n_correct_seeds']} | {rec['qwen_awq_n_correct_seeds']} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Recompute and fail if the frozen JSON/markdown drifted.",
    )
    args = parser.parse_args(argv)
    report = compute_report()
    md = render_markdown(report)
    if args.check:
        if not os.path.isfile(OUT_JSON) or not os.path.isfile(OUT_MD):
            print("ERROR: missing frozen item-level report files", file=sys.stderr)
            return 1
        with open(OUT_JSON) as fp:
            expected = json.load(fp)
        diffs = rev.json_diff(expected, report)
        if diffs:
            print(f"ERROR: {len(diffs)} drift(s) vs {OUT_JSON}", file=sys.stderr)
            for line in diffs[:40]:
                print(f"  {line}", file=sys.stderr)
            return 1
        with open(OUT_MD) as fp:
            frozen_md = fp.read()
        if frozen_md != md:
            print(f"ERROR: markdown drift vs {OUT_MD}", file=sys.stderr)
            return 1
        print(f"OK: item-level descriptive report matches {OUT_JSON}")
        return 0
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fp:
        json.dump(report, fp, indent=2)
        fp.write("\n")
    with open(OUT_MD, "w") as fp:
        fp.write(md)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
