#!/usr/bin/env python3
"""Compare a scored summary JSON against QRM / literature baseline targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runners.config_utils import load_yaml


def _load_targets(path: Path) -> dict[str, Any]:
    return load_yaml(path) or {}


def _resolve_model_key(summary: dict[str, Any]) -> str | None:
    cell_id = str(summary.get("cell_id", "")).lower()
    model_path = str(summary.get("model_path", "")).lower()
    text = f"{cell_id} {model_path}"
    if "llama" in text and "8b" in text:
        return "DeepSeek-R1-Distill-Llama-8B"
    if "gptq" in text and "qwen" in text:
        return "DeepSeek-R1-Distill-Qwen-7B-GPTQ-W4G128"
    if "qwen" in text and "7b" in text:
        return "DeepSeek-R1-Distill-Qwen-7B"
    return None


def _task_key(summary: dict[str, Any]) -> str:
    task = str(summary.get("task", "math500")).lower()
    if task.startswith("gsm8k"):
        return "GSM8K"
    if task.startswith("gpqa"):
        return "GPQA-Diamond"
    return "MATH-500"


def compare_summary(summary: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    model_key = _resolve_model_key(summary)
    task_key = _task_key(summary)
    report: dict[str, Any] = {
        "cell_id": summary.get("cell_id"),
        "model_key": model_key,
        "task": task_key,
        "checks": [],
        "passed": True,
    }

    if not model_key:
        report["checks"].append({"status": "SKIP", "message": "Could not infer model from summary"})
        report["passed"] = None
        return report

    model_targets = (targets.get("models") or {}).get(model_key, {})
    task_targets = model_targets.get(task_key)
    if not task_targets:
        report["checks"].append({"status": "SKIP", "message": f"No targets for {model_key} / {task_key}"})
        report["passed"] = None
        return report

    pass_pct = float(summary.get("pass_at_1", 0.0)) * 100.0
    pass_cfg = task_targets.get("pass_at_1_pct") or {}
    ref = pass_cfg.get("reference")
    lo = pass_cfg.get("sanity_min")
    hi = pass_cfg.get("sanity_max")

    if lo is not None and hi is not None:
        in_band = lo <= pass_pct <= hi
        report["checks"].append({
            "metric": "pass_at_1_pct",
            "observed": round(pass_pct, 2),
            "reference": ref,
            "sanity_band": [lo, hi],
            "status": "PASS" if in_band else "FAIL",
        })
        if not in_band:
            report["passed"] = False

    trunc_max = task_targets.get("truncation_rate_max")
    if trunc_max is not None and "truncation_rate" in summary:
        trunc = float(summary["truncation_rate"])
        ok = trunc <= trunc_max
        report["checks"].append({
            "metric": "truncation_rate",
            "observed": round(trunc, 4),
            "max": trunc_max,
            "status": "PASS" if ok else "FAIL",
        })
        if not ok:
            report["passed"] = False

    parse_max = task_targets.get("parse_failure_rate_max")
    if parse_max is not None and "parse_failure_rate" in summary:
        pf = float(summary["parse_failure_rate"])
        ok = pf <= parse_max
        report["checks"].append({
            "metric": "parse_failure_rate",
            "observed": round(pf, 4),
            "max": parse_max,
            "status": "PASS" if ok else "FAIL",
        })
        if not ok:
            report["passed"] = False

    steps_cfg = task_targets.get("reasoning_steps_mean") or {}
    if steps_cfg and "reasoning_steps_mean" in summary:
        steps = float(summary["reasoning_steps_mean"])
        lo_s = steps_cfg.get("sanity_min")
        hi_s = steps_cfg.get("sanity_max")
        if lo_s is not None and hi_s is not None:
            ok = lo_s <= steps <= hi_s
            report["checks"].append({
                "metric": "reasoning_steps_mean",
                "observed": round(steps, 2),
                "sanity_band": [lo_s, hi_s],
                "status": "PASS" if ok else "WARN",
            })

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare summary JSON to QRM literature targets.")
    parser.add_argument("--summary", required=True, help="results/*_summary.json path")
    parser.add_argument(
        "--targets",
        default="configs/baselines/qrm_literature_targets.yaml",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    summary_path = ROOT / args.summary
    targets_path = ROOT / args.targets
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    targets = _load_targets(targets_path)

    report = compare_summary(summary, targets)
    print(json.dumps(report, indent=2))

    if args.output:
        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")

    if report.get("passed") is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
