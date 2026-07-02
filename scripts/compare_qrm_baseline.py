#!/usr/bin/env python3
"""Compare a scored summary JSON against QRM / literature baseline targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runners.config_utils import load_yaml


def _load_targets(path: Path) -> dict[str, Any]:
    return load_yaml(path) or {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_head_commit(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _resolve_model_key(summary: dict[str, Any]) -> str | None:
    cell_id = str(summary.get("cell_id", "")).lower()
    model_path = str(summary.get("model_path", "")).lower()
    text = f"{cell_id} {model_path}"
    if "llama" in text and "8b" in text:
        return "DeepSeek-R1-Distill-Llama-8B"
    if "gptq" in text and "qwen" in text and "7b" in text:
        return "DeepSeek-R1-Distill-Qwen-7B-GPTQ-W4G128"
    if "qwen" in text and "1.5b" in text:
        return "DeepSeek-R1-Distill-Qwen-1.5B"
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


def _default_tolerance(targets: dict[str, Any], task_key: str) -> float:
    tol = targets.get("tolerance") or {}
    if task_key == "GPQA-Diamond":
        return float(tol.get("gpqa_absolute_pp", 8.0))
    return float(tol.get("pass_at_1_absolute_pp_default", 5.0))


def _pass_band(
    pass_cfg: dict[str, Any],
    *,
    default_tol: float,
) -> tuple[float | None, float | None, float]:
    tol = float(pass_cfg.get("tolerance_pp", default_tol))
    lo = pass_cfg.get("sanity_min")
    hi = pass_cfg.get("sanity_max")
    if lo is not None and hi is not None:
        return float(lo), float(hi), tol
    ref = pass_cfg.get("reference")
    if ref is None:
        return None, None, tol
    ref_f = float(ref)
    return ref_f - tol, ref_f + tol, tol


def _targets_provenance(targets_path: Path, targets: dict[str, Any]) -> dict[str, Any]:
    tol = targets.get("tolerance") or {}
    return {
        "yaml_path": str(targets_path.resolve()),
        "yaml_sha256": _sha256_file(targets_path),
        "repo_git_commit": _git_head_commit(ROOT),
        "tolerance_default_pp": tol.get("pass_at_1_absolute_pp_default"),
        "tolerance_gpqa_pp": tol.get("gpqa_absolute_pp"),
        "tolerance_note": tol.get("note"),
    }


def compare_summary(
    summary: dict[str, Any],
    targets: dict[str, Any],
    *,
    targets_path: Path,
) -> dict[str, Any]:
    model_key = _resolve_model_key(summary)
    task_key = _task_key(summary)
    report: dict[str, Any] = {
        "targets_provenance": _targets_provenance(targets_path, targets),
        "cell_id": summary.get("cell_id"),
        "model_key": model_key,
        "task": task_key,
        "gate": None,
        "checks": [],
        "hard_passed": True,
        "passed": True,
    }

    if not model_key:
        report["checks"].append({"status": "SKIP", "message": "Could not infer model from summary"})
        report["passed"] = None
        report["hard_passed"] = None
        return report

    model_targets = (targets.get("models") or {}).get(model_key, {})
    task_targets = model_targets.get(task_key)
    if not task_targets:
        report["checks"].append({"status": "SKIP", "message": f"No targets for {model_key} / {task_key}"})
        report["passed"] = None
        report["hard_passed"] = None
        return report

    if task_targets.get("status") == "unused":
        report["checks"].append({
            "status": "SKIP",
            "message": task_targets.get("note") or "Row marked unused in yaml",
        })
        report["passed"] = None
        report["hard_passed"] = None
        return report

    gate_type = str(task_targets.get("gate", "sanity"))
    default_tol = _default_tolerance(targets, task_key)
    pass_cfg = task_targets.get("pass_at_1_pct") or {}
    ref = pass_cfg.get("reference")
    lo, hi, tol_pp = _pass_band(pass_cfg, default_tol=default_tol)
    source = pass_cfg.get("source")

    report["gate"] = {
        "model_key": model_key,
        "task": task_key,
        "gate_type": gate_type,
        "prompt_profile": task_targets.get("prompt_profile"),
        "quant_config": task_targets.get("quant_config"),
        "reference_pct": ref,
        "reference_std": pass_cfg.get("reference_std"),
        "reference_deepseek_report": pass_cfg.get("reference_deepseek_report"),
        "tolerance_pp": tol_pp,
        "sanity_band_pct": [lo, hi] if lo is not None and hi is not None else None,
        "source": source,
        "source_secondary": pass_cfg.get("source_secondary"),
        "yaml_path": str(targets_path.resolve()),
        "yaml_sha256": report["targets_provenance"]["yaml_sha256"],
    }

    pass_pct = float(summary.get("pass_at_1", 0.0)) * 100.0
    if lo is not None and hi is not None:
        in_band = lo <= pass_pct <= hi
        if gate_type == "hard":
            status = "PASS" if in_band else "FAIL"
            if not in_band:
                report["hard_passed"] = False
                report["passed"] = False
        else:
            status = "PASS" if in_band else "SANITY_WARN"
        report["checks"].append({
            "metric": "pass_at_1_pct",
            "observed": round(pass_pct, 2),
            "reference": ref,
            "reference_std": pass_cfg.get("reference_std"),
            "reference_deepseek_report": pass_cfg.get("reference_deepseek_report"),
            "source": source,
            "source_secondary": pass_cfg.get("source_secondary"),
            "tolerance_pp": tol_pp,
            "gate_type": gate_type,
            "sanity_band": [lo, hi],
            "status": status,
        })

    trunc_max = task_targets.get("truncation_rate_max")
    if trunc_max is not None and "truncation_rate" in summary:
        trunc = float(summary["truncation_rate"])
        ok = trunc <= trunc_max
        check = {
            "metric": "truncation_rate",
            "observed": round(trunc, 4),
            "max": trunc_max,
            "gate_type": gate_type,
            "status": "PASS" if ok else ("FAIL" if gate_type == "hard" else "SANITY_WARN"),
        }
        report["checks"].append(check)
        if not ok and gate_type == "hard":
            report["hard_passed"] = False
            report["passed"] = False

    parse_max = task_targets.get("parse_failure_rate_max")
    if parse_max is not None and "parse_failure_rate" in summary:
        pf = float(summary["parse_failure_rate"])
        ok = pf <= parse_max
        check = {
            "metric": "parse_failure_rate",
            "observed": round(pf, 4),
            "max": parse_max,
            "gate_type": gate_type,
            "status": "PASS" if ok else ("FAIL" if gate_type == "hard" else "SANITY_WARN"),
        }
        report["checks"].append(check)
        if not ok and gate_type == "hard":
            report["hard_passed"] = False
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

    tokens_cfg = task_targets.get("completion_tokens_mean") or {}
    if tokens_cfg and "completion_tokens_mean" in summary:
        tokens_mean = float(summary["completion_tokens_mean"])
        lo_t = tokens_cfg.get("sanity_min")
        hi_t = tokens_cfg.get("sanity_max")
        if lo_t is not None:
            ok = tokens_mean >= lo_t if hi_t is None else lo_t <= tokens_mean <= hi_t
            check = {
                "metric": "completion_tokens_mean",
                "observed": round(tokens_mean, 1),
                "sanity_min": lo_t,
                "sanity_max": hi_t,
                "gate_type": gate_type,
                "status": "PASS" if ok else ("FAIL" if gate_type == "hard" else "SANITY_WARN"),
            }
            report["checks"].append(check)
            if not ok and gate_type == "hard":
                report["hard_passed"] = False
                report["passed"] = False

    return report


def _print_provenance_banner(report: dict[str, Any]) -> None:
    prov = report.get("targets_provenance") or {}
    gate = report.get("gate") or {}
    print("=== QRM baseline gate provenance ===", file=sys.stderr)
    print(f"  yaml:        {prov.get('yaml_path')}", file=sys.stderr)
    print(f"  yaml sha256: {prov.get('yaml_sha256')}", file=sys.stderr)
    print(f"  git commit:  {prov.get('repo_git_commit')}", file=sys.stderr)
    print(
        f"  tolerance:   default ±{prov.get('tolerance_default_pp')} pp; "
        f"GPQA ±{prov.get('tolerance_gpqa_pp')} pp",
        file=sys.stderr,
    )
    if gate:
        band = gate.get("sanity_band_pct")
        print(
            f"  gate:        {gate.get('gate_type')} — {gate.get('model_key')} / {gate.get('task')} "
            f"ref={gate.get('reference_pct')}% ±{gate.get('tolerance_pp')}pp band={band}",
            file=sys.stderr,
        )
        print(f"  source:      {gate.get('source')}", file=sys.stderr)
        if gate.get("source_secondary"):
            print(f"  cross-check: {gate.get('source_secondary')}", file=sys.stderr)
        if gate.get("prompt_profile"):
            print(f"  profile:     {gate.get('prompt_profile')}", file=sys.stderr)
    print(f"  hard_passed: {report.get('hard_passed')}", file=sys.stderr)
    print("====================================", file=sys.stderr)


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

    report = compare_summary(summary, targets, targets_path=targets_path)
    _print_provenance_banner(report)
    print(json.dumps(report, indent=2))

    if args.output:
        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")

    if report.get("hard_passed") is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
