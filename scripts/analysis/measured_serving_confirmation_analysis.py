#!/usr/bin/env python3
"""Statistical analysis and cost modeling for measured serving confirmation benchmark.

Aggregates:
- Confirmatory runs across 8 configs × 2 conditions × R technical reps
- Secondary fixed-token microbenchmarks
- Speedups, latencies, empirical GPU-sec/query, and Cost-of-Pass under $1.50/GPU-hr
- Automated diff check against committed JSON
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS = ["Qwen-7B", "Llama-8B"]
FORMATS = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]
CONDITIONS = ["A_single_stream_c1", "B_batched_throughput_c8"]

# Canonical 40-cell MATH-500 accuracy numbers (from Paper 1 campaign)
CANONICAL_PASS1 = {
    ("Qwen-7B", "BF16"): 0.9400,
    ("Qwen-7B", "FP8"): 0.9440,
    ("Qwen-7B", "AWQ-4"): 0.9312,
    ("Qwen-7B", "GPTQ-4"): 0.9348,
    ("Llama-8B", "BF16"): 0.8924,
    ("Llama-8B", "FP8"): 0.8952,
    ("Llama-8B", "AWQ-4"): 0.8648,
    ("Llama-8B", "GPTQ-4"): 0.8892,
}

# Old fixed-throughput (65 tok/s) proxy baseline
OLD_PROXY_COST = {
    ("Qwen-7B", "BF16"): {"mean_tokens": 3789.4, "cost_query_dollars": 0.02429, "cost_pass_dollars": 0.02584},
    ("Qwen-7B", "FP8"): {"mean_tokens": 3792.1, "cost_query_dollars": 0.02431, "cost_pass_dollars": 0.02575},
    ("Qwen-7B", "AWQ-4"): {"mean_tokens": 4028.6, "cost_query_dollars": 0.02582, "cost_pass_dollars": 0.02773},
    ("Qwen-7B", "GPTQ-4"): {"mean_tokens": 4051.8, "cost_query_dollars": 0.02597, "cost_pass_dollars": 0.02778},
    ("Llama-8B", "BF16"): {"mean_tokens": 4447.8, "cost_query_dollars": 0.02851, "cost_pass_dollars": 0.03195},
    ("Llama-8B", "FP8"): {"mean_tokens": 4346.5, "cost_query_dollars": 0.02786, "cost_pass_dollars": 0.03112},
    ("Llama-8B", "AWQ-4"): {"mean_tokens": 4524.3, "cost_query_dollars": 0.02900, "cost_pass_dollars": 0.03353},
    ("Llama-8B", "GPTQ-4"): {"mean_tokens": 4625.7, "cost_query_dollars": 0.02965, "cost_pass_dollars": 0.03335},
}


def load_raw_confirmation_data(
    raw_dir: Path,
) -> Tuple[Dict[Tuple[str, str, str], List[Dict[str, Any]]], Dict[Tuple[str, str], Dict[str, Any]]]:
    """Load all raw JSON confirmation runs into indexed dictionaries."""
    task_runs: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    micro_runs: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for f in raw_dir.glob("*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        btype = data.get("benchmark_type", "")
        model = data.get("model", "")
        fmt = data.get("format", "")

        if btype == "task_realistic_confirmation":
            cond = data.get("condition", "")
            key = (model, fmt, cond)
            task_runs.setdefault(key, []).append(data)
        elif btype == "fixed_token_microbenchmark_confirmation":
            key_m = (model, fmt)
            micro_runs[key_m] = data

    for k in task_runs:
        task_runs[k].sort(key=lambda x: x.get("repetition", 0))

    return task_runs, micro_runs


def analyze_confirmation_data(raw_dir: Path) -> Dict[str, Any]:
    """Perform full statistical analysis and Cost-of-Pass modeling on confirmation data."""
    task_runs, micro_runs = load_raw_confirmation_data(raw_dir)

    report: Dict[str, Any] = {
        "metadata": {
            "analysis_title": "Measured Serving Confirmation Analysis (Controlled Apples-to-Apples)",
            "benchmark_stack": "qrm-official (vLLM 0.7.0 eager, PyTorch 2.5.1+cu124, CUDA 12.4)",
            "gpu_hardware": "NVIDIA A100-PCIE-80GB",
            "gpu_dollar_per_hour": 1.50,
            "gpu_dollar_per_second": 1.50 / 3600.0,
            "condition_a_workload": "Balanced 20 MATH-500 prompts (4 per Level 1-5), concurrency=1",
            "condition_b_workload": "Balanced 100 MATH-500 prompts (20 per Level 1-5), max_num_seqs=8 pinned",
            "secondary_microbenchmark": "10 sample prompts, fixed 512 tokens pure decode",
        },
        "configurations": {},
        "summary_table_latex": "",
    }

    # First pass: compute absolute metrics per configuration
    config_data: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for model in MODELS:
        for fmt in FORMATS:
            cfg_entry: Dict[str, Any] = {
                "model": model,
                "format": fmt,
                "pass1_canonical": CANONICAL_PASS1.get((model, fmt), 0.0),
                "conditions": {},
            }

            for cond in CONDITIONS:
                runs = task_runs.get((model, fmt, cond), [])
                if not runs:
                    continue

                tok_speeds = [r["output_tokens_per_second"] for r in runs]
                req_speeds = [r["requests_per_second"] for r in runs]
                gpu_secs = [r["gpu_seconds_per_query"] for r in runs]
                elapseds = [r["elapsed_seconds"] for r in runs]
                tot_out_toks = [r["total_output_tokens"] for r in runs]
                peak_vrams = [r["peak_vram_allocated_gb"] for r in runs]

                mean_tok_speed = statistics.mean(tok_speeds)
                std_tok_speed = statistics.stdev(tok_speeds) if len(tok_speeds) > 1 else 0.0
                cv_pct = (std_tok_speed / mean_tok_speed * 100.0) if mean_tok_speed > 0 else 0.0

                mean_req_speed = statistics.mean(req_speeds)
                mean_gpu_sec = statistics.mean(gpu_secs)
                std_gpu_sec = statistics.stdev(gpu_secs) if len(gpu_secs) > 1 else 0.0
                mean_peak_vram = statistics.mean(peak_vrams)
                weights_mem = runs[0].get("model_weights_memory_gb")

                # Cost calculations ($1.50/GPU-hr)
                cost_per_query_dollars = mean_gpu_sec * (1.50 / 3600.0)
                pass1 = CANONICAL_PASS1.get((model, fmt), 1.0)
                cost_pass_dollars = cost_per_query_dollars / pass1 if pass1 > 0 else 0.0

                cond_dict: Dict[str, Any] = {
                    "repetition_count": len(runs),
                    "mean_tokens_per_second": mean_tok_speed,
                    "std_tokens_per_second": std_tok_speed,
                    "cv_percent": cv_pct,
                    "mean_requests_per_second": mean_req_speed,
                    "mean_gpu_seconds_per_query": mean_gpu_sec,
                    "std_gpu_seconds_per_query": std_gpu_sec,
                    "mean_total_output_tokens": statistics.mean(tot_out_toks),
                    "mean_elapsed_seconds": statistics.mean(elapseds),
                    "model_weights_memory_gb": weights_mem,
                    "peak_allocated_engine_vram_gb": mean_peak_vram,
                    "cost_per_query_dollars": cost_per_query_dollars,
                    "measured_cost_pass_dollars": cost_pass_dollars,
                }

                # Condition A specific latencies
                if cond == "A_single_stream_c1":
                    med_lats = [r["latency_median_sec"] for r in runs]
                    p90_lats = [r["latency_p90_sec"] for r in runs]
                    p95_lats = [r["latency_p95_sec"] for r in runs]
                    cond_dict["latency_median_sec"] = statistics.mean(med_lats)
                    cond_dict["latency_p90_sec"] = statistics.mean(p90_lats)
                    cond_dict["latency_p95_sec"] = statistics.mean(p95_lats)

                cfg_entry["conditions"][cond] = cond_dict

            # Secondary microbenchmark
            m_run = micro_runs.get((model, fmt))
            if m_run:
                cfg_entry["microbenchmark"] = {
                    "fixed_tokens_per_req": m_run.get("fixed_tokens_per_request", 512),
                    "raw_decode_tokens_per_second": m_run.get("raw_decode_tokens_per_second", 0.0),
                }

            config_data[(model, fmt)] = cfg_entry

    # Second pass: Compute deltas and speedups relative to BF16 anchor
    for model in MODELS:
        bf16_cfg = config_data.get((model, "BF16"))
        if not bf16_cfg:
            continue

        for fmt in FORMATS:
            cfg = config_data.get((model, fmt))
            if not cfg:
                continue

            for cond in CONDITIONS:
                c_dict = cfg["conditions"].get(cond)
                b_dict = bf16_cfg["conditions"].get(cond)
                if not (c_dict and b_dict):
                    continue

                b_tok = b_dict["mean_tokens_per_second"]
                c_tok = c_dict["mean_tokens_per_second"]
                speedup = (c_tok / b_tok) if b_tok > 0 else 1.0

                b_gpu_sec = b_dict["mean_gpu_seconds_per_query"]
                c_gpu_sec = c_dict["mean_gpu_seconds_per_query"]
                gpu_sec_delta_pct = ((c_gpu_sec - b_gpu_sec) / b_gpu_sec * 100.0) if b_gpu_sec > 0 else 0.0

                b_cpass = b_dict["measured_cost_pass_dollars"]
                c_cpass = c_dict["measured_cost_pass_dollars"]
                cpass_delta_pct = ((c_cpass - b_cpass) / b_cpass * 100.0) if b_cpass > 0 else 0.0

                c_dict["speedup_vs_bf16"] = speedup
                c_dict["gpu_seconds_delta_vs_bf16_pct"] = gpu_sec_delta_pct
                c_dict["cost_pass_delta_vs_bf16_pct"] = cpass_delta_pct

                # Compare with old fixed-throughput proxy
                old_p = OLD_PROXY_COST.get((model, fmt), {})
                if old_p:
                    c_dict["old_proxy_cost_pass_dollars"] = old_p.get("cost_pass_dollars", 0.0)
                    old_cpass = old_p.get("cost_pass_dollars", 0.0)
                    proxy_gap_pct = ((c_cpass - old_cpass) / old_cpass * 100.0) if old_cpass > 0 else 0.0
                    c_dict["measured_vs_old_proxy_gap_pct"] = proxy_gap_pct

            # Microbenchmark speedup vs BF16
            m_curr = cfg.get("microbenchmark")
            m_bf16 = bf16_cfg.get("microbenchmark")
            if m_curr and m_bf16:
                b_raw = m_bf16.get("raw_decode_tokens_per_second", 0.0)
                c_raw = m_curr.get("raw_decode_tokens_per_second", 0.0)
                m_curr["raw_decode_speedup_vs_bf16"] = (c_raw / b_raw) if b_raw > 0 else 1.0

            key_name = f"{model}_{fmt}"
            report["configurations"][key_name] = cfg

    return report


def generate_confirmation_markdown_reports(
    report: Dict[str, Any],
    output_md: Path,
    output_val_md: Path,
) -> None:
    """Generate human-readable summary and validation markdown reports."""
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_val_md.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# MEASURED SERVING CONFIRMATION BENCHMARK REPORT")
    lines.append(f"**Cluster:** PARAM Rudra HPC (NVIDIA A100-PCIE-80GB)  ")
    lines.append(f"**Serving Stack:** `qrm-official` (vLLM 0.7.0 eager, PyTorch 2.5.1+cu124, CUDA 12.4)  ")
    lines.append(f"**Pricing Baseline:** $1.50 / A100 GPU-Hour ($0.00041667 / GPU-sec)  ")
    lines.append("")

    lines.append("## 1. Executive Summary Table")
    lines.append("")
    lines.append(
        "| Model | Format | Pass@1 (MATH-500) | Cond A Tok/s (C=1) | Cond A Median Lat (s) | Cond B Tok/s (C=8) | Cond B Req/s | Empirical GPU-sec/q | Measured $C_{\\text{pass}}$ | $C_{\\text{pass}}$ Delta vs BF16 |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|"
    )

    for model in MODELS:
        for fmt in FORMATS:
            cfg = report["configurations"].get(f"{model}_{fmt}", {})
            p1 = cfg.get("pass1_canonical", 0.0) * 100.0
            cA = cfg.get("conditions", {}).get("A_single_stream_c1", {})
            cB = cfg.get("conditions", {}).get("B_batched_throughput_c8", {})

            cA_tok = f"{cA.get('mean_tokens_per_second', 0.0):.1f} ± {cA.get('std_tokens_per_second', 0.0):.1f}"
            cA_med = f"{cA.get('latency_median_sec', 0.0):.2f}s"
            cB_tok = f"{cB.get('mean_tokens_per_second', 0.0):.1f} ± {cB.get('std_tokens_per_second', 0.0):.1f}"
            cB_req = f"{cB.get('mean_requests_per_second', 0.0):.3f}"
            gpu_sec = f"{cB.get('mean_gpu_seconds_per_query', 0.0):.2f}s"
            cpass = f"${cB.get('measured_cost_pass_dollars', 0.0):.4f}"
            delta_bf16 = cB.get("cost_pass_delta_vs_bf16_pct", 0.0)
            delta_str = f"{delta_bf16:+.1f}%" if fmt != "BF16" else "Anchor"

            lines.append(
                f"| **{model}** | **{fmt}** | {p1:.2f}% | {cA_tok} | {cA_med} | {cB_tok} | {cB_req} | {gpu_sec} | {cpass} | {delta_str} |"
            )

    lines.append("")
    lines.append("## 2. Secondary Fixed-Token Microbenchmark (Pure Decode Speed)")
    lines.append("")
    lines.append("| Model | Format | Fixed Tokens | Raw Decode Tok/s | Speedup vs BF16 |")
    lines.append("|---|---|---|---|---|")
    for model in MODELS:
        for fmt in FORMATS:
            cfg = report["configurations"].get(f"{model}_{fmt}", {})
            m = cfg.get("microbenchmark", {})
            toks = m.get("fixed_tokens_per_req", 512)
            raw_spd = m.get("raw_decode_tokens_per_second", 0.0)
            raw_spdup = m.get("raw_decode_speedup_vs_bf16", 1.0)
            lines.append(
                f"| **{model}** | **{fmt}** | {toks} | {raw_spd:.2f} tok/s | {raw_spdup:.2f}× |"
            )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Confirmation markdown report written to: {output_md}")


def json_diff(expected: Any, actual: Any, path: str = "") -> List[str]:
    """Compare two nested structures."""
    diffs = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for k in set(expected.keys()).union(actual.keys()):
            if k not in expected:
                diffs.append(f"{path}.{k}: key missing in expected")
            elif k not in actual:
                diffs.append(f"{path}.{k}: key missing in actual")
            else:
                diffs.extend(json_diff(expected[k], actual[k], f"{path}.{k}"))
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: list length mismatch {len(expected)} vs {len(actual)}")
        else:
            for i, (e, a) in enumerate(zip(expected, actual)):
                diffs.extend(json_diff(e, a, f"{path}[{i}]"))
    elif isinstance(expected, float) and isinstance(actual, (float, int)):
        if not math.isclose(expected, actual, rel_tol=1e-4, abs_tol=1e-5):
            diffs.append(f"{path}: float mismatch {expected} vs {actual}")
    else:
        if expected != actual:
            diffs.append(f"{path}: value mismatch {expected!r} vs {actual!r}")
    return diffs


def main():
    parser = argparse.ArgumentParser(description="Analyze measured serving confirmation benchmark data.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "results" / "measured_serving_confirmation" / "raw",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=REPO_ROOT / "results" / "reports" / "measured_serving_confirmation" / "measured_serving_confirmation_report.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=REPO_ROOT / "results" / "reports" / "measured_serving_confirmation" / "measured_serving_confirmation_report.md",
    )
    parser.add_argument(
        "--validation-md",
        type=Path,
        default=REPO_ROOT / "results" / "reports" / "measured_serving_confirmation" / "measured_serving_confirmation_validation.md",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Recompute and exit nonzero if the result differs from checked-in JSON.",
    )
    args = parser.parse_args()

    results = analyze_confirmation_data(args.raw_dir)

    if args.check:
        if not args.report_json.is_file():
            print(f"ERROR: missing checked-in report {args.report_json}", file=sys.stderr)
            return 1
        expected = json.loads(args.report_json.read_text(encoding="utf-8"))
        diffs = json_diff(expected, results)
        if diffs:
            print(f"ERROR: {len(diffs)} drift(s) vs {args.report_json}", file=sys.stderr)
            for line in diffs[:50]:
                print(f"  {line}", file=sys.stderr)
            return 1
        n_cfg = len(results["configurations"])
        print(f"OK: recomputed confirmation report matches {args.report_json} ({n_cfg} configs)")
        return 0

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved confirmation JSON report: {args.report_json}")

    generate_confirmation_markdown_reports(results, args.report_md, args.validation_md)
    print("\nMEASURED SERVING CONFIRMATION ANALYSIS COMPLETED SUCCESSFULLY!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
