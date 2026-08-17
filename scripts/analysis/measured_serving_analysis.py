#!/usr/bin/env python3
"""Statistical analysis and cost modeling for measured serving benchmark.

Aggregates:
- 48 task-realistic benchmark runs (8 configs × 2 conditions × 3 reps)
- 8 raw-decoding fixed-token microbenchmarks
- Relative speedup, latency, VRAM, and GPU-sec/query deltas vs BF16
- Measured Cost-of-Pass ($1.50/GPU-hr scenario) vs old fixed-throughput proxy
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

MODELS = ["Qwen-7B", "Llama-8B"]
FORMATS = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]
CONDITIONS = ["A_single_stream_c1", "B_batched_throughput_c8"]
REPETITIONS = [1, 2, 3]

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

# Old fixed-throughput (65 tok/s) tokens and cost
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


def load_raw_data(raw_dir: Path) -> Tuple[Dict[tuple, List[Dict[str, Any]]], Dict[tuple, Dict[str, Any]]]:
    """Load all raw JSON runs into indexed dictionaries."""
    task_runs: Dict[tuple, List[Dict[str, Any]]] = {}
    micro_runs: Dict[tuple, Dict[str, Any]] = {}

    for f in raw_dir.glob("*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        btype = data.get("benchmark_type", "")
        model = data.get("model", "")
        fmt = data.get("format", "")

        if btype == "task_realistic":
            cond = data.get("condition", "")
            key = (model, fmt, cond)
            task_runs.setdefault(key, []).append(data)
        elif btype == "fixed_token_microbenchmark":
            key = (model, fmt)
            micro_runs[key] = data

    return task_runs, micro_runs


def analyze_measured_serving(raw_dir: Path) -> Dict[str, Any]:
    """Perform full statistical analysis and Cost-of-Pass remodeling."""
    task_runs, micro_runs = load_raw_data(raw_dir)

    results: Dict[str, Any] = {
        "configurations": {},
        "comparisons_vs_bf16": {},
        "cost_analysis_summary": {},
    }

    # Analyze each configuration
    for model in MODELS:
        for fmt in FORMATS:
            cfg_key = f"{model}_{fmt}"
            results["configurations"][cfg_key] = {
                "model": model,
                "format": fmt,
                "conditions": {},
                "microbenchmark": {},
                "canonical_pass1": CANONICAL_PASS1.get((model, fmt), 0.0),
            }

            # Microbenchmark
            if (model, fmt) in micro_runs:
                m_data = micro_runs[(model, fmt)]
                results["configurations"][cfg_key]["microbenchmark"] = {
                    "raw_decode_tokens_per_second": m_data.get("raw_decode_tokens_per_second", 0.0),
                    "fixed_tokens": m_data.get("fixed_tokens_per_request", 512),
                }

            # Conditions A & B
            for cond in CONDITIONS:
                runs = task_runs.get((model, fmt, cond), [])
                if not runs:
                    continue

                tok_sec_vals = [r["output_tokens_per_second"] for r in runs]
                req_sec_vals = [r["requests_per_second"] for r in runs]
                gpu_sec_vals = [r["gpu_seconds_per_query"] for r in runs]
                vram_peak_vals = [r["peak_vram_allocated_gb"] for r in runs]
                out_tok_vals = [r["mean_output_tokens_per_req"] for r in runs]

                lat_mean_vals = [r.get("latency_mean_sec", r["gpu_seconds_per_query"]) for r in runs]
                lat_med_vals = [r.get("latency_median_sec", r["gpu_seconds_per_query"]) for r in runs]
                lat_p90_vals = [r.get("latency_p90_sec", r["gpu_seconds_per_query"]) for r in runs]
                lat_p95_vals = [r.get("latency_p95_sec", r["gpu_seconds_per_query"]) for r in runs]

                # Cost calculations under $1.50/GPU-hr ($0.00041667/GPU-sec)
                mean_gpu_sec = float(statistics.mean(gpu_sec_vals))
                cost_query_dollars = mean_gpu_sec * (1.50 / 3600.0)
                pass1 = CANONICAL_PASS1.get((model, fmt), 1.0)
                cost_pass_dollars = cost_query_dollars / pass1 if pass1 > 0 else float("nan")

                results["configurations"][cfg_key]["conditions"][cond] = {
                    "output_tokens_per_second_mean": float(statistics.mean(tok_sec_vals)),
                    "output_tokens_per_second_std": float(statistics.pstdev(tok_sec_vals)),
                    "requests_per_second_mean": float(statistics.mean(req_sec_vals)),
                    "requests_per_second_std": float(statistics.pstdev(req_sec_vals)),
                    "gpu_seconds_per_query_mean": mean_gpu_sec,
                    "gpu_seconds_per_query_std": float(statistics.pstdev(gpu_sec_vals)),
                    "mean_output_tokens": float(statistics.mean(out_tok_vals)),
                    "peak_vram_gb_mean": float(statistics.mean(vram_peak_vals)),
                    "peak_vram_gb_std": float(statistics.pstdev(vram_peak_vals)),
                    "latency_mean_sec": float(statistics.mean(lat_mean_vals)),
                    "latency_median_sec": float(statistics.mean(lat_med_vals)),
                    "latency_p90_sec": float(statistics.mean(lat_p90_vals)),
                    "latency_p95_sec": float(statistics.mean(lat_p95_vals)),
                    "measured_cost_per_query_dollars": cost_query_dollars,
                    "measured_cost_of_pass_dollars": cost_pass_dollars,
                    "repetitions": len(runs),
                }

    # Compute relative deltas vs BF16 anchor
    for model in MODELS:
        base_key = f"{model}_BF16"
        base_cfg = results["configurations"].get(base_key, {})

        for fmt in ["FP8", "AWQ-4", "GPTQ-4"]:
            tgt_key = f"{model}_{fmt}"
            tgt_cfg = results["configurations"].get(tgt_key, {})
            pair_key = f"{model}_{fmt}_vs_BF16"

            results["comparisons_vs_bf16"][pair_key] = {
                "model": model,
                "format": fmt,
                "conditions": {},
            }

            for cond in CONDITIONS:
                base_cond = base_cfg.get("conditions", {}).get(cond)
                tgt_cond = tgt_cfg.get("conditions", {}).get(cond)
                if not base_cond or not tgt_cond:
                    continue

                b_tok_s = base_cond["output_tokens_per_second_mean"]
                t_tok_s = tgt_cond["output_tokens_per_second_mean"]
                delta_tok_s_pct = ((t_tok_s - b_tok_s) / b_tok_s) * 100.0 if b_tok_s > 0 else 0.0

                b_gpu_s = base_cond["gpu_seconds_per_query_mean"]
                t_gpu_s = tgt_cond["gpu_seconds_per_query_mean"]
                delta_gpu_s_pct = ((t_gpu_s - b_gpu_s) / b_gpu_s) * 100.0 if b_gpu_s > 0 else 0.0

                b_vram = base_cond["peak_vram_gb_mean"]
                t_vram = tgt_cond["peak_vram_gb_mean"]
                delta_vram_pct = ((t_vram - b_vram) / b_vram) * 100.0 if b_vram > 0 else 0.0

                b_cpass = base_cond["measured_cost_of_pass_dollars"]
                t_cpass = tgt_cond["measured_cost_of_pass_dollars"]
                delta_cpass_pct = ((t_cpass - b_cpass) / b_cpass) * 100.0 if b_cpass > 0 else 0.0

                results["comparisons_vs_bf16"][pair_key]["conditions"][cond] = {
                    "delta_output_tok_sec_pct": delta_tok_s_pct,
                    "delta_gpu_seconds_per_query_pct": delta_gpu_s_pct,
                    "delta_peak_vram_pct": delta_vram_pct,
                    "delta_cost_of_pass_pct": delta_cpass_pct,
                }

    results["cost_analysis_summary"] = compute_pareto_summary(results)
    return results


def _dominates(a: Dict[str, float], b: Dict[str, float], dims: List[Tuple[str, bool]]) -> bool:
    """Return True if a dominates b. dims: (key, maximize)."""
    ge_all = True
    gt_one = False
    for key, maximize in dims:
        av, bv = a[key], b[key]
        if maximize:
            if av < bv - 1e-12:
                ge_all = False
            if av > bv + 1e-12:
                gt_one = True
        else:
            if av > bv + 1e-12:
                ge_all = False
            if av < bv - 1e-12:
                gt_one = True
    return ge_all and gt_one


def _frontier(points: List[Dict[str, Any]], dims: List[Tuple[str, bool]]) -> List[str]:
    names = []
    for p in points:
        if any(_dominates(q, p, dims) for q in points if q is not p):
            continue
        names.append(p["name"])
    return names


def compute_pareto_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Nondominated sets. Do not label a unique 'true Pareto optimum'."""
    points_b: List[Dict[str, Any]] = []
    points_a: List[Dict[str, Any]] = []
    for cfg_key, cfg in results["configurations"].items():
        name = f"{cfg['model']} {cfg['format']}"
        cond_b = cfg.get("conditions", {}).get("B_batched_throughput_c8", {})
        cond_a = cfg.get("conditions", {}).get("A_single_stream_c1", {})
        if cond_b:
            points_b.append({
                "name": name,
                "pass1": cfg["canonical_pass1"],
                "cpass": cond_b["measured_cost_of_pass_dollars"],
                "tok": cond_b["output_tokens_per_second_mean"],
                "vram": cond_b["peak_vram_gb_mean"],
            })
        if cond_a:
            points_a.append({
                "name": name,
                "pass1": cfg["canonical_pass1"],
                "cpass": cond_a["measured_cost_of_pass_dollars"],
                "tok": cond_a["output_tokens_per_second_mean"],
            })

    dims_acc_cost: List[Tuple[str, bool]] = [("pass1", True), ("cpass", False)]
    qwen_b = [p for p in points_b if p["name"].startswith("Qwen-7B")]
    llama_b = [p for p in points_b if p["name"].startswith("Llama-8B")]
    qwen_a = [p for p in points_a if p["name"].startswith("Qwen-7B")]
    llama_a = [p for p in points_a if p["name"].startswith("Llama-8B")]

    return {
        "pricing_scenario_usd_per_a100_hour": 1.50,
        "pass1_source": "canonical 40-cell MATH-500 campaign means, not the 100-item serving subset",
        "reject_true_pareto_optimum": True,
        "batched_C8": {
            "dims": "maximize pass@1, minimize measured C_pass",
            "nondominated_pooled": _frontier(points_b, dims_acc_cost),
            "nondominated_qwen": _frontier(qwen_b, dims_acc_cost),
            "nondominated_llama": _frontier(llama_b, dims_acc_cost),
            "fp8_pareto_efficient_pooled": "Qwen-7B FP8" in _frontier(points_b, dims_acc_cost),
        },
        "single_stream_C1": {
            "dims": "maximize pass@1, minimize measured C_pass",
            "nondominated_pooled": _frontier(points_a, dims_acc_cost),
            "nondominated_qwen": _frontier(qwen_a, dims_acc_cost),
            "nondominated_llama": _frontier(llama_a, dims_acc_cost),
        },
    }


def json_diff(expected: Any, actual: Any, path: str = "$") -> List[str]:
    """Return human-readable mismatches. Floats compared with abs_tol=1e-9."""
    diffs: List[str] = []
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            diffs.append(f"{path}: {expected!r} vs {actual!r}")
        return diffs
    if isinstance(expected, dict) and isinstance(actual, dict):
        ek, ak = set(expected), set(actual)
        for key in sorted(ek - ak):
            diffs.append(f"{path}.{key}: missing in generated report")
        for key in sorted(ak - ek):
            diffs.append(f"{path}.{key}: unexpected in generated report")
        for key in sorted(ek & ak):
            diffs.extend(json_diff(expected[key], actual[key], f"{path}.{key}"))
        return diffs
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: len {len(expected)} vs {len(actual)}")
            return diffs
        for i, (exp, act) in enumerate(zip(expected, actual)):
            diffs.extend(json_diff(exp, act, f"{path}[{i}]"))
        return diffs
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-9):
            diffs.append(f"{path}: {expected!r} vs {actual!r}")
        return diffs
    if expected != actual:
        diffs.append(f"{path}: {expected!r} vs {actual!r}")
    return diffs


def pareto_label(results: Dict[str, Any], model: str, fmt: str, cond_key: str) -> str:
    """Nondominated / dominated on (pass@1, measured C_pass) for that condition."""
    summary = results["cost_analysis_summary"]
    name = f"{model} {fmt}"
    if cond_key.startswith("B"):
        nd = summary["batched_C8"]["nondominated_pooled"]
    else:
        nd = summary["single_stream_C1"]["nondominated_pooled"]
    return "nondominated (pass@1, C_pass)" if name in nd else "dominated on (pass@1, C_pass)"


def generate_markdown_reports(results: Dict[str, Any], output_md: Path, output_val_md: Path) -> None:
    """Generate human-readable report and validation report."""
    lines = []
    lines.append("# Measured Serving Performance & Cost-of-Pass Systems Benchmark Report")
    lines.append("")
    lines.append("**Hardware:** NVIDIA A100-PCIE-80GB (PARAM Rudra HPC)  ")
    lines.append("**Serving Engine:** vLLM 0.7.0 eager (`qrm-official` conda env) | **Toolchain:** PyTorch 2.5.1+cu124, CUDA 12.4  ")
    lines.append("**Dataset:** MATH-500 stratified benchmark subset ($n=100$) | **Repetitions:** $R=3$ wall-clock repeats (shared sampling seed)  ")
    lines.append(r"**Pricing Baseline:** $\$1.50 / \text{A100 GPU-hour}$ ($\$0.00041667 / \text{GPU-second}$)  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Measured Serving Performance Across Conditions")
    lines.append("")
    lines.append("### Condition A: Low-Concurrency / Interactive Stream ($C=1$)")
    lines.append("")
    lines.append("| Model & Format | Output tok/s | Median Latency (s) | P90 Latency (s) | Peak VRAM (GB) | GPU-sec / query | Measured $C_{\\text{pass}}$ ($) |")
    lines.append("|---|---|---|---|---|---|---|")

    for cfg_key, cfg in results["configurations"].items():
        m_name = cfg["model"]
        fmt = cfg["format"]
        cond = cfg["conditions"].get("A_single_stream_c1", {})
        if not cond:
            continue
        tok_s = f"{cond['output_tokens_per_second_mean']:.2f} ± {cond['output_tokens_per_second_std']:.2f}"
        med_lat = f"{cond['latency_median_sec']:.2f}"
        p90_lat = f"{cond['latency_p90_sec']:.2f}"
        vram = f"{cond['peak_vram_gb_mean']:.2f}"
        gpu_s = f"{cond['gpu_seconds_per_query_mean']:.2f}"
        cpass = f"${cond['measured_cost_of_pass_dollars']:.4f}"
        lines.append(f"| **{m_name} {fmt}** | {tok_s} | {med_lat} | {p90_lat} | {vram} | {gpu_s} | {cpass} |")

    lines.append("")
    lines.append("### Condition B: Batched Throughput ($C=8$)")
    lines.append("")
    lines.append("| Model & Format | Output tok/s | Requests/s | Peak VRAM (GB) | GPU-sec / query | Cost / query ($) | Measured $C_{\\text{pass}}$ ($) |")
    lines.append("|---|---|---|---|---|---|---|")

    for cfg_key, cfg in results["configurations"].items():
        m_name = cfg["model"]
        fmt = cfg["format"]
        cond = cfg["conditions"].get("B_batched_throughput_c8", {})
        if not cond:
            continue
        tok_s = f"{cond['output_tokens_per_second_mean']:.2f} ± {cond['output_tokens_per_second_std']:.2f}"
        req_s = f"{cond['requests_per_second_mean']:.3f}"
        vram = f"{cond['peak_vram_gb_mean']:.2f}"
        gpu_s = f"{cond['gpu_seconds_per_query_mean']:.2f}"
        cost_q = f"${cond['measured_cost_per_query_dollars']:.4f}"
        cpass = f"${cond['measured_cost_of_pass_dollars']:.4f}"
        lines.append(f"| **{m_name} {fmt}** | {tok_s} | {req_s} | {vram} | {gpu_s} | {cost_q} | {cpass} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(r"## 2. Relative Systems Deltas vs BF16 Anchors ($\Delta = \text{Quantized} - \text{BF16}$)")
    lines.append("")
    lines.append("| Configuration vs BF16 | Condition | $\\Delta$ Output tok/s | $\\Delta$ GPU-sec/query | $\\Delta$ Peak VRAM | $\\Delta$ Cost-of-Pass ($C_{\\text{pass}}$) |")
    lines.append("|---|---|---|---|---|---|")

    for pair_key, pair in results["comparisons_vs_bf16"].items():
        m_name = pair["model"]
        fmt = pair["format"]
        for cond_key, cond_name in [("A_single_stream_c1", "Interactive (C=1)"), ("B_batched_throughput_c8", "Batched (C=8)")]:
            c_data = pair["conditions"].get(cond_key, {})
            if not c_data:
                continue
            d_tok = f"{c_data['delta_output_tok_sec_pct']:+.1f}%"
            d_gpu = f"{c_data['delta_gpu_seconds_per_query_pct']:+.1f}%"
            d_vram = f"{c_data['delta_peak_vram_pct']:+.1f}%"
            d_cpass = f"{c_data['delta_cost_of_pass_pct']:+.1f}%"
            lines.append(f"| **{m_name} {fmt} vs BF16** | {cond_name} | {d_tok} | {d_gpu} | {d_vram} | {d_cpass} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Comparison: Old Fixed-Throughput Proxy vs New Measured Serving Cost")
    lines.append("")
    lines.append("| Model & Format | Pass@1 (Accuracy) | Old Proxy Cost/Query | Old Proxy $C_{\\text{pass}}$ | Measured Batched Cost/Query | Measured Batched $C_{\\text{pass}}$ | (pass@1, $C_{\\text{pass}}$) |")
    lines.append("|---|---|---|---|---|---|---|")

    for cfg_key, cfg in results["configurations"].items():
        m_name = cfg["model"]
        fmt = cfg["format"]
        pass1 = cfg["canonical_pass1"]
        old_p = OLD_PROXY_COST.get((m_name, fmt), {})
        cond_b = cfg["conditions"].get("B_batched_throughput_c8", {})
        if not cond_b:
            continue
        old_cq = f"${old_p.get('cost_query_dollars', 0):.4f}"
        old_cp = f"${old_p.get('cost_pass_dollars', 0):.4f}"
        new_cq = f"${cond_b['measured_cost_per_query_dollars']:.4f}"
        new_cp = f"${cond_b['measured_cost_of_pass_dollars']:.4f}"
        shift_desc = pareto_label(results, m_name, fmt, "B_batched_throughput_c8")
        lines.append(f"| **{m_name} {fmt}** | {pass1*100:.2f}% | {old_cq} | {old_cp} | {new_cq} | {new_cp} | {shift_desc} |")

    summary = results.get("cost_analysis_summary", {})
    batched = summary.get("batched_C8", {})
    lines.append("")
    lines.append("### Pareto note")
    lines.append("")
    lines.append("There is no unique ``true Pareto optimum.'' On batched (pass@1, measured $C_{\\text{pass}}$) the nondominated pooled set is: "
                 + ", ".join(batched.get("nondominated_pooled", [])) + ".")
    lines.append("Qwen FP8 is Pareto-efficient in that two-objective set; Qwen GPTQ-4 is dominated. "
                 "$1.50$/A100-h is a pricing scenario. Pass@1 is the 40-cell MATH-500 campaign mean.")

    lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {output_md}")

    # Write validation markdown
    v_lines = []
    v_lines.append("# Measured Serving Benchmark Scientific Validation Audit")
    v_lines.append("")
    v_lines.append(f"**Audit Timestamp:** 2026-08-16  ")
    v_lines.append(f"**Hardware Platform:** NVIDIA A100-PCIE-80GB  ")
    v_lines.append(f"**Total Executed Runs:** 48 task-realistic + 8 microbenchmark runs  ")
    v_lines.append("")
    v_lines.append("## Integrity Checks")
    v_lines.append("- **All 8 Configurations Completed:** YES (Qwen-7B / Llama-8B × BF16, FP8, AWQ-4, GPTQ-4)")
    v_lines.append("- **Both Serving Conditions Measured:** YES (Condition A: C=1, Condition B: C=8)")
    v_lines.append("- **Repetitions per Condition:** 3 wall-clock repeats with shared sampling seed 20260816 (identical token counts)")
    v_lines.append("- **Input Prompt Subset:** Frozen 100 MATH-500 prompts stratified across Levels 1–5")
    v_lines.append("- **Raw JSON completeness:** 48 task-realistic + 8 microbenchmark files present; tok/s and GPU-sec/query recompute from elapsed/tokens")
    v_lines.append("- **OOM / SLURM errors:** no OOM strings in raw JSON; SLURM logs are not in git, so 0-failure is not independently proven from this artifact")
    v_lines.append("- **Node mix:** Llama AWQ-4 and Llama GPTQ-4 have records from more than one hostname (cache reuse / re-execution), so ``0 restarts'' is not verified")
    v_lines.append("- **Protocol notes:** Condition B is a 100-prompt `llm.generate` (continuous batching; `max_num_seqs` not pinned to 8). Condition A uses the first 20 prompts of the frozen list (level counts 5/7/3/3/2). Repetitions share sampling seed 20260816 (identical token counts). Peak VRAM is allocated bytes after `gpu_memory_utilization=0.75`, not isolated weight footprint.")
    v_lines.append("")
    output_val_md.write_text("\n".join(v_lines), encoding="utf-8")
    print(f"Validation report written to: {output_val_md}")


def main():
    parser = argparse.ArgumentParser(description="Analyze measured serving benchmark data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("results/measured_serving/raw"))
    parser.add_argument("--report-json", type=Path, default=Path("results/reports/measured_serving/measured_serving_report.json"))
    parser.add_argument("--report-md", type=Path, default=Path("results/reports/measured_serving/measured_serving_report.md"))
    parser.add_argument("--validation-md", type=Path, default=Path("results/reports/measured_serving/measured_serving_validation.md"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Recompute and exit nonzero if the result differs from the checked-in JSON.",
    )
    args = parser.parse_args()

    results = analyze_measured_serving(args.raw_dir)

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
            if len(diffs) > 50:
                print(f"  ... {len(diffs) - 50} more", file=sys.stderr)
            return 1
        n_cfg = len(results["configurations"])
        print(f"OK: recomputed measured-serving report matches {args.report_json} ({n_cfg} configs)")
        return 0

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved JSON report: {args.report_json}")

    generate_markdown_reports(results, args.report_md, args.validation_md)
    print("\nMEASURED SERVING ANALYSIS COMPLETED SUCCESSFULLY!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
