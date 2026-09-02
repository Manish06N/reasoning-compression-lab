#!/usr/bin/env python3
"""Modal-answer agreement analysis for MATH-500 reasoning completions (CPU-only).

This script implements gold-free modal-answer agreement analysis:
1. Extracts mathematical answers from generated completions using the frozen
   LightEval / math-verify normalization policy (docs/ANSWER_NORMALIZATION.md).
2. Verifies 100% reproduction of campaign gold correctness (20,000 / 20,000).
3. Clusters predictions into answer-equivalence classes without consulting gold.
4. Determines modal agreement (k/5) and unique modal predictions.
5. Computes risk-coverage-cost trade-offs across thresholds (>=3/5, >=4/5, 5/5).
6. Computes problem-level bootstrap confidence intervals (10,000 replicates).
7. Computes paired differences against BF16 anchors.
8. Incorporates 5x generation token cost (T5).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - MacBook stdlib --check-artifact path
    np = None  # type: ignore[assignment]

# Ensure external QRM and LightEval are in path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
QRM_DIR = REPO_ROOT / "external" / "Quantized-Reasoning-Models"
LIGHTEVAL_SRC = QRM_DIR / "third-party" / "lighteval" / "src"

if str(QRM_DIR) not in sys.path:
    sys.path.insert(0, str(QRM_DIR))
if str(LIGHTEVAL_SRC) not in sys.path:
    sys.path.insert(0, str(LIGHTEVAL_SRC))

try:
    from lighteval.metrics.utils.extractive_match_utils import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        extract_target_from_pred,
        get_extraction_regexes,
    )
    from lighteval.metrics.utils.math_comparison import compare_gold_target
    from lighteval.tasks.requests import Doc
    from lighteval.utils.language import Language
except ImportError as err:
    print(f"ERROR importing lighteval dependencies: {err}", file=sys.stderr)
    print("Please run using the qrm-official conda environment.", file=sys.stderr)


MODELS = ["DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Llama-8B"]
FORMATS = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]
SEEDS = [42, 43, 44, 45, 46]
NUM_PROBLEMS = 500
THRESHOLDS = [3, 4, 5]
EXPECTED_DERIVED_SHA256 = "23e9ead021111959cf047323572889c95be0496e9475d6870b06c8b2c9a6149b"
CONFIG_NAMES = [
    ("Qwen-7B", "BF16"),
    ("Qwen-7B", "FP8"),
    ("Qwen-7B", "AWQ-4"),
    ("Qwen-7B", "GPTQ-4"),
    ("Llama-8B", "BF16"),
    ("Llama-8B", "FP8"),
    ("Llama-8B", "AWQ-4"),
    ("Llama-8B", "GPTQ-4"),
]
WILSON_Z = 1.959963984540054


def wilson_interval(k: int, n: int, z: float = WILSON_Z) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / float(n)
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    spread = (z * math.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2))) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def clopper_pearson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Binomial CI on k/n.

    Exact Clopper–Pearson at 0 and n (the 0.00% risk cells). Interior points use
    Wilson so the frozen JSON stays stdlib-deterministic.
    """
    if n <= 0:
        return 0.0, 0.0
    if k <= 0:
        return 0.0, 1.0 - (alpha / 2.0) ** (1.0 / n)
    if k >= n:
        return (alpha / 2.0) ** (1.0 / n), 1.0
    return wilson_interval(k, n)


def attach_binomial_risk_intervals(th: Dict[str, Any]) -> None:
    """Wilson / Clopper–Pearson on selective risk = (served - correct) / served."""
    served = int(th["served_count"])
    correct = int(th["correct_served_count"])
    errors = max(0, served - correct)
    w_lo, w_hi = wilson_interval(errors, served)
    cp_lo, cp_hi = clopper_pearson_interval(errors, served)
    th["selective_risk_errors"] = errors
    th["selective_risk_wilson_ci_95"] = [w_lo, w_hi]
    th["selective_risk_clopper_pearson_ci_95"] = [cp_lo, cp_hi]
    th["zero_observed_not_zero_true"] = errors == 0 and served > 0


def attach_binomial_risk_intervals_report(report: Dict[str, Any]) -> None:
    for cfg in report.get("configurations", {}).values():
        for th in cfg.get("thresholds", {}).values():
            attach_binomial_risk_intervals(th)


def get_run_dir_name(model: str, weight_format: str, seed: int) -> str:
    """Map model, format, and seed to campaign directory name."""
    if weight_format == "BF16":
        base = model
    elif weight_format == "FP8":
        base = f"{model}-FP8"
    elif weight_format == "AWQ-4":
        base = f"{model}-AWQ-4"
    elif weight_format == "GPTQ-4":
        base = f"{model}-GPTQ-4"
    else:
        raise ValueError(f"Unknown format: {weight_format}")
    return f"{base}-seed{seed}"


def get_validation_filename(model: str, weight_format: str, seed: int) -> str:
    """Map model, format, seed to validation JSON filename."""
    if weight_format == "BF16":
        stem = model
    elif weight_format == "FP8":
        stem = f"{model}-FP8"
    elif weight_format == "AWQ-4":
        stem = f"{model}-AWQ-4"
    elif weight_format == "GPTQ-4":
        stem = f"{model}-GPTQ-4"
    else:
        raise ValueError(f"Unknown format: {weight_format}")
    return f"{stem}_math500_n500_seed{seed}.json"


def init_extractors() -> Tuple[Any, Any]:
    """Initialize LightEval regex extractors for math-verify."""
    gold_target = (LatexExtractionConfig(),)
    pred_target = (ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0))
    doc = Doc(query="", choices=[], gold_index=0, instruction="")
    gold_regexes = get_extraction_regexes(doc, gold_target, Language.ENGLISH)
    pred_regexes = get_extraction_regexes(doc, pred_target, Language.ENGLISH)
    return gold_regexes, pred_regexes


def extract_prediction(
    gen_text: str,
    pred_regexes: Any,
    fallback_mode: str = "first_match",
    extraction_mode: str = "any_match",
    timeout_seconds: int = 5,
) -> List[Any]:
    """Extract primary parsed prediction target from generated text."""
    try:
        raw_list = extract_target_from_pred(
            str(gen_text), pred_regexes, fallback_mode, extraction_mode, timeout_seconds
        )
        return raw_list[:1] if len(raw_list) > 0 else []
    except Exception:
        return []


def extract_gold(
    golds: List[str],
    gold_regexes: Any,
    fallback_mode: str = "first_match",
    extraction_mode: str = "any_match",
    timeout_seconds: int = 5,
) -> List[List[Any]]:
    """Extract parsed gold targets from reference answers."""
    extracted_golds = []
    for g in golds:
        try:
            extracted_golds.append(
                extract_target_from_pred(
                    str(g), gold_regexes, fallback_mode, extraction_mode, timeout_seconds
                )
            )
        except Exception:
            extracted_golds.append([])
    if any(len(g) == 0 for g in extracted_golds):
        extracted_golds = [[g] for g in golds]
    return extracted_golds


def are_answers_equivalent(
    p1: List[Any], p2: List[Any], precision: int = 5, timeout_seconds: int = 5
) -> bool:
    """Check mathematical equivalence between two extracted predictions without gold."""
    if len(p1) == 0 and len(p2) == 0:
        return True
    if len(p1) == 0 or len(p2) == 0:
        return False
    try:
        m1 = compare_gold_target(p1, p2, precision=precision, timeout_seconds=timeout_seconds)
        m2 = compare_gold_target(p2, p1, precision=precision, timeout_seconds=timeout_seconds)
        return bool(m1 and m2)
    except Exception:
        return False


def is_prediction_correct(
    pred: List[Any],
    extracted_golds: List[List[Any]],
    precision: int = 5,
    timeout_seconds: int = 5,
) -> float:
    """Check correctness of prediction against gold."""
    if len(pred) == 0:
        return 0.0
    try:
        match = any(
            compare_gold_target(g, pred, precision=precision, timeout_seconds=timeout_seconds)
            for g in extracted_golds
        )
        return 1.0 if match else 0.0
    except Exception:
        return 0.0


def run_full_modal_analysis(
    campaign_root: Path,
    validation_root: Path,
    derived_output: Path,
    bootstrap_reps: int = 10000,
    seed: int = 20260816,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute complete end-to-end extraction and modal agreement analysis."""
    gold_regexes, pred_regexes = init_extractors()
    
    print("=" * 80)
    print("1. EXTRACTING PREDICTIONS & REPRODUCING CAMPAIGN CORRECTNESS (20,000 rows)")
    print("=" * 80)
    
    all_rows: List[Dict[str, Any]] = []
    total_mismatches = 0
    extraction_failures = 0
    
    # Store token counts by (model, format, seed, problem_idx)
    token_counts: Dict[Tuple[str, str, int, int], int] = {}
    
    for model in MODELS:
        for fmt in FORMATS:
            for s in SEEDS:
                val_file = validation_root / get_validation_filename(model, fmt, s)
                if not val_file.exists():
                    raise FileNotFoundError(f"Missing validation file: {val_file}")
                val_data = json.loads(val_file.read_text(encoding="utf-8"))
                for det in val_data.get("details", []):
                    row_idx = det["row"] - 1  # 0-indexed
                    token_counts[(model, fmt, s, row_idx)] = det.get("completion_tokens", 0)

    for model in MODELS:
        model_family = "Qwen-7B" if "Qwen" in model else "Llama-8B"
        for fmt in FORMATS:
            for s in SEEDS:
                run_dir = campaign_root / get_run_dir_name(model, fmt, s)
                jsonl_file = run_dir / "MATH-500.jsonl"
                if not jsonl_file.exists():
                    raise FileNotFoundError(f"Missing campaign file: {jsonl_file}")
                
                raw_data = json.loads(jsonl_file.read_text(encoding="utf-8"))
                if len(raw_data) != NUM_PROBLEMS:
                    raise ValueError(f"{jsonl_file}: expected {NUM_PROBLEMS} rows, got {len(raw_data)}")
                
                for idx, item in enumerate(raw_data):
                    gen_text = item["generated_text"]
                    golds = item["gold"]
                    campaign_score = float(item["metrics"]["extractive_match"])
                    
                    pred_list = extract_prediction(gen_text, pred_regexes)
                    if len(pred_list) == 0:
                        extraction_failures += 1
                        
                    gold_list = extract_gold(golds, gold_regexes)
                    recomputed_score = is_prediction_correct(pred_list, gold_list)
                    
                    if recomputed_score != campaign_score:
                        total_mismatches += 1
                        print(f"MISMATCH at {model} {fmt} seed {s} problem {idx}: campaign={campaign_score}, recomputed={recomputed_score}")
                    
                    tok_count = token_counts.get((model, fmt, s, idx), 0)
                    
                    # Store compact representation
                    pred_repr = [str(p) for p in pred_list]
                    all_rows.append({
                        "benchmark": "MATH-500",
                        "problem_index": idx,
                        "model": model_family,
                        "format": fmt,
                        "seed": s,
                        "extracted_pred_repr": pred_repr,
                        "pred_list": pred_list,
                        "gold_list": gold_list,
                        "campaign_extractive_match": campaign_score,
                        "recomputed_match": recomputed_score,
                        "completion_tokens": tok_count,
                    })

    print(f"Extraction complete: total rows = {len(all_rows)}")
    print(f"Total campaign score mismatches = {total_mismatches} (required: 0)")
    print(f"Total extraction empty fallback rows = {extraction_failures}")
    
    if total_mismatches > 0:
        raise RuntimeError(f"FATAL: {total_mismatches} correctness mismatches detected! Halting.")

    # Write compact derived artifact
    derived_output.parent.mkdir(parents=True, exist_ok=True)
    with open(derived_output, "w", encoding="utf-8") as f:
        for r in all_rows:
            export_row = {
                "benchmark": r["benchmark"],
                "problem_index": r["problem_index"],
                "model": r["model"],
                "format": r["format"],
                "seed": r["seed"],
                "extracted_pred_repr": r["extracted_pred_repr"],
                "campaign_extractive_match": r["campaign_extractive_match"],
                "completion_tokens": r["completion_tokens"],
            }
            f.write(json.dumps(export_row) + "\n")
    print(f"Compact reproducibility artifact saved to: {derived_output}")

    print("\n" + "=" * 80)
    print("2. BUILDING ANSWER EQUIVALENCE CLASSES & MODAL CONSENSUS (4,000 groups)")
    print("=" * 80)

    # Group by (model, format, problem_index)
    groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in all_rows:
        key = (r["model"], r["format"], r["problem_index"])
        groups.setdefault(key, []).append(r)

    if len(groups) != 4000:
        raise ValueError(f"Expected 4000 groups, got {len(groups)}")

    group_results: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    symmetry_violations = 0
    transitivity_violations = 0
    total_ties = 0

    for key, group in groups.items():
        model_family, fmt, prob_idx = key
        if len(group) != 5:
            raise ValueError(f"Group {key} has {len(group)} items, expected 5")
        seeds_found = sorted([r["seed"] for r in group])
        if seeds_found != SEEDS:
            raise ValueError(f"Group {key} has seeds {seeds_found}, expected {SEEDS}")

        # Pairwise equivalence matrix
        preds = [r["pred_list"] for r in group]
        eq_matrix = np.zeros((5, 5), dtype=bool)
        for i in range(5):
            for j in range(5):
                if i == j:
                    eq_matrix[i, j] = True
                elif i < j:
                    eq_ij = are_answers_equivalent(preds[i], preds[j])
                    eq_ji = are_answers_equivalent(preds[j], preds[i])
                    if eq_ij != eq_ji:
                        symmetry_violations += 1
                    eq = eq_ij and eq_ji
                    eq_matrix[i, j] = eq
                    eq_matrix[j, i] = eq

        # Transitivity check
        for i in range(5):
            for j in range(5):
                for k_idx in range(5):
                    if eq_matrix[i, j] and eq_matrix[j, k_idx] and not eq_matrix[i, k_idx]:
                        transitivity_violations += 1

        # Connected component clustering
        visited = [False] * 5
        clusters: List[List[int]] = []
        for i in range(5):
            if not visited[i]:
                cluster = []
                queue = [i]
                visited[i] = True
                while queue:
                    curr = queue.pop(0)
                    cluster.append(curr)
                    for neighbor in range(5):
                        if eq_matrix[curr, neighbor] and not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                clusters.append(cluster)

        clusters.sort(key=lambda c: len(c), reverse=True)
        class_sizes = [len(c) for c in clusters]
        modal_count = class_sizes[0]
        
        # Unique mode check
        has_unique_mode = (len(class_sizes) == 1) or (class_sizes[0] > class_sizes[1])
        if not has_unique_mode:
            total_ties += 1

        # Check correctness of modal cluster (using predictions in the largest cluster)
        if has_unique_mode:
            lead_idx = clusters[0][0]
            lead_pred = group[lead_idx]["pred_list"]
            gold_list = group[lead_idx]["gold_list"]
            modal_correct = is_prediction_correct(lead_pred, gold_list)
            modal_repr = group[lead_idx]["extracted_pred_repr"]
        else:
            modal_correct = 0.0
            modal_repr = []

        # T5 token expenditure
        t5_tokens = sum(r["completion_tokens"] for r in group)

        group_results[key] = {
            "model": model_family,
            "format": fmt,
            "problem_index": prob_idx,
            "class_sizes": class_sizes,
            "modal_count": modal_count,
            "agreement_ratio": modal_count / 5.0,
            "unique_mode": has_unique_mode,
            "modal_correct": modal_correct,
            "modal_repr": modal_repr,
            "t5_tokens": t5_tokens,
        }

    print(f"Answer clustering complete across 4,000 groups.")
    print(f"Symmetry violations = {symmetry_violations}")
    print(f"Transitivity violations = {transitivity_violations}")
    print(f"Total ties (non-unique mode) = {total_ties}")

    if symmetry_violations > 0 or transitivity_violations > 0:
        raise RuntimeError(f"FATAL: Equivalence relations showed {symmetry_violations} symmetry and {transitivity_violations} transitivity violations!")

    print("\n" + "=" * 80)
    print("3. COMPUTING SELECTIVE PREDICTION & PROBLEM-LEVEL BOOTSTRAP (10,000 reps)")
    print("=" * 80)

    # Problem-level bootstrap resampling across all 8 configurations
    if np is None:
        raise RuntimeError("numpy is required for the full modal analysis path")

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, NUM_PROBLEMS, size=(bootstrap_reps, NUM_PROBLEMS))

    config_names = CONFIG_NAMES

    report_matrix: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []

    # Store per-config per-threshold metrics for paired analysis
    boot_metrics: Dict[Tuple[str, str, int], Dict[str, np.ndarray]] = {}

    for model_family, fmt in config_names:
        cfg_key = f"{model_family}_{fmt}"
        report_matrix[cfg_key] = {
            "model": model_family,
            "format": fmt,
            "thresholds": {},
            "token_economics": {},
            "secondary_calibration": {},
        }

        # Collect problem records (0..499)
        prob_records = [group_results[(model_family, fmt, p)] for p in range(NUM_PROBLEMS)]
        t5_arr = np.array([r["t5_tokens"] for r in prob_records], dtype=np.float64)

        report_matrix[cfg_key]["token_economics"] = {
            "mean_t5_output_tokens": float(np.mean(t5_arr)),
            "median_t5_output_tokens": float(np.median(t5_arr)),
            "p90_t5_output_tokens": float(np.percentile(t5_arr, 90)),
            "total_5sample_tokens": int(np.sum(t5_arr)),
        }

        # Secondary calibration
        confidences = np.array([r["modal_count"] / 5.0 for r in prob_records], dtype=np.float64)
        labels = np.array([r["modal_correct"] if r["unique_mode"] else 0.0 for r in prob_records], dtype=np.float64)
        brier = float(np.mean((confidences - labels) ** 2))
        
        # 10-bin ECE
        bin_edges = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        for b in range(10):
            in_bin = (confidences > bin_edges[b]) & (confidences <= bin_edges[b + 1]) if b > 0 else (confidences >= bin_edges[b]) & (confidences <= bin_edges[b + 1])
            if np.sum(in_bin) > 0:
                bin_acc = np.mean(labels[in_bin])
                bin_conf = np.mean(confidences[in_bin])
                ece += (np.sum(in_bin) / NUM_PROBLEMS) * abs(bin_acc - bin_conf)
        report_matrix[cfg_key]["secondary_calibration"] = {
            "brier_score": brier,
            "ece_10bin": float(ece),
        }

        for tau in THRESHOLDS:
            served_mask = np.array([
                (r["modal_count"] >= tau) and r["unique_mode"]
                for r in prob_records
            ], dtype=bool)
            correct_mask = np.array([
                r["modal_correct"] == 1.0 for r in prob_records
            ], dtype=bool)

            served_count = int(np.sum(served_mask))
            abstained_count = NUM_PROBLEMS - served_count
            correct_served = int(np.sum(served_mask & correct_mask))
            coverage = served_count / float(NUM_PROBLEMS)
            sel_acc = (correct_served / float(served_count)) if served_count > 0 else 1.0
            sel_risk = 1.0 - sel_acc

            # Token cost metrics
            tokens_per_served = (float(np.sum(t5_arr)) / served_count) if served_count > 0 else float("nan")
            tokens_per_correct_served = (float(np.sum(t5_arr)) / correct_served) if correct_served > 0 else float("nan")

            # Bootstrap distributions
            boot_cov = np.zeros(bootstrap_reps, dtype=np.float64)
            boot_acc = np.zeros(bootstrap_reps, dtype=np.float64)
            boot_risk = np.zeros(bootstrap_reps, dtype=np.float64)

            for b_idx in range(bootstrap_reps):
                idxs = boot_indices[b_idx]
                b_served = served_mask[idxs]
                b_correct = correct_mask[idxs]
                b_s_cnt = np.sum(b_served)
                b_c_cnt = np.sum(b_served & b_correct)
                b_cov = b_s_cnt / float(NUM_PROBLEMS)
                b_acc = (b_c_cnt / float(b_s_cnt)) if b_s_cnt > 0 else 1.0
                b_risk = 1.0 - b_acc
                boot_cov[b_idx] = b_cov
                boot_acc[b_idx] = b_acc
                boot_risk[b_idx] = b_risk

            boot_metrics[(model_family, fmt, tau)] = {
                "coverage": boot_cov,
                "accuracy": boot_acc,
                "risk": boot_risk,
            }

            cov_ci = [float(np.percentile(boot_cov, 2.5)), float(np.percentile(boot_cov, 97.5))]
            acc_ci = [float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))]
            risk_ci = [float(np.percentile(boot_risk, 2.5)), float(np.percentile(boot_risk, 97.5))]

            th_data = {
                "threshold": f">={tau}/5" if tau < 5 else "5/5",
                "tau": tau,
                "served_count": served_count,
                "abstained_count": abstained_count,
                "correct_served_count": correct_served,
                "coverage": coverage,
                "coverage_ci_95": cov_ci,
                "selective_accuracy": sel_acc,
                "selective_accuracy_ci_95": acc_ci,
                "selective_risk": sel_risk,
                "selective_risk_ci_95": risk_ci,
                "five_sample_token_cost_per_served": tokens_per_served,
                "five_sample_token_cost_per_correct_served": tokens_per_correct_served,
            }
            attach_binomial_risk_intervals(th_data)
            report_matrix[cfg_key]["thresholds"][f"tau_{tau}"] = th_data

            csv_rows.append({
                "model": model_family,
                "format": fmt,
                "threshold": th_data["threshold"],
                "served": served_count,
                "abstained": abstained_count,
                "coverage": f"{coverage * 100:.2f}%",
                "coverage_ci_95": f"[{cov_ci[0]*100:.2f}%, {cov_ci[1]*100:.2f}%]",
                "selective_accuracy": f"{sel_acc * 100:.2f}%",
                "selective_accuracy_ci_95": f"[{acc_ci[0]*100:.2f}%, {acc_ci[1]*100:.2f}%]",
                "selective_risk": f"{sel_risk * 100:.2f}%",
                "selective_risk_ci_95": f"[{risk_ci[0]*100:.2f}%, {risk_ci[1]*100:.2f}%]",
                "t5_tokens_mean": f"{report_matrix[cfg_key]['token_economics']['mean_t5_output_tokens']:.1f}",
                "tokens_per_correct_served": f"{tokens_per_correct_served:.1f}",
            })

    print("\n" + "=" * 80)
    print("4. COMPUTING PAIRED DIFFERENCES VS BF16 ANCHORS")
    print("=" * 80)

    paired_differences: Dict[str, Any] = {}
    for model_family in ["Qwen-7B", "Llama-8B"]:
        for fmt in ["FP8", "AWQ-4", "GPTQ-4"]:
            pair_key = f"{model_family}_{fmt}_vs_BF16"
            paired_differences[pair_key] = {"model": model_family, "format": fmt, "thresholds": {}}
            for tau in THRESHOLDS:
                base_cov = boot_metrics[(model_family, "BF16", tau)]["coverage"]
                base_acc = boot_metrics[(model_family, "BF16", tau)]["accuracy"]
                base_risk = boot_metrics[(model_family, "BF16", tau)]["risk"]

                tgt_cov = boot_metrics[(model_family, fmt, tau)]["coverage"]
                tgt_acc = boot_metrics[(model_family, fmt, tau)]["accuracy"]
                tgt_risk = boot_metrics[(model_family, fmt, tau)]["risk"]

                diff_cov = tgt_cov - base_cov
                diff_acc = tgt_acc - base_acc
                diff_risk = tgt_risk - base_risk

                th_tag = f">={tau}/5" if tau < 5 else "5/5"
                paired_differences[pair_key]["thresholds"][f"tau_{tau}"] = {
                    "threshold": th_tag,
                    "delta_coverage_mean": float(np.mean(diff_cov)),
                    "delta_coverage_ci_95": [float(np.percentile(diff_cov, 2.5)), float(np.percentile(diff_cov, 97.5))],
                    "delta_selective_accuracy_mean": float(np.mean(diff_acc)),
                    "delta_selective_accuracy_ci_95": [float(np.percentile(diff_acc, 2.5)), float(np.percentile(diff_acc, 97.5))],
                    "delta_selective_risk_mean": float(np.mean(diff_risk)),
                    "delta_selective_risk_ci_95": [float(np.percentile(diff_risk, 2.5)), float(np.percentile(diff_risk, 97.5))],
                }

    final_report = {
        "metadata": {
            "dataset": "MATH-500",
            "n_problems": NUM_PROBLEMS,
            "models": MODELS,
            "formats": FORMATS,
            "seeds": SEEDS,
            "total_records": len(all_rows),
            "bootstrap_replicates": bootstrap_reps,
            "analysis_seed": seed,
            "normalization_policy": "docs/ANSWER_NORMALIZATION.md (LightEval math-verify)",
            "lighteval_version": "0.8.0",
            "gold_free_modal_selection": True,
        },
        "validation_diagnostics": {
            "total_rows_evaluated": len(all_rows),
            "campaign_correctness_reproduced": f"{len(all_rows) - total_mismatches}/{len(all_rows)} (100.0%)",
            "mismatch_count": total_mismatches,
            "empty_extraction_fallbacks": extraction_failures,
            "total_groups": len(groups),
            "symmetry_violations": symmetry_violations,
            "transitivity_violations": transitivity_violations,
            "total_ties": total_ties,
        },
        "configurations": report_matrix,
        "paired_differences_vs_bf16": paired_differences,
    }

    return final_report, csv_rows


def generate_markdown_report(report: Dict[str, Any], output_md: Path) -> None:
    """Generate comprehensive human-readable Markdown summary report."""
    lines = []
    lines.append("# Gold-Free Modal-Answer Agreement & Selective Prediction Report")
    lines.append("")
    lines.append(f"**Dataset:** MATH-500 ($n=500$) | **Seeds:** 42, 43, 44, 45, 46 (5 seeds) | **Total Evaluated Completions:** 20,000  ")
    lines.append(f"**Normalization Policy:** `docs/ANSWER_NORMALIZATION.md` (LightEval math-verify) | **Bootstrap Replicates:** 10,000  ")
    lines.append(f"**Gold Role:** Gold answers were strictly **NOT** used for modal selection, clustering, or abstention. Gold was used solely to score final served predictions.  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Coverage, selective risk, and five-sample token proxy across thresholds")
    lines.append("")
    lines.append("| Model & Format | Threshold | Served / 500 | Coverage (95% CI) | Selective Acc (95% CI) | Selective Risk (95% CI) | Mean $T_5$ Tokens | Tokens / Correct Served |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for cfg_key, cfg in report["configurations"].items():
        m_name = cfg["model"]
        fmt = cfg["format"]
        t5_mean = cfg["token_economics"]["mean_t5_output_tokens"]
        for tau_key in ["tau_3", "tau_4", "tau_5"]:
            th = cfg["thresholds"][tau_key]
            th_lbl = th["threshold"]
            srv = th["served_count"]
            cov = th["coverage"] * 100.0
            cov_ci = th["coverage_ci_95"]
            acc = th["selective_accuracy"] * 100.0
            acc_ci = th["selective_accuracy_ci_95"]
            risk = th["selective_risk"] * 100.0
            risk_ci = th["selective_risk_ci_95"]
            w_ci = th.get("selective_risk_wilson_ci_95", risk_ci)
            tok_per_corr = th["five_sample_token_cost_per_correct_served"]

            cov_str = f"{cov:.1f}% [{cov_ci[0]*100:.1f}%, {cov_ci[1]*100:.1f}%]"
            acc_str = f"{acc:.2f}% [{acc_ci[0]*100:.2f}%, {acc_ci[1]*100:.2f}%]"
            risk_str = (
                f"{risk:.2f}% boot[{risk_ci[0]*100:.2f}%, {risk_ci[1]*100:.2f}%] "
                f"Wilson[{w_ci[0]*100:.2f}%, {w_ci[1]*100:.2f}%]"
            )

            lines.append(f"| **{m_name} {fmt}** | {th_lbl} | {srv} | {cov_str} | {acc_str} | {risk_str} | {t5_mean:.0f} | {tok_per_corr:.0f} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(r"## 2. Paired Differences Against BF16 Anchors ($\Delta = \text{Quantized} - \text{BF16}$)")
    lines.append("")
    lines.append("| Comparison | Threshold | $\\Delta$ Coverage (95% CI) | $\\Delta$ Selective Accuracy (95% CI) | $\\Delta$ Selective Risk (95% CI) |")
    lines.append("|---|---|---|---|---|")

    for pair_key, pair in report["paired_differences_vs_bf16"].items():
        m_name = pair["model"]
        fmt = pair["format"]
        for tau_key in ["tau_3", "tau_4", "tau_5"]:
            th = pair["thresholds"][tau_key]
            th_lbl = th["threshold"]
            d_cov = th["delta_coverage_mean"] * 100.0
            d_cov_ci = th["delta_coverage_ci_95"]
            d_acc = th["delta_selective_accuracy_mean"] * 100.0
            d_acc_ci = th["delta_selective_accuracy_ci_95"]
            d_risk = th["delta_selective_risk_mean"] * 100.0
            d_risk_ci = th["delta_selective_risk_ci_95"]

            d_cov_str = f"{d_cov:+.1f}% [{d_cov_ci[0]*100:+.1f}%, {d_cov_ci[1]*100:+.1f}%]"
            d_acc_str = f"{d_acc:+.2f}% [{d_acc_ci[0]*100:+.2f}%, {d_acc_ci[1]*100:+.2f}%]"
            d_risk_str = f"{d_risk:+.2f}% [{d_risk_ci[0]*100:+.2f}%, {d_risk_ci[1]*100:+.2f}%]"

            lines.append(f"| **{m_name} {fmt} vs BF16** | {th_lbl} | {d_cov_str} | {d_acc_str} | {d_risk_str} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Secondary Calibration & Uncertainty Metrics (Appendix)")
    lines.append("")
    lines.append("| Model & Format | ECE (10-bin) | Brier Score | Notes |")
    lines.append("|---|---|---|---|")
    for cfg_key, cfg in report["configurations"].items():
        m_name = cfg["model"]
        fmt = cfg["format"]
        ece = cfg["secondary_calibration"]["ece_10bin"]
        brier = cfg["secondary_calibration"]["brier_score"]
        lines.append(f"| {m_name} {fmt} | {ece:.4f} | {brier:.4f} | Confidence = Modal Frequency ($k/5$) |")

    lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Human-readable Markdown report written to: {output_md}")


def generate_validation_md(report: Dict[str, Any], output_val_md: Path) -> None:
    """Generate scientific validation report."""
    diag = report["validation_diagnostics"]
    meta = report["metadata"]
    lines = []
    lines.append("# Modal Agreement Scientific Validation Report")
    lines.append("")
    lines.append(f"**Validation Date:** 2026-08-16  ")
    lines.append(f"**Target Dataset:** {meta['dataset']} ($n={meta['n_problems']}$)  ")
    lines.append(f"**Total Records:** {meta['total_records']} completions across {diag['total_groups']} groups  ")
    lines.append("")
    lines.append("## Integrity & Reproduction Audit")
    lines.append("")
    lines.append(f"- **20,000 Rows Confirmed:** YES ({diag['total_rows_evaluated']} evaluated)")
    lines.append(f"- **4,000 Groups Confirmed:** YES ({diag['total_groups']} groups of 5 seeds)")
    lines.append(f"- **Seeds Present per Group:** Exactly seeds 42, 43, 44, 45, 46 (0 missing seeds)")
    lines.append(f"- **Duplicates Detected:** 0")
    lines.append(f"- **Campaign Correctness Reproduction:** {diag['campaign_correctness_reproduced']}")
    lines.append(f"- **Mismatch Count:** {diag['mismatch_count']}")
    lines.append(f"- **Empty Extraction Fallbacks:** {diag['empty_extraction_fallbacks']}")
    lines.append(f"- **Equivalence Symmetry Violations:** {diag['symmetry_violations']}")
    lines.append(f"- **Equivalence Transitivity Violations:** {diag['transitivity_violations']}")
    lines.append(f"- **Tie Count (Non-Unique Mode):** {diag['total_ties']}")
    lines.append(f"- **Gold Used Before Modal Selection:** **NO** (Gold strictly insulated until post-clustering scoring)")
    lines.append("")
    output_val_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Validation report written to: {output_val_md}")


def campaign_jsonl_available(campaign_root: Path) -> bool:
    sample = campaign_root / get_run_dir_name(MODELS[0], FORMATS[0], SEEDS[0]) / "MATH-500.jsonl"
    return sample.exists()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_derived_artifact(derived_path: Path, report_path: Path) -> None:
    """Stdlib check of the committed compact artifact against the frozen JSON report.

    This does *not* re-extract from campaign traces and does *not* re-cluster with a
    second judge. Full LightEval clustering remains the HPC ``--check`` path.
    """
    if not derived_path.exists():
        raise FileNotFoundError(f"Missing compact artifact: {derived_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report JSON: {report_path}")

    digest = sha256_file(derived_path)
    if digest != EXPECTED_DERIVED_SHA256:
        raise AssertionError(
            f"Compact artifact SHA256 mismatch: got {digest}, expected {EXPECTED_DERIVED_SHA256}"
        )

    rows: List[Dict[str, Any]] = []
    forbidden_keys = {
        "generated_text",
        "gold",
        "gold_list",
        "pred_list",
        "question",
        "problem",
        "prompt",
    }
    seen_keys = set()
    for line in derived_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        overlap = forbidden_keys.intersection(row)
        if overlap:
            raise AssertionError(f"Compact artifact contains forbidden keys: {sorted(overlap)}")
        rows.append(row)
        seen_keys.update(row.keys())

    if len(rows) != 20_000:
        raise AssertionError(f"Expected 20,000 rows, got {len(rows)}")

    expected_keys = {
        "benchmark",
        "problem_index",
        "model",
        "format",
        "seed",
        "extracted_pred_repr",
        "campaign_extractive_match",
        "completion_tokens",
    }
    if seen_keys != expected_keys:
        raise AssertionError(f"Unexpected compact keys: {sorted(seen_keys)}")

    groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    pair_keys = set()
    missing_pred = 0
    for row in rows:
        if row["benchmark"] != "MATH-500":
            raise AssertionError(f"Unexpected benchmark: {row['benchmark']}")
        key = (row["model"], row["format"], int(row["problem_index"]))
        groups.setdefault(key, []).append(row)
        pair = (row["model"], row["format"], int(row["problem_index"]), int(row["seed"]))
        if pair in pair_keys:
            raise AssertionError(f"Duplicate row: {pair}")
        pair_keys.add(pair)
        pred = row["extracted_pred_repr"]
        if not isinstance(pred, list) or len(pred) == 0:
            missing_pred += 1

    if len(groups) != 4_000:
        raise AssertionError(f"Expected 4,000 groups, got {len(groups)}")
    if missing_pred != 0:
        raise AssertionError(f"Missing extracted predictions: {missing_pred}")

    t5_sums: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for key, group in groups.items():
        if len(group) != 5:
            raise AssertionError(f"Group {key} has {len(group)} rows")
        seeds_found = sorted(int(r["seed"]) for r in group)
        if seeds_found != SEEDS:
            raise AssertionError(f"Group {key} seeds {seeds_found}")
        t5_sums[(key[0], key[1])].append(sum(int(r["completion_tokens"]) for r in group))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    diag = report["validation_diagnostics"]
    assert diag["total_rows_evaluated"] == 20_000
    assert diag["total_groups"] == 4_000
    assert diag["mismatch_count"] == 0
    assert diag["symmetry_violations"] == 0
    assert diag["transitivity_violations"] == 0
    assert diag["total_ties"] == 116
    assert diag["empty_extraction_fallbacks"] == 0
    assert report["metadata"]["gold_free_modal_selection"] is True
    assert report["metadata"].get("lighteval_version") == "0.8.0"

    for model_family, fmt in CONFIG_NAMES:
        cfg = report["configurations"][f"{model_family}_{fmt}"]
        t5_arr = t5_sums[(model_family, fmt)]
        if len(t5_arr) != NUM_PROBLEMS:
            raise AssertionError(f"{model_family} {fmt}: expected 500 T5 sums")
        mean_t5 = sum(t5_arr) / float(NUM_PROBLEMS)
        total_t5 = float(sum(t5_arr))
        if not math.isclose(mean_t5, cfg["token_economics"]["mean_t5_output_tokens"], rel_tol=0, abs_tol=1e-6):
            raise AssertionError(f"{model_family} {fmt}: T5 mean drift")
        if int(total_t5) != int(cfg["token_economics"]["total_5sample_tokens"]):
            raise AssertionError(f"{model_family} {fmt}: T5 total drift")
        for tau in THRESHOLDS:
            th = cfg["thresholds"][f"tau_{tau}"]
            served = th["served_count"]
            correct = th["correct_served_count"]
            if not math.isclose(th["coverage"], served / float(NUM_PROBLEMS), abs_tol=1e-12):
                raise AssertionError(f"{model_family} {fmt} tau={tau}: coverage != served/500")
            if served > 0:
                acc = correct / float(served)
                if not math.isclose(th["selective_accuracy"], acc, abs_tol=1e-12):
                    raise AssertionError(f"{model_family} {fmt} tau={tau}: accuracy drift")
                if not math.isclose(th["selective_risk"], 1.0 - acc, abs_tol=1e-12):
                    raise AssertionError(f"{model_family} {fmt} tau={tau}: risk drift")
                paid = total_t5 / float(served)
                if not math.isclose(th["five_sample_token_cost_per_served"], paid, rel_tol=0, abs_tol=1e-6):
                    raise AssertionError(
                        f"{model_family} {fmt} tau={tau}: tokens/served must use all {NUM_PROBLEMS} T5 sums"
                    )
            errors = served - correct
            w_lo, w_hi = wilson_interval(errors, served)
            if "selective_risk_wilson_ci_95" not in th:
                raise AssertionError(f"{model_family} {fmt} tau={tau}: missing Wilson risk interval")
            got = th["selective_risk_wilson_ci_95"]
            if not math.isclose(got[0], w_lo, abs_tol=1e-12) or not math.isclose(got[1], w_hi, abs_tol=1e-12):
                raise AssertionError(f"{model_family} {fmt} tau={tau}: Wilson risk CI drift")
            cp_lo, cp_hi = clopper_pearson_interval(errors, served)
            got_cp = th["selective_risk_clopper_pearson_ci_95"]
            if not math.isclose(got_cp[0], cp_lo, abs_tol=1e-9) or not math.isclose(got_cp[1], cp_hi, abs_tol=1e-9):
                raise AssertionError(f"{model_family} {fmt} tau={tau}: Clopper-Pearson risk CI drift")
            if errors == 0 and not th.get("zero_observed_not_zero_true"):
                raise AssertionError(f"{model_family} {fmt} tau={tau}: 0/n must flag zero_observed_not_zero_true")

    print("OK: compact artifact SHA256, 20,000/4,000 structure, T5 accounting, and report internals match.")
    print("Note: this path does not re-run LightEval extraction or answer clustering.")


def main():
    parser = argparse.ArgumentParser(description="Modal-answer agreement reanalysis on MATH-500.")
    parser.add_argument("--campaign-root", type=Path, default=REPO_ROOT / "outputs-hpc-campaign-2026-08-14" / "inference")
    parser.add_argument("--validation-root", type=Path, default=REPO_ROOT / "outputs-hpc-campaign-2026-08-14" / "validation")
    parser.add_argument("--derived-output", type=Path, default=REPO_ROOT / "results" / "recovered" / "math500_modal_inputs.jsonl")
    parser.add_argument("--report-json", type=Path, default=REPO_ROOT / "results" / "reports" / "modal_agreement_report.json")
    parser.add_argument("--report-md", type=Path, default=REPO_ROOT / "results" / "reports" / "modal_agreement_report.md")
    parser.add_argument("--table-csv", type=Path, default=REPO_ROOT / "results" / "reports" / "modal_agreement_table.csv")
    parser.add_argument("--validation-md", type=Path, default=REPO_ROOT / "results" / "reports" / "modal_agreement_validation.md")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--check", action="store_true", help="Verify recomputed results against existing report JSON.")
    parser.add_argument(
        "--attach-intervals",
        action="store_true",
        help="Add Wilson/Clopper–Pearson risk intervals to the existing report JSON (no campaign traces).",
    )
    parser.add_argument(
        "--check-artifact",
        action="store_true",
        help="Stdlib check of the committed compact JSONL + report internals (no campaign traces).",
    )
    args = parser.parse_args()

    if args.attach_intervals:
        if not args.report_json.exists():
            print(f"ERROR: {args.report_json} does not exist.", file=sys.stderr)
            sys.exit(1)
        report = json.loads(args.report_json.read_text(encoding="utf-8"))
        attach_binomial_risk_intervals_report(report)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Attached Wilson/Clopper–Pearson risk intervals: {args.report_json}")
        sys.exit(0)

    if args.check_artifact or (args.check and not campaign_jsonl_available(args.campaign_root)):
        if args.check and not args.check_artifact:
            print(
                "Campaign JSONLs are not present; running compact-artifact --check "
                "(full LightEval re-extraction requires outputs-hpc-campaign-2026-08-14)."
            )
        check_derived_artifact(args.derived_output, args.report_json)
        sys.exit(0)

    if args.check:
        if not args.report_json.exists():
            print(f"ERROR: Cannot --check because {args.report_json} does not exist.", file=sys.stderr)
            sys.exit(1)
        print(f"Running in --check mode against {args.report_json}...")
        current_report = json.loads(args.report_json.read_text(encoding="utf-8"))
        new_report, _ = run_full_modal_analysis(
            args.campaign_root,
            args.validation_root,
            args.derived_output,
            bootstrap_reps=args.bootstrap_reps,
            seed=args.seed,
        )
        # Compare key metrics
        for cfg_key in current_report["configurations"]:
            curr_cfg = current_report["configurations"][cfg_key]
            new_cfg = new_report["configurations"][cfg_key]
            for tau in ["tau_3", "tau_4", "tau_5"]:
                curr_th = curr_cfg["thresholds"][tau]
                new_th = new_cfg["thresholds"][tau]
                assert curr_th["served_count"] == new_th["served_count"], f"Mismatch in served_count for {cfg_key} {tau}"
                assert curr_th["correct_served_count"] == new_th["correct_served_count"], f"Mismatch in correct_served_count for {cfg_key} {tau}"
                assert math.isclose(curr_th["coverage"], new_th["coverage"], abs_tol=1e-5), f"Mismatch in coverage for {cfg_key} {tau}"
                assert math.isclose(curr_th["selective_risk"], new_th["selective_risk"], abs_tol=1e-5), f"Mismatch in selective_risk for {cfg_key} {tau}"
        print("OK: Modal agreement reanalysis verified successfully against canonical report!")
        sys.exit(0)

    report, csv_rows = run_full_modal_analysis(
        args.campaign_root,
        args.validation_root,
        args.derived_output,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved canonical JSON report: {args.report_json}")

    generate_markdown_report(report, args.report_md)
    generate_validation_md(report, args.validation_md)

    if csv_rows:
        with open(args.table_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved CSV table: {args.table_csv}")

    print("\nALL MODAL AGREEMENT ANALYSES SUCCESSFULLY COMPLETED!")


if __name__ == "__main__":
    main()
