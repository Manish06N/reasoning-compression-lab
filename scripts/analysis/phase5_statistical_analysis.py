#!/usr/bin/env python3
"""
Phase 5 Statistical Analysis & Calibration Engine
Analyzes all 40 completed validation cells on MATH-500 across 5 seeds (42, 43, 44, 45, 46),
4 quantization formats (BF16, FP8, AWQ-4, GPTQ-4), and 2 architectures (Qwen-7B, Llama-8B).

Computes:
1. Mean ± Std Pass@1 and Wilson 95% Confidence Intervals
2. Problem-level Paired McNemar Tests with Holm-Bonferroni correction
3. maj@5 Sample-Consistency Calibration (ECE, Brier Score, AURC)
4. Token length distribution and Truncation / Repetition pathology verification
5. Deployment Economics & Cost-of-Pass (C_pass) under explicit A100 GPU pricing ($1.50/GPU-hr)
"""

import json
import glob
import os
import math
from collections import defaultdict

def wilson_score_interval(k, n, confidence=0.95):
    """Calculates Wilson score interval for binomial proportions."""
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95% confidence
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - spread), min(1.0, center + spread)

def n_choose_k(n, k):
    """Computes combinations n choose k in pure Python."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c

def mcnemar_exact_p_value(b, c):
    """Exact binomial two-tailed p-value for paired discordance (b = n10, c = n01)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Sum binomial(n, i) * 0.5^n for i in 0..k, multiplied by 2
    cum_prob = 0.0
    for i in range(k + 1):
        cum_prob += n_choose_k(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * cum_prob)

def compute_ece(confidences, accuracies, num_bins=10):
    """Computes Expected Calibration Error over equal-width bins."""
    bin_size = 1.0 / num_bins
    ece = 0.0
    total_samples = len(confidences)
    if total_samples == 0:
        return 0.0

    for i in range(num_bins):
        bin_lower = i * bin_size
        bin_upper = (i + 1) * bin_size
        # Collect items in bin
        bin_indices = [
            j for j in range(total_samples)
            if (bin_lower <= confidences[j] < bin_upper) or (i == num_bins - 1 and confidences[j] == 1.0)
        ]
        if len(bin_indices) > 0:
            bin_conf = sum(confidences[j] for j in bin_indices) / len(bin_indices)
            bin_acc = sum(accuracies[j] for j in bin_indices) / len(bin_indices)
            ece += (len(bin_indices) / total_samples) * abs(bin_acc - bin_conf)
    return ece

def compute_brier_score(confidences, labels):
    """Computes Brier Score: Mean squared difference between confidence and binary correctness."""
    if not confidences:
        return 0.0
    return sum((c - y)**2 for c, y in zip(confidences, labels)) / len(confidences)

def compute_aurc(confidences, errors):
    """Computes Area Under Risk-Coverage curve (AURC)."""
    n = len(confidences)
    if n == 0:
        return 0.0
    # Sort pairs by confidence descending
    sorted_pairs = sorted(zip(confidences, errors), key=lambda x: x[0], reverse=True)
    
    cumulative_errors = 0
    risk_coverage_points = []
    for k in range(1, n + 1):
        cumulative_errors += sorted_pairs[k - 1][1]
        risk = cumulative_errors / k
        coverage = k / n
        risk_coverage_points.append((coverage, risk))
    
    # Integrate risk over coverage using trapezoidal rule
    aurc = 0.0
    for i in range(1, len(risk_coverage_points)):
        cov_prev, risk_prev = risk_coverage_points[i - 1]
        cov_curr, risk_curr = risk_coverage_points[i]
        aurc += 0.5 * (risk_prev + risk_curr) * (cov_curr - cov_prev)
    return aurc

def main():
    validation_dir = "results/math500"
    if not glob.glob(os.path.join(validation_dir, "*.json")):
        validation_dir = "outputs-hpc-campaign-2026-08-14/validation"
    files = sorted(glob.glob(os.path.join(validation_dir, "*.json")))
    print(f"Loaded {len(files)} validation files from {validation_dir}")

    # Hierarchy: data[model][format][seed] = validation_data
    campaign_data = defaultdict(lambda: defaultdict(dict))

    for f in files:
        with open(f, "r") as fp:
            d = json.load(fp)
        
        bn = os.path.basename(f).replace(".json", "")
        # Parse model, format, seed from filename
        # Examples:
        # DeepSeek-R1-Distill-Qwen-7B_math500_n500_seed42.json -> Qwen-7B, BF16, 42
        # DeepSeek-R1-Distill-Qwen-7B-FP8_math500_n500_seed42.json -> Qwen-7B, FP8, 42
        # DeepSeek-R1-Distill-Llama-8B-AWQ-4_math500_n500_seed42.json -> Llama-8B, AWQ-4, 42
        # DeepSeek-R1-Distill-Llama-8B-GPTQ-4_math500_n500_seed42.json -> Llama-8B, GPTQ-4, 42
        
        parts = bn.split("_math500_n500_seed")
        model_part = parts[0]
        seed = int(parts[1])

        if "Qwen-7B" in model_part:
            model = "Qwen-7B"
        elif "Llama-8B" in model_part:
            model = "Llama-8B"
        else:
            continue

        if "-FP8" in model_part:
            fmt = "FP8"
        elif "-AWQ-4" in model_part:
            fmt = "AWQ-4"
        elif "-GPTQ-4" in model_part:
            fmt = "GPTQ-4"
        else:
            fmt = "BF16"

        campaign_data[model][fmt][seed] = d

    seeds = [42, 43, 44, 45, 46]
    models = ["Qwen-7B", "Llama-8B"]
    formats = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]

    summary_stats = {}

    print("\n" + "="*100)
    print("PHASE 5 HEADLINE CONFIRMATORY GRID SUMMARY (MATH-500, n=500, 5 SEEDS)")
    print("="*100)
    print(f"{'Model & Format':<20} | {'S42':<6} {'S43':<6} {'S44':<6} {'S45':<6} {'S46':<6} | {'Mean ± Std':<16} | {'95% Wilson CI':<16} | {'Mean Tok':<9} | {'Trunc':<5} {'Loops':<5}")
    print("-"*100)

    for m in models:
        for fmt in formats:
            cell_seeds = campaign_data[m][fmt]
            accs = [cell_seeds[s]["accuracy"] * 100 for s in seeds]
            correct_counts = [cell_seeds[s]["correct"] for s in seeds]
            mean_acc = sum(accs) / len(accs)
            variance = sum((a - mean_acc)**2 for a in accs) / (len(accs) - 1)
            std_acc = math.sqrt(variance)

            # Pooled Wilson CI across 2500 trials
            pooled_correct = sum(correct_counts)
            w_low, w_high = wilson_score_interval(pooled_correct, 2500)

            # Token length
            mean_tokens = sum(cell_seeds[s]["completion_tokens_mean"] for s in seeds) / len(seeds)
            trunc_count = sum(
                cell_seeds[s].get("token_limit_hits", cell_seeds[s].get("truncation_count", cell_seeds[s].get("hit_token_limit_count", 0)))
                for s in seeds
            )
            loop_count = sum(
                cell_seeds[s].get("repetition_rows", cell_seeds[s].get("repetition_flag_count", 0))
                for s in seeds
            )

            key = f"{m}_{fmt}"
            summary_stats[key] = {
                "model": m,
                "format": fmt,
                "seed_accs": {s: accs[i] for i, s in enumerate(seeds)},
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "pooled_correct": pooled_correct,
                "wilson_ci_95": (w_low * 100, w_high * 100),
                "mean_tokens": mean_tokens,
                "truncations": trunc_count,
                "loops": loop_count
            }

            acc_str = " ".join([f"{a:5.1f}%" for a in accs])
            print(f"{m + ' ' + fmt:<20} | {acc_str} | {mean_acc:6.2f}% ± {std_acc:4.2f}% | [{w_low*100:5.1f}%, {w_high*100:5.1f}%] | {mean_tokens:8.1f}  | {trunc_count:<5} {loop_count:<5}")
        print("-"*100)

    # 2. Problem-level Paired Comparisons (BF16 vs FP8, AWQ-4, GPTQ-4)
    print("\n" + "="*100)
    print("PAIRED PROBLEM-LEVEL STATISTICAL CONTRASTS (VS BF16 BASELINE, n=500, 5-SEED VOTING)")
    print("="*100)
    print(f"{'Model & Contrast':<28} | {'Both OK (n11)':<14} | {'BF16 Only (n10)':<16} | {'Quant Only (n01)':<16} | {'Both Fail (n00)':<15} | {'Exact McNemar p':<16} | {'Holm-Bonferroni Sig.'}")
    print("-"*100)

    mcnemar_results = []

    for m in models:
        bf16_cells = campaign_data[m]["BF16"]
        # Build problem-level correctness vectors for BF16: problem_idx -> count of correct seeds (0..5)
        bf16_prob_counts = [0] * 500
        for s in seeds:
            details = bf16_cells[s]["details"]
            for row_idx, r in enumerate(details):
                if r.get("extractive_match", 0.0) == 1.0:
                    bf16_prob_counts[row_idx] += 1

        for fmt in ["FP8", "AWQ-4", "GPTQ-4"]:
            fmt_cells = campaign_data[m][fmt]
            fmt_prob_counts = [0] * 500
            for s in seeds:
                details = fmt_cells[s]["details"]
                for row_idx, r in enumerate(details):
                    if r.get("extractive_match", 0.0) == 1.0:
                        fmt_prob_counts[row_idx] += 1

            # Paired problem contingency using majority-vote threshold (>= 3 seeds correct)
            n11 = n10 = n01 = n00 = 0
            for row_idx in range(500):
                bf16_maj = (bf16_prob_counts[row_idx] >= 3)
                fmt_maj = (fmt_prob_counts[row_idx] >= 3)
                if bf16_maj and fmt_maj:
                    n11 += 1
                elif bf16_maj and not fmt_maj:
                    n10 += 1
                elif not bf16_maj and fmt_maj:
                    n01 += 1
                else:
                    n00 += 1

            p_val = mcnemar_exact_p_value(n10, n01)
            contrast_name = f"{m} (BF16 vs {fmt})"
            mcnemar_results.append({
                "contrast": contrast_name,
                "model": m,
                "format": fmt,
                "n11": n11,
                "n10": n10,
                "n01": n01,
                "n00": n00,
                "p_value": p_val
            })

    # Sort by p-value for Holm-Bonferroni correction
    mcnemar_results.sort(key=lambda x: x["p_value"])
    num_tests = len(mcnemar_results)

    for rank, res in enumerate(mcnemar_results):
        alpha_adjusted = 0.05 / (num_tests - rank)
        is_significant = res["p_value"] < alpha_adjusted
        sig_str = "SIGNIFICANT (*)" if is_significant else "Not Significant (Parity)"
        res["holm_alpha"] = alpha_adjusted
        res["is_significant"] = is_significant
        print(f"{res['contrast']:<28} | {res['n11']:<14} | {res['n10']:<16} | {res['n01']:<16} | {res['n00']:<15} | p = {res['p_value']:<12.4e} | {sig_str} (alpha={alpha_adjusted:.4f})")

    print("-"*100)

    # 3. Sample-Consistency Calibration & maj@5 Reliability
    print("\n" + "="*100)
    print("SAMPLE-CONSISTENCY CALIBRATION & maj@5 METROLOGY (n=500 PROBLEMS, 5 COMPLETIONS/PROMPT)")
    print("="*100)
    print(f"{'Model & Format':<20} | {'maj@5 Accuracy':<15} | {'Expected Calib. Error (ECE)':<28} | {'Brier Score':<14} | {'AURC (Risk-Coverage)':<20}")
    print("-"*100)

    calibration_results = {}

    for m in models:
        for fmt in formats:
            cell_seeds = campaign_data[m][fmt]
            confidences = []
            maj_labels = []
            errors = []

            for row_idx in range(500):
                # Count correct seeds for this problem
                c_correct = 0
                for s in seeds:
                    if cell_seeds[s]["details"][row_idx].get("extractive_match", 0.0) == 1.0:
                        c_correct += 1
                
                # Sample consistency confidence = fraction of seeds in agreement
                # For binary pass/fail under greedy/sampling:
                conf = c_correct / 5.0
                maj_correct = 1.0 if c_correct >= 3 else 0.0
                confidences.append(conf)
                maj_labels.append(maj_correct)
                errors.append(1.0 - maj_correct)

            maj_acc = (sum(maj_labels) / 500.0) * 100
            ece = compute_ece(confidences, maj_labels, num_bins=10)
            brier = compute_brier_score(confidences, maj_labels)
            aurc = compute_aurc(confidences, errors)

            key = f"{m}_{fmt}"
            calibration_results[key] = {
                "maj5_accuracy": maj_acc,
                "ece": ece,
                "brier_score": brier,
                "aurc": aurc
            }

            print(f"{m + ' ' + fmt:<20} | {maj_acc:6.2f}%         | {ece:6.4f}                       | {brier:6.4f}         | {aurc:6.4f}")
        print("-"*100)

    # 4. Deployment Economics & Cost-of-Pass Frontier
    print("\n" + "="*100)
    print("DEPLOYMENT ECONOMICS & COST-OF-PASS (C_pass) FRONTIER ($1.50/A100 GPU-Hour Cloud Baseline)")
    print("="*100)
    print(f"{'Model & Format':<20} | {'Mean Tok/Gen':<14} | {'Est. Time/Q (sec)':<18} | {'Cost/Question ($)':<18} | {'Cost-of-Pass (C_pass)':<22}")
    print("-"*100)

    # Throughput baseline on A100 eager vLLM: ~65 tok/s for 7B/8B
    gpu_cost_per_sec = 1.50 / 3600.0  # $0.0004167 per GPU-sec
    tok_per_sec = 65.0

    economics_results = {}

    for m in models:
        for fmt in formats:
            key = f"{m}_{fmt}"
            mean_tok = summary_stats[key]["mean_tokens"]
            pass1 = summary_stats[key]["mean_acc"] / 100.0
            time_per_q = mean_tok / tok_per_sec
            cost_per_q = time_per_q * gpu_cost_per_sec
            c_pass = cost_per_q / pass1

            economics_results[key] = {
                "mean_tokens": mean_tok,
                "time_per_q_sec": time_per_q,
                "cost_per_q_dollars": cost_per_q,
                "c_pass_dollars": c_pass
            }

            print(f"{m + ' ' + fmt:<20} | {mean_tok:8.1f}       | {time_per_q:6.2f} s             | ${cost_per_q:7.5f}          | ${c_pass:7.5f}")
        print("-"*100)

    # Save comprehensive JSON report
    report = {
        "dataset": "HuggingFaceH4/MATH-500",
        "sample_count_per_cell": 500,
        "total_evaluated_completions": 20000,
        "seeds": seeds,
        "summary_statistics": summary_stats,
        "mcnemar_paired_contrasts": mcnemar_results,
        "calibration_metrology": calibration_results,
        "deployment_economics": economics_results
    }

    out_path = "results/reports/phase5_statistical_analysis_report.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"\nPhase 5 Complete! Full statistical report written to: {out_path}")

if __name__ == "__main__":
    main()
