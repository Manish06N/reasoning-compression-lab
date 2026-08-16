#!/usr/bin/env python3
"""Oracle gold-hit selective-prediction diagnostic (NOT deployable).

The compact validation JSON has no extracted answer strings. This script therefore
gates on gold-correct seed counts, not on modal-answer agreement. For K=5, k=3
coverage is identically 100% because max(c, 5-c) >= 3 always. Do not cite these
numbers as an operational safety gate. Prefer scripts/analysis/revision_reanalysis.py.
"""

import json
import glob
import os
from collections import defaultdict

def main():
    val_dir = "results/math500"
    files = sorted(glob.glob(os.path.join(val_dir, "*.json")))
    if not files:
        val_dir = "outputs-hpc-campaign-2026-08-14/validation"
        files = sorted(glob.glob(os.path.join(val_dir, "*.json")))

    data = defaultdict(lambda: defaultdict(dict))

    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        bn = os.path.basename(f).replace(".json", "")
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

        data[model][fmt][seed] = d["details"]

    seeds = [42, 43, 44, 45, 46]
    models = ["Qwen-7B", "Llama-8B"]
    formats = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]

    thresholds = [1, 2, 3, 4, 5]  # Minimum correct agreement required to answer (out of 5)

    selective_results = defaultdict(lambda: defaultdict(dict))

    print("\n" + "="*110)
    print("SELECTIVE PREDICTION RISK-COVERAGE TRADEOFF (MATH-500, n=500, 5-SAMPLE CONSENSUS)")
    print("="*110)
    print(f"{'Model & Format':<18} | {'Min Agree':<10} | {'Coverage (%)':<15} | {'Selective Acc (%)':<20} | {'Selective Risk (%)':<20}")
    print("-"*110)

    for m in models:
        for fmt in formats:
            cell_seeds = data[m][fmt]
            for th in [3, 4, 5]:  # Practical production operational thresholds
                answered_count = 0
                correct_among_answered = 0
                for row_idx in range(500):
                    c_correct = sum(1 for s in seeds if cell_seeds[s][row_idx].get("extractive_match", 0.0) == 1.0)
                    # Policy: If model is confident (consensus >= th), it answers with majority vote
                    # Majority vote answer is correct if c_correct >= 3
                    if c_correct >= th:
                        answered_count += 1
                        correct_among_answered += 1
                    elif (5 - c_correct) >= th:
                        # Model is confident it reached a consensus (all agree on same wrong answer)
                        answered_count += 1
                        # this answer was incorrect

                coverage = (answered_count / 500.0) * 100
                sel_acc = (correct_among_answered / answered_count * 100) if answered_count > 0 else 100.0
                sel_risk = 100.0 - sel_acc

                selective_results[f"{m}_{fmt}"][f"th_{th}"] = {
                    "threshold": f"{th}/5",
                    "coverage": coverage,
                    "selective_accuracy": sel_acc,
                    "selective_risk": sel_risk
                }

                th_label = f">={th}/5 seeds"
                print(f"{m + ' ' + fmt:<18} | {th_label:<10} | {coverage:6.2f}%         | {sel_acc:6.2f}%             | {sel_risk:6.2f}%")
        print("-"*110)

    out_file = "results/reports/selective_prediction_report.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(selective_results, fp, indent=2)
    print(f"\nSelective prediction report saved to: {out_file}")

if __name__ == "__main__":
    main()
