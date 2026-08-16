#!/usr/bin/env python3
"""
MATH-500 Fine-Grained Subject and Difficulty Stratification Analysis.
Cross-references validation files with MATH-500 dataset metadata (level and subject)
to compute accuracy, completion tokens, and token inflation across:
- 5 Difficulty Levels (Level 1 to Level 5)
- 7 Subject Categories (Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus)
"""

import json
import glob
import os
import math
from collections import defaultdict
from datasets import load_from_disk

def main():
    val_dir = "results/math500"
    files = sorted(glob.glob(os.path.join(val_dir, "*.json")))
    if not files:
        val_dir = "outputs-hpc-campaign-2026-08-14/validation"
        files = sorted(glob.glob(os.path.join(val_dir, "*.json")))
    
    print(f"Loaded {len(files)} validation files for stratified difficulty analysis...")

    # Load dataset features
    ds = load_from_disk("external/Quantized-Reasoning-Models/datasets/MATH-500")
    print(f"Loaded MATH-500 dataset with {len(ds)} rows.")
    levels_ds = [ds[i]["level"] for i in range(len(ds))]
    subjects_ds = [ds[i]["subject"] for i in range(len(ds))]

    # Load validation data: data[model][format][seed] = details list of 500 dicts
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

    # 1. Stratification by Difficulty Level (1 to 5)
    level_results = defaultdict(lambda: defaultdict(dict))
    levels = [1, 2, 3, 4, 5]
    
    for lvl in levels:
        lvl_indices = [i for i, l in enumerate(levels_ds) if l == lvl]
        n_lvl = len(lvl_indices)
        
        for m in models:
            for fmt in formats:
                seed_accs = []
                seed_tokens = []
                for s in seeds:
                    details = data[m][fmt][s]
                    correct_count = sum(1 for i in lvl_indices if details[i].get("extractive_match", 0.0) == 1.0)
                    tok_mean = sum(details[i].get("completion_tokens", 0) for i in lvl_indices) / n_lvl if n_lvl > 0 else 0
                    seed_accs.append(correct_count / n_lvl * 100 if n_lvl > 0 else 0)
                    seed_tokens.append(tok_mean)
                
                mean_acc = sum(seed_accs) / len(seed_accs)
                std_acc = math.sqrt(sum((a - mean_acc)**2 for a in seed_accs) / (len(seed_accs) - 1))
                mean_tok = sum(seed_tokens) / len(seed_tokens)

                level_results[lvl][f"{m}_{fmt}"] = {
                    "count": n_lvl,
                    "accuracy_mean": mean_acc,
                    "accuracy_std": std_acc,
                    "tokens_mean": mean_tok
                }

    # 2. Stratification by Subject Category
    subjects = sorted(list(set(subjects_ds)))
    subject_results = defaultdict(lambda: defaultdict(dict))

    for subj in subjects:
        subj_indices = [i for i, s_name in enumerate(subjects_ds) if s_name == subj]
        n_subj = len(subj_indices)
        for m in models:
            for fmt in formats:
                seed_accs = []
                seed_tokens = []
                for s in seeds:
                    details = data[m][fmt][s]
                    correct_count = sum(1 for i in subj_indices if details[i].get("extractive_match", 0.0) == 1.0)
                    tok_mean = sum(details[i].get("completion_tokens", 0) for i in subj_indices) / n_subj if n_subj > 0 else 0
                    seed_accs.append(correct_count / n_subj * 100 if n_subj > 0 else 0)
                    seed_tokens.append(tok_mean)
                
                mean_acc = sum(seed_accs) / len(seed_accs)
                std_acc = math.sqrt(sum((a - mean_acc)**2 for a in seed_accs) / (len(seed_accs) - 1)) if len(seed_accs) > 1 else 0
                mean_tok = sum(seed_tokens) / len(seed_tokens)

                subject_results[subj][f"{m}_{fmt}"] = {
                    "count": n_subj,
                    "accuracy_mean": mean_acc,
                    "accuracy_std": std_acc,
                    "tokens_mean": mean_tok
                }

    # Print Summary Tables
    print("\n" + "="*105)
    print("MATH-500 DIFFICULTY STRATIFICATION (LEVELS 1-5, 5 SEEDS MEAN ± STD)")
    print("="*105)
    print(f"{'Difficulty Level':<18} | {'Qwen BF16':<16} | {'Qwen FP8':<16} | {'Qwen AWQ-4':<16} | {'Qwen GPTQ-4':<16}")
    print("-"*105)
    for lvl in levels:
        n_count = level_results[lvl]['Qwen-7B_BF16']['count']
        q_bf16 = f"{level_results[lvl]['Qwen-7B_BF16']['accuracy_mean']:.2f}% ± {level_results[lvl]['Qwen-7B_BF16']['accuracy_std']:.2f}%"
        q_fp8 = f"{level_results[lvl]['Qwen-7B_FP8']['accuracy_mean']:.2f}% ± {level_results[lvl]['Qwen-7B_FP8']['accuracy_std']:.2f}%"
        q_awq = f"{level_results[lvl]['Qwen-7B_AWQ-4']['accuracy_mean']:.2f}% ± {level_results[lvl]['Qwen-7B_AWQ-4']['accuracy_std']:.2f}%"
        q_gptq = f"{level_results[lvl]['Qwen-7B_GPTQ-4']['accuracy_mean']:.2f}% ± {level_results[lvl]['Qwen-7B_GPTQ-4']['accuracy_std']:.2f}%"
        print(f"Level {lvl} (n={n_count:<3})      | {q_bf16:<16} | {q_fp8:<16} | {q_awq:<16} | {q_gptq:<16}")
    
    print("\n" + "="*105)
    print(f"{'Difficulty Level':<18} | {'Llama BF16':<16} | {'Llama FP8':<16} | {'Llama AWQ-4':<16} | {'Llama GPTQ-4':<16}")
    print("-"*105)
    for lvl in levels:
        n_count = level_results[lvl]['Llama-8B_BF16']['count']
        l_bf16 = f"{level_results[lvl]['Llama-8B_BF16']['accuracy_mean']:.2f}% ± {level_results[lvl]['Llama-8B_BF16']['accuracy_std']:.2f}%"
        l_fp8 = f"{level_results[lvl]['Llama-8B_FP8']['accuracy_mean']:.2f}% ± {level_results[lvl]['Llama-8B_FP8']['accuracy_std']:.2f}%"
        l_awq = f"{level_results[lvl]['Llama-8B_AWQ-4']['accuracy_mean']:.2f}% ± {level_results[lvl]['Llama-8B_AWQ-4']['accuracy_std']:.2f}%"
        l_gptq = f"{level_results[lvl]['Llama-8B_GPTQ-4']['accuracy_mean']:.2f}% ± {level_results[lvl]['Llama-8B_GPTQ-4']['accuracy_std']:.2f}%"
        print(f"Level {lvl} (n={n_count:<3})      | {l_bf16:<16} | {l_fp8:<16} | {l_awq:<16} | {l_gptq:<16}")
    print("="*105)

    # Print Subject Stratification
    print("\n" + "="*105)
    print("MATH-500 SUBJECT STRATIFICATION (7 SUBJECTS, 5 SEEDS MEAN ± STD)")
    print("="*105)
    print(f"{'Subject Category':<24} | {'Qwen BF16':<16} | {'Qwen FP8':<16} | {'Qwen AWQ-4':<16} | {'Qwen GPTQ-4':<16}")
    print("-"*105)
    for subj in subjects:
        n_count = subject_results[subj]['Qwen-7B_BF16']['count']
        q_bf16 = f"{subject_results[subj]['Qwen-7B_BF16']['accuracy_mean']:.1f}% ± {subject_results[subj]['Qwen-7B_BF16']['accuracy_std']:.1f}%"
        q_fp8 = f"{subject_results[subj]['Qwen-7B_FP8']['accuracy_mean']:.1f}% ± {subject_results[subj]['Qwen-7B_FP8']['accuracy_std']:.1f}%"
        q_awq = f"{subject_results[subj]['Qwen-7B_AWQ-4']['accuracy_mean']:.1f}% ± {subject_results[subj]['Qwen-7B_AWQ-4']['accuracy_std']:.1f}%"
        q_gptq = f"{subject_results[subj]['Qwen-7B_GPTQ-4']['accuracy_mean']:.1f}% ± {subject_results[subj]['Qwen-7B_GPTQ-4']['accuracy_std']:.1f}%"
        print(f"{subj:<24} (n={n_count:<3})| {q_bf16:<16} | {q_fp8:<16} | {q_awq:<16} | {q_gptq:<16}")
    print("="*105)

    # Save to results/reports/stratified_difficulty_report.json
    out_dict = {
        "difficulty_levels": level_results,
        "subject_categories": subject_results
    }
    out_file = "results/reports/stratified_difficulty_report.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(out_dict, fp, indent=2)
    print(f"\nStratified difficulty report saved to: {out_file}")

if __name__ == "__main__":
    main()
