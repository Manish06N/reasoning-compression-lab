#!/usr/bin/env python3
"""
DEPRECATED.

This script is not the manuscript pipeline. Pathology keys and
summary tables must come from scripts/analysis/revision_reanalysis.py.
"""

import json
import glob
import os
import math
from collections import defaultdict

def parse_val_file(path):
    with open(path) as f:
        data = json.load(f)
    acc = data.get("accuracy", data.get("pass@1", data.get("accuracy_mean", None)))
    if acc is None and "summary" in data:
        acc = data["summary"].get("pass@1", data["summary"].get("accuracy", None))
    if acc is None:
        records = data.get("sample_records", data.get("results", []))
        if records:
            acc = sum(1 for r in records if r.get("is_correct", False)) / len(records)
    
    tokens = data.get("completion_tokens_mean", 0)
    trunc = data.get("token_limit_hits", data.get("truncation_count", data.get("hit_token_limit_count", 0)))
    loops = data.get("repetition_rows", data.get("repetition_flag_count", 0))
    return acc, tokens, trunc, loops

def compute_grid(dir_path, seeds, models, formats):
    grid = {}
    for m_label, m_key in models:
        for f_label, f_key in formats:
            accs = []
            tokens_list = []
            total_trunc = 0
            total_loops = 0
            
            for s in seeds:
                if f_key == "BF16":
                    pattern = f"{dir_path}/DeepSeek-R1-Distill-{m_key}_*seed{s}*.json"
                else:
                    pattern = f"{dir_path}/*{m_key}*{f_key}*seed{s}*.json"
                
                matches = glob.glob(pattern)
                if matches:
                    acc, tok, tr, lp = parse_val_file(matches[0])
                    if acc is not None:
                        accs.append(acc * 100)
                    if tok:
                        tokens_list.append(tok)
                    total_trunc += tr
                    total_loops += lp
            
            if accs:
                mean_acc = sum(accs) / len(accs)
                std_acc = math.sqrt(sum((a - mean_acc)**2 for a in accs) / (len(accs) - 1)) if len(accs) > 1 else 0.0
            else:
                mean_acc, std_acc = 0.0, 0.0
            
            mean_tok = sum(tokens_list) / len(tokens_list) if tokens_list else 0.0
            grid[f"{m_label}_{f_label}"] = {
                "model": m_label,
                "format": f_label,
                "completed_seeds": len(accs),
                "total_seeds": len(seeds),
                "accuracy_mean": mean_acc,
                "accuracy_std": std_acc,
                "mean_tokens": mean_tok,
                "truncations": total_trunc,
                "loops": total_loops
            }
    return grid

def main():
    models = [("Qwen-7B", "Qwen-7B"), ("Llama-8B", "Llama-8B")]
    formats = [("BF16", "BF16"), ("FP8", "FP8"), ("AWQ-4", "AWQ-4"), ("GPTQ-4", "GPTQ-4")]

    math500_grid = compute_grid("results/math500", [42, 43, 44, 45, 46], models, formats)
    gsm8k_grid = compute_grid("results/gsm8k", [42, 43, 44], models, formats)
    
    # Check gpqa directory in results/gpqa and outputs-hpc-breadth-gpqa-2026-08-16/validation
    gpqa_dir = "outputs-hpc-breadth-gpqa-2026-08-16/validation" if os.path.exists("outputs-hpc-breadth-gpqa-2026-08-16/validation") else "results/gpqa"
    gpqa_grid = compute_grid(gpqa_dir, [42, 43, 44], models, formats)

    master_summary = {
        "math500": math500_grid,
        "gsm8k": gsm8k_grid,
        "gpqa_diamond": gpqa_grid
    }

    out_file = "results/reports/multitask_benchmark_summary.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(master_summary, fp, indent=2)

    print("\n" + "="*110)
    print("CROSS-BENCHMARK GENERALIZATION MATRIX (PASS@1 ACCURACY MEAN ± STD)")
    print("="*110)
    print(f"{'Model & Format':<20} | {'MATH-500 (n=500)':<22} | {'GSM8K (n=1,319)':<22} | {'GPQA-Diamond (n=198)':<22} | {'Pathologies (T/L)':<18}")
    print("-"*110)

    for m_label, _ in models:
        for f_label, _ in formats:
            key = f"{m_label}_{f_label}"
            m_res = math500_grid.get(key, {})
            g_res = gsm8k_grid.get(key, {})
            q_res = gpqa_grid.get(key, {})

            m_str = f"{m_res.get('accuracy_mean', 0.0):.2f}% ± {m_res.get('accuracy_std', 0.0):.2f}%" if m_res.get('completed_seeds', 0) > 0 else "—"
            g_str = f"{g_res.get('accuracy_mean', 0.0):.2f}% ± {g_res.get('accuracy_std', 0.0):.2f}%" if g_res.get('completed_seeds', 0) > 0 else "—"
            q_str = f"{q_res.get('accuracy_mean', 0.0):.2f}% ± {q_res.get('accuracy_std', 0.0):.2f}%" if q_res.get('completed_seeds', 0) > 0 else "—"

            total_t = m_res.get("truncations", 0) + g_res.get("truncations", 0) + q_res.get("truncations", 0)
            total_l = m_res.get("loops", 0) + g_res.get("loops", 0) + q_res.get("loops", 0)
            pathology_str = f"{total_t} trunc / {total_l} loops"

            print(f"{m_label + ' ' + f_label:<20} | {m_str:<22} | {g_str:<22} | {q_str:<22} | {pathology_str:<18}")
        print("-"*110)

    print(f"\nMulti-task master summary written to: {out_file}")

if __name__ == "__main__":
    main()
