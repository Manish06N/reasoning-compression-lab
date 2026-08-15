#!/usr/bin/env python3
"""
Generate publication-quality vector and raster figures for Paper 1 (J1).
Plots:
1. Figure 1: Reliability-Cost Pareto Frontier (C_pass vs Pass@1 Accuracy)
2. Figure 2: Token Inflation Dynamics across Precision Formats
3. Figure 3: Calibration Metrics (ECE & Brier Score)
4. Figure 4: Multi-Seed Stability & Pass@1 Distribution (Seeds 42-46)
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })

def main():
    setup_style()
    os.makedirs("paper_figures", exist_ok=True)
    
    with open("results/phase5_statistical_analysis_report.json", "r") as f:
        data = json.load(f)

    stats = data["summary_statistics"]
    econ = data["deployment_economics"]
    calib = data["calibration_metrology"]

    formats = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]
    colors = {
        "BF16": "#1f77b4",   # Blue
        "FP8": "#2ca02c",    # Green (Optimal)
        "AWQ-4": "#d62728",  # Red
        "GPTQ-4": "#ff7f0e"  # Orange
    }
    markers = {
        "BF16": "o",
        "FP8": "s",
        "AWQ-4": "^",
        "GPTQ-4": "D"
    }

    # =========================================================================
    # Figure 1: Cost-of-Pass (C_pass) vs Pass@1 Accuracy Pareto Frontier
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for m, ls in [("Qwen-7B", "-"), ("Llama-8B", "--")]:
        accs = []
        c_passes = []
        for fmt in formats:
            key = f"{m}_{fmt}"
            acc = stats[key]["mean_acc"]
            c_pass = econ[key]["c_pass_dollars"] * 100  # cents per correct
            accs.append(acc)
            c_passes.append(c_pass)
            
            ax.scatter(
                c_pass, acc,
                color=colors[fmt],
                marker=markers[fmt],
                s=120,
                zorder=5,
                label=f"{fmt}" if m == "Qwen-7B" else ""
            )
            # Label point
            offset_x = 0.05
            offset_y = 0.2 if fmt != "FP8" else -0.3
            ax.annotate(
                f"{fmt}",
                (c_pass, acc),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=9,
                fontweight="bold" if fmt == "FP8" else "normal"
            )
        
        ax.plot(c_passes, accs, linestyle=ls, color="#555555", alpha=0.6, label=f"{m} Frontier")

    ax.set_xlabel("Cost-of-Pass $C_{\\mathrm{pass}}$ (cents / correct answer)")
    ax.set_ylabel("Mean Pass@1 Accuracy (%) on MATH-500")
    ax.set_title("Reliability–Cost Pareto Frontier (NVIDIA A100 @ $1.50/hr)")
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    fig.savefig("paper_figures/figure1_pareto_frontier.png")
    fig.savefig("paper_figures/figure1_pareto_frontier.pdf")
    plt.close()
    print("Saved Figure 1: paper_figures/figure1_pareto_frontier.png")

    # =========================================================================
    # Figure 2: Token Inflation Dynamics across Precision Formats
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(formats))
    width = 0.35

    qwen_toks = [stats[f"Qwen-7B_{fmt}"]["mean_tokens"] for fmt in formats]
    llama_toks = [stats[f"Llama-8B_{fmt}"]["mean_tokens"] for fmt in formats]

    rects1 = ax.bar(x - width/2, qwen_toks, width, label="Qwen-7B", color="#2b5c8f", alpha=0.85)
    rects2 = ax.bar(x + width/2, llama_toks, width, label="Llama-8B", color="#c0504d", alpha=0.85)

    ax.set_ylabel("Mean Completion Tokens per Problem")
    ax.set_title("Token Inflation Penalty under 4-bit Quantization")
    ax.set_xticks(x)
    ax.set_xticklabels(formats, fontweight="bold")
    ax.legend(frameon=True)
    ax.set_ylim(0, 5500)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f"{int(height):,}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig("paper_figures/figure2_token_inflation.png")
    fig.savefig("paper_figures/figure2_token_inflation.pdf")
    plt.close()
    print("Saved Figure 2: paper_figures/figure2_token_inflation.png")

    # =========================================================================
    # Figure 3: Calibration Metrics (ECE and Brier Score)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    
    qwen_ece = [calib[f"Qwen-7B_{fmt}"]["ece"] for fmt in formats]
    llama_ece = [calib[f"Llama-8B_{fmt}"]["ece"] for fmt in formats]
    qwen_brier = [calib[f"Qwen-7B_{fmt}"]["brier_score"] for fmt in formats]
    llama_brier = [calib[f"Llama-8B_{fmt}"]["brier_score"] for fmt in formats]

    ax1.plot(formats, qwen_ece, marker="o", linewidth=2, color="#2b5c8f", label="Qwen-7B")
    ax1.plot(formats, llama_ece, marker="s", linewidth=2, color="#c0504d", label="Llama-8B")
    ax1.set_title("Expected Calibration Error (ECE)")
    ax1.set_ylabel("ECE (lower is better)")
    ax1.legend(frameon=True)

    ax2.plot(formats, qwen_brier, marker="o", linewidth=2, color="#2b5c8f", label="Qwen-7B")
    ax2.plot(formats, llama_brier, marker="s", linewidth=2, color="#c0504d", label="Llama-8B")
    ax2.set_title("Brier Score Metrology")
    ax2.set_ylabel("Brier Score (lower is better)")
    ax2.legend(frameon=True)

    plt.tight_layout()
    fig.savefig("paper_figures/figure3_calibration_reliability.png")
    fig.savefig("paper_figures/figure3_calibration_reliability.pdf")
    plt.close()
    print("Saved Figure 3: paper_figures/figure3_calibration_reliability.png")

    # =========================================================================
    # Figure 4: Multi-Seed Pass@1 Variance across 5 Seeds (Seeds 42-46)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 4.5))
    seeds = [42, 43, 44, 45, 46]
    
    for m in ["Qwen-7B", "Llama-8B"]:
        for fmt in formats:
            key = f"{m}_{fmt}"
            seed_accs = [stats[key]["seed_accs"][str(s)] for s in seeds]
            ls = "-" if m == "Qwen-7B" else "--"
            ax.plot(
                seeds, seed_accs,
                marker=markers[fmt],
                linestyle=ls,
                color=colors[fmt],
                alpha=0.8,
                label=f"{m} {fmt}"
            )

    ax.set_xlabel("Sampling Random Seed")
    ax.set_ylabel("Pass@1 Accuracy (%)")
    ax.set_title("Seed-to-Seed Stability across 5 Seeds (MATH-500, n=500)")
    ax.set_xticks(seeds)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    fig.savefig("paper_figures/figure4_seed_variance.png")
    fig.savefig("paper_figures/figure4_seed_variance.pdf")
    plt.close()
    print("Saved Figure 4: paper_figures/figure4_seed_variance.png")

    print("\nAll publication figures generated successfully in paper_figures/")

if __name__ == "__main__":
    main()
