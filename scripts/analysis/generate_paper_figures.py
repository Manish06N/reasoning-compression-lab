#!/usr/bin/env python3
"""Optional matplotlib copies of the pinned-stack figures.

The submitted PDF uses TikZ in paper/main.tex. These PNGs are not the manuscript
and must not be cited as a Pareto frontier.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FORMATS = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]
COLORS = {"BF16": "#1f77b4", "FP8": "#2ca02c", "AWQ-4": "#d62728", "GPTQ-4": "#ff7f0e"}
MARKERS = {"BF16": "o", "FP8": "s", "AWQ-4": "^", "GPTQ-4": "D"}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )


def main():
    setup_style()
    os.makedirs("paper_figures", exist_ok=True)
    os.makedirs("paper", exist_ok=True)

    with open("results/reports/revision_reanalysis_report.json") as f:
        rev = json.load(f)
    math = rev["math500"]
    stats = math["summary_statistics"]
    econ = math["deployment_economics"]
    tokens = math["token_analysis"]

    # Figure 1: modeled token-cost ranking (not measured latency)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, ls in [("Qwen-7B", "-"), ("Llama-8B", "--")]:
        accs, costs = [], []
        for fmt in FORMATS:
            key = f"{m}_{fmt}"
            acc = stats[key]["mean_acc"]
            c_pass = econ[key]["c_pass_dollars"] * 100
            accs.append(acc)
            costs.append(c_pass)
            ax.scatter(c_pass, acc, color=COLORS[fmt], marker=MARKERS[fmt], s=120, zorder=5)
            ax.annotate(fmt, (c_pass, acc), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.plot(costs, accs, linestyle=ls, color="#555555", alpha=0.6, label=f"{m}")
    ax.set_xlabel(r"Token-implied $C_{\mathrm{pass}}$ (cents / correct) at 65 tok/s")
    ax.set_ylabel("MATH-500 mean pass@1 (%)")
    ax.set_title("Modeled token-cost ranking (throughput held equal)")
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()
    for dest in ("paper_figures", "paper"):
        fig.savefig(os.path.join(dest, "figure1_pareto_frontier.pdf"))
        fig.savefig(os.path.join(dest, "figure1_pareto_frontier.png"))
    plt.close()

    # Figure 2: mean tokens
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(FORMATS))
    width = 0.35
    qwen = [stats[f"Qwen-7B_{fmt}"]["mean_tokens"] for fmt in FORMATS]
    llama = [stats[f"Llama-8B_{fmt}"]["mean_tokens"] for fmt in FORMATS]
    r1 = ax.bar(x - width / 2, qwen, width, label="Qwen-7B", color="#2b5c8f", alpha=0.85)
    r2 = ax.bar(x + width / 2, llama, width, label="Llama-8B", color="#c0504d", alpha=0.85)
    ax.set_ylabel("Mean completion tokens (MATH-500, 5 seeds)")
    ax.set_title("Full-grid mean length (ratio of means)")
    ax.set_xticks(x)
    ax.set_xticklabels(FORMATS)
    ax.legend(frameon=True)
    ax.set_ylim(0, 6000)
    for rect in list(r1) + list(r2):
        h = rect.get_height()
        ax.annotate(f"{int(h):,}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    for dest in ("paper_figures", "paper"):
        fig.savefig(os.path.join(dest, "figure2_token_inflation.pdf"))
        fig.savefig(os.path.join(dest, "figure2_token_inflation.png"))
    plt.close()

    # Figure 3: token delta by correctness stratum
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = []
    both_ok = []
    bf16_only = []
    for m in ["Qwen-7B", "Llama-8B"]:
        for fmt in ["FP8", "AWQ-4", "GPTQ-4"]:
            st = tokens[f"{m}_{fmt}"]["strata"]
            labels.append(f"{m.split('-')[0]}\n{fmt}")
            both_ok.append(st["both_correct"]["mean"])
            bf16_only.append(st["bf16_only"]["mean"])
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, both_ok, width, label="Both correct", color="#4c9a2a")
    ax.bar(x + width / 2, bf16_only, width, label="BF16 correct, quantized wrong", color="#c44e52")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean paired token delta vs BF16")
    ax.set_title("Length change concentrates on format-induced failures")
    ax.legend(frameon=True)
    fig.tight_layout()
    for dest in ("paper_figures", "paper"):
        fig.savefig(os.path.join(dest, "figure3_calibration_reliability.pdf"))
        fig.savefig(os.path.join(dest, "figure3_calibration_reliability.png"))
    plt.close()

    # Figure 4: seed variance, full y-axis
    fig, ax = plt.subplots(figsize=(8, 4.8))
    seeds = [42, 43, 44, 45, 46]
    for m in ["Qwen-7B", "Llama-8B"]:
        for fmt in FORMATS:
            key = f"{m}_{fmt}"
            accs = [stats[key]["seed_accs"][str(s)] for s in seeds]
            ax.plot(seeds, accs, marker=MARKERS[fmt], linestyle="-" if m == "Qwen-7B" else "--",
                    color=COLORS[fmt], alpha=0.85, label=f"{m} {fmt}")
    ax.set_xlabel("Sampling seed")
    ax.set_ylabel("Pass@1 (%)")
    ax.set_title("MATH-500 pass@1 by seed")
    ax.set_xticks(seeds)
    ax.set_ylim(80, 100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=8)
    fig.tight_layout()
    for dest in ("paper_figures", "paper"):
        fig.savefig(os.path.join(dest, "figure4_seed_variance.pdf"))
        fig.savefig(os.path.join(dest, "figure4_seed_variance.png"))
    plt.close()
    print("Wrote figures to paper_figures/ and paper/")


if __name__ == "__main__":
    main()
