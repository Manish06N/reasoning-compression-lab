#!/usr/bin/env python3
"""P2 publication-phase Qwen-7B cells on GSM8K (self-quantized AWQ-4 and GPTQ-4, 3 calibration draws x seeds 42-44) against the campaign BF16 cells.
Rows are joined to the campaign's row order through dataset_index (GSM8K_item_table.tsv). Scoring: extractive_match (campaign scorer)."""
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
V4 = ROOT / "results" / "followup" / "p2p4"
sys.path.insert(0, str(ROOT / "scripts" / "analysis")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_sensitivities as rs, seed_placebo as sp

B = 10_000; N = 1319
inv = [r for r in csv.DictReader(open(V4 / "tables" / "cell_inventory.tsv"), delimiter="\t")]
tab = list(csv.DictReader(open(ROOT / "results" / "followup" / "provenance" / "GSM8K_item_table.tsv"), delimiter="\t"))
camp_pos = {int(t["dataset_index"]): int(t["campaign_row"]) - 1 for t in tab}      # dataset_index -> campaign row (0-based)


def cell(method, cal, seed, exp="p2_publication_qwen7b_gsm8k_v1"):
    r = [x for x in inv if x["experiment_id"] == exp and x["variant_id"].startswith(f"{method}_cal{cal}") and x["seed"] == str(seed)]
    assert len(r) == 1, (method, cal, seed, len(r))
    d = json.load(open(V4 / "compact" / Path(r[0]["compact_file"]).name))
    A = np.zeros(N); T = np.zeros(N)
    for x in d["details"]:
        i = camp_pos[x["dataset_index"]]; A[i] = x["extractive_match"]; T[i] = x["completion_tokens"]
    return A, T


def pct(x): return [float(v) for v in np.percentile(x, [2.5, 97.5])]
def fmt(v, ci, d=0): return f"{v:+.{d}f} [{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]"


Ab, Tb = rs.load_cell("gsm8k", "Qwen-7B", "BF16")                                    # 3 seeds, campaign order
spl, npl = sp.placebo_sums(Ab, Tb); pi = spl.sum() / npl.sum()
idx = np.random.default_rng(0).integers(0, N, (B, N)); bp = spl[idx].sum(1) / npl[idx].sum(1)
res = {"placebo_3seed": pi, "placebo_ci": pct(bp)}
print(f"campaign BF16 Qwen GSM8K (seeds 42-44) pass@1 {Ab.mean()*100:.2f}, mean tokens {Tb.mean():.0f}; 3-seed placebo {fmt(pi, pct(bp))}")


def analyse(label, arms):
    A = np.vstack([a for a, t in arms]); T = np.vstack([t for a, t in arms]); k = len(arms)
    pairs = [(s, c * 3 + s) for c in range(k) for s in range(3)]
    s_, n_ = sp.item_sums(Ab, Tb, T, pairs)
    b = s_[idx].sum(1) / n_[idx].sum(1); lam = s_.sum() / n_.sum()
    dA = (A.reshape(k, 3, N).mean((0, 1)) - Ab.mean(0)) * 100; dT = T.reshape(k, 3, N).mean((0, 1)) - Tb.mean(0)
    r = dict(acc_pp=float(dA.mean()), acc_ci=pct(dA[idx].mean(1)), uncond=float(dT.mean()), uncond_ci=pct(dT[idx].mean(1)),
             lam=float(lam), lam_ci=pct(b), excess=float(lam - pi), excess_ci=pct(b - bp))
    print(f"  {label:30s} acc {fmt(r['acc_pp'], r['acc_ci'], 2)}pp | uncond {fmt(r['uncond'], r['uncond_ci'])} | Lambda_b {fmt(lam, r['lam_ci'])} | excess {fmt(r['excess'], r['excess_ci'])}")
    return r


print("\nSelf-quantized Qwen-7B vs campaign BF16 (GSM8K, seeds 42-44, seed-matched; excess against the 3-seed placebo)")
arms = {}
for method in ("awq4", "gptq4"):
    arms[method] = [tuple(np.array(z) for z in zip(*[cell(method, c, s) for s in (42, 43, 44)])) for c in range(3)]
    res[method] = {}
    for c in range(3): res[method][f"cal{c}"] = analyse(f"{method} calibration draw {c}", [arms[method][c]])
    res[method]["pooled"] = analyse(f"{method} all three draws pooled", arms[method])
print("\nCommunity checkpoints on the same three seeds")
for fmt_name in ("AWQ-4", "GPTQ-4"):
    Aq, Tq = rs.load_cell("gsm8k", "Qwen-7B", fmt_name)
    res["community_" + fmt_name] = analyse(f"community {fmt_name}", [(Aq, Tq)])

print("\nDraw-to-draw spread")
res["draw_spread"] = {}
for method in ("awq4", "gptq4"):
    for i, j in ((0, 1), (0, 2), (1, 2)):
        Ai, Ti = arms[method][i]; Aj, Tj = arms[method][j]
        da = (Ai.mean(0) - Aj.mean(0)) * 100; dt = Ti.mean(0) - Tj.mean(0)
        r = dict(acc_pp=float(da.mean()), acc_ci=pct(da[idx].mean(1)), tokens=float(dt.mean()), tokens_ci=pct(dt[idx].mean(1)))
        res["draw_spread"][f"{method} cal{i}-cal{j}"] = r
        print(f"  {method} draw{i}-draw{j}: accuracy {fmt(r['acc_pp'], r['acc_ci'], 2)}pp | mean tokens {fmt(r['tokens'], r['tokens_ci'])}")

print("\nDtype control: GPTQ draws at float16 vs bfloat16 (GSM8K)")
res["dtype"] = {}
for c in range(3):
    f16 = tuple(np.array(z) for z in zip(*[cell("gptq4", c, s, "p2_dtype_control_qwen7b_gsm8k_v1") for s in (42, 43, 44)]))
    bf = arms["gptq4"][c]
    da = (f16[0].mean(0) - bf[0].mean(0)) * 100; dt = f16[1].mean(0) - bf[1].mean(0)
    res["dtype"][f"cal{c}"] = dict(acc_pp=float(da.mean()), acc_ci=pct(da[idx].mean(1)), tokens=float(dt.mean()), tokens_ci=pct(dt[idx].mean(1)))
    print(f"  GPTQ draw {c}: float16 minus bfloat16: accuracy {fmt(da.mean(), pct(da[idx].mean(1)), 2)}pp | mean tokens {fmt(dt.mean(), pct(dt[idx].mean(1)))}")
out = ROOT / "results" / "followup" / "reports"; out.mkdir(exist_ok=True)
(out / "p2_qwen_gsm8k_results.json").write_text(json.dumps(res, indent=1)); print("\nwrote p2_qwen_gsm8k_results.json")
