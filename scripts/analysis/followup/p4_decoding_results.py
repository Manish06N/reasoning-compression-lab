#!/usr/bin/env python3
"""P4 decoding-grid cells (Qwen-7B, 300 MATH-500 items, seeds 42-44; temperature {0.2,0.6,1.0} x token budget {8192,32768}; BF16 and AWQ)
against the campaign BF16 reference at T=0.6 / 32768 restricted to the same 300 items. Real changes to the sampling setting, to compare with the placebo."""
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
V = ROOT / "results" / "followup" / "p2p4"
sys.path.insert(0, str(ROOT / "scripts" / "analysis")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_sensitivities as rs, seed_placebo as sp

B = 10_000
inv = [r for r in csv.DictReader(open(V / "tables" / "cell_inventory.tsv"), delimiter="\t") if "math300" in r["experiment_id"]]
item = list(csv.DictReader(open(ROOT / "results" / "followup" / "provenance" / "MATH500_item_table.tsv"), delimiter="\t"))
camp_row = {t["unique_id"]: int(t["row"]) - 1 for t in item}


def load(variant, seed):
    r = [x for x in inv if x["variant_id"] == variant and x["seed"] == str(seed)]; assert len(r) == 1, (variant, seed, len(r))
    d = json.load(open(V / "compact" / Path(r[0]["compact_file"]).name)); det = sorted(d["details"], key=lambda z: z["row"])
    ids = [z["item_id"] for z in det]
    return ids, np.array([z["extractive_match"] for z in det], float), np.array([z["completion_tokens"] for z in det], float), np.array([z.get("finish_reason") == "length" for z in det], float)


ids0, *_ = load("bf16_t1.0_b8k", 42)
rows = np.array([camp_row[i] for i in ids0]); assert len(set(rows)) == 300, "items must be 300 distinct campaign rows"
Ab5, Tb5 = rs.load_cell("math500", "Qwen-7B", "BF16"); Ab5, Tb5 = Ab5[:, rows], Tb5[:, rows]; Ab, Tb = Ab5[:3], Tb5[:3]
idx = np.random.default_rng(0).integers(0, 300, (B, 300))
def pct(x): return [float(v) for v in np.percentile(x, [2.5, 97.5])]
def fmt(v, ci, d=0): return f"{v:+.{d}f} [{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]"
spl, npl = sp.placebo_sums(Ab5, Tb5); pi = spl.sum() / npl.sum(); bp = spl[idx].sum(1) / npl[idx].sum(1)
print(f"reference: campaign BF16 Qwen, T=0.6, 32768, same 300 items. pass@1 (seeds 42-44) {Ab.mean()*100:.2f}, mean tokens {Tb.mean():.0f}; 5-seed placebo on these items {fmt(pi, pct(bp))}")
a0, t0 = load("bf16_t0.6_b8k", 42)[1:3]; print(f"alignment check (bf16 T0.6 8k seed 42 vs campaign seed 42): agreement {(a0 == Ab[0]).mean():.3f}, token corr {np.corrcoef(t0, Tb[0])[0,1]:.3f}")
res = {"placebo": pi, "placebo_ci": pct(bp), "n_items": 300}
print("\nsetting            | acc diff (pp)          | uncond tokens          | Lambda_b (matched)     | excess                 | length-stops")
for variant in ("bf16_t0.2_b32k", "bf16_t1.0_b32k", "bf16_t0.6_b8k", "bf16_t0.2_b8k", "bf16_t1.0_b8k", "awq_t0.6_b32k", "awq_t0.2_b32k", "awq_t1.0_b32k", "awq_t0.6_b8k", "awq_t0.2_b8k", "awq_t1.0_b8k"):
    L = [load(variant, s) for s in (42, 43, 44)]
    assert all(l[0] == ids0 for l in L), "item order differs between cells"
    A = np.array([l[1] for l in L]); T = np.array([l[2] for l in L]); F = np.array([l[3] for l in L])
    s_, n_ = sp.item_sums(Ab, Tb, T, [(s, s) for s in range(3)]); lam = s_.sum() / n_.sum(); b = s_[idx].sum(1) / n_[idx].sum(1)
    dA = (A.mean(0) - Ab.mean(0)) * 100; dT = T.mean(0) - Tb.mean(0)
    r = dict(acc=float(dA.mean()), acc_ci=pct(dA[idx].mean(1)), uncond=float(dT.mean()), uncond_ci=pct(dT[idx].mean(1)), lam=float(lam), lam_ci=pct(b),
             excess=float(lam - pi), excess_ci=pct(b - bp), length_stop_share=float(F.mean()), pass1=float(A.mean() * 100), tokens=float(T.mean()))
    res[variant] = r
    print(f"{variant:18s} | {fmt(r['acc'], r['acc_ci'], 2):22s} | {fmt(r['uncond'], r['uncond_ci']):22s} | {fmt(lam, r['lam_ci']):22s} | {fmt(r['excess'], r['excess_ci']):22s} | {F.mean()*100:5.2f}%")
out = ROOT / "results" / "followup" / "reports"; out.mkdir(exist_ok=True)
(out / "p4_decoding_results.json").write_text(json.dumps(res, indent=1)); print("\nwrote p4_decoding_results.json")
