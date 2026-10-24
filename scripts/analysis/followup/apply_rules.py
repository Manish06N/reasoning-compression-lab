#!/usr/bin/env python3
"""Apply the pre-registered verdict rules of the pending experiments, literally, to compact per-item records.

    python scripts/analysis/followup/apply_rules.py g1  --x5 <X5 compact dir> --g1 <G1 compact dir>
    python scripts/analysis/followup/apply_rules.py e4  --bf16 '<glob of BF16 cells>' --awq '<glob of AWQ-4 cells>'
    python scripts/analysis/followup/apply_rules.py d1lite --w3 '<glob of GPTQ-3 cells>' [--e2 '<glob of E2 float16 Qwen cells>']
    python scripts/analysis/followup/apply_rules.py x2 --dir <dir with the X2 compact cells of ONE model, both AIME sets>

Rules: results/followup/PREREG_G1_C2_D1.md (G1), PREREG_E4_1p5B_D1lite.md (E4, D1-lite), PREREG_X2_E4_as_received_2026-10-18.md (X2).
Item bootstrap B=10,000, numpy default_rng(0), shared resamples across the quantities that are compared. Nothing is decided here that the
pre-registration does not decide; when a rule cannot be applied (cells missing) the output says INCOMPLETE and prints no verdict for it.
"""
import argparse, glob, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "analysis")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import seed_placebo as sp

B = 10_000
OUT = ROOT / "results" / "followup" / "reports"; OUT.mkdir(exist_ok=True)


def pct(x): return [float(v) for v in np.percentile(x, [2.5, 97.5])]
def excl0(ci): return not (ci[0] <= 0 <= ci[1])
def fmt(v, ci, d=0): return f"{v:+.{d}f} [{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]"
def seed_of(p): return int(re.search(r"seed(\d+)", Path(p).name).group(1))


def load_cells(files):
    """-> seeds, A (correct), T (completion tokens), L (stopped by the length limit); arrays (seeds, items), rows in 'row' order."""
    A, T, L, seeds, rows0 = [], [], [], [], None
    for f in sorted(files, key=seed_of):
        d = sorted(json.load(open(f, encoding="utf-8"))["details"], key=lambda r: r["row"])
        rows = [r["row"] for r in d]
        assert rows0 is None or rows == rows0, f"item rows differ in {f}"
        rows0 = rows
        A.append([r["extractive_match"] for r in d]); T.append([r["completion_tokens"] for r in d])
        L.append([r.get("finish_reason") == "length" for r in d]); seeds.append(seed_of(f))
    return seeds, np.array(A, float), np.array(T, float), np.array(L, bool)


def boot_idx(n): return np.random.default_rng(0).integers(0, n, (B, n))


# ---------------------------------------------------------------------------------------------- seed-placebo contrast (E4, D1-lite)
def contrast(ref, oth, name):
    """ref = (seeds, A, T, L) of the reference arm with >= 3 seeds; oth = same for the other arm. The placebo uses ALL reference seeds; the
    seed-matched pairing uses the seeds that both arms have; the cross-seed pairing uses every (reference seed, other seed) pair."""
    sr, Ar, Tr, Lr = ref; so, Ao, To, Lo = oth
    assert len(sr) >= 3, "the placebo needs at least 3 reference seeds (6 ordered pairs)"
    common = [s for s in so if s in sr]
    rr = [sr.index(s) for s in common]; ro = [so.index(s) for s in common]
    n = Ar.shape[1]; idx = boot_idx(n)
    spl, npl = sp.placebo_sums(Ar, Tr); pi = spl.sum() / npl.sum(); bp = spl[idx].sum(1) / npl[idx].sum(1)
    pc = [(r, o) for r in rr for o in range(len(so)) if sr[r] != so[o]]; pm = list(zip(rr, ro))      # cross-seed: different seeds only
    sx, nx = sp.item_sums(Ar, Tr, To, pc); sm, nm = sp.item_sums(Ar, Tr, To, pm)
    bx = sx[idx].sum(1) / nx[idx].sum(1); bm = sm[idx].sum(1) / nm[idx].sum(1)
    dA = (Ao[ro].mean(0) - Ar[rr].mean(0)) * 100; dT = To[ro].mean(0) - Tr[rr].mean(0)
    # placebo-free item-matched difference in the mean length of CORRECT runs (Proposition 2 of the draft)
    kb, kq = Ar.sum(0), Ao.sum(0); ok = (kb > 0) & (kq > 0)
    d = np.where(ok, (Ao * To).sum(0) / np.maximum(kq, 1) - (Ar * Tr).sum(0) / np.maximum(kb, 1), 0.0)
    crd = d.sum() / ok.sum(); crd_ci = pct(d[idx].sum(1) / ok[idx].sum(1))

    def ndr(Ax, Ay, pairs):
        nn = np.zeros(Ax.shape[1]); dd = np.zeros(Ax.shape[1])
        for r, o in pairs: nn += Ax[r] * (1 - Ay[o]); dd += Ax[r]
        return nn, dd
    nb, db = ndr(Ar, Ar, [(r, o) for r in range(len(sr)) for o in range(len(sr)) if r != o]); ne, de = ndr(Ar, Ao, pm)
    flip_ci = pct((ne[idx].sum(1) / de[idx].sum(1) - nb[idx].sum(1) / db[idx].sum(1)) * 100)
    # finished-only Lambda: drop (pair, item) cases in which the OTHER run stopped at the length limit
    fo_n = np.zeros(n); fo_d = np.zeros(n)
    for r, o in pc:
        keep = Ar[r] * (1 - Lo[o]); fo_n += keep * (To[o] - Tr[r]); fo_d += keep
    res = dict(arm=name, seeds_ref=sr, seeds_oth=so, acc_pp=float(dA.mean()), acc_ci=pct(dA[idx].mean(1)),
               uncond=float(dT.mean()), uncond_ci=pct(dT[idx].mean(1)),
               placebo=float(pi), placebo_ci=pct(bp),
               lam_cross=float(sx.sum() / nx.sum()), lam_cross_ci=pct(bx), lam_matched=float(sm.sum() / nm.sum()), lam_matched_ci=pct(bm),
               excess_cross=float(sx.sum() / nx.sum() - pi), excess_cross_ci=pct(bx - bp),
               excess_matched=float(sm.sum() / nm.sum() - pi), excess_matched_ci=pct(bm - bp),
               lam_cross_finished_only=float(fo_n.sum() / max(fo_d.sum(), 1)),
               crd=float(crd), crd_ci=crd_ci, flip_baseline=float(nb.sum() / db.sum() * 100), flip_other=float(ne.sum() / de.sum() * 100),
               flip_excess_ci=flip_ci, cap_share_ref=float(Lr.mean()), cap_share_oth=float(Lo.mean()),
               cap_share_oth_per_seed={int(s): float(Lo[i].mean()) for i, s in enumerate(so)})
    res["accuracy_changed"] = excl0(res["acc_ci"])
    res["loss_detected"] = bool(excl0(res["acc_ci"]) and res["acc_pp"] < 0)
    res["excess_detected"] = excl0(res["excess_cross_ci"])
    res["truncation_dominated"] = bool(res["cap_share_oth"] > 0.30 or (res["cap_share_ref"] or 0) > 0.30)
    return res


def print_contrast(r):
    print(f"\n=== {r['arm']}  (reference seeds {r['seeds_ref']}, other seeds {r['seeds_oth']})")
    print("accuracy diff (pp)          ", fmt(r["acc_pp"], r["acc_ci"], 2), "  excludes 0:", r["accuracy_changed"])
    print("unconditional length diff   ", fmt(r["uncond"], r["uncond_ci"]))
    print("placebo Pi_b                ", fmt(r["placebo"], r["placebo_ci"]))
    print("Lambda_b cross-seed         ", fmt(r["lam_cross"], r["lam_cross_ci"]), "| matched", fmt(r["lam_matched"], r["lam_matched_ci"]))
    print("excess cross-seed           ", fmt(r["excess_cross"], r["excess_cross_ci"]), "  excludes 0:", r["excess_detected"], "| matched", fmt(r["excess_matched"], r["excess_matched_ci"]))
    print("correct-run difference      ", fmt(r["crd"], r["crd_ci"]))
    print(f"flips (descriptive)         baseline {r['flip_baseline']:.2f}%  other same-seed {r['flip_other']:.2f}%  excess {fmt(r['flip_other'] - r['flip_baseline'], r['flip_excess_ci'], 2)}")
    ref_cap = "n/a (not stored)" if r["cap_share_ref"] is None else f"{r['cap_share_ref']:.3f}"
    print(f"cap share: reference {ref_cap}, other {r['cap_share_oth']:.3f}; Lambda_b cross-seed, finished only: {r['lam_cross_finished_only']:+.0f}")
    if r["truncation_dominated"]: print("NOTE: more than 30% of the completions of an arm stop at the length limit -> report the length analysis as truncation-dominated")


# ---------------------------------------------------------------------------------------------- E4 (1.5B) and D1-lite
def cmd_e4(a):
    ref = load_cells(glob.glob(a.bf16)); oth = load_cells(glob.glob(a.awq))
    print(f"E4: BF16 seeds {ref[0]}, AWQ-4 seeds {oth[0]}")
    r = contrast(ref, oth, "E4 self-AWQ-4 vs BF16")
    print_contrast(r)
    print("RULE 2 (accuracy):", "loss detected" if r["loss_detected"] else "not detected")
    print("RULE 3 (excess, cross-seed):", "excess detected" if r["excess_detected"] else "not detected")
    (OUT / "rules_e4.json").write_text(json.dumps(r, indent=1))


def cmd_d1lite(a):
    import revision_sensitivities as rs
    A5, T5 = rs.load_cell("math500", "Qwen-7B", "BF16")                      # campaign BF16, seeds 42-46
    ref = ([42, 43, 44, 45, 46], A5, T5, np.zeros_like(A5, bool))              # finish_reason is not stored in the campaign compact files
    w3 = load_cells(glob.glob(a.w3)); print(f"D1-lite: W3 seeds {w3[0]}")
    r = contrast(ref, w3, "D1-lite GPTQ-3 (float16) vs campaign BF16"); r["cap_share_ref"] = None; print_contrast(r)
    out = {"vs_campaign_bf16": r}
    if len(w3[0]) < 3: print("NOTE: fewer than 3 W3 seeds -> report accuracy and length only; no placebo statement")
    print("RULE 1 (accuracy vs campaign BF16):", "loss detected" if r["loss_detected"] else "not detected")
    print("RULE 3 (excess, cross-seed):", "excess detected" if r["excess_detected"] else "not detected")
    if a.e2:
        e2 = load_cells(glob.glob(a.e2)); r2 = contrast(e2, w3, "D1-lite GPTQ-3 vs E2 float16 BF16 (same dtype)"); print_contrast(r2)
        print("RULE 1 (accuracy vs E2 float16):", "loss detected" if r2["loss_detected"] else "not detected"); out["vs_e2_float16"] = r2
    (OUT / "rules_d1lite.json").write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------------------------- G1 (greedy configuration placebo)
def ctir_boot(Ar, Tr, To, idx):
    """CTIR over all items / items the reference answers correctly / wrongly, as ratio of sums, with bootstrap draws."""
    d, m = To - Tr, {"all": np.ones_like(Ar), "correct": Ar, "wrong": 1 - Ar}
    out = {}
    for k, mk in m.items():
        pt = 100 * (mk * d).sum() / (mk * Tr).sum()
        bs = 100 * (mk * d)[idx].sum(1) / (mk * Tr)[idx].sum(1)
        out[k] = (float(pt), bs)
    return out


def classify(path):
    n = Path(path).name
    q = "FP8" if "FP8" in n else "AWQ-4" if "AWQ" in n else "GPTQ-4" if "GPTQ" in n else "BF16"
    fam = "Qwen-7B" if "Qwen" in n else "Llama-8B" if "Llama" in n else "?"
    return fam, q


def cmd_g1(a):
    cells = {}
    for f in glob.glob(str(Path(a.x5) / "*.json")):
        fam, _ = classify(f); mns = re.search(r"mns(\d+)", Path(f).name)
        if mns: cells.setdefault(fam, {})["R0" if mns.group(1) == "256" else "R1" if mns.group(1) == "64" else "Rx" + mns.group(1)] = f
    if a.g1:
        for f in glob.glob(str(Path(a.g1) / "*.json")):
            fam, q = classify(f)
            cells.setdefault(fam, {})["R2" if q == "BF16" else q] = f
    out = {}
    for fam, c in sorted(cells.items()):
        print(f"\n######## G1 {fam}: cells found {sorted(c)}")
        data = {k: load_cells([v]) for k, v in c.items()}
        n = next(iter(data.values()))[1].shape[1]; idx = boot_idx(n)
        for k, (s, A, T, L) in data.items(): print(f"  {k}: pass@1 {A.mean():.3f}  mean tokens {T.mean():.0f}  stopped at the length limit {L.mean():.3f}")
        Rs = [k for k in ("R0", "R1", "R2") if k in data]
        pairs = [(r, o) for r in Rs for o in Rs if r != o]; pl = {}
        for r, o in pairs:
            t = ctir_boot(data[r][1][0], data[r][2][0], data[o][2][0], idx); pl[(r, o)] = t
            print(f"  placebo {r}->{o}: CTIR_all {fmt(t['all'][0], pct(t['all'][1]), 1)}%  correct {fmt(t['correct'][0], pct(t['correct'][1]), 1)}%  wrong {fmt(t['wrong'][0], pct(t['wrong'][1]), 1)}%")
        res = {"cells": sorted(c), "pairs": {f"{r}->{o}": {k: v[0] for k, v in t.items()} for (r, o), t in pl.items()}}
        if len(Rs) < 3:
            print(f"  INCOMPLETE: the placebo set needs R0, R1 and R2; found {Rs}. No verdict is applied.")
        else:
            mean = {k: float(np.mean([t[k][0] for t in pl.values()])) for k in ("all", "correct", "wrong")}
            rng_ = {k: [float(min(t[k][0] for t in pl.values())), float(max(t[k][0] for t in pl.values()))] for k in mean}
            print("  mean over the six placebo pairs:", {k: round(v, 1) for k, v in mean.items()}, "range:", {k: [round(x, 1) for x in v] for k, v in rng_.items()})
            produced = mean["correct"] > 0 and mean["wrong"] < 0
            print("  RULE 2:", "the published pattern IS produced by configuration placebos alone" if produced else "not produced")
            res.update(mean=mean, range=rng_, rule2_produced=bool(produced))
            for q in ("FP8", "AWQ-4", "GPTQ-4"):
                if q not in data: print(f"  {q}: INCOMPLETE (cell missing)"); continue
                t = ctir_boot(data["R0"][1][0], data["R0"][2][0], data[q][2][0], idx); dec = {}
                for k in ("correct", "all"):
                    base = np.mean([pl[("R0", "R1")][k][1], pl[("R0", "R2")][k][1]], axis=0)         # placebo pairs with R0 as reference, same resamples
                    dpt = t[k][0] - float(np.mean([pl[("R0", "R1")][k][0], pl[("R0", "R2")][k][0]]))
                    ci = pct(t[k][1] - base); dec[k] = dict(diff=dpt, ci=ci, detected=excl0(ci))
                    print(f"  RULE 3 {q} CTIR_{k}: Q {t[k][0]:+.1f}%, minus placebo {fmt(dpt, ci, 1)}  ->", "inflation beyond the placebo detected" if excl0(ci) else "not detected")
                print(f"  {q}: pass@1 {data[q][1].mean():.3f}, stopped at the length limit {data[q][3].mean():.3f}")
                res[q] = dec
        out[fam] = res
    (OUT / "rules_g1.json").write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------------------------- X2 (placebo on a hard benchmark)
def cmd_x2(a):
    by = {}
    for f in glob.glob(str(Path(a.dir) / "*.json")):
        ds = "aime2024" if "2024" in Path(f).name else "aime2025" if "2025" in Path(f).name else None
        if ds: by.setdefault(ds, []).append(f)
    assert len(by) == 2, f"expected both AIME sets in {a.dir}, found {sorted(by)}"
    sets = {k: load_cells(v) for k, v in by.items()}
    assert sets["aime2024"][0] == sets["aime2025"][0], "seed sets differ between the two AIME sets"
    seeds = sets["aime2024"][0]; assert len(seeds) >= 4, "never fewer than 4 seeds"
    A = np.hstack([sets[k][1] for k in sorted(sets)]); T = np.hstack([sets[k][2] for k in sorted(sets)]); L = np.hstack([sets[k][3] for k in sorted(sets)])
    n = A.shape[1]; idx = boot_idx(n)
    spl, npl = sp.placebo_sums(A, T); pi = spl.sum() / npl.sum(); bp = spl[idx].sum(1) / npl[idx].sum(1)
    mean_len = T.mean(); k = A.sum(0)
    nb = np.zeros(n); db = np.zeros(n)
    for r in range(len(seeds)):
        for o in range(len(seeds)):
            if r != o: nb += A[r] * (1 - A[o]); db += A[r]
    res = dict(seeds=seeds, n_items=int(n), placebo=float(pi), placebo_ci=pct(bp), placebo_pct_of_mean=float(pi / mean_len * 100),
               placebo_pct_ci=[float(v / mean_len * 100) for v in pct(bp)], intermediate_share=float(((k > 0) & (k < len(seeds))).mean()),
               stopped_at_limit_share=float(L.mean()), mean_completion_tokens=float(mean_len), ndr_flip_placebo=float(nb.sum() / max(db.sum(), 1) * 100),
               pass_at_1=float(A.mean()), detected=excl0(pct(bp)))
    print(f"X2 {Path(a.dir).name}: {len(seeds)} seeds {seeds}, {n} items")
    print("RULE 1 placebo Pi_b       ", fmt(pi, res["placebo_ci"]), f"= {res['placebo_pct_of_mean']:.1f}% of mean length (MATH-500 reference: 3.4% Qwen, 5.1% Llama)")
    print(f"RULE 2 intermediate share {res['intermediate_share']:.3f}, stopped at the limit {res['stopped_at_limit_share']:.3f}, mean length {mean_len:.0f}, NDR flip placebo {res['ndr_flip_placebo']:.2f}%, pass@1 {res['pass_at_1']:.3f}")
    print("RULE 3:", "placebo detected" if res["detected"] else "not detected at n=60")
    (OUT / f"rules_x2_{Path(a.dir).name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("e4"); s.add_argument("--bf16", required=True); s.add_argument("--awq", required=True); s.set_defaults(f=cmd_e4)
    s = sub.add_parser("d1lite"); s.add_argument("--w3", required=True); s.add_argument("--e2"); s.set_defaults(f=cmd_d1lite)
    s = sub.add_parser("g1"); s.add_argument("--x5", required=True); s.add_argument("--g1"); s.set_defaults(f=cmd_g1)
    s = sub.add_parser("x2"); s.add_argument("--dir", required=True); s.set_defaults(f=cmd_x2)
    a = ap.parse_args(); a.f(a)
