#!/usr/bin/env python3
"""Independent re-check of the new numbers in paper_v2.tex.

Deliberately does NOT import seed_placebo.py, revision_sensitivities.py or any of the analysis scripts. It reads the raw per-item JSON
itself, uses explicit loops for the conditional estimators, gets the placebo from the closed-form identity (and checks it against the
pair loops), and bootstraps with a different seed and twice the resamples.

Stage A: recompute every new result and compare with the saved results JSON (point estimates exactly, interval endpoints within
         Monte-Carlo tolerance, and the include-zero decision exactly).
Stage B: find every  value [lo,hi]  triplet in the paper text and tables and match it to the saved results (exact after rounding).
"""
import csv, glob, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CAMP = ROOT / "results"
EXP = V3 = ROOT / "results" / "followup"
V4 = ROOT / "results" / "followup" / "p2p4"
PROV = ROOT / "results" / "followup" / "provenance"
OUTD = ROOT / "results" / "followup" / "reports"
B = 20_000
RNG = np.random.default_rng(987654321)
FAIL, OK = [], []


# ---------------------------------------------------------------- loading (own code)
def rows_of(path):
    d = json.load(open(path, encoding="utf-8"))
    return sorted(d["details"], key=lambda r: r["row"])


def load_plain(path):
    r = rows_of(path)
    return np.array([x["extractive_match"] for x in r], float), np.array([x["completion_tokens"] for x in r], float)


def camp(bench, fam, fmt, seeds):
    tag = "math500_n500" if bench == "math500" else "gsm8k_n1319"
    stem = f"DeepSeek-R1-Distill-{fam}" + ("" if fmt == "BF16" else "-" + fmt)
    out = [load_plain(CAMP / bench / f"{stem}_{tag}_seed{s}.json") for s in seeds]
    return np.array([a for a, t in out]), np.array([t for a, t in out])


def stack(paths):
    out = [load_plain(p) for p in paths]
    return np.array([a for a, t in out]), np.array([t for a, t in out])


MITEM = {r["unique_id"]: int(r["row"]) - 1 for r in csv.DictReader(open(PROV / "MATH500_item_table.tsv"), delimiter="\t")}
GITEM = {int(r["dataset_index"]): int(r["campaign_row"]) - 1 for r in csv.DictReader(open(PROV / "GSM8K_item_table.tsv"), delimiter="\t")}
INV = list(csv.DictReader(open(V4 / "tables" / "cell_inventory.tsv"), delimiter="\t"))


def p2cell(exp, variant_prefix, seed, bench):
    r = [x for x in INV if x["experiment_id"] == exp and x["variant_id"].startswith(variant_prefix) and x["seed"] == str(seed)]
    assert len(r) == 1, (exp, variant_prefix, seed, len(r))
    d = json.load(open(V4 / "compact" / Path(r[0]["compact_file"]).name, encoding="utf-8"))
    n = 500 if bench == "math500" else 1319
    A = np.full(n, np.nan); T = np.full(n, np.nan)
    for x in d["details"]:
        i = MITEM[x["item_id"]] if bench == "math500" else GITEM[x["dataset_index"]]
        A[i] = x["extractive_match"]; T[i] = x["completion_tokens"]
    assert not np.isnan(A).any(), "unmapped items"
    return A, T


# ---------------------------------------------------------------- estimators (own formulation)
def pair_sums_loop(Aref, Tref, Toth, pairs):
    N = Aref.shape[1]; num = np.zeros(N); den = np.zeros(N)
    for (r, o) in pairs:
        for i in range(N):                                           # explicit loops on purpose
            num[i] += Aref[r][i] * (Toth[o][i] - Tref[r][i]); den[i] += Aref[r][i]
    return num, den


def placebo_identity(A, T):
    """Pi_b from the closed form  sum_i k_i (m-k_i)(mu0-mu1) / ((m-1) sum_i k_i)."""
    m, N = A.shape; num = 0.0; den = 0.0
    for i in range(N):
        k = int(A[:, i].sum()); den += k
        if 0 < k < m:
            mu1 = T[A[:, i] == 1, i].mean(); mu0 = T[A[:, i] == 0, i].mean()
            num += k * (m - k) * (mu0 - mu1)
    return num / ((m - 1) * den)


def placebo_loop(A, T):
    m = A.shape[0]
    return pair_sums_loop(A, T, T, [(r, o) for r in range(m) for o in range(m) if r != o])


def pct(x): return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def arm(Aref, Tref, Aoth, Toth, plA, plT, pairs):
    """Lambda_b, placebo and excess (shared item resamples), accuracy and unconditional differences."""
    N = Aref.shape[1]
    sl, nl = pair_sums_loop(Aref, Tref, Toth, pairs)
    sp, npl = placebo_loop(plA, plT)
    pi = sp.sum() / npl.sum(); assert abs(pi - placebo_identity(plA, plT)) < 1e-8, "identity mismatch"
    idx = RNG.integers(0, N, (B, N))
    bl = sl[idx].sum(1) / nl[idx].sum(1); bp = sp[idx].sum(1) / npl[idx].sum(1)
    lam = sl.sum() / nl.sum()
    dA = (Aoth.mean(0) - Aref.mean(0)) * 100; dT = Toth.mean(0) - Tref.mean(0)
    return dict(lam=lam, lam_ci=pct(bl), pi=pi, pi_ci=pct(bp), excess=lam - pi, excess_ci=pct(bl - bp),
                acc=float(dA.mean()), acc_ci=pct(dA[idx].mean(1)), unc=float(dT.mean()), unc_ci=pct(dT[idx].mean(1)))


# ---------------------------------------------------------------- comparison bookkeeping
def chk(label, mine, ref, mine_ci=None, ref_ci=None):
    ok = abs(mine - ref) <= max(1e-6, 1e-7 * abs(ref))
    msg = f"{label}: point {mine:+.3f} vs saved {ref:+.3f}"
    if mine_ci is not None:
        for j, (a, b) in enumerate(zip(mine_ci, ref_ci)):
            w = abs(ref_ci[1] - ref_ci[0]) + 1e-9
            if abs(a - b) > 0.06 * w: ok = False; msg += f" | CI end {j}: {a:+.2f} vs {b:+.2f}"
        if (mine_ci[0] <= 0 <= mine_ci[1]) != (ref_ci[0] <= 0 <= ref_ci[1]): ok = False; msg += " | include-zero decision DIFFERS"
    (OK if ok else FAIL).append(label if ok else msg)


def J(name): return json.load(open(OUTD / name, encoding="utf-8"))


FAMS = {"Qwen-7B": "DeepSeek-R1-Distill-Qwen-7B", "Llama-8B": "DeepSeek-R1-Distill-Llama-8B"}
R2 = J("revision2_results.json"); EX = J("e1c_x3_results.json"); P2 = J("p2_qwen_results.json"); P2G = J("p2_qwen_gsm8k_results.json"); P4 = J("p4_decoding_results.json")


def cmp_r2(label, mine, ref, kind="matched"):
    if kind == "matched":
        chk(f"{label} Lambda(matched)", mine["lam"], ref["lam_m"], mine["lam_ci"], ref["lam_m_ci"])
        chk(f"{label} excess(matched)", mine["excess"], ref["excess_m"], mine["excess_ci"], ref["excess_m_ci"])
    else:
        chk(f"{label} Lambda(cross)", mine["lam"], ref["lam_b"], mine["lam_ci"], ref["lam_ci"])
        chk(f"{label} excess(cross)", mine["excess"], ref["excess"], mine["excess_ci"], ref["excess_ci"])
    chk(f"{label} placebo", mine["pi"], ref["pi_b"], mine["pi_ci"], ref["pi_ci"])
    chk(f"{label} accuracy diff pp", mine["acc"], ref["acc_diff_pp"], mine["acc_ci"], ref["acc_diff_ci"])
    chk(f"{label} uncond tokens", mine["unc"], ref["uncond_tokens"], mine["unc_ci"], ref["uncond_ci"])


def cross_pairs(m_ref, m_oth): return [(r, o) for r in range(m_ref) for o in range(m_oth) if r != o]
def matched_pairs(m): return [(s, s) for s in range(m)]


# ================================================================= STAGE A
print("Stage A: independent recomputation ...", flush=True)
BF = {f: camp("math500", f, "BF16", range(42, 47)) for f in FAMS}
X1D = EXP / "revision3" / "compact" / "X1_heldout"

# --- E2 (fp16), E3 (T=0.8): both families
for fam, stem in FAMS.items():
    A, T = BF[fam]
    for name, folder, tag in (("E2", "revision2/compact/E2_fp16", "fp16"), ("E3", "revision2/compact/E3_T08", "T0.8")):
        Ao, To = stack([EXP / folder / f"{stem}_{tag}_math500_n500_seed{s}.json" for s in range(42, 47)])
        cmp_r2(f"{name} {fam}", arm(A, T, Ao, To, A, T, matched_pairs(5)), R2[name][fam], "matched")
        cmp_r2(f"{name} {fam}", arm(A, T, Ao, To, A, T, cross_pairs(5, 5)), R2[name][fam], "cross")
print("   E2/E3 done", flush=True)

# --- E1 (self-AWQ Llama): MATH-500 and GSM8K, plus community comparison
A, T = BF["Llama-8B"]
sa, st = stack([EXP / "revision2/compact/E1_awq" / f"DeepSeek-R1-Distill-Llama-8B-selfawq_math500_n500_seed{s}.json" for s in range(42, 47)])
cmp_r2("E1 self-AWQ MATH", arm(A, T, sa, st, A, T, matched_pairs(5)), R2["E1"]["math500"], "matched")
cmp_r2("E1 self-AWQ MATH", arm(A, T, sa, st, A, T, cross_pairs(5, 5)), R2["E1"]["math500"], "cross")
ca, ct = camp("math500", "Llama-8B", "AWQ-4", range(42, 47))
cmp_r2("community AWQ MATH", arm(A, T, ca, ct, A, T, matched_pairs(5)), R2["E1"]["math500_community"], "matched")
cmp_r2("community AWQ MATH", arm(A, T, ca, ct, A, T, cross_pairs(5, 5)), R2["E1"]["math500_community"], "cross")
dd = (sa.mean(0) - ca.mean(0)) * 100; ix = RNG.integers(0, 500, (B, 500))
chk("E1 self minus community pp", float(dd.mean()), R2["E1"]["selfawq_minus_community_pp"]["diff"], pct(dd[ix].mean(1)), R2["E1"]["selfawq_minus_community_pp"]["ci"])
Ag, Tg = camp("gsm8k", "Llama-8B", "BF16", range(42, 45))
ga, gt = stack([EXP / "revision2/compact/E1_awq" / f"DeepSeek-R1-Distill-Llama-8B-selfawq_gsm8k_n1319_seed{s}.json" for s in range(42, 45)])
cmp_r2("E1 self-AWQ GSM8K", arm(Ag, Tg, ga, gt, Ag, Tg, matched_pairs(3)), R2["E1"]["gsm8k"], "matched")
cg, ctg = camp("gsm8k", "Llama-8B", "AWQ-4", range(42, 45))
cmp_r2("community AWQ GSM8K", arm(Ag, Tg, cg, ctg, Ag, Tg, matched_pairs(3)), R2["E1"]["gsm8k_community"], "matched")
print("   E1 done", flush=True)

# --- per-level accuracy change (Llama, MATH-500)
lev = {int(r["row"]) - 1: int(r["level"]) for r in csv.DictReader(open(PROV / "MATH500_item_table.tsv"), delimiter="\t")}
levels = np.array([lev[i] for i in range(500)])
for arm_name, AA in (("self", sa), ("community", ca)):
    for g, mask in (("L1", levels == 1), ("L2", levels == 2), ("L3", levels == 3), ("L4", levels == 4), ("L5", levels == 5), ("L1-2", levels <= 2), ("L3-5", levels >= 3)):
        d = (AA[:, mask].mean(0) - A[:, mask].mean(0)) * 100; ix = RNG.integers(0, int(mask.sum()), (B, int(mask.sum()))); ref = R2["levels"][arm_name][g]
        chk(f"level {arm_name} {g}", float(d.mean()), ref["diff"], pct(d[ix].mean(1)), ref["ci"])

# --- E1c and X3 (pre-registered rules)
c_a, c_t = stack(sorted(glob.glob(str(V3 / "revision2/compact/E1c_awq_gemm/*seed4[2-6].json"))))
e1c_m = arm(A, T, c_a, c_t, A, T, matched_pairs(5)); e1c_c = arm(A, T, c_a, c_t, A, T, cross_pairs(5, 5)); E = EX["E1c"]
chk("E1c accuracy diff", e1c_m["acc"], E["acc_diff_pp"], e1c_m["acc_ci"], E["acc_diff_ci"])
chk("E1c Lambda matched", e1c_m["lam"], E["lam_m"], e1c_m["lam_ci"], E["lam_m_ci"]); chk("E1c excess matched", e1c_m["excess"], E["excess"], e1c_m["excess_ci"], E["excess_ci"])
chk("E1c Lambda cross (pre-registered)", e1c_c["lam"], E["cross_seed"]["lam"], e1c_c["lam_ci"], E["cross_seed"]["lam_ci"])
chk("E1c excess cross (pre-registered)", e1c_c["excess"], E["cross_seed"]["excess"], e1c_c["excess_ci"], E["cross_seed"]["excess_ci"])
if not (e1c_c["excess_ci"][0] > 0): FAIL.append("E1c: cross-seed excess no longer excludes zero (verdict changes!)")
Aq, Tq = BF["Qwen-7B"]
x_a, x_t = stack(sorted(glob.glob(str(V3 / "revision3/compact/X3_batch/*seed4[2-6].json"))))
x_c = arm(Aq, Tq, x_a, x_t, Aq, Tq, cross_pairs(5, 5)); x_m = arm(Aq, Tq, x_a, x_t, Aq, Tq, matched_pairs(5)); X = EX["X3"]
chk("X3 Lambda cross", x_c["lam"], X["lam_cross"], x_c["lam_ci"], X["lam_cross_ci"]); chk("X3 excess cross (pre-registered)", x_c["excess"], X["excess_cross"], x_c["excess_ci"], X["excess_cross_ci"])
chk("X3 Lambda matched", x_m["lam"], X["lam_matched"], x_m["lam_ci"], X["lam_matched_ci"]); chk("X3 excess matched", x_m["excess"], X["excess_matched"], x_m["excess_ci"], X["excess_matched_ci"])
chk("X3 accuracy diff", x_c["acc"], X["acc_diff_pp"], x_c["acc_ci"], X["acc_diff_ci"]); chk("X3 uncond", x_c["unc"], X["uncond_tokens"], x_c["unc_ci"], X["uncond_ci"])
print("   E1c/X3 done", flush=True)

# --- X1 (held-out seeds): placebo old/new/pooled via the identity formula, paired difference
for fam, stem in FAMS.items():
    A_, T_ = BF[fam]; An, Tn = stack([X1D / f"{stem}_heldout_math500_n500_seed{s}.json" for s in range(47, 52)])
    po, pn = placebo_identity(A_, T_), placebo_identity(An, Tn); pall = placebo_identity(np.vstack([A_, An]), np.vstack([T_, Tn])); x = R2["X1"][fam]
    chk(f"X1 {fam} Pi_old", po, x["pi_old"]); chk(f"X1 {fam} Pi_new", pn, x["pi_new"]); chk(f"X1 {fam} Pi_pooled10", pall, x["pi_pooled10"])
    so, no_ = placebo_loop(A_, T_); sn, nn = placebo_loop(An, Tn); idx = RNG.integers(0, 500, (B, 500))
    chk(f"X1 {fam} diff", pn - po, x["diff"], pct(sn[idx].sum(1) / nn[idx].sum(1) - so[idx].sum(1) / no_[idx].sum(1)), x["diff_ci"])
    for nm, AA in (("old", A_), ("new", An)):
        k = AA.sum(0); chk(f"X1 {fam} intermediate share {nm}", float(((k > 0) & (k < 5)).mean()), x["intermediate_" + nm])
print("   X1 done", flush=True)

# --- excess over the 10-seed placebo for the six campaign arms
for fam, stem in FAMS.items():
    A_, T_ = BF[fam]; An, Tn = stack([X1D / f"{stem}_heldout_math500_n500_seed{s}.json" for s in range(47, 52)])
    A10, T10 = np.vstack([A_, An]), np.vstack([T_, Tn])
    for f in ("FP8", "AWQ-4", "GPTQ-4"):
        qa, qt = camp("math500", fam, f, range(42, 47)); r = R2["excess10"][f"{fam} {f}"]
        sl, nl = pair_sums_loop(A_, T_, qt, matched_pairs(5)); sp, npl = placebo_loop(A10, T10); idx = RNG.integers(0, 500, (B, 500))
        b = sl[idx].sum(1) / nl[idx].sum(1) - sp[idx].sum(1) / npl[idx].sum(1); lam = sl.sum() / nl.sum(); pi10 = sp.sum() / npl.sum()
        chk(f"excess10 {fam} {f}", lam - pi10, r["excess"], pct(b), r["ci"]); chk(f"pi10 {fam} {f}", pi10, r["pi10"])
print("   excess vs 10-seed placebo done", flush=True)

# --- P2 Qwen self-quantized (MATH-500, GSM8K), community reference, draw spread, dtype control
for bench, J2, seeds_ref in (("math500", P2, range(42, 47)), ("gsm8k", P2G, range(42, 45))):
    Ab5, Tb5 = camp(bench, "Qwen-7B", "BF16", seeds_ref); Ab, Tb = Ab5[:3], Tb5[:3]; N = Ab.shape[1]
    exp_pub = f"p2_publication_qwen7b_{bench}_v1"; exp_dt = f"p2_dtype_control_qwen7b_{bench}_v1"
    chk(f"P2 {bench} placebo (identity)", placebo_identity(Ab5, Tb5), J2["placebo_5seed"] if bench == "math500" else J2["placebo_3seed"])
    arms = {}
    for method in ("awq4", "gptq4"):
        arms[method] = []
        for c in range(3):
            cells = [p2cell(exp_pub, f"{method}_cal{c}", s, bench) for s in (42, 43, 44)]
            arms[method].append((np.array([a for a, t in cells]), np.array([t for a, t in cells])))
        for c in range(3):
            Ao, To = arms[method][c]; mine = arm(Ab, Tb, Ao, To, Ab5, Tb5, matched_pairs(3)); ref = J2[method][f"cal{c}"]
            chk(f"P2 {bench} {method} cal{c} Lambda", mine["lam"], ref["lam"], mine["lam_ci"], ref["lam_ci"]); chk(f"P2 {bench} {method} cal{c} excess", mine["excess"], ref["excess"], mine["excess_ci"], ref["excess_ci"])
            chk(f"P2 {bench} {method} cal{c} acc", mine["acc"], ref["acc_pp"], mine["acc_ci"], ref["acc_ci"]); chk(f"P2 {bench} {method} cal{c} uncond", mine["unc"], ref["uncond"], mine["unc_ci"], ref["uncond_ci"])
        Ap = np.vstack([a for a, t in arms[method]]); Tp = np.vstack([t for a, t in arms[method]])
        mine = arm(Ab, Tb, Ap, Tp, Ab5, Tb5, [(s, c * 3 + s) for c in range(3) for s in range(3)]); ref = J2[method]["pooled"]
        chk(f"P2 {bench} {method} pooled excess", mine["excess"], ref["excess"], mine["excess_ci"], ref["excess_ci"]); chk(f"P2 {bench} {method} pooled acc", mine["acc"], ref["acc_pp"], mine["acc_ci"], ref["acc_ci"])
        chk(f"P2 {bench} {method} pooled uncond", mine["unc"], ref["uncond"], mine["unc_ci"], ref["uncond_ci"])
        for i, j in ((0, 1), (0, 2), (1, 2)):
            ds = J2["draw_spread"][f"{method} cal{i}-cal{j}"]; dt = arms[method][i][1].mean(0) - arms[method][j][1].mean(0); da = (arms[method][i][0].mean(0) - arms[method][j][0].mean(0)) * 100
            ix = RNG.integers(0, N, (B, N)); chk(f"P2 {bench} {method} draw{i}-{j} tokens", float(dt.mean()), ds["tokens"], pct(dt[ix].mean(1)), ds["tokens_ci"]); chk(f"P2 {bench} {method} draw{i}-{j} acc", float(da.mean()), ds["acc_pp"], pct(da[ix].mean(1)), ds["acc_ci"])
    for f in ("AWQ-4", "GPTQ-4"):
        qa, qt = camp(bench, "Qwen-7B", f, range(42, 45)); mine = arm(Ab, Tb, qa, qt, Ab5, Tb5, matched_pairs(3)); ref = J2["community_" + f]
        chk(f"P2 {bench} community {f} excess", mine["excess"], ref["excess"], mine["excess_ci"], ref["excess_ci"]); chk(f"P2 {bench} community {f} acc", mine["acc"], ref["acc_pp"], mine["acc_ci"], ref["acc_ci"])
    for c in range(3):
        cells = [p2cell(exp_dt, f"gptq4_cal{c}", s, bench) for s in (42, 43, 44)]; fa, ft = np.array([a for a, t in cells]), np.array([t for a, t in cells]); ba, bt = arms["gptq4"][c]
        da = (fa.mean(0) - ba.mean(0)) * 100; dt = ft.mean(0) - bt.mean(0); ix = RNG.integers(0, N, (B, N)); ref = J2["dtype"][f"cal{c}"]
        chk(f"P2 {bench} dtype cal{c} acc", float(da.mean()), ref["acc_pp"], pct(da[ix].mean(1)), ref["acc_ci"]); chk(f"P2 {bench} dtype cal{c} tokens", float(dt.mean()), ref["tokens"], pct(dt[ix].mean(1)), ref["tokens_ci"])
    print(f"   P2 {bench} done", flush=True)

# --- P4 decoding grid (300 items, reference campaign BF16 restricted to the same items)
p4 = [x for x in INV if x["experiment_id"] == "p4_decoding_qwen7b_math300_rep3_v1"]
first = json.load(open(V4 / "compact" / Path([x for x in p4 if x["variant_id"] == "bf16_t1.0_b8k" and x["seed"] == "42"][0]["compact_file"]).name))
order = [MITEM[z["item_id"]] for z in sorted(first["details"], key=lambda z: z["row"])]; Ab5, Tb5 = BF["Qwen-7B"]; Ab5, Tb5 = Ab5[:, order], Tb5[:, order]; Ab, Tb = Ab5[:3], Tb5[:3]
chk("P4 placebo on 300 items", placebo_identity(Ab5, Tb5), P4["placebo"])
for variant in [k for k in P4 if k not in ("placebo", "placebo_ci", "n_items")]:
    A_, T_, F_ = [], [], []
    for s in (42, 43, 44):
        r = [x for x in p4 if x["variant_id"] == variant and x["seed"] == str(s)][0]; d = sorted(json.load(open(V4 / "compact" / Path(r["compact_file"]).name))["details"], key=lambda z: z["row"])
        assert [MITEM[z["item_id"]] for z in d] == order; A_.append([z["extractive_match"] for z in d]); T_.append([z["completion_tokens"] for z in d]); F_.append([z["finish_reason"] == "length" for z in d])
    Ao, To = np.array(A_, float), np.array(T_, float); mine = arm(Ab, Tb, Ao, To, Ab5, Tb5, matched_pairs(3)); ref = P4[variant]
    chk(f"P4 {variant} excess", mine["excess"], ref["excess"], mine["excess_ci"], ref["excess_ci"]); chk(f"P4 {variant} acc", mine["acc"], ref["acc"], mine["acc_ci"], ref["acc_ci"]); chk(f"P4 {variant} stop share", float(np.mean(F_)), ref["length_stop_share"])
print("   P4 done", flush=True)

# --- length variance table
LV = J("lenvar.json")
for fam, stem in FAMS.items():
    A_, T_ = BF[fam]; base = T_.std(0, ddof=1)
    sets = {"BF16, campaign seeds 42--46": T_, "BF16, held-out seeds 47--51 (placebo)": stack([X1D / f"{stem}_heldout_math500_n500_seed{s}.json" for s in range(47, 52)])[1],
            "BF16 at \\texttt{float16}": stack([EXP / "revision2/compact/E2_fp16" / f"{stem}_fp16_math500_n500_seed{s}.json" for s in range(42, 47)])[1],
            "BF16 at $T=0.8$": stack([EXP / "revision2/compact/E3_T08" / f"{stem}_T0.8_math500_n500_seed{s}.json" for s in range(42, 47)])[1]}
    for f in ("FP8", "AWQ-4", "GPTQ-4"): sets[f] = camp("math500", fam, f, range(42, 47))[1]
    for name, TT in sets.items():
        s = TT.std(0, ddof=1); ref = LV[f"{fam} | {name}"]; ix = RNG.integers(0, 500, (B, 500))
        chk(f"lenvar {fam} {name} mean SD", float(s.mean()), ref["mean_sd"], pct(s[ix].mean(1)), ref["ci"]); chk(f"lenvar {fam} {name} median", float(np.median(s)), ref["median_sd"])
        if "campaign" not in name: chk(f"lenvar {fam} {name} diff", float((s - base).mean()), ref["diff"], pct((s - base)[ix].mean(1)), ref["diff_ci"])
print("   length variance done", flush=True)

# --- numbers that were computed ad hoc and appear only in the text
A, T = BF["Llama-8B"]
err_nc = lambda AA, TT: ((1 - AA) * (TT >= 32500)).mean(0) * 100
err_fin = lambda AA, TT: ((1 - AA) * (TT < 32500)).mean(0) * 100
ix = RNG.integers(0, 500, (B, 500)); d1 = err_fin(sa, st) - err_fin(A, T); d2 = err_nc(sa, st) - err_nc(A, T)
for lbl, d, ref, rc in (("self-AWQ finished-error change (text +1.76 [0.44,3.08])", d1, 1.76, [0.44, 3.08]), ("self-AWQ near-cap change (text -0.08 [-0.52,0.32])", d2, -0.08, [-0.52, 0.32])):
    ok = abs(d.mean() - ref) < 0.006 and all(abs(a - b) < 0.12 for a, b in zip(pct(d[ix].mean(1)), rc))
    (OK if ok else FAIL).append(lbl if ok else f"{lbl}: {d.mean():+.3f} {pct(d[ix].mean(1))}")
def ndr(AR, AO, pairs):
    n = np.zeros(AR.shape[1]); dd_ = np.zeros(AR.shape[1])
    for r, o in pairs:
        n += AR[r] * (1 - AO[o]); dd_ += AR[r]
    return n, dd_
nb, db = ndr(A, A, cross_pairs(5, 5)); ns, ds = ndr(A, sa, matched_pairs(5)); fl = (ns.sum() / ds.sum() - nb.sum() / db.sum()) * 100
fci = pct((ns[ix].sum(1) / ds[ix].sum(1) - nb[ix].sum(1) / db[ix].sum(1)) * 100)
ok = abs(fl - 0.81) < 0.006 and all(abs(a - b) < 0.12 for a, b in zip(fci, [-0.40, 2.04]))
(OK if ok else FAIL).append("self-AWQ flip excess (text +0.81 [-0.40,+2.04])" if ok else f"self-AWQ flip excess {fl:+.3f} {fci}")
# X3 flips (text +0.34 [-0.36,+1.07]): same-seed X3 against the BF16-vs-BF16 baseline, shared resamples
nb, db = ndr(Aq, Aq, cross_pairs(5, 5)); ns, ds = ndr(Aq, x_a, matched_pairs(5)); fl3 = (ns.sum() / ds.sum() - nb.sum() / db.sum()) * 100
fci3 = pct((ns[ix].sum(1) / ds[ix].sum(1) - nb[ix].sum(1) / db[ix].sum(1)) * 100)
ok = abs(fl3 - 0.34) < 0.006 and all(abs(a - b) < 0.12 for a, b in zip(fci3, [-0.36, 1.07]))
(OK if ok else FAIL).append("X3 flip excess (text +0.34 [-0.36,+1.07])" if ok else f"X3 flip excess {fl3:+.3f} {fci3}")
ADHOC_VERIFIED = {(1.76, 0.44, 3.08), (-0.08, -0.52, 0.32), (0.81, -0.40, 2.04), (0.34, -0.36, 1.07)}   # checked above, not stored in a results file

# --- k-share of the selection term (text: old 77%/75%, new 81%/75% for Qwen/Llama; items solved in 3 or 4 of 5 seeds)
def kshare34(Am, Tm):
    m = Am.shape[0]; tot = 0.0; part = 0.0
    for i in range(Am.shape[1]):
        k = int(Am[:, i].sum())
        if 0 < k < m:
            v = k * (m - k) * (Tm[Am[:, i] == 0, i].mean() - Tm[Am[:, i] == 1, i].mean()); tot += v; part += v if k in (3, 4) else 0.0
    return part / tot
for fam, stem, want_old, want_new in (("Qwen-7B", "DeepSeek-R1-Distill-Qwen-7B", 0.77, 0.81), ("Llama-8B", "DeepSeek-R1-Distill-Llama-8B", 0.75, 0.75)):
    An, Tn = stack([X1D / f"{stem}_heldout_math500_n500_seed{s}.json" for s in range(47, 52)])
    for nm, got, want in (("old", kshare34(*BF[fam]), want_old), ("new", kshare34(An, Tn), want_new)):
        (OK if abs(got - want) < 0.005 else FAIL).append(f"k-share {fam} {nm}" if abs(got - want) < 0.005 else f"k-share {fam} {nm}: {got:.3f} vs text {want}")
# --- cell and completion counts quoted in the appendix
n_comp = 5 * 500 + 3 * 1319 + 5 * 500 + 10 * 500 + 10 * 500 + 10 * 500 + 5 * 500
(OK if n_comp == 26457 else FAIL).append("completions 26,457" if n_comp == 26457 else f"completions {n_comp} vs 26,457")
n_comp_cells = sum(1 for x in INV if x["experiment_id"] in {"p2_publication_qwen7b_math500_v1", "p2_dtype_control_qwen7b_math500_v1", "p4_decoding_qwen7b_math300_rep3_v1", "p2_publication_qwen7b_gsm8k_v1", "p2_dtype_control_qwen7b_gsm8k_v1"})
(OK if n_comp_cells == 87 else FAIL).append("87 companion-study cells" if n_comp_cells == 87 else f"companion cells {n_comp_cells} vs 87")

# --- prompt-token offset claims (P2 / P4 vs campaign-pipeline E2 cells, MATH-500)
ref_pt = np.array([z["prompt_tokens"] for z in rows_of(EXP / "revision2/compact/E2_fp16/DeepSeek-R1-Distill-Qwen-7B_fp16_math500_n500_seed42.json")])
offs = {}
for x in INV:
    if x["benchmark"] != "MATH-500": continue
    d = json.load(open(V4 / "compact" / Path(x["compact_file"]).name))
    for z in d["details"]: offs.setdefault((x["experiment_id"].split("_qwen7b")[0], x["variant_id"].split("_cal")[0].split("_t")[0]), set()).add(int(z["prompt_tokens"] - ref_pt[MITEM[z["item_id"]]]))
claims = {("p2_publication", "gptq4"): {-1}, ("p2_publication", "awq4"): {-1}, ("p2_dtype_control", "gptq4"): {-1}, ("p4_decoding", "awq"): {-1}, ("p4_decoding", "bf16"): {0}}
for k, v in claims.items():
    got = offs.get(k); (OK if got == v else FAIL).append(f"prompt-token offset {k}" if got == v else f"prompt-token offset {k}: found {got}, paper says {v}")

# --- token identity claims (transcription only: completion text is not on this machine)
TI = json.load(open(V3 / "X3_vs_campaign_token_identity.json"))["pooled"]
for lbl, got, want in (("X3 token-identical count", TI["token_identical"], 5), ("X3 pairs", TI["n_pairs"], 2500), ("X3 median first divergence", TI["first_divergent_token_quantiles_0_10_25_50_75_90_100"][3], 140), ("X3 share diverging within 100 tokens (37%)", round(TI["share_diverging_within_first_100_tokens"] * 100), 37)):
    (OK if got == want else FAIL).append(lbl if got == want else f"{lbl}: {got} vs {want}")

print(f"\nStage A: {len(OK)} checks agree, {len(FAIL)} disagree")
for m in FAIL: print("  MISMATCH:", m)

# ================================================================= STAGE B
print("\nStage B: every  value [lo,hi]  triplet in the text vs the saved results ...")
def num(s): return float(s.replace("{,}", "").replace(",", "").replace("+", ""))
TRIP = re.compile(r"\$([+-]?[\d{},.]+)\$(?:\\,pp)?(?:\}\{|\s)+\$\[([+-]?[\d{},.]+),\s*([+-]?[\d{},.]+)\]\$")
def triplets(text): return [(num(a), num(b), num(c), (a, b, c)) for a, b, c in TRIP.findall(text)]
def walk(o, out):
    if isinstance(o, dict):
        cis = [v for v in o.values() if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)]
        nums = [v for v in o.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        for ci in cis:
            for v in nums: out.append((float(v), float(ci[0]), float(ci[1])))
        for v in o.values(): walk(v, out)
    elif isinstance(o, list):
        for v in o: walk(v, out)
REF = []
for f in OUTD.glob("*.json"):
    if f.name != "verify_new_numbers.json": walk(json.load(open(f, encoding="utf-8")), REF)
def known(v, lo, hi, raw):
    d = 2 if "." in raw[0] else 0
    return any(abs(round(a, d) - v) < 1e-9 and abs(round(b, d) - lo) < 1e-9 and abs(round(c, d) - hi) < 1e-9 for (a, b, c) in REF)
OLDSET = {(v, lo, hi) for v, lo, hi, _ in triplets(open(ROOT / "paper" / "main.tex", encoding="utf-8").read())}
tex = open(ROOT / "paper" / "v2_followups" / "paper_v2.tex", encoding="utf-8").read()
for f in sorted((ROOT / "paper" / "v2_followups" / "generated").glob("tab_*.tex")): tex += "\n" + open(f, encoding="utf-8").read()
tr = triplets(tex); matched = old = 0; bad = []
for v, lo, hi, raw in tr:
    if known(v, lo, hi, raw) or (v, lo, hi) in ADHOC_VERIFIED: matched += 1
    elif (v, lo, hi) in OLDSET: old += 1
    else: bad.append(raw)
print(f"   {len(tr)} triplets in text/tables: {matched} match a saved result, {old} appear in paper/main.tex, {len(bad)} not matched")
print("   (not matched = numbers of the earlier campaign analysis, e.g. Table 1 and the accuracy table, that this script does not re-derive;")
print("    they are listed for review and do not affect the exit code. Exit code 1 means a Stage A recomputation disagrees.)")
for r in bad[:80]: print("   NOT MATCHED:", r)
(OUTD / "verify_new_numbers.json").write_text(json.dumps(dict(stageA_ok=len(OK), stageA_fail=FAIL, stageB_triplets=len(tr), stageB_matched=matched, stageB_old=old, stageB_unmatched=bad), indent=1))
sys.exit(1 if FAIL else 0)
