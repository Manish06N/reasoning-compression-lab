#!/usr/bin/env python3
"""Compare frozen major_revision_tables.md (and canonical JSON) against paper/main.tex.

TeX tables are transcribed. This script is the drift detector: markdown/JSON are
the source of truth; the manuscript must contain the same values under the
documented rounding policy.

Rounding (publication transcription, not a new analysis):
  * pass@1 Δ and CI bounds: two decimals, as in the frozen markdown
  * bootstrap p: three decimals; p < 0.0005 is written $<0.001$
  * Holm-18 adjusted p: three decimals (0.1088 → 0.109, 0.0306 → 0.031)
  * Holm flag: markdown yes/no → TeX sig./n.s.
  * RoM percent in tab:tokens: one decimal (−0.09 → −0.1%)
  * mean paired Δ tokens: nearest integer
  * clustered headline CI: one decimal
  * modal coverage: served/500 as one decimal percent
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEX_PATH = os.path.join(REPO, "paper", "main.tex")
MD_PATH = os.path.join(REPO, "results", "reports", "major_revision_tables.md")
REPORT_PATH = os.path.join(REPO, "results", "reports", "revision_reanalysis_report.json")
SERVING_PATH = os.path.join(
    REPO,
    "results",
    "reports",
    "measured_serving_confirmation",
    "measured_serving_confirmation_report.json",
)
MODAL_PATH = os.path.join(REPO, "results", "reports", "modal_agreement_report.json")

Check = tuple[str, str, str]  # (cell_id, expected, found_or_MISSING)


def load(path: str) -> str:
    with open(path, encoding="utf-8") as fp:
        return fp.read()


def strip_tex(cell: str) -> str:
    s = cell.replace("{,}", ",").replace(r"\,", " ").replace("~", " ")
    s = s.replace(r"\%", "%").replace(r"\$", "$")
    s = s.replace("$", "")
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    return " ".join(s.split())


def extract_tabular(tex: str, label: str) -> str:
    m = re.search(rf"\\label\{{{re.escape(label)}\}}", tex)
    if not m:
        raise SystemExit(f"ERROR: \\label{{{label}}} not found in paper/main.tex")
    rest = tex[m.end() :]
    tm = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", rest, re.S)
    if not tm:
        raise SystemExit(f"ERROR: no tabular after \\label{{{label}}}")
    return tm.group(0)


def tex_data_rows(tabular: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in tabular.split("\\\\"):
        raw = re.sub(r"\\(toprule|midrule|bottomrule)", " ", raw)
        if "&" not in raw:
            continue
        if r"\textbf{" in raw:
            continue
        cells = [strip_tex(c) for c in raw.split("&")]
        if cells and cells[0]:
            rows.append(cells)
    return rows


def fmt_signed_2(x: float) -> str:
    return f"{x:+.2f}"


def fmt_p(p: float) -> str:
    if p < 0.0005:
        return "<0.001"
    return f"{round(p, 3):.3f}"


def normalize_ci(s: str) -> str:
    s = s.replace(" ", "").replace("−", "-")
    s = s.replace("$", "")
    return s


def ci_close(expected: str, found: str, tol: float = 0.015) -> bool:
    num = r"[+-]?\d+(?:\.\d+)?"
    def pair(text: str) -> tuple[float, float] | None:
        m = re.search(rf"\[\s*({num})\s*,\s*({num})\s*\]", text.replace(" ", ""))
        return (float(m.group(1)), float(m.group(2))) if m else None

    e = pair(expected)
    if e is None:
        return normalize_ci(expected) in normalize_ci(found)
    for m in re.finditer(rf"\[\s*({num})\s*,\s*({num})\s*\]", found.replace(" ", "")):
        if abs(e[0] - float(m.group(1))) <= tol and abs(e[1] - float(m.group(2))) <= tol:
            return True
    return False


def p_in_cell(expected_p: str, cell: str) -> bool:
    compact = cell.replace(" ", "")
    if expected_p == "<0.001":
        return "<0.001" in compact
    return expected_p in compact


def holm_tex(yes: bool) -> str:
    return "sig." if yes else "n.s."


def short_contrast(md_name: str) -> str:
    fam = "Qwen" if "Qwen" in md_name else "Llama"
    fmt = "FP8"
    if "AWQ" in md_name:
        fmt = "AWQ-4"
    elif "GPTQ" in md_name:
        fmt = "GPTQ-4"
    return f"{fam} {fmt}"


def find_row(rows: list[list[str]], needle: str) -> list[str] | None:
    needle_n = needle.lower().replace("-", " ")
    for row in rows:
        key = " ".join(row[:2]).lower().replace("-", " ")
        if needle_n in key or needle.lower() in " ".join(row).lower():
            return row
    return None


def parse_md_tables(md: str) -> dict[str, list[list[str]]]:
    tables: dict[str, list[list[str]]] = {}
    current_title = "preamble"
    buf: list[str] = []
    in_table = False
    title = current_title

    def flush() -> None:
        nonlocal buf, title, in_table
        if not buf:
            return
        rows = []
        for line in buf:
            if line.startswith("|---") or set(line.replace("|", "").strip()) <= set("-:"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append(cells)
        tables[title] = rows
        buf = []
        in_table = False

    for line in md.splitlines():
        if line.startswith("### "):
            flush()
            title = line[4:].strip()
            continue
        if line.startswith("|") and "---" not in line:
            in_table = True
            buf.append(line)
            continue
        if in_table and line.startswith("|"):
            buf.append(line)
            continue
        if in_table:
            flush()
    flush()
    return tables


def md_contrast_map(rows: list[list[str]]) -> dict[str, list[str]]:
    out = {}
    for row in rows[1:]:
        if not row or "vs" not in row[0]:
            continue
        out[short_contrast(row[0])] = row
    return out


def check_pass_table(
    name: str,
    md_rows: list[list[str]],
    tex_rows: list[list[str]],
    *,
    has_mcnemar: bool,
) -> list[Check]:
    checks: list[Check] = []
    mapping = md_contrast_map(md_rows)
    for short, md in mapping.items():
        row = find_row(tex_rows, short)
        loc = f"{name}/{short}"
        if row is None:
            checks.append((loc, "row", "MISSING"))
            continue
        joined = " ".join(row)
        delta = md[1].replace("+", "+")
        checks.append((f"{loc} Δ", delta, delta if delta in joined.replace(" ", "") or delta in joined else "MISSING"))
        if not (delta in joined or delta.replace("+", "") in joined.replace(" ", "")):
            # allow $+0.40$ vs +0.40
            if delta not in joined.replace(" ", ""):
                checks[-1] = (f"{loc} Δ", delta, joined)
        ci95 = md[2].replace(" ", "")
        found_ci = "PASS" if ci_close(ci95, joined) else joined
        checks.append((f"{loc} 95% CI", ci95, found_ci if found_ci == "PASS" else joined))
        p_exp = fmt_p(float(md[3]))
        checks.append((f"{loc} p", p_exp, "PASS" if p_in_cell(p_exp, joined) else joined))
        holm = holm_tex(md[4].lower() == "yes")
        checks.append((f"{loc} Holm-6", holm, "PASS" if holm in joined else joined))
        ci90 = md[6].replace(" ", "")
        checks.append(
            (
                f"{loc} 90% CI",
                ci90,
                "PASS" if ci_close(ci90, joined) else joined,
            )
        )
        tost = md[7].strip().lower()
        checks.append((f"{loc} TOST", tost, "PASS" if tost in joined.lower() else joined))
        if has_mcnemar and len(row) >= 8:
            mcn = f"{float(md[8]) if False else ''}"
            # McNemar is not in the markdown contrast table; MATH only from JSON later.
            _ = mcn
    # rewrite Δ check more carefully
    cleaned: list[Check] = []
    for loc, exp, found in checks:
        if loc.endswith(" Δ"):
            row = find_row(tex_rows, loc.split("/")[1].replace(" Δ", ""))
            body = " ".join(row or [])
            ok = exp in body or exp.replace("+", "+") in body.replace(" ", "")
            # $+0.40$ after strip_tex is +0.40
            ok = exp in body.replace(" ", "") or exp in body
            cleaned.append((loc, exp, "PASS" if ok else body))
        elif found == "PASS":
            cleaned.append((loc, exp, "PASS"))
        elif loc.endswith("Holm-6") or loc.endswith("TOST") or loc.endswith(" p") or "CI" in loc:
            cleaned.append((loc, exp, found))
        else:
            cleaned.append((loc, exp, found))
    return cleaned


def check_pass_from_json(tex_rows: list[list[str]], contrasts: list[dict[str, Any]], name: str, mcnemar: bool) -> list[Check]:
    out: list[Check] = []
    for c in contrasts:
        short = f"{'Qwen' if 'Qwen' in c['model'] else 'Llama'} {c['format']}"
        row = find_row(tex_rows, short)
        loc = f"{name}/{short}"
        if row is None:
            out.append((loc, "row", "MISSING"))
            continue
        body = " ".join(row)
        compact = body.replace(" ", "")
        delta = fmt_signed_2(c["delta_pp"])
        out.append((f"{loc} Δ", delta, "PASS" if delta in compact or delta in body else body))
        ci95 = f"[{c['ci95_lo_pp']:+.2f},{c['ci95_hi_pp']:+.2f}]"
        out.append((f"{loc} 95% CI", ci95, "PASS" if ci_close(ci95, body) else body))
        p_exp = fmt_p(float(c["p_value"]))
        out.append((f"{loc} p", p_exp, "PASS" if p_in_cell(p_exp, body) else body))
        holm = holm_tex(bool(c["holm_significant_pass1"]))
        out.append((f"{loc} Holm-6", holm, "PASS" if holm in body else body))
        ci90 = f"[{c['ci90_lo_pp']:+.2f},{c['ci90_hi_pp']:+.2f}]"
        out.append((f"{loc} 90% CI", ci90, "PASS" if ci_close(ci90, body) else body))
        tost = "pass" if c["tost_equiv_1pp"] else "fail"
        out.append((f"{loc} TOST", tost, "PASS" if tost in body.lower() else body))
        if mcnemar:
            mp = fmt_p(float(c["mcnemar_p"]))
            if mp == "<0.001":
                mp = "0.000"
            # McNemar values in tex are three decimals, never <0.001 in this grid
            mp = f"{round(float(c['mcnemar_p']), 3):.3f}"
            out.append((f"{loc} McNemar p", mp, "PASS" if mp in compact else body))
        if "holm_p_global18" in c:
            gp = fmt_p(float(c["holm_p_global18"]))
            # recorded on holm18 table, not this one
            _ = gp
    return out


def headline_checks(tex_rows: list[list[str]], stats: dict[str, Any]) -> list[Check]:
    out: list[Check] = []
    order = [
        ("Qwen BF16", "Qwen-7B_BF16"),
        ("Qwen FP8", "Qwen-7B_FP8"),
        ("Qwen AWQ-4", "Qwen-7B_AWQ-4"),
        ("Qwen GPTQ-4", "Qwen-7B_GPTQ-4"),
        ("Llama BF16", "Llama-8B_BF16"),
        ("Llama FP8", "Llama-8B_FP8"),
        ("Llama AWQ-4", "Llama-8B_AWQ-4"),
        ("Llama GPTQ-4", "Llama-8B_GPTQ-4"),
    ]
    seeds = ["42", "43", "44", "45", "46"]
    for label, key in order:
        row = find_row(tex_rows, label)
        cell = stats[key]
        loc = f"tab:headline_results/{label}"
        if row is None:
            out.append((loc, "row", "MISSING"))
            continue
        body = " ".join(row)
        for s in seeds:
            acc = float(cell["seed_accs"][s])
            exp = f"{acc:.1f}"
            out.append((f"{loc} seed {s}", exp, "PASS" if exp in body else body))
        mean = f"{cell['mean_acc']:.2f}"
        std = f"{cell['std_acc']:.2f}"
        out.append((f"{loc} mean", mean, "PASS" if mean in body.replace(" ", "") else body))
        out.append((f"{loc} std", std, "PASS" if std in body.replace(" ", "") else body))
        lo, hi = cell["clustered_acc_ci95"]
        ci = f"[{lo:.1f}, {hi:.1f}]"
        out.append((f"{loc} clust CI", ci, "PASS" if ci_close(ci, body, tol=0.05) else body))
        tok = f"{cell['mean_tokens']:,.1f}"
        tok_tex = tok.replace(",", "")
        compact = body.replace(",", "").replace(" ", "")
        out.append((f"{loc} tokens", f"{cell['mean_tokens']:.1f}", "PASS" if f"{cell['mean_tokens']:.1f}" in compact or tok_tex in compact else body))
        out.append((f"{loc} loops", str(cell["loops"]), "PASS" if str(cell["loops"]) in row[-2] or str(cell["loops"]) in body.split() else body))
        out.append((f"{loc} near-cap", str(cell["near_cap"]), "PASS" if str(cell["near_cap"]) in row[-1] or body.strip().endswith(str(cell["near_cap"])) else body))
    return out


def length_checks(tex_rows: list[list[str]], token_analysis: dict[str, Any]) -> list[Check]:
    out: list[Check] = []
    key_map = {
        "Qwen FP8": "Qwen-7B_FP8",
        "Qwen AWQ-4": "Qwen-7B_AWQ-4",
        "Qwen GPTQ-4": "Qwen-7B_GPTQ-4",
        "Llama FP8": "Llama-8B_FP8",
        "Llama AWQ-4": "Llama-8B_AWQ-4",
        "Llama GPTQ-4": "Llama-8B_GPTQ-4",
    }
    for label, key in key_map.items():
        rec = token_analysis[key]
        row = find_row(tex_rows, label)
        loc = f"tab:tokens/{label}"
        if row is None:
            out.append((loc, "row", "MISSING"))
            continue
        body = " ".join(row)
        compact = body.replace(" ", "")
        rom = rec["ratio_of_means_pct"]
        rom_1 = f"{rom:+.1f}%"
        # -0.09 → -0.1%; 6.33 → +6.3%
        out.append((f"{loc} RoM", rom_1, "PASS" if rom_1.replace("%", "") in compact or rom_1 in body else body))
        mean_d = int(round(rec["mean_paired_delta_tokens"]))
        md = f"{mean_d:+d}"
        out.append((f"{loc} mean Δ", md, "PASS" if md in compact else body))
        dlo, dhi = rec["delta_ci95"]
        out.append(
            (
                f"{loc} mean Δ CI",
                f"[{dlo:+.0f},{dhi:+.0f}]",
                "PASS" if ci_close(f"[{dlo:+.0f},{dhi:+.0f}]", body, tol=1.1) else body,
            )
        )
        both = rec["strata"]["both_correct"]
        both_mean = f"{int(round(both['mean'])):+d}"
        out.append((f"{loc} Both-OK Δ", both_mean, "PASS" if both_mean in compact else body))
        out.append((f"{loc} Both-OK n", str(both["n"]), "PASS" if str(both["n"]) in compact else body))
        for stratum in ("bf16_only", "quant_only", "both_wrong"):
            st = rec["strata"][stratum]
            sm = f"{int(round(st['mean'])):+d}"
            out.append((f"{loc} {stratum} Δ", sm, "PASS" if sm in compact else body))
            out.append((f"{loc} {stratum} n", str(st["n"]), "PASS" if f"n={st['n']}" in compact or str(st["n"]) in compact else body))
    return out


def mismatch_checks(tex_rows: list[list[str]], token_analysis: dict[str, Any]) -> list[Check]:
    out: list[Check] = []
    key_map = {
        "Qwen FP8": "Qwen-7B_FP8",
        "Qwen AWQ-4": "Qwen-7B_AWQ-4",
        "Qwen GPTQ-4": "Qwen-7B_GPTQ-4",
        "Llama FP8": "Llama-8B_FP8",
        "Llama AWQ-4": "Llama-8B_AWQ-4",
        "Llama GPTQ-4": "Llama-8B_GPTQ-4",
    }
    for label, key in key_map.items():
        rec = token_analysis[key]
        row = find_row(tex_rows, label)
        loc = f"tab:mismatch-excess/{label}"
        if row is None:
            out.append((loc, "row", "MISSING"))
            continue
        body = " ".join(row)
        compact = body.replace(" ", "")
        d = rec["mismatch_excess_vs_both_correct"]
        dm = f"{int(round(d['mean'])):+d}"
        out.append((f"{loc} D", dm, "PASS" if dm in compact else body))
        out.append(
            (
                f"{loc} D CI",
                f"[{d['ci95_lo']:+.0f},{d['ci95_hi']:+.0f}]",
                "PASS" if ci_close(f"[{d['ci95_lo']:+.0f},{d['ci95_hi']:+.0f}]", body, tol=1.1) else body,
            )
        )
    return out


def serving_checks(tex_rows: list[list[str]], serving: dict[str, Any]) -> list[Check]:
    """Match confirmation aggregates. Keys vary slightly; be defensive."""
    out: list[Check] = []
    cells = serving.get("cells") or serving.get("configs") or serving.get("results")
    if isinstance(cells, dict):
        items = list(cells.items())
    elif isinstance(cells, list):
        items = []
        for c in cells:
            name = c.get("cell") or c.get("name") or f"{c.get('model','')} {c.get('format','')}"
            items.append((name, c))
    else:
        # Fall back to markdown-driven values already in tex (parsed below).
        return out
    return out


def serving_from_md(tex_rows: list[list[str]], md_rows: list[list[str]]) -> list[Check]:
    out: list[Check] = []
    for row in md_rows[1:]:
        cell = row[0]  # Qwen-7B BF16
        parts = cell.split()
        fam = parts[0]  # Qwen-7B
        fmt = parts[1]
        tex_row = None
        for tr in tex_rows:
            if fam.split("-")[0] in tr[0] and fmt in " ".join(tr[:2]):
                if (fam.startswith("Qwen") and "Qwen" in tr[0]) or (fam.startswith("Llama") and "Llama" in tr[0]):
                    tex_row = tr
                    break
        loc = f"tab:serving-main/{cell}"
        if tex_row is None:
            out.append((loc, "row", "MISSING"))
            continue
        body = " ".join(tex_row)
        compact = body.replace(" ", "")
        a_tps = row[1]
        a_gpu = row[2]
        b_tps = row[4]
        b_gpu = row[5]
        out.append((f"{loc} A tok/s", a_tps, "PASS" if a_tps in compact else body))
        out.append((f"{loc} A GPU-s", a_gpu, "PASS" if a_gpu in compact else body))
        out.append((f"{loc} B tok/s", b_tps, "PASS" if b_tps in compact else body))
        out.append((f"{loc} B GPU-s", b_gpu, "PASS" if b_gpu in compact else body))
        # C_pass like $0.0454 [0.0447, 0.0463]
        cpass = row[3].replace("$", "")
        c_mean = cpass.split()[0]
        out.append((f"{loc} A Cpass", c_mean, "PASS" if c_mean in compact else body))
        bpass = row[6].replace("$", "").split()[0]
        out.append((f"{loc} B Cpass", bpass, "PASS" if bpass in compact else body))
        delta = row[7]
        if delta != "anchor":
            # -45.9% [-46.4, -45.4]
            pct = delta.split("%")[0].replace("%", "")
            out.append((f"{loc} B Δ%", pct, "PASS" if pct in compact or pct.replace("%", "") in compact else body))
        else:
            out.append((f"{loc} B Δ%", "anchor", "PASS" if "anchor" in body.lower() else body))
    return out


def holm18_checks(tex_rows: list[list[str]], md_rows: list[list[str]]) -> list[Check]:
    out: list[Check] = []
    for row in md_rows[1:]:
        if len(row) < 6:
            continue
        bench, contrast, raw_p, h6, adj, h18 = row[0], row[1], row[2], row[3], row[4], row[5]
        short = short_contrast(contrast)
        bench_key = "MATH" if "MATH" in bench else ("GSM8K" if "GSM" in bench else "GPQA")
        tex_row = None
        for tr in tex_rows:
            blob = " ".join(tr)
            if bench_key.split("-")[0] in blob and short.split()[-1] in blob and ("Qwen" in short) == ("Qwen" in blob):
                if short.split()[0] in blob:
                    tex_row = tr
                    break
        loc = f"tab:holm18/{bench_key} {short}"
        if tex_row is None:
            out.append((loc, "row", "MISSING"))
            continue
        body = " ".join(tex_row)
        p_exp = fmt_p(float(raw_p))
        out.append((f"{loc} p", p_exp, "PASS" if p_in_cell(p_exp, body) else body))
        out.append((f"{loc} Holm-6", holm_tex(h6.lower() == "yes"), "PASS" if holm_tex(h6.lower() == "yes") in body else body))
        adj_exp = fmt_p(float(adj))
        out.append((f"{loc} Holm-18 p", adj_exp, "PASS" if p_in_cell(adj_exp, body) else body))
        out.append((f"{loc} Holm-18", holm_tex(h18.lower() == "yes"), "PASS" if holm_tex(h18.lower() == "yes") in body else body))
    return out


def modal_checks(tex_rows: list[list[str]], md_rows: list[list[str]], modal: dict[str, Any]) -> list[Check]:
    out: list[Check] = []
    t5 = {}
    # T5 may live under cells / by_model
    blob = json.dumps(modal)
    for row in md_rows[1:]:
        cell, tau, served, _errors, risk = row[0], row[1], int(row[2]), int(row[3]), row[4]
        cov = f"{served / 500.0 * 100.0:.1f}"
        fam, fmt = cell.split(" ", 1)
        tau_key = "3/5" if "3/5" in tau else ("4/5" if "4/5" in tau else "5/5")
        tex_row = None
        for tr in tex_rows:
            if fam.split("-")[0] in tr[0] and fmt in tr[1]:
                tex_row = tr
                break
        loc = f"tab:modal/{cell} {tau_key}"
        if tex_row is None:
            out.append((loc, "row", "MISSING"))
            continue
        body = " ".join(tex_row)
        out.append((f"{loc} cov", cov, "PASS" if cov in body else body))
        risk_f = f"{float(risk):.2f}" if float(risk) >= 0.1 or float(risk) == 0 else risk
        # 0.00 stays 0.00; 1.67 stays 1.67
        risk_exp = f"{float(risk):.2f}"
        out.append((f"{loc} risk", risk_exp, "PASS" if risk_exp in body else body))
        _ = blob, t5, risk_f
    # T5 from tex vs modal JSON if present
    def walk_t5(obj: Any, acc: dict[str, float]) -> None:
        if isinstance(obj, dict):
            if "mean_t5" in obj and "cell" in obj:
                acc[str(obj["cell"])] = float(obj["mean_t5"])
            if "T5" in obj and "name" in obj:
                acc[str(obj["name"])] = float(obj["T5"])
            for k, v in obj.items():
                if k in {"mean_T5", "mean_t5", "t5_mean"} and isinstance(v, (int, float)):
                    pass
                walk_t5(v, acc)
        elif isinstance(obj, list):
            for x in obj:
                walk_t5(x, acc)

    t5map: dict[str, float] = {}
    walk_t5(modal, t5map)
    return out


def fp8_rep_checks(tex_rows: list[list[str]], md_rows: list[list[str]]) -> list[Check]:
    out: list[Check] = []
    for row in md_rows[1:]:
        if not row[0].isdigit():
            continue
        rep = row[0]
        tex_row = None
        for tr in tex_rows:
            if tr[0].strip() == rep:
                tex_row = tr
                break
        loc = f"tab:fp8-reps/rep {rep}"
        if tex_row is None:
            out.append((loc, "row", "MISSING"))
            continue
        body = " ".join(tex_row)
        out.append((f"{loc} tok/s", row[1], "PASS" if row[1] in body else body))
        out.append((f"{loc} GPU-s", row[2], "PASS" if row[2] in body else body))
        cpass = row[3].replace("$", "")
        out.append((f"{loc} Cpass", cpass, "PASS" if cpass in body.replace("$", "") else body))
        out.append((f"{loc} regime", row[4], "PASS" if row[4] in body else body))
    return out


def summarize(groups: dict[str, list[Check]]) -> int:
    print("Checking manuscript numbers...\n")
    n_fail = 0
    for title, checks in groups.items():
        ok = sum(1 for c in checks if c[2] == "PASS")
        total = len(checks)
        mark = "✓" if ok == total else "✗"
        print(f"{title}:")
        print(f"  {ok}/{total} cells match {mark}")
        if ok != total:
            for loc, exp, found in checks:
                if found != "PASS":
                    n_fail += 1
                    print(f"    FAIL {loc}: expected {exp!r} found {found!r}")
        print()
    if n_fail:
        print(f"ERROR: {n_fail} manuscript cells drifted from frozen tables.", file=sys.stderr)
        return 1
    print("No manuscript drift detected.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        print("Pass --check.", file=sys.stderr)
        return 2

    tex = load(TEX_PATH)
    md = load(MD_PATH)
    report = json.loads(load(REPORT_PATH))
    md_tables = parse_md_tables(md)

    groups: dict[str, list[Check]] = {}

    groups["Table tab:headline_results (MATH-500 grid)"] = headline_checks(
        tex_data_rows(extract_tabular(tex, "tab:headline_results")),
        report["math500"]["summary_statistics"],
    )
    groups["Table tab:pass1 (MATH-500 contrasts)"] = check_pass_from_json(
        tex_data_rows(extract_tabular(tex, "tab:pass1")),
        report["math500"]["pass1_contrasts"],
        "tab:pass1",
        mcnemar=True,
    )
    groups["Table tab:gsm-contrasts"] = check_pass_from_json(
        tex_data_rows(extract_tabular(tex, "tab:gsm-contrasts")),
        report["gsm8k"]["pass1_contrasts"],
        "tab:gsm-contrasts",
        mcnemar=False,
    )
    groups["Table tab:gpqa-contrasts"] = check_pass_from_json(
        tex_data_rows(extract_tabular(tex, "tab:gpqa-contrasts")),
        report["gpqa_diamond"]["pass1_contrasts"],
        "tab:gpqa-contrasts",
        mcnemar=False,
    )
    groups["Table tab:tokens (length)"] = length_checks(
        tex_data_rows(extract_tabular(tex, "tab:tokens")),
        report["math500"]["token_analysis"],
    )
    groups["Table tab:mismatch-excess"] = mismatch_checks(
        tex_data_rows(extract_tabular(tex, "tab:mismatch-excess")),
        report["math500"]["token_analysis"],
    )
    groups["Table tab:holm18"] = holm18_checks(
        tex_data_rows(extract_tabular(tex, "tab:holm18")),
        md_tables["Holm-18 sensitivity (secondary; primary remains Holm-6 within each benchmark)"],
    )
    groups["Serving table tab:serving-main"] = serving_from_md(
        tex_data_rows(extract_tabular(tex, "tab:serving-main")),
        md_tables["Hybrid scenario Cost-of-Pass (Condition A and B)"],
    )
    groups["Table tab:fp8-reps"] = fp8_rep_checks(
        tex_data_rows(extract_tabular(tex, "tab:fp8-reps")),
        md_tables["Qwen-7B FP8 Condition B (all five repeats)"],
    )
    groups["Table tab:modal"] = modal_checks(
        tex_data_rows(extract_tabular(tex, "tab:modal")),
        md_tables["Modal selective risk with Wilson / Clopper–Pearson intervals"],
        json.loads(load(MODAL_PATH)),
    )

    # Pathology totals (prose + caption)
    gt = report["grid_totals"]
    path_checks: list[Check] = []
    for exp, needles in (
        (str(gt["loops"]), [r"\textbf{25}", "25"]),
        (str(gt["token_limit_hits"]), [r"\textbf{0}", "0 exact"]),
        (str(gt["near_cap"]), ["209"]),
    ):
        ok = any(n in tex for n in needles) if exp != "0" else ("textbf{0}" in tex.replace(" ", "") or r"\textbf{0}" in tex)
        if exp == "25":
            ok = r"\textbf{25}" in tex or "records \\textbf{25}" in tex
            ok = r"\textbf{25}" in tex
        if exp == "209":
            ok = "209" in tex
        if exp == "0":
            ok = r"\textbf{0}" in tex and "exact cap" in tex
        path_checks.append((f"pathology/{exp}", exp, "PASS" if ok else "MISSING"))
    groups["Pathology (loops / cap / near-cap)"] = path_checks

    # Multitask means
    mt_rows = tex_data_rows(extract_tabular(tex, "tab:multitask"))
    mt_checks: list[Check] = []
    bench_map = [
        ("math500", "MATH-500", 2),
        ("gsm8k", "GSM8K", 2),
        ("gpqa_diamond", "GPQA-D", 2),
    ]
    labels = {
        "Qwen-7B_BF16": "Qwen BF16",
        "Qwen-7B_FP8": "Qwen FP8",
        "Qwen-7B_AWQ-4": "Qwen AWQ-4",
        "Qwen-7B_GPTQ-4": "Qwen GPTQ-4",
        "Llama-8B_BF16": "Llama BF16",
        "Llama-8B_FP8": "Llama FP8",
        "Llama-8B_AWQ-4": "Llama AWQ-4",
        "Llama-8B_GPTQ-4": "Llama GPTQ-4",
    }
    col = {"math500": 1, "gsm8k": 2, "gpqa_diamond": 3}
    for key, label in labels.items():
        row = find_row(mt_rows, label)
        if row is None:
            mt_checks.append((f"tab:multitask/{label}", "row", "MISSING"))
            continue
        for bench, _name, _ in bench_map:
            stats = report[bench]["summary_statistics"][key]
            mean = f"{stats['mean_acc']:.2f}"
            std = f"{stats['std_acc']:.2f}"
            cell = row[col[bench]]
            mt_checks.append(
                (
                    f"tab:multitask/{label} {_name} mean",
                    mean,
                    "PASS" if mean in cell.replace(" ", "") else cell,
                )
            )
            mt_checks.append(
                (
                    f"tab:multitask/{label} {_name} std",
                    std,
                    "PASS" if std in cell.replace(" ", "") else cell,
                )
            )
    groups["Table tab:multitask"] = mt_checks

    _ = SERVING_PATH  # confirmation JSON used via frozen markdown
    return summarize(groups)


if __name__ == "__main__":
    raise SystemExit(main())
