#!/usr/bin/env python3
"""Build static HTML dashboard from archive summaries (V8.2 §Appendix D)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_summaries(archive: Path) -> list[dict]:
    results = archive / "results"
    if not results.exists():
        return []
    out = []
    for p in sorted(results.glob("*_summary.json")):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def _html_table(rows: list[dict]) -> str:
    if not rows:
        return "<p>No summaries found.</p>"
    keys = ["cell_id", "quant_config", "task", "pass_at_1", "truncation_rate", "cost_of_pass"]
    header = "".join(f"<th>{k}</th>" for k in keys)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{row.get(k, '')}</td>" for k in keys) + "</tr>"
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, help="outputs-hpc-2a100-main-* root")
    parser.add_argument("--output", default="dashboards/index.html")
    args = parser.parse_args()

    archive = Path(args.archive)
    if not archive.is_absolute():
        archive = ROOT / archive
    summaries = _load_summaries(archive)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Reasoning Compression Lab Dashboard</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
th {{ background: #f5f5f5; }}
</style></head><body>
<h1>Experiment Dashboard</h1>
<p>Archive: {archive}</p>
<h2>Cell summaries</h2>
{_html_table(summaries)}
</body></html>"""

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote dashboard → {out}")


if __name__ == "__main__":
    main()
