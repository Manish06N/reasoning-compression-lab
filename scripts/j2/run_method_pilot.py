#!/usr/bin/env python3
"""J2 method-selection pilot gate (V8.2 §8.2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_REPOS = ROOT.parent / "external_repos"
sys.path.insert(0, str(ROOT))

from src.generation.sglang.runner import SGLangPilotConfig, check_sglang_available, pilot_manifest
from src.runners.config_utils import load_yaml


CANDIDATES = {
    "eagle": "07-j2-serving-acceleration/EAGLE",
    "specreason": "07-j2-serving-acceleration/specreason",
    "kivi": "07-j2-serving-acceleration/KIVI",
    "kvquant": "07-j2-serving-acceleration/KVQuant",
    "quarot": "07-j2-serving-acceleration/QuaRot",
    "medusa": "07-j2-serving-acceleration/Medusa",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="J2 method-selection pilot manifest")
    parser.add_argument("--backend", choices=["vllm", "sglang", "all"], default="all")
    parser.add_argument("--model-path", default="${QREASON_MODEL_QWEN7B}")
    parser.add_argument("--output", default="results/j2_pilot_manifest.json")
    args = parser.parse_args()

    serving_vllm = load_yaml("configs/serving/vllm.yaml")
    serving_sglang = load_yaml("configs/serving/sglang.yaml")

    report = {
        "paper": "j2",
        "protocol": "papers/j2/protocol.yaml",
        "candidates": {},
        "backends": {},
    }

    for name, repo_rel in CANDIDATES.items():
        repo_path = EXTERNAL_REPOS / repo_rel
        report["candidates"][name] = {
            "repo": f"external_repos/{repo_rel}",
            "cloned": repo_path.exists(),
        }

    if args.backend in ("vllm", "all"):
        report["backends"]["vllm"] = serving_vllm
    if args.backend in ("sglang", "all"):
        cfg = SGLangPilotConfig(model_path=args.model_path)
        report["backends"]["sglang"] = {
            **pilot_manifest(cfg),
            "installed": check_sglang_available(),
            "serving_config": serving_sglang,
        }

    report["next_steps"] = [
        "Run 4-6 week pilot on ONE candidate with end-to-end TTFT/TPOT at concurrency 1 and 4",
        "Document acceptance rate and quality delta",
        "Select one main + one reference mechanism before full J2 grid",
    ]

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
