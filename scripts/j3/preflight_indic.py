#!/usr/bin/env python3
"""J3 Indic deployment preflight (V8.2 §10)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.generation.llamacpp.runner import LlamaCppPilotConfig, pilot_manifest
from src.runners.config_utils import load_yaml

EXTERNAL_REPOS = Path(__file__).resolve().parents[2].parent / "external_repos"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/j3_preflight.json")
    args = parser.parse_args()

    lang_matrix = load_yaml("papers/j3/language_matrix.yaml")
    protocol_path = ROOT / "papers/j3/protocol.yaml"
    serving = load_yaml("configs/serving/llamacpp.yaml")

    repos = {
        "IndicIFEval": EXTERNAL_REPOS / "08-j3-indic-deployment/IndicIFEval",
        "IndicParam": EXTERNAL_REPOS / "08-j3-indic-deployment/IndicParam",
        "indic-gen-bench": EXTERNAL_REPOS / "08-j3-indic-deployment/indic-gen-bench",
        "llama.cpp": EXTERNAL_REPOS / "09-local-edge/llama.cpp",
    }

    report = {
        "paper": "j3",
        "protocol_exists": protocol_path.exists(),
        "language_matrix": lang_matrix,
        "benchmark_repos": {k: {"path": str(v), "exists": v.exists()} for k, v in repos.items()},
        "local_transfer": pilot_manifest(
            LlamaCppPilotConfig(model_gguf_path="models/placeholder.gguf")
        ),
        "serving": serving,
        "gate": "language_annotation",
        "status": "preflight_ready",
    }

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
