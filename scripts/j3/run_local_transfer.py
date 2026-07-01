#!/usr/bin/env python3
"""J3 local GGUF transfer pilot stub."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.generation.llamacpp.runner import LlamaCppPilotConfig, pilot_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True, help="Path to GGUF model")
    parser.add_argument("--output", default="results/j3_local_transfer_manifest.json")
    args = parser.parse_args()

    cfg = LlamaCppPilotConfig(model_gguf_path=args.gguf)
    report = pilot_manifest(cfg)
    report["status"] = "awaiting_llama_cpp_benchmark_run"
    report["commands"] = [
        f"llama-cli -m {args.gguf} --prompt-file prompts/qrm_math500.txt",
        "Record TTFT, TPOT, tokens/s, peak RAM/VRAM",
    ]

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
