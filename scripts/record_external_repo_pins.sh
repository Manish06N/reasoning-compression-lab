#!/usr/bin/env bash
# Record git SHAs of external reference repos for reproducibility manifests.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EX="$ROOT/../external_repos"
OUT="$ROOT/configs/external_repo_pins.json"

python3 - <<PY
import json
import subprocess
from pathlib import Path

ex = Path("$EX")
pins = {}
for git_dir in sorted(ex.rglob(".git")):
    if git_dir.is_dir() and git_dir.name == ".git":
        repo = git_dir.parent
        rel = repo.relative_to(ex).as_posix()
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
            ).strip()
            pins[rel] = sha
        except Exception:
            pins[rel] = "unknown"
Path("$OUT").write_text(json.dumps({"external_repos": pins}, indent=2) + "\n")
print(f"Recorded {len(pins)} repo pins → $OUT")
PY
