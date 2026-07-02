#!/usr/bin/env bash
# Push to GitHub using PAT from repo-root .github-token (gitignored).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

TOKEN_FILE="$ROOT/.github-token"
if [[ ! -f "$TOKEN_FILE" ]]; then
  echo "Missing $TOKEN_FILE — create it with one line: your GitHub PAT (scopes: repo, workflow)." >&2
  exit 1
fi

export GH_TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
if [[ -z "$GH_TOKEN" ]]; then
  echo "$TOKEN_FILE is empty." >&2
  exit 1
fi

remote="${1:-origin}"
branch="${2:-main}"

# Use gh credential helper (run `gh auth login --with-token < .github-token` first).
gh auth setup-git 2>/dev/null || true
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push "$remote" "$branch"
