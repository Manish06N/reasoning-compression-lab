# Git credentials — do not commit tokens

GitHub Personal Access Tokens must **never** be committed to this repo, pasted in README/CHANGELOG, or shared in chat.

**MacBook (current, 2026-08-17):** PAT file is gitignored `.github-token` (one line, no quotes). Required scopes: `repo` **and** `workflow` (needed to push `.github/workflows/ci.yml`). Push with `bash scripts/macbook/github_push.sh origin paper-major-revision`. HPC never pushes. Do not merge to `main` until asked.

The Windows Credential Manager section below is a historical local-dev note.

## Where tokens are stored (Windows)

After you run the one-time setup below, your token lives in **Windows Credential Manager**:

1. Open **Control Panel** → **Credential Manager** → **Windows Credentials**
2. Look for `git:https://github.com`

Git for Windows uses **Git Credential Manager** to read it automatically on `git push` / `git pull`.

## One-time setup (PowerShell)

Replace `YOUR_TOKEN_HERE` with a PAT from https://github.com/settings/tokens (scopes: **repo** + **workflow**).

```powershell
@"
protocol=https
host=github.com
username=Manish06N
password=YOUR_TOKEN_HERE
"@ | git credential approve
```

Test:

```powershell
cd "G:\ALL MY Projects\2026\03-paper1-experiments"
git push origin main
```

## If a token was exposed

1. Revoke it immediately: https://github.com/settings/tokens
2. Generate a new token
3. Run the `git credential approve` block again with the new token

## HPC

HPC uses `git pull` over HTTPS or SSH — configure separately on the cluster. Do not copy PAT files to `/scratch`.
