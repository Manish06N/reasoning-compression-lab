# Push to GitHub (do this before deep HPC work)

Protects your repo. Run on MacBook.

## 1. Create empty repo on GitHub

Name: `reasoning-compression-lab` (no README — you already have one locally)

## 2. Add remote and push

```bash
cd "/Users/manish/Projects/2026/paper 1/reasoning-compression-lab"

git remote add origin https://github.com/YOUR_USERNAME/reasoning-compression-lab.git
git branch -M main
git push -u origin main
```

If `origin` already exists with wrong URL:

```bash
git remote set-url origin https://github.com/YOUR_USERNAME/reasoning-compression-lab.git
git push -u origin main
```

## 3. Verify

```bash
git remote -v
git status
```

## 4. Clone on HPC

```bash
export QR=/scratch/$USER/reasoning-compression-lab
git clone https://github.com/YOUR_USERNAME/reasoning-compression-lab.git $QR
```

**Alternative without GitHub:** `scripts/macbook/rsync_to_hpc.sh`
