#!/usr/bin/env python3
"""
Queue Manager Daemon for Reasoning Compression Lab Publication Campaign.
Maintains continuous 24/7 execution across 2 A100 GPUs (1 Qwen channel + 1 Llama channel).

Features:
- Compatible with Python 3.6+ and Python 3.12+.
- Automatically detects completed cells via validation JSON reports.
- Automatically retries missing/failed cells.
- Maintains single active job per channel (max 2 GPUs total).
- Chains pending jobs with --dependency=afterany.
- Prevents exceeding cluster job limits (keeps total queued <= 6).
"""

import json
import os
import subprocess
import time
from pathlib import Path

QR = Path(os.environ.get("QR", f"/scratch/{os.environ.get('USER', 'manishn_iitp')}/reasoning-compression-lab"))
CELLS_FILE = QR / "configs" / "campaign_cells.json"
OUTPUT_ROOT = QR / "outputs-hpc-campaign-2026-08-14"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
SLURM_SCRIPT = QR / "slurm" / "qrm_official_math500_n10.slurm"
STATE_FILE = OUTPUT_ROOT / "campaign_state.json"
VALIDATION_DIR = OUTPUT_ROOT / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


def get_active_slurm_jobs():
    """Returns list of active/pending jobs for current user."""
    try:
        user = os.environ.get("USER", "manishn_iitp")
        raw = subprocess.check_output(
            ["squeue", "-u", user, "-h", "-o", "%i %j %T"]
        )
        if isinstance(raw, bytes):
            out = raw.decode("utf-8", errors="replace")
        else:
            out = str(raw)

        jobs = []
        for line in out.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                jobs.append({"id": parts[0], "name": parts[1], "state": parts[2]})
        return jobs
    except Exception as e:
        print(f"[queue_manager] Error checking squeue: {e}")
        return []


def is_cell_completed(cell):
    """Checks if a cell produced a valid output report."""
    model_name = os.path.basename(cell["model"])
    seed = cell["seed"]
    max_samples = 500
    report_file = VALIDATION_DIR / f"{model_name}_math500_n{max_samples}_seed{seed}.json"
    if not report_file.exists():
        return False, None

    try:
        data = json.loads(report_file.read_text())
        if isinstance(data, dict) and "accuracy" in data:
            acc = data.get("accuracy", 0.0)
            correct = data.get("correct", 0)
            return True, f"accuracy={acc:.1%}, correct={correct}/500"
    except Exception:
        pass
    return False, None


def submit_cell(cell, prev_job_id=None):
    """Submits a single cell to SLURM with optional dependency."""
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name={cell['name']}",
    ]
    if prev_job_id:
        cmd.append(f"--dependency=afterany:{prev_job_id}")
    cmd.append(str(SLURM_SCRIPT))

    env = os.environ.copy()
    env["QRM_MODEL_PATH"] = cell["model"]
    env["QRM_OUTPUT_ROOT"] = str(OUTPUT_ROOT)
    env["QRM_MAX_SAMPLES"] = "500"
    env["QRM_SEED"] = str(cell["seed"])

    if "AWQ" in cell["model"] or "awq" in cell["model"].lower():
        env["QRM_DTYPE"] = "float16"

    try:
        raw = subprocess.check_output(cmd, env=env, cwd=str(QR))
        if isinstance(raw, bytes):
            out = raw.decode("utf-8", errors="replace")
        else:
            out = str(raw)
        job_id = out.strip()
        print(f"[queue_manager] ==> SUBMITTED {cell['name']} (Seed {cell['seed']}) -> SLURM Job {job_id}" + (f" [dep: {prev_job_id}]" if prev_job_id else " [ACTIVE]"))
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"[queue_manager] ERROR: Failed to submit {cell['name']}: {e}")
        return None


def run_daemon_step():
    if not CELLS_FILE.exists():
        print(f"[queue_manager] Missing cells file: {CELLS_FILE}")
        return

    cells = json.loads(CELLS_FILE.read_text())
    active_jobs = get_active_slurm_jobs()
    total_active_queued = len(active_jobs)
    active_names = {j["name"]: j for j in active_jobs}

    qwen_cells = [c for c in cells if c["channel"] == "qwen"]
    llama_cells = [c for c in cells if c["channel"] == "llama"]

    print("\n" + "=" * 70)
    print(f"[queue_manager] [{time.strftime('%Y-%m-%d %H:%M:%S')}] Campaign Heartbeat")
    print(f"[queue_manager] SLURM Queue: {total_active_queued} active/pending jobs")
    for j in active_jobs:
        print(f"  - Job {j['id']} ({j['name']}): {j['state']}")

    for channel_name, channel_cells in [("Qwen-7B", qwen_cells), ("Llama-8B", llama_cells)]:
        print(f"\n--- Channel: {channel_name} ---")
        
        # Check active or pending jobs in this channel
        channel_active_jobs = [j for j in active_jobs if any(c["name"] == j["name"] for c in channel_cells)]
        last_job_id_in_channel = None
        if channel_active_jobs:
            # Sort by job id to find the tail of dependency chain
            sorted_jobs = sorted(channel_active_jobs, key=lambda x: int(x["id"]))
            last_job_id_in_channel = sorted_jobs[-1]["id"]

        # Number of queued/running jobs in this channel
        channel_job_count = len(channel_active_jobs)

        for cell in channel_cells:
            cname = cell["name"]
            done, detail = is_cell_completed(cell)
            if done:
                print(f"  [COMPLETED] {cname} -> {detail}")
                continue

            if cname in active_names:
                j = active_names[cname]
                print(f"  [{j['state']}] {cname} (Job {j['id']})")
                continue

            # Need to submit if queue headroom allows (max 3 queued per channel, total <= 6)
            if channel_job_count >= 3 or total_active_queued >= 6:
                print(f"  [WAITING] {cname} (Queue depth: {channel_job_count} in channel, {total_active_queued} total)")
                break

            job_id = submit_cell(cell, prev_job_id=last_job_id_in_channel)
            if job_id:
                last_job_id_in_channel = job_id
                channel_job_count += 1
                total_active_queued += 1
                active_names[cname] = {"id": job_id, "name": cname, "state": "PENDING"}


def main():
    print("[queue_manager] Starting Upgraded 24/7 Publication Queue Manager...")
    while True:
        try:
            run_daemon_step()
        except Exception as e:
            print(f"[queue_manager] Daemon error: {e}")
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    main()
