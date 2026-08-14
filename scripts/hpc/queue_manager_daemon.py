#!/usr/bin/env python3
"""
Queue Manager Daemon for Reasoning Compression Lab Publication Campaign.
Automatically submits remaining publication jobs to SLURM as slots free up,
strictly respecting:
1. Max 2 active GPUs per user (QOSMaxGRESPerUser)
2. Max 10 total submitted/queued jobs per user (QOSMaxSubmitJobPerUserLimit)
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


def get_active_slurm_jobs():
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", "manishn_iitp"), "-h", "-o", "%i %j %T"],
            text=True,
        )
        jobs = []
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                jobs.append({"id": parts[0], "name": parts[1], "state": parts[2]})
        return jobs
    except Exception as e:
        print(f"[queue_manager] Error checking squeue: {e}")
        return []


def load_campaign_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"submitted": {}, "completed": {}}


def save_campaign_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def submit_cell(cell, prev_job_id=None):
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

    try:
        out = subprocess.check_output(cmd, env=env, text=True, cwd=str(QR))
        job_id = out.strip()
        print(f"[queue_manager] Submitted {cell['name']} -> Job {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"[queue_manager] Failed to submit {cell['name']}: {e}")
        return None


def run_daemon_step():
    cells = json.loads(CELLS_FILE.read_text())
    state = load_campaign_state()
    active_jobs = get_active_slurm_jobs()
    total_active_queued = len(active_jobs)
    active_names = {j["name"] for j in active_jobs}

    print(f"[queue_manager] Active/queued jobs in SLURM: {total_active_queued}/10 max limit")

    # Group cells by channel
    qwen_cells = [c for c in cells if c["channel"] == "qwen"]
    llama_cells = [c for c in cells if c["channel"] == "llama"]

    for channel_cells in (qwen_cells, llama_cells):
        last_submitted_id = None
        for cell in channel_cells:
            cname = cell["name"]
            if cname in state["submitted"]:
                last_submitted_id = state["submitted"][cname]
                continue

            # Need to submit this cell if queue limit allows (keep <= 8 queued)
            if total_active_queued >= 8:
                print(f"[queue_manager] Queue full ({total_active_queued} jobs). Waiting.")
                break

            job_id = submit_cell(cell, prev_job_id=last_submitted_id)
            if job_id:
                state["submitted"][cname] = job_id
                save_campaign_state(state)
                last_submitted_id = job_id
                total_active_queued += 1
            else:
                break


def main():
    print("[queue_manager] Starting Campaign Queue Manager Daemon...")
    while True:
        try:
            run_daemon_step()
        except Exception as e:
            print(f"[queue_manager] Daemon loop error: {e}")
        time.sleep(120)  # Check every 2 minutes


if __name__ == "__main__":
    main()
