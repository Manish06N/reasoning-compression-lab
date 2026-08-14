#!/usr/bin/env python3
"""
Queue Manager Daemon with Real-Time Telegram Notifications for
Reasoning Compression Lab Publication Campaign (24/7 Multi-Seed Evaluation).

Features:
- 24/7 Continuous Execution across 2 A100 GPUs (1 Qwen channel + 1 Llama channel).
- Real-time Telegram Notifications:
    * Job Submissions & Queue Chaining
    * Job Started (with allocated node)
    * Milestone Progress Updates (every ~20% or 100 prompts)
    * Job Completion with Pass@1 Accuracy & validation report details
    * Job Failure alerts with error cause
    * Hourly Campaign Status Summary Dashboard
- Self-healing retry for failed/missing cells.
- Fully compatible with Python 3.6+ and Python 3.12+.
"""

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

QR = Path(os.environ.get("QR", f"/scratch/{os.environ.get('USER', 'manishn_iitp')}/reasoning-compression-lab"))
CELLS_FILE = QR / "configs" / "campaign_cells.json"
OUTPUT_ROOT = QR / "outputs-hpc-campaign-2026-08-14"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
SLURM_SCRIPT = QR / "slurm" / "qrm_official_math500_n10.slurm"
STATE_FILE = OUTPUT_ROOT / "campaign_state.json"
VALIDATION_DIR = OUTPUT_ROOT / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = QR / "logs"

# Telegram credentials loader
def get_telegram_credentials():
    token = os.environ.get("TG_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    if token and chat_id:
        return token, chat_id

    # Fallback to watch-job script in home directory
    watch_script = Path(os.path.expanduser("~/watch-job-52772.sh"))
    if watch_script.exists():
        content = watch_script.read_text()
        m_tok = re.search(r'TG_TOKEN=["\']?([^"\'\s]+)["\']?', content)
        m_chat = re.search(r'TG_CHAT_ID=["\']?([^"\'\s]+)["\']?', content)
        if m_tok and m_chat:
            return m_tok.group(1), m_chat.group(1)

    return "8738869628:AAEtrsLVoqvDXeNz6CEa-ym5-AY7VJKScZ4", "638098622"


TG_TOKEN, TG_CHAT_ID = get_telegram_credentials()


def send_telegram(text):
    """Sends HTML formatted message to Telegram."""
    if not TG_TOKEN or not TG_CHAT_ID:
        print("[telegram] Warning: No Telegram credentials configured.")
        return False

    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }
    try:
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            return res.get("ok", False)
    except Exception as e:
        print(f"[telegram] Failed to send message: {e}")
        return False


def get_active_slurm_jobs():
    """Returns list of active/pending jobs for current user."""
    try:
        user = os.environ.get("USER", "manishn_iitp")
        raw = subprocess.check_output(
            ["squeue", "-u", user, "-h", "-o", "%i %j %T %R %M"]
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
                node = parts[3] if len(parts) >= 4 else "-"
                elapsed = parts[4] if len(parts) >= 5 else "0:00"
                jobs.append({
                    "id": parts[0],
                    "name": parts[1],
                    "state": parts[2],
                    "node": node,
                    "elapsed": elapsed,
                })
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
        return False, None, None

    try:
        data = json.loads(report_file.read_text())
        if isinstance(data, dict) and "accuracy" in data:
            acc = data.get("accuracy", 0.0)
            correct = data.get("correct", 0)
            total = data.get("total", max_samples)
            return True, f"{acc:.1%}", f"{correct}/{total}"
    except Exception:
        pass
    return False, None, None


def get_job_progress(job_id):
    """Parses live prompt progress from SLURM error log."""
    err_file = LOGS_DIR / f"qrm_official_{job_id}.err"
    if not err_file.exists():
        return None, None, None

    try:
        # Read last 4KB to get the latest progress line
        with open(err_file, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            lines = f.read().decode("utf-8", errors="replace").split("\n")

        for line in reversed(lines):
            if "Processed prompts:" in line:
                m = re.search(r"Processed prompts:\s*(\d+)%.*?(\d+)/500.*?output:\s*([\d\.]+\s*toks/s)", line)
                if m:
                    pct = int(m.group(1))
                    count = int(m.group(2))
                    speed = m.group(3)
                    return pct, count, speed
    except Exception:
        pass
    return None, None, None


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
        
        # Send Telegram notification for submission
        dep_str = f" (Chained after Job {prev_job_id})" if prev_job_id else " (Active Immediately)"
        msg = (
            f"🚀 <b>SLURM Job Submitted</b>\n"
            f"• <b>Job Name:</b> <code>{cell['name']}</code>\n"
            f"• <b>Job ID:</b> <code>{job_id}</code>{dep_str}\n"
            f"• <b>Model:</b> <code>{os.path.basename(cell['model'])}</code>\n"
            f"• <b>Dataset:</b> MATH-500 | <b>Seed:</b> {cell['seed']}"
        )
        send_telegram(msg)
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"[queue_manager] ERROR: Failed to submit {cell['name']}: {e}")
        return None


class CampaignTracker:
    def __init__(self):
        self.known_completed = set()
        self.known_running = {}  # job_id -> node
        self.last_progress_pct = {}  # job_id -> pct
        self.last_summary_time = 0

    def load_initial_completed(self, cells):
        for cell in cells:
            done, acc, score = is_cell_completed(cell)
            if done:
                self.known_completed.add(cell["name"])

    def step(self):
        if not CELLS_FILE.exists():
            return

        cells = json.loads(CELLS_FILE.read_text())
        if not self.known_completed:
            self.load_initial_completed(cells)

        active_jobs = get_active_slurm_jobs()
        total_active_queued = len(active_jobs)
        active_names = {j["name"]: j for j in active_jobs}

        qwen_cells = [c for c in cells if c["channel"] == "qwen"]
        llama_cells = [c for c in cells if c["channel"] == "llama"]

        print("\n" + "=" * 70)
        print(f"[queue_manager] [{time.strftime('%Y-%m-%d %H:%M:%S')}] 24/7 Campaign Heartbeat")
        print(f"[queue_manager] Active/Queued Jobs in SLURM: {total_active_queued}")

        # Check job status transitions & progress
        for j in active_jobs:
            jid = j["id"]
            jname = j["name"]
            jstate = j["state"]
            jnode = j["node"]

            # Check if job started running
            if jstate == "RUNNING" and jid not in self.known_running:
                self.known_running[jid] = jnode
                msg = (
                    f"▶️ <b>Job Started Running</b>\n"
                    f"• <b>Job Name:</b> <code>{jname}</code> (Job <code>{jid}</code>)\n"
                    f"• <b>Node:</b> <code>{jnode}</code>\n"
                    f"• <b>Time:</b> {time.strftime('%H:%M:%S IST')}"
                )
                send_telegram(msg)

            # Check prompt progress
            if jstate == "RUNNING":
                pct, count, speed = get_job_progress(jid)
                if pct is not None:
                    last_pct = self.last_progress_pct.get(jid, 0)
                    # Notify every ~20% milestone or on first progress
                    if pct >= last_pct + 20 or (last_pct == 0 and pct >= 10):
                        self.last_progress_pct[jid] = pct
                        msg = (
                            f"⏳ <b>Milestone Progress ({pct}%)</b>\n"
                            f"• <b>Job:</b> <code>{jname}</code> (<code>{jid}</code>)\n"
                            f"• <b>Progress:</b> {count} / 500 prompts ({pct}%)\n"
                            f"• <b>Gen Speed:</b> <code>{speed}</code>\n"
                            f"• <b>Node:</b> <code>{jnode}</code> | <b>Elapsed:</b> {j['elapsed']}"
                        )
                        send_telegram(msg)

        # Check for newly completed cells
        for cell in cells:
            cname = cell["name"]
            done, acc, score = is_cell_completed(cell)
            if done and cname not in self.known_completed:
                self.known_completed.add(cname)
                msg = (
                    f"🎉 <b>Job COMPLETED &amp; VALIDATED</b>\n"
                    f"• <b>Cell:</b> <code>{cname}</code>\n"
                    f"• <b>Model:</b> <code>{os.path.basename(cell['model'])}</code>\n"
                    f"• <b>Accuracy (Pass@1):</b> <b>{acc}</b> ({score})\n"
                    f"• <b>Seed:</b> {cell['seed']} | <b>Dataset:</b> MATH-500 ($n=500$)"
                )
                send_telegram(msg)

        # Queue Management across both channels
        for channel_name, channel_cells in [("Qwen-7B", qwen_cells), ("Llama-8B", llama_cells)]:
            channel_active_jobs = [j for j in active_jobs if any(c["name"] == j["name"] for c in channel_cells)]
            last_job_id_in_channel = None
            if channel_active_jobs:
                sorted_jobs = sorted(channel_active_jobs, key=lambda x: int(x["id"]))
                last_job_id_in_channel = sorted_jobs[-1]["id"]

            channel_job_count = len(channel_active_jobs)

            for cell in channel_cells:
                cname = cell["name"]
                done, acc, score = is_cell_completed(cell)
                if done:
                    continue

                if cname in active_names:
                    continue

                # Max 3 queued per channel, max 6 queued total
                if channel_job_count >= 3 or total_active_queued >= 6:
                    break

                job_id = submit_cell(cell, prev_job_id=last_job_id_in_channel)
                if job_id:
                    last_job_id_in_channel = job_id
                    channel_job_count += 1
                    total_active_queued += 1
                    active_names[cname] = {"id": job_id, "name": cname, "state": "PENDING", "node": "-", "elapsed": "0:00"}

        # Periodic hourly summary ping
        now = time.time()
        if now - self.last_summary_time > 3600:
            self.last_summary_time = now
            completed_count = len(self.known_completed)
            total_cells = len(cells)
            msg = (
                f"📊 <b>24/7 Campaign Hourly Status</b>\n"
                f"• <b>Completed Cells:</b> {completed_count} / {total_cells} ({completed_count / total_cells:.1%})\n"
                f"• <b>Active SLURM Jobs:</b> {total_active_queued}\n"
                f"• <b>Channel 1 (Qwen):</b> {len([j for j in active_jobs if 'qwen' in j['name'].lower()])} queued/active\n"
                f"• <b>Channel 2 (Llama):</b> {len([j for j in active_jobs if 'llama' in j['name'].lower()])} queued/active"
            )
            send_telegram(msg)


def main():
    print("[queue_manager] Starting Upgraded 24/7 Publication Queue Manager with Telegram Alerts...")
    send_telegram("🔔 <b>24/7 Publication Queue Manager Daemon Active</b>\nAutonomous execution across 2 A100 channels initialized.")
    tracker = CampaignTracker()
    while True:
        try:
            tracker.step()
        except Exception as e:
            print(f"[queue_manager] Daemon loop error: {e}")
        time.sleep(45)  # Poll every 45 seconds


if __name__ == "__main__":
    main()
