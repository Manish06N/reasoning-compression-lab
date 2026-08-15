#!/usr/bin/env python3
"""
Autonomous 24/7 Queue Manager Daemon with Real-Time Telegram Notifications.
Manages 2-Channel execution (1 GPU Qwen + 1 GPU Llama) on PARAM Rudra HPC.

Features:
- Maintains exactly 1 running job per model family concurrently (2 GPUs total).
- Chained dependencies with max 2 jobs queued per channel (avoids QOS submit limits).
- Dynamic configuration for any task (MATH-500, GSM8K, GPQA-Diamond).
- Real-time Telegram Notifications:
    * Job Submissions & Queue Chaining
    * Job Started (with assigned compute node)
    * Milestone Progress Updates (every ~20% or 100 prompts)
    * Job Completion with Pass@1 Accuracy & validation report details
    * Hourly Campaign Status Summary Dashboard
- Self-healing retry for failed or preempted cells.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

QR = Path(os.environ.get("QR", f"/scratch/{os.environ.get('USER', 'manishn_iitp')}/reasoning-compression-lab"))
LOGS_DIR = QR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SLURM_SCRIPT = QR / "slurm" / "qrm_official_math500_n10.slurm"


def get_telegram_credentials():
    token = os.environ.get("TG_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    if token and chat_id:
        return token, chat_id

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


def is_cell_completed(cell, validation_dir, max_samples, task_name):
    """Checks if a cell produced a valid output report."""
    model_name = os.path.basename(cell["model"])
    seed = cell["seed"]
    clean_task = task_name.lower().replace("-", "")
    report_file = validation_dir / f"{model_name}_{clean_task}_n{max_samples}_seed{seed}.json"
    if not report_file.exists():
        # Also check fallback naming
        report_file = validation_dir / f"{model_name}_math500_n{max_samples}_seed{seed}.json"
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


def get_job_progress(job_id, max_samples):
    """Parses live prompt progress from SLURM error log."""
    err_file = LOGS_DIR / f"qrm_official_{job_id}.err"
    if not err_file.exists():
        return None, None, None

    try:
        with open(err_file, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            lines = f.read().decode("utf-8", errors="replace").split("\n")

        for line in reversed(lines):
            if "Processed prompts:" in line:
                m = re.search(r"Processed prompts:\s*(\d+)%.*?(\d+)/" + str(max_samples) + r".*?output:\s*([\d\.]+\s*toks/s)", line)
                if not m:
                    m = re.search(r"Processed prompts:\s*(\d+)%.*?(\d+)/\d+.*?output:\s*([\d\.]+\s*toks/s)", line)
                if m:
                    pct = int(m.group(1))
                    count = int(m.group(2))
                    speed = m.group(3)
                    return pct, count, speed
    except Exception:
        pass
    return None, None, None


def submit_cell(cell, output_root, task_name, max_samples, prev_job_id=None):
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
    env["QRM_DATASET"] = str(task_name)
    env["QRM_MODEL_PATH"] = cell["model"]
    env["QRM_OUTPUT_ROOT"] = str(output_root)
    env["QRM_MAX_SAMPLES"] = str(max_samples)
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
        
        dep_str = f" (Chained after Job {prev_job_id})" if prev_job_id else " (Active Immediately)"
        msg = (
            f"🚀 <b>SLURM Job Submitted</b>\n"
            f"• <b>Job Name:</b> <code>{cell['name']}</code>\n"
            f"• <b>Job ID:</b> <code>{job_id}</code>{dep_str}\n"
            f"• <b>Model:</b> <code>{os.path.basename(cell['model'])}</code>\n"
            f"• <b>Task:</b> {task_name} ($n={max_samples}$) | <b>Seed:</b> {cell['seed']}"
        )
        send_telegram(msg)
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"[queue_manager] ERROR: Failed to submit {cell['name']}: {e}")
        return None


class CampaignTracker:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text())
        self.task_name = self.config.get("task", "GSM8K")
        self.max_samples = self.config.get("max_samples", 1319)
        self.output_root = Path(self.config.get("output_root", QR / "outputs-hpc-breadth-gsm8k-2026-08-15"))
        self.validation_dir = self.output_root / "validation"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.validation_dir.mkdir(parents=True, exist_ok=True)

        self.qwen_cells = self.config["channels"]["qwen"]
        self.llama_cells = self.config["channels"]["llama"]
        self.all_cells = self.qwen_cells + self.llama_cells

        self.known_completed = set()
        self.known_running = {}
        self.last_progress_pct = {}
        self.last_summary_time = 0

        self.load_initial_completed()

    def load_initial_completed(self):
        for cell in self.all_cells:
            done, acc, score = is_cell_completed(cell, self.validation_dir, self.max_samples, self.task_name)
            if done:
                self.known_completed.add(cell["name"])

    def step(self):
        active_jobs = get_active_slurm_jobs()
        total_active_queued = len(active_jobs)
        active_names = {j["name"]: j for j in active_jobs}

        print("\n" + "=" * 70)
        print(f"[queue_manager] [{time.strftime('%Y-%m-%d %H:%M:%S')}] 24/7 Breadth Campaign Heartbeat ({self.task_name})")
        print(f"[queue_manager] Active/Queued Jobs in SLURM: {total_active_queued} | Completed: {len(self.known_completed)}/{len(self.all_cells)}")

        # Check job transitions
        for j in active_jobs:
            jid = j["id"]
            jname = j["name"]
            jstate = j["state"]
            jnode = j["node"]

            # Job started
            if jstate == "RUNNING" and jid not in self.known_running:
                self.known_running[jid] = jnode
                msg = (
                    f"▶️ <b>Job Started Running</b>\n"
                    f"• <b>Job Name:</b> <code>{jname}</code> (Job <code>{jid}</code>)\n"
                    f"• <b>Node:</b> <code>{jnode}</code>\n"
                    f"• <b>Task:</b> {self.task_name} | <b>Time:</b> {time.strftime('%H:%M:%S IST')}"
                )
                send_telegram(msg)

            # Prompt progress
            if jstate == "RUNNING":
                pct, count, speed = get_job_progress(jid, self.max_samples)
                if pct is not None:
                    last_pct = self.last_progress_pct.get(jid, 0)
                    if pct >= last_pct + 20 or (last_pct == 0 and pct >= 10):
                        self.last_progress_pct[jid] = pct
                        msg = (
                            f"⏳ <b>Milestone Progress ({pct}%)</b>\n"
                            f"• <b>Job:</b> <code>{jname}</code> (<code>{jid}</code>)\n"
                            f"• <b>Progress:</b> {count} / {self.max_samples} prompts ({pct}%)\n"
                            f"• <b>Gen Speed:</b> <code>{speed}</code>\n"
                            f"• <b>Node:</b> <code>{jnode}</code> | <b>Elapsed:</b> {j['elapsed']}"
                        )
                        send_telegram(msg)

        # Check newly completed cells
        for cell in self.all_cells:
            cname = cell["name"]
            done, acc, score = is_cell_completed(cell, self.validation_dir, self.max_samples, self.task_name)
            if done and cname not in self.known_completed:
                self.known_completed.add(cname)
                msg = (
                    f"🎉 <b>Job COMPLETED &amp; VALIDATED</b>\n"
                    f"• <b>Cell:</b> <code>{cname}</code>\n"
                    f"• <b>Model:</b> <code>{os.path.basename(cell['model'])}</code>\n"
                    f"• <b>Accuracy (Pass@1):</b> <b>{acc}</b> ({score})\n"
                    f"• <b>Seed:</b> {cell['seed']} | <b>Task:</b> {self.task_name} ($n={self.max_samples}$)"
                )
                send_telegram(msg)

        # 2-Channel Autonomous Dispatching with Work-Stealing / Load-Balancing
        # (If Qwen finishes all cells first, GPU 1 automatically helps run remaining Llama cells)
        for channel_name, channel_cells in [("qwen", self.qwen_cells), ("llama", self.llama_cells)]:
            channel_active_jobs = [j for j in active_jobs if any(c["name"] == j["name"] for c in self.all_cells)]
            
            # Find active jobs currently assigned to this channel
            chan_jobs = [j for j in active_jobs if any(c["name"] == j["name"] for c in channel_cells)]
            last_job_id = None
            if chan_jobs:
                sorted_jobs = sorted(chan_jobs, key=lambda x: int(x["id"]))
                last_job_id = sorted_jobs[-1]["id"]

            channel_job_count = len(chan_jobs)

            # Primary queue: submit cells designated for this channel
            for cell in channel_cells:
                cname = cell["name"]
                done, acc, score = is_cell_completed(cell, self.validation_dir, self.max_samples, self.task_name)
                if done or cname in active_names:
                    continue

                if channel_job_count >= 2 or total_active_queued >= 4:
                    break

                job_id = submit_cell(
                    cell,
                    output_root=self.output_root,
                    task_name=self.task_name,
                    max_samples=self.max_samples,
                    prev_job_id=last_job_id
                )
                if job_id:
                    last_job_id = job_id
                    channel_job_count += 1
                    total_active_queued += 1
                    active_names[cname] = {"id": job_id, "name": cname, "state": "PENDING", "node": "-", "elapsed": "0:00"}

            # Work-Stealing: If this channel has 0 active jobs (all primary cells done),
            # steal remaining unassigned cells from the other model to keep both GPUs 100% busy!
            if channel_job_count == 0 and total_active_queued < 4:
                other_cells = self.llama_cells if channel_name == "qwen" else self.qwen_cells
                for cell in other_cells:
                    cname = cell["name"]
                    done, acc, score = is_cell_completed(cell, self.validation_dir, self.max_samples, self.task_name)
                    if done or cname in active_names:
                        continue

                    if total_active_queued >= 4 or channel_job_count >= 2:
                        break

                    print(f"[queue_manager] ⚡ WORK-STEALING: Channel {channel_name.upper()} taking over {cname} on idle GPU!")
                    job_id = submit_cell(
                        cell,
                        output_root=self.output_root,
                        task_name=self.task_name,
                        max_samples=self.max_samples,
                        prev_job_id=None  # Can run immediately since this GPU is completely free
                    )
                    if job_id:
                        channel_job_count += 1
                        total_active_queued += 1
                        active_names[cname] = {"id": job_id, "name": cname, "state": "ACTIVE", "node": "-", "elapsed": "0:00"}

        # Hourly status ping
        now = time.time()
        if now - self.last_summary_time > 3600:
            self.last_summary_time = now
            completed_count = len(self.known_completed)
            total_cells = len(self.all_cells)
            msg = (
                f"📊 <b>24/7 Breadth Campaign Hourly Status</b>\n"
                f"• <b>Task:</b> {self.task_name} ($n={self.max_samples}$)\n"
                f"• <b>Completed:</b> {completed_count} / {total_cells} ({completed_count / max(1, total_cells):.1%})\n"
                f"• <b>Active SLURM Jobs:</b> {total_active_queued}\n"
                f"• <b>Channel 1 (Qwen):</b> {len([j for j in active_jobs if 'qwen' in j['name'].lower()])} active/queued\n"
                f"• <b>Channel 2 (Llama):</b> {len([j for j in active_jobs if 'llama' in j['name'].lower()])} active/queued"
            )
            send_telegram(msg)


def main():
    parser = argparse.ArgumentParser(description="Autonomous 24/7 HPC Queue Manager Daemon")
    parser.add_argument("--config", type=str, default="configs/campaign_cells_gsm8k.json", help="Path to campaign cell config JSON")
    args = parser.parse_args()

    config_path = QR / args.config if not os.path.isabs(args.config) else Path(args.config)
    print(f"[queue_manager] Starting 24/7 Queue Manager Daemon using config: {config_path}")
    
    tracker = CampaignTracker(config_path)
    send_telegram(
        f"🔔 <b>24/7 Breadth Queue Manager Daemon Active</b>\n"
        f"• <b>Task:</b> {tracker.task_name} ($n={tracker.max_samples}$)\n"
        f"• <b>Allocations:</b> 2 Channels (1 GPU Qwen + 1 GPU Llama)\n"
        f"• <b>Total Cells:</b> {len(tracker.all_cells)} across 4 formats &amp; 3 seeds."
    )

    while True:
        try:
            tracker.step()
        except Exception as e:
            print(f"[queue_manager] Error in daemon step: {e}")
        time.sleep(45)


if __name__ == "__main__":
    main()
