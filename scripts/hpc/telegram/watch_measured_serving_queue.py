#!/usr/bin/env python3
"""Autonomous Telegram Queue Watcher Daemon for Measured Serving Benchmark."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path("/scratch/manishn_iitp/reasoning-compression-lab")
RAW_DIR = REPO_ROOT / "results" / "measured_serving" / "raw"
LOGS_DIR = REPO_ROOT / "logs" / "measured_serving"


def get_telegram_creds() -> tuple[str, str]:
    """Load Telegram bot token and chat ID."""
    token = os.environ.get("TG_TOKEN", "8738869628:AAEtrsLVoqvDXeNz6CEa-ym5-AY7VJKScZ4")
    chat_id = os.environ.get("TG_CHAT_ID", "638098622")
    sh_file = Path(os.path.expanduser("~/watch-job-52772.sh"))
    if sh_file.exists():
        for line in sh_file.read_text().splitlines():
            if line.startswith("TG_TOKEN="):
                token = line.split("=", 1)[1].strip("\"' ")
            elif line.startswith("TG_CHAT_ID="):
                chat_id = line.split("=", 1)[1].strip("\"' ")
    return token, chat_id


TG_TOKEN, TG_CHAT_ID = get_telegram_creds()


def send_tg(text: str) -> bool:
    """Send message to Telegram with HTML formatting."""
    if not TG_TOKEN or not TG_CHAT_ID:
        print(f"[WARN] Missing Telegram credentials: token={bool(TG_TOKEN)}, chat_id={TG_CHAT_ID}")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": TG_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=15) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            return bool(res.get("ok"))
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram message: {e}")
        return False


def get_active_slurm_jobs() -> Dict[str, Dict[str, str]]:
    """Query current user jobs from SLURM."""
    jobs: Dict[str, Dict[str, str]] = {}
    try:
        out = subprocess.check_output(
            ["squeue", "-u", "manishn_iitp", "-o", "%i|%j|%T|%M|%N", "--noheader"],
            text=True,
            timeout=10,
        )
        for line in out.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                job_id, name, state, elapsed, node = parts[:5]
                jobs[job_id] = {
                    "job_id": job_id,
                    "name": name,
                    "state": state,
                    "elapsed": elapsed,
                    "node": node,
                }
    except Exception as e:
        print(f"[WARN] squeue query failed: {e}")
    return jobs


def main():
    print(f"Starting Telegram Queue Watcher Daemon for {REPO_ROOT}...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    seen_raw_files: Set[str] = set(f.name for f in RAW_DIR.glob("*.json"))
    previous_jobs = get_active_slurm_jobs()
    last_heartbeat = time.time()

    initial_running = [f"{j['name']} ({j['node']})" for j in previous_jobs.values() if j["state"] == "RUNNING"]
    initial_pending = [j["name"] for j in previous_jobs.values() if j["state"] == "PENDING"]

    start_msg = (
        "🚀 <b>PARAM Rudra Serving Benchmark Watcher Started</b>\n\n"
        f"• <b>Active Jobs:</b> {len(previous_jobs)}\n"
        f"• <b>Currently Running:</b> {', '.join(initial_running) if initial_running else 'None'}\n"
        f"• <b>Queued:</b> {len(initial_pending)} jobs\n"
        "• <b>Polling Interval:</b> 30s"
    )
    send_tg(start_msg)

    while True:
        try:
            time.sleep(30)
            current_jobs = get_active_slurm_jobs()

            # 1. Check for newly written raw result JSONs
            current_raw_files = set(f.name for f in RAW_DIR.glob("*.json"))
            new_raw_files = current_raw_files - seen_raw_files
            for rf_name in sorted(new_raw_files):
                seen_raw_files.add(rf_name)
                f_path = RAW_DIR / rf_name
                try:
                    data = json.loads(f_path.read_text(encoding="utf-8"))
                    btype = data.get("benchmark_type", "")
                    model = data.get("model", "")
                    fmt = data.get("format", "")

                    if btype == "task_realistic":
                        cond = data.get("condition", "")
                        rep = data.get("repetition", 1)
                        tok_s = data.get("output_tokens_per_second", 0.0)
                        lat_med = data.get("latency_median_sec", data.get("gpu_seconds_per_query", 0.0))
                        vram = data.get("peak_vram_allocated_gb", 0.0)
                        gpu_s = data.get("gpu_seconds_per_query", 0.0)

                        cond_label = "Interactive (C=1)" if "condA" in rf_name else "Batched (C=8)"
                        msg = (
                            f"⚡ <b>Repetition Finished: {model} {fmt}</b>\n\n"
                            f"• <b>Condition:</b> {cond_label} (Rep {rep}/3)\n"
                            f"• <b>Output Speed:</b> <code>{tok_s:.2f} tok/s</code>\n"
                            f"• <b>Median Latency / GPU-sec:</b> <code>{lat_med:.2f}s</code> (GPU-sec/q: <code>{gpu_s:.2f}s</code>)\n"
                            f"• <b>Peak VRAM:</b> <code>{vram:.2f} GB</code>\n"
                            f"• <b>File:</b> <code>{rf_name}</code>"
                        )
                        send_tg(msg)

                    elif btype == "fixed_token_microbenchmark":
                        tok_s = data.get("raw_decode_tokens_per_second", 0.0)
                        msg = (
                            f"🔬 <b>Microbenchmark Done: {model} {fmt}</b>\n\n"
                            f"• <b>Fixed Token Decode:</b> 512 tokens\n"
                            f"• <b>Raw Decode Speed:</b> <code>{tok_s:.2f} tok/s</code>"
                        )
                        send_tg(msg)
                except Exception as e:
                    print(f"[WARN] Failed to parse {rf_name}: {e}")

            # 2. Check for state transitions (Pending -> Running, Running -> Done)
            for jid, jinfo in current_jobs.items():
                if jid in previous_jobs:
                    prev_state = previous_jobs[jid]["state"]
                    curr_state = jinfo["state"]
                    if prev_state != curr_state:
                        if curr_state == "RUNNING":
                            send_tg(
                                f"🟢 <b>Job Started Running: {jinfo['name']}</b>\n\n"
                                f"• <b>Job ID:</b> <code>{jid}</code>\n"
                                f"• <b>Compute Node:</b> <code>{jinfo['node']}</code> (A100 80GB)"
                            )
                else:
                    # New job discovered
                    if jinfo["state"] == "RUNNING":
                        send_tg(
                            f"🟢 <b>Job Active: {jinfo['name']}</b>\n\n"
                            f"• <b>Job ID:</b> <code>{jid}</code>\n"
                            f"• <b>Node:</b> <code>{jinfo['node']}</code>"
                        )

            for jid, pinfo in previous_jobs.items():
                if jid not in current_jobs:
                    # Job finished or left queue
                    send_tg(
                        f"🏁 <b>Job Exited Queue: {pinfo['name']}</b> (ID: <code>{jid}</code>)"
                    )

            previous_jobs = current_jobs

            # 3. Heartbeat every 20 minutes
            if time.time() - last_heartbeat > 1200:
                last_heartbeat = time.time()
                n_completed_runs = len(current_raw_files)
                running_names = [f"{j['name']} ({j['node']})" for j in current_jobs.values() if j["state"] == "RUNNING"]
                pending_count = sum(1 for j in current_jobs.values() if j["state"] == "PENDING")
                send_tg(
                    f"💓 <b>Benchmark Heartbeat (20m)</b>\n\n"
                    f"• <b>Completed Run Files:</b> {n_completed_runs}/56\n"
                    f"• <b>Active Running:</b> {', '.join(running_names) if running_names else 'None'}\n"
                    f"• <b>Pending in Queue:</b> {pending_count}"
                )

            # 4. Check if all jobs finished
            if len(current_jobs) == 0 and len(seen_raw_files) >= 56:
                send_tg(
                    "🎉 <b>ALL 8 SERVING BENCHMARK JOBS COMPLETED!</b>\n\n"
                    "All 48 task-realistic + 8 microbenchmark runs are saved on disk.\n"
                    "Executing automated validation & report generation now..."
                )
                break

        except Exception as e:
            print(f"[ERROR] Watcher loop exception: {e}")
            time.sleep(30)

    # Automatically run validation and statistical analysis once finished
    try:
        val_script = REPO_ROOT / "scripts" / "hpc" / "qrm_parity" / "validate_measured_serving.py"
        analysis_script = REPO_ROOT / "scripts" / "analysis" / "measured_serving_analysis.py"
        py_bin = "/home/manishn_iitp/.conda/envs/qrm-official/bin/python3"

        subprocess.run([py_bin, str(val_script)], check=True)
        subprocess.run([py_bin, str(analysis_script)], check=True)

        send_tg(
            "📊 <b>Measured Serving Analysis & Validation Complete!</b>\n\n"
            "Full reports and tables generated in <code>results/reports/measured_serving/</code>."
        )
    except Exception as e:
        send_tg(f"⚠️ <b>Post-processing error:</b> {e}")


if __name__ == "__main__":
    main()
