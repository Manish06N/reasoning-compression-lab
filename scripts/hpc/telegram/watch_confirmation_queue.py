#!/usr/bin/env python3
"""Autonomous Telegram Queue Watcher Daemon for Measured Serving Confirmation Benchmark."""

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
RAW_DIR = REPO_ROOT / "results" / "measured_serving_confirmation" / "raw"
LOGS_DIR = REPO_ROOT / "logs"


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
            ["squeue", "-u", os.environ.get("USER", "manishn_iitp"), "--format=%i|%j|%T|%M|%N"],
            text=True,
            timeout=10,
        )
        for line in out.strip().splitlines()[1:]:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                jobs[parts[0]] = {
                    "name": parts[1],
                    "state": parts[2],
                    "time": parts[3],
                    "node": parts[4],
                }
    except Exception as e:
        print(f"[ERROR] squeue check failed: {e}")
    return jobs


def count_completed_runs() -> Dict[str, int]:
    """Count completed confirmation runs in raw directory."""
    if not RAW_DIR.exists():
        return {"condA": 0, "condB": 0, "micro": 0, "total": 0}

    condA = len(list(RAW_DIR.glob("*_condA.json")))
    condB = len(list(RAW_DIR.glob("*_condB.json")))
    micro = len(list(RAW_DIR.glob("*_microbenchmark.json")))
    return {"condA": condA, "condB": condB, "micro": micro, "total": condA + condB + micro}


def main():
    print(f"[{datetime.now().isoformat()}] Starting Confirmation Queue Watcher Daemon...")
    send_tg(
        "🚀 <b>Confirmation Serving Benchmark Daemon Started</b>\n"
        "• Tracking SLURM job `srv_confirm`\n"
        "• Target: 8 configurations on single physical A100 node\n"
        "• Strict apples-to-apples protocol active"
    )

    prev_counts = count_completed_runs()
    prev_jobs: Dict[str, Dict[str, str]] = {}
    total_target = 8 * 3 * 2 + 8  # 48 task runs + 8 micro = 56 total

    while True:
        curr_jobs = get_active_slurm_jobs()
        curr_counts = count_completed_runs()

        # Check job transitions
        for jid, jinfo in curr_jobs.items():
            if jid not in prev_jobs:
                send_tg(
                    f"▶️ <b>SLURM Job Active</b>: {jinfo['name']} (ID: <code>{jid}</code>)\n"
                    f"• State: <code>{jinfo['state']}</code>\n"
                    f"• Node: <code>{jinfo['node']}</code>"
                )
            elif jinfo["state"] != prev_jobs[jid]["state"] or jinfo["node"] != prev_jobs[jid]["node"]:
                send_tg(
                    f"🔄 <b>Job Update</b>: {jinfo['name']} (ID: <code>{jid}</code>)\n"
                    f"• State: <code>{jinfo['state']}</code>\n"
                    f"• Node: <code>{jinfo['node']}</code> | Time: {jinfo['time']}"
                )

        # Notify on completed runs progress
        if curr_counts["total"] > prev_counts["total"]:
            delta = curr_counts["total"] - prev_counts["total"]
            pct = (curr_counts["total"] / total_target) * 100.0
            send_tg(
                f"📊 <b>Confirmation Progress</b>: +{delta} runs complete ({curr_counts['total']}/{total_target}, {pct:.1f}%)\n"
                f"• Condition A (C=1): {curr_counts['condA']}/24\n"
                f"• Condition B (C=8, max_num_seqs=8): {curr_counts['condB']}/24\n"
                f"• Microbenchmarks: {curr_counts['micro']}/8"
            )

        # Check if all runs are complete and queue is drained
        if curr_counts["total"] >= total_target and not curr_jobs:
            send_tg(
                "✅ <b>All 56 Confirmation Runs Completed!</b>\n"
                "• Running audit validation and statistical analysis..."
            )
            # Run validation & analysis
            val_cmd = [
                "/home/manishn_iitp/.conda/envs/qrm-official/bin/python3",
                str(REPO_ROOT / "scripts" / "hpc" / "qrm_parity" / "validate_measured_serving_confirmation.py"),
            ]
            ana_cmd = [
                "/home/manishn_iitp/.conda/envs/qrm-official/bin/python3",
                str(REPO_ROOT / "scripts" / "analysis" / "measured_serving_confirmation_analysis.py"),
            ]
            try:
                subprocess.run(val_cmd, check=True)
                subprocess.run(ana_cmd, check=True)
                send_tg("🎉 <b>Confirmation Validation & Analysis Completed Cleanly!</b>")
            except Exception as e:
                send_tg(f"⚠️ <b>Post-run Analysis Error</b>: {e}")
            break

        # If job completed/disappeared before reaching total target
        for jid in prev_jobs:
            if jid not in curr_jobs and curr_counts["total"] < total_target:
                send_tg(
                    f"⚠️ <b>SLURM Job Terminated</b> (ID: <code>{jid}</code>)\n"
                    f"• Runs completed: {curr_counts['total']}/{total_target}"
                )

        prev_jobs = curr_jobs
        prev_counts = curr_counts
        time.sleep(30)


if __name__ == "__main__":
    main()
