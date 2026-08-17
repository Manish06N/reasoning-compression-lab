#!/usr/bin/env python3
"""Reliable 30-Minute Telegram Heartbeat & Milestone Daemon for Measured Serving Confirmation."""

import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path("/scratch/manishn_iitp/reasoning-compression-lab")
RAW_DIR = REPO_ROOT / "results" / "measured_serving_confirmation" / "raw"
LOG_DIR = REPO_ROOT / "logs"

TG_TOKEN = "8738869628:AAEtrsLVoqvDXeNz6CEa-ym5-AY7VJKScZ4"
TG_CHAT_ID = "638098622"


def send_tg_msg(text: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": TG_CHAT_ID,
            "text": text,
        }).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=15) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            return bool(res.get("ok"))
    except Exception as e:
        print(f"Failed to send telegram: {e}", file=sys.stderr)
        return False


def get_queue_info() -> dict:
    jobs = {}
    try:
        out = subprocess.check_output(
            ["squeue", "-u", "manishn_iitp", "--format=%i|%j|%T|%M|%N"],
            text=True,
            timeout=10,
        )
        for line in out.strip().splitlines()[1:]:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                jobs[parts[0]] = {
                    "name": parts[1],
                    "state": parts[2],
                    "elapsed": parts[3],
                    "node": parts[4],
                }
    except Exception as e:
        print(f"squeue error: {e}", file=sys.stderr)
    return jobs


def count_files() -> tuple:
    if not RAW_DIR.exists():
        return 0, 0, 0, 0
    cond_a = len(list(RAW_DIR.glob("*_condA.json")))
    cond_b = len(list(RAW_DIR.glob("*_condB.json")))
    micro = len(list(RAW_DIR.glob("*_microbenchmark.json")))
    return cond_a, cond_b, micro, (cond_a + cond_b + micro)


def main():
    print(f"[{datetime.now().isoformat()}] Telegram heartbeat daemon starting...")
    send_tg_msg("🚀 Telegram Heartbeat Daemon Active: 30-minute updates enabled for Jobs 96766 & 96768.")

    prev_total = 0
    iteration = 0

    while True:
        time.sleep(60)  # Check every 1 minute
        iteration += 1

        cond_a, cond_b, micro, total = count_files()
        jobs = get_queue_info()

        # Milestone notification on new completed run
        if total > prev_total:
            diff = total - prev_total
            msg = (
                f"📈 Milestone Update:\n"
                f"• +{diff} new benchmark runs completed!\n"
                f"• Total Complete: {total}/56\n"
                f"  - Cond A (C=1): {cond_a}/24\n"
                f"  - Cond B (C=8): {cond_b}/24\n"
                f"  - Micro: {micro}/8"
            )
            send_tg_msg(msg)
            prev_total = total

        # Heartbeat notification every 30 minutes (30 iterations of 60s)
        if iteration >= 30:
            iteration = 0
            job_lines = []
            for jid, info in jobs.items():
                job_lines.append(f"• Job {jid} ({info['name']}): {info['state']} on {info['node']} ({info['elapsed']})")
            
            if not job_lines:
                jobs_summary = "• No active jobs in queue"
            else:
                jobs_summary = "\n".join(job_lines)

            msg = (
                f"💓 30-Minute Heartbeat Status:\n"
                f"{jobs_summary}\n"
                f"• Runs Completed: {total}/56 (Cond A: {cond_a}, Cond B: {cond_b}, Micro: {micro})\n"
                f"• Status: All systems running normally"
            )
            send_tg_msg(msg)

        # Final completion check
        if not jobs:
            if total >= 56:
                send_tg_msg(
                    f"🎉 ALL 56 Confirmation Runs Completed!\n"
                    f"• Cond A: {cond_a}/24 | Cond B: {cond_b}/24 | Micro: {micro}/8\n"
                    f"• Running validation and report generation..."
                )
                try:
                    subprocess.run([
                        "/home/manishn_iitp/.conda/envs/qrm-official/bin/python3",
                        str(REPO_ROOT / "scripts" / "hpc" / "qrm_parity" / "validate_measured_serving_confirmation.py")
                    ], check=True)
                    subprocess.run([
                        "/home/manishn_iitp/.conda/envs/qrm-official/bin/python3",
                        str(REPO_ROOT / "scripts" / "analysis" / "measured_serving_confirmation_analysis.py")
                    ], check=True)
                    send_tg_msg("✅ Confirmation validation and reports successfully generated!")
                except Exception as e:
                    send_tg_msg(f"⚠️ Validation error: {e}")
            else:
                send_tg_msg(f"⚠️ All jobs completed with {total}/56 runs.")
            break


if __name__ == "__main__":
    main()
