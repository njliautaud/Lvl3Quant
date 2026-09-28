#!/usr/bin/env python3
import subprocess
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [track2_queue] %(message)s",
    handlers=[
        logging.FileHandler("/home/jupiter/Lvl3Quant/logs/track2_queue.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("track2_queue")

LVL3 = Path("/home/jupiter/Lvl3Quant")
scripts = [
    ("queue_position_sweep.py",        "queue_pos_sweep.log"),
    ("time_of_day_fillsim.py",          "time_of_day_fillsim.log"),
    ("signal_strength_tpsl_interaction.py", "signal_strength_tpsl.log"),
]


def is_running(name):
    r = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True)
    return bool(r.stdout.strip())


# Wait for 2A, then launch 2B, then 2C
for script, logname in scripts[1:]:  # skip 2A, already running
    log.info(f"Waiting for queue_position_sweep.py to finish before launching {script}...")
    while is_running("queue_position_sweep.py"):
        time.sleep(30)
    # Also wait for any previous script in sequence
    prev_script = scripts[scripts.index((script, logname)) - 1][0]
    while is_running(prev_script):
        time.sleep(30)

    log.info(f"Launching {script}...")
    log_path = LVL3 / "logs" / logname
    proc = subprocess.Popen(
        ["python3", str(LVL3 / "scripts" / script)],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        cwd=str(LVL3)
    )
    log.info(f"Launched {script} PID={proc.pid}")
    # Wait for it to complete before launching next
    proc.wait()
    log.info(f"{script} completed (exit code {proc.returncode}).")

log.info("All Track 2 experiments complete.")
