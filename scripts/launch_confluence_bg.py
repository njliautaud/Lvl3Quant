"""Background launcher for confluence_meta_v1 training on Windows.
Spawns the training as a fully detached process that survives SSH disconnect.
"""
import subprocess, sys, os

script = r"C:\Users\claude\Lvl3Quant\scripts\train_confluence_meta_v1.py"
log_out = r"C:\Users\claude\Lvl3Quant\output\confluence_meta_v1\full_run.log"
log_err = r"C:\Users\claude\Lvl3Quant\output\confluence_meta_v1\full_run_err.log"

# DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP = survives parent exit
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

with open(log_out, "w") as fout, open(log_err, "w") as ferr:
    p = subprocess.Popen(
        [sys.executable, "-u", script, "--resume"],
        stdout=fout,
        stderr=ferr,
        cwd=r"C:\Users\claude\Lvl3Quant",
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
    )
    print(f"Launched PID {p.pid}")
