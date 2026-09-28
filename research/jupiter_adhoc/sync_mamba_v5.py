"""Ray job that writes sync_mamba_v5.py to disk, then runs it inline."""
import subprocess
import os
import time

PASS = os.environ.get("CLUSTER_SSH_PASSWORD", "")
HOST = "nick@winnode"
SRC = "/home/jupiter/Lvl3Quant/data/processed/mbo_tensors_mamba/"
DEST_SCP = "C:/Users/nick/Lvl3Quant/data/processed/mbo_tensors_mamba/"
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
            "-o", "PasswordAuthentication=yes", "-o", "PubkeyAuthentication=no"]

# Ensure dest dir exists via PowerShell
mkdir_cmd = "powershell -Command New-Item -ItemType Directory -Force -Path C:\\Users\\nick\\Lvl3Quant\\data\\processed\\mbo_tensors_mamba"
r = subprocess.run(
    ["sshpass", "-p", PASS, "ssh"] + SSH_OPTS + [HOST, mkdir_cmd],
    capture_output=True, text=True, timeout=30
)
print(f"mkdir rc={r.returncode} out={r.stdout[:100]!r}")

# Get files already on Uranus
r2 = subprocess.run(
    ["sshpass", "-p", PASS, "ssh"] + SSH_OPTS + [HOST,
     "dir /b C:\\Users\\nick\\Lvl3Quant\\data\\processed\\mbo_tensors_mamba\\*.pt"],
    capture_output=True, text=True, timeout=30
)
existing = set()
if r2.returncode == 0:
    for line in r2.stdout.splitlines():
        line = line.strip()
        if line.endswith(".pt"):
            existing.add(line)
print(f"Already on Uranus: {len(existing)} files")

# All source files
files = sorted([f for f in os.listdir(SRC) if f.endswith(".pt")])
todo = [f for f in files if f not in existing]
print(f"Total: {len(files)}, To sync: {len(todo)}, Skipping: {len(existing)}")

completed = 0
errors = 0
for f in todo:
    src_path = SRC + f
    t0 = time.time()
    r = subprocess.run(
        ["sshpass", "-p", PASS, "scp"] + SSH_OPTS[:4] + [src_path, f"{HOST}:{DEST_SCP}"],
        capture_output=True, text=True, timeout=600
    )
    elapsed = time.time() - t0
    fsize_mb = os.path.getsize(src_path) / 1e6
    if r.returncode != 0:
        errors += 1
        print(f"  FAIL {f}: rc={r.returncode} t={elapsed:.0f}s err={r.stderr[:150]!r}")
    else:
        completed += 1
        print(f"  OK {f}: {fsize_mb:.0f}MB in {elapsed:.0f}s ({fsize_mb/max(elapsed,1):.1f}MB/s) [{completed}/{len(todo)}]")

raise RuntimeError(f"DONE completed={completed} errors={errors} total={len(todo)} skipped={len(existing)}")
