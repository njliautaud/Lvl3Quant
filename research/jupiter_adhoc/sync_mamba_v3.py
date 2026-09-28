import subprocess
import os

PASS = os.environ.get("CLUSTER_SSH_PASSWORD", "")
HOST = "nick@winnode"
SRC = "/home/jupiter/Lvl3Quant/data/processed/mbo_tensors_mamba/"
DEST_WIN = r"C:\Users\nick\Lvl3Quant\data\processed\mbo_tensors_mamba"
DEST_SCP = "C:/Users/nick/Lvl3Quant/data/processed/mbo_tensors_mamba/"
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
            "-o", "PasswordAuthentication=yes", "-o", "PubkeyAuthentication=no"]

# Step 1: Create dest dir — try powershell (most reliable on Windows SSH)
mkdir_cmds = [
    "powershell -Command \"New-Item -ItemType Directory -Force -Path 'C:\\Users\\nick\\Lvl3Quant\\data\\processed\\mbo_tensors_mamba'\"",
    "mkdir C:\\Users\\nick\\Lvl3Quant\\data\\processed\\mbo_tensors_mamba",
    "md C:\\Users\\nick\\Lvl3Quant\\data\\processed\\mbo_tensors_mamba",
]
dir_ok = False
for cmd in mkdir_cmds:
    r = subprocess.run(
        ["sshpass", "-p", PASS, "ssh"] + SSH_OPTS + [HOST, cmd],
        capture_output=True, text=True, timeout=30
    )
    print(f"mkdir rc={r.returncode} cmd={cmd[:60]} out={r.stdout[:100]!r}")
    if r.returncode == 0:
        dir_ok = True
        print("DIR created OK")
        break

if not dir_ok:
    print("WARNING: mkdir failed, attempting scp anyway")

# Step 2: Verify dir exists via dir command
r = subprocess.run(
    ["sshpass", "-p", PASS, "ssh"] + SSH_OPTS + [HOST,
     "dir C:\\Users\\nick\\Lvl3Quant\\data\\processed\\mbo_tensors_mamba"],
    capture_output=True, text=True, timeout=30
)
print(f"dir check rc={r.returncode} out={r.stdout[:300]!r}")

# Step 3: SCP in batches of 20
files = sorted([f for f in os.listdir(SRC) if f.endswith(".pt")])
print(f"Files to sync: {len(files)}")

BATCH = 5  # ~2.25GB per batch at ~450MB/file — fits in 300s on LAN
completed = 0
errors = 0
for i in range(0, len(files), BATCH):
    batch = files[i:i+BATCH]
    src_files = [SRC + f for f in batch]
    r = subprocess.run(
        ["sshpass", "-p", PASS, "scp",
         "-o", "StrictHostKeyChecking=no",
         "-o", "PasswordAuthentication=yes",
         "-o", "PubkeyAuthentication=no",
         "-o", "ConnectTimeout=30"] + src_files + [f"{HOST}:{DEST_SCP}"],
        capture_output=True, text=True, timeout=300
    )
    completed += len(batch)
    if r.returncode != 0:
        errors += 1
        print(f"Batch {i//BATCH+1}: rc={r.returncode} STDERR={r.stderr[:300]!r}")
    else:
        print(f"Batch {i//BATCH+1}: OK done={completed}/{len(files)}")

raise RuntimeError(f"DONE transferred={completed} errors={errors} total={len(files)}")
