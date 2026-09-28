import subprocess
import os

PASS = os.environ.get("CLUSTER_SSH_PASSWORD", "")
HOST = 'nick@winnode'
SRC = '/home/jupiter/Lvl3Quant/data/processed/mbo_tensors_mamba/'
DEST = 'C:/Users/nick/Lvl3Quant/data/processed/mbo_tensors_mamba/'
SSH_OPTS = ['-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=30',
            '-o', 'PasswordAuthentication=yes', '-o', 'PubkeyAuthentication=no']

# Step 1: Create dest dir
r = subprocess.run(
    ['sshpass', '-p', PASS, 'ssh'] + SSH_OPTS + [HOST, 'mkdir', '-p', DEST],
    capture_output=True, text=True, timeout=30
)
print('mkdir stdout:', r.stdout)
print('mkdir stderr:', r.stderr)
print('mkdir rc:', r.returncode)

# Step 2: Count source files
files = sorted([f for f in os.listdir(SRC) if f.endswith('.pt')])
print(f'Files to sync: {len(files)}')

# Step 3: SCP in batches of 20 to avoid arg list too long
BATCH = 20
completed = 0
for i in range(0, len(files), BATCH):
    batch = files[i:i+BATCH]
    src_files = [SRC + f for f in batch]
    r = subprocess.run(
        ['sshpass', '-p', PASS, 'scp',
         '-o', 'StrictHostKeyChecking=no',
         '-o', 'PasswordAuthentication=yes',
         '-o', 'PubkeyAuthentication=no',
         '-o', 'ConnectTimeout=30'] + src_files + [f'{HOST}:{DEST}'],
        capture_output=True, text=True, timeout=600
    )
    completed += len(batch)
    print(f'Batch {i//BATCH+1}: {len(batch)} files, rc={r.returncode}, done={completed}/{len(files)}')
    if r.returncode != 0:
        print('STDERR:', r.stderr[:500])

print(f'SYNC_COMPLETE {completed} files transferred')
raise RuntimeError(f'DONE transferred={completed} total_files={len(files)}')
