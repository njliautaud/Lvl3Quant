"""Transfer MBO files from PC to Jupiter via SFTP over LAN."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'teleclaude-main'))
from pathlib import Path
import paramiko

JUPITER_IP = 'jupiter'
JUPITER_USER = 'jupiter'
JUPITER_PW = os.environ.get("CLUSTER_SSH_PASSWORD", "")
LOCAL_MBO = Path(r'C:\Users\Footb\Documents\Github\Lvl3Quant\mbo')
REMOTE_MBO = '/home/jupiter/lvl3quant/mbo'

def main():
    local_files = sorted(LOCAL_MBO.glob('*.zst'))
    print(f"Found {len(local_files)} MBO files to transfer")
    total_bytes = sum(f.stat().st_size for f in local_files)
    print(f"Total size: {total_bytes / 1e6:.0f} MB")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(JUPITER_IP, username=JUPITER_USER, password=JUPITER_PW, timeout=10)
    sftp = ssh.open_sftp()

    # Create remote dir
    try:
        sftp.mkdir(REMOTE_MBO)
    except:
        pass

    # Check what already exists
    existing = set()
    try:
        existing = set(sftp.listdir(REMOTE_MBO))
    except:
        pass

    to_transfer = [f for f in local_files if f.name not in existing]
    print(f"Already on Jupiter: {len(local_files) - len(to_transfer)}")
    print(f"Need to transfer: {len(to_transfer)}")

    transferred = 0
    t0 = time.time()
    for f in to_transfer:
        remote_path = f"{REMOTE_MBO}/{f.name}"
        size_mb = f.stat().st_size / 1e6
        sftp.put(str(f), remote_path)
        transferred += 1
        elapsed = time.time() - t0
        rate = transferred / elapsed if elapsed > 0 else 0
        remaining = (len(to_transfer) - transferred) / max(rate, 0.01)
        print(f"  [{transferred}/{len(to_transfer)}] {f.name} ({size_mb:.1f}MB) — {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"\nDone! Transferred {transferred} files in {elapsed:.0f}s")

    sftp.close()
    ssh.close()

if __name__ == '__main__':
    main()
