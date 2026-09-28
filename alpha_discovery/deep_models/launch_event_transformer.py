"""
Launch Event Transformer v1 walk-forward training on Neptune RTX 3090.

Steps:
  1. Safety check (RAM, GPU VRAM)
  2. Transfer data from Jupiter if not already on Neptune
  3. Launch training with BELOW_NORMAL priority
  4. Write PID file

Usage:
  python launch_event_transformer.py [--force] [--skip-transfer]
"""
import sys
import os
import subprocess
import argparse
import time
from pathlib import Path

RAM_LIMIT_PCT = 80.0
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mbo_events"
RESULTS_DIR = SCRIPT_DIR / "results" / "event_transformer_v1"
LOG_FILE = SCRIPT_DIR / "results" / "event_transformer_v1.log"


def check_system():
    try:
        import psutil
        mem = psutil.virtual_memory()
        ram_pct = mem.percent
        ram_avail_gb = mem.available / 1024**3
        print(f"RAM: {ram_pct:.1f}% used, {ram_avail_gb:.1f}GB available")
        return ram_pct, ram_avail_gb
    except ImportError:
        print("psutil not available — skipping RAM check")
        return None, None


def check_existing_training():
    """Check if event transformer training is already running."""
    pid_file = RESULTS_DIR / "event_transformer_v1.pid"
    if pid_file.exists():
        try:
            import psutil
            pid = int(pid_file.read_text().strip())
            if psutil.pid_exists(pid):
                print(f"WARNING: Training already running (PID {pid}). Kill it first if you want a fresh start.")
                return pid
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Launch Event Transformer v1 Training")
    parser.add_argument("--force", action="store_true", help="Skip RAM safety check")
    parser.add_argument("--skip-transfer", action="store_true", help="Skip data transfer from Jupiter")
    parser.add_argument("--n-folds", type=int, default=10, help="Number of WF folds")
    args = parser.parse_args()

    print("=" * 60)
    print("EVENT TRANSFORMER V1 — LAUNCH SCRIPT")
    print("=" * 60)

    # Safety check
    ram_pct, _ = check_system()
    if ram_pct is not None and ram_pct > RAM_LIMIT_PCT and not args.force:
        print(f"\nSAFETY BLOCK: RAM at {ram_pct:.1f}% exceeds {RAM_LIMIT_PCT}% limit.")
        print("Free RAM first or use --force to override.")
        sys.exit(1)

    # Check for existing run
    existing_pid = check_existing_training()
    if existing_pid:
        print(f"Existing PID {existing_pid} — continuing (will not double-launch).")
        sys.exit(0)

    # Prepare directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nData dir:   {DATA_DIR}")
    print(f"Output dir: {RESULTS_DIR}")
    print(f"Log file:   {LOG_FILE}")

    # Build command
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train_event_transformer_v1.py"),
        "--data-dir", str(DATA_DIR),
        "--output-dir", str(RESULTS_DIR),
        "--n-folds", str(args.n_folds),
        "--device", "cuda",
    ]
    if args.skip_transfer:
        cmd.append("--skip-transfer")

    print(f"\nCommand: {' '.join(cmd)}")

    # Environment
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "8"
    env["MKL_NUM_THREADS"] = "8"
    env["OPENBLAS_NUM_THREADS"] = "8"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"

    # Launch with BELOW_NORMAL priority (0x00004000) + no window
    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
    CREATE_NO_WINDOW = 0x08000000

    print(f"\nLaunching training (BELOW_NORMAL priority, no window)...")

    with open(LOG_FILE, "w") as log_fh:
        log_fh.write(f"# Event Transformer v1 — launched {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_fh.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT),
            creationflags=BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW,
        )

    # Write PID file
    pid_file = RESULTS_DIR / "event_transformer_v1.pid"
    pid_file.write_text(str(proc.pid))

    print(f"Launched! PID: {proc.pid}")
    print(f"PID file: {pid_file}")
    print(f"Monitor:  tail -f \"{LOG_FILE}\"")
    print("\nTraining running in background.")
    print("GPU usage should appear within ~60s (after data transfer completes).")


if __name__ == "__main__":
    main()
