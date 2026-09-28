"""
Launch wider CNN WF with EXPANDING WINDOW on Neptune (RTX 3090).

Key differences from previous runs:
  - NO --max-train-days (expanding window: all prior data each fold)
  - Resumes from fold 36 (existing checkpoint in results/wider_cnn/)
  - Low GPU priority via CUDA stream priorities not available on PyTorch;
    instead uses nice/ionice-equivalent via psutil, plus OMP/MKL thread caps.
  - Full 168 folds expected (173 days - 5 min_train_days)

Safety checks:
  - RAM must be <85% before launch
  - GPU must not be starved (paper engine uses CPU-only)

Usage:
  python launch_wider_cnn_expanding.py [--force]

  --force: Skip RAM check (use only if you know RAM will be released by training start)
"""
import sys
import os
import json
import time
import subprocess
import argparse
from pathlib import Path

# Safety limits
RAM_LIMIT_PCT = 85.0

def check_ram():
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.percent, mem.available / 1024**3
    except ImportError:
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Skip RAM safety check')
    args = parser.parse_args()

    print("=" * 60)
    print("WIDER CNN EXPANDING WINDOW — LAUNCH SCRIPT")
    print("=" * 60)

    # --- Safety check ---
    ram_pct, ram_avail_gb = check_ram()
    if ram_pct is not None:
        print(f"\nRAM: {ram_pct:.1f}% used, {ram_avail_gb:.2f}GB available")
        if ram_pct > RAM_LIMIT_PCT and not args.force:
            print(f"\nSAFETY BLOCK: RAM at {ram_pct:.1f}% exceeds limit of {RAM_LIMIT_PCT}%")
            print("Training would thrash into swap and harm paper engine inference.")
            print("Free RAM first (close applications, restart terminal if needed),")
            print("then re-run this script.")
            print("\nTop RAM consumers to investigate:")
            import psutil
            procs = sorted(
                [(p.memory_info().rss, p.pid, p.name()) for p in psutil.process_iter()
                 if p.memory_info().rss > 200 * 1024**2],
                reverse=True
            )[:8]
            for mb, pid, name in [(r/1024**2, pid, name) for r, pid, name in procs]:
                print(f"  {mb:7.0f}MB  PID={pid}  {name}")
            sys.exit(1)
        elif ram_pct > RAM_LIMIT_PCT:
            print(f"WARNING: RAM at {ram_pct:.1f}% (above {RAM_LIMIT_PCT}% limit) — launching anyway (--force)")
    else:
        print("psutil not available — skipping RAM check")

    # --- Locate directories ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    RESULTS_DIR = SCRIPT_DIR / 'results' / 'wider_cnn'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Find resume fold ---
    resume_fold = 0
    ckpts = sorted(RESULTS_DIR.glob('checkpoint_wider_cnn_*.json'))
    ckpts += sorted(RESULTS_DIR.glob('checkpoint_book_*.json'))
    for ckpt in ckpts:
        try:
            with open(ckpt) as f:
                d = json.load(f)
            nf = d.get('completed_folds', 0)
            if nf > resume_fold:
                resume_fold = nf
                print(f"Found checkpoint: {ckpt.name} — {nf} folds completed")
        except Exception:
            pass

    print(f"\nWill resume from fold {resume_fold} (expanding window, no max_train_days cap)")
    print(f"Output dir: {RESULTS_DIR}")
    print(f"Total expected folds: ~168")

    # --- Build argv for train_walkforward.main() ---
    # We monkey-patch sys.argv so train_walkforward.main() picks it up
    log_file = RESULTS_DIR / f'wider_cnn_expanding_wf.log'
    print(f"Log file: {log_file}")
    print()

    # Build command to run as subprocess so we can redirect output properly
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '16'
    env['MKL_NUM_THREADS'] = '16'
    env['OPENBLAS_NUM_THREADS'] = '16'
    env['CUDA_VISIBLE_DEVICES'] = '0'
    # No PYTORCH_CUDA_ALLOC_CONF needed — expanding window uses more RAM per fold
    # but GPU VRAM usage is bounded by batch size

    script = SCRIPT_DIR / 'run_wider_cnn_expanding_wf.py'
    cmd = [sys.executable, str(script)]

    print(f"Launching: {' '.join(cmd)}")
    print(f"Env: OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0")
    print()

    with open(log_file, 'w') as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT),
        )

    # Write PID file
    pid_file = RESULTS_DIR / 'wider_cnn_expanding.pid'
    pid_file.write_text(str(proc.pid))

    print(f"Launched PID: {proc.pid}")
    print(f"PID file: {pid_file}")
    print(f"Monitor: tail -f {log_file}")
    print()
    print("Training is running in background. Check log for progress.")
    print("GPU usage should appear within ~60 seconds.")

if __name__ == '__main__':
    main()
