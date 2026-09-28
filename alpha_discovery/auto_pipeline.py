"""
Auto-Pipeline Monitor — Watches Phase 1A completion and chains Phase 1B.

Monitors the Phase 1A log file for completion, then automatically launches
Phase 1B (MFE targets) using the pre-computed Rust feature cache.

Usage:
    python alpha_discovery/auto_pipeline.py --watch-log LOGFILE [--pid PID]
"""

import os
import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
FEATURE_CACHE = str(LVL3_ROOT / "data" / "processed" / "mbo_features_cache")
PYTHON = sys.executable

from alpha_discovery.process_registry import ProcessRegistry
registry = ProcessRegistry()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [auto_pipeline] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(str(RESULTS_DIR / f"auto_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("auto_pipeline")


def check_log_complete(log_path: str) -> dict:
    """Check if a pipeline log shows completion."""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
    except Exception:
        return {'complete': False, 'error': 'cannot read log'}

    if 'PIPELINE COMPLETE' in content:
        # Extract key results
        results = {'complete': True, 'lines': content.count('\n')}

        # Parse comparison table
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'Best method:' in line:
                results['best_method'] = line.split('Best method:')[1].strip()
            if 'MULTI-ALPHA EVIDENCE:' in line:
                results['multi_alpha'] = line.split('MULTI-ALPHA EVIDENCE:')[1].strip()
            if 'Pipeline Complete' in line or 'PIPELINE COMPLETE' in line:
                # Try to extract total time
                if 'Total time:' in line:
                    time_str = line.split('Total time:')[1].strip()
                    results['total_time'] = time_str

        return results

    if 'PIPELINE FAILED' in content:
        return {'complete': True, 'failed': True, 'lines': content.count('\n')}

    return {'complete': False, 'lines': content.count('\n')}


def check_process_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}'],
            capture_output=True, text=True, timeout=10
        )
        return str(pid) in result.stdout
    except Exception:
        return False


def launch_phase_1b():
    """Launch Phase 1B: MFE target scan."""
    script = str(LVL3_ROOT / "alpha_discovery" / "run_multi_alpha.py")
    cmd = [
        PYTHON, script,
        '--horizon', 'ret_5s',
        '--target-type', 'mfe_net',
        '--min-train-days', '5',
    ]
    if os.path.isdir(FEATURE_CACHE):
        cmd.extend(['--feature-cache', FEATURE_CACHE])
        logger.info("Using pre-computed Rust feature cache (saves ~3h)")

    logger.info(f"Launching Phase 1B: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(LVL3_ROOT),
    )
    registry.register(pid=proc.pid, name="phase_1b_mfe",
                      launched_by="auto_pipeline", tags=["pipeline", "phase_1b"])
    logger.info(f"Phase 1B launched as PID {proc.pid} (registered)")
    return proc


def launch_multi_horizon(horizon: str, target_type: str):
    """Launch a multi-horizon scan."""
    script = str(LVL3_ROOT / "alpha_discovery" / "run_multi_alpha.py")
    cmd = [
        PYTHON, script,
        '--horizon', horizon,
        '--target-type', target_type,
        '--min-train-days', '5',
    ]
    if os.path.isdir(FEATURE_CACHE):
        cmd.extend(['--feature-cache', FEATURE_CACHE])

    logger.info(f"Launching {horizon} + {target_type}: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(LVL3_ROOT),
    )
    registry.register(pid=proc.pid, name=f"multi_horizon_{horizon}_{target_type}",
                      launched_by="auto_pipeline", tags=["pipeline", "multi_horizon"])
    logger.info(f"Scan launched as PID {proc.pid} (registered)")
    return proc


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Auto-Pipeline Monitor')
    parser.add_argument('--watch-log', type=str, required=True,
                        help='Log file to watch for Phase 1A completion')
    parser.add_argument('--pid', type=int, default=None,
                        help='PID of Phase 1A process (optional)')
    parser.add_argument('--chain', type=str, default='1b',
                        choices=['1b', '1b+multi', 'multi'],
                        help='What to launch after Phase 1A completes')
    args = parser.parse_args()

    logger.info("Auto-Pipeline Monitor Starting")
    logger.info(f"  Watching: {args.watch_log}")
    logger.info(f"  Chain: {args.chain}")
    if args.pid:
        logger.info(f"  PID: {args.pid}")

    # Watch loop
    check_interval = 30  # seconds
    last_lines = 0

    while True:
        status = check_log_complete(args.watch_log)

        if status['complete']:
            if status.get('failed'):
                logger.error("Phase 1A FAILED!")
                logger.info("Not launching chained phases.")
                break

            logger.info("=" * 70)
            logger.info("Phase 1A COMPLETE!")
            if 'best_method' in status:
                logger.info(f"  Best: {status['best_method']}")
            if 'total_time' in status:
                logger.info(f"  Time: {status['total_time']}")
            if 'multi_alpha' in status:
                logger.info(f"  {status['multi_alpha']}")
            logger.info("=" * 70)

            # Save completion marker
            marker = {
                'phase': '1a',
                'completed_at': datetime.now().isoformat(),
                'status': status,
            }
            marker_path = RESULTS_DIR / 'phase_1a_complete.json'
            with open(str(marker_path), 'w') as f:
                json.dump(marker, f, indent=2)

            # Launch chained phases
            if args.chain in ('1b', '1b+multi'):
                logger.info("\nChaining Phase 1B...")
                time.sleep(5)  # Let resources free up
                proc = launch_phase_1b()

                if args.chain == '1b+multi':
                    # Wait for 1B to complete, then launch multi-horizon
                    logger.info("Waiting for Phase 1B to complete...")
                    proc.wait()
                    logger.info(f"Phase 1B exit code: {proc.returncode}")

                    if proc.returncode == 0:
                        # Launch multi-horizon scans sequentially
                        horizons_to_run = [
                            ('ret_3s', 'return'),
                            ('ret_3s', 'mfe_net'),
                            ('ret_10s', 'return'),
                            ('ret_10s', 'mfe_net'),
                            ('ret_30s', 'return'),
                            ('ret_30s', 'mfe_net'),
                            ('ret_1m', 'return'),
                            ('ret_1m', 'mfe_net'),
                        ]
                        for hz, tt in horizons_to_run:
                            logger.info(f"\nLaunching {hz} + {tt}...")
                            p = launch_multi_horizon(hz, tt)
                            p.wait()
                            logger.info(f"  {hz} + {tt} exit code: {p.returncode}")

            elif args.chain == 'multi':
                horizons_to_run = [
                    ('ret_3s', 'return'),
                    ('ret_10s', 'return'),
                    ('ret_30s', 'return'),
                    ('ret_1m', 'return'),
                ]
                for hz, tt in horizons_to_run:
                    logger.info(f"\nLaunching {hz} + {tt}...")
                    p = launch_multi_horizon(hz, tt)
                    p.wait()
                    logger.info(f"  {hz} + {tt} exit code: {p.returncode}")

            break

        # Progress update
        current_lines = status.get('lines', 0)
        if current_lines > last_lines:
            logger.info(f"Phase 1A still running... ({current_lines} log lines)")
            last_lines = current_lines

        # Check if process died
        if args.pid and not check_process_alive(args.pid):
            logger.warning(f"PID {args.pid} is no longer running!")
            logger.info("Checking log for final status...")
            final = check_log_complete(args.watch_log)
            if not final['complete']:
                logger.error("Process died without completing!")
            break

        time.sleep(check_interval)

    logger.info("Auto-Pipeline Monitor Exiting")


if __name__ == '__main__':
    main()
