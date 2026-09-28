"""
Overnight V2 Orchestrator — Waits for current tasks to finish, then runs novel targets.

Uses ProcessRegistry to track launched processes and identify what's running.
Only waits for REGISTERED processes with matching tags — unknown processes are
logged and skipped (no more 4-hour blind waits).

Monitors:
1. Multi-timeframe stacking (PID check via registry)
2. EventTransformer DL training (PID check via registry)
Then launches:
3. Novel targets queue (all 6 tests, 70 days)
4. Novel targets at 30s and 1m horizons

Saves results and sends progress to log for monitoring.
"""

import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

from alpha_discovery.process_registry import ProcessRegistry
from alpha_discovery.compute_notifier import notify_complete

RESULTS_DIR = LVL3_ROOT / "production" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log = RESULTS_DIR / f"overnight_v2_{_ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(str(_log), mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("overnight_v2")


registry = ProcessRegistry()


def is_process_running(pid):
    """Check if a process is still running."""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}'],
            capture_output=True, text=True, timeout=10)
        return str(pid) in result.stdout
    except Exception:
        return False


def run_phase(name, cmd, timeout_sec=7200, tags=None):
    """Run a phase with logging and process registry tracking."""
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE: {name}")
    logger.info(f"{'='*60}")
    logger.info(f"  Command: {' '.join(cmd)}")

    t0 = time.time()
    proc = None
    try:
        # Use Popen so we can register the PID immediately
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(LVL3_ROOT))

        # Register in process registry
        registry.register(
            pid=proc.pid, name=name,
            launched_by="overnight_v2",
            tags=tags or ["overnight"],
            cmd=' '.join(cmd))

        # Wait for completion with timeout
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        elapsed = time.time() - t0

        # Unregister now that it's done
        registry.unregister(proc.pid)

        if proc.returncode == 0:
            logger.info(f"  COMPLETED in {elapsed:.0f}s ({elapsed/60:.1f}m)")
            lines = stdout.strip().split('\n')
            for line in lines[-30:]:
                logger.info(f"    {line}")
        else:
            logger.error(f"  FAILED (rc={proc.returncode}) in {elapsed:.0f}s")
            if stderr:
                logger.error(f"  stderr: {stderr[-500:]}")

        return {'phase': name, 'success': proc.returncode == 0, 'elapsed': elapsed}
    except subprocess.TimeoutExpired:
        logger.error(f"  TIMEOUT after {timeout_sec}s")
        if proc:
            proc.kill()
            proc.communicate()
            registry.unregister(proc.pid)
        return {'phase': name, 'success': False, 'elapsed': timeout_sec, 'error': 'timeout'}
    except Exception as e:
        logger.error(f"  ERROR: {e}")
        if proc:
            registry.unregister(proc.pid)
        return {'phase': name, 'success': False, 'error': str(e)}


def main():
    logger.info("=" * 70)
    logger.info("OVERNIGHT V2 ORCHESTRATOR")
    logger.info(f"  Started: {datetime.now()}")
    logger.info(f"  Log: {_log}")
    logger.info("=" * 70)

    python = sys.executable
    feature_cache = str(LVL3_ROOT / "data" / "processed" / "mbo_features_cache")
    results = []
    t_total = time.time()

    # Phase 0: Smart wait — only wait for REGISTERED processes, skip unknowns
    logger.info("\nPhase 0: Checking for running compute tasks (registry-based)...")
    wait_result = registry.smart_wait(
        wait_tags=["overnight", "multi_timeframe", "dl_training"],
        check_interval=120,
        max_wait=7200,  # 2hr max (not 4hr) — tagged processes should be known duration
        warn_unknown=True,
    )
    if wait_result['unknown_skipped']:
        logger.warning(f"  Skipped {len(wait_result['unknown_skipped'])} unknown processes")
    if not wait_result['tagged_finished']:
        logger.warning("  Timeout waiting for tagged processes — proceeding anyway")
    logger.info(f"  Wait phase took {wait_result['elapsed']:.0f}s")

    # Phase 1: Novel Targets at 10s (most important horizon)
    results.append(run_phase(
        "Novel Targets V2 — 10s (all 6 objectives, 70 days)",
        [python, "alpha_discovery/run_novel_targets_queue.py",
         "--n-days", "70", "--horizon", "10s",
         "--feature-cache", feature_cache],
        timeout_sec=10800,  # 3 hours
    ))

    # Phase 2: Novel Targets at 30s
    results.append(run_phase(
        "Novel Targets V2 — 30s (all 6 objectives, 70 days)",
        [python, "alpha_discovery/run_novel_targets_queue.py",
         "--n-days", "70", "--horizon", "30s",
         "--feature-cache", feature_cache],
        timeout_sec=10800,
    ))

    # Phase 3: Novel Targets at 1m (longer horizon test)
    results.append(run_phase(
        "Novel Targets V2 — 1m (all 6 objectives, 70 days)",
        [python, "alpha_discovery/run_novel_targets_queue.py",
         "--n-days", "70", "--horizon", "1m",
         "--feature-cache", feature_cache],
        timeout_sec=10800,
    ))

    # Phase 4: Production backtest with best target
    results.append(run_phase(
        "Production Backtest (70 days, full sim)",
        [python, "production/trading_system.py",
         "--mode", "backtest", "--n-days", "70",
         "--feature-cache", feature_cache],
        timeout_sec=7200,
    ))

    # Summary
    total_elapsed = time.time() - t_total
    logger.info(f"\n\n{'='*70}")
    logger.info(f"OVERNIGHT V2 COMPLETE — {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    logger.info(f"{'='*70}")
    for r in results:
        status = "OK" if r.get('success') else "FAIL"
        logger.info(f"  [{status}] {r['phase']} ({r.get('elapsed', 0):.0f}s)")

    summary = {
        'started': _ts,
        'total_elapsed': total_elapsed,
        'phases': results,
    }
    summary_path = RESULTS_DIR / f"overnight_v2_summary_{_ts}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Notify completion signal
    n_ok = sum(1 for r in results if r.get('success'))
    notify_complete(
        task_name="overnight_v2",
        status="completed",
        result_summary=f"{n_ok}/{len(results)} phases OK, elapsed={total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)",
        result_file=str(summary_path),
    )


if __name__ == '__main__':
    main()
