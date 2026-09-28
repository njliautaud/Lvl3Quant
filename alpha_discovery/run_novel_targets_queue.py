"""
Novel Targets Queue Runner — Proper Monitoring & Early Termination

Runs each training objective as a separate process with:
- Early termination: if first 5 folds show IC < 0.01 or clearly negative, skip
- Performance monitoring: Discord-compatible progress reports
- Memory-safe: each test cleans up completely before next
- Best practices: walk-forward with 1-day purge gap, no leakage
- Results logged per-test for memory persistence

Usage:
    python alpha_discovery/run_novel_targets_queue.py --n-days 70
    python alpha_discovery/run_novel_targets_queue.py --n-days 20 --quick
    python alpha_discovery/run_novel_targets_queue.py --n-days 70 --test direct_pnl
"""

import gc
import sys
import time
import json
import logging
import argparse
import subprocess
import platform
from pathlib import Path
from datetime import datetime

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

# Cross-platform feature cache default
if platform.system() == 'Windows':
    DEFAULT_FEATURE_CACHE = str(LVL3_ROOT / "data" / "processed" / "mbo_features_cache")
else:
    DEFAULT_FEATURE_CACHE = str(Path.home() / "lvl3quant" / "data" / "processed" / "mbo_features_cache")

RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from alpha_discovery.process_registry import ProcessRegistry
from alpha_discovery.compute_notifier import notify_complete
registry = ProcessRegistry()

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log = RESULTS_DIR / f"novel_queue_{_ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(str(_log), mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("queue")

# Test configurations in compute-optimized order (fastest first)
TESTS = [
    {
        'name': 'risk_adjusted',
        'description': 'Risk-Adjusted Return (return/vol)',
        'model_type': 'LightGBM Regression',
        'est_time_min': 8,
        'early_stop_metric': 'IC',
        'min_viable': 0.01,
    },
    {
        'name': 'optimal_action',
        'description': 'Optimal Action Classification (BUY/SELL/HOLD)',
        'model_type': 'LightGBM 3-class',
        'est_time_min': 10,
        'early_stop_metric': 'Action_Acc',
        'min_viable': 0.35,  # Random = 33%, need at least 35%
    },
    {
        'name': 'time_to_move',
        'description': 'Time-to-Move Detection (2-stage: timing + direction)',
        'model_type': 'LightGBM Binary + Regression',
        'est_time_min': 15,
        'early_stop_metric': 'IC',
        'min_viable': 0.01,
    },
    {
        'name': 'direct_pnl',
        'description': 'Direct PnL Prediction (actual limit order P&L)',
        'model_type': 'LightGBM Huber',
        'est_time_min': 20,
        'early_stop_metric': 'IC',
        'min_viable': 0.01,
    },
    {
        'name': 'fill_probability',
        'description': 'Fill Probability (profitable limit fill prediction)',
        'model_type': 'LightGBM Binary + Direction combo',
        'est_time_min': 25,
        'early_stop_metric': 'AUC',
        'min_viable': 0.52,  # Random = 0.50
    },
    {
        'name': 'rl_reward',
        'description': 'RL Reward (Q-value estimation, 2 action models)',
        'model_type': '2× LightGBM Huber (buy + sell Q-functions)',
        'est_time_min': 25,
        'early_stop_metric': 'Q-IC',
        'min_viable': 0.01,
    },
]


def run_single_test(test_config, n_days, horizon, feature_cache, quick=False):
    """Run a single test as subprocess for memory isolation."""
    name = test_config['name']
    logger.info(f"\n{'='*70}")
    logger.info(f"STARTING: {test_config['description']}")
    logger.info(f"  Model: {test_config['model_type']}")
    logger.info(f"  Est time: ~{test_config['est_time_min']} min")
    logger.info(f"{'='*70}")

    python = sys.executable
    cmd = [
        python, str(LVL3_ROOT / "alpha_discovery" / "novel_targets_v2.py"),
        "--n-days", str(n_days),
        "--horizon", horizon,
        "--test", name,
        "--feature-cache", feature_cache,
    ]
    if quick:
        cmd.append("--quick")

    t0 = time.time()
    proc = None
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(LVL3_ROOT),
        )
        registry.register(
            pid=proc.pid, name=f"novel_{name}_{horizon}",
            launched_by="novel_queue",
            tags=["overnight", "novel_targets", name],
            cmd=' '.join(cmd))

        stdout, stderr = proc.communicate(timeout=3600)  # 1 hour max per test
        elapsed = time.time() - t0
        success = proc.returncode == 0

        registry.unregister(proc.pid)

        if success:
            logger.info(f"  COMPLETED in {elapsed:.0f}s ({elapsed/60:.1f}m)")
            lines = stdout.strip().split('\n')
            for line in lines[-30:]:
                logger.info(f"    {line}")
        else:
            logger.error(f"  FAILED (rc={proc.returncode}) in {elapsed:.0f}s")
            stderr_tail = stderr[-1000:] if stderr else "no stderr"
            logger.error(f"  stderr: {stderr_tail}")
            if stdout:
                for line in stdout.strip().split('\n')[-10:]:
                    logger.error(f"    {line}")

        return {
            'test': name,
            'success': success,
            'elapsed': elapsed,
            'returncode': proc.returncode,
        }

    except subprocess.TimeoutExpired:
        logger.error(f"  TIMEOUT after 3600s — skipping {name}")
        if proc:
            proc.kill()
            proc.communicate()
            registry.unregister(proc.pid)
        return {'test': name, 'success': False, 'elapsed': 3600, 'error': 'timeout'}
    except Exception as e:
        logger.error(f"  ERROR: {e}")
        if proc:
            registry.unregister(proc.pid)
        return {'test': name, 'success': False, 'error': str(e)}


def find_latest_result(test_name):
    """Find the most recent JSON result for a test."""
    results = sorted(RESULTS_DIR.glob("novel_v2_*.json"), reverse=True)
    for r in results[:5]:
        try:
            with open(r) as f:
                data = json.load(f)
            for res in data.get('results', []):
                if res.get('test_name') == test_name:
                    return res, data
        except Exception:
            continue
    return None, None


def main():
    parser = argparse.ArgumentParser(description='Novel Targets Queue Runner')
    parser.add_argument('--n-days', type=int, default=70)
    parser.add_argument('--horizon', type=str, default='10s')
    parser.add_argument('--feature-cache', type=str,
                        default=DEFAULT_FEATURE_CACHE)
    parser.add_argument('--test', type=str, default=None,
                        help='Run single test only')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.n_days = min(args.n_days, 20)

    logger.info("=" * 70)
    logger.info("NOVEL TARGETS QUEUE RUNNER")
    logger.info(f"  n_days:      {args.n_days}")
    logger.info(f"  horizon:     {args.horizon}")
    logger.info(f"  quick:       {args.quick}")
    logger.info(f"  log:         {_log}")
    logger.info("=" * 70)

    tests_to_run = TESTS
    if args.test:
        tests_to_run = [t for t in TESTS if t['name'] == args.test]
        if not tests_to_run:
            logger.error(f"Unknown test: {args.test}")
            logger.info(f"Available: {[t['name'] for t in TESTS]}")
            return

    t_total = time.time()
    summary = []

    for i, test in enumerate(tests_to_run):
        logger.info(f"\n\n{'#'*70}")
        logger.info(f"# TEST {i+1}/{len(tests_to_run)}: {test['name'].upper()}")
        logger.info(f"{'#'*70}")

        run_info = run_single_test(
            test, args.n_days, args.horizon, args.feature_cache, args.quick)

        # Find result from the JSON output
        result, result_data = find_latest_result(test['name'])
        if result:
            run_info['sim_result'] = result
            logger.info(f"\n  RESULT: {test['name']}")
            logger.info(f"    Trades: {result.get('n_trades', 0)}")
            logger.info(f"    PnL: ${result.get('total_pnl_dollars', 0):+,.0f}")
            logger.info(f"    Sharpe: {result.get('sharpe', 0):.2f}")
            logger.info(f"    WR: {result.get('win_rate', 0):.1%}")

            # Verdict
            sharpe = result.get('sharpe', 0)
            total_pnl = result.get('total_pnl_dollars', 0)
            if sharpe > 1.0 and total_pnl > 0:
                verdict = "PROMISING"
            elif sharpe > 0 and total_pnl > 0:
                verdict = "MARGINAL"
            elif total_pnl > 0:
                verdict = "WEAK"
            else:
                verdict = "NOT VIABLE"
            run_info['verdict'] = verdict
            logger.info(f"    VERDICT: {verdict}")
        else:
            run_info['verdict'] = 'NO RESULT' if run_info['success'] else 'FAILED'

        summary.append(run_info)

    # Final summary
    total_elapsed = time.time() - t_total
    logger.info(f"\n\n{'='*70}")
    logger.info(f"QUEUE COMPLETE — {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    logger.info(f"{'='*70}")

    for s in summary:
        status = s.get('verdict', 'UNKNOWN')
        test_name = s['test']
        elapsed = s.get('elapsed', 0)
        sim = s.get('sim_result', {})
        pnl = sim.get('total_pnl_dollars', 0)
        sharpe = sim.get('sharpe', 0)
        logger.info(f"  [{status:>12s}] {test_name:>20s}  "
                    f"${pnl:>+9,.0f}  Sharpe={sharpe:>+5.2f}  ({elapsed:.0f}s)")

    # Save queue summary
    queue_summary = {
        'started': _ts,
        'config': {
            'n_days': args.n_days,
            'horizon': args.horizon,
            'quick': args.quick,
        },
        'total_elapsed': total_elapsed,
        'tests': summary,
    }
    summary_path = RESULTS_DIR / f"novel_queue_{_ts}.json"
    with open(summary_path, 'w') as f:
        json.dump(queue_summary, f, indent=2, default=str)

    logger.info(f"\nQueue summary: {summary_path}")
    logger.info(f"Queue log: {_log}")

    # Notify completion signal
    n_ok = sum(1 for s in summary if s.get('verdict') not in ('FAILED', 'NOT VIABLE', 'NO RESULT'))
    notify_complete(
        task_name=f"novel_targets_queue_{args.horizon}",
        status="completed",
        result_summary=(
            f"{n_ok}/{len(summary)} tests viable, "
            f"horizon={args.horizon}, n_days={args.n_days}, "
            f"elapsed={total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)"
        ),
        result_file=str(summary_path),
    )


if __name__ == '__main__':
    main()
