#!/usr/bin/env python3
"""
Adaptive TP/SL Time-of-Day Sweep
=================================
Tests the hypothesis: TP/SL should vary with time-of-day and volatility.

Approach:
  1. Split each trading day into time windows (pre-open, morning, midday, afternoon, close)
  2. For each window, sweep TP/SL combinations through fill_sim_cli
  3. Find the optimal TP/SL per window
  4. Compare: static (best single TP/SL) vs adaptive (per-window optimal)

This directly tests the user's hypothesis that "2am moves are slower than 2pm moves"
so static TP/SL is suboptimal.

Usage:
    python adaptive_tpsl_tod_sweep.py [--workers 8] [--threshold 2.3]
"""

import sys
import json
import time
import argparse
import subprocess
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).resolve().parent.parent
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
PRED_DIR = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results' / 'adaptive_tpsl_tod'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(str(RESULTS_DIR / f'adaptive_tpsl_{_ts}.log'), mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Time Windows (ET) ──────────────────────────────────────────────────────
# Designed to capture different volatility regimes throughout the trading day
TIME_WINDOWS = {
    'overnight':    ('18:00', '09:30'),   # Globex overnight session (prior evening)
    'open_30min':   ('09:30', '10:00'),   # First 30min — high vol, wide spreads
    'morning':      ('10:00', '11:30'),   # Morning — good liquidity, trending
    'midday':       ('11:30', '13:30'),   # Lunch — low vol, choppy
    'afternoon':    ('13:30', '15:30'),   # Afternoon — vol picks up
    'close':        ('15:30', '16:00'),   # MOC/close — high vol spike
}

# TP/SL combinations to sweep per window
TP_TICKS = [5, 8, 10, 13, 16, 20, 25]
SL_TICKS = [8, 12, 15, 20, 25, 30]
HOLD_MS = [30000, 60000, 120000]  # 30s, 60s, 120s

# ── Helper Functions ───────────────────────────────────────────────────────

PRED_CACHE = RESULTS_DIR / 'pred_cache'
PRED_CACHE.mkdir(parents=True, exist_ok=True)


def discover_fold_dates() -> List[Tuple[str, Path, Path]]:
    """Find all folds with matching MBO data. Returns [(date, pred_1d_path, mbo_path)].

    Preprocesses multi-horizon (N,3) predictions into 1D (N,) NPZ files
    using the 10s horizon (index 2) since fill_sim_cli expects 1D predictions.
    """
    results = []
    for pred_file in sorted(PRED_DIR.glob('fold_*_oot_predictions.npz')):
        if 'concat' in pred_file.name:
            continue
        try:
            data = np.load(str(pred_file), allow_pickle=True)
            oot_path = str(data['oot_files'][0])
            basename = oot_path.replace('\\', '/').split('/')[-1]
            date_str = basename.split('_')[0]
            # Find matching MBO file
            mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
            if not mbo_file.exists():
                log.warning(f"No MBO file for {date_str}, skipping")
                continue

            # Preprocess: extract 10s horizon (index 2) as 1D predictions
            cache_file = PRED_CACHE / f'pred_10s_{date_str}.npz'
            if not cache_file.exists():
                preds = data['predictions']
                if preds.ndim == 2:
                    # Multi-horizon: take 10s (index 2)
                    preds_1d = preds[:, 2].astype(np.float64)
                else:
                    preds_1d = preds.astype(np.float64)
                np.savez_compressed(str(cache_file), predictions=preds_1d)
                log.info(f"  Cached 1D preds for {date_str}: {preds_1d.shape[0]} samples, "
                         f"mean={preds_1d.mean():.4f}, std={preds_1d.std():.4f}")

            results.append((date_str, cache_file, mbo_file))
        except Exception as e:
            log.warning(f"Failed to process {pred_file}: {e}")
    return results


def run_fill_sim(
    mbo_file: Path, pred_file: Path,
    tp: int, sl: int, hold_ms: int,
    threshold: float = 2.3,
    time_start: str = "", time_end: str = "",
    extra_args: list = None,
) -> Dict:
    """Run fill_sim_cli with given parameters. Returns parsed JSON result."""
    output_file = Path(f'/tmp/fillsim_{os.getpid()}_{time.time_ns()}.json')
    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(output_file),
        '--take-profit-ticks', str(tp),
        '--stop-loss-ticks', str(sl),
        '--hold-ms', str(hold_ms),
        '--signal-threshold', str(threshold),
        '--chase-entry',
        '--chase-max-ticks', '2',
        '--chase-max-reprices', '5',
        '--vol-exit-ticks', '5',
        '--vol-exit-bars', '5',
        '--latency-ms', '1',
        '--quiet',
    ]
    if time_start and time_end:
        cmd += ['--time-window-start', time_start, '--time-window-end', time_end]
    if extra_args:
        cmd += extra_args

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return {'error': result.stderr[:200]}
        if output_file.exists():
            with open(output_file) as f:
                data = json.load(f)
            output_file.unlink(missing_ok=True)
            return data
        return {'error': 'no output file'}
    except subprocess.TimeoutExpired:
        return {'error': 'timeout'}
    except Exception as e:
        return {'error': str(e)}


def extract_metrics(result: Dict) -> Dict:
    """Extract key metrics from fill_sim result."""
    if 'error' in result:
        return {'error': result['error'], 'n_trades': 0, 'pnl': 0}

    trades = result.get('trades', [])
    n_trades = len(trades)
    if n_trades == 0:
        return {'n_trades': 0, 'pnl': 0, 'win_rate': 0, 'sharpe': 0, 'sortino': 0}

    pnls = [t.get('pnl_dollars', t.get('pnl_ticks', 0) * TICK_VALUE) for t in trades]
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)

    pnl_arr = np.array(pnls)
    mean_pnl = pnl_arr.mean()
    std_pnl = pnl_arr.std() if len(pnl_arr) > 1 else 1

    sharpe = mean_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0
    downside = pnl_arr[pnl_arr < 0].std() if (pnl_arr < 0).sum() > 1 else 1
    sortino = mean_pnl / downside * np.sqrt(252) if downside > 0 else 0

    return {
        'n_trades': n_trades,
        'pnl': round(total_pnl, 2),
        'mean_pnl': round(mean_pnl, 2),
        'win_rate': round(wins / n_trades, 3),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pnl_per_trade': round(total_pnl / n_trades, 2),
    }


# ── Main Sweep ─────────────────────────────────────────────────────────────

def sweep_window(
    window_name: str, time_start: str, time_end: str,
    fold_dates: List[Tuple[str, Path, Path]],
    threshold: float, workers: int,
) -> Dict:
    """Sweep TP/SL/hold combos for a single time window across all dates."""
    log.info(f"  Sweeping {window_name} ({time_start}-{time_end})")

    configs = []
    for tp in TP_TICKS:
        for sl in SL_TICKS:
            for hold in HOLD_MS:
                configs.append((tp, sl, hold))

    # Run all configs across all dates
    results = {}
    tasks = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for tp, sl, hold in configs:
            config_key = f'tp{tp}_sl{sl}_h{hold//1000}s'
            for date_str, pred_file, mbo_file in fold_dates:
                future = executor.submit(
                    run_fill_sim,
                    mbo_file, pred_file, tp, sl, hold, threshold,
                    time_start, time_end,
                )
                tasks.append((future, config_key, date_str))

        # Collect
        config_date_metrics = defaultdict(dict)
        done = 0
        total = len(tasks)
        for future, config_key, date_str in tasks:
            try:
                raw = future.result(timeout=180)
                metrics = extract_metrics(raw)
                config_date_metrics[config_key][date_str] = metrics
            except Exception as e:
                config_date_metrics[config_key][date_str] = {'error': str(e), 'n_trades': 0, 'pnl': 0}
            done += 1
            if done % 100 == 0:
                log.info(f"    {window_name}: {done}/{total} sims complete")

    # Aggregate per config
    config_agg = {}
    for config_key, date_metrics in config_date_metrics.items():
        total_trades = sum(m.get('n_trades', 0) for m in date_metrics.values())
        total_pnl = sum(m.get('pnl', 0) for m in date_metrics.values())
        daily_pnls = [m.get('pnl', 0) for m in date_metrics.values()]

        if total_trades == 0:
            continue

        pnl_arr = np.array(daily_pnls)
        mean_daily = pnl_arr.mean()
        std_daily = pnl_arr.std() if len(pnl_arr) > 1 else 1
        downside = pnl_arr[pnl_arr < 0].std() if (pnl_arr < 0).sum() > 1 else 1

        config_agg[config_key] = {
            'total_trades': total_trades,
            'total_pnl': round(total_pnl, 2),
            'mean_daily_pnl': round(mean_daily, 2),
            'daily_sharpe': round(mean_daily / std_daily * np.sqrt(252), 3) if std_daily > 0 else 0,
            'daily_sortino': round(mean_daily / downside * np.sqrt(252), 3) if downside > 0 else 0,
            'avg_trades_per_day': round(total_trades / len(date_metrics), 1),
            'win_rate': round(
                np.mean([m.get('win_rate', 0) for m in date_metrics.values() if m.get('n_trades', 0) > 0]), 3
            ),
        }

    # Sort by daily_sortino
    sorted_configs = sorted(config_agg.items(), key=lambda x: x[1].get('daily_sortino', -999), reverse=True)

    return {
        'window': window_name,
        'time_range': f'{time_start}-{time_end}',
        'n_configs_tested': len(configs),
        'n_configs_with_trades': len(config_agg),
        'top_5': {k: v for k, v in sorted_configs[:5]},
        'bottom_3': {k: v for k, v in sorted_configs[-3:]},
        'best_config': sorted_configs[0] if sorted_configs else None,
        'all_configs': dict(sorted_configs),
    }


def run_static_baseline(
    fold_dates: List[Tuple[str, Path, Path]],
    threshold: float, workers: int,
) -> Dict:
    """Run the static baseline (TP=8, SL=15, hold=60s) across full day for comparison."""
    log.info("Running static baseline (TP=8, SL=15, hold=60s, full day)...")

    daily_results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for date_str, pred_file, mbo_file in fold_dates:
            f = executor.submit(run_fill_sim, mbo_file, pred_file, 8, 15, 60000, threshold)
            futures[f] = date_str

        for future in as_completed(futures):
            date_str = futures[future]
            try:
                raw = future.result(timeout=180)
                daily_results[date_str] = extract_metrics(raw)
            except Exception as e:
                daily_results[date_str] = {'error': str(e), 'n_trades': 0, 'pnl': 0}

    total_pnl = sum(m.get('pnl', 0) for m in daily_results.values())
    total_trades = sum(m.get('n_trades', 0) for m in daily_results.values())
    daily_pnls = np.array([m.get('pnl', 0) for m in daily_results.values()])
    mean_d = daily_pnls.mean()
    std_d = daily_pnls.std() if len(daily_pnls) > 1 else 1
    down_d = daily_pnls[daily_pnls < 0].std() if (daily_pnls < 0).sum() > 1 else 1

    return {
        'config': 'tp8_sl15_h60s',
        'total_pnl': round(total_pnl, 2),
        'total_trades': total_trades,
        'daily_sharpe': round(mean_d / std_d * np.sqrt(252), 3) if std_d > 0 else 0,
        'daily_sortino': round(mean_d / down_d * np.sqrt(252), 3) if down_d > 0 else 0,
        'win_rate': round(np.mean([m.get('win_rate', 0) for m in daily_results.values() if m.get('n_trades', 0) > 0]), 3),
        'per_date': daily_results,
    }


def main():
    parser = argparse.ArgumentParser(description='Adaptive TP/SL Time-of-Day Sweep')
    parser.add_argument('--workers', type=int, default=10, help='Parallel fill_sim workers')
    parser.add_argument('--threshold', type=float, default=2.3, help='Signal z-score threshold')
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("ADAPTIVE TP/SL TIME-OF-DAY SWEEP")
    log.info(f"  Threshold: z >= {args.threshold}")
    log.info(f"  Workers: {args.workers}")
    log.info(f"  TP range: {TP_TICKS}")
    log.info(f"  SL range: {SL_TICKS}")
    log.info(f"  Hold range: {[h//1000 for h in HOLD_MS]}s")
    log.info(f"  Time windows: {list(TIME_WINDOWS.keys())}")
    log.info("=" * 60)

    # Discover folds
    fold_dates = discover_fold_dates()
    log.info(f"Found {len(fold_dates)} folds with MBO data")
    if not fold_dates:
        log.error("No folds found!")
        return

    for date_str, _, _ in fold_dates:
        log.info(f"  {date_str}")

    t0 = time.time()

    # 1. Static baseline
    baseline = run_static_baseline(fold_dates, args.threshold, args.workers)
    log.info(f"\nSTATIC BASELINE: PnL=${baseline['total_pnl']}, "
             f"Sharpe={baseline['daily_sharpe']}, Sortino={baseline['daily_sortino']}, "
             f"Trades={baseline['total_trades']}, WinRate={baseline['win_rate']}")

    # 2. Per-window sweep (skip overnight for now — the fill_sim time windows
    #    don't handle crossing midnight well)
    window_results = {}
    for window_name, (t_start, t_end) in TIME_WINDOWS.items():
        if window_name == 'overnight':
            log.info(f"Skipping {window_name} (crosses midnight)")
            continue
        window_results[window_name] = sweep_window(
            window_name, t_start, t_end, fold_dates, args.threshold, args.workers
        )
        best = window_results[window_name].get('best_config')
        if best:
            log.info(f"  BEST for {window_name}: {best[0]} → "
                     f"PnL=${best[1]['total_pnl']}, Sortino={best[1]['daily_sortino']}, "
                     f"WinRate={best[1]['win_rate']}")

    # 3. Construct adaptive composite
    # For each window, take the best config and sum PnL across windows
    adaptive_pnl = 0
    adaptive_trades = 0
    adaptive_detail = {}
    for window_name, wr in window_results.items():
        best = wr.get('best_config')
        if best:
            adaptive_pnl += best[1]['total_pnl']
            adaptive_trades += best[1]['total_trades']
            adaptive_detail[window_name] = {
                'config': best[0],
                'pnl': best[1]['total_pnl'],
                'sortino': best[1]['daily_sortino'],
                'trades': best[1]['total_trades'],
            }

    elapsed = time.time() - t0

    # 4. Summary
    log.info("\n" + "=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)
    log.info(f"\nSTATIC (TP=8, SL=15, 60s hold, full day):")
    log.info(f"  PnL: ${baseline['total_pnl']}")
    log.info(f"  Sharpe: {baseline['daily_sharpe']}")
    log.info(f"  Sortino: {baseline['daily_sortino']}")
    log.info(f"  Trades: {baseline['total_trades']}")

    log.info(f"\nADAPTIVE (best TP/SL per time window):")
    log.info(f"  Total PnL: ${adaptive_pnl:.2f}")
    log.info(f"  Total trades: {adaptive_trades}")
    for wn, wd in adaptive_detail.items():
        log.info(f"  {wn}: {wd['config']} → ${wd['pnl']}, sortino={wd['sortino']}, trades={wd['trades']}")

    improvement = adaptive_pnl - baseline['total_pnl']
    log.info(f"\nADAPTIVE vs STATIC improvement: ${improvement:.2f}")
    log.info(f"Elapsed: {elapsed/60:.1f} minutes")

    # Save full results
    output = {
        'timestamp': _ts,
        'params': {'threshold': args.threshold, 'tp_range': TP_TICKS, 'sl_range': SL_TICKS, 'hold_range': HOLD_MS},
        'baseline_static': baseline,
        'adaptive_composite': {
            'total_pnl': round(adaptive_pnl, 2),
            'total_trades': adaptive_trades,
            'per_window': adaptive_detail,
            'improvement_vs_static': round(improvement, 2),
        },
        'per_window_results': {k: {kk: vv for kk, vv in v.items() if kk != 'all_configs'} for k, v in window_results.items()},
        'per_window_all_configs': {k: v.get('all_configs', {}) for k, v in window_results.items()},
    }

    out_file = RESULTS_DIR / f'adaptive_tpsl_results_{_ts}.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nFull results saved to {out_file}")


if __name__ == '__main__':
    main()
