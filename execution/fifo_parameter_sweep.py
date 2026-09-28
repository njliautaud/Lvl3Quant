#!/usr/bin/env python3
"""
FIFO Market Replay Parameter Sweep
====================================
Systematically tests different execution configurations against the FIFO
market replay engine to find profitable parameter combinations.

The existing FIFO replay with gate=0.70, TP=6t, SL=3t produces -$9,036 over
39 days at 36% WR and 57.7% SL rate. The signals have directional alpha
(R:R=1.30) but the execution config is wrong. This sweep searches for the
right one.

Usage:
    # Full sweep on Neptune (16 cores):
    python fifo_parameter_sweep.py

    # Quick test with fewer configs:
    python fifo_parameter_sweep.py --max-configs 20

    # Use specific dates:
    python fifo_parameter_sweep.py --dates 20260316,20260317,20260318

    # Custom output path:
    python fifo_parameter_sweep.py --output results/sweep_results.csv

    # Dry run (show configs without running):
    python fifo_parameter_sweep.py --dry-run
"""

import os
import sys
import csv
import json
import time
import socket
import logging
import argparse
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from multiprocessing import Pool, cpu_count

# ── Auto-detect root based on hostname ──────────────────────────────────────
hostname = socket.gethostname().lower()
if 'neptune' in hostname or hostname == 'nick-desktop':
    LVL3_ROOT = Path('/home/nick/Lvl3Quant')
elif 'saturn' in hostname:
    LVL3_ROOT = Path('/home/saturn/Lvl3Quant')
elif 'jupiter' in hostname:
    LVL3_ROOT = Path('/home/jupiter/Lvl3Quant')
else:
    # Fallback: walk up from this file
    LVL3_ROOT = Path(__file__).resolve().parents[1]

# Add project root and deep_models to path so we can import the replay engine
sys.path.insert(0, str(LVL3_ROOT))
sys.path.insert(0, str(LVL3_ROOT / 'alpha_discovery' / 'deep_models'))

from fifo_market_replay import (
    FIFOReplayEngine,
    load_decay_predictions,
    generate_signals,
    compute_metrics,
    compute_daily_metrics,
    TICK_USD,
    COMMISSION_TICKS,
)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,  # Quiet during sweep; per-config logs are noisy
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger('fifo_sweep')
log.setLevel(logging.INFO)

# Suppress the replay engine's verbose logging during sweep
logging.getLogger('fifo_market_replay').setLevel(logging.WARNING)

# ── Default paths ───────────────────────────────────────────────────────────
DEFAULT_PRED_DIR = str(LVL3_ROOT / 'output' / 'decay_v4_comprehensive' / 'CNN-Mamba_v2')
DEFAULT_MBO_DIR  = str(LVL3_ROOT / 'data' / 'processed' / 'mbo_events')
DEFAULT_OUTPUT   = str(LVL3_ROOT / 'execution' / 'results' / 'fifo_parameter_sweep.csv')


# ── Parameter grid ──────────────────────────────────────────────────────────
DEFAULT_GRID = {
    'gate_threshold': [0.50, 0.60, 0.70, 0.80, 0.90],
    'tp_ticks':       [3, 4, 6, 8, 12, 20],
    'sl_ticks':       [2, 3, 4, 6, 8],
    'max_hold_ms':    [10000, 30000, 60000, 120000],
    'cancel_ms':      [5000, 15000, 30000],
    'horizon':        [0],  # 1s only for now
}


@dataclass
class SweepConfig:
    """One parameter configuration to test."""
    config_id:      int
    gate_threshold: float
    tp_ticks:       float
    sl_ticks:       float
    max_hold_ms:    float
    cancel_ms:      float
    horizon:        int


@dataclass
class SweepResult:
    """Results from one configuration across all dates."""
    config_id:      int
    gate_threshold: float
    tp_ticks:       float
    sl_ticks:       float
    max_hold_ms:    float
    cancel_ms:      float
    horizon:        int
    # Aggregate metrics
    n_dates:        int
    n_signals:      int
    n_trades:       int
    fill_rate:      float
    total_pnl_ticks:   float
    total_pnl_dollars: float
    mean_pnl_per_trade: float
    win_rate:       float
    profit_factor:  float
    sharpe:         float
    sortino:        float
    avg_rr:         float
    tp_rate:        float
    sl_rate:        float
    max_hold_rate:  float
    eod_rate:       float
    avg_queue_wait_ms:  float
    avg_hold_time_ms:   float
    avg_slippage_ticks: float
    # Daily portfolio metrics
    daily_sharpe:   float
    daily_sortino:  float
    max_drawdown:   float
    win_days:       int
    loss_days:      int
    best_day:       float
    worst_day:      float
    pnl_per_day:    float


def build_grid(grid_params: dict) -> List[SweepConfig]:
    """Build full cartesian product of parameter grid."""
    keys = sorted(grid_params.keys())
    values = [grid_params[k] for k in keys]
    configs = []
    for i, combo in enumerate(itertools.product(*values)):
        params = dict(zip(keys, combo))
        configs.append(SweepConfig(
            config_id=i,
            gate_threshold=params['gate_threshold'],
            tp_ticks=params['tp_ticks'],
            sl_ticks=params['sl_ticks'],
            max_hold_ms=params['max_hold_ms'],
            cancel_ms=params['cancel_ms'],
            horizon=params['horizon'],
        ))
    return configs


def select_representative_dates(all_dates: List[str], n_dates: int = 10) -> List[str]:
    """
    Select a representative subset of dates for speed.
    Picks evenly spaced dates to capture different market regimes.
    """
    if len(all_dates) <= n_dates:
        return all_dates
    indices = np.linspace(0, len(all_dates) - 1, n_dates, dtype=int)
    return [all_dates[i] for i in indices]


# ── Global cache for loaded data (shared across worker calls via fork) ──────
_GLOBAL_PRED_DATA = {}   # date_str -> prediction data dict
_GLOBAL_ENGINES = {}     # date_str -> FIFOReplayEngine (pre-loaded MBO data)


def preload_data(pred_dir: str, mbo_dir: str, dates: List[str]):
    """
    Pre-load all prediction data and MBO replay engines into globals.
    This avoids redundant I/O across configs when using fork-based multiprocessing.
    """
    global _GLOBAL_PRED_DATA, _GLOBAL_ENGINES

    log.info(f"Pre-loading predictions from {pred_dir}")
    all_data = load_decay_predictions(Path(pred_dir), Path(mbo_dir))

    # Filter to requested dates
    for d in all_data:
        if d['date'] in dates:
            _GLOBAL_PRED_DATA[d['date']] = d

    loaded_dates = sorted(_GLOBAL_PRED_DATA.keys())
    log.info(f"  Loaded predictions for {len(loaded_dates)} dates: "
             f"{loaded_dates[0]}..{loaded_dates[-1]}")

    # Pre-load MBO replay engines (this is the expensive part)
    log.info("Pre-loading MBO replay engines (this takes a while)...")
    for i, date_str in enumerate(loaded_dates):
        try:
            engine = FIFOReplayEngine(date=date_str)
            _GLOBAL_ENGINES[date_str] = engine
            log.info(f"  [{i+1}/{len(loaded_dates)}] Loaded {date_str}: "
                     f"{len(engine.records):,} MBO events")
        except FileNotFoundError as e:
            log.warning(f"  [{i+1}/{len(loaded_dates)}] Skipping {date_str}: {e}")
        except Exception as e:
            log.warning(f"  [{i+1}/{len(loaded_dates)}] Error loading {date_str}: {e}")

    log.info(f"  Pre-loaded {len(_GLOBAL_ENGINES)} replay engines")
    return loaded_dates


def run_single_config(config: SweepConfig) -> Optional[SweepResult]:
    """
    Run one parameter configuration across all pre-loaded dates.
    Designed to be called from multiprocessing pool.
    """
    try:
        all_trade_results = []
        daily_pnls = []
        total_signals = 0

        for date_str, engine in sorted(_GLOBAL_ENGINES.items()):
            pred_data = _GLOBAL_PRED_DATA.get(date_str)
            if pred_data is None:
                continue

            # Generate signals with this config's gate/horizon
            signals = generate_signals(
                pred_data,
                gate_threshold=config.gate_threshold,
                horizon=config.horizon,
                min_interval_ns=500_000_000,  # 500ms anti-churn
            )
            total_signals += len(signals)

            if not signals:
                daily_pnls.append(0.0)
                continue

            # Temporarily override engine's hold/cancel params for this config
            engine_cancel = engine.cancel_after_ns
            engine_hold = engine.max_hold_ns
            engine.cancel_after_ns = int(config.cancel_ms * 1e6)
            engine.max_hold_ns = int(config.max_hold_ms * 1e6)

            trades = engine.simulate(
                signals,
                tp_ticks=config.tp_ticks,
                sl_ticks=config.sl_ticks,
                order_type='limit',
            )

            # Restore defaults for other configs
            engine.cancel_after_ns = engine_cancel
            engine.max_hold_ns = engine_hold

            all_trade_results.extend(trades)
            day_pnl = sum(t.pnl_dollars for t in trades)
            daily_pnls.append(day_pnl)

        # Compute aggregate metrics
        if not all_trade_results:
            return SweepResult(
                config_id=config.config_id,
                gate_threshold=config.gate_threshold,
                tp_ticks=config.tp_ticks,
                sl_ticks=config.sl_ticks,
                max_hold_ms=config.max_hold_ms,
                cancel_ms=config.cancel_ms,
                horizon=config.horizon,
                n_dates=len(daily_pnls),
                n_signals=total_signals,
                n_trades=0, fill_rate=0, total_pnl_ticks=0,
                total_pnl_dollars=0, mean_pnl_per_trade=0,
                win_rate=0, profit_factor=0, sharpe=0, sortino=0,
                avg_rr=0, tp_rate=0, sl_rate=0, max_hold_rate=0,
                eod_rate=0, avg_queue_wait_ms=0, avg_hold_time_ms=0,
                avg_slippage_ticks=0, daily_sharpe=0, daily_sortino=0,
                max_drawdown=0, win_days=0, loss_days=0,
                best_day=0, worst_day=0, pnl_per_day=0,
            )

        metrics = compute_metrics(all_trade_results, total_signals)
        daily = compute_daily_metrics(daily_pnls)

        return SweepResult(
            config_id=config.config_id,
            gate_threshold=config.gate_threshold,
            tp_ticks=config.tp_ticks,
            sl_ticks=config.sl_ticks,
            max_hold_ms=config.max_hold_ms,
            cancel_ms=config.cancel_ms,
            horizon=config.horizon,
            n_dates=len(daily_pnls),
            n_signals=total_signals,
            n_trades=metrics['n_trades'],
            fill_rate=metrics['fill_rate'],
            total_pnl_ticks=metrics['total_pnl_ticks'],
            total_pnl_dollars=metrics['total_pnl_dollars'],
            mean_pnl_per_trade=metrics.get('mean_pnl_ticks', 0),
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            sharpe=metrics['sharpe'],
            sortino=metrics['sortino'],
            avg_rr=metrics['avg_rr'],
            tp_rate=metrics['tp_rate'],
            sl_rate=metrics['sl_rate'],
            max_hold_rate=metrics['max_hold_rate'],
            eod_rate=metrics['eod_rate'],
            avg_queue_wait_ms=metrics['avg_queue_wait_ms'],
            avg_hold_time_ms=metrics['avg_hold_time_ms'],
            avg_slippage_ticks=metrics['avg_slippage_ticks'],
            daily_sharpe=daily.get('daily_sharpe', 0),
            daily_sortino=daily.get('daily_sortino', 0),
            max_drawdown=daily.get('max_drawdown', 0),
            win_days=daily.get('win_days', 0),
            loss_days=daily.get('loss_days', 0),
            best_day=daily.get('best_day', 0),
            worst_day=daily.get('worst_day', 0),
            pnl_per_day=daily.get('mean_daily_pnl', 0),
        )

    except Exception as e:
        log.error(f"Config {config.config_id} failed: {e}")
        return None


def run_single_config_sequential(args: Tuple) -> Optional[SweepResult]:
    """
    Run one config WITHOUT relying on global pre-loaded data.
    Used when multiprocessing with spawn (not fork) or for sequential mode.
    Each call loads its own data. Slower but safer.
    """
    config, pred_dir, mbo_dir, dates = args
    try:
        all_trade_results = []
        daily_pnls = []
        total_signals = 0

        # Load predictions
        all_data = load_decay_predictions(Path(pred_dir), Path(mbo_dir))
        date_map = {d['date']: d for d in all_data if d['date'] in dates}

        for date_str in sorted(dates):
            pred_data = date_map.get(date_str)
            if pred_data is None:
                continue

            signals = generate_signals(
                pred_data,
                gate_threshold=config.gate_threshold,
                horizon=config.horizon,
                min_interval_ns=500_000_000,
            )
            total_signals += len(signals)

            if not signals:
                daily_pnls.append(0.0)
                continue

            try:
                engine = FIFOReplayEngine(
                    date=date_str,
                    cancel_after_ns=int(config.cancel_ms * 1e6),
                    max_hold_ns=int(config.max_hold_ms * 1e6),
                )
            except FileNotFoundError:
                continue

            trades = engine.simulate(
                signals,
                tp_ticks=config.tp_ticks,
                sl_ticks=config.sl_ticks,
                order_type='limit',
            )
            all_trade_results.extend(trades)
            daily_pnls.append(sum(t.pnl_dollars for t in trades))

        if not all_trade_results:
            return SweepResult(
                config_id=config.config_id,
                gate_threshold=config.gate_threshold,
                tp_ticks=config.tp_ticks,
                sl_ticks=config.sl_ticks,
                max_hold_ms=config.max_hold_ms,
                cancel_ms=config.cancel_ms,
                horizon=config.horizon,
                n_dates=len(daily_pnls), n_signals=total_signals,
                n_trades=0, fill_rate=0, total_pnl_ticks=0,
                total_pnl_dollars=0, mean_pnl_per_trade=0,
                win_rate=0, profit_factor=0, sharpe=0, sortino=0,
                avg_rr=0, tp_rate=0, sl_rate=0, max_hold_rate=0,
                eod_rate=0, avg_queue_wait_ms=0, avg_hold_time_ms=0,
                avg_slippage_ticks=0, daily_sharpe=0, daily_sortino=0,
                max_drawdown=0, win_days=0, loss_days=0,
                best_day=0, worst_day=0, pnl_per_day=0,
            )

        metrics = compute_metrics(all_trade_results, total_signals)
        daily = compute_daily_metrics(daily_pnls)

        return SweepResult(
            config_id=config.config_id,
            gate_threshold=config.gate_threshold,
            tp_ticks=config.tp_ticks,
            sl_ticks=config.sl_ticks,
            max_hold_ms=config.max_hold_ms,
            cancel_ms=config.cancel_ms,
            horizon=config.horizon,
            n_dates=len(daily_pnls), n_signals=total_signals,
            n_trades=metrics['n_trades'],
            fill_rate=metrics['fill_rate'],
            total_pnl_ticks=metrics['total_pnl_ticks'],
            total_pnl_dollars=metrics['total_pnl_dollars'],
            mean_pnl_per_trade=metrics.get('mean_pnl_ticks', 0),
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            sharpe=metrics['sharpe'],
            sortino=metrics['sortino'],
            avg_rr=metrics['avg_rr'],
            tp_rate=metrics['tp_rate'],
            sl_rate=metrics['sl_rate'],
            max_hold_rate=metrics['max_hold_rate'],
            eod_rate=metrics['eod_rate'],
            avg_queue_wait_ms=metrics['avg_queue_wait_ms'],
            avg_hold_time_ms=metrics['avg_hold_time_ms'],
            avg_slippage_ticks=metrics['avg_slippage_ticks'],
            daily_sharpe=daily.get('daily_sharpe', 0),
            daily_sortino=daily.get('daily_sortino', 0),
            max_drawdown=daily.get('max_drawdown', 0),
            win_days=daily.get('win_days', 0),
            loss_days=daily.get('loss_days', 0),
            best_day=daily.get('best_day', 0),
            worst_day=daily.get('worst_day', 0),
            pnl_per_day=daily.get('mean_daily_pnl', 0),
        )

    except Exception as e:
        log.error(f"Config {config.config_id} failed: {e}")
        return None


def save_results_csv(results: List[SweepResult], output_path: str):
    """Save sweep results to CSV."""
    if not results:
        log.warning("No results to save")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fields = list(asdict(results[0]).keys())

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    log.info(f"Saved {len(results)} results to {output_path}")


def print_top_results(results: List[SweepResult], sort_key: str, n: int = 10,
                      label: str = ""):
    """Print top N configs sorted by a metric."""
    valid = [r for r in results if r.n_trades > 0]
    if not valid:
        log.info(f"No valid results to rank by {sort_key}")
        return

    sorted_results = sorted(valid, key=lambda r: getattr(r, sort_key), reverse=True)

    print(f"\n{'='*120}")
    print(f"TOP {n} CONFIGS BY {sort_key.upper()} {label}")
    print(f"{'='*120}")
    print(f"{'Rank':>4} | {'Gate':>5} | {'TP':>4} | {'SL':>4} | {'Hold_ms':>8} | "
          f"{'Cancel':>7} | {'Trades':>6} | {'WR':>5} | {'PF':>5} | "
          f"{'Sharpe':>7} | {'Sortino':>8} | {'PnL$':>9} | {'SL%':>5} | "
          f"{'TP%':>5} | {'MH%':>5} | {'$/day':>8} | {'MDD$':>8}")
    print(f"{'-'*120}")

    for rank, r in enumerate(sorted_results[:n], 1):
        print(f"{rank:>4} | {r.gate_threshold:>5.2f} | {r.tp_ticks:>4.0f} | "
              f"{r.sl_ticks:>4.0f} | {r.max_hold_ms:>8.0f} | {r.cancel_ms:>7.0f} | "
              f"{r.n_trades:>6} | {r.win_rate:>5.1%} | {r.profit_factor:>5.2f} | "
              f"{r.sharpe:>7.3f} | {r.sortino:>8.3f} | "
              f"${r.total_pnl_dollars:>+8,.0f} | {r.sl_rate:>5.1%} | "
              f"{r.tp_rate:>5.1%} | {r.max_hold_rate:>5.1%} | "
              f"${r.pnl_per_day:>+7,.0f} | ${r.max_drawdown:>7,.0f}")

    # Print the absolute best config details
    best = sorted_results[0]
    print(f"\n  BEST by {sort_key}: config_id={best.config_id}")
    print(f"    gate={best.gate_threshold}, TP={best.tp_ticks}t, SL={best.sl_ticks}t, "
          f"hold={best.max_hold_ms}ms, cancel={best.cancel_ms}ms")
    print(f"    {best.n_trades} trades over {best.n_dates} days, "
          f"fill_rate={best.fill_rate:.1%}")
    print(f"    PnL: ${best.total_pnl_dollars:+,.0f} "
          f"({best.total_pnl_ticks:+,.1f} ticks)")
    print(f"    WR={best.win_rate:.1%}, R:R={best.avg_rr:.2f}, "
          f"PF={best.profit_factor:.2f}")
    print(f"    Exit mix: TP={best.tp_rate:.1%} SL={best.sl_rate:.1%} "
          f"MH={best.max_hold_rate:.1%} EOD={best.eod_rate:.1%}")


def print_analysis(results: List[SweepResult]):
    """Print analysis of parameter sensitivities."""
    valid = [r for r in results if r.n_trades > 0]
    if not valid:
        return

    print(f"\n{'='*80}")
    print("PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*80}")

    # Analyze each parameter's marginal effect on PnL
    for param in ['gate_threshold', 'tp_ticks', 'sl_ticks', 'max_hold_ms', 'cancel_ms']:
        values = sorted(set(getattr(r, param) for r in valid))
        print(f"\n  {param}:")
        for val in values:
            subset = [r for r in valid if getattr(r, param) == val]
            pnls = [r.total_pnl_dollars for r in subset]
            avg_pnl = np.mean(pnls)
            avg_wr = np.mean([r.win_rate for r in subset])
            avg_sl = np.mean([r.sl_rate for r in subset])
            n_profitable = sum(1 for p in pnls if p > 0)
            print(f"    {val:>10} -> avg PnL=${avg_pnl:>+9,.0f}, "
                  f"WR={avg_wr:.1%}, SL%={avg_sl:.1%}, "
                  f"profitable={n_profitable}/{len(subset)}")

    # Best combos: configs where SL_rate < 40%
    low_sl = [r for r in valid if r.sl_rate < 0.40 and r.total_pnl_dollars > 0]
    if low_sl:
        print(f"\n  Profitable configs with SL_rate < 40%: {len(low_sl)}")
        top = sorted(low_sl, key=lambda r: r.total_pnl_dollars, reverse=True)[:5]
        for r in top:
            print(f"    gate={r.gate_threshold}, TP={r.tp_ticks}t, SL={r.sl_ticks}t, "
                  f"hold={r.max_hold_ms}ms -> ${r.total_pnl_dollars:+,.0f}, "
                  f"WR={r.win_rate:.1%}, SL={r.sl_rate:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description='FIFO Market Replay Parameter Sweep')

    # Paths
    parser.add_argument('--pred-dir', default=DEFAULT_PRED_DIR,
                        help='CNN-Mamba v2 prediction directory')
    parser.add_argument('--mbo-event-dir', default=DEFAULT_MBO_DIR,
                        help='Processed MBO event directory')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='Output CSV path')

    # Date selection
    parser.add_argument('--dates', type=str, default=None,
                        help='Comma-separated dates to use (e.g., 20260316,20260317)')
    parser.add_argument('--n-dates', type=int, default=10,
                        help='Number of representative dates to use (default: 10)')
    parser.add_argument('--all-dates', action='store_true',
                        help='Use all available dates (slower)')

    # Grid overrides
    parser.add_argument('--gate', type=float, nargs='+', default=None,
                        help='Gate thresholds to test')
    parser.add_argument('--tp', type=float, nargs='+', default=None,
                        help='TP ticks to test')
    parser.add_argument('--sl', type=float, nargs='+', default=None,
                        help='SL ticks to test')
    parser.add_argument('--hold', type=float, nargs='+', default=None,
                        help='Max hold ms to test')
    parser.add_argument('--cancel', type=float, nargs='+', default=None,
                        help='Cancel ms to test')
    parser.add_argument('--horizons', type=int, nargs='+', default=None,
                        help='Horizons to test (0=1s, 1=5s, 2=10s)')

    # Execution
    parser.add_argument('--max-configs', type=int, default=None,
                        help='Limit number of configs to test (for quick runs)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count-1)')
    parser.add_argument('--sequential', action='store_true',
                        help='Run sequentially (no multiprocessing)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show configs without running')

    args = parser.parse_args()

    # ── Build parameter grid ────────────────────────────────────────────
    grid = dict(DEFAULT_GRID)
    if args.gate:
        grid['gate_threshold'] = args.gate
    if args.tp:
        grid['tp_ticks'] = args.tp
    if args.sl:
        grid['sl_ticks'] = args.sl
    if args.hold:
        grid['max_hold_ms'] = args.hold
    if args.cancel:
        grid['cancel_ms'] = args.cancel
    if args.horizons:
        grid['horizon'] = args.horizons

    configs = build_grid(grid)

    if args.max_configs and len(configs) > args.max_configs:
        # Random subset for quick testing
        np.random.seed(42)
        indices = np.random.choice(len(configs), args.max_configs, replace=False)
        configs = [configs[i] for i in sorted(indices)]
        # Re-number
        for i, c in enumerate(configs):
            c.config_id = i

    log.info(f"Parameter sweep: {len(configs)} configurations")
    for k, v in grid.items():
        log.info(f"  {k}: {v}")

    if args.dry_run:
        print(f"\nDry run: {len(configs)} configs would be tested")
        print(f"Grid: {grid}")
        for c in configs[:20]:
            print(f"  #{c.config_id}: gate={c.gate_threshold}, TP={c.tp_ticks}, "
                  f"SL={c.sl_ticks}, hold={c.max_hold_ms}, cancel={c.cancel_ms}")
        if len(configs) > 20:
            print(f"  ... and {len(configs) - 20} more")
        return

    # ── Determine dates ─────────────────────────────────────────────────
    pred_dir = Path(args.pred_dir)
    available_dates = sorted([
        d.name for d in pred_dir.iterdir()
        if d.is_dir() and (d / 'predictions.npz').exists()
    ])

    if args.dates:
        dates = args.dates.split(',')
        # Validate
        dates = [d for d in dates if d in available_dates]
    elif args.all_dates:
        dates = available_dates
    else:
        dates = select_representative_dates(available_dates, args.n_dates)

    log.info(f"Using {len(dates)} dates: {dates[0]}..{dates[-1]}")
    log.info(f"Available: {len(available_dates)} total dates")

    # ── Pre-load data ───────────────────────────────────────────────────
    t0 = time.time()
    loaded_dates = preload_data(args.pred_dir, args.mbo_event_dir, dates)
    actual_dates = sorted(_GLOBAL_ENGINES.keys())
    t_load = time.time() - t0
    log.info(f"Data pre-load: {t_load:.1f}s ({len(actual_dates)} dates with MBO data)")

    if not actual_dates:
        log.error("No dates with both predictions and MBO data. Exiting.")
        return

    # ── Run sweep ───────────────────────────────────────────────────────
    n_workers = args.workers or max(1, cpu_count() - 1)
    total = len(configs)

    log.info(f"\nStarting sweep: {total} configs x {len(actual_dates)} dates")
    log.info(f"Workers: {'sequential' if args.sequential else n_workers}")
    log.info(f"Estimated: ~{total * len(actual_dates) * 30 / max(1, n_workers) / 60:.0f} "
             f"minutes (rough estimate)")

    t_start = time.time()
    results = []

    if args.sequential:
        # Sequential mode: use pre-loaded globals directly
        for i, config in enumerate(configs):
            result = run_single_config(config)
            if result:
                results.append(result)
            if (i + 1) % 10 == 0 or i == total - 1:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                profitable = sum(1 for r in results if r.total_pnl_dollars > 0)
                log.info(f"  [{i+1}/{total}] {elapsed:.0f}s elapsed, "
                         f"ETA {eta:.0f}s, {profitable}/{len(results)} profitable")
    else:
        # Parallel mode: use fork to share pre-loaded globals
        # Note: on Linux, Pool uses fork by default, so globals are shared
        with Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(run_single_config, configs)):
                if result:
                    results.append(result)
                if (i + 1) % max(1, total // 20) == 0 or i == total - 1:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed
                    eta = (total - i - 1) / rate if rate > 0 else 0
                    profitable = sum(1 for r in results if r.total_pnl_dollars > 0)
                    log.info(f"  [{i+1}/{total}] {elapsed:.0f}s elapsed, "
                             f"ETA {eta:.0f}s, {profitable}/{len(results)} profitable")

    t_total = time.time() - t_start
    log.info(f"\nSweep complete: {len(results)} results in {t_total:.1f}s "
             f"({t_total/max(1,len(results)):.2f}s/config)")

    # ── Save results ────────────────────────────────────────────────────
    # Add timestamp to filename
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base, ext = os.path.splitext(args.output)
    output_path = f"{base}_{ts}{ext}"
    save_results_csv(results, output_path)

    # Also save as latest (overwrite)
    save_results_csv(results, args.output)

    # Also save full results as JSON for richer analysis
    json_path = output_path.replace('.csv', '.json')
    with open(json_path, 'w') as f:
        json.dump({
            'sweep_config': {
                'grid': {k: [float(x) for x in v] for k, v in grid.items()},
                'n_configs': len(configs),
                'dates': actual_dates,
                'n_dates': len(actual_dates),
                'elapsed_s': round(t_total, 1),
                'timestamp': ts,
                'hostname': hostname,
            },
            'results': [asdict(r) for r in results],
        }, f, indent=2)
    log.info(f"JSON results saved to {json_path}")

    # ── Print top results ───────────────────────────────────────────────
    print_top_results(results, 'total_pnl_dollars', n=10, label="(best absolute PnL)")
    print_top_results(results, 'sharpe', n=10, label="(best per-trade Sharpe)")
    print_top_results(results, 'daily_sortino', n=10, label="(best daily Sortino)")
    print_top_results(results, 'profit_factor', n=10, label="(best Profit Factor)")
    print_analysis(results)

    # ── Summary ─────────────────────────────────────────────────────────
    valid = [r for r in results if r.n_trades > 0]
    profitable = [r for r in valid if r.total_pnl_dollars > 0]

    print(f"\n{'='*80}")
    print("SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"  Configs tested:     {len(results)}")
    print(f"  With trades:        {len(valid)}")
    print(f"  Profitable:         {len(profitable)} ({len(profitable)/max(1,len(valid)):.1%})")
    print(f"  Dates tested:       {len(actual_dates)}")
    print(f"  Runtime:            {t_total:.0f}s ({t_total/60:.1f} min)")
    print(f"  Output:             {output_path}")

    if profitable:
        best = max(profitable, key=lambda r: r.total_pnl_dollars)
        print(f"\n  BEST CONFIG:")
        print(f"    gate={best.gate_threshold}, TP={best.tp_ticks}t, "
              f"SL={best.sl_ticks}t, hold={best.max_hold_ms}ms, "
              f"cancel={best.cancel_ms}ms")
        print(f"    PnL: ${best.total_pnl_dollars:+,.0f} over {best.n_dates} days")
        print(f"    WR={best.win_rate:.1%}, PF={best.profit_factor:.2f}, "
              f"Sharpe={best.sharpe:.3f}")
        print(f"    SL_rate={best.sl_rate:.1%} (vs 57.7% baseline)")
    else:
        print(f"\n  No profitable configs found. Consider:")
        print(f"    - Wider TP (let winners run)")
        print(f"    - Wider SL (reduce stop-outs)")
        print(f"    - Longer hold time")
        print(f"    - Higher gate (trade only strongest signals)")


if __name__ == '__main__':
    main()
