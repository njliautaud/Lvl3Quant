#!/usr/bin/env python3
"""
Optuna-Based FIFO Execution Parameter Optimizer
================================================
Wraps the FIFO-approximate fill simulator from fifo_rules_optimizer.py
and uses Optuna TPE sampler to search a continuous parameter space.

Key findings from the grid sweep (19,854 configs):
  - SHORT-ONLY is profitable (long side has no edge after FIFO fills)
  - Top 3% threshold: 0.7 trades/day, 74% WR, PF 3.0
  - Top 5% threshold: 2.7 trades/day, marginal edge (57% WR, PF 1.15)
  - Top 10%+: completely negative

This optimizer:
  1. Fixes direction='short' based on sweep findings
  2. Searches continuous parameter space via TPE
  3. Adds spread_max_ticks and min_pred_z filters
  4. Maximizes drawdown-adjusted Daily Sortino
  5. Runs 500 trials with 14 parallel workers

Usage:
    python execution/optuna_fifo_optimizer.py --trials 500 --workers 14
    python execution/optuna_fifo_optimizer.py --trials 50 --workers 4 --quick
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from multiprocessing import Pool, cpu_count, Manager

# Ensure the execution directory is on the path for imports
EXEC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXEC_DIR))

from fifo_rules_optimizer import (
    load_date_data, simulate_date, RuleConfig, TradeResult,
    discover_oot_dates, compute_metrics, TICK_USD, COMMISSION_TICKS,
    generate_signals_for_config,
)

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

# ── Paths ───────────────────────────────────────────────────────────────────
LVL3_ROOT = Path('/home/jupiter/Lvl3Quant')
OUT_DIR   = LVL3_ROOT / 'execution' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger('optuna_fifo')
log.setLevel(logging.INFO)

# Suppress verbose Optuna trial logs (keep warnings/errors)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Global data cache (populated before fork, shared via COW)
# =============================================================================
_GLOBAL_DATE_DATA: Dict[str, dict] = {}


# =============================================================================
# Extended signal filtering with spread and z-score floors
# =============================================================================

def filter_signals_extended(
    data: dict,
    config: RuleConfig,
    spread_max_ticks: float,
    min_pred_z: float,
) -> List[dict]:
    """
    Generate signals using the standard pipeline, then apply additional
    filters for spread and minimum z-score.
    """
    signals = generate_signals_for_config(
        data['preds'], data['zscores'], data['pred_ts'],
        data['valid_indices'], config,
    )

    if not signals:
        return signals

    event_spreads = data['event_spreads']
    n_events = data['n_events']

    filtered = []
    for sig in signals:
        eidx = sig['event_idx']

        # Bounds check
        if eidx >= n_events:
            continue

        # Spread filter: skip signals when spread is too wide
        spread = float(event_spreads[eidx])
        if spread > spread_max_ticks:
            continue

        # Minimum z-score floor
        if sig['strength'] < min_pred_z:
            continue

        filtered.append(sig)

    return filtered


def simulate_date_extended(
    data: dict,
    config: RuleConfig,
    spread_max_ticks: float,
    min_pred_z: float,
) -> Tuple[List[TradeResult], int]:
    """
    Wrapper around simulate_date that pre-filters signals with extended
    criteria (spread, z-score floor), then runs the standard simulation.

    We achieve this by temporarily replacing the predictions with a
    filtered subset before calling simulate_date, then restoring them.

    Actually, since simulate_date calls generate_signals_for_config
    internally, we need a different approach: we'll replicate the
    simulation loop but with our filtered signals. Instead of duplicating
    the entire engine, we modify the data dict to include only the
    prediction indices that pass our filters, then let simulate_date
    handle the rest.
    """
    # Get all signals first to count them
    all_signals = generate_signals_for_config(
        data['preds'], data['zscores'], data['pred_ts'],
        data['valid_indices'], config,
    )

    if not all_signals:
        return [], 0

    event_spreads = data['event_spreads']
    n_events = data['n_events']

    # Build mask of prediction indices that pass our extended filters
    valid_pred_indices = set()
    for sig in all_signals:
        eidx = sig['event_idx']
        if eidx >= n_events:
            continue
        spread = float(event_spreads[eidx])
        if spread > spread_max_ticks:
            continue
        if sig['strength'] < min_pred_z:
            continue
        valid_pred_indices.add(sig['pred_idx'])

    if not valid_pred_indices:
        return [], len(all_signals)

    # Create a modified data dict with only passing predictions
    # This is the cleanest way to filter without modifying the engine
    valid_mask = np.array([i in valid_pred_indices for i in range(len(data['preds']))], dtype=bool)

    # We need to remap: the engine uses indices into preds array
    # So we create a new preds/zscores/pred_ts/valid_indices with only passing entries
    # and adjust the threshold to pass everything (since we already filtered)
    new_preds = data['preds'][valid_mask]
    new_zscores = data['zscores'][valid_mask]
    new_pred_ts = data['pred_ts'][valid_mask]
    new_valid_indices = data['valid_indices'][valid_mask]

    if len(new_preds) == 0:
        return [], len(all_signals)

    # Create modified data dict
    data_filtered = dict(data)
    data_filtered['preds'] = new_preds
    data_filtered['zscores'] = new_zscores
    data_filtered['pred_ts'] = new_pred_ts
    data_filtered['valid_indices'] = new_valid_indices
    data_filtered['n_preds'] = len(new_preds)

    # Recompute z-scores for the filtered subset so threshold_pct works correctly
    zscores_new = np.zeros_like(new_preds)
    for h in range(3):
        col = new_preds[:, h]
        mu, sigma = col.mean(), col.std()
        if sigma > 1e-8:
            zscores_new[:, h] = (col - mu) / sigma
        else:
            zscores_new[:, h] = 0.0
    data_filtered['zscores'] = zscores_new

    # Now run standard simulation on filtered data
    trades, n_sigs = simulate_date(data_filtered, config)
    return trades, n_sigs


# =============================================================================
# Objective function
# =============================================================================

def evaluate_trial_params(params: dict) -> dict:
    """
    Evaluate one parameter set across all OOT dates.
    Returns metrics dict or None on failure.
    """
    # Build RuleConfig from params
    config = RuleConfig(
        config_id=params.get('trial_number', 0),
        horizon_mode=params['horizon_mode'],
        threshold_pct=params['threshold_pct'],
        direction='short',  # FIXED: short-only based on sweep findings
        tp_ticks=params['tp_ticks'],
        sl_ticks=params['sl_ticks'],
        max_hold_ms=params['max_hold_ms'],
        cancel_ms=params['cancel_ms'],
        signal_decay_exit=params['signal_decay_exit'],
        decay_threshold=params['decay_threshold'],
        signal_flip_exit=params['signal_flip_exit'],
        min_interval_ms=params['min_interval_ms'],
    )

    spread_max_ticks = params['spread_max_ticks']
    min_pred_z = params['min_pred_z']

    all_trades = []
    total_signals = 0
    n_dates = 0

    for date_str in sorted(_GLOBAL_DATE_DATA.keys()):
        data = _GLOBAL_DATE_DATA[date_str]
        trades, n_sigs = simulate_date_extended(
            data, config, spread_max_ticks, min_pred_z
        )
        all_trades.extend(trades)
        total_signals += n_sigs
        n_dates += 1

    metrics = compute_metrics(all_trades, total_signals, n_dates)

    # Compute the objective: drawdown-adjusted daily Sortino
    n_trades = metrics['n_trades']
    daily_sortino = metrics['daily_sortino']
    total_pnl = metrics['total_pnl_dollars']
    max_dd = metrics['max_drawdown']

    # Prune: too few trades
    if n_trades < 15:
        objective = -999.0
    elif total_pnl <= 0:
        # Negative or zero P&L — penalize
        objective = daily_sortino  # will be negative or zero
    else:
        # Drawdown penalty: sortino * (1 - max_dd / total_pnl)
        dd_ratio = max_dd / total_pnl if total_pnl > 0 else 1.0
        dd_penalty = max(0.0, 1.0 - dd_ratio)
        objective = daily_sortino * dd_penalty

    metrics['objective'] = round(float(objective), 6)
    metrics['params'] = params
    return metrics


def _worker_evaluate(params: dict) -> Optional[dict]:
    """Multiprocessing-safe wrapper for evaluate_trial_params."""
    try:
        return evaluate_trial_params(params)
    except Exception as e:
        log.error(f"Trial failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Optuna study
# =============================================================================

def create_objective(study_results: list):
    """
    Create an Optuna objective function that samples parameters,
    dispatches evaluation, and returns the objective value.

    For parallel execution we use a different approach:
    we batch-sample from the study, evaluate in parallel, then
    tell the study the results. But Optuna's native parallelism
    via storage works better. Since we're using in-memory storage
    with a single process orchestrating, we'll use a simpler pattern:
    sequential Optuna sampling + parallel evaluation in batches.
    """

    def objective(trial: optuna.Trial) -> float:
        # ── Sample continuous parameter space ──
        params = {}

        params['threshold_pct'] = trial.suggest_float('threshold_pct', 1.0, 8.0)
        params['tp_ticks'] = trial.suggest_int('tp_ticks', 2, 16)
        params['sl_ticks'] = trial.suggest_int('sl_ticks', 2, 8)
        params['max_hold_ms'] = trial.suggest_int('max_hold_ms', 3000, 120000, log=True)
        params['cancel_ms'] = trial.suggest_int('cancel_ms', 500, 30000, log=True)
        params['min_interval_ms'] = trial.suggest_int('min_interval_ms', 500, 10000)
        params['horizon_mode'] = trial.suggest_categorical(
            'horizon_mode', ['1s', '5s', '10s', '2of3', '3of3']
        )

        # Direction fixed to short
        params['direction'] = 'short'

        # Signal-based exits
        params['signal_decay_exit'] = trial.suggest_categorical(
            'signal_decay_exit', [True, False]
        )
        if params['signal_decay_exit']:
            params['decay_threshold'] = trial.suggest_float('decay_threshold', 0.1, 0.8)
        else:
            params['decay_threshold'] = 0.3  # default, unused

        params['signal_flip_exit'] = trial.suggest_categorical(
            'signal_flip_exit', [True, False]
        )

        # NEW filters
        params['spread_max_ticks'] = trial.suggest_float('spread_max_ticks', 0.25, 2.0)
        params['min_pred_z'] = trial.suggest_float('min_pred_z', 1.0, 4.0)

        params['trial_number'] = trial.number

        # Evaluate
        result = evaluate_trial_params(params)
        if result is None:
            return -999.0

        # Store metrics for later retrieval
        trial.set_user_attr('n_trades', result['n_trades'])
        trial.set_user_attr('total_pnl_dollars', result['total_pnl_dollars'])
        trial.set_user_attr('daily_sortino', result['daily_sortino'])
        trial.set_user_attr('daily_sharpe', result['daily_sharpe'])
        trial.set_user_attr('win_rate', result['win_rate'])
        trial.set_user_attr('profit_factor', result['profit_factor'])
        trial.set_user_attr('max_drawdown', result['max_drawdown'])
        trial.set_user_attr('trades_per_day', result['trades_per_day'])
        trial.set_user_attr('pnl_per_day', result['pnl_per_day'])
        trial.set_user_attr('annual_sortino', result['annual_sortino'])
        trial.set_user_attr('rr_ratio', result['rr_ratio'])
        trial.set_user_attr('tp_rate', result['tp_rate'])
        trial.set_user_attr('sl_rate', result['sl_rate'])
        trial.set_user_attr('fill_rate', result['fill_rate'])
        trial.set_user_attr('avg_hold_time_ms', result['avg_hold_time_ms'])
        trial.set_user_attr('avg_queue_wait_ms', result['avg_queue_wait_ms'])

        study_results.append(result)

        obj = result['objective']
        return obj

    return objective


def run_parallel_optuna(n_trials: int, n_workers: int, seed: int = 42) -> Tuple[optuna.Study, list]:
    """
    Run Optuna with parallel workers using batch evaluation.

    Strategy: We sample `n_workers` trials at a time from the study,
    evaluate them in parallel with a process pool, then report results
    back. This gives us true parallelism while keeping Optuna's TPE
    informed of all results.
    """
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=max(20, n_workers * 2),  # random exploration before TPE kicks in
        multivariate=True,  # model parameter correlations
    )

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='fifo_short_optimizer',
    )

    all_results = []
    completed = 0
    t0 = time.time()

    log.info(f"Starting Optuna optimization: {n_trials} trials, {n_workers} workers")

    # We use Optuna's built-in parallel support via threading
    # Since our objective does CPU work in the same process (numpy),
    # and the data is shared via fork COW, we use a process pool pattern:
    # sample batch -> evaluate in parallel -> enqueue results

    # For simplicity and correctness with TPE, we'll use a process pool
    # where each worker runs its own trial sequence, but they share
    # an SQLite-backed study for synchronization.
    # However, for in-memory studies, Optuna supports threading.
    # Since our workload is CPU-bound (numpy), threads won't help.
    #
    # Best approach: use Optuna with SQLite storage + multiprocess workers.

    import tempfile
    db_path = tempfile.mktemp(suffix='.db')
    storage_url = f'sqlite:///{db_path}'

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(
            seed=seed,
            n_startup_trials=max(20, n_workers * 2),
            multivariate=True,
        ),
        study_name='fifo_short_optimizer',
        storage=storage_url,
        load_if_exists=True,
    )

    # Batch evaluation approach: sample params, eval in pool, tell study
    batch_size = n_workers
    remaining = n_trials

    with Pool(processes=n_workers) as pool:
        while remaining > 0:
            current_batch = min(batch_size, remaining)

            # Sample parameters for this batch using ask()
            trials_and_params = []
            for _ in range(current_batch):
                trial = study.ask()
                params = _sample_params_from_trial(trial)
                params['trial_number'] = trial.number
                trials_and_params.append((trial, params))

            # Evaluate batch in parallel
            param_list = [tp[1] for tp in trials_and_params]
            results = pool.map(_worker_evaluate, param_list)

            # Report results back to study
            for (trial, params), result in zip(trials_and_params, results):
                if result is None:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    continue

                obj = result['objective']

                # Store user attrs BEFORE tell() (trial becomes read-only after tell)
                for key in ['n_trades', 'total_pnl_dollars', 'daily_sortino',
                            'daily_sharpe', 'win_rate', 'profit_factor',
                            'max_drawdown', 'trades_per_day', 'pnl_per_day',
                            'annual_sortino', 'rr_ratio', 'tp_rate', 'sl_rate',
                            'fill_rate', 'avg_hold_time_ms', 'avg_queue_wait_ms']:
                    trial.set_user_attr(key, result.get(key, 0))

                study.tell(trial, obj)

                all_results.append(result)

            completed += current_batch
            remaining -= current_batch

            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (n_trials - completed) / rate if rate > 0 else 0

            # Find best so far
            try:
                best_val = study.best_value
                best_params = study.best_params
                best_trades = study.best_trial.user_attrs.get('n_trades', '?')
                best_pnl = study.best_trial.user_attrs.get('total_pnl_dollars', '?')
            except ValueError:
                best_val = -999
                best_trades = '?'
                best_pnl = '?'

            log.info(
                f"  [{completed}/{n_trials}] "
                f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining | "
                f"Best obj={best_val:.4f} ({best_trades} trades, ${best_pnl})"
            )

    # Clean up temp db
    try:
        os.unlink(db_path)
        os.unlink(db_path + '-journal')
    except FileNotFoundError:
        pass

    total_time = time.time() - t0
    log.info(f"Optimization complete: {len(all_results)} successful trials in {total_time:.1f}s")

    return study, all_results


def _sample_params_from_trial(trial: optuna.Trial) -> dict:
    """Sample all parameters from an Optuna trial using suggest_*."""
    params = {}

    params['threshold_pct'] = trial.suggest_float('threshold_pct', 1.0, 8.0)
    params['tp_ticks'] = trial.suggest_int('tp_ticks', 2, 16)
    params['sl_ticks'] = trial.suggest_int('sl_ticks', 2, 8)
    params['max_hold_ms'] = trial.suggest_int('max_hold_ms', 3000, 120000, log=True)
    params['cancel_ms'] = trial.suggest_int('cancel_ms', 500, 30000, log=True)
    params['min_interval_ms'] = trial.suggest_int('min_interval_ms', 500, 10000)
    params['horizon_mode'] = trial.suggest_categorical(
        'horizon_mode', ['1s', '5s', '10s', '2of3', '3of3']
    )
    params['direction'] = 'short'

    params['signal_decay_exit'] = trial.suggest_categorical(
        'signal_decay_exit', [True, False]
    )
    if params['signal_decay_exit']:
        params['decay_threshold'] = trial.suggest_float('decay_threshold', 0.1, 0.8)
    else:
        params['decay_threshold'] = 0.3

    params['signal_flip_exit'] = trial.suggest_categorical(
        'signal_flip_exit', [True, False]
    )

    params['spread_max_ticks'] = trial.suggest_float('spread_max_ticks', 0.25, 2.0)
    params['min_pred_z'] = trial.suggest_float('min_pred_z', 1.0, 4.0)

    return params


# =============================================================================
# Results formatting and output
# =============================================================================

def save_results(study: optuna.Study, all_results: list, output_path: str,
                 n_dates: int, dates: list, total_time: float, load_time: float,
                 n_workers: int):
    """Save all trial results to JSON."""
    # Sort results by objective
    all_results.sort(key=lambda r: r.get('objective', -999), reverse=True)

    # Build serializable trial data
    trials_data = []
    for r in all_results:
        trial_entry = {
            'objective': r.get('objective', -999),
            'params': {
                k: v for k, v in r.get('params', {}).items()
                if k != 'trial_number'
            },
        }
        # Add all metrics
        for key in ['n_trades', 'total_pnl_dollars', 'total_pnl_ticks',
                     'pnl_per_day', 'trades_per_day', 'win_rate', 'profit_factor',
                     'rr_ratio', 'daily_sharpe', 'daily_sortino',
                     'annual_sharpe', 'annual_sortino', 'max_drawdown',
                     'sharpe', 'sortino', 'fill_rate',
                     'tp_rate', 'sl_rate', 'max_hold_rate', 'decay_rate',
                     'flip_rate', 'eod_rate',
                     'avg_queue_wait_ms', 'avg_hold_time_ms',
                     'avg_spread_at_signal', 'win_days', 'loss_days',
                     'pct_profitable_days', 'best_day', 'worst_day',
                     'mean_pnl_per_trade', 'n_signals']:
            if key in r:
                trial_entry[key] = r[key]

        trials_data.append(trial_entry)

    # Best trial params
    try:
        best_params = study.best_params
        best_value = study.best_value
    except ValueError:
        best_params = {}
        best_value = -999

    output = {
        'metadata': {
            'optimizer': 'optuna_tpe',
            'n_trials': len(all_results),
            'n_dates': n_dates,
            'dates': dates,
            'direction': 'short (fixed)',
            'total_time_s': round(total_time, 1),
            'load_time_s': round(load_time, 1),
            'workers': n_workers,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'best_objective': round(best_value, 6),
            'best_params': best_params,
        },
        'trials': trials_data,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    log.info(f"Results saved to {output_path}")


def print_top_results(all_results: list, n_top: int = 20):
    """Print formatted table of top trials."""
    # Sort by objective
    sorted_results = sorted(all_results, key=lambda r: r.get('objective', -999), reverse=True)
    top = sorted_results[:n_top]

    if not top:
        print("No successful trials.")
        return

    print("\n" + "=" * 175)
    print(f"TOP {n_top} TRIALS BY OBJECTIVE (Drawdown-Adjusted Daily Sortino)")
    print("=" * 175)

    header = (
        f"{'#':>3} {'Obj':>7} "
        f"{'Hz':>5} {'Thr%':>5} {'TP':>3} {'SL':>3} "
        f"{'Hold':>6} {'Can':>5} {'Intv':>5} "
        f"{'SprdMax':>7} {'MinZ':>5} "
        f"{'Decay':>5} {'Flip':>4} "
        f"{'Trades':>6} {'T/Day':>5} {'WR%':>5} {'PF':>5} "
        f"{'Daily$':>8} {'Tot$':>9} "
        f"{'DSortino':>9} {'DSharpe':>8} "
        f"{'AnnSort':>8} {'MDD':>8} {'W/L':>5} {'R:R':>5} {'FillR':>5}"
    )
    print(header)
    print("-" * 175)

    for i, r in enumerate(top):
        p = r.get('params', {})
        decay_str = f"{p.get('decay_threshold', 0.3):.1f}" if p.get('signal_decay_exit') else '-'
        flip_str = 'Y' if p.get('signal_flip_exit') else '-'

        line = (
            f"{i+1:>3} "
            f"{r.get('objective', -999):>7.3f} "
            f"{p.get('horizon_mode', '?'):>5} "
            f"{p.get('threshold_pct', 0):>5.1f} "
            f"{p.get('tp_ticks', 0):>3} "
            f"{p.get('sl_ticks', 0):>3} "
            f"{p.get('max_hold_ms', 0):>6} "
            f"{p.get('cancel_ms', 0):>5} "
            f"{p.get('min_interval_ms', 0):>5} "
            f"{p.get('spread_max_ticks', 0):>7.2f} "
            f"{p.get('min_pred_z', 0):>5.1f} "
            f"{decay_str:>5} "
            f"{flip_str:>4} "
            f"{r.get('n_trades', 0):>6} "
            f"{r.get('trades_per_day', 0):>5.1f} "
            f"{r.get('win_rate', 0)*100:>5.1f} "
            f"{r.get('profit_factor', 0):>5.2f} "
            f"{r.get('pnl_per_day', 0):>8.1f} "
            f"{r.get('total_pnl_dollars', 0):>9.1f} "
            f"{r.get('daily_sortino', 0):>9.3f} "
            f"{r.get('daily_sharpe', 0):>8.3f} "
            f"{r.get('annual_sortino', 0):>8.2f} "
            f"{r.get('max_drawdown', 0):>8.1f} "
            f"{r.get('win_days', 0):>2}/{r.get('loss_days', 0):<2} "
            f"{r.get('rr_ratio', 0):>5.2f} "
            f"{r.get('fill_rate', 0):>5.2f}"
        )
        print(line)

    print("=" * 175)

    # Print best config summary
    best = top[0]
    bp = best.get('params', {})

    print(f"\n{'='*65}")
    print("BEST TRIAL SUMMARY")
    print(f"{'='*65}")
    print(f"  Objective:      {best.get('objective', 0):.4f}")
    print(f"  Horizon:        {bp.get('horizon_mode')}")
    print(f"  Threshold:      top {bp.get('threshold_pct', 0):.2f}%")
    print(f"  Direction:      short (fixed)")
    print(f"  TP/SL:          {bp.get('tp_ticks')}/{bp.get('sl_ticks')} ticks")
    print(f"  Max Hold:       {bp.get('max_hold_ms')}ms")
    print(f"  Cancel:         {bp.get('cancel_ms')}ms")
    print(f"  Min Interval:   {bp.get('min_interval_ms')}ms")
    print(f"  Spread Max:     {bp.get('spread_max_ticks', 0):.2f} ticks")
    print(f"  Min Pred Z:     {bp.get('min_pred_z', 0):.2f}")
    print(f"  Signal Decay:   {'Yes (th={:.2f})'.format(bp.get('decay_threshold', 0.3)) if bp.get('signal_decay_exit') else 'No'}")
    print(f"  Signal Flip:    {'Yes' if bp.get('signal_flip_exit') else 'No'}")
    print(f"  ---")
    print(f"  Trades:         {best.get('n_trades', 0)} ({best.get('trades_per_day', 0):.1f}/day)")
    print(f"  Fill Rate:      {best.get('fill_rate', 0)*100:.1f}%")
    print(f"  Win Rate:       {best.get('win_rate', 0)*100:.1f}%")
    print(f"  Profit Factor:  {best.get('profit_factor', 0):.2f}")
    print(f"  R:R Ratio:      {best.get('rr_ratio', 0):.2f}")
    print(f"  ---")
    print(f"  Total P&L:      ${best.get('total_pnl_dollars', 0):,.2f}")
    print(f"  Daily P&L:      ${best.get('pnl_per_day', 0):,.2f}")
    print(f"  Daily Sortino:  {best.get('daily_sortino', 0):.3f}")
    print(f"  Daily Sharpe:   {best.get('daily_sharpe', 0):.3f}")
    print(f"  Annual Sortino: {best.get('annual_sortino', 0):.2f}")
    print(f"  Max Drawdown:   ${best.get('max_drawdown', 0):,.2f}")
    print(f"  Win/Loss Days:  {best.get('win_days', 0)}/{best.get('loss_days', 0)}")
    print(f"  ---")
    print(f"  Exit Reasons:   TP={best.get('tp_rate',0)*100:.0f}% "
          f"SL={best.get('sl_rate',0)*100:.0f}% "
          f"MaxHold={best.get('max_hold_rate',0)*100:.0f}% "
          f"EOD={best.get('eod_rate',0)*100:.0f}%")
    print(f"  Avg Queue Wait: {best.get('avg_queue_wait_ms', 0):.0f}ms")
    print(f"  Avg Hold Time:  {best.get('avg_hold_time_ms', 0):.0f}ms")
    print(f"  Avg Spread:     {best.get('avg_spread_at_signal', 0):.1f} ticks")
    print(f"{'='*65}\n")

    # Parameter importance (approximate from top vs bottom half)
    if len(sorted_results) >= 20:
        print(f"{'='*65}")
        print("PARAMETER TENDENCIES (top 20% vs bottom 50%)")
        print(f"{'='*65}")
        n = len(sorted_results)
        top_20 = sorted_results[:max(1, n // 5)]
        bot_50 = sorted_results[n // 2:]

        numeric_params = ['threshold_pct', 'tp_ticks', 'sl_ticks', 'max_hold_ms',
                          'cancel_ms', 'min_interval_ms', 'spread_max_ticks', 'min_pred_z']
        for param in numeric_params:
            top_vals = [r['params'].get(param, 0) for r in top_20 if 'params' in r]
            bot_vals = [r['params'].get(param, 0) for r in bot_50 if 'params' in r]
            if top_vals and bot_vals:
                top_mean = np.mean(top_vals)
                bot_mean = np.mean(bot_vals)
                print(f"  {param:>20}: top20%={top_mean:>10.2f}  bot50%={bot_mean:>10.2f}")

        cat_params = ['horizon_mode', 'signal_decay_exit', 'signal_flip_exit']
        for param in cat_params:
            top_vals = [str(r['params'].get(param, '?')) for r in top_20 if 'params' in r]
            bot_vals = [str(r['params'].get(param, '?')) for r in bot_50 if 'params' in r]
            if top_vals:
                from collections import Counter
                top_counts = Counter(top_vals).most_common(3)
                bot_counts = Counter(bot_vals).most_common(3)
                top_str = ', '.join(f"{v}({c})" for v, c in top_counts)
                bot_str = ', '.join(f"{v}({c})" for v, c in bot_counts)
                print(f"  {param:>20}: top20%=[{top_str}]  bot50%=[{bot_str}]")

        print(f"{'='*65}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Optuna FIFO Execution Parameter Optimizer (SHORT-ONLY)'
    )
    parser.add_argument('--trials', type=int, default=500,
                        help='Number of Optuna trials (default: 500)')
    parser.add_argument('--workers', type=int, default=14,
                        help='Number of parallel workers (default: 14)')
    parser.add_argument('--dates', type=str, default=None,
                        help='Comma-separated dates (default: all OOT dates)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for TPE sampler')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('--quick', action='store_true',
                        help='Fewer trials (50) for quick testing')
    args = parser.parse_args()

    if args.quick:
        args.trials = min(args.trials, 50)
        log.info("Quick mode: limiting to 50 trials")

    # ── Discover dates ──
    if args.dates:
        dates = args.dates.split(',')
    else:
        dates = discover_oot_dates()

    if not dates:
        log.error("No OOT dates found. Check MBO_DIR and PRED_DIR paths.")
        return

    log.info(f"Found {len(dates)} OOT dates: {dates[0]}..{dates[-1]}")

    # ── Preload all date data (before fork) ──
    global _GLOBAL_DATE_DATA
    log.info("Loading MBO event data and predictions...")
    t_load = time.time()

    for i, date_str in enumerate(dates):
        data = load_date_data(date_str)
        if data is not None:
            _GLOBAL_DATE_DATA[date_str] = data
            log.info(f"  [{i+1}/{len(dates)}] {date_str}: "
                     f"{data['n_events']:,} events, {data['n_preds']} predictions")
        else:
            log.warning(f"  [{i+1}/{len(dates)}] Skipping {date_str}: missing data")

    load_time = time.time() - t_load
    total_events = sum(d['n_events'] for d in _GLOBAL_DATE_DATA.values())
    log.info(f"Loaded {len(_GLOBAL_DATE_DATA)} dates in {load_time:.1f}s "
             f"({total_events:,} total events)")

    if not _GLOBAL_DATE_DATA:
        log.error("No data loaded. Exiting.")
        return

    # ── Run Optuna optimization ──
    t_start = time.time()

    study, all_results = run_parallel_optuna(
        n_trials=args.trials,
        n_workers=args.workers,
        seed=args.seed,
    )

    total_time = time.time() - t_start

    if not all_results:
        log.error("No successful trials. Exiting.")
        return

    # ── Save results ──
    output_path = args.output or str(OUT_DIR / 'optuna_fifo_results.json')
    save_results(
        study, all_results, output_path,
        n_dates=len(_GLOBAL_DATE_DATA),
        dates=sorted(_GLOBAL_DATE_DATA.keys()),
        total_time=total_time,
        load_time=load_time,
        n_workers=args.workers,
    )

    # ── Print summary ──
    print_top_results(all_results, n_top=20)

    log.info(f"Total wall time: {total_time:.1f}s "
             f"({total_time / max(1, len(all_results)):.2f}s/trial)")


if __name__ == '__main__':
    main()
