#!/usr/bin/env python3
"""
Midday Optimization v6 — Exhaustive Midday Window Parameter Sweep
==================================================================
Cross-validation showed ONLY midday-filtered strategies (10:00-14:00 ET)
are robust across both CNN-Mamba v2 and Mamba v7. This script exhaustively
optimizes parameters within the midday window.

Strategy Matrix (126 total):
  Base grid: 6 z-thresholds x 6 hold times x 3 exit modes = 108
  Time window variants: 3 windows x 6 z-thresholds (30s, signal_flip) = 18

Data: CNN-Mamba v2, folds 0-4 (Feb 23-27)
Fill sim: Rust fill_sim_cli with real MBO data

Usage:
    python midday_optimization_v6.py
    python midday_optimization_v6.py --workers 12
"""

import sys
import json
import time
import argparse
import subprocess
import logging
import os
import gc
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).resolve().parent.parent
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
EVENT_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3'
PRED_DIR = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results' / 'midday_v6'
PRED_CACHE_DIR = LVL3_ROOT / 'execution' / 'pred_cache' / 'midday_v6'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── ES Futures Constants ──────────────────────────────────────────────────
TICK_VALUE = 12.50
POINT_VALUE = 50.00
COMMISSION_RT = 4.70
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE

# ── Bar/Timing Constants ──
BARS_PER_SEC = 10
BAR_NS = 100_000_000  # 100ms
RTH_HOURS = 6.5
N_RTH_BARS = int(RTH_HOURS * 3600 * BARS_PER_SEC)  # 234000

# ── Model Constants ──
WINDOW = 1000
STRIDE = 500

# ── Fold Discovery (only folds 0-4, skip fold 05 with 758 preds) ──
FOLD_FILES = sorted(PRED_DIR.glob('fold_0[0-4]_oot_predictions.npz'))

# ── Timestamp ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Logging ──
_log_file = str(RESULTS_DIR / f'midday_v6_{_ts}.log')
log = logging.getLogger('midday_v6')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(_log_file, mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# RTH Timestamp Utilities
# ============================================================

def rth_start_ns_for_date(date_str: str) -> int:
    """Compute RTH start timestamp (9:30 AM ET) for date YYYYMMDD."""
    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    d = datetime(year, month, day)
    dst_start_2025 = datetime(2025, 3, 9)
    dst_end_2025 = datetime(2025, 11, 2)
    dst_start_2026 = datetime(2026, 3, 8)
    dst_end_2026 = datetime(2026, 11, 1)
    if (dst_start_2025 <= d < dst_end_2025) or (dst_start_2026 <= d < dst_end_2026):
        utc_offset = -4
    else:
        utc_offset = -5
    rth_start_utc_hours = 9.5 - utc_offset
    midnight_utc = datetime(year, month, day, tzinfo=timezone.utc)
    rth_start = midnight_utc + timedelta(hours=rth_start_utc_hours)
    return int(rth_start.timestamp() * 1e9)


# ============================================================
# Prediction Loading
# ============================================================

def load_fold(fold_path: Path) -> Optional[Dict]:
    """Load a CNN-Mamba v2 fold prediction file."""
    try:
        data = np.load(str(fold_path), allow_pickle=True)
        oot_path = str(data['oot_files'][0])
        basename = oot_path.replace('\\', '/').split('/')[-1]
        date_str = basename.split('_')[0]
        result = {
            'predictions': data['predictions'].astype(np.float64),
            'labels': data['labels'].astype(np.float64),
            'date_str': date_str,
            'n_samples': data['predictions'].shape[0],
            'fold_path': str(fold_path),
        }
        return result
    except Exception as e:
        log.warning(f"Failed to load {fold_path}: {e}")
        return None


def load_event_timestamps(date_str: str) -> Optional[np.ndarray]:
    """Load event timestamps for a date."""
    ev_file = EVENT_DIR / f'{date_str}_mbo_events.npz'
    if not ev_file.exists():
        return None
    try:
        ev_data = np.load(str(ev_file), allow_pickle=True)
        ts = ev_data['timestamps']
        del ev_data
        return ts
    except Exception as e:
        log.warning(f"Failed to load events for {date_str}: {e}")
        return None


# ============================================================
# Signal Generation Core
# ============================================================

def map_predictions_to_bars(
    predictions: np.ndarray,
    event_timestamps: np.ndarray,
    date_str: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map per-window predictions to bar indices. Returns only RTH-valid entries."""
    n_events = len(event_timestamps)
    n_preds = len(predictions)

    starts = np.arange(0, n_events - WINDOW + 1, STRIDE, dtype=np.int64)
    label_idxs = starts + WINDOW - 1

    if len(label_idxs) > n_preds:
        label_idxs = label_idxs[:n_preds]
    elif n_preds > len(label_idxs):
        predictions = predictions[:len(label_idxs)]

    if len(predictions) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    pred_timestamps = event_timestamps[label_idxs]
    rth_start = rth_start_ns_for_date(date_str)
    bar_indices = ((pred_timestamps - rth_start) // BAR_NS).astype(np.int64)
    rth_mask = (bar_indices >= 0) & (bar_indices < N_RTH_BARS)

    return bar_indices[rth_mask], predictions[rth_mask]


def expanding_zscore_bar_signal(
    raw_preds: np.ndarray,
    bar_indices: np.ndarray,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Apply expanding z-score to bar-level signal (no lookahead)."""
    if running_stats is None:
        running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}

    bar_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
    for bi, val in zip(bar_indices, raw_preds):
        bar_preds[bi] = val

    zscore = np.zeros(N_RTH_BARS, dtype=np.float64)
    rs = running_stats['sum']
    rsq = running_stats['sq']
    cnt = running_stats['count']

    for i in range(N_RTH_BARS):
        v = bar_preds[i]
        if v == 0.0:
            continue
        rs += v
        rsq += v * v
        cnt += 1
        if cnt >= 50:
            mean = rs / cnt
            var = (rsq / cnt) - mean * mean
            std = max(np.sqrt(max(var, 0)), 1e-8)
            zscore[i] = (v - mean) / std

    return zscore, {'sum': rs, 'sq': rsq, 'count': cnt}


def generate_base_10s_signal(
    fold_data: Dict,
    event_timestamps: np.ndarray,
    date_str: str,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Standard 10s horizon z-score signal."""
    preds_10s = fold_data['predictions'][:, 2]
    bar_idx, preds_rth = map_predictions_to_bars(preds_10s, event_timestamps, date_str)
    return expanding_zscore_bar_signal(preds_rth, bar_idx, running_stats)


# ============================================================
# Strategy Specification
# ============================================================

@dataclass
class StrategySpec:
    """Strategy specification for midday optimization."""
    label: str
    description: str

    # fill_sim_cli params
    signal_threshold: float = 2.5
    hold_ms: int = 30000
    chase_entry: bool = True
    stop_loss_ticks: Optional[int] = None
    take_profit_ticks: Optional[int] = None
    signal_flip_exit: bool = False
    time_window_start: str = "10:00"
    time_window_end: str = "14:00"

    def to_cli_args(self) -> List[str]:
        args = []
        if self.chase_entry:
            args.append('--chase-entry')
            args.extend(['--chase-max-ticks', '1'])
            args.extend(['--chase-max-reprices', '3'])

        args.extend(['--signal-threshold', str(self.signal_threshold)])
        args.extend(['--hold-ms', str(self.hold_ms)])

        if self.stop_loss_ticks is not None:
            args.extend(['--stop-loss-ticks', str(self.stop_loss_ticks)])
        if self.take_profit_ticks is not None:
            args.extend(['--take-profit-ticks', str(self.take_profit_ticks)])
        if self.signal_flip_exit:
            args.append('--signal-flip-exit')

        if self.time_window_start:
            args.extend(['--time-window-start', self.time_window_start])
            args.extend(['--time-window-end', self.time_window_end])

        args.append('--quiet')
        return args


def build_all_strategies() -> List[StrategySpec]:
    """Build the 126-strategy matrix for midday optimization."""
    strategies = []

    # ── Base parameters ──
    z_thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    hold_times_ms = [10000, 20000, 30000, 45000, 60000, 120000]
    hold_labels = ['10s', '20s', '30s', '45s', '60s', '120s']

    # ── 108 base strategies: 6 z x 6 hold x 3 exit modes ──
    for z in z_thresholds:
        for hold_ms, hold_label in zip(hold_times_ms, hold_labels):
            z_str = str(z).replace('.', '')

            # Exit mode 1: time_only (just hold for max time, exit at market)
            strategies.append(StrategySpec(
                label=f'mid_z{z_str}_{hold_label}_time',
                description=f'Midday 10-14, z>{z}, {hold_label} hold, time exit',
                signal_threshold=z,
                hold_ms=hold_ms,
                signal_flip_exit=False,
                time_window_start='10:00',
                time_window_end='14:00',
            ))

            # Exit mode 2: signal_flip (exit when prediction flips)
            strategies.append(StrategySpec(
                label=f'mid_z{z_str}_{hold_label}_flip',
                description=f'Midday 10-14, z>{z}, {hold_label} hold, signal flip exit',
                signal_threshold=z,
                hold_ms=hold_ms,
                signal_flip_exit=True,
                time_window_start='10:00',
                time_window_end='14:00',
            ))

            # Exit mode 3: bracket SL=3 TP=4
            strategies.append(StrategySpec(
                label=f'mid_z{z_str}_{hold_label}_brk34',
                description=f'Midday 10-14, z>{z}, {hold_label} hold, SL3/TP4 bracket',
                signal_threshold=z,
                hold_ms=hold_ms,
                stop_loss_ticks=3,
                take_profit_ticks=4,
                signal_flip_exit=False,
                time_window_start='10:00',
                time_window_end='14:00',
            ))

    # ── 18 time window variants: 3 windows x 6 z-thresholds ──
    # All use 30s hold + signal_flip (baseline exit mode)
    time_windows = [
        ('early', '10:00', '12:00', 'Early midday 10-12'),
        ('late', '12:00', '14:00', 'Late midday 12-14'),
        ('ext', '10:00', '15:00', 'Extended 10-15'),
    ]

    for tw_name, tw_start, tw_end, tw_desc in time_windows:
        for z in z_thresholds:
            z_str = str(z).replace('.', '')
            strategies.append(StrategySpec(
                label=f'{tw_name}_z{z_str}_30s_flip',
                description=f'{tw_desc}, z>{z}, 30s hold, signal flip exit',
                signal_threshold=z,
                hold_ms=30000,
                signal_flip_exit=True,
                time_window_start=tw_start,
                time_window_end=tw_end,
            ))

    return strategies


# ============================================================
# Signal Preparation
# ============================================================

def prepare_signals() -> Dict[str, Path]:
    """Prepare base_10s z-score NPZ files for each date.

    Returns: {date_str: pred_npz_path}
    """
    log.info("Preparing prediction signals...")

    folds_data = []
    for fp in FOLD_FILES:
        fd = load_fold(fp)
        if fd:
            folds_data.append(fd)
            log.info(f"  Loaded fold: {fd['date_str']} ({fd['n_samples']} samples)")

    if not folds_data:
        log.error("No fold data loaded!")
        return {}

    folds_data.sort(key=lambda x: x['date_str'])

    # Load event timestamps
    timestamps_map = {}
    for fd in folds_data:
        ts = load_event_timestamps(fd['date_str'])
        if ts is not None:
            timestamps_map[fd['date_str']] = ts
        else:
            log.warning(f"  No event timestamps for {fd['date_str']}")

    # Generate base_10s z-score signals (expanding window across folds)
    signal_files = {}
    running_stats = None

    for fd in folds_data:
        date_str = fd['date_str']
        if date_str not in timestamps_map:
            continue
        cache_path = PRED_CACHE_DIR / f'base_10s_{date_str}.npz'

        if not cache_path.exists():
            z_signal, running_stats = generate_base_10s_signal(
                fd, timestamps_map[date_str], date_str, running_stats
            )
            np.savez_compressed(str(cache_path), predictions=z_signal)
            n_nz = int(np.count_nonzero(z_signal))
            log.info(f"    {date_str}: {n_nz} non-zero z-scores")
        else:
            # Advance running stats from cache
            data = np.load(str(cache_path))
            nz = data['predictions'][data['predictions'] != 0]
            if running_stats is None:
                running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}
            running_stats['sum'] += float(np.sum(nz))
            running_stats['sq'] += float(np.sum(nz ** 2))
            running_stats['count'] += len(nz)
            log.info(f"    {date_str}: cached ({len(nz)} signals)")

        signal_files[date_str] = cache_path

    del timestamps_map
    gc.collect()

    log.info(f"  Signal preparation complete: {len(signal_files)} dates")
    return signal_files


# ============================================================
# Fill Simulator Interface
# ============================================================

def run_fill_sim(
    date_str: str,
    pred_file: Path,
    strategy: StrategySpec,
    out_dir: Path,
) -> Optional[Dict]:
    """Run Rust fill_sim_cli for a single day + strategy."""
    mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
    if not mbo_file.exists():
        mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
    if not mbo_file.exists():
        return None

    out_file = out_dir / f'{strategy.label}_{date_str}.json'

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
    ] + strategy.to_cli_args()

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            log.debug(f"Sim failed {strategy.label}/{date_str}: {r.stderr[:200]}")
            return None
        if not out_file.exists():
            return None
        with open(out_file) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        log.warning(f"Timeout: {strategy.label}/{date_str}")
        return None
    except Exception as e:
        log.debug(f"Error {strategy.label}/{date_str}: {e}")
        return None


def run_sweep(
    strategies: List[StrategySpec],
    signal_files: Dict[str, Path],
    workers: int = 8,
) -> Dict[str, Dict[str, Dict]]:
    """Run all strategies across all days in parallel.

    Returns: {strategy_label: {date_str: sim_result_dict}}
    """
    sim_out = RESULTS_DIR / f'sim_{_ts}'
    sim_out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for strategy in strategies:
        for date_str, pred_file in sorted(signal_files.items()):
            jobs.append({
                'date': date_str,
                'pred_file': pred_file,
                'strategy': strategy,
            })

    log.info(f"\nRunning {len(jobs)} sim jobs ({workers} workers)")
    log.info(f"  Strategies: {len(strategies)}")
    log.info(f"  Dates: {len(signal_files)}")

    results = {}
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for job in jobs:
            future = executor.submit(
                run_fill_sim,
                job['date'], job['pred_file'], job['strategy'], sim_out,
            )
            futures[future] = job

        for future in as_completed(futures):
            done += 1
            job = futures[future]
            try:
                result = future.result()
                if result:
                    label = job['strategy'].label
                    if label not in results:
                        results[label] = {}
                    results[label][job['date']] = result
            except Exception as e:
                log.debug(f"Job error: {e}")

            if done % 50 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(jobs) - done) / max(rate, 0.01)
                log.info(f"  [{done}/{len(jobs)}] {rate:.1f} jobs/s, "
                         f"~{remaining:.0f}s remaining")

    elapsed = time.time() - t0
    log.info(f"Sweep done: {done} jobs in {elapsed:.1f}s")
    return results


# ============================================================
# Analysis & Reporting
# ============================================================

def aggregate_results(
    results: Dict[str, Dict[str, Dict]],
    strategy_map: Dict[str, StrategySpec],
) -> List[Dict]:
    """Aggregate per-day sim results into per-strategy summaries."""
    summaries = []

    for label, date_results in results.items():
        total_pnl = 0.0
        total_trades = 0
        total_signals = 0
        total_filled = 0
        total_wins = 0
        daily_pnls = []
        all_trade_pnls = []
        dates = []
        long_pnls = []
        short_pnls = []

        for date_str, res in sorted(date_results.items()):
            dates.append(date_str)
            day_pnl = res.get('total_pnl_dollars', 0)
            total_pnl += day_pnl
            total_trades += res.get('total_trades', 0)
            total_signals += res.get('total_signals', 0)
            total_filled += res.get('total_filled', 0)
            daily_pnls.append(day_pnl)

            if 'trades' in res:
                for trade in res['trades']:
                    pnl = trade.get('pnl_dollars', 0)
                    all_trade_pnls.append(pnl)
                    if pnl > 0:
                        total_wins += 1
                    sig = trade.get('signal_strength', trade.get('entry_signal', 0))
                    if sig > 0:
                        long_pnls.append(pnl)
                    else:
                        short_pnls.append(pnl)

        n_days = len(date_results)
        if n_days == 0:
            continue

        win_rate = total_wins / max(total_trades, 1)
        fill_rate = total_filled / max(total_signals, 1)
        avg_daily = np.mean(daily_pnls) if daily_pnls else 0

        # Sortino ratio
        if len(daily_pnls) > 1:
            downside = [min(0, x) for x in daily_pnls]
            downside_std = np.std(downside)
            sortino = (avg_daily / max(downside_std, 1e-8)) * np.sqrt(252)
        else:
            sortino = 0.0

        gross_profit = sum(p for p in all_trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in all_trade_pnls if p < 0))
        profit_factor = gross_profit / max(gross_loss, 0.01)

        avg_trade_pnl = np.mean(all_trade_pnls) if all_trade_pnls else 0

        # Max drawdown
        cum = np.cumsum(daily_pnls) if daily_pnls else np.array([0])
        peak = np.maximum.accumulate(cum)
        max_dd = abs(float((cum - peak).min())) if len(cum) > 0 else 0

        # Consistency: how many days profitable?
        profitable_days = sum(1 for p in daily_pnls if p > 0)

        spec = strategy_map.get(label)
        description = spec.description if spec else label

        summaries.append({
            'label': label,
            'description': description,
            'total_pnl': round(total_pnl, 2),
            'n_days': n_days,
            'n_trades': total_trades,
            'n_signals': total_signals,
            'trades_per_day': round(total_trades / max(n_days, 1), 1),
            'fill_rate': round(fill_rate, 4),
            'win_rate': round(win_rate, 4),
            'sortino': round(sortino, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_daily_pnl': round(avg_daily, 2),
            'avg_trade_pnl': round(avg_trade_pnl, 2),
            'avg_trade_ticks': round(avg_trade_pnl / TICK_VALUE, 3),
            'max_dd': round(max_dd, 2),
            'profitable_days': profitable_days,
            'consistency': round(profitable_days / max(n_days, 1), 2),
            'long_trades': len(long_pnls),
            'long_pnl': round(sum(long_pnls), 2),
            'short_trades': len(short_pnls),
            'short_pnl': round(sum(short_pnls), 2),
            'daily_pnls': {d: round(p, 2) for d, p in zip(dates, daily_pnls)},
        })

    summaries.sort(key=lambda x: x['sortino'], reverse=True)
    return summaries


def print_top_results(summaries: List[Dict], title: str, sort_key: str, n: int = 20):
    """Print top N results sorted by a given key."""
    sorted_s = sorted(summaries, key=lambda x: x.get(sort_key, 0), reverse=True)

    log.info(f"\n{'=' * 170}")
    log.info(f" {title} (Top {n} by {sort_key})")
    log.info(f"{'=' * 170}")

    if not sorted_s:
        log.info("  No results.")
        return

    header = (
        f"{'#':>3} "
        f"{'Strategy':<32} "
        f"{'Total P&L':>10} "
        f"{'Trades':>7} "
        f"{'T/Day':>6} "
        f"{'FillR':>6} "
        f"{'WinR':>6} "
        f"{'AvgTrd':>8} "
        f"{'Sortino':>8} "
        f"{'PF':>5} "
        f"{'MaxDD':>8} "
        f"{'Days+':>5} "
        f"{'Long$':>8} "
        f"{'Short$':>8} "
        f"{'DailyPnL':>40}"
    )
    log.info(header)
    log.info("-" * 170)

    for rank, s in enumerate(sorted_s[:n]):
        pnl_marker = '+' if s['total_pnl'] > 0 else ' '
        daily_str = ' | '.join(
            f"{d[-4:]}:{'+' if p >= 0 else ''}{p:.0f}"
            for d, p in sorted(s['daily_pnls'].items())
        )
        line = (
            f"{rank + 1:>3} "
            f"{s['label']:<32} "
            f"{pnl_marker}${abs(s['total_pnl']):>8,.0f} "
            f"{s['n_trades']:>7} "
            f"{s['trades_per_day']:>5.1f} "
            f"{s['fill_rate']:>5.1%} "
            f"{s['win_rate']:>5.1%} "
            f"${s['avg_trade_pnl']:>7.2f} "
            f"{s['sortino']:>8.2f} "
            f"{s['profit_factor']:>5.2f} "
            f"${s['max_dd']:>7,.0f} "
            f"{s['profitable_days']}/{s['n_days']} "
            f"${s['long_pnl']:>7,.0f} "
            f"${s['short_pnl']:>7,.0f} "
            f"{daily_str}"
        )
        log.info(line)


def print_detailed_top(summaries: List[Dict], n: int = 5):
    """Print detailed breakdown for top N strategies."""
    log.info(f"\n{'=' * 80}")
    log.info(f"  DETAILED TOP {n} STRATEGIES (by Sortino)")
    log.info(f"{'=' * 80}")

    for rank, s in enumerate(summaries[:n]):
        log.info(f"\n  #{rank + 1} {s['label']}")
        log.info(f"  {s['description']}")
        log.info(f"  Total P&L:     ${s['total_pnl']:,.2f}")
        log.info(f"  Trades:        {s['n_trades']} ({s['trades_per_day']:.1f}/day)")
        log.info(f"  Fill rate:     {s['fill_rate']:.1%}")
        log.info(f"  Win rate:      {s['win_rate']:.1%}")
        log.info(f"  Avg trade:     ${s['avg_trade_pnl']:.2f} ({s['avg_trade_ticks']:.2f} ticks)")
        log.info(f"  Sortino:       {s['sortino']:.2f}")
        log.info(f"  Profit factor: {s['profit_factor']:.2f}")
        log.info(f"  Max DD:        ${s['max_dd']:,.2f}")
        log.info(f"  Consistency:   {s['profitable_days']}/{s['n_days']} days profitable")
        log.info(f"  Long:          {s['long_trades']} trades -> ${s['long_pnl']:,.2f}")
        log.info(f"  Short:         {s['short_trades']} trades -> ${s['short_pnl']:,.2f}")
        log.info(f"  Daily P&L:")
        for date, pnl in sorted(s['daily_pnls'].items()):
            marker = '+' if pnl >= 0 else '-'
            log.info(f"    {date}: {marker}${abs(pnl):,.2f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Midday Optimization v6 — Exhaustive Midday Window Parameter Sweep',
    )
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--clear-cache', action='store_true')
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("MIDDAY OPTIMIZATION v6 — Exhaustive Parameter Sweep")
    log.info("=" * 80)
    log.info(f"  Model:       CNN-Mamba v2 (5 folds, Feb 23-27)")
    log.info(f"  Instrument:  ES (tick=$12.50, commission=$4.70 RT)")
    log.info(f"  Workers:     {args.workers}")
    log.info(f"  Focus:       Midday window optimization (10:00-14:00 ET)")
    log.info("=" * 80)

    # Clear cache if requested
    if args.clear_cache:
        import shutil
        if PRED_CACHE_DIR.exists():
            shutil.rmtree(PRED_CACHE_DIR)
            PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        log.info("  Cache cleared")

    # Build strategies
    strategies = build_all_strategies()
    strategy_map = {s.label: s for s in strategies}

    log.info(f"\nBuilt {len(strategies)} strategies:")
    log.info(f"  Base grid (6z x 6hold x 3exit): 108 strategies")
    log.info(f"  Time window variants (3win x 6z): 18 strategies")
    log.info(f"  Total: {len(strategies)}")

    if args.dry_run:
        for s in strategies:
            log.info(f"  {s.label:<35} {s.description}")
        log.info(f"\nDry run complete. {len(strategies)} strategies.")
        return

    # Check binary
    if not BINARY.exists():
        log.error(f"fill_sim_cli not found: {BINARY}")
        sys.exit(1)

    # ── Phase 1: Prepare signals ──
    log.info(f"\n{'=' * 60}")
    log.info(f"  PHASE 1: Signal Preparation")
    log.info(f"{'=' * 60}")

    signal_files = prepare_signals()
    if not signal_files:
        log.error("No signals prepared.")
        sys.exit(1)

    # ── Phase 2: Fill simulation sweep ──
    log.info(f"\n{'=' * 60}")
    log.info(f"  PHASE 2: Fill Simulation Sweep ({len(strategies)} x {len(signal_files)} = {len(strategies) * len(signal_files)} jobs)")
    log.info(f"{'=' * 60}")

    sim_results = run_sweep(strategies, signal_files, workers=args.workers)

    # ── Phase 3: Aggregate and report ──
    log.info(f"\n{'=' * 60}")
    log.info(f"  PHASE 3: Analysis & Reporting")
    log.info(f"{'=' * 60}")

    summaries = aggregate_results(sim_results, strategy_map)

    # Print top 20 by Sortino
    print_top_results(summaries, "TOP 20 BY SORTINO", 'sortino', n=20)

    # Best by P&L
    print_top_results(summaries, "TOP 10 BY TOTAL P&L", 'total_pnl', n=10)

    # Best by win rate (min 10 trades)
    filtered = [s for s in summaries if s['n_trades'] >= 10]
    print_top_results(filtered, "TOP 10 BY WIN RATE (min 10 trades)", 'win_rate', n=10)

    # Best by profit factor (min 10 trades)
    print_top_results(filtered, "TOP 10 BY PROFIT FACTOR (min 10 trades)", 'profit_factor', n=10)

    # Detailed top 5
    print_detailed_top(summaries, n=5)

    # ── Summary statistics ──
    profitable = [s for s in summaries if s['total_pnl'] > 0]
    consistent = [s for s in summaries if s['profitable_days'] >= 3]
    robust = [s for s in summaries if s['total_pnl'] > 0 and s['profitable_days'] >= 3]

    log.info(f"\n{'=' * 80}")
    log.info(f"  SWEEP SUMMARY")
    log.info(f"{'=' * 80}")
    log.info(f"  Total strategies tested:     {len(summaries)}")
    log.info(f"  Profitable strategies:       {len(profitable)}")
    log.info(f"  Consistent (3+ days green):  {len(consistent)}")
    log.info(f"  Robust (profitable + cons.): {len(robust)}")

    if robust:
        log.info(f"\n  ROBUST STRATEGIES (profitable + 3+ green days):")
        for s in sorted(robust, key=lambda x: x['sortino'], reverse=True)[:20]:
            log.info(f"    {s['label']:<32} Sortino={s['sortino']:>6.2f}  "
                     f"P&L=${s['total_pnl']:>8,.0f}  "
                     f"PF={s['profit_factor']:>5.2f}  "
                     f"WR={s['win_rate']:.1%}  "
                     f"Days={s['profitable_days']}/{s['n_days']}")

    # ── Exit mode comparison ──
    log.info(f"\n  EXIT MODE COMPARISON (avg across z/hold combos):")
    for mode, suffix in [('time_only', '_time'), ('signal_flip', '_flip'), ('bracket_SL3_TP4', '_brk34')]:
        mode_strats = [s for s in summaries if s['label'].startswith('mid_') and s['label'].endswith(suffix)]
        if mode_strats:
            avg_pnl = np.mean([s['total_pnl'] for s in mode_strats])
            avg_sortino = np.mean([s['sortino'] for s in mode_strats])
            n_profitable = sum(1 for s in mode_strats if s['total_pnl'] > 0)
            log.info(f"    {mode:<20} avg_pnl=${avg_pnl:>8,.0f}  avg_sortino={avg_sortino:>6.2f}  "
                     f"profitable={n_profitable}/{len(mode_strats)}")

    # ── Time window comparison ──
    log.info(f"\n  TIME WINDOW COMPARISON (avg across z-thresholds, 30s flip):")
    for tw_name, label_prefix in [('midday 10-14', 'mid_'), ('early 10-12', 'early_'),
                                   ('late 12-14', 'late_'), ('extended 10-15', 'ext_')]:
        if label_prefix == 'mid_':
            tw_strats = [s for s in summaries if s['label'].startswith('mid_') and s['label'].endswith('_flip')
                         and '30s' in s['label']]
        else:
            tw_strats = [s for s in summaries if s['label'].startswith(label_prefix)]
        if tw_strats:
            avg_pnl = np.mean([s['total_pnl'] for s in tw_strats])
            avg_sortino = np.mean([s['sortino'] for s in tw_strats])
            n_profitable = sum(1 for s in tw_strats if s['total_pnl'] > 0)
            log.info(f"    {tw_name:<20} avg_pnl=${avg_pnl:>8,.0f}  avg_sortino={avg_sortino:>6.2f}  "
                     f"profitable={n_profitable}/{len(tw_strats)}")

    # ── Save results ──
    out_file = RESULTS_DIR / f'midday_v6_results_{_ts}.json'
    save_data = {
        'timestamp': _ts,
        'instrument': 'ES',
        'tick_value': TICK_VALUE,
        'commission_rt': COMMISSION_RT,
        'n_strategies_tested': len(summaries),
        'n_profitable': len(profitable),
        'n_consistent': len(consistent),
        'n_robust': len(robust),
        'strategies': summaries,
    }
    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    # Also save a compact version with just the top strategies
    top_file = RESULTS_DIR / f'midday_v6_top20_{_ts}.json'
    top_data = {
        'timestamp': _ts,
        'description': 'Top 20 midday strategies by Sortino for Monday paper trading',
        'strategies': summaries[:20],
    }
    with open(top_file, 'w') as f:
        json.dump(top_data, f, indent=2, default=str)

    log.info(f"\n{'=' * 80}")
    log.info(f"  MIDDAY OPTIMIZATION v6 COMPLETE")
    log.info(f"{'=' * 80}")
    log.info(f"  Strategies tested:  {len(summaries)}")
    log.info(f"  Profitable:         {len(profitable)}")
    log.info(f"  Robust:             {len(robust)}")
    if summaries:
        best = summaries[0]
        log.info(f"  Best strategy:      {best['label']} "
                 f"(Sortino={best['sortino']:.2f}, P&L=${best['total_pnl']:,.0f})")
    log.info(f"  Full results:       {out_file}")
    log.info(f"  Top 20 results:     {top_file}")
    log.info(f"  Log file:           {_log_file}")
    log.info(f"{'=' * 80}")


if __name__ == '__main__':
    main()
