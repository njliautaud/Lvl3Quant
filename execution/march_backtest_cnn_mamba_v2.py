#!/usr/bin/env python3
"""
March Backtest — CNN-Mamba v2 Folds 6-9 (March Dates)
======================================================
Tests the TOP strategies on CNN-Mamba v2 predictions for March 2-5, 2026.
Folds 6-9 are true OOT — trained on earlier data, predicting March.

Strategies tested (3 + 2 controls):
  1. midday_z3.5_60s       — midday time filter, z>3.5, 60s hold
  2. momentum_3_z2.5_30s   — momentum 3 consecutive, z>2.5, 30s
  3. agree_all3_z3.0_midday — all horizons agree, z>3.0, midday
  4. midday_z2.5_30s        — midday control
  5. baseline_z2.5_30s      — no filter control

Data:
  Predictions: CNN-Mamba v2, folds 6-9 (Mar 2-5), smart_v3 features
  Fill sim: Rust fill_sim_cli with real MBO data
"""

import sys
import gc
import json
import time
import argparse
import subprocess
import logging
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
EVENT_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3'
PRED_DIR = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results' / 'march_backtest_v2'
PRED_CACHE_DIR = LVL3_ROOT / 'execution' / 'pred_cache' / 'march_backtest_v2'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── ES Futures Constants ──
TICK_VALUE = 12.50
POINT_VALUE = 50.00
COMMISSION_RT = 4.70

# ── Bar/Timing Constants ──
BARS_PER_SEC = 10
BAR_NS = 100_000_000  # 100ms
RTH_HOURS = 6.5
N_RTH_BARS = int(RTH_HOURS * 3600 * BARS_PER_SEC)  # 234000

# ── Model Constants ──
WINDOW = 1000
STRIDE = 500

# ── Fold Discovery: folds 6-9 only ──
FOLD_FILES = sorted(PRED_DIR.glob('fold_0[6-9]_oot_predictions.npz'))

# ── Timestamp ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Logging ──
_log_file = str(RESULTS_DIR / f'march_backtest_{_ts}.log')
log = logging.getLogger('march_backtest')
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
    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    d = datetime(year, month, day)
    dst_start_2026 = datetime(2026, 3, 8)
    dst_end_2026 = datetime(2026, 11, 1)
    dst_start_2025 = datetime(2025, 3, 9)
    dst_end_2025 = datetime(2025, 11, 2)
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
    try:
        data = np.load(str(fold_path), allow_pickle=True)
        oot_path = str(data['oot_files'][0])
        basename = oot_path.replace('\\', '/').split('/')[-1]
        date_str = basename.split('_')[0]
        result = {
            'predictions': data['predictions'].astype(np.float64),  # (N, 3)
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


def multi_horizon_expanding_zscore(
    predictions_3h: np.ndarray,
    event_timestamps: np.ndarray,
    date_str: str,
    running_stats_list: Optional[List[Dict]] = None,
) -> Tuple[List[np.ndarray], List[Dict]]:
    if running_stats_list is None:
        running_stats_list = [None, None, None]

    z_signals = []
    new_stats = []
    for h in range(3):
        preds_h = predictions_3h[:, h]
        bar_idx, preds_rth = map_predictions_to_bars(preds_h, event_timestamps, date_str)
        z_h, stats_h = expanding_zscore_bar_signal(preds_rth, bar_idx, running_stats_list[h])
        z_signals.append(z_h)
        new_stats.append(stats_h)

    return z_signals, new_stats


# ============================================================
# Signal Generators
# ============================================================

def generate_base_10s_signal(fold_data, event_timestamps, date_str, running_stats=None):
    preds_10s = fold_data['predictions'][:, 2]
    bar_idx, preds_rth = map_predictions_to_bars(preds_10s, event_timestamps, date_str)
    return expanding_zscore_bar_signal(preds_rth, bar_idx, running_stats)


def generate_momentum_signal(fold_data, event_timestamps, date_str, n_consecutive, running_stats=None):
    preds_10s = fold_data['predictions'][:, 2]
    bar_idx, preds_rth = map_predictions_to_bars(preds_10s, event_timestamps, date_str)
    z_signal, new_stats = expanding_zscore_bar_signal(preds_rth, bar_idx, running_stats)

    nonzero_bars = np.where(z_signal != 0)[0]
    if len(nonzero_bars) < n_consecutive:
        return np.zeros(N_RTH_BARS, dtype=np.float64), new_stats

    momentum_signal = np.zeros(N_RTH_BARS, dtype=np.float64)
    recent_signs = []
    for bar in nonzero_bars:
        sign = 1 if z_signal[bar] > 0 else -1
        recent_signs.append(sign)
        if len(recent_signs) > n_consecutive:
            recent_signs = recent_signs[-n_consecutive:]
        if len(recent_signs) >= n_consecutive:
            if all(s == recent_signs[-1] for s in recent_signs[-n_consecutive:]):
                momentum_signal[bar] = z_signal[bar]

    return momentum_signal, new_stats


def generate_agree_all3_signal(fold_data, event_timestamps, date_str, running_stats_list=None):
    z_signals, new_stats = multi_horizon_expanding_zscore(
        fold_data['predictions'], event_timestamps, date_str, running_stats_list
    )

    agreement = (z_signals[0] != 0) & (z_signals[1] != 0) & (z_signals[2] != 0)
    ref_sign = np.sign(z_signals[0])
    agreement &= (np.sign(z_signals[1]) == ref_sign)
    agreement &= (np.sign(z_signals[2]) == ref_sign)

    result = np.where(agreement, z_signals[2], 0.0)
    return result, new_stats


# ============================================================
# Strategy Specification
# ============================================================

@dataclass
class StrategySpec:
    label: str
    description: str
    signal_type: str  # 'base_10s', 'momentum_3', 'agree_all3'
    z_threshold: float
    hold_ms: int = 30000
    stop_loss_ticks: Optional[int] = None
    take_profit_ticks: Optional[int] = None
    time_window_start: str = ""
    time_window_end: str = ""

    def to_cli_args(self) -> List[str]:
        args = [
            '--chase-entry',
            '--chase-max-ticks', '1',
            '--chase-max-reprices', '3',
            '--signal-threshold', str(self.z_threshold),
            '--hold-ms', str(self.hold_ms),
        ]
        if self.stop_loss_ticks is not None:
            args.extend(['--stop-loss-ticks', str(self.stop_loss_ticks)])
        if self.take_profit_ticks is not None:
            args.extend(['--take-profit-ticks', str(self.take_profit_ticks)])
        if self.time_window_start:
            args.extend(['--time-window-start', self.time_window_start])
            args.extend(['--time-window-end', self.time_window_end])
        args.append('--quiet')
        return args


def build_strategies() -> List[StrategySpec]:
    """Build the 5 strategies to test on March dates."""
    return [
        # TOP strategy from midday optimization
        StrategySpec(
            label='midday_z3.5_60s',
            description='Midday 10-14, z>3.5, 60s hold',
            signal_type='base_10s',
            z_threshold=3.5, hold_ms=60000,
            time_window_start='10:00', time_window_end='14:00',
        ),
        # #1 from v4/v5: momentum 3 consecutive
        StrategySpec(
            label='momentum_3_z2.5_30s',
            description='Momentum 3 consecutive agree, z>2.5, 30s hold',
            signal_type='momentum_3',
            z_threshold=2.5, hold_ms=30000,
        ),
        # All 3 horizons agree + midday
        StrategySpec(
            label='agree_all3_z3.0_midday',
            description='All 3 horizons agree, z>3.0, midday 10:00-14:00',
            signal_type='agree_all3',
            z_threshold=3.0, hold_ms=30000,
            time_window_start='10:00', time_window_end='14:00',
        ),
        # Midday control
        StrategySpec(
            label='midday_z2.5_30s',
            description='Base 10s, z>2.5, midday 10:00-14:00, 30s hold',
            signal_type='base_10s',
            z_threshold=2.5, hold_ms=30000,
            time_window_start='10:00', time_window_end='14:00',
        ),
        # Baseline control (no filters)
        StrategySpec(
            label='baseline_z2.5_30s',
            description='Base 10s, z>2.5, 30s hold (no filter)',
            signal_type='base_10s',
            z_threshold=2.5, hold_ms=30000,
        ),
    ]


# ============================================================
# Signal Preparation
# ============================================================

def prepare_all_signals() -> Tuple[Dict[str, Dict[str, Path]], List[str]]:
    log.info("\nPreparing prediction signals from CNN-Mamba v2 folds 6-9...")

    folds_data = []
    for fp in FOLD_FILES:
        fd = load_fold(fp)
        if fd:
            folds_data.append(fd)
            log.info(f"  Loaded fold: {fd['date_str']} ({fd['n_samples']} samples) from {fp.name}")

    if not folds_data:
        log.error("No fold data loaded!")
        return {}, []

    folds_data.sort(key=lambda x: x['date_str'])
    valid_dates = [f['date_str'] for f in folds_data]
    log.info(f"  Valid dates ({len(valid_dates)}): {valid_dates}")

    timestamps_map = {}
    for fd in folds_data:
        ts = load_event_timestamps(fd['date_str'])
        if ts is not None:
            timestamps_map[fd['date_str']] = ts
        else:
            log.warning(f"  No event timestamps for {fd['date_str']}")

    signal_cache = {}

    # ── base_10s signals ──
    log.info("  Generating base_10s signals...")
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
            log.info(f"    {date_str}: {n_nz} signals")
        else:
            data = np.load(str(cache_path))
            nz = data['predictions'][data['predictions'] != 0]
            if running_stats is None:
                running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}
            running_stats['sum'] += float(np.sum(nz))
            running_stats['sq'] += float(np.sum(nz ** 2))
            running_stats['count'] += len(nz)
            log.info(f"    {date_str}: cached ({len(nz)} nonzero)")
        signal_cache.setdefault('base_10s', {})[date_str] = cache_path

    # ── momentum_3 signals ──
    log.info("  Generating momentum_3 signals...")
    running_stats = None
    for fd in folds_data:
        date_str = fd['date_str']
        if date_str not in timestamps_map:
            continue
        cache_path = PRED_CACHE_DIR / f'momentum_3_{date_str}.npz'
        if not cache_path.exists():
            z_signal, running_stats = generate_momentum_signal(
                fd, timestamps_map[date_str], date_str, 3, running_stats
            )
            np.savez_compressed(str(cache_path), predictions=z_signal)
            n_nz = int(np.count_nonzero(z_signal))
            log.info(f"    {date_str}: {n_nz} signals")
        else:
            data = np.load(str(cache_path))
            nz = data['predictions'][data['predictions'] != 0]
            if running_stats is None:
                running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}
            running_stats['sum'] += float(np.sum(nz))
            running_stats['sq'] += float(np.sum(nz ** 2))
            running_stats['count'] += len(nz)
            log.info(f"    {date_str}: cached ({len(nz)} nonzero)")
        signal_cache.setdefault('momentum_3', {})[date_str] = cache_path

    # ── agree_all3 signals ──
    log.info("  Generating agree_all3 signals...")
    running_stats_list = None
    for fd in folds_data:
        date_str = fd['date_str']
        if date_str not in timestamps_map:
            continue
        cache_path = PRED_CACHE_DIR / f'agree_all3_{date_str}.npz'
        if not cache_path.exists():
            z_signal, running_stats_list = generate_agree_all3_signal(
                fd, timestamps_map[date_str], date_str, running_stats_list
            )
            np.savez_compressed(str(cache_path), predictions=z_signal)
            n_nz = int(np.count_nonzero(z_signal))
            log.info(f"    {date_str}: {n_nz} signals")
        else:
            log.info(f"    {date_str}: cached")
            if running_stats_list is None:
                running_stats_list = [None, None, None]
        signal_cache.setdefault('agree_all3', {})[date_str] = cache_path

    del timestamps_map
    gc.collect()

    log.info(f"  Signal types prepared: {list(signal_cache.keys())}")
    return signal_cache, valid_dates


# ============================================================
# Fill Simulator Interface
# ============================================================

def run_fill_sim(date_str, pred_file, strategy, out_dir):
    mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
    if not mbo_file.exists():
        mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
    if not mbo_file.exists():
        log.warning(f"No MBO file for {date_str}")
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
            log.warning(f"Sim failed {strategy.label}/{date_str}: {r.stderr[:300]}")
            return None
        if not out_file.exists():
            return None
        with open(out_file) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        log.warning(f"Timeout: {strategy.label}/{date_str}")
        return None
    except Exception as e:
        log.warning(f"Error {strategy.label}/{date_str}: {e}")
        return None


def run_strategy_sweep(strategies, signal_cache, workers=8):
    sim_out = RESULTS_DIR / f'sim_{_ts}'
    sim_out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for strat in strategies:
        date_files = signal_cache.get(strat.signal_type, {})
        for date_str, pred_file in sorted(date_files.items()):
            jobs.append({
                'date': date_str,
                'pred_file': pred_file,
                'strategy': strat,
            })

    log.info(f"\nRunning {len(jobs)} sim jobs ({workers} workers)")

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

            if done % 5 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                log.info(f"  [{done}/{len(jobs)}] {rate:.1f} jobs/s")

    elapsed = time.time() - t0
    log.info(f"Sweep done: {done} jobs in {elapsed:.1f}s")
    return results


# ============================================================
# Analysis & Reporting
# ============================================================

def aggregate_results(results, strategy_map):
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
        profitable_days = 0

        for date_str, res in sorted(date_results.items()):
            dates.append(date_str)
            day_pnl = res.get('total_pnl_dollars', 0)
            total_pnl += day_pnl
            total_trades += res.get('total_trades', 0)
            total_signals += res.get('total_signals', 0)
            total_filled += res.get('total_filled', 0)
            daily_pnls.append(day_pnl)
            if day_pnl > 0:
                profitable_days += 1

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

        # Sortino
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

        strat = strategy_map.get(label)

        summaries.append({
            'label': label,
            'description': strat.description if strat else '',
            'signal_type': strat.signal_type if strat else '',
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
            'profitable_day_rate': round(profitable_days / max(n_days, 1), 4),
            'long_trades': len(long_pnls),
            'long_pnl': round(sum(long_pnls), 2),
            'short_trades': len(short_pnls),
            'short_pnl': round(sum(short_pnls), 2),
            'daily_pnls': {d: round(p, 2) for d, p in zip(dates, daily_pnls)},
        })

    summaries.sort(key=lambda x: x['sortino'], reverse=True)
    return summaries


def print_results(summaries):
    log.info(f"\n{'=' * 170}")
    log.info(f" CNN-Mamba v2 — March Dates (Folds 6-9) — Execution Strategy Backtest")
    log.info(f"{'=' * 170}")

    if not summaries:
        log.info("  No results.")
        return

    header = (
        f"{'Strategy':<35} "
        f"{'Total P&L':>10} "
        f"{'Trades':>7} "
        f"{'T/Day':>6} "
        f"{'FillR':>6} "
        f"{'WinR':>6} "
        f"{'AvgTrd':>8} "
        f"{'Sortino':>8} "
        f"{'PF':>5} "
        f"{'MaxDD':>8} "
        f"{'ProfD':>6} "
        f"{'Long$':>8} "
        f"{'Short$':>8}"
    )
    log.info(header)
    log.info("-" * 170)

    for s in summaries:
        pnl_marker = '+' if s['total_pnl'] > 0 else ' '
        line = (
            f"{s['label']:<35} "
            f"{pnl_marker}${abs(s['total_pnl']):>8,.0f} "
            f"{s['n_trades']:>7} "
            f"{s['trades_per_day']:>5.1f} "
            f"{s['fill_rate']:>5.1%} "
            f"{s['win_rate']:>5.1%} "
            f"${s['avg_trade_pnl']:>7.2f} "
            f"{s['sortino']:>8.2f} "
            f"{s['profit_factor']:>5.2f} "
            f"${s['max_dd']:>7,.0f} "
            f"{s['profitable_days']}/{s['n_days']:>2} "
            f"${s['long_pnl']:>7,.0f} "
            f"${s['short_pnl']:>7,.0f}"
        )
        log.info(line)

    n_days = summaries[0]['n_days'] if summaries else 0
    log.info(f"\n  CNN-Mamba v2 | ES Futures | Tick=$12.50 | Commission=$4.70 RT | {n_days} OOT days (Mar 2-5)")

    # Detailed per-day breakdown
    log.info(f"\n{'=' * 100}")
    log.info(f"  DETAILED STRATEGY RESULTS")
    log.info(f"{'=' * 100}")

    for rank, s in enumerate(summaries):
        profitable_marker = "PROFITABLE" if s['total_pnl'] > 0 else "UNPROFITABLE"
        log.info(f"\n  #{rank + 1} {s['label']} [{profitable_marker}]")
        log.info(f"  {s['description']}")
        log.info(f"  Total P&L:       ${s['total_pnl']:,.2f}")
        log.info(f"  Trades:          {s['n_trades']} ({s['trades_per_day']:.1f}/day)")
        log.info(f"  Fill rate:       {s['fill_rate']:.1%}")
        log.info(f"  Win rate:        {s['win_rate']:.1%}")
        log.info(f"  Avg trade:       ${s['avg_trade_pnl']:.2f} ({s['avg_trade_ticks']:.2f} ticks)")
        log.info(f"  Sortino:         {s['sortino']:.2f}")
        log.info(f"  Profit factor:   {s['profit_factor']:.2f}")
        log.info(f"  Max DD:          ${s['max_dd']:,.2f}")
        log.info(f"  Profitable days: {s['profitable_days']}/{s['n_days']} ({s['profitable_day_rate']:.0%})")
        log.info(f"  Long:            {s['long_trades']} trades -> ${s['long_pnl']:,.2f}")
        log.info(f"  Short:           {s['short_trades']} trades -> ${s['short_pnl']:,.2f}")
        log.info(f"  Daily P&L:")
        for date, pnl in sorted(s['daily_pnls'].items()):
            marker = '+' if pnl >= 0 else '-'
            log.info(f"    {date}: {marker}${abs(pnl):,.2f}")

    # Verdict
    log.info(f"\n{'=' * 100}")
    log.info(f"  MARCH OOT VERDICT — CNN-Mamba v2")
    log.info(f"{'=' * 100}")

    n_profitable = len([s for s in summaries if s['total_pnl'] > 0])
    log.info(f"\n  Profitable strategies: {n_profitable}/{len(summaries)}")
    for s in summaries:
        status = "+" if s['total_pnl'] > 0 else "-"
        log.info(f"    [{status}] {s['label']:<35} P&L=${s['total_pnl']:>8,.2f}  "
                 f"Sortino={s['sortino']:>6.2f}  PF={s['profit_factor']:>5.2f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='March Backtest — CNN-Mamba v2')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--clear-cache', action='store_true')
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("MARCH BACKTEST — CNN-Mamba v2, Folds 6-9 (Mar 2-5)")
    log.info("=" * 80)
    log.info(f"  Model:       CNN-Mamba v2 (cnn_mamba_v2_smart_v3_mar)")
    log.info(f"  Folds:       6-9 (OOT)")
    log.info(f"  Dates:       Mar 2-5, 2026 (4 trading days)")
    log.info(f"  Instrument:  ES (tick=$12.50, commission=$4.70 RT)")
    log.info(f"  Workers:     {args.workers}")
    log.info("=" * 80)

    if args.clear_cache:
        import shutil
        if PRED_CACHE_DIR.exists():
            shutil.rmtree(PRED_CACHE_DIR)
            PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        log.info("  Cache cleared")

    strategies = build_strategies()
    strategy_map = {s.label: s for s in strategies}

    log.info(f"\n  Strategies to test: {len(strategies)}")
    for s in strategies:
        log.info(f"    {s.label:<35} {s.description}")

    if not BINARY.exists():
        log.error(f"fill_sim_cli not found: {BINARY}")
        sys.exit(1)

    # Phase 1: Prepare signals
    signal_cache, valid_dates = prepare_all_signals()
    if not signal_cache:
        log.error("No signals prepared. Check prediction files and event data.")
        sys.exit(1)

    # Phase 2: Run fill sim
    results = run_strategy_sweep(strategies, signal_cache, workers=args.workers)

    # Phase 3: Aggregate and report
    summaries = aggregate_results(results, strategy_map)
    print_results(summaries)

    # Save results
    out_file = RESULTS_DIR / f'march_backtest_results_{_ts}.json'
    with open(out_file, 'w') as f:
        json.dump(summaries, f, indent=2)
    log.info(f"\n  Results saved to: {out_file}")
    log.info(f"  Log saved to: {_log_file}")


if __name__ == '__main__':
    main()
