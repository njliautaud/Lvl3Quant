#!/usr/bin/env python3
"""
Decision Tree Execution Sweep v1
=================================
Combines four execution optimizations into a unified sweep:
  1. Conviction quintile-based adaptive TP/SL
  2. Hour-of-day filters (skip chop hours)
  3. Trailing stop (vs fixed timeout)
  4. Vol-exit (fast adverse move detection)

Runs all combinations through Rust fill_sim_cli on real MBO data.
CPU-only — designed for Jupiter.

Usage:
    python decision_tree_sweep_v1.py
    python decision_tree_sweep_v1.py --workers 12 --model cnn-mamba-v2
"""

import sys
import json
import time
import argparse
import subprocess
import logging
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).resolve().parent.parent
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
EVENT_DIR_V3 = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3'
EVENT_DIR_V2 = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v2'
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results' / 'decision_tree_v1'
PRED_CACHE_DIR = LVL3_ROOT / 'execution' / 'pred_cache' / 'decision_tree_v1'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Model Prediction Directories ──
CNN_MAMBA_V2_DIR = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'
MAMBA_V7_DIR = LVL3_ROOT / 'output' / 'mamba_v7_tiny_smart_v3_mar_apr'

# ── Constants ──
TICK_VALUE = 12.50
COMMISSION_RT = 4.70
BARS_PER_SEC = 10
BAR_NS = 100_000_000
RTH_HOURS = 6.5
N_RTH_BARS = int(RTH_HOURS * 3600 * BARS_PER_SEC)
WINDOW = 1000
STRIDE = 500

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Logging ──
log = logging.getLogger('decision_tree')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(
    str(RESULTS_DIR / f'decision_tree_{_ts}.log'), mode='w', encoding='utf-8'
)
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)

# ============================================================
# Empirical findings from exec_deepdive_20260427
# ============================================================

# Best TP/SL per conviction quintile (from adaptive_tp_best_per_quintile.csv)
QUINTILE_TPSL = {
    1: {'tp': 8, 'sl': 20},   # Low conviction: wide SL, moderate TP
    2: {'tp': 15, 'sl': 15},  # Moderate: balanced
    3: {'tp': 6, 'sl': 20},   # Mid: tight TP, wide SL
    4: {'tp': 15, 'sl': 20},  # High: wide both
    5: {'tp': 10, 'sl': 6},   # Top conviction: tight SL, moderate TP
}

# Hours with negative Sortino (chop hours to potentially skip)
CHOP_HOURS = [9, 12]  # 9AM and noon ET are worst
BEST_HOURS = [10, 13]  # 10AM and 1PM ET are best


# ============================================================
# Decision Tree Config Generator
# ============================================================

def generate_decision_tree_configs() -> List[Dict]:
    """Generate all decision tree combinations for the sweep.

    Dimensions:
    - Z-threshold: [2.0, 2.5, 3.0]
    - TP/SL mode: [fixed_best, adaptive_quintile (requires per-quintile runs)]
    - Hour filter: [all_hours, skip_chop, prime_only, best_2h]
    - Exit mode: [timeout_60s, timeout_120s, trailing_3t, trailing_5t, signal_flip]
    - Vol-exit: [off, 5t_5bars, 3t_3bars]
    """
    configs = []

    # ── Dimension 1: Z-threshold entry gates ──
    z_thresholds = [2.0, 2.5, 3.0, 3.5]

    # ── Dimension 2: TP/SL combos ──
    # Fixed combos from empirical best-per-quintile plus sweeps
    tp_sl_combos = [
        {'tp': 8, 'sl': 15, 'label': 'tp8sl15'},     # Baseline
        {'tp': 9, 'sl': 15, 'label': 'tp9sl15'},      # Current default
        {'tp': 10, 'sl': 20, 'label': 'tp10sl20'},    # Wide SL (MAE-aware: 93% go red)
        {'tp': 8, 'sl': 20, 'label': 'tp8sl20'},      # Q1 best
        {'tp': 15, 'sl': 20, 'label': 'tp15sl20'},    # Q4 best (let winners run)
        {'tp': 10, 'sl': 6, 'label': 'tp10sl6'},      # Q5 best (tight SL)
        {'tp': 12, 'sl': 13, 'label': 'tp12sl13'},    # Balanced near MAE p50
        {'tp': 6, 'sl': 20, 'label': 'tp6sl20'},      # Q3 best (quick scalp)
    ]

    # ── Dimension 3: Time filters ──
    time_filters = [
        {'start': '', 'end': '', 'label': 'all_hours'},
        {'start': '10:00', 'end': '15:00', 'label': 'skip_open'},
        {'start': '10:30', 'end': '14:30', 'label': 'prime'},
        {'start': '10:00', 'end': '14:00', 'label': 'best_4h'},
        {'start': '12:30', 'end': '14:00', 'label': 'best_90m'},
    ]

    # ── Dimension 4: Exit modes ──
    exit_modes = [
        {'hold_ms': 60000, 'trailing': 0, 'flip': False, 'label': 'hold60s'},
        {'hold_ms': 120000, 'trailing': 0, 'flip': False, 'label': 'hold120s'},
        {'hold_ms': 300000, 'trailing': 0, 'flip': False, 'label': 'hold300s'},
        {'hold_ms': 120000, 'trailing': 3, 'flip': False, 'label': 'trail3t_120s'},
        {'hold_ms': 120000, 'trailing': 5, 'flip': False, 'label': 'trail5t_120s'},
        {'hold_ms': 300000, 'trailing': 3, 'flip': False, 'label': 'trail3t_300s'},
        {'hold_ms': 300000, 'trailing': 0, 'flip': True, 'label': 'flip_300s'},
    ]

    # ── Dimension 5: Vol-exit ──
    vol_exits = [
        {'ticks': 0, 'bars': 0, 'label': 'novol'},
        {'ticks': 5, 'bars': 5, 'label': 'vol5t5b'},
        {'ticks': 3, 'bars': 3, 'label': 'vol3t3b'},
    ]

    # ── Generate priority configs (not full cartesian — too many) ──
    # Strategy: Fix vol-exit=off, sweep z × tp/sl × time × exit
    # Then add vol-exit variants for the best combos

    # Phase 1: Core sweep (z × tp/sl × time × exit) — no vol-exit
    for z in z_thresholds:
        for tpsl in tp_sl_combos:
            for tf in time_filters:
                for em in exit_modes:
                    label = f'z{z:.1f}_{tpsl["label"]}_{tf["label"]}_{em["label"]}'
                    cli_args = [
                        '--chase-entry',
                        '--signal-threshold', str(z),
                        '--take-profit-ticks', str(tpsl['tp']),
                        '--stop-loss-ticks', str(tpsl['sl']),
                        '--hold-ms', str(em['hold_ms']),
                        '--quiet',
                    ]
                    if em['trailing'] > 0:
                        cli_args.extend(['--trailing-ticks', str(em['trailing'])])
                    if em['flip']:
                        cli_args.append('--signal-flip-exit')
                    if tf['start']:
                        cli_args.extend([
                            '--time-window-start', tf['start'],
                            '--time-window-end', tf['end'],
                        ])

                    configs.append({
                        'label': label,
                        'cli_args': cli_args,
                        'z': z,
                        'tp': tpsl['tp'],
                        'sl': tpsl['sl'],
                        'time_filter': tf['label'],
                        'exit_mode': em['label'],
                        'vol_exit': 'none',
                    })

    # Phase 2: Vol-exit variants for promising z/tp/sl combos only
    promising_combos = [
        (2.5, 'tp9sl15'), (2.5, 'tp10sl20'), (2.5, 'tp8sl20'),
        (3.0, 'tp9sl15'), (3.0, 'tp10sl20'),
        (2.0, 'tp10sl20'), (2.0, 'tp8sl20'),
    ]
    for z, tpsl_label in promising_combos:
        tpsl = next(t for t in tp_sl_combos if t['label'] == tpsl_label)
        for ve in vol_exits[1:]:  # skip novol
            for tf in [time_filters[0], time_filters[2]]:  # all_hours + prime
                for em in [exit_modes[0], exit_modes[3]]:  # hold60s + trail3t_120s
                    label = f'z{z:.1f}_{tpsl_label}_{tf["label"]}_{em["label"]}_{ve["label"]}'
                    cli_args = [
                        '--chase-entry',
                        '--signal-threshold', str(z),
                        '--take-profit-ticks', str(tpsl['tp']),
                        '--stop-loss-ticks', str(tpsl['sl']),
                        '--hold-ms', str(em['hold_ms']),
                        '--vol-exit-ticks', str(ve['ticks']),
                        '--vol-exit-bars', str(ve['bars']),
                        '--quiet',
                    ]
                    if em['trailing'] > 0:
                        cli_args.extend(['--trailing-ticks', str(em['trailing'])])
                    if tf['start']:
                        cli_args.extend([
                            '--time-window-start', tf['start'],
                            '--time-window-end', tf['end'],
                        ])
                    configs.append({
                        'label': label,
                        'cli_args': cli_args,
                        'z': z,
                        'tp': tpsl['tp'],
                        'sl': tpsl['sl'],
                        'time_filter': tf['label'],
                        'exit_mode': em['label'],
                        'vol_exit': ve['label'],
                    })

    return configs


# ============================================================
# Prediction & MBO Loading (reused from advanced_strategies_v1)
# ============================================================

from datetime import timezone, timedelta

def rth_start_ns_for_date(date_str: str) -> int:
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


def load_fold(fold_path: Path) -> Optional[Dict]:
    try:
        data = np.load(str(fold_path), allow_pickle=True)
        preds = data['predictions']
        labels = data['labels']
        oot_path = str(data['oot_files'][0])
        basename = oot_path.replace('\\', '/').split('/')[-1]
        date_str = basename.split('_')[0]
        return {
            'predictions': preds.astype(np.float64),
            'labels': labels.astype(np.float64),
            'date_str': date_str,
            'n_samples': preds.shape[0],
            'fold_path': str(fold_path),
        }
    except Exception as e:
        log.warning(f"Failed to load {fold_path}: {e}")
        return None


def discover_folds(pred_dir: Path) -> Dict[str, Dict]:
    folds = {}
    for f in sorted(pred_dir.glob('fold_*_oot_predictions.npz')):
        if 'concat' in f.name:
            continue
        data = load_fold(f)
        if data:
            folds[data['date_str']] = data
            log.info(f"  Fold: {data['date_str']} -> {f.name} ({data['n_samples']} samples)")
    return folds


def load_event_timestamps(date_str: str) -> Optional[np.ndarray]:
    for edir in [EVENT_DIR_V3, EVENT_DIR_V2]:
        candidate = edir / f'{date_str}_mbo_events.npz'
        if candidate.exists():
            try:
                return np.load(str(candidate), allow_pickle=True)['timestamps']
            except Exception as e:
                log.warning(f"  Failed to load events {candidate}: {e}")
    return None


def predictions_to_bar_signal(
    predictions: np.ndarray,
    event_timestamps: np.ndarray,
    date_str: str,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Convert per-window predictions to bar-indexed z-scored signal."""
    n_events = len(event_timestamps)
    n_preds = len(predictions)

    starts = np.arange(0, n_events - WINDOW + 1, STRIDE, dtype=np.int64)
    label_idxs = starts + WINDOW - 1

    if len(label_idxs) > n_preds:
        label_idxs = label_idxs[:n_preds]
    elif n_preds > len(label_idxs):
        predictions = predictions[:len(label_idxs)]
        n_preds = len(predictions)

    if n_preds == 0:
        return np.zeros(N_RTH_BARS, dtype=np.float64), running_stats or {}

    pred_timestamps = event_timestamps[label_idxs]
    rth_start = rth_start_ns_for_date(date_str)
    bar_indices = ((pred_timestamps - rth_start) // BAR_NS).astype(np.int64)
    rth_mask = (bar_indices >= 0) & (bar_indices < N_RTH_BARS)

    bar_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
    for bi, sig in zip(bar_indices[rth_mask], predictions[rth_mask]):
        bar_preds[bi] = sig

    if running_stats is None:
        running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}

    zscore_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
    rs, rsq, cnt = running_stats['sum'], running_stats['sq'], running_stats['count']

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
            zscore_preds[i] = (v - mean) / std

    running_stats = {'sum': rs, 'sq': rsq, 'count': cnt}
    return zscore_preds, running_stats


def prepare_signals(folds: Dict[str, Dict]) -> Dict[str, Path]:
    """Prepare z-scored bar signals for all folds. Returns {date: npz_path}."""
    saved = {}
    running_stats = None

    for date_str in sorted(folds.keys()):
        fold_data = folds[date_str]
        cache_file = PRED_CACHE_DIR / f'standard_{date_str}.npz'

        # Check MBO file exists
        mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
        if not mbo_file.exists():
            mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
        if not mbo_file.exists():
            log.warning(f"  No MBO file for {date_str}, skipping")
            continue

        if cache_file.exists():
            saved[date_str] = cache_file
            # Update running stats for walk-forward consistency
            timestamps = load_event_timestamps(date_str)
            if timestamps is not None:
                p_10s = fold_data['predictions'][:, 2]
                _, running_stats = predictions_to_bar_signal(
                    p_10s, timestamps, date_str, running_stats
                )
            continue

        timestamps = load_event_timestamps(date_str)
        if timestamps is None:
            continue

        p_10s = fold_data['predictions'][:, 2]
        bar_signal, running_stats = predictions_to_bar_signal(
            p_10s, timestamps, date_str, running_stats
        )

        np.savez_compressed(str(cache_file), predictions=bar_signal)
        saved[date_str] = cache_file
        log.info(f"  Cached signal for {date_str}: {np.count_nonzero(bar_signal)} non-zero bars")

    return saved


# ============================================================
# Fill Sim Execution
# ============================================================

def run_fill_sim(
    date_str: str,
    pred_file: Path,
    config: Dict,
    out_dir: Path,
) -> Optional[Dict]:
    """Run Rust fill_sim_cli for a single day + config."""
    if not BINARY.exists():
        log.error(f"Binary not found: {BINARY}")
        return None

    mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
    if not mbo_file.exists():
        mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
    if not mbo_file.exists():
        return None

    out_file = out_dir / f'{config["label"]}_{date_str}.json'

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
    ] + config['cli_args']

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            if 'no trades' not in r.stderr.lower():
                log.debug(f"Sim failed {config['label']}/{date_str}: {r.stderr[:200]}")
            return None
        if not out_file.exists():
            return None
        with open(out_file) as f:
            result = json.load(f)
            result['_config_label'] = config['label']
            result['_date'] = date_str
            return result
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        log.debug(f"Error {config['label']}/{date_str}: {e}")
        return None


# ============================================================
# Results Aggregation
# ============================================================

def aggregate_results(results: List[Dict]) -> Dict[str, Dict]:
    """Aggregate per-config results across all dates."""
    by_config = defaultdict(list)
    for r in results:
        label = r.get('_config_label', 'unknown')
        by_config[label].append(r)

    summaries = {}
    for label, runs in by_config.items():
        n_dates = len(runs)
        total_pnl = sum(r.get('total_pnl_dollars', 0) for r in runs)
        total_trades = sum(r.get('total_trades', 0) for r in runs)

        if total_trades == 0:
            continue

        # Per-trade stats
        mean_pnl = total_pnl / total_trades if total_trades > 0 else 0

        # Win rate (weighted by trades per day)
        total_wins = sum(
            r.get('total_trades', 0) * r.get('win_rate', 0) for r in runs
        )
        win_rate = total_wins / total_trades if total_trades > 0 else 0

        # Profit factor (weighted average of daily PFs)
        # Compute from avg_win/avg_loss and win_rate
        total_gross_profit = sum(
            r.get('total_trades', 0) * r.get('win_rate', 0) * r.get('avg_win', 0)
            for r in runs
        )
        total_gross_loss = abs(sum(
            r.get('total_trades', 0) * (1 - r.get('win_rate', 0)) * r.get('avg_loss', 0)
            for r in runs
        ))
        pf = total_gross_profit / total_gross_loss if total_gross_loss > 0 else float('inf')

        # Daily PnL for Sharpe/Sortino
        daily_pnl = [r.get('total_pnl_dollars', 0) for r in runs]
        daily_mean = np.mean(daily_pnl) if daily_pnl else 0
        daily_std = np.std(daily_pnl) if len(daily_pnl) > 1 else 1
        daily_sharpe = daily_mean / daily_std if daily_std > 0 else 0

        # Sortino (downside deviation only)
        neg_returns = [d for d in daily_pnl if d < 0]
        downside_std = np.std(neg_returns) if len(neg_returns) > 1 else daily_std
        daily_sortino = daily_mean / downside_std if downside_std > 0 else 0

        # Fill rate
        total_signals = sum(r.get('total_signals', r.get('total_trades', 0)) for r in runs)
        fill_rate = total_trades / total_signals if total_signals > 0 else 0

        summaries[label] = {
            'n_dates': n_dates,
            'n_trades': total_trades,
            'total_pnl': round(total_pnl, 2),
            'mean_pnl_per_trade': round(mean_pnl, 4),
            'win_rate': round(win_rate, 4),
            'profit_factor': round(pf, 4),
            'daily_sharpe': round(daily_sharpe, 4),
            'daily_sortino': round(daily_sortino, 4),
            'fill_rate': round(fill_rate, 4),
            'gross_profit': round(total_gross_profit, 2),
            'gross_loss': round(-total_gross_loss, 2),
        }

    return summaries


def print_top_results(summaries: Dict[str, Dict], top_n: int = 30):
    """Print top results sorted by daily Sortino."""
    # Filter: require minimum trades
    filtered = {k: v for k, v in summaries.items() if v['n_trades'] >= 20}

    # Sort by daily Sortino
    sorted_by_sortino = sorted(
        filtered.items(), key=lambda x: x[1]['daily_sortino'], reverse=True
    )

    log.info(f"\n{'='*120}")
    log.info(f"TOP {top_n} CONFIGS BY DAILY SORTINO (min 20 trades)")
    log.info(f"{'='*120}")
    log.info(f"{'Config':<55} {'Trades':>7} {'PnL($)':>10} {'$/Trade':>9} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'Sortino':>8} {'Fill':>5}")
    log.info('-' * 120)

    for label, s in sorted_by_sortino[:top_n]:
        log.info(
            f"{label:<55} {s['n_trades']:>7} {s['total_pnl']:>10.0f} "
            f"{s['mean_pnl_per_trade']:>9.2f} {s['win_rate']:>5.1%} {s['profit_factor']:>6.3f} "
            f"{s['daily_sharpe']:>7.3f} {s['daily_sortino']:>8.3f} {s['fill_rate']:>5.1%}"
        )

    # Also sort by total PnL
    sorted_by_pnl = sorted(
        filtered.items(), key=lambda x: x[1]['total_pnl'], reverse=True
    )

    log.info(f"\n{'='*120}")
    log.info(f"TOP {top_n} CONFIGS BY TOTAL PnL (min 20 trades)")
    log.info(f"{'='*120}")
    log.info(f"{'Config':<55} {'Trades':>7} {'PnL($)':>10} {'$/Trade':>9} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'Sortino':>8}")
    log.info('-' * 120)

    for label, s in sorted_by_pnl[:top_n]:
        log.info(
            f"{label:<55} {s['n_trades']:>7} {s['total_pnl']:>10.0f} "
            f"{s['mean_pnl_per_trade']:>9.2f} {s['win_rate']:>5.1%} {s['profit_factor']:>6.3f} "
            f"{s['daily_sharpe']:>7.3f} {s['daily_sortino']:>8.3f}"
        )


def analyze_dimensions(summaries: Dict[str, Dict]):
    """Analyze which decision tree dimensions matter most."""
    # Parse config labels back to dimensions
    dimension_pnl = {
        'z': defaultdict(list),
        'tpsl': defaultdict(list),
        'time': defaultdict(list),
        'exit': defaultdict(list),
        'vol': defaultdict(list),
    }

    for label, s in summaries.items():
        parts = label.split('_')
        # Extract z-threshold
        z_part = [p for p in parts if p.startswith('z')]
        if z_part:
            dimension_pnl['z'][z_part[0]].append(s['total_pnl'])

        # Extract tp/sl combo
        tpsl_parts = [p for p in parts if p.startswith('tp')]
        if tpsl_parts:
            dimension_pnl['tpsl'][tpsl_parts[0]].append(s['total_pnl'])

    log.info(f"\n{'='*80}")
    log.info("DIMENSION IMPORTANCE ANALYSIS (avg PnL per dimension value)")
    log.info(f"{'='*80}")

    for dim_name, values in dimension_pnl.items():
        if not values:
            continue
        log.info(f"\n  {dim_name.upper()}:")
        for val, pnls in sorted(values.items()):
            avg = np.mean(pnls) if pnls else 0
            med = np.median(pnls) if pnls else 0
            log.info(f"    {val:<20} avg=${avg:>8.0f}  med=${med:>8.0f}  n={len(pnls)}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Decision Tree Execution Sweep')
    parser.add_argument('--model', default='cnn-mamba-v2', choices=['cnn-mamba-v2', 'mamba-v7'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-configs', type=int, default=0, help='Limit configs (0=all)')
    args = parser.parse_args()

    pred_dir = CNN_MAMBA_V2_DIR if args.model == 'cnn-mamba-v2' else MAMBA_V7_DIR
    model_name = args.model.replace('-', '_')

    log.info(f"Decision Tree Execution Sweep v1")
    log.info(f"  Model: {args.model}")
    log.info(f"  Predictions: {pred_dir}")
    log.info(f"  Workers: {args.workers}")
    log.info(f"  Binary: {BINARY}")

    # ── Discover folds ──
    log.info(f"\nDiscovering folds...")
    folds = discover_folds(pred_dir)
    log.info(f"  Found {len(folds)} folds: {sorted(folds.keys())}")

    if not folds:
        log.error("No folds found!")
        return

    # ── Prepare signals ──
    log.info(f"\nPreparing bar-indexed z-scored signals...")
    signal_files = prepare_signals(folds)
    log.info(f"  Prepared {len(signal_files)} date signals")

    # ── Generate configs ──
    all_configs = generate_decision_tree_configs()
    if args.max_configs > 0:
        all_configs = all_configs[:args.max_configs]
    log.info(f"\nGenerated {len(all_configs)} decision tree configs")

    # ── Build job list ──
    jobs = []
    for config in all_configs:
        for date_str, pred_file in sorted(signal_files.items()):
            jobs.append({
                'config': config,
                'date': date_str,
                'pred_file': pred_file,
            })

    log.info(f"  Total sim jobs: {len(jobs)} ({len(all_configs)} configs × {len(signal_files)} dates)")

    # ── Run all sims ──
    sim_out = RESULTS_DIR / f'sim_{model_name}_{_ts}'
    sim_out.mkdir(parents=True, exist_ok=True)

    results = []
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for job in jobs:
            future = executor.submit(
                run_fill_sim,
                job['date'], job['pred_file'], job['config'], sim_out,
            )
            futures[future] = job

        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                results.append(result)

            if done % 500 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                log.info(f"  Progress: {done}/{len(jobs)} ({done/len(jobs):.0%}) "
                         f"| {len(results)} results | {rate:.0f} sims/s "
                         f"| ETA: {(len(jobs)-done)/rate/60:.1f}min" if rate > 0 else "")

    elapsed = time.time() - t0
    log.info(f"\nCompleted {len(jobs)} sims in {elapsed:.0f}s ({len(results)} produced results)")

    # ── Aggregate ──
    summaries = aggregate_results(results)
    log.info(f"  {len(summaries)} unique configs with trades")

    # ── Report ──
    print_top_results(summaries, top_n=30)
    analyze_dimensions(summaries)

    # ── Save full results ──
    out_json = RESULTS_DIR / f'decision_tree_results_{model_name}_{_ts}.json'
    with open(out_json, 'w') as f:
        json.dump(summaries, f, indent=2)
    log.info(f"\nFull results saved to: {out_json}")

    # ── Save top configs ──
    filtered = {k: v for k, v in summaries.items() if v['n_trades'] >= 20}
    top_by_sortino = sorted(filtered.items(), key=lambda x: x[1]['daily_sortino'], reverse=True)[:10]
    top_json = RESULTS_DIR / f'top_configs_{model_name}_{_ts}.json'
    with open(top_json, 'w') as f:
        json.dump(dict(top_by_sortino), f, indent=2)
    log.info(f"Top 10 configs saved to: {top_json}")


if __name__ == '__main__':
    main()
