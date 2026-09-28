"""
Longer-Horizon Market Order Simulator

Tests signals at 15m, 30m, 1hr hold periods where cost-to-move ratio
is favorable (3-6% vs 50%+ at 10s horizons).

Key findings from IC study:
  - ask_L1_orders: IC=0.103 at 1hr (t=6.8), INCREASES at longer horizons
  - slow_decay_combo: IC=0.094 at 1hr (t=6.9), also persistent
  - Cost ratio: 3.0% at 1hr, 4.3% at 30m, 6.2% at 15m

CRITICAL: Tests first 50 days vs holdout 50 days separately.

Usage:
    python alpha_discovery/longer_horizon_sim.py
    python alpha_discovery/longer_horizon_sim.py --signal slow_decay_combo
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
from scipy import stats

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(str(RESULTS_DIR / f"longer_horizon_sim_{_ts}.log"), mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("lh_sim")

# Constants
TICK = 0.25
TICK_VAL = 12.50
COMM_TICKS = 3.00 / TICK_VAL  # 0.24 ticks RT commission
BARS_PER_SEC = 10

# Directories
SIGNAL_DIR = LVL3_ROOT / "data" / "processed" / "signal_predictions"
FEAT_CACHE = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"

# Hold periods to test (bars at 10/sec)
HOLD_CONFIGS = {
    '5m':  3000,   # 300 seconds
    '10m': 6000,   # 600 seconds
    '15m': 9000,   # 900 seconds
    '20m': 12000,  # 1200 seconds
    '30m': 18000,  # 1800 seconds
    '45m': 27000,  # 2700 seconds
    '1h':  36000,  # 3600 seconds
}

# Date split for holdout testing
HOLDOUT_START = "2025-09-22"  # First 50 days: Jul 14 - Sep 19; Holdout: Sep 22 - Nov 28


def load_day_data(date_str):
    """Load mid/spread from feature cache."""
    feat_path = FEAT_CACHE / f"{date_str}_mbo_features.npz"
    if not feat_path.exists():
        return None, None
    data = np.load(str(feat_path))
    feats = data['mbo_features']
    mid = feats[:, 0].astype(np.float64)
    spread = feats[:, 1].astype(np.float64)
    return mid, spread


def load_signal(sig_name, date_str):
    """Load signal predictions for a day."""
    path = SIGNAL_DIR / f"{sig_name}_{date_str}.npz"
    if not path.exists():
        return None
    return np.load(str(path))['predictions'].astype(np.float64)


def simulate_day(mid, spread, signal, thresh, hold_bars, cooldown=100):
    """
    Market-order sim for a single day with longer hold periods.

    Entry: abs(signal) > thresh after cooldown
    Exit: after hold_bars bars (fixed hold)
    Cost: spread/2 per side + commission
    """
    n = min(len(mid), len(signal))
    trades = []
    in_trade = False
    entry_bar = -9999
    entry_px = 0.0
    direction = 0
    last_exit = -cooldown - 1

    for i in range(n):
        if in_trade:
            if (i - entry_bar) >= hold_bars:
                # Exit at market
                if direction == 1:
                    exit_px = mid[i] - spread[i] / 2.0
                else:
                    exit_px = mid[i] + spread[i] / 2.0

                pnl_ticks = direction * (exit_px - entry_px) / TICK - COMM_TICKS
                pnl_dollars = pnl_ticks * TICK_VAL

                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'direction': direction,
                    'pnl_ticks': pnl_ticks,
                    'pnl_dollars': pnl_dollars,
                })
                in_trade = False
                last_exit = i

        elif (i - last_exit) >= cooldown:
            sig = signal[i]
            if abs(sig) > thresh:
                direction = 1 if sig > 0 else -1
                if direction == 1:
                    entry_px = mid[i] + spread[i] / 2.0
                else:
                    entry_px = mid[i] - spread[i] / 2.0
                in_trade = True
                entry_bar = i

    # Force close at EOD
    if in_trade:
        idx = n - 1
        exit_px = mid[idx]  # mid price at EOD (no spread penalty)
        pnl_ticks = direction * (exit_px - entry_px) / TICK - COMM_TICKS
        pnl_dollars = pnl_ticks * TICK_VAL
        trades.append({
            'entry_bar': entry_bar,
            'exit_bar': idx,
            'direction': direction,
            'pnl_ticks': pnl_ticks,
            'pnl_dollars': pnl_dollars,
            'eod_close': True,
        })

    return trades


def analyze_period(day_results, period_name):
    """Analyze a set of day results."""
    if not day_results:
        return None

    pnls = np.array([r['pnl'] for r in day_results])
    trades_per_day = np.array([r['trades'] for r in day_results])
    wrs = np.array([r['wr'] for r in day_results if r['trades'] > 0])

    n_days = len(pnls)
    mean_pnl = np.mean(pnls)
    median_pnl = np.median(pnls)
    std_pnl = np.std(pnls, ddof=1) if n_days > 1 else 0
    sharpe = (mean_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0
    pos_days = np.sum(pnls > 0)
    total_trades = np.sum(trades_per_day)
    mean_wr = np.mean(wrs) if len(wrs) > 0 else 0
    t_stat, p_val = stats.ttest_1samp(pnls, 0) if n_days > 1 else (0, 1)

    return {
        'period': period_name,
        'n_days': n_days,
        'mean_pnl': float(mean_pnl),
        'median_pnl': float(median_pnl),
        'std_pnl': float(std_pnl),
        'sharpe': float(sharpe),
        'pct_pos': float(pos_days / n_days * 100),
        'total_trades': int(total_trades),
        'trades_per_day': float(np.mean(trades_per_day)),
        'win_rate': float(mean_wr * 100),
        't_stat': float(t_stat),
        'p_val': float(p_val),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal', type=str, default=None,
                        help='Signal name (default: test all)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer thresholds')
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("LONGER-HORIZON MARKET ORDER SIMULATOR")
    log.info("=" * 80)
    log.info("  Testing signals at 5m-1hr holds where cost ratio is favorable")
    log.info("  HOLDOUT SPLIT: First 50 days (Jul-Sep) vs Last 50 days (Oct-Nov)")
    log.info("")

    # Discover available signals
    if args.signal:
        signals_to_test = [args.signal]
    else:
        signals_to_test = ['slow_decay_combo', 'ask_orders', 'cancel_asym_chain',
                          'causal_chain', 'depth_ratio', 'depth_ratio_z',
                          'book_imb_z', 'order_frag', 'top5_ensemble']

    # Discover available dates
    all_dates = sorted(set(
        f.stem.replace("_mbo_features", "")
        for f in FEAT_CACHE.glob("*_mbo_features.npz")
    ))
    log.info("  Available dates: %d (%s to %s)", len(all_dates), all_dates[0], all_dates[-1])

    # Threshold range
    if args.quick:
        thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]
    else:
        thresholds = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    # Hold periods to test
    hold_labels = ['5m', '10m', '15m', '30m', '1h']

    all_results = []

    for sig_name in signals_to_test:
        # Check how many days have predictions
        sig_dates = sorted([
            d for d in all_dates
            if (SIGNAL_DIR / f"{sig_name}_{d}.npz").exists()
        ])

        if not sig_dates:
            log.info("  SKIP %s: no prediction files found", sig_name)
            continue

        log.info("")
        log.info("=" * 80)
        log.info("SIGNAL: %s (%d days available)", sig_name, len(sig_dates))
        log.info("=" * 80)

        # Split into IS and OOS
        is_dates = [d for d in sig_dates if d < HOLDOUT_START]
        oos_dates = [d for d in sig_dates if d >= HOLDOUT_START]
        log.info("  IS (training): %d days | OOS (holdout): %d days", len(is_dates), len(oos_dates))

        # Preload all data
        day_cache = {}
        for date in sig_dates:
            mid, spread = load_day_data(date)
            signal = load_signal(sig_name, date)
            if mid is not None and signal is not None:
                day_cache[date] = (mid, spread, signal)
        log.info("  Loaded %d days into cache", len(day_cache))

        for hold_label in hold_labels:
            hold_bars = HOLD_CONFIGS[hold_label]
            cooldown = max(100, hold_bars // 10)  # Scale cooldown with hold

            for thresh in thresholds:
                # Simulate all days
                is_results = []
                oos_results = []

                for date in sig_dates:
                    if date not in day_cache:
                        continue
                    mid, spread, signal = day_cache[date]
                    trades = simulate_day(mid, spread, signal, thresh, hold_bars, cooldown)

                    day_pnl = sum(t['pnl_dollars'] for t in trades)
                    n_trades = len(trades)
                    wins = sum(1 for t in trades if t['pnl_dollars'] > 0)
                    wr = wins / n_trades if n_trades > 0 else 0

                    day_result = {'date': date, 'pnl': day_pnl, 'trades': n_trades, 'wr': wr}

                    if date < HOLDOUT_START:
                        is_results.append(day_result)
                    else:
                        oos_results.append(day_result)

                # Analyze periods
                is_stats = analyze_period(is_results, 'IS')
                oos_stats = analyze_period(oos_results, 'OOS')
                all_stats = analyze_period(is_results + oos_results, 'ALL')

                if all_stats and all_stats['total_trades'] > 0:
                    result = {
                        'signal': sig_name,
                        'hold': hold_label,
                        'hold_bars': hold_bars,
                        'thresh': thresh,
                        'is': is_stats,
                        'oos': oos_stats,
                        'all': all_stats,
                    }
                    all_results.append(result)

                    # Log noteworthy results
                    is_pnl = is_stats['mean_pnl'] if is_stats else 0
                    oos_pnl = oos_stats['mean_pnl'] if oos_stats else 0
                    all_pnl = all_stats['mean_pnl']
                    all_tr = all_stats['trades_per_day']
                    all_wr = all_stats['win_rate']

                    # Only log if there are meaningful trades
                    if all_tr >= 0.5:
                        marker = ""
                        if is_stats and oos_stats:
                            if is_pnl > 0 and oos_pnl > 0:
                                marker = " *** BOTH POSITIVE ***"
                            elif is_pnl > 0 and oos_pnl < 0:
                                marker = " [IS+/OOS-]"
                            elif is_pnl < 0 and oos_pnl > 0:
                                marker = " [IS-/OOS+]"

                        log.info("  %s t=%.1f hold=%s: IS=%+.0f OOS=%+.0f ALL=%+.0f  Tr/d=%.1f WR=%.0f%%%s",
                                sig_name, thresh, hold_label,
                                is_pnl, oos_pnl, all_pnl, all_tr, all_wr, marker)

    # ==================== GRAND SUMMARY ====================
    log.info("")
    log.info("=" * 80)
    log.info("GRAND SUMMARY — Configs where BOTH IS and OOS are positive")
    log.info("=" * 80)

    profitable = [r for r in all_results
                  if r.get('is') and r.get('oos')
                  and r['is']['mean_pnl'] > 0
                  and r['oos']['mean_pnl'] > 0]

    if not profitable:
        log.info("  NONE — No configs profitable in both IS and OOS periods.")
    else:
        profitable.sort(key=lambda r: r['oos']['mean_pnl'], reverse=True)
        for r in profitable[:20]:
            log.info("  %s t=%.1f %s: IS=%+.0f/d OOS=%+.0f/d ALL=%+.0f/d  Tr=%.1f/d WR=%.0f%%  IS_p=%.3f OOS_p=%.3f",
                    r['signal'], r['thresh'], r['hold'],
                    r['is']['mean_pnl'], r['oos']['mean_pnl'], r['all']['mean_pnl'],
                    r['all']['trades_per_day'], r['all']['win_rate'],
                    r['is']['p_val'], r['oos']['p_val'])

    # Summary: best OOS configs regardless of IS
    log.info("")
    log.info("=" * 80)
    log.info("TOP 20 CONFIGS BY OOS PnL (regardless of IS)")
    log.info("=" * 80)

    oos_ranked = [r for r in all_results if r.get('oos') and r['oos']['total_trades'] > 10]
    oos_ranked.sort(key=lambda r: r['oos']['mean_pnl'], reverse=True)
    for r in oos_ranked[:20]:
        is_pnl = r['is']['mean_pnl'] if r.get('is') else 0
        log.info("  %s t=%.1f %s: IS=%+.0f OOS=%+.0f  Tr/d=%.1f WR=%.0f%%  OOS_p=%.3f",
                r['signal'], r['thresh'], r['hold'],
                is_pnl, r['oos']['mean_pnl'],
                r['oos']['trades_per_day'], r['oos']['win_rate'],
                r['oos']['p_val'])

    # Cost-ratio analysis
    log.info("")
    log.info("=" * 80)
    log.info("COST-RATIO ANALYSIS BY HORIZON")
    log.info("=" * 80)

    for hold_label in hold_labels:
        configs_at_hold = [r for r in all_results
                          if r['hold'] == hold_label
                          and r.get('all')
                          and r['all']['total_trades'] > 20]
        if not configs_at_hold:
            continue

        best = max(configs_at_hold, key=lambda r: r['all']['mean_pnl'])
        worst = min(configs_at_hold, key=lambda r: r['all']['mean_pnl'])

        log.info("  %s: Best=%s t=%.1f %+.0f/d | Worst=%s t=%.1f %+.0f/d | Tested=%d configs",
                hold_label,
                best['signal'], best['thresh'], best['all']['mean_pnl'],
                worst['signal'], worst['thresh'], worst['all']['mean_pnl'],
                len(configs_at_hold))

    # Save results
    out_path = RESULTS_DIR / f"longer_horizon_sim_{_ts}.json"
    with open(str(out_path), 'w') as f:
        json.dump({
            'timestamp': _ts,
            'signals_tested': signals_to_test,
            'holds': hold_labels,
            'thresholds': thresholds,
            'holdout_start': HOLDOUT_START,
            'n_configs_tested': len(all_results),
            'n_profitable_both': len(profitable),
            'results': all_results,
        }, f, indent=2)
    log.info("")
    log.info("Saved: %s", out_path)
    log.info("Total configs tested: %d", len(all_results))


if __name__ == '__main__':
    main()
