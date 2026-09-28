"""
Dynamic Exit Strategy Simulator — Tests NON-STATIC exit approaches.

Strategies:
1. signal_flip: Hold until prediction sign flips
2. signal_weaken: Hold until |pred| drops below decay_factor * entry_strength
3. dynamic_trail: Trail tightens as signal weakens (adaptive)
4. multi_signal: Use one signal for entry, another for exit confirmation
5. regime_filter: Only trade when signal density is low (selective regime)

All use MARKET ORDER execution (cross the spread, 1.24 ticks cost per RT).

Usage:
    python alpha_discovery/dynamic_exit_sim.py --n-days 50
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
        logging.FileHandler(str(RESULTS_DIR / f"dynamic_exit_{_ts}.log"), mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("dyn_exit")

TICK = 0.25
TICK_VAL = 12.50
COMM_TICKS = 3.00 / TICK_VAL  # 0.24
TOTAL_COST_TICKS = 1.0 + COMM_TICKS  # 1.24 (spread + commission)
BARS_SEC = 10

FEAT_CACHE = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
SIG_DIR = LVL3_ROOT / "data" / "processed" / "signal_predictions"


def preload_days(n_days):
    """Preload mid + spread for all days."""
    days = {}
    for f in sorted(FEAT_CACHE.glob("*_mbo_features.npz"))[:n_days]:
        date = f.stem.replace("_mbo_features", "")
        data = np.load(str(f))
        feats = data['mbo_features']
        days[date] = {'mid': feats[:, 0].copy(), 'spread': feats[:, 1].copy()}
        del feats, data
    return days


def load_signal(sig_name, date):
    path = SIG_DIR / f"{sig_name}_{date}.npz"
    if not path.exists():
        return None
    return np.load(str(path))['predictions']


# ============================================================================
# Strategy 1: Signal-Flip Exit
# ============================================================================
def sim_signal_flip(mid, spread, preds, thresh, max_hold=3000, cooldown=10):
    """
    Entry: |pred| > thresh
    Exit: prediction sign flips OR max_hold reached
    """
    n = min(len(mid), len(preds))
    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0

    for i in range(n):
        if in_pos:
            # Check exit conditions
            bars = i - entry_bar
            pred_sign = 1 if preds[i] > 0 else (-1 if preds[i] < 0 else 0)

            do_exit = False
            # Signal flipped
            if direction == 1 and pred_sign == -1:
                do_exit = True
            elif direction == -1 and pred_sign == 1:
                do_exit = True
            # Max hold
            elif bars >= max_hold:
                do_exit = True

            if do_exit:
                if direction == 1:
                    unrealized = (mid[i] - entry_price) / TICK
                else:
                    unrealized = (entry_price - mid[i]) / TICK
                exit_cost = spread[i] / 2.0 / TICK + COMM_TICKS
                pnl = unrealized - exit_cost
                total_pnl += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                in_pos = False
                last_exit = i

        elif i - last_exit >= cooldown:
            p = preds[i]
            if abs(p) > thresh:
                if p > 0:
                    entry_price = mid[i] + spread[i] / 2.0
                    direction = 1
                else:
                    entry_price = mid[i] - spread[i] / 2.0
                    direction = -1
                in_pos = True
                entry_bar = i

    if in_pos:
        i = n - 1
        if direction == 1:
            unrealized = (mid[i] - entry_price) / TICK
        else:
            unrealized = (entry_price - mid[i]) / TICK
        pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl
        trades += 1
        if pnl > 0:
            wins += 1

    return total_pnl * TICK_VAL, trades, wins


# ============================================================================
# Strategy 2: Signal-Weaken Exit
# ============================================================================
def sim_signal_weaken(mid, spread, preds, thresh, decay_factor=0.3, max_hold=3000, cooldown=10):
    """
    Entry: |pred| > thresh
    Exit: |pred| drops below entry_strength * decay_factor OR max_hold
    """
    n = min(len(mid), len(preds))
    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0
    entry_strength = 0.0

    for i in range(n):
        if in_pos:
            bars = i - entry_bar
            current_strength = abs(preds[i])

            do_exit = False
            if current_strength < entry_strength * decay_factor:
                do_exit = True
            elif bars >= max_hold:
                do_exit = True

            if do_exit:
                if direction == 1:
                    unrealized = (mid[i] - entry_price) / TICK
                else:
                    unrealized = (entry_price - mid[i]) / TICK
                pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                in_pos = False
                last_exit = i

        elif i - last_exit >= cooldown:
            p = preds[i]
            if abs(p) > thresh:
                if p > 0:
                    entry_price = mid[i] + spread[i] / 2.0
                    direction = 1
                else:
                    entry_price = mid[i] - spread[i] / 2.0
                    direction = -1
                in_pos = True
                entry_bar = i
                entry_strength = abs(p)

    if in_pos:
        i = n - 1
        if direction == 1:
            unrealized = (mid[i] - entry_price) / TICK
        else:
            unrealized = (entry_price - mid[i]) / TICK
        pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl
        trades += 1
        if pnl > 0:
            wins += 1

    return total_pnl * TICK_VAL, trades, wins


# ============================================================================
# Strategy 3: Dynamic Trailing Stop
# ============================================================================
def sim_dynamic_trail(mid, spread, preds, thresh, base_trail=8.0, max_hold=3000, cooldown=10):
    """
    Entry: |pred| > thresh
    Exit: adaptive trailing stop that TIGHTENS as signal weakens.
    trail = base_trail * (current_strength / entry_strength)
    When signal is strong: wide trail (let it run)
    When signal weakens: tight trail (protect profits)
    """
    n = min(len(mid), len(preds))
    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0
    entry_strength = 0.0
    peak = 0.0

    for i in range(n):
        if in_pos:
            if direction == 1:
                unrealized = (mid[i] - entry_price) / TICK
            else:
                unrealized = (entry_price - mid[i]) / TICK

            peak = max(peak, unrealized)
            bars = i - entry_bar

            # Adaptive trail: tightens as signal weakens
            current_strength = abs(preds[i])
            ratio = min(current_strength / max(entry_strength, 1e-10), 1.0)
            adaptive_trail = max(base_trail * ratio, 2.0)  # minimum 2 ticks

            do_exit = False
            if (peak - unrealized) >= adaptive_trail:
                do_exit = True
            elif bars >= max_hold:
                do_exit = True

            if do_exit:
                pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                in_pos = False
                last_exit = i

        elif i - last_exit >= cooldown:
            p = preds[i]
            if abs(p) > thresh:
                if p > 0:
                    entry_price = mid[i] + spread[i] / 2.0
                    direction = 1
                else:
                    entry_price = mid[i] - spread[i] / 2.0
                    direction = -1
                in_pos = True
                entry_bar = i
                entry_strength = abs(p)
                peak = 0.0

    if in_pos:
        i = n - 1
        if direction == 1:
            unrealized = (mid[i] - entry_price) / TICK
        else:
            unrealized = (entry_price - mid[i]) / TICK
        pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl
        trades += 1
        if pnl > 0:
            wins += 1

    return total_pnl * TICK_VAL, trades, wins


# ============================================================================
# Strategy 4: Multi-Signal Entry/Exit
# ============================================================================
def sim_multi_signal(mid, spread, entry_preds, exit_preds,
                     entry_thresh=1.0, exit_thresh=0.5, max_hold=3000, cooldown=10):
    """
    Entry: entry_signal > entry_thresh (e.g., cancel_asym for timing)
    Exit: exit_signal reverses OR weakens below exit_thresh (e.g., depth_ratio for confirmation)
    """
    n = min(len(mid), len(entry_preds), len(exit_preds))
    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0

    for i in range(n):
        if in_pos:
            bars = i - entry_bar

            # Exit when exit signal no longer confirms direction
            exit_sig = exit_preds[i]
            do_exit = False

            if direction == 1 and exit_sig < -exit_thresh:
                do_exit = True  # exit signal says sell now
            elif direction == -1 and exit_sig > exit_thresh:
                do_exit = True  # exit signal says buy now
            elif bars >= max_hold:
                do_exit = True

            if do_exit:
                if direction == 1:
                    unrealized = (mid[i] - entry_price) / TICK
                else:
                    unrealized = (entry_price - mid[i]) / TICK
                pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                in_pos = False
                last_exit = i

        elif i - last_exit >= cooldown:
            p = entry_preds[i]
            if abs(p) > entry_thresh:
                if p > 0:
                    entry_price = mid[i] + spread[i] / 2.0
                    direction = 1
                else:
                    entry_price = mid[i] - spread[i] / 2.0
                    direction = -1
                in_pos = True
                entry_bar = i

    if in_pos:
        i = n - 1
        if direction == 1:
            unrealized = (mid[i] - entry_price) / TICK
        else:
            unrealized = (entry_price - mid[i]) / TICK
        pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl
        trades += 1
        if pnl > 0:
            wins += 1

    return total_pnl * TICK_VAL, trades, wins


# ============================================================================
# Strategy 5: Regime-Filtered Trading
# ============================================================================
def sim_regime_filtered(mid, spread, preds, thresh, hold_bars=600, trail=8.0,
                        max_signals_first_hour=200, cooldown=10):
    """
    Same as static sim BUT only trades if signal density in first hour is below threshold.
    If too many signals fire early → skip the day (noisy regime).
    """
    n = min(len(mid), len(preds))

    # Count signals in first hour (36000 bars at 10Hz)
    first_hour = min(36000, n)
    signals_first_hour = sum(1 for i in range(first_hour) if abs(preds[i]) > thresh)

    if signals_first_hour > max_signals_first_hour:
        return 0.0, 0, 0  # skip day

    # Standard market order sim
    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0
    peak = 0.0

    for i in range(n):
        if in_pos:
            if direction == 1:
                unrealized = (mid[i] - entry_price) / TICK
            else:
                unrealized = (entry_price - mid[i]) / TICK
            peak = max(peak, unrealized)
            bars = i - entry_bar
            do_exit = bars >= hold_bars or (trail > 0 and (peak - unrealized) >= trail)
            if do_exit:
                pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                in_pos = False
                last_exit = i
        elif i - last_exit >= cooldown:
            if abs(preds[i]) > thresh:
                if preds[i] > 0:
                    entry_price = mid[i] + spread[i] / 2.0
                    direction = 1
                else:
                    entry_price = mid[i] - spread[i] / 2.0
                    direction = -1
                in_pos = True
                entry_bar = i
                peak = 0.0

    if in_pos:
        i = n - 1
        if direction == 1:
            unrealized = (mid[i] - entry_price) / TICK
        else:
            unrealized = (entry_price - mid[i]) / TICK
        pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl
        trades += 1
        if pnl > 0:
            wins += 1

    return total_pnl * TICK_VAL, trades, wins


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-days', type=int, default=50)
    args = parser.parse_args()

    log.info("Dynamic Exit Strategy Simulator")
    log.info(f"  Market orders: {TOTAL_COST_TICKS:.2f} ticks/RT cost")

    # Preload days
    log.info(f"\nPreloading {args.n_days} days...")
    t0 = time.time()
    days = preload_days(args.n_days)
    log.info(f"  Loaded {len(days)} days in {time.time()-t0:.1f}s")

    # Signals to test — ordered by IC strength
    signal_names = [
        'depth_ratio',               # IC=+0.0943
        'depth_ratio_z',             # IC=+0.0924
        'top5_ensemble',             # IC=+0.0896
        'order_frag',                # IC=+0.0862
        'novel_risk_adjusted_return',# IC=+0.0827 (39 days)
        'ask_orders',                # IC=+0.0819
        'book_imb_z',                # IC=+0.0717
        'cancel_asym_chain',         # IC=+0.0589
        'slow_decay_combo',          # generating, ~IC=0.08
        'cancel_asym',               # IC=+0.0356
        'volgated',                  # IC=+0.043
    ]

    # Load all signals
    signals = {}
    for sig in signal_names:
        sig_data = {}
        for date in days:
            s = load_signal(sig, date)
            if s is not None:
                sig_data[date] = s
        if sig_data:
            signals[sig] = sig_data
            log.info(f"  {sig}: {len(sig_data)} days")

    all_results = {}

    # ── Strategy 1: Signal-Flip Exit ──────────────────────────────────────
    log.info(f"\n{'='*80}")
    log.info("STRATEGY 1: Signal-Flip Exit")
    log.info(f"{'='*80}")

    thresholds = [0.3, 0.5, 1.0, 1.5, 2.0]
    max_holds = [600, 1800, 3000, 6000]  # 60s, 3min, 5min, 10min

    for sig_name, sig_data in signals.items():
        for thresh in thresholds:
            for max_hold in max_holds:
                key = f"flip_{sig_name}_t{thresh}_mh{max_hold}"
                day_pnls = []
                total_trades = 0
                total_wins = 0

                for date in sorted(sig_data.keys()):
                    pnl, tr, w = sim_signal_flip(
                        days[date]['mid'], days[date]['spread'],
                        sig_data[date], thresh, max_hold
                    )
                    day_pnls.append(pnl)
                    total_trades += tr
                    total_wins += w

                n = len(day_pnls)
                active = sum(1 for p, t in zip(day_pnls, [1]*n) if True)  # all days count
                pos = sum(1 for p in day_pnls if p > 0)
                total = sum(day_pnls)

                all_results[key] = {
                    'strategy': 'signal_flip',
                    'signal': sig_name,
                    'thresh': thresh,
                    'max_hold': max_hold,
                    'total_pnl': total,
                    'mean_daily': total / max(n, 1),
                    'trades': total_trades,
                    'trades_per_day': total_trades / max(n, 1),
                    'wr': total_wins / max(total_trades, 1),
                    'pos_days': pos,
                    'n_days': n,
                    'pct_pos': 100 * pos / max(n, 1),
                }

    # Print strategy 1 results
    flip_results = {k: v for k, v in all_results.items() if v['strategy'] == 'signal_flip'}
    sorted_flip = sorted(flip_results.items(), key=lambda x: x[1]['mean_daily'], reverse=True)
    log.info(f"\n  Top 20 signal-flip configs:")
    log.info(f"  {'signal':>25} {'thresh':>6} {'mh':>6} {'mean_pnl':>10} {'total':>10} {'wr':>6} {'tr/d':>6} {'%pos':>6}")
    for k, r in sorted_flip[:20]:
        log.info(f"  {r['signal']:>25} {r['thresh']:>6.1f} {r['max_hold']:>6} "
                f"{r['mean_daily']:>+10.2f} {r['total_pnl']:>+10.0f} "
                f"{r['wr']*100:>5.1f}% {r['trades_per_day']:>5.1f} {r['pct_pos']:>5.1f}%")

    # ── Strategy 2: Signal-Weaken Exit ────────────────────────────────────
    log.info(f"\n{'='*80}")
    log.info("STRATEGY 2: Signal-Weaken Exit")
    log.info(f"{'='*80}")

    decay_factors = [0.1, 0.3, 0.5, 0.7]

    for sig_name, sig_data in signals.items():
        for thresh in [0.5, 1.0, 1.5]:
            for decay in decay_factors:
                key = f"weaken_{sig_name}_t{thresh}_d{decay}"
                day_pnls = []
                total_trades = 0
                total_wins = 0

                for date in sorted(sig_data.keys()):
                    pnl, tr, w = sim_signal_weaken(
                        days[date]['mid'], days[date]['spread'],
                        sig_data[date], thresh, decay
                    )
                    day_pnls.append(pnl)
                    total_trades += tr
                    total_wins += w

                n = len(day_pnls)
                pos = sum(1 for p in day_pnls if p > 0)
                total = sum(day_pnls)

                all_results[key] = {
                    'strategy': 'signal_weaken',
                    'signal': sig_name,
                    'thresh': thresh,
                    'decay': decay,
                    'total_pnl': total,
                    'mean_daily': total / max(n, 1),
                    'trades': total_trades,
                    'trades_per_day': total_trades / max(n, 1),
                    'wr': total_wins / max(total_trades, 1),
                    'pos_days': pos,
                    'n_days': n,
                    'pct_pos': 100 * pos / max(n, 1),
                }

    weaken_results = {k: v for k, v in all_results.items() if v['strategy'] == 'signal_weaken'}
    sorted_weaken = sorted(weaken_results.items(), key=lambda x: x[1]['mean_daily'], reverse=True)
    log.info(f"\n  Top 20 signal-weaken configs:")
    log.info(f"  {'signal':>25} {'thresh':>6} {'decay':>6} {'mean_pnl':>10} {'total':>10} {'wr':>6} {'tr/d':>6} {'%pos':>6}")
    for k, r in sorted_weaken[:20]:
        log.info(f"  {r['signal']:>25} {r['thresh']:>6.1f} {r['decay']:>6.1f} "
                f"{r['mean_daily']:>+10.2f} {r['total_pnl']:>+10.0f} "
                f"{r['wr']*100:>5.1f}% {r['trades_per_day']:>5.1f} {r['pct_pos']:>5.1f}%")

    # ── Strategy 3: Dynamic Trailing ──────────────────────────────────────
    log.info(f"\n{'='*80}")
    log.info("STRATEGY 3: Dynamic Trailing Stop")
    log.info(f"{'='*80}")

    base_trails = [4, 8, 12, 16]

    for sig_name, sig_data in signals.items():
        for thresh in [0.5, 1.0, 1.5]:
            for bt in base_trails:
                key = f"dyntrail_{sig_name}_t{thresh}_bt{bt}"
                day_pnls = []
                total_trades = 0
                total_wins = 0

                for date in sorted(sig_data.keys()):
                    pnl, tr, w = sim_dynamic_trail(
                        days[date]['mid'], days[date]['spread'],
                        sig_data[date], thresh, bt
                    )
                    day_pnls.append(pnl)
                    total_trades += tr
                    total_wins += w

                n = len(day_pnls)
                pos = sum(1 for p in day_pnls if p > 0)
                total = sum(day_pnls)

                all_results[key] = {
                    'strategy': 'dynamic_trail',
                    'signal': sig_name,
                    'thresh': thresh,
                    'base_trail': bt,
                    'total_pnl': total,
                    'mean_daily': total / max(n, 1),
                    'trades': total_trades,
                    'trades_per_day': total_trades / max(n, 1),
                    'wr': total_wins / max(total_trades, 1),
                    'pos_days': pos,
                    'n_days': n,
                    'pct_pos': 100 * pos / max(n, 1),
                }

    dyn_results = {k: v for k, v in all_results.items() if v['strategy'] == 'dynamic_trail'}
    sorted_dyn = sorted(dyn_results.items(), key=lambda x: x[1]['mean_daily'], reverse=True)
    log.info(f"\n  Top 20 dynamic-trail configs:")
    log.info(f"  {'signal':>25} {'thresh':>6} {'bt':>4} {'mean_pnl':>10} {'total':>10} {'wr':>6} {'tr/d':>6} {'%pos':>6}")
    for k, r in sorted_dyn[:20]:
        log.info(f"  {r['signal']:>25} {r['thresh']:>6.1f} {r['base_trail']:>4} "
                f"{r['mean_daily']:>+10.2f} {r['total_pnl']:>+10.0f} "
                f"{r['wr']*100:>5.1f}% {r['trades_per_day']:>5.1f} {r['pct_pos']:>5.1f}%")

    # ── Strategy 4: Multi-Signal ──────────────────────────────────────────
    log.info(f"\n{'='*80}")
    log.info("STRATEGY 4: Multi-Signal Entry/Exit")
    log.info(f"{'='*80}")

    # Test combinations: leading signals for entry, confirming signals for exit
    multi_pairs = [
        # cancel_asym LEADS price — best for entry timing
        ('cancel_asym', 'depth_ratio'),
        ('cancel_asym', 'order_frag'),
        ('cancel_asym', 'top5_ensemble'),
        ('cancel_asym', 'volgated'),
        ('cancel_asym_chain', 'depth_ratio'),
        ('cancel_asym_chain', 'order_frag'),
        # depth_ratio has highest IC — test as entry too
        ('depth_ratio', 'cancel_asym'),
        ('depth_ratio', 'order_frag'),
        ('depth_ratio', 'ask_orders'),
        # order_frag is stable — test both ways
        ('order_frag', 'depth_ratio'),
        ('order_frag', 'cancel_asym'),
        # novel targets
        ('novel_risk_adjusted_return', 'depth_ratio'),
        ('novel_risk_adjusted_return', 'cancel_asym'),
    ]

    for entry_sig, exit_sig in multi_pairs:
        if entry_sig not in signals or exit_sig not in signals:
            continue
        for entry_thresh in [0.5, 1.0, 1.5]:
            for exit_thresh in [0.3, 0.5, 1.0]:
                key = f"multi_{entry_sig}>{exit_sig}_et{entry_thresh}_xt{exit_thresh}"
                day_pnls = []
                total_trades = 0
                total_wins = 0

                dates = sorted(set(signals[entry_sig].keys()) & set(signals[exit_sig].keys()) & set(days.keys()))
                for date in dates:
                    pnl, tr, w = sim_multi_signal(
                        days[date]['mid'], days[date]['spread'],
                        signals[entry_sig][date],
                        signals[exit_sig][date],
                        entry_thresh, exit_thresh
                    )
                    day_pnls.append(pnl)
                    total_trades += tr
                    total_wins += w

                n = len(day_pnls)
                pos = sum(1 for p in day_pnls if p > 0)
                total = sum(day_pnls)

                all_results[key] = {
                    'strategy': 'multi_signal',
                    'entry_signal': entry_sig,
                    'exit_signal': exit_sig,
                    'entry_thresh': entry_thresh,
                    'exit_thresh': exit_thresh,
                    'total_pnl': total,
                    'mean_daily': total / max(n, 1),
                    'trades': total_trades,
                    'trades_per_day': total_trades / max(n, 1),
                    'wr': total_wins / max(total_trades, 1),
                    'pos_days': pos,
                    'n_days': n,
                    'pct_pos': 100 * pos / max(n, 1),
                }

    multi_results = {k: v for k, v in all_results.items() if v['strategy'] == 'multi_signal'}
    sorted_multi = sorted(multi_results.items(), key=lambda x: x[1]['mean_daily'], reverse=True)
    log.info(f"\n  Top 20 multi-signal configs:")
    log.info(f"  {'entry':>15}→{'exit':>15} {'et':>4} {'xt':>4} {'mean_pnl':>10} {'total':>10} {'wr':>6} {'tr/d':>6} {'%pos':>6}")
    for k, r in sorted_multi[:20]:
        log.info(f"  {r['entry_signal']:>15}→{r['exit_signal']:>15} {r['entry_thresh']:>4.1f} "
                f"{r['exit_thresh']:>4.1f} {r['mean_daily']:>+10.2f} {r['total_pnl']:>+10.0f} "
                f"{r['wr']*100:>5.1f}% {r['trades_per_day']:>5.1f} {r['pct_pos']:>5.1f}%")

    # ── Strategy 5: Regime-Filtered ───────────────────────────────────────
    log.info(f"\n{'='*80}")
    log.info("STRATEGY 5: Regime-Filtered Trading")
    log.info(f"{'='*80}")

    max_sig_limits = [100, 200, 300, 500]

    for sig_name, sig_data in signals.items():
        for thresh in [0.5, 1.0, 1.5]:
            for hold in [600, 1200]:
                for trail in [8, 12]:
                    for max_sig in max_sig_limits:
                        key = f"regime_{sig_name}_t{thresh}_h{hold}_tr{trail}_ms{max_sig}"
                        day_pnls = []
                        total_trades = 0
                        total_wins = 0
                        skipped = 0

                        for date in sorted(sig_data.keys()):
                            pnl, tr, w = sim_regime_filtered(
                                days[date]['mid'], days[date]['spread'],
                                sig_data[date], thresh, hold, trail, max_sig
                            )
                            day_pnls.append(pnl)
                            total_trades += tr
                            total_wins += w
                            if tr == 0 and pnl == 0:
                                skipped += 1

                        n = len(day_pnls)
                        active = n - skipped
                        pos = sum(1 for p in day_pnls if p > 0)
                        total = sum(day_pnls)

                        all_results[key] = {
                            'strategy': 'regime_filter',
                            'signal': sig_name,
                            'thresh': thresh,
                            'hold': hold,
                            'trail': trail,
                            'max_signals_1h': max_sig,
                            'total_pnl': total,
                            'mean_daily': total / max(active, 1),
                            'trades': total_trades,
                            'trades_per_day': total_trades / max(active, 1),
                            'wr': total_wins / max(total_trades, 1),
                            'pos_days': pos,
                            'active_days': active,
                            'skipped_days': skipped,
                            'n_days': n,
                            'pct_pos': 100 * pos / max(active, 1),
                        }

    regime_results = {k: v for k, v in all_results.items() if v['strategy'] == 'regime_filter'}
    sorted_regime = sorted(regime_results.items(), key=lambda x: x[1]['mean_daily'], reverse=True)
    log.info(f"\n  Top 20 regime-filtered configs:")
    log.info(f"  {'signal':>25} {'thresh':>6} {'hold':>5} {'trail':>5} {'ms1h':>5} {'mean_pnl':>10} {'wr':>6} {'%pos':>6} {'skip':>5}")
    for k, r in sorted_regime[:20]:
        log.info(f"  {r['signal']:>25} {r['thresh']:>6.1f} {r['hold']:>5} {r['trail']:>5} "
                f"{r['max_signals_1h']:>5} {r['mean_daily']:>+10.2f} "
                f"{r['wr']*100:>5.1f}% {r['pct_pos']:>5.1f}% {r.get('skipped_days',0):>5}")

    # ── GRAND SUMMARY ─────────────────────────────────────────────────────
    log.info(f"\n\n{'='*80}")
    log.info("GRAND SUMMARY — All Strategies Ranked")
    log.info(f"{'='*80}")

    grand = sorted(all_results.items(), key=lambda x: x[1]['mean_daily'], reverse=True)
    log.info(f"\n  {'strategy':>15} {'signal':>25} {'mean_pnl':>10} {'total':>10} {'wr':>6} {'tr/d':>6} {'%pos':>6}")
    log.info("  " + "-" * 100)
    for k, r in grand[:40]:
        sig = r.get('signal', r.get('entry_signal', ''))
        log.info(f"  {r['strategy']:>15} {sig:>25} {r['mean_daily']:>+10.2f} "
                f"{r['total_pnl']:>+10.0f} {r['wr']*100:>5.1f}% "
                f"{r['trades_per_day']:>5.1f} {r['pct_pos']:>5.1f}%")

    # Save
    out = RESULTS_DIR / f"dynamic_exit_{_ts}.json"
    with open(str(out), 'w') as f:
        json.dump({
            'timestamp': _ts,
            'cost_ticks_rt': TOTAL_COST_TICKS,
            'n_days': len(days),
            'total_configs': len(all_results),
            'top_50': [{
                'key': k,
                **v,
            } for k, v in grand[:50]],
        }, f, indent=2)
    log.info(f"\nSaved: {out}")
    log.info(f"Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
