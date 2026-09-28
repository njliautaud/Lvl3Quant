"""
Latency-Aware Market Order Simulator

Tests if profitable market order signals survive realistic execution latency.
Latency = delay between signal generation and order execution.

At 10 bars/second: 1 bar = 100ms latency.

Tests top 3 profitable configs:
1. cancel_asym at t=2.0 (fast scalp, 30s hold)
2. cancel_asym_chain at t=3.0-3.5 (slow momentum, 300s hold)
3. volgated at t=0.3-1.0 (vol-gated, 10s hold)

Usage:
    python alpha_discovery/latency_test.py --n-days 50
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

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
        logging.FileHandler(str(RESULTS_DIR / f"latency_test_{_ts}.log"), mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("latency")

TICK = 0.25
TICK_VAL = 12.50
COMM_TICKS = 3.00 / TICK_VAL  # 0.24
TOTAL_COST = 1.0 + COMM_TICKS  # 1.24

FEAT_CACHE = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
SIG_DIR = LVL3_ROOT / "data" / "processed" / "signal_predictions"


def preload_days(n_days):
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


def sim_day_latency(mid, spread, preds, thresh, hold_bars, trail, latency_bars, cooldown=10):
    """Market order sim with execution latency.

    When signal fires at bar i, we enter at bar i + latency_bars.
    Entry price uses mid/spread at the delayed bar.
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
    peak = 0.0
    pending_entry = None  # (bar_to_enter, direction)

    for i in range(n):
        # Check pending entry
        if pending_entry is not None and i >= pending_entry[0]:
            if not in_pos and (i - last_exit >= cooldown):
                edir = pending_entry[1]
                if edir == 1:
                    entry_price = mid[i] + spread[i] / 2.0
                else:
                    entry_price = mid[i] - spread[i] / 2.0
                direction = edir
                in_pos = True
                entry_bar = i
                peak = 0.0
            pending_entry = None

        if in_pos:
            if direction == 1:
                unrealized = (mid[i] - entry_price) / TICK
            else:
                unrealized = (entry_price - mid[i]) / TICK

            peak = max(peak, unrealized)
            bars = i - entry_bar

            do_exit = False
            if bars >= hold_bars:
                do_exit = True
            elif trail > 0 and (peak - unrealized) >= trail:
                do_exit = True

            if do_exit:
                exit_cost = spread[i] / 2.0 / TICK + COMM_TICKS
                pnl = unrealized - exit_cost
                total_pnl += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                in_pos = False
                last_exit = i

        elif not in_pos and pending_entry is None and (i - last_exit >= cooldown):
            p = preds[i]
            if abs(p) > thresh:
                if latency_bars == 0:
                    # Instant entry
                    if p > 0:
                        entry_price = mid[i] + spread[i] / 2.0
                        direction = 1
                    else:
                        entry_price = mid[i] - spread[i] / 2.0
                        direction = -1
                    in_pos = True
                    entry_bar = i
                    peak = 0.0
                else:
                    # Queue entry with delay
                    pending_entry = (i + latency_bars, 1 if p > 0 else -1)

    # Force close
    if in_pos:
        idx = n - 1
        if direction == 1:
            unrealized = (mid[idx] - entry_price) / TICK
        else:
            unrealized = (entry_price - mid[idx]) / TICK
        exit_cost = spread[idx] / 2.0 / TICK + COMM_TICKS
        pnl = unrealized - exit_cost
        total_pnl += pnl
        trades += 1
        if pnl > 0:
            wins += 1

    return total_pnl, trades, (wins / max(trades, 1))


def to_native(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-days', type=int, default=50)
    args = parser.parse_args()

    log.info("Latency-Aware Market Order Simulator")
    log.info("  Tests if profitable signals survive execution latency")
    log.info("  Latency values: 0, 1, 2, 5, 10 bars (0ms, 100ms, 200ms, 500ms, 1s)")
    log.info("")

    # Preload
    log.info("Preloading %d days...", args.n_days)
    t0 = time.time()
    days = preload_days(args.n_days)
    log.info("  Loaded %d days in %.1fs", len(days), time.time() - t0)

    # Define test configs
    LATENCY_BARS = [0, 1, 2, 5, 10, 20, 50]  # 0ms to 5s

    CONFIGS = [
        # (signal, thresh, hold_bars, trail, label)
        ("cancel_asym", 2.0, 300, 0, "cancel_asym t=2.0 30s no-trail"),
        ("cancel_asym", 2.0, 300, 4, "cancel_asym t=2.0 30s trail=4"),
        ("cancel_asym", 2.0, 600, 0, "cancel_asym t=2.0 60s no-trail"),
        ("cancel_asym", 2.5, 300, 0, "cancel_asym t=2.5 30s no-trail"),
        ("cancel_asym", 3.0, 300, 0, "cancel_asym t=3.0 30s no-trail"),
        ("cancel_asym_chain", 3.0, 3000, 0, "ca_chain t=3.0 300s no-trail"),
        ("cancel_asym_chain", 3.5, 3000, 0, "ca_chain t=3.5 300s no-trail"),
        ("cancel_asym_chain", 3.0, 1200, 0, "ca_chain t=3.0 120s no-trail"),
        ("cancel_asym_chain", 3.5, 1200, 0, "ca_chain t=3.5 120s no-trail"),
        ("volgated", 0.3, 100, 0, "volgated t=0.3 10s"),
        ("volgated", 0.5, 100, 0, "volgated t=0.5 10s"),
        ("volgated", 1.0, 100, 0, "volgated t=1.0 10s"),
    ]

    # Load signals
    signal_cache = {}
    for sig, _, _, _, _ in CONFIGS:
        if sig not in signal_cache:
            signal_cache[sig] = {}
            count = 0
            for date in days:
                preds = load_signal(sig, date)
                if preds is not None:
                    signal_cache[sig][date] = preds
                    count += 1
            log.info("  %s: %d days", sig, count)

    all_results = []

    for sig, thresh, hold, trail, label in CONFIGS:
        log.info("")
        log.info("=" * 70)
        log.info("CONFIG: %s", label)
        log.info("=" * 70)

        for lat in LATENCY_BARS:
            lat_ms = lat * 100  # 100ms per bar
            day_pnls = []
            total_trades = 0
            total_wins = 0

            for date in sorted(days.keys()):
                if date not in signal_cache.get(sig, {}):
                    continue
                mid = days[date]['mid']
                spread = days[date]['spread']
                preds = signal_cache[sig][date]

                pnl, trades, wr = sim_day_latency(
                    mid, spread, preds, thresh, hold, trail, lat
                )
                pnl_dollars = pnl * TICK_VAL
                day_pnls.append(pnl_dollars)
                total_trades += trades
                total_wins += int(trades * wr)

            if not day_pnls:
                continue

            mean_pnl = np.mean(day_pnls)
            total_pnl = np.sum(day_pnls)
            pos_days = sum(1 for p in day_pnls if p > 0)
            n_days = len(day_pnls)
            pct_pos = pos_days / n_days * 100
            wr = total_wins / max(total_trades, 1) * 100
            tr_per_day = total_trades / n_days

            log.info("  Latency=%4dms: PnL=%+8.1f/d  Total=%+8.0f  WR=%5.1f%%  Pos=%4.0f%%  Tr/d=%5.1f  Trades=%d",
                     lat_ms, mean_pnl, total_pnl, wr, pct_pos, tr_per_day, total_trades)

            all_results.append({
                'signal': sig,
                'thresh': thresh,
                'hold_bars': hold,
                'trail': trail,
                'label': label,
                'latency_bars': lat,
                'latency_ms': lat_ms,
                'mean_daily_pnl': to_native(mean_pnl),
                'total_pnl': to_native(total_pnl),
                'win_rate': to_native(wr / 100),
                'pct_pos_days': to_native(pct_pos),
                'trades_per_day': to_native(tr_per_day),
                'total_trades': total_trades,
                'n_days': n_days,
            })

    # Grand summary
    log.info("")
    log.info("=" * 80)
    log.info("LATENCY DECAY SUMMARY")
    log.info("=" * 80)

    for sig, thresh, hold, trail, label in CONFIGS:
        entries = [r for r in all_results if r['label'] == label]
        if not entries:
            continue
        log.info("")
        log.info("  %s:", label)
        baseline = entries[0]['mean_daily_pnl'] if entries else 0
        for r in entries:
            decay = 0 if baseline == 0 else (1 - r['mean_daily_pnl'] / baseline) * 100
            log.info("    %4dms: %+8.1f/d  (%+5.1f%% decay)  WR=%5.1f%%  Pos=%4.0f%%",
                     r['latency_ms'], r['mean_daily_pnl'], -decay if baseline > 0 else decay,
                     r['win_rate'] * 100, r['pct_pos_days'])

    # Save
    out_path = RESULTS_DIR / f"latency_test_{_ts}.json"
    with open(str(out_path), 'w') as f:
        json.dump({'configs': len(CONFIGS), 'latencies': LATENCY_BARS, 'results': all_results}, f, indent=2)
    log.info("")
    log.info("Saved: %s", out_path)
    log.info("Total time: %.0fs", time.time() - t0)


if __name__ == '__main__':
    main()
