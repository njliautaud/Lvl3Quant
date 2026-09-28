"""
Test ALL raw microstructure signals with market order execution.
Memory-efficient: loads one signal at a time, processes all dates, then frees.

Tests each signal across 100 OOS days with first/last half split.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

TICK = 0.25
TICK_VAL = 12.50
COMM_TICKS = 3.00 / TICK_VAL
BARS_SEC = 10

LVL3_ROOT = Path(__file__).parent.parent
FEAT_CACHE = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'
SIG_DIR = LVL3_ROOT / 'data' / 'processed' / 'signal_predictions'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'

THRESHOLDS = [2.0, 3.0, 4.0, 5.0]
HOLDS = [100, 300, 600, 1200]  # 10s, 30s, 60s, 120s
TRAILS = [0, 8]


def load_mid_spread():
    """Load only mid and spread for all days (minimal memory)."""
    days = {}
    for f in sorted(FEAT_CACHE.glob('*_mbo_features.npz')):
        date = f.stem.replace('_mbo_features', '')
        data = np.load(str(f))
        feats = data['mbo_features']
        days[date] = {
            'mid': feats[:, 0].astype(np.float32).copy(),
            'spread': feats[:, 1].astype(np.float32).copy(),
        }
        del feats, data
    return days


def get_signal_names():
    """Parse all unique signal names from the predictions directory."""
    sig_dates = defaultdict(set)
    for f in sorted(SIG_DIR.glob('*.npz')):
        parts = f.stem.split('_')
        for i in range(len(parts)):
            if parts[i].startswith('2025-'):
                sig_name = '_'.join(parts[:i])
                date = '_'.join(parts[i:])
                sig_dates[sig_name].add(date)
                break
    return sig_dates


def load_signal(sig_name, date):
    """Load a single signal prediction file."""
    path = SIG_DIR / f'{sig_name}_{date}.npz'
    if path.exists():
        return np.load(str(path))['predictions']
    return None


def sim_day(mid, spread, preds, thresh, hold_bars, trail, cooldown=10):
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

    for i in range(n):
        if in_pos:
            if direction == 1:
                unr = (mid[i] - entry_price) / TICK
            else:
                unr = (entry_price - mid[i]) / TICK
            peak = max(peak, unr)
            if (i - entry_bar >= hold_bars) or (trail > 0 and (peak - unr) >= trail):
                pnl = unr - spread[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl
                trades += 1
                wins += (1 if pnl > 0 else 0)
                in_pos = False
                last_exit = i
        elif (i - last_exit >= cooldown) and abs(preds[i]) > thresh:
            d = 1 if preds[i] > 0 else -1
            entry_price = mid[i] + d * spread[i] / 2.0
            direction = d
            in_pos = True
            entry_bar = i
            peak = 0.0

    if in_pos:
        i = n - 1
        if direction == 1:
            unr = (mid[i] - entry_price) / TICK
        else:
            unr = (entry_price - mid[i]) / TICK
        pnl = unr - spread[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl
        trades += 1
        wins += (1 if pnl > 0 else 0)

    return total_pnl * TICK_VAL, trades, wins


def main():
    print("=" * 70)
    print("ALL SIGNALS Market Order Test — 100 OOS Days")
    print("=" * 70)
    print(f"Cost: {TICK_VAL * (1.0 + COMM_TICKS):.2f}/RT")

    # Load mid/spread data
    print("\nLoading mid/spread data...")
    t0 = time.time()
    days = load_mid_spread()
    all_dates = sorted(days.keys())
    mid_idx = len(all_dates) // 2
    print(f"Loaded {len(all_dates)} days in {time.time()-t0:.1f}s")

    # Get signal names
    sig_dates = get_signal_names()
    print(f"Found {len(sig_dates)} signal types")

    total_combos = len(THRESHOLDS) * len(HOLDS) * len(TRAILS)
    results_all = {}

    for sig_name in sorted(sig_dates.keys()):
        n_available = len(sig_dates[sig_name])
        if n_available < 50:
            continue

        # Load signal data for all matching dates
        sig_data = {}
        for date in all_dates:
            preds = load_signal(sig_name, date)
            if preds is not None:
                sig_data[date] = preds

        if len(sig_data) < 50:
            del sig_data
            continue

        matched_dates = sorted(sig_data.keys())
        mid = len(matched_dates) // 2

        best_pnl = -999999
        best_config = None
        best_h1 = best_h2 = best_trades = 0
        best_win = 0.0

        for thresh in THRESHOLDS:
            for hold in HOLDS:
                for trail in TRAILS:
                    h1_pnl = h2_pnl = 0
                    tot_tr = tot_w = 0
                    for i, date in enumerate(matched_dates):
                        p, t, w = sim_day(
                            days[date]['mid'], days[date]['spread'],
                            sig_data[date], thresh, hold, trail
                        )
                        if i < mid:
                            h1_pnl += p
                        else:
                            h2_pnl += p
                        tot_tr += t
                        tot_w += w

                    total = h1_pnl + h2_pnl
                    if total > best_pnl:
                        best_pnl = total
                        best_config = f"t{thresh}_h{hold//BARS_SEC}s_tr{trail}"
                        best_h1 = h1_pnl
                        best_h2 = h2_pnl
                        best_trades = tot_tr
                        best_win = tot_w / max(tot_tr, 1)

        both_pos = best_h1 > 0 and best_h2 > 0
        results_all[sig_name] = {
            'best_pnl': best_pnl, 'config': best_config,
            'h1': best_h1, 'h2': best_h2,
            'trades': best_trades, 'win': best_win,
            'both_positive': both_pos, 'n_days': len(sig_data),
        }

        flag = " *** BOTH +" if both_pos else ""
        print(f"  {sig_name:<30} best={best_config:<18} "
              f"PnL=${best_pnl:>+9,.0f}  H1=${best_h1:>+7,.0f} H2=${best_h2:>+7,.0f}  "
              f"trades={best_trades:>5}  win={best_win:.1%}{flag}")

        del sig_data

    # Summary
    sorted_sigs = sorted(results_all.items(), key=lambda x: x[1]['best_pnl'], reverse=True)

    print(f"\n{'='*70}")
    print("SUMMARY — ALL SIGNALS RANKED")
    print(f"{'='*70}")
    print(f"{'Signal':<30} {'Config':<20} {'Total':>10} {'H1':>8} {'H2':>8} {'Trades':>7} {'Win%':>6} {'Both':>5}")
    print("-" * 100)
    for name, r in sorted_sigs:
        flag = "YES" if r['both_positive'] else ""
        print(f"{name:<30} {r['config']:<20} ${r['best_pnl']:>+9,.0f} "
              f"${r['h1']:>+7,.0f} ${r['h2']:>+7,.0f} {r['trades']:>7} "
              f"{r['win']:>5.1%} {flag:>5}")

    both_positive = [(n, r) for n, r in sorted_sigs if r['both_positive']]
    print(f"\nSignals positive in BOTH halves: {len(both_positive)}/{len(sorted_sigs)}")
    for n, r in both_positive:
        print(f"  {n}: +${r['best_pnl']:,.0f} "
              f"(H1: +${r['h1']:,.0f}, H2: +${r['h2']:,.0f}, trades={r['trades']})")

    # Save
    ts = time.strftime('%Y%m%d_%H%M%S')
    out = RESULTS_DIR / f'all_signals_mktorder_{ts}.json'
    with open(out, 'w') as f:
        json.dump({'results': results_all, 'n_days': len(all_dates)}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
