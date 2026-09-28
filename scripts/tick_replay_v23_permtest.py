#!/usr/bin/env python3
"""
Tick Replay v23 — Permutation Test on v22 Best Configs
=======================================================
HC #659: Every config must pass permutation test (random directions LOSE money).

v22 found promising configs on 21 dates but process was killed before
permutation test ran. This script runs JUST the permutation test on
the top 3 configs, plus per-day and long/short breakdowns.

Top configs from v22:
1. q5/tp12sl8  → +0.970 net/trade, Sharpe +1.5, 136 trades
2. q10/tp12sl8 → +0.838 net/trade, Sharpe +1.3, 294 trades
3. q5/tp8sl4   → +0.455 net/trade, Sharpe +1.1, 136 trades
"""

import numpy as np
import os
import sys
import json
import time
from pathlib import Path

PRED_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_tick_replay_preds'
MBO_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo'
OUTPUT = Path('/home/jupiter/Lvl3Quant/output/tick_replay_v23_permtest')
OUTPUT.mkdir(parents=True, exist_ok=True)

PRED_STRIDE = 250
N_PERMS = 200  # 200 permutations for tighter p-value

# Configs to test
TEST_CONFIGS = [
    {'head': 'composite_signal', 'quantile': 0.05, 'tp': 12, 'sl': 8, 'hold_s': 30, 'cancel_s': 10},
    {'head': 'composite_signal', 'quantile': 0.10, 'tp': 12, 'sl': 8, 'hold_s': 30, 'cancel_s': 10},
    {'head': 'composite_signal', 'quantile': 0.05, 'tp': 8, 'sl': 4, 'hold_s': 30, 'cancel_s': 10},
    {'head': 'composite_signal', 'quantile': 0.10, 'tp': 8, 'sl': 4, 'hold_s': 30, 'cancel_s': 10},
    {'head': 'composite_signal', 'quantile': 0.15, 'tp': 12, 'sl': 8, 'hold_s': 30, 'cancel_s': 10},
]

print(f"{'='*70}")
print(f"TICK REPLAY v23 — PERMUTATION TEST + DIAGNOSTICS")
print(f"{'='*70}")

# Load data
pred_files = sorted(os.listdir(PRED_DIR))
pred_dates = [f.replace('oot_', '').replace('.npz', '') for f in pred_files]
mbo_files = {f.replace('mbo_', '').replace('.npz', ''): os.path.join(MBO_DIR, f)
             for f in os.listdir(MBO_DIR) if f.endswith('.npz')}
common_dates = sorted([d for d in pred_dates if d in mbo_files])
print(f"Common dates: {len(common_dates)}")

all_data = {}
for date in common_dates:
    try:
        pred_data = np.load(os.path.join(PRED_DIR, f'oot_{date}.npz'))
        mbo_data = np.load(mbo_files[date])
        all_data[date] = {'preds': dict(pred_data), 'mbo': dict(mbo_data)}
    except Exception as e:
        print(f"  Skip {date}: {e}")
print(f"Loaded {len(all_data)} dates\n")


def run_config(data, cfg, randomize=False, seed=None):
    """Run a single config across all dates. Returns list of trades."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = None

    head = cfg['head']
    q = cfg['quantile']
    tp = cfg['tp']
    sl = cfg['sl']
    hold = cfg['hold_s']
    cancel = cfg['cancel_s']
    tick = 0.25

    trades = []

    for date, ddata in data.items():
        preds = ddata['preds']
        mbo = ddata['mbo']

        if head not in preds:
            continue

        signal = preds[head]
        abs_signal = np.abs(signal)
        threshold = np.quantile(abs_signal, 1 - q)
        if threshold <= 0:
            continue

        try:
            timestamps = mbo['ts_ns'] if 'ts_ns' in mbo else mbo['ts_event']
            prices = mbo['price']
            sizes = mbo['size']
            sides_arr = mbo['side']
        except KeyError:
            continue

        n_events = len(timestamps)
        n_preds = len(signal)

        for pi in range(n_preds):
            sig = signal[pi]
            if abs(sig) < threshold:
                continue

            if randomize:
                direction = 'long' if rng.random() > 0.5 else 'short'
            else:
                direction = 'long' if sig > 0 else 'short'

            event_idx = pi * PRED_STRIDE
            if event_idx >= n_events - 100:
                continue

            entry_time = timestamps[event_idx]

            # Find BBO
            bid_price = 0.0
            ask_price = 0.0
            for ei in range(max(0, event_idx - 200), event_idx):
                p = prices[ei]
                if p <= 0 or np.isnan(p):
                    continue
                s = sides_arr[ei]
                if s == 0:
                    bid_price = max(bid_price, p)
                elif s == 1:
                    ask_price = min(ask_price, p) if ask_price > 0 else p

            if bid_price <= 0 or ask_price <= 0 or ask_price <= bid_price:
                continue

            entry_price = bid_price if direction == 'long' else ask_price

            if direction == 'long':
                tp_price = entry_price + tp * tick
                sl_price = entry_price - sl * tick
            else:
                tp_price = entry_price - tp * tick
                sl_price = entry_price + sl * tick

            cancel_deadline = entry_time + cancel * 10**9

            # Fill sim
            filled = False
            fill_time = 0
            fill_idx = event_idx
            for ei in range(event_idx, min(event_idx + 5000, n_events)):
                t = timestamps[ei]
                p = prices[ei]
                if t > cancel_deadline:
                    break
                if direction == 'long' and p <= entry_price and sizes[ei] > 0:
                    filled = True; fill_time = t; fill_idx = ei; break
                elif direction == 'short' and p >= entry_price and sizes[ei] > 0:
                    filled = True; fill_time = t; fill_idx = ei; break

            if not filled:
                continue

            # Exit sim
            exit_deadline = fill_time + hold * 10**9
            exit_price = entry_price
            exit_reason = 'time_stop'
            last_p = entry_price
            mfe = 0.0
            mae = 0.0

            for ei in range(fill_idx + 1, min(fill_idx + 50000, n_events)):
                t = timestamps[ei]
                p = prices[ei]
                if p <= 0 or np.isnan(p):
                    continue
                last_p = p

                unrealized = ((p - entry_price) / tick) if direction == 'long' else ((entry_price - p) / tick)
                mfe = max(mfe, unrealized)
                mae = min(mae, unrealized)

                if direction == 'long' and p >= tp_price:
                    exit_price = tp_price; exit_reason = 'tp'; break
                elif direction == 'short' and p <= tp_price:
                    exit_price = tp_price; exit_reason = 'tp'; break

                if direction == 'long' and p <= sl_price:
                    exit_price = sl_price; exit_reason = 'sl'; break
                elif direction == 'short' and p >= sl_price:
                    exit_price = sl_price; exit_reason = 'sl'; break

                if t >= exit_deadline:
                    exit_price = p; exit_reason = 'time_stop'; break

            if exit_price <= 0:
                exit_price = last_p

            gross = ((exit_price - entry_price) / tick) if direction == 'long' else ((entry_price - exit_price) / tick)
            cost = 0.376 if exit_reason == 'tp' else 1.376
            net = gross - cost

            trades.append({
                'date': date,
                'direction': direction,
                'signal': float(sig),
                'gross': float(gross),
                'net': float(net),
                'exit_reason': exit_reason,
                'mfe': float(mfe),
                'mae': float(mae),
            })

    return trades


all_results = []

for cfg in TEST_CONFIGS:
    label = f"q{int(cfg['quantile']*100)}/tp{cfg['tp']}sl{cfg['sl']}"
    print(f"\n{'='*70}")
    print(f"CONFIG: {label}")
    print(f"{'='*70}")

    # Run real config
    trades = run_config(all_data, cfg, randomize=False)
    n = len(trades)
    if n < 5:
        print(f"  Only {n} trades — skipping")
        continue

    nets = np.array([t['net'] for t in trades])
    real_mean = float(np.mean(nets))
    real_total = float(np.sum(nets))
    wr = float(np.mean(nets > 0))
    sharpe = float(np.mean(nets) / np.std(nets) * np.sqrt(252)) if np.std(nets) > 0 else 0

    print(f"  Trades: {n} ({n/len(all_data):.1f}/day)")
    print(f"  Net/trade: {real_mean:+.3f} ticks")
    print(f"  Total net: {real_total:+.1f} ticks")
    print(f"  WR: {wr:.1%}")
    print(f"  Sharpe: {sharpe:+.1f}")

    # Long/short breakdown
    longs = [t for t in trades if t['direction'] == 'long']
    shorts = [t for t in trades if t['direction'] == 'short']
    l_net = np.mean([t['net'] for t in longs]) if longs else 0
    s_net = np.mean([t['net'] for t in shorts]) if shorts else 0
    print(f"  Long: {len(longs)} trades, net/trade={l_net:+.3f}")
    print(f"  Short: {len(shorts)} trades, net/trade={s_net:+.3f}")

    # Exit reason breakdown
    exits = {}
    for t in trades:
        exits[t['exit_reason']] = exits.get(t['exit_reason'], 0) + 1
    print(f"  Exits: {exits}")

    # MFE/MAE
    avg_mfe = np.mean([t['mfe'] for t in trades])
    avg_mae = np.mean([t['mae'] for t in trades])
    p90_mfe = np.percentile([t['mfe'] for t in trades], 90)
    print(f"  Avg MFE: {avg_mfe:.1f} ticks, p90 MFE: {p90_mfe:.1f} ticks")
    print(f"  Avg MAE: {avg_mae:.1f} ticks")

    # HC #428 R2 check: TP ≤ p90 of realized MFE
    if cfg['tp'] > p90_mfe:
        print(f"  ⚠️  HC #428 R2 WARNING: TP {cfg['tp']} > p90 MFE {p90_mfe:.1f} — target too aggressive")

    # Per-day breakdown
    from collections import defaultdict
    day_data = defaultdict(lambda: {'net': 0, 'trades': 0})
    for t in trades:
        day_data[t['date']]['net'] += t['net']
        day_data[t['date']]['trades'] += 1

    green = sum(1 for d in day_data.values() if d['net'] > 0)
    red = sum(1 for d in day_data.values() if d['net'] < 0)
    flat = sum(1 for d in day_data.values() if d['net'] == 0)
    print(f"  Days: {green} green, {red} red, {flat} flat (day WR: {green/(green+red)*100:.0f}%)")

    # Day-level Sharpe
    day_nets = [d['net'] for d in day_data.values()]
    day_sharpe = float(np.mean(day_nets) / np.std(day_nets) * np.sqrt(252)) if np.std(day_nets) > 0 else 0
    print(f"  Day-level Sharpe: {day_sharpe:+.2f}")

    # PERMUTATION TEST (HC #659 mandatory)
    print(f"\n  Running {N_PERMS} permutations...")
    perm_means = []
    t0 = time.time()
    for pi in range(N_PERMS):
        perm_trades = run_config(all_data, cfg, randomize=True, seed=pi*42+7)
        if len(perm_trades) > 0:
            perm_nets = [t['net'] for t in perm_trades]
            perm_means.append(float(np.mean(perm_nets)))
        if (pi+1) % 50 == 0:
            print(f"    {pi+1}/{N_PERMS} done ({time.time()-t0:.0f}s)")

    if perm_means:
        perm_mean = float(np.mean(perm_means))
        perm_std = float(np.std(perm_means))
        p_value = float(np.mean([p >= real_mean for p in perm_means]))

        print(f"\n  PERMUTATION RESULTS:")
        print(f"    Real net/trade:  {real_mean:+.3f}")
        print(f"    Perm mean:       {perm_mean:+.3f}")
        print(f"    Perm std:        {perm_std:.3f}")
        print(f"    p-value:         {p_value:.3f}")
        if perm_mean > 0:
            print(f"    ⚠️  RANDOM DIRECTIONS ALSO PROFITABLE ({perm_mean:+.3f}) — structural bias!")
        verdict = 'PASS' if p_value < 0.05 and perm_mean <= 0 else 'FAIL'
        print(f"    VERDICT:         {verdict}")
    else:
        p_value = 1.0
        perm_mean = 0
        verdict = 'FAIL'

    result = {
        'label': label,
        'config': cfg,
        'n_trades': n,
        'trades_per_day': round(n / len(all_data), 1),
        'net_per_trade': round(real_mean, 3),
        'total_net_ticks': round(real_total, 1),
        'win_rate': round(wr, 3),
        'sharpe': round(sharpe, 1),
        'day_sharpe': round(day_sharpe, 2),
        'long_trades': len(longs),
        'short_trades': len(shorts),
        'long_net': round(l_net, 3),
        'short_net': round(s_net, 3),
        'exits': exits,
        'avg_mfe': round(avg_mfe, 1),
        'p90_mfe': round(p90_mfe, 1),
        'avg_mae': round(avg_mae, 1),
        'green_days': green,
        'red_days': red,
        'perm_p_value': round(p_value, 3),
        'perm_mean': round(perm_mean, 3),
        'perm_verdict': verdict,
    }
    all_results.append(result)

# Summary
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
for r in all_results:
    status = '✅' if r['perm_verdict'] == 'PASS' else '❌'
    print(f"  {status} {r['label']:20s} net={r['net_per_trade']:+.3f} Sharpe={r['sharpe']:+.1f} "
          f"DaySharpe={r['day_sharpe']:+.2f} perm_p={r['perm_p_value']:.3f} → {r['perm_verdict']}")

# Save
with open(OUTPUT / 'v23_results.json', 'w') as f:
    json.dump({
        'version': 'v23_permtest',
        'n_dates': len(all_data),
        'n_perms': N_PERMS,
        'results': all_results,
    }, f, indent=2, default=float)

print(f"\nSaved to {OUTPUT}/v23_results.json")
print("Done.")
