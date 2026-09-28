#!/usr/bin/env python3
"""
Tick Replay v24 — Regime Overlay Analysis
==========================================
Runs the winning LONG_q10_tp12sl8 config and stratifies by SPY daily return.
Uses the same fill simulation as v24 (proper BBO from MBO events).
HC #428 R1: Must pass regime gap < 0.50.
"""

import numpy as np
import os
import json
import time
import yfinance as yf
import pandas as pd
from pathlib import Path
from collections import defaultdict

PRED_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_tick_replay_preds'
MBO_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo'
OUTPUT = Path('/home/jupiter/Lvl3Quant/output/tick_replay_v24_longonly')

PRED_STRIDE = 250  # ms between predictions
TICK = 0.25

# Get common dates
pred_files = sorted(os.listdir(PRED_DIR))
pred_dates = [f.replace('oot_', '').replace('.npz', '') for f in pred_files]
mbo_files = {f.replace('mbo_', '').replace('.npz', ''): os.path.join(MBO_DIR, f)
             for f in os.listdir(MBO_DIR) if f.endswith('.npz')}
common_dates = sorted([d for d in pred_dates if d in mbo_files])

print(f"Loading {len(common_dates)} dates...")
all_data = {}
for date in common_dates:
    try:
        pred_data = np.load(os.path.join(PRED_DIR, f'oot_{date}.npz'))
        mbo_data = np.load(mbo_files[date])
        all_data[date] = {'preds': dict(pred_data), 'mbo': dict(mbo_data)}
    except Exception as e:
        print(f"  Skip {date}: {e}")

print(f"Loaded {len(all_data)} dates")

# Get SPY returns for regime classification
print("Fetching SPY data...")
spy = yf.download("SPY", start="2025-12-20", end="2026-04-01", progress=False)
spy['return'] = spy['Close'].pct_change()
spy_returns = {}
for d in common_dates:
    dt_str = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    try:
        ret = spy.loc[dt_str, 'return']
        if hasattr(ret, 'item'):
            ret = ret.item()
        elif hasattr(ret, 'iloc'):
            ret = float(ret.iloc[0])
        spy_returns[d] = float(ret)
    except:
        spy_returns[d] = 0.0


def classify_regime(spy_ret):
    if spy_ret > 0.001:
        return "GREEN"
    elif spy_ret < -0.001:
        return "RED"
    return "FLAT"


def run_long_only_with_per_day(data, q=0.10, tp=12, sl=8, hold_s=30, cancel_s=10):
    """Run long-only config and return per-day trade breakdown."""
    head = 'composite_signal'
    per_day = {}

    for date, ddata in sorted(data.items()):
        preds = ddata['preds']
        mbo = ddata['mbo']

        if head not in preds:
            per_day[date] = {'trades': [], 'n': 0, 'total_net': 0}
            continue

        signal = preds[head]
        n_preds = len(signal)

        try:
            timestamps = mbo['ts_ns'] if 'ts_ns' in mbo else mbo['ts_event']
            prices = mbo['price']
            sizes = mbo['size']
            sides_arr = mbo['side']
        except KeyError:
            per_day[date] = {'trades': [], 'n': 0, 'total_net': 0}
            continue

        n_events = len(timestamps)

        # Threshold for long signals
        positive_signals = signal[signal > 0]
        if len(positive_signals) < 10:
            per_day[date] = {'trades': [], 'n': 0, 'total_net': 0}
            continue

        threshold = np.quantile(positive_signals, 1 - q)

        # Find candidates
        candidates = []
        for pi in range(n_preds):
            if signal[pi] > 0 and signal[pi] >= threshold:
                candidates.append(pi)

        # Execute trades with proper fill simulation
        day_trades = []
        for pi in candidates:
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

            entry_price = bid_price  # Long: enter at bid (limit order)
            tp_price = entry_price + tp * TICK
            sl_price = entry_price - sl * TICK
            cancel_deadline = entry_time + cancel_s * 10**9

            # Fill sim
            filled = False
            fill_time = 0
            fill_idx = event_idx
            for ei in range(event_idx, min(event_idx + 5000, n_events)):
                t = timestamps[ei]
                if t > cancel_deadline:
                    break
                p = prices[ei]
                if p <= entry_price and sizes[ei] > 0:
                    filled = True
                    fill_time = t
                    fill_idx = ei
                    break

            if not filled:
                continue

            # Exit sim
            exit_deadline = fill_time + hold_s * 10**9
            exit_price = entry_price
            exit_reason = 'time_stop'
            mfe = 0.0
            mae = 0.0
            last_p = entry_price

            for ei in range(fill_idx + 1, min(fill_idx + 50000, n_events)):
                t = timestamps[ei]
                p = prices[ei]
                if p <= 0 or np.isnan(p):
                    continue
                last_p = p

                unrealized = (p - entry_price) / TICK
                mfe = max(mfe, unrealized)
                mae = min(mae, unrealized)

                if p >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break
                if p <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                if t >= exit_deadline:
                    exit_price = p
                    exit_reason = 'time_stop'
                    break

            if exit_price <= 0:
                exit_price = last_p

            gross = (exit_price - entry_price) / TICK
            cost = 0.376 if exit_reason == 'tp' else 1.376
            net = gross - cost

            day_trades.append({
                'net': net,
                'gross': gross,
                'exit_reason': exit_reason,
                'mfe': mfe,
                'mae': mae,
            })

        total_net = sum(t['net'] for t in day_trades)
        per_day[date] = {
            'trades': day_trades,
            'n': len(day_trades),
            'total_net': total_net,
            'avg_net': total_net / len(day_trades) if day_trades else 0,
            'wr': sum(1 for t in day_trades if t['net'] > 0) / len(day_trades) if day_trades else 0,
        }

    return per_day


print("\nRunning LONG_q10_tp12sl8...")
t0 = time.time()
per_day = run_long_only_with_per_day(all_data, q=0.10, tp=12, sl=8, hold_s=30, cancel_s=10)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s\n")

# Regime analysis
print(f"{'='*80}")
print(f"LONG_q10_tp12sl8 — PER-DAY REGIME ANALYSIS")
print(f"{'='*80}")
print(f"{'Date':<12} {'SPY':>7} {'Regime':<6} {'Trades':>6} {'Net':>8} {'Avg':>7} {'WR':>6}")
print(f"{'-'*60}")

green_days = []
red_days = []
flat_days = []

for d in common_dates:
    r = per_day.get(d, {'n': 0, 'total_net': 0, 'avg_net': 0, 'wr': 0})
    spy_ret = spy_returns.get(d, 0)
    regime = classify_regime(spy_ret)

    if r['n'] > 0:
        print(f"{d:<12} {spy_ret*100:>+6.2f}% {regime:<6} {r['n']:>6} {r['total_net']:>+8.1f} {r['avg_net']:>+7.2f} {r['wr']:>5.0%}")
    else:
        print(f"{d:<12} {spy_ret*100:>+6.2f}% {regime:<6} {r['n']:>6}     ---     ---   ---")

    if r['n'] > 0:
        entry = {'date': d, 'net': r['total_net'], 'n_trades': r['n'], 'avg': r['avg_net']}
        if regime == "GREEN":
            green_days.append(entry)
        elif regime == "RED":
            red_days.append(entry)
        else:
            flat_days.append(entry)

# Stratified analysis
print(f"\n{'='*80}")
print(f"REGIME STRATIFICATION (HC #428 R1)")
print(f"{'='*80}")

def regime_sharpe(days, label):
    if not days:
        print(f"  {label}: No trading days")
        return 0, 0
    nets = [d['net'] for d in days]
    total = sum(nets)
    avg = np.mean(nets)
    std = np.std(nets) if len(nets) > 1 else 1
    sharpe = avg / std * np.sqrt(252) if std > 0 else 0
    wr = sum(1 for n in nets if n > 0) / len(nets)
    total_trades = sum(d['n_trades'] for d in days)
    avg_per_trade = total / total_trades if total_trades > 0 else 0
    print(f"  {label}: {len(days)} days, {total_trades} trades")
    print(f"    Total: {total:+.1f} ticks, Avg/day: {avg:+.1f}, Avg/trade: {avg_per_trade:+.2f}")
    print(f"    Day WR: {wr:.0%}, Sharpe: {sharpe:+.2f}")
    return sharpe, len(days)

s_green, n_green = regime_sharpe(green_days, "GREEN (SPY up)")
s_red, n_red = regime_sharpe(red_days, "RED   (SPY down)")
s_flat, n_flat = regime_sharpe(flat_days, "FLAT  (SPY flat)")

# HC #428 regime gap
print(f"\n  --- HC #428 REGIME GAP TEST ---")
if n_green > 0 and n_red > 0:
    denom = max(abs(s_green), abs(s_red))
    if denom > 0:
        gap = abs(s_green - s_red) / denom
        verdict = "✅ PASS" if gap < 0.50 else "❌ FAIL"
        print(f"  Green Sharpe: {s_green:+.2f}")
        print(f"  Red Sharpe:   {s_red:+.2f}")
        print(f"  Gap: {gap:.3f} (threshold < 0.50) → {verdict}")
    else:
        print(f"  Cannot compute (zero denominator)")
else:
    print(f"  Insufficient regime coverage (need both green and red days with trades)")

# Overall
all_days = green_days + red_days + flat_days
if all_days:
    total_trades = sum(d['n_trades'] for d in all_days)
    total_net = sum(d['net'] for d in all_days)
    avg_day = np.mean([d['net'] for d in all_days])
    std_day = np.std([d['net'] for d in all_days]) if len(all_days) > 1 else 1
    sharpe = avg_day / std_day * np.sqrt(252) if std_day > 0 else 0
    day_wr = sum(1 for d in all_days if d['net'] > 0) / len(all_days)

    print(f"\n  --- OVERALL ---")
    print(f"  {len(all_days)} days, {total_trades} trades")
    print(f"  Total: {total_net:+.1f} ticks (${total_net * 12.50:+,.0f})")
    print(f"  Avg/day: {avg_day:+.1f} ticks (${avg_day * 12.50:+,.0f})")
    print(f"  Day WR: {day_wr:.0%}, Sharpe: {sharpe:+.2f}")
    print(f"  Avg/trade: {total_net/total_trades:+.3f} ticks (${total_net/total_trades * 12.50:+,.2f})")

# Save
results = {
    'config': 'LONG_q10_tp12sl8',
    'n_dates': len(common_dates),
    'green': {'sharpe': s_green, 'n_days': n_green, 'days': green_days},
    'red': {'sharpe': s_red, 'n_days': n_red, 'days': red_days},
    'flat': {'sharpe': s_flat, 'n_days': n_flat, 'days': flat_days},
    'spy_returns': spy_returns,
}
if n_green > 0 and n_red > 0 and max(abs(s_green), abs(s_red)) > 0:
    results['regime_gap'] = abs(s_green - s_red) / max(abs(s_green), abs(s_red))

with open(OUTPUT / 'v24_regime_overlay.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nSaved to {OUTPUT / 'v24_regime_overlay.json'}")
