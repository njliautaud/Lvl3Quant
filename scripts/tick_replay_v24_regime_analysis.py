#!/usr/bin/env python3
"""
Tick Replay v24 — Regime Stratification Analysis
=================================================
HC #428 R1: Must check regime gap for the winning LONG_q10_tp12sl8 config.
Stratify by SPY close-to-close return (green/red/flat).
"""

import numpy as np
import os
import json
import yfinance as yf
import pandas as pd
from pathlib import Path
from collections import defaultdict

PRED_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_tick_replay_preds'
MBO_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo'
OUTPUT = Path('/home/jupiter/Lvl3Quant/output/tick_replay_v24_longonly')

PRED_STRIDE = 250  # ms
TICK = 0.25

# Get common dates
pred_files = sorted(os.listdir(PRED_DIR))
pred_dates = [f.replace('oot_', '').replace('.npz', '') for f in pred_files]
mbo_files = {f.replace('mbo_', '').replace('.npz', ''): os.path.join(MBO_DIR, f)
             for f in os.listdir(MBO_DIR) if f.endswith('.npz')}
common_dates = sorted([d for d in pred_dates if d in mbo_files])

# Load all data
all_data = {}
for date in common_dates:
    try:
        pred_data = np.load(os.path.join(PRED_DIR, f'oot_{date}.npz'))
        mbo_data = np.load(mbo_files[date])
        all_data[date] = {'preds': dict(pred_data), 'mbo': dict(mbo_data)}
    except Exception as e:
        print(f"  Skip {date}: {e}")

print(f"Loaded {len(all_data)} dates\n")

# Get SPY returns for regime classification
date_strs = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in common_dates]
spy = yf.download("SPY", start="2025-12-20", end="2026-04-01", progress=False)
spy['return'] = spy['Close'].pct_change()
spy_returns = {}
for d in common_dates:
    dt_str = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    try:
        ret = spy.loc[dt_str, 'return']
        if hasattr(ret, 'item'):
            ret = ret.item()
        spy_returns[d] = float(ret)
    except:
        spy_returns[d] = 0.0

print("SPY returns per date:")
for d in common_dates:
    regime = "GREEN" if spy_returns.get(d, 0) > 0.001 else ("RED" if spy_returns.get(d, 0) < -0.001 else "FLAT")
    print(f"  {d}: SPY {spy_returns.get(d, 0)*100:+.2f}% [{regime}]")

# Run the winning config: LONG_q10_tp12sl8
HEAD = 'composite_signal'
Q = 0.10  # top 10%
TP = 12  # ticks
SL = 8   # ticks
HOLD = 60  # seconds
CANCEL = 10  # seconds

per_day_results = {}

for date, ddata in sorted(all_data.items()):
    preds = ddata['preds']
    mbo = ddata['mbo']

    if HEAD not in preds:
        print(f"  {date}: no {HEAD} in preds, skipping")
        continue

    signal = preds[HEAD]
    n_preds = len(signal)

    # Get MBO data
    if 'mid_prices' in mbo:
        mid = mbo['mid_prices']
        ts = mbo['timestamps'] if 'timestamps' in mbo else np.arange(len(mid))
    elif 'prices' in mbo:
        mid = mbo['prices']
        ts = mbo['timestamps'] if 'timestamps' in mbo else np.arange(len(mid))
    else:
        print(f"  {date}: no price data, skipping")
        continue

    # Compute threshold — top Q% of LONG signals (positive values)
    positive_signals = signal[signal > 0]
    if len(positive_signals) < 10:
        per_day_results[date] = {'trades': 0, 'net_ticks': 0}
        continue

    threshold = np.quantile(positive_signals, 1 - Q)

    # Find entry points (long only, signal above threshold)
    entries = []
    cooldown_until = -1

    for pi in range(n_preds):
        if pi <= cooldown_until:
            continue
        if signal[pi] > threshold:
            # Map prediction index to MBO index
            pred_time_ms = pi * PRED_STRIDE
            mbo_idx = int(pred_time_ms / 250) if 'timestamps' not in mbo else \
                      np.searchsorted(ts, pred_time_ms / 1000.0)

            if mbo_idx < len(mid) - 1:
                entries.append({
                    'pred_idx': pi,
                    'mbo_idx': mbo_idx,
                    'entry_price': mid[mbo_idx],
                    'signal_val': signal[pi],
                    'direction': 1  # long only
                })
                # Cooldown: skip next HOLD/STRIDE predictions
                cooldown_until = pi + max(1, int(HOLD * 1000 / PRED_STRIDE))

    # Simulate trades
    day_trades = []
    for entry in entries:
        entry_px = entry['entry_price']
        entry_idx = entry['mbo_idx']

        # Scan forward for TP/SL/timeout
        max_idx = min(len(mid), entry_idx + int(HOLD * 4000 / 250))  # rough
        cancel_idx = entry_idx + int(CANCEL * 4000 / 250)
        hold_idx = entry_idx + int(HOLD * 4000 / 250)

        exit_px = None
        exit_reason = None
        mfe = 0
        mae = 0

        for i in range(entry_idx + 1, min(len(mid), hold_idx + 1)):
            px = mid[i]
            move_ticks = (px - entry_px) / TICK  # long
            mfe = max(mfe, move_ticks)
            mae = min(mae, move_ticks)

            if move_ticks >= TP:
                exit_px = entry_px + TP * TICK
                exit_reason = 'tp'
                break
            elif move_ticks <= -SL:
                exit_px = entry_px - SL * TICK
                exit_reason = 'sl'
                break

        if exit_px is None:
            # Timeout — exit at current price
            timeout_idx = min(len(mid) - 1, hold_idx)
            exit_px = mid[timeout_idx]
            exit_reason = 'timeout'

        pnl_ticks = (exit_px - entry_px) / TICK
        # Apply costs: commission = 0.376 ticks RT
        net_ticks = pnl_ticks - 0.376

        day_trades.append({
            'net_ticks': net_ticks,
            'pnl_ticks': pnl_ticks,
            'mfe': mfe,
            'mae': mae,
            'exit_reason': exit_reason,
        })

    total_net = sum(t['net_ticks'] for t in day_trades)
    n_trades = len(day_trades)
    avg_net = total_net / n_trades if n_trades > 0 else 0
    wr = sum(1 for t in day_trades if t['net_ticks'] > 0) / n_trades if n_trades > 0 else 0

    per_day_results[date] = {
        'trades': n_trades,
        'net_ticks': total_net,
        'avg_net': avg_net,
        'wr': wr,
        'exits': {r: sum(1 for t in day_trades if t['exit_reason'] == r) for r in ['tp', 'sl', 'timeout']},
        'avg_mfe': np.mean([t['mfe'] for t in day_trades]) if day_trades else 0,
        'avg_mae': np.mean([t['mae'] for t in day_trades]) if day_trades else 0,
    }

# Print per-day results with regime
print(f"\n{'='*80}")
print(f"LONG_q10_tp12sl8 — PER-DAY REGIME ANALYSIS (21 dates)")
print(f"{'='*80}")
print(f"{'Date':<12} {'SPY':>7} {'Regime':<6} {'Trades':>6} {'Net':>8} {'Avg':>7} {'WR':>6} {'TP':>4} {'SL':>4} {'TO':>4}")
print(f"{'-'*80}")

green_pnl = []
red_pnl = []
flat_pnl = []
green_trades = 0
red_trades = 0

for d in common_dates:
    r = per_day_results.get(d, {'trades': 0, 'net_ticks': 0, 'avg_net': 0, 'wr': 0, 'exits': {}})
    spy_ret = spy_returns.get(d, 0)
    regime = "GREEN" if spy_ret > 0.001 else ("RED" if spy_ret < -0.001 else "FLAT")

    exits = r.get('exits', {})
    print(f"{d:<12} {spy_ret*100:>+6.2f}% {regime:<6} {r['trades']:>6} {r['net_ticks']:>+8.1f} {r['avg_net']:>+7.2f} {r['wr']:>5.0%} {exits.get('tp',0):>4} {exits.get('sl',0):>4} {exits.get('timeout',0):>4}")

    if r['trades'] > 0:
        if regime == "GREEN":
            green_pnl.append(r['net_ticks'])
            green_trades += r['trades']
        elif regime == "RED":
            red_pnl.append(r['net_ticks'])
            red_trades += r['trades']
        else:
            flat_pnl.append(r['net_ticks'])

print(f"\n{'='*80}")
print(f"REGIME STRATIFICATION")
print(f"{'='*80}")

def regime_stats(pnl_list, label, n_trades):
    if not pnl_list:
        print(f"  {label}: No days")
        return 0
    total = sum(pnl_list)
    avg_day = np.mean(pnl_list)
    std_day = np.std(pnl_list) if len(pnl_list) > 1 else 0
    sharpe = avg_day / std_day * np.sqrt(252) if std_day > 0 else 0
    wr = sum(1 for p in pnl_list if p > 0) / len(pnl_list)
    print(f"  {label}: {len(pnl_list)} days, {n_trades} trades, total {total:+.1f} ticks, avg/day {avg_day:+.1f}, Sharpe {sharpe:.2f}, WR {wr:.0%}")
    return sharpe

s_green = regime_stats(green_pnl, "GREEN", green_trades)
s_red = regime_stats(red_pnl, "RED  ", red_trades)
regime_stats(flat_pnl, "FLAT ", 0)

# HC #428 regime gap test
if s_green != 0 and s_red != 0:
    gap = abs(s_green - s_red) / max(abs(s_green), abs(s_red))
    verdict = "PASS" if gap < 0.50 else "FAIL"
    print(f"\n  HC #428 Regime Gap: {gap:.3f} (threshold <0.50) → {verdict}")
    print(f"  Green Sharpe: {s_green:.2f}, Red Sharpe: {s_red:.2f}")
elif s_green == 0 and s_red == 0:
    print(f"\n  HC #428 Regime Gap: N/A (insufficient data)")
else:
    print(f"\n  HC #428 Regime Gap: WARN — one regime has zero Sharpe")

# Overall summary
all_pnl = green_pnl + red_pnl + flat_pnl
total_trades = green_trades + red_trades
if all_pnl:
    total = sum(all_pnl)
    avg = np.mean(all_pnl)
    std = np.std(all_pnl) if len(all_pnl) > 1 else 0
    sharpe = avg / std * np.sqrt(252) if std > 0 else 0
    print(f"\n  OVERALL: {len(all_pnl)} days, {total_trades} trades, {total:+.1f} ticks total")
    print(f"  Daily avg: {avg:+.1f} ticks, Sharpe: {sharpe:.2f}")
    print(f"  At $12.50/tick: ${total * 12.50:+,.0f} total, ${avg * 12.50:+,.0f}/day avg")

# Save results
results = {
    'config': 'LONG_q10_tp12sl8',
    'n_dates': len(common_dates),
    'total_trades': total_trades,
    'per_day': {d: per_day_results.get(d, {}) for d in common_dates},
    'spy_returns': spy_returns,
    'regime': {
        'green_sharpe': s_green,
        'red_sharpe': s_red,
        'gap': abs(s_green - s_red) / max(abs(s_green), abs(s_red)) if s_green and s_red else None,
    }
}
with open(OUTPUT / 'v24_regime_analysis.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nSaved to {OUTPUT / 'v24_regime_analysis.json'}")
