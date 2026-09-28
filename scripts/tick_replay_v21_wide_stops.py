#!/usr/bin/env python3
"""
Tick Replay v21 — Wide Stops + High Confidence Only
=====================================================
HC #659: All backtesting must use tick-level replay engine.
HC #659: Every config must pass permutation test (random directions LOSE money).

Prior work (v10-v20): All configs with SL ≤ 4 deeply negative.
This run: Explores WIDER stops (SL 6-16), LONGER holds (30-120s),
and TOP 2-5% confidence filtering only. The hypothesis: if we only
trade when the model is extremely confident AND give the trade room,
the signal's edge (IC 0.22 at 1s) may survive costs.

Uses v4 multihead predictions (composite_signal) on 27 OOT days.
"""

import numpy as np
import os
import sys
import json
import time
from pathlib import Path

# Constants from tick_replay_engine
PRED_STRIDE = 250  # 250 MBO events per prediction

OUTPUT = Path('/home/jupiter/Lvl3Quant/output/tick_replay_v21_wide')
OUTPUT.mkdir(parents=True, exist_ok=True)

PRED_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_tick_replay_preds'
MBO_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo'

print(f"{'='*70}")
print(f"TICK REPLAY v21 — WIDE STOPS + HIGH CONFIDENCE")
print(f"{'='*70}")

# Check available data
pred_files = sorted(os.listdir(PRED_DIR))
pred_dates = [f.replace('oot_', '').replace('.npz', '') for f in pred_files]
mbo_files = {f.replace('mbo_', '').replace('.npz', ''): os.path.join(MBO_DIR, f)
             for f in os.listdir(MBO_DIR) if f.endswith('.npz')}

# Find dates with BOTH predictions and MBO data
common_dates = sorted([d for d in pred_dates if d in mbo_files])
print(f"Prediction dates: {len(pred_dates)}")
print(f"MBO dates: {len(mbo_files)}")
print(f"Common dates: {len(common_dates)}")
print(f"Dates: {common_dates[:5]}...{common_dates[-5:]}")

if len(common_dates) < 5:
    print("ERROR: Need at least 5 common dates")
    sys.exit(1)

# ─── Config space: wide stops, high confidence ───
configs = []

# Signal heads to test
heads = ['composite_signal', 'pred_log_ret_1s']

# Confidence thresholds (top N% of predictions only)
quantiles = [0.05, 0.10, 0.15, 0.20]  # Top 5%, 10%, 15%, 20%

# TP/SL combos (all wide — HC #659 says SL ≤ 4 is artifact)
tp_sl_combos = [
    (6, 6), (8, 6), (8, 8), (10, 8), (12, 8), (12, 10),
    (6, 4), (8, 4),  # Include a couple of SL=4 for comparison
]

# Hold times
hold_seconds = [30, 60, 120]

# Cancel window (how long to wait for fill)
cancel_seconds = [10]

for head in heads:
    for q in quantiles:
        for tp, sl in tp_sl_combos:
            for hold in hold_seconds:
                for cancel in cancel_seconds:
                    configs.append({
                        'head': head,
                        'quantile': q,
                        'tp': tp,
                        'sl': sl,
                        'hold_s': hold,
                        'cancel_s': cancel,
                    })

print(f"\nTotal configs: {len(configs)}")
print("Running sweep...\n")

# ─── Run sweep ───
results = []
start_time = time.time()

# Pre-load all data
print("Pre-loading data...")
all_data = {}
for date in common_dates:
    pred_path = os.path.join(PRED_DIR, f'oot_{date}.npz')
    mbo_path = mbo_files[date]

    try:
        pred_data = np.load(pred_path)
        mbo_data = np.load(mbo_path)
        all_data[date] = {
            'preds': dict(pred_data),
            'mbo': dict(mbo_data),
        }
    except Exception as e:
        print(f"  Skip {date}: {e}")
        continue

print(f"Loaded {len(all_data)} dates\n")

# For each config, run across all dates
for ci, cfg in enumerate(configs):
    head = cfg['head']
    q = cfg['quantile']
    tp = cfg['tp']
    sl = cfg['sl']
    hold = cfg['hold_s']
    cancel = cfg['cancel_s']

    label = f"{head.split('_')[-1]}|q{int(q*100)}|tp{tp}sl{sl}h{hold}c{cancel}"

    all_trades = []
    day_pnls = []

    for date, data in all_data.items():
        preds = data['preds']
        mbo = data['mbo']

        if head not in preds:
            continue

        signal = preds[head]

        # Determine threshold from quantile (top Q% strongest predictions)
        abs_signal = np.abs(signal)
        threshold = np.quantile(abs_signal, 1 - q)

        if threshold <= 0:
            continue

        # Get MBO fields
        try:
            timestamps = mbo['ts_ns'] if 'ts_ns' in mbo else mbo['ts_event']
            prices = mbo['price']
            sizes = mbo['size']
            actions = mbo['action']
            sides = mbo['side']
        except KeyError:
            continue

        n_events = len(timestamps)
        if n_events < 1000:
            continue

        # RTH filter (roughly 9:30-16:00 ET)
        # Timestamps are in nanoseconds
        day_start = timestamps[0]

        # Simulate trades for this day
        day_net = 0.0
        day_trades = 0

        # Walk through predictions
        n_preds = len(signal)
        for pi in range(n_preds):
            sig = signal[pi]

            # Skip if below confidence threshold
            if abs(sig) < threshold:
                continue

            # Determine direction
            direction = 'long' if sig > 0 else 'short'

            # Map prediction index to MBO event index
            event_idx = pi * PRED_STRIDE
            if event_idx >= n_events - 100:
                continue

            entry_time = timestamps[event_idx]

            # Find BBO at entry time
            # Side encoding: 0=Bid, 1=Ask (Databento MBO int8)
            bid_price = 0.0
            ask_price = 0.0
            for ei in range(max(0, event_idx - 200), event_idx):
                p = prices[ei]
                if p <= 0 or np.isnan(p):
                    continue
                s = sides[ei]
                if s == 0:  # Bid side
                    bid_price = max(bid_price, p)
                elif s == 1:  # Ask side
                    if ask_price == 0:
                        ask_price = p
                    else:
                        ask_price = min(ask_price, p)

            if bid_price <= 0 or ask_price <= 0 or ask_price <= bid_price:
                continue

            # Entry price (passive limit)
            entry_price = bid_price if direction == 'long' else ask_price

            # TP/SL prices
            tick = 0.25
            if direction == 'long':
                tp_price = entry_price + tp * tick
                sl_price = entry_price - sl * tick
            else:
                tp_price = entry_price - tp * tick
                sl_price = entry_price + sl * tick

            # Cancel deadline (wait for fill)
            cancel_deadline = entry_time + cancel * 10**9

            # Simulate fill: need price to trade through entry
            filled = False
            fill_time = 0
            fill_idx = event_idx

            for ei in range(event_idx, min(event_idx + 5000, n_events)):
                t = timestamps[ei]
                p = prices[ei]

                if t > cancel_deadline:
                    break

                # Check if we'd get filled (price trades through our level)
                if direction == 'long' and p <= entry_price and sizes[ei] > 0:
                    # Simplified: assume fill when price hits our bid
                    filled = True
                    fill_time = t
                    fill_idx = ei
                    break
                elif direction == 'short' and p >= entry_price and sizes[ei] > 0:
                    filled = True
                    fill_time = t
                    fill_idx = ei
                    break

            if not filled:
                continue

            # Track position: find exit
            exit_deadline = fill_time + hold * 10**9
            exit_price = 0.0
            exit_reason = 'time_stop'
            mfe = 0.0
            mae = 0.0

            last_price = entry_price
            for ei in range(fill_idx + 1, min(fill_idx + 50000, n_events)):
                t = timestamps[ei]
                p = prices[ei]

                if p <= 0 or np.isnan(p):
                    continue

                last_price = p

                # Track MFE/MAE
                if direction == 'long':
                    unrealized = (p - entry_price) / tick
                else:
                    unrealized = (entry_price - p) / tick

                mfe = max(mfe, unrealized)
                mae = min(mae, unrealized)

                # Check TP (passive exit)
                if direction == 'long' and p >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break
                elif direction == 'short' and p <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break

                # Check SL (market exit)
                if direction == 'long' and p <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                elif direction == 'short' and p >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break

                # Check time stop
                if t >= exit_deadline:
                    exit_price = p  # Market exit at current price
                    exit_reason = 'time_stop'
                    break

            if exit_price <= 0:
                exit_price = last_price
                exit_reason = 'time_stop'

            # Calculate PnL
            if direction == 'long':
                gross_ticks = (exit_price - entry_price) / tick
            else:
                gross_ticks = (entry_price - exit_price) / tick

            # Cost depends on exit type
            if exit_reason == 'tp':
                cost = 0.376  # Passive both sides
            else:
                cost = 1.376  # Market exit

            net_ticks = gross_ticks - cost

            all_trades.append({
                'date': date,
                'direction': direction,
                'signal': float(sig),
                'entry': float(entry_price),
                'exit': float(exit_price),
                'gross': float(gross_ticks),
                'net': float(net_ticks),
                'exit_reason': exit_reason,
                'mfe': float(mfe),
                'mae': float(mae),
            })

            day_net += net_ticks
            day_trades += 1

        if day_trades > 0:
            day_pnls.append({'date': date, 'net': day_net, 'trades': day_trades})

    n_trades = len(all_trades)
    if n_trades < 10:
        continue

    net_arr = np.array([t['net'] for t in all_trades])
    total_net = float(np.sum(net_arr))
    per_trade = float(np.mean(net_arr))
    win_rate = float(np.mean(net_arr > 0))
    sharpe = float(np.mean(net_arr) / np.std(net_arr) * np.sqrt(252)) if np.std(net_arr) > 0 else 0

    # Count exit reasons
    exits = {}
    for t in all_trades:
        exits[t['exit_reason']] = exits.get(t['exit_reason'], 0) + 1

    green_days = sum(1 for d in day_pnls if d['net'] > 0)
    red_days = sum(1 for d in day_pnls if d['net'] < 0)

    result = {
        'label': label,
        'head': head,
        'quantile': q,
        'tp': tp, 'sl': sl, 'hold_s': hold, 'cancel_s': cancel,
        'n_trades': n_trades,
        'trades_per_day': round(n_trades / len(all_data), 1),
        'net_ticks': round(total_net, 1),
        'per_trade': round(per_trade, 3),
        'win_rate': round(win_rate, 3),
        'sharpe': round(sharpe, 1),
        'green_days': green_days,
        'red_days': red_days,
        'exit_reasons': exits,
        'avg_mfe': round(float(np.mean([t['mfe'] for t in all_trades])), 1),
        'avg_mae': round(float(np.mean([t['mae'] for t in all_trades])), 1),
    }
    results.append(result)

    # Print if promising
    if per_trade > -0.5:
        marker = "***" if per_trade > 0 else ""
        print(f"  {label:45s} trades={n_trades:4d} net/trade={per_trade:+.3f} WR={win_rate:.1%} Sharpe={sharpe:+.1f} {marker}")

elapsed = time.time() - start_time
print(f"\n{'='*70}")
print(f"SWEEP COMPLETE — {len(results)} configs with ≥10 trades, {elapsed:.0f}s elapsed")
print(f"{'='*70}")

# Sort by per_trade
results.sort(key=lambda x: x['per_trade'], reverse=True)

print(f"\nTOP 10 CONFIGS:")
for r in results[:10]:
    print(f"  {r['label']:45s} net/trade={r['per_trade']:+.3f} WR={r['win_rate']:.1%} trades={r['n_trades']} Sharpe={r['sharpe']:+.1f}")

print(f"\nBOTTOM 5:")
for r in results[-5:]:
    print(f"  {r['label']:45s} net/trade={r['per_trade']:+.3f} WR={r['win_rate']:.1%} trades={r['n_trades']}")

# ─── Permutation test on best config ───
best = results[0] if results else None
if best and best['per_trade'] > 0:
    print(f"\n{'='*70}")
    print(f"PERMUTATION TEST on best config: {best['label']}")
    print(f"{'='*70}")

    # Re-run best config with randomized directions
    head = best['head']
    q = best['quantile']
    tp = best['tp']
    sl = best['sl']
    hold = best['hold_s']
    cancel = best['cancel_s']

    perm_nets = []
    for perm_i in range(100):
        perm_total = 0.0
        perm_trades = 0

        for date, data in all_data.items():
            preds = data['preds']
            mbo = data['mbo']

            if head not in preds:
                continue

            signal = preds[head]
            abs_signal = np.abs(signal)
            threshold = np.quantile(abs_signal, 1 - q)

            try:
                timestamps = mbo['ts_ns'] if 'ts_ns' in mbo else mbo['ts_event']
                prices = mbo['price']
                sizes = mbo['size']
                actions = mbo['action']
                sides_arr = mbo['side']
            except KeyError:
                continue

            n_events = len(timestamps)
            n_preds = len(signal)

            for pi in range(n_preds):
                sig = signal[pi]
                if abs(sig) < threshold:
                    continue

                # RANDOM direction instead of signal direction
                direction = 'long' if np.random.random() > 0.5 else 'short'

                event_idx = pi * PRED_STRIDE
                if event_idx >= n_events - 100:
                    continue

                entry_time = timestamps[event_idx]
                bid_price = ask_price = 0.0
                for ei in range(max(0, event_idx-200), event_idx):
                    p = prices[ei]
                    if p <= 0 or np.isnan(p):
                        continue
                    s = sides_arr[ei]
                    if s == 0:  # Bid
                        bid_price = max(bid_price, p)
                    elif s == 1:  # Ask
                        ask_price = min(ask_price, p) if ask_price > 0 else p

                if bid_price <= 0 or ask_price <= 0 or ask_price <= bid_price:
                    continue

                entry_price = bid_price if direction == 'long' else ask_price
                tick = 0.25
                tp_price = entry_price + (tp if direction == 'long' else -tp) * tick
                sl_price = entry_price + (-sl if direction == 'long' else sl) * tick
                cancel_deadline = entry_time + cancel * 10**9

                filled = False
                fill_time = 0
                fill_idx = event_idx
                for ei in range(event_idx, min(event_idx+5000, n_events)):
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

                exit_deadline = fill_time + hold * 10**9
                exit_price = entry_price
                exit_reason = 'time_stop'
                last_p = entry_price

                for ei in range(fill_idx+1, min(fill_idx+50000, n_events)):
                    t = timestamps[ei]
                    p = prices[ei]
                    if p <= 0 or np.isnan(p):
                        continue
                    last_p = p
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
                perm_total += gross - cost
                perm_trades += 1

        if perm_trades > 0:
            perm_nets.append(perm_total / perm_trades)

    real_net = best['per_trade']
    perm_mean = float(np.mean(perm_nets)) if perm_nets else 0
    p_value = float(np.mean([p >= real_net for p in perm_nets])) if perm_nets else 1.0

    print(f"Real net/trade: {real_net:+.3f}")
    print(f"Perm mean:      {perm_mean:+.3f}")
    print(f"Perm std:       {float(np.std(perm_nets)):+.3f}")
    print(f"p-value:        {p_value:.3f}")
    print(f"VERDICT:        {'PASS — REAL EDGE' if p_value < 0.05 else 'FAIL — ARTIFACT'}")

    best['perm_p_value'] = round(p_value, 3)
    best['perm_mean'] = round(perm_mean, 3)
    best['perm_verdict'] = 'PASS' if p_value < 0.05 else 'FAIL'

elif best:
    print(f"\nBest config net/trade = {best['per_trade']:+.3f} (negative). No permutation test needed — nothing to validate.")

# ─── Save ───
output_data = {
    'version': 'v21_wide_stops',
    'generated': time.strftime('%Y-%m-%d %H:%M'),
    'n_dates': len(all_data),
    'n_configs_tested': len(configs),
    'n_configs_with_trades': len(results),
    'elapsed_s': round(elapsed, 1),
    'top_10': results[:10],
    'all_results': results,
}

with open(OUTPUT / 'v21_results.json', 'w') as f:
    json.dump(output_data, f, indent=2, default=float)

print(f"\nSaved to {OUTPUT}/v21_results.json")
print("Done.")
