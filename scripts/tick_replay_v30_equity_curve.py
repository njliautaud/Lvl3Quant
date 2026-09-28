#!/usr/bin/env python3
"""
Tick Replay v30 — Per-Day Equity Curve + Drawdown Analysis for Production Candidates
====================================================================================
Uses SAME sim engine as v28/v29, but outputs per-day and per-trade detail
for the TOP production candidates only (no permutation, those already passed).

Purpose: understand drawdown characteristics, max consecutive losses,
daily P&L distribution, and regime-stratified equity curves.
"""

import numpy as np
import os
import json
import time
from pathlib import Path
from collections import defaultdict

PRED_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_tick_replay_preds'
MBO_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo'
OUTPUT = Path('/home/jupiter/Lvl3Quant/output/tick_replay_v30_equity')
OUTPUT.mkdir(parents=True, exist_ok=True)

PRED_STRIDE = 250
TICK = 0.25
TICK_VALUE = 12.50  # $ per tick

# Only test production candidates that passed v29
CONFIGS = [
    {'q': 0.10, 'tp': 16, 'sl': 4, 'hold': 30, 'cancel': 10, 'label': 'champion_tp16sl4'},
    {'q': 0.10, 'tp': 16, 'sl': 6, 'hold': 30, 'cancel': 10, 'label': 'champion_tp16sl6'},
    {'q': 0.10, 'tp': 16, 'sl': 8, 'hold': 30, 'cancel': 10, 'label': 'champion_tp16sl8'},
]


def classify_date_regime(mbo_data):
    """Same classifier as v29 (fixed version)."""
    prices = mbo_data['price']
    valid = prices[prices > 0]
    if len(valid) < 100:
        return 'flat', 0
    n = len(valid)
    pct5 = max(50, n // 20)
    open_price = np.median(valid[:pct5])
    close_price = np.median(valid[-pct5:])
    change_ticks = (close_price - open_price) / TICK
    if change_ticks > 20:
        regime = 'green'
    elif change_ticks < -20:
        regime = 'red'
    else:
        regime = 'flat'
    return regime, float(change_ticks)


def run_long_only_detailed(data, head, q, tp, sl, hold_s, cancel_s):
    """Same sim as v28/v29 but returns detailed per-trade records."""
    trades = []

    for date, ddata in data.items():
        preds = ddata['preds']
        mbo = ddata['mbo']
        if head not in preds:
            continue

        signal = preds[head]
        n_preds = len(signal)

        try:
            timestamps = mbo['ts_ns'] if 'ts_ns' in mbo else mbo['ts_event']
            prices = mbo['price']
            sizes = mbo['size']
            sides_arr = mbo['side']
        except KeyError:
            continue

        n_events = len(timestamps)

        pos_signals = signal.copy()
        pos_signals[pos_signals <= 0] = 0
        if np.max(pos_signals) <= 0:
            continue
        pos_nonzero = pos_signals[pos_signals > 0]
        if len(pos_nonzero) == 0:
            continue
        threshold = np.quantile(pos_nonzero, 1 - q)

        candidates = [pi for pi in range(n_preds) if signal[pi] > 0 and signal[pi] >= threshold]
        if not candidates:
            continue

        for pi in candidates:
            event_idx = pi * PRED_STRIDE
            if event_idx >= n_events - 100:
                continue

            entry_time = timestamps[event_idx]

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

            entry_price = bid_price
            tp_price = entry_price + tp * TICK
            sl_price = entry_price - sl * TICK
            cancel_deadline = entry_time + cancel_s * 10**9

            filled = False
            fill_time = 0
            fill_idx = event_idx
            for ei in range(event_idx, min(event_idx + 5000, n_events)):
                t = timestamps[ei]
                if t > cancel_deadline:
                    break
                if prices[ei] <= entry_price and sizes[ei] > 0:
                    filled = True
                    fill_time = t
                    fill_idx = ei
                    break

            if not filled:
                continue

            exit_deadline = fill_time + hold_s * 10**9
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

            # Compute hold time in seconds
            exit_time = timestamps[min(ei, n_events-1)] if ei < n_events else fill_time + hold_s * 10**9
            hold_time_s = (exit_time - fill_time) / 1e9

            trades.append({
                'date': date,
                'entry_price': float(entry_price),
                'exit_price': float(exit_price),
                'gross_ticks': float(gross),
                'cost_ticks': float(cost),
                'net_ticks': float(net),
                'net_dollars': float(net * TICK_VALUE),
                'exit_reason': exit_reason,
                'mfe_ticks': float(mfe),
                'mae_ticks': float(mae),
                'hold_time_s': float(hold_time_s),
                'signal_strength': float(signal[pi]),
            })

    return trades


print(f"{'='*70}")
print(f"TICK REPLAY v30 — EQUITY CURVE + DRAWDOWN ANALYSIS")
print(f"{'='*70}\n")

# Load data
pred_files = sorted(os.listdir(PRED_DIR))
pred_dates = [f.replace('oot_', '').replace('.npz', '') for f in pred_files if f.startswith('oot_')]
mbo_files = {f.replace('mbo_', '').replace('.npz', ''): os.path.join(MBO_DIR, f)
             for f in os.listdir(MBO_DIR) if f.endswith('.npz')}
common_dates = sorted([d for d in pred_dates if d in mbo_files])

all_data = {}
for date in common_dates:
    try:
        pred_data = np.load(os.path.join(PRED_DIR, f'oot_{date}.npz'))
        mbo_data = np.load(mbo_files[date])
        n_preds = int(pred_data.get('n_preds', len(pred_data['composite_signal'])))
        if n_preds < 50:
            continue
        all_data[date] = {'preds': dict(pred_data), 'mbo': dict(mbo_data)}
    except:
        pass

print(f"Loaded {len(all_data)} dates\n")

# Classify regimes
date_info = {}
for date, ddata in all_data.items():
    regime, change = classify_date_regime(ddata['mbo'])
    date_info[date] = {'regime': regime, 'change_ticks': change}

# Run each config
for cfg in CONFIGS:
    label = cfg['label']
    print(f"\n{'='*60}")
    print(f"CONFIG: {label} (q={cfg['q']}, TP={cfg['tp']}, SL={cfg['sl']})")
    print(f"{'='*60}")

    trades = run_long_only_detailed(all_data, 'composite_signal', cfg['q'], cfg['tp'], cfg['sl'], cfg['hold'], cfg['cancel'])

    if not trades:
        print("  No trades")
        continue

    # Per-day aggregation
    day_pnl = defaultdict(lambda: {'trades': 0, 'net_ticks': 0, 'net_dollars': 0, 'wins': 0, 'losses': 0, 'tp': 0, 'sl': 0, 'time_stop': 0})
    for t in trades:
        d = day_pnl[t['date']]
        d['trades'] += 1
        d['net_ticks'] += t['net_ticks']
        d['net_dollars'] += t['net_dollars']
        if t['net_ticks'] > 0:
            d['wins'] += 1
        else:
            d['losses'] += 1
        d[t['exit_reason']] += 1

    # Equity curve
    sorted_dates = sorted(day_pnl.keys())
    cum_ticks = 0
    cum_dollars = 0
    peak_ticks = 0
    max_dd_ticks = 0
    max_dd_dollars = 0
    consecutive_red = 0
    max_consecutive_red = 0

    print(f"\n{'Date':<12} {'Regime':<6} {'Trades':>6} {'Net':>8} {'$Net':>8} {'Cum$':>10} {'DD$':>8} {'WR':>5} {'TP':>3} {'SL':>3} {'TO':>3}")
    print('-' * 90)

    daily_records = []
    for date in sorted_dates:
        d = day_pnl[date]
        regime = date_info.get(date, {}).get('regime', '?')
        cum_ticks += d['net_ticks']
        cum_dollars += d['net_dollars']
        peak_ticks = max(peak_ticks, cum_ticks)
        dd_ticks = cum_ticks - peak_ticks
        dd_dollars = dd_ticks * TICK_VALUE
        max_dd_ticks = min(max_dd_ticks, dd_ticks)
        max_dd_dollars = min(max_dd_dollars, dd_dollars)

        wr = d['wins'] / max(d['trades'], 1)

        if d['net_ticks'] < 0:
            consecutive_red += 1
            max_consecutive_red = max(max_consecutive_red, consecutive_red)
        else:
            consecutive_red = 0

        marker = '🟢' if d['net_ticks'] > 0 else '🔴' if d['net_ticks'] < 0 else '⚪'
        print(f"{date:<12} {regime:<6} {d['trades']:>6} {d['net_ticks']:>+8.1f} {d['net_dollars']:>+8.0f} {cum_dollars:>+10.0f} {dd_dollars:>+8.0f} {wr:>5.0%} {d['tp']:>3} {d['sl']:>3} {d['time_stop']:>3} {marker}")

        daily_records.append({
            'date': date,
            'regime': regime,
            'trades': d['trades'],
            'net_ticks': round(d['net_ticks'], 2),
            'net_dollars': round(d['net_dollars'], 2),
            'cum_dollars': round(cum_dollars, 2),
            'drawdown_dollars': round(dd_dollars, 2),
            'win_rate': round(wr, 3),
            'exits': {'tp': d['tp'], 'sl': d['sl'], 'time_stop': d['time_stop']},
        })

    # Summary statistics
    nets = np.array([t['net_ticks'] for t in trades])
    day_nets = np.array([day_pnl[d]['net_ticks'] for d in sorted_dates])
    day_dollars = np.array([day_pnl[d]['net_dollars'] for d in sorted_dates])

    # Daily Sharpe
    daily_sharpe = float(np.mean(day_nets) / np.std(day_nets) * np.sqrt(252)) if np.std(day_nets) > 0 else 0

    # Sortino (downside only)
    neg_days = day_nets[day_nets < 0]
    downside_std = np.std(neg_days) if len(neg_days) > 1 else np.std(day_nets)
    sortino = float(np.mean(day_nets) / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    # Profit factor
    gross_wins = float(np.sum(nets[nets > 0]))
    gross_losses = float(abs(np.sum(nets[nets < 0])))
    pf = gross_wins / max(gross_losses, 0.001)

    # Calmar
    annual_return = float(np.mean(day_dollars) * 252)
    calmar = annual_return / abs(max_dd_dollars) if max_dd_dollars < 0 else float('inf')

    # MFE/MAE analysis
    mfes = [t['mfe_ticks'] for t in trades]
    maes = [t['mae_ticks'] for t in trades]
    hold_times = [t['hold_time_s'] for t in trades]

    print(f"\n{'='*60}")
    print(f"SUMMARY — {label}")
    print(f"{'='*60}")
    print(f"  Total trades: {len(trades)} over {len(sorted_dates)} days ({len(trades)/max(len(sorted_dates),1):.1f}/day)")
    print(f"  Net/trade: {float(np.mean(nets)):+.3f} ticks (${float(np.mean(nets)*TICK_VALUE):+.2f})")
    print(f"  Total P&L: {float(np.sum(nets)):+.1f} ticks (${float(np.sum(nets)*TICK_VALUE):+.0f})")
    print(f"  Win rate: {float(np.mean(nets>0)):.1%}")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Daily Sharpe: {daily_sharpe:.2f}")
    print(f"  Daily Sortino: {sortino:.2f}")
    print(f"  Calmar: {calmar:.2f}")
    print(f"  Max drawdown: {max_dd_ticks:+.1f} ticks (${max_dd_dollars:+.0f})")
    print(f"  Max consecutive red days: {max_consecutive_red}")
    print(f"  Green days: {sum(1 for d in day_nets if d>0)} / {len(day_nets)} ({sum(1 for d in day_nets if d>0)/max(len(day_nets),1):.0%})")
    print(f"  Avg hold time: {np.mean(hold_times):.1f}s")
    print(f"  MFE: mean={np.mean(mfes):.1f}, median={np.median(mfes):.1f}, p90={np.percentile(mfes,90):.1f}")
    print(f"  MAE: mean={np.mean(maes):.1f}, median={np.median(maes):.1f}, p10={np.percentile(maes,10):.1f}")

    # Regime breakdown
    print(f"\n  Regime Breakdown:")
    for regime in ['green', 'red', 'flat']:
        regime_dates = [d for d in sorted_dates if date_info.get(d,{}).get('regime') == regime]
        if not regime_dates:
            continue
        r_nets = [day_pnl[d]['net_ticks'] for d in regime_dates]
        r_arr = np.array(r_nets)
        r_sharpe = float(np.mean(r_arr) / np.std(r_arr) * np.sqrt(252)) if len(r_arr) > 1 and np.std(r_arr) > 0 else 0
        print(f"    {regime:>5}: {len(regime_dates)} days, avg {np.mean(r_arr):+.1f} ticks/day, Sharpe {r_sharpe:.1f}, {sum(1 for r in r_arr if r>0)}/{len(r_arr)} green")

    # Save results
    summary = {
        'label': label,
        'params': cfg,
        'n_trades': len(trades),
        'n_days': len(sorted_dates),
        'trades_per_day': round(len(trades) / max(len(sorted_dates), 1), 1),
        'net_per_trade_ticks': round(float(np.mean(nets)), 4),
        'total_net_ticks': round(float(np.sum(nets)), 1),
        'total_net_dollars': round(float(np.sum(nets) * TICK_VALUE), 0),
        'win_rate': round(float(np.mean(nets > 0)), 3),
        'profit_factor': round(pf, 2),
        'daily_sharpe': round(daily_sharpe, 2),
        'daily_sortino': round(sortino, 2),
        'calmar': round(calmar, 2),
        'max_drawdown_ticks': round(max_dd_ticks, 1),
        'max_drawdown_dollars': round(max_dd_dollars, 0),
        'max_consecutive_red_days': max_consecutive_red,
        'daily_records': daily_records,
        'mfe_p90': round(float(np.percentile(mfes, 90)), 1),
        'mae_p10': round(float(np.percentile(maes, 10)), 1),
        'avg_hold_time_s': round(float(np.mean(hold_times)), 1),
    }

    with open(OUTPUT / f'{label}_detail.json', 'w') as f:
        json.dump(summary, f, indent=2)

print(f"\n\nResults saved to {OUTPUT}")
