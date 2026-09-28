#!/usr/bin/env python3
"""
Tick Replay v25 — Deep Diagnostic of LONG_q10_tp12sl8
=====================================================
The ONLY config that passed permutation in v24 (p=0.005).
This script does a deep dive:
  1. Per-trade breakdown with signal strength
  2. Date concentration (is edge from 1-2 outlier days?)
  3. Consecutive loss streak analysis
  4. Signal quintile analysis (does stronger signal = better trades?)
  5. Leave-one-out day stability test
  6. Subsample stability (first half vs second half of each day)
  7. Bootstrap confidence interval on net/trade
"""

import numpy as np
import os
import json
from pathlib import Path
from collections import defaultdict

PRED_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_tick_replay_preds'
MBO_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo'
OUTPUT = Path('/home/jupiter/Lvl3Quant/output/tick_replay_v25_diagnostic')
OUTPUT.mkdir(parents=True, exist_ok=True)

PRED_STRIDE = 250
TICK = 0.25

# Config that passed
CFG = {
    'head': 'composite_signal',
    'quantile': 0.10,  # top 10% of positive signals
    'tp': 12,
    'sl': 8,
    'hold_s': 120,
    'cancel_s': 15,
}

print("=" * 70)
print("TICK REPLAY v25 — DIAGNOSTIC: LONG_q10_tp12sl8")
print("=" * 70)

# Load data
pred_files = sorted(os.listdir(PRED_DIR))
pred_dates = [f.replace('oot_', '').replace('.npz', '') for f in pred_files if f.endswith('.npz')]
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


def run_single_config(data, cfg):
    """Run the exact passing config and return detailed per-trade data."""
    head = cfg['head']
    q = cfg['quantile']
    tp = cfg['tp']
    sl = cfg['sl']
    hold = cfg['hold_s']
    cancel = cfg['cancel_s']

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

        # Long-only: top q of positive signals
        pos_signals = signal.copy()
        pos_signals[pos_signals <= 0] = 0
        if np.max(pos_signals) <= 0:
            continue
        threshold = np.quantile(pos_signals[pos_signals > 0], 1 - q) if np.sum(pos_signals > 0) > 0 else float('inf')

        candidates = []
        for pi in range(n_preds):
            if signal[pi] > 0 and signal[pi] >= threshold:
                candidates.append(pi)

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

            entry_price = bid_price  # long only, enter at bid (passive)

            tp_price = entry_price + tp * TICK
            sl_price = entry_price - sl * TICK

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
                if p <= entry_price and sizes[ei] > 0:
                    filled = True
                    fill_time = t
                    fill_idx = ei
                    break

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

            # Relative time within day (fraction of trading session)
            day_start = timestamps[0]
            day_end = timestamps[-1]
            day_frac = (entry_time - day_start) / max(1, day_end - day_start)

            trades.append({
                'date': date,
                'signal_strength': float(signal[pi]),
                'signal_rank': float(signal[pi] / threshold) if threshold > 0 else 0,
                'gross': float(gross),
                'net': float(net),
                'exit_reason': exit_reason,
                'mfe': float(mfe),
                'mae': float(mae),
                'entry_price': float(entry_price),
                'day_fraction': float(day_frac),
                'pred_idx': int(pi),
            })

    return trades


# ========== RUN THE CONFIG ==========
print("Running LONG_q10_tp12sl8...")
trades = run_single_config(all_data, CFG)
print(f"Total trades: {len(trades)}\n")

if len(trades) < 10:
    print("INSUFFICIENT TRADES FOR DIAGNOSTIC")
    import sys; sys.exit(1)

nets = np.array([t['net'] for t in trades])
dates_list = [t['date'] for t in trades]

# ========== 1. BASIC STATS ==========
print("=" * 70)
print("1. BASIC STATS")
print("=" * 70)
print(f"  Trades: {len(trades)}")
print(f"  Net/trade: {np.mean(nets):+.4f} ticks")
print(f"  Total: {np.sum(nets):+.1f} ticks (${np.sum(nets)*12.50:+,.0f})")
print(f"  WR: {np.mean(nets > 0):.1%}")
print(f"  Sharpe (per-trade): {np.mean(nets)/np.std(nets)*np.sqrt(252):+.2f}")
print(f"  Median trade: {np.median(nets):+.4f} ticks")
print(f"  Std dev: {np.std(nets):.4f} ticks")
print(f"  Skew: {float(np.mean(((nets - np.mean(nets))/np.std(nets))**3)):.2f}")
exits = {}
for t in trades:
    exits[t['exit_reason']] = exits.get(t['exit_reason'], 0) + 1
print(f"  Exits: {exits}")

# ========== 2. PER-DATE BREAKDOWN ==========
print(f"\n{'='*70}")
print("2. PER-DATE BREAKDOWN (checking for outlier concentration)")
print("=" * 70)
day_data = defaultdict(lambda: {'net': 0, 'trades': 0, 'nets': []})
for t in trades:
    day_data[t['date']]['net'] += t['net']
    day_data[t['date']]['trades'] += 1
    day_data[t['date']]['nets'].append(t['net'])

print(f"  {'Date':<12} {'Trades':>7} {'Net/Trade':>10} {'Total':>10} {'WR':>7}")
print(f"  {'-'*50}")
for date in sorted(day_data.keys()):
    d = day_data[date]
    wr = np.mean(np.array(d['nets']) > 0)
    print(f"  {date:<12} {d['trades']:>7} {d['net']/d['trades']:>+10.3f} {d['net']:>+10.1f} {wr:>7.1%}")

day_nets = [d['net'] for d in day_data.values()]
n_days = len(day_nets)
green = sum(1 for n in day_nets if n > 0)
red = sum(1 for n in day_nets if n < 0)
day_sharpe = np.mean(day_nets) / np.std(day_nets) * np.sqrt(252) if np.std(day_nets) > 0 else 0
print(f"\n  Day stats: {green}G/{red}R ({n_days} days), Day Sharpe: {day_sharpe:+.2f}")

# Date concentration: what % of total P&L comes from top 2 days?
sorted_day_nets = sorted(day_nets, reverse=True)
total = sum(day_nets)
top2 = sum(sorted_day_nets[:2])
print(f"  Top 2 days: {top2:+.1f} ticks ({top2/total*100:.0f}% of total)")
top3 = sum(sorted_day_nets[:3])
print(f"  Top 3 days: {top3:+.1f} ticks ({top3/total*100:.0f}% of total)")

# ========== 3. LEAVE-ONE-OUT STABILITY ==========
print(f"\n{'='*70}")
print("3. LEAVE-ONE-OUT STABILITY (does edge survive dropping any day?)")
print("=" * 70)
loo_results = []
for drop_date in sorted(day_data.keys()):
    remaining_nets = [t['net'] for t in trades if t['date'] != drop_date]
    if len(remaining_nets) < 5:
        continue
    loo_mean = np.mean(remaining_nets)
    loo_results.append({'drop': drop_date, 'mean': loo_mean, 'n': len(remaining_nets)})
    status = "✅" if loo_mean > 0 else "❌ FRAGILE"
    print(f"  Drop {drop_date}: net/trade = {loo_mean:+.4f} ({len(remaining_nets)} trades) {status}")

loo_means = [r['mean'] for r in loo_results]
n_positive = sum(1 for m in loo_means if m > 0)
print(f"\n  Stability: {n_positive}/{len(loo_means)} LOO variants profitable")
if n_positive < len(loo_means):
    fragile_dates = [r['drop'] for r in loo_results if r['mean'] <= 0]
    print(f"  ⚠️ FRAGILE: removing {fragile_dates} kills profitability")

# ========== 4. SIGNAL STRENGTH VS PERFORMANCE ==========
print(f"\n{'='*70}")
print("4. SIGNAL STRENGTH ANALYSIS (does stronger signal = better trades?)")
print("=" * 70)
signals = np.array([t['signal_strength'] for t in trades])
signal_ranks = np.array([t['signal_rank'] for t in trades])

# Split into tertiles by signal strength
n_tert = len(trades) // 3
sorted_idx = np.argsort(signals)
for i, label in enumerate(['Bottom 1/3', 'Middle 1/3', 'Top 1/3']):
    if i < 2:
        idx = sorted_idx[i*n_tert:(i+1)*n_tert]
    else:
        idx = sorted_idx[i*n_tert:]
    tert_nets = nets[idx]
    print(f"  {label}: net/trade={np.mean(tert_nets):+.4f}, WR={np.mean(tert_nets>0):.1%}, "
          f"signal=[{signals[idx].min():.4f}, {signals[idx].max():.4f}]")

# Correlation between signal strength and trade outcome
corr = np.corrcoef(signals, nets)[0, 1]
print(f"\n  Signal-outcome correlation: {corr:+.3f}")
if abs(corr) < 0.05:
    print(f"  ⚠️ WEAK: Signal magnitude barely predicts trade quality")

# ========== 5. CONSECUTIVE LOSS STREAKS ==========
print(f"\n{'='*70}")
print("5. CONSECUTIVE LOSS STREAKS")
print("=" * 70)
streak = 0
max_streak = 0
max_dd = 0
running_pnl = 0
peak_pnl = 0
streaks = []
for n in nets:
    running_pnl += n
    peak_pnl = max(peak_pnl, running_pnl)
    dd = running_pnl - peak_pnl
    max_dd = min(max_dd, dd)
    if n < 0:
        streak += 1
        max_streak = max(max_streak, streak)
    else:
        if streak > 0:
            streaks.append(streak)
        streak = 0
if streak > 0:
    streaks.append(streak)

print(f"  Max consecutive losses: {max_streak}")
print(f"  Max drawdown: {max_dd:+.1f} ticks (${max_dd*12.50:+,.0f})")
print(f"  Avg loss streak: {np.mean(streaks):.1f}" if streaks else "  No loss streaks")
streak_hist = defaultdict(int)
for s in streaks:
    streak_hist[s] += 1
print(f"  Streak distribution: {dict(sorted(streak_hist.items()))}")

# ========== 6. TIME-OF-DAY ANALYSIS ==========
print(f"\n{'='*70}")
print("6. TIME-OF-DAY ANALYSIS (early vs mid vs late session)")
print("=" * 70)
fracs = np.array([t['day_fraction'] for t in trades])
for label, lo, hi in [('Early (0-33%)', 0, 0.33), ('Mid (33-66%)', 0.33, 0.66), ('Late (66-100%)', 0.66, 1.01)]:
    mask = (fracs >= lo) & (fracs < hi)
    if mask.sum() == 0:
        continue
    seg_nets = nets[mask]
    print(f"  {label}: {mask.sum()} trades, net/trade={np.mean(seg_nets):+.4f}, WR={np.mean(seg_nets>0):.1%}")

# ========== 7. BOOTSTRAP CONFIDENCE INTERVAL ==========
print(f"\n{'='*70}")
print("7. BOOTSTRAP CI (10000 resamples of net/trade)")
print("=" * 70)
np.random.seed(42)
n_boot = 10000
boot_means = np.array([np.mean(np.random.choice(nets, size=len(nets), replace=True)) for _ in range(n_boot)])
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
prob_positive = np.mean(boot_means > 0)
print(f"  Mean net/trade: {np.mean(nets):+.4f}")
print(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
print(f"  P(profitable): {prob_positive:.1%}")
if ci_lo < 0:
    print(f"  ⚠️ CI includes zero — NOT statistically significant at 95%")
else:
    print(f"  ✅ CI entirely positive — edge is statistically robust")

# ========== 8. HALF-SAMPLE STABILITY ==========
print(f"\n{'='*70}")
print("8. HALF-SAMPLE STABILITY (first 11 vs last 10 dates)")
print("=" * 70)
sorted_dates = sorted(day_data.keys())
mid = len(sorted_dates) // 2
first_half_dates = set(sorted_dates[:mid])
second_half_dates = set(sorted_dates[mid:])

h1_nets = [t['net'] for t in trades if t['date'] in first_half_dates]
h2_nets = [t['net'] for t in trades if t['date'] in second_half_dates]

if h1_nets and h2_nets:
    print(f"  First half ({len(first_half_dates)} dates, {len(h1_nets)} trades): "
          f"net/trade={np.mean(h1_nets):+.4f}, WR={np.mean(np.array(h1_nets)>0):.1%}")
    print(f"  Second half ({len(second_half_dates)} dates, {len(h2_nets)} trades): "
          f"net/trade={np.mean(h2_nets):+.4f}, WR={np.mean(np.array(h2_nets)>0):.1%}")
    if np.mean(h1_nets) > 0 and np.mean(h2_nets) > 0:
        print(f"  ✅ Both halves profitable — edge not period-specific")
    else:
        print(f"  ⚠️ Edge concentrated in one half")

# ========== 9. PROFIT FACTOR & RISK METRICS ==========
print(f"\n{'='*70}")
print("9. RISK METRICS")
print("=" * 70)
wins = nets[nets > 0]
losses = nets[nets < 0]
pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 else float('inf')
avg_win = np.mean(wins) if len(wins) > 0 else 0
avg_loss = np.mean(losses) if len(losses) > 0 else 0
rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
sortino = np.mean(nets) / np.std(nets[nets < 0]) * np.sqrt(252) if np.std(nets[nets < 0]) > 0 else 0

print(f"  Profit Factor: {pf:.2f}")
print(f"  Avg Win: {avg_win:+.2f} ticks | Avg Loss: {avg_loss:+.2f} ticks | R:R = {rr:.2f}")
print(f"  Sortino: {sortino:+.2f}")
print(f"  Calmar (annualized): {(np.sum(nets)/n_days*252) / abs(max_dd):.2f}" if max_dd < 0 else "  Calmar: infinite (no DD)")

# ========== SAVE RESULTS ==========
results = {
    'config': 'LONG_q10_tp12sl8',
    'n_trades': len(trades),
    'n_dates': n_days,
    'net_per_trade': float(np.mean(nets)),
    'total_ticks': float(np.sum(nets)),
    'win_rate': float(np.mean(nets > 0)),
    'sharpe': float(np.mean(nets)/np.std(nets)*np.sqrt(252)),
    'day_sharpe': float(day_sharpe),
    'sortino': float(sortino),
    'profit_factor': float(pf),
    'max_dd_ticks': float(max_dd),
    'max_loss_streak': int(max_streak),
    'bootstrap_ci': [float(ci_lo), float(ci_hi)],
    'bootstrap_p_positive': float(prob_positive),
    'loo_stability': f"{n_positive}/{len(loo_means)}",
    'signal_outcome_corr': float(corr),
    'top2_day_concentration': float(top2/total*100) if total > 0 else 0,
    'green_days': green,
    'red_days': red,
    'per_date': {date: {'net': d['net'], 'trades': d['trades']}
                 for date, d in day_data.items()},
    'trades': trades,  # full per-trade data
}

with open(OUTPUT / 'v25_diagnostic.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print("DIAGNOSTIC COMPLETE — saved to output/tick_replay_v25_diagnostic/")
print("=" * 70)
