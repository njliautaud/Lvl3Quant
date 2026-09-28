#!/usr/bin/env python3
"""
Tick Replay v28 — Expanded Sample Validation (33 dates vs prior 21)
====================================================================
Uses EXACT v26 simulation logic (passive fill sim, BBO reconstruction).
Tests confirmed configs on 33 dates to validate edge on unseen April data.
"""

import numpy as np
import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

PRED_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_tick_replay_preds'
MBO_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo'
OUTPUT = Path('/home/jupiter/Lvl3Quant/output/tick_replay_v28_expanded')
OUTPUT.mkdir(parents=True, exist_ok=True)

PRED_STRIDE = 250  # MBO events between predictions (matches v26)
N_PERMS = 200
TICK = 0.25

# Configs to test
CONFIGS = [
    {'q': 0.10, 'tp': 16, 'sl': 4, 'hold': 30, 'cancel': 10, 'label': 'champion_tp16sl4'},
    {'q': 0.10, 'tp': 16, 'sl': 8, 'hold': 30, 'cancel': 10, 'label': 'champion_tp16sl8'},
    {'q': 0.20, 'tp': 16, 'sl': 4, 'hold': 30, 'cancel': 10, 'label': 'highvol_tp16sl4'},
    {'q': 0.10, 'tp': 12, 'sl': 4, 'hold': 30, 'cancel': 10, 'label': 'alt_tp12sl4'},
]

print(f"{'='*70}")
print(f"TICK REPLAY v28 — EXPANDED SAMPLE VALIDATION")
print(f"{'='*70}")

# Load data
pred_files = sorted(os.listdir(PRED_DIR))
pred_dates = [f.replace('oot_', '').replace('.npz', '') for f in pred_files if f.startswith('oot_')]
mbo_files = {f.replace('mbo_', '').replace('.npz', ''): os.path.join(MBO_DIR, f)
             for f in os.listdir(MBO_DIR) if f.endswith('.npz')}
common_dates = sorted([d for d in pred_dates if d in mbo_files])
print(f"Prediction dates: {len(pred_dates)}")
print(f"MBO dates: {len(mbo_files)}")
print(f"Common dates: {len(common_dates)}")

all_data = {}
for date in common_dates:
    try:
        pred_data = np.load(os.path.join(PRED_DIR, f'oot_{date}.npz'))
        mbo_data = np.load(mbo_files[date])
        # Skip dates with very few predictions
        n_preds = int(pred_data.get('n_preds', len(pred_data['composite_signal'])))
        if n_preds < 50:
            print(f"  Skipping {date}: only {n_preds} predictions")
            continue
        all_data[date] = {'preds': dict(pred_data), 'mbo': dict(mbo_data)}
    except Exception as e:
        print(f"  Skip {date}: {e}")

usable_dates = sorted(all_data.keys())
n_original = len([d for d in usable_dates if d < '20260320'])
n_new = len([d for d in usable_dates if d >= '20260401'])
print(f"Loaded {len(all_data)} dates (original: {n_original}, new: {n_new})")
print()


def run_long_only(data, head, q, tp, sl, hold_s, cancel_s, randomize_timing=False, seed=None):
    """EXACT copy of v26 simulation logic. Passive fill sim with BBO reconstruction."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = None

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

        # Long-only: positive signals above quantile threshold
        pos_signals = signal.copy()
        pos_signals[pos_signals <= 0] = 0
        if np.max(pos_signals) <= 0:
            continue
        pos_nonzero = pos_signals[pos_signals > 0]
        if len(pos_nonzero) == 0:
            continue
        threshold = np.quantile(pos_nonzero, 1 - q)

        candidates = []
        for pi in range(n_preds):
            if signal[pi] > 0 and signal[pi] >= threshold:
                candidates.append(pi)

        if not candidates:
            continue

        # Randomize timing for permutation test
        if randomize_timing and rng is not None:
            n_cands = len(candidates)
            valid_indices = [pi for pi in range(n_preds) if pi * PRED_STRIDE < n_events - 100]
            if len(valid_indices) >= n_cands:
                candidates = sorted(rng.choice(valid_indices, size=n_cands, replace=False).tolist())

        # Execute trades
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

            entry_price = bid_price  # long = buy at bid (passive)

            tp_price = entry_price + tp * TICK
            sl_price = entry_price - sl * TICK
            cancel_deadline = entry_time + cancel_s * 10**9

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
                    filled = True; fill_time = t; fill_idx = ei; break

            if not filled:
                continue

            # Exit sim
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
                    exit_price = tp_price; exit_reason = 'tp'; break
                if p <= sl_price:
                    exit_price = sl_price; exit_reason = 'sl'; break
                if t >= exit_deadline:
                    exit_price = p; exit_reason = 'time_stop'; break

            if exit_price <= 0:
                exit_price = last_p

            gross = (exit_price - entry_price) / TICK
            cost = 0.376 if exit_reason == 'tp' else 1.376
            net = gross - cost

            trades.append({
                'date': date,
                'gross': float(gross),
                'net': float(net),
                'exit_reason': exit_reason,
                'mfe': float(mfe),
                'mae': float(mae),
            })

    return trades


def summarize_with_split(trades, label=""):
    """Summarize trades with original vs new date split."""
    if len(trades) < 5:
        return None

    nets = np.array([t['net'] for t in trades])
    n = len(nets)
    mean_net = float(np.mean(nets))
    total_net = float(np.sum(nets))
    wr = float(np.mean(nets > 0))
    sharpe = float(np.mean(nets) / np.std(nets) * np.sqrt(n)) if np.std(nets) > 0 else 0

    # Per-day stats
    day_data = defaultdict(float)
    for t in trades:
        day_data[t['date']] += t['net']
    day_nets = list(day_data.values())
    n_days = len(day_nets)
    green = sum(1 for d in day_nets if d > 0)
    red = sum(1 for d in day_nets if d < 0)
    day_sharpe = float(np.mean(day_nets) / np.std(day_nets) * np.sqrt(252)) if len(day_nets) > 1 and np.std(day_nets) > 0 else 0

    # Exit breakdown
    exits = {}
    for t in trades:
        exits[t['exit_reason']] = exits.get(t['exit_reason'], 0) + 1

    # Day concentration
    sorted_day_nets = sorted(day_nets, reverse=True)
    top2 = sum(sorted_day_nets[:2]) if len(sorted_day_nets) >= 2 else sum(sorted_day_nets)
    concentration = (top2 / total_net * 100) if total_net > 0 else 999

    # Split by original vs new dates
    orig_trades = [t for t in trades if t['date'] < '20260320']
    new_trades = [t for t in trades if t['date'] >= '20260401']

    orig_nets = np.array([t['net'] for t in orig_trades]) if orig_trades else np.array([])
    new_nets = np.array([t['net'] for t in new_trades]) if new_trades else np.array([])

    orig_day_data = defaultdict(float)
    for t in orig_trades:
        orig_day_data[t['date']] += t['net']
    new_day_data = defaultdict(float)
    for t in new_trades:
        new_day_data[t['date']] += t['net']

    orig_day_vals = list(orig_day_data.values())
    new_day_vals = list(new_day_data.values())

    orig_day_sharpe = float(np.mean(orig_day_vals) / np.std(orig_day_vals) * np.sqrt(252)) if len(orig_day_vals) > 1 and np.std(orig_day_vals) > 0 else 0
    new_day_sharpe = float(np.mean(new_day_vals) / np.std(new_day_vals) * np.sqrt(252)) if len(new_day_vals) > 1 and np.std(new_day_vals) > 0 else 0

    return {
        'n_trades': n,
        'trades_per_day': round(n / max(n_days, 1), 1),
        'net_per_trade': round(mean_net, 4),
        'total_net_ticks': round(total_net, 1),
        'win_rate': round(wr, 3),
        'sharpe': round(sharpe, 2),
        'day_sharpe': round(day_sharpe, 2),
        'n_days': n_days,
        'green_days': green,
        'red_days': red,
        'day_wr': round(green / max(green + red, 1), 3),
        'exits': exits,
        'avg_mfe': round(float(np.mean([t['mfe'] for t in trades])), 1),
        'p90_mfe': round(float(np.percentile([t['mfe'] for t in trades], 90)), 1),
        'avg_mae': round(float(np.mean([t['mae'] for t in trades])), 1),
        'top2_day_concentration': round(concentration, 1),
        'label': label,
        # Split analysis
        'original_n_trades': len(orig_trades),
        'original_n_days': len(orig_day_data),
        'original_net_per_trade': round(float(np.mean(orig_nets)), 4) if len(orig_nets) > 0 else 0,
        'original_day_sharpe': round(orig_day_sharpe, 2),
        'original_green_days': sum(1 for d in orig_day_vals if d > 0),
        'new_n_trades': len(new_trades),
        'new_n_days': len(new_day_data),
        'new_net_per_trade': round(float(np.mean(new_nets)), 4) if len(new_nets) > 0 else 0,
        'new_day_sharpe': round(new_day_sharpe, 2),
        'new_green_days': sum(1 for d in new_day_vals if d > 0),
    }


# === MAIN ===
results = []
t0 = time.time()

for cfg in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Config: {cfg['label']} (q={cfg['q']}, TP={cfg['tp']}, SL={cfg['sl']})")
    print(f"{'='*60}")

    # Phase 1: Run on full sample
    print("Phase 1: Full sample test...")
    trades = run_long_only(all_data, 'composite_signal', cfg['q'], cfg['tp'], cfg['sl'], cfg['hold'], cfg['cancel'])
    stats = summarize_with_split(trades, label=cfg['label'])

    if stats is None:
        print("  Too few trades — skipping")
        continue

    print(f"  Total: {stats['n_trades']} trades over {stats['n_days']} days")
    print(f"  Net/trade: {stats['net_per_trade']:+.4f} ticks, WR: {stats['win_rate']:.1%}, Day WR: {stats['day_wr']:.1%}")
    print(f"  Day Sharpe: {stats['day_sharpe']:.2f}")
    print(f"  ORIGINAL ({stats['original_n_days']} days): {stats['original_net_per_trade']:+.4f}/trade, Day Sharpe {stats['original_day_sharpe']:.2f}, {stats['original_green_days']} green days")
    print(f"  NEW APRIL ({stats['new_n_days']} days): {stats['new_net_per_trade']:+.4f}/trade, Day Sharpe {stats['new_day_sharpe']:.2f}, {stats['new_green_days']} green days")

    # Phase 2: Permutation test
    print(f"\nPhase 2: Permutation test ({N_PERMS} permutations)...", flush=True)
    real_mean = stats['net_per_trade']
    perm_means = []
    for p in range(N_PERMS):
        if (p + 1) % 50 == 0:
            print(f"  Perm {p+1}/{N_PERMS}...", flush=True)
        perm_trades = run_long_only(all_data, 'composite_signal', cfg['q'], cfg['tp'], cfg['sl'],
                                     cfg['hold'], cfg['cancel'], randomize_timing=True, seed=p*137)
        if len(perm_trades) > 0:
            perm_means.append(float(np.mean([t['net'] for t in perm_trades])))

    if perm_means:
        perm_arr = np.array(perm_means)
        p_value = float(np.mean(perm_arr >= real_mean))
        perm_result = {
            'perm_real_mean': real_mean,
            'perm_random_mean': round(float(np.mean(perm_arr)), 4),
            'perm_random_std': round(float(np.std(perm_arr)), 4),
            'perm_edge': round(real_mean - float(np.mean(perm_arr)), 4),
            'perm_p_value': p_value,
            'perm_verdict': 'PASS' if p_value < 0.05 else 'FAIL',
        }
        stats.update(perm_result)
        print(f"  Model: {perm_result['perm_real_mean']:+.4f}, Random: {perm_result['perm_random_mean']:+.4f}")
        print(f"  Edge: {perm_result['perm_edge']:+.4f} ticks, p={perm_result['perm_p_value']:.3f} → {perm_result['perm_verdict']}")

    stats['params'] = cfg
    results.append(stats)

elapsed = time.time() - t0

# Save results
output_data = {
    'n_configs': len(results),
    'total_dates': len(all_data),
    'original_dates': n_original,
    'new_dates': n_new,
    'elapsed_s': elapsed,
    'results': results,
}

with open(OUTPUT / 'v28_results.json', 'w') as f:
    json.dump(output_data, f, indent=2)

# Summary
print(f"\n{'='*70}")
print(f"SUMMARY ({elapsed/60:.1f} min)")
print(f"{'='*70}")
print(f"{'Config':<22} {'Net/Tr':>8} {'Trades':>6} {'p-val':>6} {'Edge':>7} | {'Orig/Tr':>8} {'New/Tr':>8}")
print('-' * 85)
for r in results:
    pval = r.get('perm_p_value', 1)
    edge = r.get('perm_edge', 0)
    print(f"{r['label']:<22} {r['net_per_trade']:>+8.4f} {r['n_trades']:>6} {pval:>6.3f} {edge:>+7.4f} | {r['original_net_per_trade']:>+8.4f} {r['new_net_per_trade']:>+8.4f}")

print(f"\nResults saved to {OUTPUT / 'v28_results.json'}")
