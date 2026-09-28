#!/usr/bin/env python3
"""
Tick Replay v27 — SHORT-ONLY Validation
=========================================
HC #659: Tick-level replay mandatory. Permutation test required.

v26 found 97% of LONG configs profitable. But is that model edge or market drift?
This script tests SHORT-ONLY to answer:
  - If shorts also profit → genuine directional prediction
  - If shorts all lose → long profits were market drift artifact

Also tests BOTH sides (long when signal positive, short when negative)
to see if the model genuinely predicts direction.

Uses same parameter grid as v26 but focused: only Phase 2 candidates
from v26 to avoid redundant compute.
"""

import numpy as np
import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from itertools import product

PRED_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_tick_replay_preds'
MBO_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo'
OUTPUT = Path('/home/jupiter/Lvl3Quant/output/tick_replay_v27_short_test')
OUTPUT.mkdir(parents=True, exist_ok=True)

PRED_STRIDE = 250
N_PERMS = 200
TICK = 0.25

# Parameter grid — focused on v26 sweet spots + short-specific
TP_RANGE = [6, 8, 10, 12, 14, 16]
SL_RANGE = [4, 6, 8]  # Tight SLs only (v26 showed wider SLs worse)
Q_RANGE = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]

# Fixed hold/cancel since v26 showed they don't matter
HOLD = 30
CANCEL = 10

# Phase 2 thresholds
MIN_NET_PER_TRADE = 0.3  # Lower for shorts (may have less edge)
MIN_TRADES = 30

print(f"{'='*70}")
print(f"TICK REPLAY v27 — SHORT-ONLY + BOTH-SIDES VALIDATION")
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


def run_directional(data, head, q, tp, sl, hold_s, cancel_s, side='short',
                     randomize_timing=False, seed=None):
    """
    Run directional tick replay.
    side='short': only short (negative) signals
    side='long': only long (positive) signals
    side='both': long on positive, short on negative
    """
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

        # Select candidates based on side
        if side == 'short':
            # Short: negative signals, most negative
            neg_signals = -signal.copy()  # Flip so we can use same quantile logic
            neg_signals[neg_signals <= 0] = 0
            neg_nonzero = neg_signals[neg_signals > 0]
            if len(neg_nonzero) == 0:
                continue
            threshold = np.quantile(neg_nonzero, 1 - q)
            candidates = [(pi, 'short') for pi in range(n_preds) if neg_signals[pi] >= threshold]
        elif side == 'long':
            pos_signals = signal.copy()
            pos_signals[pos_signals <= 0] = 0
            pos_nonzero = pos_signals[pos_signals > 0]
            if len(pos_nonzero) == 0:
                continue
            threshold = np.quantile(pos_nonzero, 1 - q)
            candidates = [(pi, 'long') for pi in range(n_preds) if pos_signals[pi] >= threshold]
        elif side == 'both':
            # Top q% of positive → long, top q% of negative → short
            pos_signals = signal.copy()
            pos_signals[pos_signals <= 0] = 0
            neg_signals = -signal.copy()
            neg_signals[neg_signals <= 0] = 0

            cands = []
            pos_nonzero = pos_signals[pos_signals > 0]
            if len(pos_nonzero) > 0:
                pos_thresh = np.quantile(pos_nonzero, 1 - q)
                cands += [(pi, 'long') for pi in range(n_preds) if pos_signals[pi] >= pos_thresh]
            neg_nonzero = neg_signals[neg_signals > 0]
            if len(neg_nonzero) > 0:
                neg_thresh = np.quantile(neg_nonzero, 1 - q)
                cands += [(pi, 'short') for pi in range(n_preds) if neg_signals[pi] >= neg_thresh]
            candidates = sorted(cands, key=lambda x: x[0])

        if not candidates:
            continue

        # Randomize timing for permutation test
        if randomize_timing and rng is not None:
            n_cands = len(candidates)
            valid_indices = [pi for pi in range(n_preds) if pi * PRED_STRIDE < n_events - 100]
            if len(valid_indices) >= n_cands:
                rand_pis = sorted(rng.choice(valid_indices, size=n_cands, replace=False).tolist())
                # Keep original sides but shuffle timing
                candidates = [(rand_pis[i], candidates[i][1]) for i in range(n_cands)]

        # Execute trades
        for pi, trade_side in candidates:
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

            # Entry price depends on side
            if trade_side == 'long':
                entry_price = bid_price  # buy at bid (passive)
                tp_price = entry_price + tp * TICK
                sl_price = entry_price - sl * TICK
            else:  # short
                entry_price = ask_price  # sell at ask (passive)
                tp_price = entry_price - tp * TICK
                sl_price = entry_price + sl * TICK

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
                if trade_side == 'long':
                    if p <= entry_price and sizes[ei] > 0:
                        filled = True; fill_time = t; fill_idx = ei; break
                else:  # short
                    if p >= entry_price and sizes[ei] > 0:
                        filled = True; fill_time = t; fill_idx = ei; break

            if not filled:
                continue

            # Exit sim
            exit_deadline = fill_time + hold_s * 10**9
            exit_price = entry_price
            exit_reason = 'time_stop'
            mfe = 0.0
            mae = 0.0

            for ei in range(fill_idx + 1, min(fill_idx + 50000, n_events)):
                t = timestamps[ei]
                p = prices[ei]
                if p <= 0 or np.isnan(p):
                    continue

                if trade_side == 'long':
                    unrealized = (p - entry_price) / TICK
                else:  # short
                    unrealized = (entry_price - p) / TICK

                mfe = max(mfe, unrealized)
                mae = min(mae, unrealized)

                if trade_side == 'long':
                    if p >= tp_price:
                        exit_price = tp_price; exit_reason = 'tp'; break
                    if p <= sl_price:
                        exit_price = sl_price; exit_reason = 'sl'; break
                else:  # short
                    if p <= tp_price:
                        exit_price = tp_price; exit_reason = 'tp'; break
                    if p >= sl_price:
                        exit_price = sl_price; exit_reason = 'sl'; break

                if t >= exit_deadline:
                    exit_price = p; exit_reason = 'time_stop'; break

            if trade_side == 'long':
                gross = (exit_price - entry_price) / TICK
            else:
                gross = (entry_price - exit_price) / TICK

            cost = 0.376 if exit_reason == 'tp' else 1.376
            net = gross - cost

            trades.append({
                'date': date,
                'side': trade_side,
                'gross': float(gross),
                'net': float(net),
                'exit_reason': exit_reason,
                'mfe': float(mfe),
                'mae': float(mae),
            })

    return trades


def summarize(trades, label=''):
    """Quick summary stats."""
    if len(trades) < 3:
        return None
    nets = np.array([t['net'] for t in trades])
    n = len(nets)
    mean_net = float(np.mean(nets))
    total_net = float(np.sum(nets))
    wr = float(np.mean(nets > 0))
    sharpe = float(np.mean(nets) / np.std(nets) * np.sqrt(252)) if np.std(nets) > 0 else 0

    day_data = defaultdict(float)
    for t in trades:
        day_data[t['date']] += t['net']
    day_nets = list(day_data.values())
    n_days = len(day_nets)
    green = sum(1 for d in day_nets if d > 0)
    red = sum(1 for d in day_nets if d < 0)
    day_sharpe = float(np.mean(day_nets) / np.std(day_nets) * np.sqrt(252)) if len(day_nets) > 1 and np.std(day_nets) > 0 else 0

    exits = {}
    for t in trades:
        exits[t['exit_reason']] = exits.get(t['exit_reason'], 0) + 1

    sorted_day_nets = sorted(day_nets, reverse=True)
    top2 = sum(sorted_day_nets[:2]) if len(sorted_day_nets) >= 2 else sum(sorted_day_nets)
    concentration = (top2 / total_net * 100) if total_net > 0 else 999

    return {
        'label': label,
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
        'avg_mae': round(float(np.mean([t['mae'] for t in trades])), 1),
        'top2_day_concentration': round(concentration, 1),
    }


# =====================================================
# PHASE 1: SHORT-ONLY + BOTH-SIDES sweep
# =====================================================
total_combos = len(TP_RANGE) * len(SL_RANGE) * len(Q_RANGE) * 3  # 3 sides
print(f"Parameter grid: {len(TP_RANGE)} TP x {len(SL_RANGE)} SL x {len(Q_RANGE)} Q x 3 sides = {total_combos} combos")
print(f"Hold={HOLD}s, Cancel={CANCEL}s (fixed per v26 findings)")

results = {'short': [], 'long': [], 'both': []}
t0 = time.time()
done = 0

for side in ['short', 'long', 'both']:
    print(f"\n--- {side.upper()} ---")
    for q, tp, sl in product(Q_RANGE, TP_RANGE, SL_RANGE):
        if sl >= tp:
            done += 1
            continue

        trades = run_directional(all_data, 'composite_signal', q, tp, sl, HOLD, CANCEL, side=side)
        label = f"{side}_q{int(q*100)}/tp{tp}sl{sl}"
        stats = summarize(trades, label)
        done += 1

        if stats is None:
            continue

        stats['params'] = {'q': q, 'tp': tp, 'sl': sl, 'side': side}
        results[side].append(stats)

        if done % 50 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_combos - done) / rate if rate > 0 else 0
            print(f"  {done}/{total_combos} ({elapsed:.0f}s, ETA {eta:.0f}s)")

elapsed_p1 = time.time() - t0
print(f"\nPhase 1 complete: {elapsed_p1:.0f}s")

# =====================================================
# ANALYSIS: Compare short vs long vs both
# =====================================================
print(f"\n{'='*70}")
print(f"DIRECTION COMPARISON")
print(f"{'='*70}")

for side in ['short', 'long', 'both']:
    res = results[side]
    if not res:
        print(f"\n  {side.upper()}: no results")
        continue

    profitable = sum(1 for r in res if r['net_per_trade'] > 0)
    avg_net = np.mean([r['net_per_trade'] for r in res])
    best = max(res, key=lambda x: x['net_per_trade'])

    print(f"\n  {side.upper()}: {profitable}/{len(res)} profitable ({profitable/len(res)*100:.0f}%)")
    print(f"    Average net/trade: {avg_net:+.3f} ticks")
    print(f"    Best: {best['label']} net={best['net_per_trade']:+.3f} n={best['n_trades']} WR={best['win_rate']:.1%}")

# CRITICAL: Direction asymmetry test
long_results = results['long']
short_results = results['short']

if long_results and short_results:
    long_profitable_pct = sum(1 for r in long_results if r['net_per_trade'] > 0) / len(long_results)
    short_profitable_pct = sum(1 for r in short_results if r['net_per_trade'] > 0) / len(short_results)

    print(f"\n{'='*70}")
    print(f"CRITICAL DIRECTION ASYMMETRY TEST")
    print(f"{'='*70}")
    print(f"  Long profitable:  {long_profitable_pct*100:.1f}%")
    print(f"  Short profitable: {short_profitable_pct*100:.1f}%")

    if long_profitable_pct > 0.80 and short_profitable_pct < 0.30:
        print(f"\n  ⚠️  VERDICT: MARKET DRIFT ARTIFACT")
        print(f"  Long profits are likely market drift (upward bias in test period).")
        print(f"  Model prediction is NOT adding directional value.")
    elif long_profitable_pct > 0.60 and short_profitable_pct > 0.60:
        print(f"\n  ✅  VERDICT: GENUINE DIRECTIONAL PREDICTION")
        print(f"  Both sides profitable — model predicts direction, not just market trend.")
    elif short_profitable_pct > long_profitable_pct:
        print(f"\n  📊  VERDICT: SHORT SIDE STRONGER")
        print(f"  Consistent with decay analysis (short edge > long edge).")
    else:
        print(f"\n  📊  VERDICT: MIXED — need permutation test to confirm")

# =====================================================
# PHASE 2: Permutation test on best configs per side
# =====================================================
print(f"\n{'='*70}")
print(f"PHASE 2: PERMUTATION TESTS — best configs per direction")
print(f"{'='*70}")

phase2_results = {}
for side in ['short', 'long', 'both']:
    res = results[side]
    # Select top candidates for perm testing
    candidates = [r for r in res if r['net_per_trade'] >= MIN_NET_PER_TRADE and r['n_trades'] >= MIN_TRADES]
    candidates.sort(key=lambda x: x['net_per_trade'], reverse=True)
    test_list = candidates[:10]  # Top 10 per side

    if not test_list:
        print(f"\n  {side.upper()}: no candidates above threshold")
        continue

    print(f"\n  {side.upper()}: testing {len(test_list)} candidates")
    perm_results = []

    for i, cand in enumerate(test_list):
        p = cand['params']
        label = cand['label']

        # Run real
        real_trades = run_directional(all_data, 'composite_signal', p['q'], p['tp'], p['sl'], HOLD, CANCEL, side=side)
        if not real_trades:
            continue
        real_mean = float(np.mean([t['net'] for t in real_trades]))

        # Run permutations
        perm_means = []
        t1 = time.time()
        for pi in range(N_PERMS):
            perm_trades = run_directional(all_data, 'composite_signal', p['q'], p['tp'], p['sl'], HOLD, CANCEL,
                                           side=side, randomize_timing=True, seed=pi * 42 + 7)
            if perm_trades:
                perm_means.append(float(np.mean([t['net'] for t in perm_trades])))

        if perm_means:
            perm_mean = float(np.mean(perm_means))
            p_value = float(np.mean([pm >= real_mean for pm in perm_means]))
            edge = real_mean - perm_mean
            verdict = "PASS" if p_value < 0.05 else "FAIL"

            print(f"    [{i+1}/{len(test_list)}] {label}: real={real_mean:+.4f} rand={perm_mean:+.4f} edge={edge:+.4f} p={p_value:.3f} → {verdict}")

            result = dict(cand)
            result['perm_real_mean'] = round(real_mean, 4)
            result['perm_random_mean'] = round(perm_mean, 4)
            result['perm_edge'] = round(edge, 4)
            result['perm_p_value'] = round(p_value, 3)
            result['perm_verdict'] = verdict
            perm_results.append(result)

    phase2_results[side] = perm_results

# =====================================================
# SAVE & SUMMARY
# =====================================================
# Save all results
with open(OUTPUT / 'results.json', 'w') as f:
    json.dump({
        'phase1': {side: res for side, res in results.items()},
        'phase2': phase2_results,
        'meta': {
            'n_dates': len(all_data),
            'dates': sorted(all_data.keys()),
            'elapsed_s': round(time.time() - t0, 1),
            'hold': HOLD,
            'cancel': CANCEL,
        }
    }, f, indent=2)

print(f"\n{'='*70}")
print(f"FINAL SUMMARY")
print(f"{'='*70}")

for side in ['short', 'long', 'both']:
    pr = phase2_results.get(side, [])
    if not pr:
        print(f"\n  {side.upper()}: no perm tests run")
        continue
    passing = [r for r in pr if r['perm_verdict'] == 'PASS']
    print(f"\n  {side.upper()}: {len(passing)}/{len(pr)} PASS permutation test")
    for r in passing:
        print(f"    {r['label']}: net={r['net_per_trade']:+.3f} p={r['perm_p_value']:.3f} edge={r['perm_edge']:+.4f} n={r['n_trades']}")

total_time = time.time() - t0
print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}m)")
print(f"Output: {OUTPUT}")
