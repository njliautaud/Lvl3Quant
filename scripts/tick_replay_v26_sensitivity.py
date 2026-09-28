#!/usr/bin/env python3
"""
Tick Replay v26 — Sensitivity Sweep Around Winning Config
==========================================================
HC #659: Tick-level replay mandatory. Permutation test required.

v24 found LONG_q10_tp12sl8 passes permutation (p=0.005).
This script maps the full profitability surface:
  - TP: 6, 8, 10, 12, 14, 16
  - SL: 4, 6, 8, 10, 12
  - Quantile: 0.03, 0.05, 0.07, 0.10, 0.15, 0.20
  - Hold: 15, 20, 30, 45, 60
  - Cancel: 8, 10, 15

Phase 1: Fast sweep (no permutation) — all combos
Phase 2: Permutation test ONLY configs with net/trade > 0.5 and >50 trades
Phase 3: Regime analysis on permutation-passing configs

Output: JSON with full surface + heatmaps
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
OUTPUT = Path('/home/jupiter/Lvl3Quant/output/tick_replay_v26_sensitivity')
OUTPUT.mkdir(parents=True, exist_ok=True)

PRED_STRIDE = 250
N_PERMS = 200
TICK = 0.25

# Parameter grid
TP_RANGE = [6, 8, 10, 12, 14, 16]
SL_RANGE = [4, 6, 8, 10, 12]
Q_RANGE = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
HOLD_RANGE = [15, 20, 30, 45, 60]
CANCEL_RANGE = [8, 10, 15]

# Thresholds for Phase 2
MIN_NET_PER_TRADE = 0.5  # ticks
MIN_TRADES = 50
MIN_DAY_WR = 0.45

print(f"{'='*70}")
print(f"TICK REPLAY v26 — SENSITIVITY SWEEP")
print(f"{'='*70}")

total_combos = len(TP_RANGE) * len(SL_RANGE) * len(Q_RANGE) * len(HOLD_RANGE) * len(CANCEL_RANGE)
print(f"Parameter grid: {len(TP_RANGE)} TP x {len(SL_RANGE)} SL x {len(Q_RANGE)} Q x {len(HOLD_RANGE)} hold x {len(CANCEL_RANGE)} cancel = {total_combos} combos")

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


def run_long_only(data, head, q, tp, sl, hold_s, cancel_s, randomize_timing=False, seed=None):
    """Run long-only tick replay. Returns list of trade dicts."""
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


def summarize(trades):
    """Quick summary stats for a set of trades."""
    if len(trades) < 5:
        return None
    nets = np.array([t['net'] for t in trades])
    n = len(nets)
    mean_net = float(np.mean(nets))
    total_net = float(np.sum(nets))
    wr = float(np.mean(nets > 0))
    sharpe = float(np.mean(nets) / np.std(nets) * np.sqrt(252)) if np.std(nets) > 0 else 0

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

    # Day concentration — top 2 day P&L as % of total
    sorted_day_nets = sorted(day_nets, reverse=True)
    top2 = sum(sorted_day_nets[:2]) if len(sorted_day_nets) >= 2 else sum(sorted_day_nets)
    concentration = (top2 / total_net * 100) if total_net > 0 else 999

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
    }


# =====================================================
# PHASE 1: Fast sweep (no permutation)
# =====================================================
print(f"{'='*70}")
print(f"PHASE 1: FAST SWEEP — {total_combos} configs")
print(f"{'='*70}")

phase1_results = []
phase2_candidates = []
t0 = time.time()
done = 0

for q, tp, sl, hold, cancel in product(Q_RANGE, TP_RANGE, SL_RANGE, HOLD_RANGE, CANCEL_RANGE):
    # Skip obviously bad combos
    if sl >= tp:  # SL should be tighter than TP for positive expectancy
        done += 1
        continue

    trades = run_long_only(all_data, 'composite_signal', q, tp, sl, hold, cancel)
    stats = summarize(trades)
    done += 1

    if stats is None:
        continue

    label = f"q{int(q*100)}/tp{tp}sl{sl}/h{hold}c{cancel}"
    stats['label'] = label
    stats['params'] = {'q': q, 'tp': tp, 'sl': sl, 'hold': hold, 'cancel': cancel}
    phase1_results.append(stats)

    # Check if candidate for Phase 2
    if (stats['net_per_trade'] >= MIN_NET_PER_TRADE and
        stats['n_trades'] >= MIN_TRADES and
        stats['day_wr'] >= MIN_DAY_WR):
        phase2_candidates.append(stats)
        marker = " ★"
    else:
        marker = ""

    if done % 100 == 0:
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total_combos - done) / rate if rate > 0 else 0
        print(f"  {done}/{total_combos} ({elapsed:.0f}s, ETA {eta:.0f}s) — {len(phase2_candidates)} candidates so far")

elapsed = time.time() - t0
print(f"\nPhase 1 complete: {done} configs in {elapsed:.0f}s")
print(f"  Profitable (>0): {sum(1 for r in phase1_results if r['net_per_trade'] > 0)} / {len(phase1_results)}")
print(f"  Phase 2 candidates: {len(phase2_candidates)}")

# Sort phase1 by net_per_trade
phase1_results.sort(key=lambda x: x['net_per_trade'], reverse=True)

# Save Phase 1 results
with open(OUTPUT / 'phase1_sweep.json', 'w') as f:
    json.dump({
        'n_configs': len(phase1_results),
        'n_profitable': sum(1 for r in phase1_results if r['net_per_trade'] > 0),
        'n_phase2_candidates': len(phase2_candidates),
        'elapsed_s': round(elapsed, 1),
        'top_20': phase1_results[:20],
        'all_results': phase1_results,
    }, f, indent=2)
print(f"  Saved phase1_sweep.json")

# Print top 20
print(f"\n{'='*70}")
print(f"TOP 20 CONFIGS BY NET/TRADE")
print(f"{'='*70}")
print(f"{'Label':<25} {'N':>4} {'Net/T':>7} {'Total':>8} {'WR':>6} {'Shrp':>5} {'DShrp':>6} {'Days':>5} {'DWR':>5} {'Conc':>5}")
for r in phase1_results[:20]:
    print(f"{r['label']:<25} {r['n_trades']:>4} {r['net_per_trade']:>+7.3f} {r['total_net_ticks']:>+8.1f} {r['win_rate']:>6.1%} {r['sharpe']:>+5.1f} {r['day_sharpe']:>+6.1f} {r['green_days']}G/{r['red_days']}R {r['day_wr']:>5.1%} {r['top2_day_concentration']:>5.1f}")

# =====================================================
# PHASE 2: Permutation tests on candidates
# =====================================================
if phase2_candidates:
    print(f"\n{'='*70}")
    print(f"PHASE 2: PERMUTATION TESTS — {len(phase2_candidates)} candidates")
    print(f"{'='*70}")

    # Sort by net_per_trade descending, test best first
    phase2_candidates.sort(key=lambda x: x['net_per_trade'], reverse=True)

    # Limit to top 30 to avoid excessive runtime
    test_list = phase2_candidates[:30]

    phase2_results = []

    for i, cand in enumerate(test_list):
        p = cand['params']
        label = cand['label']
        print(f"\n  [{i+1}/{len(test_list)}] {label} (net={cand['net_per_trade']:+.3f}, {cand['n_trades']} trades)")

        # Run real config to get exact mean
        real_trades = run_long_only(all_data, 'composite_signal', p['q'], p['tp'], p['sl'], p['hold'], p['cancel'])
        real_mean = float(np.mean([t['net'] for t in real_trades]))

        # Run permutations
        perm_means = []
        t1 = time.time()
        for pi in range(N_PERMS):
            perm_trades = run_long_only(all_data, 'composite_signal', p['q'], p['tp'], p['sl'], p['hold'], p['cancel'],
                                         randomize_timing=True, seed=pi * 42 + 7)
            if len(perm_trades) > 0:
                perm_means.append(float(np.mean([t['net'] for t in perm_trades])))
            if (pi + 1) % 50 == 0:
                print(f"    {pi+1}/{N_PERMS} ({time.time()-t1:.0f}s)")

        if perm_means:
            perm_mean = float(np.mean(perm_means))
            p_value = float(np.mean([pm >= real_mean for pm in perm_means]))
            edge_over_random = real_mean - perm_mean

            verdict = "PASS" if p_value < 0.05 else "FAIL"
            print(f"    PERM: real={real_mean:+.4f} vs random={perm_mean:+.4f} edge={edge_over_random:+.4f} p={p_value:.3f} → {verdict}")

            result = dict(cand)
            result['perm_real_mean'] = round(real_mean, 4)
            result['perm_random_mean'] = round(perm_mean, 4)
            result['perm_edge'] = round(edge_over_random, 4)
            result['perm_p_value'] = round(p_value, 3)
            result['perm_verdict'] = verdict
            phase2_results.append(result)

    # Save Phase 2
    passing = [r for r in phase2_results if r['perm_verdict'] == 'PASS']
    failing = [r for r in phase2_results if r['perm_verdict'] == 'FAIL']

    with open(OUTPUT / 'phase2_permtest.json', 'w') as f:
        json.dump({
            'n_tested': len(phase2_results),
            'n_pass': len(passing),
            'n_fail': len(failing),
            'passing': passing,
            'failing': failing,
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"PHASE 2 SUMMARY: {len(passing)} PASS / {len(failing)} FAIL out of {len(phase2_results)} tested")
    print(f"{'='*70}")

    if passing:
        print(f"\nPASSING CONFIGS:")
        for r in passing:
            print(f"  {r['label']}: net={r['net_per_trade']:+.3f}, p={r['perm_p_value']:.3f}, edge={r['perm_edge']:+.4f}, trades={r['n_trades']}, DWR={r['day_wr']:.0%}")

        # =====================================================
        # PHASE 3: Regime analysis on passing configs
        # =====================================================
        print(f"\n{'='*70}")
        print(f"PHASE 3: REGIME ANALYSIS — {len(passing)} passing configs")
        print(f"{'='*70}")

        # Get SPY returns for regime classification
        # Using same dates as v24 overlay
        spy_closes = {}
        try:
            import pandas as pd
            spy_csv = '/home/jupiter/Lvl3Quant/data/feature_store/macro_regime/spy_daily.csv'
            if os.path.exists(spy_csv):
                df = pd.read_csv(spy_csv)
                for _, row in df.iterrows():
                    dt = str(row.get('date', '')).replace('-', '')[:8]
                    spy_closes[dt] = float(row.get('close', row.get('Close', 0)))
        except:
            pass

        for r in passing:
            p = r['params']
            trades = run_long_only(all_data, 'composite_signal', p['q'], p['tp'], p['sl'], p['hold'], p['cancel'])

            # Per-day P&L
            day_pnl = defaultdict(lambda: {'net': 0, 'trades': 0})
            for t in trades:
                day_pnl[t['date']]['net'] += t['net']
                day_pnl[t['date']]['trades'] += 1

            # Classify days by SPY return
            green_days, red_days, flat_days = [], [], []
            for date, dp in day_pnl.items():
                prev_dates = sorted([d for d in spy_closes if d < date])
                if prev_dates and date in spy_closes:
                    spy_ret = (spy_closes[date] - spy_closes[prev_dates[-1]]) / spy_closes[prev_dates[-1]]
                    if spy_ret > 0.001:
                        green_days.append(dp['net'])
                    elif spy_ret < -0.001:
                        red_days.append(dp['net'])
                    else:
                        flat_days.append(dp['net'])
                else:
                    flat_days.append(dp['net'])

            green_sharpe = float(np.mean(green_days) / np.std(green_days) * np.sqrt(252)) if len(green_days) > 1 and np.std(green_days) > 0 else 0
            red_sharpe = float(np.mean(red_days) / np.std(red_days) * np.sqrt(252)) if len(red_days) > 1 and np.std(red_days) > 0 else 0

            # HC #428 regime gap
            if max(abs(green_sharpe), abs(red_sharpe)) > 0:
                regime_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe))
            else:
                regime_gap = 0

            regime_verdict = "PASS" if regime_gap < 0.50 else "FAIL"

            print(f"\n  {r['label']}:")
            print(f"    Green days: {len(green_days)}, Sharpe={green_sharpe:+.2f}, avg={np.mean(green_days):+.1f}" if green_days else f"    Green days: 0")
            print(f"    Red days: {len(red_days)}, Sharpe={red_sharpe:+.2f}, avg={np.mean(red_days):+.1f}" if red_days else f"    Red days: 0")
            print(f"    Flat days: {len(flat_days)}" if flat_days else f"    Flat days: 0")
            print(f"    Regime gap: {regime_gap:.3f} → {regime_verdict} (threshold <0.50)")

            r['regime_green_sharpe'] = round(green_sharpe, 2)
            r['regime_red_sharpe'] = round(red_sharpe, 2)
            r['regime_gap'] = round(regime_gap, 3)
            r['regime_verdict'] = regime_verdict
            r['n_green'] = len(green_days)
            r['n_red'] = len(red_days)

        # Save final results
        fully_passing = [r for r in passing if r.get('regime_verdict') == 'PASS']
        with open(OUTPUT / 'phase3_final.json', 'w') as f:
            json.dump({
                'n_perm_pass': len(passing),
                'n_regime_pass': len(fully_passing),
                'fully_passing': fully_passing,
                'perm_pass_regime_fail': [r for r in passing if r.get('regime_verdict') != 'PASS'],
            }, f, indent=2)

        print(f"\n{'='*70}")
        print(f"FINAL: {len(fully_passing)} configs pass BOTH permutation AND regime gates")
        print(f"{'='*70}")
        for r in fully_passing:
            print(f"  {r['label']}: net={r['net_per_trade']:+.3f}, p={r['perm_p_value']:.3f}, regime_gap={r['regime_gap']:.3f}, trades={r['n_trades']}")

        # Robustness assessment
        if len(fully_passing) >= 5:
            print(f"\n  ASSESSMENT: ROBUST — {len(fully_passing)} configs in a region pass all gates")
            print(f"  This suggests genuine model edge, not single-point curve-fitting")
        elif len(fully_passing) >= 2:
            print(f"\n  ASSESSMENT: MODERATE — {len(fully_passing)} configs pass, edge exists but narrow")
        elif len(fully_passing) == 1:
            print(f"\n  ASSESSMENT: FRAGILE — only 1 config passes, possible overfitting")
        else:
            print(f"\n  ASSESSMENT: NO EDGE — zero configs pass all gates")

    else:
        print(f"\nNo configs passed permutation test.")
        print(f"ASSESSMENT: NO EDGE — the v24 result may have been overfitted")

else:
    print(f"\nNo Phase 2 candidates — all configs below threshold")
    print(f"ASSESSMENT: NO EDGE at these thresholds")

total_time = time.time() - t0
print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}m)")
print(f"Output: {OUTPUT}")
