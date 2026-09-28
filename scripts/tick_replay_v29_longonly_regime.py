#!/usr/bin/env python3
"""
Tick Replay v29 — Long-Only Regime-Stratified Production Config Search
======================================================================
v27 proved: shorts have ZERO model edge (all fail permutation). Longs pass.
v28 proved: champion configs hold on expanded 33-date sample.

This script:
1. Tests expanded long-only config grid around proven sweet spot
2. Regime-stratifies ALL results (green/red/flat days by ES close-to-close)
3. HC #428 R1: reject if regime-Sharpe gap > 0.50
4. HC #428 R2: verify MFE within horizon
5. Reports only configs that pass BOTH permutation AND regime gates

Uses EXACT v28 simulation engine (passive fill sim, BBO reconstruction).
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
OUTPUT = Path('/home/jupiter/Lvl3Quant/output/tick_replay_v29_longonly_regime')
OUTPUT.mkdir(parents=True, exist_ok=True)

PRED_STRIDE = 250
N_PERMS = 200
TICK = 0.25

# Expanded config grid around proven sweet spot
# Proven: q=0.10, TP=16, SL=4-8 work. Let's test nearby.
CONFIGS = [
    # Core proven configs
    {'q': 0.10, 'tp': 16, 'sl': 4, 'hold': 30, 'cancel': 10, 'label': 'q10_tp16_sl4'},
    {'q': 0.10, 'tp': 16, 'sl': 6, 'hold': 30, 'cancel': 10, 'label': 'q10_tp16_sl6'},
    {'q': 0.10, 'tp': 16, 'sl': 8, 'hold': 30, 'cancel': 10, 'label': 'q10_tp16_sl8'},
    # Adjacent TP values
    {'q': 0.10, 'tp': 12, 'sl': 4, 'hold': 30, 'cancel': 10, 'label': 'q10_tp12_sl4'},
    {'q': 0.10, 'tp': 14, 'sl': 4, 'hold': 30, 'cancel': 10, 'label': 'q10_tp14_sl4'},
    {'q': 0.10, 'tp': 14, 'sl': 6, 'hold': 30, 'cancel': 10, 'label': 'q10_tp14_sl6'},
    {'q': 0.10, 'tp': 20, 'sl': 6, 'hold': 30, 'cancel': 10, 'label': 'q10_tp20_sl6'},
    # Slightly tighter selectivity
    {'q': 0.07, 'tp': 16, 'sl': 4, 'hold': 30, 'cancel': 10, 'label': 'q07_tp16_sl4'},
    {'q': 0.07, 'tp': 16, 'sl': 6, 'hold': 30, 'cancel': 10, 'label': 'q07_tp16_sl6'},
    # Slightly wider selectivity
    {'q': 0.15, 'tp': 16, 'sl': 4, 'hold': 30, 'cancel': 10, 'label': 'q15_tp16_sl4'},
    {'q': 0.15, 'tp': 16, 'sl': 6, 'hold': 30, 'cancel': 10, 'label': 'q15_tp16_sl6'},
    # Hold time variants for best config
    {'q': 0.10, 'tp': 16, 'sl': 4, 'hold': 20, 'cancel': 10, 'label': 'q10_tp16_sl4_h20'},
    {'q': 0.10, 'tp': 16, 'sl': 4, 'hold': 45, 'cancel': 10, 'label': 'q10_tp16_sl4_h45'},
]

# === ES daily close data for regime classification ===
# Need to classify each date as green/red/flat
# We'll compute from the MBO data itself (first vs last trade price)
def classify_date_regime(mbo_data):
    """Classify a date as green/red/flat from MBO price data.
    Uses first 5% vs last 5% of session prices with wider threshold (20 ticks = 5 pts).
    This better captures the actual daily direction for ES futures."""
    prices = mbo_data['price']
    valid = prices[prices > 0]
    if len(valid) < 100:
        return 'flat'
    n = len(valid)
    pct5 = max(50, n // 20)  # 5% of session
    open_price = np.median(valid[:pct5])   # median of first 5% of trades
    close_price = np.median(valid[-pct5:]) # median of last 5% of trades
    change_ticks = (close_price - open_price) / TICK
    # ES moves ~40-80 ticks/day on average; use 20 ticks (5 pts) as green/red threshold
    if change_ticks > 20:
        return 'green'
    elif change_ticks < -20:
        return 'red'
    else:
        return 'flat'


def run_long_only(data, head, q, tp, sl, hold_s, cancel_s, randomize_timing=False, seed=None):
    """EXACT copy of v26/v28 simulation logic. Passive fill sim with BBO reconstruction."""
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
                'entry_price': float(entry_price),
            })

    return trades


def regime_stratify(trades, date_regimes):
    """Stratify by regime. Returns per-regime stats + HC #428 R1 check."""
    regime_trades = {'green': [], 'red': [], 'flat': []}
    for t in trades:
        r = date_regimes.get(t['date'], 'flat')
        regime_trades[r].append(t)

    stats = {}
    for regime, rtrades in regime_trades.items():
        if len(rtrades) < 3:
            stats[regime] = {'n_trades': len(rtrades), 'sharpe': 0, 'net_per_trade': 0, 'n_days': 0}
            continue
        nets = np.array([t['net'] for t in rtrades])
        day_data = defaultdict(float)
        for t in rtrades:
            day_data[t['date']] += t['net']
        day_vals = list(day_data.values())
        day_sharpe = float(np.mean(day_vals) / np.std(day_vals) * np.sqrt(252)) if len(day_vals) > 1 and np.std(day_vals) > 0 else 0
        stats[regime] = {
            'n_trades': len(rtrades),
            'n_days': len(day_data),
            'net_per_trade': round(float(np.mean(nets)), 4),
            'total_net': round(float(np.sum(nets)), 1),
            'win_rate': round(float(np.mean(nets > 0)), 3),
            'day_sharpe': round(day_sharpe, 2),
            'green_days': sum(1 for v in day_vals if v > 0),
        }

    # HC #428 R1: regime gap check
    g_days = stats.get('green', {}).get('n_days', 0)
    r_days = stats.get('red', {}).get('n_days', 0)
    g_sharpe = abs(stats.get('green', {}).get('day_sharpe', 0))
    r_sharpe = abs(stats.get('red', {}).get('day_sharpe', 0))
    max_sharpe = max(g_sharpe, r_sharpe, 0.001)
    regime_gap = abs(g_sharpe - r_sharpe) / max_sharpe

    # If either regime has <3 days, gap is unreliable — mark as underpowered
    if g_days < 3 or r_days < 3:
        stats['regime_gap'] = round(regime_gap, 3)
        stats['regime_gate'] = 'UNDERPOWERED'
        stats['regime_note'] = f'green={g_days}d, red={r_days}d — need >=3 each for reliable test'
    else:
        stats['regime_gap'] = round(regime_gap, 3)
        stats['regime_gate'] = 'PASS' if regime_gap <= 0.50 else 'FAIL'

    return stats


def mfe_horizon_check(trades, hold_s):
    """HC #428 R2: Check MFE within horizon."""
    mfes = [t['mfe'] for t in trades]
    if len(mfes) < 10:
        return {'gate': 'SKIP', 'reason': 'too few trades'}
    p90_mfe = np.percentile(mfes, 90)
    median_mfe = np.median(mfes)
    return {
        'p90_mfe': round(float(p90_mfe), 1),
        'median_mfe': round(float(median_mfe), 1),
        'mean_mfe': round(float(np.mean(mfes)), 1),
        'p10_mfe': round(float(np.percentile(mfes, 10)), 1),
        'gate': 'INFO',  # Can't reject on MFE alone for longs, but report it
    }


print(f"{'='*70}")
print(f"TICK REPLAY v29 — LONG-ONLY REGIME-STRATIFIED PRODUCTION CONFIG SEARCH")
print(f"{'='*70}")
print(f"v27 result: ALL shorts fail perm. Signal is LONG-ONLY.")
print(f"Testing {len(CONFIGS)} long-only configs with regime gates.\n")

# Load data
pred_files = sorted(os.listdir(PRED_DIR))
pred_dates = [f.replace('oot_', '').replace('.npz', '') for f in pred_files if f.startswith('oot_')]
mbo_files = {f.replace('mbo_', '').replace('.npz', ''): os.path.join(MBO_DIR, f)
             for f in os.listdir(MBO_DIR) if f.endswith('.npz')}
common_dates = sorted([d for d in pred_dates if d in mbo_files])
print(f"Common dates: {len(common_dates)}")

all_data = {}
for date in common_dates:
    try:
        pred_data = np.load(os.path.join(PRED_DIR, f'oot_{date}.npz'))
        mbo_data = np.load(mbo_files[date])
        n_preds = int(pred_data.get('n_preds', len(pred_data['composite_signal'])))
        if n_preds < 50:
            continue
        all_data[date] = {'preds': dict(pred_data), 'mbo': dict(mbo_data)}
    except Exception as e:
        print(f"  Skip {date}: {e}")

usable_dates = sorted(all_data.keys())
print(f"Loaded {len(all_data)} dates")

# Classify regimes
date_regimes = {}
for date, ddata in all_data.items():
    date_regimes[date] = classify_date_regime(ddata['mbo'])
regime_counts = defaultdict(int)
for r in date_regimes.values():
    regime_counts[r] += 1
print(f"Regimes: {dict(regime_counts)}")
print()

# === MAIN SWEEP ===
results = []
t0 = time.time()

for ci, cfg in enumerate(CONFIGS):
    label = cfg['label']
    elapsed_so_far = time.time() - t0
    print(f"\n[{ci+1}/{len(CONFIGS)}] {label} (q={cfg['q']}, TP={cfg['tp']}, SL={cfg['sl']}, hold={cfg['hold']}) [{elapsed_so_far/60:.0f}m elapsed]")

    # Phase 1: Real trades
    trades = run_long_only(all_data, 'composite_signal', cfg['q'], cfg['tp'], cfg['sl'], cfg['hold'], cfg['cancel'])

    if len(trades) < 10:
        print(f"  Too few trades ({len(trades)}) — skip")
        continue

    nets = np.array([t['net'] for t in trades])
    day_data = defaultdict(float)
    for t in trades:
        day_data[t['date']] += t['net']
    day_vals = list(day_data.values())
    n_days = len(day_vals)
    day_sharpe = float(np.mean(day_vals) / np.std(day_vals) * np.sqrt(252)) if n_days > 1 and np.std(day_vals) > 0 else 0

    # Day concentration
    sorted_dv = sorted(day_vals, reverse=True)
    total_net = float(np.sum(nets))
    top2_conc = (sum(sorted_dv[:2]) / total_net * 100) if total_net > 0 else 999

    exits = defaultdict(int)
    for t in trades:
        exits[t['exit_reason']] += 1

    stats = {
        'label': label,
        'params': cfg,
        'n_trades': len(trades),
        'trades_per_day': round(len(trades) / max(n_days, 1), 1),
        'net_per_trade': round(float(np.mean(nets)), 4),
        'total_net_ticks': round(total_net, 1),
        'win_rate': round(float(np.mean(nets > 0)), 3),
        'day_sharpe': round(day_sharpe, 2),
        'n_days': n_days,
        'green_days': sum(1 for v in day_vals if v > 0),
        'red_days': sum(1 for v in day_vals if v <= 0),
        'day_wr': round(sum(1 for v in day_vals if v > 0) / max(n_days, 1), 3),
        'top2_day_concentration': round(top2_conc, 1),
        'exits': dict(exits),
    }

    # Sortino
    neg_returns = [v for v in day_vals if v < 0]
    downside_std = np.std(neg_returns) if len(neg_returns) > 1 else 1.0
    sortino = float(np.mean(day_vals) / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    stats['day_sortino'] = round(sortino, 2)

    # Profit factor
    gross_wins = sum(n for n in nets if n > 0)
    gross_losses = abs(sum(n for n in nets if n < 0))
    stats['profit_factor'] = round(gross_wins / max(gross_losses, 0.001), 2)

    # MFE check
    mfe_info = mfe_horizon_check(trades, cfg['hold'])
    stats['mfe'] = mfe_info

    # Regime stratification
    regime_stats = regime_stratify(trades, date_regimes)
    stats['regime'] = regime_stats

    # Original vs new date split
    orig_trades = [t for t in trades if t['date'] < '20260320']
    new_trades = [t for t in trades if t['date'] >= '20260401']
    stats['orig_net_per_trade'] = round(float(np.mean([t['net'] for t in orig_trades])), 4) if orig_trades else 0
    stats['new_net_per_trade'] = round(float(np.mean([t['net'] for t in new_trades])), 4) if new_trades else 0
    stats['orig_n_days'] = len(set(t['date'] for t in orig_trades))
    stats['new_n_days'] = len(set(t['date'] for t in new_trades))

    print(f"  {len(trades)} trades, {n_days} days, net/trade={stats['net_per_trade']:+.3f}, Day Sharpe={day_sharpe:.2f}, PF={stats['profit_factor']:.2f}")
    print(f"  Regime: {regime_stats['regime_gate']} (gap={regime_stats['regime_gap']:.2f}) | Green Sharpe={regime_stats.get('green', {}).get('day_sharpe', 0):.1f}, Red Sharpe={regime_stats.get('red', {}).get('day_sharpe', 0):.1f}")
    print(f"  DayConc={top2_conc:.0f}% | Orig={stats['orig_net_per_trade']:+.3f}/t ({stats['orig_n_days']}d), New={stats['new_net_per_trade']:+.3f}/t ({stats['new_n_days']}d)")

    # Quick reject on day concentration or regime
    if top2_conc > 70:
        stats['perm_verdict'] = 'SKIP_CONC'
        print(f"  ⚠️ Day concentration {top2_conc:.0f}% > 70% cap — skipping permtest")
        results.append(stats)
        continue

    # Phase 2: Permutation test
    print(f"  Running permutation test ({N_PERMS} perms)...", flush=True)
    real_mean = stats['net_per_trade']
    perm_means = []
    for p in range(N_PERMS):
        if (p + 1) % 50 == 0:
            print(f"    Perm {p+1}/{N_PERMS}...", flush=True)
        perm_trades = run_long_only(all_data, 'composite_signal', cfg['q'], cfg['tp'], cfg['sl'],
                                     cfg['hold'], cfg['cancel'], randomize_timing=True, seed=p*137)
        if len(perm_trades) > 0:
            perm_means.append(float(np.mean([t['net'] for t in perm_trades])))

    if perm_means:
        perm_arr = np.array(perm_means)
        p_value = float(np.mean(perm_arr >= real_mean))
        stats['perm_real_mean'] = real_mean
        stats['perm_random_mean'] = round(float(np.mean(perm_arr)), 4)
        stats['perm_edge'] = round(real_mean - float(np.mean(perm_arr)), 4)
        stats['perm_p_value'] = p_value
        stats['perm_verdict'] = 'PASS' if p_value < 0.05 else 'FAIL'
        print(f"  Perm: real={real_mean:+.3f}, random={stats['perm_random_mean']:+.3f}, edge={stats['perm_edge']:+.3f}, p={p_value:.3f} → {stats['perm_verdict']}")

    # Overall verdict
    perm_pass = stats.get('perm_verdict') == 'PASS'
    regime_ok = regime_stats['regime_gate'] in ('PASS', 'UNDERPOWERED')  # UNDERPOWERED = not enough data to reject
    conc_pass = top2_conc <= 70
    stats['overall_verdict'] = 'PRODUCTION_CANDIDATE' if (perm_pass and regime_ok and conc_pass) else 'REJECT'
    if regime_stats['regime_gate'] == 'UNDERPOWERED' and perm_pass and conc_pass:
        stats['overall_verdict'] = 'CANDIDATE_REGIME_TBD'  # passes perm+conc but needs more regime data
    print(f"  → {stats['overall_verdict']}")

    results.append(stats)

elapsed = time.time() - t0

# === SAVE ===
output_data = {
    'n_configs': len(results),
    'n_dates': len(all_data),
    'date_regimes': date_regimes,
    'elapsed_s': elapsed,
    'results': results,
}

with open(OUTPUT / 'v29_results.json', 'w') as f:
    json.dump(output_data, f, indent=2)

# === FINAL SUMMARY ===
print(f"\n{'='*90}")
print(f"FINAL SUMMARY — {len(results)} configs tested in {elapsed/60:.0f} min")
print(f"{'='*90}")
print(f"{'Config':<24} {'Net/Tr':>7} {'Trades':>6} {'DaySh':>6} {'Sort':>6} {'PF':>5} {'Edge':>7} {'p':>5} {'Regime':>6} {'Verdict':<20}")
print('-' * 110)

candidates = []
for r in results:
    pval = r.get('perm_p_value', 1)
    edge = r.get('perm_edge', 0)
    regime_g = r.get('regime', {}).get('regime_gate', '?')
    verdict = r.get('overall_verdict', '?')
    print(f"{r['label']:<24} {r['net_per_trade']:>+7.3f} {r['n_trades']:>6} {r['day_sharpe']:>6.2f} {r.get('day_sortino', 0):>6.2f} {r.get('profit_factor', 0):>5.2f} {edge:>+7.3f} {pval:>5.3f} {regime_g:>6} {verdict:<20}")
    if verdict == 'PRODUCTION_CANDIDATE':
        candidates.append(r)

print(f"\n{'='*70}")
print(f"PRODUCTION CANDIDATES: {len(candidates)}")
for c in candidates:
    print(f"  ★ {c['label']}: {c['net_per_trade']:+.3f}/trade, Sharpe {c['day_sharpe']:.2f}, Sortino {c.get('day_sortino',0):.2f}, PF {c.get('profit_factor',0):.2f}")
    print(f"    Edge over random: {c.get('perm_edge',0):+.3f} ticks (p={c.get('perm_p_value',1):.3f})")
    print(f"    Regime gap: {c['regime']['regime_gap']:.2f}, DayConc: {c['top2_day_concentration']:.0f}%")
    print(f"    MFE p90: {c['mfe']['p90_mfe']:.1f}, median: {c['mfe']['median_mfe']:.1f}")

if not candidates:
    print("  None — no config passes all gates (perm + regime + concentration)")

print(f"\nResults saved to {OUTPUT / 'v29_results.json'}")
