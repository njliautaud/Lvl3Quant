#!/usr/bin/env python3
"""
Tick Replay v24 — Long-Only + Direction-Filtered Variants
==========================================================
HC #659: Tick-level replay mandatory. Permutation test required.

KEY INSIGHT from v22/v23:
- Long trades consistently profitable (+0.77 to +1.73 net/trade)
- Short trades near zero or negative (-0.16 to -0.46 net/trade)
- All configs FAIL permutation when mixing long+short

HYPOTHESIS: The model has genuine long-side directional edge that gets
diluted by weak short predictions. Testing:
  A) Long-only (skip all shorts)
  B) Asymmetric confidence (q5 for shorts, q3 for longs — more selective on shorts)
  C) Long-only with tighter stops (TP 8-12, SL 4-8)

PERMUTATION: Randomize entry TIMING (not direction) for long-only,
since direction is fixed. This tests whether the MODEL's timing
adds value vs random long entries at same frequency.
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
OUTPUT = Path('/home/jupiter/Lvl3Quant/output/tick_replay_v24_longonly')
OUTPUT.mkdir(parents=True, exist_ok=True)

PRED_STRIDE = 250
N_PERMS = 200
TICK = 0.25

# Load data once
print(f"{'='*70}")
print(f"TICK REPLAY v24 — LONG-ONLY + DIRECTION FILTERING")
print(f"{'='*70}")

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


def run_config(data, cfg, mode='long_only', randomize_timing=False, seed=None):
    """
    Run tick replay with direction filtering.

    Modes:
      'long_only' — only take long trades (signal > 0 above threshold)
      'short_only' — only take short trades (signal < 0 above threshold)
      'both' — original behavior (both directions)
      'asymmetric' — different quantile thresholds for long vs short

    randomize_timing: if True, keep same number of trades per day but
    randomize which prediction indices trigger entries. Tests whether
    MODEL TIMING adds value vs random timing.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = None

    head = cfg['head']
    q = cfg['quantile']
    q_short = cfg.get('quantile_short', q)  # for asymmetric mode
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

        # Build candidate list based on mode
        candidates = []

        if mode == 'long_only':
            # Only positive signals above threshold
            pos_signals = signal.copy()
            pos_signals[pos_signals <= 0] = 0
            if np.max(pos_signals) <= 0:
                continue
            threshold = np.quantile(pos_signals[pos_signals > 0], 1 - q) if np.sum(pos_signals > 0) > 0 else float('inf')
            for pi in range(n_preds):
                if signal[pi] > 0 and signal[pi] >= threshold:
                    candidates.append((pi, 'long'))

        elif mode == 'short_only':
            neg_signals = -signal.copy()
            neg_signals[neg_signals <= 0] = 0
            if np.max(neg_signals) <= 0:
                continue
            threshold = np.quantile(neg_signals[neg_signals > 0], 1 - q) if np.sum(neg_signals > 0) > 0 else float('inf')
            for pi in range(n_preds):
                if signal[pi] < 0 and abs(signal[pi]) >= threshold:
                    candidates.append((pi, 'short'))

        elif mode == 'asymmetric':
            # Different thresholds for long vs short
            pos_sigs = signal[signal > 0]
            neg_sigs = np.abs(signal[signal < 0])
            long_thresh = np.quantile(pos_sigs, 1 - q) if len(pos_sigs) > 0 else float('inf')
            short_thresh = np.quantile(neg_sigs, 1 - q_short) if len(neg_sigs) > 0 else float('inf')
            for pi in range(n_preds):
                if signal[pi] > 0 and signal[pi] >= long_thresh:
                    candidates.append((pi, 'long'))
                elif signal[pi] < 0 and abs(signal[pi]) >= short_thresh:
                    candidates.append((pi, 'short'))

        else:  # 'both'
            abs_signal = np.abs(signal)
            threshold = np.quantile(abs_signal, 1 - q)
            for pi in range(n_preds):
                if abs(signal[pi]) >= threshold:
                    direction = 'long' if signal[pi] > 0 else 'short'
                    candidates.append((pi, direction))

        if not candidates:
            continue

        # If randomizing timing, shuffle which prediction indices we use
        # but keep same count and direction distribution
        if randomize_timing and rng is not None:
            n_cands = len(candidates)
            directions = [c[1] for c in candidates]
            # Pick random prediction indices
            valid_indices = [pi for pi in range(n_preds) if pi * PRED_STRIDE < n_events - 100]
            if len(valid_indices) >= n_cands:
                random_pis = rng.choice(valid_indices, size=n_cands, replace=False)
                candidates = list(zip(random_pis, directions))

        # Execute trades
        for pi, direction in candidates:
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
                tp_price = entry_price + tp * TICK
                sl_price = entry_price - sl * TICK
            else:
                tp_price = entry_price - tp * TICK
                sl_price = entry_price + sl * TICK

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

                unrealized = ((p - entry_price) / TICK) if direction == 'long' else ((entry_price - p) / TICK)
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

            gross = ((exit_price - entry_price) / TICK) if direction == 'long' else ((entry_price - exit_price) / TICK)
            cost = 0.376 if exit_reason == 'tp' else 1.376
            net = gross - cost

            trades.append({
                'date': date,
                'direction': direction,
                'signal': float(signal[pi]) if pi < len(signal) else 0.0,
                'gross': float(gross),
                'net': float(net),
                'exit_reason': exit_reason,
                'mfe': float(mfe),
                'mae': float(mae),
            })

    return trades


def analyze_trades(trades, label, run_perm=True):
    """Analyze trade results and run permutation test."""
    n = len(trades)
    if n < 5:
        print(f"  {label}: Only {n} trades — insufficient")
        return None

    nets = np.array([t['net'] for t in trades])
    real_mean = float(np.mean(nets))
    real_total = float(np.sum(nets))
    wr = float(np.mean(nets > 0))
    sharpe = float(np.mean(nets) / np.std(nets) * np.sqrt(252)) if np.std(nets) > 0 else 0

    # Per-day
    day_data = defaultdict(lambda: {'net': 0, 'trades': 0})
    for t in trades:
        day_data[t['date']]['net'] += t['net']
        day_data[t['date']]['trades'] += 1

    green = sum(1 for d in day_data.values() if d['net'] > 0)
    red = sum(1 for d in day_data.values() if d['net'] < 0)
    day_nets = [d['net'] for d in day_data.values()]
    day_sharpe = float(np.mean(day_nets) / np.std(day_nets) * np.sqrt(252)) if len(day_nets) > 1 and np.std(day_nets) > 0 else 0

    # Exit breakdown
    exits = {}
    for t in trades:
        exits[t['exit_reason']] = exits.get(t['exit_reason'], 0) + 1

    # MFE/MAE
    avg_mfe = float(np.mean([t['mfe'] for t in trades]))
    p90_mfe = float(np.percentile([t['mfe'] for t in trades], 90))
    avg_mae = float(np.mean([t['mae'] for t in trades]))

    # Long/short split
    longs = [t for t in trades if t['direction'] == 'long']
    shorts = [t for t in trades if t['direction'] == 'short']

    print(f"\n  {label}")
    print(f"    Trades: {n} ({n/len(all_data):.1f}/day)")
    print(f"    Net/trade: {real_mean:+.4f} ticks | Total: {real_total:+.1f} ticks")
    print(f"    WR: {wr:.1%} | Sharpe: {sharpe:+.2f} | Day Sharpe: {day_sharpe:+.2f}")
    print(f"    Days: {green}G/{red}R (day WR: {green/(green+red)*100:.0f}%)")
    print(f"    Exits: {exits}")
    print(f"    MFE avg={avg_mfe:.1f} p90={p90_mfe:.1f} | MAE avg={avg_mae:.1f}")
    if longs:
        print(f"    Longs: {len(longs)} trades, net={np.mean([t['net'] for t in longs]):+.3f}")
    if shorts:
        print(f"    Shorts: {len(shorts)} trades, net={np.mean([t['net'] for t in shorts]):+.3f}")

    result = {
        'label': label,
        'n_trades': n,
        'trades_per_day': round(n / len(all_data), 1),
        'net_per_trade': round(real_mean, 4),
        'total_net_ticks': round(real_total, 1),
        'win_rate': round(wr, 3),
        'sharpe': round(sharpe, 2),
        'day_sharpe': round(day_sharpe, 2),
        'green_days': green,
        'red_days': red,
        'day_wr': round(green / max(green + red, 1), 3),
        'exits': exits,
        'avg_mfe': round(avg_mfe, 1),
        'p90_mfe': round(p90_mfe, 1),
        'avg_mae': round(avg_mae, 1),
        'n_longs': len(longs),
        'n_shorts': len(shorts),
        'long_net': round(float(np.mean([t['net'] for t in longs])), 4) if longs else None,
        'short_net': round(float(np.mean([t['net'] for t in shorts])), 4) if shorts else None,
    }

    if not run_perm:
        result['perm_verdict'] = 'SKIPPED'
        return result

    # PERMUTATION TEST
    # For long-only: randomize TIMING (not direction)
    # This tests: does the MODEL know WHEN to go long? Or is any random long timing equally good?
    print(f"    Running {N_PERMS} permutations (randomized timing)...")
    perm_means = []
    t0 = time.time()
    for pi in range(N_PERMS):
        perm_trades = run_config(all_data, cfg_for_perm, mode=mode_for_perm,
                                  randomize_timing=True, seed=pi * 42 + 7)
        if len(perm_trades) > 0:
            perm_nets = [t['net'] for t in perm_trades]
            perm_means.append(float(np.mean(perm_nets)))
        if (pi + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"      {pi+1}/{N_PERMS} ({elapsed:.0f}s)")

    if perm_means:
        perm_mean = float(np.mean(perm_means))
        p_value = float(np.mean([p >= real_mean for p in perm_means]))

        print(f"    PERMUTATION: real={real_mean:+.4f} vs random={perm_mean:+.4f} p={p_value:.3f}")

        # For long-only: if random longs are also profitable, that's a bullish-bias issue
        # The test is whether MODEL-TIMED longs beat RANDOM-TIMED longs
        if perm_mean > 0:
            print(f"    ⚠️  Random longs also profitable ({perm_mean:+.4f}) — bullish sample bias")
            # Adjust: real edge = real_mean - perm_mean (excess over random)
            excess = real_mean - perm_mean
            print(f"    Excess over random: {excess:+.4f} ticks/trade")
            verdict = 'PASS' if p_value < 0.05 else 'FAIL'
        else:
            verdict = 'PASS' if p_value < 0.05 else 'FAIL'

        result['perm_p'] = round(p_value, 3)
        result['perm_mean'] = round(perm_mean, 4)
        result['perm_excess'] = round(real_mean - perm_mean, 4)
        result['perm_verdict'] = verdict

        status = '✅' if verdict == 'PASS' else '❌'
        print(f"    {status} VERDICT: {verdict}")
    else:
        result['perm_verdict'] = 'FAIL'
        result['perm_p'] = 1.0

    return result


# ============================================================
# TEST SUITE
# ============================================================

all_results = []

# --- SECTION A: LONG-ONLY at various confidence thresholds ---
print(f"\n{'='*70}")
print(f"SECTION A: LONG-ONLY (model says UP + high confidence)")
print(f"{'='*70}")

long_only_configs = [
    {'head': 'composite_signal', 'quantile': 0.05, 'tp': 12, 'sl': 8, 'hold_s': 30, 'cancel_s': 10},
    {'head': 'composite_signal', 'quantile': 0.10, 'tp': 12, 'sl': 8, 'hold_s': 30, 'cancel_s': 10},
    {'head': 'composite_signal', 'quantile': 0.15, 'tp': 12, 'sl': 8, 'hold_s': 30, 'cancel_s': 10},
    {'head': 'composite_signal', 'quantile': 0.05, 'tp': 8, 'sl': 4, 'hold_s': 30, 'cancel_s': 10},
    {'head': 'composite_signal', 'quantile': 0.10, 'tp': 8, 'sl': 4, 'hold_s': 30, 'cancel_s': 10},
    {'head': 'composite_signal', 'quantile': 0.05, 'tp': 10, 'sl': 6, 'hold_s': 30, 'cancel_s': 10},
    {'head': 'composite_signal', 'quantile': 0.10, 'tp': 10, 'sl': 6, 'hold_s': 30, 'cancel_s': 10},
    # Tight: TP6/SL4
    {'head': 'composite_signal', 'quantile': 0.05, 'tp': 6, 'sl': 4, 'hold_s': 30, 'cancel_s': 10},
    {'head': 'composite_signal', 'quantile': 0.10, 'tp': 6, 'sl': 4, 'hold_s': 30, 'cancel_s': 10},
]

for cfg in long_only_configs:
    label = f"LONG_q{int(cfg['quantile']*100)}_tp{cfg['tp']}sl{cfg['sl']}"
    cfg_for_perm = cfg
    mode_for_perm = 'long_only'

    trades = run_config(all_data, cfg, mode='long_only')
    result = analyze_trades(trades, label, run_perm=True)
    if result:
        result['mode'] = 'long_only'
        result['config'] = {k: v for k, v in cfg.items()}
        all_results.append(result)


# --- SECTION B: SHORT-ONLY (for comparison — expect to fail) ---
print(f"\n{'='*70}")
print(f"SECTION B: SHORT-ONLY (model says DOWN + high confidence)")
print(f"{'='*70}")

short_only_configs = [
    {'head': 'composite_signal', 'quantile': 0.05, 'tp': 12, 'sl': 8, 'hold_s': 30, 'cancel_s': 10},
    {'head': 'composite_signal', 'quantile': 0.10, 'tp': 12, 'sl': 8, 'hold_s': 30, 'cancel_s': 10},
    {'head': 'composite_signal', 'quantile': 0.05, 'tp': 8, 'sl': 4, 'hold_s': 30, 'cancel_s': 10},
]

for cfg in short_only_configs:
    label = f"SHORT_q{int(cfg['quantile']*100)}_tp{cfg['tp']}sl{cfg['sl']}"
    cfg_for_perm = cfg
    mode_for_perm = 'short_only'

    trades = run_config(all_data, cfg, mode='short_only')
    result = analyze_trades(trades, label, run_perm=True)
    if result:
        result['mode'] = 'short_only'
        result['config'] = {k: v for k, v in cfg.items()}
        all_results.append(result)


# --- SECTION C: INDIVIDUAL HEADS (not just composite) ---
print(f"\n{'='*70}")
print(f"SECTION C: INDIVIDUAL HEADS — LONG-ONLY")
print(f"{'='*70}")

# Check what heads are available
sample_date = list(all_data.keys())[0]
available_heads = list(all_data[sample_date]['preds'].keys())
print(f"Available heads: {available_heads}")

# Test each head with best config from Section A
for head in available_heads:
    if head in ('composite_signal',):
        continue  # already tested above
    cfg = {'head': head, 'quantile': 0.10, 'tp': 12, 'sl': 8, 'hold_s': 30, 'cancel_s': 10}
    label = f"LONG_{head}_q10_tp12sl8"
    cfg_for_perm = cfg
    mode_for_perm = 'long_only'

    trades = run_config(all_data, cfg, mode='long_only')
    result = analyze_trades(trades, label, run_perm=False)  # No perm for screening
    if result:
        result['mode'] = 'long_only'
        result['config'] = {k: v for k, v in cfg.items()}
        all_results.append(result)


# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*70}")
print(f"FINAL SUMMARY — ALL CONFIGS")
print(f"{'='*70}")

# Sort by net/trade descending
sorted_results = sorted(all_results, key=lambda x: x['net_per_trade'], reverse=True)

for r in sorted_results:
    pv = r.get('perm_verdict', '?')
    status = '✅' if pv == 'PASS' else ('⏭️' if pv == 'SKIPPED' else '❌')
    excess = r.get('perm_excess', '')
    excess_str = f" excess={excess:+.4f}" if isinstance(excess, (int, float)) else ''
    print(f"  {status} {r['label']:35s} n={r['n_trades']:4d} net={r['net_per_trade']:+.4f} "
          f"WR={r['win_rate']:.1%} Sharpe={r['sharpe']:+.2f} DayS={r['day_sharpe']:+.2f} "
          f"p={r.get('perm_p', '-'):>5}{excess_str} → {pv}")

# Count passes
passes = [r for r in all_results if r.get('perm_verdict') == 'PASS']
print(f"\n{len(passes)}/{len(all_results)} configs PASSED permutation test")

if passes:
    print(f"\n{'='*70}")
    print(f"PASSING CONFIGS (GENUINE MODEL EDGE)")
    print(f"{'='*70}")
    for r in passes:
        print(f"  ✅ {r['label']}")
        print(f"     {r['n_trades']} trades, {r['net_per_trade']:+.4f} net/trade, "
              f"Sharpe {r['sharpe']:+.2f}, Day Sharpe {r['day_sharpe']:+.2f}")
        print(f"     Permutation: p={r['perm_p']:.3f}, random_mean={r['perm_mean']:+.4f}, "
              f"excess={r['perm_excess']:+.4f}")
        print(f"     Days: {r['green_days']}G/{r['red_days']}R ({r['day_wr']:.0%})")

# Save results
with open(OUTPUT / 'v24_results.json', 'w') as f:
    json.dump({
        'version': 'v24_longonly',
        'n_dates': len(all_data),
        'n_perms': N_PERMS,
        'hypothesis': 'Long-side has genuine model edge, shorts dilute. Test timing value.',
        'results': all_results,
    }, f, indent=2, default=str)

print(f"\nSaved to {OUTPUT}/v24_results.json")
print("Done.")
