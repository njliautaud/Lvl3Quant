#!/usr/bin/env python3
"""
Tick-Level Replay v10b: NUMBA-ACCELERATED with ALIGNED PREDICTIONS
===================================================================

v10 was correct but used pure Python (~days to finish).
v10b uses the EXISTING Numba kernel (_simulate_day_numba) which already
accepts custom pred_event_indices. Just pass aligned indices directly.

v10 fix recap: v8/v9 FAILED because predictions were mapped to wrong
event indices (training had ~280K extra pre-RTH events). The aligned
indices fix this by timestamp-matching predictions to MBO events.

HC #659 compliant: tick-level, permutation test on any positive result.
"""

import numpy as np
import os
import sys
import time
import json
from collections import defaultdict

sys.path.insert(0, '/home/jupiter/Lvl3Quant/engines')

# Import the fast engine directly
from tick_replay_fast import (
    HAS_NUMBA, N_TRADE_COLS, TICK_SIZE,
    COST_PASSIVE_EXIT, COST_MARKET_EXIT,
    EXIT_TP, EXIT_SL, EXIT_TIME, EXIT_EOD
)

if HAS_NUMBA:
    from tick_replay_fast import _simulate_day_numba
    print("✓ Numba engine loaded")
else:
    from tick_replay_fast import _simulate_day_python as _simulate_day_numba
    print("⚠ Numba not available, using Python fallback")

PREPROC_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo'
PRED_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/oot_47day_perdate'
ALIGNED_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo/pred_indices_aligned'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/tick_replay_v10b_numba'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_day_data(date_str):
    """Load and cache MBO arrays + aligned predictions for a date."""
    aligned_path = os.path.join(ALIGNED_DIR, f'aligned_{date_str}.npz')
    pred_path = os.path.join(PRED_DIR, f'oot_{date_str}.npz')
    preproc_path = os.path.join(PREPROC_DIR, f'mbo_{date_str}.npz')

    if not all(os.path.exists(p) for p in [aligned_path, pred_path, preproc_path]):
        return None

    # Load MBO
    mbo = np.load(preproc_path)
    ts_ns = mbo['ts_ns']
    action = mbo['action']
    side = mbo['side']
    price = mbo['price']
    size = mbo['size'].astype(np.int32)
    order_id = mbo['order_id']

    # Load aligned prediction indices
    aligned = np.load(aligned_path)
    event_indices = aligned['event_indices']
    in_range = aligned['in_mbo_range']

    # Load predictions
    pred_file = np.load(pred_path)
    if 'pred_log_ret_1s' in pred_file:
        preds = pred_file['pred_log_ret_1s'].astype(np.float64)
    elif 'predictions' in pred_file:
        preds = pred_file['predictions'].astype(np.float64)
    else:
        return None

    # Filter to in-range only
    valid_preds = preds[in_range]
    valid_indices = event_indices[in_range].astype(np.int64)

    # Further filter to events within MBO array bounds
    n_events = len(ts_ns)
    bounds_mask = (valid_indices >= 0) & (valid_indices < n_events)
    valid_preds = valid_preds[bounds_mask]
    valid_indices = valid_indices[bounds_mask]

    return {
        'ts_ns': ts_ns, 'action': action, 'side': side,
        'price': price, 'size': size, 'order_id': order_id,
        'predictions': valid_preds, 'event_indices': valid_indices,
        'n_events': n_events, 'n_preds': len(valid_preds)
    }


def run_config(day_data_list, tp_ticks, sl_ticks, hold_seconds, cancel_seconds,
               signal_threshold, max_concurrent, short_only, long_only):
    """Run a single config across all dates using Numba engine."""
    hold_ns = int(hold_seconds * 1e9)
    cancel_ns = int(cancel_seconds * 1e9)

    all_trades = []
    daily_pnls = []

    for dd in day_data_list:
        preds = dd['predictions'].copy()
        indices = dd['event_indices'].copy()

        # Direction filter
        if short_only:
            mask = preds < -signal_threshold
        elif long_only:
            mask = preds > signal_threshold
        else:
            mask = np.abs(preds) > signal_threshold

        filtered_preds = preds[mask]
        filtered_indices = indices[mask]

        if len(filtered_preds) == 0:
            daily_pnls.append(0.0)
            continue

        # Sort by index
        sort_idx = np.argsort(filtered_indices)
        filtered_preds = filtered_preds[sort_idx]
        filtered_indices = filtered_indices[sort_idx]

        # Call Numba kernel directly with aligned indices
        trades = _simulate_day_numba(
            dd['ts_ns'], dd['action'], dd['side'], dd['price'],
            dd['size'], dd['order_id'],
            filtered_preds, filtered_indices,
            tp_ticks, sl_ticks, hold_ns, cancel_ns,
            signal_threshold, max_concurrent
        )

        if len(trades) > 0:
            all_trades.append(trades)
            daily_pnls.append(trades[:, 10].sum())
        else:
            daily_pnls.append(0.0)

    return all_trades, np.array(daily_pnls)


def permutation_test(day_data_list, tp_ticks, sl_ticks, hold_seconds, cancel_seconds,
                     signal_threshold, max_concurrent, short_only, long_only,
                     n_perms=100, real_sharpe=0.0):
    """
    HC #659 R3: Random direction permutation test.
    Flip prediction signs randomly. If random is also profitable -> artifact.
    """
    hold_ns = int(hold_seconds * 1e9)
    cancel_ns = int(cancel_seconds * 1e9)

    random_sharpes = []

    for perm_i in range(n_perms):
        daily_pnls = []
        for dd in day_data_list:
            preds = dd['predictions'].copy()
            # Random sign flip
            signs = np.random.choice([-1.0, 1.0], size=len(preds))
            preds = preds * signs

            indices = dd['event_indices'].copy()

            if short_only:
                mask = preds < -signal_threshold
            elif long_only:
                mask = preds > signal_threshold
            else:
                mask = np.abs(preds) > signal_threshold

            filtered_preds = preds[mask]
            filtered_indices = indices[mask]

            if len(filtered_preds) == 0:
                daily_pnls.append(0.0)
                continue

            sort_idx = np.argsort(filtered_indices)
            filtered_preds = filtered_preds[sort_idx]
            filtered_indices = filtered_indices[sort_idx]

            trades = _simulate_day_numba(
                dd['ts_ns'], dd['action'], dd['side'], dd['price'],
                dd['size'], dd['order_id'],
                filtered_preds, filtered_indices,
                tp_ticks, sl_ticks, hold_ns, cancel_ns,
                signal_threshold, max_concurrent
            )

            if len(trades) > 0:
                daily_pnls.append(trades[:, 10].sum())
            else:
                daily_pnls.append(0.0)

        daily_arr = np.array(daily_pnls)
        if daily_arr.std() > 0:
            s = daily_arr.mean() / daily_arr.std() * np.sqrt(252)
        else:
            s = 0.0
        random_sharpes.append(s)

    random_sharpes = np.array(random_sharpes)
    p_value = (random_sharpes >= real_sharpe).mean()
    return p_value, random_sharpes.mean(), random_sharpes.std()


def main():
    print("=" * 70)
    print("TICK REPLAY v10b — NUMBA ACCELERATED, ALIGNED PREDICTIONS")
    print("=" * 70)

    # Discover dates
    dates = sorted([f[8:16] for f in os.listdir(ALIGNED_DIR) if f.startswith('aligned_')])
    print(f"Loading {len(dates)} dates...")

    # Pre-load all day data (saves repeated I/O)
    day_data_list = []
    for date_str in dates:
        dd = load_day_data(date_str)
        if dd is not None:
            day_data_list.append(dd)
            print(f"  {date_str}: {dd['n_events']:,} events, {dd['n_preds']:,} predictions")

    print(f"\nLoaded {len(day_data_list)} dates successfully")
    total_preds = sum(dd['n_preds'] for dd in day_data_list)
    print(f"Total predictions: {total_preds:,}")
    print()

    # Config sweep — focused on high-selectivity short-only (where edge was shown)
    configs = [
        # (tp, sl, hold_s, cancel_s, threshold, short_only, long_only, max_conc, label)
        # High-selectivity shorts (most promising from alignment validation)
        (2, 3, 5, 5, 0.6, True, False, 4, "S_TP2_SL3_h5_thr0.6"),
        (2, 4, 5, 5, 0.6, True, False, 4, "S_TP2_SL4_h5_thr0.6"),
        (3, 4, 5, 5, 0.6, True, False, 4, "S_TP3_SL4_h5_thr0.6"),
        (3, 6, 8, 8, 0.6, True, False, 4, "S_TP3_SL6_h8_thr0.6"),
        (2, 3, 5, 5, 0.8, True, False, 4, "S_TP2_SL3_h5_thr0.8"),
        (2, 4, 5, 5, 0.8, True, False, 4, "S_TP2_SL4_h5_thr0.8"),
        (3, 4, 5, 5, 0.8, True, False, 4, "S_TP3_SL4_h5_thr0.8"),
        (3, 6, 8, 8, 0.8, True, False, 4, "S_TP3_SL6_h8_thr0.8"),
        (4, 6, 10, 8, 0.8, True, False, 4, "S_TP4_SL6_h10_thr0.8"),
        (2, 3, 5, 5, 1.0, True, False, 4, "S_TP2_SL3_h5_thr1.0"),
        (3, 4, 5, 5, 1.0, True, False, 4, "S_TP3_SL4_h5_thr1.0"),
        (4, 6, 10, 8, 1.0, True, False, 4, "S_TP4_SL6_h10_thr1.0"),
        # Long-only (weaker edge expected)
        (2, 3, 5, 5, 0.6, False, True, 4, "L_TP2_SL3_h5_thr0.6"),
        (3, 4, 5, 5, 0.6, False, True, 4, "L_TP3_SL4_h5_thr0.6"),
        (2, 3, 5, 5, 0.8, False, True, 4, "L_TP2_SL3_h5_thr0.8"),
        (3, 4, 5, 5, 0.8, False, True, 4, "L_TP3_SL4_h5_thr0.8"),
        # Both directions
        (2, 3, 5, 5, 0.6, False, False, 4, "B_TP2_SL3_h5_thr0.6"),
        (3, 4, 5, 5, 0.8, False, False, 4, "B_TP3_SL4_h5_thr0.8"),
    ]

    print(f"Testing {len(configs)} configs across {len(day_data_list)} dates")
    print("=" * 70)
    print()

    results = {}
    promising = []  # Configs that pass initial filter for permutation test

    for ci, (tp, sl, hold_s, cancel_s, thr, short_only, long_only, max_conc, label) in enumerate(configs):
        t0 = time.time()

        all_trades, daily_pnls = run_config(
            day_data_list, tp, sl, hold_s, cancel_s, thr, max_conc, short_only, long_only
        )

        elapsed = time.time() - t0

        if not all_trades:
            print(f"[{ci+1}/{len(configs)}] {label}: NO TRADES [{elapsed:.1f}s]")
            results[label] = {'n_trades': 0, 'status': 'no_trades'}
            continue

        trades = np.vstack(all_trades)
        n_trades = len(trades)
        n_per_day = n_trades / len(day_data_list)

        pnl = trades[:, 10]
        total_pnl = pnl.sum()
        winners = pnl > 0
        wr = winners.mean()

        gross_win = pnl[winners].sum() if winners.any() else 0
        gross_loss = abs(pnl[~winners].sum()) if (~winners).any() else 1e-9
        pf = gross_win / gross_loss

        sharpe = (daily_pnls.mean() / daily_pnls.std() * np.sqrt(252)) if daily_pnls.std() > 0 else 0

        green = (daily_pnls > 0).sum()
        red = (daily_pnls < 0).sum()

        # Exit reason breakdown
        exits = trades[:, 7]
        tp_pct = (exits == 0).mean() * 100
        sl_pct = (exits == 1).mean() * 100
        time_pct = (exits == 2).mean() * 100

        results[label] = {
            'n_trades': int(n_trades), 'per_day': round(float(n_per_day), 1),
            'total_pnl_ticks': round(float(total_pnl), 1),
            'wr': round(float(wr), 4), 'pf': round(float(pf), 3),
            'sharpe': round(float(sharpe), 2),
            'green_days': int(green), 'red_days': int(red),
            'tp_pct': round(float(tp_pct), 1),
            'sl_pct': round(float(sl_pct), 1),
            'time_pct': round(float(time_pct), 1),
        }

        status = "✓" if sharpe > 0 and pf > 1.0 else "✗"
        print(f"[{ci+1}/{len(configs)}] {status} {label}")
        print(f"  {n_trades} trades ({n_per_day:.0f}/day), PnL={total_pnl:+.1f}t, "
              f"WR={wr:.3f}, PF={pf:.3f}, Sharpe={sharpe:+.2f}")
        print(f"  Exits: TP={tp_pct:.0f}% SL={sl_pct:.0f}% Time={time_pct:.0f}% | "
              f"Days: {green}G/{red}R  [{elapsed:.1f}s]")
        print()

        # Mark promising configs for permutation test
        if sharpe > 0.5 and pf > 1.0 and wr > 0.45:
            promising.append((ci, tp, sl, hold_s, cancel_s, thr, short_only, long_only, max_conc, label))

    # Save intermediate results
    out_path = os.path.join(OUTPUT_DIR, 'v10b_sweep_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # =====================================================================
    # PERMUTATION TEST on promising configs (HC #659 R3)
    # =====================================================================
    if promising:
        print()
        print("=" * 70)
        print(f"PERMUTATION TEST — {len(promising)} promising configs (100 trials each)")
        print("=" * 70)
        print()

        for ci, tp, sl, hold_s, cancel_s, thr, short_only, long_only, max_conc, label in promising:
            real_sharpe = results[label]['sharpe']
            print(f"Testing {label} (real Sharpe={real_sharpe:+.2f})...")
            t0 = time.time()

            p_val, mean_rand, std_rand = permutation_test(
                day_data_list, tp, sl, hold_s, cancel_s, thr, max_conc,
                short_only, long_only,
                n_perms=100, real_sharpe=real_sharpe
            )

            elapsed = time.time() - t0
            results[label]['perm_p_value'] = round(float(p_val), 4)
            results[label]['perm_random_sharpe_mean'] = round(float(mean_rand), 2)
            results[label]['perm_random_sharpe_std'] = round(float(std_rand), 2)

            verdict = "REAL EDGE ✓✓✓" if p_val < 0.05 else "ARTIFACT ✗"
            print(f"  p={p_val:.3f} | Random Sharpe: {mean_rand:+.2f} ± {std_rand:.2f} | "
                  f"{verdict} [{elapsed:.0f}s]")
            print()

    # Final summary
    print()
    print("=" * 70)
    print("FINAL SUMMARY (sorted by Sharpe)")
    print("=" * 70)
    print(f"{'Config':<30} {'Trades':>7} {'PnL':>8} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'G/R':>5} {'p-val':>6}")
    print("-" * 78)
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v.get('n_trades', 0) > 0],
        key=lambda x: x[1].get('sharpe', -999), reverse=True
    )
    for label, r in sorted_results:
        p_str = f"{r['perm_p_value']:.3f}" if 'perm_p_value' in r else "  —"
        print(f"{label:<30} {r['n_trades']:>7} {r['total_pnl_ticks']:>+8.0f} "
              f"{r['wr']:>6.3f} {r['pf']:>6.3f} {r['sharpe']:>+7.2f} "
              f"{r['green_days']}/{r['red_days']:>2} {p_str:>6}")

    # Save final results
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")

    # Flag any validated configs
    validated = [k for k, v in results.items()
                 if v.get('perm_p_value', 1.0) < 0.05 and v.get('sharpe', 0) > 0.5]
    if validated:
        print(f"\n🎯 VALIDATED CONFIGS (p<0.05, Sharpe>0.5): {validated}")
        # Save separately for easy access
        val_path = os.path.join(OUTPUT_DIR, 'VALIDATED_CONFIGS.json')
        with open(val_path, 'w') as f:
            json.dump({k: results[k] for k in validated}, f, indent=2)
        print(f"Saved to {val_path}")
    else:
        print("\n❌ No configs passed permutation test. Model edge may not survive FIFO fills.")


if __name__ == "__main__":
    main()
