#!/usr/bin/env python3
"""
Tick-Level Replay v11: 5-SECOND HORIZON PREDICTIONS
====================================================

v10b/v10d proved: 1s predictions (IC=0.253) can't overcome tick-level costs
because expected 1s moves (~1 tick) are too small relative to RT costs (1.75-2.75 ticks).

v11 tests: 5s predictions (IC=0.129 single-day) with wider TP/SL and longer hold.
5s moves are ~2-3× larger → TP=6-8 ticks might work.

Additionally tests pred_fifo_tp8sl5_hit_tp (IC=0.138) as a trade filter.

Uses existing Numba engine (FIFO passive entry) with aligned indices.
HC #659 compliant: tick-level replay, permutation test on positive results.
"""

import numpy as np
import os
import sys
import time
import json
from collections import defaultdict

sys.path.insert(0, '/home/jupiter/Lvl3Quant/engines')
from tick_replay_fast import _simulate_day_numba, HAS_NUMBA, N_TRADE_COLS

if not HAS_NUMBA:
    from tick_replay_fast import _simulate_day_python as _simulate_day_numba

PREPROC_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo'
PRED_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/oot_47day_perdate'
ALIGNED_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo/pred_indices_aligned'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/tick_replay_v11_5s'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_day_data(date_str, pred_head='pred_log_ret_5s', filter_head=None):
    """Load MBO + aligned predictions for a specific head."""
    aligned_path = os.path.join(ALIGNED_DIR, f'aligned_{date_str}.npz')
    pred_path = os.path.join(PRED_DIR, f'oot_{date_str}.npz')
    preproc_path = os.path.join(PREPROC_DIR, f'mbo_{date_str}.npz')

    if not all(os.path.exists(p) for p in [aligned_path, pred_path, preproc_path]):
        return None

    mbo = np.load(preproc_path)
    ts_ns = mbo['ts_ns']
    action = mbo['action']
    side = mbo['side']
    price = mbo['price']
    size = mbo['size'].astype(np.int32)
    order_id = mbo['order_id']

    aligned = np.load(aligned_path)
    event_indices = aligned['event_indices']
    in_range = aligned['in_mbo_range']

    pred_file = np.load(pred_path)
    if pred_head not in pred_file:
        return None
    preds = pred_file[pred_head].astype(np.float64)

    # Optional filter head (e.g., fifo_tp8sl5_hit_tp)
    filter_vals = None
    if filter_head and filter_head in pred_file:
        filter_vals = pred_file[filter_head].astype(np.float64)
        filter_vals = filter_vals[in_range]

    valid_preds = preds[in_range]
    valid_indices = event_indices[in_range].astype(np.int64)

    n_events = len(ts_ns)
    bounds_mask = (valid_indices >= 0) & (valid_indices < n_events)
    valid_preds = valid_preds[bounds_mask]
    valid_indices = valid_indices[bounds_mask]
    if filter_vals is not None:
        filter_vals = filter_vals[bounds_mask]

    sort_idx = np.argsort(valid_indices)
    valid_preds = valid_preds[sort_idx]
    valid_indices = valid_indices[sort_idx]
    if filter_vals is not None:
        filter_vals = filter_vals[sort_idx]

    return {
        'ts_ns': ts_ns, 'action': action, 'side': side,
        'price': price, 'size': size, 'order_id': order_id,
        'predictions': valid_preds, 'event_indices': valid_indices,
        'filter_vals': filter_vals,
    }


def run_config(day_data_list, tp_ticks, sl_ticks, hold_seconds, cancel_seconds,
               signal_threshold, max_concurrent, short_only, long_only,
               filter_threshold=None):
    """Run config using Numba FIFO engine."""
    hold_ns = int(hold_seconds * 1e9)
    cancel_ns = int(cancel_seconds * 1e9)
    all_trades = []
    daily_pnls = []

    for dd in day_data_list:
        preds = dd['predictions'].copy()
        indices = dd['event_indices'].copy()

        # Apply filter if present
        if filter_threshold is not None and dd['filter_vals'] is not None:
            # Only keep predictions where filter_vals > threshold
            # (filter_head predicts probability of TP hit — higher = better)
            keep = dd['filter_vals'] > filter_threshold
            preds = preds[keep]
            indices = indices[keep]

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
            all_trades.append(trades)
            daily_pnls.append(trades[:, 10].sum())
        else:
            daily_pnls.append(0.0)

    return all_trades, np.array(daily_pnls)


def permutation_test(day_data_list, tp_ticks, sl_ticks, hold_seconds, cancel_seconds,
                     signal_threshold, max_concurrent, short_only, long_only,
                     filter_threshold, n_perms=100, real_sharpe=0.0):
    """HC #659 R3: Permutation test."""
    hold_ns = int(hold_seconds * 1e9)
    cancel_ns = int(cancel_seconds * 1e9)
    random_sharpes = []

    for _ in range(n_perms):
        daily_pnls = []
        for dd in day_data_list:
            preds = dd['predictions'].copy()
            signs = np.random.choice([-1.0, 1.0], size=len(preds))
            preds = preds * signs
            indices = dd['event_indices'].copy()

            if filter_threshold is not None and dd['filter_vals'] is not None:
                keep = dd['filter_vals'] > filter_threshold
                preds = preds[keep]
                indices = indices[keep]

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
            daily_pnls.append(trades[:, 10].sum() if len(trades) > 0 else 0.0)

        daily_arr = np.array(daily_pnls)
        s = (daily_arr.mean() / daily_arr.std() * np.sqrt(252)) if daily_arr.std() > 0 else 0.0
        random_sharpes.append(s)

    random_sharpes = np.array(random_sharpes)
    return (random_sharpes >= real_sharpe).mean(), random_sharpes.mean(), random_sharpes.std()


def main():
    print("=" * 70, flush=True)
    print("TICK REPLAY v11 — 5-SECOND HORIZON + FIFO FILTER", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print("Rationale: 1s predictions (IC=0.25) failed because moves too small.", flush=True)
    print("5s predictions (IC=0.13) have 2× larger moves → wider TP feasible.", flush=True)
    print("Also testing fifo_tp8sl5_hit_tp (IC=0.14) as trade quality filter.", flush=True)
    print(flush=True)

    dates = sorted([f[8:16] for f in os.listdir(ALIGNED_DIR) if f.startswith('aligned_')])
    print(f"Loading {len(dates)} dates (5s head + fifo filter)...", flush=True)

    day_data_list = []
    for date_str in dates:
        dd = load_day_data(date_str, pred_head='pred_log_ret_5s',
                          filter_head='pred_fifo_tp8sl5_hit_tp')
        if dd is not None and len(dd['predictions']) > 1000:
            day_data_list.append(dd)

    print(f"Loaded {len(day_data_list)} dates", flush=True)
    total_preds = sum(len(dd['predictions']) for dd in day_data_list)
    print(f"Total predictions: {total_preds:,}", flush=True)
    print(flush=True)

    # Configs: 5s horizon → hold 10-30s, TP 4-10 ticks, SL 4-8 ticks
    # Also test with/without fifo filter
    configs = [
        # (tp, sl, hold_s, cancel_s, threshold, short_only, long_only, max_conc, filter_thr, label)
        # 5s predictions, short-only (stronger edge historically)
        (4, 4, 10, 8, 0.3, True, False, 4, None, "5s_S_TP4_SL4_h10_thr0.3"),
        (6, 4, 15, 10, 0.3, True, False, 4, None, "5s_S_TP6_SL4_h15_thr0.3"),
        (8, 5, 20, 15, 0.3, True, False, 4, None, "5s_S_TP8_SL5_h20_thr0.3"),
        (6, 4, 15, 10, 0.5, True, False, 4, None, "5s_S_TP6_SL4_h15_thr0.5"),
        (8, 5, 20, 15, 0.5, True, False, 4, None, "5s_S_TP8_SL5_h20_thr0.5"),
        (8, 5, 20, 15, 0.8, True, False, 4, None, "5s_S_TP8_SL5_h20_thr0.8"),
        (10, 6, 30, 20, 0.5, True, False, 4, None, "5s_S_TP10_SL6_h30_thr0.5"),
        (10, 6, 30, 20, 0.8, True, False, 4, None, "5s_S_TP10_SL6_h30_thr0.8"),
        # With FIFO filter (pred_fifo_tp8sl5_hit_tp > threshold)
        (8, 5, 20, 15, 0.3, True, False, 4, -2.5, "5s_S_TP8_SL5_h20_t0.3_filt-2.5"),
        (8, 5, 20, 15, 0.3, True, False, 4, -2.0, "5s_S_TP8_SL5_h20_t0.3_filt-2.0"),
        (8, 5, 20, 15, 0.5, True, False, 4, -2.5, "5s_S_TP8_SL5_h20_t0.5_filt-2.5"),
        (8, 5, 20, 15, 0.5, True, False, 4, -2.0, "5s_S_TP8_SL5_h20_t0.5_filt-2.0"),
        # Both directions
        (6, 4, 15, 10, 0.3, False, False, 4, None, "5s_B_TP6_SL4_h15_thr0.3"),
        (8, 5, 20, 15, 0.5, False, False, 4, None, "5s_B_TP8_SL5_h20_thr0.5"),
        # Long only (weaker edge)
        (8, 5, 20, 15, 0.5, False, True, 4, None, "5s_L_TP8_SL5_h20_thr0.5"),
    ]

    print(f"Testing {len(configs)} configs", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    results = {}
    promising = []

    for ci, (tp, sl, hold_s, cancel_s, thr, short_only, long_only, max_conc, filt_thr, label) in enumerate(configs):
        t0 = time.time()
        all_trades, daily_pnls = run_config(
            day_data_list, tp, sl, hold_s, cancel_s, thr, max_conc,
            short_only, long_only, filter_threshold=filt_thr
        )
        elapsed = time.time() - t0

        if not all_trades:
            print(f"[{ci+1}/{len(configs)}] {label}: NO TRADES [{elapsed:.1f}s]", flush=True)
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
        green = int((daily_pnls > 0).sum())
        red = int((daily_pnls < 0).sum())

        exits = trades[:, 7] if trades.shape[1] > 7 else np.zeros(len(trades))
        tp_pct = (exits == 0).mean() * 100
        sl_pct = (exits == 1).mean() * 100
        time_pct = (exits == 2).mean() * 100

        results[label] = {
            'n_trades': int(n_trades), 'per_day': round(float(n_per_day), 1),
            'total_pnl_ticks': round(float(total_pnl), 1),
            'avg_pnl': round(float(pnl.mean()), 3),
            'wr': round(float(wr), 4), 'pf': round(float(pf), 3),
            'sharpe': round(float(sharpe), 2),
            'green_days': green, 'red_days': red,
            'tp_pct': round(float(tp_pct), 1), 'sl_pct': round(float(sl_pct), 1),
        }

        status = "✓" if sharpe > 0 and pf > 1.0 else "✗"
        print(f"[{ci+1}/{len(configs)}] {status} {label}", flush=True)
        print(f"  {n_trades} trades ({n_per_day:.0f}/day), PnL={total_pnl:+.1f}t, avg={pnl.mean():+.3f}t "
              f"WR={wr:.3f}, PF={pf:.3f}, Sharpe={sharpe:+.2f}", flush=True)
        print(f"  Exits: TP={tp_pct:.0f}% SL={sl_pct:.0f}% Time={time_pct:.0f}% | "
              f"Days: {green}G/{red}R [{elapsed:.1f}s]", flush=True)
        print(flush=True)

        if sharpe > 0.5 and pf > 1.0 and wr > 0.45:
            promising.append((ci, tp, sl, hold_s, cancel_s, thr, short_only, long_only, max_conc, filt_thr, label))

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'v11_5s_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Permutation test
    if promising:
        print("=" * 70, flush=True)
        print(f"PERMUTATION TEST — {len(promising)} configs", flush=True)
        print("=" * 70, flush=True)
        print(flush=True)

        for ci, tp, sl, hold_s, cancel_s, thr, short_only, long_only, max_conc, filt_thr, label in promising:
            real_sharpe = results[label]['sharpe']
            print(f"Testing {label} (Sharpe={real_sharpe:+.2f})...", flush=True)
            t0 = time.time()
            p_val, mean_rand, std_rand = permutation_test(
                day_data_list, tp, sl, hold_s, cancel_s, thr, max_conc,
                short_only, long_only, filt_thr,
                n_perms=100, real_sharpe=real_sharpe
            )
            elapsed = time.time() - t0
            results[label]['perm_p_value'] = round(float(p_val), 4)
            results[label]['perm_random_mean'] = round(float(mean_rand), 2)
            results[label]['perm_random_std'] = round(float(std_rand), 2)
            verdict = "REAL EDGE ✓" if p_val < 0.05 else "ARTIFACT ✗"
            print(f"  p={p_val:.3f} | Random: {mean_rand:+.2f}±{std_rand:.2f} | {verdict} [{elapsed:.0f}s]", flush=True)
            print(flush=True)

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    print("=" * 70, flush=True)
    print("FINAL SUMMARY — 5s HORIZON (sorted by Sharpe)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Config':<40} {'N':>5} {'PnL':>7} {'WR':>6} {'PF':>6} {'Sh':>6} {'G/R':>5} {'p':>5}", flush=True)
    print("-" * 82, flush=True)
    sorted_r = sorted(
        [(k, v) for k, v in results.items() if v.get('n_trades', 0) > 0],
        key=lambda x: x[1].get('sharpe', -999), reverse=True
    )
    for label, r in sorted_r:
        p_str = f"{r['perm_p_value']:.2f}" if 'perm_p_value' in r else " —"
        print(f"{label:<40} {r['n_trades']:>5} {r['total_pnl_ticks']:>+7.0f} "
              f"{r['wr']:>6.3f} {r['pf']:>6.3f} {r['sharpe']:>+6.2f} "
              f"{r['green_days']}/{r['red_days']:>2} {p_str:>5}", flush=True)

    validated = [k for k, v in results.items()
                 if v.get('perm_p_value', 1.0) < 0.05 and v.get('sharpe', 0) > 0.5]
    if validated:
        print(f"\n🎯 VALIDATED: {validated}", flush=True)
    else:
        print("\n❌ No validated configs with 5s horizon either.", flush=True)

    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
