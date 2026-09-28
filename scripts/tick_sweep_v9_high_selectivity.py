#!/usr/bin/env python3
"""
Tick-Level Replay v9: HIGH SELECTIVITY + SHORT-ONLY (Fast Engine)
==================================================================

v8 proved threshold=0.2 is garbage (60% pass rate, Sharpe -30 to -50).
This tests the TOP confidence predictions only using the fast Numba engine.

Key insight from decay analysis:
- Short side has +1.56 ticks avg move at top 10% confidence
- Long side is weaker across the board
- Signal decays by 30s

Sweep design:
- Thresholds: 0.4, 0.5, 0.6, 0.8 (keeping 30%, 17%, 8%, 2%)
- Side modes: short_only (primary), long_only (comparison), both
- TP/SL: 2-4 ticks (tight, matching signal horizon)
- Hold: 3s, 5s (not 7+, signal dead by then)
- Cancel: 5s (stale predictions)

HC #659: tick-level replay mandatory, permutation test on every config.
"""

import numpy as np
import glob
import os
import sys
import json
import time

# Add engines to path
sys.path.insert(0, '/home/jupiter/Lvl3Quant/engines')
from tick_replay_fast import (
    load_predictions, find_preprocessed, match_dates,
    run_one_config, run_permutation, compute_metrics,
    N_TRADE_COLS
)


def main():
    PREPROC_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo'
    PRED_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/oot_47day_perdate'
    OUT_DIR = '/home/jupiter/Lvl3Quant/output/tick_level_replay'

    # Fallback pred dir
    if not os.path.exists(PRED_DIR) or not glob.glob(os.path.join(PRED_DIR, 'oot_*.npz')):
        PRED_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate'

    print("=" * 70)
    print("TICK REPLAY v9: HIGH SELECTIVITY (Fast Numba Engine)")
    print("=" * 70)
    print(f"  Preprocessed MBO: {PREPROC_DIR}")
    print(f"  Predictions:      {PRED_DIR}")

    # Load data
    preproc_map = find_preprocessed(PREPROC_DIR)
    pred_map = load_predictions(PRED_DIR, head='pred_log_ret_1s')
    all_matched = match_dates(preproc_map, pred_map)

    print(f"  Preprocessed dates: {len(preproc_map)}")
    print(f"  Prediction dates: {len(pred_map)}")
    print(f"  Matched dates: {len(all_matched)}")

    if not all_matched:
        print("ERROR: No matching dates!")
        sys.exit(1)

    # Distribution analysis
    all_p = np.concatenate([m[2] for m in all_matched])
    print(f"\n  Prediction distribution (N={len(all_p):,}):")
    print(f"    mean={all_p.mean():.4f}, std={all_p.std():.4f}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        n_l = np.sum(all_p > thr)
        n_s = np.sum(all_p < -thr)
        pct = 100 * (n_l + n_s) / len(all_p)
        print(f"    |pred|>{thr:.1f}: {pct:.1f}% ({n_l:,}L, {n_s:,}S per {len(all_matched)} days)")

    # ==========================================================================
    # Phase 1: SHORT-ONLY at high thresholds
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: SHORT-ONLY (High-confidence shorts — best edge per decay analysis)")
    print("=" * 70)

    configs = []

    for threshold in [0.4, 0.5, 0.6, 0.8]:
        for tp in [2, 3, 4]:
            for sl in [3, 4, 6]:
                for hold_s in [3.0, 5.0]:
                    configs.append({
                        'tp': tp, 'sl': sl, 'hold': hold_s,
                        'cancel': 5.0, 'threshold': threshold,
                        'side': 'short_only', 'max_concurrent': 1,
                    })

    # Phase 2: LONG-ONLY comparison (smaller set)
    for threshold in [0.5, 0.6, 0.8]:
        for tp in [2, 3]:
            for sl in [3, 4]:
                configs.append({
                    'tp': tp, 'sl': sl, 'hold': 5.0,
                    'cancel': 5.0, 'threshold': threshold,
                    'side': 'long_only', 'max_concurrent': 1,
                })

    # Phase 3: BOTH at very high threshold
    for threshold in [0.6, 0.8]:
        for tp in [2, 3, 4]:
            for sl in [3, 4]:
                configs.append({
                    'tp': tp, 'sl': sl, 'hold': 5.0,
                    'cancel': 5.0, 'threshold': threshold,
                    'side': 'both', 'max_concurrent': 1,
                })

    print(f"\nTotal configs: {len(configs)}")

    results = []
    t_start = time.time()

    for i, cfg in enumerate(configs):
        label = f"TP{cfg['tp']}_SL{cfg['sl']}_h{cfg['hold']:.0f}_thr{cfg['threshold']:.1f}_{cfg['side']}"

        # Filter predictions by side
        if cfg['side'] == 'short_only':
            # Keep only negative predictions (short signals), zero out longs
            filtered = [(d, p, np.where(preds < 0, preds, 0.0)) for d, p, preds in all_matched]
        elif cfg['side'] == 'long_only':
            # Keep only positive predictions (long signals), zero out shorts
            filtered = [(d, p, np.where(preds > 0, preds, 0.0)) for d, p, preds in all_matched]
        else:
            filtered = all_matched

        print(f"\n[{i+1}/{len(configs)}] {label}", flush=True)

        trades = run_one_config(
            filtered,
            tp_ticks=cfg['tp'], sl_ticks=cfg['sl'],
            hold_seconds=cfg['hold'], cancel_seconds=cfg['cancel'],
            signal_threshold=cfg['threshold'],
            max_concurrent=cfg['max_concurrent'],
            verbose=False,
        )

        if len(trades) == 0:
            print(f"  NO TRADES")
            results.append({'label': label, 'n_trades': 0, **cfg})
            continue

        metrics = compute_metrics(trades, label)
        n_days = len(all_matched)
        trades_per_day = metrics['n_trades'] / n_days

        print(f"  {metrics['n_trades']} trades ({trades_per_day:.0f}/day), "
              f"PnL={metrics['net_pnl_ticks']:.0f}t, "
              f"WR={metrics['win_rate']:.3f}, PF={metrics['profit_factor']:.3f}, "
              f"Sharpe={metrics['sharpe']:.2f}")

        result = {**metrics, **cfg, 'trades_per_day': trades_per_day}

        # Permutation test for anything with PF > 0.8 or positive PnL
        if metrics['profit_factor'] > 0.8 or metrics['net_pnl_ticks'] > 0:
            print(f"  ⚡ Promising — running 50 permutation trials...")
            random_pnls = run_permutation(
                filtered,
                tp_ticks=cfg['tp'], sl_ticks=cfg['sl'],
                hold_seconds=cfg['hold'], cancel_seconds=cfg['cancel'],
                signal_threshold=cfg['threshold'],
                max_concurrent=cfg['max_concurrent'],
                n_perms=50,
            )
            model_pnl = metrics['net_pnl_ticks']
            p_value = float(np.mean(random_pnls >= model_pnl))
            result['p_value'] = p_value
            result['random_mean_pnl'] = float(np.mean(random_pnls))
            result['random_std_pnl'] = float(np.std(random_pnls))
            print(f"  p-value: {p_value:.4f} (random mean: {np.mean(random_pnls):.0f}t)")
        else:
            result['p_value'] = None

        results.append(result)

        # Save intermediate
        out_path = os.path.join(OUT_DIR, 'v9_high_selectivity.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t_start

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f"FINAL SUMMARY (elapsed: {elapsed/60:.1f} min)")
    print("=" * 70)

    valid = [r for r in results if r.get('n_trades', 0) > 0]
    valid.sort(key=lambda x: x.get('profit_factor', 0), reverse=True)

    print(f"\n{'Label':<50} {'N':>5} {'T/d':>5} {'PnL':>8} {'WR':>5} {'PF':>5} {'Sh':>6} {'p':>5}")
    print("-" * 95)
    for r in valid[:30]:
        p_str = f"{r['p_value']:.3f}" if r.get('p_value') is not None else "  -  "
        print(f"{r['label']:<50} {r['n_trades']:>5} {r.get('trades_per_day',0):>5.0f} "
              f"{r.get('net_pnl_ticks',0):>8.0f} {r['win_rate']:>5.3f} "
              f"{r['profit_factor']:>5.3f} {r.get('sharpe',0):>6.2f} {p_str:>5}")

    # Highlight winners
    winners = [r for r in valid if r.get('p_value') is not None and r['p_value'] < 0.05 and r['profit_factor'] > 1.0]
    if winners:
        print(f"\n🟢 {len(winners)} configs PASS permutation test (p<0.05, PF>1.0):")
        for w in winners:
            print(f"   {w['label']}: PF={w['profit_factor']:.2f}, Sharpe={w['sharpe']:.2f}, p={w['p_value']:.4f}")
    else:
        # Check if anything is close
        close = [r for r in valid if r.get('profit_factor', 0) > 0.8]
        if close:
            print(f"\n🟡 {len(close)} configs approaching profitability (PF > 0.8)")
            print("   None pass permutation test yet")
        else:
            print(f"\n🔴 No profitable configs found")
            print("   MODEL EDGE DOES NOT SURVIVE PASSIVE FIFO FILLS AT TICK LEVEL")
            print("   Next test: market-entry (pay 1 tick spread but instant fill)")

    # Save final
    out_path = os.path.join(OUT_DIR, 'v9_high_selectivity.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
