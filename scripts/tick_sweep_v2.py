#!/usr/bin/env python3
"""
Tick-Level Sweep v2: Systematic config search for CNN-Mamba v3.4.2
===================================================================

Key improvements over v1:
- Tests configs matched to signal horizon (1s IC=0.10, 5s IC=0.05)
- Sweeps signal thresholds to find the right selectivity
- Phase 1: quick no-permutation scan across all 34 OOT days
- Phase 2: permutation test on promising configs only
- Tests both long-only, short-only, and both-sides
- Also tests using higher-horizon signals (5s, 10s) for entry

Author: Claude (autonomous, 2026-07-01)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'engines'))

import numpy as np
import glob
import json
import time
from collections import defaultdict
from tick_replay_engine import (
    TickReplayEngine, compute_metrics, TICK_SIZE, TICK_VALUE,
    COST_PASSIVE_EXIT, COST_MARKET_EXIT
)


def load_all_predictions(pred_dir):
    """Load predictions for all dates, with multiple signal types."""
    preds = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, 'oot_*.npz'))):
        d = np.load(f, allow_pickle=True)
        date_str = os.path.basename(f).replace('oot_', '').replace('.npz', '')
        if 'pred_log_ret_1s' not in d.files:
            print(f"  Skipping {date_str} (incomplete predictions)")
            d.close()
            continue
        # Extract arrays and copy to avoid lazy-load issues
        preds[date_str] = {
            'lr1s': np.array(d['pred_log_ret_1s']),
            'lr5s': np.array(d['pred_log_ret_5s']),
            'lr10s': np.array(d['pred_log_ret_10s']),
            'lr30s': np.array(d['pred_log_ret_30s']),
            'pup5s': np.array(d['pred_p_up_5s']),
            'pup10s': np.array(d['pred_p_up_10s']),
        }
        d.close()
    return preds


def find_mbo_files(mbo_dir):
    """Find MBO files, indexed by date."""
    files = {}
    for f in sorted(glob.glob(os.path.join(mbo_dir, 'glbx-mdp3-*.mbo.dbn.zst'))):
        date8 = os.path.basename(f).split('-')[2].split('.')[0]
        files[date8] = f
    return files


def run_config(mbo_files, predictions, tp, sl, hold_s, cancel_s,
               signal_key='lr1s', threshold=0.0, side_filter='both',
               max_days=None):
    """Run a single config across all matching dates."""
    matched = sorted(set(mbo_files.keys()) & set(predictions.keys()))
    if max_days:
        matched = matched[:max_days]

    all_trades = []
    daily_stats = {}

    for date_key in matched:
        mbo_path = mbo_files[date_key]
        preds_raw = predictions[date_key][signal_key]

        # Apply side filter
        if side_filter == 'long':
            # Zero out negative predictions (short signals)
            preds = np.where(preds_raw > 0, preds_raw, 0.0)
        elif side_filter == 'short':
            # Zero out positive predictions (long signals)
            preds = np.where(preds_raw < 0, preds_raw, 0.0)
        else:
            preds = preds_raw

        engine = TickReplayEngine(
            tp_ticks=tp,
            sl_ticks=sl,
            hold_seconds=hold_s,
            signal_threshold=threshold,
            cancel_seconds=cancel_s,
            max_concurrent=1,
        )

        try:
            trades = engine.run_day(mbo_path, preds)
            all_trades.extend(trades)
            day_pnl = sum(t.pnl_ticks for t in trades)
            daily_stats[date_key] = {
                'trades': len(trades),
                'pnl_ticks': day_pnl,
                'longs': sum(1 for t in trades if t.side == 'long'),
                'shorts': sum(1 for t in trades if t.side == 'short'),
            }
        except Exception as e:
            print(f"  ERROR on {date_key}: {e}")
            continue

    metrics = compute_metrics(all_trades, f"TP{tp}_SL{sl}")
    return all_trades, metrics, daily_stats


def run_permutation(mbo_files, predictions, tp, sl, hold_s, cancel_s,
                    signal_key='lr1s', threshold=0.0, side_filter='both',
                    n_perms=50, max_days=None):
    """Run permutation test for a config."""
    matched = sorted(set(mbo_files.keys()) & set(predictions.keys()))
    if max_days:
        matched = matched[:max_days]

    rng = np.random.RandomState(42)
    random_pnls = []

    for perm_i in range(n_perms):
        perm_trades = []
        for date_key in matched:
            mbo_path = mbo_files[date_key]
            preds_raw = predictions[date_key][signal_key]

            # Randomize directions
            random_preds = preds_raw * rng.choice([-1, 1], size=len(preds_raw))

            if side_filter == 'long':
                random_preds = np.where(random_preds > 0, random_preds, 0.0)
            elif side_filter == 'short':
                random_preds = np.where(random_preds < 0, random_preds, 0.0)

            engine = TickReplayEngine(
                tp_ticks=tp, sl_ticks=sl,
                hold_seconds=hold_s,
                signal_threshold=threshold,
                cancel_seconds=cancel_s,
                max_concurrent=1,
            )

            try:
                trades = engine.run_day(mbo_path, random_preds)
                perm_trades.extend(trades)
            except:
                continue

        perm_pnl = sum(t.pnl_ticks for t in perm_trades)
        random_pnls.append(perm_pnl)

        if (perm_i + 1) % 10 == 0:
            print(f"    Perm {perm_i+1}/{n_perms}: random PnL = {perm_pnl:.1f}t")

    return np.array(random_pnls)


def phase1_scan(mbo_files, predictions, max_days=None):
    """
    Phase 1: Quick scan of many configs WITHOUT permutation test.
    Goal: identify which configs even show positive PnL.
    """
    print("=" * 80)
    print("PHASE 1: CONFIG SCAN (no permutation test)")
    print("=" * 80)

    configs = []

    # Tight configs matching 1s-5s signal horizon
    for signal_key in ['lr1s', 'lr5s', 'lr10s']:
        for tp in [1, 2, 3, 4, 5, 6, 8]:
            for sl in [1, 2, 3, 4]:
                if tp <= sl:
                    continue  # Need asymmetry
                for threshold in [0.0, 0.05, 0.10, 0.15, 0.20]:
                    for hold_s in [5.0, 10.0, 15.0, 30.0]:
                        for cancel_s in [5.0, 10.0]:
                            configs.append({
                                'signal': signal_key,
                                'tp': tp, 'sl': sl,
                                'hold': hold_s, 'cancel': cancel_s,
                                'threshold': threshold,
                                'side': 'both',
                            })

    print(f"Total configs to test: {len(configs)}")
    print(f"This is too many. Reducing to key combos...")

    # Focused set: match signal horizon
    configs = []
    for signal_key in ['lr1s', 'lr5s']:
        for tp in [2, 3, 4, 6]:
            for sl in [1, 2, 3]:
                if tp <= sl:
                    continue
                for threshold in [0.0, 0.10, 0.20]:
                    for hold_s in [10.0, 30.0]:
                        configs.append({
                            'signal': signal_key,
                            'tp': tp, 'sl': sl,
                            'hold': hold_s,
                            'cancel': 10.0,
                            'threshold': threshold,
                            'side': 'both',
                        })

    print(f"Reduced to {len(configs)} configs")

    results = []
    t0_total = time.time()

    for ci, cfg in enumerate(configs):
        t0 = time.time()
        _, metrics, daily = run_config(
            mbo_files, predictions,
            tp=cfg['tp'], sl=cfg['sl'],
            hold_s=cfg['hold'], cancel_s=cfg['cancel'],
            signal_key=cfg['signal'],
            threshold=cfg['threshold'],
            side_filter=cfg['side'],
            max_days=max_days,
        )
        elapsed = time.time() - t0

        metrics.update(cfg)
        metrics['elapsed_s'] = elapsed
        results.append(metrics)

        # Progress
        if (ci + 1) % 10 == 0 or metrics['net_pnl_ticks'] > 0:
            marker = "✓" if metrics['net_pnl_ticks'] > 0 else "✗"
            print(f"  [{ci+1}/{len(configs)}] {marker} "
                  f"sig={cfg['signal']} TP{cfg['tp']}/SL{cfg['sl']} "
                  f"th={cfg['threshold']} hold={cfg['hold']}s side={cfg['side']}: "
                  f"trades={metrics['n_trades']}, "
                  f"PnL={metrics['net_pnl_ticks']:.0f}t, "
                  f"WR={metrics['win_rate']:.1%}, "
                  f"Sharpe={metrics['sharpe']:.2f} "
                  f"({elapsed:.0f}s)")

    total_time = time.time() - t0_total

    # Sort by Sharpe
    results.sort(key=lambda x: x['sharpe'], reverse=True)

    print(f"\n{'='*80}")
    print(f"PHASE 1 COMPLETE — {total_time:.0f}s total")
    print(f"{'='*80}")

    # Show top 20
    print(f"\nTOP 20 configs by Sharpe:")
    print(f"{'Config':<35} {'Trades':>7} {'PnL(t)':>8} {'WR':>6} {'PF':>6} {'Sharpe':>7}")
    print("-" * 75)

    for r in results[:20]:
        label = (f"{r['signal']} TP{r['tp']}/SL{r['sl']} "
                 f"th={r['threshold']:.2f} h={r['hold']:.0f}s {r['side'][:1]}")
        print(f"{label:<35} {r['n_trades']:>7} "
              f"{r['net_pnl_ticks']:>8.0f} "
              f"{r['win_rate']:>5.1%} "
              f"{r['profit_factor']:>6.2f} "
              f"{r['sharpe']:>7.2f}")

    # Show worst 5 too
    print(f"\nBOTTOM 5 configs:")
    for r in results[-5:]:
        label = (f"{r['signal']} TP{r['tp']}/SL{r['sl']} "
                 f"th={r['threshold']:.2f} h={r['hold']:.0f}s {r['side'][:1]}")
        print(f"{label:<35} {r['n_trades']:>7} "
              f"{r['net_pnl_ticks']:>8.0f} "
              f"{r['win_rate']:>5.1%} "
              f"{r['profit_factor']:>6.2f} "
              f"{r['sharpe']:>7.2f}")

    # Count positive PnL configs
    positive = [r for r in results if r['net_pnl_ticks'] > 0]
    print(f"\n{len(positive)}/{len(results)} configs show positive PnL (before permutation test)")

    return results


def phase2_validate(mbo_files, predictions, promising_configs, n_perms=50, max_days=None):
    """
    Phase 2: Run permutation test on promising configs.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: PERMUTATION VALIDATION")
    print("=" * 80)

    validated = []

    for ci, cfg in enumerate(promising_configs):
        label = (f"{cfg['signal']} TP{cfg['tp']}/SL{cfg['sl']} "
                 f"th={cfg['threshold']:.2f} h={cfg['hold']:.0f}s {cfg['side'][:1]}")
        print(f"\n[{ci+1}/{len(promising_configs)}] Testing: {label}")
        print(f"  Model PnL: {cfg['net_pnl_ticks']:.0f}t, Sharpe: {cfg['sharpe']:.2f}")

        random_pnls = run_permutation(
            mbo_files, predictions,
            tp=cfg['tp'], sl=cfg['sl'],
            hold_s=cfg['hold'], cancel_s=cfg['cancel'],
            signal_key=cfg['signal'],
            threshold=cfg['threshold'],
            side_filter=cfg['side'],
            n_perms=n_perms,
            max_days=max_days,
        )

        model_pnl = cfg['net_pnl_ticks']
        p_value = float(np.mean(random_pnls >= model_pnl))
        model_edge = model_pnl - np.mean(random_pnls)
        pct_from_model = (model_edge / abs(model_pnl) * 100) if model_pnl != 0 else 0

        cfg['p_value'] = p_value
        cfg['random_mean'] = float(np.mean(random_pnls))
        cfg['random_std'] = float(np.std(random_pnls))
        cfg['model_edge_ticks'] = float(model_edge)
        cfg['pct_from_model'] = float(pct_from_model)

        sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""

        print(f"  p-value: {p_value:.3f} {sig}")
        print(f"  Random mean: {np.mean(random_pnls):.0f}t, Model edge: {model_edge:.0f}t")
        print(f"  % from model: {pct_from_model:.1f}%")

        validated.append(cfg)

    # Summary
    print(f"\n{'='*80}")
    print("PHASE 2 SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<35} {'PnL(t)':>8} {'Sharpe':>7} {'p-val':>7} {'Edge':>8} {'%Model':>7}")
    print("-" * 80)

    for r in sorted(validated, key=lambda x: x.get('p_value', 1.0)):
        label = (f"{r['signal']} TP{r['tp']}/SL{r['sl']} "
                 f"th={r['threshold']:.2f} h={r['hold']:.0f}s {r['side'][:1]}")
        sig = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else ""
        print(f"{label:<35} {r['net_pnl_ticks']:>8.0f} "
              f"{r['sharpe']:>7.2f} "
              f"{r['p_value']:>6.3f}{sig:>2} "
              f"{r['model_edge_ticks']:>8.0f} "
              f"{r['pct_from_model']:>6.1f}%")

    significant = [r for r in validated if r['p_value'] < 0.05]
    if significant:
        print(f"\n*** {len(significant)} configs beat random at p < 0.05 ***")
    else:
        print(f"\n*** NO configs beat random at p < 0.05 ***")
        print("Model directional signal does not translate to tradeable edge at tick level.")

    return validated


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-days', type=int, default=None)
    parser.add_argument('--phase', choices=['1', '2', 'both'], default='both')
    parser.add_argument('--perms', type=int, default=50)
    parser.add_argument('--top-n', type=int, default=10,
                        help='How many top configs to validate in Phase 2')
    args = parser.parse_args()

    MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
    PRED_DIR = "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
    OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/tick_level_replay"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading predictions...")
    predictions = load_all_predictions(PRED_DIR)
    print(f"  Loaded {len(predictions)} dates")

    print("Finding MBO files...")
    mbo_files = find_mbo_files(MBO_DIR)
    matched = len(set(mbo_files.keys()) & set(predictions.keys()))
    print(f"  Found {len(mbo_files)} MBO files, {matched} matching prediction dates")

    if args.phase in ('1', 'both'):
        results = phase1_scan(mbo_files, predictions, max_days=args.max_days)

        # Save Phase 1 results
        output_path = os.path.join(OUTPUT_DIR, 'phase1_scan_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nPhase 1 results saved to {output_path}")

        if args.phase == 'both':
            # Phase 2: validate top positive-PnL configs
            promising = [r for r in results if r['net_pnl_ticks'] > 0][:args.top_n]

            if not promising:
                print("\nNo positive-PnL configs found. Trying top by Sharpe instead...")
                promising = results[:args.top_n]

            if promising:
                validated = phase2_validate(
                    mbo_files, predictions, promising,
                    n_perms=args.perms, max_days=args.max_days,
                )

                output_path = os.path.join(OUTPUT_DIR, 'phase2_validated_results.json')
                with open(output_path, 'w') as f:
                    json.dump(validated, f, indent=2, default=str)
                print(f"\nPhase 2 results saved to {output_path}")

    elif args.phase == '2':
        # Load Phase 1 results and validate
        p1_path = os.path.join(OUTPUT_DIR, 'phase1_scan_results.json')
        with open(p1_path) as f:
            results = json.load(f)

        promising = [r for r in results if r['net_pnl_ticks'] > 0][:args.top_n]
        if not promising:
            promising = results[:args.top_n]

        validated = phase2_validate(
            mbo_files, predictions, promising,
            n_perms=args.perms, max_days=args.max_days,
        )

        output_path = os.path.join(OUTPUT_DIR, 'phase2_validated_results.json')
        with open(output_path, 'w') as f:
            json.dump(validated, f, indent=2, default=str)


if __name__ == '__main__':
    main()
