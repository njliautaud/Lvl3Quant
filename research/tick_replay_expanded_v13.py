#!/usr/bin/env python3
"""
Tick Replay v13 — Expanded FIFO passive-limit sweep with permutation test.

Prior v4 FIFO results (13 days): hold=5s +0.202t/trade, 53.6% WR.
Now: run ALL 34 available prediction days, permutation-validated.

HC #659: Every config must pass permutation test before reporting.
Focused on time-stop exits (no TP/SL — prior results showed passive time exit wins).
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path("/home/jupiter/Lvl3Quant/engines")))
from tick_replay_engine import (
    TickReplayEngine, compute_metrics, load_predictions, find_mbo_files
)

MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/tick_replay_v13_expanded")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_config(matched_dates, predictions, hold_s, threshold, cancel_s,
               tp=99, sl=99, label=""):
    """Run one config across all days, return (trades_list, metrics_dict)."""
    all_trades = []
    day_pnls = []
    for mbo_path, date_key in matched_dates:
        engine = TickReplayEngine(
            tp_ticks=tp, sl_ticks=sl,
            hold_seconds=hold_s,
            signal_threshold=threshold,
            cancel_seconds=cancel_s,
        )
        trades = engine.run_day(mbo_path, predictions[date_key])
        all_trades.extend(trades)
        day_pnl = sum(t.pnl_ticks for t in trades)
        day_pnls.append(day_pnl)

    m = compute_metrics(all_trades, label)
    m['day_pnls'] = day_pnls
    m['green_days'] = sum(1 for p in day_pnls if p > 0)
    m['red_days'] = sum(1 for p in day_pnls if p < 0)
    m['flat_days'] = sum(1 for p in day_pnls if p == 0)
    return all_trades, m


def permutation_test(matched_dates, predictions, hold_s, threshold, cancel_s,
                     tp, sl, real_pnl, n_perms=100, seed=42):
    """Run N permutations with random sign flips, return p-value and random PnLs."""
    rng = np.random.RandomState(seed)
    random_pnls = []

    for i in range(n_perms):
        perm_pnl = 0.0
        for mbo_path, date_key in matched_dates:
            engine = TickReplayEngine(
                tp_ticks=tp, sl_ticks=sl,
                hold_seconds=hold_s,
                signal_threshold=threshold,
                cancel_seconds=cancel_s,
            )
            preds = predictions[date_key]
            random_preds = preds * rng.choice([-1, 1], size=len(preds))
            trades = engine.run_day(mbo_path, random_preds)
            perm_pnl += sum(t.pnl_ticks for t in trades)

        random_pnls.append(perm_pnl)
        if (i + 1) % 25 == 0:
            p_so_far = sum(1 for x in random_pnls if x >= real_pnl) / len(random_pnls)
            print(f"    Perm {i+1}/{n_perms}: rand={perm_pnl:.0f}t, p={p_so_far:.3f}")

    random_pnls = np.array(random_pnls)
    p_value = float(np.mean(random_pnls >= real_pnl))
    return p_value, random_pnls


def main():
    print("=" * 70)
    print("TICK REPLAY v13 — EXPANDED FIFO PASSIVE LIMIT SWEEP")
    print("=" * 70)

    # Load data
    print("\nLoading predictions...")
    predictions = load_predictions(PRED_DIR)
    print(f"  {len(predictions)} prediction dates")

    mbo_files = find_mbo_files(MBO_DIR)
    print(f"  {len(mbo_files)} MBO files")

    # Match
    matched = []
    for mbo_path in mbo_files:
        date8 = os.path.basename(mbo_path).split('-')[2].split('.')[0]
        if date8 in predictions:
            matched.append((mbo_path, date8))

    print(f"  {len(matched)} matched dates for simulation")
    print(f"  Dates: {[d for _, d in matched[:5]]}...{[d for _, d in matched[-3:]]}")

    # Config sweep: time-stop only (TP=99/SL=99 = effectively disabled)
    # Focus on hold_seconds and threshold as primary parameters
    configs = []
    for hold_s in [3, 5, 7, 10, 15]:
        for threshold in [0.2, 0.3, 0.4, 0.5]:
            for cancel_s in [10, 20]:
                configs.append({
                    'hold_s': hold_s,
                    'threshold': threshold,
                    'cancel_s': cancel_s,
                    'tp': 99, 'sl': 99,  # disabled — pure time stop
                })

    # Also test with modest TP/SL for comparison
    for hold_s in [5, 10]:
        for tp in [3, 4, 6]:
            for sl in [4, 6, 8]:
                configs.append({
                    'hold_s': hold_s,
                    'threshold': 0.3,
                    'cancel_s': 15,
                    'tp': tp, 'sl': sl,
                })

    print(f"\n{len(configs)} configs to test")
    print("Phase 1: Screen all configs (no permutation)")
    print("Phase 2: Permutation test top configs")

    # Phase 1: Screen
    phase1_results = []
    t0 = time.time()

    for i, cfg in enumerate(configs):
        label = (f"h{cfg['hold_s']}s_t{cfg['threshold']}_c{cfg['cancel_s']}"
                 f"_TP{cfg['tp']}_SL{cfg['sl']}")
        _, m = run_config(
            matched, predictions,
            hold_s=cfg['hold_s'],
            threshold=cfg['threshold'],
            cancel_s=cfg['cancel_s'],
            tp=cfg['tp'],
            sl=cfg['sl'],
            label=label,
        )
        m['config'] = cfg
        m['label'] = label
        phase1_results.append(m)

        net = m.get('net_pnl_ticks', 0)
        wr = m.get('win_rate', 0)
        sharpe = m.get('sharpe', 0)
        n = m.get('n_trades', 0)
        sign = '+' if net > 0 else ''
        elapsed = time.time() - t0

        if (i + 1) % 10 == 0 or net > 0:
            print(f"  [{i+1}/{len(configs)}] {label}: "
                  f"{sign}{net:.0f}t n={n} WR={wr:.1%} Sh={sharpe:.1f} "
                  f"G/R={m.get('green_days',0)}/{m.get('red_days',0)} "
                  f"({elapsed:.0f}s)")

    # Sort by Sharpe
    phase1_results.sort(key=lambda x: x.get('sharpe', -99), reverse=True)

    print(f"\n{'='*70}")
    print("PHASE 1 SCREENING — TOP 10")
    print(f"{'='*70}")
    print(f"{'Label':<35} {'Trades':>7} {'PnL(t)':>8} {'WR':>6} "
          f"{'Sharpe':>7} {'G/R':>5}")
    print("-" * 75)

    for r in phase1_results[:10]:
        net = r.get('net_pnl_ticks', 0)
        sign = '+' if net > 0 else ''
        print(f"{r['label']:<35} {r.get('n_trades',0):>7} "
              f"{sign}{net:>7.0f} {r.get('win_rate',0):>5.1%} "
              f"{r.get('sharpe',0):>7.1f} "
              f"{r.get('green_days',0):>2}/{r.get('red_days',0):<2}")

    # Phase 2: Permutation test top 5 profitable configs
    profitable = [r for r in phase1_results if r.get('net_pnl_ticks', 0) > 0]
    top_n = min(5, len(profitable))
    top_configs = profitable[:top_n]

    print(f"\n{'='*70}")
    print(f"PHASE 2 — PERMUTATION TEST ON TOP {top_n} PROFITABLE CONFIGS")
    print(f"{'='*70}")

    final_results = []
    for r in top_configs:
        cfg = r['config']
        print(f"\n  Testing: {r['label']} (PnL={r['net_pnl_ticks']:.0f}t)")
        real_pnl = r['net_pnl_ticks']

        p_value, rand_pnls = permutation_test(
            matched, predictions,
            hold_s=cfg['hold_s'],
            threshold=cfg['threshold'],
            cancel_s=cfg['cancel_s'],
            tp=cfg['tp'],
            sl=cfg['sl'],
            real_pnl=real_pnl,
            n_perms=100,
        )

        r['p_value'] = p_value
        r['random_mean_pnl'] = float(np.mean(rand_pnls))
        r['random_std_pnl'] = float(np.std(rand_pnls))

        sig = "✅ PASS" if p_value < 0.05 else "❌ FAIL"
        print(f"  → p={p_value:.3f} {sig}")
        print(f"    Real: {real_pnl:.0f}t vs Random: {np.mean(rand_pnls):.0f}±{np.std(rand_pnls):.0f}t")

        final_results.append(r)

    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    passing = [r for r in final_results if r.get('p_value', 1) < 0.05]
    if passing:
        print(f"\n{len(passing)} configs pass permutation test (p < 0.05):")
        for r in passing:
            print(f"  {r['label']}: Sharpe={r.get('sharpe',0):.1f} "
                  f"PnL={r['net_pnl_ticks']:.0f}t WR={r.get('win_rate',0):.1%} "
                  f"p={r['p_value']:.3f}")
    else:
        print("\nNO configs pass permutation test — signal insufficient for tick-level trading")

    # Save all
    output = {
        'generated': time.strftime('%Y-%m-%d %H:%M'),
        'n_days': len(matched),
        'dates': [d for _, d in matched],
        'phase1_top10': [{k: v for k, v in r.items() if k != 'day_pnls'}
                         for r in phase1_results[:10]],
        'phase2_permutation': [{k: v for k, v in r.items() if k != 'day_pnls'}
                                for r in final_results],
        'verdict': 'PASS' if passing else 'FAIL',
        'n_passing': len(passing),
    }

    out_path = OUT_DIR / 'v13_results.json'
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
