#!/usr/bin/env python3
"""
Tick Replay Broad Sweep — Longer Holds + Wider Stops
=====================================================
HC #659: tick-level replay mandatory. Previous runs (v15/v16) tested
3 dates with short holds (3-30s) and tight TP/SL (2-5 ticks) → 0 profitable.

This sweep tests:
- ALL 34 matched dates (not just 3)
- Longer holds: 60s, 120s, 300s (matching model's longer horizons)
- Wider TP/SL: TP 6-20, SL 6-20
- Multiple signal thresholds
- Permutation test on any promising configs

Hypothesis: the signal may be real at longer horizons but too thin
at short horizons to cover FIFO queue + cost friction.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'engines'))
from tick_replay_engine import (
    TickReplayEngine, load_predictions, find_mbo_files,
    compute_metrics, Trade
)

MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/tick_replay_broad_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("=" * 70)
print("TICK REPLAY BROAD SWEEP — LONGER HOLDS + WIDER STOPS")
print("=" * 70)

predictions = load_predictions(PRED_DIR)
mbo_files = find_mbo_files(MBO_DIR)

# Match dates
matched = []
for mbo_path in mbo_files:
    date8 = os.path.basename(mbo_path).split('-')[2].split('.')[0]
    if date8 in predictions:
        matched.append((mbo_path, date8))

print(f"\nMatched {len(matched)} dates for simulation")
print(f"Date range: {matched[0][1]} to {matched[-1][1]}")

# ─────────────────────────────────────────────
# CONFIG SPACE
# ─────────────────────────────────────────────
configs = []

# Wider TP/SL with longer holds
for hold_s in [60, 120, 300]:
    for tp in [8, 12, 16, 20]:
        for sl in [6, 10, 16, 20]:
            for thresh in [0.2, 0.3, 0.5]:
                cancel_s = min(hold_s // 2, 30)
                configs.append({
                    'tp': tp, 'sl': sl, 'hold_s': hold_s,
                    'thresh': thresh, 'cancel_s': cancel_s,
                    'label': f'h{hold_s}_tp{tp}_sl{sl}_t{thresh}'
                })

# Also test no-TP/no-SL (pure time exit) to measure raw signal quality
for hold_s in [60, 120, 300]:
    for thresh in [0.2, 0.3, 0.5]:
        configs.append({
            'tp': 999, 'sl': 999, 'hold_s': hold_s,
            'thresh': thresh, 'cancel_s': min(hold_s // 2, 30),
            'label': f'h{hold_s}_pure_time_t{thresh}'
        })

print(f"Testing {len(configs)} configurations")

# ─────────────────────────────────────────────
# RUN SWEEP (use subset of days first for speed)
# ─────────────────────────────────────────────
# Phase 1: Run on 10 evenly-spaced dates for quick screening
phase1_dates = matched[::3][:12]  # Every 3rd date, up to 12
print(f"\nPhase 1: Screening on {len(phase1_dates)} dates")

results = []
t_start = time.time()

for ci, cfg in enumerate(configs):
    all_trades = []

    for mbo_path, date_key in phase1_dates:
        try:
            engine = TickReplayEngine(
                tp_ticks=cfg['tp'],
                sl_ticks=cfg['sl'],
                hold_seconds=cfg['hold_s'],
                signal_threshold=cfg['thresh'],
                cancel_seconds=cfg['cancel_s'],
            )
            preds = predictions[date_key]
            trades = engine.run_day(mbo_path, preds)
            all_trades.extend(trades)
        except Exception as e:
            print(f"  ERROR {date_key}: {e}")

    if not all_trades:
        continue

    # Compute metrics
    n_trades = len(all_trades)
    total_pnl = sum(t.pnl_ticks for t in all_trades)
    per_trade = total_pnl / n_trades if n_trades > 0 else 0
    wins = sum(1 for t in all_trades if t.pnl_ticks > 0)
    wr = wins / n_trades if n_trades > 0 else 0
    avg_mfe = np.mean([t.mfe_ticks for t in all_trades]) if all_trades else 0
    avg_mae = np.mean([t.mae_ticks for t in all_trades]) if all_trades else 0

    # Exit reason breakdown
    exit_reasons = {}
    for t in all_trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    result = {
        'label': cfg['label'],
        'tp': cfg['tp'], 'sl': cfg['sl'],
        'hold_s': cfg['hold_s'], 'thresh': cfg['thresh'],
        'n_trades': n_trades,
        'total_pnl': round(total_pnl, 1),
        'per_trade': round(per_trade, 3),
        'win_rate': round(wr, 3),
        'avg_mfe': round(avg_mfe, 1),
        'avg_mae': round(avg_mae, 1),
        'n_days': len(phase1_dates),
        'exit_reasons': exit_reasons,
    }
    results.append(result)

    # Progress
    if (ci + 1) % 20 == 0 or per_trade > 0:
        status = "🔥 PROFITABLE" if per_trade > 0 else ""
        elapsed = time.time() - t_start
        print(f"  [{ci+1}/{len(configs)}] {cfg['label']:30s} | "
              f"trades={n_trades:5d} | per_trade={per_trade:+.3f}t | "
              f"WR={wr:.3f} | MFE={avg_mfe:.1f} | MAE={avg_mae:.1f} | "
              f"{elapsed:.0f}s {status}")

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
elapsed_total = time.time() - t_start
print(f"\n{'=' * 70}")
print(f"RESULTS — {len(results)} configs tested in {elapsed_total:.0f}s")
print(f"{'=' * 70}")

# Sort by per-trade P&L
results_sorted = sorted(results, key=lambda x: x['per_trade'], reverse=True)

# How many profitable?
profitable = [r for r in results if r['per_trade'] > 0]
print(f"\nProfitable configs: {len(profitable)}/{len(results)}")

# Top 10
print(f"\nTop 10 by per-trade P&L:")
for r in results_sorted[:10]:
    exits = r['exit_reasons']
    tp_pct = exits.get('tp', 0) / r['n_trades'] * 100 if r['n_trades'] > 0 else 0
    sl_pct = exits.get('sl', 0) / r['n_trades'] * 100 if r['n_trades'] > 0 else 0
    ts_pct = exits.get('time_stop', 0) / r['n_trades'] * 100 if r['n_trades'] > 0 else 0
    print(f"  {r['label']:30s} | trades={r['n_trades']:5d} | per_trade={r['per_trade']:+.3f}t | "
          f"WR={r['win_rate']:.3f} | MFE={r['avg_mfe']:.1f} MAE={r['avg_mae']:.1f} | "
          f"TP:{tp_pct:.0f}% SL:{sl_pct:.0f}% Time:{ts_pct:.0f}%")

# Bottom 5 (worst)
print(f"\nBottom 5:")
for r in results_sorted[-5:]:
    print(f"  {r['label']:30s} | per_trade={r['per_trade']:+.3f}t | WR={r['win_rate']:.3f}")

# Phase 2: If any profitable, run permutation test on ALL dates
if profitable:
    print(f"\n{'=' * 70}")
    print(f"PHASE 2: PERMUTATION TEST ON {len(profitable)} PROMISING CONFIGS")
    print(f"Running on ALL {len(matched)} dates...")
    print(f"{'=' * 70}")

    validated = []
    for r in profitable[:10]:  # Test top 10 max
        print(f"\n  Testing {r['label']}...")

        # Run real signal on all dates
        real_trades = []
        for mbo_path, date_key in matched:
            try:
                engine = TickReplayEngine(
                    tp_ticks=r['tp'], sl_ticks=r['sl'],
                    hold_seconds=r['hold_s'],
                    signal_threshold=r['thresh'],
                    cancel_seconds=min(r['hold_s'] // 2, 30),
                )
                trades = engine.run_day(mbo_path, predictions[date_key])
                real_trades.extend(trades)
            except:
                pass

        real_per_trade = sum(t.pnl_ticks for t in real_trades) / len(real_trades) if real_trades else 0
        real_wr = sum(1 for t in real_trades if t.pnl_ticks > 0) / len(real_trades) if real_trades else 0

        # Permutation: random directions
        n_perms = 100
        perm_per_trades = []
        for p in range(n_perms):
            perm_pnl = 0
            for t in real_trades:
                # Flip direction randomly
                if np.random.random() < 0.5:
                    perm_pnl += t.pnl_ticks
                else:
                    perm_pnl -= t.pnl_ticks  # Reverse the P&L
            perm_per_trade = perm_pnl / len(real_trades) if real_trades else 0
            perm_per_trades.append(perm_per_trade)

        perm_mean = np.mean(perm_per_trades)
        perm_p = np.mean([p >= real_per_trade for p in perm_per_trades])

        r['full_trades'] = len(real_trades)
        r['full_per_trade'] = round(real_per_trade, 3)
        r['full_wr'] = round(real_wr, 3)
        r['perm_p'] = round(perm_p, 3)
        r['perm_mean'] = round(perm_mean, 3)

        status = "✅ VALIDATED" if perm_p < 0.05 and real_per_trade > 0 else "❌ ARTIFACT"
        print(f"    Full: {len(real_trades)} trades, per_trade={real_per_trade:+.3f}t, WR={real_wr:.3f}")
        print(f"    Perm: mean={perm_mean:+.3f}t, p={perm_p:.3f} {status}")

        if perm_p < 0.05 and real_per_trade > 0:
            validated.append(r)

    if validated:
        print(f"\n🔥 {len(validated)} CONFIGS PASS PERMUTATION TEST!")
        for v in validated:
            print(f"  {v['label']}: per_trade={v['full_per_trade']:+.3f}t, WR={v['full_wr']:.3f}, perm_p={v['perm_p']:.3f}")
    else:
        print(f"\n❌ No configs survive permutation test. Signal too weak for tick-level execution.")

# Save all results
output = {
    'generated': time.strftime('%Y-%m-%d %H:%M'),
    'phase1_dates': len(phase1_dates),
    'total_configs': len(configs),
    'profitable_count': len(profitable),
    'results': results_sorted,
    'elapsed_s': round(elapsed_total, 1),
}
with open(OUTPUT_DIR / 'broad_sweep_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved. Total time: {elapsed_total:.0f}s")
