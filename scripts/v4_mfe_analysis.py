#!/usr/bin/env python3
"""
V4 MFE/MAE Analysis using multi-horizon labels
================================================
For top-5% v4 directional signals, analyze:
- What's the typical move trajectory over 1s, 5s, 10s?
- What % of signals reach various TP levels within each horizon?
- What's the maximum adverse excursion?
- Optimal TP/SL sizing based on realized move profiles.

We use labels at 1s, 5s, 10s as snapshots of the price path.
This is approximate MFE (actual MFE from tick data would be better,
but these give us the key parameters).
"""

import numpy as np
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
V4_DIR = ROOT / "output" / "v4_multihead_pressure_v1"

# Load all predictions
fold_files = sorted(V4_DIR.glob("fold_*_oot_predictions.npz"))
print(f"Loading {len(fold_files)} folds...")

all_preds = []
all_labels = []
all_dates = []

for fp in fold_files:
    data = np.load(str(fp), allow_pickle=True)
    date_str = Path(str(data['oot_files'][0])).stem.split('_')[0]

    preds_dir = data['preds_dir']  # (N, 3) - 1s, 5s, 10s predictions
    labels_dir = data['labels_dir']  # (N, 3) - 1s, 5s, 10s actual returns in ticks

    all_preds.append(preds_dir)
    all_labels.append(labels_dir)
    all_dates.extend([date_str] * len(preds_dir))

preds = np.concatenate(all_preds)  # (N, 3)
labels = np.concatenate(all_labels)  # (N, 3)
dates = np.array(all_dates)
N = len(preds)

print(f"Total: {N:,} predictions across {len(fold_files)} dates")

# Use 10s prediction magnitude for top-5% selection
# (this had the best net edge in deep validation)
pred_10s = preds[:, 2]
abs_pred_10s = np.abs(pred_10s)
valid = ~np.isnan(pred_10s) & ~np.isnan(labels[:, 2])
pred_10s_v = pred_10s[valid]
abs_pred_v = abs_pred_10s[valid]
labels_v = labels[valid]  # (N_valid, 3)
dirs = np.sign(pred_10s_v)

# Top percentile thresholds
for pct_name, pct_threshold in [("ALL", 0), ("TOP 20%", 80), ("TOP 10%", 90), ("TOP 5%", 95), ("TOP 2%", 98)]:
    if pct_threshold > 0:
        cutoff = np.percentile(abs_pred_v, pct_threshold)
        mask = abs_pred_v >= cutoff
    else:
        mask = np.ones(len(abs_pred_v), dtype=bool)

    n = mask.sum()
    d = dirs[mask]
    l = labels_v[mask]  # (n, 3) - 1s, 5s, 10s

    # Signed move in predicted direction at each horizon
    move_1s = d * l[:, 0]  # favorable = positive
    move_5s = d * l[:, 1]
    move_10s = d * l[:, 2]

    print(f"\n{'='*70}")
    print(f"{pct_name} PREDICTIONS (n={n:,})")
    print(f"{'='*70}")

    for h_name, move in [("1s", move_1s), ("5s", move_5s), ("10s", move_10s)]:
        v = move[~np.isnan(move)]
        if len(v) < 100:
            continue
        print(f"\n  {h_name} move in predicted direction:")
        print(f"    Mean:   {np.mean(v):+.3f} ticks")
        print(f"    Median: {np.median(v):+.3f} ticks")
        print(f"    Std:    {np.std(v):.3f} ticks")
        print(f"    WR:     {np.mean(v > 0):.1%}")

        # MFE proxy: what fraction reach various tick levels favorably?
        for tp in [1, 2, 3, 4, 5, 6, 8, 10]:
            reach_pct = np.mean(v >= tp)
            print(f"    TP≥{tp}t:  {reach_pct:>5.1%} ({int(reach_pct * len(v))} trades)")

        # MAE proxy: adverse moves
        print(f"    --- Adverse ---")
        for sl in [1, 2, 3, 4, 5]:
            hit_pct = np.mean(v <= -sl)
            print(f"    SL≤-{sl}t: {hit_pct:>5.1%} ({int(hit_pct * len(v))} trades)")

    # Trajectory analysis: how does the move evolve?
    print(f"\n  TRAJECTORY (mean move in predicted direction):")
    valid_traj = ~np.isnan(move_1s) & ~np.isnan(move_5s) & ~np.isnan(move_10s)
    if valid_traj.sum() > 100:
        m1 = np.mean(move_1s[valid_traj])
        m5 = np.mean(move_5s[valid_traj])
        m10 = np.mean(move_10s[valid_traj])
        print(f"    1s: {m1:+.3f}t → 5s: {m5:+.3f}t → 10s: {m10:+.3f}t")
        print(f"    Signal builds from 1s to 10s: {m10/m1:.1f}x amplification")

    # Profit factor
    for h_name, move in [("1s", move_1s), ("5s", move_5s), ("10s", move_10s)]:
        v = move[~np.isnan(move)]
        if len(v) < 100:
            continue
        wins = v[v > 0].sum()
        losses = np.abs(v[v < 0]).sum()
        pf = wins / losses if losses > 0 else float('inf')
        print(f"    {h_name} PF: {pf:.2f}")

# ================================================================
# ECONOMIC VIABILITY: TP/SL configurations
# ================================================================
print(f"\n{'#'*70}")
print("TP/SL CONFIGURATION ANALYSIS (top 5%, 10s horizon labels)")
print(f"{'#'*70}")

cutoff_5 = np.percentile(abs_pred_v, 95)
mask_5 = abs_pred_v >= cutoff_5
d5 = dirs[mask_5]
l5 = labels_v[mask_5]
move_10s_5 = d5 * l5[:, 2]
v_10s = move_10s_5[~np.isnan(move_10s_5)]

print(f"\nUsing {len(v_10s):,} top-5% signals, 10s realized move distribution")

for tp in [2, 3, 4, 5, 6]:
    for sl in [1, 2, 3, 4]:
        if sl >= tp:
            continue

        # Simplified model: if move ≥ TP, we win TP; if move ≤ -SL, we lose SL
        # Signals that don't hit either: we use actual move as PnL
        tp_hits = v_10s >= tp
        sl_hits = v_10s <= -sl
        neither = ~tp_hits & ~sl_hits

        pnl_tp = tp * tp_hits.sum()
        pnl_sl = -sl * sl_hits.sum()
        pnl_mid = v_10s[neither].sum()

        total_pnl = pnl_tp + pnl_sl + pnl_mid
        n_trades = len(v_10s)
        avg_gross = total_pnl / n_trades

        # Cost depends on exit type
        # TP = passive exit → 0.376 ticks
        # SL = market exit → 1.376 ticks
        # Neither = market exit → 1.376 ticks
        cost = (COST_TP := 0.376) * tp_hits.sum() + 1.376 * (sl_hits.sum() + neither.sum())
        avg_cost = cost / n_trades
        avg_net = avg_gross - avg_cost

        wr = (tp_hits.sum() + (v_10s[neither] > 0).sum()) / n_trades

        prof = '✓' if avg_net > 0 else ''
        print(f"  TP{tp}/SL{sl}: TP={tp_hits.mean():.0%}, SL={sl_hits.mean():.0%}, "
              f"gross={avg_gross:+.3f}t, cost={avg_cost:.3f}t, "
              f"net={avg_net:+.3f}t, WR={wr:.1%} {prof}")

# Pure passive analysis (no TP/SL, just hold to horizon)
print(f"\n  HOLD-TO-HORIZON (passive exit):")
for h_name, h_idx in [("1s", 0), ("5s", 1), ("10s", 2)]:
    move = d5 * l5[:, h_idx]
    v = move[~np.isnan(move)]
    avg_gross = np.mean(v)
    avg_net = avg_gross - 0.376  # passive both sides
    print(f"    {h_name}: gross={avg_gross:+.3f}t, net={avg_net:+.3f}t, WR={np.mean(v>0):.1%}")

print("\nDone.")
