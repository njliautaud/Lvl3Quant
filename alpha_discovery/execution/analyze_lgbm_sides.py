#!/usr/bin/env python3
"""
Analyze LGBM execution filter results by SIDE (long vs short).
Uses the saved concat_oot_predictions.npz to break down performance.
"""

import numpy as np
import sys
from pathlib import Path

LVL3_ROOT = Path("/home/nick/Lvl3Quant") if Path("/home/nick/Lvl3Quant").exists() else Path("/home/jupiter/Lvl3Quant")

COMMISSION_TICKS = 0.376

def analyze_side(results_dir: str = None):
    if results_dir is None:
        results_dir = str(LVL3_ROOT / "output" / "exec_lgbm_v1")

    pred_file = Path(results_dir) / "concat_oot_predictions.npz"
    if not pred_file.exists():
        print(f"No predictions found at {pred_file}")
        return

    data = np.load(str(pred_file), allow_pickle=True)
    probs = data["probs"]  # LGBM predicted probability of profitable trade
    actual_move = data["actual_move"]  # actual move in predicted direction

    print(f"Loaded {len(probs)} OOT samples from {pred_file}")

    # Load the fold prediction files to get signal directions
    # For now, use actual_move sign as a proxy — positive = signal was correct direction
    pnl = actual_move - COMMISSION_TICKS

    # We need the original predictions to know if it was a long or short signal
    # Load from the fold files
    all_oot_dir = Path(results_dir).parent / "cnn_mamba_v2_all_oot"
    if not all_oot_dir.exists():
        all_oot_dir = Path(results_dir).parent / "cnn_mamba_v2_smart_v3_mar"

    # Since we can't easily map back, analyze by actual_move sign
    # positive actual_move = market moved in predicted direction
    is_correct = actual_move > 0
    is_wrong = actual_move <= 0

    for tier_name, pct in [("top_1pct", 0.99), ("top_5pct", 0.95), ("top_10pct", 0.90), ("top_20pct", 0.80)]:
        threshold = np.percentile(probs, pct * 100)
        mask = probs >= threshold

        for side_name, side_mask in [("ALL", np.ones(len(probs), dtype=bool)), ("LONG", is_long), ("SHORT", is_short)]:
            combined = mask & side_mask
            n = combined.sum()
            if n < 10:
                continue

            tier_pnl = pnl_10s[combined]  # use 10s horizon for P&L
            wr = (tier_pnl > 0).mean()
            avg_pnl = tier_pnl.mean()

            wins = tier_pnl[tier_pnl > 0]
            losses = tier_pnl[tier_pnl <= 0]
            pf = wins.sum() / (-losses.sum() + 1e-10) if len(losses) > 0 else float('inf')

            print(f"  {tier_name:12s} {side_name:6s}: n={n:6d} WR={wr:.3f} PnL={avg_pnl:+.3f}t PF={pf:.2f}")

    # Side distribution analysis
    print(f"\n  Overall: {is_long.sum()} longs ({is_long.mean()*100:.1f}%), {is_short.sum()} shorts ({is_short.mean()*100:.1f}%)")

    # Among top predictions, side distribution
    for tier_name, pct in [("top_1pct", 0.99), ("top_5pct", 0.95)]:
        threshold = np.percentile(probs, pct * 100)
        mask = probs >= threshold
        n_long = (mask & is_long).sum()
        n_short = (mask & is_short).sum()
        total = mask.sum()
        print(f"  {tier_name}: {n_long} longs ({n_long/total*100:.1f}%), {n_short} shorts ({n_short/total*100:.1f}%)")


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else None
    analyze_side(rd)
