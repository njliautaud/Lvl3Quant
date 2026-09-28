#!/usr/bin/env python3
"""
Quick 3-tier evaluation for LGBM DA classifier (smart_v3).
Loads all fold predictions, computes DA at confidence tiers,
and estimates cost-adjusted P&L.
"""
import numpy as np
import glob
import json
import os
from scipy.stats import spearmanr

PRED_DIR = "/home/jupiter/Lvl3Quant/output/lgbm_da_smart_v3_1d_oot"
OUTPUT_FILE = os.path.join(PRED_DIR, "eval_3tier_results.json")

# Cost assumptions (ES futures)
TICK_VALUE = 12.50
COST_TICKS = 2.0  # spread + slippage

def evaluate():
    preds_files = sorted(glob.glob(f"{PRED_DIR}/fold*_preds.npz"))
    print(f"Loading {len(preds_files)} fold predictions...")

    all_probs, all_labels, all_conf = [], [], []
    fold_results = []

    for pf in preds_files:
        d = np.load(pf)
        probs = d['probs']
        labels = d['labels']
        conf = d['confidence']
        all_probs.append(probs)
        all_labels.append(labels)
        all_conf.append(conf)

        # Per-fold DA
        da = np.mean((probs > 0.5) == labels)
        fold_name = os.path.basename(pf).replace('_preds.npz', '')
        fold_results.append({'fold': fold_name, 'da': float(da), 'n': len(labels)})

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    conf = np.concatenate(all_conf)

    # Direction: prob > 0.5 = long, else short
    pred_dir = (probs > 0.5).astype(float)  # 1=long, 0=short
    correct = (pred_dir == labels)

    print(f"\nTotal samples: {len(labels):,}")
    print(f"Long predictions: {pred_dir.sum():,.0f} ({pred_dir.mean()*100:.1f}%)")
    print(f"Short predictions: {(1-pred_dir).sum():,.0f} ({(1-pred_dir.mean())*100:.1f}%)")

    # Confidence tiers
    tiers = {
        "All": 0, "Top50%": 50, "Top25%": 75, "Top10%": 90,
        "Top5%": 95, "Top1%": 99, "Top0.5%": 99.5, "Top0.1%": 99.9
    }

    results = {"model": "LGBM DA smart_v3", "total_folds": len(preds_files),
               "total_samples": int(len(labels)), "tiers": {}}

    print(f"\n{'Tier':<10} {'n':>8} {'DA':>7} {'WinRate':>8} {'AvgConf':>8} {'NetPnL/trade':>13} {'Sortino':>8} {'PF':>6}")
    print("-" * 80)

    for tier_name, pct in tiers.items():
        if pct == 0:
            mask = np.ones(len(conf), dtype=bool)
        else:
            thresh = np.percentile(conf, pct)
            mask = conf >= thresh

        n = mask.sum()
        if n < 10:
            continue

        tier_correct = correct[mask]
        tier_probs = probs[mask]
        tier_labels = labels[mask]
        tier_pred_dir = pred_dir[mask]
        tier_conf = conf[mask]

        da = tier_correct.mean()

        # Estimate P&L per trade (in ticks)
        # Assume: correct trade = avg_win ticks, wrong = avg_loss ticks
        # For ES, typical movement at 10s: ~2-4 ticks
        # Simple model: win = confidence * scale, loss = -(1-confidence) * scale
        # Better: use DA directly
        # Net edge per trade = DA - (1-DA) = 2*DA - 1 (in probability units)
        edge = 2 * da - 1  # fraction of "net correct"

        # Gross P&L per trade (assume avg move = 2 ticks)
        avg_move_ticks = 2.0
        gross_pnl = edge * avg_move_ticks  # ticks per trade
        net_pnl = gross_pnl - COST_TICKS  # after costs

        # Win/loss for Sortino
        wins = tier_correct.sum()
        losses = n - wins
        win_rate = wins / n

        # Simulate trade-by-trade P&L for Sortino
        trade_pnl = np.where(tier_correct, avg_move_ticks - COST_TICKS, -avg_move_ticks - COST_TICKS)
        mean_pnl = trade_pnl.mean()
        downside = trade_pnl[trade_pnl < 0]
        downside_std = downside.std() if len(downside) > 1 else 1.0
        sortino = mean_pnl / downside_std if downside_std > 0 else 0

        # Profit factor
        gross_wins = trade_pnl[trade_pnl > 0].sum() if (trade_pnl > 0).any() else 0
        gross_losses = abs(trade_pnl[trade_pnl < 0].sum()) if (trade_pnl < 0).any() else 1
        pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

        # Long/short breakdown
        long_mask = tier_pred_dir == 1
        short_mask = tier_pred_dir == 0
        long_da = tier_correct[long_mask].mean() if long_mask.sum() > 0 else 0
        short_da = tier_correct[short_mask].mean() if short_mask.sum() > 0 else 0

        tier_result = {
            "n": int(n),
            "DA": float(da),
            "win_rate": float(win_rate),
            "avg_confidence": float(tier_conf.mean()),
            "net_pnl_per_trade_ticks": float(net_pnl),
            "sortino": float(sortino),
            "profit_factor": float(pf),
            "long_DA": float(long_da),
            "short_DA": float(short_da),
            "n_long": int(long_mask.sum()),
            "n_short": int(short_mask.sum()),
            "total_net_pnl_usd": float(mean_pnl * n * TICK_VALUE),
        }
        results["tiers"][tier_name] = tier_result

        print(f"{tier_name:<10} {n:>8,} {da:>7.4f} {win_rate:>8.4f} {tier_conf.mean():>8.4f} {net_pnl:>13.3f} {sortino:>8.4f} {pf:>6.3f}")

    # Per-fold trend (last 20 folds for recent performance)
    print(f"\n--- Last 20 folds trend ---")
    for fr in fold_results[-20:]:
        marker = "✓" if fr['da'] > 0.55 else "·"
        print(f"  {fr['fold']}: DA={fr['da']:.4f} n={fr['n']:>6} {marker}")

    # Save results
    results["fold_results"] = fold_results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    evaluate()
