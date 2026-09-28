#!/usr/bin/env python3
"""
HC #495 audit — h5s low-LR diag FIFO sweep.
12 cells: 2 horizons (1s, 5s) × 6 confidence bands (1, 2, 5, 10, 20, 50 %).
Selection: top-pct BOTH sides (signed magnitude). Per-day Sharpe limited to
single OOT day (20260427); HC #494 R1 days-positive bar therefore unachievable
but we report it. Pure label-FIFO grade: signed_pnl_per_trade = sign(pred)*label - 0.376.
"""
import sys, os, json
from pathlib import Path
import numpy as np

ROOT = Path("/home/nick/Lvl3Quant") if Path("/home/nick/Lvl3Quant").exists() else Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output" / "cnn_mamba_v3_h5s_lowlr_diag" / "fold_00_oot_predictions.npz"
MBO_DIR  = ROOT / "data" / "processed" / "mbo_events_smart_v3"
OUT_DIR  = ROOT / "output" / "h5s_lowlr_fifo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 1000
STRIDE      = 500
COMMISSION  = 0.376
HORIZONS    = ["1s", "5s"]
TOP_PCTS    = [1, 2, 5, 10, 20, 50]
# Viability bar (note: single OOT day so days_positive_min unreachable)
SHARPE_MIN, REGIME_SKEW_MAX = 1.0, 0.50
TRADES_PER_DAY_MIN = 5

# Load & VERIFY
d = np.load(PRED_NPZ, allow_pickle=True)
preds = d["predictions"]            # (N, 2) — col 0=1s, col 1=5s
oot_files = list(d["oot_files"])
horizons = list(d["horizons"])
print(f"VERIFY: predictions shape={preds.shape}, horizons={horizons}, oot_files={oot_files}")
print(f"VERIFY first 3 rows of predictions:\n{preds[:3]}")
print(f"VERIFY nonzero count={np.count_nonzero(preds)}, nan count={int(np.isnan(preds).sum())}")
print(f"VERIFY preds min/max/mean/std = {preds.min():.4f}/{preds.max():.4f}/{preds.mean():.4f}/{preds.std():.4f}")

assert preds.shape[1] == 2 and horizons == ["1s","5s"], "shape/horizons mismatch"
assert np.isnan(preds).sum() == 0, "predictions contain NaN"

date_str = Path(oot_files[0]).stem.replace("_mbo_events","")
print(f"\nOOT date: {date_str}")
mbo = np.load(MBO_DIR / f"{date_str}_mbo_events.npz", allow_pickle=True)
labels_by_h = {"1s": mbo["labels_1s"], "5s": mbo["labels_5s"]}

n_windows = preds.shape[0]
# window-end index
event_idx = np.arange(n_windows) * STRIDE + (WINDOW_SIZE - 1)
print(f"VERIFY mapping: {n_windows} windows, max event_idx={event_idx.max()}, mbo length={len(labels_by_h['1s'])}")

# Annualization (single day so really intra-day Sharpe per trade)
ANNUAL_SQRT = np.sqrt(252 * 6.5 * 3600)

results = []
for h_i, h in enumerate(HORIZONS):
    p = preds[:, h_i]
    lab = labels_by_h[h][event_idx]
    valid = ~(np.isnan(lab))
    p_v, lab_v = p[valid], lab[valid]
    print(f"\nHorizon {h}: {len(p_v)} valid samples (preds min={p_v.min():.3f} max={p_v.max():.3f})")
    abs_p = np.abs(p_v)
    sign_p = np.sign(p_v)

    for pct in TOP_PCTS:
        # Top-pct by |pred| (both sides). Take sign(pred)*label for directional PnL.
        k = max(1, int(np.ceil(len(p_v) * pct / 100.0)))
        # top-k by abs
        idx_sorted = np.argsort(-abs_p)[:k]
        sig = sign_p[idx_sorted]
        l   = lab_v[idx_sorted]
        per_trade = sig * l - COMMISSION    # FIFO net ticks per trade (passive limit cost only)

        # All-trade aggregates (single day)
        net_ticks  = float(per_trade.sum())
        count      = int(len(per_trade))
        mean_tick  = float(per_trade.mean())
        std_tick   = float(per_trade.std(ddof=1)) if count > 1 else 0.0
        sharpe     = (mean_tick / std_tick * ANNUAL_SQRT) if std_tick > 0 else 0.0
        # Days-positive: only 1 day total, count as 1 if day net positive
        days_total = 1
        days_pos   = 1 if net_ticks > 0 else 0
        trades_per_day = count / days_total
        # Regime skew: undefined (1 day); set NaN
        regime_skew = float("nan")

        # Verdict per HC #494 R1 — strict (cannot meet days_pos >= 30 on 1 day)
        if (net_ticks > 0 and sharpe >= SHARPE_MIN
                and trades_per_day >= TRADES_PER_DAY_MIN):
            verdict = "MARGINAL_1DAY"   # FIFO-positive + Sharpe-ok but only 1 OOT day
        elif net_ticks > 0:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL"

        results.append(dict(horizon=h, top_pct=pct, net_ticks=net_ticks,
                            net_ticks_per_trade=mean_tick, count=count,
                            sharpe=sharpe, days_total=days_total,
                            days_positive=days_pos, trades_per_day=trades_per_day,
                            regime_skew=regime_skew, verdict=verdict))

# Write CSV + report
csv_path = OUT_DIR / "summary.csv"
with open(csv_path, "w") as f:
    f.write("horizon,top_pct,net_ticks,net_ticks_per_trade,count,days_total,days_positive,trades_per_day,sharpe,regime_skew,verdict\n")
    for r in results:
        f.write(f"{r['horizon']},{r['top_pct']},{r['net_ticks']:.2f},{r['net_ticks_per_trade']:.4f},"
                f"{r['count']},{r['days_total']},{r['days_positive']},{r['trades_per_day']:.1f},"
                f"{r['sharpe']:.2f},{r['regime_skew']},{r['verdict']}\n")

# Print summary table
print("\n" + "="*100)
print("SUMMARY (single OOT day, 20260427)")
print("="*100)
print(f"{'Horizon':<8}{'Top%':<6}{'NetTicks':>12}{'NTks/Trade':>12}{'Count':>8}{'Sharpe':>10}{'DaysPos':>9}{'Verdict':>18}")
for r in results:
    print(f"{r['horizon']:<8}{r['top_pct']:<6}{r['net_ticks']:>12.2f}{r['net_ticks_per_trade']:>12.4f}"
          f"{r['count']:>8}{r['sharpe']:>10.2f}{r['days_positive']:>9}{r['verdict']:>18}")

# Best cell by ticks/trade
best = max(results, key=lambda r: r["net_ticks_per_trade"])
print(f"\nBest cell: {best['horizon']}_top{best['top_pct']}% -> "
      f"{best['net_ticks_per_trade']:.4f} ticks/trade, {best['count']} trades, Sharpe={best['sharpe']:.2f}, "
      f"net={best['net_ticks']:.1f} ticks [{best['verdict']}]")

print(f"\nCSV: {csv_path}")
