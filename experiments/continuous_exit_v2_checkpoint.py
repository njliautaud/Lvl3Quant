#!/usr/bin/env python3
"""
Continuous-Exit Backtester v2 (HC #515 + HC #469 R5f)
=====================================================
Uses CNN-Mamba v2 predictions (1s, 5s, 10s horizons) and smart_v3 CHECKPOINT
labels (labels_1s, labels_5s, labels_10s, labels_30s — known good ticks).

Does NOT use the broken MFE/MAE V2 data (per RUN_HISTORY 2026-06-03 14:55).

Logic:
- Entry: |pred_1s| >= entry_threshold (taker entry, cost commission only per HC #512)
- Continuous monitoring (HC #515): at each subsequent checkpoint (5s, 10s, 30s),
  decide HOLD vs EXIT based on:
    (a) pred at that horizon still aligned with entry side → HOLD
    (b) pred reversed AND realized so far >= take_profit_ticks → EXIT (lock gains)
    (c) pred reversed AND realized so far <= -stop_loss_ticks → EXIT (stop)
- Final exit at max_horizon if no earlier trigger.

Output: per-config aggregate metrics across all OOT days.
"""
import sys, json, time
from pathlib import Path
from itertools import product
import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_DIR = ROOT / "output/cnn_mamba_v2_all_oot"
LABEL_DIR = ROOT / "data/processed/mbo_events_smart_v3"
OUT_DIR = ROOT / "output/continuous_exit_v2_checkpoint"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION_RT = 0.376  # ticks (HC #512)
TICK_VALUE = 12.5  # $

# Sweep grid
ENTRY_THRESHOLDS = [0.5, 0.75, 1.0, 1.5]  # standardized prediction units
TP_TICKS = [1.0, 2.0, 3.0]
SL_TICKS = [1.5, 2.5, 4.0]
SIDES = ["both", "long", "short"]


def load_aligned_day(date_str):
    """Returns (pred_1s, pred_5s, pred_10s, lab_1s, lab_5s, lab_10s, lab_30s)
    all aligned to prediction stride positions."""
    pf = PRED_DIR / f"{date_str}_predictions.npz"
    lf = LABEL_DIR / f"{date_str}_mbo_events.npz"
    if not pf.exists() or not lf.exists():
        return None
    p = np.load(pf)
    preds = p["predictions"]  # (N, 3) for 1s, 5s, 10s
    stride = int(p["stride"])
    window_size = int(p["window_size"])
    n_windows = int(p["n_windows"])
    # Each prediction i corresponds to event index = window_size + i*stride - 1
    end_event_idx = window_size + np.arange(n_windows) * stride - 1
    L = np.load(lf)
    lab_1s = L["labels_1s"]
    lab_5s = L["labels_5s"]
    lab_10s = L["labels_10s"]
    lab_30s = L["labels_30s"]
    # Clip event idx to label length
    valid = end_event_idx < len(lab_1s)
    end_event_idx = end_event_idx[valid]
    preds = preds[valid]
    return {
        "pred_1s": preds[:, 0].astype(np.float64),
        "pred_5s": preds[:, 1].astype(np.float64),
        "pred_10s": preds[:, 2].astype(np.float64),
        "lab_1s": lab_1s[end_event_idx].astype(np.float64),
        "lab_5s": lab_5s[end_event_idx].astype(np.float64),
        "lab_10s": lab_10s[end_event_idx].astype(np.float64),
        "lab_30s": lab_30s[end_event_idx].astype(np.float64),
        "date": date_str,
    }


def simulate_config(day, entry_thr, tp, sl, side):
    """Returns dict with per-trade net ticks for this config on this day."""
    p1 = day["pred_1s"]; p5 = day["pred_5s"]; p10 = day["pred_10s"]
    l1 = day["lab_1s"]; l5 = day["lab_5s"]; l10 = day["lab_10s"]; l30 = day["lab_30s"]

    # Entry signal: |pred_1s| >= entry_thr
    if side == "long":
        entry = p1 >= entry_thr
        sign = +1.0
    elif side == "short":
        entry = p1 <= -entry_thr
        sign = -1.0
    else:  # both — take side from pred
        entry = np.abs(p1) >= entry_thr
        sign = np.sign(p1)

    n_signals = int(entry.sum())
    if n_signals < 50:
        return None

    idx = np.where(entry)[0]
    # Drop NaN labels
    keep = ~(np.isnan(l1[idx]) | np.isnan(l5[idx]) | np.isnan(l10[idx]) | np.isnan(l30[idx]))
    idx = idx[keep]
    if len(idx) < 50:
        return None

    s = sign if side != "both" else sign[idx]
    realized_1s = s * l1[idx]
    realized_5s = s * l5[idx]
    realized_10s = s * l10[idx]
    realized_30s = s * l30[idx]

    pred5_at = s * p5[idx]
    pred10_at = s * p10[idx]

    # Exit logic per trade
    # Default: hold until 30s checkpoint, capture realized_30s
    exit_pnl = realized_30s.copy()
    exit_horizon = np.full(len(idx), 30.0)

    # At 5s checkpoint:
    #  - if pred_5s reversed (pred5_at < 0) AND realized_1s >= tp → exit at 5s with realized_5s (or use 1s)
    #  - if pred_5s reversed AND realized_1s <= -sl → exit at 5s with realized_5s
    rev_5s = pred5_at < 0
    tp_hit_at_1s = realized_1s >= tp
    sl_hit_at_1s = realized_1s <= -sl
    exit_5s_tp = rev_5s & tp_hit_at_1s
    exit_5s_sl = rev_5s & sl_hit_at_1s
    exit_pnl[exit_5s_tp] = realized_5s[exit_5s_tp]
    exit_pnl[exit_5s_sl] = realized_5s[exit_5s_sl]
    exit_horizon[exit_5s_tp | exit_5s_sl] = 5.0

    # At 10s checkpoint (only if not already exited):
    still_in = ~(exit_5s_tp | exit_5s_sl)
    rev_10s = (pred10_at < 0) & still_in
    tp_hit_at_5s = (realized_5s >= tp) & still_in
    sl_hit_at_5s = (realized_5s <= -sl) & still_in
    exit_10s_tp = rev_10s & tp_hit_at_5s
    exit_10s_sl = rev_10s & sl_hit_at_5s
    exit_pnl[exit_10s_tp] = realized_10s[exit_10s_tp]
    exit_pnl[exit_10s_sl] = realized_10s[exit_10s_sl]
    exit_horizon[exit_10s_tp | exit_10s_sl] = 10.0

    net = exit_pnl - COMMISSION_RT
    return {
        "n_trades": len(idx),
        "gross_mean": float(exit_pnl.mean()),
        "net_mean": float(net.mean()),
        "net_sum": float(net.sum()),
        "wr": float((net > 0).mean()),
        "pf": float(net[net > 0].sum() / abs(net[net < 0].sum())) if (net < 0).any() else float("inf"),
        "exit_horizon_mean": float(exit_horizon.mean()),
        "pct_exit_5s": float((exit_horizon == 5).mean()),
        "pct_exit_10s": float((exit_horizon == 10).mean()),
        "pct_exit_30s": float((exit_horizon == 30).mean()),
    }


def main():
    pred_dates = sorted(set(
        p.name.split("_")[0]
        for p in PRED_DIR.glob("*_predictions.npz")
        if p.name[:8].isdigit() and p.name.startswith("2026")
    ))
    print(f"[{time.strftime('%H:%M:%S')}] Loading {len(pred_dates)} prediction dates")

    days = []
    for ds in pred_dates:
        d = load_aligned_day(ds)
        if d is not None and len(d["pred_1s"]) > 100:
            days.append(d)
    print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(days)} aligned days")
    if len(days) < 5:
        print("ERROR: too few days")
        return

    configs = list(product(ENTRY_THRESHOLDS, TP_TICKS, SL_TICKS, SIDES))
    print(f"[{time.strftime('%H:%M:%S')}] Sweeping {len(configs)} configs × {len(days)} days = {len(configs)*len(days)} sims")

    results = []
    for ci, (et, tp, sl, side) in enumerate(configs):
        per_day = []
        for d in days:
            r = simulate_config(d, et, tp, sl, side)
            if r is not None:
                r["date"] = d["date"]
                per_day.append(r)
        if len(per_day) < 5:
            continue
        # Aggregate
        net_means = np.array([r["net_mean"] for r in per_day])
        n_trades = np.array([r["n_trades"] for r in per_day])
        total_trades = int(n_trades.sum())
        weighted_net = float((net_means * n_trades).sum() / n_trades.sum())
        daily_sharpe = float(net_means.mean() / net_means.std()) if net_means.std() > 0 else 0.0
        profitable_days = int((net_means > 0).sum())
        agg = {
            "entry_thr": et, "tp": tp, "sl": sl, "side": side,
            "n_days": len(per_day),
            "total_trades": total_trades,
            "trades_per_day": float(n_trades.mean()),
            "weighted_net_per_trade": weighted_net,
            "daily_sharpe": daily_sharpe,
            "profitable_days": profitable_days,
            "pct_profitable_days": profitable_days / len(per_day),
            "mean_exit_horizon": float(np.mean([r["exit_horizon_mean"] for r in per_day])),
        }
        results.append(agg)
        if ci % 10 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Config {ci+1}/{len(configs)} | {side} et={et} tp={tp} sl={sl} | net={weighted_net:+.3f}t Sharpe={daily_sharpe:+.2f} profDays={profitable_days}/{len(per_day)}")

    # Sort by daily Sharpe
    results.sort(key=lambda x: -x["daily_sharpe"])
    out = OUT_DIR / f"results_{time.strftime('%Y%m%d_%H%M')}.json"
    with open(out, "w") as f:
        json.dump({"n_days": len(days), "results": results}, f, indent=2)

    profitable = [r for r in results if r["weighted_net_per_trade"] > 0]
    sharpe_pos = [r for r in results if r["daily_sharpe"] > 0.5]
    print(f"\n=== FINAL ===")
    print(f"Total configs evaluated: {len(results)}")
    print(f"Profitable on weighted net: {len(profitable)}")
    print(f"Sharpe > 0.5: {len(sharpe_pos)}")
    print(f"Top 5 by daily Sharpe:")
    for r in results[:5]:
        print(f"  {r['side']:5s} et={r['entry_thr']} tp={r['tp']} sl={r['sl']} | net={r['weighted_net_per_trade']:+.3f}t Sharpe={r['daily_sharpe']:+.2f} profDays={r['profitable_days']}/{r['n_days']} trades={r['total_trades']}")
    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()
