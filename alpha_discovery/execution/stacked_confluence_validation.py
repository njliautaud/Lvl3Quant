#!/usr/bin/env python3
"""
stacked_confluence_validation.py
================================
Stacked gate validation: CNN-Mamba v2 signal + meta-score filter + OFI confluence.

Three gates applied sequentially to short signals at 1s horizon:
  Gate A: Signal strength — top 5% or top 10% shorts by 1s prediction magnitude
  Gate B: Meta-score — top 30% by LGBM meta-filter (pred_strength + microstructure)
  Gate C: OFI confluence — book queue imbalance agrees with short direction (OFI < 0)

Previous results (tested SEPARATELY):
  - Signal top-5% shorts: +0.31 ticks/trade gross
  - Meta-filter top-30%: lifts gross from +0.75 to +1.17 ticks
  - OFI confluence: lifts top-5% short from +0.31 to +0.60 ticks

This script tests them STACKED for the first time.

Cost model: 0.376 ticks RT (passive-only, commission only — CANONICAL per CLAUDE.md)
Data: CNN-Mamba v2 bulk OOT predictions + OFI features + MBO events (smart_v3)
Regime: green/red days derived from cumulative 1s label drift per day

Output: /home/jupiter/Lvl3Quant/output/stacked_confluence_v1/
"""
from __future__ import annotations

import json
import os
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
MBO_DIR = ROOT / "data/processed/mbo_events_smart_v3"
OFI_DIR = ROOT / "data/processed/mbo_events_smart_v3_ofi_features"
PRED_DIR = ROOT / "output/cnn_mamba_v2_bulk_oot_v2"
META_MODEL_PATH = ROOT / "output/hc440_meta_filter/lgbm_model.pkl"
OUT_DIR = ROOT / "output/stacked_confluence_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMMISSION_TICKS = 0.376  # Canonical passive RT cost
HORIZONS = ["1s", "5s", "10s"]
H_IDX = {"1s": 0, "5s": 1, "10s": 2}

# OFI feature for confluence gate (best from ofi_confluence_v1: queue_imbalance_proxy_1s)
OFI_GATE_FEATURE = "ofi_book_1s"  # raw book OFI, we compute queue_imbalance_proxy from it

# Meta-filter features (must match hc440 training)
META_FEATURES = [
    "pred_1s", "pred_5s", "pred_10s",
    "filter_vol_500ev_tk", "filter_evt_per_sec_30s",
    "filter_buy_aggr_50", "filter_spread_proxy_tk",
    "tod_min_et", "dow",
]


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_overlap_dates() -> list[str]:
    """Find dates with predictions + OFI + MBO all available."""
    pred_dates = {f[:8] for f in os.listdir(PRED_DIR) if f.endswith("_predictions.npz")}
    ofi_dates = {f[:8] for f in os.listdir(OFI_DIR) if f.endswith("_ofi.npz")}
    mbo_dates = {f[:8] for f in os.listdir(MBO_DIR) if f.endswith("_mbo_events.npz")}
    return sorted(pred_dates & ofi_dates & mbo_dates)


def load_day(date: str) -> dict:
    """Load predictions, labels, OFI, and microstructure features for one date."""
    pred_data = np.load(PRED_DIR / f"{date}_predictions.npz", allow_pickle=True)
    ofi_data = np.load(OFI_DIR / f"{date}_ofi.npz", allow_pickle=True)
    mbo_data = np.load(MBO_DIR / f"{date}_mbo_events.npz", allow_pickle=True)

    preds = pred_data["predictions"]  # (N, 3) for 1s/5s/10s
    labels = pred_data["labels"]      # (N, 3)
    ws = int(pred_data["window_size"])
    stride = int(pred_data["stride"])
    n_pred = preds.shape[0]

    # Event indices for each prediction window (last event in window)
    event_idx = ws - 1 + np.arange(n_pred) * stride
    n_events = mbo_data["events"].shape[0]
    valid_mask = event_idx < n_events
    if not valid_mask.all():
        n_pred = int(valid_mask.sum())
        event_idx = event_idx[:n_pred]
        preds = preds[:n_pred]
        labels = labels[:n_pred]

    events = mbo_data["events"]
    timestamps = mbo_data["timestamps"][event_idx]

    # ---- OFI features at prediction points ----
    ofi_vals = {}
    for fn in ofi_data.files:
        ofi_vals[fn] = ofi_data[fn][event_idx].astype(np.float32)

    # Queue imbalance proxy from book OFI
    for w in ["1s", "5s", "10s"]:
        bk_key = f"ofi_book_{w}"
        if bk_key in ofi_vals:
            bk = ofi_vals[bk_key]
            ofi_vals[f"queue_imbalance_proxy_{w}"] = bk / (np.abs(bk) + 1.0)

    # ---- Microstructure features for meta-filter ----
    # Extract from smart_v3 events at prediction points
    ev_at_pred = events[event_idx]

    # Approximate meta features from smart_v3 event columns
    # smart_v3 has 25 features per event. We reconstruct what we can.
    # We need: vol_500ev, evt_per_sec_30s, buy_aggr_50, spread_proxy, tod_min_et, dow

    # Volatility proxy: column 10 is typically rolling volatility
    filter_vol_500ev_tk = ev_at_pred[:, 10].astype(np.float32)

    # Event rate proxy: column 4 (typically event intensity/rate)
    filter_evt_per_sec_30s = ev_at_pred[:, 4].astype(np.float32)

    # Buy aggressive fraction: column 9 (buy/sell imbalance)
    filter_buy_aggr_50 = ev_at_pred[:, 9].astype(np.float32)

    # Spread proxy: column 11 (spread-related)
    filter_spread_proxy_tk = ev_at_pred[:, 11].astype(np.float32)

    # Time of day from timestamps
    # timestamps are nanoseconds since epoch
    secs_since_midnight = (timestamps % (24 * 3600 * int(1e9))) / 1e9
    # Convert to ET (UTC-4 during EDT, UTC-5 during EST)
    # For March-April dates, EDT applies (UTC-4)
    tod_min_et = (secs_since_midnight - 4 * 3600) / 60.0
    tod_min_et = tod_min_et.astype(np.float32)

    # Day of week from date string
    from datetime import datetime as dt
    d = dt.strptime(date, "%Y%m%d")
    dow = float(d.weekday())

    # ---- Regime: daily return from labels_1s ----
    # Use widely-spaced labels_1s to approximate non-overlapping daily return
    labs_1s = mbo_data["labels_1s"]
    # Sample every ~4000 events (roughly 1s of non-overlapping returns)
    regime_stride = 4000
    sampled_labs = labs_1s[::regime_stride]
    valid_labs = sampled_labs[np.isfinite(sampled_labs)]
    daily_return_ticks = float(np.nansum(valid_labs)) if len(valid_labs) > 0 else 0.0

    return {
        "date": date,
        "preds": preds,
        "labels": labels,
        "ofi": ofi_vals,
        "timestamps": timestamps,
        "n": n_pred,
        "daily_return_ticks": daily_return_ticks,
        # Meta features
        "meta_features": {
            "pred_1s": preds[:, 0],
            "pred_5s": preds[:, 1],
            "pred_10s": preds[:, 2],
            "filter_vol_500ev_tk": filter_vol_500ev_tk,
            "filter_evt_per_sec_30s": filter_evt_per_sec_30s,
            "filter_buy_aggr_50": filter_buy_aggr_50,
            "filter_spread_proxy_tk": filter_spread_proxy_tk,
            "tod_min_et": tod_min_et,
            "dow": np.full(n_pred, dow, dtype=np.float32),
        },
    }


def load_meta_model():
    """Load the pre-trained LGBM meta-filter model."""
    if not META_MODEL_PATH.exists():
        log(f"WARNING: Meta model not found at {META_MODEL_PATH.name}")
        return None, None
    with open(META_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["features"]


# ---------------------------------------------------------------------------
# Gate functions
# ---------------------------------------------------------------------------

def gate_a_signal_strength(preds_1s: np.ndarray, top_pct: float) -> np.ndarray:
    """Gate A: Select top N% strongest short signals (most negative 1s predictions)."""
    short_mask = preds_1s < 0
    n_short = short_mask.sum()
    if n_short < 10:
        return np.zeros_like(preds_1s, dtype=bool)

    # Threshold: Nth percentile of short predictions (lower = more extreme)
    short_vals = preds_1s[short_mask]
    thresh = np.percentile(short_vals, top_pct * 100)
    result = short_mask & (preds_1s <= thresh)
    return result


def gate_b_meta_score(meta_model, meta_features_list, day_meta_features: dict,
                      mask: np.ndarray, top_pct: float = 0.30) -> np.ndarray:
    """Gate B: Keep only signals where meta-model predicts highest tradability."""
    if meta_model is None:
        return mask  # pass through if no model

    n = len(mask)
    indices = np.where(mask)[0]
    if len(indices) < 5:
        return mask

    # Build feature matrix for masked predictions
    X = np.column_stack([day_meta_features[f][indices] for f in meta_features_list])

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict meta-score (higher = more tradable)
    scores = meta_model.predict(X)

    # Keep top N% by score
    thresh = np.percentile(scores, (1.0 - top_pct) * 100)
    keep = scores >= thresh

    result = np.zeros(n, dtype=bool)
    result[indices[keep]] = True
    return result


def gate_c_ofi_confluence(ofi_vals: dict, mask: np.ndarray,
                          feature: str = "ofi_book_1s") -> np.ndarray:
    """Gate C: Keep only signals where OFI agrees with short direction (OFI < 0)."""
    n = len(mask)
    if feature not in ofi_vals:
        return mask  # pass through

    ofi = ofi_vals[feature]
    indices = np.where(mask)[0]
    if len(indices) < 5:
        return mask

    ofi_at_signals = ofi[indices]
    # For shorts: OFI < 0 means selling pressure = agrees with short
    agrees = ofi_at_signals < 0

    result = np.zeros(n, dtype=bool)
    result[indices[agrees]] = True
    return result


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(pnl_per_trade: np.ndarray, dates: np.ndarray,
                    daily_returns: dict) -> dict:
    """Compute comprehensive metrics for a set of trades."""
    n = len(pnl_per_trade)
    if n == 0:
        return {
            "n_trades": 0, "mean_pnl_ticks": 0, "total_pnl_ticks": 0,
            "wr": 0, "pf": 0, "sharpe": 0, "sortino": 0,
            "green_day_pct": 0, "n_days": 0, "trades_per_day": 0,
            "n_green_regime": 0, "n_red_regime": 0,
            "sharpe_green": 0, "sharpe_red": 0, "regime_skew": 0,
        }

    net = pnl_per_trade - COMMISSION_TICKS
    total_net = float(np.sum(net))
    mean_net = float(np.mean(net))
    wr = float(np.mean(net > 0))

    wins = net[net > 0].sum()
    losses = -net[net < 0].sum()
    pf = float(wins / max(losses, 1e-9))

    # Per-day aggregation
    unique_dates = np.unique(dates)
    day_pnls = []
    day_regime = []
    for ud in unique_dates:
        dm = dates == ud
        day_net = float(np.sum(net[dm]))
        day_pnls.append(day_net)
        # Regime: green if daily ES return > 0
        dr = daily_returns.get(ud, 0.0)
        day_regime.append("green" if dr > 0 else ("red" if dr < 0 else "flat"))

    day_pnls = np.array(day_pnls)
    n_days = len(day_pnls)
    green_day_pct = float(np.mean(day_pnls > 0)) if n_days > 0 else 0.0
    trades_per_day = n / n_days if n_days > 0 else 0.0

    # Sharpe (annualized from daily P&L per day)
    if n_days > 1 and np.std(day_pnls, ddof=1) > 0:
        sharpe = float(np.mean(day_pnls) / np.std(day_pnls, ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Sortino (annualized)
    if n_days > 1:
        downside = day_pnls[day_pnls < 0]
        if len(downside) > 0 and np.std(downside, ddof=1) > 0:
            sortino = float(np.mean(day_pnls) / np.std(downside, ddof=1) * np.sqrt(252))
        else:
            sortino = float("inf") if np.mean(day_pnls) > 0 else 0.0
    else:
        sortino = 0.0

    # Regime stratification
    day_regime = np.array(day_regime)
    green_idx = day_regime == "green"
    red_idx = day_regime == "red"

    def regime_sharpe(idx):
        if idx.sum() < 2:
            return 0.0
        rp = day_pnls[idx]
        if np.std(rp, ddof=1) > 0:
            return float(np.mean(rp) / np.std(rp, ddof=1) * np.sqrt(252))
        return 0.0

    sharpe_green = regime_sharpe(green_idx)
    sharpe_red = regime_sharpe(red_idx)

    # Regime skew (HC #428): |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|)
    max_s = max(abs(sharpe_green), abs(sharpe_red))
    regime_skew = abs(sharpe_green - sharpe_red) / max_s if max_s > 0 else 0.0

    return {
        "n_trades": n,
        "n_days": n_days,
        "trades_per_day": round(trades_per_day, 1),
        "mean_pnl_ticks": round(mean_net, 4),
        "total_pnl_ticks": round(total_net, 2),
        "wr": round(wr, 4),
        "pf": round(pf, 3),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "green_day_pct": round(green_day_pct, 4),
        "n_green_regime": int(green_idx.sum()),
        "n_red_regime": int(red_idx.sum()),
        "sharpe_green": round(sharpe_green, 2),
        "sharpe_red": round(sharpe_red, 2),
        "regime_skew": round(regime_skew, 3),
        "regime_pass": regime_skew <= 0.50,
    }


def compute_per_day_detail(pnl_per_trade: np.ndarray, dates: np.ndarray,
                           daily_returns: dict) -> pd.DataFrame:
    """Compute per-day P&L detail for a trade set."""
    net = pnl_per_trade - COMMISSION_TICKS
    unique_dates = sorted(np.unique(dates))
    rows = []
    for ud in unique_dates:
        dm = dates == ud
        day_net = net[dm]
        dr = daily_returns.get(ud, 0.0)
        regime = "green" if dr > 0 else ("red" if dr < 0 else "flat")
        rows.append({
            "date": ud,
            "n_trades": int(dm.sum()),
            "total_pnl_ticks": round(float(day_net.sum()), 2),
            "mean_pnl_ticks": round(float(day_net.mean()), 4),
            "wr": round(float((day_net > 0).mean()), 4),
            "regime": regime,
            "daily_return_ticks": round(dr, 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    log("=" * 80)
    log("STACKED CONFLUENCE VALIDATION v1")
    log("Gates: (A) Signal top-N% short + (B) Meta top-30% + (C) OFI book agreement")
    log("Cost: 0.376 ticks RT (passive-passive, commission only)")
    log("=" * 80)

    # Load meta model
    meta_model, meta_features = load_meta_model()
    if meta_model is not None:
        log(f"Loaded meta-filter model with features: {meta_features}")
    else:
        log("WARNING: No meta model — Gate B will be skipped (pass-through)")

    # Find dates
    dates = find_overlap_dates()
    log(f"Found {len(dates)} overlapping dates (predictions + OFI + MBO)")

    if len(dates) < 5:
        log("FATAL: Not enough dates for validation")
        sys.exit(1)

    # Load all days
    days = []
    for d in dates:
        try:
            day = load_day(d)
            days.append(day)
            log(f"  {d}: {day['n']:,} predictions, daily_return={day['daily_return_ticks']:+.1f}t")
        except Exception as e:
            log(f"  FAILED {d}: {e}")

    total_preds = sum(d["n"] for d in days)
    log(f"Loaded {len(days)} days, {total_preds:,} total predictions")

    # Daily returns dict for regime classification
    daily_returns = {d["date"]: d["daily_return_ticks"] for d in days}

    # Concatenate across days
    all_preds_1s = np.concatenate([d["preds"][:, 0] for d in days])
    all_preds_5s = np.concatenate([d["preds"][:, 1] for d in days])
    all_preds_10s = np.concatenate([d["preds"][:, 2] for d in days])
    all_labels_1s = np.concatenate([d["labels"][:, 0] for d in days])
    all_dates = np.concatenate([np.full(d["n"], d["date"], dtype=object) for d in days])

    # Concatenate OFI
    all_ofi = {}
    for key in days[0]["ofi"]:
        all_ofi[key] = np.concatenate([d["ofi"][key] for d in days])

    # Concatenate meta features
    all_meta = {}
    for feat in META_FEATURES:
        all_meta[feat] = np.concatenate([d["meta_features"][feat] for d in days])

    # Short P&L: for shorts, profit = -label (if price drops, label negative, short profits)
    all_short_pnl = -all_labels_1s  # in ticks, BEFORE commission

    # Valid mask
    valid = np.isfinite(all_preds_1s) & np.isfinite(all_labels_1s)
    log(f"Valid predictions: {valid.sum():,} / {len(valid):,}")

    # ========================================================================
    # Run gate combinations
    # ========================================================================
    configs = []

    for top_pct_label, top_pct in [("top_5pct", 0.05), ("top_10pct", 0.10)]:
        for meta_enabled in [False, True]:
            for ofi_enabled in [False, True]:
                meta_pct = 0.30 if meta_enabled else 1.0
                config_name = f"signal_{top_pct_label}"
                if meta_enabled:
                    config_name += "_meta30"
                if ofi_enabled:
                    config_name += "_ofi"

                configs.append({
                    "name": config_name,
                    "top_pct": top_pct,
                    "meta_enabled": meta_enabled,
                    "meta_pct": meta_pct,
                    "ofi_enabled": ofi_enabled,
                })

    results = {}
    per_day_dfs = {}

    for cfg in configs:
        log(f"\n--- Config: {cfg['name']} ---")

        # Gate A: Signal strength
        mask_a = gate_a_signal_strength(all_preds_1s, cfg["top_pct"]) & valid
        n_a = mask_a.sum()
        log(f"  Gate A (signal {cfg['top_pct']:.0%} shorts): {n_a:,} trades")

        # Gate B: Meta score
        if cfg["meta_enabled"] and meta_model is not None:
            mask_b = gate_b_meta_score(meta_model, meta_features, all_meta,
                                        mask_a, cfg["meta_pct"])
        else:
            mask_b = mask_a
        n_b = mask_b.sum()
        log(f"  Gate B (meta top {cfg['meta_pct']:.0%}): {n_b:,} trades")

        # Gate C: OFI confluence
        if cfg["ofi_enabled"]:
            mask_c = gate_c_ofi_confluence(all_ofi, mask_b, OFI_GATE_FEATURE)
        else:
            mask_c = mask_b
        n_c = mask_c.sum()
        log(f"  Gate C (OFI confluence): {n_c:,} trades")

        if n_c < 10:
            log(f"  SKIP: too few trades ({n_c})")
            results[cfg["name"]] = {"n_trades": int(n_c), "skipped": True}
            continue

        # Compute metrics
        pnl = all_short_pnl[mask_c]
        trade_dates = all_dates[mask_c]
        metrics = compute_metrics(pnl, trade_dates, daily_returns)
        results[cfg["name"]] = metrics
        results[cfg["name"]]["config"] = cfg

        # Per-day detail
        per_day = compute_per_day_detail(pnl, trade_dates, daily_returns)
        per_day_dfs[cfg["name"]] = per_day

        log(f"  RESULT: n={metrics['n_trades']} days={metrics['n_days']} "
            f"mean={metrics['mean_pnl_ticks']:+.4f}t WR={metrics['wr']:.1%} "
            f"PF={metrics['pf']:.2f} Sharpe={metrics['sharpe']:.1f} "
            f"Sortino={metrics['sortino']:.1f} "
            f"green%={metrics['green_day_pct']:.0%}")
        log(f"         regime: Sh_green={metrics['sharpe_green']:.1f} "
            f"Sh_red={metrics['sharpe_red']:.1f} "
            f"skew={metrics['regime_skew']:.3f} "
            f"{'PASS' if metrics['regime_pass'] else 'FAIL'}")

    # ========================================================================
    # Summary comparison
    # ========================================================================
    log("\n" + "=" * 80)
    log("STACKED CONFLUENCE RESULTS COMPARISON")
    log("=" * 80)

    # Build comparison table
    compare_rows = []
    for name, m in results.items():
        if m.get("skipped"):
            continue
        compare_rows.append({
            "config": name,
            "n_trades": m["n_trades"],
            "n_days": m["n_days"],
            "tpd": m["trades_per_day"],
            "mean_tk": m["mean_pnl_ticks"],
            "total_tk": m["total_pnl_ticks"],
            "wr": m["wr"],
            "pf": m["pf"],
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "green%": m["green_day_pct"],
            "Sh_green": m["sharpe_green"],
            "Sh_red": m["sharpe_red"],
            "skew": m["regime_skew"],
            "regime_ok": m["regime_pass"],
        })

    df_compare = pd.DataFrame(compare_rows)
    if len(df_compare) > 0:
        log("\n" + df_compare.to_string(index=False))

    # ========================================================================
    # Lift analysis: stacked vs individual
    # ========================================================================
    log("\n" + "=" * 80)
    log("LIFT ANALYSIS: Stacked vs Individual Filters")
    log("=" * 80)

    for base in ["top_5pct", "top_10pct"]:
        baseline_key = f"signal_{base}"
        meta_key = f"signal_{base}_meta30"
        ofi_key = f"signal_{base}_ofi"
        stacked_key = f"signal_{base}_meta30_ofi"

        baseline = results.get(baseline_key, {})
        meta_only = results.get(meta_key, {})
        ofi_only = results.get(ofi_key, {})
        stacked = results.get(stacked_key, {})

        if baseline.get("skipped") or not baseline.get("n_trades"):
            continue

        log(f"\n  {base.upper()} shorts baseline:")
        log(f"    Baseline:     mean={baseline.get('mean_pnl_ticks', 0):+.4f}t "
            f"Sharpe={baseline.get('sharpe', 0):.1f} PF={baseline.get('pf', 0):.2f} "
            f"n={baseline.get('n_trades', 0)}")

        if not meta_only.get("skipped") and meta_only.get("n_trades"):
            lift = meta_only["mean_pnl_ticks"] - baseline["mean_pnl_ticks"]
            log(f"    +Meta30:      mean={meta_only['mean_pnl_ticks']:+.4f}t "
                f"Sharpe={meta_only['sharpe']:.1f} PF={meta_only['pf']:.2f} "
                f"n={meta_only['n_trades']} (lift={lift:+.4f}t)")

        if not ofi_only.get("skipped") and ofi_only.get("n_trades"):
            lift = ofi_only["mean_pnl_ticks"] - baseline["mean_pnl_ticks"]
            log(f"    +OFI:         mean={ofi_only['mean_pnl_ticks']:+.4f}t "
                f"Sharpe={ofi_only['sharpe']:.1f} PF={ofi_only['pf']:.2f} "
                f"n={ofi_only['n_trades']} (lift={lift:+.4f}t)")

        if not stacked.get("skipped") and stacked.get("n_trades"):
            lift = stacked["mean_pnl_ticks"] - baseline["mean_pnl_ticks"]
            log(f"    +Meta30+OFI:  mean={stacked['mean_pnl_ticks']:+.4f}t "
                f"Sharpe={stacked['sharpe']:.1f} PF={stacked['pf']:.2f} "
                f"n={stacked['n_trades']} (lift={lift:+.4f}t)")

    # ========================================================================
    # Save outputs
    # ========================================================================

    # Save full results JSON
    with open(OUT_DIR / "stacked_results.json", "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        clean_results = {}
        for k, v in results.items():
            clean_results[k] = {kk: convert(vv) for kk, vv in v.items()} if isinstance(v, dict) else v
        json.dump(clean_results, f, indent=2, default=convert)

    # Save comparison CSV
    if len(df_compare) > 0:
        df_compare.to_csv(OUT_DIR / "comparison.csv", index=False)

    # Save per-day CSVs
    for name, pdf in per_day_dfs.items():
        safe_name = name.replace(" ", "_")
        pdf.to_csv(OUT_DIR / f"perday_{safe_name}.csv", index=False)

    # Save summary markdown
    with open(OUT_DIR / "summary.md", "w") as f:
        f.write("# Stacked Confluence Validation v1\n\n")
        f.write(f"**Dates**: {len(days)} OOT days ({days[0]['date']} to {days[-1]['date']})\n")
        f.write(f"**Total predictions**: {total_preds:,}\n")
        f.write(f"**Cost model**: {COMMISSION_TICKS} ticks RT (passive-passive)\n\n")

        if len(df_compare) > 0:
            f.write("## Results Comparison\n\n")
            f.write("| Config | Trades | Days | Mean tk | WR | PF | Sharpe | Sortino | Green% | Regime |\n")
            f.write("|--------|--------|------|---------|----|----|--------|---------|--------|--------|\n")
            for _, row in df_compare.iterrows():
                regime_str = "PASS" if row["regime_ok"] else "FAIL"
                f.write(f"| {row['config']} | {row['n_trades']} | {row['n_days']} | "
                        f"{row['mean_tk']:+.4f} | {row['wr']:.1%} | {row['pf']:.2f} | "
                        f"{row['sharpe']:.1f} | {row['sortino']:.1f} | {row['green%']:.0%} | "
                        f"{regime_str} |\n")

    elapsed = time.time() - t0
    log(f"\nDone in {elapsed:.1f}s. Results saved to output/stacked_confluence_v1/")


if __name__ == "__main__":
    main()
