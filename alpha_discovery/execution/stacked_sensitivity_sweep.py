#!/usr/bin/env python3
"""
stacked_sensitivity_sweep.py
============================
Parameter sensitivity sweep over the stacked confluence gates.

Sweeps:
  - Signal threshold: [2%, 3%, 5%, 7%, 10%, 15%, 20%] (shorts only)
  - Meta-filter percentile: [10%, 20%, 30%, 40%, 50%, 100% (=OFF)]
  - OFI gate: [ON, OFF]

Total: 7 x 6 x 2 = 84 configs

For each config: mean_pnl_ticks, WR, PF, Sharpe, trades_per_day, green_day_pct, regime_skew.
Cost = 0.376 ticks (passive-passive, canonical).
Pareto frontier: configs NOT dominated on both Sharpe AND mean_pnl_ticks.
HC #428 regime gate: reject if regime_skew > 0.50.

Output: /home/jupiter/Lvl3Quant/output/stacked_sensitivity_v1/
"""
from __future__ import annotations

import json
import os
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (same as stacked_confluence_validation.py)
# ---------------------------------------------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
MBO_DIR = ROOT / "data/processed/mbo_events_smart_v3"
OFI_DIR = ROOT / "data/processed/mbo_events_smart_v3_ofi_features"
PRED_DIR = ROOT / "output/cnn_mamba_v2_bulk_oot_v2"
META_MODEL_PATH = ROOT / "output/hc440_meta_filter/lgbm_model.pkl"
OUT_DIR = ROOT / "output/stacked_sensitivity_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMMISSION_TICKS = 0.376

# Sweep grid
SIGNAL_THRESHOLDS = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
META_PERCENTILES = [0.10, 0.20, 0.30, 0.40, 0.50, 1.00]  # 1.0 = no filter
OFI_OPTIONS = [True, False]

OFI_GATE_FEATURE = "ofi_book_1s"

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
# Data loading (reused from stacked_confluence_validation.py)
# ---------------------------------------------------------------------------

def find_overlap_dates() -> list[str]:
    pred_dates = {f[:8] for f in os.listdir(PRED_DIR) if f.endswith("_predictions.npz")}
    ofi_dates = {f[:8] for f in os.listdir(OFI_DIR) if f.endswith("_ofi.npz")}
    mbo_dates = {f[:8] for f in os.listdir(MBO_DIR) if f.endswith("_mbo_events.npz")}
    return sorted(pred_dates & ofi_dates & mbo_dates)


def load_day(date: str) -> dict:
    from datetime import datetime as dt

    pred_data = np.load(PRED_DIR / f"{date}_predictions.npz", allow_pickle=True)
    ofi_data = np.load(OFI_DIR / f"{date}_ofi.npz", allow_pickle=True)
    mbo_data = np.load(MBO_DIR / f"{date}_mbo_events.npz", allow_pickle=True)

    preds = pred_data["predictions"]
    labels = pred_data["labels"]
    ws = int(pred_data["window_size"])
    stride = int(pred_data["stride"])
    n_pred = preds.shape[0]

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

    # OFI features
    ofi_vals = {}
    for fn in ofi_data.files:
        ofi_vals[fn] = ofi_data[fn][event_idx].astype(np.float32)
    for w in ["1s", "5s", "10s"]:
        bk_key = f"ofi_book_{w}"
        if bk_key in ofi_vals:
            bk = ofi_vals[bk_key]
            ofi_vals[f"queue_imbalance_proxy_{w}"] = bk / (np.abs(bk) + 1.0)

    # Microstructure features for meta-filter
    ev_at_pred = events[event_idx]
    filter_vol_500ev_tk = ev_at_pred[:, 10].astype(np.float32)
    filter_evt_per_sec_30s = ev_at_pred[:, 4].astype(np.float32)
    filter_buy_aggr_50 = ev_at_pred[:, 9].astype(np.float32)
    filter_spread_proxy_tk = ev_at_pred[:, 11].astype(np.float32)

    secs_since_midnight = (timestamps % (24 * 3600 * int(1e9))) / 1e9
    tod_min_et = (secs_since_midnight - 4 * 3600) / 60.0
    tod_min_et = tod_min_et.astype(np.float32)

    d = dt.strptime(date, "%Y%m%d")
    dow = float(d.weekday())

    # Regime
    labs_1s = mbo_data["labels_1s"]
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
    if not META_MODEL_PATH.exists():
        log(f"WARNING: Meta model not found at {META_MODEL_PATH.name}")
        return None, None
    with open(META_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["features"]


# ---------------------------------------------------------------------------
# Gate functions
# ---------------------------------------------------------------------------

def gate_signal(preds_1s: np.ndarray, top_pct: float) -> np.ndarray:
    short_mask = preds_1s < 0
    n_short = short_mask.sum()
    if n_short < 10:
        return np.zeros_like(preds_1s, dtype=bool)
    short_vals = preds_1s[short_mask]
    thresh = np.percentile(short_vals, top_pct * 100)
    return short_mask & (preds_1s <= thresh)


def gate_meta(meta_model, meta_features_list, day_meta_features: dict,
              mask: np.ndarray, top_pct: float) -> np.ndarray:
    if meta_model is None or top_pct >= 1.0:
        return mask
    n = len(mask)
    indices = np.where(mask)[0]
    if len(indices) < 5:
        return mask
    X = np.column_stack([day_meta_features[f][indices] for f in meta_features_list])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scores = meta_model.predict(X)
    thresh = np.percentile(scores, (1.0 - top_pct) * 100)
    keep = scores >= thresh
    result = np.zeros(n, dtype=bool)
    result[indices[keep]] = True
    return result


def gate_ofi(ofi_vals: dict, mask: np.ndarray) -> np.ndarray:
    n = len(mask)
    if OFI_GATE_FEATURE not in ofi_vals:
        return mask
    ofi = ofi_vals[OFI_GATE_FEATURE]
    indices = np.where(mask)[0]
    if len(indices) < 5:
        return mask
    agrees = ofi[indices] < 0
    result = np.zeros(n, dtype=bool)
    result[indices[agrees]] = True
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pnl_per_trade: np.ndarray, dates: np.ndarray,
                    daily_returns: dict) -> dict:
    n = len(pnl_per_trade)
    if n < 10:
        return None

    net = pnl_per_trade - COMMISSION_TICKS
    mean_net = float(np.mean(net))
    wr = float(np.mean(net > 0))
    wins = net[net > 0].sum()
    losses = -net[net < 0].sum()
    pf = float(wins / max(losses, 1e-9))

    unique_dates = np.unique(dates)
    day_pnls = []
    day_regime = []
    for ud in unique_dates:
        dm = dates == ud
        day_pnls.append(float(np.sum(net[dm])))
        dr = daily_returns.get(ud, 0.0)
        day_regime.append("green" if dr > 0 else ("red" if dr < 0 else "flat"))

    day_pnls = np.array(day_pnls)
    n_days = len(day_pnls)
    green_day_pct = float(np.mean(day_pnls > 0)) if n_days > 0 else 0.0
    trades_per_day = n / n_days if n_days > 0 else 0.0

    if n_days > 1 and np.std(day_pnls, ddof=1) > 0:
        sharpe = float(np.mean(day_pnls) / np.std(day_pnls, ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0

    if n_days > 1:
        downside = day_pnls[day_pnls < 0]
        if len(downside) > 0 and np.std(downside, ddof=1) > 0:
            sortino = float(np.mean(day_pnls) / np.std(downside, ddof=1) * np.sqrt(252))
        else:
            sortino = float("inf") if np.mean(day_pnls) > 0 else 0.0
    else:
        sortino = 0.0

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
    max_s = max(abs(sharpe_green), abs(sharpe_red))
    regime_skew = abs(sharpe_green - sharpe_red) / max_s if max_s > 0 else 0.0

    return {
        "n_trades": n,
        "n_days": n_days,
        "trades_per_day": round(trades_per_day, 1),
        "mean_pnl_ticks": round(mean_net, 4),
        "total_pnl_ticks": round(float(np.sum(net)), 2),
        "wr": round(wr, 4),
        "pf": round(pf, 3),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "green_day_pct": round(green_day_pct, 4),
        "sharpe_green": round(sharpe_green, 2),
        "sharpe_red": round(sharpe_red, 2),
        "regime_skew": round(regime_skew, 3),
        "regime_pass": regime_skew <= 0.50,
    }


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def pareto_frontier(df: pd.DataFrame, obj1: str = "sharpe", obj2: str = "mean_pnl_ticks") -> pd.DataFrame:
    """Find rows NOT dominated on both objectives (higher is better for both)."""
    is_pareto = np.ones(len(df), dtype=bool)
    vals = df[[obj1, obj2]].values
    for i in range(len(vals)):
        for j in range(len(vals)):
            if i == j:
                continue
            # j dominates i if j >= i on both and strictly > on at least one
            if vals[j, 0] >= vals[i, 0] and vals[j, 1] >= vals[i, 1]:
                if vals[j, 0] > vals[i, 0] or vals[j, 1] > vals[i, 1]:
                    is_pareto[i] = False
                    break
    return df[is_pareto].copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    log("=" * 80)
    log("STACKED CONFLUENCE SENSITIVITY SWEEP")
    log(f"Signal thresholds: {[f'{x:.0%}' for x in SIGNAL_THRESHOLDS]}")
    log(f"Meta percentiles:  {[f'{x:.0%}' for x in META_PERCENTILES]}")
    log(f"OFI options:       ON/OFF")
    log(f"Total configs:     {len(SIGNAL_THRESHOLDS) * len(META_PERCENTILES) * len(OFI_OPTIONS)}")
    log(f"Cost: {COMMISSION_TICKS} ticks RT (passive-passive)")
    log("=" * 80)

    # Load meta model
    meta_model, meta_features = load_meta_model()
    if meta_model is not None:
        log(f"Loaded meta-filter model ({len(meta_features)} features)")
    else:
        log("WARNING: No meta model — meta gate will pass through at all percentiles")

    # Load data
    dates = find_overlap_dates()
    log(f"Found {len(dates)} overlapping dates")

    days = []
    for d in dates:
        try:
            day = load_day(d)
            days.append(day)
        except Exception as e:
            log(f"  FAILED {d}: {e}")

    total_preds = sum(d["n"] for d in days)
    log(f"Loaded {len(days)} days, {total_preds:,} total predictions")

    daily_returns = {d["date"]: d["daily_return_ticks"] for d in days}

    # Concatenate all data
    all_preds_1s = np.concatenate([d["preds"][:, 0] for d in days])
    all_labels_1s = np.concatenate([d["labels"][:, 0] for d in days])
    all_dates = np.concatenate([np.full(d["n"], d["date"], dtype=object) for d in days])
    all_ofi = {}
    for key in days[0]["ofi"]:
        all_ofi[key] = np.concatenate([d["ofi"][key] for d in days])
    all_meta = {}
    for feat in META_FEATURES:
        all_meta[feat] = np.concatenate([d["meta_features"][feat] for d in days])

    all_short_pnl = -all_labels_1s  # shorts profit when price drops
    valid = np.isfinite(all_preds_1s) & np.isfinite(all_labels_1s)
    log(f"Valid predictions: {valid.sum():,} / {len(valid):,}")

    # ========================================================================
    # Sweep
    # ========================================================================
    results = []
    total_configs = len(SIGNAL_THRESHOLDS) * len(META_PERCENTILES) * len(OFI_OPTIONS)
    cfg_num = 0

    for sig_pct in SIGNAL_THRESHOLDS:
        # Gate A only depends on signal threshold — compute once
        mask_a = gate_signal(all_preds_1s, sig_pct) & valid
        n_a = mask_a.sum()

        for meta_pct in META_PERCENTILES:
            # Gate B
            mask_b = gate_meta(meta_model, meta_features, all_meta, mask_a, meta_pct)
            n_b = mask_b.sum()

            for ofi_on in OFI_OPTIONS:
                cfg_num += 1
                # Gate C
                if ofi_on:
                    mask_c = gate_ofi(all_ofi, mask_b)
                else:
                    mask_c = mask_b
                n_c = mask_c.sum()

                label = (f"sig{sig_pct:.0%}_meta{meta_pct:.0%}"
                         f"{'_ofi' if ofi_on else ''}")

                if n_c < 10:
                    if cfg_num % 20 == 0:
                        log(f"  [{cfg_num}/{total_configs}] {label}: SKIP ({n_c} trades)")
                    continue

                pnl = all_short_pnl[mask_c]
                trade_dates = all_dates[mask_c]
                m = compute_metrics(pnl, trade_dates, daily_returns)
                if m is None:
                    continue

                m["config"] = label
                m["signal_pct"] = sig_pct
                m["meta_pct"] = meta_pct
                m["ofi_on"] = ofi_on
                results.append(m)

                if cfg_num % 10 == 0:
                    log(f"  [{cfg_num}/{total_configs}] {label}: "
                        f"n={m['n_trades']} Sharpe={m['sharpe']:.1f} "
                        f"mean={m['mean_pnl_ticks']:+.4f}t "
                        f"skew={m['regime_skew']:.3f}")

    log(f"\nSweep done. {len(results)} valid configs out of {total_configs}.")

    # ========================================================================
    # Build results DataFrame
    # ========================================================================
    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

    # ========================================================================
    # HC #428 regime gate
    # ========================================================================
    df["regime_pass"] = df["regime_skew"] <= 0.50
    n_pass = df["regime_pass"].sum()
    n_fail = (~df["regime_pass"]).sum()
    log(f"Regime gate (HC #428): {n_pass} PASS, {n_fail} FAIL (skew > 0.50)")

    # ========================================================================
    # Pareto frontier (on regime-passing configs only)
    # ========================================================================
    df_pass = df[df["regime_pass"]].copy()
    if len(df_pass) > 0:
        pareto = pareto_frontier(df_pass, "sharpe", "mean_pnl_ticks")
        pareto = pareto.sort_values("sharpe", ascending=False).reset_index(drop=True)
        log(f"Pareto frontier: {len(pareto)} configs (non-dominated on Sharpe + mean_pnl_ticks)")
    else:
        pareto = pd.DataFrame()
        log("WARNING: No configs pass regime gate!")

    # ========================================================================
    # Print top 5 Pareto configs
    # ========================================================================
    display_cols = ["config", "n_trades", "trades_per_day", "mean_pnl_ticks",
                    "wr", "pf", "sharpe", "sortino", "green_day_pct",
                    "sharpe_green", "sharpe_red", "regime_skew"]

    log("\n" + "=" * 80)
    log("TOP 5 PARETO-OPTIMAL CONFIGS (sorted by Sharpe, regime-pass only)")
    log("=" * 80)

    if len(pareto) > 0:
        top5 = pareto.head(5)
        for i, row in top5.iterrows():
            log(f"\n  #{top5.index.get_loc(i)+1}: {row['config']}")
            log(f"     Trades: {row['n_trades']} ({row['trades_per_day']:.1f}/day)")
            log(f"     Mean P&L: {row['mean_pnl_ticks']:+.4f} ticks/trade")
            log(f"     WR: {row['wr']:.1%}  PF: {row['pf']:.2f}")
            log(f"     Sharpe: {row['sharpe']:.2f}  Sortino: {row['sortino']:.2f}")
            log(f"     Green days: {row['green_day_pct']:.0%}")
            log(f"     Regime: Sh_green={row['sharpe_green']:.1f} Sh_red={row['sharpe_red']:.1f} skew={row['regime_skew']:.3f}")
    else:
        log("  No Pareto-optimal configs found.")

    # ========================================================================
    # Also print top 5 overall by Sharpe (regime-pass)
    # ========================================================================
    log("\n" + "=" * 80)
    log("TOP 5 CONFIGS BY SHARPE (regime-pass only)")
    log("=" * 80)

    if len(df_pass) > 0:
        top5_sharpe = df_pass.sort_values("sharpe", ascending=False).head(5)
        log("\n" + top5_sharpe[display_cols].to_string(index=False))

    # ========================================================================
    # Print full grid summary
    # ========================================================================
    log("\n" + "=" * 80)
    log("FULL GRID (sorted by Sharpe)")
    log("=" * 80)
    if len(df) > 0:
        summary_cols = ["config", "n_trades", "trades_per_day", "mean_pnl_ticks",
                        "wr", "pf", "sharpe", "green_day_pct", "regime_skew", "regime_pass"]
        log("\n" + df[summary_cols].to_string(index=False))

    # ========================================================================
    # Save outputs
    # ========================================================================
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

    # Full grid CSV
    df.to_csv(OUT_DIR / "full_grid.csv", index=False)
    log(f"\nSaved full grid ({len(df)} rows) to full_grid.csv")

    # Pareto CSV
    if len(pareto) > 0:
        pareto.to_csv(OUT_DIR / "pareto_frontier.csv", index=False)
        log(f"Saved Pareto frontier ({len(pareto)} rows) to pareto_frontier.csv")

    # Full results JSON
    with open(OUT_DIR / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)
    log("Saved sweep_results.json")

    # Pareto JSON
    if len(pareto) > 0:
        pareto_records = pareto.to_dict(orient="records")
        with open(OUT_DIR / "pareto_results.json", "w") as f:
            json.dump(pareto_records, f, indent=2, default=convert)
        log("Saved pareto_results.json")

    elapsed = time.time() - t0
    log(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
