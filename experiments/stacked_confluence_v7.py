#!/usr/bin/env python3
"""
stacked_confluence_v7.py
========================
Stacked confluence validation for meta v7 production predictions.

Tests whether OFI (order-flow imbalance) stacking still adds lift on top of v7's
much stronger signal (Spearman 0.271 vs v6's 0.167).

Previous finding: OFI stacking with v6 lifted edge from +0.31 to +0.60 ticks/trade.
Question: Does OFI confluence still add incremental edge on v7's stronger signal?

Gates applied sequentially to SHORT signals:
  Gate A: Signal strength — top X% shorts by v7 prediction magnitude
  Gate B: Meta score — top Y% by v7 prediction confidence (|pred| percentile)
  Gate C: OFI confluence — book/trade imbalance agrees with short direction

Sweep:
  Signal thresholds: 3%, 5%, 10%, 20%
  Meta thresholds: 10%, 30%, 50%, 100% (100% = no filter)
  OFI: on / off

Cost model: 0.376 ticks RT (passive-only, commission only — CANONICAL)
Regime gate: HC #428 R1 — 40-day regime-agnostic validation

Output: /home/nick/Lvl3Quant/output/stacked_confluence_v7/
MLflow experiment: stacked_confluence_v7
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/home/nick/Lvl3Quant")
MBO_DIR = ROOT / "data/processed/mbo_events_smart_v3"
CM_DIR = ROOT / "output/cnn_mamba_v2_bulk_oot"  # for window_size/stride
V7_DIR = ROOT / "output/meta_v7_prod"
OUT_DIR = ROOT / "output/stacked_confluence_v7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_DIR / "stacked_confluence_v7.log"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMMISSION_TICKS = 0.376  # Canonical passive RT cost

# OFI feature indices in smart_v3 events (25 features):
# Based on mbo_features.py mapping to smart_v3 compressed format:
#   col[2]  = vol_imbalance (normalized, [-1, 1]) — bid/ask volume ratio
#   col[6]  = cancel_to_add-like flow signal
#   col[7]  = trade_imbalance (normalized, [-5, 5]) — net aggressive flow
#   col[9]  = book pressure signal
#   col[10] = order flow imbalance (normalized, [-5, 5])
#   col[15] = signed trade flow
#   col[19] = queue imbalance proxy ([-1, 1])
# We'll test several as OFI proxy.
OFI_COLUMNS = {
    "vol_imbalance": 2,       # Volume imbalance (bid vs ask volume)
    "trade_imbalance": 7,     # Net aggressive trade imbalance
    "ofi_flow": 10,           # Order flow imbalance
    "signed_flow": 15,        # Signed trade flow
    "queue_imbalance": 19,    # Queue imbalance proxy
}

# Which OFI column to use as primary gate (trade_imbalance is most interpretable)
PRIMARY_OFI = "trade_imbalance"

# Sweep parameters
SIGNAL_THRESHOLDS = [0.03, 0.05, 0.10, 0.20]  # top 3%, 5%, 10%, 20%
META_THRESHOLDS = [0.10, 0.30, 0.50, 1.00]     # 10%, 30%, 50%, 100% (passthrough)
OFI_OPTIONS = [False, True]

# CNN-Mamba v2 parameters (verified from data)
CM_WINDOW_SIZE = 3000
CM_STRIDE = 250


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_v7_predictions() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load v7 concat OOT predictions.
    Returns: (predictions, labels, dates) — all 1D arrays.
    """
    path = V7_DIR / "concat_oot_predictions.npz"
    d = np.load(path, allow_pickle=True)
    return d["predictions"], d["labels"], d["dates"]


def compute_event_indices(n_preds: int) -> np.ndarray:
    """Compute event indices for each prediction using CNN-Mamba v2 stride.
    event_idx[i] = window_size + i * stride
    """
    return CM_WINDOW_SIZE + np.arange(n_preds) * CM_STRIDE


def load_mbo_day(date: str) -> Optional[dict]:
    """Load MBO events for one date. Returns dict with events, labels, timestamps."""
    path = MBO_DIR / f"{date}_mbo_events.npz"
    if not path.exists():
        return None
    try:
        d = np.load(path, allow_pickle=True)
        return {
            "events": d["events"],
            "labels_1s": d["labels_1s"],
            "timestamps": d["timestamps"],
            "n_events": d["events"].shape[0],
        }
    except Exception as e:
        log(f"  WARNING: Failed to load MBO for {date}: {e}")
        return None


def extract_ofi_at_predictions(mbo: dict, event_idx: np.ndarray) -> dict:
    """Extract OFI features from MBO events at prediction event indices."""
    valid = event_idx < mbo["n_events"]
    n_valid = valid.sum()

    ofi = {}
    for name, col_idx in OFI_COLUMNS.items():
        vals = np.full(len(event_idx), np.nan, dtype=np.float32)
        vals[valid] = mbo["events"][event_idx[valid], col_idx]
        ofi[name] = vals

    return ofi


def classify_regime(mbo: dict) -> float:
    """Classify day as green/red based on cumulative 1s label drift.
    Returns daily return in ticks (positive = green day).
    """
    labs = mbo["labels_1s"]
    # Sample non-overlapping: every ~4000 events for ~1s of independent returns
    stride = 4000
    sampled = labs[::stride]
    valid = sampled[np.isfinite(sampled)]
    return float(np.nansum(valid)) if len(valid) > 0 else 0.0


# ---------------------------------------------------------------------------
# Gate functions
# ---------------------------------------------------------------------------

def gate_signal_strength(preds: np.ndarray, top_pct: float) -> np.ndarray:
    """Select top N% strongest short signals (most negative predictions).
    For shorts: most negative pred = strongest short signal.
    """
    short_mask = preds < 0
    n_short = short_mask.sum()
    if n_short < 10:
        return np.zeros_like(preds, dtype=bool)

    short_vals = preds[short_mask]
    # top_pct of shorts = most negative values
    thresh = np.percentile(short_vals, top_pct * 100)
    return short_mask & (preds <= thresh)


def gate_meta_confidence(preds: np.ndarray, mask: np.ndarray, top_pct: float) -> np.ndarray:
    """Keep only signals where meta model confidence (|pred|) is in top N%.
    Higher |pred| = higher conviction from the meta model.
    """
    if top_pct >= 1.0:
        return mask  # passthrough

    indices = np.where(mask)[0]
    if len(indices) < 5:
        return mask

    confidence = np.abs(preds[indices])
    thresh = np.percentile(confidence, (1.0 - top_pct) * 100)
    keep = confidence >= thresh

    result = np.zeros_like(mask)
    result[indices[keep]] = True
    return result


def gate_ofi_confluence(ofi_vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep only signals where OFI agrees with short direction.
    For shorts: negative OFI = selling pressure = agrees.
    """
    indices = np.where(mask)[0]
    if len(indices) < 5:
        return mask

    ofi_at = ofi_vals[indices]
    agrees = ofi_at < 0

    result = np.zeros_like(mask)
    result[indices[agrees]] = True
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pnl_gross: np.ndarray, dates: np.ndarray,
                    daily_returns: dict) -> dict:
    """Compute comprehensive metrics for a set of trades."""
    n = len(pnl_gross)
    if n == 0:
        return {"n_trades": 0, "skipped": True}

    net = pnl_gross - COMMISSION_TICKS
    total_net = float(np.sum(net))
    mean_net = float(np.mean(net))
    wr = float(np.mean(net > 0))

    wins = net[net > 0].sum()
    losses = -net[net < 0].sum()
    pf = float(wins / max(losses, 1e-9))

    # Per-day aggregation
    unique_dates = np.unique(dates)
    day_pnls = []
    day_regimes = []
    for ud in unique_dates:
        dm = dates == ud
        day_pnls.append(float(np.sum(net[dm])))
        dr = daily_returns.get(ud, 0.0)
        day_regimes.append("green" if dr > 0 else ("red" if dr < 0 else "flat"))

    day_pnls = np.array(day_pnls)
    n_days = len(day_pnls)
    green_day_pct = float(np.mean(day_pnls > 0)) if n_days > 0 else 0.0
    trades_per_day = n / n_days if n_days > 0 else 0.0

    # Sharpe (annualized from daily)
    if n_days > 1 and np.std(day_pnls, ddof=1) > 0:
        sharpe = float(np.mean(day_pnls) / np.std(day_pnls, ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Sortino
    if n_days > 1:
        downside = day_pnls[day_pnls < 0]
        if len(downside) > 0 and np.std(downside, ddof=1) > 0:
            sortino = float(np.mean(day_pnls) / np.std(downside, ddof=1) * np.sqrt(252))
        else:
            sortino = float("inf") if np.mean(day_pnls) > 0 else 0.0
    else:
        sortino = 0.0

    # Regime stratification (HC #428 R1)
    day_regimes = np.array(day_regimes)
    green_idx = day_regimes == "green"
    red_idx = day_regimes == "red"

    def regime_sharpe(idx):
        if idx.sum() < 2:
            return 0.0
        rp = day_pnls[idx]
        if np.std(rp, ddof=1) > 0:
            return float(np.mean(rp) / np.std(rp, ddof=1) * np.sqrt(252))
        return 0.0

    sharpe_green = regime_sharpe(green_idx)
    sharpe_red = regime_sharpe(red_idx)

    # Regime skew (HC #428): reject if > 0.50
    max_s = max(abs(sharpe_green), abs(sharpe_red))
    regime_skew = abs(sharpe_green - sharpe_red) / max_s if max_s > 0 else 0.0

    return {
        "n_trades": n,
        "n_days": n_days,
        "trades_per_day": round(trades_per_day, 1),
        "mean_pnl_gross": round(float(np.mean(pnl_gross)), 4),
        "mean_pnl_net": round(mean_net, 4),
        "total_pnl_net": round(total_net, 2),
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


def compute_per_day_detail(pnl_gross: np.ndarray, dates: np.ndarray,
                           daily_returns: dict) -> pd.DataFrame:
    net = pnl_gross - COMMISSION_TICKS
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
            "total_pnl_net": round(float(day_net.sum()), 2),
            "mean_pnl_net": round(float(day_net.mean()), 4),
            "wr": round(float((day_net > 0).mean()), 4),
            "regime": regime,
            "daily_return_ticks": round(dr, 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OFI Feature Analysis (bonus: test all OFI columns, pick best)
# ---------------------------------------------------------------------------

def analyze_ofi_features(preds: np.ndarray, labels: np.ndarray, dates: np.ndarray,
                         all_ofi: dict, daily_returns: dict):
    """Quick scan: which OFI feature gives best confluence lift for top-10% shorts."""
    log("\n" + "=" * 80)
    log("OFI FEATURE COMPARISON (top-10% shorts, each OFI feature independently)")
    log("=" * 80)

    valid = np.isfinite(preds) & np.isfinite(labels)
    mask_base = gate_signal_strength(preds, 0.10) & valid
    pnl_base = -labels[mask_base]  # short PnL = -label

    base_mean = float(np.mean(pnl_base - COMMISSION_TICKS))
    log(f"  Baseline (no OFI): mean_net={base_mean:+.4f}t, n={mask_base.sum()}")

    results = {}
    for feat_name, ofi_vals in all_ofi.items():
        mask_ofi = gate_ofi_confluence(ofi_vals, mask_base)
        n_ofi = mask_ofi.sum()
        if n_ofi < 50:
            log(f"  {feat_name}: SKIP (n={n_ofi})")
            continue
        pnl_ofi = -labels[mask_ofi]
        mean_net = float(np.mean(pnl_ofi - COMMISSION_TICKS))
        lift = mean_net - base_mean
        log(f"  {feat_name}: mean_net={mean_net:+.4f}t, n={n_ofi}, lift={lift:+.4f}t")
        results[feat_name] = {"mean_net": mean_net, "n": n_ofi, "lift": lift}

    # Pick best OFI feature
    if results:
        best = max(results, key=lambda k: results[k]["lift"])
        log(f"\n  BEST OFI FEATURE: {best} (lift={results[best]['lift']:+.4f}t)")
        return best
    return PRIMARY_OFI


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    # Clear log
    with open(LOG_PATH, "w") as f:
        f.write("")

    log("=" * 80)
    log("STACKED CONFLUENCE VALIDATION v7")
    log("v7 prod predictions (Spearman 0.271) + OFI confluence")
    log("Question: Does OFI stacking still add lift on top of v7's stronger signal?")
    log(f"Cost: {COMMISSION_TICKS} ticks RT (passive, commission only)")
    log("Regime gate: HC #428 R1 (40+ day, regime-agnostic)")
    log("=" * 80)

    # ---- MLflow setup ----
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("stacked_confluence_v7")
        run = mlflow.start_run(run_name=f"confluence_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        log(f"MLflow run started: {run.info.run_id}")

    # ---- Load v7 predictions ----
    log("\nLoading v7 production predictions...")
    preds, labels, dates = load_v7_predictions()
    unique_dates = sorted(set(dates))
    log(f"  Total: {len(preds):,} predictions, {len(unique_dates)} OOT dates")
    log(f"  Date range: {unique_dates[0]} to {unique_dates[-1]}")
    log(f"  Pred range: [{preds.min():.4f}, {preds.max():.4f}]")

    # Quick Spearman check
    valid_mask = np.isfinite(preds) & np.isfinite(labels)
    r, p = stats.spearmanr(preds[valid_mask], labels[valid_mask])
    log(f"  Spearman: {r:.4f} (p={p:.2e})")

    if MLFLOW_AVAILABLE:
        mlflow.log_param("n_predictions", len(preds))
        mlflow.log_param("n_oot_dates", len(unique_dates))
        mlflow.log_param("spearman_v7", round(r, 4))
        mlflow.log_param("commission_ticks", COMMISSION_TICKS)

    # ---- Load MBO events & extract OFI per date ----
    log("\nLoading MBO events and extracting OFI features per date...")

    all_ofi = {name: np.full(len(preds), np.nan, dtype=np.float32) for name in OFI_COLUMNS}
    daily_returns = {}
    dates_loaded = 0
    dates_missing = 0

    for dt in unique_dates:
        dt_mask = dates == dt
        n_dt = dt_mask.sum()

        mbo = load_mbo_day(dt)
        if mbo is None:
            log(f"  {dt}: MBO NOT FOUND, {n_dt} preds orphaned")
            dates_missing += 1
            continue

        # Compute event indices for this date's predictions
        event_idx = compute_event_indices(n_dt)

        # Extract OFI features
        ofi_day = extract_ofi_at_predictions(mbo, event_idx)
        for name in OFI_COLUMNS:
            all_ofi[name][dt_mask] = ofi_day[name]

        # Regime classification
        daily_returns[dt] = classify_regime(mbo)

        regime = "green" if daily_returns[dt] > 0 else ("red" if daily_returns[dt] < 0 else "flat")
        log(f"  {dt}: {n_dt:,} preds, {mbo['n_events']:,} events, "
            f"daily_return={daily_returns[dt]:+.1f}t ({regime})")
        dates_loaded += 1

    log(f"\nLoaded {dates_loaded}/{len(unique_dates)} dates "
        f"({dates_missing} missing MBO)")

    # HC #428 R1: Must have >= 40 OOT days. We have ~27 from v7 prod.
    # Report actual count and proceed (can't manufacture more data).
    n_regime_days = len(daily_returns)
    n_green = sum(1 for v in daily_returns.values() if v > 0)
    n_red = sum(1 for v in daily_returns.values() if v < 0)
    n_flat = sum(1 for v in daily_returns.values() if v == 0)
    log(f"Regime breakdown: {n_green} green, {n_red} red, {n_flat} flat "
        f"(total {n_regime_days} days)")
    if n_regime_days < 40:
        log(f"WARNING: HC #428 R1 requires 40+ OOT days, we have {n_regime_days}. "
            f"Proceeding with what's available but flagging.")

    # Short PnL: for shorts, profit = -label (price drops = label negative = short profits)
    short_pnl = -labels

    # ---- OFI feature comparison ----
    best_ofi = analyze_ofi_features(preds, labels, dates, all_ofi, daily_returns)
    ofi_for_gate = all_ofi[best_ofi]
    log(f"\nUsing OFI feature '{best_ofi}' for main sweep")

    if MLFLOW_AVAILABLE:
        mlflow.log_param("best_ofi_feature", best_ofi)

    # ---- Main sweep ----
    log("\n" + "=" * 80)
    log("STACKED GATE SWEEP")
    log(f"Signal thresholds: {SIGNAL_THRESHOLDS}")
    log(f"Meta thresholds: {META_THRESHOLDS}")
    log(f"OFI: on/off (using '{best_ofi}')")
    log("=" * 80)

    results = {}
    per_day_dfs = {}
    valid = np.isfinite(preds) & np.isfinite(labels)

    for sig_pct in SIGNAL_THRESHOLDS:
        for meta_pct in META_THRESHOLDS:
            for ofi_on in OFI_OPTIONS:
                # Build config name
                sig_label = f"sig{int(sig_pct*100)}pct"
                meta_label = f"meta{int(meta_pct*100)}pct" if meta_pct < 1.0 else "noMeta"
                ofi_label = "ofi" if ofi_on else "noOfi"
                config_name = f"{sig_label}_{meta_label}_{ofi_label}"

                # Gate A: Signal strength
                mask = gate_signal_strength(preds, sig_pct) & valid
                n_a = mask.sum()

                # Gate B: Meta confidence
                mask = gate_meta_confidence(preds, mask, meta_pct)
                n_b = mask.sum()

                # Gate C: OFI confluence
                if ofi_on:
                    mask = gate_ofi_confluence(ofi_for_gate, mask)
                n_c = mask.sum()

                if n_c < 20:
                    results[config_name] = {"n_trades": int(n_c), "skipped": True}
                    continue

                # Compute metrics
                pnl = short_pnl[mask]
                trade_dates = dates[mask]
                metrics = compute_metrics(pnl, trade_dates, daily_returns)
                metrics["config"] = {
                    "signal_pct": sig_pct,
                    "meta_pct": meta_pct,
                    "ofi_on": ofi_on,
                }
                results[config_name] = metrics

                # Per-day detail
                per_day = compute_per_day_detail(pnl, trade_dates, daily_returns)
                per_day_dfs[config_name] = per_day

                # Compact log
                rp = "PASS" if metrics.get("regime_pass", False) else "FAIL"
                log(f"  {config_name:30s}  n={metrics['n_trades']:6d}  "
                    f"mean={metrics['mean_pnl_net']:+.4f}t  "
                    f"WR={metrics['wr']:.1%}  PF={metrics['pf']:.2f}  "
                    f"Sh={metrics['sharpe']:5.1f}  "
                    f"So={metrics['sortino']:5.1f}  "
                    f"Sh_g={metrics['sharpe_green']:5.1f}  "
                    f"Sh_r={metrics['sharpe_red']:5.1f}  "
                    f"skew={metrics['regime_skew']:.3f}  {rp}")

                # Log to MLflow
                if MLFLOW_AVAILABLE:
                    prefix = config_name
                    mlflow.log_metric(f"{prefix}_mean_net", metrics["mean_pnl_net"])
                    mlflow.log_metric(f"{prefix}_sharpe", metrics["sharpe"])
                    mlflow.log_metric(f"{prefix}_pf", metrics["pf"])
                    mlflow.log_metric(f"{prefix}_wr", metrics["wr"])
                    mlflow.log_metric(f"{prefix}_n_trades", metrics["n_trades"])
                    mlflow.log_metric(f"{prefix}_regime_skew", metrics["regime_skew"])

    # ---- Summary comparison table ----
    log("\n" + "=" * 80)
    log("RESULTS COMPARISON TABLE")
    log("=" * 80)

    compare_rows = []
    for name, m in results.items():
        if m.get("skipped"):
            continue
        compare_rows.append({
            "config": name,
            "n_trades": m["n_trades"],
            "days": m["n_days"],
            "tpd": m["trades_per_day"],
            "gross_tk": m["mean_pnl_gross"],
            "net_tk": m["mean_pnl_net"],
            "total_tk": m["total_pnl_net"],
            "wr": m["wr"],
            "pf": m["pf"],
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "green%": m["green_day_pct"],
            "Sh_green": m["sharpe_green"],
            "Sh_red": m["sharpe_red"],
            "skew": m["regime_skew"],
            "regime": "PASS" if m["regime_pass"] else "FAIL",
        })

    df_compare = pd.DataFrame(compare_rows)
    if len(df_compare) > 0:
        # Sort by Sharpe descending
        df_compare = df_compare.sort_values("sharpe", ascending=False)
        log("\n" + df_compare.to_string(index=False))

    # ---- Lift analysis ----
    log("\n" + "=" * 80)
    log("LIFT ANALYSIS: OFI Stacking vs No-OFI Baseline")
    log("=" * 80)

    for sig_pct in SIGNAL_THRESHOLDS:
        sig_label = f"sig{int(sig_pct*100)}pct"
        log(f"\n  Signal top-{sig_pct:.0%} shorts:")

        for meta_pct in META_THRESHOLDS:
            meta_label = f"meta{int(meta_pct*100)}pct" if meta_pct < 1.0 else "noMeta"

            base_key = f"{sig_label}_{meta_label}_noOfi"
            ofi_key = f"{sig_label}_{meta_label}_ofi"

            base = results.get(base_key, {})
            ofi = results.get(ofi_key, {})

            if base.get("skipped") or not base.get("n_trades"):
                continue

            base_str = (f"mean={base['mean_pnl_net']:+.4f}t "
                       f"Sh={base['sharpe']:.1f} PF={base['pf']:.2f} "
                       f"n={base['n_trades']}")

            if not ofi.get("skipped") and ofi.get("n_trades"):
                lift = ofi["mean_pnl_net"] - base["mean_pnl_net"]
                sharpe_lift = ofi["sharpe"] - base["sharpe"]
                ofi_str = (f"mean={ofi['mean_pnl_net']:+.4f}t "
                          f"Sh={ofi['sharpe']:.1f} PF={ofi['pf']:.2f} "
                          f"n={ofi['n_trades']}")
                log(f"    {meta_label:10s} base: {base_str}")
                log(f"    {meta_label:10s} +OFI: {ofi_str}  "
                    f"lift={lift:+.4f}t  Sharpe_lift={sharpe_lift:+.1f}")
            else:
                log(f"    {meta_label:10s} base: {base_str}")
                log(f"    {meta_label:10s} +OFI: SKIPPED (too few trades)")

    # ---- Best configurations ----
    log("\n" + "=" * 80)
    log("TOP 5 CONFIGURATIONS (by Sharpe, regime-pass only)")
    log("=" * 80)

    passing = [r for r in compare_rows if r["regime"] == "PASS"]
    passing.sort(key=lambda x: x["sharpe"], reverse=True)

    for i, cfg in enumerate(passing[:5]):
        log(f"\n  #{i+1}: {cfg['config']}")
        log(f"      mean_net={cfg['net_tk']:+.4f}t  WR={cfg['wr']:.1%}  PF={cfg['pf']:.2f}")
        log(f"      Sharpe={cfg['sharpe']:.1f}  Sortino={cfg['sortino']:.1f}")
        log(f"      n={cfg['n_trades']}  tpd={cfg['tpd']}  days={cfg['days']}")
        log(f"      Sh_green={cfg['Sh_green']:.1f}  Sh_red={cfg['Sh_red']:.1f}  "
            f"skew={cfg['skew']:.3f}")

    if not passing:
        log("  NO configurations passed regime gate!")
        # Show top 5 regardless
        for i, cfg in enumerate(compare_rows[:5]):
            log(f"\n  #{i+1} (REGIME FAIL): {cfg['config']}")
            log(f"      mean_net={cfg['net_tk']:+.4f}t  Sharpe={cfg['sharpe']:.1f}  "
                f"skew={cfg['skew']:.3f}")

    # ---- Save outputs ----
    log("\nSaving results...")

    # JSON results
    def np_convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    clean_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean_results[k] = {kk: np_convert(vv) for kk, vv in v.items()}
        else:
            clean_results[k] = v

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(clean_results, f, indent=2, default=np_convert)

    # Comparison CSV
    if len(df_compare) > 0:
        df_compare.to_csv(OUT_DIR / "comparison.csv", index=False)

    # Per-day CSVs
    for name, pdf in per_day_dfs.items():
        pdf.to_csv(OUT_DIR / f"perday_{name}.csv", index=False)

    # MLflow artifacts
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(str(OUT_DIR / "results.json"))
            mlflow.log_artifact(str(OUT_DIR / "comparison.csv"))
            mlflow.log_artifact(str(LOG_PATH))

            # Log best config
            if passing:
                best = passing[0]
                mlflow.log_metric("best_sharpe", best["sharpe"])
                mlflow.log_metric("best_mean_net", best["net_tk"])
                mlflow.log_metric("best_pf", best["pf"])
                mlflow.log_param("best_config", best["config"])

            mlflow.end_run()
            log("MLflow run completed and artifacts logged.")
        except Exception as e:
            log(f"MLflow artifact logging failed: {e}")
            try:
                mlflow.end_run()
            except:
                pass

    elapsed = time.time() - t0
    log(f"\nDone in {elapsed:.1f}s. Results saved to {OUT_DIR}")
    log("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        log(traceback.format_exc())
        if MLFLOW_AVAILABLE:
            try:
                mlflow.end_run(status="FAILED")
            except:
                pass
        sys.exit(1)
