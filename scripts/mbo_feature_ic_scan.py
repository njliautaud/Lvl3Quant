#!/usr/bin/env python3
"""
MBO Feature IC Scan — Phase 1 Alpha Test
Computes univariate Spearman rank IC for each feature against forward returns
at multiple horizons. No look-ahead bias, no overnight leakage.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

DATA_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/mbo_feature_ic_scan")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "1h": 60, "2h": 120}

# Base features to test directly
BASE_FEATURES = [
    "volume", "vwap", "trade_count", "signed_volume",
    "spread_mean", "ofi_1min", "microprice_close"
]

# Derived features: (name, computation function)
# All use only current/past data — no look-ahead

def derive_features(df):
    """Add derived features to a single-day dataframe. No look-ahead."""
    df = df.copy()
    df["microprice_dev"] = df["microprice_close"] - df["close"]
    df["ofi_momentum_5"] = df["ofi_1min"].rolling(5, min_periods=5).sum()
    df["volume_ratio_5"] = df["volume"] / df["volume"].rolling(5, min_periods=5).mean()
    df["signed_volume_ratio"] = df["signed_volume"] / df["volume"].replace(0, np.nan)
    df["spread_change"] = df["spread_mean"].diff()
    df["vwap_dev"] = df["vwap"] - df["close"]
    df["trade_intensity"] = df["trade_count"] / df["volume"].replace(0, np.nan)
    return df

DERIVED_FEATURES = [
    "microprice_dev", "ofi_momentum_5", "volume_ratio_5",
    "signed_volume_ratio", "spread_change", "vwap_dev", "trade_intensity"
]

ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES


def load_all_days():
    """Load all parquet files, derive features, return list of per-day DataFrames."""
    files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"Found {len(files)} day files")
    days = []
    for f in files:
        df = pd.read_parquet(f)
        df = df.sort_values("ts_minute").reset_index(drop=True)
        df = derive_features(df)
        days.append(df)
    return days


def compute_forward_returns(df, h):
    """Compute forward return within a single day. Drop last h bars."""
    fwd = df["close"].shift(-h)
    fwd_ret = (fwd - df["close"]) / df["close"]
    # Null out last h bars (can't compute forward return)
    fwd_ret.iloc[-h:] = np.nan
    return fwd_ret


def compute_ic_for_feature_horizon(days, feature, horizon_bars):
    """
    Compute concat Spearman IC across all days for one feature at one horizon.
    Returns (ic, t_stat, p_value, n_obs).
    """
    all_feat = []
    all_ret = []

    for df in days:
        if len(df) <= horizon_bars:
            continue
        fwd_ret = compute_forward_returns(df, horizon_bars)
        feat_vals = df[feature]

        # Mask: both feature and forward return must be non-null and finite
        mask = feat_vals.notna() & fwd_ret.notna() & np.isfinite(feat_vals) & np.isfinite(fwd_ret)
        if mask.sum() < 10:
            continue

        all_feat.append(feat_vals[mask].values)
        all_ret.append(fwd_ret[mask].values)

    if not all_feat:
        return np.nan, np.nan, np.nan, 0

    concat_feat = np.concatenate(all_feat)
    concat_ret = np.concatenate(all_ret)

    # Spearman rank correlation
    ic, p_value = stats.spearmanr(concat_feat, concat_ret)

    # t-statistic: t = IC * sqrt(N-2) / sqrt(1 - IC^2)
    n = len(concat_feat)
    if abs(ic) >= 1.0:
        t_stat = np.inf if ic > 0 else -np.inf
    else:
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2)

    return ic, t_stat, p_value, n


def main():
    days = load_all_days()
    print(f"Loaded {len(days)} days, bars per day: {[len(d) for d in days[:5]]}...")

    results = []
    for feat in ALL_FEATURES:
        for h_name, h_bars in HORIZONS.items():
            ic, t_stat, p_val, n_obs = compute_ic_for_feature_horizon(days, feat, h_bars)
            results.append({
                "feature": feat,
                "horizon": h_name,
                "horizon_bars": h_bars,
                "IC": round(ic, 6) if not np.isnan(ic) else None,
                "t_stat": round(t_stat, 4) if not np.isnan(t_stat) else None,
                "p_value": round(p_val, 6) if not np.isnan(p_val) else None,
                "n_obs": int(n_obs)
            })
            status = "***" if (abs(ic) > 0.03 and abs(t_stat) > 3.0 and h_bars >= 5) else ""
            print(f"  {feat:25s} @ {h_name:5s} -> IC={ic:+.5f}  t={t_stat:+.2f}  p={p_val:.4f}  n={n_obs}  {status}")

    # Sort by |IC| descending
    results_sorted = sorted(results, key=lambda r: abs(r["IC"]) if r["IC"] is not None else 0, reverse=True)

    # Save full results
    output_path = OUTPUT_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(results_sorted, f, indent=2)
    print(f"\nFull results saved to {output_path}")

    # Print top 20
    print("\n" + "=" * 90)
    print("TOP 20 FEATURES BY |IC|")
    print("=" * 90)
    print(f"{'Feature':25s} {'Horizon':8s} {'IC':>10s} {'t-stat':>10s} {'p-value':>12s} {'n_obs':>8s} {'Pass?':>6s}")
    print("-" * 90)
    for r in results_sorted[:20]:
        ic = r["IC"] if r["IC"] is not None else 0
        t = r["t_stat"] if r["t_stat"] is not None else 0
        p = r["p_value"] if r["p_value"] is not None else 1
        h_bars = r["horizon_bars"]
        passes = "|IC|>0.03 & t>3 & h>=5" if (abs(ic) > 0.03 and abs(t) > 3.0 and h_bars >= 5) else ""
        print(f"{r['feature']:25s} {r['horizon']:8s} {ic:+10.5f} {t:+10.3f} {p:12.6f} {r['n_obs']:8d}   {passes}")

    # Summary: features passing the success criterion
    print("\n" + "=" * 90)
    print("FEATURES PASSING SUCCESS CRITERION: |IC| > 0.03, t-stat > 3.0, horizon >= 5min")
    print("=" * 90)
    passing = [r for r in results_sorted
               if r["IC"] is not None and abs(r["IC"]) > 0.03
               and r["t_stat"] is not None and abs(r["t_stat"]) > 3.0
               and r["horizon_bars"] >= 5]
    if passing:
        print(f"{'Feature':25s} {'Horizon':8s} {'IC':>10s} {'t-stat':>10s} {'p-value':>12s}")
        print("-" * 70)
        for r in passing:
            print(f"{r['feature']:25s} {r['horizon']:8s} {r['IC']:+10.5f} {r['t_stat']:+10.3f} {r['p_value']:12.6f}")
        print(f"\n{len(passing)} feature-horizon pairs pass the criterion.")
    else:
        print("NO features pass the criterion.")

    # Also save passing results separately
    passing_path = OUTPUT_DIR / "passing_features.json"
    with open(passing_path, "w") as f:
        json.dump(passing, f, indent=2)

    return results_sorted


if __name__ == "__main__":
    main()
