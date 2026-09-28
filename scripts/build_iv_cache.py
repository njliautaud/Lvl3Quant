#!/usr/bin/env python3
"""
Build IV Cache for Wheel Backtest Engine
==========================================

The wheel engine expects an IV DataFrame with columns:
  date, ticker, sigma (annualized vol), iv_rank (percentile 0-1)

Currently no iv_cache.parquet exists, so the engine falls back to
rv_20 from prices.parquet with iv_rank=0.50 (flat). This produces
inaccurate pricing and makes parameter sweeps unreliable.

This script builds a proper IV cache by:
1. Computing rolling realized vol at multiple windows (20d, 60d)
2. Computing IV rank: percentile of current 20d vol vs 252d history
3. Adding VIX-adjusted IV proxy (rv_20 * vix_adjustment)
4. Saving as iv_cache.parquet

Author: Claude (2026-07-10)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"


def build_iv_cache():
    print("Loading prices...", flush=True)
    prices = pd.read_parquet(CACHE / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    print(f"  {len(prices)} rows, {prices['ticker'].nunique()} tickers", flush=True)

    # Load macro for VIX
    print("Loading macro...", flush=True)
    macro = pd.read_parquet(CACHE / "macro.parquet")
    macro["date"] = pd.to_datetime(macro["date"])
    vix_map = macro.set_index("date")["vix"].to_dict()

    # Compute IV for each ticker
    print("Computing IV cache...", flush=True)
    results = []

    for i, (ticker, grp) in enumerate(prices.groupby("ticker")):
        grp = grp.sort_values("date").copy()

        # rv_20 is already in the data, but let's ensure consistency
        if "rv_20" in grp.columns:
            sigma = grp["rv_20"].copy()
        else:
            log_ret = np.log(grp["close"] / grp["close"].shift(1))
            sigma = log_ret.rolling(20).std() * np.sqrt(252)

        # Fill NaN with 60d vol, then 252d vol, then 0.30
        if "rv_60" in grp.columns:
            sigma = sigma.fillna(grp["rv_60"])
        if "rv_252" in grp.columns:
            sigma = sigma.fillna(grp["rv_252"])
        sigma = sigma.fillna(0.30)

        # Clip to reasonable range
        sigma = sigma.clip(0.05, 3.0)

        # IV rank: percentile of current 20d vol vs trailing 252d history
        # Higher rank = vol is elevated vs recent history
        iv_rank = sigma.rolling(252, min_periods=40).apply(
            lambda x: (x.iloc[:-1] <= x.iloc[-1]).mean() if len(x) > 1 else 0.5,
            raw=False
        )
        iv_rank = iv_rank.fillna(0.50)

        # VIX adjustment: if VIX is elevated, IV tends to be higher than rv_20
        # Simple proxy: sigma_adjusted = rv_20 * max(1.0, vix/20)
        # This captures the fact that implied vol > realized vol when VIX is high
        vix_values = grp["date"].map(vix_map).fillna(20.0)
        vix_adj = np.maximum(1.0, vix_values / 20.0)
        sigma_adjusted = sigma * vix_adj

        ticker_df = pd.DataFrame({
            "date": grp["date"].values,
            "ticker": ticker,
            "sigma": sigma_adjusted.values,
            "sigma_raw": sigma.values,
            "iv_rank": iv_rank.values,
        })
        results.append(ticker_df)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{prices['ticker'].nunique()} tickers processed", flush=True)

    iv_cache = pd.concat(results, ignore_index=True)

    # Quality checks
    print(f"\nIV Cache Stats:", flush=True)
    print(f"  Rows: {len(iv_cache)}", flush=True)
    print(f"  Tickers: {iv_cache['ticker'].nunique()}", flush=True)
    print(f"  Date range: {iv_cache['date'].min()} to {iv_cache['date'].max()}", flush=True)
    print(f"  Sigma: mean={iv_cache['sigma'].mean():.3f}, "
          f"median={iv_cache['sigma'].median():.3f}, "
          f"min={iv_cache['sigma'].min():.3f}, "
          f"max={iv_cache['sigma'].max():.3f}", flush=True)
    print(f"  IV Rank: mean={iv_cache['iv_rank'].mean():.3f}, "
          f"median={iv_cache['iv_rank'].median():.3f}", flush=True)
    print(f"  NaN sigma: {iv_cache['sigma'].isna().sum()}", flush=True)
    print(f"  NaN iv_rank: {iv_cache['iv_rank'].isna().sum()}", flush=True)

    # Save
    output_path = CACHE / "iv_cache.parquet"
    iv_cache.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    # Verify it works with the engine
    print("\nVerifying engine compatibility...", flush=True)
    test_iv = pd.read_parquet(output_path)
    assert "date" in test_iv.columns
    assert "ticker" in test_iv.columns
    assert "sigma" in test_iv.columns
    assert "iv_rank" in test_iv.columns
    print("  ✓ All required columns present", flush=True)
    print("  ✓ IV cache ready for wheel engine", flush=True)


if __name__ == "__main__":
    build_iv_cache()
