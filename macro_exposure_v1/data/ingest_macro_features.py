"""
ingest_macro_features.py — Derive macro feature panel from market.parquet.

Input:  data/cache/market.parquet  (from ingest_market.py)
Output: data/cache/macro_features.parquet
        date-indexed wide table of features:
          vix, vix3m, vix9d, vix_pct_20d,           (vol regime)
          vix_ts_slope = vix3m / vix,
          vix_ts_slope_short = vix9d / vix (NaN if vix9d missing),
          spy_close, spy_dma50_dist, spy_dma100_dist, spy_dma200_dist,
          spy_mom_3m, spy_mom_6m, spy_mom_12m, spy_52w_high_dist,
          spy_atr_pct,
          yield_2s10s = tnx - fvx       (proxy: TNX 10y, FVX 5y — true 2s10s needs FRED)
          yield_2s10s_alt = tnx - irx   (10y - 3m steepness, fallback)
          dxy, dxy_trend_20d,
          gold_close, copper_close, gold_copper_ratio.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"


def _pivot_close(df: pd.DataFrame) -> pd.DataFrame:
    """Return a wide date-indexed table of close prices by ticker."""
    w = df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    w = w.sort_index().ffill()
    return w


def _atr_pct(df: pd.DataFrame, ticker: str, window: int = 14) -> pd.Series:
    d = df[df["ticker"] == ticker].set_index("date").sort_index()
    if d.empty:
        return pd.Series(dtype=float)
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return (atr / d["close"]).rename("spy_atr_pct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""

    src = CACHE / f"market{sfx}.parquet"
    if not src.exists():
        print(f"[macro_feat] missing {src} — run ingest_market.py first", flush=True)
        return 1

    df = pd.read_parquet(src)
    close = _pivot_close(df)
    out = pd.DataFrame(index=close.index)

    def col(t):
        return close[t] if t in close.columns else pd.Series(np.nan, index=close.index)

    # --- VIX family
    out["vix"]    = col("^VIX")
    out["vix3m"]  = col("^VIX3M")
    out["vix9d"]  = col("^VIX9D")
    out["vix_pct_20d"] = out["vix"].rolling(20).rank(pct=True)
    out["vix_ts_slope"] = out["vix3m"] / out["vix"]
    out["vix_ts_slope_short"] = out["vix9d"] / out["vix"]

    # --- SPY trend / momentum
    spy = col("SPY")
    out["spy_close"] = spy
    out["spy_dma50_dist"]  = spy / spy.rolling(50).mean()  - 1.0
    out["spy_dma100_dist"] = spy / spy.rolling(100).mean() - 1.0
    out["spy_dma200_dist"] = spy / spy.rolling(200).mean() - 1.0
    # ~63 trading days = 3 months, 126 = 6m, 252 = 12m
    out["spy_mom_3m"]  = spy.pct_change(63)
    out["spy_mom_6m"]  = spy.pct_change(126)
    out["spy_mom_12m"] = spy.pct_change(252)
    out["spy_52w_high_dist"] = spy / spy.rolling(252).max() - 1.0

    # --- SPY ATR%
    atr = _atr_pct(df, "SPY", window=14)
    out = out.join(atr, how="left")

    # --- Yield curve (proxy: TNX = 10y *10, FVX = 5y *10, IRX = 3m *10 in yfinance)
    # Convert with /10 so the units are %; difference units are also %.
    tnx = col("^TNX") / 10.0
    fvx = col("^FVX") / 10.0
    irx = col("^IRX") / 10.0
    out["yield_10y"] = tnx
    out["yield_5y"]  = fvx
    out["yield_3m"]  = irx
    out["yield_2s10s"]      = tnx - fvx       # true-2s would be FRED DGS2; 5y is close substitute
    out["yield_2s10s_alt"]  = tnx - irx

    # --- DXY trend (20d % change)
    dxy = col("DX-Y.NYB")
    out["dxy"] = dxy
    out["dxy_trend_20d"] = dxy.pct_change(20)

    # --- Gold / Copper ratio
    gld = col("GC=F")
    if gld.isna().mean() > 0.5:
        gld = col("GLD")  # fallback
    cop = col("HG=F")
    out["gold_close"] = gld
    out["copper_close"] = cop
    out["gold_copper_ratio"] = gld / cop

    out = out.reset_index()
    out_path = CACHE / f"macro_features{sfx}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[macro_feat] wrote {len(out)} rows, {out.shape[1]-1} features -> {out_path}", flush=True)
    print(f"[macro_feat] date span {out['date'].min()} -> {out['date'].max()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
