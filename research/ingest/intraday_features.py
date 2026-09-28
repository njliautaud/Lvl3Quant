"""
Family: intraday_features  (HC #561 R1 — intraday-style features)

WHAT: Per-(ticker, date) intraday-style features. HC #561 R1 mandates this
family; mandates: "if 1-minute or 5-minute bars exist anywhere in the repo,
use them; else compute proxies from OHLCV daily and flag as such".

This implementation uses DAILY OHLCV proxies (no 1m/5m bars currently on disk
per data/INVENTORY.md). Every column is flagged accordingly in the schema
comment so downstream consumers know the provenance.

SOURCE: wheel_strategy_v1/data/cache/prices_v2.parquet
  (ticker, date, open, high, low, close, volume, ret, log_ret, rv_20, rv_60, rv_252)

OUTPUT: data/feature_store/intraday/daily_proxies.parquet
Schema:
  ticker, date,
  overnight_gap          = open / prev_close - 1                    (real)
  intraday_range_pct     = (high - low) / open                      (real)
  open_to_close_ret      = close / open - 1                         (real)
  close_minus_typprice   = (close - (high+low+close)/3) / close     (proxy, no VWAP)
  upper_shadow_pct       = (high - max(open,close)) / open          (proxy for late-day buying pressure)
  lower_shadow_pct       = (min(open,close) - low)   / open         (proxy for late-day selling pressure)
  max_intraday_dd_pct    = (low - prev_close) / prev_close          (proxy for worst-of-day vs prior close)
  rv_5m_intraday         = NaN  (UNAVAILABLE — needs intraday bars; column kept for schema continuity)
  dollar_volume_first30m = NaN  (UNAVAILABLE — needs intraday bars)
  dollar_volume_last30m  = NaN  (UNAVAILABLE — needs intraday bars)
  dollar_volume          = close * volume                           (real)

PIT-SAFE: every feature is available at end-of-day D for day D.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, "/home/jupiter/Lvl3Quant/research/ingest")
from _common import write_parquet, smoke_log  # type: ignore

FAMILY = "intraday"
SRC = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/prices_v2.parquet")


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"source prices missing: {SRC}")
    print(f"[{FAMILY}] reading {SRC}")
    df = pd.read_parquet(SRC)
    print(f"[{FAMILY}] input shape={df.shape}")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df.groupby("ticker", group_keys=False)
    df["prev_close"] = g["close"].shift(1)

    # Real columns
    df["overnight_gap"]        = df["open"] / df["prev_close"] - 1
    df["intraday_range_pct"]   = (df["high"] - df["low"]) / df["open"]
    df["open_to_close_ret"]    = df["close"] / df["open"] - 1
    typprice                   = (df["high"] + df["low"] + df["close"]) / 3.0
    df["close_minus_typprice"] = (df["close"] - typprice) / df["close"]
    df["upper_shadow_pct"]     = (df["high"] - df[["open","close"]].max(axis=1)) / df["open"]
    df["lower_shadow_pct"]     = (df[["open","close"]].min(axis=1) - df["low"]) / df["open"]
    df["max_intraday_dd_pct"]  = (df["low"] - df["prev_close"]) / df["prev_close"]
    df["dollar_volume"]        = df["close"] * df["volume"]

    # Placeholder columns kept for schema continuity (NaN until intraday bars exist)
    df["rv_5m_intraday"]         = np.nan
    df["dollar_volume_first30m"] = np.nan
    df["dollar_volume_last30m"]  = np.nan

    cols = ["ticker","date",
            "overnight_gap","intraday_range_pct","open_to_close_ret",
            "close_minus_typprice","upper_shadow_pct","lower_shadow_pct",
            "max_intraday_dd_pct","dollar_volume",
            "rv_5m_intraday","dollar_volume_first30m","dollar_volume_last30m"]
    out = df[cols].copy()

    # Drop rows where we have no prev_close (first day per ticker)
    out = out.dropna(subset=["overnight_gap"])

    p = write_parquet(out, FAMILY, "daily_proxies.parquet")
    smoke_log(FAMILY, True, f"{len(out)} rows, {out['ticker'].nunique()} tickers -> {p}")
    print(f"OK {FAMILY}: {len(out)} rows, {out['ticker'].nunique()} tickers -> {p}")

    # Sanity sample
    samp = out.tail(3).to_string()
    print(samp)
    return p


def run_full():
    return main()


if __name__ == "__main__":
    main()
