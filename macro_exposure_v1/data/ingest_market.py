"""
ingest_market.py — Daily OHLCV for the macro-exposure universe.

Tickers (small-set, ~15, safe even while wheel pipeline is running):
  ETFs:        SPY, QQQ, IWM, GLD
  VIX family:  ^VIX, ^VIX3M, ^VIX9D
  Yields:      ^TNX (10y), ^FVX (5y), ^IRX (3m), ^TYX (30y)
  FX/Comms:    DX-Y.NYB (DXY), GC=F (gold front), HG=F (copper front)

Window: 2010-01-01 -> today.

Output: data/cache/market.parquet  (long form: date, ticker, open, high, low, close, volume)
"""
from __future__ import annotations
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "SPY", "QQQ", "IWM", "GLD",
    "^VIX", "^VIX3M", "^VIX9D",
    "^TNX", "^FVX", "^IRX", "^TYX",
    "DX-Y.NYB", "GC=F", "HG=F",
]


def _yf_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False,
                             auto_adjust=True, threads=False)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.columns = [str(c).lower() for c in df.columns]
            keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[keep].copy()
            df["date"] = pd.to_datetime(df["date"])
            df["ticker"] = ticker
            return df
        except Exception as e:
            print(f"[market] {ticker} attempt {attempt+1} err: {e}", flush=True)
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.end is None:
        args.end = pd.Timestamp.today().strftime("%Y-%m-%d")
    if args.smoke:
        args.start, args.end = "2023-01-01", "2024-12-31"

    pieces = []
    ok, fail = 0, []
    for i, t in enumerate(TICKERS, 1):
        df = _yf_one(t, args.start, args.end)
        if df.empty:
            fail.append(t)
        else:
            pieces.append(df)
            ok += 1
        if i % 5 == 0:
            print(f"[market] {i}/{len(TICKERS)} ok={ok}", flush=True)
        time.sleep(0.4)  # be polite

    if not pieces:
        print("[market] NO DATA — aborting", flush=True)
        return 1

    out = pd.concat(pieces, ignore_index=True).sort_values(["ticker", "date"])
    out = out[["date", "ticker", "open", "high", "low", "close", "volume"]]
    sfx = "_smoke" if args.smoke else ""
    path = CACHE / f"market{sfx}.parquet"
    out.to_parquet(path, index=False)
    print(f"[market] wrote {len(out)} rows ({out['ticker'].nunique()} tickers) -> {path}", flush=True)
    if fail:
        print(f"[market] failed: {fail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
