"""
Family: cross_asset  (HC #563 R2 — cross-asset state)

WHAT: Daily close + returns for the canonical cross-asset basket:
gold (GC=F), oil (CL=F), copper (HG=F), BTC (BTC-USD as risk-on proxy),
10Y UST yield (^TNX), DXY dollar index (DX-Y.NYB), EUR/USD (EURUSD=X).

SOURCE: yfinance (free).

OUTPUT: data/feature_store/cross_asset/daily.parquet
Schema: (asset, date, close, ret_1d, ret_5d, ret_20d, ret_60d, zscore_60d)
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, "/home/jupiter/Lvl3Quant/research/ingest")
from _common import write_parquet, smoke_log  # type: ignore

FAMILY = "cross_asset"

ASSETS = {
    "GOLD":    "GC=F",
    "OIL":     "CL=F",
    "COPPER":  "HG=F",
    "BTC":     "BTC-USD",
    "UST10Y":  "^TNX",
    "DXY":     "DX-Y.NYB",
    "EURUSD":  "EURUSD=X",
}


def main():
    import yfinance as yf
    tickers = list(ASSETS.values())
    print(f"[{FAMILY}] fetching {len(tickers)} assets (5y daily)")
    df = yf.download(tickers, period="5y", interval="1d",
                     progress=False, auto_adjust=False,
                     group_by="column", threads=True)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned empty")

    close = df["Close"].stack(dropna=True).rename("close").reset_index()
    close.columns = ["date", "ticker", "close"]
    # Map ticker → friendly asset name
    inv = {v: k for k, v in ASSETS.items()}
    close["asset"] = close["ticker"].map(inv)
    close = close.dropna(subset=["asset"]).drop(columns=["ticker"])
    close["date"] = pd.to_datetime(close["date"]).dt.tz_localize(None)
    close = close.sort_values(["asset", "date"]).reset_index(drop=True)

    g = close.groupby("asset", group_keys=False)
    close["ret_1d"]  = g["close"].pct_change(1)
    close["ret_5d"]  = g["close"].pct_change(5)
    close["ret_20d"] = g["close"].pct_change(20)
    close["ret_60d"] = g["close"].pct_change(60)
    # 60d z-score of close
    rolling = g["close"].transform(lambda s: (s - s.rolling(60, min_periods=20).mean())
                                    / s.rolling(60, min_periods=20).std())
    close["zscore_60d"] = rolling

    out = close[["asset","date","close","ret_1d","ret_5d","ret_20d","ret_60d","zscore_60d"]]
    p = write_parquet(out, FAMILY, "daily.parquet")
    smoke_log(FAMILY, True, f"{len(out)} rows, {out['asset'].nunique()} assets -> {p}")
    print(f"OK {FAMILY}: {len(out)} rows, {out['asset'].nunique()} assets -> {p}")
    print(out.tail(3).to_string())
    return p


def run_full():
    return main()


if __name__ == "__main__":
    main()
