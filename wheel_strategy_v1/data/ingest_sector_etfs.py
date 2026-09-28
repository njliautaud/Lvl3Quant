"""
ingest_sector_etfs.py — HC #558 R3 prerequisite.

Pulls daily OHLCV for sector SPDR ETFs + thematic ETFs from yfinance,
caches to data/cache/sector_etfs.parquet. Used by macro_picker for sector
rotation + flow-proxy features.

Tickers:
  SPDR sectors: XLK, XLF, XLV, XLE, XLY, XLP, XLI, XLU, XLB, XLRE, XLC
  Broad index:  SPY, QQQ, IWM, DIA
  Thematic:     SMH (semis), SOXX (semis 2), XBI (biotech), ARKK (innovation/SaaS),
                IGV (software/SaaS), KWEB (china tech), ITA (defense),
                XME (metals/mining), KBE (banks), JETS (airlines)
  Bond/gold:    TLT, GLD, USO, UUP, HYG

Window: 2015-01-01 to today.
"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import yfinance as yf

ROOT = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1")
OUT = ROOT / "data" / "cache" / "sector_etfs.parquet"

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"]
INDEX_ETFS = ["SPY", "QQQ", "IWM", "DIA"]
THEMATIC = ["SMH", "SOXX", "XBI", "ARKK", "IGV", "KWEB", "ITA", "XME", "KBE", "JETS"]
MACRO_ETFS = ["TLT", "GLD", "USO", "UUP", "HYG"]

ALL_TICKERS = SECTOR_ETFS + INDEX_ETFS + THEMATIC + MACRO_ETFS

START = "2015-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")


def main():
    print(f"[etf] downloading {len(ALL_TICKERS)} tickers {START}..{END}")
    df = yf.download(
        ALL_TICKERS,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    rows = []
    for tk in ALL_TICKERS:
        try:
            sub = df[tk].copy()
        except KeyError:
            print(f"[etf] {tk}: not in download — skipping")
            continue
        sub = sub.dropna(how="all")
        if len(sub) == 0:
            print(f"[etf] {tk}: empty — skipping")
            continue
        sub = sub.reset_index().rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        sub["ticker"] = tk
        # category tag
        if tk in SECTOR_ETFS:
            sub["category"] = "sector"
        elif tk in INDEX_ETFS:
            sub["category"] = "index"
        elif tk in THEMATIC:
            sub["category"] = "thematic"
        elif tk in MACRO_ETFS:
            sub["category"] = "macro"
        rows.append(sub[["date", "ticker", "category", "open", "high", "low", "close", "volume"]])

    if not rows:
        print("[etf] no data downloaded", file=sys.stderr)
        sys.exit(1)

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Derived: dollar volume (cheap fund-flow proxy)
    out["dollar_volume"] = out["close"] * out["volume"]
    # Returns
    out["ret_1d"] = out.groupby("ticker")["close"].pct_change()
    out["log_ret_1d"] = (out["close"] / out.groupby("ticker")["close"].shift(1)).apply(
        lambda x: pd.NA if pd.isna(x) else __import__("math").log(x)
    )

    print(f"[etf] writing {len(out):,} rows × {len(out.columns)} cols → {OUT}")
    print(f"[etf] coverage: {out['date'].min().date()}..{out['date'].max().date()}")
    print(f"[etf] per-ticker rows: {out.groupby('ticker').size().to_dict()}")
    out.to_parquet(OUT)
    print("[etf] done")


if __name__ == "__main__":
    main()
