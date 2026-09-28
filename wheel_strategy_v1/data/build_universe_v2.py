"""
build_universe_v2.py — Expand universe from 70 curated tickers to full S&P 500
intersected with FMP archive (≈242 tickers as of 2026-06-07).

Per HC #559 R2: macro picker universe = S&P 1500 minimum target. This is the
first step (S&P 500 ⨯ FMP coverage). A future v3 will pull mid-caps via IWM
holdings.

Outputs (v2 lives next to v1 — does not overwrite):
    data/cache/universe_v2.parquet      (ticker, name, sector, source)
    data/cache/prices_v2.parquet        (ticker, date, open, high, low, close, volume, ret, log_ret, rv_20/60/252)

Run:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python3 -m data.build_universe_v2
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1")
CACHE = ROOT / "data" / "cache"
FMP_ROOT = Path("/home/jupiter/teleclaude-main/data/fmp_archive")
PRICES_DIR = FMP_ROOT / "prices"
INDICES_DIR = FMP_ROOT / "indices"

START = pd.Timestamp("2015-01-01")
END = pd.Timestamp("2026-07-01")


def fmp_price_tickers() -> set:
    return {f.stem.replace("_daily", "") for f in PRICES_DIR.glob("*_daily.json")}


def load_sp500_meta() -> pd.DataFrame:
    with open(INDICES_DIR / "sp500.json") as f:
        sp = json.load(f)
    df = pd.DataFrame(sp).rename(columns={"symbol": "ticker"})
    return df[["ticker", "name", "sector"]]


def build_universe() -> pd.DataFrame:
    sp_meta = load_sp500_meta()
    have = fmp_price_tickers()
    # Always keep ETFs we use for sector flow / theme features
    sector_etfs = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLY", "XLP",
                   "XLV", "XLI", "XLU", "XLB", "XLRE", "XLC",
                   "TLT", "GLD", "MTUM", "QUAL", "VLUE", "USMV"]

    # HC #573 — niche-theme tickers outside S&P 500 the sub-industry rotator
    # needs (space, nuclear, quantum, AI infra, eVTOL, defense small-caps, etc.).
    # Mapped to broad sector for join keys; sub_industry_taxonomy assigns finer
    # buckets downstream.
    niche_themes = [
        # space / new-space
        ("RKLB", "Space",     "Industrials"),
        ("RDW",  "Space",     "Industrials"),
        ("ASTS", "Space",     "Communication Services"),
        ("LUNR", "Space",     "Industrials"),
        ("SPCE", "Space",     "Industrials"),
        # nuclear / uranium / SMR
        ("CCJ",  "Nuclear",   "Energy"),
        ("OKLO", "Nuclear",   "Utilities"),
        ("NNE",  "Nuclear",   "Utilities"),
        ("SMR",  "Nuclear",   "Industrials"),
        ("BWXT", "Nuclear",   "Industrials"),
        ("LEU",  "Nuclear",   "Energy"),
        ("UEC",  "Nuclear",   "Energy"),
        ("URA",  "Nuclear",   "ETF"),
        ("URNM", "Nuclear",   "ETF"),
        ("UUUU", "Nuclear",   "Energy"),
        # quantum
        ("IONQ", "Quantum",   "Information Technology"),
        ("QUBT", "Quantum",   "Information Technology"),
        ("RGTI", "Quantum",   "Information Technology"),
        # AI infra / physical AI / robotics
        ("AI",   "AI_Apps",   "Information Technology"),
        ("PATH", "AI_Apps",   "Information Technology"),
        ("SOUN", "AI_Apps",   "Information Technology"),
        ("BBAI", "AI_Apps",   "Information Technology"),
        ("TEM",  "AI_Apps",   "Health Care"),
        ("RXRX", "AI_Apps",   "Health Care"),
        # autonomy / lidar / robotics
        ("AVAV", "Robotics",  "Industrials"),
        ("OUST", "Robotics",  "Information Technology"),
        ("RCAT", "Robotics",  "Industrials"),
        ("ACHR", "Robotics",  "Industrials"),
        ("JOBY", "Robotics",  "Industrials"),
        # solar / batteries / energy storage
        ("ENPH", "CleanEnergy", "Information Technology"),
        ("PLUG", "CleanEnergy", "Industrials"),
        ("MP",   "RareEarth",  "Materials"),
        ("AMPX", "CleanEnergy", "Industrials"),
        # defense small/mid (overflows beyond S&P)
        ("KTOS", "Defense",   "Industrials"),
        ("LMT",  "Defense",   "Industrials"),
        ("NOC",  "Defense",   "Industrials"),
        ("RTX",  "Defense",   "Industrials"),
        ("GD",   "Defense",   "Industrials"),
        ("HII",  "Defense",   "Industrials"),
        ("BAH",  "Defense",   "Industrials"),
    ]

    keep = sp_meta[sp_meta["ticker"].isin(have)].copy()
    keep["source"] = "sp500_fmp"
    # Add ETFs we already have prices for
    extra = []
    for t in sector_etfs:
        if t in have and t not in set(keep["ticker"]):
            extra.append({"ticker": t, "name": t, "sector": "ETF", "source": "etf"})
    # Add niche theme names (HC #573)
    have_tk = set(keep["ticker"])
    for tk, theme, sector in niche_themes:
        if tk in have and tk not in have_tk:
            extra.append({"ticker": tk, "name": tk,
                          "sector": sector, "source": f"niche_{theme.lower()}"})
            have_tk.add(tk)
    if extra:
        keep = pd.concat([keep, pd.DataFrame(extra)], ignore_index=True)
    keep = keep.reset_index(drop=True)
    return keep


def load_fmp_daily(ticker: str) -> pd.DataFrame:
    p = PRICES_DIR / f"{ticker}_daily.json"
    if not p.exists():
        return pd.DataFrame()
    with open(p) as f:
        d = json.load(f)
    # FMP daily format: {"symbol": "...", "historical": [{date, open, high, low, close, volume}, ...]}
    if isinstance(d, dict) and "historical" in d:
        rows = d["historical"]
    elif isinstance(d, list):
        rows = d
    else:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"])
    keep_cols = [c for c in ["ticker", "date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep_cols].sort_values("date").reset_index(drop=True)
    return df


def build_prices(universe: pd.DataFrame) -> pd.DataFrame:
    out = []
    t0 = time.time()
    n = len(universe)
    for i, t in enumerate(universe["ticker"]):
        df = load_fmp_daily(t)
        if df.empty:
            continue
        df = df[(df["date"] >= START) & (df["date"] < END)].copy()
        if len(df) < 50:
            continue
        df["ret"] = df["close"].pct_change()
        df["log_ret"] = np.log1p(df["ret"].fillna(0)).replace(0, np.nan)
        df["rv_20"] = df["log_ret"].rolling(20).std() * np.sqrt(252)
        df["rv_60"] = df["log_ret"].rolling(60).std() * np.sqrt(252)
        df["rv_252"] = df["log_ret"].rolling(252).std() * np.sqrt(252)
        out.append(df)
        if (i + 1) % 50 == 0:
            print(f"  prices: {i+1}/{n} ({(time.time()-t0):.1f}s)", flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    print("=== build_universe_v2 ===", flush=True)
    u = build_universe()
    print(f"universe v2: {len(u)} tickers", flush=True)
    print(u["sector"].value_counts(), flush=True)
    u.to_parquet(CACHE / "universe_v2.parquet", index=False)
    print(f"wrote {CACHE/'universe_v2.parquet'}", flush=True)

    print("\n=== build prices ===", flush=True)
    p = build_prices(u)
    if p.empty:
        print("NO PRICES — abort", flush=True)
        sys.exit(1)
    print(f"prices_v2: {p.shape}, dates {p.date.min()} → {p.date.max()}", flush=True)
    print(f"  tickers covered: {p.ticker.nunique()}/{len(u)}", flush=True)
    p.to_parquet(CACHE / "prices_v2.parquet", index=False)
    print(f"wrote {CACHE/'prices_v2.parquet'}", flush=True)


if __name__ == "__main__":
    main()
