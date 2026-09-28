"""
Momentum Strategy — Data Module
Fetches historical prices for a liquid US equity universe via yfinance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import json
import datetime as dt

CACHE_DIR = Path(__file__).resolve().parent / "_cache"
CACHE_DIR.mkdir(exist_ok=True)


# Large-cap core (S&P 500 top ~110)
LARGE_CAP = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADP", "ADSK", "AIG", "AMAT",
    "AMD", "AMGN", "AMZN", "AVGO", "AXP", "BA", "BAC", "BDX", "BLK", "BMY",
    "BRK-B", "C", "CAT", "CCI", "CDNS", "CI", "CL", "CMCSA", "CME", "COF",
    "COP", "COST", "CRM", "CSCO", "CVX", "D", "DE", "DHR", "DIS", "DUK",
    "ECL", "EL", "EMR", "EOG", "EXC", "F", "FDX", "GD", "GE", "GILD",
    "GM", "GOOG", "GS", "HD", "HON", "IBM", "ICE", "INTC", "INTU", "ISRG",
    "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ",
    "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX",
    "NKE", "NOC", "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL",
    "QCOM", "RTX", "SBUX", "SCHW", "SHW", "SO", "SPG", "T", "TGT", "TMO",
    "TMUS", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT",
    "XOM", "ZTS",
]

# Mid-cap growth/momentum names (S&P 400 + NASDAQ mid-tier)
# These are where the biggest gains often come from — $5-50B market cap,
# less institutional coverage, more room to run
MID_CAP = [
    # Tech / Software
    "BILL", "CFLT", "CRWD", "CYBR", "DDOG", "DOCN", "DUOL", "ESTC", "FIVE",
    "FRSH", "GLOB", "GTLB", "HUBS", "IOT", "MDB", "MNDY", "NET", "OKTA",
    "PCTY", "PLTR", "QLYS", "SHOP", "SNAP", "SQSP", "TOST", "TTD", "TWLO",
    "U", "ZI", "ZS",
    # Semis / Hardware
    "ACLS", "AMKR", "COHR", "DIOD", "ENTG", "FORM", "LSCC", "MRVL",
    "ONTO", "RMBS", "SMTC", "WOLF",
    # Fintech / Financial
    "AFRM", "COIN", "FOUR", "HOOD", "LC", "LPLA", "NUVEI", "SOFI", "SQ",
    # Healthcare / Biotech
    "ALGN", "DXCM", "EXAS", "INSP", "IONS", "NBIX", "NTRA", "RARE",
    "RXRX", "SGEN", "TEM", "VEEV",
    # Consumer / E-commerce
    "BIRK", "CELH", "CAVA", "DKS", "DUOL", "ELF", "ETSY", "GRAB",
    "LULU", "ONON", "PINS", "RBLX", "RIVN", "SE", "WDAY",
    # Industrial / Energy / Defense
    "ASTS", "AXON", "BWX", "GNRC", "KRATOS", "LTCH", "RKLB", "TDG",
    "TRTX", "VST",
    # REITs / Specialty
    "AMT", "ARE", "DLR", "EQIX", "IRM", "PSA",
]

# Combined universe — ~220 liquid stocks across large + mid cap
FULL_UNIVERSE = sorted(set(LARGE_CAP + MID_CAP))

# For backward compat
LIQUID_100 = LARGE_CAP


def fetch_prices(
    tickers: list[str] | None = None,
    start: str = "2010-01-01",
    end: str | None = None,
    use_cache: bool = True,
    cache_max_age_hours: int = 24,
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.

    Returns:
        DataFrame with DatetimeIndex (trading days) and ticker columns.
        Missing data forward-filled then back-filled (for IPO dates).
    """
    if tickers is None:
        tickers = FULL_UNIVERSE
    if end is None:
        end = dt.date.today().isoformat()

    cache_file = CACHE_DIR / f"prices_{start}_{end}_{len(tickers)}.parquet"
    meta_file = cache_file.with_suffix(".json")

    # Check cache freshness
    if use_cache and cache_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        cached_at = dt.datetime.fromisoformat(meta["cached_at"])
        if (dt.datetime.now() - cached_at).total_seconds() < cache_max_age_hours * 3600:
            if set(meta["tickers"]) == set(tickers):
                print(f"[data] Loading cached prices from {cache_file.name}")
                return pd.read_parquet(cache_file)

    print(f"[data] Downloading {len(tickers)} tickers from {start} to {end} ...")
    # yfinance bulk download
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker case
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Clean: drop tickers with >30% missing data
    missing_pct = prices.isna().mean()
    bad = missing_pct[missing_pct > 0.30].index.tolist()
    if bad:
        print(f"[data] Dropping {len(bad)} tickers with >30% missing data: {bad[:10]}...")
        prices = prices.drop(columns=bad)

    prices = prices.ffill().bfill()

    # Cache
    prices.to_parquet(cache_file)
    meta_file.write_text(json.dumps({
        "cached_at": dt.datetime.now().isoformat(),
        "tickers": prices.columns.tolist(),
        "start": start,
        "end": end,
        "shape": list(prices.shape),
    }))
    print(f"[data] Cached {prices.shape[0]} days x {prices.shape[1]} tickers")

    return prices


def fetch_spy(start: str = "2010-01-01", end: str | None = None) -> pd.Series:
    """Fetch SPY adjusted close for regime classification and trend filter."""
    if end is None:
        end = dt.date.today().isoformat()

    cache_file = CACHE_DIR / f"spy_{start}_{end}.parquet"
    meta_file = cache_file.with_suffix(".json")

    if cache_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        cached_at = dt.datetime.fromisoformat(meta["cached_at"])
        if (dt.datetime.now() - cached_at).total_seconds() < 24 * 3600:
            return pd.read_parquet(cache_file).squeeze()

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    result = spy["Close"].squeeze()
    result.to_frame().to_parquet(cache_file)
    meta_file.write_text(json.dumps({"cached_at": dt.datetime.now().isoformat()}))
    return result


def fetch_bond(
    ticker: str = "SHY",
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.Series:
    """Fetch bond ETF prices (e.g. SHY, BIL) for cash alternative in trend filter."""
    if end is None:
        end = dt.date.today().isoformat()

    cache_file = CACHE_DIR / f"bond_{ticker}_{start}_{end}.parquet"
    meta_file = cache_file.with_suffix(".json")

    if cache_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        cached_at = dt.datetime.fromisoformat(meta["cached_at"])
        if (dt.datetime.now() - cached_at).total_seconds() < 24 * 3600:
            return pd.read_parquet(cache_file).squeeze()

    bond = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    result = bond["Close"].squeeze()
    result.to_frame().to_parquet(cache_file)
    meta_file.write_text(json.dumps({"cached_at": dt.datetime.now().isoformat()}))
    return result


if __name__ == "__main__":
    prices = fetch_prices(start="2015-01-01")
    print(f"Shape: {prices.shape}")
    print(prices.tail())
