#!/usr/bin/env python3
"""
Shared Market Data Downloader — runs ONCE before paper engines.
Downloads all market data to a shared cache so paper engines don't
each hit yfinance independently (which causes rate limiting).

Gap Fix: 8 paper engines hitting yfinance simultaneously at 4:30 PM
caused ALL of them to fail with TypeError("'NoneType' object is not subscriptable").

This script:
1. Downloads all needed tickers in ONE call
2. Saves to /home/jupiter/Lvl3Quant/data/shared_market_cache.pkl
3. Paper engines import from cache instead of downloading

Run at 4:00 PM (before any paper engine) and 9:15 AM (before morning engines).
"""

import os
import sys
import json
import pickle
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CACHE_PATH = BASE / "data" / "shared_market_cache.pkl"
LOG_PATH = BASE / "logs" / "shared_market_downloader.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DataDownloader] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# All tickers needed by any paper engine
ALL_TICKERS = [
    # Sector ETFs
    'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLB', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE',
    # Macro
    'SPY', 'QQQ', 'IWM', 'TLT', 'SHY', 'GLD', 'HYG', 'DBC', 'DBA',
    # VIX
    '^VIX', '^VIX3M',
    # Commodity/Intl ETFs used by some engines
    'EFA', 'EEM', 'UUP',
    # Quality mega-caps used by stock-level engines
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK-B', 'JPM', 'JNJ',
    'UNH', 'PG', 'HD', 'MA', 'ABBV', 'KO', 'PEP', 'COST', 'LIN', 'CRM',
    'AVGO', 'TMO', 'MRK', 'ACN', 'INTC', 'CSCO', 'CAT', 'TXN',
    # Additional sector rotation stocks
    'TSM', 'ASML', 'NVO', 'SHOP',
    # Sub-sector rotation tracker ETFs
    'SMH', 'IGV', 'CIBR', 'IBB', 'IHI', 'KRE', 'KIE', 'IPAY',
    'OIH', 'AMLP', 'ITA', 'IYT', 'CARZ', 'PEJ', 'XME',
    # Sub-sector rotation representative tickers (not already covered)
    'AMD', 'QCOM', 'MU', 'AMAT', 'LRCX', 'KLAC', 'TER',
    'ORCL', 'NOW', 'ADBE', 'PANW', 'CRWD', 'FTNT', 'ZS', 'OKTA',
    'DELL', 'HPQ', 'ANET', 'REGN', 'GILD', 'VRTX', 'MRNA', 'BIIB',
    'ISRG', 'ABT', 'MDT', 'SYK', 'EW', 'LLY', 'PFE',
    'HCA', 'CNC', 'ELV', 'CI', 'BAC', 'WFC', 'C', 'GS',
    'USB', 'PNC', 'TFC', 'FITB', 'KEY', 'PGR', 'AIG', 'MET', 'ALL',
    'V', 'PYPL', 'AFRM', 'FIS',
    'XOM', 'CVX', 'COP', 'EOG', 'OXY', 'SLB', 'HAL', 'BKR',
    'WMB', 'KMI', 'OKE',
    'LMT', 'RTX', 'NOC', 'GD', 'LHX', 'DE', 'CMI', 'PCAR',
    'EMR', 'ROK', 'ETN', 'IR', 'AME', 'UNP', 'UPS', 'FDX', 'CSX', 'DAL',
    'LOW', 'TJX', 'TSLA', 'GM', 'F', 'RIVN', 'ON',
    'MCD', 'SBUX', 'CMG', 'DRI', 'YUM',
    'MDLZ', 'GIS', 'WMT', 'TGT', 'DG', 'KR',
    'FCX', 'NEM', 'SCCO', 'CLF', 'APD', 'ECL', 'SHW', 'DD',
    'PLD', 'AMT', 'EQIX', 'DLR', 'SPG',
    'NEE', 'DUK', 'SO', 'AEP', 'SRE',
    'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS',
]


def download_with_retry(tickers, start='2024-01-01', max_retries=3):
    """Download with exponential backoff retry."""
    import yfinance as yf

    for attempt in range(max_retries):
        try:
            data = yf.download(tickers, start=start, progress=False, threads=False)
            if data is None or data.empty:
                raise ValueError("Empty download result")

            # Verify we got actual data
            if isinstance(data.columns, pd.MultiIndex):
                close = data['Close']
            else:
                close = data

            n_tickers = close.shape[1] if len(close.shape) > 1 else 1
            n_rows = len(close)

            if n_rows < 5:
                raise ValueError(f"Only {n_rows} rows — likely rate limited")

            log.info(f"Downloaded {n_tickers} tickers, {n_rows} rows (attempt {attempt+1})")
            return data

        except Exception as e:
            wait = 2 ** (attempt + 1)
            log.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    log.error(f"All {max_retries} download attempts failed")
    return None


def main():
    log.info("=" * 60)
    log.info("Shared Market Data Downloader — starting")

    # Download in smaller batches to avoid rate limits
    all_data = {}

    # Batch 1: ETFs and indices
    batch1 = [t for t in ALL_TICKERS if t.startswith('^') or t.startswith('X') or
              t in ('SPY', 'QQQ', 'IWM', 'TLT', 'SHY', 'GLD', 'HYG', 'DBC', 'DBA', 'EFA', 'EEM', 'UUP')]

    # Batch 2: Individual stocks
    batch2 = [t for t in ALL_TICKERS if t not in batch1]

    results = {}
    for i, batch in enumerate([batch1, batch2], 1):
        if not batch:
            continue
        log.info(f"Batch {i}: downloading {len(batch)} tickers...")
        data = download_with_retry(batch)
        if data is not None:
            results[f'batch_{i}'] = data
            log.info(f"Batch {i} success")
        else:
            log.error(f"Batch {i} FAILED — {len(batch)} tickers lost")

        if i < 2:  # Wait between batches
            time.sleep(2)

    if not results:
        log.error("ALL downloads failed — cache NOT updated")
        return 1

    # Merge batches
    if len(results) > 1:
        merged = pd.concat(results.values(), axis=1)
        # Remove duplicate columns
        merged = merged.loc[:, ~merged.columns.duplicated()]
    else:
        merged = list(results.values())[0]

    # Extract OHLCV
    cache = {}
    if isinstance(merged.columns, pd.MultiIndex):
        for field in ['Close', 'High', 'Low', 'Open', 'Volume']:
            try:
                df = merged[field]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)
                cache[field] = df.ffill()
            except KeyError:
                log.warning(f"No {field} data in download")
    else:
        cache['Close'] = merged.ffill()

    cache['timestamp'] = datetime.now().isoformat()
    cache['tickers'] = list(cache.get('Close', pd.DataFrame()).columns)

    # Save cache
    os.makedirs(CACHE_PATH.parent, exist_ok=True)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)

    n_tickers = len(cache.get('tickers', []))
    n_rows = len(cache.get('Close', []))
    log.info(f"Cache saved: {n_tickers} tickers, {n_rows} rows → {CACHE_PATH}")
    log.info("=" * 60)
    return 0


def load_cache():
    """Helper function for paper engines to load cached data.

    Usage in paper engines:
        from scripts.shared_market_data_downloader import load_cache
        cache = load_cache()
        close = cache['Close']  # DataFrame with all tickers
        high = cache['High']
        low = cache['Low']
    """
    if not CACHE_PATH.exists():
        return None

    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)

    # Check freshness — cache should be from today
    ts = cache.get('timestamp', '')
    if ts:
        cache_date = datetime.fromisoformat(ts).date()
        today = datetime.now().date()
        if (today - cache_date).days > 1:
            log.warning(f"Cache is stale: {cache_date} vs today {today}")

    return cache


if __name__ == "__main__":
    sys.exit(main())
