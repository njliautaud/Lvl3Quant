#!/usr/bin/env python3
"""
Flow Data Enricher — Pulls fresh strike-level options chain data daily via yfinance.

For each ticker in the quality universe, fetches:
- All available expiration dates (filters to <=60 days out)
- Strike, volume, OI, IV (impliedVolatility), bid, ask, last price
- For both calls and puts

Stores as parquet in /home/jupiter/Lvl3Quant/data/flow_enriched/{TICKER}_{DATE}.parquet
with a rolling 30-day retention to keep disk usage manageable.

Run daily (e.g., 6pm ET after market close) to capture end-of-day OI snapshots.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# Paths
BASE = Path("/home/jupiter/Lvl3Quant")
UNIVERSE_JSON = BASE / "data" / "quality_universe.json"
ENRICHED_DIR = BASE / "data" / "flow_enriched"
LOG_DIR = BASE / "logs" / "flow_screener"

ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "enricher.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("flow_enricher")

# Config
MAX_DTE = 60          # Only fetch expirations within 60 days
RETENTION_DAYS = 30   # Keep 30 days of historical enriched data
RATE_LIMIT_DELAY = 0.5  # seconds between tickers to avoid throttling


def load_universe() -> list:
    with open(UNIVERSE_JSON) as f:
        return json.load(f)["tickers"]


def fetch_ticker_chains(ticker: str, today: datetime) -> pd.DataFrame:
    """Fetch options chain data for a single ticker via yfinance."""
    try:
        tk = yf.Ticker(ticker)

        # Get spot price
        info = tk.fast_info
        spot_price = getattr(info, "last_price", None)
        if spot_price is None:
            hist = tk.history(period="1d")
            if not hist.empty:
                spot_price = hist["Close"].iloc[-1]

        # Get available expiration dates
        try:
            expirations = tk.options
        except Exception:
            log.warning("%s: no options data available", ticker)
            return pd.DataFrame()

        if not expirations:
            return pd.DataFrame()

        all_rows = []
        cutoff = today + timedelta(days=MAX_DTE)

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            if exp_date > cutoff:
                continue

            try:
                chain = tk.option_chain(exp_str)
            except Exception as e:
                log.debug("%s exp %s failed: %s", ticker, exp_str, e)
                continue

            dte = (exp_date - today).days

            for opt_type, df_chain in [("call", chain.calls), ("put", chain.puts)]:
                if df_chain.empty:
                    continue
                df_chain = df_chain.copy()
                df_chain["ticker"] = ticker
                df_chain["expiration"] = exp_str
                df_chain["dte"] = dte
                df_chain["option_type"] = opt_type
                df_chain["spot_price"] = spot_price
                df_chain["snapshot_date"] = today.strftime("%Y-%m-%d")

                # Standardize column names
                rename_map = {
                    "contractSymbol": "contract",
                    "lastTradeDate": "last_trade_date",
                    "lastPrice": "last_price",
                    "impliedVolatility": "implied_vol",
                    "openInterest": "openInterest",
                    "volume": "volume",
                    "strike": "strike",
                    "bid": "bid",
                    "ask": "ask",
                    "inTheMoney": "itm",
                }
                df_chain = df_chain.rename(columns=rename_map)

                # Select columns we care about
                keep_cols = [
                    "ticker", "snapshot_date", "expiration", "dte", "option_type",
                    "strike", "volume", "openInterest", "implied_vol",
                    "bid", "ask", "last_price", "spot_price", "itm", "contract",
                ]
                existing = [c for c in keep_cols if c in df_chain.columns]
                all_rows.append(df_chain[existing])

        if not all_rows:
            return pd.DataFrame()

        result = pd.concat(all_rows, ignore_index=True)

        # Calculate moneyness
        if "spot_price" in result.columns and "strike" in result.columns:
            result["moneyness"] = result["strike"] / result["spot_price"]

        log.info(
            "%s: fetched %d option contracts across %d expirations (spot=$%.2f)",
            ticker, len(result),
            result["expiration"].nunique() if "expiration" in result.columns else 0,
            spot_price or 0,
        )
        return result

    except Exception as e:
        log.error("%s: fetch failed — %s", ticker, e)
        return pd.DataFrame()


def cleanup_old_data(retention_days: int = RETENTION_DAYS):
    """Remove enriched data files older than retention period."""
    cutoff = datetime.now() - timedelta(days=retention_days)
    removed = 0
    for f in ENRICHED_DIR.glob("*.parquet"):
        try:
            # Filename format: TICKER_YYYY-MM-DD.parquet
            date_str = f.stem.split("_")[-1]
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date < cutoff:
                f.unlink()
                removed += 1
        except (ValueError, IndexError):
            pass
    if removed:
        log.info("Cleaned up %d old enriched data files", removed)


def run():
    log.info("=== Flow Data Enricher starting ===")
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")

    universe = load_universe()
    log.info("Universe: %d tickers", len(universe))

    # Check if already ran today
    today_files = list(ENRICHED_DIR.glob(f"*_{today_str}.parquet"))
    if len(today_files) >= len(universe) * 0.8:
        log.info("Already have %d/%d tickers for today. Skipping.", len(today_files), len(universe))
        return

    success = 0
    failed = 0
    total_contracts = 0

    for i, ticker in enumerate(universe):
        # Skip if already fetched today
        outfile = ENRICHED_DIR / f"{ticker}_{today_str}.parquet"
        if outfile.exists():
            log.debug("%s: already fetched today, skipping", ticker)
            success += 1
            continue

        df = fetch_ticker_chains(ticker, today)

        if df.empty:
            failed += 1
            log.warning("%s: no data returned", ticker)
        else:
            df.to_parquet(outfile, index=False)
            success += 1
            total_contracts += len(df)

        # Rate limit
        if i < len(universe) - 1:
            time.sleep(RATE_LIMIT_DELAY)

        # Progress every 10 tickers
        if (i + 1) % 10 == 0:
            log.info("Progress: %d/%d tickers processed", i + 1, len(universe))

    # Cleanup old data
    cleanup_old_data()

    log.info(
        "=== Enricher complete: %d success, %d failed, %d total contracts ===",
        success, failed, total_contracts,
    )


if __name__ == "__main__":
    run()
