#!/usr/bin/env python3
"""
Daily Options Chain Data Logger
================================
Captures full strike-level options chain data (prices, IV, Greeks, volume, OI)
for our key tickers. Designed to build a real options dataset for backtesting
instead of relying on Black-Scholes approximations.

Runs daily at 4:15 PM ET via PM2 cron.
Saves parquet files organized by date: data/options_chains/YYYY-MM-DD/{TICKER}.parquet

Columns saved per contract:
  ticker, expiry, strike, option_type, bid, ask, last, mid, iv,
  delta, gamma, theta, vega, volume, open_interest,
  underlying_price, in_the_money, dte_days,
  snapshot_timestamp

Greeks are computed from IV using Black-Scholes (yfinance provides IV but not Greeks).

Usage:
    python options_data_logger.py              # collect today
    python options_data_logger.py 2026-07-14   # collect specific date
    python options_data_logger.py --force      # re-collect even if data exists
"""

import sys
import os
import time
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("/home/jupiter/Lvl3Quant/data/options_chains")
LOG_FILE = DATA_DIR / "collection.log"
MAX_DTE = 90  # capture expirations up to 90 days out
SLEEP_BETWEEN = 0.4  # seconds between tickers (rate limiting)
RISK_FREE_RATE = 0.043  # ~4.3% fed funds rate, update periodically

# Tickers we actively trade options on + key indices
TICKERS = [
    # Core wheel strategy tickers
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "AMD", "TSLA",
    # ETFs we trade
    "SPY", "QQQ", "IWM",
    # Sector ETFs (primary strategy universe for bull call spreads)
    "XLE", "XLK", "XLF", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC", "XLY",
    # Additional wheel candidates
    "AVGO", "CRM", "NFLX", "ADBE", "INTC", "QCOM", "MU", "AMAT",
    # Financials
    "JPM", "BAC", "GS",
    # Healthcare
    "UNH", "LLY", "ABBV",
    # Consumer
    "WMT", "COST", "HD",
    # Energy
    "XOM", "CVX",
    # Volatility products
    "GLD", "SLV", "TLT",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("options_chain_logger")


# ---------------------------------------------------------------------------
# Black-Scholes Greeks from IV
# ---------------------------------------------------------------------------

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Compute Black-Scholes Greeks from implied volatility.

    Args:
        S: underlying price
        K: strike price
        T: time to expiry in years (must be > 0)
        r: risk-free rate
        sigma: implied volatility
        option_type: 'call' or 'put'

    Returns:
        dict with delta, gamma, theta, vega (or None values if inputs invalid)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": None, "gamma": None, "theta": None, "vega": None}

    try:
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * sqrt_T)

        # Vega (same for calls and puts, per 1% move in vol)
        vega = S * norm.pdf(d1) * sqrt_T / 100.0

        if option_type == "call":
            delta = norm.cdf(d1)
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * sqrt_T)
                - r * K * math.exp(-r * T) * norm.cdf(d2)
            ) / 365.0  # per calendar day
        else:
            delta = norm.cdf(d1) - 1.0
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * sqrt_T)
                + r * K * math.exp(-r * T) * norm.cdf(-d2)
            ) / 365.0

        return {
            "delta": round(delta, 6),
            "gamma": round(gamma, 6),
            "theta": round(theta, 6),
            "vega": round(vega, 6),
        }
    except (ValueError, ZeroDivisionError, OverflowError):
        return {"delta": None, "gamma": None, "theta": None, "vega": None}


# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------

def collect_ticker_chain(ticker: str, today: datetime) -> pd.DataFrame | None:
    """
    Collect full options chain for a single ticker.
    Returns DataFrame with one row per contract, or None on failure.
    """
    try:
        tk = yf.Ticker(ticker)

        # Get current price
        hist = tk.history(period="1d")
        if hist.empty:
            log.warning(f"{ticker}: no price data, skipping")
            return None
        underlying_price = float(hist["Close"].iloc[-1])

        # Get expirations within MAX_DTE days
        try:
            expirations = tk.options
        except Exception:
            log.warning(f"{ticker}: no options chain available")
            return None

        if not expirations:
            log.warning(f"{ticker}: no expirations found")
            return None

        cutoff = today + timedelta(days=MAX_DTE)
        valid_exps = [
            exp for exp in expirations
            if datetime.strptime(exp, "%Y-%m-%d") <= cutoff
        ]

        if not valid_exps:
            log.warning(f"{ticker}: no expirations within {MAX_DTE} days")
            return None

        all_rows = []
        snapshot_ts = datetime.utcnow().isoformat(timespec="seconds")

        for exp_str in valid_exps:
            try:
                chain = tk.option_chain(exp_str)
            except Exception as e:
                log.debug(f"{ticker} {exp_str}: chain error: {e}")
                continue

            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            dte_days = (exp_date - today).days
            T_years = max(dte_days / 365.0, 1 / 365.0)  # floor at 1 day

            for opt_type, df_chain in [("call", chain.calls), ("put", chain.puts)]:
                if df_chain.empty:
                    continue

                for _, row in df_chain.iterrows():
                    strike = float(row["strike"])
                    bid = float(row.get("bid", 0) or 0)
                    ask = float(row.get("ask", 0) or 0)
                    last = float(row.get("lastPrice", 0) or 0)
                    mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else last
                    iv = float(row.get("impliedVolatility", 0) or 0)
                    raw_vol = row.get("volume", 0)
                    volume = int(raw_vol) if pd.notna(raw_vol) else 0
                    raw_oi = row.get("openInterest", 0)
                    oi = int(raw_oi) if pd.notna(raw_oi) else 0
                    itm = bool(row.get("inTheMoney", False))

                    # Compute Greeks from IV
                    greeks = bs_greeks(
                        S=underlying_price,
                        K=strike,
                        T=T_years,
                        r=RISK_FREE_RATE,
                        sigma=iv,
                        option_type=opt_type,
                    )

                    all_rows.append({
                        "ticker": ticker,
                        "expiry": exp_str,
                        "strike": strike,
                        "option_type": opt_type,
                        "bid": round(bid, 4),
                        "ask": round(ask, 4),
                        "last": round(last, 4),
                        "mid": round(mid, 4),
                        "iv": round(iv, 6),
                        "delta": greeks["delta"],
                        "gamma": greeks["gamma"],
                        "theta": greeks["theta"],
                        "vega": greeks["vega"],
                        "volume": volume,
                        "open_interest": oi,
                        "underlying_price": round(underlying_price, 4),
                        "in_the_money": itm,
                        "dte_days": dte_days,
                        "snapshot_timestamp": snapshot_ts,
                    })

            # Small delay between expirations to be nice to yfinance
            time.sleep(0.1)

        if not all_rows:
            log.warning(f"{ticker}: no contract data collected")
            return None

        return pd.DataFrame(all_rows)

    except Exception as e:
        log.error(f"{ticker}: unexpected error: {e}")
        return None


def main():
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--force"]

    if args:
        date_str = args[0]
        today = datetime.strptime(date_str, "%Y-%m-%d")
    else:
        today = datetime.now()

    date_label = today.strftime("%Y-%m-%d")
    day_dir = DATA_DIR / date_label

    # Weekend check
    if today.weekday() >= 5:
        log.info(f"{date_label} is a weekend, skipping collection")
        return

    # Idempotency: skip if directory exists with data (unless --force)
    if day_dir.exists() and not force:
        existing = list(day_dir.glob("*.parquet"))
        if len(existing) >= len(TICKERS) * 0.5:
            log.info(
                f"Already collected {len(existing)} tickers for {date_label}, "
                f"skipping (use --force to re-collect)"
            )
            return

    day_dir.mkdir(parents=True, exist_ok=True)

    # Also set up file logging
    fh = logging.FileHandler(LOG_FILE, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    log.info(f"Starting options chain collection for {date_label} — {len(TICKERS)} tickers")
    t0 = time.time()

    results = {}
    errors = []
    total_contracts = 0

    for i, ticker in enumerate(TICKERS):
        df = collect_ticker_chain(ticker, today)
        if df is not None and len(df) > 0:
            # Save per-ticker parquet
            ticker_path = day_dir / f"{ticker}.parquet"
            df.to_parquet(ticker_path, index=False)
            results[ticker] = len(df)
            total_contracts += len(df)
            log.info(f"  {ticker}: {len(df)} contracts saved")
        else:
            errors.append(ticker)

        # Progress log every 10 tickers
        if (i + 1) % 10 == 0:
            log.info(
                f"  Progress: {i+1}/{len(TICKERS)} tickers "
                f"({len(results)} ok, {len(errors)} failed, {total_contracts} contracts)"
            )

        if i < len(TICKERS) - 1:
            time.sleep(SLEEP_BETWEEN)

    elapsed = time.time() - t0

    # Also save a combined file for the day (easier for bulk analysis)
    if results:
        all_dfs = []
        for ticker in results:
            ticker_path = day_dir / f"{ticker}.parquet"
            all_dfs.append(pd.read_parquet(ticker_path))
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = day_dir / "_all_tickers.parquet"
        combined.to_parquet(combined_path, index=False)

    summary = (
        f"Options chain collection {date_label}: "
        f"{len(results)}/{len(TICKERS)} tickers, "
        f"{total_contracts} total contracts, "
        f"{len(errors)} failures"
        + (f" (failed: {', '.join(errors)})" if errors else "")
        + f", elapsed={elapsed:.0f}s"
    )
    log.info(summary)
    print(summary)


if __name__ == "__main__":
    main()
