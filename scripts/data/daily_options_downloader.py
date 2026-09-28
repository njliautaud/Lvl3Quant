#!/usr/bin/env python3
"""
daily_options_downloader.py — Daily options chain data for backtest-quality pricing.

## The Problem This Solves

Our Dolt chain parquets were materialized once and are now stale (stop at June 2026).
The underlying Dolt database IS updated daily by the author, and has been tracking
daily snapshots since Oct 2024. We need to:

  1. Pull latest Dolt commits (already on origin, just need to merge)
  2. Re-materialize the per-ticker chain parquets for the date gap
  3. Provide supplemental coverage from Alpaca bars for the 2024 period where
     Dolt had only Mon/Wed/Fri (Tue/Thu gaps ~28% of days)

## Data Source Assessment (as of July 2026)

### SOURCE 1: DoltHub post-no-preference/options (PRIMARY — USE THIS)
- Cost: FREE. Public dataset, no API key needed.
- History: 2019-present
- Frequency:
    * 2019: WEEKLY (Saturday snapshots)
    * 2020 – Sep 2024: Mon/Wed/Fri only (~60-72% of trading days)
    * Oct 2024 – present: FULL DAILY (every weekday)
- Coverage: ~70+ large-cap tickers
- Data: full chain (bid/ask/greeks/IV for all strikes/expirations), daily EOD snapshot
- Format: Dolt SQL database, materialized to per-ticker Parquet
- Update: author pushes daily ~2:30 AM ET. Pull via `dolt fetch + merge`.
- STATUS: Already cloned at wheel_strategy_v1/data/cache/options_real/options/
          Chain parquets stale (last materialized June 2026). Need re-materialization.
- VERDICT: Best source for our 70-ticker universe. Free, has greeks. Use daily.

### SOURCE 2: Alpaca API (SUPPLEMENTAL for 2024 gap-filling)
- Cost: FREE (paper account, indicative feed). OPRA feed requires paid Options subscription (~$9/mo).
- History: Jan 19, 2024 – present (no data before that date)
- Frequency: DAILY bars per option contract (OHLCV). NOT snapshots with greeks.
- Coverage: Any optionable US equity
- Data: OHLCV trade bars per OCC contract symbol. NO bid/ask, NO greeks, NO IV.
- Rate limit: 200 requests/minute (paper tier)
- Gap-fill potential: Can reconstruct historical close price per strike/expiry
  by generating OCC symbols and batch-requesting bars. Works for 2024 period where
  Dolt had only Mon/Wed/Fri. BUT: without bid/ask, mid-price estimation is rough.
- VERDICT: Useful ONLY to fill Dolt's Tue/Thu gaps in Jan-Sep 2024 (close price only,
  no greeks). Low priority given Dolt covers Oct 2024+ fully.

### SOURCE 3: Yahoo Finance / yfinance
- Cost: FREE
- History: Current chain only (no historical snapshots). `option_chain()` returns
  today's quotes. Cannot retrieve past dates. USELESS for historical backtesting.
- VERDICT: NOT suitable for historical backtest data.

### SOURCE 4: Polygon.io
- Cost: Free tier: real-time delayed + limited history. Stocks Starter ($29/mo) has
  2-year options history. Options data requires "Options Add-On" ($79+/mo extra).
  Full options history: ~$200+/mo.
- History: 2004+ (paid). Free tier: recent only.
- Frequency: Daily + intraday
- Data: Snapshots with greeks, bid/ask, open interest
- VERDICT: Good quality but expensive (~$200+/mo for full access). Not needed given
  Dolt is free and covers our universe.

### SOURCE 5: ThetaData
- Cost: $0 for EOD options, $35/mo for minute-level. Historical: 2005-present.
- Frequency: EOD + intraday options
- Coverage: All US optionable equities
- Data: NBBO quotes, open interest, greeks (calculated), IV
- VERDICT: Best value for institutional-quality EOD data if Dolt fails. ~$0-35/mo.
  No action needed now since Dolt is working and free.

### SOURCE 6: CBOE DataShop
- Cost: Institutional pricing, $500-5000+/month for historical data products.
- VERDICT: Way too expensive for our scale.

### SOURCE 7: IVolatility
- Cost: Academic/retail: ~$40-200/month depending on plan.
- VERDICT: More expensive than ThetaData with similar coverage.

### SOURCE 8: OptionsDX / Dolt expanded cadence
- The Dolt dataset IS daily now (since Oct 2024). No expanded product needed.

## BOTTOM LINE

The #1 action item is simple: pull latest Dolt data and re-materialize parquets.
The Dolt author updates daily. We just haven't pulled since June 2026.

---

## Usage

    # Full refresh — pull Dolt, re-materialize all tickers, fill gaps
    python scripts/data/daily_options_downloader.py

    # Just materialize new dates (fast, skip Dolt pull if already fresh)
    python scripts/data/daily_options_downloader.py --no-pull

    # Only specific tickers
    python scripts/data/daily_options_downloader.py --tickers SPY,AAPL,TSLA

    # Check current status without doing anything
    python scripts/data/daily_options_downloader.py --status

    # Alpaca gap-fill for Jan-Sep 2024 Tue/Thu holes (close-price only, no greeks)
    python scripts/data/daily_options_downloader.py --alpaca-gapfill --tickers SPY

---

## Schedule

Run daily after 2:30 AM ET (when Dolt author pushes):
    0 3 * * 1-5  cd /home/jupiter/Lvl3Quant && python scripts/data/daily_options_downloader.py >> logs/dolt_daily.log 2>&1
"""
from __future__ import annotations

import os
import argparse
import shutil
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
DOLT_DIR = CACHE / "options_real" / "options"
OUT_DIR = CACHE / "options_real"
CHAINS_DIR = OUT_DIR / "chains"
LOG_DIR = ROOT / "logs"

UNIVERSE_PARQUET = CACHE / "universe.parquet"

# ---------------------------------------------------------------------------
# Dolt helpers
# ---------------------------------------------------------------------------

def _find_dolt() -> str:
    for p in (shutil.which("dolt"), "/home/jupiter/.local/bin/dolt", "/usr/local/bin/dolt"):
        if p and Path(p).exists():
            return p
    raise SystemExit("dolt binary not found — install from https://github.com/dolthub/dolt")


def dolt_query(sql: str, dolt_bin: Optional[str] = None, timeout: int = 300) -> pd.DataFrame:
    """Run a Dolt SQL query and return a DataFrame."""
    bin_ = dolt_bin or _find_dolt()
    proc = subprocess.run(
        [bin_, "sql", "-q", sql, "-r", "csv"],
        cwd=str(DOLT_DIR),
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"dolt sql failed:\n{proc.stderr.strip()[:1000]}")
    out = proc.stdout.strip()
    if not out:
        return pd.DataFrame()
    from io import StringIO
    return pd.read_csv(StringIO(out))


def dolt_max_date(dolt_bin: str) -> Optional[date]:
    """Return the latest date available in volatility_history."""
    try:
        df = dolt_query("SELECT max(date) as md FROM volatility_history", dolt_bin, timeout=30)
        if df.empty or df["md"].iloc[0] is None:
            return None
        return pd.to_datetime(df["md"].iloc[0]).date()
    except Exception as e:
        print(f"  [warn] Could not query Dolt max date: {e}")
        return None


def dolt_pull(dolt_bin: str) -> bool:
    """
    Pull latest data from DoltHub.
    The Dolt repo is public/read-only — no credentials needed for fetch.
    The author commits daily at ~2:30 AM ET.
    Returns True if new data was merged.
    """
    print("[dolt] Fetching from origin...")
    try:
        fetch = subprocess.run(
            [dolt_bin, "fetch"],
            cwd=str(DOLT_DIR),
            capture_output=True, text=True, timeout=120,
        )
        if fetch.returncode != 0:
            print(f"  [warn] dolt fetch error: {fetch.stderr[:300]}")
    except subprocess.TimeoutExpired:
        print("  [warn] dolt fetch timed out")
        return False

    # Configure dummy identity for merge commit (read-only pull doesn't actually commit)
    subprocess.run(
        [dolt_bin, "config", "--global", "--add", "user.name", "Jupiter"],
        cwd=str(DOLT_DIR), capture_output=True,
    )
    subprocess.run(
        [dolt_bin, "config", "--global", "--add", "user.email", "jupiter@lvl3quant.local"],
        cwd=str(DOLT_DIR), capture_output=True,
    )

    print("[dolt] Merging remotes/origin/master...")
    try:
        merge = subprocess.run(
            [dolt_bin, "merge", "remotes/origin/master"],
            cwd=str(DOLT_DIR),
            capture_output=True, text=True, timeout=120,
        )
        output = merge.stdout + merge.stderr
        if "Fast-forward" in output or "rows added" in output:
            print(f"  [dolt] Merge succeeded: {output.strip()[:200]}")
            return True
        elif "already up to date" in output.lower() or "nothing to merge" in output.lower():
            print("  [dolt] Already up to date.")
            return False
        else:
            print(f"  [dolt] Merge output: {output[:300]}")
            return False
    except subprocess.TimeoutExpired:
        print("  [warn] dolt merge timed out")
        return False


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def _dolt_symbol(ticker: str) -> str:
    """Map our ticker convention to Dolt act_symbol convention."""
    return ticker.replace("-", ".") if "-" in ticker else ticker


def load_universe(restrict: Optional[List[str]] = None) -> List[str]:
    if not UNIVERSE_PARQUET.exists():
        raise SystemExit(f"Universe parquet not found: {UNIVERSE_PARQUET}")
    u = pd.read_parquet(UNIVERSE_PARQUET)
    tks = u["ticker"].astype(str).tolist()
    if restrict:
        tks = [t for t in tks if t in set(restrict)]
    return tks


# ---------------------------------------------------------------------------
# Chain materialization — Dolt -> per-ticker Parquet
# ---------------------------------------------------------------------------

def _existing_max_date_for_ticker(ticker: str) -> Optional[date]:
    p = CHAINS_DIR / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, columns=["date"])
        if df.empty:
            return None
        return df["date"].max().date()
    except Exception:
        return None


def materialize_chain_for_ticker(ticker: str, dolt_bin: str,
                                 since: Optional[date] = None) -> Optional[pd.DataFrame]:
    """
    Pull option_chain rows for one ticker from Dolt, filtered to wheel-relevant tenor.

    Args:
        ticker: Universe ticker (e.g. 'SPY', 'BRK-B')
        dolt_bin: Path to dolt binary
        since: Only pull rows with date > since (for incremental refresh)

    Returns:
        DataFrame with columns: date, expiration, strike, type, bid, ask, mid,
                                vol, delta, gamma, theta, vega, rho, dte
        Or None on query failure.
    """
    sym = _dolt_symbol(ticker)

    # --- PRE-FLIGHT CHECK: verify ticker exists in volatility_history ---
    # This is a fast indexed query. If the ticker isn't in Dolt at all, we skip
    # the expensive full-scan on option_chain (which can take 5-10 minutes returning empty).
    preflight_sql = (
        f"SELECT COUNT(*) as cnt FROM volatility_history WHERE act_symbol = '{sym}' LIMIT 1"
    )
    try:
        pf = dolt_query(preflight_sql, dolt_bin, timeout=30)
        if pf.empty or int(pf["cnt"].iloc[0]) == 0:
            print(f"  [{ticker}] Not in Dolt (checked volatility_history) — skipping")
            return pd.DataFrame()  # Return empty df (not None = failure)
    except Exception as e:
        print(f"  [{ticker}] Pre-flight check failed: {e} — attempting query anyway")
    # --- END PRE-FLIGHT ---

    date_clause = f"AND date > '{since}'" if since else ""
    sql = (
        "SELECT date, expiration, strike, call_put, bid, ask, vol, "
        "delta, gamma, theta, vega, rho "
        "FROM option_chain "
        f"WHERE act_symbol = '{sym}' "
        "  AND DATEDIFF(expiration, date) BETWEEN 5 AND 90 "
        f"  {date_clause} "
        "ORDER BY date, expiration, strike"
    )
    try:
        df = dolt_query(sql, dolt_bin, timeout=1800)
    except Exception as e:
        print(f"  [{ticker}] query failed: {e}")
        return None

    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["dte"] = (df["expiration"] - df["date"]).dt.days
    for c in ["strike", "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df.rename(columns={"call_put": "type"})
    df["type"] = df["type"].str.lower().str[:1]  # 'c' or 'p'
    return df[["date", "expiration", "strike", "type", "bid", "ask", "mid",
               "vol", "delta", "gamma", "theta", "vega", "rho", "dte"]]


def materialize_all_chains(tickers: List[str], dolt_bin: str,
                           incremental: bool = True, force: bool = False) -> dict:
    """
    Materialize per-ticker chain parquets from Dolt.

    Args:
        tickers: List of tickers to process
        dolt_bin: Path to dolt binary
        incremental: If True, only pull dates after existing parquet max date
        force: If True, ignore existing data and re-pull everything

    Returns:
        dict with keys: ok, empty, fail, skipped, new_rows
    """
    CHAINS_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"ok": 0, "empty": 0, "fail": 0, "skipped": 0, "new_rows": 0}

    for i, ticker in enumerate(tickers, 1):
        out_path = CHAINS_DIR / f"{ticker}.parquet"

        # Determine incremental start date
        since = None
        if incremental and not force and out_path.exists():
            existing_max = _existing_max_date_for_ticker(ticker)
            if existing_max:
                since = existing_max
                # Check if Dolt has newer data
                # We'll just always re-query with since= which is efficient

        print(f"[{i}/{len(tickers)}] {ticker} (since={since or 'beginning'})...")

        new_df = materialize_chain_for_ticker(ticker, dolt_bin, since=since)

        if new_df is None:
            stats["fail"] += 1
            print(f"  -> FAIL")
            continue

        if new_df.empty:
            stats["empty"] += 1
            print(f"  -> EMPTY (no new data)")
            stats["skipped"] += 1
            continue

        # If incremental: append to existing
        if incremental and since and out_path.exists():
            try:
                existing = pd.read_parquet(out_path)
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["date", "expiration", "strike", "type"]
                ).sort_values(["date", "expiration", "strike"])
                combined.to_parquet(out_path, index=False)
                new_count = len(new_df)
                stats["new_rows"] += new_count
                print(f"  -> appended {new_count:,} rows (total {len(combined):,})")
            except Exception as e:
                print(f"  [warn] Failed to append, overwriting: {e}")
                new_df.to_parquet(out_path, index=False)
                stats["new_rows"] += len(new_df)
        else:
            new_df.to_parquet(out_path, index=False)
            stats["new_rows"] += len(new_df)
            print(f"  -> wrote {len(new_df):,} rows")

        stats["ok"] += 1

    return stats


# ---------------------------------------------------------------------------
# Alpaca gap-fill — fills Tue/Thu gaps in Jan-Sep 2024 with close prices
# ---------------------------------------------------------------------------
#
# IMPORTANT LIMITATION: Alpaca historical bars provide OHLCV (trade price) only.
# NO bid/ask, NO greeks, NO IV. This means:
#   - We can estimate mid-price as close price (rough approximation)
#   - We CANNOT fill delta/gamma/theta/vega/rho columns
#   - This is a second-class data source vs Dolt
#
# For the period Jan–Sep 2024, Dolt has Mon/Wed/Fri only.
# Forward-filling across Tue/Thu still gives correct pricing for backtests
# because when we're CLOSED (no chain observation), we use the last known price.
#
# RECOMMENDATION: Don't bother with Alpaca gap-fill.
# Rationale:
#   - In the backtest, a missing Tue/Thu just means we can't open/close on that day.
#   - The REAL problem was using 0 buyback (assuming $0 for missing close data).
#   - That bug is now FIXED because we have real close prices on 3 of 5 days.
#   - Alpaca would add close price for 2 more days but WITHOUT greeks,
#     so we still can't open positions on those days (no delta to pick strikes).
#
# If you truly need Alpaca gap-fill, it works like this:
#   1. For each ticker + expiry, generate OCC symbols for $1-increment strikes
#   2. Batch request bars (up to 200 symbols per request) from Alpaca data API
#   3. Extract close price for each date, use as mid (rough)
#   4. Merge into chain parquet with a flag column: pricing_source='alpaca_bar'
# This is implemented below but NOT called by default.

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BARS_URL = "https://data.alpaca.markets/v1beta1/options/bars"
ALPACA_MIN_DATE = date(2024, 1, 19)  # Confirmed: no data before this date


def _occ_symbol(ticker: str, expiry: date, strike: float, option_type: str) -> str:
    """Format OCC symbol: e.g. SPY240119P00470000"""
    otype = "P" if option_type.lower().startswith("p") else "C"
    strike_int = int(round(strike * 1000))
    return f"{ticker.upper():<6}{expiry.strftime('%y%m%d')}{otype}{strike_int:08d}".replace(" ", "")


def _alpaca_bars_batch(symbols: List[str], start: date, end: date) -> dict:
    """Fetch daily bars for a batch of OCC symbols. Returns {symbol: [bar, ...]}."""
    import requests
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    # Alpaca allows many symbols in one request
    params = {
        "symbols": ",".join(symbols),
        "timeframe": "1Day",
        "start": str(start),
        "end": str(end),
        "limit": 10000,
    }
    try:
        r = requests.get(ALPACA_BARS_URL, headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            return {}
        return r.json().get("bars", {})
    except Exception:
        return {}


def alpaca_gapfill_ticker(ticker: str, start: date = ALPACA_MIN_DATE,
                          end: date = date(2024, 9, 30),
                          strike_range_pct: float = 0.15) -> Optional[pd.DataFrame]:
    """
    Fill Tue/Thu gaps in Dolt chains for a ticker using Alpaca EOD bars.

    Strategy:
      - Load existing Dolt chain parquet to find what dates/expirations/strikes exist
      - Identify "gap dates" (trading days not in Dolt)
      - For each gap date, reconstruct the chain by fetching bars for all
        strikes that were traded around the same expiry (from adjacent Dolt dates)
      - Fill close price as mid proxy

    Args:
        ticker: Ticker symbol
        start: Start date for gap-fill (ALPACA_MIN_DATE minimum)
        end: End date (use Sep 2024 since Oct+ Dolt is fully daily)
        strike_range_pct: ATM ±% to enumerate strikes (default 15%)

    Returns:
        DataFrame of new rows to append, or None if nothing to add
    """
    import pandas_market_calendars as mcal  # type: ignore

    chain_path = CHAINS_DIR / f"{ticker}.parquet"
    if not chain_path.exists():
        print(f"  [{ticker}] No existing chain — run Dolt materialization first")
        return None

    existing = pd.read_parquet(chain_path)
    existing["date"] = pd.to_datetime(existing["date"])

    # Get trading days in window
    try:
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=str(start), end_date=str(end))
        trading_days = [pd.Timestamp(d).date() for d in schedule.index]
    except ImportError:
        # Fallback: generate weekdays and filter known holidays manually
        from pandas.tseries.offsets import BDay
        dr = pd.date_range(start=str(start), end=str(end), freq=BDay())
        trading_days = [d.date() for d in dr]

    # Find gap dates (trading days not in existing Dolt data for this window)
    existing_in_window = existing[
        (existing["date"] >= pd.Timestamp(start)) &
        (existing["date"] <= pd.Timestamp(end))
    ]
    existing_dates = set(existing_in_window["date"].dt.date.unique())
    gap_dates = [d for d in trading_days if d not in existing_dates]

    if not gap_dates:
        print(f"  [{ticker}] No gaps in {start} – {end}")
        return None

    print(f"  [{ticker}] Found {len(gap_dates)} gap dates in {start} – {end}")

    # For each gap date, find adjacent Dolt date to get known strikes/expirations
    new_rows = []
    import requests

    # Batch by expiry to reduce API calls
    # Get all expirations in window from existing data
    expirations = sorted(
        existing_in_window["expiration"].dt.date.unique()
    )

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }

    for exp in expirations:
        exp_rows = existing_in_window[existing_in_window["expiration"].dt.date == exp]
        if exp_rows.empty:
            continue

        # Get strike range for this expiry from existing data
        strikes_in_chain = sorted(exp_rows["strike"].unique())
        if not strikes_in_chain:
            continue

        # Build OCC symbols for all strikes
        # Try both puts and calls
        all_syms = []
        for strike in strikes_in_chain:
            for otype in ["p", "c"]:
                sym = _occ_symbol(ticker, exp, strike, otype)
                all_syms.append((sym, strike, otype))

        # Batch fetch for gap dates in this expiry's lifetime
        gap_dates_for_exp = [d for d in gap_dates
                             if d < exp and (exp - d).days <= 90]
        if not gap_dates_for_exp:
            continue

        # Fetch all bars for this expiry's symbols across the gap window
        occ_syms = [s[0] for s in all_syms]
        BATCH = 200
        bars_by_sym = {}
        for i in range(0, len(occ_syms), BATCH):
            batch = occ_syms[i:i+BATCH]
            result = _alpaca_bars_batch(
                batch,
                start=gap_dates_for_exp[0],
                end=gap_dates_for_exp[-1] + timedelta(days=1),
            )
            bars_by_sym.update(result)
            time.sleep(0.3)  # Rate limit

        # Build rows for gap dates
        for gap_date in gap_dates_for_exp:
            for sym, strike, otype in all_syms:
                bars = bars_by_sym.get(sym, [])
                # Find bar for this date
                date_str = f"{gap_date.strftime('%Y-%m-%d')}T"
                bar = next((b for b in bars if b.get("t", "").startswith(date_str)), None)
                if bar is None:
                    continue

                close_price = bar.get("c", 0.0)
                if close_price <= 0:
                    continue

                dte = (exp - gap_date).days
                if dte < 5 or dte > 90:
                    continue

                new_rows.append({
                    "date": pd.Timestamp(gap_date),
                    "expiration": pd.Timestamp(exp),
                    "strike": strike,
                    "type": otype[0],
                    "bid": close_price * 0.98,   # rough estimate — NOT real bid/ask
                    "ask": close_price * 1.02,
                    "mid": close_price,           # Alpaca close as mid proxy
                    "vol": float("nan"),          # no IV in Alpaca bars
                    "delta": float("nan"),        # no greeks in Alpaca bars
                    "gamma": float("nan"),
                    "theta": float("nan"),
                    "vega": float("nan"),
                    "rho": float("nan"),
                    "dte": dte,
                    "pricing_source": "alpaca_bar_close",  # tag for quality control
                })

    if not new_rows:
        print(f"  [{ticker}] No Alpaca data found for gap dates")
        return None

    df = pd.DataFrame(new_rows)
    print(f"  [{ticker}] Alpaca gap-fill: {len(df):,} new rows")
    return df


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------

def print_status(tickers: List[str]):
    """Print coverage summary for each ticker."""
    print("\n" + "=" * 70)
    print("OPTIONS CHAIN DATA STATUS")
    print("=" * 70)

    # Dolt status
    dolt_bin = _find_dolt()
    dolt_max = dolt_max_date(dolt_bin)
    print(f"\nDolt database: {DOLT_DIR}")
    print(f"Dolt latest date: {dolt_max}")

    # Chain parquet status
    print(f"\nChain parquets: {CHAINS_DIR}")
    print(f"{'Ticker':<10} {'Parquet Max Date':<20} {'Dolt Coverage':<20} {'Status'}")
    print("-" * 70)

    for ticker in sorted(tickers)[:20]:
        parquet_max = _existing_max_date_for_ticker(ticker)
        parquet_path = CHAINS_DIR / f"{ticker}.parquet"
        stale_days = (dolt_max - parquet_max).days if (dolt_max and parquet_max) else None
        status = "UP TO DATE" if (stale_days is not None and stale_days <= 3) else \
                 f"STALE ({stale_days}d)" if stale_days is not None else "MISSING"
        print(f"{ticker:<10} {str(parquet_max) if parquet_max else 'MISSING':<20} "
              f"{str(dolt_max) if dolt_max else 'N/A':<20} {status}")

    if len(tickers) > 20:
        print(f"  ... and {len(tickers)-20} more tickers")

    print(f"\nSummary:")
    print(f"  Dolt data frequency:")
    print(f"    2019: WEEKLY (Saturdays)")
    print(f"    2020 – Sep 2024: Mon/Wed/Fri only (~60-72% of trading days)")
    print(f"    Oct 2024 – present: FULL DAILY")
    print(f"  Alpaca bars: Jan 19, 2024 – present (OHLCV only, no greeks)")
    print(f"  yfinance: CURRENT only, no historical snapshots")
    print(f"  Polygon/ThetaData: paid services, not configured")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Daily options chain downloader")
    ap.add_argument("--tickers", default=None,
                    help="Comma-separated ticker list (default: full universe)")
    ap.add_argument("--no-pull", action="store_true",
                    help="Skip Dolt pull (assume already current)")
    ap.add_argument("--force", action="store_true",
                    help="Re-materialize from scratch (ignore existing parquets)")
    ap.add_argument("--status", action="store_true",
                    help="Print coverage status and exit")
    ap.add_argument("--alpaca-gapfill", action="store_true",
                    help="Fill Jan-Sep 2024 Tue/Thu gaps using Alpaca bars (OHLCV only, no greeks)")
    ap.add_argument("--alpaca-start", default="2024-01-19",
                    help="Alpaca gap-fill start date (default: 2024-01-19)")
    ap.add_argument("--alpaca-end", default="2024-09-30",
                    help="Alpaca gap-fill end date (default: 2024-09-30)")
    args = ap.parse_args()

    restrict = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    tickers = load_universe(restrict)
    print(f"[options_dl] Processing {len(tickers)} tickers")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    dolt_bin = _find_dolt()

    # Status only
    if args.status:
        print_status(tickers)
        return

    # Step 1: Dolt pull
    if not args.no_pull:
        if not DOLT_DIR.exists() or not (DOLT_DIR / ".dolt").exists():
            raise SystemExit(
                f"Dolt repo not found at {DOLT_DIR}\n"
                "Clone it first:\n"
                f"  mkdir -p {DOLT_DIR.parent}\n"
                f"  cd {DOLT_DIR.parent}\n"
                f"  {dolt_bin} clone post-no-preference/options options"
            )
        new_data = dolt_pull(dolt_bin)
        if new_data:
            print("[dolt] New data available — re-materializing affected tickers")
        else:
            print("[dolt] No new Dolt data. Checking for stale parquets...")

    # Step 2: Materialize chains
    print(f"\n[chains] Materializing chain parquets (incremental={not args.force})...")
    t0 = time.time()
    stats = materialize_all_chains(
        tickers, dolt_bin,
        incremental=not args.force,
        force=args.force,
    )
    elapsed = time.time() - t0
    print(f"\n[chains] Done in {elapsed:.0f}s")
    print(f"  ok={stats['ok']} empty={stats['empty']} fail={stats['fail']} "
          f"skipped={stats['skipped']} new_rows={stats['new_rows']:,}")

    # Step 3 (optional): Alpaca gap-fill
    if args.alpaca_gapfill:
        print(f"\n[alpaca-gapfill] Filling Tue/Thu gaps in "
              f"{args.alpaca_start} – {args.alpaca_end}...")
        print("  NOTE: Alpaca bars provide OHLCV only. No greeks/IV will be filled.")
        print("  Greeks/delta columns will be NaN for gap-fill rows.")
        print("  These rows are tagged with pricing_source='alpaca_bar_close'")

        alpaca_start = date.fromisoformat(args.alpaca_start)
        alpaca_end = date.fromisoformat(args.alpaca_end)

        if alpaca_start < ALPACA_MIN_DATE:
            print(f"  [warn] Alpaca data starts {ALPACA_MIN_DATE}. Adjusting start.")
            alpaca_start = ALPACA_MIN_DATE

        for ticker in tickers:
            gapfill_df = alpaca_gapfill_ticker(ticker, alpaca_start, alpaca_end)
            if gapfill_df is not None and not gapfill_df.empty:
                out_path = CHAINS_DIR / f"{ticker}.parquet"
                if out_path.exists():
                    existing = pd.read_parquet(out_path)
                    combined = pd.concat([existing, gapfill_df], ignore_index=True)
                    combined = combined.drop_duplicates(
                        subset=["date", "expiration", "strike", "type"]
                    ).sort_values(["date", "expiration", "strike"])
                    combined.to_parquet(out_path, index=False)
                    print(f"  [{ticker}] Appended {len(gapfill_df):,} Alpaca rows")
                else:
                    gapfill_df.to_parquet(out_path, index=False)

    # Final status
    print_status(tickers[:10])
    print(f"\n[options_dl] Complete.")


if __name__ == "__main__":
    main()
