"""
Fetch SPY 1-minute bar data from Alpaca Markets API.
Date range: Jul 14 - Nov 28, 2025 (aligned to ES MBO data).

Alpaca free-tier historical data API supports up to 10,000 bars per request.
SPY has ~390 1-min bars per trading day, ~100 trading days in range = ~39,000 bars.
We paginate using the next_page_token.

Output: Lvl3Quant/data/spy/spy_1min_bars.parquet
"""

import os
import requests
import pandas as pd
import time
import sys
from datetime import datetime

# Alpaca API credentials (paper trading keys work for market data)
API_KEY = os.environ.get("ALPACA_API_KEY", "")
API_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")

# Use the SIP feed for historical data (paper keys get IEX by default,
# but we'll try SIP first; fall back to IEX if needed)
BASE_URL = "https://data.alpaca.markets/v2"

headers = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Accept": "application/json",
}

OUTPUT_PATH = r"C:\Users\Footb\Documents\Github\Lvl3Quant\data\spy\spy_1min_bars.parquet"

def fetch_spy_bars():
    """Fetch SPY 1-min bars from Jul 14 to Nov 28, 2025."""

    start = "2025-07-14T00:00:00Z"
    end = "2025-11-28T23:59:59Z"

    all_bars = []
    page_token = None
    page = 0

    print(f"Fetching SPY 1-min bars: {start} to {end}")
    print(f"Using Alpaca data API: {BASE_URL}")

    while True:
        params = {
            "start": start,
            "end": end,
            "timeframe": "1Min",
            "limit": 10000,
            "adjustment": "split",  # adjust for splits
            "feed": "sip",  # try SIP first
        }

        if page_token:
            params["page_token"] = page_token

        try:
            resp = requests.get(
                f"{BASE_URL}/stocks/SPY/bars",
                headers=headers,
                params=params,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            time.sleep(2)
            continue

        if resp.status_code == 403:
            print("SIP feed not available on free tier, falling back to IEX...")
            params["feed"] = "iex"
            resp = requests.get(
                f"{BASE_URL}/stocks/SPY/bars",
                headers=headers,
                params=params,
                timeout=30,
            )

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 5))
            print(f"Rate limited. Waiting {retry_after}s...")
            time.sleep(retry_after)
            continue

        if resp.status_code != 200:
            print(f"Error {resp.status_code}: {resp.text}")
            if resp.status_code == 422:
                # Try without SIP feed
                params["feed"] = "iex"
                resp = requests.get(
                    f"{BASE_URL}/stocks/SPY/bars",
                    headers=headers,
                    params=params,
                    timeout=30,
                )
                if resp.status_code != 200:
                    print(f"IEX fallback also failed: {resp.status_code} {resp.text}")
                    break
            else:
                break

        data = resp.json()
        bars = data.get("bars", [])

        if not bars:
            print("No more bars returned.")
            break

        all_bars.extend(bars)
        page += 1

        # Progress
        first_t = bars[0]["t"]
        last_t = bars[-1]["t"]
        print(f"  Page {page}: {len(bars)} bars ({first_t} to {last_t}), total: {len(all_bars)}")

        # Check for next page
        page_token = data.get("next_page_token")
        if not page_token:
            print("No more pages.")
            break

        # Small delay to be nice to the API
        time.sleep(0.3)

    if not all_bars:
        print("ERROR: No data fetched!")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_bars)

    # Parse timestamp
    df["t"] = pd.to_datetime(df["t"])
    df = df.rename(columns={
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "n": "trade_count",
        "vw": "vwap",
    })

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Summary stats
    print(f"\n=== SPY 1-Min Bar Summary ===")
    print(f"Total bars: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Trading days: {df['timestamp'].dt.date.nunique()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"Avg volume/bar: {df['volume'].mean():,.0f}")

    # Save to parquet
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"File size: {pd.io.common.file_exists(OUTPUT_PATH)}")

    return df


def fetch_with_yfinance_fallback():
    """Fallback: use yfinance for daily data if Alpaca fails."""
    try:
        import yfinance as yf
        print("\nFalling back to yfinance (daily bars only for historical range)...")
        spy = yf.Ticker("SPY")
        df = spy.history(start="2025-07-14", end="2025-11-28", interval="1d")

        if df.empty:
            print("yfinance also returned no data.")
            return None

        df = df.reset_index()
        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        output = OUTPUT_PATH.replace("1min", "daily")
        df.to_parquet(output, index=False)
        print(f"Saved daily bars to: {output}")
        print(f"Total bars: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df

    except ImportError:
        print("yfinance not installed. Install with: pip install yfinance")
        return None


if __name__ == "__main__":
    df = fetch_spy_bars()

    if df is None:
        print("\nAlpaca fetch failed. Trying yfinance fallback...")
        df = fetch_with_yfinance_fallback()

    if df is not None:
        print("\nDone! Data ready for alignment.")
    else:
        print("\nFailed to fetch SPY data from any source.")
        sys.exit(1)
