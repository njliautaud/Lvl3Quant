"""
Fetch SPY tick/bar data from Databento for cross-venue analysis.

Usage:
    python fetch_spy_data.py [--schema trades|ohlcv-1s|ohlcv-100ms] [--cost-only]

Requires:
    pip install databento

Data specs:
    - Symbol: SPY (SPDR S&P 500 ETF)
    - Dataset: XNAS.ITCH (Nasdaq TotalView) for best SPY liquidity
      Alt: ARCX.PILLAR (NYSE Arca - SPY primary listing)
    - Date range: 2025-07-14 to 2025-11-28 (matching ES MBO data)
    - Output: data/spy/ directory as CSV + DBN

Set your API key:
    export DATABENTO_KEY=db-xxxxx
    OR edit the KEY variable below.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import date

# ── Configuration ──────────────────────────────────────────────
KEY = os.environ.get("DATABENTO_KEY", "")  # Set via env or paste here
START = "2025-07-14"
END = "2025-11-28"
SYMBOL = "SPY"

# Databento dataset options for SPY:
#   XNAS.ITCH  - Nasdaq TotalView (SPY trades heavily here)
#   ARCX.PILLAR - NYSE Arca (SPY primary listing exchange)
#   OPRA.PILLAR - Options (if we want SPY options later)
DATASET = "XNAS.ITCH"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "spy"


def estimate_cost(client, schema: str):
    """Get cost estimate before downloading."""
    print(f"\n{'='*60}")
    print(f"Cost estimate for {SYMBOL} on {DATASET}")
    print(f"Schema: {schema}, Range: {START} to {END}")
    print(f"{'='*60}")

    try:
        cost = client.metadata.get_cost(
            dataset=DATASET,
            symbols=[SYMBOL],
            schema=schema,
            start=START,
            end=END,
        )
        print(f"Estimated cost: ${cost:.2f}")
        return cost
    except Exception as e:
        print(f"Cost estimate failed: {e}")
        # Try alternative dataset
        print(f"\nTrying alternative dataset ARCX.PILLAR...")
        try:
            cost = client.metadata.get_cost(
                dataset="ARCX.PILLAR",
                symbols=[SYMBOL],
                schema=schema,
                start=START,
                end=END,
            )
            print(f"Estimated cost (ARCX.PILLAR): ${cost:.2f}")
            return cost
        except Exception as e2:
            print(f"ARCX.PILLAR also failed: {e2}")
            return None


def fetch_data(client, schema: str, batch: bool = True):
    """Download SPY data from Databento."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if batch:
        # Batch download (cheaper, async)
        print(f"\nSubmitting batch job for {SYMBOL} {schema}...")
        job = client.batch.submit_job(
            dataset=DATASET,
            symbols=[SYMBOL],
            schema=schema,
            start=START,
            end=END,
            encoding="dbn",
            compression="zstd",
            split_duration="day",  # One file per day, matching ES structure
        )
        print(f"Job submitted: {job.job_id}")
        print(f"Status: {job.status}")
        print(f"\nTo check status later:")
        print(f"  python -c \"import databento as db; c=db.Historical(key='{KEY[:8]}...'); print(c.batch.get_job('{job.job_id}').status)\"")
        print(f"\nTo download when ready:")
        print(f"  python -c \"import databento as db; c=db.Historical(key='{KEY[:8]}...'); c.batch.download('{job.job_id}', '{OUTPUT_DIR}')\"")
        return job
    else:
        # Streaming download (immediate, costs more)
        print(f"\nStreaming {SYMBOL} {schema} data...")
        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[SYMBOL],
            schema=schema,
            start=START,
            end=END,
        )

        # Save as DBN
        dbn_path = OUTPUT_DIR / f"spy_{schema.replace('-', '_')}_{START}_{END}.dbn.zst"
        data.to_file(str(dbn_path))
        print(f"Saved DBN: {dbn_path} ({dbn_path.stat().st_size / 1e6:.1f} MB)")

        # Also save as CSV for quick inspection
        csv_path = OUTPUT_DIR / f"spy_{schema.replace('-', '_')}_{START}_{END}.csv"
        df = data.to_df()
        df.to_csv(str(csv_path))
        print(f"Saved CSV: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:\n{df.head()}")
        return df


def fetch_ohlcv_1s_free_alternatives():
    """
    Alternative FREE sources for SPY 1-second or 1-minute data.
    Use these if Databento credits are exhausted.
    """
    print("\n" + "=" * 60)
    print("FREE ALTERNATIVES FOR SPY BAR DATA")
    print("=" * 60)

    alternatives = """
    1. Yahoo Finance (yfinance) - 1-minute bars, 30-day lookback max
       pip install yfinance
       import yfinance as yf
       spy = yf.download("SPY", interval="1m", start="2025-10-28", end="2025-11-28")
       # Only last 30 days available at 1m resolution

    2. Polygon.io - Free tier: 5 API calls/min, delayed data
       pip install polygon-api-client
       # Supports 1s bars for stocks, but rate-limited on free tier
       # $29/mo Starter plan for full historical access

    3. Alpha Vantage - Free tier: 25 API calls/day
       # Supports 1-min intraday, but limited history
       # Premium: $49.99/mo for extended history

    4. Alpaca Markets (Paper Account) - Free real-time + historical
       pip install alpaca-trade-api
       # Supports 1-min bars, good historical depth
       # Already have Alpaca paper account (see .env)

    5. FirstRate Data - Paid, one-time purchase
       # Tick-level SPY data, ~$50 for the date range needed
       # https://firstratedata.com/

    6. Tiingo - Free tier: 500 requests/hr
       # 1-min IEX data, good for SPY

    RECOMMENDED: Alpaca (free, already have account) for 1-min bars.
    For 1-second or tick data: Databento (new account) or FirstRate Data.
    """
    print(alternatives)


def main():
    parser = argparse.ArgumentParser(description="Fetch SPY data from Databento")
    parser.add_argument(
        "--schema",
        default="ohlcv-1s",
        choices=["trades", "ohlcv-1s", "ohlcv-100ms", "mbp-1", "mbp-10"],
        help="Data schema (default: ohlcv-1s)",
    )
    parser.add_argument(
        "--cost-only",
        action="store_true",
        help="Only estimate cost, don't download",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming (immediate) instead of batch (cheaper)",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET,
        help=f"Databento dataset (default: {DATASET})",
    )
    parser.add_argument(
        "--alternatives",
        action="store_true",
        help="Show free alternative data sources",
    )
    args = parser.parse_args()

    if args.alternatives:
        fetch_ohlcv_1s_free_alternatives()
        return

    if not KEY:
        print("ERROR: No Databento API key found.")
        print("Set via: export DATABENTO_KEY=db-xxxxx")
        print("Or edit the KEY variable in this script.")
        print("\nCurrent key status: ACCOUNT LOCKED ($DATABENTO_API_KEY)")
        print("You need a new Databento account + key.")
        fetch_ohlcv_1s_free_alternatives()
        sys.exit(1)

    import databento as db

    global DATASET
    DATASET = args.dataset
    client = db.Historical(key=KEY)

    # Step 1: Estimate cost
    cost = estimate_cost(client, args.schema)
    if cost is None:
        print("\nFailed to estimate cost. Check API key and dataset.")
        sys.exit(1)

    if args.cost_only:
        print("\n(--cost-only mode, not downloading)")
        return

    # Step 2: Confirm and download
    print(f"\nProceed with download? Estimated cost: ${cost:.2f}")
    confirm = input("Type 'yes' to proceed: ")
    if confirm.lower() != "yes":
        print("Aborted.")
        return

    # Step 3: Fetch
    result = fetch_data(client, args.schema, batch=not args.stream)
    print("\nDone!")


if __name__ == "__main__":
    main()
