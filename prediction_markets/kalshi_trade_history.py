#!/usr/bin/env python3
"""
Kalshi Historical Trade Downloader — Downloads public trade data via the Kalshi API.

Uses the FREE (no auth required) Kalshi public API endpoint:
    GET https://api.elections.kalshi.com/trade-api/v2/markets/trades

Parameters:
    ticker:  Market ticker (e.g., KXINX-26MAR07H1600-B5875)
    min_ts:  Minimum timestamp (epoch seconds)
    max_ts:  Maximum timestamp (epoch seconds)
    limit:   Max trades per request (up to 1000)
    cursor:  Pagination cursor

Note on SPX markets:
    The S&P 500 bracket markets on Kalshi use the following series:
    - KXINX:  S&P 500 range (daily brackets, settlement at 4pm ET)
    - KXINXB: S&P 500 range (another bracket series)
    - KXINXW: S&P 500 weekly range
    - KXINXM: S&P 500 monthly range
    - KXINXI: S&P 500 hourly
    - KXINXU: S&P 500 above/below
    - KXINXAB: S&P 500 close above/below
    - KXINXZ: S&P 500 up/down

    Market tickers follow: {SERIES}-{DATE}H{HOUR}-B{STRIKE} for brackets
                           {SERIES}-{DATE}H{HOUR}-T{STRIKE} for above/below

Usage:
    python kalshi_trade_history.py                          # Download SPX bracket trades
    python kalshi_trade_history.py --days 30                # Last 30 days
    python kalshi_trade_history.py --series KXINX KXINXW    # Specific series
    python kalshi_trade_history.py --all-trades             # All recent trades (no filter)
    python kalshi_trade_history.py --summary                # Show volume summary

Output:
    data/kalshi_trades/YYYY-MM-DD.jsonl — Trades by date
    data/kalshi_trades/summary.json     — Volume and spread summary
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logger = logging.getLogger("kalshi_trade_history")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# Kalshi API (new elections endpoint, no auth for public market data)
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DATA_DIR = Path(__file__).parent / "data" / "kalshi_trades"

# SPX bracket series on Kalshi
SPX_SERIES = [
    "KXINX",     # S&P 500 range (daily brackets)
    "KXINXB",    # S&P 500 range (alt bracket series)
    "KXINXW",    # S&P 500 weekly range
    "KXINXM",    # S&P 500 monthly range
    "KXINXI",    # S&P 500 hourly
    "KXINXU",    # S&P 500 above/below
    "KXINXAB",   # S&P 500 close above/below
    "KXINXZ",    # S&P 500 up/down
]

# Additional series for SPX direction markets
SPX_DIRECTION_SERIES = [
    "KXINXZ",    # S&P 500 positive/negative day
    "KXINXPOS",  # S&P 500 positive year
]


def _get(endpoint: str, params: dict = None, retries: int = 3) -> dict:
    """Make a GET request to the Kalshi API with retry."""
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                # Rate limited
                wait = int(resp.headers.get("Retry-After", 5))
                logger.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            logger.error(f"API request failed: {e}")
            raise
    return {}


def get_markets_for_series(
    series_ticker: str,
    status: str = "settled",
    limit: int = 200,
) -> list[dict]:
    """Get all markets for a given series.

    Args:
        series_ticker: Series ticker (e.g., KXINX)
        status: 'open', 'closed', 'settled', or None for all
        limit: Max markets to return

    Returns:
        List of market dicts
    """
    params = {"series_ticker": series_ticker, "limit": limit}
    if status:
        params["status"] = status

    all_markets = []
    cursor = ""

    while True:
        if cursor:
            params["cursor"] = cursor
        data = _get("markets", params)
        markets = data.get("markets", [])
        all_markets.extend(markets)

        cursor = data.get("cursor", "")
        if not cursor or not markets:
            break

        time.sleep(0.3)  # Rate limiting

    return all_markets


def get_trades_for_ticker(
    ticker: str,
    min_ts: int = None,
    max_ts: int = None,
    limit: int = 1000,
) -> list[dict]:
    """Get all trades for a specific market ticker.

    Paginates through all results using cursor.

    Args:
        ticker: Market ticker
        min_ts: Minimum timestamp (epoch seconds)
        max_ts: Maximum timestamp (epoch seconds)
        limit: Max trades per page (up to 1000)

    Returns:
        List of trade dicts
    """
    params = {"ticker": ticker, "limit": min(limit, 1000)}
    if min_ts:
        params["min_ts"] = min_ts
    if max_ts:
        params["max_ts"] = max_ts

    all_trades = []
    cursor = ""
    pages = 0

    while True:
        if cursor:
            params["cursor"] = cursor

        data = _get("markets/trades", params)
        trades = data.get("trades", [])
        all_trades.extend(trades)
        pages += 1

        cursor = data.get("cursor", "")
        if not cursor or not trades:
            break

        time.sleep(0.2)  # Rate limiting

    return all_trades


def get_recent_trades(
    min_ts: int = None,
    max_ts: int = None,
    limit: int = 1000,
    max_pages: int = 50,
) -> list[dict]:
    """Get recent trades across all markets (no ticker filter).

    Useful for finding what's actually trading.

    Args:
        min_ts: Minimum timestamp
        max_ts: Maximum timestamp
        limit: Trades per page
        max_pages: Max pagination pages

    Returns:
        List of trade dicts
    """
    params = {"limit": min(limit, 1000)}
    if min_ts:
        params["min_ts"] = min_ts
    if max_ts:
        params["max_ts"] = max_ts

    all_trades = []
    cursor = ""

    for page in range(max_pages):
        if cursor:
            params["cursor"] = cursor

        data = _get("markets/trades", params)
        trades = data.get("trades", [])
        all_trades.extend(trades)

        cursor = data.get("cursor", "")
        if not cursor or not trades:
            break

        time.sleep(0.2)

        if (page + 1) % 10 == 0:
            logger.info(f"  Page {page+1}, total trades: {len(all_trades)}")

    return all_trades


def download_spx_trades(
    days: int = 30,
    series_list: list = None,
    dest_dir: Path = None,
) -> dict:
    """Download all SPX bracket trades for the given period.

    Strategy:
    1. Get all markets (open + settled) for each SPX series
    2. For each market with volume > 0, download all trades
    3. Save trades as JSONL by date

    Args:
        days: Number of days to look back
        series_list: List of series tickers. Default: SPX_SERIES
        dest_dir: Output directory. Default: data/kalshi_trades/

    Returns:
        Summary dict
    """
    if series_list is None:
        series_list = SPX_SERIES + SPX_DIRECTION_SERIES
    if dest_dir is None:
        dest_dir = DATA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    min_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

    summary = {
        "series_scanned": [],
        "markets_found": 0,
        "markets_with_volume": 0,
        "total_trades": 0,
        "trades_by_date": {},
        "volume_by_series": {},
    }

    # Collect all trades grouped by date
    trades_by_date = defaultdict(list)

    for series in series_list:
        logger.info(f"Scanning series: {series}")
        summary["series_scanned"].append(series)

        # Get both open and settled markets
        markets = []
        for status in ["open", "settled", "closed"]:
            try:
                batch = get_markets_for_series(series, status=status)
                markets.extend(batch)
            except Exception as e:
                logger.warning(f"Failed to get {series} {status} markets: {e}")

        summary["markets_found"] += len(markets)
        logger.info(f"  Found {len(markets)} markets for {series}")

        # Filter for markets with volume
        vol_markets = [m for m in markets if m.get("volume", 0) > 0]
        summary["markets_with_volume"] += len(vol_markets)

        series_volume = 0
        series_trade_count = 0

        for market in vol_markets:
            ticker = market.get("ticker", "")
            volume = market.get("volume", 0)

            logger.info(f"  Downloading trades for {ticker} (vol={volume})...")

            try:
                trades = get_trades_for_ticker(ticker, min_ts=min_ts)
            except Exception as e:
                logger.warning(f"  Failed to get trades for {ticker}: {e}")
                continue

            if not trades:
                continue

            logger.info(f"    Got {len(trades)} trades")
            series_volume += sum(t.get("count", 0) for t in trades)
            series_trade_count += len(trades)

            # Group by date
            for trade in trades:
                ts = trade.get("created_time", "")
                date_str = ts[:10] if ts else "unknown"
                trades_by_date[date_str].append(trade)

            time.sleep(0.3)

        summary["volume_by_series"][series] = {
            "volume_contracts": series_volume,
            "trade_count": series_trade_count,
            "markets": len(vol_markets),
        }

    # If no SPX-specific trades found, also grab recent trades and filter
    if sum(len(v) for v in trades_by_date.values()) == 0:
        logger.info("No SPX trades found via series query. Fetching recent trades...")
        recent = get_recent_trades(min_ts=min_ts, max_pages=20)
        logger.info(f"Got {len(recent)} recent trades across all markets")

        # Filter for any that look SPX-related
        spx_prefixes = tuple(s + "-" for s in series_list)
        for trade in recent:
            ticker = trade.get("ticker", "")
            if any(ticker.startswith(p) for p in spx_prefixes):
                ts = trade.get("created_time", "")
                date_str = ts[:10] if ts else "unknown"
                trades_by_date[date_str].append(trade)

        # Also save ALL recent trades for analysis
        all_trades_path = dest_dir / "all_recent_trades.jsonl"
        with open(all_trades_path, "w") as f:
            for trade in recent:
                f.write(json.dumps(trade) + "\n")
        logger.info(f"Saved {len(recent)} recent trades to {all_trades_path}")

        # Compute volume breakdown by ticker prefix
        volume_by_prefix = defaultdict(lambda: {"count": 0, "volume": 0})
        for trade in recent:
            ticker = trade.get("ticker", "")
            # Extract prefix (series + event)
            parts = ticker.split("-")
            prefix = parts[0] if parts else "unknown"
            volume_by_prefix[prefix]["count"] += 1
            volume_by_prefix[prefix]["volume"] += trade.get("count", 0)

        # Sort by volume
        top_prefixes = sorted(
            volume_by_prefix.items(), key=lambda x: -x[1]["volume"]
        )[:20]
        summary["top_trading_series"] = {
            k: v for k, v in top_prefixes
        }

    # Save trades by date
    for date_str, trades in sorted(trades_by_date.items()):
        date_file = dest_dir / f"{date_str}.jsonl"
        with open(date_file, "w") as f:
            for trade in trades:
                f.write(json.dumps(trade) + "\n")
        summary["trades_by_date"][date_str] = len(trades)
        summary["total_trades"] += len(trades)

    # Save summary
    summary["download_time"] = datetime.now(timezone.utc).isoformat()
    summary["period_days"] = days
    summary_path = dest_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {summary_path}")
    return summary


def compute_volume_summary(dest_dir: Path = None) -> dict:
    """Compute volume, trade count, and average spread from downloaded data.

    Returns:
        Summary dict with statistics
    """
    if dest_dir is None:
        dest_dir = DATA_DIR

    if not dest_dir.exists():
        return {"error": "No data directory found. Run download first."}

    jsonl_files = sorted(dest_dir.glob("*.jsonl"))
    if not jsonl_files:
        return {"error": "No JSONL files found"}

    all_trades = []
    for f in jsonl_files:
        if f.name == "all_recent_trades.jsonl":
            continue
        with open(f) as fh:
            for line in fh:
                try:
                    all_trades.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    if not all_trades:
        return {"error": "No trades loaded"}

    # Aggregate stats
    total_volume = sum(t.get("count", 0) for t in all_trades)
    total_trades = len(all_trades)

    # Volume by date
    volume_by_date = defaultdict(int)
    trades_by_date = defaultdict(int)
    for t in all_trades:
        date_str = t.get("created_time", "")[:10]
        volume_by_date[date_str] += t.get("count", 0)
        trades_by_date[date_str] += 1

    # Price stats (average spread proxy: |yes_price - 50|)
    prices = [t.get("yes_price", 50) for t in all_trades if "yes_price" in t]
    price_dollars = [float(t.get("yes_price_dollars", "0.50")) for t in all_trades
                     if "yes_price_dollars" in t]

    # Volume by ticker series
    series_vol = defaultdict(lambda: {"volume": 0, "trades": 0})
    for t in all_trades:
        ticker = t.get("ticker", "")
        series = ticker.split("-")[0] if "-" in ticker else ticker
        series_vol[series]["volume"] += t.get("count", 0)
        series_vol[series]["trades"] += 1

    return {
        "total_trades": total_trades,
        "total_volume_contracts": total_volume,
        "date_range": {
            "start": min(volume_by_date.keys()) if volume_by_date else None,
            "end": max(volume_by_date.keys()) if volume_by_date else None,
        },
        "avg_daily_volume": round(total_volume / max(len(volume_by_date), 1), 1),
        "avg_daily_trades": round(total_trades / max(len(trades_by_date), 1), 1),
        "avg_price_cents": round(sum(prices) / max(len(prices), 1), 1) if prices else None,
        "avg_price_dollars": round(sum(price_dollars) / max(len(price_dollars), 1), 4) if price_dollars else None,
        "volume_by_date": dict(sorted(volume_by_date.items())),
        "top_series": dict(sorted(series_vol.items(), key=lambda x: -x[1]["volume"])[:15]),
    }


def format_summary(summary: dict) -> str:
    """Format summary dict as readable text."""
    lines = [
        "=" * 60,
        "  KALSHI TRADE HISTORY SUMMARY",
        "=" * 60,
        "",
        f"  Total trades:       {summary.get('total_trades', 0):>8,d}",
        f"  Total volume:       {summary.get('total_volume_contracts', 0):>8,d} contracts",
        f"  Date range:         {summary.get('date_range', {}).get('start', 'N/A')} to "
        f"{summary.get('date_range', {}).get('end', 'N/A')}",
        f"  Avg daily volume:   {summary.get('avg_daily_volume', 0):>8,.0f} contracts",
        f"  Avg daily trades:   {summary.get('avg_daily_trades', 0):>8,.0f}",
        "",
    ]

    if summary.get("avg_price_dollars"):
        lines.append(f"  Avg trade price:    ${summary['avg_price_dollars']:.4f}")

    # Volume by series
    if summary.get("top_series"):
        lines.extend([
            "",
            "-" * 60,
            "  VOLUME BY SERIES",
            "-" * 60,
            f"  {'Series':<25s} | {'Volume':>10s} | {'Trades':>8s}",
            "  " + "-" * 50,
        ])
        for series, data in summary["top_series"].items():
            lines.append(
                f"  {series:<25s} | {data['volume']:>10,d} | {data['trades']:>8,d}"
            )

    # Volume by date
    if summary.get("volume_by_date"):
        lines.extend([
            "",
            "-" * 60,
            "  DAILY VOLUME",
            "-" * 60,
        ])
        for date_str, vol in sorted(summary["volume_by_date"].items()):
            lines.append(f"  {date_str}:  {vol:>8,d} contracts")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download Kalshi trade history for SPX bracket markets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days to look back")
    parser.add_argument("--series", nargs="+", default=None,
                        help="Series tickers to download (default: all SPX)")
    parser.add_argument("--all-trades", action="store_true",
                        help="Download all recent trades (not just SPX)")
    parser.add_argument("--summary", action="store_true",
                        help="Show volume summary from downloaded data")
    parser.add_argument("--dest", type=str, default=None,
                        help="Destination directory")
    parser.add_argument("--list-markets", action="store_true",
                        help="List available SPX markets without downloading trades")
    parser.add_argument("--test", action="store_true",
                        help="Test API connection with a small request")

    args = parser.parse_args()

    dest_dir = Path(args.dest) if args.dest else DATA_DIR

    # Test mode
    if args.test:
        print("Testing Kalshi API connection...")
        try:
            data = _get("markets/trades", {"limit": 3})
            trades = data.get("trades", [])
            print(f"API OK. Got {len(trades)} recent trades:")
            for t in trades:
                print(f"  {t.get('ticker', 'N/A')} — "
                      f"count={t.get('count', 0)} "
                      f"price=${t.get('yes_price_dollars', 'N/A')} "
                      f"at {t.get('created_time', '')[:19]}")
        except Exception as e:
            print(f"API Error: {e}")
        return

    # Summary mode
    if args.summary:
        vol_summary = compute_volume_summary(dest_dir)
        if "error" in vol_summary:
            print(f"Error: {vol_summary['error']}")
        else:
            print(format_summary(vol_summary))
        return

    # List markets mode
    if args.list_markets:
        series_list = args.series or SPX_SERIES
        for series in series_list:
            print(f"\n{series}:")
            for status in ["open", "settled"]:
                try:
                    markets = get_markets_for_series(series, status=status, limit=10)
                    if markets:
                        print(f"  {status}: {len(markets)} markets")
                        for m in markets[:3]:
                            print(f"    {m.get('ticker')} | {m.get('subtitle', '')[:40]} | "
                                  f"vol={m.get('volume', 0)} | result={m.get('result', '-')}")
                except Exception as e:
                    print(f"  {status}: error — {e}")
        return

    # All-trades mode
    if args.all_trades:
        min_ts = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp())
        logger.info(f"Downloading all trades for last {args.days} days...")
        trades = get_recent_trades(min_ts=min_ts, max_pages=100)
        logger.info(f"Got {len(trades)} trades")

        # Save by date
        dest_dir.mkdir(parents=True, exist_ok=True)
        trades_by_date = defaultdict(list)
        for t in trades:
            date_str = t.get("created_time", "")[:10]
            trades_by_date[date_str].append(t)

        for date_str, day_trades in sorted(trades_by_date.items()):
            path = dest_dir / f"{date_str}.jsonl"
            with open(path, "w") as f:
                for t in day_trades:
                    f.write(json.dumps(t) + "\n")
            print(f"  {date_str}: {len(day_trades)} trades")

        print(f"\nTotal: {len(trades)} trades across {len(trades_by_date)} dates")
        return

    # Default: download SPX trades
    series_list = args.series or (SPX_SERIES + SPX_DIRECTION_SERIES)
    logger.info(f"Downloading SPX trades for {args.days} days, "
                f"series: {', '.join(series_list)}")

    summary = download_spx_trades(
        days=args.days,
        series_list=series_list,
        dest_dir=dest_dir,
    )

    print(f"\n{'='*60}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"  Series scanned:    {len(summary['series_scanned'])}")
    print(f"  Markets found:     {summary['markets_found']}")
    print(f"  Markets w/ volume: {summary['markets_with_volume']}")
    print(f"  Total trades:      {summary['total_trades']}")

    if summary.get("top_trading_series"):
        print(f"\n  Top trading series (from recent trades):")
        for series, data in list(summary["top_trading_series"].items())[:10]:
            print(f"    {series:<25s} vol={data['volume']:>6d} trades={data['count']:>4d}")

    print(f"\n  Data saved to: {dest_dir}")


if __name__ == "__main__":
    main()
