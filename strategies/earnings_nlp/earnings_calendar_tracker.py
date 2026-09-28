#!/usr/bin/env python3
"""
Earnings Calendar Tracker
===========================
Tracks upcoming earnings dates for our quality universe using yfinance.
Alerts when earnings are within 3 days so we can prepare.

Usage:
    python earnings_calendar_tracker.py                    # Check all tickers
    python earnings_calendar_tracker.py --ticker AAPL      # Check single ticker
    python earnings_calendar_tracker.py --days-ahead 7     # Custom lookahead
    python earnings_calendar_tracker.py --json              # Output JSON only
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

# Paths
BASE_DIR = Path("/home/jupiter/Lvl3Quant")
UNIVERSE_FILE = BASE_DIR / "data" / "quality_universe.json"
CALENDAR_FILE = BASE_DIR / "state" / "earnings_calendar.json"
LOG_DIR = BASE_DIR / "logs" / "earnings_nlp"

LOG_DIR.mkdir(parents=True, exist_ok=True)
CALENDAR_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "calendar_tracker.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def load_universe() -> list:
    """Load ticker universe."""
    if UNIVERSE_FILE.exists():
        with open(UNIVERSE_FILE) as f:
            return json.load(f).get("tickers", [])
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "LLY",
        "UNH", "JNJ", "JPM", "V", "MA", "PG", "HD", "COST", "ABBV", "MRK",
        "AVGO", "PEP", "KO", "TMO", "ACN", "MCD", "LIN", "AMD", "CRM",
        "ISRG", "NFLX", "INTU", "TXN", "LOW", "AMAT",
    ]


def get_earnings_date(ticker: str) -> dict:
    """Get upcoming earnings date for a ticker using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar

        if cal is None or (isinstance(cal, dict) and not cal):
            # Try earnings_dates instead
            try:
                dates = stock.earnings_dates
                if dates is not None and not dates.empty:
                    future_dates = dates.index[dates.index >= datetime.now()]
                    if len(future_dates) > 0:
                        next_date = future_dates[0]
                        return {
                            "ticker": ticker,
                            "earnings_date": next_date.strftime("%Y-%m-%d"),
                            "source": "earnings_dates",
                            "status": "found",
                        }
                    else:
                        # Get the most recent past date
                        if len(dates.index) > 0:
                            last_date = dates.index[0]
                            return {
                                "ticker": ticker,
                                "earnings_date": last_date.strftime("%Y-%m-%d"),
                                "source": "earnings_dates_past",
                                "status": "past_only",
                            }
            except Exception:
                pass

            return {
                "ticker": ticker,
                "earnings_date": None,
                "source": "none",
                "status": "not_found",
            }

        # Extract from calendar
        if isinstance(cal, dict):
            earnings_date = cal.get("Earnings Date")
            if isinstance(earnings_date, list) and len(earnings_date) > 0:
                earnings_date = earnings_date[0]
            if earnings_date:
                if hasattr(earnings_date, 'strftime'):
                    date_str = earnings_date.strftime("%Y-%m-%d")
                else:
                    date_str = str(earnings_date)[:10]
                return {
                    "ticker": ticker,
                    "earnings_date": date_str,
                    "source": "calendar",
                    "status": "found",
                }

        return {
            "ticker": ticker,
            "earnings_date": None,
            "source": "calendar_empty",
            "status": "not_found",
        }

    except Exception as e:
        log.error(f"{ticker}: Error fetching earnings date: {e}")
        return {
            "ticker": ticker,
            "earnings_date": None,
            "source": "error",
            "status": "error",
            "error": str(e),
        }


def check_all_earnings(tickers: list = None, days_ahead: int = 30) -> dict:
    """Check earnings dates for all tickers and categorize urgency."""
    if tickers is None:
        tickers = load_universe()

    now = datetime.now()
    results = {
        "checked_at": now.isoformat(),
        "total_tickers": len(tickers),
        "imminent": [],      # Within 3 days
        "upcoming": [],      # 3-7 days
        "scheduled": [],     # 7-30 days
        "reported_recently": [],  # Already reported within last 30 days
        "unknown": [],       # No date found
        "all_dates": [],
    }

    for ticker in tickers:
        log.info(f"Checking {ticker}...")
        info = get_earnings_date(ticker)

        if not info.get("earnings_date"):
            results["unknown"].append(info)
            results["all_dates"].append(info)
            continue

        try:
            earnings_dt = datetime.strptime(info["earnings_date"], "%Y-%m-%d")
            days_until = (earnings_dt - now).days
            info["days_until"] = days_until

            if days_until < 0:
                info["category"] = "reported_recently"
                info["days_since"] = abs(days_until)
                results["reported_recently"].append(info)
            elif days_until <= 3:
                info["category"] = "imminent"
                info["alert"] = True
                results["imminent"].append(info)
                log.warning(f"*** {ticker}: EARNINGS IN {days_until} DAYS ({info['earnings_date']}) ***")
            elif days_until <= 7:
                info["category"] = "upcoming"
                results["upcoming"].append(info)
            elif days_until <= days_ahead:
                info["category"] = "scheduled"
                results["scheduled"].append(info)

            results["all_dates"].append(info)
        except ValueError:
            info["days_until"] = None
            info["category"] = "unknown"
            results["unknown"].append(info)
            results["all_dates"].append(info)

    # Sort each category by date
    for cat in ["imminent", "upcoming", "scheduled", "reported_recently"]:
        results[cat].sort(key=lambda x: x.get("earnings_date", "9999"))

    results["summary"] = {
        "imminent_count": len(results["imminent"]),
        "upcoming_count": len(results["upcoming"]),
        "scheduled_count": len(results["scheduled"]),
        "recently_reported_count": len(results["reported_recently"]),
        "unknown_count": len(results["unknown"]),
    }

    return results


def format_report(results: dict) -> str:
    """Format a readable earnings calendar report."""
    lines = [
        "=" * 60,
        "EARNINGS CALENDAR REPORT",
        f"Generated: {results['checked_at'][:19]}",
        f"Universe: {results['total_tickers']} tickers",
        "=" * 60,
    ]

    if results["imminent"]:
        lines.append("\n*** IMMINENT (within 3 days) — PREPARE NOW ***")
        for item in results["imminent"]:
            days = item.get("days_until", "?")
            lines.append(f"  {item['ticker']:6s} | {item['earnings_date']} | {days} day(s) away")

    if results["upcoming"]:
        lines.append("\nUPCOMING (3-7 days)")
        for item in results["upcoming"]:
            days = item.get("days_until", "?")
            lines.append(f"  {item['ticker']:6s} | {item['earnings_date']} | {days} day(s) away")

    if results["scheduled"]:
        lines.append("\nSCHEDULED (7-30 days)")
        for item in results["scheduled"]:
            days = item.get("days_until", "?")
            lines.append(f"  {item['ticker']:6s} | {item['earnings_date']} | {days} day(s) away")

    if results["reported_recently"]:
        lines.append("\nRECENTLY REPORTED")
        for item in results["reported_recently"]:
            days = item.get("days_since", "?")
            lines.append(f"  {item['ticker']:6s} | {item['earnings_date']} | {days} day(s) ago")

    if results["unknown"]:
        lines.append(f"\nUNKNOWN: {', '.join(i['ticker'] for i in results['unknown'])}")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Track upcoming earnings dates")
    parser.add_argument("--ticker", type=str, help="Check single ticker")
    parser.add_argument("--days-ahead", type=int, default=30, help="How many days ahead to look")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Earnings Calendar Tracker Starting")

    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = load_universe()

    results = check_all_earnings(tickers, args.days_ahead)

    # Save results
    with open(CALENDAR_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Saved calendar to {CALENDAR_FILE}")

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        report = format_report(results)
        print(report)

    # Summary
    s = results["summary"]
    log.info(
        f"Summary: {s['imminent_count']} imminent, {s['upcoming_count']} upcoming, "
        f"{s['scheduled_count']} scheduled, {s['recently_reported_count']} recently reported, "
        f"{s['unknown_count']} unknown"
    )

    return results


if __name__ == "__main__":
    main()
