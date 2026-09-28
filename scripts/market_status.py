#!/usr/bin/env python3
"""
Market status utility — answers "is the market open right now?" definitively.

Usage:
    python3 market_status.py              # JSON output
    python3 market_status.py --oneliner   # Single line for injection into prompts
    python3 market_status.py --guard      # Exit code 0 if market open, 1 if closed

Covers:
- Weekends
- US market holidays (NYSE calendar through 2027)
- Pre-market / regular / after-hours / closed windows
- Early close days (day before Independence Day, Black Friday, Christmas Eve)

State file written to: state/market_status_now.json (updated each call)
"""

import json
import sys
from datetime import datetime, date, time
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# NYSE holidays — static list through 2027
# Source: NYSE holiday calendar
NYSE_HOLIDAYS = {
    # 2026
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day (observed)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
    # 2027
    date(2027, 1, 1),   # New Year's Day
    date(2027, 1, 18),  # MLK Day
    date(2027, 2, 15),  # Presidents' Day
    date(2027, 3, 26),  # Good Friday
    date(2027, 5, 31),  # Memorial Day
    date(2027, 6, 18),  # Juneteenth (observed, falls on Sat)
    date(2027, 7, 5),   # Independence Day (observed)
    date(2027, 9, 6),   # Labor Day
    date(2027, 11, 25), # Thanksgiving
    date(2027, 12, 24), # Christmas (observed, falls on Sat)
}

# Early close days (1:00 PM ET close)
EARLY_CLOSE = {
    date(2026, 7, 2),   # Day before Independence Day (observed)
    date(2026, 11, 27), # Black Friday
    date(2026, 12, 24), # Christmas Eve
    date(2027, 11, 26), # Black Friday
}


def get_market_status() -> dict:
    """Return full market status dict."""
    now = datetime.now(ET)
    today = now.date()
    day_of_week = now.strftime("%A")
    current_time = now.time()

    result = {
        "timestamp_et": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "day_of_week": day_of_week,
        "date": str(today),
    }

    # Weekend check
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        result["is_trading_day"] = False
        result["market_open"] = False
        result["session"] = "weekend"
        result["reason"] = f"Weekend ({day_of_week})"
        next_open = today
        while next_open.weekday() >= 5 or next_open in NYSE_HOLIDAYS:
            from datetime import timedelta
            next_open += timedelta(days=1)
        result["next_open"] = str(next_open)
        return result

    # Holiday check
    if today in NYSE_HOLIDAYS:
        result["is_trading_day"] = False
        result["market_open"] = False
        result["session"] = "holiday"
        result["reason"] = "NYSE holiday"
        from datetime import timedelta
        next_open = today + timedelta(days=1)
        while next_open.weekday() >= 5 or next_open in NYSE_HOLIDAYS:
            next_open += timedelta(days=1)
        result["next_open"] = str(next_open)
        return result

    # It's a trading day
    result["is_trading_day"] = True

    # Determine close time
    close_time = time(13, 0) if today in EARLY_CLOSE else time(16, 0)
    early = today in EARLY_CLOSE
    result["early_close"] = early

    # Session windows
    pre_open = time(4, 0)
    market_open = time(9, 30)

    if current_time < pre_open:
        result["market_open"] = False
        result["session"] = "overnight"
        result["reason"] = "Before pre-market (opens 4:00 AM ET)"
    elif current_time < market_open:
        result["market_open"] = False
        result["session"] = "pre_market"
        result["reason"] = f"Pre-market hours (regular session opens 9:30 AM ET)"
    elif current_time < close_time:
        result["market_open"] = True
        result["session"] = "regular"
        close_str = "1:00 PM" if early else "4:00 PM"
        result["reason"] = f"Regular trading hours (closes {close_str} ET)"
    elif current_time < time(20, 0):
        result["market_open"] = False
        result["session"] = "after_hours"
        result["reason"] = "After-hours (regular session closed)"
    else:
        result["market_open"] = False
        result["session"] = "closed"
        result["reason"] = "Market closed for the day"

    return result


def oneliner(status: dict) -> str:
    """One-line summary for prompt injection."""
    ts = status["timestamp_et"]
    day = status["day_of_week"]
    if not status["is_trading_day"]:
        reason = status["reason"]
        next_open = status.get("next_open", "unknown")
        return f"[{ts}] {day} — MARKET CLOSED ({reason}). Next open: {next_open}. No signals, no execution, no position changes."
    elif status["market_open"]:
        return f"[{ts}] {day} — MARKET OPEN ({status['reason']}). Signals and execution ACTIVE."
    else:
        return f"[{ts}] {day} — Trading day but {status['session']} ({status['reason']}). No execution until regular session."


def main():
    status = get_market_status()

    # Write state file
    state_path = Path(__file__).parent.parent / "state" / "market_status_now.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(status, f, indent=2)

    if "--oneliner" in sys.argv:
        print(oneliner(status))
    elif "--guard" in sys.argv:
        sys.exit(0 if status.get("market_open") else 1)
    else:
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
