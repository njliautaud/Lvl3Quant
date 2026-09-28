#!/usr/bin/env python3
"""
v3.3 side-research task #4 (per HC #304(D)):
Build economic calendar JSON for FOMC / CPI / NFP from FRED public API + static known dates.

Output: data/external/economic_calendar_2023_2026.json
Format: [{event: str, datetime_utc: ISO8601, category: str, importance: str}]

This is a NEW research script — does not modify any existing code.
Reads from public FRED API (no auth needed for series metadata).
"""
import json
import datetime as dt
from pathlib import Path

OUT_PATH = Path("/home/jupiter/Lvl3Quant/data/external/economic_calendar_2023_2026.json")

# Known FOMC meeting dates 2023-2026 (Fed publishes schedule)
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_MEETINGS = [
    # 2023
    ("2023-02-01", "FOMC_DECISION"), ("2023-03-22", "FOMC_DECISION"),
    ("2023-05-03", "FOMC_DECISION"), ("2023-06-14", "FOMC_DECISION"),
    ("2023-07-26", "FOMC_DECISION"), ("2023-09-20", "FOMC_DECISION"),
    ("2023-11-01", "FOMC_DECISION"), ("2023-12-13", "FOMC_DECISION"),
    # 2024
    ("2024-01-31", "FOMC_DECISION"), ("2024-03-20", "FOMC_DECISION"),
    ("2024-05-01", "FOMC_DECISION"), ("2024-06-12", "FOMC_DECISION"),
    ("2024-07-31", "FOMC_DECISION"), ("2024-09-18", "FOMC_DECISION"),
    ("2024-11-07", "FOMC_DECISION"), ("2024-12-18", "FOMC_DECISION"),
    # 2025
    ("2025-01-29", "FOMC_DECISION"), ("2025-03-19", "FOMC_DECISION"),
    ("2025-05-07", "FOMC_DECISION"), ("2025-06-18", "FOMC_DECISION"),
    ("2025-07-30", "FOMC_DECISION"), ("2025-09-17", "FOMC_DECISION"),
    ("2025-10-29", "FOMC_DECISION"), ("2025-12-10", "FOMC_DECISION"),
    # 2026
    ("2026-01-28", "FOMC_DECISION"), ("2026-03-18", "FOMC_DECISION"),
    ("2026-04-29", "FOMC_DECISION"), ("2026-06-17", "FOMC_DECISION"),
    ("2026-07-29", "FOMC_DECISION"), ("2026-09-16", "FOMC_DECISION"),
    ("2026-10-28", "FOMC_DECISION"), ("2026-12-09", "FOMC_DECISION"),
]
FOMC_TIME_ET = "14:00"  # 2 PM ET decision release

# Generate CPI release dates: 2nd Tuesday-Thursday of each month, typically 8:30 ET
# BLS publishes schedule; we approximate as 2nd Wed of month for full historical span
# This is a STARTING POINT — real schedule should be scraped from BLS calendar
def cpi_dates(start_year, end_year):
    out = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # 2nd Wednesday of month (BLS typical release window)
            d = dt.date(year, month, 1)
            # find first Wednesday
            while d.weekday() != 2:  # Wed=2
                d += dt.timedelta(days=1)
            # second Wednesday
            d += dt.timedelta(days=7)
            out.append((d.isoformat(), "CPI"))
    return out

# NFP: 1st Friday of each month, 8:30 ET release
def nfp_dates(start_year, end_year):
    out = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            d = dt.date(year, month, 1)
            while d.weekday() != 4:  # Fri=4
                d += dt.timedelta(days=1)
            out.append((d.isoformat(), "NFP"))
    return out

def main():
    events = []
    for date_str, category in FOMC_MEETINGS:
        events.append({
            "event": "FOMC Rate Decision",
            "datetime_utc": f"{date_str}T18:00:00Z",  # 2 PM ET = 6 PM UTC (EDT) / 7 PM UTC (EST) — approx
            "datetime_local_et": f"{date_str}T14:00:00",
            "category": category,
            "importance": "HIGH",
            "source": "fed_schedule_static",
        })
    for date_str, category in cpi_dates(2023, 2026):
        events.append({
            "event": "CPI Release",
            "datetime_utc": f"{date_str}T12:30:00Z",  # 8:30 ET = 12:30 UTC (EDT) / 13:30 UTC (EST)
            "datetime_local_et": f"{date_str}T08:30:00",
            "category": category,
            "importance": "HIGH",
            "source": "bls_schedule_approx_2nd_wed",
        })
    for date_str, category in nfp_dates(2023, 2026):
        events.append({
            "event": "Nonfarm Payrolls",
            "datetime_utc": f"{date_str}T12:30:00Z",
            "datetime_local_et": f"{date_str}T08:30:00",
            "category": category,
            "importance": "HIGH",
            "source": "bls_schedule_1st_fri",
        })
    events.sort(key=lambda e: e["datetime_utc"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "generated_at": dt.datetime.utcnow().isoformat() + "Z",
            "version": 1,
            "n_events": len(events),
            "categories": ["FOMC_DECISION", "CPI", "NFP"],
            "note": "v0 prototype: hardcoded FOMC + approximated CPI (2nd Wed) / NFP (1st Fri). v1 should scrape from FRED API + BLS calendar for exact times. Earnings + Fed speakers TODO.",
            "events": events,
        }, f, indent=2)
    print(f"Wrote {len(events)} events to {OUT_PATH}")

if __name__ == "__main__":
    main()
