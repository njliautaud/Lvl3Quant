#!/usr/bin/env python3
"""
Daily Stock Prediction Pipeline
==================================
Combined runner for PM2 scheduling.
Runs the scanner first, then the paper trading engine.

Scheduled to run at 4:30 PM ET on weekdays.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from daily_scanner import run_scan
from options_paper_engine import run_daily


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    weekday = datetime.now().weekday()

    print(f"[PIPELINE] Stock Prediction Pipeline — {today}")
    print(f"[PIPELINE] Day: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][weekday]}")

    # Skip weekends
    if weekday >= 5:
        print(f"[PIPELINE] Weekend — skipping.")
        return

    # Step 1: Run scanner
    print(f"\n{'#'*70}")
    print(f"# STEP 1: DAILY SCANNER")
    print(f"{'#'*70}")
    t0 = time.time()
    try:
        signals = run_scan(today)
        print(f"[PIPELINE] Scanner completed in {time.time() - t0:.0f}s — {len(signals)} signals")
    except Exception as e:
        print(f"[PIPELINE] Scanner FAILED: {e}")
        import traceback
        traceback.print_exc()
        signals = []

    # Brief pause
    time.sleep(5)

    # Step 2: Run paper engine
    print(f"\n{'#'*70}")
    print(f"# STEP 2: PAPER TRADING ENGINE")
    print(f"{'#'*70}")
    t1 = time.time()
    try:
        state, trades = run_daily(today)
        print(f"[PIPELINE] Paper engine completed in {time.time() - t1:.0f}s")
    except Exception as e:
        print(f"[PIPELINE] Paper engine FAILED: {e}")
        import traceback
        traceback.print_exc()

    total_time = time.time() - t0
    print(f"\n[PIPELINE] Total runtime: {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"[PIPELINE] Done.")


if __name__ == "__main__":
    main()
