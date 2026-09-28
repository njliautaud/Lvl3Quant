#!/usr/bin/env python3
"""
Unified Growth Scanner Runner
===============================
Runs all growth scanners and produces a consolidated report.

Usage:
  python growth/run_all_scanners.py [--quick]  # quick = momentum + trend only
  python growth/run_all_scanners.py             # full = all 4 scanners
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from momentum_scanner import run_scanner as run_momentum
from trend_scanner import run_scanner as run_trend


def run_all(quick: bool = False):
    print(f"\n{'#'*70}")
    print(f"  GROWTH SCANNER SUITE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Mode: {'QUICK (momentum + trend)' if quick else 'FULL (all scanners)'}")
    print(f"{'#'*70}\n")

    # Always run these (fast)
    print("\n" + "="*70)
    print("  [1/4] MOMENTUM SCANNER")
    print("="*70)
    mom_results = run_momentum(top_n=20)

    print("\n" + "="*70)
    print("  [2/4] TREND SCANNER")
    print("="*70)
    trend_results = run_trend()

    if not quick:
        # These are slower (individual ticker API calls)
        print("\n" + "="*70)
        print("  [3/4] LEAPS SCREENER")
        print("="*70)
        from leaps_screener import run_screener as run_leaps
        leaps_results = run_leaps(max_cost=500)

        print("\n" + "="*70)
        print("  [4/4] FACTOR SCANNER")
        print("="*70)
        # Factor scanner is the slowest (fetches fundamentals per ticker)
        # Skip earnings momentum — only useful during earnings season
        from factor_scanner import run_scanner as run_factor
        factor_results = run_factor(top_n=20)

    print(f"\n{'#'*70}")
    print(f"  ALL SCANS COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick mode: momentum + trend only")
    args = parser.parse_args()
    run_all(quick=args.quick)
