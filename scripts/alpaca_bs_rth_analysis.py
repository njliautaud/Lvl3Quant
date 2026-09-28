#!/usr/bin/env python3
"""
Alpaca vs BS Pricing — RTH-Only Analysis
==========================================
Filters the alpaca_vs_bs_pricing.jsonl log to ONLY include entries
during Regular Trading Hours (9:30-16:00 ET) on weekdays.

This resolves the item #134 question: Sunday data showed BS underprices
by ~41% vs bid, but weekend bid-ask spreads are abnormally wide.

Run: python3 scripts/alpaca_bs_rth_analysis.py
Output: output/alpaca_bs_rth_analysis/
"""

import json
import os
import warnings
from collections import defaultdict
from datetime import datetime, time
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
LOG_FILE = ROOT / "logs" / "alpaca_vs_bs_pricing.jsonl"
OUTPUT = ROOT / "output" / "alpaca_bs_rth_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

RTH_START = time(9, 30)
RTH_END = time(16, 0)
WEEKDAYS = {0, 1, 2, 3, 4}  # Mon-Fri


def is_rth(ts_str):
    """Check if timestamp falls within RTH on a weekday."""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.weekday() in WEEKDAYS and RTH_START <= dt.time() <= RTH_END
    except:
        return False


def main():
    if not LOG_FILE.exists():
        print("ERROR: No pricing log file found")
        return

    # Collect data
    rth_entries = []
    non_rth_entries = []

    with open(LOG_FILE) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                bs = d.get('bs_price', 0)
                mid = d.get('alpaca_mid', 0)
                bid = d.get('alpaca_bid', 0)
                ask = d.get('alpaca_ask', 0)
                ts = d.get('ts', '')

                if bs < 0.05 or mid < 0.05 or bid <= 0:
                    continue

                entry = {
                    'ticker': d.get('ticker', '?'),
                    'bs': bs, 'mid': mid, 'bid': bid, 'ask': ask,
                    'delta': abs(d.get('alpaca_delta', 0)),
                    'gap_vs_mid_pct': (bs / mid - 1) * 100,
                    'gap_vs_bid_pct': (bs / bid - 1) * 100,
                    'spread_pct': (ask - bid) / mid * 100 if mid > 0 else 0,
                    'ts': ts,
                }

                if is_rth(ts):
                    rth_entries.append(entry)
                else:
                    non_rth_entries.append(entry)
            except:
                continue

    print("=" * 70)
    print("ALPACA vs BS PRICING — RTH vs NON-RTH COMPARISON")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    print("=" * 70)

    for label, entries in [("RTH (9:30-16:00 weekday)", rth_entries),
                           ("NON-RTH (overnight/weekend)", non_rth_entries)]:
        if not entries:
            print(f"\n{label}: NO DATA YET")
            continue

        gaps_mid = [e['gap_vs_mid_pct'] for e in entries]
        gaps_bid = [e['gap_vs_bid_pct'] for e in entries]
        spreads = [e['spread_pct'] for e in entries]

        print(f"\n{label} ({len(entries):,} comparisons)")
        print("-" * 50)
        print(f"  BS vs Mid:  mean {np.mean(gaps_mid):+.1f}%, median {np.median(gaps_mid):+.1f}%")
        print(f"  BS vs Bid:  mean {np.mean(gaps_bid):+.1f}%, median {np.median(gaps_bid):+.1f}%")
        print(f"  Bid-Ask:    mean {np.mean(spreads):.1f}%, median {np.median(spreads):.1f}%")

        # By delta bucket
        buckets = {
            'OTM (10-25δ)': [e for e in entries if 0.10 <= e['delta'] < 0.25],
            'ATM-ish (25-40δ)': [e for e in entries if 0.25 <= e['delta'] < 0.40],
        }
        for bname, bentries in buckets.items():
            if bentries:
                bg = [e['gap_vs_bid_pct'] for e in bentries]
                bs = [e['spread_pct'] for e in bentries]
                print(f"  {bname}: BS vs bid {np.median(bg):+.1f}%, spread {np.median(bs):.1f}% (n={len(bentries)})")

    # Verdict
    if rth_entries:
        rth_bid_gap = np.median([e['gap_vs_bid_pct'] for e in rth_entries])
        rth_spread = np.median([e['spread_pct'] for e in rth_entries])

        print(f"\n{'=' * 70}")
        print("VERDICT (RTH data only)")
        print(f"{'=' * 70}")

        if rth_bid_gap > 10:
            print(f"  BS OVERESTIMATES what we'd receive by {rth_bid_gap:.0f}%")
            print(f"  Backtests INFLATED → apply ~{rth_bid_gap:.0f}% haircut to CAGR estimates")
        elif rth_bid_gap < -10:
            print(f"  BS UNDERESTIMATES what we'd receive by {abs(rth_bid_gap):.0f}%")
            print(f"  Backtests CONSERVATIVE → real P&L would be higher")
        else:
            print(f"  BS is within ~10% of real bid — backtests are REASONABLE")
        print(f"  Typical bid-ask spread: {rth_spread:.1f}% of mid")
    else:
        print(f"\n⚠️  NO RTH DATA YET — run this after Monday market hours (9:30+ ET)")
        print(f"  Current data is {len(non_rth_entries):,} non-RTH entries (weekend/overnight)")

    # Save
    results = {
        "generated": datetime.now().isoformat(),
        "rth_count": len(rth_entries),
        "non_rth_count": len(non_rth_entries),
    }
    if rth_entries:
        results["rth_bs_vs_bid_median"] = float(np.median([e['gap_vs_bid_pct'] for e in rth_entries]))
        results["rth_spread_median"] = float(np.median([e['spread_pct'] for e in rth_entries]))

    with open(OUTPUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUTPUT}/")


if __name__ == "__main__":
    main()
