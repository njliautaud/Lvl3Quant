#!/usr/bin/env python3
"""
NFLX Earnings Gap Checker — Run at 9:35 AM ET Jul 17, 2026
Data backing: Earnings Gap Buyer v1 — 10%+ gaps: Sharpe 5.55, WR 63%, PF 2.61, perm p=0.000

Usage: python3 nflx_earnings_gap_check.py
  Checks gap, prints recommendation, does NOT auto-execute.
"""

import yfinance as yf
import datetime
import sys

def check_nflx_gap():
    nflx = yf.Ticker('NFLX')
    hist = nflx.history(period='5d', interval='1d')

    if len(hist) < 2:
        print("ERROR: Not enough price data")
        return None

    # Previous close (Jul 16)
    prev_close = hist['Close'].iloc[-2]
    # Today's open (Jul 17)
    today_open = hist['Open'].iloc[-1]
    today_date = hist.index[-1].strftime('%Y-%m-%d')

    gap_pct = ((today_open - prev_close) / prev_close) * 100

    print(f"=" * 60)
    print(f"NFLX EARNINGS GAP CHECK — {today_date}")
    print(f"=" * 60)
    print(f"  Previous Close (Jul 16): ${prev_close:.2f}")
    print(f"  Today's Open   (Jul 17): ${today_open:.2f}")
    print(f"  Gap: {gap_pct:+.2f}%")
    print()

    if abs(gap_pct) >= 10:
        direction = "UP" if gap_pct > 0 else "DOWN"
        print(f"  ✅ SIGNAL: Gap {direction} {abs(gap_pct):.1f}% — ABOVE 10% threshold")
        print()
        print(f"  DATA BACKING:")
        print(f"    - Earnings Gap Buyer v1: Sharpe 5.55, WR 63.2%, PF 2.61")
        print(f"    - 57 trades backtested (2019-2026)")
        print(f"    - Permutation p=0.000 (REAL signal)")
        print(f"    - Average return: +1.53% per trade")
        print()
        print(f"  TRADE PLAN:")
        if gap_pct > 0:
            atm = round(today_open)
            print(f"    - BUY call debit spread")
            print(f"    - Lower strike: ~${atm} (near ATM)")
            print(f"    - Upper strike: ~${atm + 5}")
            print(f"    - Expiration: Jul 24")
            print(f"    - Max budget: $110")
        else:
            atm = round(today_open)
            print(f"    - BUY put debit spread")
            print(f"    - Upper strike: ~${atm} (near ATM)")
            print(f"    - Lower strike: ~${atm - 5}")
            print(f"    - Expiration: Jul 24")
            print(f"    - Max budget: $110")
        print(f"    - Exit: By close Jul 18 (1-day) or Jul 21 (2-day)")
        print()
        print(f"  ⚠️  RISK NOTE: Fails R1 (regime-dependent). Individual play, not systematic.")
        return {"signal": True, "direction": direction, "gap_pct": gap_pct, "open": today_open}
    else:
        print(f"  ❌ NO SIGNAL: Gap {abs(gap_pct):.1f}% — BELOW 10% threshold")
        print(f"  Action: SKIP this trade")
        return {"signal": False, "gap_pct": gap_pct}

if __name__ == "__main__":
    result = check_nflx_gap()
    if result and result.get("signal"):
        print("\n  >>> PROCEED TO ROBINHOOD MCP TO EXECUTE <<<")
    else:
        print("\n  >>> NO TRADE — WAIT FOR INTC (Jul 24) <<<")
