#!/bin/bash
# Earnings gap check for Jul 16 AM reporters (UNH, ISRG, MS)
# Fires at 9:35 AM ET Jul 16
set -u

LOG="/home/jupiter/Lvl3Quant/logs/earnings_gap_jul16.log"
echo "=== Earnings Gap Check $(date) ===" >> "$LOG"

python3 -u - << 'PYTHON' 2>&1 | tee -a "$LOG"
import yfinance as yf
import json

# AM reporters Jul 16
tickers = {"UNH": 419, "ISRG": 389, "MS": 229}
gaps = {}

for sym, prev_close in tickers.items():
    try:
        data = yf.download(sym, period="2d", progress=False)
        if len(data) >= 1:
            current = float(data['Open'].iloc[-1])
            gap_pct = (current - prev_close) / prev_close * 100
            gaps[sym] = {"prev_close": prev_close, "open": current, "gap_pct": gap_pct}
            flag = " *** 10%+ GAP ***" if abs(gap_pct) >= 10 else ""
            print(f"  {sym}: prev ${prev_close} -> open ${current:.2f} = {gap_pct:+.1f}%{flag}")
    except Exception as e:
        print(f"  {sym}: ERROR - {e}")

# Report
big_gaps = {s: g for s, g in gaps.items() if abs(g["gap_pct"]) >= 10}
if big_gaps:
    print(f"\n*** {len(big_gaps)} STOCKS WITH 10%+ GAP - TRADE SIGNAL ***")
    for s, g in big_gaps.items():
        direction = "UP" if g["gap_pct"] > 0 else "DOWN"
        print(f"  {s}: {direction} {abs(g['gap_pct']):.1f}% - BUY signal per validated strategy")
else:
    print(f"\nNo 10%+ gaps. No trade today.")
PYTHON

# Inject into Claude session
/home/jupiter/Lvl3Quant/scripts/autonomy_inject.sh "EARNINGS_GAP_CHECK Jul 16 AM results: $(tail -10 $LOG). If any 10%+ gaps found, alert user on Discord and look up option chains. Max budget $110 per trade."
