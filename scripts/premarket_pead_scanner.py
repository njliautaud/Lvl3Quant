#!/usr/bin/env python3
"""
Premarket PEAD Scanner v1
==========================
Run at 9:00-9:25 AM ET before market open.
Checks premarket prices for stocks that reported earnings yesterday.
If gap > threshold, generates specific option trade recommendations.

Uses Robinhood quotes for premarket data (must be run when RH MCP available).
Otherwise uses yfinance pre/post market data.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json, os, sys, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

STATE_DIR = '/home/jupiter/Lvl3Quant/state'

print("=" * 60)
print("PREMARKET PEAD SCANNER v1")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# 1. LOAD TRADE PLAN
# ══════════════════════════════════════════════════════════════
plan_path = os.path.join(STATE_DIR, 'trade_plan_tomorrow.json')
if os.path.exists(plan_path):
    with open(plan_path) as f:
        plan = json.load(f)
    print(f"\n  Loaded trade plan from: {plan.get('date', 'unknown')}")
    print(f"  Regime: {plan.get('regime', 'unknown')}")
    print(f"  Kill switch: {'YES' if plan.get('kill_switch') else 'NO'}")
else:
    print("  No trade plan found. Run next_day_trade_planner.py first.")
    sys.exit(0)

# ══════════════════════════════════════════════════════════════
# 2. CHECK PREMARKET PRICES
# ══════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("PREMARKET GAP CHECK")
print("-" * 60)

pead_candidates = plan.get('pead_candidates', [])
if not pead_candidates:
    print("  No PEAD candidates in today's plan.")
    sys.exit(0)

# Get current quotes via yfinance
tickers = [c['ticker'] for c in pead_candidates]
try:
    data = yf.download(tickers, period='5d', progress=False, prepost=True)
    if len(tickers) == 1:
        closes = pd.DataFrame({tickers[0]: data['Close']})
    else:
        closes = data['Close']
except Exception as e:
    print(f"  Error downloading: {e}")
    sys.exit(1)

actionable = []

for candidate in pead_candidates:
    sym = candidate['ticker']
    prev_close = candidate['prev_close']
    direction = candidate['direction']
    
    if sym not in closes.columns:
        print(f"\n  {sym}: No data available")
        continue
    
    current = closes[sym].dropna().iloc[-1]
    gap_pct = (current - prev_close) / prev_close * 100
    
    # PEAD criteria
    gap_threshold = 3.0  # minimum gap to trade
    
    is_gap_direction_match = (gap_pct > 0 and direction == 'LONG') or (gap_pct < 0 and direction == 'SHORT')
    is_significant = abs(gap_pct) >= gap_threshold
    is_affordable = candidate.get('affordable', False)
    
    status = '✅ ACTIONABLE' if (is_gap_direction_match and is_significant and is_affordable) else '❌ SKIP'
    
    print(f"\n  {sym}: {status}")
    print(f"    Prev close: ${prev_close:.2f}")
    print(f"    Current: ${current:.2f}")
    print(f"    Gap: {gap_pct:+.1f}%")
    print(f"    Surprise: {candidate['surprise_pct']:+.1f}%")
    print(f"    Direction match: {'YES' if is_gap_direction_match else 'NO'}")
    print(f"    Gap significant (>{gap_threshold}%): {'YES' if is_significant else 'NO'}")
    print(f"    Affordable: {'YES' if is_affordable else 'NO'}")
    
    if status == '✅ ACTIONABLE':
        # Generate specific trade recommendation
        option_type = 'call' if direction == 'LONG' else 'put'
        
        # Strike selection: ATM or slightly OTM
        if option_type == 'call':
            strike = round(current * 1.02, 0)  # 2% OTM
        else:
            strike = round(current * 0.98, 0)  # 2% OTM
        
        # Expiry: 2-3 weeks out
        expiry_target = datetime.now() + timedelta(days=14)
        # Round to next Friday
        days_to_friday = (4 - expiry_target.weekday()) % 7
        expiry = expiry_target + timedelta(days=days_to_friday)
        
        trade = {
            'ticker': sym,
            'action': f'BUY {option_type.upper()}',
            'strike': strike,
            'expiry': expiry.strftime('%Y-%m-%d'),
            'direction': direction,
            'gap_pct': round(gap_pct, 1),
            'surprise_pct': candidate['surprise_pct'],
            'expected_drift': candidate['expected_drift_pct'],
            'max_cost': 200,
            'exit_rules': {
                'take_profit': '+30%',
                'stop_loss': '-25%',
                'max_hold': '5 trading days',
                'trailing_stop': '50% giveback after +15%'
            }
        }
        actionable.append(trade)
        
        print(f"    >>> TRADE: BUY {sym} ${strike} {option_type} exp {expiry.strftime('%Y-%m-%d')}")
        print(f"    >>> Max cost: $200, TP: +30%, SL: -25%, 5-day max hold")

# ══════════════════════════════════════════════════════════════
# 3. SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if actionable:
    print(f"\n  {len(actionable)} actionable trade(s):")
    for t in actionable:
        print(f"    {t['action']} {t['ticker']} ${t['strike']} {t['expiry']}")
    
    # Save actionable trades
    scan_path = os.path.join(STATE_DIR, 'premarket_pead_scan.json')
    with open(scan_path, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'trades': actionable}, f, indent=2)
    print(f"\n  Trades saved.")
else:
    print("\n  No actionable PEAD trades today.")
    print("  Reasons: gaps too small, wrong direction, or options too expensive.")

print("\nDone.")
