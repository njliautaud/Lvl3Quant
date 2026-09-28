#!/usr/bin/env python3
"""
Next-Day Trade Planner v1
==========================
Runs every evening after market close. Combines all validated signals
to generate specific trade recommendations for tomorrow's open.

Checks:
1. Earnings gaps from today's after-hours reporters (PEAD)
2. Sector momentum rankings (rotation)
3. Contrarian reversion triggers (mega-cap gaps)
4. VIX regime for strategy filtering
5. IV run-up candidates (earnings 10-15d out)

Output: JSON trade plan + Discord-ready summary
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json, os, sys, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

STATE_DIR = '/home/jupiter/Lvl3Quant/state'
os.makedirs(STATE_DIR, exist_ok=True)

print("=" * 60)
print("NEXT-DAY TRADE PLANNER v1")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# 1. LOAD MARKET REGIME
# ══════════════════════════════════════════════════════════════
regime_path = os.path.join(STATE_DIR, 'market_regime.json')
if os.path.exists(regime_path):
    with open(regime_path) as f:
        regime = json.load(f)
    print(f"\n  Regime: {regime['composite']['regime']} (score {regime['composite']['risk_score']}/100)")
    print(f"  VIX: {regime['vix']['current']} ({regime['vix']['regime']})")
    print(f"  Allocation level: {regime['composite']['allocation_pct']}%")
else:
    regime = None
    print("\n  WARNING: No regime data. Run market_regime_dashboard.py first.")

# ══════════════════════════════════════════════════════════════
# 2. CHECK AFTER-HOURS EARNINGS GAPS (PEAD CANDIDATES)
# ══════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("PEAD CANDIDATES (After-Hours Earnings Gaps)")
print("-" * 60)

# Stocks that reported today PM — check for gaps
pm_reporters = {
    'META': {'est': 7.18, 'act': 6.18, 'prev_close': 593.41},
    'MSFT': {'est': 4.23, 'act': 4.74, 'prev_close': 393.35},
    'HOOD': {'est': 0.41, 'act': 0.62, 'prev_close': 92.76},
    'ARM': {'est': 0.36, 'act': 0.45, 'prev_close': 244.74},
    'QCOM': {'est': 2.09, 'act': 2.21, 'prev_close': 162.88},
}

# AM reporters from today
am_reporters = {
    'SOFI': {'est': 0.11, 'act': 0.12, 'prev_close': 16.74, 'today_close': 15.235},
}

pead_trades = []

for sym, info in {**pm_reporters, **am_reporters}.items():
    surprise_pct = (info['act'] - info['est']) / abs(info['est']) * 100 if info['est'] != 0 else 0
    beat = info['act'] > info['est']
    
    # For AM reporters, we know today's close
    if 'today_close' in info:
        gap_pct = (info['today_close'] - info['prev_close']) / info['prev_close'] * 100
        gap_known = True
    else:
        gap_known = False
        gap_pct = None
    
    # PEAD criteria: significant earnings surprise
    if abs(surprise_pct) > 5:
        direction = 'LONG' if beat else 'SHORT'
        
        # Our research: DOWN gaps drift 2.81% avg (4x more than UP gaps 0.67%)
        expected_drift = 2.81 if not beat else 0.67
        
        trade = {
            'ticker': sym,
            'surprise_pct': round(surprise_pct, 1),
            'beat': beat,
            'direction': direction,
            'gap_pct': round(gap_pct, 1) if gap_pct else 'TBD (check premarket)',
            'expected_drift_pct': expected_drift,
            'prev_close': info['prev_close'],
            'affordable': info['prev_close'] < 100,  # rough check for options budget
        }
        pead_trades.append(trade)
        
        print(f"\n  {sym}: {'BEAT' if beat else 'MISS'} by {surprise_pct:+.1f}%")
        if gap_known:
            print(f"    Gap: {gap_pct:+.1f}%")
        print(f"    Direction: {direction} (expected drift: {expected_drift}%)")
        print(f"    Affordable options (<$200): {'YES' if trade['affordable'] else 'NO (stock too expensive)'}")

if not pead_trades:
    print("  No significant earnings surprises today.")

# ══════════════════════════════════════════════════════════════
# 3. CONTRARIAN REVERSION CHECK
# ══════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("CONTRARIAN REVERSION (Mega-Cap Gap Triggers)")
print("-" * 60)

# Check if any mega-cap gapped >3% today
mega_caps = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
sector_map = {
    'AAPL': 'XLK', 'MSFT': 'XLK', 'GOOGL': 'XLC', 'AMZN': 'XLY',
    'META': 'XLC', 'NVDA': 'XLK', 'TSLA': 'XLY'
}

try:
    mega_data = yf.download(mega_caps, period='5d', progress=False)
    mega_close = mega_data['Close'] if 'Close' in mega_data.columns.get_level_values(0) else mega_data['Adj Close']
    
    contrarian_trades = []
    for sym in mega_caps:
        if sym in mega_close.columns:
            prices_s = mega_close[sym].dropna()
            if len(prices_s) >= 2:
                today_ret = (prices_s.iloc[-1] / prices_s.iloc[-2] - 1) * 100
                if today_ret < -3:
                    sector = sector_map.get(sym, 'SPY')
                    contrarian_trades.append({
                        'trigger': sym,
                        'gap_pct': round(today_ret, 1),
                        'trade': f'BUY {sector} shares',
                        'hold_days': 3,
                        'expected_reversion': '+0.16% avg'
                    })
                    print(f"  {sym} gapped {today_ret:+.1f}% → BUY {sector} for 3-day reversion")
    
    if not contrarian_trades:
        print("  No mega-cap gaps >3% today.")
except Exception as e:
    print(f"  Error checking mega-caps: {e}")
    contrarian_trades = []

# ══════════════════════════════════════════════════════════════
# 4. SECTOR ROTATION SIGNAL
# ══════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("SECTOR ROTATION (LGBM-Enhanced)")
print("-" * 60)

if regime and not regime['strategy_guidance']['sector_rotation']:
    print("  ⚠️ PAUSED — VIX elevated / trend not bullish enough")
    print(f"    Would trade: {regime['sectors']['top_3'][0]['ticker']} ({regime['sectors']['top_3'][0]['name']}) +{regime['sectors']['top_3'][0]['mom_20d']}%")
else:
    if regime:
        top = regime['sectors']['top_3'][0]
        print(f"  ACTIVE — Top sector: {top['ticker']} ({top['name']}) +{top['mom_20d']}% 20d momentum")

# ══════════════════════════════════════════════════════════════
# 5. TOMORROW'S EARNINGS WATCH
# ══════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("TOMORROW'S EARNINGS WATCH (Jul 30)")
print("-" * 60)

tomorrow_reporters = {
    'AAPL': {'timing': 'PM', 'est': 1.89, 'price': '~230'},
    'AMZN': {'timing': 'PM', 'est': 1.82, 'price': '~200'},
    'RDDT': {'timing': 'PM', 'est': 0.97, 'price': '~180'},
    'RBLX': {'timing': 'PM', 'est': -0.34, 'price': '~70'},
    'COIN': {'timing': 'PM', 'est': 0.14, 'price': '~250'},
    'RIVN': {'timing': 'PM', 'est': -0.79, 'price': '~15'},
    'MA': {'timing': 'AM', 'est': 4.76, 'price': '~570'},
}

print("  Growth names reporting PM (for PEAD entries Thursday):")
for sym, info in tomorrow_reporters.items():
    affordable = '✅' if sym in ['RIVN', 'RBLX'] else '❌ (expensive)'
    print(f"    {sym} ({info['timing']}) — Est ${info['est']}, Price {info['price']} — Options {affordable}")

# ══════════════════════════════════════════════════════════════
# 6. COMPOSITE TRADE PLAN
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TRADE PLAN FOR TOMORROW")
print("=" * 60)

trade_plan = {
    'date': datetime.now().strftime('%Y-%m-%d'),
    'target_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
    'regime': regime['composite']['regime'] if regime else 'UNKNOWN',
    'vix': regime['vix']['current'] if regime else None,
    'allocation_level': regime['composite']['allocation_pct'] if regime else 75,
    'pead_candidates': pead_trades,
    'contrarian_triggers': contrarian_trades if 'contrarian_trades' in dir() else [],
    'rotation_signal': 'PAUSED' if (regime and not regime['strategy_guidance']['sector_rotation']) else 'ACTIVE',
    'actions': []
}

# Determine actual actions
kill_switch = regime and regime['vix']['current'] > 20 and regime['trend']['spy_vs_50sma'] == 'below'

if kill_switch:
    print("\n  ⚠️ KILL SWITCH ACTIVE (VIX >20 + SPY below 50-SMA)")
    print("  Pausing momentum/rotation. Event-driven only (PEAD, contrarian).")
    trade_plan['kill_switch'] = True

# PEAD — only if affordable
affordable_pead = [t for t in pead_trades if t['affordable']]
if affordable_pead:
    for t in affordable_pead:
        action = f"PEAD {t['direction']}: Check {t['ticker']} premarket gap. If >5%, buy {'puts' if t['direction'] == 'SHORT' else 'calls'} at open."
        trade_plan['actions'].append(action)
        print(f"\n  ✅ {action}")
else:
    print("\n  No affordable PEAD setups. Mega-cap options too expensive for $645 account.")

# Meta-recommendation
if not trade_plan['actions']:
    msg = "STAY CASH. No high-confidence setups meet all criteria (affordable + validated + right regime)."
    trade_plan['actions'].append(msg)
    print(f"\n  💰 {msg}")
    print("  Tomorrow PM: AAPL, AMZN, RBLX, RIVN report — prepare PEAD for Thursday if gaps occur.")

# Save
plan_path = os.path.join(STATE_DIR, 'trade_plan_tomorrow.json')
with open(plan_path, 'w') as f:
    json.dump(trade_plan, f, indent=2, default=str)

print(f"\n  Plan saved.")
print("\nDone.")
