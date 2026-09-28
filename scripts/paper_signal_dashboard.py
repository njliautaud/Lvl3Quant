#!/usr/bin/env python3
"""
Paper Signal Dashboard
========================
Reads all paper engine state files and produces a consolidated view.
Run anytime to see current signals across all 13 paper engines.
"""

import json
from pathlib import Path
from datetime import datetime

OUTPUT_ROOT = Path("/home/jupiter/Lvl3Quant/output")

# Define all paper engines and their state locations
ENGINES = {
    # Growth strategies
    'Gold/Silver Ratio': 'ml_gold_silver_paper/state.json',
    'Yield Curve Trade': 'ml_yield_curve_paper/state.json',
    'Bond Duration': 'ml_bond_duration_paper/state.json',
    'Currency Carry': 'ml_currency_carry_paper/state.json',
    'Tail Risk Hedge': 'ml_tail_risk_paper/state.json',
    'Stat Arb Pairs': 'ml_stat_arb_paper/state.json',
    'Carry+Momentum': 'ml_carry_momentum_paper/state.json',
    'CTA Trend': 'ml_trend_paper/state.json',
    'Sector Rotation': 'ml_sector_paper/state.json',
    'Commodity Trend': 'commodity_trend_paper/state.json',
    'Vol Breakout': 'vol_breakout_paper/state.json',
    # Income strategies (wheel engines)
    'Earnings Vol': 'earnings_vol_paper/state.json',
    'Strangle': 'strangle_paper/state.json',
}

print(f"{'='*70}")
print(f"  PAPER ENGINE SIGNAL DASHBOARD — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
print(f"{'='*70}")
print()

active = 0
total = 0

for name, state_path in ENGINES.items():
    total += 1
    full_path = OUTPUT_ROOT / state_path

    if not full_path.exists():
        print(f"  {name:25s} | NO STATE FILE")
        continue

    try:
        with open(full_path) as f:
            state = json.load(f)

        # Extract signal based on different state formats
        signal = '?'
        confidence = '?'
        date = '?'

        # Format 1: position/signal_date (gold/silver style)
        if 'position' in state:
            signal = state.get('position', '?')
            confidence = f"{state.get('probability', 0):.0%}" if isinstance(state.get('probability'), (int, float)) else '?'
            date = state.get('signal_date', state.get('updated', '?'))

        # Format 2: current_allocation (carry momentum style)
        elif 'current_allocation' in state:
            alloc = state['current_allocation']
            signal = alloc.get('category', '?').upper()
            conf_val = alloc.get('confidence', 0)
            confidence = f"{conf_val:.0%}" if isinstance(conf_val, (int, float)) else '?'
            date = alloc.get('date', '?')

        # Format 3: signal/prediction field
        elif 'signal' in state:
            signal = state.get('signal', '?')
            confidence = f"{state.get('confidence', 0):.0%}" if isinstance(state.get('confidence'), (int, float)) else '?'
            date = state.get('date', state.get('last_run', '?'))

        # Format 4: allocation field
        elif 'allocation' in state:
            signal = state.get('allocation', '?')
            confidence = f"{state.get('confidence', 0):.0%}" if isinstance(state.get('confidence'), (int, float)) else '?'
            date = state.get('date', '?')

        # Generic fallback
        else:
            # Try to find any useful info
            for key in ['action', 'recommendation', 'trade', 'direction']:
                if key in state:
                    signal = state[key]
                    break
            date = state.get('last_run', state.get('date', state.get('updated', '?')))

        active += 1
        # Truncate signal if too long
        signal_str = str(signal)[:25]
        print(f"  {name:25s} | {signal_str:25s} | conf: {str(confidence):6s} | {str(date)[:10]}")

    except Exception as e:
        print(f"  {name:25s} | ERROR: {str(e)[:40]}")

print()
print(f"  Active: {active}/{total} engines have state files")
print(f"  Next fire: Monday 4:45pm ET (cron weekdays)")
print(f"{'='*70}")
