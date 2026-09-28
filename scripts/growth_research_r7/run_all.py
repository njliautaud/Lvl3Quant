#!/usr/bin/env python3
"""
R7 Master Runner: Execute all 5 research directions and produce summary.
"""
import sys
import os
import json
import traceback
from datetime import datetime

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r7'
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research_r7')

results = {}

# Direction 1
print("\n" + "#" * 80)
print("# DIRECTION 1: ML-Driven Dynamic Leverage")
print("#" * 80)
try:
    from dir1_ml_dynamic_leverage import run_direction1
    results['D1'] = run_direction1()
except Exception as e:
    print(f"ERROR in D1: {e}")
    traceback.print_exc()
    results['D1'] = {'error': str(e)}

# Direction 2
print("\n" + "#" * 80)
print("# DIRECTION 2: Carry Trade")
print("#" * 80)
try:
    from dir2_carry_trade import run_direction2
    results['D2'] = run_direction2()
except Exception as e:
    print(f"ERROR in D2: {e}")
    traceback.print_exc()
    results['D2'] = {'error': str(e)}

# Direction 3
print("\n" + "#" * 80)
print("# DIRECTION 3: Commodities Trend Following")
print("#" * 80)
try:
    from dir3_commodities_trend import run_direction3
    results['D3'] = run_direction3()
except Exception as e:
    print(f"ERROR in D3: {e}")
    traceback.print_exc()
    results['D3'] = {'error': str(e)}

# Direction 4
print("\n" + "#" * 80)
print("# DIRECTION 4: Earnings-Driven Stock Selection")
print("#" * 80)
try:
    from dir4_earnings_stock_selection import run_direction4
    results['D4'] = run_direction4()
except Exception as e:
    print(f"ERROR in D4: {e}")
    traceback.print_exc()
    results['D4'] = {'error': str(e)}

# Direction 5
print("\n" + "#" * 80)
print("# DIRECTION 5: Portfolio Insurance Timing")
print("#" * 80)
try:
    from dir5_portfolio_insurance_timing import run_direction5
    results['D5'] = run_direction5()
except Exception as e:
    print(f"ERROR in D5: {e}")
    traceback.print_exc()
    results['D5'] = {'error': str(e)}

# Summary
print("\n" + "=" * 80)
print("GRAND SUMMARY — R7 Growth Research")
print("=" * 80)
print(f"Benchmark: 3x Risk Parity = 21.7% CAGR\n")

for key in ['D1', 'D2', 'D3', 'D4', 'D5']:
    r = results.get(key, {})
    if 'error' in r:
        print(f"{key}: ERROR — {r['error']}")
        continue
    assessment = r.get('honest_assessment', 'No assessment')
    print(f"{key} ({r.get('direction', key)}):")
    print(f"  {assessment}")
    print()

# Save grand summary
with open(os.path.join(OUT_DIR, 'r7_grand_summary.json'), 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'benchmark_cagr': 0.217,
        'results': {k: v.get('honest_assessment', str(v)) for k, v in results.items()},
    }, f, indent=2, default=str)

print(f"Grand summary saved to {OUT_DIR}/r7_grand_summary.json")
