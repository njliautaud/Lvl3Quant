#!/usr/bin/env python3
"""Compare Strategy B-MA across ES, SPY IBKR, SPY free-commission."""
import sys, json, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from high_conviction_strategy import (
    load_aligned_data, precompute_signals, simulate_trades,
    get_precomputed, compute_metrics,
    HOLD_PERIODS, COST_STRUCTURES
)

print(f"{time.strftime('%H:%M:%S')} Loading data...")
days = load_aligned_data()
print(f"{time.strftime('%H:%M:%S')} Pre-computing signals...")
precompute_signals(days)

# Best config from robustness: vol=60, conv=0.75, 30min, morning_afternoon
vol_pct = 60
conv = 0.75
hold_bars = HOLD_PERIODS['30min']

for cost_key, cost in COST_STRUCTURES.items():
    all_trades = []
    for day in days[20:]:
        signal, c, n_agree, vol_pred = get_precomputed(day)
        trades = simulate_trades(
            day, signal, c, n_agree, vol_pred,
            hold_bars=hold_bars,
            cost_spread_ticks=cost['spread_ticks'],
            cost_comm_ticks=cost['comm_ticks'],
            conviction_threshold=conv,
            min_agreement=1,
            vol_percentile_min=vol_pct,
            time_filter='morning_afternoon',
        )
        for t in trades:
            t['date'] = day['date']
        all_trades.extend(trades)

    m = compute_metrics(all_trades, label=f'{cost_key}')
    mult = cost['multiplier']
    print(f"\n{'='*60}")
    print(f"VENUE: {cost['label']} (mult={mult}, spread={cost['spread_ticks']}t, comm={cost['comm_ticks']}t)")
    print(f"  Trades: {m['total_trades']}, Days: {m['n_days']}")
    print(f"  Gross: {m['gross_pnl_ticks']:+.1f} ticks")
    print(f"  Net:   {m['net_pnl_ticks']:+.1f} ticks (${m['net_pnl_ticks'] * 12.50:+,.0f} ES-equiv)")
    print(f"  Avg/trade: {m['avg_pnl_per_trade_ticks']:+.3f} ticks")
    print(f"  Win Rate: {m['win_rate']:.1f}%")
    print(f"  Sharpe: {m['sharpe']:.2f}")
    print(f"  Max DD: {m['max_drawdown_ticks']:.1f} ticks")

print(f"\n{time.strftime('%H:%M:%S')} Done.")
