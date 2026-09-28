#!/usr/bin/env python3
"""Run Strategy B with morning_afternoon time restriction comparison."""
import sys, json, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from high_conviction_strategy import (
    load_aligned_data, precompute_signals, simulate_trades,
    get_precomputed, compute_metrics,
    HOLD_PERIODS, COST_STRUCTURES, TICK_VAL
)

print(f"{time.strftime('%H:%M:%S')} Loading data...")
days = load_aligned_data()
print(f"{time.strftime('%H:%M:%S')} Pre-computing signals...")
precompute_signals(days)  # modifies in-place

# Strategy B parameters (from corrected results)
vol_pct = 60
conviction = 0.5
hold_key = '30min'
hold_bars = HOLD_PERIODS[hold_key]
cost = COST_STRUCTURES['ES_futures']

results = {}
for tf_name in ['none', 'morning_afternoon', 'no_lunch', 'first_hour']:
    all_trades = []
    for day in days[20:]:  # skip first 20 for param tuning
        signal, conv, n_agree, vol_pred = get_precomputed(day)
        trades = simulate_trades(
            day, signal, conv, n_agree, vol_pred,
            hold_bars=hold_bars,
            cost_spread_ticks=cost['spread_ticks'],
            cost_comm_ticks=cost['comm_ticks'],
            conviction_threshold=conviction,
            min_agreement=1,
            vol_percentile_min=vol_pct,
            time_filter=tf_name,
        )
        for t in trades:
            t['date'] = day['date']
        all_trades.extend(trades)

    metrics = compute_metrics(all_trades, label=f'B_{tf_name}|ES')
    results[tf_name] = metrics
    print(f"\n{'='*60}")
    print(f"TIME FILTER: {tf_name}")
    print(f"  Trades: {metrics['total_trades']}, Days: {metrics['n_days']}")
    print(f"  Net P&L: {metrics['net_pnl_ticks']:+.1f} ticks (${metrics['net_pnl_dollars']:+,.0f})")
    print(f"  Avg/trade: {metrics['avg_pnl_per_trade_ticks']:+.3f} ticks")
    print(f"  Win Rate: {metrics['win_rate']:.1f}%")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  Max DD: {metrics['max_drawdown_ticks']:.1f} ticks (${metrics['max_drawdown_dollars']:+,.0f})")

# Save results
out_path = Path(__file__).parent / 'results' / f'time_restricted_{time.strftime("%Y%m%d_%H%M%S")}.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n{time.strftime('%H:%M:%S')} Results saved to {out_path}")
