#!/usr/bin/env python3
"""Strategy B robustness test — vary parameters around optimal."""
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

cost = COST_STRUCTURES['ES_futures']
results = []

# Test grid around optimal: vol_pct in [40,50,60,70,80], conviction in [0.25,0.5,0.75,1.0], hold in [10min,30min,1hr]
for vol_pct in [40, 50, 60, 70, 80]:
    for conv in [0.25, 0.5, 0.75, 1.0]:
        for hold_key in ['10min', '30min', '1hr']:
            hold_bars = HOLD_PERIODS[hold_key]
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
            
            m = compute_metrics(all_trades, label=f'v{vol_pct}_c{conv}_{hold_key}')
            results.append({
                'vol_pct': vol_pct, 'conviction': conv, 'hold': hold_key,
                'trades': m['total_trades'], 'pnl': m['net_pnl_ticks'],
                'pnl_dollars': m['net_pnl_dollars'],
                'avg_trade': m['avg_pnl_per_trade_ticks'],
                'wr': m['win_rate'], 'sharpe': m['sharpe'],
                'max_dd': m['max_drawdown_ticks'],
            })

# Sort by P&L
results.sort(key=lambda r: r['pnl'], reverse=True)

print(f"\n{'='*80}")
print(f"ROBUSTNESS GRID: {len(results)} configs tested")
print(f"{'='*80}")
print(f"{'Vol':>4} {'Conv':>5} {'Hold':>5} {'Trades':>7} {'PnL$':>9} {'$/Tr':>7} {'WR':>5} {'Sharpe':>7} {'MaxDD':>7}")
print(f"{'-'*80}")

# Show top 15 and bottom 5
for r in results[:15]:
    print(f"{r['vol_pct']:>4} {r['conviction']:>5.2f} {r['hold']:>5} {r['trades']:>7} {r['pnl_dollars']:>+9,.0f} {r['pnl_dollars']/max(r['trades'],1):>+7.0f} {r['wr']:>5.1f} {r['sharpe']:>7.2f} {r['max_dd']:>7.1f}")

print(f"\n--- Bottom 5 ---")
for r in results[-5:]:
    print(f"{r['vol_pct']:>4} {r['conviction']:>5.2f} {r['hold']:>5} {r['trades']:>7} {r['pnl_dollars']:>+9,.0f} {r['pnl_dollars']/max(r['trades'],1):>+7.0f} {r['wr']:>5.1f} {r['sharpe']:>7.2f} {r['max_dd']:>7.1f}")

# Count profitable configs
n_profitable = sum(1 for r in results if r['pnl'] > 0)
print(f"\nProfitable configs: {n_profitable}/{len(results)} ({100*n_profitable/len(results):.0f}%)")
print(f"Median P&L: ${np.median([r['pnl_dollars'] for r in results]):+,.0f}")
print(f"Mean P&L: ${np.mean([r['pnl_dollars'] for r in results]):+,.0f}")

out_path = Path(__file__).parent / 'results' / f'robustness_{time.strftime("%Y%m%d_%H%M%S")}.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n{time.strftime('%H:%M:%S')} Saved to {out_path}")
