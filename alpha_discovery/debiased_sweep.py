#!/usr/bin/env python3
"""Full OOS sweep of de-biased CNN strategy across all configs."""

import sys, json, gc, numpy as np
from pathlib import Path
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
from high_conviction_strategy import (
    load_all_dates, load_cnn_predictions, load_mbo_day,
    zscore_per_day, compute_trailing_vol, _precompute_vol_percentiles,
    simulate_trades, compute_metrics, HOLD_PERIODS, COST_STRUCTURES,
    PARAM_TUNE_DAYS
)
from datetime import datetime

# Load CNN-only data
all_dates = load_all_dates()
cnn_data = load_cnn_predictions()

CNN_OFFSET = 99
cost = COST_STRUCTURES['ES_futures']

cnn_dates_sorted = sorted(cnn_data.keys())
all_days = []
for date in cnn_dates_sorted:
    mbo = load_mbo_day(date)
    if mbo is None:
        continue
    mid, spread = mbo
    n_bars = len(mid)
    cp, ct = cnn_data[date]
    cp_aligned = np.full(n_bars, np.nan)
    end_idx = min(CNN_OFFSET + len(cp), n_bars)
    cp_aligned[CNN_OFFSET:end_idx] = cp[:end_idx - CNN_OFFSET]
    signal = zscore_per_day(cp_aligned)
    conviction = np.abs(signal)
    n_agree = np.ones(n_bars, dtype=int)
    vol_pred = compute_trailing_vol(mid)
    vol_pct_thresholds = _precompute_vol_percentiles(vol_pred)
    all_days.append({
        'date': date, 'mid': mid.astype(np.float32), 'spread': spread.astype(np.float32),
        'n_bars': n_bars,
        '_signal': signal.astype(np.float32), '_conviction': conviction.astype(np.float32),
        '_n_agree': n_agree,
        '_vol_pred': vol_pred.astype(np.float32),
        '_vol_pct_thresholds': {k: v.astype(np.float32) for k, v in vol_pct_thresholds.items()},
    })
    # Free mbo data immediately
    del mid, spread, cp_aligned, signal, conviction, vol_pred, vol_pct_thresholds
    gc.collect()
del cnn_data; gc.collect()
print(f"Loaded {len(all_days)} CNN days")

# Use full OOS (skip first 20 tune days)
oos_days = all_days[PARAM_TUNE_DAYS:]
print(f"OOS: {len(oos_days)} days")

# Test a wider range of configs on OOS DIRECTLY
configs = []
for vol_pct in [50, 60, 70, 80]:
    for conv in [0.5, 1.0, 1.5, 2.0]:
        for hold in ['5min', '10min', '30min', '1hr']:
            for tf in ['none', 'morning_afternoon']:
                configs.append({'vol_pct': vol_pct, 'conv': conv, 'hold': hold, 'time': tf})

print(f"Testing {len(configs)} configs...")

results = []
for i, cfg in enumerate(configs):
    trades = []
    for day in oos_days:
        t = simulate_trades(
            day, day['_signal'], day['_conviction'], day['_n_agree'],
            day['_vol_pred'], hold_bars=HOLD_PERIODS[cfg['hold']],
            cost_spread_ticks=cost['spread_ticks'],
            cost_comm_ticks=cost['comm_ticks'],
            conviction_threshold=cfg['conv'],
            min_agreement=1,
            vol_percentile_min=cfg['vol_pct'],
            time_filter=cfg['time'],
        )
        for tt in t:
            tt['date'] = day['date']
        trades.extend(t)

    label = f"v{cfg['vol_pct']}|c{cfg['conv']}|{cfg['hold']}|{cfg['time']}"
    m = compute_metrics(trades, label=label)
    m['config'] = cfg
    results.append(m)
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(configs)} done...")

# Sort by Sharpe
sorted_r = sorted(results, key=lambda x: x['sharpe'], reverse=True)

print('\nFULL OOS SWEEP (74 days, no tuning bias):')
print('=' * 100)
profitable = [r for r in sorted_r if r['profitable'] and r['total_trades'] >= 10]
print(f"Profitable configs (trades>=10): {len(profitable)} / {len(sorted_r)}")
print(f"\nTop 20 by Sharpe:")
for r in sorted_r[:20]:
    cfg = r['config']
    dollars = r['net_pnl_dollars']
    print(f"  v>={cfg['vol_pct']:2d} c>={cfg['conv']:.1f} {cfg['hold']:5s} {cfg['time']:20s} | "
          f"trades={r['total_trades']:4d} PnL={r['net_pnl_ticks']:+8.1f}t "
          f"(${dollars:+8.0f}) Sharpe={r['sharpe']:+5.2f} WR={r['win_rate']:5.1f}% "
          f"DD={r['max_drawdown_ticks']:6.0f}t")

print(f"\nBottom 5 (worst):")
for r in sorted_r[-5:]:
    cfg = r['config']
    print(f"  v>={cfg['vol_pct']:2d} c>={cfg['conv']:.1f} {cfg['hold']:5s} {cfg['time']:20s} | "
          f"trades={r['total_trades']:4d} PnL={r['net_pnl_ticks']:+8.1f}t "
          f"Sharpe={r['sharpe']:+5.2f}")

# Robustness analysis: what fraction of configs are profitable?
n_profitable = sum(1 for r in results if r['net_pnl_ticks'] > 0 and r['total_trades'] >= 10)
n_eligible = sum(1 for r in results if r['total_trades'] >= 10)
print(f"\nRobustness: {n_profitable}/{n_eligible} configs profitable ({100*n_profitable/max(n_eligible,1):.0f}%)")

# By hold period
for hold in ['5min', '10min', '30min', '1hr']:
    subset = [r for r in results if r['config']['hold'] == hold and r['total_trades'] >= 10]
    if subset:
        avg_sharpe = np.mean([r['sharpe'] for r in subset])
        pct_profit = 100 * sum(1 for r in subset if r['net_pnl_ticks'] > 0) / len(subset)
        avg_pnl = np.mean([r['net_pnl_ticks'] for r in subset])
        print(f"  {hold:5s}: avg Sharpe={avg_sharpe:+.2f}, {pct_profit:.0f}% profitable, avg PnL={avg_pnl:+.0f}t ({len(subset)} configs)")

# Save
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
outfile = f"alpha_discovery/results/debiased_full_sweep_{ts}.json"
with open(outfile, 'w') as f:
    json.dump({
        'configs_tested': len(results),
        'profitable_count': len(profitable),
        'robustness_pct': round(100 * n_profitable / max(n_eligible, 1), 1),
        'top_20': sorted_r[:20],
        'all_results': results,
    }, f, indent=2, default=str)
print(f"\nSaved to {outfile}")
