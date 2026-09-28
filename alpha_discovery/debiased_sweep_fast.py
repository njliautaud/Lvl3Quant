#!/usr/bin/env python3
"""
Fast de-biased sweep — optimized vol percentile computation.
The bisect.insort approach in _precompute_vol_percentiles is O(n^2).
Replace with a much faster approximate approach.
"""

import sys, json, gc, time, numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# Import only what we need — avoid triggering slow module-level code
from high_conviction_strategy import (
    load_all_dates, load_cnn_predictions, load_mbo_day,
    zscore_per_day, compute_trailing_vol, compute_time_features,
    HOLD_PERIODS, COST_STRUCTURES, PARAM_TUNE_DAYS,
    TICK, TICK_VAL, BARS_PER_SEC,
)


def fast_expanding_percentile(arr, percentiles=(50, 60, 70, 80, 90), block_size=1000):
    """
    Fast expanding-window percentile using block updates.
    Instead of maintaining a sorted list of all values, we:
    1. Process in blocks of block_size bars
    2. At end of each block, compute percentile thresholds
    3. Apply those thresholds to the next block
    This is O(n * (n/block_size) * log(n/block_size)) instead of O(n^2).
    """
    n = len(arr)
    result = {p: np.full(n, -np.inf, dtype=np.float32) for p in percentiles}
    collected = []

    for block_start in range(0, n, block_size):
        block_end = min(block_start + block_size, n)
        # Add valid values from this block
        for i in range(block_start, block_end):
            if not np.isnan(arr[i]):
                collected.append(arr[i])

        if len(collected) < 100:
            continue

        # Compute percentiles from collected values
        sorted_vals = np.sort(collected)
        for p in percentiles:
            idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
            thresh = sorted_vals[idx]
            # Apply to the NEXT block (no look-ahead)
            next_start = block_end
            next_end = min(block_end + block_size, n)
            if next_start < n:
                result[p][next_start:next_end] = thresh

    return result


def simulate_trades_fast(mid, signal, vol_pred, vol_thresh_arr,
                         hold_bars, cost_spread_ticks, cost_comm_ticks,
                         conviction_threshold, vol_percentile_min,
                         time_filter, max_trades_per_day=50):
    """Minimal trade simulation — CNN only, no ensemble agreement."""
    n = len(mid)
    rt_cost_points = (cost_spread_ticks + cost_comm_ticks) * 0.25  # TICK = 0.25

    # Time filter
    seconds = np.arange(n) / 10.0
    minutes = seconds / 60.0
    if time_filter == 'morning_afternoon':
        time_ok = (minutes < 120) | ((minutes >= 240) & (minutes < 330))
    elif time_filter == 'first_hour':
        time_ok = minutes < 60
    elif time_filter == 'no_lunch':
        lunch_dead = (minutes > 120) & (minutes < 240)
        time_ok = ~lunch_dead
    else:
        time_ok = np.ones(n, dtype=bool)

    trades = []
    last_exit_bar = -1

    for i in range(100, n - hold_bars):
        if i < last_exit_bar + hold_bars:
            continue
        if len(trades) >= max_trades_per_day:
            break
        if not time_ok[i]:
            continue
        if np.isnan(signal[i]) or abs(signal[i]) < conviction_threshold:
            continue
        if np.isnan(vol_pred[i]) or vol_pred[i] < vol_thresh_arr[i]:
            continue

        direction = 1.0 if signal[i] > 0 else -1.0
        entry_price = mid[i]
        exit_bar = i + hold_bars
        exit_price = mid[exit_bar]

        gross_pnl_points = direction * (exit_price - entry_price)
        net_pnl_points = gross_pnl_points - rt_cost_points
        gross_ticks = gross_pnl_points / 0.25
        net_ticks = net_pnl_points / 0.25

        trades.append({
            'gross_pnl_ticks': float(gross_ticks),
            'net_pnl_ticks': float(net_ticks),
        })
        last_exit_bar = exit_bar

    return trades


def main():
    t0 = time.time()
    print("Loading dates...")
    all_dates = load_all_dates()

    print("Loading CNN predictions...")
    cnn_data = load_cnn_predictions()
    cnn_dates = sorted(cnn_data.keys())
    print(f"  {len(cnn_dates)} CNN days")

    CNN_OFFSET = 99
    cost = COST_STRUCTURES['ES_futures']

    # Process days one at a time, keeping only minimal data
    print("Processing days...")
    all_days = []
    for di, date in enumerate(cnn_dates):
        mbo = load_mbo_day(date)
        if mbo is None:
            continue
        mid, spread = mbo
        n_bars = len(mid)
        cp, ct = cnn_data[date]

        # Align CNN predictions
        cp_aligned = np.full(n_bars, np.nan, dtype=np.float32)
        end_idx = min(CNN_OFFSET + len(cp), n_bars)
        cp_aligned[CNN_OFFSET:end_idx] = cp[:end_idx - CNN_OFFSET]

        # Expanding z-score signal
        signal = zscore_per_day(cp_aligned).astype(np.float32)

        # Trailing vol
        vol_pred = compute_trailing_vol(mid).astype(np.float32)

        # Fast expanding percentiles
        vol_pct_thresholds = fast_expanding_percentile(vol_pred)

        all_days.append({
            'date': date,
            'mid': mid.astype(np.float32),
            'signal': signal,
            'vol_pred': vol_pred,
            'vol_pct': vol_pct_thresholds,
            'n_bars': n_bars,
        })

        if (di + 1) % 20 == 0:
            print(f"  {di+1}/{len(cnn_dates)} days processed")

    del cnn_data
    gc.collect()
    print(f"Loaded {len(all_days)} days in {time.time()-t0:.0f}s")

    # OOS split
    oos_days = all_days[PARAM_TUNE_DAYS:]
    print(f"OOS: {len(oos_days)} days")

    # Full sweep
    configs = []
    for vol_pct in [50, 60, 70, 80]:
        for conv in [0.5, 1.0, 1.5, 2.0]:
            for hold in ['5min', '10min', '30min', '1hr']:
                for tf in ['none', 'morning_afternoon']:
                    configs.append({
                        'vol_pct': vol_pct, 'conv': conv,
                        'hold': hold, 'time': tf,
                    })

    print(f"Testing {len(configs)} configs...")
    t1 = time.time()

    results = []
    for ci, cfg in enumerate(configs):
        hold_bars = HOLD_PERIODS[cfg['hold']]
        all_trades = []
        daily_pnl = defaultdict(float)

        for day in oos_days:
            # Get vol threshold array for this percentile
            vol_thresh = day['vol_pct'].get(cfg['vol_pct'], np.full(day['n_bars'], -np.inf))

            trades = simulate_trades_fast(
                day['mid'], day['signal'], day['vol_pred'], vol_thresh,
                hold_bars=hold_bars,
                cost_spread_ticks=cost['spread_ticks'],
                cost_comm_ticks=cost['comm_ticks'],
                conviction_threshold=cfg['conv'],
                vol_percentile_min=cfg['vol_pct'],
                time_filter=cfg['time'],
            )
            for t in trades:
                daily_pnl[day['date']] += t['net_pnl_ticks']
            all_trades.extend(trades)

        # Compute metrics inline (avoid heavy function)
        n_trades = len(all_trades)
        if n_trades < 1:
            results.append({
                'config': cfg, 'trades': 0, 'pnl_ticks': 0,
                'pnl_dollars': 0, 'sharpe': 0, 'win_rate': 0,
                'max_dd_ticks': 0, 'profitable': False,
                'gross_per_trade': 0, 'net_per_trade': 0,
            })
            continue

        pnls = np.array([t['net_pnl_ticks'] for t in all_trades])
        gross = np.array([t['gross_pnl_ticks'] for t in all_trades])
        daily_returns = np.array(list(daily_pnl.values()))

        # Sharpe
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Max drawdown
        cumsum = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumsum)
        max_dd = (running_max - cumsum).max()

        results.append({
            'config': cfg,
            'trades': n_trades,
            'n_days': len(daily_pnl),
            'pnl_ticks': round(float(pnls.sum()), 1),
            'pnl_dollars': round(float(pnls.sum() * TICK_VAL), 0),
            'gross_per_trade': round(float(gross.mean()), 3),
            'net_per_trade': round(float(pnls.mean()), 3),
            'sharpe': round(float(sharpe), 2),
            'win_rate': round(float((pnls > 0).mean() * 100), 1),
            'max_dd_ticks': round(float(max_dd), 1),
            'max_dd_dollars': round(float(max_dd * TICK_VAL), 0),
            'profitable': bool(pnls.sum() > 0),
        })

        if (ci + 1) % 32 == 0:
            elapsed = time.time() - t1
            remaining = elapsed / (ci + 1) * (len(configs) - ci - 1)
            print(f"  {ci+1}/{len(configs)} configs done ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    # Sort by Sharpe
    sorted_r = sorted(results, key=lambda x: x['sharpe'], reverse=True)

    print(f"\n{'='*100}")
    print(f"FULL OOS SWEEP ({len(oos_days)} days, {len(configs)} configs)")
    print(f"{'='*100}")

    n_eligible = sum(1 for r in results if r['trades'] >= 10)
    n_profitable = sum(1 for r in results if r['profitable'] and r['trades'] >= 10)
    print(f"Eligible configs (>=10 trades): {n_eligible}")
    print(f"Profitable configs: {n_profitable} ({100*n_profitable/max(n_eligible,1):.0f}%)")

    print(f"\nTop 20 by Sharpe:")
    for r in sorted_r[:20]:
        cfg = r['config']
        print(f"  v>={cfg['vol_pct']:2d} c>={cfg['conv']:.1f} {cfg['hold']:5s} {cfg['time']:20s} | "
              f"trades={r['trades']:4d} PnL={r['pnl_ticks']:+8.1f}t "
              f"(${r['pnl_dollars']:+8.0f}) Sharpe={r['sharpe']:+5.2f} "
              f"WR={r['win_rate']:5.1f}% DD={r['max_dd_ticks']:6.0f}t "
              f"gross/t={r['gross_per_trade']:+.3f} net/t={r['net_per_trade']:+.3f}")

    print(f"\nBottom 5:")
    for r in sorted_r[-5:]:
        cfg = r['config']
        print(f"  v>={cfg['vol_pct']:2d} c>={cfg['conv']:.1f} {cfg['hold']:5s} {cfg['time']:20s} | "
              f"trades={r['trades']:4d} PnL={r['pnl_ticks']:+8.1f}t Sharpe={r['sharpe']:+5.2f}")

    # By hold period
    print(f"\nBy hold period:")
    for hold in ['5min', '10min', '30min', '1hr']:
        subset = [r for r in results if r['config']['hold'] == hold and r['trades'] >= 10]
        if subset:
            avg_sharpe = np.mean([r['sharpe'] for r in subset])
            pct_prof = 100 * sum(1 for r in subset if r['profitable']) / len(subset)
            avg_pnl = np.mean([r['pnl_ticks'] for r in subset])
            print(f"  {hold:5s}: avg Sharpe={avg_sharpe:+.2f}, {pct_prof:.0f}% profitable, "
                  f"avg PnL={avg_pnl:+.0f}t ({len(subset)} configs)")

    # By vol filter
    print(f"\nBy vol filter:")
    for vol in [50, 60, 70, 80]:
        subset = [r for r in results if r['config']['vol_pct'] == vol and r['trades'] >= 10]
        if subset:
            avg_sharpe = np.mean([r['sharpe'] for r in subset])
            pct_prof = 100 * sum(1 for r in subset if r['profitable']) / len(subset)
            print(f"  v>={vol:2d}: avg Sharpe={avg_sharpe:+.2f}, {pct_prof:.0f}% profitable ({len(subset)} configs)")

    # Save
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = f"alpha_discovery/results/debiased_full_sweep_{ts}.json"
    with open(outfile, 'w') as f:
        json.dump({
            'configs_tested': len(configs),
            'oos_days': len(oos_days),
            'profitable_pct': round(100 * n_profitable / max(n_eligible, 1), 1),
            'top_20': sorted_r[:20],
            'all_results': results,
            'total_time_seconds': round(time.time() - t0, 1),
        }, f, indent=2, default=str)
    print(f"\nSaved to {outfile}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
