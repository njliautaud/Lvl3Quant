#!/usr/bin/env python3
"""
Re-run the best CNN config saving per-trade data for distribution analysis.
Config: vol>=80, conv>=1.5, 30min hold, morning_afternoon filter.
"""
import sys, json, gc, time, numpy as np
from pathlib import Path
from collections import defaultdict

_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from high_conviction_strategy import (
    load_all_dates, load_cnn_predictions, load_mbo_day,
    zscore_per_day, compute_trailing_vol,
    HOLD_PERIODS, COST_STRUCTURES, PARAM_TUNE_DAYS,
    TICK, TICK_VAL,
)
from debiased_sweep_fast import fast_expanding_percentile

def main():
    t0 = time.time()
    print("Loading data...")
    cnn_data = load_cnn_predictions()
    cnn_dates = sorted(cnn_data.keys())
    CNN_OFFSET = 99
    cost = COST_STRUCTURES['ES_futures']
    hold_bars = HOLD_PERIODS['30min']
    rt_cost_points = (cost['spread_ticks'] + cost['comm_ticks']) * 0.25

    # Config
    VOL_PCT = 80
    CONV = 1.5
    TIME_FILTER = 'morning_afternoon'

    print(f"Config: vol>={VOL_PCT}, conv>={CONV}, hold=30min, time={TIME_FILTER}")
    print(f"Processing {len(cnn_dates)} days (streaming, no memory accumulation)...", flush=True)

    # Stream process: load one day at a time, extract trades, free memory
    all_trades = []
    daily_pnl = defaultdict(float)
    daily_trades = defaultdict(int)
    day_count = 0

    for di, date in enumerate(cnn_dates):
        mbo = load_mbo_day(date)
        if mbo is None:
            continue
        day_count += 1

        # Skip IS days
        if day_count <= PARAM_TUNE_DAYS:
            del mbo
            if (di + 1) % 20 == 0:
                print(f"  {di+1}/{len(cnn_dates)} days (IS, skipping)", flush=True)
            continue

        mid, spread = mbo
        n_bars = len(mid)
        mid = mid.astype(np.float32)

        if date not in cnn_data:
            del mbo, mid
            continue

        cp, ct = cnn_data[date]
        cp_aligned = np.full(n_bars, np.nan, dtype=np.float32)
        end_idx = min(CNN_OFFSET + len(cp), n_bars)
        cp_aligned[CNN_OFFSET:end_idx] = cp[:end_idx - CNN_OFFSET]

        signal = zscore_per_day(cp_aligned).astype(np.float32)
        vol_pred = compute_trailing_vol(mid).astype(np.float32)
        vol_pct = fast_expanding_percentile(vol_pred)

        n = n_bars
        vol_thresh = vol_pct.get(VOL_PCT, np.full(n, -np.inf))

        # Time filter
        seconds = np.arange(n) / 10.0
        minutes = seconds / 60.0
        time_ok = (minutes < 120) | ((minutes >= 240) & (minutes < 330))

        last_exit_bar = -1
        day_trade_count = 0

        for i in range(100, n - hold_bars):
            if i < last_exit_bar + hold_bars:
                continue
            if day_trade_count >= 50:
                break
            if not time_ok[i]:
                continue
            if np.isnan(signal[i]) or abs(signal[i]) < CONV:
                continue
            if np.isnan(vol_pred[i]) or vol_pred[i] < vol_thresh[i]:
                continue

            direction = 1.0 if signal[i] > 0 else -1.0
            entry_price = mid[i]
            exit_bar = i + hold_bars
            exit_price = mid[exit_bar]

            gross_pnl_points = direction * (exit_price - entry_price)
            net_pnl_points = gross_pnl_points - rt_cost_points
            gross_ticks = gross_pnl_points / 0.25
            net_ticks = net_pnl_points / 0.25

            trade = {
                'date': date,
                'bar': int(i),
                'time_minutes': float(minutes[i]),
                'direction': 'LONG' if direction > 0 else 'SHORT',
                'entry': float(entry_price),
                'exit': float(exit_price),
                'signal_z': float(signal[i]),
                'vol': float(vol_pred[i]),
                'gross_ticks': float(gross_ticks),
                'net_ticks': float(net_ticks),
                'gross_dollars': float(gross_ticks * TICK_VAL),
                'net_dollars': float(net_ticks * TICK_VAL),
            }
            all_trades.append(trade)
            daily_pnl[date] += net_ticks
            daily_trades[date] += 1
            last_exit_bar = exit_bar
            day_trade_count += 1

        # Free memory for this day
        del mbo, mid, cp_aligned, signal, vol_pred, vol_pct, vol_thresh
        if (di + 1) % 20 == 0:
            print(f"  {di+1}/{len(cnn_dates)} days ({len(all_trades)} trades so far)", flush=True)
            gc.collect()

    del cnn_data; gc.collect()
    print(f"OOS complete: {day_count - PARAM_TUNE_DAYS} days, {len(all_trades)} trades")

    # Analysis
    n_trades = len(all_trades)
    if n_trades == 0:
        print("No trades!")
        return

    net_pnls = np.array([t['net_ticks'] for t in all_trades])
    gross_pnls = np.array([t['gross_ticks'] for t in all_trades])

    print(f"\n{'='*70}")
    print(f"TRADE DISTRIBUTION ANALYSIS ({n_trades} trades)")
    print(f"{'='*70}")

    # Basic stats
    print(f"\n  Total net PnL: {net_pnls.sum():+.1f} ticks (${net_pnls.sum() * TICK_VAL:+,.0f})")
    print(f"  Mean: {net_pnls.mean():+.2f} ticks | Median: {np.median(net_pnls):+.2f} ticks")
    print(f"  Std: {net_pnls.std():.2f} ticks")
    print(f"  Skew: {float(((net_pnls - net_pnls.mean())**3).mean() / net_pnls.std()**3):.2f}")
    print(f"  Win rate: {(net_pnls > 0).mean()*100:.1f}%")
    print(f"  Best trade: {net_pnls.max():+.1f} ticks (${net_pnls.max()*TICK_VAL:+,.0f})")
    print(f"  Worst trade: {net_pnls.min():+.1f} ticks (${net_pnls.min()*TICK_VAL:+,.0f})")

    # Percentile analysis
    sorted_pnls = np.sort(net_pnls)[::-1]  # Descending
    top_10pct = int(np.ceil(n_trades * 0.10))
    bottom_10pct = int(np.ceil(n_trades * 0.10))
    top_5 = min(5, n_trades)
    bottom_5 = min(5, n_trades)

    top10_sum = sorted_pnls[:top_10pct].sum()
    bottom10_sum = sorted_pnls[-bottom_10pct:].sum()
    top5_sum = sorted_pnls[:top_5].sum()
    bottom5_sum = sorted_pnls[-bottom_5:].sum()
    middle_sum = sorted_pnls[top_10pct:-bottom_10pct].sum() if n_trades > 2*top_10pct else 0

    total = net_pnls.sum()
    print(f"\n  TOP/BOTTOM ANALYSIS:")
    print(f"  Top 10% ({top_10pct} trades): {top10_sum:+.1f} ticks ({top10_sum/total*100:.1f}% of total PnL)")
    print(f"  Bottom 10% ({bottom_10pct} trades): {bottom10_sum:+.1f} ticks ({bottom10_sum/total*100:.1f}% of total PnL)")
    print(f"  Middle 80% ({n_trades - 2*top_10pct} trades): {middle_sum:+.1f} ticks ({middle_sum/total*100:.1f}% of total PnL)")
    print(f"  Top 5 trades: {top5_sum:+.1f} ticks ({top5_sum/total*100:.1f}%)")
    print(f"  Bottom 5 trades: {bottom5_sum:+.1f} ticks ({bottom5_sum/total*100:.1f}%)")

    # Remove top/bottom 5
    without_top5 = sorted_pnls[top_5:].sum()
    without_bottom5 = sorted_pnls[:-bottom_5].sum()
    print(f"\n  WITHOUT EXTREMES:")
    print(f"  Remove top 5: {without_top5:+.1f} ticks (${without_top5*TICK_VAL:+,.0f}) — {'STILL PROFITABLE' if without_top5 > 0 else 'UNPROFITABLE'}")
    print(f"  Remove bottom 5: {without_bottom5:+.1f} ticks (${without_bottom5*TICK_VAL:+,.0f})")
    print(f"  Remove both extremes: {sorted_pnls[top_5:-bottom_5].sum():+.1f} ticks")

    # Daily PnL
    daily_vals = np.array(list(daily_pnl.values()))
    all_oos_dates = sorted(set(t['date'] for t in all_trades) | set(daily_pnl.keys()))
    # Add dates with 0 trades (approximate — use all dates from cnn_dates after IS cutoff)
    all_oos_dates = cnn_dates[PARAM_TUNE_DAYS:]
    full_daily = np.array([daily_pnl.get(d, 0.0) for d in all_oos_dates])

    print(f"\n  DAILY PnL ({len(daily_pnl)} trading days, {len(all_oos_dates)} total OOS days):")
    print(f"  Mean daily (trading days): {daily_vals.mean():+.2f} ticks")
    print(f"  Mean daily (all days): {full_daily.mean():+.2f} ticks")
    print(f"  Daily std: {full_daily.std():.2f} ticks")
    print(f"  Sharpe (all days): {full_daily.mean()/full_daily.std()*np.sqrt(252):.2f}" if full_daily.std() > 0 else "")
    print(f"  Best day: {full_daily.max():+.1f} ticks")
    print(f"  Worst day: {full_daily.min():+.1f} ticks")
    print(f"  Days positive: {(full_daily > 0).sum()}/{len(full_daily)} ({(full_daily > 0).mean()*100:.0f}%)")

    # Winning/losing streaks (on full daily including zeros)
    signs = np.sign(full_daily)
    max_win_streak = max_lose_streak = current_streak = 0
    current_sign = 0
    for s in signs:
        if s > 0:
            if current_sign > 0:
                current_streak += 1
            else:
                current_streak = 1
                current_sign = 1
            max_win_streak = max(max_win_streak, current_streak)
        elif s < 0:
            if current_sign < 0:
                current_streak += 1
            else:
                current_streak = 1
                current_sign = -1
            max_lose_streak = max(max_lose_streak, current_streak)
        else:
            current_sign = 0
            current_streak = 0

    print(f"  Longest win streak: {max_win_streak} days")
    print(f"  Longest lose streak: {max_lose_streak} days")

    # Top 5 best/worst days
    day_list = [(d, daily_pnl.get(d, 0)) for d in all_oos_dates if daily_pnl.get(d, 0) != 0]
    day_list.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  TOP 5 DAYS:")
    for d, pnl in day_list[:5]:
        print(f"    {d}: {pnl:+.1f} ticks (${pnl*TICK_VAL:+,.0f}) — {daily_trades[d]} trades")
    print(f"  BOTTOM 5 DAYS:")
    for d, pnl in day_list[-5:]:
        print(f"    {d}: {pnl:+.1f} ticks (${pnl*TICK_VAL:+,.0f}) — {daily_trades[d]} trades")

    # Monthly breakdown
    monthly = defaultdict(float)
    monthly_trades = defaultdict(int)
    for t in all_trades:
        month = t['date'][:7]
        monthly[month] += t['net_ticks']
        monthly_trades[month] += 1

    print(f"\n  MONTHLY BREAKDOWN:")
    for month in sorted(monthly.keys()):
        pnl = monthly[month]
        nt = monthly_trades[month]
        print(f"    {month}: {pnl:+.1f} ticks (${pnl*TICK_VAL:+,.0f}) — {nt} trades — avg {pnl/nt:+.1f}t/trade")

    # Direction breakdown
    longs = [t for t in all_trades if t['direction'] == 'LONG']
    shorts = [t for t in all_trades if t['direction'] == 'SHORT']
    long_pnl = sum(t['net_ticks'] for t in longs)
    short_pnl = sum(t['net_ticks'] for t in shorts)
    print(f"\n  DIRECTION:")
    print(f"  Long: {len(longs)} trades, {long_pnl:+.1f} ticks, avg {long_pnl/max(len(longs),1):+.1f}t")
    print(f"  Short: {len(shorts)} trades, {short_pnl:+.1f} ticks, avg {short_pnl/max(len(shorts),1):+.1f}t")

    # Save full trade list
    outfile = Path(_this_dir) / 'results' / 'trade_distribution_best_config.json'
    with open(outfile, 'w') as f:
        json.dump({
            'config': {'vol_pct': VOL_PCT, 'conv': CONV, 'hold': '30min', 'time': TIME_FILTER},
            'n_trades': n_trades,
            'total_net_ticks': float(net_pnls.sum()),
            'total_net_dollars': float(net_pnls.sum() * TICK_VAL),
            'mean_net_ticks': float(net_pnls.mean()),
            'median_net_ticks': float(np.median(net_pnls)),
            'std_net_ticks': float(net_pnls.std()),
            'win_rate': float((net_pnls > 0).mean()),
            'top10pct_of_total': float(top10_sum / total),
            'bottom10pct_of_total': float(bottom10_sum / total),
            'profitable_without_top5': bool(without_top5 > 0),
            'sharpe_all_days': float(full_daily.mean()/full_daily.std()*np.sqrt(252)) if full_daily.std() > 0 else 0,
            'trades': all_trades,
            'daily_pnl': dict(daily_pnl),
        }, f, indent=2)
    print(f"\nSaved to {outfile}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
