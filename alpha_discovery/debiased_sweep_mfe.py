#!/usr/bin/env python3
"""
MFE-enhanced de-biased sweep.
Adds per-trade: direction, MFE, MAE, time-to-MFE, entry/exit bars.
Also tests dynamic exit strategies (trailing stop from MFE).
Based on debiased_sweep_fast.py with full MFE instrumentation.
"""

import sys, json, gc, time, csv, numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from high_conviction_strategy import (
    load_all_dates, load_cnn_predictions, load_mbo_day,
    zscore_per_day, compute_trailing_vol, compute_time_features,
    HOLD_PERIODS, COST_STRUCTURES, PARAM_TUNE_DAYS,
    TICK, TICK_VAL, BARS_PER_SEC,
)


def fast_expanding_percentile(arr, percentiles=(50, 60, 70, 80, 90), block_size=1000):
    n = len(arr)
    result = {p: np.full(n, -np.inf, dtype=np.float32) for p in percentiles}
    collected = []
    for block_start in range(0, n, block_size):
        block_end = min(block_start + block_size, n)
        for i in range(block_start, block_end):
            if not np.isnan(arr[i]):
                collected.append(arr[i])
        if len(collected) < 100:
            continue
        sorted_vals = np.sort(collected)
        for p in percentiles:
            idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
            thresh = sorted_vals[idx]
            next_start = block_end
            next_end = min(block_end + block_size, n)
            if next_start < n:
                result[p][next_start:next_end] = thresh
    return result


def simulate_trades_mfe(mid, signal, vol_pred, vol_thresh_arr,
                        hold_bars, cost_spread_ticks, cost_comm_ticks,
                        conviction_threshold, vol_percentile_min,
                        time_filter, date_str='', max_trades_per_day=50):
    """Trade simulation with full MFE/MAE instrumentation."""
    n = len(mid)
    rt_cost_points = (cost_spread_ticks + cost_comm_ticks) * 0.25

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

        # MFE/MAE: extract intra-trade price path
        path_slice = mid[i:exit_bar + 1]
        # Directional P&L path in ticks (positive = favorable)
        path_ticks = direction * (path_slice - entry_price) / 0.25
        mfe_ticks = float(np.max(path_ticks))
        mae_ticks = float(np.max(-path_ticks))  # worst adverse excursion (positive number)
        time_to_mfe = int(np.argmax(path_ticks))

        # Also track P&L at various checkpoints (for dynamic exit analysis)
        checkpoints = {}
        for pct in [0.25, 0.50, 0.75]:
            cp_bar = int(hold_bars * pct)
            if cp_bar < len(path_ticks):
                checkpoints[f'pnl_at_{int(pct*100)}pct'] = float(path_ticks[cp_bar])

        trades.append({
            'date': date_str,
            'direction': int(direction),
            'entry_bar': int(i),
            'exit_bar': int(exit_bar),
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'gross_pnl_ticks': float(gross_ticks),
            'net_pnl_ticks': float(net_ticks),
            'mfe_ticks': mfe_ticks,
            'mae_ticks': mae_ticks,
            'time_to_mfe_bars': time_to_mfe,
            'signal_strength': float(abs(signal[i])),
            'vol_pred': float(vol_pred[i]),
            **checkpoints,
        })
        last_exit_bar = exit_bar

    return trades


def simulate_dynamic_exit(mid, signal, vol_pred, vol_thresh_arr,
                          hold_bars, cost_spread_ticks, cost_comm_ticks,
                          conviction_threshold, vol_percentile_min,
                          time_filter, trail_stop_ticks=None,
                          profit_target_ticks=None, max_trades_per_day=50):
    """Trade simulation with dynamic exits: trailing stop and/or profit target."""
    n = len(mid)
    rt_cost_points = (cost_spread_ticks + cost_comm_ticks) * 0.25

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
        max_bar = i + hold_bars

        # Dynamic exit: scan bar by bar
        best_pnl_ticks = 0.0
        actual_exit_bar = max_bar
        for j in range(i + 1, max_bar + 1):
            current_pnl_ticks = direction * (mid[j] - entry_price) / 0.25
            best_pnl_ticks = max(best_pnl_ticks, current_pnl_ticks)

            # Profit target hit
            if profit_target_ticks and current_pnl_ticks >= profit_target_ticks:
                actual_exit_bar = j
                break

            # Trailing stop: if we've had MFE >= some threshold and now pulled back
            if trail_stop_ticks and best_pnl_ticks >= trail_stop_ticks:
                pullback = best_pnl_ticks - current_pnl_ticks
                if pullback >= trail_stop_ticks * 0.5:  # give back 50% of trail threshold
                    actual_exit_bar = j
                    break

        exit_price = mid[actual_exit_bar]
        gross_pnl_points = direction * (exit_price - entry_price)
        net_pnl_points = gross_pnl_points - rt_cost_points
        gross_ticks = gross_pnl_points / 0.25
        net_ticks = net_pnl_points / 0.25

        trades.append({
            'gross_pnl_ticks': float(gross_ticks),
            'net_pnl_ticks': float(net_ticks),
            'hold_bars_actual': int(actual_exit_bar - i),
        })
        last_exit_bar = actual_exit_bar

    return trades


def main():
    t0 = time.time()

    # Parse CLI args
    mode = 'full'  # 'full' = MFE sweep, 'dynamic' = dynamic exit sweep, 'both' = both
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    print(f"Mode: {mode}")
    print("Loading dates...")
    all_dates = load_all_dates()

    print("Loading CNN predictions...")
    cnn_data = load_cnn_predictions()
    cnn_dates = sorted(cnn_data.keys())
    print(f"  {len(cnn_dates)} CNN days")

    CNN_OFFSET = 99
    cost = COST_STRUCTURES['ES_futures']

    print("Processing days...")
    all_days = []
    for di, date in enumerate(cnn_dates):
        mbo = load_mbo_day(date)
        if mbo is None:
            continue
        mid, spread = mbo
        n_bars = len(mid)
        cp, ct = cnn_data[date]

        cp_aligned = np.full(n_bars, np.nan, dtype=np.float32)
        end_idx = min(CNN_OFFSET + len(cp), n_bars)
        cp_aligned[CNN_OFFSET:end_idx] = cp[:end_idx - CNN_OFFSET]

        signal = zscore_per_day(cp_aligned).astype(np.float32)
        vol_pred = compute_trailing_vol(mid).astype(np.float32)
        vol_pct_thresholds = fast_expanding_percentile(vol_pred)

        all_days.append({
            'date': date, 'mid': mid.astype(np.float32),
            'signal': signal, 'vol_pred': vol_pred,
            'vol_pct': vol_pct_thresholds, 'n_bars': n_bars,
        })
        if (di + 1) % 20 == 0:
            print(f"  {di+1}/{len(cnn_dates)} days processed")

    del cnn_data
    gc.collect()
    print(f"Loaded {len(all_days)} days in {time.time()-t0:.0f}s")

    oos_days = all_days[PARAM_TUNE_DAYS:]
    print(f"OOS: {len(oos_days)} days")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path('alpha_discovery/results')
    results_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # PHASE 1: MFE-instrumented sweep (same configs as original)
    # ============================================================
    if mode in ('full', 'both'):
        print("\n" + "="*80)
        print("PHASE 1: MFE-instrumented sweep")
        print("="*80)

        configs = []
        for vol_pct in [50, 60, 70, 80]:
            for conv in [0.5, 1.0, 1.5, 2.0]:
                for hold in ['5min', '10min', '30min', '1hr']:
                    for tf in ['none', 'morning_afternoon']:
                        configs.append({
                            'vol_pct': vol_pct, 'conv': conv,
                            'hold': hold, 'time': tf,
                        })

        print(f"Testing {len(configs)} configs with MFE tracking...")
        t1 = time.time()

        results = []
        all_trade_records = []  # For CSV export

        for ci, cfg in enumerate(configs):
            hold_bars = HOLD_PERIODS[cfg['hold']]
            config_trades = []
            daily_pnl = defaultdict(float)

            for day in oos_days:
                vol_thresh = day['vol_pct'].get(cfg['vol_pct'], np.full(day['n_bars'], -np.inf))
                trades = simulate_trades_mfe(
                    day['mid'], day['signal'], day['vol_pred'], vol_thresh,
                    hold_bars=hold_bars,
                    cost_spread_ticks=cost['spread_ticks'],
                    cost_comm_ticks=cost['comm_ticks'],
                    conviction_threshold=cfg['conv'],
                    vol_percentile_min=cfg['vol_pct'],
                    time_filter=cfg['time'],
                    date_str=day['date'],
                )
                for t in trades:
                    daily_pnl[day['date']] += t['net_pnl_ticks']
                config_trades.extend(trades)

            n_trades = len(config_trades)
            if n_trades < 1:
                results.append({
                    'config': cfg, 'trades': 0, 'pnl_ticks': 0,
                    'pnl_dollars': 0, 'sharpe': 0, 'win_rate': 0,
                    'max_dd_ticks': 0, 'profitable': False,
                    'gross_per_trade': 0, 'net_per_trade': 0,
                })
                continue

            pnls = np.array([t['net_pnl_ticks'] for t in config_trades])
            gross = np.array([t['gross_pnl_ticks'] for t in config_trades])
            mfes = np.array([t['mfe_ticks'] for t in config_trades])
            maes = np.array([t['mae_ticks'] for t in config_trades])
            ttmfes = np.array([t['time_to_mfe_bars'] for t in config_trades])
            directions = np.array([t['direction'] for t in config_trades])

            daily_returns = np.array(list(daily_pnl.values()))
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                sharpe = 0

            cumsum = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumsum)
            max_dd = (running_max - cumsum).max()

            # Long/short breakdown
            long_mask = directions > 0
            short_mask = directions < 0
            n_long = int(long_mask.sum())
            n_short = int(short_mask.sum())
            long_pnl = float(pnls[long_mask].sum()) if n_long > 0 else 0
            short_pnl = float(pnls[short_mask].sum()) if n_short > 0 else 0

            r = {
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
                # NEW: MFE/MAE/direction stats
                'n_long': n_long,
                'n_short': n_short,
                'long_pnl_ticks': round(long_pnl, 1),
                'short_pnl_ticks': round(short_pnl, 1),
                'avg_mfe_ticks': round(float(mfes.mean()), 2),
                'avg_mae_ticks': round(float(maes.mean()), 2),
                'avg_time_to_mfe_bars': round(float(ttmfes.mean()), 1),
                'median_mfe_ticks': round(float(np.median(mfes)), 2),
                'p75_mfe_ticks': round(float(np.percentile(mfes, 75)), 2),
                'p90_mfe_ticks': round(float(np.percentile(mfes, 90)), 2),
                'mfe_capture_ratio': round(float(gross.mean() / mfes.mean()) if mfes.mean() > 0 else 0, 3),
            }
            results.append(r)

            # Save per-trade records for best configs (30min hold only to save space)
            if cfg['hold'] == '30min' and cfg['vol_pct'] >= 70:
                for t in config_trades:
                    t['config_key'] = f"v{cfg['vol_pct']}_c{cfg['conv']}_{cfg['hold']}_{cfg['time']}"
                    all_trade_records.append(t)

            if (ci + 1) % 32 == 0:
                elapsed = time.time() - t1
                remaining = elapsed / (ci + 1) * (len(configs) - ci - 1)
                print(f"  {ci+1}/{len(configs)} configs done ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        # Sort and print results
        sorted_r = sorted(results, key=lambda x: x['sharpe'], reverse=True)

        print(f"\n{'='*120}")
        print(f"MFE SWEEP RESULTS ({len(oos_days)} OOS days, {len(configs)} configs)")
        print(f"{'='*120}")

        n_eligible = sum(1 for r in results if r['trades'] >= 10)
        n_profitable = sum(1 for r in results if r['profitable'] and r['trades'] >= 10)
        print(f"Eligible (>=10 trades): {n_eligible}, Profitable: {n_profitable} ({100*n_profitable/max(n_eligible,1):.0f}%)")

        print(f"\nTop 20 by Sharpe (with MFE/direction data):")
        for r in sorted_r[:20]:
            cfg = r['config']
            print(f"  v>={cfg['vol_pct']:2d} c>={cfg['conv']:.1f} {cfg['hold']:5s} {cfg['time']:20s} | "
                  f"trades={r['trades']:4d} (L:{r.get('n_long',0):3d}/S:{r.get('n_short',0):3d}) "
                  f"PnL={r['pnl_ticks']:+8.1f}t (L:{r.get('long_pnl_ticks',0):+.0f}/S:{r.get('short_pnl_ticks',0):+.0f}) "
                  f"Sharpe={r['sharpe']:+5.2f} WR={r['win_rate']:5.1f}% "
                  f"MFE={r.get('avg_mfe_ticks',0):5.1f} MAE={r.get('avg_mae_ticks',0):5.1f} "
                  f"ttMFE={r.get('avg_time_to_mfe_bars',0):5.0f}bars "
                  f"capture={r.get('mfe_capture_ratio',0):.1%}")

        # MFE analysis by hold period
        print(f"\nMFE Analysis by Hold Period:")
        for hold in ['5min', '10min', '30min', '1hr']:
            subset = [r for r in results if r['config']['hold'] == hold and r['trades'] >= 10]
            if subset:
                avg_mfe = np.mean([r.get('avg_mfe_ticks', 0) for r in subset])
                avg_mae = np.mean([r.get('avg_mae_ticks', 0) for r in subset])
                avg_ttmfe = np.mean([r.get('avg_time_to_mfe_bars', 0) for r in subset])
                avg_capture = np.mean([r.get('mfe_capture_ratio', 0) for r in subset])
                avg_sharpe = np.mean([r['sharpe'] for r in subset])
                pct_prof = 100 * sum(1 for r in subset if r['profitable']) / len(subset)
                print(f"  {hold:5s}: Sharpe={avg_sharpe:+.2f} {pct_prof:.0f}%prof "
                      f"MFE={avg_mfe:.1f}t MAE={avg_mae:.1f}t "
                      f"ttMFE={avg_ttmfe:.0f}bars capture={avg_capture:.1%}")

        # Long/short breakdown
        print(f"\nLong/Short Breakdown (profitable configs only):")
        for r in sorted_r[:10]:
            if r['profitable'] and r['trades'] >= 10:
                cfg = r['config']
                print(f"  v>={cfg['vol_pct']:2d} c>={cfg['conv']:.1f} {cfg['hold']:5s} | "
                      f"Long: {r.get('n_long',0)} trades, {r.get('long_pnl_ticks',0):+.0f}t | "
                      f"Short: {r.get('n_short',0)} trades, {r.get('short_pnl_ticks',0):+.0f}t")

        # Save JSON
        outfile = results_dir / f"mfe_sweep_{ts}.json"
        with open(outfile, 'w') as f:
            json.dump({
                'mode': 'mfe_sweep',
                'configs_tested': len(configs),
                'oos_days': len(oos_days),
                'profitable_pct': round(100 * n_profitable / max(n_eligible, 1), 1),
                'top_20': sorted_r[:20],
                'all_results': results,
                'total_time_seconds': round(time.time() - t0, 1),
            }, f, indent=2, default=str)
        print(f"\nSaved JSON: {outfile}")

        # Save per-trade CSV for deep analysis
        if all_trade_records:
            csvfile = results_dir / f"mfe_trades_{ts}.csv"
            fieldnames = ['config_key', 'date', 'direction', 'entry_bar', 'exit_bar',
                          'entry_price', 'exit_price', 'gross_pnl_ticks', 'net_pnl_ticks',
                          'mfe_ticks', 'mae_ticks', 'time_to_mfe_bars', 'signal_strength',
                          'vol_pred', 'pnl_at_25pct', 'pnl_at_50pct', 'pnl_at_75pct']
            with open(csvfile, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_trade_records)
            print(f"Saved {len(all_trade_records)} trade records: {csvfile}")

    # ============================================================
    # PHASE 2: Dynamic exit strategies
    # ============================================================
    if mode in ('dynamic', 'both'):
        print("\n" + "="*80)
        print("PHASE 2: Dynamic exit strategies")
        print("="*80)

        # Only test on best hold period (30min) with top vol/conv configs
        base_hold = '30min'
        hold_bars = HOLD_PERIODS[base_hold]

        dynamic_configs = []
        for vol_pct in [70, 80]:
            for conv in [1.0, 1.5, 2.0]:
                for tf in ['none', 'morning_afternoon']:
                    # Fixed 30min (baseline)
                    dynamic_configs.append({
                        'vol_pct': vol_pct, 'conv': conv, 'time': tf,
                        'exit_type': 'fixed_30min',
                        'trail_stop': None, 'profit_target': None,
                    })
                    # Trailing stops at various levels
                    for trail in [5, 8, 10, 15, 20]:
                        dynamic_configs.append({
                            'vol_pct': vol_pct, 'conv': conv, 'time': tf,
                            'exit_type': f'trail_{trail}t',
                            'trail_stop': trail, 'profit_target': None,
                        })
                    # Profit targets
                    for target in [10, 15, 20, 30]:
                        dynamic_configs.append({
                            'vol_pct': vol_pct, 'conv': conv, 'time': tf,
                            'exit_type': f'target_{target}t',
                            'trail_stop': None, 'profit_target': target,
                        })
                    # Combined: trail + target
                    for trail, target in [(8, 20), (10, 25), (10, 30), (15, 30)]:
                        dynamic_configs.append({
                            'vol_pct': vol_pct, 'conv': conv, 'time': tf,
                            'exit_type': f'trail{trail}_target{target}',
                            'trail_stop': trail, 'profit_target': target,
                        })

        print(f"Testing {len(dynamic_configs)} dynamic exit configs...")
        t2 = time.time()

        dynamic_results = []
        for ci, cfg in enumerate(dynamic_configs):
            all_trades = []
            daily_pnl = defaultdict(float)

            for day in oos_days:
                vol_thresh = day['vol_pct'].get(cfg['vol_pct'], np.full(day['n_bars'], -np.inf))
                trades = simulate_dynamic_exit(
                    day['mid'], day['signal'], day['vol_pred'], vol_thresh,
                    hold_bars=hold_bars,
                    cost_spread_ticks=cost['spread_ticks'],
                    cost_comm_ticks=cost['comm_ticks'],
                    conviction_threshold=cfg['conv'],
                    vol_percentile_min=cfg['vol_pct'],
                    time_filter=cfg['time'],
                    trail_stop_ticks=cfg['trail_stop'],
                    profit_target_ticks=cfg['profit_target'],
                )
                for t in trades:
                    daily_pnl[day['date']] += t['net_pnl_ticks']
                all_trades.extend(trades)

            n_trades = len(all_trades)
            if n_trades < 1:
                dynamic_results.append({
                    'config': cfg, 'trades': 0, 'sharpe': 0, 'pnl_ticks': 0,
                })
                continue

            pnls = np.array([t['net_pnl_ticks'] for t in all_trades])
            gross = np.array([t['gross_pnl_ticks'] for t in all_trades])
            daily_returns = np.array(list(daily_pnl.values()))

            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                sharpe = 0

            cumsum = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumsum)
            max_dd = (running_max - cumsum).max()

            avg_hold = np.mean([t.get('hold_bars_actual', hold_bars) for t in all_trades])

            dynamic_results.append({
                'config': cfg,
                'trades': n_trades,
                'pnl_ticks': round(float(pnls.sum()), 1),
                'pnl_dollars': round(float(pnls.sum() * TICK_VAL), 0),
                'sharpe': round(float(sharpe), 2),
                'win_rate': round(float((pnls > 0).mean() * 100), 1),
                'max_dd_ticks': round(float(max_dd), 1),
                'gross_per_trade': round(float(gross.mean()), 3),
                'net_per_trade': round(float(pnls.mean()), 3),
                'avg_hold_bars': round(float(avg_hold), 0),
                'profitable': bool(pnls.sum() > 0),
            })

            if (ci + 1) % 50 == 0:
                elapsed = time.time() - t2
                remaining = elapsed / (ci + 1) * (len(dynamic_configs) - ci - 1)
                print(f"  {ci+1}/{len(dynamic_configs)} dynamic configs ({elapsed:.0f}s, ~{remaining:.0f}s left)")

        # Sort and display
        sorted_dyn = sorted(dynamic_results, key=lambda x: x['sharpe'], reverse=True)

        print(f"\n{'='*120}")
        print(f"DYNAMIC EXIT RESULTS ({len(oos_days)} OOS days)")
        print(f"{'='*120}")

        print(f"\nTop 30 by Sharpe:")
        for r in sorted_dyn[:30]:
            cfg = r['config']
            print(f"  v>={cfg['vol_pct']:2d} c>={cfg['conv']:.1f} {cfg['time']:20s} "
                  f"exit={cfg['exit_type']:20s} | "
                  f"trades={r['trades']:4d} PnL={r['pnl_ticks']:+8.1f}t "
                  f"(${r.get('pnl_dollars',0):+8.0f}) Sharpe={r['sharpe']:+5.2f} "
                  f"WR={r.get('win_rate',0):5.1f}% avgHold={r.get('avg_hold_bars',0):.0f}bars")

        # Compare exit types
        print(f"\nBy Exit Type (averaged across configs):")
        exit_types = sorted(set(cfg['config']['exit_type'] for cfg in dynamic_results if cfg['trades'] >= 5))
        for et in exit_types:
            subset = [r for r in dynamic_results if r['config']['exit_type'] == et and r['trades'] >= 5]
            if subset:
                avg_sharpe = np.mean([r['sharpe'] for r in subset])
                avg_pnl = np.mean([r['pnl_ticks'] for r in subset])
                pct_prof = 100 * sum(1 for r in subset if r['profitable']) / len(subset)
                avg_hold = np.mean([r.get('avg_hold_bars', 0) for r in subset])
                print(f"  {et:25s}: Sharpe={avg_sharpe:+.2f} PnL={avg_pnl:+.0f}t "
                      f"{pct_prof:.0f}%prof avgHold={avg_hold:.0f}bars")

        # Save
        dyn_outfile = results_dir / f"dynamic_exit_sweep_{ts}.json"
        with open(dyn_outfile, 'w') as f:
            json.dump({
                'mode': 'dynamic_exit',
                'base_hold': base_hold,
                'configs_tested': len(dynamic_configs),
                'oos_days': len(oos_days),
                'top_30': sorted_dyn[:30],
                'all_results': dynamic_results,
                'total_time_seconds': round(time.time() - t0, 1),
            }, f, indent=2, default=str)
        print(f"\nSaved: {dyn_outfile}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
