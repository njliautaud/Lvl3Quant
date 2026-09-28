#!/usr/bin/env python3
"""
Time-of-Day and Day-of-Week Filter Analysis for Champion Strategy
================================================================
Independently validates time/day filters on the 210-trade champion dataset
(FILTERED_bias_1.50x from multi_scale_combo_v1).

Computes full regime validation (green/red Sharpe, regime gap < 0.50) for each filter.

HC #428: Regime-agnostic validation required.
HC #432: MFE-within-horizon validation required.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

# Constants
TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376

TRADE_CSV = Path("/home/jupiter/Lvl3Quant/output/multi_scale_combo_v1/champion_trade_details.csv")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/multi_scale_combo_v1")


def compute_metrics(df, label):
    """Compute full strategy metrics with regime validation."""
    n = len(df)
    if n < 5:
        return {'name': label, 'n_trades': n, 'error': 'too few trades'}

    pnl = df['pnl_ticks'].values
    dates = df['date'].values
    dirs = df['direction'].values

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    wr = len(wins) / n
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float('inf')

    # Daily aggregation for Sharpe/Sortino
    unique_dates = sorted(set(dates))
    daily_pnl = np.array([pnl[dates == d].sum() for d in unique_dates])
    n_days = len(unique_dates)

    mean_d = daily_pnl.mean()
    std_d = daily_pnl.std(ddof=1) if n_days > 1 else np.nan
    sharpe = (mean_d / std_d * np.sqrt(252)) if std_d and std_d > 0 else np.nan

    downside = daily_pnl[daily_pnl < 0]
    ds_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = (mean_d / ds_std * np.sqrt(252)) if ds_std and ds_std > 0 else np.nan

    # Max drawdown
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = dd.max() if len(dd) > 0 else 0

    # Day concentration
    daily_pnl_abs = np.abs(daily_pnl)
    day_conc = daily_pnl_abs.max() / daily_pnl_abs.sum() if daily_pnl_abs.sum() > 0 else 1.0

    # Regime analysis
    regime_metrics = {}
    for reg in ['green', 'red', 'flat']:
        reg_mask = df['regime'].values == reg
        if reg_mask.sum() < 3:
            continue
        r_pnl = pnl[reg_mask]
        r_dates = dates[reg_mask]
        r_unique = sorted(set(r_dates))
        r_daily = np.array([r_pnl[r_dates == d].sum() for d in r_unique])
        r_mean = r_daily.mean()
        r_std = r_daily.std(ddof=1) if len(r_daily) > 1 else np.nan
        r_sharpe = (r_mean / r_std * np.sqrt(252)) if r_std and r_std > 0 else np.nan
        regime_metrics[reg] = {
            'n_trades': int(reg_mask.sum()),
            'n_days': len(r_unique),
            'sharpe': round(r_sharpe, 2) if not np.isnan(r_sharpe) else None,
            'wr': round(len(r_pnl[r_pnl > 0]) / len(r_pnl), 3),
            'mean_pnl': round(r_pnl.mean(), 2),
            'total_pnl': round(r_pnl.sum(), 1),
        }

    # Regime gap (using green and red only, per HC #428)
    regime_gap = None
    regime_pass = None
    green_sharpe = regime_metrics.get('green', {}).get('sharpe')
    red_sharpe = regime_metrics.get('red', {}).get('sharpe')
    if green_sharpe is not None and red_sharpe is not None:
        max_abs = max(abs(green_sharpe), abs(red_sharpe))
        if max_abs > 0:
            regime_gap = round(abs(green_sharpe - red_sharpe) / max_abs, 3)
            regime_pass = regime_gap <= 0.50

    # Long/short breakdown
    long_mask = dirs == 1
    short_mask = dirs == -1
    long_pnl = pnl[long_mask]
    short_pnl = pnl[short_mask]

    return {
        'name': label,
        'n_trades': n,
        'n_trading_days': n_days,
        'wr': round(wr, 3),
        'pf': round(pf, 2) if pf != float('inf') else 'inf',
        'sharpe': round(sharpe, 2) if not np.isnan(sharpe) else None,
        'sortino': round(sortino, 2) if not np.isnan(sortino) else None,
        'total_pnl_ticks': round(pnl.sum(), 1),
        'total_pnl_dollars': round(pnl.sum() * TICK_VALUE, 0),
        'max_dd_ticks': round(max_dd, 1),
        'max_dd_dollars': round(max_dd * TICK_VALUE, 0),
        'day_concentration': round(day_conc, 3),
        'n_long': int(long_mask.sum()),
        'n_short': int(short_mask.sum()),
        'long_wr': round(len(long_pnl[long_pnl > 0]) / max(len(long_pnl), 1), 3),
        'short_wr': round(len(short_pnl[short_pnl > 0]) / max(len(short_pnl), 1), 3),
        'regime': regime_metrics,
        'green_sharpe': green_sharpe,
        'red_sharpe': red_sharpe,
        'regime_gap': regime_gap,
        'regime_pass': regime_pass,
    }


def main():
    print("=" * 80)
    print("TIME/DAY FILTER ANALYSIS — Champion Strategy (210 trades)")
    print("=" * 80)

    df = pd.read_csv(TRADE_CSV)
    print(f"Loaded {len(df)} trades, {df['date'].nunique()} unique dates")
    print(f"Baseline PnL: {df['pnl_ticks'].sum():.1f} ticks, WR: {len(df[df['pnl_ticks']>0])/len(df):.3f}")
    print()

    # Define filter configs
    filters = {
        'BASELINE (no filter)': df,
        'Morning (<1PM ET)': df[df['fill_hour_et'] < 13],
        'Morning (<12PM ET)': df[df['fill_hour_et'] < 12],
        'Morning (<2PM ET)': df[df['fill_hour_et'] < 14],
        'No Friday': df[df['day_of_week'] != 4],
        'Mon-Wed only': df[df['day_of_week'] <= 2],
        'Mon-Thu (no Fri)': df[df['day_of_week'] <= 3],
        'Morning(<1PM) + No Fri': df[(df['fill_hour_et'] < 13) & (df['day_of_week'] != 4)],
        'Morning(<1PM) + Mon-Wed': df[(df['fill_hour_et'] < 13) & (df['day_of_week'] <= 2)],
        'Morning(<12PM) + No Fri': df[(df['fill_hour_et'] < 12) & (df['day_of_week'] != 4)],
        'Morning(<2PM) + No Fri': df[(df['fill_hour_et'] < 14) & (df['day_of_week'] != 4)],
        'Tue-Thu only': df[df['day_of_week'].isin([1, 2, 3])],
        'Mon-Wed + Morning(<1PM) + No Fri': df[(df['fill_hour_et'] < 13) & (df['day_of_week'] <= 2)],  # same as Mon-Wed Morning
    }

    results = []
    for label, subset in filters.items():
        if len(subset) < 10:
            print(f"  SKIP {label}: only {len(subset)} trades")
            continue
        m = compute_metrics(subset, label)
        results.append(m)

    # Sort by Sharpe descending
    results.sort(key=lambda x: x.get('sharpe') or -999, reverse=True)

    # Print results table
    print()
    print("=" * 140)
    print(f"{'Config':<30} {'Trades':>6} {'Days':>5} {'WR':>6} {'PF':>5} {'Sharpe':>7} {'Sortino':>8} "
          f"{'PnL(t)':>8} {'MaxDD(t)':>9} {'DayConc':>8} "
          f"{'Grn_Sh':>7} {'Red_Sh':>7} {'Gap':>6} {'PASS':>5}")
    print("-" * 140)

    for m in results:
        rp = m.get('regime_pass')
        if rp is None:
            regime_pass_str = '?'
        elif rp:
            regime_pass_str = 'YES'
        else:
            regime_pass_str = 'NO'

        print(f"{m['name']:<30} {m['n_trades']:>6} {m['n_trading_days']:>5} "
              f"{m['wr']:>6.3f} {str(m['pf']):>5} {m.get('sharpe', '?'):>7} {m.get('sortino', '?'):>8} "
              f"{m['total_pnl_ticks']:>8.1f} {m['max_dd_ticks']:>9.1f} {m['day_concentration']:>8.3f} "
              f"{m.get('green_sharpe', '?'):>7} {m.get('red_sharpe', '?'):>7} "
              f"{m.get('regime_gap', '?'):>6} {regime_pass_str:>5}")

    print("=" * 140)

    # Detailed regime breakdown for passing configs
    print("\n\n" + "=" * 80)
    print("DETAILED REGIME BREAKDOWN — Configs that PASS regime gate (gap < 0.50)")
    print("=" * 80)

    passing = [m for m in results if m.get('regime_pass')]
    if not passing:
        print("  NO CONFIGS PASS THE REGIME GATE!")
    else:
        for m in passing:
            print(f"\n--- {m['name']} ---")
            print(f"  Overall: {m['n_trades']} trades, {m['n_trading_days']} days, "
                  f"Sharpe={m['sharpe']}, Sortino={m['sortino']}, WR={m['wr']}, PF={m['pf']}")
            print(f"  PnL: {m['total_pnl_ticks']:.1f} ticks (${m['total_pnl_dollars']:.0f}), "
                  f"MaxDD: {m['max_dd_ticks']:.1f}t (${m['max_dd_dollars']:.0f})")
            print(f"  Day concentration: {m['day_concentration']:.3f} (cap: 0.70)")
            print(f"  Long: {m['n_long']} trades (WR {m['long_wr']:.3f}), "
                  f"Short: {m['n_short']} trades (WR {m['short_wr']:.3f})")
            for reg in ['green', 'red', 'flat']:
                if reg in m['regime']:
                    r = m['regime'][reg]
                    print(f"  {reg.upper():>6}: {r['n_trades']:>3} trades, {r['n_days']:>3} days, "
                          f"Sharpe={r['sharpe']}, WR={r['wr']:.3f}, "
                          f"MeanPnL={r['mean_pnl']:.2f}t, TotalPnL={r['total_pnl']:.1f}t")
            print(f"  Regime gap: {m['regime_gap']:.3f} {'PASS' if m['regime_pass'] else 'FAIL'}")

    # Detailed regime breakdown for FAILING configs (for comparison)
    print("\n\n" + "=" * 80)
    print("FAILING CONFIGS — Regime gap > 0.50")
    print("=" * 80)
    failing = [m for m in results if m.get('regime_pass') is not None and not m.get('regime_pass')]
    for m in failing:
        print(f"  {m['name']:<30} Sharpe={m['sharpe']}, Gap={m['regime_gap']}, "
              f"Green={m.get('green_sharpe')}, Red={m.get('red_sharpe')}")

    # Save full results
    output_path = OUTPUT_DIR / 'timeday_filter_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'run_time': pd.Timestamp.now().isoformat(),
            'n_champion_trades': len(df),
            'results': results,
            'passing_configs': [m['name'] for m in passing],
            'best_passing': passing[0] if passing else None,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Summary recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    if passing:
        best = passing[0]
        print(f"Best regime-passing config: {best['name']}")
        print(f"  Sharpe {best['sharpe']}, Sortino {best['sortino']}, WR {best['wr']}, PF {best['pf']}")
        print(f"  {best['n_trades']} trades over {best['n_trading_days']} days")
        print(f"  Regime gap: {best['regime_gap']:.3f} (threshold: 0.50)")
        print(f"  PnL: {best['total_pnl_ticks']:.1f} ticks (${best['total_pnl_dollars']:.0f})")
    else:
        print("No configs pass the regime gate. The base champion (gap 0.121) is the best option.")


if __name__ == '__main__':
    main()
