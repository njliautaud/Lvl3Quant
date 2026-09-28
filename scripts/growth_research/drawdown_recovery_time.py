#!/usr/bin/env python3
"""
Drawdown Recovery Time Analysis
================================
How long does the vol-adjusted system take to recover from drawdowns?
Compare to SPY and naked UPRO.

Also: what's the probability of recovery within N months?
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/recovery_time'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL = 500
WEEKLY_DCA = 100


def download_data():
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT']
    data = yf.download(tickers, start='2012-01-01', period='max',
                       auto_adjust=True, threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data
    if hasattr(closes.columns, 'droplevel'):
        try:
            closes.columns = closes.columns.droplevel(1)
        except:
            pass
    return closes.dropna(how='all').dropna(subset=['SPY', 'UPRO'])


def simulate_vol_adjusted(closes):
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    spy_sma50 = spy.rolling(50).mean()
    returns = closes.pct_change().fillna(0)

    warmup = 63
    holdings = {}
    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None
    daily_values = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        for ticker in list(holdings.keys()):
            if ticker in returns.columns:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    holdings[ticker] *= (1 + r)

        vol = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 0.15
        protection = spy.iloc[i] > spy_sma50.iloc[i] if not np.isnan(spy_sma50.iloc[i]) else True

        if not protection:
            regime = 'CASH'
            target = {'SPY': 1.0}
        elif vol < 0.20:
            regime = 'UPRO'
            target = {'UPRO': 1.0}
        elif vol < 0.30:
            regime = 'SPY'
            target = {'SPY': 1.0}
        else:
            regime = 'SAFE'
            target = {'GLD': 0.5, 'TLT': 0.5}

        if regime != last_regime:
            total_val = cash + sum(holdings.values())
            holdings = {t: total_val * w for t, w in target.items()
                       if w > 0 and t in closes.columns}
            cash = 0
            last_regime = regime
        elif cash > 50 and holdings:
            total_h = sum(holdings.values())
            if total_h > 0:
                for t in holdings:
                    holdings[t] += cash * (holdings[t] / total_h)
                cash = 0

        portfolio_val = cash + sum(holdings.values())
        daily_values.append(portfolio_val)

    return pd.Series(daily_values, index=closes.index[warmup:])


def simulate_dca(closes, ticker):
    """Simple DCA into a single ticker."""
    returns = closes[ticker].pct_change().fillna(0)
    warmup = 63

    holdings = float(INITIAL)
    cash = 0
    last_week = None
    daily_values = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            last_week = week_key

        r = returns.iloc[i]
        if not np.isnan(r):
            holdings *= (1 + r)

        if cash > 0:
            holdings += cash
            cash = 0

        daily_values.append(holdings)

    return pd.Series(daily_values, index=closes.index[warmup:])


def analyze_drawdowns(portfolio, name):
    """Analyze all drawdown events and recovery times."""
    peak = portfolio.expanding().max()
    dd = (portfolio - peak) / peak

    events = []
    in_dd = False
    dd_start = None
    dd_trough = None
    dd_trough_idx = None
    dd_depth = 0

    for i in range(len(dd)):
        if dd.iloc[i] < -0.05 and not in_dd:
            dd_start = i
            in_dd = True
            dd_depth = dd.iloc[i]
            dd_trough = i
            dd_trough_idx = i

        if in_dd:
            if dd.iloc[i] < dd_depth:
                dd_depth = dd.iloc[i]
                dd_trough = i
                dd_trough_idx = i

            if dd.iloc[i] >= 0:
                # Recovery complete
                recovery_days = i - dd_trough
                total_days = i - dd_start
                events.append({
                    'start_date': portfolio.index[dd_start],
                    'trough_date': portfolio.index[dd_trough],
                    'recovery_date': portfolio.index[i],
                    'depth': dd_depth * 100,
                    'drawdown_days': dd_trough - dd_start,
                    'recovery_days': recovery_days,
                    'total_days': total_days,
                })
                in_dd = False
                dd_depth = 0

    # Handle ongoing drawdown
    if in_dd:
        events.append({
            'start_date': portfolio.index[dd_start],
            'trough_date': portfolio.index[dd_trough],
            'recovery_date': None,
            'depth': dd_depth * 100,
            'drawdown_days': dd_trough - dd_start,
            'recovery_days': len(dd) - dd_trough,
            'total_days': len(dd) - dd_start,
        })

    return events


def main():
    print("="*70)
    print("DRAWDOWN RECOVERY TIME ANALYSIS")
    print("="*70)

    closes = download_data()
    print(f"  Data: {len(closes)} days")

    # Generate portfolios
    print("\n  Simulating strategies...")
    vol_adj = simulate_vol_adjusted(closes)
    spy_dca = simulate_dca(closes, 'SPY')
    upro_dca = simulate_dca(closes, 'UPRO')

    portfolios = {
        'Vol-Adjusted': vol_adj,
        'SPY DCA': spy_dca,
        'UPRO DCA': upro_dca,
    }

    for name, pf in portfolios.items():
        events = analyze_drawdowns(pf, name)
        completed = [e for e in events if e['recovery_date'] is not None]
        ongoing = [e for e in events if e['recovery_date'] is None]

        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")
        print(f"  Total drawdowns (>5%): {len(events)}")
        print(f"  Completed recoveries: {len(completed)}")
        print(f"  Ongoing: {len(ongoing)}")

        if completed:
            depths = [abs(e['depth']) for e in completed]
            recovery_days = [e['recovery_days'] for e in completed]
            total_days = [e['total_days'] for e in completed]

            print(f"\n  DRAWDOWN DEPTH DISTRIBUTION:")
            print(f"    Mean: {np.mean(depths):.1f}%")
            print(f"    Median: {np.median(depths):.1f}%")
            print(f"    Max: {max(depths):.1f}%")
            print(f"    Min: {min(depths):.1f}%")

            print(f"\n  RECOVERY TIME (trading days):")
            print(f"    Mean: {np.mean(recovery_days):.0f} days ({np.mean(recovery_days)/21:.1f} months)")
            print(f"    Median: {np.median(recovery_days):.0f} days ({np.median(recovery_days)/21:.1f} months)")
            print(f"    Max: {max(recovery_days)} days ({max(recovery_days)/21:.1f} months)")
            print(f"    Min: {min(recovery_days)} days")

            print(f"\n  TOTAL UNDERWATER TIME:")
            print(f"    Mean: {np.mean(total_days):.0f} days ({np.mean(total_days)/21:.1f} months)")
            print(f"    Median: {np.median(total_days):.0f} days ({np.median(total_days)/21:.1f} months)")
            print(f"    Max: {max(total_days)} days ({max(total_days)/21:.1f} months)")

            # Recovery time by depth bucket
            print(f"\n  RECOVERY TIME BY DEPTH:")
            buckets = [(5, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
            for lo, hi in buckets:
                bucket_events = [e for e in completed if lo <= abs(e['depth']) < hi]
                if bucket_events:
                    avg_rec = np.mean([e['recovery_days'] for e in bucket_events])
                    avg_total = np.mean([e['total_days'] for e in bucket_events])
                    print(f"    {lo}-{hi}% depth: {len(bucket_events)} events, "
                          f"avg recovery {avg_rec:.0f} days ({avg_rec/21:.1f}mo), "
                          f"avg total underwater {avg_total:.0f} days ({avg_total/21:.1f}mo)")

            # All events detail
            print(f"\n  ALL DRAWDOWN EVENTS:")
            print(f"  {'Start':<12s} {'Trough':<12s} {'Recovery':<12s} {'Depth':>7s} "
                  f"{'DD days':>8s} {'Rec days':>9s} {'Total':>7s}")
            print("  " + "-"*72)
            for e in sorted(events, key=lambda x: x['depth']):
                rec_str = e['recovery_date'].strftime('%Y-%m-%d') if e['recovery_date'] else 'ONGOING'
                print(f"  {e['start_date'].strftime('%Y-%m-%d'):<12s} "
                      f"{e['trough_date'].strftime('%Y-%m-%d'):<12s} "
                      f"{rec_str:<12s} {e['depth']:>6.1f}% "
                      f"{e['drawdown_days']:>8d} {e['recovery_days']:>9d} {e['total_days']:>7d}")

    # --- Cross-strategy comparison ---
    print(f"\n{'='*70}")
    print("CROSS-STRATEGY COMPARISON")
    print(f"{'='*70}")

    print(f"\n  {'Metric':<30s} {'Vol-Adjusted':>14s} {'SPY DCA':>14s} {'UPRO DCA':>14s}")
    print("  " + "-"*74)

    for name, pf in portfolios.items():
        events = analyze_drawdowns(pf, name)
        completed = [e for e in events if e['recovery_date'] is not None]

    # Compute side by side
    metrics = {}
    for name, pf in portfolios.items():
        events = analyze_drawdowns(pf, name)
        completed = [e for e in events if e['recovery_date'] is not None]
        if completed:
            metrics[name] = {
                'num_dd': len(events),
                'avg_depth': np.mean([abs(e['depth']) for e in completed]),
                'max_depth': max([abs(e['depth']) for e in completed]),
                'avg_rec_days': np.mean([e['recovery_days'] for e in completed]),
                'med_rec_days': np.median([e['recovery_days'] for e in completed]),
                'max_rec_days': max([e['recovery_days'] for e in completed]),
                'avg_total_days': np.mean([e['total_days'] for e in completed]),
                'pct_underwater': sum(e['total_days'] for e in events) / len(pf) * 100,
            }

    metric_names = [
        ('Number of drawdowns', 'num_dd', '{:.0f}'),
        ('Avg depth', 'avg_depth', '{:.1f}%'),
        ('Max depth', 'max_depth', '{:.1f}%'),
        ('Avg recovery (days)', 'avg_rec_days', '{:.0f}'),
        ('Med recovery (days)', 'med_rec_days', '{:.0f}'),
        ('Max recovery (days)', 'max_rec_days', '{:.0f}'),
        ('Avg total underwater', 'avg_total_days', '{:.0f}'),
        ('% of time underwater', 'pct_underwater', '{:.1f}%'),
    ]

    for display_name, key, fmt in metric_names:
        print(f"  {display_name:<30s}", end="")
        for strat_name in ['Vol-Adjusted', 'SPY DCA', 'UPRO DCA']:
            val = metrics.get(strat_name, {}).get(key, 0)
            formatted = fmt.format(val)
            print(f" {formatted:>14s}", end="")
        print()

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'metrics': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else str(vv) for kk, vv in v.items()} for k, v in metrics.items()},
    }
    with open(os.path.join(OUTPUT_DIR, 'recovery_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
