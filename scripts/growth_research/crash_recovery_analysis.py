#!/usr/bin/env python3
"""
Crash Recovery Analysis
========================
Analyzes how our vol-adjusted strategy behaves during and after drawdowns:

1. Recovery time from various drawdown depths
2. Should we be more aggressive in recovery (buy the dip)?
3. V-shape vs L-shape recovery patterns
4. Does adding back leverage quickly after a crash improve returns?
5. Comparison with buy-and-hold recovery times
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/crash_recovery'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_data():
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT', 'VIXY']
    data = yf.download(tickers, start='2010-01-01', period='max',
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
    closes = closes.dropna(how='all').dropna(subset=['UPRO', 'SPY'])
    return closes

def simulate_vol_adjusted(closes, vol_low=0.20, vol_high=0.30, recovery_mode=None):
    """
    recovery_mode options:
    - None: standard vol-adjusted
    - 'aggressive': after >10% DD, use UPRO even at higher vol (25% threshold)
    - 'conservative': after >10% DD, use SPY until new high
    - 'double_down': after >15% DD, go 100% UPRO regardless of vol
    """
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    spy_sma50 = spy.rolling(50).mean()
    returns = closes.pct_change().fillna(0)

    warmup = 63
    portfolio_val = 10000.0  # Start bigger for better DD analysis
    holdings = {}
    cash = 10000.0
    last_regime = None
    peak_val = portfolio_val

    daily_values = []
    daily_dates = []
    dd_events = []
    in_recovery = False
    dd_start = None
    dd_depth = 0

    for i in range(warmup, len(closes)):
        date = closes.index[i]

        # Update
        for ticker in list(holdings.keys()):
            if ticker in returns.columns:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    holdings[ticker] *= (1 + r)

        portfolio_val = cash + sum(holdings.values())

        # Track drawdown
        if portfolio_val > peak_val:
            if in_recovery and dd_start is not None:
                # Recovery complete — log event
                dd_events.append({
                    'start': dd_start,
                    'trough': dd_trough_date,
                    'recovery': date,
                    'depth': dd_depth,
                    'recovery_days': (date - dd_trough_date).days,
                    'total_days': (date - dd_start).days,
                })
            peak_val = portfolio_val
            in_recovery = False
            dd_start = None

        current_dd = (portfolio_val - peak_val) / peak_val

        if current_dd < -0.05 and dd_start is None:
            dd_start = date
            dd_depth = current_dd
            dd_trough_date = date
            in_recovery = True

        if in_recovery and current_dd < dd_depth:
            dd_depth = current_dd
            dd_trough_date = date

        # Vol regime
        vol = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 0.15
        protection = spy.iloc[i] > spy_sma50.iloc[i] if not np.isnan(spy_sma50.iloc[i]) else True

        # Adjust thresholds based on recovery mode
        vl = vol_low
        vh = vol_high

        if recovery_mode == 'aggressive' and in_recovery and dd_depth < -0.10:
            vl = 0.25  # Stay in UPRO at higher vol during recovery
        elif recovery_mode == 'conservative' and in_recovery:
            vl = 0.10  # Switch to SPY at lower vol during recovery
            vh = 0.20
        elif recovery_mode == 'double_down' and in_recovery and dd_depth < -0.15:
            vl = 0.50  # Always UPRO during deep recovery
            vh = 0.60

        if not protection:
            regime = 'CASH'
            target = {'SPY': 1.0}
        elif vol < vl:
            regime = 'UPRO'
            target = {'UPRO': 1.0}
        elif vol < vh:
            regime = 'SPY'
            target = {'SPY': 1.0}
        else:
            regime = 'SAFE'
            target = {'GLD': 0.5, 'TLT': 0.5}

        if regime != last_regime:
            total_val = cash + sum(holdings.values())
            holdings = {t: total_val * w for t, w in target.items() if w > 0}
            cash = 0
            last_regime = regime

        portfolio_val = cash + sum(holdings.values())
        daily_values.append(portfolio_val)
        daily_dates.append(date)

    return pd.Series(daily_values, index=daily_dates), dd_events

def compute_metrics(portfolio):
    r = portfolio.pct_change().dropna()
    if len(r) < 63:
        return None
    years = len(r) / 252
    final = portfolio.iloc[-1]
    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    neg = r[r < 0]
    downside_vol = neg.std() * np.sqrt(252) if len(neg) > 0 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0
    peak = portfolio.expanding().max()
    dd = (portfolio - peak) / peak
    max_dd = dd.min()
    cagr = (final / portfolio.iloc[0]) ** (1/years) - 1
    return {
        'final_value': float(final),
        'cagr': float(cagr * 100),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd * 100),
    }

def main():
    print("="*70)
    print("CRASH RECOVERY ANALYSIS")
    print("="*70)

    closes = download_data()
    print(f"  Data: {len(closes)} days, {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")

    # --- 1. Baseline drawdown events ---
    print("\n1. DRAWDOWN EVENTS (Vol-Adjusted Strategy)")
    print("-"*50)

    portfolio, dd_events = simulate_vol_adjusted(closes)

    if dd_events:
        print(f"\n  Found {len(dd_events)} drawdown events (>5% from peak):\n")
        print(f"  {'Start':>12s} {'Trough':>12s} {'Recovery':>12s} {'Depth':>7s} {'To Trough':>10s} {'To Recovery':>12s}")
        print("  " + "-"*70)

        depths = []
        recovery_days = []
        total_days = []

        for ev in dd_events:
            start_str = ev['start'].strftime('%Y-%m-%d')
            trough_str = ev['trough'].strftime('%Y-%m-%d')
            recovery_str = ev['recovery'].strftime('%Y-%m-%d')
            trough_days = (ev['trough'] - ev['start']).days
            print(f"  {start_str} {trough_str} {recovery_str} {ev['depth']*100:>+6.1f}% "
                  f"{trough_days:>8d}d {ev['total_days']:>10d}d")
            depths.append(ev['depth'])
            recovery_days.append(ev['recovery_days'])
            total_days.append(ev['total_days'])

        print(f"\n  Average depth: {np.mean(depths)*100:.1f}%")
        print(f"  Average recovery time (from trough): {np.mean(recovery_days):.0f} days")
        print(f"  Average total underwater: {np.mean(total_days):.0f} days")
        print(f"  Worst depth: {min(depths)*100:.1f}%")
        print(f"  Longest recovery: {max(total_days)} days")

    # --- 2. Compare UPRO buy-and-hold drawdowns ---
    print("\n\n2. COMPARISON: VOL-ADJUSTED vs UPRO BUY-AND-HOLD")
    print("-"*50)

    upro = closes['UPRO']
    upro_peak = upro.expanding().max()
    upro_dd = (upro - upro_peak) / upro_peak

    # Find UPRO drawdown events
    upro_events = []
    in_dd = False
    dd_start = None
    dd_depth = 0
    dd_trough_date = None

    for i in range(len(upro)):
        date = upro.index[i]
        dd = upro_dd.iloc[i]

        if dd >= 0 and in_dd:
            upro_events.append({
                'start': dd_start,
                'trough': dd_trough_date,
                'recovery': date,
                'depth': dd_depth,
                'total_days': (date - dd_start).days,
            })
            in_dd = False
            dd_start = None

        if dd < -0.10 and not in_dd:
            in_dd = True
            dd_start = date
            dd_depth = dd
            dd_trough_date = date

        if in_dd and dd < dd_depth:
            dd_depth = dd
            dd_trough_date = date

    print(f"\n  UPRO drawdowns >10%: {len(upro_events)}")
    for ev in upro_events:
        print(f"    {ev['start'].strftime('%Y-%m-%d')}: depth {ev['depth']*100:+.1f}%, "
              f"recovery {ev['total_days']}d")

    if dd_events:
        vol_avg_depth = np.mean([abs(e['depth']) for e in dd_events]) * 100
        vol_avg_recovery = np.mean([e['total_days'] for e in dd_events])

        upro_avg_depth = np.mean([abs(e['depth']) for e in upro_events]) * 100 if upro_events else 0
        upro_avg_recovery = np.mean([e['total_days'] for e in upro_events]) if upro_events else 0

        print(f"\n  Vol-adjusted: avg depth {vol_avg_depth:.1f}%, avg recovery {vol_avg_recovery:.0f} days")
        print(f"  UPRO B&H:     avg depth {upro_avg_depth:.1f}%, avg recovery {upro_avg_recovery:.0f} days")

    # --- 3. Recovery mode comparison ---
    print("\n\n3. RECOVERY MODE COMPARISON")
    print("-"*50)

    modes = {
        'Standard (no adjustment)': None,
        'Aggressive (stay UPRO longer)': 'aggressive',
        'Conservative (stay SPY)': 'conservative',
        'Double down (UPRO always in deep DD)': 'double_down',
    }

    mode_results = {}
    for name, mode in modes.items():
        portfolio, events = simulate_vol_adjusted(closes, recovery_mode=mode)
        m = compute_metrics(portfolio)
        if m:
            m['n_dd_events'] = len(events)
            m['avg_recovery'] = np.mean([e['total_days'] for e in events]) if events else 0
            mode_results[name] = m

    print(f"\n  {'Mode':<40s} {'Final $':>10s} {'Sharpe':>7s} {'MaxDD':>7s} {'CAGR':>7s} {'Avg Recovery':>12s}")
    print("  " + "-"*86)
    for name, m in sorted(mode_results.items(), key=lambda x: x[1]['final_value'], reverse=True):
        print(f"  {name:<40s} ${m['final_value']:>9,.0f} {m['sharpe']:>7.3f} {m['max_dd']:>6.1f}% "
              f"{m['cagr']:>6.1f}% {m['avg_recovery']:>10.0f}d")

    # --- 4. After-crash analysis ---
    print("\n\n4. POST-CRASH RETURN ANALYSIS (SPY)")
    print("-"*50)

    spy = closes['SPY']
    spy_ret = spy.pct_change()
    spy_peak = spy.expanding().max()
    spy_dd = (spy - spy_peak) / spy_peak

    # Find dates when SPY recovers from >10% drawdown
    recovery_dates = []
    in_dd = False
    for i in range(1, len(spy)):
        if spy_dd.iloc[i] < -0.10 and not in_dd:
            in_dd = True
        if spy_dd.iloc[i] >= 0 and in_dd:
            in_dd = False
            recovery_dates.append(spy.index[i])

    print(f"  Found {len(recovery_dates)} SPY recovery points (after >10% DD)")

    # Forward returns after recovery
    fwd_periods = [5, 10, 21, 63, 126, 252]
    print(f"\n  Forward returns after recovery:")
    print(f"  {'Period':>8s} {'Mean':>7s} {'Median':>7s} {'WR':>6s} {'Min':>7s} {'Max':>7s}")
    print("  " + "-"*42)

    for period in fwd_periods:
        fwd_rets = []
        for date in recovery_dates:
            idx = spy.index.get_loc(date)
            if idx + period < len(spy):
                fwd_ret = spy.iloc[idx + period] / spy.iloc[idx] - 1
                fwd_rets.append(fwd_ret)

        if fwd_rets:
            arr = np.array(fwd_rets)
            print(f"  {period:>6d}d {arr.mean()*100:>+6.1f}% {np.median(arr)*100:>+6.1f}% "
                  f"{(arr > 0).mean()*100:>5.0f}% {arr.min()*100:>+6.1f}% {arr.max()*100:>+6.1f}%")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'dd_events': [{k: str(v) if isinstance(v, pd.Timestamp) else v for k, v in e.items()} for e in dd_events],
        'mode_results': mode_results,
    }

    with open(os.path.join(OUTPUT_DIR, 'crash_recovery_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved.")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == '__main__':
    main()
