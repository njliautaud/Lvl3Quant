#!/usr/bin/env python3
"""
Quick validation: No SMA protection vs SMA50 vs EMA50 vs Crossover 20/200
across sub-periods, drawdown events, and regimes.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

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

def simulate(closes, protection_func, start_idx=260, end_idx=None):
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    returns = closes.pct_change().fillna(0)

    if end_idx is None:
        end_idx = len(closes)

    holdings = {}
    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None
    daily_values = []

    for i in range(start_idx, end_idx):
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
        protection = protection_func(spy, i)

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
            holdings = {t: total_val * w for t, w in target.items() if w > 0 and t in closes.columns}
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

    portfolio = pd.Series(daily_values, index=closes.index[start_idx:end_idx])
    return portfolio, total_contributed

def metrics(portfolio):
    r = portfolio.pct_change().dropna()
    if len(r) < 20:
        return None
    years = max(len(r) / 252, 0.1)
    final = portfolio.iloc[-1]
    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    peak = portfolio.expanding().max()
    max_dd = ((portfolio - peak) / peak).min()
    return {'final': final, 'sharpe': sharpe, 'max_dd': max_dd * 100, 'ann_ret': ann_ret * 100}

def main():
    print("="*70)
    print("SMA PROTECTION — SUB-PERIOD VALIDATION")
    print("="*70)

    closes = download_data()
    spy = closes['SPY']
    n = len(closes)

    # Define protection functions
    sma50 = spy.rolling(50).mean()
    ema50 = spy.ewm(span=50, adjust=False).mean()
    sma20 = spy.rolling(20).mean()
    sma200 = spy.rolling(200).mean()

    configs = {
        'No Protection': lambda s, i: True,
        'SMA50': lambda s, i: s.iloc[i] > sma50.iloc[i] if not np.isnan(sma50.iloc[i]) else True,
        'EMA50': lambda s, i: s.iloc[i] > ema50.iloc[i] if not np.isnan(ema50.iloc[i]) else True,
        'Cross 20/200': lambda s, i: (
            sma20.iloc[i] > sma200.iloc[i]
            if not np.isnan(sma20.iloc[i]) and not np.isnan(sma200.iloc[i]) else True
        ),
    }

    # === Sub-period analysis ===
    warmup = 260
    usable = n - warmup
    third = usable // 3

    periods = {
        'Early (2013-2017)': (warmup, warmup + third),
        'Mid (2017-2022)': (warmup + third, warmup + 2*third),
        'Late (2022-2026)': (warmup + 2*third, n),
    }

    print("\n  SUB-PERIOD SHARPE RATIOS:")
    print(f"  {'Period':<22s}", end="")
    for cfg in configs:
        print(f" {cfg:>16s}", end="")
    print()
    print("  " + "-"*82)

    for pname, (start, end) in periods.items():
        print(f"  {pname:<22s}", end="")
        for cname, func in configs.items():
            p, _ = simulate(closes, func, start, end)
            m = metrics(p)
            if m:
                print(f" {m['sharpe']:>16.3f}", end="")
            else:
                print(f" {'N/A':>16s}", end="")
        print()

    # === Drawdown event analysis ===
    # Focus on major drawdowns: COVID (Feb-Mar 2020), Rate Hikes (Jan-Oct 2022),
    # Late 2018, Aug 2015
    print("\n  BEHAVIOR DURING MAJOR DRAWDOWNS:")

    # Find drawdown periods
    spy_peak = spy.expanding().max()
    spy_dd = (spy - spy_peak) / spy_peak

    events = []
    in_dd = False
    dd_start = None
    for i in range(260, len(closes)):
        if spy_dd.iloc[i] < -0.10 and not in_dd:
            dd_start = i
            in_dd = True
        elif spy_dd.iloc[i] > -0.02 and in_dd:
            events.append((dd_start, i, spy_dd.iloc[dd_start:i].min()))
            in_dd = False

    print(f"\n  Found {len(events)} drawdowns > 10%")
    print(f"  {'Event':<30s}", end="")
    for cfg in configs:
        print(f" {cfg:>16s}", end="")
    print()
    print("  " + "-"*98)

    for start, end, depth in events[:6]:
        event_dates = f"{closes.index[start].strftime('%Y-%m')}-{closes.index[end].strftime('%Y-%m')}"
        print(f"  {event_dates:<15s} ({depth*100:.0f}% SPY)", end="")

        for cname, func in configs.items():
            p, _ = simulate(closes, func, start, end)
            m = metrics(p)
            if m:
                print(f" {m['max_dd']:>15.1f}%", end="")
            else:
                print(f" {'N/A':>16s}", end="")
        print()

    # === Time in cash (protection triggered) ===
    print("\n  PROTECTION TRIGGER FREQUENCY:")
    for cname, func in configs.items():
        if cname == 'No Protection':
            print(f"    {cname:<20s}: 0.0% of days in cash")
            continue
        cash_days = 0
        total_days = 0
        for i in range(260, len(closes)):
            total_days += 1
            if not func(spy, i):
                cash_days += 1
        print(f"    {cname:<20s}: {cash_days/total_days*100:.1f}% of days in cash "
              f"({cash_days} days, {cash_days/14:.0f}/yr)")

    # === Full-period comparison ===
    print("\n  FULL PERIOD RESULTS:")
    for cname, func in configs.items():
        p, total = simulate(closes, func)
        m = metrics(p)
        if m:
            print(f"    {cname:<20s}: ${m['final']:>10,.0f} | Sharpe {m['sharpe']:.3f} | "
                  f"MaxDD {m['max_dd']:.1f}% | Ann ret {m['ann_ret']:.1f}%")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
