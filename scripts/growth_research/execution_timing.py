#!/usr/bin/env python3
"""
Execution Timing Analysis for Vol-Adjusted System
===================================================
Tests whether timing of DCA deployment and regime switches matters.

Questions:
1. Best day of week for DCA deployment?
2. Best time of month (1st week, 2nd week, etc)?
3. Does executing regime switches at open vs close matter?
4. Monday morning vs Friday close for vol checks?
5. Intraday timing (open vs close) for entries?

We simulate daily granularity since we don't have intraday data for ETFs.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/timing'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL = 500
WEEKLY_DCA = 100


def download_data():
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT']
    data = yf.download(tickers, start='2012-01-01', period='max',
                       auto_adjust=True, threads=True, progress=False)

    # Get OHLC for SPY to test open vs close
    spy_ohlc = yf.download('SPY', start='2012-01-01', period='max',
                           auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data
    if hasattr(closes.columns, 'droplevel'):
        try:
            closes.columns = closes.columns.droplevel(1)
        except:
            pass

    closes = closes.dropna(how='all').dropna(subset=['SPY', 'UPRO'])
    print(f"  Data: {len(closes)} days")
    return closes, spy_ohlc


def simulate_vol_adjusted(closes, dca_day=None, dca_week_of_month=None,
                          check_day=None, name=""):
    """
    dca_day: 0=Mon, 1=Tue, ..., 4=Fri (when to deploy DCA cash)
    dca_week_of_month: 1-4 (which week of month to deploy)
    check_day: when to check vol/regime (0=Mon..4=Fri), None=daily
    """
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
    pending_dca = 0  # DCA waiting to be deployed

    daily_values = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]
        dow = date.dayofweek  # 0=Mon, 4=Fri
        wom = (date.day - 1) // 7 + 1  # week of month 1-4

        # Accumulate DCA weekly
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            pending_dca += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        # Deploy DCA on specific day
        deploy_today = False
        if dca_day is not None:
            if dow == dca_day:
                deploy_today = True
        elif dca_week_of_month is not None:
            if wom == dca_week_of_month and dow == 0:  # Monday of target week
                deploy_today = True
        else:
            deploy_today = True  # Deploy immediately

        if deploy_today and pending_dca > 0:
            cash += pending_dca
            pending_dca = 0

        # Apply returns
        for ticker in list(holdings.keys()):
            if ticker in returns.columns:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    holdings[ticker] *= (1 + r)

        # Check regime (optionally only on specific day)
        do_check = True
        if check_day is not None and dow != check_day:
            do_check = False

        if do_check:
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
        else:
            # Still deploy cash into current holdings
            if cash > 50 and holdings:
                total_h = sum(holdings.values())
                if total_h > 0:
                    for t in holdings:
                        holdings[t] += cash * (holdings[t] / total_h)
                    cash = 0

        portfolio_val = cash + pending_dca + sum(holdings.values())
        daily_values.append(portfolio_val)

    return pd.Series(daily_values, index=closes.index[warmup:]), total_contributed


def compute_metrics(portfolio, total_contributed):
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
    max_dd = ((portfolio - peak) / peak).min()
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
    print("EXECUTION TIMING ANALYSIS — VOL-ADJUSTED SYSTEM")
    print("="*70)

    closes, spy_ohlc = download_data()

    # --- Day-of-week returns for SPY and UPRO ---
    print("\n  DAY-OF-WEEK RETURNS (annualized):")
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    for ticker in ['SPY', 'UPRO']:
        if ticker in closes.columns:
            ret = closes[ticker].pct_change()
            print(f"    {ticker}:")
            for d in range(5):
                mask = closes.index.dayofweek == d
                day_ret = ret[mask].dropna()
                ann = day_ret.mean() * 252
                vol = day_ret.std() * np.sqrt(252)
                sr = ann / vol if vol > 0 else 0
                wr = (day_ret > 0).mean() * 100
                print(f"      {day_names[d]}: ann {ann*100:>6.1f}%, vol {vol*100:>5.1f}%, "
                      f"Sharpe {sr:>5.2f}, WR {wr:.1f}%")

    # --- Week-of-month returns ---
    print(f"\n  WEEK-OF-MONTH RETURNS (SPY, annualized):")
    spy_ret = closes['SPY'].pct_change()
    for wom in range(1, 5):
        mask = ((closes.index.day - 1) // 7 + 1) == wom
        wom_ret = spy_ret[mask].dropna()
        ann = wom_ret.mean() * 252
        vol = wom_ret.std() * np.sqrt(252)
        sr = ann / vol if vol > 0 else 0
        print(f"    Week {wom}: ann {ann*100:>6.1f}%, vol {vol*100:>5.1f}%, Sharpe {sr:>5.2f}")

    # --- DCA day-of-week test ---
    print(f"\n  DCA DEPLOYMENT DAY TEST:")
    results_dow = {}
    for d in range(5):
        portfolio, total = simulate_vol_adjusted(closes, dca_day=d, name=f"DCA {day_names[d]}")
        m = compute_metrics(portfolio, total)
        if m:
            results_dow[day_names[d]] = m
            print(f"    {day_names[d]}: ${m['final_value']:,.0f} | Sharpe {m['sharpe']:.4f} | MaxDD {m['max_dd']:.1f}%")

    # Baseline (deploy immediately)
    portfolio, total = simulate_vol_adjusted(closes, name="Immediate")
    m_base = compute_metrics(portfolio, total)
    if m_base:
        results_dow['Immediate'] = m_base
        print(f"    Immediate: ${m_base['final_value']:,.0f} | Sharpe {m_base['sharpe']:.4f} | MaxDD {m_base['max_dd']:.1f}%")

    # --- Week-of-month DCA test ---
    print(f"\n  DCA WEEK-OF-MONTH TEST:")
    results_wom = {}
    for wom in range(1, 5):
        portfolio, total = simulate_vol_adjusted(closes, dca_week_of_month=wom, name=f"Week {wom}")
        m = compute_metrics(portfolio, total)
        if m:
            results_wom[f'Week {wom}'] = m
            print(f"    Week {wom}: ${m['final_value']:,.0f} | Sharpe {m['sharpe']:.4f}")

    # --- Vol check frequency ---
    print(f"\n  VOL CHECK DAY TEST (check only on specific day):")
    results_check = {}
    for d in range(5):
        portfolio, total = simulate_vol_adjusted(closes, check_day=d, name=f"Check {day_names[d]}")
        m = compute_metrics(portfolio, total)
        if m:
            results_check[f'Check {day_names[d]}'] = m
            print(f"    Check {day_names[d]}: ${m['final_value']:,.0f} | Sharpe {m['sharpe']:.4f} | MaxDD {m['max_dd']:.1f}%")

    portfolio, total = simulate_vol_adjusted(closes, name="Check Daily")
    m_daily = compute_metrics(portfolio, total)
    if m_daily:
        results_check['Check Daily'] = m_daily
        print(f"    Check Daily: ${m_daily['final_value']:,.0f} | Sharpe {m_daily['sharpe']:.4f} | MaxDD {m_daily['max_dd']:.1f}%")

    # --- Open vs Close execution ---
    print(f"\n  OPEN vs CLOSE EXECUTION (SPY):")
    if isinstance(spy_ohlc.columns, pd.MultiIndex):
        try:
            spy_open = spy_ohlc[('Open', 'SPY')]
            spy_close = spy_ohlc[('Close', 'SPY')]
        except:
            spy_open = spy_ohlc['Open'].iloc[:, 0] if isinstance(spy_ohlc['Open'], pd.DataFrame) else spy_ohlc['Open']
            spy_close = spy_ohlc['Close'].iloc[:, 0] if isinstance(spy_ohlc['Close'], pd.DataFrame) else spy_ohlc['Close']
    else:
        spy_open = spy_ohlc['Open']
        spy_close = spy_ohlc['Close']

    # Open-to-close (day session) vs close-to-open (overnight)
    day_ret = (spy_close / spy_open - 1).dropna()
    overnight_ret = (spy_open / spy_close.shift(1) - 1).dropna()

    day_ann = day_ret.mean() * 252
    overnight_ann = overnight_ret.mean() * 252
    print(f"    Day session (open→close): {day_ann*100:.1f}% annualized")
    print(f"    Overnight (close→open): {overnight_ann*100:.1f}% annualized")
    print(f"    Day Sharpe: {day_ann / (day_ret.std() * np.sqrt(252)):.3f}")
    print(f"    Overnight Sharpe: {overnight_ann / (overnight_ret.std() * np.sqrt(252)):.3f}")

    # --- Spread analysis ---
    print(f"\n  SUMMARY:")

    # DCA day spread
    if results_dow:
        vals = {k: v['final_value'] for k, v in results_dow.items() if k != 'Immediate'}
        best_day = max(vals, key=vals.get)
        worst_day = min(vals, key=vals.get)
        spread = vals[best_day] - vals[worst_day]
        spread_pct = spread / vals[worst_day] * 100
        print(f"    DCA day spread: ${spread:,.0f} ({spread_pct:.1f}%) — best={best_day}, worst={worst_day}")

    # Check day spread
    if results_check:
        vals = {k: v['final_value'] for k, v in results_check.items() if k != 'Check Daily'}
        best_check = max(vals, key=vals.get)
        worst_check = min(vals, key=vals.get)
        spread = vals[best_check] - vals[worst_check]
        spread_pct = spread / vals[worst_check] * 100
        print(f"    Vol check day spread: ${spread:,.0f} ({spread_pct:.1f}%) — best={best_check}, worst={worst_check}")
        print(f"    Daily check vs best weekly: ${results_check.get('Check Daily', {}).get('final_value', 0) - vals[best_check]:+,.0f}")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'dca_day': results_dow,
        'dca_week': results_wom,
        'check_day': results_check,
    }
    with open(os.path.join(OUTPUT_DIR, 'timing_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
