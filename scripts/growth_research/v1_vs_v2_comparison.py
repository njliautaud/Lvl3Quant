#!/usr/bin/env python3
"""
GAMEPLAN v1 vs v2 Comparison
=============================
Compare the original system (SMA50, GLD+TLT safe haven)
vs the improved v2 (20/200 crossover, GLD-only, September hedge).
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/v1_v2_comparison'
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


def simulate(closes, protection_func, safe_haven, sep_hedge=False, earnings_aggressive=False):
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    returns = closes.pct_change().fillna(0)

    warmup = 260  # Max lookback for 200d SMA
    holdings = {}
    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None
    switches = 0
    daily_values = []

    # Earnings months: mid-Jan to mid-Feb, mid-Apr to mid-May, mid-Jul to mid-Aug, mid-Oct to mid-Nov
    def is_earnings(date):
        m, d = date.month, date.day
        return ((m == 1 and d >= 15) or (m == 2 and d <= 15) or
                (m == 4 and d >= 15) or (m == 5 and d <= 15) or
                (m == 7 and d >= 15) or (m == 8 and d <= 15) or
                (m == 10 and d >= 15) or (m == 11 and d <= 15))

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
        protection = protection_func(spy, i)

        # September hedge
        if sep_hedge and date.month == 9 and protection:
            regime = 'SPY'
            target = {'SPY': 1.0}
        elif not protection:
            regime = 'CASH'
            target = {'SPY': 1.0}
        else:
            # Vol thresholds
            low_thresh = 0.20
            high_thresh = 0.30
            if earnings_aggressive and is_earnings(date):
                low_thresh = 0.25
                high_thresh = 0.35

            if vol < low_thresh:
                regime = 'UPRO'
                target = {'UPRO': 1.0}
            elif vol < high_thresh:
                regime = 'SPY'
                target = {'SPY': 1.0}
            else:
                regime = 'SAFE'
                target = safe_haven.copy()

        if regime != last_regime:
            total_val = cash + sum(holdings.values())
            valid = {t: w for t, w in target.items() if t in closes.columns}
            tw = sum(valid.values())
            if tw > 0:
                valid = {t: w/tw for t, w in valid.items()}
            else:
                valid = {'SPY': 1.0}
            holdings = {t: total_val * w for t, w in valid.items() if w > 0}
            cash = 0
            switches += 1
            last_regime = regime
        elif cash > 50 and holdings:
            total_h = sum(holdings.values())
            if total_h > 0:
                for t in holdings:
                    holdings[t] += cash * (holdings[t] / total_h)
                cash = 0

        portfolio_val = cash + sum(holdings.values())
        daily_values.append(portfolio_val)

    portfolio = pd.Series(daily_values, index=closes.index[warmup:])
    return portfolio, total_contributed, switches


def compute_metrics(portfolio, total_contributed, switches):
    r = portfolio.pct_change().dropna()
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

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Time underwater
    underwater = (portfolio < peak).sum() / len(portfolio) * 100

    return {
        'final_value': float(final),
        'profit': float(final - total_contributed),
        'cagr': float(cagr * 100),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd * 100),
        'calmar': float(calmar),
        'ann_vol': float(ann_vol * 100),
        'switches': switches,
        'switches_per_year': switches / years,
        'underwater_pct': float(underwater),
        'total_contributed': float(total_contributed),
    }


def main():
    print("="*70)
    print("GAMEPLAN v1 vs v2 COMPARISON")
    print("="*70)

    closes = download_data()
    spy = closes['SPY']
    print(f"  Data: {len(closes)} days")

    # Protection functions
    sma50 = spy.rolling(50).mean()
    sma20 = spy.rolling(20).mean()
    sma200 = spy.rolling(200).mean()

    prot_sma50 = lambda s, i: s.iloc[i] > sma50.iloc[i] if not np.isnan(sma50.iloc[i]) else True
    prot_cross20_200 = lambda s, i: (
        sma20.iloc[i] > sma200.iloc[i]
        if not np.isnan(sma20.iloc[i]) and not np.isnan(sma200.iloc[i]) else True
    )
    prot_none = lambda s, i: True

    systems = {
        'v1: Original GAMEPLAN': {
            'protection': prot_sma50,
            'safe_haven': {'GLD': 0.5, 'TLT': 0.5},
            'sep_hedge': False,
            'earnings': False,
        },
        'v1 + Earnings': {
            'protection': prot_sma50,
            'safe_haven': {'GLD': 0.5, 'TLT': 0.5},
            'sep_hedge': False,
            'earnings': True,
        },
        'v2: Full GAMEPLAN v2': {
            'protection': prot_cross20_200,
            'safe_haven': {'GLD': 1.0},
            'sep_hedge': True,
            'earnings': True,
        },
        'v2 no Sep hedge': {
            'protection': prot_cross20_200,
            'safe_haven': {'GLD': 1.0},
            'sep_hedge': False,
            'earnings': True,
        },
        'Vol-only (no SMA)': {
            'protection': prot_none,
            'safe_haven': {'GLD': 1.0},
            'sep_hedge': True,
            'earnings': True,
        },
        'SPY Buy & Hold': {
            'protection': prot_none,
            'safe_haven': {'SPY': 1.0},
            'sep_hedge': False,
            'earnings': False,
        },
        'UPRO Buy & Hold': {
            'protection': prot_none,
            'safe_haven': {'UPRO': 1.0},
            'sep_hedge': False,
            'earnings': False,
        },
    }

    # Override for buy & hold
    results = {}

    for name, config in systems.items():
        if name == 'SPY Buy & Hold':
            # Simple SPY DCA
            returns = closes['SPY'].pct_change().fillna(0)
            warmup = 260
            holdings = float(INITIAL)
            cash = 0
            total_contributed = float(INITIAL)
            last_week = None
            daily_values = []
            for i in range(warmup, len(closes)):
                date = closes.index[i]
                week_key = (date.year, date.isocalendar()[1])
                if week_key != last_week:
                    cash += WEEKLY_DCA
                    total_contributed += WEEKLY_DCA
                    last_week = week_key
                holdings *= (1 + returns.iloc[i])
                if cash > 0:
                    holdings += cash
                    cash = 0
                daily_values.append(holdings)
            portfolio = pd.Series(daily_values, index=closes.index[warmup:])
            m = compute_metrics(portfolio, total_contributed, 0)
        elif name == 'UPRO Buy & Hold':
            returns = closes['UPRO'].pct_change().fillna(0)
            warmup = 260
            holdings = float(INITIAL)
            cash = 0
            total_contributed = float(INITIAL)
            last_week = None
            daily_values = []
            for i in range(warmup, len(closes)):
                date = closes.index[i]
                week_key = (date.year, date.isocalendar()[1])
                if week_key != last_week:
                    cash += WEEKLY_DCA
                    total_contributed += WEEKLY_DCA
                    last_week = week_key
                holdings *= (1 + returns.iloc[i])
                if cash > 0:
                    holdings += cash
                    cash = 0
                daily_values.append(holdings)
            portfolio = pd.Series(daily_values, index=closes.index[warmup:])
            m = compute_metrics(portfolio, total_contributed, 0)
        else:
            portfolio, total, switches = simulate(
                closes,
                config['protection'],
                config['safe_haven'],
                config['sep_hedge'],
                config['earnings'],
            )
            m = compute_metrics(portfolio, total, switches)

        results[name] = m
        print(f"  {name:<30s}: ${m['final_value']:>10,.0f} | Sharpe {m['sharpe']:.3f} | "
              f"MaxDD {m['max_dd']:.1f}% | SW/yr {m['switches_per_year']:.1f}")

    # --- Comparison table ---
    print(f"\n{'='*70}")
    print("DETAILED COMPARISON")
    print(f"{'='*70}")

    metrics_to_show = [
        ('Final Value', 'final_value', '${:>10,.0f}'),
        ('Profit', 'profit', '${:>10,.0f}'),
        ('CAGR', 'cagr', '{:>9.1f}%'),
        ('Sharpe', 'sharpe', '{:>10.3f}'),
        ('Sortino', 'sortino', '{:>10.3f}'),
        ('Max Drawdown', 'max_dd', '{:>9.1f}%'),
        ('Calmar', 'calmar', '{:>10.3f}'),
        ('Ann Volatility', 'ann_vol', '{:>9.1f}%'),
        ('Switches/yr', 'switches_per_year', '{:>10.1f}'),
        ('% Underwater', 'underwater_pct', '{:>9.1f}%'),
        ('Contributed', 'total_contributed', '${:>10,.0f}'),
    ]

    key_systems = ['v1: Original GAMEPLAN', 'v2: Full GAMEPLAN v2', 'SPY Buy & Hold', 'UPRO Buy & Hold']

    print(f"\n  {'Metric':<18s}", end="")
    for sys in key_systems:
        short = sys.split(':')[0] if ':' in sys else sys[:15]
        print(f" {short:>16s}", end="")
    print()
    print("  " + "-"*82)

    for display_name, key, fmt in metrics_to_show:
        print(f"  {display_name:<18s}", end="")
        for sys in key_systems:
            val = results[sys][key]
            formatted = fmt.format(val)
            print(f" {formatted:>16s}", end="")
        print()

    # --- v1 vs v2 delta ---
    v1 = results['v1: Original GAMEPLAN']
    v2 = results['v2: Full GAMEPLAN v2']

    print(f"\n  v2 IMPROVEMENT vs v1:")
    print(f"    Final value: +${v2['final_value'] - v1['final_value']:,.0f} "
          f"({(v2['final_value']/v1['final_value'] - 1)*100:+.1f}%)")
    print(f"    Sharpe: {v2['sharpe'] - v1['sharpe']:+.3f}")
    print(f"    MaxDD: {v2['max_dd'] - v1['max_dd']:+.1f}pp")
    print(f"    Switches/yr: {v2['switches_per_year'] - v1['switches_per_year']:+.1f}")
    print(f"    Calmar: {v2['calmar'] - v1['calmar']:+.3f}")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'results': results,
    }
    with open(os.path.join(OUTPUT_DIR, 'v1_v2_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
