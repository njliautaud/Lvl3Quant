#!/usr/bin/env python3
"""
Sector Rotation Enhancement for Vol-Adjusted Strategy
======================================================
Tests whether replacing UPRO (3x S&P) with sector-specific leveraged ETFs
during the low-vol UPRO phase can add alpha.

Hypothesis: when vol is low and we're in UPRO, maybe we can do better by
overweighting winning sectors (momentum) or underweighting losing ones.

Tests:
1. Sector momentum: rotate into top 3 sector ETFs by 1-month momentum
2. Sector mean-reversion: buy beaten-down sectors
3. Tech-heavy: TQQQ instead of UPRO during low vol
4. Equal-weight: XLK+XLF+XLV+XLI equally weighted
5. Adaptive: top 2 sectors by 3-month momentum, rebalanced monthly

All compared to plain UPRO with vol-adjusted switching.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/sector_rotation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

WEEKLY_DCA = 100
INITIAL_CAPITAL = 500
TX_COST = 0.001

# Sector ETFs (unleveraged — we'll calculate synthetic 3x returns)
SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLI', 'XLE', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']
# Also test with actual leveraged sector ETFs where available
LEVERAGED_SECTOR = {
    'XLK': 'TECL',  # 3x Technology
    'XLF': 'FAS',   # 3x Financials
    'XLE': 'ERX',   # 3x Energy (actually 2x now)
}

def download_data():
    all_tickers = ['SPY', 'UPRO', 'TQQQ', 'GLD', 'TLT', 'VIXY'] + SECTOR_ETFS
    all_tickers += list(LEVERAGED_SECTOR.values())

    print(f"Downloading {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start='2012-01-01', period='max',
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

    closes = closes.dropna(how='all').dropna(subset=['SPY', 'UPRO'])
    print(f"  Data: {len(closes)} days, {closes.shape[1]} tickers")
    print(f"  Range: {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")
    return closes

def simulate_strategy(closes, get_allocation, name=""):
    """
    Simulate vol-adjusted strategy with custom allocation function.
    get_allocation(date, closes, i, vol_regime) -> dict {ticker: weight}
    """
    spy_ret = closes['SPY'].pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    spy = closes['SPY']
    spy_sma50 = spy.rolling(50).mean()
    returns = closes.pct_change().fillna(0)

    warmup = 63
    portfolio_val = INITIAL_CAPITAL
    holdings = {}
    cash = float(INITIAL_CAPITAL)
    total_contributed = INITIAL_CAPITAL
    last_week = None
    last_alloc = None
    n_switches = 0

    daily_values = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]

        # DCA
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        # Update holdings
        for ticker in list(holdings.keys()):
            if ticker in returns.columns:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    holdings[ticker] *= (1 + r)

        # Vol regime
        vol = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 0.15
        protection = spy.iloc[i] > spy_sma50.iloc[i] if not np.isnan(spy_sma50.iloc[i]) else True

        if not protection:
            vol_regime = 'CASH'
        elif vol < 0.20:
            vol_regime = 'LOW'
        elif vol < 0.30:
            vol_regime = 'MEDIUM'
        else:
            vol_regime = 'HIGH'

        # Get target allocation
        if vol_regime == 'CASH':
            target = {'SPY': 1.0}  # Cash proxy
        elif vol_regime == 'MEDIUM':
            target = {'SPY': 1.0}
        elif vol_regime == 'HIGH':
            target = {'GLD': 0.5, 'TLT': 0.5}
        else:  # LOW — this is where we test sector rotation
            target = get_allocation(date, closes, i, vol_regime)

        # Check if allocation changed
        should_switch = False
        if last_alloc is None:
            should_switch = True
        else:
            for t in set(list(target.keys()) + list(last_alloc.keys())):
                if abs(target.get(t, 0) - last_alloc.get(t, 0)) > 0.10:
                    should_switch = True
                    break

        if should_switch:
            total_val = cash + sum(holdings.values())
            turnover = sum(abs(holdings.get(t, 0) - total_val * target.get(t, 0))
                          for t in set(list(holdings.keys()) + list(target.keys())))
            cost = (turnover / total_val * TX_COST * total_val) if total_val > 0 else 0
            total_val -= cost

            holdings = {t: total_val * w for t, w in target.items() if w > 0}
            cash = 0
            last_alloc = dict(target)
            n_switches += 1
        elif cash > WEEKLY_DCA * 0.5 and holdings:
            total_h = sum(holdings.values())
            if total_h > 0:
                for t in holdings:
                    holdings[t] += cash * (holdings[t] / total_h)
                cash = 0

        portfolio_val = cash + sum(holdings.values())
        daily_values.append(portfolio_val)

    portfolio = pd.Series(daily_values, index=closes.index[warmup:])
    return portfolio, total_contributed, n_switches

def compute_metrics(portfolio, total_contributed):
    r = portfolio.pct_change().dropna()
    if len(r) < 63:
        return None

    years = len(r) / 252
    final = portfolio.iloc[-1]
    profit = final - total_contributed

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
        'profit': float(profit),
        'cagr': float(cagr * 100),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd * 100),
    }

def main():
    print("="*70)
    print("SECTOR ROTATION ENHANCEMENT TEST")
    print("="*70)

    closes = download_data()

    # Available sector ETFs
    avail_sectors = [s for s in SECTOR_ETFS if s in closes.columns]
    print(f"  Available sectors: {', '.join(avail_sectors)}")

    # --- Strategy 1: Baseline UPRO ---
    def alloc_upro(date, c, i, regime):
        return {'UPRO': 1.0}

    # --- Strategy 2: TQQQ instead ---
    def alloc_tqqq(date, c, i, regime):
        return {'TQQQ': 1.0}

    # --- Strategy 3: Top 3 sectors by 1-month momentum ---
    def alloc_mom_top3(date, c, i, regime):
        rets = {}
        for s in avail_sectors:
            if s in c.columns and i >= 21:
                r = c[s].iloc[i] / c[s].iloc[i-21] - 1
                if not np.isnan(r):
                    rets[s] = r
        if len(rets) < 3:
            return {'UPRO': 1.0}
        top3 = sorted(rets, key=rets.get, reverse=True)[:3]
        return {s: 1/3 for s in top3}

    # --- Strategy 4: Top 2 sectors by 3-month momentum ---
    def alloc_mom_top2_3m(date, c, i, regime):
        rets = {}
        for s in avail_sectors:
            if s in c.columns and i >= 63:
                r = c[s].iloc[i] / c[s].iloc[i-63] - 1
                if not np.isnan(r):
                    rets[s] = r
        if len(rets) < 2:
            return {'UPRO': 1.0}
        top2 = sorted(rets, key=rets.get, reverse=True)[:2]
        return {s: 0.5 for s in top2}

    # --- Strategy 5: Bottom 3 sectors (mean reversion) ---
    def alloc_meanrev(date, c, i, regime):
        rets = {}
        for s in avail_sectors:
            if s in c.columns and i >= 21:
                r = c[s].iloc[i] / c[s].iloc[i-21] - 1
                if not np.isnan(r):
                    rets[s] = r
        if len(rets) < 3:
            return {'UPRO': 1.0}
        bottom3 = sorted(rets, key=rets.get)[:3]
        return {s: 1/3 for s in bottom3}

    # --- Strategy 6: 50/50 UPRO + TQQQ ---
    def alloc_upro_tqqq(date, c, i, regime):
        return {'UPRO': 0.5, 'TQQQ': 0.5}

    # --- Strategy 7: Tech momentum with leveraged ETFs ---
    def alloc_lev_momentum(date, c, i, regime):
        # Use available leveraged sector ETFs
        available = {}
        for unlev, lev in LEVERAGED_SECTOR.items():
            if lev in c.columns and unlev in c.columns and i >= 21:
                r = c[unlev].iloc[i] / c[unlev].iloc[i-21] - 1
                if not np.isnan(r):
                    available[lev] = r

        if len(available) < 2:
            return {'UPRO': 1.0}

        # Top leveraged sector by momentum
        best = max(available, key=available.get)
        return {best: 0.5, 'UPRO': 0.5}

    # --- Strategy 8: Equal weight sectors (diversified) ---
    def alloc_equal_sectors(date, c, i, regime):
        sectors = [s for s in ['XLK', 'XLF', 'XLV', 'XLI', 'XLE'] if s in c.columns]
        if not sectors:
            return {'UPRO': 1.0}
        return {s: 1/len(sectors) for s in sectors}

    strategies = {
        '1. UPRO baseline': alloc_upro,
        '2. TQQQ instead': alloc_tqqq,
        '3. Top 3 sectors (1m mom)': alloc_mom_top3,
        '4. Top 2 sectors (3m mom)': alloc_mom_top2_3m,
        '5. Bottom 3 sectors (meanrev)': alloc_meanrev,
        '6. 50/50 UPRO+TQQQ': alloc_upro_tqqq,
        '7. Lev sector momentum': alloc_lev_momentum,
        '8. Equal weight 5 sectors': alloc_equal_sectors,
    }

    results = {}

    for name, alloc in strategies.items():
        print(f"\n  Running: {name}...", end=" ", flush=True)
        portfolio, total_cont, n_sw = simulate_strategy(closes, alloc, name)
        m = compute_metrics(portfolio, total_cont)
        if m:
            m['n_switches'] = n_sw
            results[name] = m
            print(f"${m['final_value']:,.0f} | Sharpe {m['sharpe']:.3f} | MaxDD {m['max_dd']:.1f}%")

    # Sort by final value
    sorted_val = sorted(results.items(), key=lambda x: x[1]['final_value'], reverse=True)

    print("\n" + "="*70)
    print("RESULTS — RANKED BY FINAL VALUE")
    print("="*70)

    print(f"\n  {'Strategy':<30s} {'Final $':>10s} {'Profit':>10s} {'CAGR':>7s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>7s}")
    print("  " + "-"*82)
    for name, m in sorted_val:
        print(f"  {name:<30s} ${m['final_value']:>9,.0f} ${m['profit']:>9,.0f} {m['cagr']:>6.1f}% "
              f"{m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['max_dd']:>6.1f}%")

    # Sort by Sharpe
    sorted_sharpe = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)

    print("\n" + "="*70)
    print("RESULTS — RANKED BY SHARPE")
    print("="*70)

    print(f"\n  {'Strategy':<30s} {'Sharpe':>7s} {'Final $':>10s} {'MaxDD':>7s}")
    print("  " + "-"*56)
    for name, m in sorted_sharpe:
        print(f"  {name:<30s} {m['sharpe']:>7.3f} ${m['final_value']:>9,.0f} {m['max_dd']:>6.1f}%")

    # Compare to baseline
    baseline = results.get('1. UPRO baseline', {})
    if baseline:
        print(f"\n" + "="*70)
        print("IMPROVEMENT OVER UPRO BASELINE")
        print("="*70)

        for name, m in sorted_val:
            if name == '1. UPRO baseline':
                continue
            val_diff = m['final_value'] - baseline['final_value']
            sharpe_diff = m['sharpe'] - baseline['sharpe']
            dd_diff = m['max_dd'] - baseline['max_dd']
            print(f"  {name:<30s}: value {'+' if val_diff > 0 else ''}{val_diff:,.0f}, "
                  f"Sharpe {sharpe_diff:+.3f}, MaxDD {dd_diff:+.1f}pp")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'results': results,
        'ranking_by_value': [n for n, _ in sorted_val],
        'ranking_by_sharpe': [n for n, _ in sorted_sharpe],
    }

    output_path = os.path.join(OUTPUT_DIR, 'sector_rotation_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved.")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == '__main__':
    main()
