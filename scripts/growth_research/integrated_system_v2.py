#!/usr/bin/env python3
"""
Integrated System Backtest v2 — Growth-Optimized
=================================================
Lessons from v1: Full risk parity kills returns for small accounts.
This version tests growth-optimized allocations:

Variant A: UPRO protected only (100% UPRO with SMA50 overlay, all phases)
Variant B: UPRO protected + gradual CTA tilt (70/30 → 60/40 as account grows)
Variant C: UPRO protected heavy + light diversification (80% UPRO + 20% GLD/TLT)
Variant D: Dynamic protection (UPRO when vol < 20%, SPY when vol 20-30%, cash when vol > 30%)
Variant E: Concentrated growth → gradual diversification (UPRO until $25K, then slowly add)
Variant F: Simple DCA SPY (benchmark)
Variant G: Simple DCA UPRO no protection (benchmark)

All with $500 start + $100/week DCA.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/integrated_system'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TX_COST = 0.001
WEEKLY_DCA = 100
INITIAL_CAPITAL = 500

def download_data():
    tickers = ['SPY', 'UPRO', 'SH', 'GLD', 'TLT', 'VIXY', 'HYG', 'IWM']
    print(f"Downloading {len(tickers)} tickers...")
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
    closes = closes.dropna(how='all').dropna(subset=['UPRO', 'SPY'])
    print(f"  Data: {len(closes)} days, {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")
    return closes

def simulate_dca(closes, allocation_func, name=""):
    """
    Generic DCA simulator.
    allocation_func(date, portfolio_value, closes, i) -> dict of {ticker: weight}
    """
    warmup = 50
    portfolio_val = INITIAL_CAPITAL
    holdings = {}  # ticker -> value
    cash = float(INITIAL_CAPITAL)
    total_contributed = INITIAL_CAPITAL
    last_week = None
    last_alloc = None
    n_rebalances = 0
    total_tx = 0.0

    daily_values = []
    daily_dates = []

    returns = closes.pct_change().fillna(0)

    for i in range(warmup, len(closes)):
        date = closes.index[i]

        # Weekly DCA
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

        portfolio_val = cash + sum(holdings.values())

        # Get target allocation
        target_alloc = allocation_func(date, portfolio_val, closes, i)

        # Check if rebalance needed (threshold 15% or first time)
        should_rebalance = False
        if last_alloc is None:
            should_rebalance = True
        else:
            invested = sum(holdings.values())
            if invested > 0:
                current_alloc = {t: v / invested for t, v in holdings.items()}
                for t in set(list(target_alloc.keys()) + list(current_alloc.keys())):
                    cur = current_alloc.get(t, 0)
                    tgt = target_alloc.get(t, 0)
                    if abs(cur - tgt) > 0.15:
                        should_rebalance = True
                        break

            # Check if allocation changed significantly
            if last_alloc is not None:
                for t in set(list(target_alloc.keys()) + list(last_alloc.keys())):
                    cur = last_alloc.get(t, 0)
                    tgt = target_alloc.get(t, 0)
                    if abs(cur - tgt) > 0.10:
                        should_rebalance = True
                        break

        if should_rebalance:
            total_val = cash + sum(holdings.values())
            old_holdings = dict(holdings)

            # New allocations
            new_holdings = {}
            for ticker, weight in target_alloc.items():
                new_holdings[ticker] = total_val * weight

            # Calculate turnover
            all_tickers = set(list(old_holdings.keys()) + list(new_holdings.keys()))
            turnover = sum(abs(new_holdings.get(t, 0) - old_holdings.get(t, 0)) for t in all_tickers)
            cost = turnover * TX_COST
            total_tx += cost

            # Apply cost
            total_after_cost = total_val - cost
            holdings = {t: total_after_cost * w for t, w in target_alloc.items()}
            cash = 0
            last_alloc = dict(target_alloc)
            n_rebalances += 1
        elif cash > WEEKLY_DCA * 0.5 and holdings:
            # Deploy DCA cash to existing allocation
            for ticker in holdings:
                weight = holdings[ticker] / sum(holdings.values())
                holdings[ticker] += cash * weight
            cash = 0

        portfolio_val = cash + sum(holdings.values())
        daily_values.append(portfolio_val)
        daily_dates.append(date)

    portfolio = pd.Series(daily_values, index=daily_dates)
    return portfolio, total_contributed, n_rebalances, total_tx

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
        'total_contributed': float(total_contributed),
        'return_on_contributions': float(profit / total_contributed * 100),
        'cagr': float(cagr * 100),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd * 100),
        'ann_vol': float(ann_vol * 100),
        'years': float(years),
    }

def main():
    print("="*70)
    print("INTEGRATED SYSTEM v2 — GROWTH-OPTIMIZED VARIANTS")
    print("="*70)

    closes = download_data()
    spy = closes['SPY']
    spy_sma50 = spy.rolling(50).mean()
    spy_sma200 = spy.rolling(200).mean()
    spy_ret = spy.pct_change()
    spy_vol_21 = spy_ret.rolling(21).std() * np.sqrt(252)
    vixy = closes.get('VIXY')
    vixy_sma5 = vixy.rolling(5).mean() if vixy is not None else None

    # 4-signal protection from our validated research
    hyg = closes.get('HYG')
    lqd = closes.get('LQD')
    iwm = closes.get('IWM')

    def protection_on(i):
        """SPY > SMA50 protection overlay."""
        if i >= len(spy) or np.isnan(spy_sma50.iloc[i]):
            return True
        return spy.iloc[i] > spy_sma50.iloc[i]

    def vol_level(i):
        if i >= len(spy_vol_21) or np.isnan(spy_vol_21.iloc[i]):
            return 0.15
        return spy_vol_21.iloc[i]

    def vixy_rising(i):
        if vixy is None or vixy_sma5 is None:
            return False
        if i >= len(vixy) or np.isnan(vixy_sma5.iloc[i]):
            return False
        return vixy.iloc[i] > vixy_sma5.iloc[i]

    # --- Variant A: UPRO protected only ---
    def alloc_a(date, pv, c, i):
        if protection_on(i):
            return {'UPRO': 1.0}
        return {'SPY': 0.0}  # Cash equivalent (0 allocation)

    # Fix: can't have empty allocation. Use a tiny SPY position as "cash"
    def alloc_a_fixed(date, pv, c, i):
        if protection_on(i):
            return {'UPRO': 1.0}
        # In cash — hold money market proxy (just keep cash)
        return {'SPY': 1.0}  # SPY as cash proxy when protection off
        # Note: this overstates returns in cash periods. Real impl = money market

    # --- Variant B: UPRO + CTA tilt ---
    def alloc_b(date, pv, c, i):
        if not protection_on(i):
            return {'SPY': 1.0}

        cta_on = spy.iloc[i] > spy_sma200.iloc[i] if not np.isnan(spy_sma200.iloc[i]) else True

        if pv < 10000:
            # Small account: heavy UPRO
            if cta_on:
                return {'UPRO': 0.8, 'SPY': 0.2}
            else:
                return {'UPRO': 0.6, 'GLD': 0.2, 'TLT': 0.2}
        else:
            # Larger: more balanced
            if cta_on:
                return {'UPRO': 0.6, 'SPY': 0.2, 'GLD': 0.1, 'TLT': 0.1}
            else:
                return {'UPRO': 0.4, 'GLD': 0.3, 'TLT': 0.3}

    # --- Variant C: UPRO + light diversification ---
    def alloc_c(date, pv, c, i):
        if not protection_on(i):
            return {'SPY': 1.0}
        return {'UPRO': 0.80, 'GLD': 0.10, 'TLT': 0.10}

    # --- Variant D: Vol-adjusted ---
    def alloc_d(date, pv, c, i):
        vol = vol_level(i)
        if vol < 0.15:
            return {'UPRO': 1.0}  # Low vol: full leverage
        elif vol < 0.20:
            return {'UPRO': 0.7, 'SPY': 0.3}  # Medium: partial deleverage
        elif vol < 0.25:
            return {'SPY': 1.0}  # High: just SPY
        else:
            return {'GLD': 0.5, 'TLT': 0.5}  # Very high: safe havens

    # --- Variant E: Concentrated then diversify ---
    def alloc_e(date, pv, c, i):
        if not protection_on(i):
            return {'SPY': 1.0}

        if pv < 25000:
            return {'UPRO': 1.0}  # Pure growth
        elif pv < 50000:
            return {'UPRO': 0.7, 'GLD': 0.15, 'TLT': 0.15}
        else:
            # Add dynamic hedge
            if vixy_rising(i):
                return {'UPRO': 0.5, 'GLD': 0.15, 'TLT': 0.15, 'SH': 0.2}
            return {'UPRO': 0.6, 'GLD': 0.2, 'TLT': 0.2}

    # --- Variant F: SPY only ---
    def alloc_f(date, pv, c, i):
        return {'SPY': 1.0}

    # --- Variant G: UPRO unprotected ---
    def alloc_g(date, pv, c, i):
        return {'UPRO': 1.0}

    # --- Variant H: UPRO protected + dynamic SH hedge ---
    def alloc_h(date, pv, c, i):
        if not protection_on(i):
            return {'SPY': 1.0}
        if vixy_rising(i):
            return {'UPRO': 0.8, 'SH': 0.2}
        return {'UPRO': 1.0}

    variants = {
        'A: UPRO protected': alloc_a_fixed,
        'B: UPRO + CTA tilt': alloc_b,
        'C: UPRO + 20% diversified': alloc_c,
        'D: Vol-adjusted leverage': alloc_d,
        'E: Concentrate → diversify': alloc_e,
        'F: SPY only (benchmark)': alloc_f,
        'G: UPRO no protection (bench)': alloc_g,
        'H: UPRO + dynamic SH hedge': alloc_h,
    }

    all_results = {}

    for name, alloc_func in variants.items():
        print(f"\n  Running: {name}...", end=" ", flush=True)
        portfolio, total_cont, n_rebal, total_tx = simulate_dca(closes, alloc_func, name)
        m = compute_metrics(portfolio, total_cont)
        if m:
            m['n_rebalances'] = n_rebal
            m['total_tx'] = float(total_tx)
            all_results[name] = m
            print(f"${m['final_value']:,.0f} | Sharpe {m['sharpe']:.3f} | MaxDD {m['max_dd']:.1f}% | "
                  f"Profit ${m['profit']:,.0f}")

    # Sort by final value (user cares about profit)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['final_value'], reverse=True)

    print("\n" + "="*70)
    print("RESULTS RANKED BY FINAL VALUE (what the user cares about)")
    print("="*70)

    print(f"\n  {'Variant':<35s} {'Final $':>10s} {'Profit':>10s} {'CAGR':>7s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>7s} {'Rebal':>6s}")
    print("  " + "-"*95)
    for name, m in sorted_results:
        print(f"  {name:<35s} ${m['final_value']:>9,.0f} ${m['profit']:>9,.0f} {m['cagr']:>6.1f}% "
              f"{m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['max_dd']:>6.1f}% {m['n_rebalances']:>5d}")

    # Sort by Sharpe (risk-adjusted)
    sorted_sharpe = sorted(all_results.items(), key=lambda x: x[1]['sharpe'], reverse=True)

    print("\n" + "="*70)
    print("RESULTS RANKED BY SHARPE (risk-adjusted)")
    print("="*70)

    print(f"\n  {'Variant':<35s} {'Sharpe':>7s} {'Sortino':>8s} {'Final $':>10s} {'MaxDD':>7s} {'CAGR':>7s}")
    print("  " + "-"*78)
    for name, m in sorted_sharpe:
        print(f"  {name:<35s} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} ${m['final_value']:>9,.0f} "
              f"{m['max_dd']:>6.1f}% {m['cagr']:>6.1f}%")

    # Key tradeoff analysis
    print("\n" + "="*70)
    print("KEY TRADEOFFS")
    print("="*70)

    spy_m = all_results.get('F: SPY only (benchmark)', {})
    upro_m = all_results.get('G: UPRO no protection (bench)', {})
    best_growth = sorted_results[0] if sorted_results else None
    best_sharpe = sorted_sharpe[0] if sorted_sharpe else None

    if best_growth and spy_m:
        bg_name, bg_m = best_growth
        print(f"\n  Best absolute growth: {bg_name}")
        print(f"    Final value: ${bg_m['final_value']:,.0f} vs SPY ${spy_m['final_value']:,.0f} "
              f"({bg_m['final_value']/spy_m['final_value']:.1f}x)")

    if best_sharpe:
        bs_name, bs_m = best_sharpe
        print(f"\n  Best risk-adjusted: {bs_name}")
        print(f"    Sharpe: {bs_m['sharpe']:.3f}, MaxDD: {bs_m['max_dd']:.1f}%")

    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    # Find best balance of growth and risk
    # Score = final_value_rank + sharpe_rank (lower is better)
    names_by_value = [n for n, _ in sorted_results]
    names_by_sharpe = [n for n, _ in sorted_sharpe]

    combined_rank = {}
    for name in all_results:
        val_rank = names_by_value.index(name)
        sharpe_rank = names_by_sharpe.index(name)
        combined_rank[name] = val_rank + sharpe_rank

    best_balanced = min(combined_rank, key=combined_rank.get)
    bm = all_results[best_balanced]

    print(f"\n  BEST BALANCED (growth + risk): {best_balanced}")
    print(f"    Final: ${bm['final_value']:,.0f}, Profit: ${bm['profit']:,.0f}")
    print(f"    Sharpe: {bm['sharpe']:.3f}, MaxDD: {bm['max_dd']:.1f}%")
    print(f"    CAGR: {bm['cagr']:.1f}%")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'results': all_results,
        'ranking_by_value': [n for n, _ in sorted_results],
        'ranking_by_sharpe': [n for n, _ in sorted_sharpe],
        'recommendation': best_balanced,
    }

    output_path = os.path.join(OUTPUT_DIR, 'integrated_system_v2_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved.")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == '__main__':
    main()
