#!/usr/bin/env python3
"""
Golden Butterfly Portfolio with Dynamic Rebalancing
====================================================
Classic golden butterfly: 20% each of SPY, small-cap value, long bonds,
short bonds, gold. But with ML-based dynamic rebalancing based on
macro conditions.

Hypothesis: The static golden butterfly has excellent risk-adjusted returns.
Adding dynamic rebalancing based on vol/macro signals could improve further.
"""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research/autonomous')
from research_template import *

def main():
    print("=" * 70)
    print("GOLDEN BUTTERFLY + DYNAMIC REBALANCING")
    print("=" * 70)

    tickers = ['SPY', 'IWN', 'TLT', 'SHY', 'GLD', 'UPRO']
    prices = download_etfs(tickers, start='2010-01-01')
    vix = download_vix()
    prices['VIX'] = vix
    prices = prices.dropna()
    print(f"Data: {len(prices)} rows, {prices.index[0].date()} → {prices.index[-1].date()}")

    rets = {t: prices[t].pct_change() for t in tickers}
    spy_ret = rets['SPY']

    configs = {}
    start = 252  # 1yr warmup

    # 1. Static Golden Butterfly (20/20/20/20/20)
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(1, len(prices)):
        r = 0.20 * rets['SPY'].iloc[i] + 0.20 * rets['IWN'].iloc[i] + \
            0.20 * rets['TLT'].iloc[i] + 0.20 * rets['SHY'].iloc[i] + \
            0.20 * rets['GLD'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + r) if np.isfinite(r) else eq.iloc[i-1]
    configs['Static GB'] = eq

    # 2. Vol-scaled: increase SHY/GLD when vol high, increase SPY/IWN when vol low
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    rvol = spy_ret.rolling(20).std() * np.sqrt(252)
    for i in range(start, len(prices)):
        v = rvol.iloc[i]
        if v < 0.12:  # Low vol
            w = {'SPY': 0.30, 'IWN': 0.25, 'TLT': 0.20, 'SHY': 0.10, 'GLD': 0.15}
        elif v < 0.20:  # Normal
            w = {'SPY': 0.20, 'IWN': 0.20, 'TLT': 0.20, 'SHY': 0.20, 'GLD': 0.20}
        elif v < 0.30:  # Elevated
            w = {'SPY': 0.15, 'IWN': 0.10, 'TLT': 0.20, 'SHY': 0.30, 'GLD': 0.25}
        else:  # Crisis
            w = {'SPY': 0.10, 'IWN': 0.05, 'TLT': 0.15, 'SHY': 0.40, 'GLD': 0.30}

        r = sum(w[t] * rets[t].iloc[i] for t in w if np.isfinite(rets[t].iloc[i]))
        eq.iloc[i] = eq.iloc[i-1] * (1 + r) if np.isfinite(r) else eq.iloc[i-1]
    configs['Vol-Scaled GB'] = eq.iloc[start:]

    # 3. Trend-filtered: increase equity when SPY above 200 SMA
    sma200 = prices['SPY'].rolling(200).mean()
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        if prices['SPY'].iloc[i] > sma200.iloc[i]:
            w = {'SPY': 0.30, 'IWN': 0.25, 'TLT': 0.15, 'SHY': 0.10, 'GLD': 0.20}
        else:
            w = {'SPY': 0.10, 'IWN': 0.10, 'TLT': 0.25, 'SHY': 0.35, 'GLD': 0.20}

        r = sum(w[t] * rets[t].iloc[i] for t in w if np.isfinite(rets[t].iloc[i]))
        eq.iloc[i] = eq.iloc[i-1] * (1 + r) if np.isfinite(r) else eq.iloc[i-1]
    configs['Trend-Filtered GB'] = eq.iloc[start:]

    # 4. Leveraged GB: UPRO replaces SPY when trend positive + vol low
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = rvol.iloc[i]
        trend_up = prices['SPY'].iloc[i] > sma200.iloc[i]

        if trend_up and v < 0.15:
            # Use UPRO for equity sleeve
            w_equity = 0.25
            r = w_equity * rets['UPRO'].iloc[i] + 0.15 * rets['IWN'].iloc[i] + \
                0.20 * rets['TLT'].iloc[i] + 0.20 * rets['SHY'].iloc[i] + \
                0.20 * rets['GLD'].iloc[i]
        elif trend_up:
            r = 0.25 * rets['SPY'].iloc[i] + 0.20 * rets['IWN'].iloc[i] + \
                0.20 * rets['TLT'].iloc[i] + 0.15 * rets['SHY'].iloc[i] + \
                0.20 * rets['GLD'].iloc[i]
        else:
            r = 0.10 * rets['SPY'].iloc[i] + 0.10 * rets['IWN'].iloc[i] + \
                0.20 * rets['TLT'].iloc[i] + 0.35 * rets['SHY'].iloc[i] + \
                0.25 * rets['GLD'].iloc[i]

        eq.iloc[i] = eq.iloc[i-1] * (1 + r) if np.isfinite(r) else eq.iloc[i-1]
    configs['Leveraged GB'] = eq.iloc[start:]

    # SPY benchmark
    spy_eq = prices['SPY'].iloc[start:] / prices['SPY'].iloc[start] * INITIAL_CAPITAL
    configs['SPY B&H'] = spy_eq

    # Results
    print(f"\n{'Strategy':<25s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'MaxDD':>7s}")
    print(f"{'-'*25} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")

    best_name, best_sharpe = None, -999
    for name, eq in configs.items():
        m = compute_metrics(eq, name)
        print(f"{m['name']:<25s} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1%} {m['max_dd']:>6.1%}")
        if name != 'SPY B&H' and m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_name = name

    print(f"\nBest: {best_name}")
    best_eq = configs[best_name]
    metrics = compute_metrics(best_eq, best_name)
    adv = full_adversarial(best_eq, prices['SPY'])

    print(f"Adversarial: {adv['gates_passed']}/3")
    print(f"  Perm p={adv['permutation']['p_value']} {'PASS' if adv['permutation']['pass'] else 'FAIL'}")
    print(f"  SubP CV={adv['subperiod']['cv']} {'PASS' if adv['subperiod']['pass'] else 'FAIL'}")
    print(f"  R1 gap={adv['regime']['gap']} {'PASS' if adv['regime']['pass'] else 'FAIL'}")

    emit_result(
        name=f"Golden Butterfly ({best_name})",
        description="Golden Butterfly portfolio with dynamic vol/trend rebalancing",
        metrics=metrics,
        adversarial=adv
    )

if __name__ == '__main__':
    main()
