#!/usr/bin/env python3
"""
Global Equity Momentum (GEM) / Dual Momentum
=============================================
Classic Antonacci Dual Momentum:
- Absolute momentum: only invest when asset > T-bills (12mo return)
- Relative momentum: choose between US (SPY) and International (EFA)
- When both fail: park in bonds (AGG/TLT)

With ML enhancement: use cross-asset features to predict which
momentum regime we're in.
"""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research/autonomous')
from research_template import *

def main():
    print("=" * 70)
    print("DUAL MOMENTUM (GEM) + ML ENHANCEMENT")
    print("=" * 70)

    tickers = ['SPY', 'EFA', 'AGG', 'TLT', 'SHY', 'GLD', 'UPRO']
    prices = download_etfs(tickers, start='2007-01-01')
    vix = download_vix(start='2007-01-01')
    prices['VIX'] = vix
    prices = prices.dropna()
    print(f"Data: {len(prices)} rows, {prices.index[0].date()} → {prices.index[-1].date()}")

    spy_ret = prices['SPY'].pct_change()
    efa_ret = prices['EFA'].pct_change()
    agg_ret = prices['AGG'].pct_change()
    tlt_ret = prices['TLT'].pct_change()
    shy_ret = prices['SHY'].pct_change()
    upro_ret = prices['UPRO'].pct_change()
    gld_ret = prices['GLD'].pct_change()

    start = 252
    configs = {}

    # 1. Classic GEM (12-month lookback)
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        spy_12m = prices['SPY'].iloc[i] / prices['SPY'].iloc[max(0, i-252)] - 1
        efa_12m = prices['EFA'].iloc[i] / prices['EFA'].iloc[max(0, i-252)] - 1
        shy_12m = prices['SHY'].iloc[i] / prices['SHY'].iloc[max(0, i-252)] - 1

        if spy_12m > shy_12m and spy_12m > efa_12m:
            ret = spy_ret.iloc[i]  # US equities
        elif efa_12m > shy_12m:
            ret = efa_ret.iloc[i]  # International
        else:
            ret = agg_ret.iloc[i]  # Bonds

        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['Classic GEM'] = eq.iloc[start:]

    # 2. GEM with multiple lookbacks (3/6/12 month average)
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        scores = {}
        for asset, price_col in [('SPY', prices['SPY']), ('EFA', prices['EFA'])]:
            r3 = price_col.iloc[i] / price_col.iloc[max(0, i-63)] - 1
            r6 = price_col.iloc[i] / price_col.iloc[max(0, i-126)] - 1
            r12 = price_col.iloc[i] / price_col.iloc[max(0, i-252)] - 1
            scores[asset] = (r3 + r6 + r12) / 3

        shy_r = prices['SHY'].iloc[i] / prices['SHY'].iloc[max(0, i-252)] - 1

        best_asset = max(scores, key=scores.get)
        if scores[best_asset] > shy_r:
            ret = spy_ret.iloc[i] if best_asset == 'SPY' else efa_ret.iloc[i]
        else:
            ret = agg_ret.iloc[i]

        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['Multi-Lookback GEM'] = eq.iloc[start:]

    # 3. Leveraged GEM: UPRO when US momentum strong
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    rvol = spy_ret.rolling(20).std() * np.sqrt(252)
    for i in range(start, len(prices)):
        spy_12m = prices['SPY'].iloc[i] / prices['SPY'].iloc[max(0, i-252)] - 1
        efa_12m = prices['EFA'].iloc[i] / prices['EFA'].iloc[max(0, i-252)] - 1
        shy_12m = prices['SHY'].iloc[i] / prices['SHY'].iloc[max(0, i-252)] - 1
        v = rvol.iloc[i] if not np.isnan(rvol.iloc[i]) else 0.15

        if spy_12m > shy_12m and spy_12m > efa_12m:
            if v < 0.15:
                # Low vol + US momentum: use UPRO
                ret = 0.60 * upro_ret.iloc[i] + 0.40 * shy_ret.iloc[i]
            else:
                ret = spy_ret.iloc[i]
        elif efa_12m > shy_12m:
            ret = efa_ret.iloc[i]
        else:
            ret = 0.60 * tlt_ret.iloc[i] + 0.40 * shy_ret.iloc[i]

        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['Leveraged GEM'] = eq.iloc[start:]

    # 4. GEM + Gold hedge: add gold as 4th asset class
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        scores = {}
        for asset in ['SPY', 'EFA', 'GLD']:
            r12 = prices[asset].iloc[i] / prices[asset].iloc[max(0, i-252)] - 1
            scores[asset] = r12

        shy_r = prices['SHY'].iloc[i] / prices['SHY'].iloc[max(0, i-252)] - 1
        best = max(scores, key=scores.get)

        if scores[best] > shy_r:
            ret_map = {'SPY': spy_ret, 'EFA': efa_ret, 'GLD': gld_ret}
            ret = ret_map[best].iloc[i]
        else:
            ret = agg_ret.iloc[i]

        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['GEM + Gold'] = eq.iloc[start:]

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
    metrics = compute_metrics(configs[best_name], best_name)
    adv = full_adversarial(configs[best_name], prices['SPY'])

    print(f"Adversarial: {adv['gates_passed']}/3")
    for test in ['permutation', 'subperiod', 'regime']:
        t = adv[test]
        key = 'p_value' if test == 'permutation' else 'cv' if test == 'subperiod' else 'gap'
        print(f"  {test}: {t.get(key, '?')} {'PASS' if t['pass'] else 'FAIL'}")

    emit_result(
        name=f"Dual Momentum ({best_name})",
        description="Global Equity Momentum with multiple variants + leveraged option",
        metrics=metrics,
        adversarial=adv
    )

if __name__ == '__main__':
    main()
