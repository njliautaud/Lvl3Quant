#!/usr/bin/env python3
"""
Anti-Fragile Portfolio
=======================
Portfolio that BENEFITS from volatility spikes, not just survives them.
Combines:
- VIX-scaled leverage (proven: be leveraged when calm)
- VIX spike buying (proven: buy fear when VIX > 30)
- Tail risk hedge (small ongoing cost for crash protection)

The idea: continuous small cost during calm markets (via VIX call spreads or
TLT allocation) that pays off hugely during crashes.
"""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research/autonomous')
from research_template import *

def main():
    print("=" * 70)
    print("ANTI-FRAGILE PORTFOLIO")
    print("=" * 70)

    tickers = ['SPY', 'UPRO', 'SHY', 'TLT', 'GLD', 'VIXY']
    prices = download_etfs(tickers, start='2012-01-01')  # VIXY starts later
    vix = download_vix(start='2012-01-01')
    prices['VIX'] = vix
    prices = prices.dropna()
    print(f"Data: {len(prices)} rows, {prices.index[0].date()} → {prices.index[-1].date()}")

    rets = {t: prices[t].pct_change() for t in tickers}
    start = 252

    configs = {}

    # 1. VIX-Scaled Leverage (our proven base)
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        if v < 15:
            ret = 0.50 * rets['UPRO'].iloc[i] + 0.50 * rets['SHY'].iloc[i]
        elif v < 20:
            ret = 0.80 * rets['SPY'].iloc[i] + 0.20 * rets['SHY'].iloc[i]
        elif v < 30:
            ret = 0.40 * rets['SPY'].iloc[i] + 0.60 * rets['SHY'].iloc[i]
        else:
            ret = rets['SHY'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['VIX Leverage Base'] = eq.iloc[start:]

    # 2. Anti-Fragile v1: VIX leverage + TLT tail hedge
    # Allocate 5% to TLT always (acts as crash insurance)
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        hedge = 0.05  # 5% in TLT always
        if v < 15:
            ret = 0.47 * rets['UPRO'].iloc[i] + 0.48 * rets['SHY'].iloc[i] + hedge * rets['TLT'].iloc[i]
        elif v < 20:
            ret = 0.75 * rets['SPY'].iloc[i] + 0.20 * rets['SHY'].iloc[i] + hedge * rets['TLT'].iloc[i]
        elif v < 30:
            ret = 0.35 * rets['SPY'].iloc[i] + 0.50 * rets['SHY'].iloc[i] + 0.15 * rets['TLT'].iloc[i]
        else:
            # During crisis: increase TLT allocation (flight to quality)
            ret = 0.40 * rets['TLT'].iloc[i] + 0.40 * rets['GLD'].iloc[i] + 0.20 * rets['SHY'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['Anti-Fragile v1 (TLT)'] = eq.iloc[start:]

    # 3. Anti-Fragile v2: During VIX spikes, go LONG crisis assets
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        vix_5d_change = (prices['VIX'].iloc[i] / prices['VIX'].iloc[max(0,i-5)] - 1) if i > 5 else 0

        if v < 15:
            ret = 0.50 * rets['UPRO'].iloc[i] + 0.50 * rets['SHY'].iloc[i]
        elif v < 20:
            ret = 0.80 * rets['SPY'].iloc[i] + 0.20 * rets['SHY'].iloc[i]
        elif v < 30:
            if vix_5d_change > 0.20:  # VIX spiking
                ret = 0.30 * rets['TLT'].iloc[i] + 0.30 * rets['GLD'].iloc[i] + 0.40 * rets['SHY'].iloc[i]
            else:
                ret = 0.40 * rets['SPY'].iloc[i] + 0.60 * rets['SHY'].iloc[i]
        else:
            # VIX > 30: maximum anti-fragile positioning
            if vix_5d_change > 0.30:  # Still spiking: defensive
                ret = 0.40 * rets['TLT'].iloc[i] + 0.40 * rets['GLD'].iloc[i] + 0.20 * rets['SHY'].iloc[i]
            else:  # VIX high but stabilizing: buy the fear
                ret = 0.30 * rets['UPRO'].iloc[i] + 0.30 * rets['TLT'].iloc[i] + 0.40 * rets['SHY'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['Anti-Fragile v2 (Crisis)'] = eq.iloc[start:]

    # 4. Anti-Fragile v3: Gold as primary hedge + VIX spike buying
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        if v < 15:
            ret = 0.45 * rets['UPRO'].iloc[i] + 0.10 * rets['GLD'].iloc[i] + 0.45 * rets['SHY'].iloc[i]
        elif v < 20:
            ret = 0.60 * rets['SPY'].iloc[i] + 0.15 * rets['GLD'].iloc[i] + 0.25 * rets['SHY'].iloc[i]
        elif v < 30:
            ret = 0.30 * rets['SPY'].iloc[i] + 0.20 * rets['GLD'].iloc[i] + 0.50 * rets['SHY'].iloc[i]
        else:
            # VIX > 30: buy the fear with UPRO + gold hedge
            ret = 0.25 * rets['UPRO'].iloc[i] + 0.25 * rets['GLD'].iloc[i] + 0.50 * rets['SHY'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['Anti-Fragile v3 (Gold)'] = eq.iloc[start:]

    # SPY benchmark
    spy_eq = prices['SPY'].iloc[start:] / prices['SPY'].iloc[start] * INITIAL_CAPITAL
    configs['SPY B&H'] = spy_eq

    # Results
    print(f"\n{'Strategy':<30s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'MaxDD':>7s}")
    print(f"{'-'*30} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")

    best_name, best_sharpe = None, -999
    for name, eq in configs.items():
        m = compute_metrics(eq, name)
        print(f"{m['name']:<30s} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
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
        name=f"Anti-Fragile Portfolio ({best_name})",
        description="Portfolio that benefits from vol spikes via crisis asset rotation",
        metrics=metrics,
        adversarial=adv
    )

if __name__ == '__main__':
    main()
