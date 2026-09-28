#!/usr/bin/env python3
"""
Risk Parity with Leveraged Equity Sleeve
==========================================
Classic risk parity (equal risk contribution across assets) but with
UPRO for the equity sleeve when conditions favor it.

Hypothesis: Risk parity has strong Sharpe but low absolute returns.
Adding selective leverage to the equity sleeve when vol is low could
boost returns without proportionally increasing risk.
"""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research/autonomous')
from research_template import *

def main():
    print("=" * 70)
    print("RISK PARITY + LEVERAGED EQUITY")
    print("=" * 70)

    tickers = ['SPY', 'UPRO', 'TLT', 'GLD', 'SHY', 'DBC', 'HYG']
    prices = download_etfs(tickers, start='2010-01-01')
    vix = download_vix()
    prices['VIX'] = vix
    prices = prices.dropna()
    print(f"Data: {len(prices)} rows, {prices.index[0].date()} → {prices.index[-1].date()}")

    rets = {t: prices[t].pct_change() for t in tickers}
    start = 252

    configs = {}

    # Compute rolling vols for risk parity weights
    vols = {}
    for t in ['SPY', 'TLT', 'GLD']:
        vols[t] = rets[t].rolling(63).std() * np.sqrt(252)

    def risk_parity_weights(vol_dict, assets):
        """Inverse-vol weights normalized to sum=1"""
        inv_vols = {a: 1.0 / max(vol_dict[a], 0.01) for a in assets}
        total = sum(inv_vols.values())
        return {a: v / total for a, v in inv_vols.items()}

    # 1. Static Risk Parity (SPY/TLT/GLD)
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = {a: vols[a].iloc[i] if not np.isnan(vols[a].iloc[i]) else 0.15 for a in ['SPY', 'TLT', 'GLD']}
        w = risk_parity_weights(v, ['SPY', 'TLT', 'GLD'])
        r = sum(w[a] * rets[a].iloc[i] for a in w if np.isfinite(rets[a].iloc[i]))
        eq.iloc[i] = eq.iloc[i-1] * (1 + r) if np.isfinite(r) else eq.iloc[i-1]
    configs['Static RP'] = eq.iloc[start:]

    # 2. Leveraged RP: UPRO replaces SPY when vol is low
    sma200 = prices['SPY'].rolling(200).mean()
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        spy_vol = vols['SPY'].iloc[i] if not np.isnan(vols['SPY'].iloc[i]) else 0.15
        trend_up = prices['SPY'].iloc[i] > sma200.iloc[i]

        v = {a: vols[a].iloc[i] if not np.isnan(vols[a].iloc[i]) else 0.15 for a in ['SPY', 'TLT', 'GLD']}
        w = risk_parity_weights(v, ['SPY', 'TLT', 'GLD'])

        if trend_up and spy_vol < 0.13:
            # Use UPRO for equity, but scale down weight (3x leverage = 1/3 weight)
            upro_w = w['SPY'] * 0.8  # slightly less than full 3x
            bond_w = w['TLT'] + w['SPY'] * 0.2  # put excess in bonds
            r = upro_w * rets['UPRO'].iloc[i] + bond_w * rets['TLT'].iloc[i] + w['GLD'] * rets['GLD'].iloc[i]
        else:
            r = sum(w[a] * rets[a].iloc[i] for a in w if np.isfinite(rets[a].iloc[i]))

        eq.iloc[i] = eq.iloc[i-1] * (1 + r) if np.isfinite(r) else eq.iloc[i-1]
    configs['Leveraged RP'] = eq.iloc[start:]

    # 3. 4-Asset RP with commodities
    for t in ['DBC']:
        vols[t] = rets[t].rolling(63).std() * np.sqrt(252)

    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = {a: vols[a].iloc[i] if not np.isnan(vols[a].iloc[i]) else 0.15
             for a in ['SPY', 'TLT', 'GLD', 'DBC']}
        w = risk_parity_weights(v, ['SPY', 'TLT', 'GLD', 'DBC'])
        r = sum(w[a] * rets[a].iloc[i] for a in w if np.isfinite(rets[a].iloc[i]))
        eq.iloc[i] = eq.iloc[i-1] * (1 + r) if np.isfinite(r) else eq.iloc[i-1]
    configs['4-Asset RP'] = eq.iloc[start:]

    # 4. Adaptive RP: shift RP weights based on VIX regime
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        vix_val = prices['VIX'].iloc[i]
        v = {a: vols[a].iloc[i] if not np.isnan(vols[a].iloc[i]) else 0.15 for a in ['SPY', 'TLT', 'GLD']}
        w = risk_parity_weights(v, ['SPY', 'TLT', 'GLD'])

        if vix_val > 25:
            # Crisis: reduce equity, increase gold
            w['SPY'] *= 0.5
            w['GLD'] *= 1.5
            total = sum(w.values())
            w = {k: v/total for k, v in w.items()}

        r = sum(w[a] * rets[a].iloc[i] for a in w if np.isfinite(rets[a].iloc[i]))
        eq.iloc[i] = eq.iloc[i-1] * (1 + r) if np.isfinite(r) else eq.iloc[i-1]
    configs['Adaptive RP'] = eq.iloc[start:]

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
        name=f"Risk Parity ({best_name})",
        description="Risk parity with leveraged equity sleeve and adaptive weighting",
        metrics=metrics,
        adversarial=adv
    )

if __name__ == '__main__':
    main()
