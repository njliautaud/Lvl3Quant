#!/usr/bin/env python3
"""Quick adversarial audit for Anti-Fragile and VIX Percentile strategies."""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research/autonomous')
from research_template import *


def run_audit(name, equity, spy_prices):
    print(f"\n{'='*70}")
    print(f"AUDITING: {name}")
    print(f"{'='*70}")
    m = compute_metrics(equity, name)
    print(f"  Base: Sharpe {m['sharpe']:.3f}, Sortino {m['sortino']:.3f}, CAGR {m['cagr']:.1%}, MaxDD {m['max_dd']:.1%}")

    print(f"\n  [1/5] Permutation (50 trials)...")
    perm = permutation_test(equity, spy_prices, n_perms=50)
    s = "PASS" if perm['pass'] else "FAIL"
    print(f"    {s}: real={perm['real_sharpe']:.3f}, perm_mean={perm['perm_mean']:.3f}, p={perm['p_value']:.3f}")

    print(f"\n  [2/5] Sub-Period Consistency...")
    subp = subperiod_test(equity)
    s = "PASS" if subp['pass'] else "FAIL"
    print(f"    {s}: blocks={subp['block_sharpes']}, CV={subp['cv']:.3f}")

    print(f"\n  [3/5] Regime Test...")
    regime = regime_test(equity, spy_prices)
    s = "PASS" if regime['pass'] else "FAIL"
    print(f"    {s}: green={regime['green_sharpe']:.2f}, red={regime['red_sharpe']:.2f}, gap={regime['gap']:.3f}")

    print(f"\n  [4/5] Outlier Sensitivity...")
    returns = equity.pct_change().dropna()
    threshold = returns.quantile(0.95)
    filtered = returns[returns <= threshold]
    orig_sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
    filt_sharpe = float(filtered.mean() / filtered.std() * np.sqrt(252))
    degradation = (orig_sharpe - filt_sharpe) / max(abs(orig_sharpe), 0.01)
    outlier_pass = degradation < 0.50
    s = "PASS" if outlier_pass else "FAIL"
    print(f"    {s}: orig={orig_sharpe:.3f}, filt={filt_sharpe:.3f}, degrad={degradation*100:.1f}%")

    print(f"\n  [5/5] Data Integrity...")
    suspicious = returns[abs(returns) > 0.15]
    integrity_pass = len(suspicious) <= len(returns) * 0.005
    s = "PASS" if integrity_pass else "FAIL"
    print(f"    {s}: suspicious={len(suspicious)}/{len(returns)}")

    gates = sum([perm['pass'], subp['pass'], regime['pass'], outlier_pass, integrity_pass])
    print(f"\n  VERDICT: {gates}/5 gates passed")
    return gates


def main():
    print("="*70)
    print("REMAINING STRATEGY AUDIT")
    print("="*70)

    tickers = ['SPY', 'TLT', 'GLD', 'SHY', 'UPRO']
    prices = download_etfs(tickers, start='2009-01-01')
    vix = download_vix(start='2009-01-01')
    prices['VIX'] = vix
    prices = prices.dropna()
    print(f"Data: {len(prices)} rows, {prices.index[0].date()} -> {prices.index[-1].date()}")

    spy = prices['SPY']
    vix_s = prices['VIX']
    start_i = 252

    # ANTI-FRAGILE
    eq_af = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start_i, len(prices)):
        v = vix_s.iloc[i]
        spy_ret = spy.pct_change().iloc[i]
        tlt_ret = prices['TLT'].pct_change().iloc[i]
        gld_ret = prices['GLD'].pct_change().iloc[i]
        if v > 25:
            port_ret = 0.30 * spy_ret + 0.40 * tlt_ret + 0.30 * gld_ret
        elif v > 20:
            port_ret = 0.50 * spy_ret + 0.30 * tlt_ret + 0.20 * gld_ret
        else:
            port_ret = 0.80 * spy_ret + 0.10 * tlt_ret + 0.10 * gld_ret
        if np.isfinite(port_ret):
            eq_af.iloc[i] = eq_af.iloc[i-1] * (1 + port_ret)
        else:
            eq_af.iloc[i] = eq_af.iloc[i-1]
    eq_af = eq_af.iloc[start_i:]
    run_audit("Anti-Fragile Portfolio", eq_af, spy.iloc[start_i:])

    # VIX PERCENTILE
    eq_vp = pd.Series(INITIAL_CAPITAL, index=prices.index)
    lookback = 252
    for i in range(start_i, len(prices)):
        v = vix_s.iloc[i]
        hist_vix = vix_s.iloc[max(0,i-lookback):i]
        pctile = (hist_vix < v).mean()
        if pctile < 0.20:
            leverage = 2.5
        elif pctile < 0.40:
            leverage = 2.0
        elif pctile < 0.60:
            leverage = 1.5
        elif pctile < 0.80:
            leverage = 1.0
        else:
            leverage = 0.5
        spy_ret = spy.pct_change().iloc[i]
        port_ret = leverage * spy_ret
        if np.isfinite(port_ret):
            eq_vp.iloc[i] = eq_vp.iloc[i-1] * (1 + port_ret)
        else:
            eq_vp.iloc[i] = eq_vp.iloc[i-1]
    eq_vp = eq_vp.iloc[start_i:]
    run_audit("Relative VIX Percentile", eq_vp, spy.iloc[start_i:])

    print(f"\n{'='*70}")
    print("DONE")


if __name__ == '__main__':
    main()
