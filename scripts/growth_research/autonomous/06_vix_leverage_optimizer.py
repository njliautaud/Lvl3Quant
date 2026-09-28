#!/usr/bin/env python3
"""
VIX-Level Leverage Optimizer
==============================
Given that VIX-level gating + UPRO is our strongest signal (Sharpe 2.83),
explore the optimal threshold levels and position sizing.

Grid search: VIX thresholds × leverage levels × cash alternatives
"""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research/autonomous')
from research_template import *

def main():
    print("=" * 70)
    print("VIX-LEVEL LEVERAGE OPTIMIZER")
    print("=" * 70)

    tickers = ['SPY', 'UPRO', 'SHY', 'TLT', 'GLD']
    prices = download_etfs(tickers, start='2010-01-01')
    vix = download_vix()
    prices['VIX'] = vix
    prices = prices.dropna()
    print(f"Data: {len(prices)} rows")

    rets = {t: prices[t].pct_change() for t in tickers}
    start = 252

    def run_vix_leverage(prices, rets, start, thresholds, weights, cash_asset='SHY'):
        """
        thresholds: list of VIX levels [t1, t2, t3]
        weights: list of UPRO/SPY weights for each regime [w1, w2, w3, w4]
          regime 0: VIX < t1
          regime 1: t1 <= VIX < t2
          regime 2: t2 <= VIX < t3
          regime 3: VIX >= t3
        weights format: (upro_pct, spy_pct, cash_pct)
        """
        eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
        cash_ret = rets[cash_asset]

        for i in range(start, len(prices)):
            v = prices['VIX'].iloc[i]
            if v < thresholds[0]:
                w = weights[0]
            elif v < thresholds[1]:
                w = weights[1]
            elif v < thresholds[2]:
                w = weights[2]
            else:
                w = weights[3]

            ret = w[0] * rets['UPRO'].iloc[i] + w[1] * rets['SPY'].iloc[i] + w[2] * cash_ret.iloc[i]
            eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]

        return eq.iloc[start:]

    # Grid search
    configs = {
        # Original
        'Original (15/20/30)': {
            'thresholds': [15, 20, 30],
            'weights': [(0.50, 0, 0.50), (0, 0.80, 0.20), (0, 0.40, 0.60), (0, 0, 1.0)]
        },
        # More aggressive
        'Aggressive (13/18/25)': {
            'thresholds': [13, 18, 25],
            'weights': [(0.70, 0, 0.30), (0.30, 0.40, 0.30), (0, 0.50, 0.50), (0, 0, 1.0)]
        },
        # More conservative
        'Conservative (17/22/28)': {
            'thresholds': [17, 22, 28],
            'weights': [(0.40, 0.20, 0.40), (0, 0.60, 0.40), (0, 0.30, 0.70), (0, 0, 1.0)]
        },
        # Binary (simple)
        'Binary (20)': {
            'thresholds': [20, 99, 100],  # Only 1 threshold matters
            'weights': [(0.50, 0.20, 0.30), (0, 0.20, 0.80), (0, 0.20, 0.80), (0, 0.20, 0.80)]
        },
        # Ultra-aggressive
        'Ultra-Agg (12/16/22)': {
            'thresholds': [12, 16, 22],
            'weights': [(0.80, 0, 0.20), (0.50, 0.20, 0.30), (0, 0.60, 0.40), (0, 0, 1.0)]
        },
        # Percentile-based thresholds (approx VIX percentiles)
        'Pctile (25/50/75)': {  # ~13/17/23 VIX
            'thresholds': [13, 17, 23],
            'weights': [(0.60, 0, 0.40), (0.30, 0.30, 0.40), (0, 0.40, 0.60), (0, 0, 1.0)]
        },
        # With TLT as cash alternative
        'TLT Cash (15/20/30)': {
            'thresholds': [15, 20, 30],
            'weights': [(0.50, 0, 0.50), (0, 0.80, 0.20), (0, 0.40, 0.60), (0, 0, 1.0)]
        },
        # With GLD as crisis hedge
        'GLD Crisis (15/20/30)': {
            'thresholds': [15, 20, 30],
            'weights': [(0.50, 0, 0.50), (0, 0.80, 0.20), (0, 0.40, 0.60), (0, 0, 1.0)]
        },
    }

    results = {}
    print(f"\n{'Config':<25s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'MaxDD':>7s} {'Calmar':>7s}")
    print(f"{'-'*25} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")

    for name, cfg in configs.items():
        cash = 'TLT' if 'TLT' in name else ('GLD' if 'GLD' in name else 'SHY')
        eq = run_vix_leverage(prices, rets, start, cfg['thresholds'], cfg['weights'], cash)
        m = compute_metrics(eq, name)
        results[name] = {'equity': eq, 'metrics': m}
        print(f"{name:<25s} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1%} {m['max_dd']:>6.1%} {m['calmar']:>7.3f}")

    # SPY benchmark
    spy_eq = prices['SPY'].iloc[start:] / prices['SPY'].iloc[start] * INITIAL_CAPITAL
    spy_m = compute_metrics(spy_eq, 'SPY B&H')
    print(f"{'SPY B&H':<25s} {spy_m['sharpe']:>7.3f} {spy_m['sortino']:>8.3f} "
          f"{spy_m['cagr']:>6.1%} {spy_m['max_dd']:>6.1%} {spy_m['calmar']:>7.3f}")

    # Find best by Sharpe
    best_name = max(results, key=lambda k: results[k]['metrics']['sharpe'])
    best_eq = results[best_name]['equity']
    best_m = results[best_name]['metrics']

    print(f"\nBest: {best_name}")

    # Full adversarial on best
    adv = full_adversarial(best_eq, prices['SPY'])
    print(f"Adversarial: {adv['gates_passed']}/3")
    for test in ['permutation', 'subperiod', 'regime']:
        t = adv[test]
        key = 'p_value' if test == 'permutation' else 'cv' if test == 'subperiod' else 'gap'
        print(f"  {test}: {t.get(key, '?')} {'PASS' if t['pass'] else 'FAIL'}")

    # Also test: does adding v4.4 SMA as confirmation improve things?
    print(f"\n{'='*70}")
    print("BONUS: VIX + SMA CONFIRMATION")
    print("=" * 70)

    sma50 = prices['SPY'].rolling(50).mean()
    sma200 = prices['SPY'].rolling(200).mean()

    eq_combined = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        trend_up = sma50.iloc[i] > sma200.iloc[i]

        if v < 15 and trend_up:
            ret = 0.60 * rets['UPRO'].iloc[i] + 0.40 * rets['SHY'].iloc[i]
        elif v < 20 and trend_up:
            ret = 0.30 * rets['UPRO'].iloc[i] + 0.70 * rets['SHY'].iloc[i]
        elif v < 20:
            ret = 0.50 * rets['SPY'].iloc[i] + 0.50 * rets['SHY'].iloc[i]
        elif v < 30:
            ret = 0.30 * rets['SPY'].iloc[i] + 0.70 * rets['SHY'].iloc[i]
        else:
            ret = rets['SHY'].iloc[i]

        eq_combined.iloc[i] = eq_combined.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq_combined.iloc[i-1]

    m_comb = compute_metrics(eq_combined.iloc[start:], "VIX + SMA Combined")
    print(f"  Sharpe: {m_comb['sharpe']:.3f}, CAGR: {m_comb['cagr']:.1%}, MaxDD: {m_comb['max_dd']:.1%}")

    adv_comb = full_adversarial(eq_combined.iloc[start:], prices['SPY'])
    print(f"  Adversarial: {adv_comb['gates_passed']}/3")

    # Final emit
    emit_result(
        name=f"VIX Leverage Optimizer ({best_name})",
        description="Grid search over VIX thresholds and leverage levels",
        metrics=best_m,
        adversarial=adv,
        extra={'combined_sharpe': m_comb['sharpe'], 'combined_cagr': m_comb['cagr']}
    )

if __name__ == '__main__':
    main()
