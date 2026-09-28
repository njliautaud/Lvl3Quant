#!/usr/bin/env python3
"""
Momentum-Quality Blend: Combine momentum (MTUM) with quality (QUAL)
to reduce momentum crash risk. Quality provides defensive ballast
during momentum drawdowns.

Hypothesis: MTUM+QUAL blend beats either alone because momentum crashes
(2009, 2020) are offset by quality's defensive nature.
"""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research/autonomous')
from research_template import *

def main():
    print("=" * 70)
    print("MOMENTUM-QUALITY BLEND")
    print("=" * 70)

    tickers = ['SPY', 'MTUM', 'QUAL', 'USMV', 'SHY', 'UPRO']
    prices = download_etfs(tickers)
    vix = download_vix()
    prices['VIX'] = vix
    prices = prices.dropna()
    print(f"Data: {len(prices)} rows, {prices.index[0].date()} → {prices.index[-1].date()}")

    spy_ret = prices['SPY'].pct_change()
    mtum_ret = prices['MTUM'].pct_change()
    qual_ret = prices['QUAL'].pct_change()
    usmv_ret = prices['USMV'].pct_change()
    shy_ret = prices['SHY'].pct_change()

    configs = {}

    # Config 1: Static 50/50 MTUM/QUAL
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(1, len(prices)):
        ret = 0.50 * mtum_ret.iloc[i] + 0.50 * qual_ret.iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['50/50 MTUM/QUAL'] = eq

    # Config 2: Momentum-tilted in uptrend, quality-tilted in downtrend
    sma200 = prices['SPY'].rolling(200).mean()
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(200, len(prices)):
        if prices['SPY'].iloc[i] > sma200.iloc[i]:
            ret = 0.70 * mtum_ret.iloc[i] + 0.30 * qual_ret.iloc[i]
        else:
            ret = 0.30 * mtum_ret.iloc[i] + 0.70 * qual_ret.iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['Adaptive MTUM/QUAL'] = eq.iloc[200:]

    # Config 3: Add low-vol as crisis hedge
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(200, len(prices)):
        vix_val = prices['VIX'].iloc[i]
        if vix_val > 25:
            ret = 0.20 * mtum_ret.iloc[i] + 0.30 * qual_ret.iloc[i] + 0.50 * usmv_ret.iloc[i]
        elif prices['SPY'].iloc[i] > sma200.iloc[i]:
            ret = 0.50 * mtum_ret.iloc[i] + 0.30 * qual_ret.iloc[i] + 0.20 * usmv_ret.iloc[i]
        else:
            ret = 0.30 * mtum_ret.iloc[i] + 0.40 * qual_ret.iloc[i] + 0.30 * usmv_ret.iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['Adaptive 3-Factor'] = eq.iloc[200:]

    # Config 4: VIX-scaled leverage (UPRO when calm, defensive when stressed)
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    upro_ret = prices['UPRO'].pct_change()
    for i in range(200, len(prices)):
        vix_val = prices['VIX'].iloc[i]
        if vix_val < 15:
            # Very calm: leverage via UPRO + quality
            ret = 0.50 * upro_ret.iloc[i] + 0.30 * qual_ret.iloc[i] + 0.20 * shy_ret.iloc[i]
        elif vix_val < 20:
            ret = 0.40 * mtum_ret.iloc[i] + 0.40 * qual_ret.iloc[i] + 0.20 * shy_ret.iloc[i]
        elif vix_val < 30:
            ret = 0.30 * qual_ret.iloc[i] + 0.30 * usmv_ret.iloc[i] + 0.40 * shy_ret.iloc[i]
        else:
            ret = 0.20 * qual_ret.iloc[i] + 0.80 * shy_ret.iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['VIX-Scaled Factor'] = eq.iloc[200:]

    # SPY benchmark
    spy_eq = prices['SPY'].iloc[200:] / prices['SPY'].iloc[200] * INITIAL_CAPITAL
    configs['SPY B&H'] = spy_eq

    # Results
    print(f"\n{'Strategy':<25s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'MaxDD':>7s}")
    print(f"{'-'*25} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")

    best_name = None
    best_sharpe = -999

    for name, eq in configs.items():
        m = compute_metrics(eq, name)
        print(f"{m['name']:<25s} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1%} {m['max_dd']:>6.1%}")
        if name != 'SPY B&H' and m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_name = name

    # Adversarial on best
    print(f"\nBest: {best_name}")
    best_eq = configs[best_name]
    metrics = compute_metrics(best_eq, best_name)
    adv = full_adversarial(best_eq, prices['SPY'])

    print(f"\nAdversarial: {adv['gates_passed']}/3 gates")
    print(f"  Perm p={adv['permutation']['p_value']} {'PASS' if adv['permutation']['pass'] else 'FAIL'}")
    print(f"  SubP CV={adv['subperiod']['cv']} {'PASS' if adv['subperiod']['pass'] else 'FAIL'}")
    print(f"  R1 gap={adv['regime']['gap']} {'PASS' if adv['regime']['pass'] else 'FAIL'}")

    emit_result(
        name=f"Momentum-Quality Blend ({best_name})",
        description="Factor blend: MTUM+QUAL with optional VIX/trend gating",
        metrics=metrics,
        adversarial=adv
    )


if __name__ == '__main__':
    main()
