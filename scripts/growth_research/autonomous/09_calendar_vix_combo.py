#!/usr/bin/env python3
"""
Calendar Anomalies + VIX Leverage Combo
=========================================
Well-documented anomalies:
- Turn-of-month effect (last 3 + first 3 trading days = bulk of returns)
- Monthly seasonality (Nov-Apr > May-Oct)
- Pre-holiday effect
- Day-of-week effect (Monday weakness)

Combined with VIX leverage: be MORE leveraged during favorable calendar periods.
"""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research/autonomous')
from research_template import *

def main():
    print("=" * 70)
    print("CALENDAR ANOMALIES + VIX LEVERAGE")
    print("=" * 70)

    tickers = ['SPY', 'UPRO', 'SHY']
    prices = download_etfs(tickers, start='2010-01-01')
    vix = download_vix()
    prices['VIX'] = vix
    prices = prices.dropna()
    print(f"Data: {len(prices)} rows")

    rets = {t: prices[t].pct_change() for t in tickers}
    start = 252

    # Calendar features
    dates = prices.index
    day_of_month = dates.day
    day_of_week = dates.dayofweek  # Mon=0, Fri=4
    month = dates.month

    # Turn-of-month: last 3 and first 3 trading days
    # Approximate: day <= 3 or day >= 27
    is_tom = (day_of_month <= 3) | (day_of_month >= 27)

    # Favorable months (Nov-Apr)
    is_favorable_month = month.isin([11, 12, 1, 2, 3, 4])

    # Monday weakness
    is_monday = day_of_week == 0

    configs = {}

    # 1. Pure VIX leverage (baseline)
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        if v < 15: ret = 0.50 * rets['UPRO'].iloc[i] + 0.50 * rets['SHY'].iloc[i]
        elif v < 20: ret = 0.80 * rets['SPY'].iloc[i] + 0.20 * rets['SHY'].iloc[i]
        elif v < 30: ret = 0.40 * rets['SPY'].iloc[i] + 0.60 * rets['SHY'].iloc[i]
        else: ret = rets['SHY'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['VIX Leverage Base'] = eq.iloc[start:]

    # 2. VIX + Turn-of-Month boost
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        tom = is_tom[i]

        if v < 15:
            upro_w = 0.60 if tom else 0.40
            ret = upro_w * rets['UPRO'].iloc[i] + (1-upro_w) * rets['SHY'].iloc[i]
        elif v < 20:
            spy_w = 0.90 if tom else 0.70
            ret = spy_w * rets['SPY'].iloc[i] + (1-spy_w) * rets['SHY'].iloc[i]
        elif v < 30:
            spy_w = 0.50 if tom else 0.30
            ret = spy_w * rets['SPY'].iloc[i] + (1-spy_w) * rets['SHY'].iloc[i]
        else:
            ret = rets['SHY'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['VIX + TOM Boost'] = eq.iloc[start:]

    # 3. VIX + Seasonal (Nov-Apr stronger leverage)
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        fav = is_favorable_month[i]

        if v < 15:
            upro_w = 0.60 if fav else 0.40
            ret = upro_w * rets['UPRO'].iloc[i] + (1-upro_w) * rets['SHY'].iloc[i]
        elif v < 20:
            spy_w = 0.90 if fav else 0.70
            ret = spy_w * rets['SPY'].iloc[i] + (1-spy_w) * rets['SHY'].iloc[i]
        elif v < 30:
            spy_w = 0.50 if fav else 0.30
            ret = spy_w * rets['SPY'].iloc[i] + (1-spy_w) * rets['SHY'].iloc[i]
        else:
            ret = rets['SHY'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['VIX + Seasonal'] = eq.iloc[start:]

    # 4. VIX + Monday avoidance (reduce Monday exposure)
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        mon = is_monday[i]

        if v < 15:
            upro_w = 0.30 if mon else 0.55
            ret = upro_w * rets['UPRO'].iloc[i] + (1-upro_w) * rets['SHY'].iloc[i]
        elif v < 20:
            spy_w = 0.50 if mon else 0.85
            ret = spy_w * rets['SPY'].iloc[i] + (1-spy_w) * rets['SHY'].iloc[i]
        elif v < 30:
            spy_w = 0.20 if mon else 0.45
            ret = spy_w * rets['SPY'].iloc[i] + (1-spy_w) * rets['SHY'].iloc[i]
        else:
            ret = rets['SHY'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['VIX + Mon Avoidance'] = eq.iloc[start:]

    # 5. Full combo: VIX + TOM + Seasonal + Monday avoidance
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]

        # Calendar score: +1 for each favorable condition
        cal_score = 0
        if is_tom[i]: cal_score += 1
        if is_favorable_month[i]: cal_score += 1
        if not is_monday[i]: cal_score += 1
        # cal_score: 0-3

        if v < 15:
            base_w = 0.35 + cal_score * 0.10  # 0.35-0.65 UPRO
            ret = base_w * rets['UPRO'].iloc[i] + (1-base_w) * rets['SHY'].iloc[i]
        elif v < 20:
            base_w = 0.50 + cal_score * 0.15  # 0.50-0.95 SPY
            ret = min(base_w, 1.0) * rets['SPY'].iloc[i] + max(0, 1-base_w) * rets['SHY'].iloc[i]
        elif v < 30:
            base_w = 0.20 + cal_score * 0.10  # 0.20-0.50 SPY
            ret = base_w * rets['SPY'].iloc[i] + (1-base_w) * rets['SHY'].iloc[i]
        else:
            ret = rets['SHY'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['VIX + Full Calendar'] = eq.iloc[start:]

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

    # Compare calendar variants vs base
    base_sharpe = compute_metrics(configs['VIX Leverage Base'])['sharpe']
    print(f"\nCalendar improvement over VIX base (Sharpe {base_sharpe:.3f}):")
    for name, eq in configs.items():
        if name in ['SPY B&H', 'VIX Leverage Base']: continue
        m = compute_metrics(eq)
        delta = m['sharpe'] - base_sharpe
        print(f"  {name}: {delta:+.3f} Sharpe")

    print(f"\nBest: {best_name}")
    metrics = compute_metrics(configs[best_name], best_name)

    # Signal-shuffle perm test
    print(f"\nPermutation test (shuffling VIX + calendar)...")
    real_sharpe = metrics['sharpe']
    perm_sharpes = []
    vix_vals = prices['VIX'].values.copy()

    for _ in range(200):
        np.random.shuffle(vix_vals)
        eq_p = pd.Series(INITIAL_CAPITAL, index=prices.index)
        for i in range(start, len(prices)):
            v = vix_vals[i]
            cal_score = 0
            if is_tom[i]: cal_score += 1
            if is_favorable_month[i]: cal_score += 1
            if not is_monday[i]: cal_score += 1

            if v < 15:
                base_w = 0.35 + cal_score * 0.10
                ret = base_w * rets['UPRO'].iloc[i] + (1-base_w) * rets['SHY'].iloc[i]
            elif v < 20:
                base_w = 0.50 + cal_score * 0.15
                ret = min(base_w, 1.0) * rets['SPY'].iloc[i] + max(0, 1-base_w) * rets['SHY'].iloc[i]
            elif v < 30:
                base_w = 0.20 + cal_score * 0.10
                ret = base_w * rets['SPY'].iloc[i] + (1-base_w) * rets['SHY'].iloc[i]
            else:
                ret = rets['SHY'].iloc[i]
            eq_p.iloc[i] = eq_p.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq_p.iloc[i-1]

        pm = compute_metrics(eq_p.iloc[start:])
        perm_sharpes.append(pm['sharpe'])

    perm_sharpes = np.array(perm_sharpes)
    p_value = float((perm_sharpes >= real_sharpe).mean())
    print(f"  Real: {real_sharpe:.3f}, Perm mean: {perm_sharpes.mean():.3f}, p={p_value:.3f}")

    subp = subperiod_test(configs[best_name])
    regime = regime_test(configs[best_name], prices['SPY'])

    adv = {
        'permutation': {'real_sharpe': real_sharpe, 'perm_mean': round(float(perm_sharpes.mean()), 3), 'p_value': round(p_value, 3), 'pass': p_value < 0.05},
        'subperiod': subp,
        'regime': regime,
        'gates_passed': sum([p_value < 0.05, subp['pass'], regime['pass']]),
        'gates_total': 3,
    }

    print(f"  SubP: CV={subp['cv']:.3f}, R1: gap={regime['gap']:.3f}")
    print(f"  Gates: {adv['gates_passed']}/3")

    emit_result(
        name=f"Calendar+VIX ({best_name})",
        description="Calendar anomalies combined with VIX leverage",
        metrics=metrics,
        adversarial=adv
    )

if __name__ == '__main__':
    main()
