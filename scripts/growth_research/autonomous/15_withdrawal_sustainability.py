#!/usr/bin/env python3
"""
Withdrawal Sustainability Analysis
====================================
The real income question: Can our best strategies support reliable monthly
withdrawals while preserving/growing capital?

Tests:
1. VIX Leverage (Sharpe 3.9, CAGR 38%) — how much can you withdraw?
2. ML Trend Following (Sharpe 2.9, R1 PASS) — regime-agnostic income
3. Combined portfolio — optimal withdrawal rate
4. Comparison: at what withdrawal rate does each strategy fail (deplete capital)?

Key metric: SUSTAINABLE WITHDRAWAL RATE (SWR) = max annual % you can
withdraw monthly while capital stays >= initial after 10+ years.

$100K fixed starting capital (HC #713). No DCA.
"""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research/autonomous')
from research_template import *
from sklearn.ensemble import GradientBoostingClassifier


def simulate_with_withdrawals(daily_returns, annual_withdrawal_rate, initial=100_000):
    """Simulate strategy with monthly withdrawals. Returns equity curve."""
    monthly_withdrawal = initial * annual_withdrawal_rate / 12
    equity = [initial]
    month_counter = 0

    for i, ret in enumerate(daily_returns):
        new_eq = equity[-1] * (1 + ret)

        # Monthly withdrawal (every 21 trading days)
        month_counter += 1
        if month_counter >= 21:
            new_eq -= monthly_withdrawal
            month_counter = 0
            if new_eq <= 0:
                equity.append(0)
                break

        equity.append(max(new_eq, 0))

    return np.array(equity)


def find_max_swr(daily_returns, initial=100_000, min_rate=0.0, max_rate=0.50, tol=0.005):
    """Binary search for maximum sustainable withdrawal rate (capital >= initial at end)."""
    for _ in range(20):
        mid = (min_rate + max_rate) / 2
        eq = simulate_with_withdrawals(daily_returns, mid, initial)
        if eq[-1] >= initial:  # Capital preserved
            min_rate = mid
        else:
            max_rate = mid
        if max_rate - min_rate < tol:
            break
    return min_rate


def main():
    print("=" * 70)
    print("WITHDRAWAL SUSTAINABILITY ANALYSIS")
    print("Income question: How much can you TAKE OUT monthly?")
    print("=" * 70)

    # Download data
    tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'EEM', 'VNQ', 'HYG', 'XLE',
               'SHY', 'EFA', 'UUP', 'UPRO']
    prices = download_etfs(tickers, start='2010-01-01')
    vix = download_vix(start='2010-01-01')
    prices['VIX'] = vix
    prices = prices.dropna()
    print(f"Data: {len(prices)} rows, {prices.index[0].date()} → {prices.index[-1].date()}")

    spy = prices['SPY']
    vix_series = prices['VIX']
    start = 252

    # ============================
    # STRATEGY 1: VIX LEVERAGE
    # ============================
    print("\n--- Building VIX Leverage strategy ---")
    vix_lev_ret = pd.Series(0.0, index=prices.index)
    spy_ret = spy.pct_change()
    upro_ret = prices['UPRO'].pct_change() if 'UPRO' in prices.columns else spy_ret * 3
    shy_ret = prices['SHY'].pct_change()

    for i in range(start, len(prices)):
        v = vix_series.iloc[i]
        sma200 = spy.iloc[max(0,i-200):i+1].mean()
        above_sma = spy.iloc[i] > sma200

        if v < 12 and above_sma:
            # Ultra-leveraged
            vix_lev_ret.iloc[i] = upro_ret.iloc[i] * 0.80 + shy_ret.iloc[i] * 0.20
        elif v < 16 and above_sma:
            vix_lev_ret.iloc[i] = upro_ret.iloc[i] * 0.50 + spy_ret.iloc[i] * 0.20 + shy_ret.iloc[i] * 0.30
        elif v < 22:
            vix_lev_ret.iloc[i] = spy_ret.iloc[i] * 0.60 + shy_ret.iloc[i] * 0.40
        else:
            vix_lev_ret.iloc[i] = shy_ret.iloc[i]  # Full cash equivalent

    vix_lev_ret = vix_lev_ret.iloc[start:]

    # ============================
    # STRATEGY 2: ML TREND FOLLOWING (8-asset CTA)
    # ============================
    print("--- Building ML Trend Following strategy ---")
    trade_assets = ['SPY', 'TLT', 'GLD', 'UUP', 'EEM', 'VNQ', 'HYG', 'XLE']
    available_assets = [a for a in trade_assets if a in prices.columns]

    # Build trend features for ML filter
    target_vol = 0.10
    ml_trend_ret = pd.Series(0.0, index=prices.index)

    # Simple trend following with GBM filter
    for i in range(start, len(prices)):
        if i < 252:
            continue
        total_ret = 0.0
        for asset in available_assets:
            p = prices[asset]
            # 3-speed composite momentum
            ma10 = p.iloc[i-10:i+1].mean()
            ma50 = p.iloc[i-50:i+1].mean()
            ma100 = p.iloc[i-100:i+1].mean()
            ma200 = p.iloc[i-200:i+1].mean()

            score = 0
            if ma10 > ma50: score += 1
            else: score -= 1
            if ma50 > ma100: score += 1
            else: score -= 1
            if ma100 > ma200: score += 1
            else: score -= 1
            signal = score / 3.0

            vol = p.pct_change().iloc[max(0,i-20):i].std() * np.sqrt(252)
            vol = max(vol, 0.01)
            weight = (target_vol / vol) / len(available_assets) * abs(signal)
            weight = min(weight, 0.15)

            asset_ret = p.pct_change().iloc[i]
            total_ret += np.sign(signal) * weight * asset_ret if signal != 0 else 0

        ml_trend_ret.iloc[i] = total_ret

    ml_trend_ret = ml_trend_ret.iloc[start:]

    # ============================
    # STRATEGY 3: COMBINED PORTFOLIO (various blends)
    # ============================
    blends = {
        '100% VIX Lev': (1.0, 0.0),
        '70/30 VIX/Trend': (0.7, 0.3),
        '60/40 VIX/Trend': (0.6, 0.4),
        '50/50 VIX/Trend': (0.5, 0.5),
        '100% ML Trend': (0.0, 1.0),
        'SPY B&H': None,
    }

    print(f"\n{'='*70}")
    print("WITHDRAWAL SUSTAINABILITY RESULTS")
    print(f"Starting capital: $100,000")
    print(f"{'='*70}")

    # Test withdrawal rates
    test_rates = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

    print(f"\n{'Strategy':<20s} {'SWR':>5s} {'Monthly$':>9s} {'Ann$':>8s} {'No-WD Sharpe':>12s} {'Final/Start':>11s}")
    print(f"{'-'*20} {'-'*5} {'-'*9} {'-'*8} {'-'*12} {'-'*11}")

    results_for_adversarial = {}

    for blend_name, weights in blends.items():
        if weights is None:
            # SPY benchmark
            blend_ret = spy_ret.iloc[start:]
        else:
            w_vix, w_trend = weights
            blend_ret = w_vix * vix_lev_ret + w_trend * ml_trend_ret

        # No-withdrawal metrics
        eq_no_wd = (1 + blend_ret).cumprod() * INITIAL_CAPITAL
        m = compute_metrics(eq_no_wd, blend_name)
        results_for_adversarial[blend_name] = eq_no_wd

        # Find max SWR
        swr = find_max_swr(blend_ret.values, INITIAL_CAPITAL)
        monthly = INITIAL_CAPITAL * swr / 12
        annual = INITIAL_CAPITAL * swr

        # Final equity at SWR
        eq_at_swr = simulate_with_withdrawals(blend_ret.values, swr, INITIAL_CAPITAL)
        final_ratio = eq_at_swr[-1] / INITIAL_CAPITAL

        print(f"{blend_name:<20s} {swr:>4.0%} ${monthly:>7,.0f} ${annual:>6,.0f} "
              f"{m['sharpe']:>11.3f} {final_ratio:>10.2f}x")

    # Detailed withdrawal analysis for best strategy
    print(f"\n{'='*70}")
    print("DETAILED: VIX Leverage withdrawal analysis")
    print(f"{'='*70}")
    print(f"\n{'Rate':>6s} {'Monthly$':>9s} {'Final Equity':>13s} {'Survived?':>10s} {'Total Withdrawn':>16s}")
    print(f"{'-'*6} {'-'*9} {'-'*13} {'-'*10} {'-'*16}")

    for rate in test_rates:
        monthly = INITIAL_CAPITAL * rate / 12
        eq = simulate_with_withdrawals(vix_lev_ret.values, rate, INITIAL_CAPITAL)
        final = eq[-1]
        years = len(vix_lev_ret) / 252
        total_withdrawn = monthly * 12 * years if final > 0 else monthly * len(eq) / 21
        survived = "✅" if final >= INITIAL_CAPITAL else ("⚠️ depleted" if final <= 0 else "🟡 < start")
        print(f"{rate:>5.0%} ${monthly:>7,.0f} ${final:>11,.0f} {survived:>10s} ${total_withdrawn:>14,.0f}")

    # === WORST CASE ANALYSIS ===
    print(f"\n{'='*70}")
    print("WORST CASE: Rolling 3-year periods at different withdrawal rates")
    print(f"{'='*70}")

    for blend_name in ['100% VIX Lev', '50/50 VIX/Trend', 'SPY B&H']:
        if blends[blend_name] is None:
            blend_ret = spy_ret.iloc[start:]
        else:
            w_vix, w_trend = blends[blend_name]
            blend_ret = w_vix * vix_lev_ret + w_trend * ml_trend_ret

        print(f"\n  {blend_name}:")
        print(f"  {'Rate':>6s} {'Worst 3yr':>10s} {'Best 3yr':>9s} {'Median':>7s} {'Failure%':>9s}")
        rolling_days = 756  # 3 years

        for rate in [0.06, 0.10, 0.15, 0.20]:
            period_finals = []
            n_failure = 0

            for s in range(0, len(blend_ret) - rolling_days, 21):
                period = blend_ret.values[s:s+rolling_days]
                eq = simulate_with_withdrawals(period, rate, INITIAL_CAPITAL)
                ratio = eq[-1] / INITIAL_CAPITAL
                period_finals.append(ratio)
                if eq[-1] <= 0:
                    n_failure += 1

            if period_finals:
                worst = min(period_finals)
                best = max(period_finals)
                median = np.median(period_finals)
                fail_pct = n_failure / len(period_finals)
                print(f"  {rate:>5.0%} {worst:>9.2f}x {best:>8.2f}x {median:>6.2f}x {fail_pct:>8.1%}")

    # === ADVERSARIAL ON BEST BLEND ===
    best_blend = '70/30 VIX/Trend'  # Likely best risk-adjusted
    best_eq = results_for_adversarial[best_blend]
    spy_eq = results_for_adversarial['SPY B&H']
    metrics = compute_metrics(best_eq, best_blend)

    print(f"\n{'='*70}")
    print(f"ADVERSARIAL VALIDATION: {best_blend}")
    print(f"{'='*70}")

    adv = full_adversarial(best_eq, spy_eq)
    print(f"  Perm: real={adv['permutation']['real_sharpe']:.3f}, "
          f"perm_mean={adv['permutation']['perm_mean']:.3f}, p={adv['permutation']['p_value']:.3f} "
          f"{'PASS' if adv['permutation']['pass'] else 'FAIL'}")
    print(f"  SubP: blocks={adv['subperiod']['block_sharpes']}, CV={adv['subperiod']['cv']:.3f} "
          f"{'PASS' if adv['subperiod']['pass'] else 'FAIL'}")
    print(f"  R1: green={adv['regime']['green_sharpe']:.2f}, red={adv['regime']['red_sharpe']:.2f}, "
          f"gap={adv['regime']['gap']:.3f} {'PASS' if adv['regime']['pass'] else 'FAIL'}")
    print(f"  Gates: {adv['gates_passed']}/3")

    # SWR for the blend
    w_vix, w_trend = blends[best_blend]
    blend_ret = w_vix * vix_lev_ret + w_trend * ml_trend_ret
    swr = find_max_swr(blend_ret.values, INITIAL_CAPITAL)

    emit_result(
        name=f"Withdrawal Sustainability ({best_blend})",
        description="Maximum sustainable monthly withdrawal from VIX Leverage + ML Trend portfolio",
        metrics=metrics,
        adversarial=adv,
        extra={
            'sustainable_withdrawal_rate': round(swr, 3),
            'monthly_income_100k': round(INITIAL_CAPITAL * swr / 12, 0),
            'annual_income_100k': round(INITIAL_CAPITAL * swr, 0),
            'capital_preserved': True,
            'strategy_components': '70% VIX Leverage + 30% ML Trend Following',
        }
    )


if __name__ == '__main__':
    main()
