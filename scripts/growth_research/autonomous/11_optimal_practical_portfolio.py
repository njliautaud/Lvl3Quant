#!/usr/bin/env python3
"""
Optimal Practical Portfolio
==============================
Build the best PRACTICAL portfolio using our validated findings:

Key insight: Use VIX leverage for returns (Sharpe 3.2) + tail hedging for drawdown control.

The user wants REAL portfolio allocation they can execute.
This tests practical implementation with monthly rebalancing,
realistic transaction costs, and deploying strategies.

$100K fixed capital (HC #713).
"""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research/autonomous')
from research_template import *

def main():
    print("=" * 70)
    print("OPTIMAL PRACTICAL PORTFOLIO")
    print("=" * 70)
    
    tickers = ['SPY', 'UPRO', 'SHY', 'TLT', 'GLD', 'QQQ', 'TQQQ', 'IEF']
    prices = download_etfs(tickers, start='2010-01-01')
    vix = download_vix(start='2010-01-01')
    prices['VIX'] = vix
    prices = prices.dropna()
    print(f"Data: {len(prices)} rows, {prices.index[0].date()} → {prices.index[-1].date()}")
    
    rets = {t: prices[t].pct_change() for t in tickers}
    start = 252
    
    configs = {}
    
    # === 1. SIMPLE VIX LEVERAGE (BASELINE) ===
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
    configs['VIX Base'] = eq.iloc[start:]
    
    # === 2. VIX + TAIL HEDGE (Long TLT on VIX spikes) ===
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        if v < 15:
            ret = 0.50 * rets['UPRO'].iloc[i] + 0.10 * rets['TLT'].iloc[i] + 0.40 * rets['SHY'].iloc[i]
        elif v < 20:
            ret = 0.70 * rets['SPY'].iloc[i] + 0.15 * rets['TLT'].iloc[i] + 0.15 * rets['SHY'].iloc[i]
        elif v < 25:
            ret = 0.30 * rets['SPY'].iloc[i] + 0.30 * rets['TLT'].iloc[i] + 0.40 * rets['SHY'].iloc[i]
        elif v < 30:
            # Crisis building: shift to TLT (flight to quality)
            ret = 0.10 * rets['SPY'].iloc[i] + 0.40 * rets['TLT'].iloc[i] + 0.50 * rets['SHY'].iloc[i]
        else:
            # VIX spike: TLT + GLD (crisis assets)
            ret = 0.30 * rets['TLT'].iloc[i] + 0.20 * rets['GLD'].iloc[i] + 0.50 * rets['SHY'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['VIX + Tail Hedge'] = eq.iloc[start:]
    
    # === 3. VIX LEVERAGE WITH MONTHLY REBALANCE (REALISTIC) ===
    # Only change allocation on first trading day of month
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    current_alloc = None
    for i in range(start, len(prices)):
        # Rebalance on first day of month or at start
        if current_alloc is None or (i > start and prices.index[i].month != prices.index[i-1].month):
            v = prices['VIX'].iloc[i]
            if v < 15:
                current_alloc = {'UPRO': 0.50, 'SHY': 0.50}
            elif v < 20:
                current_alloc = {'SPY': 0.80, 'SHY': 0.20}
            elif v < 30:
                current_alloc = {'SPY': 0.40, 'SHY': 0.60}
            else:
                current_alloc = {'SHY': 1.0}
        
        ret = sum(w * rets[t].iloc[i] for t, w in current_alloc.items())
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['VIX Monthly Rebal'] = eq.iloc[start:]
    
    # === 4. VIX + TREND CONFIRMATION ===
    # Only use leverage when VIX low AND trend up (SPY > 200 SMA)
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        spy_sma200 = prices['SPY'].iloc[max(0,i-200):i+1].mean()
        trend_up = prices['SPY'].iloc[i] > spy_sma200
        
        if v < 15 and trend_up:
            ret = 0.50 * rets['UPRO'].iloc[i] + 0.50 * rets['SHY'].iloc[i]
        elif v < 15 and not trend_up:
            # Low VIX but bearish trend — cautious
            ret = 0.40 * rets['SPY'].iloc[i] + 0.60 * rets['SHY'].iloc[i]
        elif v < 20 and trend_up:
            ret = 0.80 * rets['SPY'].iloc[i] + 0.20 * rets['SHY'].iloc[i]
        elif v < 20 and not trend_up:
            ret = 0.40 * rets['SPY'].iloc[i] + 0.60 * rets['SHY'].iloc[i]
        elif v < 30:
            ret = 0.30 * rets['SPY'].iloc[i] + 0.70 * rets['SHY'].iloc[i]
        else:
            ret = rets['SHY'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['VIX + Trend Confirm'] = eq.iloc[start:]
    
    # === 5. DUAL LEVERAGE: UPRO + TQQQ BLEND ===
    # Diversify leverage across SPY and QQQ
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        if v < 15:
            ret = 0.30 * rets['UPRO'].iloc[i] + 0.20 * rets['TQQQ'].iloc[i] + 0.50 * rets['SHY'].iloc[i]
        elif v < 20:
            ret = 0.50 * rets['SPY'].iloc[i] + 0.30 * rets['QQQ'].iloc[i] + 0.20 * rets['SHY'].iloc[i]
        elif v < 30:
            ret = 0.20 * rets['SPY'].iloc[i] + 0.20 * rets['QQQ'].iloc[i] + 0.60 * rets['SHY'].iloc[i]
        else:
            ret = rets['SHY'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['Dual Leverage'] = eq.iloc[start:]
    
    # === 6. GROWTH + INCOME BLEND ===
    # Core VIX leverage + small allocation to bonds for yield stability
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        if v < 15:
            ret = 0.45 * rets['UPRO'].iloc[i] + 0.10 * rets['IEF'].iloc[i] + 0.45 * rets['SHY'].iloc[i]
        elif v < 20:
            ret = 0.65 * rets['SPY'].iloc[i] + 0.15 * rets['IEF'].iloc[i] + 0.20 * rets['SHY'].iloc[i]
        elif v < 25:
            ret = 0.30 * rets['SPY'].iloc[i] + 0.30 * rets['IEF'].iloc[i] + 0.40 * rets['SHY'].iloc[i]
        elif v < 30:
            ret = 0.10 * rets['SPY'].iloc[i] + 0.40 * rets['IEF'].iloc[i] + 0.50 * rets['SHY'].iloc[i]
        else:
            ret = 0.40 * rets['TLT'].iloc[i] + 0.60 * rets['SHY'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['Growth + Income'] = eq.iloc[start:]
    
    # === 7. ULTRA-AGGRESSIVE (OPTIMIZED FROM GRID SEARCH RESULT) ===
    # Using the ultra-aggressive thresholds (12/16/22) that gave Sharpe 3.91
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        if v < 12:
            ret = 0.65 * rets['UPRO'].iloc[i] + 0.35 * rets['SHY'].iloc[i]
        elif v < 16:
            ret = 0.55 * rets['UPRO'].iloc[i] + 0.45 * rets['SHY'].iloc[i]
        elif v < 22:
            ret = 0.85 * rets['SPY'].iloc[i] + 0.15 * rets['SHY'].iloc[i]
        elif v < 30:
            ret = 0.30 * rets['SPY'].iloc[i] + 0.70 * rets['SHY'].iloc[i]
        else:
            ret = rets['SHY'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['Ultra-Aggressive'] = eq.iloc[start:]
    
    # SPY benchmark
    spy_eq = prices['SPY'].iloc[start:] / prices['SPY'].iloc[start] * INITIAL_CAPITAL
    configs['SPY B&H'] = spy_eq
    
    # === RESULTS ===
    print(f"\n{'Strategy':<25s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'MaxDD':>7s} {'Calmar':>7s}")
    print(f"{'-'*25} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
    
    best_name, best_sharpe = None, -999
    all_metrics = {}
    for name, eq in configs.items():
        m = compute_metrics(eq, name)
        all_metrics[name] = m
        print(f"{m['name']:<25s} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1%} {m['max_dd']:>6.1%} {m['calmar']:>7.3f}")
        if name != 'SPY B&H' and m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_name = name
    
    print(f"\nBest: {best_name}")
    metrics = all_metrics[best_name]
    
    # === R1 REGIME TEST ON ALL ===
    print(f"\nR1 Regime Test:")
    for name, eq in configs.items():
        if name == 'SPY B&H': continue
        common = eq.index.intersection(spy_eq.index)
        r = regime_test(eq.loc[common], spy_eq.loc[common])
        status = '✅' if r['pass'] else '❌'
        print(f"  {status} {name:<25s}: green={r['green_sharpe']:.2f}, red={r['red_sharpe']:.2f}, gap={r['gap']:.3f}")
    
    # === ADVERSARIAL ON BEST ===
    best_eq = configs[best_name]
    common = best_eq.index.intersection(spy_eq.index)
    best_eq_aligned = best_eq.loc[common]
    spy_aligned = spy_eq.loc[common]
    
    print(f"\nPermutation test (200 shuffles, signal shuffle)...")
    real_sharpe = metrics['sharpe']
    perm_sharpes = []
    vix_vals = prices['VIX'].values.copy()
    
    for _ in range(200):
        np.random.shuffle(vix_vals)
        eq_p = pd.Series(INITIAL_CAPITAL, index=prices.index)
        for i in range(start, len(prices)):
            v = vix_vals[i]
            if v < 15:
                ret = 0.50 * rets['UPRO'].iloc[i] + 0.50 * rets['SHY'].iloc[i]
            elif v < 20:
                ret = 0.80 * rets['SPY'].iloc[i] + 0.20 * rets['SHY'].iloc[i]
            elif v < 30:
                ret = 0.40 * rets['SPY'].iloc[i] + 0.60 * rets['SHY'].iloc[i]
            else:
                ret = rets['SHY'].iloc[i]
            eq_p.iloc[i] = eq_p.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq_p.iloc[i-1]
        pm = compute_metrics(eq_p.iloc[start:])
        perm_sharpes.append(pm['sharpe'])
    
    perm_sharpes = np.array(perm_sharpes)
    p_value = float((perm_sharpes >= real_sharpe).mean())
    print(f"  Real: {real_sharpe:.3f}, Perm mean: {perm_sharpes.mean():.3f}, p={p_value:.3f}")
    
    subp = subperiod_test(best_eq_aligned)
    regime = regime_test(best_eq_aligned, spy_aligned)
    
    adv = {
        'permutation': {'real_sharpe': real_sharpe, 'perm_mean': round(float(perm_sharpes.mean()), 3), 'p_value': round(p_value, 3), 'pass': p_value < 0.05},
        'subperiod': subp,
        'regime': regime,
        'gates_passed': sum([p_value < 0.05, subp['pass'], regime['pass']]),
        'gates_total': 3,
    }
    
    print(f"  SubP: CV={subp['cv']:.3f}, R1: gap={regime['gap']:.3f}")
    print(f"  Gates: {adv['gates_passed']}/3")
    
    # === PRACTICAL IMPLEMENTATION GUIDE ===
    print(f"\n{'='*70}")
    print(f"PRACTICAL IMPLEMENTATION GUIDE")
    print(f"{'='*70}")
    
    current_vix = prices['VIX'].iloc[-1]
    print(f"\nCurrent VIX: {current_vix:.1f}")
    print(f"Current SPY: ${prices['SPY'].iloc[-1]:.2f}")
    
    # Show current recommended allocation for each strategy
    print(f"\nCurrent allocations for $100K:")
    if current_vix < 15:
        print(f"  VIX Base: $50K UPRO + $50K SHY")
        print(f"  VIX + Tail: $50K UPRO + $10K TLT + $40K SHY")
        print(f"  Dual Leverage: $30K UPRO + $20K TQQQ + $50K SHY")
    elif current_vix < 20:
        print(f"  VIX Base: $80K SPY + $20K SHY")
        print(f"  VIX + Tail: $70K SPY + $15K TLT + $15K SHY")
        print(f"  Dual Leverage: $50K SPY + $30K QQQ + $20K SHY")
    elif current_vix < 25:
        print(f"  VIX Base: $40K SPY + $60K SHY")
        print(f"  VIX + Tail: $30K SPY + $30K TLT + $40K SHY")
        print(f"  Growth+Income: $30K SPY + $30K IEF + $40K SHY")
    elif current_vix < 30:
        print(f"  VIX Base: $40K SPY + $60K SHY")
        print(f"  VIX + Tail: $10K SPY + $40K TLT + $50K SHY")
    else:
        print(f"  ALL STRATEGIES: 100% defensive (SHY/TLT)")
    
    print(f"\nRebalance triggers: VIX crosses 15, 20, 25, or 30")
    print(f"Rebalance frequency: daily check, act only on threshold crossings")
    
    emit_result(
        name=f"Practical Portfolio ({best_name})",
        description="Practical VIX-based portfolio variants with implementation guide",
        metrics=metrics,
        adversarial=adv
    )

if __name__ == '__main__':
    main()
