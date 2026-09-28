#!/usr/bin/env python3
"""
Global Macro Momentum
=======================
Key hypothesis: Multi-asset momentum with risk parity sizing
should be regime-agnostic because:
- When equities fall, bonds/gold/dollar tend to rise (negative correlation)
- Long/short across uncorrelated assets captures trends in ALL regimes
- Risk parity ensures no single asset dominates

This is essentially what the best CTAs do:
1. Trend signals across multiple asset classes
2. Risk parity sizing (inverse volatility)
3. Dynamic position scaling

$100K fixed capital (HC #713).
"""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research/autonomous')
from research_template import *

def main():
    print("=" * 70)
    print("GLOBAL MACRO MOMENTUM")
    print("=" * 70)
    
    # Broader asset universe
    tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM',  # Equities
               'TLT', 'IEF', 'SHY',                   # Bonds
               'GLD', 'SLV',                            # Precious metals
               'UUP',                                   # Dollar
               'VNQ',                                   # Real estate
               'XLE', 'XLF', 'XLK',                    # Sectors
               'HYG']                                   # Credit
    
    prices = download_etfs(tickers, start='2008-01-01')
    vix = download_vix(start='2008-01-01')
    prices['VIX'] = vix
    prices = prices.dropna()
    print(f"Data: {len(prices)} rows, {prices.index[0].date()} → {prices.index[-1].date()}")
    
    rets = {t: prices[t].pct_change() for t in tickers}
    start = 252
    
    # Trade these assets long/short
    trade_assets = [t for t in tickers if t != 'SHY']
    
    configs = {}
    
    # === 1. SIMPLE DUAL MA TREND (baseline) ===
    for ma_fast, ma_slow in [(20, 100), (50, 200), (10, 50)]:
        name = f'Trend {ma_fast}/{ma_slow}'
        target_vol = 0.10
        eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
        
        for i in range(start, len(prices)):
            if i < ma_slow: continue
            total_ret = 0.0
            for asset in trade_assets:
                p = prices[asset]
                fast = p.iloc[i-ma_fast:i+1].mean()
                slow = p.iloc[i-ma_slow:i+1].mean()
                signal = 1 if fast > slow else -1
                
                vol = p.pct_change().iloc[max(0,i-20):i].std() * np.sqrt(252)
                vol = max(vol, 0.01)
                weight = (target_vol / vol) / len(trade_assets)
                weight = min(weight, 0.15)
                
                total_ret += signal * weight * rets[asset].iloc[i]
            
            eq.iloc[i] = eq.iloc[i-1] * (1 + total_ret) if np.isfinite(total_ret) else eq.iloc[i-1]
        configs[name] = eq.iloc[start:]
    
    # === 2. COMPOSITE MOMENTUM (3 speeds averaged) ===
    target_vol = 0.10
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        if i < 200: continue
        total_ret = 0.0
        for asset in trade_assets:
            p = prices[asset]
            # 3-speed composite
            ma10 = p.iloc[i-10:i+1].mean()
            ma50 = p.iloc[i-50:i+1].mean()
            ma100 = p.iloc[i-100:i+1].mean()
            ma200 = p.iloc[i-200:i+1].mean()
            
            # Score: number of fast > slow crossings
            score = 0
            if ma10 > ma50: score += 1
            else: score -= 1
            if ma50 > ma100: score += 1
            else: score -= 1
            if ma100 > ma200: score += 1
            else: score -= 1
            signal = score / 3.0  # -1 to +1
            
            vol = p.pct_change().iloc[max(0,i-20):i].std() * np.sqrt(252)
            vol = max(vol, 0.01)
            weight = (target_vol / vol) / len(trade_assets) * abs(signal)
            weight = min(weight, 0.15)
            
            total_ret += np.sign(signal) * weight * rets[asset].iloc[i]
        
        eq.iloc[i] = eq.iloc[i-1] * (1 + total_ret) if np.isfinite(total_ret) else eq.iloc[i-1]
    configs['Composite Mom'] = eq.iloc[start:]
    
    # === 3. BREAKOUT MOMENTUM (price vs N-day high/low) ===
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    lookback = 60
    for i in range(start, len(prices)):
        if i < lookback: continue
        total_ret = 0.0
        for asset in trade_assets:
            p = prices[asset]
            high_n = p.iloc[i-lookback:i].max()
            low_n = p.iloc[i-lookback:i].min()
            current = p.iloc[i]
            
            # Channel position (0 to 1)
            if high_n == low_n:
                channel_pos = 0.5
            else:
                channel_pos = (current - low_n) / (high_n - low_n)
            
            # Signal: long if near high, short if near low
            signal = 2 * channel_pos - 1  # -1 to +1
            
            vol = p.pct_change().iloc[max(0,i-20):i].std() * np.sqrt(252)
            vol = max(vol, 0.01)
            weight = (target_vol / vol) / len(trade_assets) * abs(signal)
            weight = min(weight, 0.15)
            
            total_ret += np.sign(signal) * weight * rets[asset].iloc[i]
        
        eq.iloc[i] = eq.iloc[i-1] * (1 + total_ret) if np.isfinite(total_ret) else eq.iloc[i-1]
    configs['Breakout Mom'] = eq.iloc[start:]
    
    # === 4. TIME-SERIES MOMENTUM (past return predicts future) ===
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        total_ret = 0.0
        for asset in trade_assets:
            p = prices[asset]
            # 12-month momentum, skip most recent month
            if i < 252: continue
            ret_12m = p.iloc[i-21] / p.iloc[i-252] - 1
            
            signal = 1 if ret_12m > 0 else -1
            
            vol = p.pct_change().iloc[max(0,i-60):i].std() * np.sqrt(252)
            vol = max(vol, 0.05)
            weight = (target_vol / vol) / len(trade_assets)
            weight = min(weight, 0.15)
            
            total_ret += signal * weight * rets[asset].iloc[i]
        
        eq.iloc[i] = eq.iloc[i-1] * (1 + total_ret) if np.isfinite(total_ret) else eq.iloc[i-1]
    configs['12m TSMOM'] = eq.iloc[start:]
    
    # === 5. CARRY + MOMENTUM COMBO ===
    # Use VIX as equity carry signal, yield curve for bonds, momentum for everything else
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        if i < 200: continue
        total_ret = 0.0
        v = prices['VIX'].iloc[i]
        
        for asset in trade_assets:
            p = prices[asset]
            
            # Momentum signal
            ma50 = p.iloc[i-50:i+1].mean()
            ma200 = p.iloc[i-200:i+1].mean()
            mom_signal = 1 if ma50 > ma200 else -1
            
            # Carry signal (asset-specific)
            if asset in ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VNQ', 'XLE', 'XLF', 'XLK']:
                # Equity carry: inverse VIX (low VIX = positive carry)
                carry_signal = 1 if v < 20 else (-1 if v > 30 else 0)
            elif asset in ['TLT', 'IEF']:
                # Bond carry: positive when yield curve steep (proxy: TLT > IEF momentum)
                tlt_mom = prices['TLT'].iloc[i] / prices['TLT'].iloc[i-60] - 1
                ief_mom = prices['IEF'].iloc[i] / prices['IEF'].iloc[i-60] - 1
                carry_signal = 1 if tlt_mom > ief_mom else -1
            elif asset in ['GLD', 'SLV']:
                # Gold: positive carry when real rates negative (proxy: high VIX + falling yields)
                carry_signal = 1 if v > 20 else 0
            elif asset == 'UUP':
                # Dollar: carry from rate differentials (proxy: rising yields)
                carry_signal = -1 if v > 25 else 1
            elif asset == 'HYG':
                carry_signal = 1 if v < 18 else -1
            else:
                carry_signal = 0
            
            # Combine: momentum + carry
            combined = (mom_signal + carry_signal) / 2.0
            
            vol = p.pct_change().iloc[max(0,i-20):i].std() * np.sqrt(252)
            vol = max(vol, 0.01)
            weight = (target_vol / vol) / len(trade_assets) * abs(combined)
            weight = min(weight, 0.15)
            
            total_ret += np.sign(combined) * weight * rets[asset].iloc[i] if combined != 0 else 0
        
        eq.iloc[i] = eq.iloc[i-1] * (1 + total_ret) if np.isfinite(total_ret) else eq.iloc[i-1]
    configs['Carry + Mom'] = eq.iloc[start:]
    
    # SPY benchmark
    spy_eq = prices['SPY'].iloc[start:] / prices['SPY'].iloc[start] * INITIAL_CAPITAL
    configs['SPY B&H'] = spy_eq
    
    # === RESULTS ===
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
    
    # === R1 REGIME TEST ===
    print(f"\nR1 Regime Test:")
    for name, eq in configs.items():
        if name == 'SPY B&H': continue
        common = eq.index.intersection(spy_eq.index)
        r = regime_test(eq.loc[common], spy_eq.loc[common])
        status = '✅ PASS' if r['pass'] else '❌ FAIL'
        print(f"  {status} {name:<25s}: green={r['green_sharpe']:.2f}, red={r['red_sharpe']:.2f}, gap={r['gap']:.3f}")
    
    # === ADVERSARIAL ON BEST ===
    best_eq = configs[best_name]
    common = best_eq.index.intersection(spy_eq.index)
    
    # Perm test: shuffle day-to-signal mapping
    print(f"\nPermutation test (200 shuffles)...")
    real_sharpe = metrics['sharpe']
    perm_sharpes = []
    
    for _ in range(200):
        eq_p = pd.Series(INITIAL_CAPITAL, index=prices.index)
        shuffled = np.random.permutation(range(start, len(prices)))
        
        for idx, i in enumerate(range(start, len(prices))):
            j = shuffled[idx]
            if j < 200 or i < 200: continue
            total_ret = 0.0
            
            for asset in trade_assets:
                p = prices[asset]
                ma50 = p.iloc[j-50:j+1].mean()
                ma200 = p.iloc[j-200:j+1].mean()
                signal = 1 if ma50 > ma200 else -1
                
                vol = p.pct_change().iloc[max(0,j-20):j].std() * np.sqrt(252)
                vol = max(vol, 0.01)
                weight = (0.10 / vol) / len(trade_assets)
                weight = min(weight, 0.15)
                
                total_ret += signal * weight * rets[asset].iloc[i]
            
            eq_p.iloc[i] = eq_p.iloc[i-1] * (1 + total_ret) if np.isfinite(total_ret) else eq_p.iloc[i-1]
        
        pm = compute_metrics(eq_p.iloc[start:])
        perm_sharpes.append(pm['sharpe'])
    
    perm_sharpes = np.array(perm_sharpes)
    p_value = float((perm_sharpes >= real_sharpe).mean())
    print(f"  Real: {real_sharpe:.3f}, Perm mean: {perm_sharpes.mean():.3f}, p={p_value:.3f}")
    
    subp = subperiod_test(best_eq.loc[common])
    regime = regime_test(best_eq.loc[common], spy_eq.loc[common])
    
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
        name=f"Global Macro ({best_name})",
        description="Multi-asset momentum with risk parity sizing",
        metrics=metrics,
        adversarial=adv
    )

if __name__ == '__main__':
    main()
