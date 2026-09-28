#!/usr/bin/env python3
"""
Regime-Agnostic Portfolio Blend
================================
Key insight from research so far:
- VIX Leverage strategies have highest Sharpe (3.2-3.9) but ALL fail R1 (bull-biased)
- ML Trend Following is the ONLY R1 PASS with strong Sharpe (2.90)
- Dynamic VIX/Trend blend: Sharpe 3.06 but VIX component still dominates

This script explores regime-agnostic strategies that maintain edge in BOTH
bull and bear markets:
1. Long/short trend following with risk parity
2. Cross-asset carry + trend combo
3. Adaptive volatility harvesting
4. Regime-switching ensemble (but with balance requirement)
"""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research/autonomous')
from research_template import *
from sklearn.ensemble import GradientBoostingClassifier

def main():
    print("=" * 70)
    print("REGIME-AGNOSTIC PORTFOLIO STRATEGIES")
    print("=" * 70)
    
    tickers = ['SPY', 'UPRO', 'SHY', 'TLT', 'GLD', 'UUP', 'EEM', 'VNQ', 'HYG', 'XLE', 'QQQ', 'IWM']
    prices = download_etfs(tickers, start='2008-01-01')
    vix = download_vix(start='2008-01-01')
    prices['VIX'] = vix
    prices = prices.dropna()
    print(f"Data: {len(prices)} rows, {prices.index[0].date()} → {prices.index[-1].date()}")
    
    rets = {t: prices[t].pct_change() for t in tickers}
    start = 252
    
    # Trend assets for long/short strategies
    trend_assets = ['SPY', 'TLT', 'GLD', 'UUP', 'EEM', 'VNQ', 'HYG', 'XLE']
    
    configs = {}
    
    # === 1. ENHANCED MULTI-ASSET TREND FOLLOWING ===
    # Long/short with vol targeting, multiple MA speeds
    target_vol = 0.10
    
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        total_ret = 0.0
        for asset in trend_assets:
            price = prices[asset]
            
            # Multi-speed trend signal (3 MAs)
            if i < 200: continue
            ma20 = price.iloc[i-20:i+1].mean()
            ma50 = price.iloc[i-50:i+1].mean()
            ma100 = price.iloc[i-100:i+1].mean()
            ma200 = price.iloc[i-200:i+1].mean()
            
            # Composite signal: average of 3 lookbacks
            signal = 0
            if ma20 > ma50: signal += 1
            else: signal -= 1
            if ma50 > ma100: signal += 1
            else: signal -= 1
            if ma100 > ma200: signal += 1
            else: signal -= 1
            signal /= 3.0  # -1 to +1
            
            # Vol scaling
            vol = prices[asset].pct_change().iloc[max(0,i-20):i].std() * np.sqrt(252)
            vol = max(vol, 0.01)
            weight = (target_vol / vol) / len(trend_assets) * abs(signal)
            weight = min(weight, 0.20)
            
            total_ret += np.sign(signal) * weight * rets[asset].iloc[i]
        
        eq.iloc[i] = eq.iloc[i-1] * (1 + total_ret) if np.isfinite(total_ret) else eq.iloc[i-1]
    configs['Multi-Speed Trend'] = eq.iloc[start:]
    
    # === 2. TREND + VOL SCALING COMBO ===
    # Trend following with dynamic vol targeting based on market regime
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        # Adaptive vol target: low vol → higher target, high vol → lower target
        if v < 15: tv = 0.15
        elif v < 20: tv = 0.10
        elif v < 30: tv = 0.06
        else: tv = 0.03
        
        total_ret = 0.0
        for asset in trend_assets:
            price = prices[asset]
            if i < 100: continue
            ma20 = price.iloc[i-20:i+1].mean()
            ma100 = price.iloc[i-100:i+1].mean()
            signal = 1 if ma20 > ma100 else -1
            
            vol = prices[asset].pct_change().iloc[max(0,i-20):i].std() * np.sqrt(252)
            vol = max(vol, 0.01)
            weight = (tv / vol) / len(trend_assets)
            weight = min(weight, 0.25)
            
            total_ret += signal * weight * rets[asset].iloc[i]
        
        eq.iloc[i] = eq.iloc[i-1] * (1 + total_ret) if np.isfinite(total_ret) else eq.iloc[i-1]
    configs['Trend + Adaptive Vol'] = eq.iloc[start:]
    
    # === 3. CROSS-ASSET MOMENTUM + MEAN REVERSION BLEND ===
    # 12-month momentum for allocation, 1-month mean reversion for timing
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    alloc_assets = ['SPY', 'TLT', 'GLD', 'EEM', 'VNQ', 'XLE']
    for i in range(start, len(prices)):
        if i < 252: continue
        
        # 12-month momentum scores
        mom_scores = {}
        for asset in alloc_assets:
            ret_12m = prices[asset].iloc[i] / prices[asset].iloc[i-252] - 1
            ret_1m = prices[asset].iloc[i] / prices[asset].iloc[i-21] - 1
            # Momentum: 12m return minus 1m (skip recent month for mean reversion)
            mom_scores[asset] = ret_12m - ret_1m
        
        # Rank and go long top 3, short bottom 3
        ranked = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)
        total_ret = 0.0
        n_long = len(ranked) // 2
        
        for j, (asset, score) in enumerate(ranked):
            vol = prices[asset].pct_change().iloc[max(0,i-60):i].std() * np.sqrt(252)
            vol = max(vol, 0.05)
            base_weight = 0.10 / vol  # Risk parity sizing
            base_weight = min(base_weight, 0.20)
            
            if j < n_long:
                total_ret += base_weight * rets[asset].iloc[i]
            else:
                total_ret -= base_weight * rets[asset].iloc[i]
        
        eq.iloc[i] = eq.iloc[i-1] * (1 + total_ret) if np.isfinite(total_ret) else eq.iloc[i-1]
    configs['Cross-Asset L/S Mom'] = eq.iloc[start:]
    
    # === 4. VOLATILITY CARRY ===
    # Short realized vol relative to implied (VIX) via UPRO/SHY position sizing
    eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    for i in range(start, len(prices)):
        v = prices['VIX'].iloc[i]
        spy_rvol = prices['SPY'].pct_change().iloc[max(0,i-20):i].std() * np.sqrt(252) * 100
        
        # Vol carry = IV - RV (positive = sell vol = be long equity)
        vol_carry = v - spy_rvol
        
        # Scale equity exposure by vol carry signal
        if vol_carry > 5:  # Significant overpricing of vol → be long
            spy_w = min(0.90, 0.50 + vol_carry * 0.05)
        elif vol_carry > 0:  # Mild overpricing
            spy_w = 0.50 + vol_carry * 0.03
        elif vol_carry > -5:  # Vol underpriced → reduce
            spy_w = max(0.10, 0.50 + vol_carry * 0.05)
        else:  # Severe underpricing → minimal
            spy_w = 0.10
        
        ret = spy_w * rets['SPY'].iloc[i] + (1 - spy_w) * rets['SHY'].iloc[i]
        eq.iloc[i] = eq.iloc[i-1] * (1 + ret) if np.isfinite(ret) else eq.iloc[i-1]
    configs['Vol Carry'] = eq.iloc[start:]
    
    # === 5. DIVERSIFIED ALPHA ENSEMBLE ===
    # Equal-risk blend of: trend following + cross-asset mom + vol carry
    # Each independently managed, combined at portfolio level
    eq_trend_component = configs.get('Multi-Speed Trend', pd.Series(INITIAL_CAPITAL, index=prices.index[start:]))
    eq_mom_component = configs.get('Cross-Asset L/S Mom', pd.Series(INITIAL_CAPITAL, index=prices.index[start:]))
    eq_vol_component = configs.get('Vol Carry', pd.Series(INITIAL_CAPITAL, index=prices.index[start:]))
    
    # Equal-weight daily returns
    common_idx = eq_trend_component.index.intersection(eq_mom_component.index).intersection(eq_vol_component.index)
    r_trend = eq_trend_component.loc[common_idx].pct_change()
    r_mom = eq_mom_component.loc[common_idx].pct_change()
    r_vol = eq_vol_component.loc[common_idx].pct_change()
    
    eq_ensemble = pd.Series(INITIAL_CAPITAL, index=common_idx)
    for i in range(1, len(common_idx)):
        combo_ret = (r_trend.iloc[i] + r_mom.iloc[i] + r_vol.iloc[i]) / 3.0
        if np.isfinite(combo_ret):
            eq_ensemble.iloc[i] = eq_ensemble.iloc[i-1] * (1 + combo_ret)
        else:
            eq_ensemble.iloc[i] = eq_ensemble.iloc[i-1]
    configs['Diversified Ensemble'] = eq_ensemble
    
    # === 6. ML-FILTERED TREND (GBM) ===
    # Train GBM to predict which trend signals are "real" (not noise)
    print("\nTraining ML trend filter...")
    eq_ml = pd.Series(INITIAL_CAPITAL, index=prices.index)
    train_window = 252
    
    for i in range(start + train_window, len(prices)):
        if i % 21 != 0 and i != start + train_window:
            # Re-use last model within month
            pass
        else:
            # Build features and labels for training
            features = []
            labels = []
            
            for j in range(i - train_window, i - 21):
                if j < 200: continue
                feats = []
                for asset in ['SPY', 'TLT', 'GLD', 'EEM']:
                    p = prices[asset]
                    ma20 = p.iloc[j-20:j+1].mean()
                    ma50 = p.iloc[j-50:j+1].mean()
                    ma100 = p.iloc[j-100:j+1].mean()
                    ret_20 = p.iloc[j] / p.iloc[j-20] - 1
                    vol_20 = p.pct_change().iloc[j-20:j].std()
                    feats.extend([ma20/p.iloc[j]-1, ma50/p.iloc[j]-1, 
                                 ma100/p.iloc[j]-1, ret_20, vol_20])
                
                feats.append(prices['VIX'].iloc[j])
                feats.append(prices['VIX'].iloc[j] / prices['VIX'].iloc[max(0,j-20):j+1].mean() - 1)
                
                # Label: was trend-following profitable next 21 days?
                future_ret = 0
                for k in range(j+1, min(j+22, len(prices))):
                    for asset in trend_assets[:4]:
                        p = prices[asset]
                        ma20 = p.iloc[k-20:k+1].mean()
                        ma100 = p.iloc[k-100:k+1].mean()
                        sig = 1 if ma20 > ma100 else -1
                        future_ret += sig * rets[asset].iloc[k] / 4
                
                features.append(feats)
                labels.append(1 if future_ret > 0 else 0)
            
            if len(features) < 50: continue
            X_train = np.array(features)
            y_train = np.array(labels)
            
            model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            model.fit(X_train, y_train)
        
        # Current features
        feats_now = []
        for asset in ['SPY', 'TLT', 'GLD', 'EEM']:
            p = prices[asset]
            ma20 = p.iloc[i-20:i+1].mean()
            ma50 = p.iloc[i-50:i+1].mean()
            ma100 = p.iloc[i-100:i+1].mean()
            ret_20 = p.iloc[i] / p.iloc[i-20] - 1
            vol_20 = p.pct_change().iloc[i-20:i].std()
            feats_now.extend([ma20/p.iloc[i]-1, ma50/p.iloc[i]-1,
                             ma100/p.iloc[i]-1, ret_20, vol_20])
        feats_now.append(prices['VIX'].iloc[i])
        feats_now.append(prices['VIX'].iloc[i] / prices['VIX'].iloc[max(0,i-20):i+1].mean() - 1)
        
        prob = model.predict_proba(np.array([feats_now]))[0]
        confidence = prob[1]  # P(trend profitable)
        
        # Use confidence as scaling factor
        if confidence > 0.55:
            scale = min(confidence * 1.5, 1.0)
        else:
            scale = 0.3  # Reduce exposure when ML says trend is weak
        
        total_ret = 0.0
        for asset in trend_assets[:4]:
            p = prices[asset]
            ma20 = p.iloc[i-20:i+1].mean()
            ma100 = p.iloc[i-100:i+1].mean()
            signal = 1 if ma20 > ma100 else -1
            
            vol = p.pct_change().iloc[max(0,i-20):i].std() * np.sqrt(252)
            vol = max(vol, 0.01)
            weight = (target_vol / vol) / 4 * scale
            weight = min(weight, 0.25)
            
            total_ret += signal * weight * rets[asset].iloc[i]
        
        eq_ml.iloc[i] = eq_ml.iloc[i-1] * (1 + total_ret) if np.isfinite(total_ret) else eq_ml.iloc[i-1]
    configs['ML-Filtered Trend'] = eq_ml.iloc[start + train_window:]
    
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
    
    # === R1 REGIME TEST (KEY GATE) ===
    print(f"\nR1 Regime Test (the critical gate):")
    for name, eq in configs.items():
        if name == 'SPY B&H': continue
        # Align with SPY for regime classification
        common = eq.index.intersection(spy_eq.index)
        eq_aligned = eq.loc[common]
        spy_aligned = spy_eq.loc[common]
        r = regime_test(eq_aligned, spy_aligned)
        status = '✅ PASS' if r['pass'] else '❌ FAIL'
        print(f"  {name:<25s}: green={r['green_sharpe']:.2f}, red={r['red_sharpe']:.2f}, gap={r['gap']:.3f} {status}")
    
    # === FULL ADVERSARIAL ON BEST ===
    best_eq = configs[best_name]
    common = best_eq.index.intersection(spy_eq.index)
    best_eq_aligned = best_eq.loc[common]
    spy_aligned = spy_eq.loc[common]
    
    # Permutation test (signal shuffle)
    print(f"\nPermutation test (200 shuffles)...")
    real_sharpe = metrics['sharpe']
    perm_sharpes = []
    
    for _ in range(200):
        eq_p = pd.Series(INITIAL_CAPITAL, index=prices.index)
        # Shuffle VIX + asset returns mapping to break signal-return relationship
        shuffled_days = np.random.permutation(range(start, len(prices)))
        
        for idx, i in enumerate(range(start, len(prices))):
            j = shuffled_days[idx]  # Use shuffled day's signals but real returns
            total_ret = 0.0
            
            for asset in trend_assets[:4]:
                p = prices[asset]
                if j < 100 or i < 100: continue
                # Signal from shuffled day j
                ma20 = p.iloc[j-20:j+1].mean()
                ma100 = p.iloc[j-100:j+1].mean()
                signal = 1 if ma20 > ma100 else -1
                
                vol = p.pct_change().iloc[max(0,j-20):j].std() * np.sqrt(252)
                vol = max(vol, 0.01)
                weight = (target_vol / vol) / 4
                weight = min(weight, 0.25)
                
                # Returns from real day i
                total_ret += signal * weight * rets[asset].iloc[i]
            
            eq_p.iloc[i] = eq_p.iloc[i-1] * (1 + total_ret) if np.isfinite(total_ret) else eq_p.iloc[i-1]
        
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
    
    # Component correlations
    print(f"\nComponent correlations:")
    comp_names = ['Multi-Speed Trend', 'Cross-Asset L/S Mom', 'Vol Carry']
    for i_c in range(len(comp_names)):
        for j_c in range(i_c+1, len(comp_names)):
            n1, n2 = comp_names[i_c], comp_names[j_c]
            if n1 in configs and n2 in configs:
                c1 = configs[n1]
                c2 = configs[n2]
                common_c = c1.index.intersection(c2.index)
                corr = c1.loc[common_c].pct_change().corr(c2.loc[common_c].pct_change())
                print(f"  {n1} vs {n2}: {corr:.3f}")
    
    emit_result(
        name=f"Regime-Agnostic ({best_name})",
        description="Strategies targeting R1 regime-agnostic performance",
        metrics=metrics,
        adversarial=adv
    )

if __name__ == '__main__':
    main()
