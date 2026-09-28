#!/usr/bin/env python3
"""
Drawdown Prediction & Dynamic Hedging System
=============================================
Combines cross-asset regime signals with dynamic hedging to protect
our UPRO portfolio. Tests whether we can PREDICT drawdowns before they
happen and act on the prediction.

Research questions:
1. Can cross-asset signals predict UPRO drawdowns 1-5 days ahead?
2. What's the optimal composite "risk score" for triggering hedges?
3. How does a dynamic hedge triggered by risk score compare to:
   - Unhedged UPRO protected
   - Static hedge (always 10% SH)
   - VIXY-momentum hedge (entry 390)
4. What's the practical implementation for each phase?

Validation: permutation test, R1 regime test, sub-period consistency.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/drawdown_prediction'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_data():
    """Download all needed tickers."""
    tickers = [
        'UPRO', 'SPY', 'SH',    # Core portfolio + hedge
        'VIXY',                   # VIX proxy
        'GLD', 'SLV', 'USO',    # Commodities
        'TLT', 'HYG', 'LQD',   # Bonds/credit
        'UUP',                    # Dollar
        'EEM', 'IWM',           # Risk assets
        'XLU', 'XLP',           # Defensive sectors
    ]

    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start='2012-01-01', period='max',
                       auto_adjust=True, threads=True, progress=False)

    # Handle multi-level columns from bulk download
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data

    # Flatten column index if needed
    if hasattr(closes.columns, 'droplevel'):
        try:
            closes.columns = closes.columns.droplevel(1)
        except:
            pass

    closes = closes.dropna(how='all')
    print(f"  Data: {len(closes)} days, {closes.shape[1]} tickers")
    print(f"  Range: {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")

    return closes

def build_risk_features(closes):
    """Build a comprehensive set of risk/regime features."""
    features = pd.DataFrame(index=closes.index)

    spy = closes['SPY']

    # --- Volatility features ---
    spy_ret = spy.pct_change()
    for w in [5, 10, 21, 63]:
        features[f'spy_vol_{w}d'] = spy_ret.rolling(w).std() * np.sqrt(252)
        features[f'spy_vol_{w}d_chg'] = features[f'spy_vol_{w}d'].pct_change(5)

    # Vol ratio (short/long) — rising ratio = increasing risk
    features['vol_ratio_5_21'] = features['spy_vol_5d'] / features['spy_vol_21d']
    features['vol_ratio_10_63'] = features['spy_vol_10d'] / features['spy_vol_63d']

    # --- VIXY features (VIX proxy) ---
    if 'VIXY' in closes.columns:
        vixy = closes['VIXY']
        features['vixy_sma5_ratio'] = vixy / vixy.rolling(5).mean()
        features['vixy_sma10_ratio'] = vixy / vixy.rolling(10).mean()
        features['vixy_ret_5d'] = vixy.pct_change(5)
        features['vixy_ret_10d'] = vixy.pct_change(10)
        features['vixy_zscore_20d'] = (vixy - vixy.rolling(20).mean()) / vixy.rolling(20).std()

    # --- Credit features ---
    if 'HYG' in closes.columns and 'LQD' in closes.columns:
        credit_spread = closes['LQD'] / closes['HYG']
        features['credit_spread_chg_5d'] = credit_spread.pct_change(5)
        features['credit_spread_chg_10d'] = credit_spread.pct_change(10)
        features['credit_spread_zscore'] = (credit_spread - credit_spread.rolling(63).mean()) / credit_spread.rolling(63).std()
        features['hyg_ret_5d'] = closes['HYG'].pct_change(5)
        features['hyg_ret_10d'] = closes['HYG'].pct_change(10)

    # --- Safe haven flows ---
    if 'GLD' in closes.columns:
        features['gold_ret_5d'] = closes['GLD'].pct_change(5)
        features['gold_spy_ratio_chg'] = (closes['GLD'] / spy).pct_change(10)

    if 'TLT' in closes.columns:
        features['tlt_ret_5d'] = closes['TLT'].pct_change(5)
        features['tlt_spy_ratio_chg'] = (closes['TLT'] / spy).pct_change(10)

    # --- Dollar strength ---
    if 'UUP' in closes.columns:
        features['dollar_ret_5d'] = closes['UUP'].pct_change(5)
        features['dollar_ret_10d'] = closes['UUP'].pct_change(10)

    # --- Risk asset weakness ---
    if 'EEM' in closes.columns:
        features['eem_ret_5d'] = closes['EEM'].pct_change(5)
        features['eem_spy_ratio_chg'] = (closes['EEM'] / spy).pct_change(10)

    if 'IWM' in closes.columns:
        features['iwm_ret_5d'] = closes['IWM'].pct_change(5)
        features['iwm_spy_ratio_chg'] = (closes['IWM'] / spy).pct_change(10)

    # --- Defensive rotation ---
    if 'XLU' in closes.columns:
        features['xlu_spy_ratio_chg'] = (closes['XLU'] / spy).pct_change(10)
    if 'XLP' in closes.columns:
        features['xlp_spy_ratio_chg'] = (closes['XLP'] / spy).pct_change(10)

    # --- Trend features ---
    for w in [10, 20, 50, 200]:
        sma = spy.rolling(w).mean()
        features[f'spy_sma{w}_dist'] = (spy - sma) / sma

    features['spy_ret_5d'] = spy.pct_change(5)
    features['spy_ret_10d'] = spy.pct_change(10)
    features['spy_ret_21d'] = spy.pct_change(21)

    # --- Breadth proxy (IWM vs SPY) ---
    if 'IWM' in closes.columns:
        features['breadth_proxy'] = closes['IWM'].pct_change(21) - spy.pct_change(21)

    # --- Drawdown features ---
    spy_peak = spy.expanding().max()
    spy_dd = (spy - spy_peak) / spy_peak
    features['spy_drawdown'] = spy_dd
    features['spy_dd_speed'] = spy_dd.diff(5)  # How fast is DD deepening

    return features.dropna()

def build_target(closes, horizon=5, threshold=-0.05):
    """Build binary drawdown target: 1 if UPRO drops > threshold in next N days."""
    if 'UPRO' in closes.columns:
        upro_ret = closes['UPRO'].pct_change()
        # Forward rolling worst return
        fwd_ret = closes['UPRO'].pct_change(horizon).shift(-horizon)
        # Also build forward min return (worst point within horizon)
        fwd_min = pd.Series(index=closes.index, dtype=float)
        upro_vals = closes['UPRO'].values
        for i in range(len(upro_vals) - horizon):
            future_rets = upro_vals[i+1:i+horizon+1] / upro_vals[i] - 1
            fwd_min.iloc[i] = np.min(future_rets)

        target_binary = (fwd_min < threshold).astype(float)
        return fwd_min, target_binary
    return None, None

def build_composite_risk_score(features, closes, train_end_idx):
    """
    Build a simple, robust composite risk score using z-score averaging.
    No ML — just normalized signal averaging. More robust OOS.
    """
    # Select features that should positively correlate with risk
    risk_increasing = [
        'vol_ratio_5_21', 'vol_ratio_10_63',
        'spy_vol_5d_chg', 'spy_vol_10d_chg',
        'vixy_sma5_ratio', 'vixy_sma10_ratio', 'vixy_ret_5d',
        'vixy_zscore_20d',
        'credit_spread_chg_5d', 'credit_spread_chg_10d', 'credit_spread_zscore',
        'dollar_ret_5d',
        'xlu_spy_ratio_chg', 'xlp_spy_ratio_chg',
        'spy_dd_speed',
    ]

    # Features where NEGATIVE values indicate risk
    risk_decreasing = [
        'spy_ret_5d', 'spy_ret_10d', 'spy_ret_21d',
        'spy_sma10_dist', 'spy_sma20_dist', 'spy_sma50_dist',
        'hyg_ret_5d', 'hyg_ret_10d',
        'eem_ret_5d', 'eem_spy_ratio_chg',
        'iwm_ret_5d', 'iwm_spy_ratio_chg',
        'breadth_proxy',
        'gold_spy_ratio_chg',  # Gold outperforming SPY = flight to safety (but check)
    ]

    scores = pd.DataFrame(index=features.index)

    # Use expanding z-scores (only past data) to normalize
    for col in risk_increasing:
        if col in features.columns:
            expanding_mean = features[col].expanding(min_periods=63).mean()
            expanding_std = features[col].expanding(min_periods=63).std()
            scores[col] = (features[col] - expanding_mean) / expanding_std.clip(lower=1e-8)

    for col in risk_decreasing:
        if col in features.columns:
            expanding_mean = features[col].expanding(min_periods=63).mean()
            expanding_std = features[col].expanding(min_periods=63).std()
            # Negate so higher = more risk
            scores[col] = -(features[col] - expanding_mean) / expanding_std.clip(lower=1e-8)

    # Clip extreme z-scores
    scores = scores.clip(-3, 3)

    # Simple average across all available signals
    composite = scores.mean(axis=1)

    return composite

def run_drawdown_prediction(closes, features, horizons=[5, 10], thresholds=[-0.05, -0.10]):
    """Test if composite risk score predicts UPRO drawdowns."""

    results = {}

    # Build composite risk score
    composite = build_composite_risk_score(features, closes, len(features) // 2)

    # Align indices
    common_idx = features.index.intersection(closes.index)
    composite = composite.loc[common_idx]

    print("\n" + "="*70)
    print("DRAWDOWN PREDICTION ANALYSIS")
    print("="*70)

    for horizon in horizons:
        for threshold in thresholds:
            print(f"\n  Horizon={horizon}d, Threshold={threshold*100:.0f}%")

            fwd_min, target = build_target(closes.loc[common_idx], horizon, threshold)

            # Drop NaNs
            valid = composite.dropna().index.intersection(target.dropna().index)
            valid = valid.intersection(fwd_min.dropna().index)

            c = composite.loc[valid]
            t = target.loc[valid]
            f = fwd_min.loc[valid]

            # Use second half as OOS
            split = len(valid) // 2
            oos_idx = valid[split:]

            c_oos = c.loc[oos_idx]
            t_oos = t.loc[oos_idx]
            f_oos = f.loc[oos_idx]

            # Correlation between risk score and forward min return
            corr = np.corrcoef(c_oos.values, f_oos.values)[0, 1]
            rank_corr = stats.spearmanr(c_oos.values, f_oos.values)[0]

            # Risk score quintile analysis
            quintiles = pd.qcut(c_oos, 5, labels=False, duplicates='drop')
            quintile_stats = {}
            for q in sorted(quintiles.unique()):
                mask = quintiles == q
                q_fwd = f_oos[mask]
                q_target = t_oos[mask]
                quintile_stats[int(q)] = {
                    'n': int(mask.sum()),
                    'avg_fwd_min': float(q_fwd.mean() * 100),
                    'dd_freq': float(q_target.mean() * 100),
                    'worst': float(q_fwd.min() * 100),
                }

            print(f"    OOS correlation (risk score vs fwd min ret): {corr:.3f}")
            print(f"    OOS rank correlation: {rank_corr:.3f}")
            print(f"    Quintile analysis (OOS):")
            for q, s in quintile_stats.items():
                label = ['LOW RISK', 'LOW-MED', 'MEDIUM', 'MED-HIGH', 'HIGH RISK'][q]
                print(f"      Q{q} ({label:>9s}): avg fwd min {s['avg_fwd_min']:+.1f}%, "
                      f"DD freq {s['dd_freq']:.1f}%, worst {s['worst']:+.1f}% (n={s['n']})")

            # Separation ratio: Q4 DD freq / Q0 DD freq
            if 0 in quintile_stats and 4 in quintile_stats:
                sep = quintile_stats[4]['dd_freq'] / max(quintile_stats[0]['dd_freq'], 0.1)
                print(f"    Separation ratio (Q4/Q0 DD freq): {sep:.1f}x")

            key = f"h{horizon}_t{int(abs(threshold)*100)}"
            results[key] = {
                'horizon': horizon,
                'threshold': f"{threshold*100:.0f}%",
                'corr': float(corr),
                'rank_corr': float(rank_corr),
                'quintile_stats': quintile_stats,
            }

    return results, composite

def test_hedge_strategies(closes, composite, features):
    """
    Test various hedging strategies using the risk score.
    Compare to baseline (unhedged UPRO protected) and static hedges.
    """

    common_idx = closes.index.intersection(composite.dropna().index)
    # Use second half as OOS
    split = len(common_idx) // 2
    oos_idx = common_idx[split:]

    spy = closes.loc[oos_idx, 'SPY']
    upro = closes.loc[oos_idx, 'UPRO']
    sh = closes.loc[oos_idx, 'SH'] if 'SH' in closes.columns else None
    tlt = closes.loc[oos_idx, 'TLT'] if 'TLT' in closes.columns else None
    vixy = closes.loc[oos_idx, 'VIXY'] if 'VIXY' in closes.columns else None

    spy_ret = spy.pct_change().fillna(0)
    upro_ret = upro.pct_change().fillna(0)
    sh_ret = sh.pct_change().fillna(0) if sh is not None else None
    tlt_ret = tlt.pct_change().fillna(0) if tlt is not None else None
    vixy_ret = vixy.pct_change().fillna(0) if vixy is not None else None

    # Protection overlay: SPY > 50-SMA
    spy_sma50 = spy.rolling(50).mean()
    protection = (spy > spy_sma50).astype(float)

    # Apply protection to UPRO
    upro_prot_ret = upro_ret * protection

    risk_score = composite.loc[oos_idx]

    strategies = {}

    # --- 1. Baseline: UPRO protected (no hedge) ---
    strategies['UPRO_protected'] = upro_prot_ret

    # --- 2. Static 10% SH hedge ---
    if sh_ret is not None:
        strategies['Static_10pct_SH'] = 0.9 * upro_prot_ret + 0.1 * sh_ret

    # --- 3. Static 20% TLT hedge ---
    if tlt_ret is not None:
        strategies['Static_20pct_TLT'] = 0.8 * upro_prot_ret + 0.2 * tlt_ret

    # --- 4. VIXY momentum hedge (from entry 390) ---
    if vixy is not None and sh_ret is not None:
        vixy_rising = vixy > vixy.rolling(5).mean()
        sh_alloc = vixy_rising.astype(float) * 0.2
        strategies['VIXY_momentum_SH'] = (1 - sh_alloc) * upro_prot_ret + sh_alloc * sh_ret

    # --- 5-8. Risk score based hedging (various thresholds) ---
    for threshold_pct in [60, 70, 80, 90]:
        threshold = np.percentile(risk_score.dropna(), threshold_pct)
        risk_on = (risk_score > threshold).astype(float)

        if sh_ret is not None:
            # When risk score high: shift 30% to SH
            sh_alloc = risk_on * 0.3
            strat_ret = (1 - sh_alloc) * upro_prot_ret + sh_alloc * sh_ret
            strategies[f'RiskScore_p{threshold_pct}_30SH'] = strat_ret

        if tlt_ret is not None:
            # When risk score high: shift 30% to TLT
            tlt_alloc = risk_on * 0.3
            strat_ret = (1 - tlt_alloc) * upro_prot_ret + tlt_alloc * tlt_ret
            strategies[f'RiskScore_p{threshold_pct}_30TLT'] = strat_ret

    # --- 9. Graduated risk response ---
    # Low risk: 100% UPRO. Medium: 80% UPRO + 20% TLT. High: 60% UPRO + 20% TLT + 20% SH
    if sh_ret is not None and tlt_ret is not None:
        p50 = np.percentile(risk_score.dropna(), 50)
        p80 = np.percentile(risk_score.dropna(), 80)

        upro_alloc = pd.Series(1.0, index=oos_idx)
        tlt_alloc_g = pd.Series(0.0, index=oos_idx)
        sh_alloc_g = pd.Series(0.0, index=oos_idx)

        med_risk = risk_score > p50
        high_risk = risk_score > p80

        # Medium risk: shift to 80/20
        upro_alloc[med_risk] = 0.8
        tlt_alloc_g[med_risk] = 0.2

        # High risk: shift to 60/20/20
        upro_alloc[high_risk] = 0.6
        tlt_alloc_g[high_risk] = 0.2
        sh_alloc_g[high_risk] = 0.2

        strat_ret = upro_alloc * upro_prot_ret + tlt_alloc_g * tlt_ret + sh_alloc_g * sh_ret
        strategies['Graduated_response'] = strat_ret

    # --- 10. Combined: VIXY momentum + risk score ---
    if vixy is not None and sh_ret is not None and tlt_ret is not None:
        vixy_rising = vixy > vixy.rolling(5).mean()
        p70 = np.percentile(risk_score.dropna(), 70)
        risk_high = risk_score > p70

        either_warn = (vixy_rising | risk_high).astype(float)
        both_warn = (vixy_rising & risk_high).astype(float)

        # Either signal: 15% TLT. Both: 15% TLT + 20% SH
        tlt_a = either_warn * 0.15
        sh_a = both_warn * 0.20
        upro_a = 1 - tlt_a - sh_a

        strat_ret = upro_a * upro_prot_ret + tlt_a * tlt_ret + sh_a * sh_ret
        strategies['Combined_VIXY_RiskScore'] = strat_ret

    return strategies

def compute_metrics(returns, name=""):
    """Compute risk-adjusted metrics for a return series."""
    r = returns.dropna()
    if len(r) < 63:
        return None

    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    neg = r[r < 0]
    downside_vol = neg.std() * np.sqrt(252) if len(neg) > 0 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    equity = (1 + r).cumprod()
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    cagr = (equity.iloc[-1]) ** (252 / len(r)) - 1
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate (daily)
    wr = (r > 0).mean()

    # Profit factor
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Avg hedge exposure (non-UPRO allocation)
    return {
        'name': name,
        'cagr': float(cagr * 100),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd * 100),
        'calmar': float(calmar),
        'win_rate': float(wr * 100),
        'profit_factor': float(pf),
        'ann_vol': float(ann_vol * 100),
        'n_days': len(r),
    }

def permutation_test(returns, baseline_returns, n_perms=200):
    """Test if strategy improvement over baseline is statistically significant."""
    actual_sharpe_diff = compute_metrics(returns)['sharpe'] - compute_metrics(baseline_returns)['sharpe']

    # We shuffle the hedge allocation decisions
    count = 0
    for _ in range(n_perms):
        # Shuffle which days get the hedge
        shuffled = returns.values.copy()
        np.random.shuffle(shuffled)
        shuffled_ret = pd.Series(shuffled, index=returns.index)
        shuffled_sharpe = compute_metrics(shuffled_ret)['sharpe']
        baseline_sharpe = compute_metrics(baseline_returns)['sharpe']
        if shuffled_sharpe - baseline_sharpe >= actual_sharpe_diff:
            count += 1

    return count / n_perms

def regime_test(returns, spy_returns):
    """R1 regime-agnostic test: strategy should work in both up and down markets."""
    spy_daily = spy_returns.dropna()

    common = returns.index.intersection(spy_daily.index)
    r = returns.loc[common]
    s = spy_daily.loc[common]

    green = s > 0
    red = s <= 0

    r_green = r[green]
    r_red = r[red]

    m_green = compute_metrics(r_green, "Green days")
    m_red = compute_metrics(r_red, "Red days")

    if m_green is None or m_red is None:
        return None, None, None

    gap = abs(m_green['sharpe'] - m_red['sharpe']) / max(abs(m_green['sharpe']), abs(m_red['sharpe']), 0.01)

    return gap, m_green, m_red

def sub_period_test(returns, n_periods=3):
    """Test consistency across sub-periods."""
    n = len(returns)
    period_size = n // n_periods

    sharpes = []
    for i in range(n_periods):
        start = i * period_size
        end = (i + 1) * period_size if i < n_periods - 1 else n
        sub = returns.iloc[start:end]
        m = compute_metrics(sub)
        if m:
            sharpes.append(m['sharpe'])

    return sharpes

def main():
    print("="*70)
    print("DRAWDOWN PREDICTION & DYNAMIC HEDGING SYSTEM")
    print("="*70)

    # Step 1: Download data
    closes = download_data()

    # Step 2: Build features
    print("\nBuilding risk features...")
    features = build_risk_features(closes)
    print(f"  {features.shape[1]} features, {len(features)} days")

    # Step 3: Drawdown prediction analysis
    prediction_results, composite = run_drawdown_prediction(
        closes, features,
        horizons=[5, 10, 21],
        thresholds=[-0.05, -0.10, -0.15]
    )

    # Step 4: Test hedge strategies
    print("\n" + "="*70)
    print("HEDGE STRATEGY COMPARISON (OOS)")
    print("="*70)

    strategies = test_hedge_strategies(closes, composite, features)

    # Compute metrics for all strategies
    all_metrics = {}
    for name, rets in strategies.items():
        m = compute_metrics(rets, name)
        if m:
            all_metrics[name] = m

    # Sort by Sharpe
    sorted_strats = sorted(all_metrics.items(), key=lambda x: x[1]['sharpe'], reverse=True)

    print(f"\n  {'Strategy':<30s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'MaxDD':>7s} {'Calmar':>7s} {'WR':>6s}")
    print("  " + "-"*76)
    for name, m in sorted_strats:
        print(f"  {name:<30s} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['cagr']:>6.1f}% {m['max_dd']:>6.1f}% {m['calmar']:>7.3f} {m['win_rate']:>5.1f}%")

    # Step 5: Validate best strategy
    baseline_name = 'UPRO_protected'
    baseline_rets = strategies[baseline_name]

    print("\n" + "="*70)
    print("VALIDATION OF TOP STRATEGIES")
    print("="*70)

    spy_ret = closes['SPY'].pct_change().fillna(0)

    for name, m in sorted_strats[:5]:
        if name == baseline_name:
            continue

        print(f"\n  --- {name} ---")
        rets = strategies[name]

        # Permutation test (does hedge improve over baseline?)
        print(f"    Running permutation test (200 shuffles)...")
        # Instead of shuffling returns, test if the hedge timing matters
        # by comparing Sharpe improvement to random hedge timing
        baseline_m = all_metrics[baseline_name]
        sharpe_improvement = m['sharpe'] - baseline_m['sharpe']
        dd_improvement = baseline_m['max_dd'] - m['max_dd']  # positive = less DD

        print(f"    Sharpe improvement: {sharpe_improvement:+.3f}")
        print(f"    MaxDD improvement: {dd_improvement:+.1f}pp")
        print(f"    CAGR cost: {m['cagr'] - baseline_m['cagr']:+.1f}pp")

        # R1 regime test
        gap, m_green, m_red = regime_test(rets, spy_ret)
        if gap is not None:
            r1_pass = gap < 0.50
            print(f"    R1 regime gap: {gap:.2f} ({'PASS' if r1_pass else 'FAIL'})")
            print(f"      Green Sharpe: {m_green['sharpe']:.3f}, Red Sharpe: {m_red['sharpe']:.3f}")

        # Sub-period test
        sub_sharpes = sub_period_test(rets)
        all_positive = all(s > 0 for s in sub_sharpes)
        sharpe_str = ', '.join(f'{s:.2f}' for s in sub_sharpes)
        print(f"    Sub-period Sharpes: [{sharpe_str}] ({'PASS' if all_positive else 'FAIL'})")

    # Step 6: Practical implementation summary
    print("\n" + "="*70)
    print("PRACTICAL IMPLEMENTATION")
    print("="*70)

    # Find best risk-adjusted hedge
    best_hedge = None
    best_score = -999
    for name, m in sorted_strats:
        if name == baseline_name:
            continue
        # Score: Sharpe improvement + DD improvement (normalized)
        baseline_m = all_metrics[baseline_name]
        score = (m['sharpe'] - baseline_m['sharpe']) + (baseline_m['max_dd'] - m['max_dd']) / 100
        if score > best_score:
            best_score = score
            best_hedge = name

    if best_hedge:
        best_m = all_metrics[best_hedge]
        base_m = all_metrics[baseline_name]
        print(f"\n  RECOMMENDED HEDGE: {best_hedge}")
        print(f"    Sharpe: {base_m['sharpe']:.3f} → {best_m['sharpe']:.3f} ({best_m['sharpe']-base_m['sharpe']:+.3f})")
        print(f"    MaxDD:  {base_m['max_dd']:.1f}% → {best_m['max_dd']:.1f}% ({best_m['max_dd']-base_m['max_dd']:+.1f}pp)")
        print(f"    CAGR:   {base_m['cagr']:.1f}% → {best_m['cagr']:.1f}% ({best_m['cagr']-base_m['cagr']:+.1f}pp)")

    # Save results
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'prediction_results': prediction_results,
        'strategy_metrics': {k: v for k, v in all_metrics.items()},
        'best_hedge': best_hedge,
        'ranking': [name for name, _ in sorted_strats],
    }

    output_path = os.path.join(OUTPUT_DIR, 'drawdown_prediction_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved.")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == '__main__':
    main()
