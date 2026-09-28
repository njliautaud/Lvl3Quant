#!/usr/bin/env python3
"""
Task 1: Validate 3-Day Flow Strategy — autocorrelation, Newey-West Sharpe, split-half, bootstrap CI.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')

FEATURE_FILE = Path("/home/nick/Lvl3Quant/output/long_horizon_flow_v2/enhanced_daily_features.parquet")
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/long_horizon_flow_v2/validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_SIZE = 0.25
TICK_VALUE = 12.50
COST_RT_TICKS = 2.376
TRAIN_DAYS = 40
OOT_DAYS = 5
SLIDE_DAYS = 5
TARGET = 'fwd_direction_3d'
THRESHOLD = 0.55


def reconstruct_daily_pnl():
    df = pd.read_parquet(FEATURE_FILE)
    print(f"Loaded features: {df.shape[0]} days, {df.shape[1]} cols")
    
    exclude_prefixes = ['fwd_', 'open', 'close', 'high', 'low', 'session_vwap', 'trade_count', 'date']
    feature_cols = [c for c in df.columns if not any(c.startswith(p) for p in exclude_prefixes)]
    print(f"Features: {len(feature_cols)}")
    
    n = len(df)
    daily_results = []
    
    start = TRAIN_DAYS
    while start + OOT_DAYS <= n:
        train_slice = slice(start - TRAIN_DAYS, start)
        test_end = min(start + OOT_DAYS, n)
        test_slice = slice(start, test_end)
        
        X_train = df.iloc[train_slice][feature_cols].values
        y_train = df.iloc[train_slice][TARGET].values
        X_test = df.iloc[test_slice][feature_cols].values
        y_test = df.iloc[test_slice][TARGET].values
        fwd_rets = df.iloc[test_slice]['fwd_return_3d_ticks'].values
        dates = df.iloc[test_slice]['date'].values if 'date' in df.columns else [f'day_{i}' for i in range(start, test_end)]
        
        valid_train = ~np.isnan(y_train)
        valid_test = ~np.isnan(y_test) & ~np.isnan(fwd_rets)
        
        if valid_train.sum() < 20:
            start += SLIDE_DAYS
            continue
        
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
            verbose=-1, random_state=42
        )
        model.fit(X_train[valid_train], y_train[valid_train])
        
        probs = model.predict_proba(X_test)[:, 1]
        
        for j in range(len(probs)):
            if not valid_test[j]:
                continue
            
            prob = probs[j]
            fwd_ret = fwd_rets[j]
            
            if prob >= THRESHOLD:
                position = 1
            elif prob <= (1 - THRESHOLD):
                position = -1
            else:
                position = 0
            
            pnl_ticks = position * fwd_ret - abs(position) * COST_RT_TICKS
            
            daily_results.append({
                'date': str(dates[j]),
                'prob': float(prob),
                'position': int(position),
                'fwd_return_ticks': float(fwd_ret),
                'pnl_ticks': float(pnl_ticks),
            })
        
        start += SLIDE_DAYS
    
    return pd.DataFrame(daily_results)


def compute_acf(series, nlags=5):
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    if var == 0:
        return np.zeros(nlags + 1)
    acf = np.zeros(nlags + 1)
    acf[0] = 1.0
    for lag in range(1, nlags + 1):
        cov = np.sum((series[lag:] - mean) * (series[:-lag] - mean)) / n
        acf[lag] = cov / var
    return acf


def ljung_box_test(series, nlags=5):
    n = len(series)
    acf = compute_acf(series, nlags)
    Q = n * (n + 2) * np.sum(acf[1:nlags+1]**2 / (n - np.arange(1, nlags+1)))
    p_value = 1 - stats.chi2.cdf(Q, df=nlags)
    return Q, p_value


def newey_west_sharpe(returns, max_lag=None):
    n = len(returns)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n/100)**(2/9)))
    mean_ret = np.mean(returns)
    gamma_0 = np.var(returns, ddof=1)
    nw_var = gamma_0
    for j in range(1, max_lag + 1):
        weight = 1 - j / (max_lag + 1)
        gamma_j = np.sum((returns[j:] - mean_ret) * (returns[:-j] - mean_ret)) / (n - 1)
        nw_var += 2 * weight * gamma_j
    if nw_var <= 0:
        return mean_ret / np.sqrt(gamma_0) * np.sqrt(252) if gamma_0 > 0 else 0
    return mean_ret / np.sqrt(nw_var) * np.sqrt(252)


def non_overlapping_block_sharpe(pnl_series, block_size=3):
    n_blocks = len(pnl_series) // block_size
    block_returns = [np.sum(pnl_series[i*block_size:(i+1)*block_size]) for i in range(n_blocks)]
    block_returns = np.array(block_returns)
    if len(block_returns) < 2 or np.std(block_returns) == 0:
        return 0.0
    blocks_per_year = 252 / block_size
    return np.mean(block_returns) / np.std(block_returns, ddof=1) * np.sqrt(blocks_per_year)


def bootstrap_sharpe_ci(returns, n_bootstrap=10000, ci=0.95):
    n = len(returns)
    rng = np.random.RandomState(42)
    sharpes = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(returns, size=n, replace=True)
        if np.std(sample) > 0:
            sharpes[i] = np.mean(sample) / np.std(sample, ddof=1) * np.sqrt(252)
    alpha = (1 - ci) / 2
    return sharpes, np.percentile(sharpes, alpha * 100), np.percentile(sharpes, (1 - alpha) * 100)


def main():
    print("=" * 70)
    print("TASK 1: 3-Day Flow Strategy Validation")
    print("=" * 70)
    
    print("\n[1] Reconstructing walk-forward daily PnL...")
    df_pnl = reconstruct_daily_pnl()
    
    if df_pnl is None or len(df_pnl) == 0:
        print("ERROR: No PnL data reconstructed")
        return
    
    trades = df_pnl[df_pnl['position'] != 0].copy()
    all_obs = df_pnl.copy()  # Include no-trade days as 0 PnL for proper Sharpe
    print(f"Total OOT observations: {len(df_pnl)}, Trades taken: {len(trades)}")
    
    pnl = trades['pnl_ticks'].values
    
    # Basic metrics
    print(f"\n--- Basic Metrics ---")
    n_trades = len(trades)
    total_pnl = np.sum(pnl)
    avg_pnl = np.mean(pnl)
    std_pnl = np.std(pnl, ddof=1)
    raw_sharpe = avg_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0
    wr = np.mean(pnl > 0)
    gross_win = np.sum(pnl[pnl > 0])
    gross_loss = abs(np.sum(pnl[pnl < 0]))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    
    print(f"N trades: {n_trades}")
    print(f"Total PnL: {total_pnl:.1f} ticks (${total_pnl * TICK_VALUE:.0f})")
    print(f"Avg PnL/trade: {avg_pnl:.2f} ticks")
    print(f"Raw Sharpe (annualized): {raw_sharpe:.2f}")
    print(f"Win rate: {wr:.1%}")
    print(f"Profit factor: {pf:.2f}")
    
    # ACF
    print(f"\n--- Autocorrelation Analysis ---")
    acf_vals = compute_acf(pnl, nlags=5)
    ci_bound = 1.96 / np.sqrt(n_trades)
    for lag in range(1, 6):
        sig = "***" if abs(acf_vals[lag]) > ci_bound else ""
        print(f"  Lag {lag}: {acf_vals[lag]:+.4f}  (95% CI: +/-{ci_bound:.4f}) {sig}")
    
    Q, lb_pval = ljung_box_test(pnl, nlags=5)
    print(f"\nLjung-Box Q(5): {Q:.2f}, p-value: {lb_pval:.4f}")
    print(f"  -> {'SIGNIFICANT serial correlation' if lb_pval < 0.05 else 'No significant serial correlation'}")
    
    # Newey-West
    nw_sharpe = newey_west_sharpe(pnl)
    max_lag = int(np.floor(4 * (n_trades/100)**(2/9)))
    print(f"\n--- Newey-West Corrected Sharpe ---")
    print(f"NW Sharpe (lag={max_lag}): {nw_sharpe:.2f}")
    print(f"Raw Sharpe: {raw_sharpe:.2f}")
    if nw_sharpe != 0:
        print(f"Inflation factor: {raw_sharpe/nw_sharpe:.2f}x")
    
    # Block Sharpe
    block3 = non_overlapping_block_sharpe(pnl, 3)
    block5 = non_overlapping_block_sharpe(pnl, 5)
    print(f"\n--- Non-Overlapping Block Sharpe ---")
    print(f"3-day blocks: {block3:.2f}")
    print(f"5-day blocks: {block5:.2f}")
    
    # Split-half
    print(f"\n--- Split-Half Stability ---")
    mid = n_trades // 2
    for label, arr in [("First half", pnl[:mid]), ("Second half", pnl[mid:])]:
        n = len(arr)
        s = np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252) if np.std(arr) > 0 else 0
        w = np.mean(arr > 0)
        gw = np.sum(arr[arr > 0])
        gl = abs(np.sum(arr[arr < 0]))
        p = gw / gl if gl > 0 else float('inf')
        print(f"  {label}: N={n}, Sharpe={s:.2f}, WR={w:.1%}, PF={p:.2f}, Total={np.sum(arr):.1f}")
    
    s1 = np.mean(pnl[:mid]) / np.std(pnl[:mid], ddof=1) * np.sqrt(252) if np.std(pnl[:mid]) > 0 else 0
    s2 = np.mean(pnl[mid:]) / np.std(pnl[mid:], ddof=1) * np.sqrt(252) if np.std(pnl[mid:]) > 0 else 0
    
    # Bootstrap
    boot_sharpes, lower, upper = bootstrap_sharpe_ci(pnl)
    median_sharpe = np.median(boot_sharpes)
    print(f"\n--- Bootstrap 95% CI on Sharpe ---")
    print(f"Median={median_sharpe:.2f}, 95% CI=[{lower:.2f}, {upper:.2f}]")
    print(f"P(Sharpe > 0): {np.mean(boot_sharpes > 0):.1%}")
    print(f"P(Sharpe > 1): {np.mean(boot_sharpes > 1):.1%}")
    print(f"P(Sharpe > 2): {np.mean(boot_sharpes > 2):.1%}")
    
    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    stable = s1 * s2 > 0 and min(abs(s1), abs(s2)) / max(abs(s1), abs(s2)) > 0.3 if max(abs(s1), abs(s2)) > 0 else False
    print(f"Raw Sharpe:           {raw_sharpe:.2f}")
    print(f"Newey-West Sharpe:    {nw_sharpe:.2f}")
    print(f"Block-3d Sharpe:      {block3:.2f}")
    print(f"Block-5d Sharpe:      {block5:.2f}")
    print(f"Bootstrap 95% CI:     [{lower:.2f}, {upper:.2f}]")
    print(f"Ljung-Box p-value:    {lb_pval:.4f}")
    print(f"ACF(1):               {acf_vals[1]:+.4f}")
    print(f"Split-half stable:    {'YES' if stable else 'NO'}")
    
    results = {
        'n_trades': int(n_trades),
        'total_pnl_ticks': float(total_pnl),
        'raw_sharpe': float(raw_sharpe),
        'newey_west_sharpe': float(nw_sharpe),
        'block_sharpe_3d': float(block3),
        'block_sharpe_5d': float(block5),
        'bootstrap_median': float(median_sharpe),
        'bootstrap_ci': [float(lower), float(upper)],
        'bootstrap_prob_positive': float(np.mean(boot_sharpes > 0)),
        'win_rate': float(wr),
        'profit_factor': float(pf),
        'acf': {str(i): float(acf_vals[i]) for i in range(1, 6)},
        'ljung_box': {'Q': float(Q), 'p_value': float(lb_pval)},
        'split_half': {'first': float(s1), 'second': float(s2), 'stable': stable}
    }
    
    with open(OUTPUT_DIR / 'validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'validation_results.json'}")


if __name__ == '__main__':
    main()
