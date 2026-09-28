#!/usr/bin/env python3
"""
Volatility Clustering + Mean Reversion Timing Strategy
=======================================================
Exploits two documented market phenomena:
  1) After extreme volatility, markets mean-revert (go long)
  2) After volatility compression, breakouts occur (LightGBM predicts direction)

Walk-forward: 252d sliding train, 21d step, 1d label gap
Allocation: predictions -> UPRO/SPY/SHY
Adversarial: permutation (200 runs), sub-period (4 blocks), outlier robustness, regime test
Transaction costs: 10 bps
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/ml_vol_clustering'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. DATA ACQUISITION
# ============================================================================

def download_data():
    """Download SPY, ^VIX, TLT, GLD from yfinance (2010-2026)."""
    import yfinance as yf

    tickers = {'SPY': 'SPY', 'VIX': '^VIX', 'TLT': 'TLT', 'GLD': 'GLD',
               'UPRO': 'UPRO', 'SHY': 'SHY'}

    data = {}
    for name, ticker in tickers.items():
        print(f"  Downloading {name} ({ticker})...")
        df = yf.download(ticker, start='2010-01-01', end='2026-07-20', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[name] = df
        time.sleep(0.3)

    return data


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def compute_features(data):
    """Compute volatility-regime features from raw price data."""
    spy = data['SPY'].copy()
    vix = data['VIX'].copy()

    # Daily returns
    spy['ret'] = spy['Close'].pct_change()
    spy['log_ret'] = np.log(spy['Close'] / spy['Close'].shift(1))

    # --- Realized volatility at multiple timeframes ---
    for w in [5, 10, 21, 63]:
        spy[f'rvol_{w}d'] = spy['log_ret'].rolling(w).std() * np.sqrt(252)

    # --- Vol-of-vol: 21d rolling std of 5d realized vol ---
    spy['vol_of_vol'] = spy['rvol_5d'].rolling(21).std()

    # --- VIX / Realized Vol ratio (variance risk premium) ---
    # Align VIX to SPY index
    vix_close = vix['Close'].reindex(spy.index, method='ffill')
    spy['vix'] = vix_close / 100.0  # VIX is annualized %
    spy['vrp'] = spy['vix'] / spy['rvol_21d']  # Variance risk premium ratio
    spy['vrp'] = spy['vrp'].clip(0.1, 10)  # Clip extremes

    # --- Mean reversion signal: z-score relative to 20d mean ---
    spy['sma_20'] = spy['Close'].rolling(20).mean()
    spy['std_20'] = spy['Close'].rolling(20).std()
    spy['zscore_20'] = (spy['Close'] - spy['sma_20']) / spy['std_20']

    # --- Consecutive days in same direction (momentum exhaustion) ---
    signs = np.sign(spy['ret'])
    groups = (signs != signs.shift(1)).cumsum()
    spy['consec_days'] = signs.groupby(groups).cumcount() + 1
    spy['consec_days'] = spy['consec_days'] * signs  # Positive for up streaks, negative for down

    # --- Gap between today's range and 20d avg range ---
    spy['daily_range'] = (spy['High'] - spy['Low']) / spy['Close']
    spy['avg_range_20'] = spy['daily_range'].rolling(20).mean()
    spy['range_ratio'] = spy['daily_range'] / spy['avg_range_20']

    # --- Distance from 10d high and low (breakout proximity) ---
    spy['high_10d'] = spy['High'].rolling(10).max()
    spy['low_10d'] = spy['Low'].rolling(10).min()
    spy['dist_high_10d'] = (spy['Close'] - spy['high_10d']) / spy['Close']
    spy['dist_low_10d'] = (spy['Close'] - spy['low_10d']) / spy['Close']

    # --- ATR ratio: 5d ATR / 21d ATR (vol acceleration) ---
    tr = pd.DataFrame({
        'hl': spy['High'] - spy['Low'],
        'hc': abs(spy['High'] - spy['Close'].shift(1)),
        'lc': abs(spy['Low'] - spy['Close'].shift(1))
    }).max(axis=1)
    spy['atr_5'] = tr.rolling(5).mean()
    spy['atr_21'] = tr.rolling(21).mean()
    spy['atr_ratio'] = spy['atr_5'] / spy['atr_21']

    # --- Additional useful features ---
    # VIX percentile (rolling 252d)
    spy['vix_pctl'] = spy['vix'].rolling(252).rank(pct=True)

    # Realized vol percentile
    spy['rvol_pctl'] = spy['rvol_21d'].rolling(252).rank(pct=True)

    # Vol term structure proxy: 5d vol / 63d vol
    spy['vol_term'] = spy['rvol_5d'] / spy['rvol_63d']

    # Skewness of returns (21d rolling)
    spy['ret_skew_21'] = spy['log_ret'].rolling(21).skew()

    # Kurtosis of returns (21d rolling)
    spy['ret_kurt_21'] = spy['log_ret'].rolling(21).apply(
        lambda x: stats.kurtosis(x, fisher=True), raw=True
    )

    # RSI-like: proportion of up days in last 14 days
    spy['up_ratio_14'] = (spy['ret'] > 0).astype(float).rolling(14).mean()

    # Cross-asset features
    for asset in ['TLT', 'GLD']:
        if asset in data:
            asset_ret = data[asset]['Close'].pct_change().reindex(spy.index, method='ffill')
            spy[f'{asset.lower()}_ret_5d'] = asset_ret.rolling(5).sum()
            spy[f'{asset.lower()}_corr_21d'] = spy['ret'].rolling(21).corr(asset_ret)

    # Forward return (label) — 1d ahead, with 1d gap
    spy['fwd_ret_1d'] = spy['ret'].shift(-2)  # -1 for next day, -1 more for gap

    return spy


def get_feature_cols():
    """Return list of feature column names."""
    return [
        'rvol_5d', 'rvol_10d', 'rvol_21d', 'rvol_63d',
        'vol_of_vol', 'vrp', 'zscore_20',
        'consec_days', 'range_ratio',
        'dist_high_10d', 'dist_low_10d',
        'atr_ratio', 'vix_pctl', 'rvol_pctl',
        'vol_term', 'ret_skew_21', 'ret_kurt_21',
        'up_ratio_14',
        'tlt_ret_5d', 'tlt_corr_21d',
        'gld_ret_5d', 'gld_corr_21d',
    ]


# ============================================================================
# 3. STRATEGY MODES
# ============================================================================

def classify_regime(row):
    """
    Classify market into vol regime:
      - 'compression': ATR ratio < 0.7 (vol breakout mode)
      - 'extreme_vol': VIX pctl > 0.80 (mean reversion mode)
      - 'normal': everything else
    """
    if row['atr_ratio'] < 0.7:
        return 'compression'
    elif row['vix_pctl'] > 0.80:
        return 'extreme_vol'
    else:
        return 'normal'


def generate_allocation(pred, regime, confidence_threshold=0.0):
    """
    Generate allocation based on regime + prediction.

    Returns: dict with weights for UPRO, SPY, SHY
    """
    if regime == 'extreme_vol':
        # Mean reversion: go long (VIX extreme = buy fear)
        # Use SPY (not UPRO) for safety during vol
        return {'UPRO': 0.0, 'SPY': 0.8, 'SHY': 0.2}

    elif regime == 'compression':
        # Vol breakout: use ML prediction for direction
        if pred > confidence_threshold:
            return {'UPRO': 0.6, 'SPY': 0.2, 'SHY': 0.2}  # Bullish breakout
        elif pred < -confidence_threshold:
            return {'UPRO': 0.0, 'SPY': 0.0, 'SHY': 1.0}  # Bearish breakout -> safety
        else:
            return {'UPRO': 0.0, 'SPY': 0.5, 'SHY': 0.5}  # Uncertain

    else:  # normal
        # Moderate allocation based on ML prediction
        if pred > confidence_threshold:
            return {'UPRO': 0.3, 'SPY': 0.4, 'SHY': 0.3}
        elif pred < -confidence_threshold:
            return {'UPRO': 0.0, 'SPY': 0.2, 'SHY': 0.8}
        else:
            return {'UPRO': 0.1, 'SPY': 0.4, 'SHY': 0.5}


# ============================================================================
# 4. WALK-FORWARD ENGINE
# ============================================================================

def walk_forward_backtest(df, feature_cols, train_days=252, step_days=21,
                          label_col='fwd_ret_1d', txn_cost_bps=10,
                          shuffle_labels=False, random_state=None):
    """
    Sliding-window walk-forward with LightGBM.

    Args:
        shuffle_labels: If True, shuffle labels (permutation test)
        random_state: RNG seed for shuffling

    Returns: DataFrame with daily strategy returns
    """
    import lightgbm as lgb

    # Clean data
    mask = df[feature_cols + [label_col]].notna().all(axis=1)
    clean = df[mask].copy()

    X = clean[feature_cols].values
    y = clean[label_col].values
    dates = clean.index
    regimes = clean.apply(classify_regime, axis=1).values

    if shuffle_labels:
        rng = np.random.RandomState(random_state)
        y = rng.permutation(y)

    predictions = np.full(len(X), np.nan)

    # Walk-forward
    start_idx = train_days
    while start_idx < len(X):
        end_idx = min(start_idx + step_days, len(X))

        train_X = X[start_idx - train_days:start_idx]
        train_y = y[start_idx - train_days:start_idx]
        test_X = X[start_idx:end_idx]

        # LightGBM
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'verbose': -1,
            'n_jobs': 2,
            'seed': 42 if random_state is None else random_state,
        }

        dtrain = lgb.Dataset(train_X, label=train_y)
        model = lgb.train(params, dtrain, num_boost_round=200,
                          valid_sets=[dtrain], callbacks=[lgb.log_evaluation(0)])

        preds = model.predict(test_X)
        predictions[start_idx:end_idx] = preds

        start_idx += step_days

    # Build results
    valid = ~np.isnan(predictions)
    result_df = pd.DataFrame(index=dates[valid])
    result_df['prediction'] = predictions[valid]
    result_df['regime'] = regimes[valid]
    result_df['actual_ret'] = y[valid]

    # Get asset returns for allocation
    spy_rets = clean['ret'].values[valid]

    # Approximate UPRO and SHY returns from SPY
    # UPRO ~ 3x daily SPY return (simplified)
    upro_rets = spy_rets * 3.0
    shy_rets = np.full_like(spy_rets, 0.02 / 252)  # ~2% annual

    # Use actual UPRO/SHY if available in the data
    result_df['spy_ret'] = spy_rets
    result_df['upro_ret'] = upro_rets
    result_df['shy_ret'] = shy_rets

    # Generate allocations
    strat_rets = []
    prev_alloc = None
    for i in range(len(result_df)):
        row_regime = result_df['regime'].iloc[i]
        pred = result_df['prediction'].iloc[i]
        alloc = generate_allocation(pred, row_regime)

        daily_ret = (alloc['UPRO'] * result_df['upro_ret'].iloc[i] +
                     alloc['SPY'] * result_df['spy_ret'].iloc[i] +
                     alloc['SHY'] * result_df['shy_ret'].iloc[i])

        # Transaction costs on turnover
        if prev_alloc is not None:
            turnover = sum(abs(alloc[k] - prev_alloc[k]) for k in alloc) / 2
            daily_ret -= turnover * txn_cost_bps / 10000

        strat_rets.append(daily_ret)
        prev_alloc = alloc

    result_df['strategy_ret'] = strat_rets
    result_df['cum_ret'] = (1 + result_df['strategy_ret']).cumprod()
    result_df['spy_cum'] = (1 + result_df['spy_ret']).cumprod()

    return result_df, model


# ============================================================================
# 5. METRICS
# ============================================================================

def compute_metrics(rets, label='Strategy'):
    """Compute risk-adjusted performance metrics."""
    rets = rets.dropna()
    if len(rets) < 30:
        return {}

    ann = np.sqrt(252)
    total_ret = (1 + rets).prod() - 1
    cagr = (1 + total_ret) ** (252 / len(rets)) - 1
    vol = rets.std() * ann
    sharpe = rets.mean() / rets.std() * ann if rets.std() > 0 else 0

    downside = rets[rets < 0].std() * ann
    sortino = rets.mean() / (rets[rets < 0].std()) * ann if len(rets[rets < 0]) > 0 else 0

    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    win_rate = (rets > 0).mean()

    # Profit factor
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        'label': label,
        'total_return': f"{total_ret:.2%}",
        'cagr': f"{cagr:.2%}",
        'annual_vol': f"{vol:.2%}",
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown': f"{max_dd:.2%}",
        'calmar': round(calmar, 3),
        'profit_factor': round(pf, 3),
        'win_rate': f"{win_rate:.2%}",
        'n_days': len(rets),
    }


# ============================================================================
# 6. ADVERSARIAL VALIDATION
# ============================================================================

def permutation_test(df, feature_cols, n_perms=200):
    """Shuffle signal-to-date mapping, run 200 times."""
    print(f"\n--- Permutation Test ({n_perms} runs) ---")
    real_result, _ = walk_forward_backtest(df, feature_cols)
    real_sharpe = real_result['strategy_ret'].mean() / real_result['strategy_ret'].std() * np.sqrt(252)

    perm_sharpes = []
    for i in range(n_perms):
        if (i + 1) % 50 == 0:
            print(f"  Permutation {i+1}/{n_perms}...")
        perm_result, _ = walk_forward_backtest(df, feature_cols, shuffle_labels=True, random_state=i)
        s = perm_result['strategy_ret'].mean() / perm_result['strategy_ret'].std() * np.sqrt(252)
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= real_sharpe)

    return {
        'real_sharpe': round(real_sharpe, 4),
        'perm_mean': round(np.mean(perm_sharpes), 4),
        'perm_std': round(np.std(perm_sharpes), 4),
        'perm_p95': round(np.percentile(perm_sharpes, 95), 4),
        'p_value': round(p_value, 4),
        'significant_5pct': p_value < 0.05,
    }


def sub_period_test(result_df, n_blocks=4):
    """Split OOT into n_blocks and check consistency."""
    print(f"\n--- Sub-Period Test ({n_blocks} blocks) ---")
    rets = result_df['strategy_ret'].dropna()
    block_size = len(rets) // n_blocks

    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else len(rets)
        block_rets = rets.iloc[start:end]

        sharpe = block_rets.mean() / block_rets.std() * np.sqrt(252) if block_rets.std() > 0 else 0
        total = (1 + block_rets).prod() - 1

        blocks.append({
            'block': i + 1,
            'start': str(block_rets.index[0].date()),
            'end': str(block_rets.index[-1].date()),
            'sharpe': round(sharpe, 3),
            'total_return': f"{total:.2%}",
            'n_days': len(block_rets),
        })

    sharpes = [b['sharpe'] for b in blocks]
    positive_blocks = sum(1 for s in sharpes if s > 0)

    return {
        'blocks': blocks,
        'positive_blocks': f"{positive_blocks}/{n_blocks}",
        'min_sharpe': round(min(sharpes), 3),
        'max_sharpe': round(max(sharpes), 3),
        'sharpe_std': round(np.std(sharpes), 3),
        'consistent': positive_blocks >= n_blocks * 0.75,
    }


def outlier_robustness_test(result_df, pct=1):
    """Remove top/bottom pct% of returns and recompute metrics."""
    print(f"\n--- Outlier Robustness (removing {pct}% tails) ---")
    rets = result_df['strategy_ret'].dropna()

    lo = rets.quantile(pct / 100)
    hi = rets.quantile(1 - pct / 100)
    trimmed = rets[(rets >= lo) & (rets <= hi)]

    full_metrics = compute_metrics(rets, 'Full')
    trimmed_metrics = compute_metrics(trimmed, f'Trimmed ({pct}%)')

    return {
        'full': full_metrics,
        'trimmed': trimmed_metrics,
        'sharpe_change': round(
            float(trimmed_metrics.get('sharpe', 0)) - float(full_metrics.get('sharpe', 0)), 3
        ) if full_metrics and trimmed_metrics else None,
    }


def regime_test(result_df, df_features):
    """R1 regime-agnostic test: stratify by green/red/flat SPY days."""
    print("\n--- R1 Regime Test (green/red/flat days) ---")
    rets = result_df['strategy_ret']
    spy_rets = result_df['spy_ret']

    green = spy_rets > 0.001   # >10 bps
    red = spy_rets < -0.001    # <-10 bps
    flat = ~green & ~red

    results = {}
    for label, mask in [('green', green), ('red', red), ('flat', flat)]:
        sub = rets[mask]
        if len(sub) > 20:
            sharpe = sub.mean() / sub.std() * np.sqrt(252) if sub.std() > 0 else 0
            results[label] = {
                'sharpe': round(sharpe, 3),
                'mean_ret': f"{sub.mean():.5f}",
                'n_days': int(mask.sum()),
            }

    # Check regime balance
    if 'green' in results and 'red' in results:
        sg, sr = results['green']['sharpe'], results['red']['sharpe']
        imbalance = abs(sg - sr) / max(abs(sg), abs(sr), 0.001)
        results['regime_imbalance'] = round(imbalance, 3)
        results['regime_balanced'] = imbalance <= 0.50

    return results


# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================

def feature_importance_report(model, feature_cols):
    """Extract and rank feature importances."""
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feature_cols, imp), key=lambda x: -x[1])
    return [{'feature': f, 'importance': round(float(v), 2)} for f, v in ranked]


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    start_time = time.time()
    print("=" * 70)
    print("VOLATILITY CLUSTERING + MEAN REVERSION TIMING")
    print("=" * 70)

    # --- Download data ---
    print("\n[1] Downloading data...")
    data = download_data()
    print(f"  SPY: {len(data['SPY'])} rows, {data['SPY'].index[0].date()} to {data['SPY'].index[-1].date()}")

    # --- Feature engineering ---
    print("\n[2] Computing features...")
    df = compute_features(data)
    feature_cols = get_feature_cols()

    # Check feature coverage
    valid_mask = df[feature_cols + ['fwd_ret_1d']].notna().all(axis=1)
    print(f"  Total rows: {len(df)}, Valid rows: {valid_mask.sum()}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Date range: {df[valid_mask].index[0].date()} to {df[valid_mask].index[-1].date()}")

    # --- Regime distribution ---
    regimes = df[valid_mask].apply(classify_regime, axis=1)
    regime_counts = regimes.value_counts()
    print(f"\n  Regime distribution:")
    for r, c in regime_counts.items():
        print(f"    {r}: {c} days ({c/len(regimes):.1%})")

    # --- Walk-forward backtest ---
    print("\n[3] Running walk-forward backtest...")
    result_df, model = walk_forward_backtest(df, feature_cols)
    print(f"  OOT period: {result_df.index[0].date()} to {result_df.index[-1].date()}")
    print(f"  OOT days: {len(result_df)}")

    # --- Primary metrics ---
    print("\n[4] Performance Metrics")
    strat_metrics = compute_metrics(result_df['strategy_ret'], 'Vol Clustering Strategy')
    spy_metrics = compute_metrics(result_df['spy_ret'], 'SPY Buy & Hold')

    print(f"\n  Strategy: Sharpe={strat_metrics['sharpe']}, Sortino={strat_metrics['sortino']}, "
          f"CAGR={strat_metrics['cagr']}, MaxDD={strat_metrics['max_drawdown']}, "
          f"PF={strat_metrics['profit_factor']}, WR={strat_metrics['win_rate']}")
    print(f"  SPY B&H:  Sharpe={spy_metrics['sharpe']}, Sortino={spy_metrics['sortino']}, "
          f"CAGR={spy_metrics['cagr']}, MaxDD={spy_metrics['max_drawdown']}")

    # --- Feature importance ---
    print("\n[5] Feature Importance")
    feat_imp = feature_importance_report(model, feature_cols)
    for fi in feat_imp[:10]:
        print(f"    {fi['feature']:25s} {fi['importance']:.1f}")

    # --- Adversarial tests ---
    print("\n[6] Adversarial Validation")

    # 6a. Permutation test
    perm_results = permutation_test(df, feature_cols, n_perms=200)
    print(f"  Permutation: real_sharpe={perm_results['real_sharpe']}, "
          f"p_value={perm_results['p_value']}, significant={perm_results['significant_5pct']}")

    # 6b. Sub-period test
    subperiod = sub_period_test(result_df, n_blocks=4)
    print(f"  Sub-period: {subperiod['positive_blocks']} positive, consistent={subperiod['consistent']}")
    for b in subperiod['blocks']:
        print(f"    Block {b['block']}: {b['start']} to {b['end']}, Sharpe={b['sharpe']}, Ret={b['total_return']}")

    # 6c. Outlier robustness
    outlier = outlier_robustness_test(result_df, pct=1)
    print(f"  Outlier robustness: Sharpe change = {outlier['sharpe_change']} after removing 1% tails")

    # 6d. Regime test
    regime_results = regime_test(result_df, df)
    for label in ['green', 'red', 'flat']:
        if label in regime_results:
            r = regime_results[label]
            print(f"  {label:6s} days: Sharpe={r['sharpe']}, n={r['n_days']}")
    if 'regime_balanced' in regime_results:
        print(f"  Regime balanced: {regime_results['regime_balanced']} "
              f"(imbalance={regime_results['regime_imbalance']})")

    # --- Per-year breakdown ---
    print("\n[7] Per-Year Breakdown")
    result_df['year'] = result_df.index.year
    yearly = []
    for yr, grp in result_df.groupby('year'):
        m = compute_metrics(grp['strategy_ret'], str(yr))
        m_spy = compute_metrics(grp['spy_ret'], f'SPY {yr}')
        yearly.append({
            'year': yr,
            'strat_sharpe': m.get('sharpe', 0),
            'strat_return': m.get('total_return', '0%'),
            'spy_sharpe': m_spy.get('sharpe', 0),
            'spy_return': m_spy.get('total_return', '0%'),
        })
        print(f"  {yr}: Strat Sharpe={m.get('sharpe',0):.3f} Ret={m.get('total_return','N/A')}, "
              f"SPY Sharpe={m_spy.get('sharpe',0):.3f} Ret={m_spy.get('total_return','N/A')}")

    # --- Save results ---
    print("\n[8] Saving results...")

    results = {
        'strategy': 'Volatility Clustering + Mean Reversion Timing',
        'run_date': datetime.now().isoformat(),
        'data_range': f"{df[valid_mask].index[0].date()} to {df[valid_mask].index[-1].date()}",
        'oot_range': f"{result_df.index[0].date()} to {result_df.index[-1].date()}",
        'oot_days': len(result_df),
        'features': feature_cols,
        'n_features': len(feature_cols),
        'walk_forward': {'train_days': 252, 'step_days': 21, 'label_gap': 1, 'window': 'sliding'},
        'txn_cost_bps': 10,
        'strategy_metrics': strat_metrics,
        'benchmark_metrics': spy_metrics,
        'feature_importance': feat_imp,
        'permutation_test': perm_results,
        'sub_period_test': subperiod,
        'outlier_robustness': outlier,
        'regime_test': regime_results,
        'yearly_breakdown': yearly,
        'regime_distribution': {k: int(v) for k, v in regime_counts.items()},
        'runtime_seconds': round(time.time() - start_time, 1),
    }

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    result_df.to_csv(os.path.join(OUTPUT_DIR, 'daily_returns.csv'))

    # --- Summary verdict ---
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    checks = []
    checks.append(('Sharpe > 0.5', strat_metrics['sharpe'] > 0.5))
    checks.append(('Sortino > 0.7', strat_metrics['sortino'] > 0.7))
    checks.append(('Permutation p < 0.05', perm_results['significant_5pct']))
    checks.append(('Sub-period consistent', subperiod['consistent']))
    checks.append(('Regime balanced', regime_results.get('regime_balanced', False)))
    checks.append(('Beats SPY Sharpe', strat_metrics['sharpe'] > spy_metrics['sharpe']))

    passed = sum(1 for _, v in checks if v)
    for name, val in checks:
        status = 'PASS' if val else 'FAIL'
        print(f"  [{status}] {name}")

    print(f"\n  Score: {passed}/{len(checks)} checks passed")
    if passed >= 5:
        print("  -> PROMISING: Worth further investigation")
    elif passed >= 3:
        print("  -> MARGINAL: Needs refinement")
    else:
        print("  -> WEAK: Unlikely to be a real edge")

    print(f"\n  Runtime: {time.time() - start_time:.1f}s")
    print(f"  Results saved to {OUTPUT_DIR}/")

    return results


if __name__ == '__main__':
    main()
