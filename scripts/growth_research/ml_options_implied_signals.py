#!/usr/bin/env python3
"""
Options-Implied Directional Signals — ML Strategy
====================================================
Concept: Options markets often lead equity markets. VIX term structure,
implied vol dynamics, and volatility regime signals carry predictive
information for next-week equity moves.

Features:
  - VIX/VIX3M ratio (term structure — contango vs backwardation)
  - VIX percentile rank (252d rolling)
  - VIX 5d change (momentum)
  - VIX mean-reversion signal (z-score vs 20d MA)
  - SKEW index (tail risk proxy)
  - Put-call ratio proxy (VIX-based construction)
  - Realized vs implied vol spread
  - VIX term structure slope
  - VIX acceleration (change of change)

Target: Next 5-day SPY return (regression) → quintile → trade signal
Strategy: Q5 (top) → UPRO, Q1 (bottom) → SHY, Q2-Q4 → SPY

Walk-forward: 252d sliding train, 21d test step, 5d label gap
Adversarial: permutation test, sub-period stability, outlier robustness
Regime stratification: R1 gate (|Sharpe_up - Sharpe_down| / max < 0.50)

Author: Claude (Opus 4.6), 2026-07-20
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
import json
import warnings
import sys
import time
import os

os.environ['PYTHONUNBUFFERED'] = '1'
warnings.filterwarnings("ignore")

# Force unbuffered output
_print = print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _print(*args, **kwargs)

# ─── Paths ───
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/ml_options_implied_signals")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ───
TRAIN_DAYS = 252        # 1 year sliding window
TEST_STEP = 21          # 3 weeks per step
LABEL_GAP = 5           # 5-day gap (predicting 5d returns)
FORWARD_DAYS = 5        # 5-day forward return target
COST_BPS = 10           # 10 bps per trade
N_PERM = 50             # Permutation test iterations
REGIME_THRESHOLD = 0.50 # R1 gate

# ─── Data Download ───
def download_data():
    """Download all required data from yfinance."""
    import yfinance as yf

    print("Downloading data from yfinance...")
    tickers = {
        'SPY': 'SPY',
        'QQQ': 'QQQ',
        'VIX': '^VIX',
        'SKEW': '^SKEW',
        'UPRO': 'UPRO',
        'SHY': 'SHY',
    }

    # Try VIX3M first, fallback to VIXMT, then construct proxy
    vix3m_tickers = ['^VIX3M', '^VIX9D']

    data = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start='2011-01-01', end='2026-07-19',
                           progress=False, auto_adjust=True)
            if len(df) > 100:
                data[name] = df['Close'].squeeze()
                print(f"  {name} ({ticker}): {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
            else:
                print(f"  {name} ({ticker}): insufficient data ({len(df)} rows)")
        except Exception as e:
            print(f"  {name} ({ticker}): FAILED - {e}")

    # VIX3M — try multiple tickers
    vix3m_found = False
    for vt in vix3m_tickers:
        try:
            df = yf.download(vt, start='2011-01-01', end='2026-07-19',
                           progress=False, auto_adjust=True)
            if len(df) > 100:
                data['VIX3M'] = df['Close'].squeeze()
                print(f"  VIX3M ({vt}): {len(df)} rows")
                vix3m_found = True
                break
        except:
            continue

    if not vix3m_found:
        print("  VIX3M: not available, will construct proxy from VIX")

    return data


def build_features(data):
    """Build options-implied feature set."""
    spy = data['SPY'].rename('spy_close')
    vix = data['VIX'].rename('vix')

    # Align all series to SPY's index
    df = pd.DataFrame({'spy_close': spy})
    df['vix'] = data['VIX'].reindex(df.index)
    df['qqq_close'] = data['QQQ'].reindex(df.index) if 'QQQ' in data else np.nan

    if 'SKEW' in data:
        df['skew'] = data['SKEW'].reindex(df.index)
    else:
        df['skew'] = np.nan

    if 'VIX3M' in data:
        df['vix3m'] = data['VIX3M'].reindex(df.index)
    else:
        # Proxy: VIX3M ≈ VIX * 1.05 + 20d EMA smoothing (crude but usable)
        df['vix3m'] = df['vix'].ewm(span=63).mean() * 1.02

    # UPRO and SHY for strategy returns
    if 'UPRO' in data:
        df['upro_close'] = data['UPRO'].reindex(df.index)
    if 'SHY' in data:
        df['shy_close'] = data['SHY'].reindex(df.index)

    df = df.dropna(subset=['spy_close', 'vix'])

    # ─── Feature Engineering ───
    print("Building features...")

    # 1. VIX/VIX3M ratio (term structure)
    df['vix_vix3m_ratio'] = df['vix'] / df['vix3m']

    # 2. VIX percentile rank (252d rolling)
    df['vix_pctrank_252'] = df['vix'].rolling(252).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) == 252 else np.nan,
        raw=False
    )

    # 3. VIX 5d change
    df['vix_5d_chg'] = df['vix'].pct_change(5)

    # 4. VIX 1d change
    df['vix_1d_chg'] = df['vix'].pct_change(1)

    # 5. VIX mean-reversion z-score (vs 20d MA)
    vix_ma20 = df['vix'].rolling(20).mean()
    vix_std20 = df['vix'].rolling(20).std()
    df['vix_zscore_20'] = (df['vix'] - vix_ma20) / vix_std20

    # 6. VIX mean-reversion z-score (vs 60d MA)
    vix_ma60 = df['vix'].rolling(60).mean()
    vix_std60 = df['vix'].rolling(60).std()
    df['vix_zscore_60'] = (df['vix'] - vix_ma60) / vix_std60

    # 7. Realized vol (20d) vs VIX spread
    spy_ret = df['spy_close'].pct_change()
    df['realized_vol_20'] = spy_ret.rolling(20).std() * np.sqrt(252) * 100
    df['rv_iv_spread'] = df['vix'] - df['realized_vol_20']

    # 8. Realized vol 60d
    df['realized_vol_60'] = spy_ret.rolling(60).std() * np.sqrt(252) * 100
    df['rv60_iv_spread'] = df['vix'] - df['realized_vol_60']

    # 9. VIX term structure slope
    df['vix_term_slope'] = df['vix3m'] - df['vix']

    # 10. Put-call ratio proxy (VIX-based)
    # Higher VIX relative to realized vol → more put buying → bearish
    # Lower VIX relative to realized vol → complacency → contrarian bearish
    df['pcr_proxy'] = df['rv_iv_spread'] / (df['realized_vol_20'] + 1e-6)

    # 11. SKEW features (if available)
    if df['skew'].notna().sum() > 100:
        df['skew_pctrank_60'] = df['skew'].rolling(60).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) == 60 else np.nan,
            raw=False
        )
        df['skew_5d_chg'] = df['skew'].pct_change(5)
    else:
        df['skew_pctrank_60'] = np.nan
        df['skew_5d_chg'] = np.nan

    # 12. VIX acceleration
    df['vix_accel'] = df['vix_1d_chg'].diff(1)

    # 13. SPY momentum features (context)
    df['spy_ret_5d'] = df['spy_close'].pct_change(5)
    df['spy_ret_21d'] = df['spy_close'].pct_change(21)
    df['spy_ma_ratio_50'] = df['spy_close'] / df['spy_close'].rolling(50).mean()

    # 14. VIX contango/backwardation regime
    df['vix_contango'] = (df['vix_vix3m_ratio'] < 1.0).astype(float)

    # 15. Composite fear gauge: VIX z-score * term structure inversion
    df['fear_composite'] = df['vix_zscore_20'] * df['vix_vix3m_ratio']

    # ─── Target: 5-day forward SPY return ───
    df['fwd_ret_5d'] = df['spy_close'].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)

    # Direction target (for classification reference)
    df['fwd_dir_5d'] = (df['fwd_ret_5d'] > 0).astype(int)

    # ─── Regime label for R1 stratification ───
    df['spy_daily_ret'] = spy_ret
    df['regime_5d'] = np.where(df['fwd_ret_5d'] > 0.005, 'up',
                               np.where(df['fwd_ret_5d'] < -0.005, 'down', 'flat'))

    print(f"  Total rows after feature engineering: {len(df)}")
    return df


def get_feature_cols(df):
    """Return list of feature columns."""
    feat_cols = [
        'vix_vix3m_ratio', 'vix_pctrank_252', 'vix_5d_chg', 'vix_1d_chg',
        'vix_zscore_20', 'vix_zscore_60', 'realized_vol_20', 'rv_iv_spread',
        'realized_vol_60', 'rv60_iv_spread', 'vix_term_slope', 'pcr_proxy',
        'vix_accel', 'spy_ret_5d', 'spy_ret_21d', 'spy_ma_ratio_50',
        'vix_contango', 'fear_composite',
    ]
    # Add SKEW features if they exist with enough data
    if df['skew_pctrank_60'].notna().sum() > 100:
        feat_cols += ['skew_pctrank_60', 'skew_5d_chg']
    return feat_cols


def walk_forward_backtest(df, feat_cols, shuffle_labels=False, seed=42, fast=False):
    """
    Sliding window walk-forward with LightGBM.

    Parameters:
        shuffle_labels: if True, shuffle target within training set (permutation test)
        seed: random seed for shuffling
        fast: if True, use fewer boosting rounds (for permutation test speed)

    Returns:
        DataFrame with predictions, actuals, and strategy returns
    """
    df_clean = df.dropna(subset=feat_cols + ['fwd_ret_5d']).copy()
    df_clean = df_clean.reset_index()  # bring Date into columns

    results = []
    n = len(df_clean)
    n_rounds = 100 if fast else 300

    # Walk-forward loop
    step = 0
    i = TRAIN_DAYS
    while i + LABEL_GAP + TEST_STEP <= n:
        train_start = i - TRAIN_DAYS
        train_end = i
        test_start = i + LABEL_GAP  # 5-day gap for label leakage prevention
        test_end = min(test_start + TEST_STEP, n)

        if test_end <= test_start:
            break

        train = df_clean.iloc[train_start:train_end]
        test = df_clean.iloc[test_start:test_end]

        X_train = train[feat_cols].values
        y_train = train['fwd_ret_5d'].values.copy()

        if shuffle_labels:
            rng = np.random.RandomState(seed + step)
            rng.shuffle(y_train)

        X_test = test[feat_cols].values
        y_test = test['fwd_ret_5d'].values

        # LightGBM regressor
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'learning_rate': 0.05 if fast else 0.03,
            'num_leaves': 15,
            'max_depth': 4,
            'min_child_samples': 30,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbose': -1,
            'n_jobs': 2,  # Keep CPU moderate
            'seed': 42,
        }

        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            callbacks=[lgb.log_evaluation(period=0)],
        )

        preds = model.predict(X_test)

        for j in range(len(test)):
            row = test.iloc[j]
            results.append({
                'date': row['Date'],
                'pred': preds[j],
                'actual': y_test[j],
                'spy_close': row['spy_close'],
                'regime': row['regime_5d'],
                'step': step,
            })

        i += TEST_STEP
        step += 1

    return pd.DataFrame(results)


def compute_strategy_returns(results_df, df_full):
    """
    Compute strategy returns:
      Q5 (top quintile prediction) → UPRO
      Q1 (bottom quintile prediction) → SHY
      Q2-Q4 → SPY

    With 10 bps transaction costs on position changes.
    """
    res = results_df.copy()
    res = res.sort_values('date').reset_index(drop=True)

    # Quintile assignment per test step (handle edge cases with few unique predictions)
    def assign_quintile(group):
        group = group.copy()
        try:
            group['quintile'] = pd.qcut(group['pred'], 5, labels=[1, 2, 3, 4, 5],
                                         duplicates='drop')
        except ValueError:
            # Fallback: rank-based quintile
            group['quintile'] = pd.cut(group['pred'].rank(method='first'),
                                        bins=5, labels=[1, 2, 3, 4, 5])
        group['quintile'] = group['quintile'].fillna(3)  # middle quintile for any NaN
        return group

    # Assign quintiles within each step
    res = res.groupby('step', group_keys=False).apply(assign_quintile)
    res['quintile'] = res['quintile'].astype(int)

    # Merge UPRO/SHY/SPY returns
    df_full_idx = df_full.reset_index() if 'Date' not in df_full.columns else df_full.copy()
    if 'Date' not in df_full_idx.columns and df_full_idx.index.name == 'Date':
        df_full_idx = df_full_idx.reset_index()

    # Compute 5-day forward returns for each instrument
    for col, ret_col in [('spy_close', 'spy_fwd5d'), ('upro_close', 'upro_fwd5d'),
                          ('shy_close', 'shy_fwd5d')]:
        if col in df_full_idx.columns:
            s = df_full_idx.set_index('Date')[col]
            fwd = s.pct_change(5).shift(-5)
            fwd_df = fwd.reset_index()
            fwd_df.columns = ['date', ret_col]
            res = res.merge(fwd_df, on='date', how='left')

    # Strategy return per row
    def strategy_return(row):
        q = row['quintile']
        if q == 5 and 'upro_fwd5d' in row.index and pd.notna(row.get('upro_fwd5d')):
            return row['upro_fwd5d']
        elif q == 1 and 'shy_fwd5d' in row.index and pd.notna(row.get('shy_fwd5d')):
            return row['shy_fwd5d']
        else:
            return row.get('spy_fwd5d', row['actual'])

    res['strat_ret_gross'] = res.apply(strategy_return, axis=1)

    # De-overlap: take every 5th observation to get non-overlapping 5-day returns
    res = res.iloc[::FORWARD_DAYS].reset_index(drop=True)

    # Transaction costs: 10 bps when position changes
    res['prev_quintile'] = res['quintile'].shift(1)
    res['position_change'] = (res['quintile'] != res['prev_quintile']).astype(float)
    res['cost'] = res['position_change'] * COST_BPS / 10000
    res['strat_ret_net'] = res['strat_ret_gross'] - res['cost']

    # Benchmark: buy-and-hold SPY
    if 'spy_fwd5d' in res.columns:
        res['bench_ret'] = res['spy_fwd5d']
    else:
        res['bench_ret'] = res['actual']

    return res


def compute_metrics(returns, name="Strategy", period_days=5):
    """
    Compute risk-adjusted metrics from a return series.
    period_days: number of trading days per return period (5 for weekly).
    """
    returns = returns.dropna()
    if len(returns) < 10:
        return {'name': name, 'n_obs': len(returns), 'sharpe': 0}

    # These are 5-day (weekly) returns, so annualize accordingly
    periods_per_year = 252 / period_days  # ~50.4

    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / periods_per_year
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(periods_per_year)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown on cumulative equity
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    win_rate = (returns > 0).mean()
    profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if (returns < 0).any() else np.inf

    return {
        'name': name,
        'n_obs': len(returns),
        'total_return': f"{total_ret:.2%}",
        'ann_return': f"{ann_ret:.2%}",
        'ann_vol': f"{ann_vol:.2%}",
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd': f"{max_dd:.2%}",
        'win_rate': f"{win_rate:.1%}",
        'profit_factor': round(profit_factor, 3),
    }


def regime_stratification(res_df):
    """R1 regime-agnostic validation. Returns regime metrics + pass/fail."""
    up = res_df[res_df['regime'] == 'up']['strat_ret_net']
    down = res_df[res_df['regime'] == 'down']['strat_ret_net']
    flat = res_df[res_df['regime'] == 'flat']['strat_ret_net']

    periods_per_year = 252 / 5  # weekly returns
    metrics = {}
    for name, rets in [('up', up), ('down', down), ('flat', flat)]:
        if len(rets) > 10:
            ann_ret = rets.mean() * periods_per_year
            ann_vol = rets.std() * np.sqrt(periods_per_year)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            metrics[name] = {
                'n': len(rets),
                'mean_ret': f"{rets.mean():.4%}",
                'sharpe': round(sharpe, 3),
                'win_rate': f"{(rets > 0).mean():.1%}",
            }

    # R1 gate check
    if 'up' in metrics and 'down' in metrics:
        s_up = metrics['up']['sharpe']
        s_down = metrics['down']['sharpe']
        denom = max(abs(s_up), abs(s_down), 0.001)
        regime_gap = abs(s_up - s_down) / denom
        r1_pass = regime_gap < REGIME_THRESHOLD
    else:
        regime_gap = np.nan
        r1_pass = False

    return metrics, regime_gap, r1_pass


def permutation_test(df, feat_cols, n_perms=N_PERM):
    """Shuffle signal-to-date mapping and re-run to get null distribution."""
    print(f"\nRunning permutation test ({n_perms} iterations)...")

    # First get the real Sharpe
    real_results = walk_forward_backtest(df, feat_cols, shuffle_labels=False)
    if len(real_results) == 0:
        return None, None, None

    real_strat = compute_strategy_returns(real_results, df)
    real_metrics = compute_metrics(real_strat['strat_ret_net'], "Real")
    real_sharpe = real_metrics.get('sharpe', 0)

    null_sharpes = []
    for i in range(n_perms):
        if (i + 1) % 10 == 0:
            print(f"  Permutation {i+1}/{n_perms}...")
        perm_results = walk_forward_backtest(df, feat_cols, shuffle_labels=True, seed=i, fast=True)
        if len(perm_results) == 0:
            continue
        perm_strat = compute_strategy_returns(perm_results, df)
        perm_m = compute_metrics(perm_strat['strat_ret_net'], "Perm")
        null_sharpes.append(perm_m.get('sharpe', 0))

    null_sharpes = np.array(null_sharpes)
    p_value = (null_sharpes >= real_sharpe).mean() if len(null_sharpes) > 0 else 1.0

    return real_sharpe, null_sharpes, p_value


def sub_period_stability(res_df, n_blocks=4):
    """Split OOT results into n_blocks and check consistency."""
    res_sorted = res_df.sort_values('date')
    block_size = len(res_sorted) // n_blocks
    blocks = []

    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else len(res_sorted)
        block = res_sorted.iloc[start:end]
        m = compute_metrics(block['strat_ret_net'], f"Block {i+1}")
        m['date_range'] = f"{block['date'].iloc[0].date()} to {block['date'].iloc[-1].date()}"
        blocks.append(m)

    # Check: how many blocks have positive Sharpe?
    positive_blocks = sum(1 for b in blocks if b.get('sharpe', 0) > 0)
    stable = positive_blocks >= n_blocks // 2

    return blocks, stable


def outlier_robustness(res_df, trim_pct=0.02):
    """Remove top/bottom 2% of returns and recompute metrics."""
    rets = res_df['strat_ret_net'].dropna()
    lo = rets.quantile(trim_pct)
    hi = rets.quantile(1 - trim_pct)
    trimmed = rets[(rets >= lo) & (rets <= hi)]

    original = compute_metrics(rets, "Original")
    trimmed_m = compute_metrics(trimmed, "Trimmed (2%)")

    # Robust if Sharpe doesn't collapse by more than 50%
    orig_sharpe = original.get('sharpe', 0)
    trim_sharpe = trimmed_m.get('sharpe', 0)
    robust = True
    if orig_sharpe > 0:
        robust = trim_sharpe / orig_sharpe > 0.50

    return original, trimmed_m, robust


def feature_importance(df, feat_cols):
    """Train a single model on all data and get feature importance."""
    df_clean = df.dropna(subset=feat_cols + ['fwd_ret_5d']).copy()
    X = df_clean[feat_cols].values
    y = df_clean['fwd_ret_5d'].values

    params = {
        'objective': 'regression', 'metric': 'mae',
        'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4,
        'min_child_samples': 30, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'verbose': -1, 'n_jobs': 2, 'seed': 42,
    }
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=300,
                      callbacks=[lgb.log_evaluation(period=0)])

    importance = dict(zip(feat_cols, model.feature_importance('gain')))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    return importance


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("  Options-Implied Directional Signals — ML Strategy")
    print("=" * 70)

    # 1. Download data
    data = download_data()
    if 'SPY' not in data or 'VIX' not in data:
        print("FATAL: Missing SPY or VIX data. Aborting.")
        sys.exit(1)

    # 2. Build features
    df = build_features(data)
    feat_cols = get_feature_cols(df)
    print(f"\nFeature columns ({len(feat_cols)}): {feat_cols}")

    # Check data sufficiency
    df_valid = df.dropna(subset=feat_cols + ['fwd_ret_5d'])
    print(f"Valid rows for modeling: {len(df_valid)} ({df_valid.index[0].date()} to {df_valid.index[-1].date()})")

    if len(df_valid) < TRAIN_DAYS + LABEL_GAP + TEST_STEP + 50:
        print("FATAL: Insufficient data for walk-forward. Aborting.")
        sys.exit(1)

    # 3. Walk-forward backtest
    print("\n" + "─" * 50)
    print("WALK-FORWARD BACKTEST")
    print("─" * 50)
    results = walk_forward_backtest(df, feat_cols, shuffle_labels=False)
    print(f"  Walk-forward produced {len(results)} predictions across {results['step'].nunique()} steps")

    if len(results) < 50:
        print("FATAL: Too few predictions. Aborting.")
        sys.exit(1)

    # 4. Strategy returns
    strat_df = compute_strategy_returns(results, df)

    # 5. Overall metrics
    print("\n" + "─" * 50)
    print("STRATEGY METRICS (net of 10 bps costs)")
    print("─" * 50)
    strat_metrics = compute_metrics(strat_df['strat_ret_net'], "Options-Implied Signal")
    bench_metrics = compute_metrics(strat_df['bench_ret'], "SPY Buy & Hold")

    for m in [strat_metrics, bench_metrics]:
        print(f"\n  {m['name']}:")
        for k, v in m.items():
            if k != 'name':
                print(f"    {k}: {v}")

    # Prediction quality
    ic = strat_df[['pred', 'actual']].corr().iloc[0, 1]
    rank_ic = strat_df[['pred', 'actual']].corr('spearman').iloc[0, 1]
    print(f"\n  Prediction IC (Pearson): {ic:.4f}")
    print(f"  Prediction IC (Spearman): {rank_ic:.4f}")

    # Quintile analysis
    print("\n  Quintile Analysis:")
    for q in sorted(strat_df['quintile'].unique()):
        q_rets = strat_df[strat_df['quintile'] == q]['actual']
        print(f"    Q{q}: mean 5d ret = {q_rets.mean():.4%}, n={len(q_rets)}")

    # 6. Regime stratification (R1)
    print("\n" + "─" * 50)
    print("R1 REGIME STRATIFICATION")
    print("─" * 50)
    regime_metrics, regime_gap, r1_pass = regime_stratification(strat_df)
    for regime, m in regime_metrics.items():
        print(f"  {regime}: n={m['n']}, mean_ret={m['mean_ret']}, Sharpe={m['sharpe']}, WR={m['win_rate']}")
    print(f"\n  Regime gap: {regime_gap:.3f} (threshold: {REGIME_THRESHOLD})")
    print(f"  R1 PASS: {'YES' if r1_pass else 'NO'}")

    # 7. Permutation test
    print("\n" + "─" * 50)
    print("PERMUTATION TEST")
    print("─" * 50)
    real_sharpe, null_sharpes, p_value = permutation_test(df, feat_cols, n_perms=N_PERM)
    if null_sharpes is not None:
        print(f"  Real Sharpe: {real_sharpe:.3f}")
        print(f"  Null Sharpe (median): {np.median(null_sharpes):.3f}")
        print(f"  Null Sharpe (95th pct): {np.percentile(null_sharpes, 95):.3f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant (p < 0.05): {'YES' if p_value < 0.05 else 'NO'}")
    else:
        print("  Permutation test FAILED (insufficient data)")

    # 8. Sub-period stability
    print("\n" + "─" * 50)
    print("SUB-PERIOD STABILITY (4 blocks)")
    print("─" * 50)
    blocks, stable = sub_period_stability(strat_df, n_blocks=4)
    for b in blocks:
        print(f"  {b.get('name', '?')}: {b.get('date_range', '?')} | "
              f"Sharpe={b.get('sharpe', '?')} | WR={b.get('win_rate', '?')} | "
              f"Return={b.get('total_return', '?')}")
    print(f"\n  Stable: {'YES' if stable else 'NO'}")

    # 9. Outlier robustness
    print("\n" + "─" * 50)
    print("OUTLIER ROBUSTNESS (2% trim)")
    print("─" * 50)
    orig_m, trim_m, robust = outlier_robustness(strat_df)
    print(f"  Original: Sharpe={orig_m.get('sharpe', '?')}, WR={orig_m.get('win_rate', '?')}")
    print(f"  Trimmed:  Sharpe={trim_m.get('sharpe', '?')}, WR={trim_m.get('win_rate', '?')}")
    print(f"  Robust: {'YES' if robust else 'NO'}")

    # 10. Feature importance
    print("\n" + "─" * 50)
    print("FEATURE IMPORTANCE (gain)")
    print("─" * 50)
    fi = feature_importance(df, feat_cols)
    total_gain = sum(fi.values())
    for feat, gain in fi.items():
        pct = gain / total_gain * 100
        print(f"  {feat:30s} {gain:10.1f}  ({pct:.1f}%)")

    # ─── Save Results ───
    print("\n" + "─" * 50)
    print("SAVING RESULTS")
    print("─" * 50)

    # Save detailed results
    strat_df.to_csv(OUT_DIR / "walk_forward_results.csv", index=False)

    # Summary JSON
    summary = {
        'run_date': datetime.now().isoformat(),
        'data_range': f"{df_valid.index[0].date()} to {df_valid.index[-1].date()}",
        'n_features': len(feat_cols),
        'features': feat_cols,
        'n_predictions': len(strat_df),
        'n_wf_steps': int(strat_df['step'].nunique()),
        'train_window': TRAIN_DAYS,
        'test_step': TEST_STEP,
        'label_gap': LABEL_GAP,
        'forward_days': FORWARD_DAYS,
        'cost_bps': COST_BPS,
        'strategy_metrics': strat_metrics,
        'benchmark_metrics': bench_metrics,
        'prediction_ic_pearson': round(ic, 4),
        'prediction_ic_spearman': round(rank_ic, 4),
        'regime_stratification': {
            'metrics': regime_metrics,
            'regime_gap': round(regime_gap, 4) if not np.isnan(regime_gap) else None,
            'r1_pass': r1_pass,
        },
        'permutation_test': {
            'real_sharpe': round(real_sharpe, 4) if real_sharpe is not None else None,
            'null_median': round(float(np.median(null_sharpes)), 4) if null_sharpes is not None else None,
            'null_95pct': round(float(np.percentile(null_sharpes, 95)), 4) if null_sharpes is not None else None,
            'p_value': round(p_value, 4) if p_value is not None else None,
            'significant': bool(p_value < 0.05) if p_value is not None else False,
            'n_permutations': N_PERM,
        },
        'sub_period_stability': {
            'blocks': blocks,
            'stable': stable,
        },
        'outlier_robustness': {
            'original': orig_m,
            'trimmed': trim_m,
            'robust': robust,
        },
        'feature_importance': {k: round(v, 1) for k, v in fi.items()},
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(OUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Overall verdict
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    all_pass = r1_pass and (p_value is not None and p_value < 0.05) and stable and robust
    print(f"  R1 Regime-Agnostic:    {'PASS' if r1_pass else 'FAIL'}")
    print(f"  Permutation Test:      {'PASS' if (p_value is not None and p_value < 0.05) else 'FAIL'}")
    print(f"  Sub-Period Stability:  {'PASS' if stable else 'FAIL'}")
    print(f"  Outlier Robustness:    {'PASS' if robust else 'FAIL'}")
    print(f"  OVERALL:               {'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'}")
    print(f"\n  Runtime: {time.time() - t0:.1f}s")
    print(f"  Results saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
