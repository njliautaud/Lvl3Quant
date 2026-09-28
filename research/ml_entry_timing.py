#!/usr/bin/env python3
"""
ML-Based Optimal Entry Timing Strategy
=======================================
Determines WHEN within a week/month is optimal to add equity exposure.
Uses LightGBM classifier on daily features to predict forward 5-day SPY returns.

Walk-forward: 504d train, 126d test, 63d slide (sliding window, HC #0).
Fixed capital $100K, no DCA (HC #713).
Adversarial validation: permutation test, sub-period, outlier robustness (HC #705).
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from datetime import datetime, timedelta
import calendar
from collections import defaultdict
import sys
import time

# ============================================================
# 1. DATA DOWNLOAD
# ============================================================

def download_data():
    """Download daily price data for all tickers."""
    tickers = ['SPY', 'QQQ', 'IWM', 'UPRO', '^VIX', 'TLT', 'GLD', 'HYG']
    print("Downloading data from 2010-present...")

    data = {}
    for t in tickers:
        tries = 3
        for attempt in range(tries):
            try:
                df = yf.download(t, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'),
                                 progress=False, auto_adjust=True)
                if len(df) > 100:
                    clean_name = t.replace('^', '')
                    data[clean_name] = df
                    print(f"  {t}: {len(df)} rows ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
                    break
                else:
                    print(f"  {t}: only {len(df)} rows, retrying...")
            except Exception as e:
                print(f"  {t} attempt {attempt+1} failed: {e}")
                time.sleep(2)

    # VIXY as VIX proxy ETF (if VIX download fails, we'll use it)
    if 'VIX' not in data:
        print("  VIX download failed, trying VIXY...")
        try:
            df = yf.download('VIXY', start='2010-01-01', progress=False, auto_adjust=True)
            if len(df) > 100:
                data['VIXY'] = df
                print(f"  VIXY: {len(df)} rows")
        except:
            pass

    return data


def get_opex_date(year, month):
    """Get monthly options expiration date (3rd Friday of the month)."""
    c = calendar.monthcalendar(year, month)
    # Find the 3rd Friday
    fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
    if len(fridays) >= 3:
        return pd.Timestamp(year=year, month=month, day=fridays[2])
    return pd.Timestamp(year=year, month=month, day=fridays[-1])


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def build_features(data):
    """Build all features from daily data. NO LOOK-AHEAD."""
    spy = data['SPY'].copy()

    # Flatten multi-level columns if present
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    df = pd.DataFrame(index=spy.index)
    df['close'] = spy['Close']
    df['high'] = spy['High']
    df['low'] = spy['Low']
    df['volume'] = spy['Volume']
    df['ret_1d'] = df['close'].pct_change()

    # --- Day of week (one-hot) ---
    for i in range(5):
        df[f'dow_{i}'] = (df.index.dayofweek == i).astype(int)

    # --- Day of month bucket ---
    dom = df.index.day
    df['dom_bucket'] = pd.cut(dom, bins=[0, 5, 10, 15, 20, 31], labels=[1,2,3,4,5]).astype(int)
    for b in range(1, 6):
        df[f'dom_b{b}'] = (df['dom_bucket'] == b).astype(int)

    # --- Distance from monthly OpEx (3rd Friday) ---
    opex_dates = []
    for year in range(df.index.year.min(), df.index.year.max() + 1):
        for month in range(1, 13):
            opex_dates.append(get_opex_date(year, month))
    opex_series = pd.Series(opex_dates).sort_values().reset_index(drop=True)

    dist_from_opex = []
    for dt in df.index:
        diffs = (opex_series - dt).dt.days
        # Nearest OpEx (could be past or future)
        abs_diffs = diffs.abs()
        nearest_idx = abs_diffs.idxmin()
        dist_from_opex.append(diffs.iloc[nearest_idx])
    df['dist_opex'] = dist_from_opex

    # --- RSI ---
    def rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    df['rsi_5'] = rsi(df['close'], 5)
    df['rsi_14'] = rsi(df['close'], 14)

    # --- SMA distances ---
    sma20 = df['close'].rolling(20).mean()
    sma50 = df['close'].rolling(50).mean()
    df['dist_sma20'] = (df['close'] - sma20) / sma20
    df['dist_sma50'] = (df['close'] - sma50) / sma50

    # --- Bollinger band position ---
    bb_std = df['close'].rolling(20).std()
    bb_upper = sma20 + 2 * bb_std
    bb_lower = sma20 - 2 * bb_std
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

    # --- Recent drawdown from 20d high ---
    rolling_high_20 = df['close'].rolling(20).max()
    df['drawdown_20d'] = (df['close'] - rolling_high_20) / rolling_high_20

    # --- VIX features ---
    if 'VIX' in data:
        vix_df = data['VIX'].copy()
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        vix_close = vix_df['Close'].reindex(df.index, method='ffill')
        df['vix_level'] = vix_close
        df['vix_5d_chg'] = vix_close.pct_change(5)
    elif 'VIXY' in data:
        vixy_df = data['VIXY'].copy()
        if isinstance(vixy_df.columns, pd.MultiIndex):
            vixy_df.columns = vixy_df.columns.get_level_values(0)
        vixy_close = vixy_df['Close'].reindex(df.index, method='ffill')
        df['vix_level'] = vixy_close
        df['vix_5d_chg'] = vixy_close.pct_change(5)

    # --- Intraday range ratio (smoothed) ---
    intraday_range = (df['high'] - df['low']) / df['close']
    df['range_ratio_5d'] = intraday_range.rolling(5).mean()

    # --- Volume ratio ---
    vol_20d = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / vol_20d

    # --- HYG/TLT ratio momentum (credit conditions) ---
    if 'HYG' in data and 'TLT' in data:
        hyg_df = data['HYG'].copy()
        tlt_df = data['TLT'].copy()
        if isinstance(hyg_df.columns, pd.MultiIndex):
            hyg_df.columns = hyg_df.columns.get_level_values(0)
        if isinstance(tlt_df.columns, pd.MultiIndex):
            tlt_df.columns = tlt_df.columns.get_level_values(0)
        hyg_close = hyg_df['Close'].reindex(df.index, method='ffill')
        tlt_close = tlt_df['Close'].reindex(df.index, method='ffill')
        credit_ratio = hyg_close / tlt_close
        df['credit_mom_5d'] = credit_ratio.pct_change(5)
        df['credit_mom_20d'] = credit_ratio.pct_change(20)

    # --- Prior returns ---
    df['ret_prior_1d'] = df['ret_1d'].shift(1)
    df['ret_prior_3d'] = df['close'].pct_change(3).shift(1)

    # --- QQQ / IWM relative strength ---
    for ticker in ['QQQ', 'IWM']:
        if ticker in data:
            t_df = data[ticker].copy()
            if isinstance(t_df.columns, pd.MultiIndex):
                t_df.columns = t_df.columns.get_level_values(0)
            t_close = t_df['Close'].reindex(df.index, method='ffill')
            df[f'{ticker.lower()}_rel_5d'] = t_close.pct_change(5) - df['close'].pct_change(5)

    # --- Target: forward 5-day return ---
    df['fwd_5d_ret'] = df['close'].pct_change(5).shift(-5)

    return df


# ============================================================
# 3. WALK-FORWARD LGBM
# ============================================================

def get_feature_cols(df):
    """Get feature column names."""
    exclude = ['close', 'high', 'low', 'volume', 'ret_1d', 'fwd_5d_ret', 'target', 'dom_bucket']
    return [c for c in df.columns if c not in exclude]


def walk_forward_lgbm(df):
    """
    Walk-forward with sliding window.
    504d train, 126d test, 63d slide.
    """
    feature_cols = get_feature_cols(df)

    # Drop rows with NaN features or target
    df_clean = df.dropna(subset=feature_cols + ['fwd_5d_ret']).copy()

    # Binary target: above median forward return = 1 (good entry)
    median_ret = df_clean['fwd_5d_ret'].median()
    df_clean['target'] = (df_clean['fwd_5d_ret'] > median_ret).astype(int)

    print(f"\nClean data: {len(df_clean)} rows, {len(feature_cols)} features")
    print(f"Median fwd 5d return: {median_ret*100:.3f}%")
    print(f"Target balance: {df_clean['target'].mean():.3f}")

    TRAIN_LEN = 504
    TEST_LEN = 126
    SLIDE = 63

    all_preds = []
    all_actuals = []
    all_dates = []
    all_probs = []
    fold_metrics = []
    feature_importance_accum = defaultdict(float)

    start = 0
    fold = 0

    while start + TRAIN_LEN + TEST_LEN <= len(df_clean):
        train_idx = df_clean.index[start:start + TRAIN_LEN]
        test_idx = df_clean.index[start + TRAIN_LEN:start + TRAIN_LEN + TEST_LEN]

        X_train = df_clean.loc[train_idx, feature_cols].values
        y_train = df_clean.loc[train_idx, 'target'].values
        X_test = df_clean.loc[test_idx, feature_cols].values
        y_test = df_clean.loc[test_idx, 'target'].values

        # LightGBM
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        valid_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_cols, reference=train_data)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 5,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbose': -1,
            'seed': 42,
            'n_jobs': -1,
        }

        callbacks = [lgb.log_evaluation(0), lgb.early_stopping(30)]
        model = lgb.train(params, train_data, num_boost_round=500,
                          valid_sets=[valid_data], callbacks=callbacks)

        probs = model.predict(X_test)
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, probs)
        except:
            auc = 0.5

        fold_metrics.append({
            'fold': fold,
            'train_start': train_idx[0].strftime('%Y-%m-%d'),
            'train_end': train_idx[-1].strftime('%Y-%m-%d'),
            'test_start': test_idx[0].strftime('%Y-%m-%d'),
            'test_end': test_idx[-1].strftime('%Y-%m-%d'),
            'acc': acc,
            'auc': auc,
            'n_test': len(y_test),
        })

        all_preds.extend(preds)
        all_actuals.extend(y_test)
        all_dates.extend(test_idx)
        all_probs.extend(probs)

        # Accumulate feature importance
        imp = model.feature_importance(importance_type='gain')
        for fname, fval in zip(feature_cols, imp):
            feature_importance_accum[fname] += fval

        fold += 1
        start += SLIDE

    print(f"\nCompleted {fold} walk-forward folds")

    # Build results DataFrame
    results = pd.DataFrame({
        'date': all_dates,
        'pred': all_preds,
        'actual': all_actuals,
        'prob': all_probs,
    })
    # Remove duplicate dates (overlapping test windows) — keep first occurrence
    results = results.drop_duplicates(subset='date', keep='first')
    results.set_index('date', inplace=True)

    # Average feature importance
    for k in feature_importance_accum:
        feature_importance_accum[k] /= fold

    return results, fold_metrics, feature_importance_accum, feature_cols, median_ret, df_clean


# ============================================================
# 4. STRATEGY BACKTEST
# ============================================================

def backtest_strategy(results, df_full, capital=100_000):
    """
    Backtest the entry timing strategy.
    Good entry (pred=1) → long SPY
    Bad entry (pred=0) → cash
    Compare vs buy-and-hold.
    Fixed capital, no DCA.
    """
    # Merge predictions with actual forward returns
    bt = results.copy()
    bt['fwd_5d_ret'] = df_full.loc[bt.index, 'fwd_5d_ret']
    bt = bt.dropna(subset=['fwd_5d_ret'])

    # Daily returns for strategy
    # When model says "good entry", we capture the fwd_5d_ret spread over 5 days
    # Simplified: use the daily returns
    daily_rets = df_full.loc[bt.index, 'ret_1d'].fillna(0)

    # Strategy: if yesterday's prediction was "good entry", be long today
    bt['signal'] = bt['pred'].shift(1).fillna(0)  # trade next day

    # Strategy returns
    bt['strat_ret'] = bt['signal'] * daily_rets
    bt['bh_ret'] = daily_rets

    # Cumulative
    bt['strat_cum'] = (1 + bt['strat_ret']).cumprod() * capital
    bt['bh_cum'] = (1 + bt['bh_ret']).cumprod() * capital

    # Also do UPRO version (3x leveraged)
    upro_daily = df_full['ret_1d'] * 3  # approximate UPRO as 3x SPY daily
    upro_rets = upro_daily.reindex(bt.index).fillna(0)
    bt['upro_strat_ret'] = bt['signal'] * upro_rets
    bt['upro_bh_ret'] = upro_rets
    bt['upro_strat_cum'] = (1 + bt['upro_strat_ret']).cumprod() * capital
    bt['upro_bh_cum'] = (1 + bt['upro_bh_ret']).cumprod() * capital

    # 50% exposure version: bad entry → 50% instead of 0%
    bt['signal_50'] = bt['pred'].shift(1).fillna(0).apply(lambda x: 1.0 if x == 1 else 0.5)
    bt['strat_50_ret'] = bt['signal_50'] * daily_rets
    bt['strat_50_cum'] = (1 + bt['strat_50_ret']).cumprod() * capital

    return bt


def compute_metrics(returns, annual_factor=252):
    """Compute risk-adjusted metrics from a return series."""
    returns = returns.dropna()
    if len(returns) < 20:
        return {}

    ann_ret = returns.mean() * annual_factor
    ann_vol = returns.std() * np.sqrt(annual_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(annual_factor)
    sortino = ann_ret / downside if downside > 0 else 0

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    # Win rate (daily)
    wr = (returns > 0).mean()

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        'Ann Return': f"{ann_ret*100:.2f}%",
        'Ann Vol': f"{ann_vol*100:.2f}%",
        'Sharpe': f"{sharpe:.3f}",
        'Sortino': f"{sortino:.3f}",
        'Max DD': f"{max_dd*100:.2f}%",
        'Calmar': f"{calmar:.3f}",
        'Win Rate': f"{wr*100:.1f}%",
        'Profit Factor': f"{pf:.3f}",
        'Exposure': f"{(returns != 0).mean()*100:.1f}%",
    }


# ============================================================
# 5. ADVERSARIAL VALIDATION (HC #705)
# ============================================================

def adversarial_tests(results, df_clean, feature_cols, median_ret):
    """
    1. Permutation test (1000 shuffles)
    2. Sub-period analysis (3 blocks)
    3. Outlier robustness (5% trim)
    """
    print("\n" + "="*60)
    print("ADVERSARIAL VALIDATION (HC #705)")
    print("="*60)

    # --- 1. Permutation test ---
    print("\n--- Permutation Test (1000 shuffles) ---")
    actual_acc = accuracy_score(results['actual'], results['pred'])
    actual_auc = roc_auc_score(results['actual'], results['prob'])

    perm_accs = []
    perm_aucs = []
    rng = np.random.RandomState(42)
    for i in range(1000):
        shuffled = rng.permutation(results['actual'].values)
        perm_accs.append(accuracy_score(shuffled, results['pred'].values))
        try:
            perm_aucs.append(roc_auc_score(shuffled, results['prob'].values))
        except:
            perm_aucs.append(0.5)

    p_value_acc = np.mean([pa >= actual_acc for pa in perm_accs])
    p_value_auc = np.mean([pa >= actual_auc for pa in perm_aucs])

    print(f"  Actual Accuracy: {actual_acc:.4f}  |  Permutation mean: {np.mean(perm_accs):.4f}  |  p-value: {p_value_acc:.4f}")
    print(f"  Actual AUC:      {actual_auc:.4f}  |  Permutation mean: {np.mean(perm_aucs):.4f}  |  p-value: {p_value_auc:.4f}")

    if p_value_auc < 0.05:
        print("  >>> PASS: AUC is statistically significant (p < 0.05)")
    else:
        print("  >>> FAIL: AUC is NOT statistically significant")

    # --- 2. Sub-period analysis (3 blocks) ---
    print("\n--- Sub-Period Analysis (3 blocks) ---")
    n = len(results)
    block_size = n // 3

    for i in range(3):
        start_i = i * block_size
        end_i = (i + 1) * block_size if i < 2 else n
        block = results.iloc[start_i:end_i]

        block_acc = accuracy_score(block['actual'], block['pred'])
        try:
            block_auc = roc_auc_score(block['actual'], block['prob'])
        except:
            block_auc = 0.5

        date_range = f"{block.index[0].strftime('%Y-%m-%d')} to {block.index[-1].strftime('%Y-%m-%d')}"
        print(f"  Block {i+1} ({date_range}): Acc={block_acc:.4f}, AUC={block_auc:.4f}, N={len(block)}")

    # --- 3. Outlier robustness (5% trim) ---
    print("\n--- Outlier Robustness (5% trim) ---")
    # Trim days where fwd return was in top/bottom 5%
    fwd_rets = results.join(df_clean[['fwd_5d_ret']], how='left')['fwd_5d_ret']
    q_low = fwd_rets.quantile(0.05)
    q_high = fwd_rets.quantile(0.95)
    mask = (fwd_rets >= q_low) & (fwd_rets <= q_high)
    trimmed = results[mask]

    trim_acc = accuracy_score(trimmed['actual'], trimmed['pred'])
    try:
        trim_auc = roc_auc_score(trimmed['actual'], trimmed['prob'])
    except:
        trim_auc = 0.5

    print(f"  Full dataset:  Acc={actual_acc:.4f}, AUC={actual_auc:.4f}, N={len(results)}")
    print(f"  5% trimmed:    Acc={trim_acc:.4f}, AUC={trim_auc:.4f}, N={len(trimmed)}")

    acc_drop = actual_acc - trim_acc
    auc_drop = actual_auc - trim_auc
    print(f"  Acc change: {acc_drop:+.4f}  |  AUC change: {auc_drop:+.4f}")

    if abs(auc_drop) < 0.02:
        print("  >>> PASS: Robust to outlier removal")
    else:
        print("  >>> CAUTION: Performance changes meaningfully with outlier removal")

    return {
        'perm_p_value_acc': p_value_acc,
        'perm_p_value_auc': p_value_auc,
        'trimmed_auc': trim_auc,
    }


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("="*70)
    print("ML ENTRY TIMING STRATEGY")
    print("LightGBM classifier — Walk-Forward — Fixed Capital $100K")
    print("="*70)

    # Download data
    data = download_data()

    if 'SPY' not in data:
        print("FATAL: Could not download SPY data")
        sys.exit(1)

    # Build features
    print("\nBuilding features...")
    df = build_features(data)
    print(f"Total rows: {len(df)}")

    # Walk-forward
    print("\n" + "="*60)
    print("WALK-FORWARD LGBM (504d train, 126d test, 63d slide)")
    print("="*60)

    results, fold_metrics, feat_imp, feature_cols, median_ret, df_clean = walk_forward_lgbm(df)

    # Overall OOS metrics
    print("\n" + "="*60)
    print("OUT-OF-SAMPLE PREDICTION METRICS (all folds concat)")
    print("="*60)

    oos_acc = accuracy_score(results['actual'], results['pred'])
    oos_auc = roc_auc_score(results['actual'], results['prob'])
    print(f"  Accuracy:  {oos_acc:.4f}  (baseline: 0.500)")
    print(f"  AUC:       {oos_auc:.4f}  (baseline: 0.500)")
    print(f"  N samples: {len(results)}")
    print(f"  Date range: {results.index[0].strftime('%Y-%m-%d')} to {results.index[-1].strftime('%Y-%m-%d')}")

    # Per-fold summary
    print("\n  Per-fold AUC range: {:.4f} - {:.4f}".format(
        min(f['auc'] for f in fold_metrics),
        max(f['auc'] for f in fold_metrics)))
    print("  Per-fold Acc range: {:.4f} - {:.4f}".format(
        min(f['acc'] for f in fold_metrics),
        max(f['acc'] for f in fold_metrics)))

    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCES (avg gain across folds)")
    print("="*60)
    sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    for i, (fname, fval) in enumerate(sorted_imp[:20]):
        bar = "█" * int(fval / sorted_imp[0][1] * 30)
        print(f"  {i+1:2d}. {fname:25s} {fval:10.1f}  {bar}")

    # Backtest
    print("\n" + "="*60)
    print("STRATEGY BACKTEST ($100K fixed capital)")
    print("="*60)

    bt = backtest_strategy(results, df)

    print("\n--- SPY Timing (100% long when good entry, 0% when bad) ---")
    strat_metrics = compute_metrics(bt['strat_ret'])
    bh_metrics = compute_metrics(bt['bh_ret'])

    header = f"{'Metric':20s} {'Strategy':>12s} {'Buy&Hold':>12s}"
    print(f"  {header}")
    print(f"  {'-'*46}")
    for key in strat_metrics:
        print(f"  {key:20s} {strat_metrics[key]:>12s} {bh_metrics.get(key, 'N/A'):>12s}")

    final_strat = bt['strat_cum'].iloc[-1]
    final_bh = bt['bh_cum'].iloc[-1]
    print(f"\n  Final equity:  Strategy=${final_strat:,.0f}  |  Buy&Hold=${final_bh:,.0f}")

    print("\n--- SPY Timing (100% when good, 50% when bad) ---")
    strat50_metrics = compute_metrics(bt['strat_50_ret'])
    header = f"{'Metric':20s} {'Strat 50%':>12s} {'Buy&Hold':>12s}"
    print(f"  {header}")
    print(f"  {'-'*46}")
    for key in strat50_metrics:
        print(f"  {key:20s} {strat50_metrics[key]:>12s} {bh_metrics.get(key, 'N/A'):>12s}")

    final_50 = bt['strat_50_cum'].iloc[-1]
    print(f"\n  Final equity:  Strategy50=${final_50:,.0f}  |  Buy&Hold=${final_bh:,.0f}")

    print("\n--- UPRO (3x) Timing ---")
    upro_strat_metrics = compute_metrics(bt['upro_strat_ret'])
    upro_bh_metrics = compute_metrics(bt['upro_bh_ret'])
    header = f"{'Metric':20s} {'UPRO Timing':>12s} {'UPRO B&H':>12s}"
    print(f"  {header}")
    print(f"  {'-'*46}")
    for key in upro_strat_metrics:
        print(f"  {key:20s} {upro_strat_metrics[key]:>12s} {upro_bh_metrics.get(key, 'N/A'):>12s}")

    final_upro_strat = bt['upro_strat_cum'].iloc[-1]
    final_upro_bh = bt['upro_bh_cum'].iloc[-1]
    print(f"\n  Final equity:  UPRO Timing=${final_upro_strat:,.0f}  |  UPRO B&H=${final_upro_bh:,.0f}")

    # Adversarial
    adv_results = adversarial_tests(results, df_clean, feature_cols, median_ret)

    # Year-by-year breakdown
    print("\n" + "="*60)
    print("YEAR-BY-YEAR BREAKDOWN")
    print("="*60)
    bt['year'] = bt.index.year
    print(f"\n  {'Year':>6s} {'Strat Ann%':>12s} {'B&H Ann%':>12s} {'Strat Sharpe':>14s} {'B&H Sharpe':>12s} {'Exposure%':>10s}")
    print(f"  {'-'*68}")
    for year, grp in bt.groupby('year'):
        if len(grp) < 20:
            continue
        s_ann = grp['strat_ret'].mean() * 252 * 100
        b_ann = grp['bh_ret'].mean() * 252 * 100
        s_vol = grp['strat_ret'].std() * np.sqrt(252)
        b_vol = grp['bh_ret'].std() * np.sqrt(252)
        s_sharpe = (grp['strat_ret'].mean() * 252) / s_vol if s_vol > 0 else 0
        b_sharpe = (grp['bh_ret'].mean() * 252) / b_vol if b_vol > 0 else 0
        exposure = grp['signal'].mean() * 100
        print(f"  {year:>6d} {s_ann:>11.1f}% {b_ann:>11.1f}% {s_sharpe:>14.3f} {b_sharpe:>12.3f} {exposure:>9.1f}%")

    # Signal analysis
    print("\n" + "="*60)
    print("SIGNAL ANALYSIS")
    print("="*60)

    # When model says "good entry", what's the avg fwd return?
    merged = results.join(df[['fwd_5d_ret']], how='left').dropna()
    good_entry = merged[merged['pred'] == 1]['fwd_5d_ret']
    bad_entry = merged[merged['pred'] == 0]['fwd_5d_ret']

    print(f"\n  Good entry days (pred=1): N={len(good_entry)}, avg 5d fwd ret = {good_entry.mean()*100:.3f}%")
    print(f"  Bad entry days  (pred=0): N={len(bad_entry)},  avg 5d fwd ret = {bad_entry.mean()*100:.3f}%")
    print(f"  Spread: {(good_entry.mean() - bad_entry.mean())*100:.3f}%")
    print(f"  Good entry median: {good_entry.median()*100:.3f}%  |  Bad entry median: {bad_entry.median()*100:.3f}%")

    # Confidence buckets
    print("\n  By prediction confidence:")
    for lo, hi, label in [(0.0, 0.3, 'Strong bad (<0.3)'),
                           (0.3, 0.45, 'Mild bad (0.3-0.45)'),
                           (0.45, 0.55, 'Neutral (0.45-0.55)'),
                           (0.55, 0.7, 'Mild good (0.55-0.7)'),
                           (0.7, 1.0, 'Strong good (>0.7)')]:
        mask = (merged['prob'] >= lo) & (merged['prob'] < hi)
        subset = merged[mask]
        if len(subset) > 0:
            avg_ret = subset['fwd_5d_ret'].mean() * 100
            print(f"    {label:25s}  N={len(subset):5d}  avg 5d ret = {avg_ret:+.3f}%")

    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)

    is_significant = adv_results['perm_p_value_auc'] < 0.05
    has_edge = (good_entry.mean() - bad_entry.mean()) > 0
    beats_bh_sharpe = float(strat_metrics['Sharpe']) > float(bh_metrics['Sharpe'])

    verdicts = []
    if is_significant:
        verdicts.append("PASS: Statistically significant (permutation test)")
    else:
        verdicts.append("FAIL: NOT statistically significant")

    if has_edge:
        verdicts.append(f"PASS: Good-entry days have higher fwd returns ({(good_entry.mean()-bad_entry.mean())*100:.3f}% spread)")
    else:
        verdicts.append("FAIL: No return spread between good/bad entry predictions")

    if beats_bh_sharpe:
        verdicts.append(f"PASS: Strategy Sharpe ({strat_metrics['Sharpe']}) > Buy&Hold Sharpe ({bh_metrics['Sharpe']})")
    else:
        verdicts.append(f"FAIL: Strategy Sharpe ({strat_metrics['Sharpe']}) <= Buy&Hold Sharpe ({bh_metrics['Sharpe']})")

    for v in verdicts:
        print(f"  {v}")

    all_pass = is_significant and has_edge and beats_bh_sharpe
    print(f"\n  OVERALL: {'DEPLOYABLE' if all_pass else 'NEEDS MORE WORK'}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
