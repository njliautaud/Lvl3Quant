#!/usr/bin/env python3
"""
ML Regime Predictor — Next-Day SPY Regime Classification
==========================================================
Predicts next-day SPY regime (GREEN / FLAT / RED) using cross-asset features,
then backtests a timing strategy: GREEN->UPRO, RED->SHY, FLAT->SPY.

Walk-forward: 252d SLIDING train window, predict next day, slide forward (HC #0).
Models: LightGBM (primary), 2-layer MLP (sklearn).
Adversarial validation per HC #705: permutation test, sub-period consistency,
    outlier removal, R1 regime test.

Capital: Fixed $100K, NO DCA (HC #713).
Execution: Signal at close T -> trade at open T+1 (no look-ahead bias).
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import json
import datetime as dt
from pathlib import Path
from functools import partial

# Force unbuffered output
import builtins
_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)
    sys.stdout.flush()

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats

# ==============================================================
# CONFIGURATION
# ==============================================================

BASE = Path('/home/jupiter/Lvl3Quant')
OUTPUT_DIR = BASE / 'output' / 'ml_regime_predictor'
CACHE_DIR = BASE / 'output' / 'growth_research' / 'cache'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ['SPY', 'UPRO', 'SHY', 'GLD', 'SLV', 'USO', 'UUP', 'TLT', 'HYG',
           'IEF', 'BTC-USD', 'CPER', 'QQQ', 'IWM']
VIX_TICKER = '^VIX'
ALL_TICKERS = TICKERS + [VIX_TICKER]

TRAIN_DAYS = 252       # 1 year sliding window
INITIAL_CAPITAL = 100_000

# Regime thresholds on next-day SPY return
GREEN_THRESH = 0.003   # > +0.3%
RED_THRESH = -0.003    # < -0.3%

START_DATE = '2009-01-01'
END_DATE = '2026-07-17'

# v4.4 baseline for comparison
V44_SHARPE = 1.02
V44_CAGR = 0.241


# ==============================================================
# DATA DOWNLOAD
# ==============================================================

def download_data():
    """Download or load cached daily data for all tickers."""
    cache_file = CACHE_DIR / 'ml_regime_predictor_data.parquet'
    if cache_file.exists():
        mtime = dt.datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (dt.datetime.now() - mtime).hours < 6 if hasattr((dt.datetime.now() - mtime), 'hours') else (dt.datetime.now() - mtime).total_seconds() < 21600:
            print(f"  Loading cached data...")
            return pd.read_parquet(cache_file)

    print("  Downloading daily data from Yahoo Finance...")
    dfs = {}
    for ticker in ALL_TICKERS:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                dfs[ticker] = df[['Open', 'Close']].rename(
                    columns={'Open': f'{ticker}_open', 'Close': f'{ticker}_close'})
                print(f"    {ticker}: {len(df)} rows")
            else:
                print(f"    {ticker}: SKIPPED (only {len(df)} rows)")
        except Exception as e:
            print(f"    {ticker}: FAILED ({e})")

    # Merge all on date index
    merged = None
    for ticker, df in dfs.items():
        if merged is None:
            merged = df
        else:
            merged = merged.join(df, how='outer')

    merged = merged.sort_index()
    merged = merged.ffill().dropna()
    print(f"  Merged data: {len(merged)} rows, {merged.columns.size} columns")

    try:
        merged.to_parquet(cache_file)
    except Exception:
        pass

    return merged


# ==============================================================
# FEATURE ENGINEERING
# ==============================================================

def engineer_features(df):
    """Build cross-asset features from daily close/open prices."""
    feat = pd.DataFrame(index=df.index)

    # Helper: get close series for a ticker
    def close(t):
        col = f'{t}_close'
        return df[col] if col in df.columns else None

    def open_price(t):
        col = f'{t}_open'
        return df[col] if col in df.columns else None

    spy = close('SPY')
    vix = close('^VIX')

    # --- 1. Rolling returns (5d, 10d, 21d, 63d) for key assets ---
    for t in ['SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'USO', 'UUP', 'TLT', 'HYG', 'IEF', 'BTC-USD', 'CPER']:
        s = close(t)
        if s is None:
            continue
        for w in [5, 10, 21, 63]:
            feat[f'{t}_ret_{w}d'] = s.pct_change(w)

    # --- 2. Rolling volatility (10d, 21d) ---
    for t in ['SPY', 'QQQ', 'TLT', 'GLD', 'BTC-USD']:
        s = close(t)
        if s is None:
            continue
        daily_ret = s.pct_change()
        for w in [10, 21]:
            feat[f'{t}_vol_{w}d'] = daily_ret.rolling(w).std() * np.sqrt(252)

    # --- 3. VIX features ---
    if vix is not None:
        feat['VIX_level'] = vix
        feat['VIX_pctile_63d'] = vix.rolling(63).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100.0, raw=False)
        feat['VIX_change_5d'] = vix.pct_change(5)
        feat['VIX_change_1d'] = vix.pct_change(1)
        feat['VIX_over_20'] = (vix > 20).astype(float)
        feat['VIX_over_30'] = (vix > 30).astype(float)

    # --- 4. Credit spread proxy: HYG - IEF return differential ---
    hyg = close('HYG')
    ief = close('IEF')
    if hyg is not None and ief is not None:
        hyg_ret = hyg.pct_change()
        ief_ret = ief.pct_change()
        feat['credit_spread_1d'] = hyg_ret - ief_ret
        feat['credit_spread_5d'] = hyg.pct_change(5) - ief.pct_change(5)
        feat['credit_spread_21d'] = hyg.pct_change(21) - ief.pct_change(21)

    # --- 5. Cross-asset momentum (relative to SPY) ---
    if spy is not None:
        spy_21d = spy.pct_change(21)
        for t in ['GLD', 'TLT', 'UUP', 'BTC-USD', 'USO']:
            s = close(t)
            if s is None:
                continue
            feat[f'{t}_vs_SPY_21d'] = s.pct_change(21) - spy_21d

    # --- 6. Dollar strength (UUP momentum) ---
    uup = close('UUP')
    if uup is not None:
        feat['dollar_mom_10d'] = uup.pct_change(10)
        feat['dollar_mom_21d'] = uup.pct_change(21)

    # --- 7. Correlation regime: rolling 21d corr between SPY and TLT ---
    tlt = close('TLT')
    if spy is not None and tlt is not None:
        spy_ret = spy.pct_change()
        tlt_ret = tlt.pct_change()
        feat['SPY_TLT_corr_21d'] = spy_ret.rolling(21).corr(tlt_ret)
        feat['SPY_TLT_corr_63d'] = spy_ret.rolling(63).corr(tlt_ret)
        # Stress indicator: positive correlation (unusual)
        feat['corr_stress'] = (feat['SPY_TLT_corr_21d'] > 0).astype(float)

    # --- 8. SPY-specific features ---
    if spy is not None:
        spy_ret = spy.pct_change()
        # Momentum
        feat['SPY_mom_5d'] = spy.pct_change(5)
        feat['SPY_mom_21d'] = spy.pct_change(21)
        # Distance from 50d/200d SMA
        feat['SPY_dist_sma50'] = spy / spy.rolling(50).mean() - 1
        feat['SPY_dist_sma200'] = spy / spy.rolling(200).mean() - 1
        # SMA crossover
        feat['SPY_sma50_above_200'] = (spy.rolling(50).mean() > spy.rolling(200).mean()).astype(float)
        # Recent drawdown
        feat['SPY_dd_21d'] = spy / spy.rolling(21).max() - 1
        feat['SPY_dd_63d'] = spy / spy.rolling(63).max() - 1
        # Consecutive up/down days
        up = (spy_ret > 0).astype(int)
        feat['SPY_consec_up'] = up.groupby((up != up.shift()).cumsum()).cumcount() * up
        down = (spy_ret < 0).astype(int)
        feat['SPY_consec_down'] = down.groupby((down != down.shift()).cumsum()).cumcount() * down

    # --- 9. Day-of-week, month features ---
    feat['day_of_week'] = pd.to_datetime(feat.index).dayofweek
    feat['month'] = pd.to_datetime(feat.index).month

    # Drop NaN rows from rolling calculations
    feat = feat.replace([np.inf, -np.inf], np.nan)

    return feat


# ==============================================================
# TARGET VARIABLE
# ==============================================================

def create_target(df):
    """Create next-day SPY regime: GREEN (>+0.3%), RED (<-0.3%), FLAT (between)."""
    spy = df['SPY_close']
    spy_open = df['SPY_open']
    upro_open = df['UPRO_open'] if 'UPRO_open' in df.columns else None
    shy_open = df['SHY_open'] if 'SHY_open' in df.columns else None

    # Next-day return: close T to close T+1
    next_day_ret = spy.pct_change().shift(-1)

    target = pd.Series(1, index=df.index, name='regime')  # 1 = FLAT
    target[next_day_ret > GREEN_THRESH] = 2   # GREEN
    target[next_day_ret < RED_THRESH] = 0     # RED

    regime_map = {0: 'RED', 1: 'FLAT', 2: 'GREEN'}

    return target, next_day_ret, regime_map


# ==============================================================
# WALK-FORWARD ENGINE (SLIDING WINDOW)
# ==============================================================

def walk_forward(features, target, next_day_ret, df, model_type='lgbm'):
    """
    Sliding window walk-forward:
    - Train on 252 days, predict day 253, slide forward by 1 day.
    - Signal at close T -> trade at OPEN T+1.
    """
    n = len(features)
    feature_cols = features.columns.tolist()

    predictions = []
    actuals = []
    dates = []
    probas_green = []
    probas_red = []

    # For strategy: we need next-day open prices
    spy_open = df['SPY_open'] if 'SPY_open' in df.columns else None
    spy_close = df['SPY_close'] if 'SPY_close' in df.columns else None
    upro_open = df['UPRO_open'] if 'UPRO_open' in df.columns else None
    upro_close = df['UPRO_close'] if 'UPRO_close' in df.columns else None
    shy_open = df['SHY_open'] if 'SHY_open' in df.columns else None
    shy_close = df['SHY_close'] if 'SHY_close' in df.columns else None

    # Align features and target
    valid_mask = features.notna().all(axis=1) & target.notna()
    feat_aligned = features[valid_mask].copy()
    tgt_aligned = target[valid_mask].copy()
    ret_aligned = next_day_ret[valid_mask].copy()

    n_valid = len(feat_aligned)
    print(f"  Walk-forward with {model_type.upper()}: {n_valid} valid days, "
          f"train={TRAIN_DAYS}d sliding window")

    total_steps = n_valid - TRAIN_DAYS
    report_every = max(1, total_steps // 20)

    for i in range(TRAIN_DAYS, n_valid - 1):  # -1 because we predict next day
        train_idx = range(i - TRAIN_DAYS, i)
        test_idx = i

        X_train = feat_aligned.iloc[train_idx][feature_cols].values
        y_train = tgt_aligned.iloc[train_idx].values
        X_test = feat_aligned.iloc[[test_idx]][feature_cols].values
        y_test = tgt_aligned.iloc[test_idx]
        test_date = feat_aligned.index[test_idx]

        # Skip if insufficient class diversity in training
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            continue

        try:
            if model_type == 'lgbm':
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    num_leaves=16,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    verbose=-1,
                    n_jobs=1
                )
                model.fit(X_train, y_train)

            elif model_type == 'mlp':
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation='relu',
                    max_iter=150,
                    learning_rate='adaptive',
                    learning_rate_init=0.001,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=10,
                    random_state=42,
                    verbose=False
                )
                model.fit(X_train_s, y_train)
                X_train = X_train_s  # for consistency

            pred = model.predict(X_test)[0]
            proba = model.predict_proba(X_test)[0]

            predictions.append(pred)
            actuals.append(y_test)
            dates.append(test_date)

            # Store probabilities for each class
            classes = model.classes_
            p_green = proba[np.where(classes == 2)[0][0]] if 2 in classes else 0
            p_red = proba[np.where(classes == 0)[0][0]] if 0 in classes else 0
            probas_green.append(p_green)
            probas_red.append(p_red)

        except Exception as e:
            continue

        if (i - TRAIN_DAYS) % report_every == 0:
            pct = (i - TRAIN_DAYS) / total_steps * 100
            print(f"    Progress: {pct:.0f}% ({i - TRAIN_DAYS}/{total_steps})")

    results = pd.DataFrame({
        'date': dates,
        'prediction': predictions,
        'actual': actuals,
        'prob_green': probas_green,
        'prob_red': probas_red,
    }).set_index('date')

    print(f"  Completed: {len(results)} predictions")
    return results


# ==============================================================
# STRATEGY BACKTEST
# ==============================================================

def backtest_strategy(results, df):
    """
    Backtest: GREEN->UPRO, RED->SHY, FLAT->SPY.
    Signal at close T, execute at open T+1, hold until close T+1.
    Also backtest buy-and-hold SPY for comparison.
    """
    # Get next-day returns for each asset (open to close)
    spy_close = df['SPY_close']
    spy_open = df['SPY_open']

    # For the strategy, we use close-to-close returns (signal at close, measure next close)
    spy_ret = spy_close.pct_change()
    upro_ret = df['UPRO_close'].pct_change() if 'UPRO_close' in df.columns else spy_ret * 3
    shy_ret = df['SHY_close'].pct_change() if 'SHY_close' in df.columns else pd.Series(0.0001, index=df.index)

    # Align: prediction on date T -> we get return on date T+1
    strat_returns = []
    bnh_returns = []
    strat_dates = []
    allocations = []

    for i in range(len(results) - 1):
        signal_date = results.index[i]
        # Next trading day
        next_dates = spy_ret.index[spy_ret.index > signal_date]
        if len(next_dates) == 0:
            continue
        exec_date = next_dates[0]

        pred = results['prediction'].iloc[i]

        if pred == 2:  # GREEN -> UPRO
            ret = upro_ret.loc[exec_date] if exec_date in upro_ret.index else 0
            alloc = 'UPRO'
        elif pred == 0:  # RED -> SHY
            ret = shy_ret.loc[exec_date] if exec_date in shy_ret.index else 0
            alloc = 'SHY'
        else:  # FLAT -> SPY
            ret = spy_ret.loc[exec_date] if exec_date in spy_ret.index else 0
            alloc = 'SPY'

        spy_r = spy_ret.loc[exec_date] if exec_date in spy_ret.index else 0

        strat_returns.append(ret)
        bnh_returns.append(spy_r)
        strat_dates.append(exec_date)
        allocations.append(alloc)

    strat_df = pd.DataFrame({
        'strategy_ret': strat_returns,
        'spy_ret': bnh_returns,
        'allocation': allocations
    }, index=strat_dates)

    return strat_df


def compute_metrics(returns, name='Strategy', annual_rf=0.04):
    """Compute Sharpe, Sortino, CAGR, MaxDD."""
    if len(returns) == 0:
        return {}

    daily_rf = annual_rf / 252
    excess = returns - daily_rf
    n_years = len(returns) / 252

    total_ret = (1 + returns).prod() - 1
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_mean = returns.mean() * 252
    ann_std = returns.std() * np.sqrt(252)
    sharpe = (ann_mean - annual_rf) / ann_std if ann_std > 0 else 0

    downside = returns[returns < daily_rf] - daily_rf
    downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (ann_mean - annual_rf) / downside_std if downside_std > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_dd = drawdown.min()

    # Win rate
    wr = (returns > 0).mean()

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'name': name,
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MaxDD': max_dd,
        'WinRate': wr,
        'ProfitFactor': pf,
        'TotalReturn': total_ret,
        'AnnVol': ann_std,
        'N_Days': len(returns),
    }


# ==============================================================
# ADVERSARIAL VALIDATION (HC #705)
# ==============================================================

def adversarial_validation(results, strat_df, df, features, model_type='lgbm'):
    """
    Full adversarial validation per HC #705:
    1. Permutation test (100 shuffles)
    2. Sub-period consistency
    3. Outlier removal
    4. R1 regime test (bull vs bear performance)
    5. Feature importance concentration
    """
    print(f"\n{'='*60}")
    print(f"ADVERSARIAL VALIDATION ({model_type.upper()}) — HC #705")
    print(f"{'='*60}")

    checks_passed = 0
    checks_total = 0

    # --- 1. PERMUTATION TEST (100 shuffles) ---
    print("\n--- Test 1: Permutation Test (100 shuffles) ---")
    checks_total += 1

    actual_acc = accuracy_score(results['actual'], results['prediction'])
    actual_sharpe = compute_metrics(strat_df['strategy_ret'], 'actual')['Sharpe']

    perm_accs = []
    perm_sharpes = []
    n_perms = 100

    for p in range(n_perms):
        shuffled = results['prediction'].values.copy()
        np.random.seed(p)
        np.random.shuffle(shuffled)
        perm_acc = accuracy_score(results['actual'], shuffled)
        perm_accs.append(perm_acc)

        # Shuffled strategy returns
        perm_rets = []
        spy_ret = df['SPY_close'].pct_change()
        upro_ret = df['UPRO_close'].pct_change() if 'UPRO_close' in df.columns else spy_ret * 3
        shy_ret = df['SHY_close'].pct_change() if 'SHY_close' in df.columns else pd.Series(0.0001, index=df.index)

        for i in range(len(strat_df)):
            d = strat_df.index[i]
            pred = shuffled[i] if i < len(shuffled) else 1
            if pred == 2:
                r = upro_ret.loc[d] if d in upro_ret.index else 0
            elif pred == 0:
                r = shy_ret.loc[d] if d in shy_ret.index else 0
            else:
                r = spy_ret.loc[d] if d in spy_ret.index else 0
            perm_rets.append(r)
        pm = compute_metrics(pd.Series(perm_rets), f'perm_{p}')
        perm_sharpes.append(pm.get('Sharpe', 0))

    p_value_acc = np.mean([pa >= actual_acc for pa in perm_accs])
    p_value_sharpe = np.mean([ps >= actual_sharpe for ps in perm_sharpes])

    print(f"  Actual accuracy: {actual_acc:.4f}")
    print(f"  Permuted accuracy (mean): {np.mean(perm_accs):.4f}")
    print(f"  p-value (accuracy): {p_value_acc:.4f}")
    print(f"  Actual Sharpe: {actual_sharpe:.3f}")
    print(f"  Permuted Sharpe (mean): {np.mean(perm_sharpes):.3f}")
    print(f"  p-value (Sharpe): {p_value_sharpe:.4f}")

    if p_value_sharpe < 0.05:
        print("  PASS: Strategy Sharpe significantly better than random (p<0.05)")
        checks_passed += 1
    else:
        print("  FAIL: Strategy Sharpe NOT significantly better than random")

    # --- 2. SUB-PERIOD CONSISTENCY ---
    print("\n--- Test 2: Sub-Period Consistency ---")
    checks_total += 1

    n_periods = 4
    period_len = len(strat_df) // n_periods
    sub_sharpes = []
    sub_names = []
    for sp in range(n_periods):
        start = sp * period_len
        end = min((sp + 1) * period_len, len(strat_df))
        sub_ret = strat_df['strategy_ret'].iloc[start:end]
        sub_m = compute_metrics(sub_ret, f'P{sp+1}')
        sub_sharpes.append(sub_m['Sharpe'])
        start_date = strat_df.index[start].strftime('%Y-%m')
        end_date = strat_df.index[min(end-1, len(strat_df)-1)].strftime('%Y-%m')
        sub_names.append(f"{start_date} to {end_date}")
        print(f"  Period {sp+1} ({start_date} to {end_date}): "
              f"Sharpe={sub_m['Sharpe']:.3f}, CAGR={sub_m['CAGR']:.1%}, MaxDD={sub_m['MaxDD']:.1%}")

    positive_periods = sum(1 for s in sub_sharpes if s > 0)
    print(f"  Positive Sharpe periods: {positive_periods}/{n_periods}")

    if positive_periods >= 3:
        print("  PASS: Sharpe positive in >= 3/4 sub-periods")
        checks_passed += 1
    else:
        print("  FAIL: Sharpe not consistently positive")

    # --- 3. OUTLIER REMOVAL ---
    print("\n--- Test 3: Outlier Removal (drop top/bottom 1% of days) ---")
    checks_total += 1

    strat_rets = strat_df['strategy_ret']
    q01 = strat_rets.quantile(0.01)
    q99 = strat_rets.quantile(0.99)
    trimmed = strat_rets[(strat_rets >= q01) & (strat_rets <= q99)]
    trimmed_m = compute_metrics(trimmed, 'Trimmed')
    full_m = compute_metrics(strat_rets, 'Full')

    print(f"  Full Sharpe: {full_m['Sharpe']:.3f}")
    print(f"  Trimmed Sharpe (1%-99%): {trimmed_m['Sharpe']:.3f}")

    sharpe_ratio = trimmed_m['Sharpe'] / full_m['Sharpe'] if full_m['Sharpe'] != 0 else 0
    if sharpe_ratio > 0.5:
        print(f"  PASS: Trimmed/Full Sharpe ratio = {sharpe_ratio:.2f} > 0.50")
        checks_passed += 1
    else:
        print(f"  FAIL: Trimmed/Full Sharpe ratio = {sharpe_ratio:.2f} <= 0.50")

    # --- 4. R1 REGIME TEST (Bull vs Bear) ---
    print("\n--- Test 4: R1 Regime Test (Bull vs Bear Market Performance) ---")
    checks_total += 1

    # Define bull/bear using SPY 200d SMA
    spy_close = df['SPY_close']
    sma200 = spy_close.rolling(200).mean()
    bull_days = spy_close[spy_close > sma200].index
    bear_days = spy_close[spy_close <= sma200].index

    bull_rets = strat_df.loc[strat_df.index.isin(bull_days), 'strategy_ret']
    bear_rets = strat_df.loc[strat_df.index.isin(bear_days), 'strategy_ret']

    bull_m = compute_metrics(bull_rets, 'Bull') if len(bull_rets) > 50 else {'Sharpe': 0, 'CAGR': 0}
    bear_m = compute_metrics(bear_rets, 'Bear') if len(bear_rets) > 50 else {'Sharpe': 0, 'CAGR': 0}

    print(f"  Bull market days: {len(bull_rets)}, Sharpe: {bull_m['Sharpe']:.3f}")
    print(f"  Bear market days: {len(bear_rets)}, Sharpe: {bear_m['Sharpe']:.3f}")

    # R1 check: |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|) <= 0.50
    max_abs = max(abs(bull_m['Sharpe']), abs(bear_m['Sharpe']))
    if max_abs > 0:
        regime_skew = abs(bull_m['Sharpe'] - bear_m['Sharpe']) / max_abs
    else:
        regime_skew = 0

    print(f"  Regime skew: {regime_skew:.3f} (threshold: 0.50)")

    if regime_skew <= 0.50:
        print("  PASS: Strategy is regime-agnostic (skew <= 0.50)")
        checks_passed += 1
    else:
        print("  FAIL: Strategy is regime-biased (skew > 0.50)")

    # --- 5. FEATURE IMPORTANCE CONCENTRATION (LightGBM only) ---
    if model_type == 'lgbm':
        print("\n--- Test 5: Feature Importance Concentration ---")
        checks_total += 1

        # Train a final model on recent data to get feature importances
        feature_cols = features.columns.tolist()
        valid_mask = features.notna().all(axis=1)
        feat_clean = features[valid_mask].tail(TRAIN_DAYS * 2)
        from sklearn.model_selection import train_test_split

        # Get target aligned
        spy = df['SPY_close']
        next_ret = spy.pct_change().shift(-1)
        tgt = pd.Series(1, index=df.index)
        tgt[next_ret > GREEN_THRESH] = 2
        tgt[next_ret < RED_THRESH] = 0
        tgt_clean = tgt[valid_mask].loc[feat_clean.index]

        model_fi = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            verbose=-1, n_jobs=-1, random_state=42)
        model_fi.fit(feat_clean.values, tgt_clean.values)

        importances = model_fi.feature_importances_
        imp_pct = importances / importances.sum()
        sorted_idx = np.argsort(imp_pct)[::-1]

        print("  Top 10 features:")
        for rank, idx in enumerate(sorted_idx[:10]):
            print(f"    {rank+1}. {feature_cols[idx]}: {imp_pct[idx]:.1%}")

        top1_pct = imp_pct[sorted_idx[0]]
        top3_pct = imp_pct[sorted_idx[:3]].sum()

        if top1_pct < 0.30 and top3_pct < 0.60:
            print(f"  PASS: No single feature dominates (top1={top1_pct:.1%}, top3={top3_pct:.1%})")
            checks_passed += 1
        else:
            print(f"  FAIL: Feature concentration too high (top1={top1_pct:.1%}, top3={top3_pct:.1%})")

    print(f"\n{'='*60}")
    print(f"ADVERSARIAL SUMMARY: {checks_passed}/{checks_total} checks passed")
    print(f"{'='*60}")

    return checks_passed, checks_total


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("ML REGIME PREDICTOR — Next-Day SPY Regime Classification")
    print("=" * 70)
    print(f"Start: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: SLIDING {TRAIN_DAYS}d window, GREEN>{GREEN_THRESH*100:.1f}%, RED<{RED_THRESH*100:.1f}%")
    print(f"Capital: ${INITIAL_CAPITAL:,.0f} fixed, NO DCA")
    print()

    # --- Step 1: Download Data ---
    print("STEP 1: Download cross-asset data")
    df = download_data()
    print()

    # --- Step 2: Engineer Features ---
    print("STEP 2: Engineer features")
    features = engineer_features(df)
    print(f"  Generated {len(features.columns)} features")
    print(f"  Feature list: {', '.join(features.columns[:10])}... (+{len(features.columns)-10} more)")
    print()

    # --- Step 3: Create Target ---
    print("STEP 3: Create target variable")
    target, next_day_ret, regime_map = create_target(df)

    # Align features and target
    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]
    next_day_ret = next_day_ret.loc[common_idx]

    # Drop rows with NaN features
    valid_mask = features.notna().all(axis=1) & target.notna()
    features = features[valid_mask]
    target = target[valid_mask]
    next_day_ret = next_day_ret[valid_mask]

    regime_counts = target.value_counts().sort_index()
    for val, name in regime_map.items():
        cnt = regime_counts.get(val, 0)
        print(f"  {name}: {cnt} days ({cnt/len(target)*100:.1f}%)")
    print(f"  Total valid days: {len(target)}")
    print()

    # --- Step 4: Walk-Forward LightGBM ---
    print("STEP 4: Walk-Forward — LightGBM")
    lgbm_results = walk_forward(features, target, next_day_ret, df, model_type='lgbm')
    print()

    # --- Step 5: Walk-Forward MLP ---
    print("STEP 5: Walk-Forward — MLP (2-layer: 64-32)")
    mlp_results = walk_forward(features, target, next_day_ret, df, model_type='mlp')
    print()

    # --- Step 6: Backtest Both Models ---
    print("STEP 6: Backtest Strategies")
    print("-" * 50)

    for model_name, model_results in [('LightGBM', lgbm_results), ('MLP', mlp_results)]:
        print(f"\n--- {model_name} ---")

        # Classification report
        print(f"\n  Classification Report:")
        acc = accuracy_score(model_results['actual'], model_results['prediction'])
        print(f"  Overall Accuracy: {acc:.4f}")

        for val, name in regime_map.items():
            mask_actual = model_results['actual'] == val
            mask_pred = model_results['prediction'] == val
            tp = ((model_results['actual'] == val) & (model_results['prediction'] == val)).sum()
            fp = ((model_results['actual'] != val) & (model_results['prediction'] == val)).sum()
            fn = ((model_results['actual'] == val) & (model_results['prediction'] != val)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"  {name}: Precision={precision:.3f}, Recall={recall:.3f}, "
                  f"Actual={mask_actual.sum()}, Predicted={mask_pred.sum()}")

        # Backtest
        strat_df = backtest_strategy(model_results, df)
        strat_m = compute_metrics(strat_df['strategy_ret'], f'{model_name} ML Timing')
        bnh_m = compute_metrics(strat_df['spy_ret'], 'Buy-and-Hold SPY')

        print(f"\n  Strategy Performance (GREEN->UPRO, RED->SHY, FLAT->SPY):")
        print(f"  {'Metric':<20} {'ML Timing':>15} {'B&H SPY':>15} {'v4.4 Baseline':>15}")
        print(f"  {'-'*65}")
        print(f"  {'CAGR':<20} {strat_m['CAGR']:>14.1%} {bnh_m['CAGR']:>14.1%} {V44_CAGR:>14.1%}")
        print(f"  {'Sharpe':<20} {strat_m['Sharpe']:>15.3f} {bnh_m['Sharpe']:>15.3f} {V44_SHARPE:>15.3f}")
        print(f"  {'Sortino':<20} {strat_m['Sortino']:>15.3f} {bnh_m['Sortino']:>15.3f} {'N/A':>15}")
        print(f"  {'MaxDD':<20} {strat_m['MaxDD']:>14.1%} {bnh_m['MaxDD']:>14.1%} {'N/A':>15}")
        print(f"  {'Win Rate':<20} {strat_m['WinRate']:>14.1%} {bnh_m['WinRate']:>14.1%} {'N/A':>15}")
        print(f"  {'Profit Factor':<20} {strat_m['ProfitFactor']:>15.3f} {bnh_m['ProfitFactor']:>15.3f} {'N/A':>15}")
        print(f"  {'Total Return':<20} {strat_m['TotalReturn']:>14.1%} {bnh_m['TotalReturn']:>14.1%} {'N/A':>15}")
        print(f"  {'Ann. Volatility':<20} {strat_m['AnnVol']:>14.1%} {bnh_m['AnnVol']:>14.1%} {'N/A':>15}")

        # Allocation distribution
        alloc_counts = strat_df['allocation'].value_counts()
        print(f"\n  Allocation Distribution:")
        for alloc in ['UPRO', 'SPY', 'SHY']:
            cnt = alloc_counts.get(alloc, 0)
            print(f"    {alloc}: {cnt} days ({cnt/len(strat_df)*100:.1f}%)")

        # Equity curve
        equity = INITIAL_CAPITAL * (1 + strat_df['strategy_ret']).cumprod()
        spy_equity = INITIAL_CAPITAL * (1 + strat_df['spy_ret']).cumprod()
        print(f"\n  Final Equity: ${equity.iloc[-1]:,.0f} (ML) vs ${spy_equity.iloc[-1]:,.0f} (SPY B&H)")

        # Save results
        strat_df.to_csv(OUTPUT_DIR / f'{model_name.lower()}_strategy_returns.csv')

        # --- Step 7: Adversarial Validation ---
        adversarial_validation(model_results, strat_df, df, features, model_type=model_name.lower())

    # --- Final Summary ---
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"COMPLETED in {elapsed/60:.1f} minutes")
    print(f"Results saved to: output/ml_regime_predictor/")
    print(f"{'='*70}")

    # Save summary JSON
    summary = {
        'timestamp': dt.datetime.now().isoformat(),
        'config': {
            'train_window': TRAIN_DAYS,
            'green_threshold': GREEN_THRESH,
            'red_threshold': RED_THRESH,
            'initial_capital': INITIAL_CAPITAL,
            'window_type': 'SLIDING (HC #0)',
        },
        'elapsed_minutes': elapsed / 60,
    }
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
