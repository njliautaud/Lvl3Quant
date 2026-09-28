#!/usr/bin/env python3
"""
VIX Spike Trading Execution Strategy
=====================================
Uses the validated GBM VIX spike predictor (AUC 0.89) to generate
defensive/offensive allocation signals.

When spike predicted (prob > threshold):
  - Buy VIX-linked ETFs (VIXY/VXZ)
  - Shift equity to defensive (TLT, GLD, SHY)
When no spike predicted:
  - Stay risk-on: SPY/QQQ equal weight

Walk-forward with SLIDING 252d window (HC #0).
5-day holding period matching the model's predictive horizon.

Benchmarks: SPY buy-and-hold, simple VIX>20 threshold strategy.
Full adversarial: permutation 100x, sub-period 5yr, outlier, R1 regime.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL   = 100_000
TRAIN_WINDOW      = 252 * 5    # 5 years sliding (match original predictor)
RETRAIN_EVERY     = 20         # retrain every 20 trading days
N_PERMUTATIONS    = 100
COST_BPS          = 2          # ETF spreads ~1-2 bps, assume market orders
HOLD_DAYS         = 5          # match 5-day prediction horizon
# Continuous defensive allocation: scale linearly from 0% defensive at
# SPIKE_PROB_LOW to 100% defensive at SPIKE_PROB_HIGH.
# Using low thresholds so the model has meaningful impact on allocation.
SPIKE_PROB_LOW    = 0.10       # below this: 100% risk-on
SPIKE_PROB_HIGH   = 0.50       # above this: max defensive + VIX long
# VIX long allocation kicks in only at high confidence
VIX_LONG_THRESH   = 0.40       # start VIX long allocation here

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_vix_spike_execution"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Assets ──────────────────────────────────────────────────────────────────
# VIX-linked ETFs for spike capture
VIX_ETFS = ['VIXY', 'VXZ']
# Defensive assets
DEFENSIVE = ['TLT', 'GLD', 'SHY']
# Risk-on assets
RISK_ON = ['SPY', 'QQQ']
# Feature signal tickers (from validated predictor)
FEATURE_TICKERS = ['SPY', 'TLT', 'GLD', 'USO', 'UUP', 'HYG', 'EEM', 'XLE']

ALL_TICKERS = list(dict.fromkeys(
    VIX_ETFS + DEFENSIVE + RISK_ON + FEATURE_TICKERS + ['^VIX']
))


def download_data():
    """Download all required data."""
    print("=" * 80)
    print("STEP 1: DOWNLOADING DATA")
    print("=" * 80)

    cache = BASE / "data" / "cache" / "vix_spike_exec_data.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 3600:
        df = pd.read_parquet(cache)
        print(f"  Cached: {df.shape}, {df.index[0].date()} -> {df.index[-1].date()}")
        return df

    print(f"  Downloading {len(ALL_TICKERS)} tickers...")
    raw = yf.download(ALL_TICKERS, start='2005-01-01', auto_adjust=True, progress=False)
    closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    if '^VIX' in closes.columns:
        closes = closes.rename(columns={'^VIX': 'VIX'})

    closes = closes.ffill().dropna(thresh=len(closes.columns) - 5)

    cache.parent.mkdir(parents=True, exist_ok=True)
    closes.to_parquet(cache)
    print(f"  Shape: {closes.shape}, {closes.index[0].date()} -> {closes.index[-1].date()}")

    for t in ALL_TICKERS:
        col = 'VIX' if t == '^VIX' else t
        if col in closes.columns:
            valid = closes[col].dropna()
            print(f"    {col:6s}: {len(valid)} days, {valid.index[0].date()} -> {valid.index[-1].date()}")
        else:
            print(f"    {col:6s}: MISSING")

    return closes


def build_features(df):
    """Build features matching the validated VIX spike predictor."""
    print("\n" + "=" * 80)
    print("STEP 2: BUILDING FEATURES (matching validated predictor)")
    print("=" * 80)

    features = pd.DataFrame(index=df.index)

    # 1. VIX level and derivatives (top feature in validated model: importance 0.73)
    if 'VIX' in df.columns:
        features['vix'] = df['VIX']
        features['vix_pctile_252d'] = df['VIX'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        features['vix_5d_change'] = df['VIX'].pct_change(5)
        features['vix_20d_change'] = df['VIX'].pct_change(20)
        features['vix_vs_ma20'] = df['VIX'] / df['VIX'].rolling(20).mean() - 1
        features['vix_vs_ma50'] = df['VIX'] / df['VIX'].rolling(50).mean() - 1

    # 2. SPY features
    if 'SPY' in df.columns:
        spy = df['SPY']
        spy_ret = spy.pct_change()
        features['spy_vol_10d'] = spy_ret.rolling(10).std() * np.sqrt(252)
        features['spy_vol_20d'] = spy_ret.rolling(20).std() * np.sqrt(252)
        features['spy_drawdown'] = spy / spy.rolling(252).max() - 1
        ma50 = spy.rolling(50).mean()
        ma200 = spy.rolling(200).mean()
        features['spy_ma_ratio_50_200'] = ma50 / ma200 - 1
        for w in [5, 20, 60]:
            features[f'spy_ret_{w}d'] = spy.pct_change(w)

    # 3. Bond features
    if 'TLT' in df.columns:
        for w in [5, 20]:
            features[f'bond_ret_{w}d'] = df['TLT'].pct_change(w)

    # 4. Gold features
    if 'GLD' in df.columns:
        for w in [5, 20]:
            features[f'gold_ret_{w}d'] = df['GLD'].pct_change(w)

    # 5. Gold/silver ratio proxy (GLD as proxy)
    if 'GLD' in df.columns and 'SPY' in df.columns:
        features['gold_silver_ratio'] = df['GLD'] / df['SPY']

    # 6. Credit spread proxy (HYG vs TLT)
    if 'HYG' in df.columns and 'TLT' in df.columns:
        features['credit_spread'] = df['TLT'].pct_change(20) - df['HYG'].pct_change(20)

    # 7. Copper proxy (XLE as commodity proxy)
    if 'XLE' in df.columns:
        features['copper_ret_5d'] = df['XLE'].pct_change(5)

    # 8. Cross-asset correlations
    if 'SPY' in df.columns and 'TLT' in df.columns:
        spy_d = df['SPY'].pct_change()
        tlt_d = df['TLT'].pct_change()
        features['corr_spy_tlt_20d'] = spy_d.rolling(20).corr(tlt_d)

    # 9. EM stress (EEM)
    if 'EEM' in df.columns:
        features['eem_ret_20d'] = df['EEM'].pct_change(20)

    # 10. Dollar strength (UUP)
    if 'UUP' in df.columns:
        features['uup_ret_20d'] = df['UUP'].pct_change(20)

    features = features.dropna(thresh=int(len(features.columns) * 0.5))

    print(f"  Features: {len(features.columns)} columns")
    print(f"  Date range: {features.index[0].date()} -> {features.index[-1].date()}")
    print(f"  Observations: {len(features)}")

    return features


def build_target(df):
    """Binary target: VIX spike within 5 days.
    Spike = VIX rises >15% within the next 5 trading days.
    This matches the validated predictor's target definition.
    """
    print("\n" + "=" * 80)
    print("STEP 3: BUILDING TARGET (VIX 5d spike > 15%)")
    print("=" * 80)

    if 'VIX' not in df.columns:
        raise ValueError("VIX data required")

    vix = df['VIX']
    # Forward max VIX over next 5 days
    vix_fwd_max = vix.rolling(HOLD_DAYS).max().shift(-HOLD_DAYS)
    vix_fwd_change = (vix_fwd_max - vix) / vix
    target = (vix_fwd_change > 0.15).astype(int)

    valid = target.dropna()
    spike_rate = valid.mean()
    print(f"  Target observations: {len(valid)}")
    print(f"  Spike rate: {spike_rate:.1%} ({int(valid.sum())} spikes)")
    print(f"  No-spike rate: {(1 - spike_rate):.1%}")

    return target


def walk_forward_ml(features, target, df):
    """Walk-forward GBM spike prediction with sliding window."""
    print("\n" + "=" * 80)
    print("STEP 4: WALK-FORWARD ML TRAINING")
    print("=" * 80)

    common_idx = features.index.intersection(target.dropna().index)
    X = features.loc[common_idx].copy()
    y = target.loc[common_idx].copy()

    feature_cols = X.columns.tolist()
    predictions = pd.Series(np.nan, index=common_idx)

    n_retrains = 0
    total_retrains = (len(common_idx) - TRAIN_WINDOW) // RETRAIN_EVERY + 1
    model = None

    t0 = time.time()
    for i in range(TRAIN_WINDOW, len(common_idx)):
        if (i - TRAIN_WINDOW) % RETRAIN_EVERY == 0:
            train_start = i - TRAIN_WINDOW
            X_train = X.iloc[train_start:i][feature_cols].fillna(0)
            y_train = y.iloc[train_start:i]

            valid_mask = ~y_train.isna()
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]

            if len(X_train) < 100:
                continue

            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=30,
                random_state=42,
            )
            model.fit(X_train, y_train)
            n_retrains += 1

            if n_retrains % 200 == 0:
                elapsed = time.time() - t0
                pct = i / len(common_idx) * 100
                print(f"  Retrain {n_retrains}/{total_retrains} ({pct:.0f}%) "
                      f"[{elapsed:.0f}s elapsed] "
                      f"date={common_idx[i].date()}")

        if model is not None:
            X_pred = X.iloc[i:i+1][feature_cols].fillna(0)
            try:
                prob = model.predict_proba(X_pred)[:, 1][0]
            except IndexError:
                prob = 0.0
            predictions.iloc[i] = prob

    predictions = predictions.dropna()
    elapsed = time.time() - t0
    print(f"\n  Total retrains: {n_retrains}")
    print(f"  Predictions: {len(predictions)}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Mean spike probability: {predictions.mean():.3f}")
    print(f"  Defensive trigger (>{SPIKE_PROB_LOW}): {(predictions > SPIKE_PROB_LOW).mean():.1%}")
    print(f"  Max defensive (>{SPIKE_PROB_HIGH}): {(predictions > SPIKE_PROB_HIGH).mean():.1%}")
    print(f"  VIX long (>{VIX_LONG_THRESH}): {(predictions > VIX_LONG_THRESH).mean():.1%}")

    if model is not None:
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=False)
        print("\n  Top 10 features:")
        for feat, imp in importances.head(10).items():
            print(f"    {feat:30s}: {imp:.4f}")

    return predictions


def backtest_vix_spike(predictions, df):
    """
    Backtest the VIX spike trading strategy.

    Continuous position sizing based on spike probability:
    - prob < SPIKE_PROB_LOW: 100% risk-on (SPY/QQQ equal weight)
    - SPIKE_PROB_LOW < prob < SPIKE_PROB_HIGH: linear interpolation
      from 100% risk-on to 100% defensive
    - prob > VIX_LONG_THRESH: add VIX long (scaled by excess prob)
    - prob > SPIKE_PROB_HIGH: max defensive + max VIX long

    Holdings are held for HOLD_DAYS (5 days) before re-evaluation.
    """
    print("\n" + "=" * 80)
    print("STEP 5: BACKTEST - VIX SPIKE STRATEGY")
    print("=" * 80)

    # Daily returns for tradeable assets
    asset_returns = pd.DataFrame(index=df.index)
    all_assets = RISK_ON + DEFENSIVE + VIX_ETFS
    for a in all_assets:
        if a in df.columns:
            asset_returns[a] = df[a].pct_change()

    # Find available assets
    avail_vix = [v for v in VIX_ETFS if v in asset_returns.columns and asset_returns[v].dropna().shape[0] > 100]
    avail_def = [d for d in DEFENSIVE if d in asset_returns.columns]
    avail_on = [r for r in RISK_ON if r in asset_returns.columns]

    print(f"  Available VIX ETFs: {avail_vix}")
    print(f"  Available defensive: {avail_def}")
    print(f"  Available risk-on: {avail_on}")

    if not avail_on:
        raise ValueError("No risk-on assets available")

    pred_dates = predictions.index
    portfolio_returns = []
    current_weights = {}
    last_signal_date = None
    prev_weights = {}
    regime = 'risk_on'
    n_switches = 0

    for day_idx, date in enumerate(pred_dates):
        if date not in asset_returns.index:
            continue

        # Re-evaluate every HOLD_DAYS
        should_reeval = (last_signal_date is None or
                         day_idx - pred_dates.get_loc(last_signal_date) >= HOLD_DAYS
                         if last_signal_date in pred_dates else True)

        if should_reeval:
            prob = predictions.loc[date]
            old_regime = regime
            prev_weights = dict(current_weights)

            # Compute defensive fraction: linear from 0 at SPIKE_PROB_LOW to 1 at SPIKE_PROB_HIGH
            if prob <= SPIKE_PROB_LOW:
                def_frac = 0.0
                regime = 'risk_on'
            elif prob >= SPIKE_PROB_HIGH:
                def_frac = 1.0
                regime = 'full_defensive'
            else:
                def_frac = (prob - SPIKE_PROB_LOW) / (SPIKE_PROB_HIGH - SPIKE_PROB_LOW)
                regime = 'partial_defensive'

            on_frac = 1.0 - def_frac

            # VIX long allocation: scales from 0 at VIX_LONG_THRESH to 0.25 at SPIKE_PROB_HIGH+
            vix_alloc = 0.0
            if prob > VIX_LONG_THRESH and avail_vix:
                vix_scale = min((prob - VIX_LONG_THRESH) / (SPIKE_PROB_HIGH - VIX_LONG_THRESH), 1.0)
                vix_alloc = 0.25 * vix_scale

            # Build weights (sum to 1.0, VIX comes out of defensive allocation)
            current_weights = {}
            n_on = max(len(avail_on), 1)
            n_def = max(len(avail_def), 1)

            for a in avail_on:
                current_weights[a] = on_frac / n_on

            def_remaining = def_frac - vix_alloc
            if def_remaining < 0:
                def_remaining = 0
                vix_alloc = def_frac  # cap VIX to defensive fraction

            for a in avail_def:
                current_weights[a] = def_remaining / n_def

            # VIX ETF: use first available, check data exists for this date
            if vix_alloc > 0 and avail_vix:
                vix_etf = avail_vix[0]
                if date in asset_returns.index and not pd.isna(asset_returns[vix_etf].get(date, np.nan)):
                    current_weights[vix_etf] = vix_alloc
                else:
                    # Redistribute to defensive
                    for a in avail_def:
                        current_weights[a] += vix_alloc / n_def

            if old_regime != regime:
                n_switches += 1
            last_signal_date = date

        # Compute daily return
        day_ret = 0.0
        for a, w in current_weights.items():
            if a in asset_returns.columns and date in asset_returns.index:
                r = asset_returns[a].loc[date]
                if not pd.isna(r):
                    day_ret += w * r

        # Transaction cost on rebalance days: based on actual turnover
        if date == last_signal_date and prev_weights:
            turnover = 0.0
            all_keys = set(list(current_weights.keys()) + list(prev_weights.keys()))
            for k in all_keys:
                turnover += abs(current_weights.get(k, 0) - prev_weights.get(k, 0))
            cost = turnover * COST_BPS / 10000
            day_ret -= cost

        portfolio_returns.append({
            'date': date,
            'return': day_ret,
            'regime': regime,
            'def_frac': def_frac if should_reeval else np.nan,
        })

    ret_df = pd.DataFrame(portfolio_returns).set_index('date')
    ret_series = ret_df['return']

    regime_counts = ret_df['regime'].value_counts()
    avg_def = ret_df['def_frac'].dropna().mean()
    print(f"  Trading days: {len(ret_series)}")
    print(f"  Regime switches: {n_switches}")
    print(f"  Average defensive fraction: {avg_def:.1%}")
    for r, c in regime_counts.items():
        print(f"    {r}: {c} days ({c/len(ret_series)*100:.1f}%)")
    print(f"  Mean daily return: {ret_series.mean()*100:.4f}%")

    return ret_series, ret_df


def simple_vix_threshold_strategy(df, pred_dates):
    """Benchmark: go defensive when VIX > 20, risk-on otherwise."""
    asset_returns = pd.DataFrame(index=df.index)
    for a in RISK_ON + DEFENSIVE:
        if a in df.columns:
            asset_returns[a] = df[a].pct_change()

    avail_def = [d for d in DEFENSIVE if d in asset_returns.columns]
    avail_on = [r for r in RISK_ON if r in asset_returns.columns]

    common = pred_dates.intersection(asset_returns.index)
    rets = []

    for date in common:
        vix_val = df['VIX'].loc[date] if 'VIX' in df.columns and date in df.index else 15

        if vix_val > 20:
            # Defensive
            n = len(avail_def)
            day_ret = sum(asset_returns[a].loc[date] / max(n, 1)
                         for a in avail_def
                         if not pd.isna(asset_returns[a].get(date, np.nan)))
        else:
            # Risk-on
            n = len(avail_on)
            day_ret = sum(asset_returns[a].loc[date] / max(n, 1)
                         for a in avail_on
                         if not pd.isna(asset_returns[a].get(date, np.nan)))

        rets.append({'date': date, 'return': day_ret})

    ret_df = pd.DataFrame(rets).set_index('date')
    return ret_df['return']


def compute_metrics(returns, name="Strategy"):
    """Compute standard performance metrics."""
    r = returns.dropna()
    if len(r) < 30:
        return {}

    mu = r.mean() * 252
    sigma = r.std() * np.sqrt(252)
    sharpe = mu / (sigma + 1e-8)

    downside = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else 1e-8
    sortino = mu / (downside + 1e-8)

    cumret = (1 + r).cumprod()
    total_return = cumret.iloc[-1] - 1
    years = len(r) / 252
    cagr = (cumret.iloc[-1]) ** (1 / max(years, 0.01)) - 1

    running_max = cumret.cummax()
    drawdown = cumret / running_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / (losses + 1e-8)
    wr = (r > 0).mean()

    return {
        'name': name,
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'cagr': round(float(cagr * 100), 2),
        'max_dd': round(float(max_dd * 100), 2),
        'calmar': round(float(calmar), 3),
        'total_return': round(float(total_return * 100), 2),
        'profit_factor': round(float(pf), 3),
        'win_rate': round(float(wr * 100), 1),
        'annual_vol': round(float(sigma * 100), 2),
        'years': round(float(years), 1),
    }


def run_adversarial(ret_series, predictions, df):
    """Full adversarial validation: 4 gates."""
    print("\n" + "=" * 80)
    print("ADVERSARIAL VALIDATION")
    print("=" * 80)

    results = {}
    real = compute_metrics(ret_series, "real")
    real_sharpe = real.get('sharpe', 0)

    # ── 1. PERMUTATION TEST ──────────────────────────────────────────────
    print("\n  [1/4] Permutation test (100 shuffles)...")
    t0 = time.time()
    perm_sharpes = []

    for trial in range(N_PERMUTATIONS):
        perm_preds = predictions.copy()
        perm_preds[:] = np.random.permutation(perm_preds.values)

        perm_ret, _ = backtest_vix_spike(perm_preds, df)
        m = compute_metrics(perm_ret, f"perm_{trial}")
        if m:
            perm_sharpes.append(m['sharpe'])

        if (trial + 1) % 20 == 0:
            print(f"    {trial+1}/{N_PERMUTATIONS} done [{time.time()-t0:.0f}s]")

    perm_p = np.mean([s >= real_sharpe for s in perm_sharpes]) if perm_sharpes else 1.0
    perm_pass = bool(perm_p < 0.05)
    print(f"    Real Sharpe: {real_sharpe:.3f}")
    print(f"    Perm mean: {np.mean(perm_sharpes):.3f} +/- {np.std(perm_sharpes):.3f}")
    print(f"    p-value: {perm_p:.3f} -> {'PASS' if perm_pass else 'FAIL'}")
    results['permutation'] = {
        'p_value': round(float(perm_p), 4),
        'pass': perm_pass,
    }

    # ── 2. SUB-PERIOD CONSISTENCY (5-year blocks) ────────────────────────
    print("\n  [2/4] Sub-period consistency (5-year blocks)...")
    years = (ret_series.index[-1] - ret_series.index[0]).days / 365.25
    block_years = 5
    n_blocks = max(int(years // block_years), 2)
    block_size = len(ret_series) // n_blocks

    block_sharpes = []
    for b in range(n_blocks):
        start = b * block_size
        end = (b + 1) * block_size if b < n_blocks - 1 else len(ret_series)
        block_ret = ret_series.iloc[start:end]
        m = compute_metrics(block_ret, f"block_{b}")
        if m:
            block_sharpes.append(m['sharpe'])
            start_date = block_ret.index[0].date()
            end_date = block_ret.index[-1].date()
            print(f"    Block {b+1} ({start_date} -> {end_date}): Sharpe {m['sharpe']:.3f}")

    if block_sharpes and abs(np.mean(block_sharpes)) > 0.01:
        cv = float(np.std(block_sharpes) / abs(np.mean(block_sharpes)))
    else:
        cv = 999.0
    sub_pass = bool(cv < 0.50)
    print(f"    CV of Sharpe: {cv:.3f} -> {'PASS' if sub_pass else 'FAIL'}")
    results['sub_period'] = {
        'cv': round(cv, 3),
        'block_sharpes': [round(s, 3) for s in block_sharpes],
        'pass': sub_pass,
    }

    # ── 3. OUTLIER ROBUSTNESS ────────────────────────────────────────────
    print("\n  [3/4] Outlier robustness (remove top/bottom 5%)...")
    p95 = ret_series.quantile(0.95)
    p05 = ret_series.quantile(0.05)
    trimmed = ret_series[(ret_series > p05) & (ret_series < p95)]
    m_full = compute_metrics(ret_series, "full")
    m_trim = compute_metrics(trimmed, "trimmed")

    if m_full.get('sharpe', 0) != 0:
        deg = float((m_full['sharpe'] - m_trim['sharpe']) / abs(m_full['sharpe']))
    else:
        deg = 0.0
    outlier_pass = bool(abs(deg) < 0.30)
    print(f"    Full Sharpe: {m_full.get('sharpe', 0):.3f}")
    print(f"    Trimmed Sharpe: {m_trim.get('sharpe', 0):.3f}")
    print(f"    Degradation: {deg:.1%} -> {'PASS' if outlier_pass else 'FAIL'}")
    results['outlier'] = {
        'degradation': round(deg, 3),
        'pass': outlier_pass,
    }

    # ── 4. R1 REGIME TEST ────────────────────────────────────────────────
    print("\n  [4/4] R1 regime test (green vs red days)...")
    spy_daily_ret = df['SPY'].pct_change()

    green_days = [d for d in ret_series.index if d in spy_daily_ret.index and spy_daily_ret.loc[d] > 0]
    red_days = [d for d in ret_series.index if d in spy_daily_ret.index and spy_daily_ret.loc[d] <= 0]

    green_ret = ret_series.loc[ret_series.index.isin(green_days)]
    red_ret = ret_series.loc[ret_series.index.isin(red_days)]

    m_green = compute_metrics(green_ret, "green")
    m_red = compute_metrics(red_ret, "red")

    if m_green and m_red:
        s_g = m_green['sharpe']
        s_r = m_red['sharpe']
        gap = abs(s_g - s_r) / max(abs(s_g), abs(s_r), 0.01)
        r1_pass = bool(gap < 0.50)
        print(f"    Green Sharpe: {s_g:.3f}")
        print(f"    Red Sharpe: {s_r:.3f}")
        print(f"    |gap|/max: {gap:.3f} -> {'PASS' if r1_pass else 'FAIL'}")
        results['r1_regime'] = {
            'gap': round(float(gap), 3),
            'pass': r1_pass,
            'green_sharpe': float(s_g),
            'red_sharpe': float(s_r),
        }
    else:
        r1_pass = False
        results['r1_regime'] = {'pass': False, 'gap': 999, 'green_sharpe': 0, 'red_sharpe': 0}

    # Summary
    gates = sum([
        results.get('permutation', {}).get('pass', False),
        results.get('sub_period', {}).get('pass', False),
        results.get('outlier', {}).get('pass', False),
        results.get('r1_regime', {}).get('pass', False),
    ])
    verdict = 'PASS' if gates >= 3 else 'FAIL'
    results['summary'] = {
        'gates_passed': gates,
        'total': 4,
        'verdict': verdict,
    }
    print(f"\n  ADVERSARIAL SUMMARY: {gates}/4 gates passed -> {verdict}")

    return results


def main():
    t0 = time.time()
    print("=" * 80)
    print("VIX SPIKE TRADING EXECUTION STRATEGY")
    print("ML-driven defensive rotation on predicted VIX spikes")
    print("=" * 80)
    print()

    # ── Step 1: Data ─────────────────────────────────────────────────────
    df = download_data()

    # ── Step 2: Features ─────────────────────────────────────────────────
    features = build_features(df)

    # ── Step 3: Target ───────────────────────────────────────────────────
    target = build_target(df)

    # ── Step 4: Walk-forward ML ──────────────────────────────────────────
    predictions = walk_forward_ml(features, target, df)

    # ── Step 5: Backtest ─────────────────────────────────────────────────
    ret_series, ret_df = backtest_vix_spike(predictions, df)

    # ── Step 6: Metrics ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 6: RESULTS")
    print("=" * 80)

    m_portfolio = compute_metrics(ret_series, "VIX Spike Strategy")

    # SPY buy-and-hold benchmark
    spy_ret = df['SPY'].pct_change().reindex(ret_series.index).fillna(0)
    m_spy = compute_metrics(spy_ret, "SPY B&H")

    # Simple VIX>20 threshold benchmark
    simple_ret = simple_vix_threshold_strategy(df, ret_series.index)
    m_simple = compute_metrics(simple_ret, "Simple VIX>20 Threshold")

    print(f"\n  {'Strategy':<28} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'PF':>8} {'WR':>8}")
    print(f"  {'-'*90}")
    for met in [m_portfolio, m_spy, m_simple]:
        if met:
            print(f"  {met['name']:<28} {met['sharpe']:>8.3f} {met['sortino']:>8.3f} "
                  f"{met['cagr']:>7.1f}% {met['max_dd']:>7.1f}% {met['calmar']:>8.3f} "
                  f"{met['profit_factor']:>8.3f} {met['win_rate']:>7.1f}%")

    # ── Step 7: Adversarial validation ───────────────────────────────────
    adv = run_adversarial(ret_series, predictions, df)

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        'strategy': 'VIX Spike Trading Execution',
        'date': pd.Timestamp.now().isoformat(),
        'metrics': {
            'portfolio': m_portfolio,
            'spy': m_spy,
            'simple_threshold': m_simple,
        },
        'adversarial': adv,
        'parameters': {
            'train_window': TRAIN_WINDOW,
            'retrain_every': RETRAIN_EVERY,
            'hold_days': HOLD_DAYS,
            'spike_prob_low': SPIKE_PROB_LOW,
            'spike_prob_high': SPIKE_PROB_HIGH,
            'vix_long_thresh': VIX_LONG_THRESH,
            'cost_bps': COST_BPS,
            'vix_etfs': VIX_ETFS,
            'defensive_assets': DEFENSIVE,
            'risk_on_assets': RISK_ON,
            'n_permutations': N_PERMUTATIONS,
            'initial_capital': INITIAL_CAPITAL,
        },
        'runtime_seconds': round(time.time() - t0, 1),
    }

    results_path = OUTPUT / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # ── Equity curve plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

    cum = (1 + ret_series).cumprod() * INITIAL_CAPITAL
    spy_cum = (1 + spy_ret).cumprod() * INITIAL_CAPITAL
    simple_cum = (1 + simple_ret.reindex(ret_series.index).fillna(0)).cumprod() * INITIAL_CAPITAL

    axes[0].plot(cum.index, cum.values,
                 label=f'VIX Spike Strategy (Sharpe={m_portfolio.get("sharpe", 0):.2f})',
                 linewidth=2, color='blue')
    axes[0].plot(spy_cum.index, spy_cum.values,
                 label=f'SPY B&H (Sharpe={m_spy.get("sharpe", 0):.2f})',
                 alpha=0.7, color='gray')
    axes[0].plot(simple_cum.index, simple_cum.values,
                 label=f'VIX>20 Threshold (Sharpe={m_simple.get("sharpe", 0):.2f})',
                 alpha=0.7, color='orange')
    axes[0].set_title('VIX Spike Trading Strategy - Equity Curves ($100K)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('Portfolio Value ($)')

    # Drawdown
    dd = cum / cum.cummax() - 1
    dd_spy = spy_cum / spy_cum.cummax() - 1
    axes[1].fill_between(dd.index, dd.values, 0, alpha=0.5, color='red', label='Strategy DD')
    axes[1].plot(dd_spy.index, dd_spy.values, alpha=0.5, color='gray', label='SPY DD')
    axes[1].set_title('Drawdown')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylabel('Drawdown')

    # Spike probabilities
    axes[2].plot(predictions.index, predictions.values, alpha=0.5, linewidth=0.5, color='purple')
    axes[2].axhline(SPIKE_PROB_THRESH, color='orange', linestyle='--', alpha=0.7,
                     label=f'Spike threshold ({SPIKE_PROB_THRESH})')
    axes[2].axhline(SPIKE_PROB_HIGH, color='red', linestyle='--', alpha=0.7,
                     label=f'High-conf threshold ({SPIKE_PROB_HIGH})')
    axes[2].set_title('VIX Spike Probability (ML Predictions)')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylabel('P(VIX Spike)')

    plt.tight_layout()
    chart_path = OUTPUT / 'equity_curve.png'
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Chart saved to {chart_path}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"COMPLETE in {elapsed:.0f}s")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
