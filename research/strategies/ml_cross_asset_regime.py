#!/usr/bin/env python3
"""
Cross-Asset Risk-On/Risk-Off Regime Rotation
=============================================
Thesis: Use ML (GBM) to classify market regime (risk-on vs risk-off)
using cross-asset features, then rotate between defensive and cyclical
sector ETFs accordingly.

Regime signals: SPY, TLT, GLD, USO, UUP, ^VIX
Trading universe: XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU (sector ETFs)

Walk-forward: SLIDING 252d (HC #0). Fixed $100K, NO DCA.
Full adversarial: permutation 100x, sub-period 5-year blocks, outlier, R1 regime.
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
TRAIN_WINDOW      = 252       # 1 year sliding
RETRAIN_EVERY     = 20        # retrain every 20 trading days
N_PERMUTATIONS    = 100
REBAL_COST_BPS    = 5         # 5 bps per rebalance (ETFs are liquid)
TARGET_VOL        = 0.10      # 10% annual vol per position
REBAL_FREQ        = 5         # rebalance every 5 trading days
RISK_ON_THRESH    = 0.55
RISK_OFF_THRESH   = 0.45
FORWARD_DAYS      = 5         # target: 5-day forward return

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_cross_asset_regime"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Regime signal assets
SIGNAL_TICKERS = ['SPY', 'TLT', 'GLD', 'USO', 'UUP']

# Trading universe — sector ETFs
CYCLICAL_SECTORS  = ['XLK', 'XLY', 'XLF', 'XLE', 'XLI']
DEFENSIVE_SECTORS = ['XLP', 'XLV', 'XLU', 'XLB']
ALL_SECTORS = CYCLICAL_SECTORS + DEFENSIVE_SECTORS


def download_data():
    """Download all required data via yfinance."""
    print("=" * 80)
    print("STEP 1: DOWNLOADING DATA")
    print("=" * 80)

    all_tickers = SIGNAL_TICKERS + ALL_SECTORS + ['^VIX']
    # Remove duplicates
    all_tickers = list(dict.fromkeys(all_tickers))

    cache = BASE / "data" / "cache" / "cross_asset_regime_data.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 3600:
        df = pd.read_parquet(cache)
        print(f"  Cached: {df.shape}, {df.index[0].date()} -> {df.index[-1].date()}")
        return df

    print(f"  Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start='2005-01-01', auto_adjust=True, progress=False)
    closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    if '^VIX' in closes.columns:
        closes = closes.rename(columns={'^VIX': 'VIX'})

    closes = closes.ffill().dropna(thresh=len(closes.columns) - 3)

    cache.parent.mkdir(parents=True, exist_ok=True)
    closes.to_parquet(cache)
    print(f"  Shape: {closes.shape}, {closes.index[0].date()} -> {closes.index[-1].date()}")

    # Print available tickers
    for t in all_tickers:
        col = 'VIX' if t == '^VIX' else t
        if col in closes.columns:
            valid = closes[col].dropna()
            print(f"    {t:6s}: {len(valid)} days, {valid.index[0].date()} -> {valid.index[-1].date()}")
        else:
            print(f"    {t:6s}: MISSING")

    return closes


def try_fred_data(df):
    """Try to get 10y-2y yield spread from FRED. Non-critical if unavailable."""
    print("\n  Attempting FRED 10y-2y spread download...")
    try:
        from pandas_datareader import data as pdr
        spread = pdr.get_data_fred('T10Y2Y', start='2005-01-01')
        spread = spread.reindex(df.index).ffill()
        print(f"    FRED T10Y2Y: {spread.dropna().shape[0]} days")
        return spread['T10Y2Y']
    except Exception as e:
        print(f"    FRED unavailable ({e}), skipping yield spread feature")
        return None


def build_features(df, fred_spread=None):
    """Build ML features from regime signal assets."""
    print("\n" + "=" * 80)
    print("STEP 2: BUILDING FEATURES")
    print("=" * 80)

    features = pd.DataFrame(index=df.index)

    # 1. Multi-horizon returns for each signal asset
    for ticker in SIGNAL_TICKERS:
        if ticker not in df.columns:
            continue
        price = df[ticker]
        for window in [5, 20, 60]:
            features[f'{ticker}_ret_{window}d'] = price.pct_change(window)

    # 2. SPY-TLT spread (equity vs bonds momentum)
    if 'SPY' in df.columns and 'TLT' in df.columns:
        spy_ret_20 = df['SPY'].pct_change(20)
        tlt_ret_20 = df['TLT'].pct_change(20)
        features['spy_tlt_spread_20d'] = spy_ret_20 - tlt_ret_20

        spy_ret_60 = df['SPY'].pct_change(60)
        tlt_ret_60 = df['TLT'].pct_change(60)
        features['spy_tlt_spread_60d'] = spy_ret_60 - tlt_ret_60

    # 3. GLD-UUP spread (gold vs dollar)
    if 'GLD' in df.columns and 'UUP' in df.columns:
        gld_ret_20 = df['GLD'].pct_change(20)
        uup_ret_20 = df['UUP'].pct_change(20)
        features['gld_uup_spread_20d'] = gld_ret_20 - uup_ret_20

    # 4. Rolling correlations
    if 'SPY' in df.columns and 'TLT' in df.columns:
        spy_daily = df['SPY'].pct_change()
        tlt_daily = df['TLT'].pct_change()
        features['corr_spy_tlt_20d'] = spy_daily.rolling(20).corr(tlt_daily)

    if 'SPY' in df.columns and 'GLD' in df.columns:
        spy_daily = df['SPY'].pct_change()
        gld_daily = df['GLD'].pct_change()
        features['corr_spy_gld_20d'] = spy_daily.rolling(20).corr(gld_daily)

    # 5. VIX features
    if 'VIX' in df.columns:
        features['vix_level'] = df['VIX']
        features['vix_5d_change'] = df['VIX'].pct_change(5)
        features['vix_20d_ma'] = df['VIX'].rolling(20).mean()
        features['vix_vs_ma'] = df['VIX'] / df['VIX'].rolling(20).mean() - 1

    # 6. Realized vol (20d) for SPY
    if 'SPY' in df.columns:
        spy_daily = df['SPY'].pct_change()
        features['spy_realized_vol_20d'] = spy_daily.rolling(20).std() * np.sqrt(252)
        features['spy_realized_vol_60d'] = spy_daily.rolling(60).std() * np.sqrt(252)

    # 7. FRED yield spread if available
    if fred_spread is not None:
        features['yield_spread_10y2y'] = fred_spread
        features['yield_spread_5d_change'] = fred_spread.diff(5)

    # 8. Additional cross-asset features
    # SPY momentum vs its own vol (risk-adjusted momentum)
    if 'SPY' in df.columns:
        spy_mom = df['SPY'].pct_change(20)
        spy_vol = df['SPY'].pct_change().rolling(20).std() * np.sqrt(252)
        features['spy_risk_adj_mom'] = spy_mom / (spy_vol + 1e-8)

    # Count how many signal assets are in uptrend (20d return > 0)
    uptrend_count = pd.DataFrame()
    for ticker in SIGNAL_TICKERS:
        if ticker in df.columns:
            uptrend_count[ticker] = (df[ticker].pct_change(20) > 0).astype(float)
    if len(uptrend_count.columns) > 0:
        features['n_assets_uptrend'] = uptrend_count.sum(axis=1)
        features['pct_assets_uptrend'] = uptrend_count.mean(axis=1)

    # Drop initial NaN rows
    features = features.dropna(thresh=int(len(features.columns) * 0.5))

    print(f"  Features: {len(features.columns)} columns")
    print(f"  Date range: {features.index[0].date()} -> {features.index[-1].date()}")
    print(f"  Observations: {len(features)}")
    for col in features.columns:
        pct_nan = features[col].isna().mean() * 100
        if pct_nan > 0:
            print(f"    {col}: {pct_nan:.1f}% NaN")

    return features


def build_target(df):
    """Binary target: is SPY 5-day forward return positive?"""
    print("\n" + "=" * 80)
    print("STEP 3: BUILDING TARGET")
    print("=" * 80)

    spy_fwd = df['SPY'].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
    target = (spy_fwd > 0).astype(int)

    valid = target.dropna()
    print(f"  Target observations: {len(valid)}")
    print(f"  Risk-on (SPY 5d > 0): {valid.mean():.1%}")
    print(f"  Risk-off (SPY 5d < 0): {(1 - valid).mean():.1%}")

    return target


def walk_forward_ml(features, target, df):
    """Walk-forward GBM classification with sliding window."""
    print("\n" + "=" * 80)
    print("STEP 4: WALK-FORWARD ML TRAINING")
    print("=" * 80)

    # Align features and target
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
        # Only retrain every RETRAIN_EVERY days
        if (i - TRAIN_WINDOW) % RETRAIN_EVERY == 0:
            train_start = i - TRAIN_WINDOW
            X_train = X.iloc[train_start:i][feature_cols].fillna(0)
            y_train = y.iloc[train_start:i]

            # Remove any remaining NaN in target
            valid_mask = ~y_train.isna()
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]

            if len(X_train) < 50:
                continue

            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=20,
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

        # Predict for current day
        if model is not None:
            X_pred = X.iloc[i:i+1][feature_cols].fillna(0)
            prob = model.predict_proba(X_pred)[:, 1][0]
            predictions.iloc[i] = prob

    predictions = predictions.dropna()
    elapsed = time.time() - t0
    print(f"\n  Total retrains: {n_retrains}")
    print(f"  Predictions: {len(predictions)}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Mean probability: {predictions.mean():.3f}")
    print(f"  Risk-on calls (>{RISK_ON_THRESH}): {(predictions > RISK_ON_THRESH).mean():.1%}")
    print(f"  Risk-off calls (<{RISK_OFF_THRESH}): {(predictions < RISK_OFF_THRESH).mean():.1%}")
    print(f"  Dead zone ({RISK_OFF_THRESH}-{RISK_ON_THRESH}): "
          f"{((predictions >= RISK_OFF_THRESH) & (predictions <= RISK_ON_THRESH)).mean():.1%}")

    # Feature importance (from last model)
    if model is not None:
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=False)
        print("\n  Top 10 features:")
        for feat, imp in importances.head(10).items():
            print(f"    {feat:30s}: {imp:.4f}")

    return predictions


def compute_sector_vols(df, sectors, lookback=60):
    """Compute rolling annualized vol for each sector."""
    vols = pd.DataFrame(index=df.index)
    for s in sectors:
        if s in df.columns:
            vols[s] = df[s].pct_change().rolling(lookback).std() * np.sqrt(252)
    return vols


def backtest_regime_rotation(predictions, df):
    """Backtest the regime rotation strategy."""
    print("\n" + "=" * 80)
    print("STEP 5: BACKTEST — REGIME ROTATION")
    print("=" * 80)

    # Compute sector vols for risk parity
    sector_vols = compute_sector_vols(df, ALL_SECTORS)

    # Daily returns for all sectors
    sector_returns = pd.DataFrame()
    for s in ALL_SECTORS:
        if s in df.columns:
            sector_returns[s] = df[s].pct_change()

    # Align with prediction dates
    pred_dates = predictions.index
    portfolio_returns = []
    current_weights = {}
    last_rebal = -REBAL_FREQ  # force initial rebalance

    for day_idx, date in enumerate(pred_dates):
        if date not in sector_returns.index:
            continue

        # Rebalance every REBAL_FREQ days
        if day_idx - last_rebal >= REBAL_FREQ:
            prob = predictions.loc[date]

            # Determine regime
            if prob > RISK_ON_THRESH:
                # Risk-on: overweight cyclical
                active_sectors = [s for s in CYCLICAL_SECTORS if s in sector_returns.columns]
                regime = 'risk_on'
            elif prob < RISK_OFF_THRESH:
                # Risk-off: overweight defensive
                active_sectors = [s for s in DEFENSIVE_SECTORS if s in sector_returns.columns]
                regime = 'risk_off'
            else:
                # Dead zone: equal weight all
                active_sectors = [s for s in ALL_SECTORS if s in sector_returns.columns]
                regime = 'neutral'

            # Risk parity sizing within active group
            current_weights = {}
            if len(active_sectors) > 0:
                for s in active_sectors:
                    vol = sector_vols[s].loc[date] if s in sector_vols.columns and date in sector_vols.index else 0.20
                    if pd.isna(vol) or vol < 0.01:
                        vol = 0.20
                    # Weight inversely proportional to vol, targeting TARGET_VOL
                    raw_w = TARGET_VOL / vol
                    current_weights[s] = raw_w

                # Normalize so total portfolio vol ~ TARGET_VOL * sqrt(n) but cap leverage
                total_w = sum(current_weights.values())
                max_leverage = 1.5
                if total_w > max_leverage:
                    scale = max_leverage / total_w
                    current_weights = {k: v * scale for k, v in current_weights.items()}

            last_rebal = day_idx

        # Compute daily return from current weights
        day_ret = 0.0
        for s, w in current_weights.items():
            if s in sector_returns.columns and date in sector_returns.index:
                r = sector_returns[s].loc[date]
                if not pd.isna(r):
                    day_ret += w * r

        # Transaction cost on rebalance days (approximation)
        if day_idx == last_rebal:
            turnover = sum(abs(w) for w in current_weights.values())
            cost = turnover * REBAL_COST_BPS / 10000
            day_ret -= cost

        portfolio_returns.append({
            'date': date,
            'return': day_ret,
            'n_positions': len(current_weights),
            'regime': regime if day_idx == last_rebal else '',
        })

    ret_df = pd.DataFrame(portfolio_returns).set_index('date')
    ret_series = ret_df['return']

    # Stats
    regime_counts = ret_df[ret_df['regime'] != '']['regime'].value_counts()
    print(f"  Trading days: {len(ret_series)}")
    print(f"  Rebalance events: {(ret_df['regime'] != '').sum()}")
    for regime, count in regime_counts.items():
        print(f"    {regime}: {count}")
    print(f"  Mean daily return: {ret_series.mean()*100:.4f}%")

    return ret_series, ret_df


def equal_weight_benchmark(df, pred_dates):
    """Equal-weight all sector ETFs (no ML) benchmark."""
    sector_returns = pd.DataFrame()
    for s in ALL_SECTORS:
        if s in df.columns:
            sector_returns[s] = df[s].pct_change()

    # Filter to prediction dates
    common = pred_dates.intersection(sector_returns.index)
    ew_rets = sector_returns.loc[common].mean(axis=1)
    return ew_rets


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
    """Full adversarial validation suite."""
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
        # Shuffle predictions, re-run backtest
        perm_preds = predictions.copy()
        perm_preds[:] = np.random.permutation(perm_preds.values)

        perm_ret, _ = backtest_regime_rotation(perm_preds, df)
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

    # Classify each day: green if SPY close > prior close
    green_days = []
    red_days = []
    for date in ret_series.index:
        if date in spy_daily_ret.index:
            if spy_daily_ret.loc[date] > 0:
                green_days.append(date)
            else:
                red_days.append(date)

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
    print("CROSS-ASSET RISK-ON/RISK-OFF REGIME ROTATION")
    print("ML-driven sector ETF rotation strategy")
    print("=" * 80)
    print()

    # ── Step 1: Data ─────────────────────────────────────────────────────
    df = download_data()
    fred_spread = try_fred_data(df)

    # ── Step 2: Features ─────────────────────────────────────────────────
    features = build_features(df, fred_spread)

    # ── Step 3: Target ───────────────────────────────────────────────────
    target = build_target(df)

    # ── Step 4: Walk-forward ML ──────────────────────────────────────────
    predictions = walk_forward_ml(features, target, df)

    # ── Step 5: Backtest ─────────────────────────────────────────────────
    ret_series, ret_df = backtest_regime_rotation(predictions, df)

    # ── Compute metrics ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 6: RESULTS")
    print("=" * 80)

    m_portfolio = compute_metrics(ret_series, "ML Regime Rotation")

    # SPY buy-and-hold benchmark
    spy_ret = df['SPY'].pct_change().reindex(ret_series.index).fillna(0)
    m_spy = compute_metrics(spy_ret, "SPY B&H")

    # Equal-weight sector rotation benchmark (no ML)
    ew_ret = equal_weight_benchmark(df, ret_series.index)
    m_ew = compute_metrics(ew_ret, "Equal-Weight Sectors")

    print(f"\n  {'Strategy':<25} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'PF':>8} {'WR':>8}")
    print(f"  {'-'*85}")
    for met in [m_portfolio, m_spy, m_ew]:
        if met:
            print(f"  {met['name']:<25} {met['sharpe']:>8.3f} {met['sortino']:>8.3f} "
                  f"{met['cagr']:>7.1f}% {met['max_dd']:>7.1f}% {met['calmar']:>8.3f} "
                  f"{met['profit_factor']:>8.3f} {met['win_rate']:>7.1f}%")

    # ── Step 7: Adversarial validation ───────────────────────────────────
    adv = run_adversarial(ret_series, predictions, df)

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        'strategy': 'ML Cross-Asset Regime Rotation',
        'metrics': {
            'portfolio': m_portfolio,
            'spy': m_spy,
            'equal_weight': m_ew,
        },
        'adversarial': adv,
        'parameters': {
            'train_window': TRAIN_WINDOW,
            'retrain_every': RETRAIN_EVERY,
            'forward_days': FORWARD_DAYS,
            'risk_on_threshold': RISK_ON_THRESH,
            'risk_off_threshold': RISK_OFF_THRESH,
            'target_vol': TARGET_VOL,
            'rebal_freq': REBAL_FREQ,
            'rebal_cost_bps': REBAL_COST_BPS,
            'signal_tickers': SIGNAL_TICKERS,
            'cyclical_sectors': CYCLICAL_SECTORS,
            'defensive_sectors': DEFENSIVE_SECTORS,
            'n_permutations': N_PERMUTATIONS,
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
    ew_cum = (1 + ew_ret).cumprod() * INITIAL_CAPITAL

    axes[0].plot(cum.index, cum.values, label=f'ML Regime Rotation (Sharpe={m_portfolio.get("sharpe", 0):.2f})',
                 linewidth=2, color='blue')
    axes[0].plot(spy_cum.index, spy_cum.values, label=f'SPY B&H (Sharpe={m_spy.get("sharpe", 0):.2f})',
                 alpha=0.7, color='gray')
    axes[0].plot(ew_cum.index, ew_cum.values, label=f'Equal-Weight Sectors (Sharpe={m_ew.get("sharpe", 0):.2f})',
                 alpha=0.7, color='orange')
    axes[0].set_title('Cross-Asset Regime Rotation — Equity Curves ($100K)')
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

    # ML predictions over time
    axes[2].plot(predictions.index, predictions.values, alpha=0.5, linewidth=0.5, color='purple')
    axes[2].axhline(RISK_ON_THRESH, color='green', linestyle='--', alpha=0.7, label=f'Risk-On > {RISK_ON_THRESH}')
    axes[2].axhline(RISK_OFF_THRESH, color='red', linestyle='--', alpha=0.7, label=f'Risk-Off < {RISK_OFF_THRESH}')
    axes[2].set_title('ML Regime Probability')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylabel('P(Risk-On)')

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
