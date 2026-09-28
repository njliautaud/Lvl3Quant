#!/usr/bin/env python3
"""
Macro Timing Overlay v1
========================
Tests whether macro timing signals can improve our validated market-neutral
L/S sector rotation strategy (baseline: Sharpe 2.64, MDD -6.9%, WR 78.6%).

Baseline: Long top-3, short bottom-3 LGBM-ranked sectors, monthly rebalance,
equal-weight, sliding 500-day LGBM train window.

8 Variants:
  A: Baseline market-neutral L/S (no timing)
  B: VIX Regime Scaling (VIX>25 double shorts, VIX<15 double longs)
  C: Trend Filter (flat when SPY < 200d SMA)
  D: Momentum Regime (positive SPY 1m mom -> 2:1 longs, negative -> 2:1 shorts)
  E: Credit Spread Regime (VIX/VIX3M > 1.0 backwardation -> half size)
  F: Correlation Regime (high cross-sector corr -> reduce, low -> full)
  G: Multi-Signal Timing (combine B+C+D, need 2/3 aligned for full size)
  H: Adaptive Concentration (high dispersion -> top2/bot2, low -> top4/bot4)

21 LGBM features (production set):
  ret_5d, ret_10d, ret_21d, ret_63d, ret_126d, ret_252d,
  vol_21d, vol_63d, sharpe_63d, maxdd_63d, pct_52w_high, mom_accel,
  pct_pos_months_12m, sortino_63d, calmar_1y, up_capture,
  trend_r2_63d, trend_slope_63d, sector_spy_beta_63d,
  sector_relative_vol_21d, cross_sector_dispersion

5-gate adversarial: permutation (300), regime stability, sub-period,
outlier removal, yearly consistency.

MLflow experiment: macro_timing_overlay_v1
Output: output/growth_research/macro_timing_v1/
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Flush-print wrapper
# ---------------------------------------------------------------------------
_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "macro_timing_overlay_v1"
USE_MLFLOW = True

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as e:
    fprint(f"[WARN] MLflow unavailable: {e}. Continuing without tracking.")
    USE_MLFLOW = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
BENCHMARK = 'SPY'
MACRO_TICKERS = ['^VIX', '^VIX3M', 'TLT']
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK] + MACRO_TICKERS

TRAIN_WINDOW = 500
REBAL_PERIOD = 21
STARTING_CAPITAL = 10000
N_LONG = 3
N_SHORT = 3

FEATURE_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y', 'up_capture',
    'trend_r2_63d', 'trend_slope_63d', 'sector_spy_beta_63d',
    'sector_relative_vol_21d', 'cross_sector_dispersion',
]
assert len(FEATURE_COLS) == 21, f"Expected 21 features, got {len(FEATURE_COLS)}"

LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 5,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
}

VARIANT_NAMES = {
    'A': 'Baseline L/S',
    'B': 'VIX Regime Scaling',
    'C': 'Trend Filter (SPY>200d SMA)',
    'D': 'Momentum Regime',
    'E': 'Credit Spread Regime',
    'F': 'Correlation Regime',
    'G': 'Multi-Signal Timing (B+C+D)',
    'H': 'Adaptive Concentration',
}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'output' / 'growth_research' / 'macro_timing_v1'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_data(start='2018-01-01'):
    """Download sector ETFs, SPY, VIX, VIX3M, TLT via yfinance."""
    import yfinance as yf
    end = datetime.now().strftime('%Y-%m-%d')
    fprint(f"[DATA] Downloading {len(ALL_TICKERS)} tickers from {start} to {end} ...")

    data = yf.download(ALL_TICKERS, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
        volume = data['Volume'] if 'Volume' in data.columns.get_level_values(0) else None
    else:
        close = data[['Close']].copy()
        volume = data[['Volume']].copy() if 'Volume' in data.columns else None

    close = close.dropna(how='all')
    # Forward-fill VIX/VIX3M on holidays
    close = close.ffill()

    fprint(f"[DATA] {len(close)} trading days, {close.index[0].date()} to {close.index[-1].date()}")
    fprint(f"[DATA] Tickers available: {list(close.columns)}")

    return close

# ---------------------------------------------------------------------------
# Feature computation — 21 production features
# ---------------------------------------------------------------------------
def compute_features(close):
    """Compute 21 features for each sector ETF on each date. Returns long-form DataFrame."""
    fprint("[FEATURES] Computing 21 features for 11 sectors ...")
    t0 = time.time()

    spy_close = close[BENCHMARK]
    spy_ret = spy_close.pct_change()

    # Pre-compute cross-sector dispersion (used as feature)
    sector_rets = close[SECTOR_ETFS].pct_change()
    cross_sector_dispersion = sector_rets.std(axis=1)

    records = []

    for ticker in SECTOR_ETFS:
        c = close[ticker].dropna()
        ret = c.pct_change()

        # Return features
        ret_5d = c.pct_change(5)
        ret_10d = c.pct_change(10)
        ret_21d = c.pct_change(21)
        ret_63d = c.pct_change(63)
        ret_126d = c.pct_change(126)
        ret_252d = c.pct_change(252)

        # Volatility
        vol_21d = ret.rolling(21).std()
        vol_63d = ret.rolling(63).std()

        # Sharpe 63d
        mean_63d = ret.rolling(63).mean()
        sharpe_63d = mean_63d / vol_63d.replace(0, np.nan)

        # Max drawdown 63d
        def _maxdd(x):
            cum = (1 + x).cumprod()
            peak = cum.cummax()
            return (cum / peak - 1).min()
        maxdd_63d = ret.rolling(63).apply(_maxdd, raw=False)

        # Pct of 52-week high
        high_252 = c.rolling(252).max()
        pct_52w_high = c / high_252.replace(0, np.nan)

        # Momentum acceleration
        mom_accel = ret_21d - ret_21d.shift(21)

        # Pct positive months in trailing 12 months
        monthly_ret = c.pct_change(21)
        pct_pos_months_12m = monthly_ret.rolling(252).apply(
            lambda x: np.sum(x[::21] > 0) / max(len(x[::21]), 1), raw=True
        )

        # Sortino 63d
        downside = ret.copy()
        downside[downside > 0] = 0
        downside_std_63d = downside.rolling(63).std()
        sortino_63d = mean_63d / downside_std_63d.replace(0, np.nan)

        # Calmar 1y
        maxdd_252d = ret.rolling(252).apply(_maxdd, raw=False)
        calmar_1y = ret_252d / maxdd_252d.abs().replace(0, np.nan)

        # Up capture vs SPY (63d)
        spy_ret_aligned = spy_ret.reindex(ret.index)
        spy_up_mask = spy_ret_aligned > 0
        up_capture = pd.Series(np.nan, index=c.index)
        for i in range(63, len(c.index)):
            window_mask = spy_up_mask.iloc[i-63:i]
            if window_mask.sum() > 5:
                spy_up_days = spy_ret_aligned.iloc[i-63:i][window_mask]
                stock_up_days = ret.iloc[i-63:i][window_mask]
                spy_mean = spy_up_days.mean()
                if spy_mean != 0:
                    up_capture.iloc[i] = stock_up_days.mean() / spy_mean
                else:
                    up_capture.iloc[i] = 1.0

        # Trend slope 63d (linear regression slope of log prices)
        log_c = np.log(c.replace(0, np.nan))
        def _trend_slope(x):
            if len(x) < 10 or np.any(np.isnan(x)):
                return np.nan
            slope, _, _, _, _ = linregress(np.arange(len(x)), x)
            return slope
        trend_slope_63d = log_c.rolling(63).apply(_trend_slope, raw=True)

        # Trend R-squared 63d
        def _trend_r2(x):
            if len(x) < 10 or np.any(np.isnan(x)):
                return np.nan
            _, _, r_value, _, _ = linregress(np.arange(len(x)), x)
            return r_value ** 2
        trend_r2_63d = log_c.rolling(63).apply(_trend_r2, raw=True)

        # Sector-SPY beta (63d rolling)
        sector_spy_beta_63d = pd.Series(np.nan, index=c.index)
        for i in range(63, len(c.index)):
            s_ret = ret.iloc[i-63:i].values
            sp_ret = spy_ret_aligned.iloc[i-63:i].values
            valid = ~(np.isnan(s_ret) | np.isnan(sp_ret))
            if valid.sum() > 20:
                cov = np.cov(s_ret[valid], sp_ret[valid])
                if cov[1, 1] > 0:
                    sector_spy_beta_63d.iloc[i] = cov[0, 1] / cov[1, 1]

        # Sector relative volatility (vs SPY, 21d)
        spy_vol_21d = spy_ret.rolling(21).std().reindex(c.index)
        sector_relative_vol_21d = vol_21d / spy_vol_21d.replace(0, np.nan)

        # Cross-sector dispersion (aligned)
        csd_aligned = cross_sector_dispersion.reindex(c.index)

        # Forward return (target)
        fwd_ret_21d = c.pct_change(21).shift(-21)

        df = pd.DataFrame({
            'date': c.index,
            'ticker': ticker,
            'ret_5d': ret_5d,
            'ret_10d': ret_10d,
            'ret_21d': ret_21d,
            'ret_63d': ret_63d,
            'ret_126d': ret_126d,
            'ret_252d': ret_252d,
            'vol_21d': vol_21d,
            'vol_63d': vol_63d,
            'sharpe_63d': sharpe_63d,
            'maxdd_63d': maxdd_63d,
            'pct_52w_high': pct_52w_high,
            'mom_accel': mom_accel,
            'pct_pos_months_12m': pct_pos_months_12m,
            'sortino_63d': sortino_63d,
            'calmar_1y': calmar_1y,
            'up_capture': up_capture.values,
            'trend_r2_63d': trend_r2_63d,
            'trend_slope_63d': trend_slope_63d,
            'sector_spy_beta_63d': sector_spy_beta_63d.values,
            'sector_relative_vol_21d': sector_relative_vol_21d,
            'cross_sector_dispersion': csd_aligned,
            'fwd_ret_21d': fwd_ret_21d,
            'close': c,
        })
        records.append(df.reset_index(drop=True))

    features_df = pd.concat(records, ignore_index=True)
    features_df = features_df.dropna(subset=FEATURE_COLS + ['fwd_ret_21d'])
    elapsed = time.time() - t0
    fprint(f"[FEATURES] {len(features_df)} rows x {len(FEATURE_COLS)} features ({elapsed:.1f}s)")
    return features_df

# ---------------------------------------------------------------------------
# Macro signal computation
# ---------------------------------------------------------------------------
def compute_macro_signals(close):
    """Compute macro overlay signals at every trading date."""
    fprint("[MACRO] Computing macro signals ...")

    spy = close[BENCHMARK]
    vix = close['^VIX'] if '^VIX' in close.columns else None
    vix3m = close['^VIX3M'] if '^VIX3M' in close.columns else None

    signals = pd.DataFrame(index=close.index)

    # VIX level
    if vix is not None:
        signals['vix'] = vix
    else:
        fprint("[WARN] VIX data not available, using neutral VIX=20")
        signals['vix'] = 20.0

    # SPY 200d SMA
    signals['spy_sma200'] = spy.rolling(200).mean()
    signals['spy_above_sma200'] = (spy > signals['spy_sma200']).astype(float)

    # SPY 1-month momentum (21 trading days)
    signals['spy_mom_1m'] = spy.pct_change(21)

    # VIX/VIX3M ratio (term structure)
    if vix is not None and vix3m is not None:
        signals['vix_ratio'] = vix / vix3m.replace(0, np.nan)
    else:
        fprint("[WARN] VIX3M not available, defaulting ratio to 0.85 (contango)")
        signals['vix_ratio'] = 0.85

    # Cross-sector correlation (rolling 21d pairwise avg)
    sector_rets = close[SECTOR_ETFS].pct_change()
    def _avg_corr(window):
        """Average pairwise correlation from rolling window."""
        corr_mat = window.corr()
        n = len(corr_mat)
        # Upper triangle excluding diagonal
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        vals = corr_mat.values[mask]
        return np.nanmean(vals) if len(vals) > 0 else np.nan

    # Compute rolling average correlation efficiently
    fprint("[MACRO] Computing rolling 21d cross-sector correlation ...")
    avg_corrs = []
    dates = close.index
    sector_rets_arr = sector_rets.values
    n_sectors = len(SECTOR_ETFS)

    for i in range(len(dates)):
        if i < 21:
            avg_corrs.append(np.nan)
            continue
        window = sector_rets_arr[i-21:i, :n_sectors]
        # Remove rows with all NaN
        valid = ~np.all(np.isnan(window), axis=1)
        window = window[valid]
        if len(window) < 10:
            avg_corrs.append(np.nan)
            continue
        corr_mat = np.corrcoef(window.T)
        mask = np.triu(np.ones((n_sectors, n_sectors), dtype=bool), k=1)
        vals = corr_mat[mask]
        vals = vals[~np.isnan(vals)]
        avg_corrs.append(np.mean(vals) if len(vals) > 0 else np.nan)

    signals['cross_sector_corr'] = avg_corrs

    # Signal dispersion: std of LGBM scores at each rebal (computed later, placeholder)
    signals['spy_close'] = spy

    fprint(f"[MACRO] Signals computed. VIX range: {signals['vix'].min():.1f}-{signals['vix'].max():.1f}")
    return signals

# ---------------------------------------------------------------------------
# Walk-forward LGBM ranking
# ---------------------------------------------------------------------------
def walk_forward_ranking(features_df, close):
    """
    Sliding 500-day LGBM walk-forward. Monthly rebalance.
    Returns: list of dicts with rebal_date, ranked tickers, scores, forward returns.
    """
    fprint("[WF] Running walk-forward LGBM ranking ...")
    t0 = time.time()

    dates = sorted(features_df['date'].unique())
    all_close_dates = close.index.tolist()

    valid_start_idx = TRAIN_WINDOW
    rebal_dates = []
    for i in range(valid_start_idx, len(all_close_dates), REBAL_PERIOD):
        d = all_close_dates[i]
        if d in set(dates):
            rebal_dates.append(d)

    fprint(f"[WF] {len(rebal_dates)} rebalance points: {rebal_dates[0].date()} to {rebal_dates[-1].date()}")

    rankings = []
    total = len(rebal_dates)

    for idx, rebal_date in enumerate(rebal_dates):
        if idx % 10 == 0:
            fprint(f"  Rebalance {idx+1}/{total}...")

        # Training data: trailing TRAIN_WINDOW days strictly before rebal_date
        mask_before = features_df['date'] < rebal_date
        train_pool = features_df[mask_before].copy()
        train_dates_sorted = sorted(train_pool['date'].unique())
        if len(train_dates_sorted) < TRAIN_WINDOW:
            continue
        cutoff = train_dates_sorted[-TRAIN_WINDOW]
        train_pool = train_pool[train_pool['date'] >= cutoff]

        # Prediction data: features at rebal_date
        pred_data = features_df[features_df['date'] == rebal_date].copy()
        if len(pred_data) < 8:
            continue

        X_train = train_pool[FEATURE_COLS].values
        y_train = train_pool['fwd_ret_21d'].values
        X_pred = pred_data[FEATURE_COLS].values

        # Clean NaN/Inf
        valid_train = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]
        if len(X_train) < 100:
            continue

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_train, y_train)

        scores = model.predict(X_pred)
        pred_data = pred_data.copy()
        pred_data['score'] = scores

        # Sort by score descending
        pred_data = pred_data.sort_values('score', ascending=False)

        # Next rebal date for computing actual returns
        next_rebal_idx = min(idx + 1, total - 1)
        if next_rebal_idx == idx:
            continue
        next_date = rebal_dates[next_rebal_idx]

        # Compute actual returns for each sector between rebal dates
        sector_returns = {}
        for _, row in pred_data.iterrows():
            tkr = row['ticker']
            try:
                p0 = close.loc[rebal_date, tkr]
                p1 = close.loc[next_date, tkr]
                sector_returns[tkr] = (p1 / p0) - 1
            except:
                sector_returns[tkr] = 0.0

        spy_ret = 0.0
        try:
            spy_ret = (close.loc[next_date, BENCHMARK] / close.loc[rebal_date, BENCHMARK]) - 1
        except:
            pass

        rankings.append({
            'date': rebal_date,
            'next_date': next_date,
            'ranked_tickers': list(pred_data['ticker'].values),
            'scores': list(pred_data['score'].values),
            'sector_returns': sector_returns,
            'spy_ret': spy_ret,
        })

    elapsed = time.time() - t0
    fprint(f"[WF] Walk-forward done: {len(rankings)} periods ({elapsed:.1f}s)")
    return rankings

# ---------------------------------------------------------------------------
# Variant strategy implementations
# ---------------------------------------------------------------------------
def run_variant_A(rankings, macro_signals):
    """Baseline L/S: long top-3, short bottom-3, equal weight."""
    returns = []
    for r in rankings:
        longs = r['ranked_tickers'][:N_LONG]
        shorts = r['ranked_tickers'][-N_SHORT:]
        long_ret = np.mean([r['sector_returns'].get(t, 0) for t in longs])
        short_ret = np.mean([r['sector_returns'].get(t, 0) for t in shorts])
        period_ret = long_ret - short_ret
        returns.append({'date': r['date'], 'return': period_ret, 'spy_ret': r['spy_ret']})
    return returns


def run_variant_B(rankings, macro_signals):
    """VIX Regime Scaling: VIX>25 double shorts, VIX<15 double longs."""
    returns = []
    for r in rankings:
        date = r['date']
        vix_val = macro_signals.loc[date, 'vix'] if date in macro_signals.index else 20.0

        longs = r['ranked_tickers'][:N_LONG]
        shorts = r['ranked_tickers'][-N_SHORT:]
        long_ret = np.mean([r['sector_returns'].get(t, 0) for t in longs])
        short_ret = np.mean([r['sector_returns'].get(t, 0) for t in shorts])

        if vix_val > 25:
            # Bearish: double short allocation
            period_ret = long_ret - 2.0 * short_ret
        elif vix_val < 15:
            # Bullish: double long allocation
            period_ret = 2.0 * long_ret - short_ret
        else:
            # Normal
            period_ret = long_ret - short_ret

        returns.append({'date': date, 'return': period_ret, 'spy_ret': r['spy_ret']})
    return returns


def run_variant_C(rankings, macro_signals):
    """Trend Filter: only trade when SPY > 200d SMA. Flat otherwise."""
    returns = []
    for r in rankings:
        date = r['date']
        above_sma = macro_signals.loc[date, 'spy_above_sma200'] if date in macro_signals.index else 1.0

        if above_sma > 0.5:
            longs = r['ranked_tickers'][:N_LONG]
            shorts = r['ranked_tickers'][-N_SHORT:]
            long_ret = np.mean([r['sector_returns'].get(t, 0) for t in longs])
            short_ret = np.mean([r['sector_returns'].get(t, 0) for t in shorts])
            period_ret = long_ret - short_ret
        else:
            period_ret = 0.0  # flat / cash

        returns.append({'date': date, 'return': period_ret, 'spy_ret': r['spy_ret']})
    return returns


def run_variant_D(rankings, macro_signals):
    """Momentum Regime: positive SPY 1m mom -> 2:1 longs, negative -> 2:1 shorts."""
    returns = []
    for r in rankings:
        date = r['date']
        spy_mom = macro_signals.loc[date, 'spy_mom_1m'] if date in macro_signals.index else 0.0

        longs = r['ranked_tickers'][:N_LONG]
        shorts = r['ranked_tickers'][-N_SHORT:]
        long_ret = np.mean([r['sector_returns'].get(t, 0) for t in longs])
        short_ret = np.mean([r['sector_returns'].get(t, 0) for t in shorts])

        if spy_mom > 0:
            period_ret = 2.0 * long_ret - 1.0 * short_ret
        else:
            period_ret = 1.0 * long_ret - 2.0 * short_ret

        returns.append({'date': date, 'return': period_ret, 'spy_ret': r['spy_ret']})
    return returns


def run_variant_E(rankings, macro_signals):
    """Credit Spread Regime: VIX/VIX3M > 1.0 (backwardation) -> half size."""
    returns = []
    for r in rankings:
        date = r['date']
        vix_ratio = macro_signals.loc[date, 'vix_ratio'] if date in macro_signals.index else 0.85

        longs = r['ranked_tickers'][:N_LONG]
        shorts = r['ranked_tickers'][-N_SHORT:]
        long_ret = np.mean([r['sector_returns'].get(t, 0) for t in longs])
        short_ret = np.mean([r['sector_returns'].get(t, 0) for t in shorts])
        period_ret = long_ret - short_ret

        if vix_ratio > 1.0:
            period_ret *= 0.5  # stress regime, half size

        returns.append({'date': date, 'return': period_ret, 'spy_ret': r['spy_ret']})
    return returns


def run_variant_F(rankings, macro_signals):
    """Correlation Regime: high cross-sector corr (>0.7) reduce, low (<0.3) full."""
    returns = []
    for r in rankings:
        date = r['date']
        corr = macro_signals.loc[date, 'cross_sector_corr'] if date in macro_signals.index else 0.5

        longs = r['ranked_tickers'][:N_LONG]
        shorts = r['ranked_tickers'][-N_SHORT:]
        long_ret = np.mean([r['sector_returns'].get(t, 0) for t in longs])
        short_ret = np.mean([r['sector_returns'].get(t, 0) for t in shorts])
        period_ret = long_ret - short_ret

        if np.isnan(corr):
            scale = 1.0
        elif corr > 0.7:
            scale = 0.5  # high correlation, L/S less effective
        elif corr < 0.3:
            scale = 1.0  # low correlation, full size
        else:
            # Linear interpolation: 0.3->1.0, 0.7->0.5
            scale = 1.0 - 0.5 * (corr - 0.3) / 0.4

        period_ret *= scale
        returns.append({'date': date, 'return': period_ret, 'spy_ret': r['spy_ret']})
    return returns


def run_variant_G(rankings, macro_signals):
    """Multi-Signal Timing: combine B+C+D. Need 2/3 aligned for full size, else half."""
    returns = []
    for r in rankings:
        date = r['date']
        vix_val = macro_signals.loc[date, 'vix'] if date in macro_signals.index else 20.0
        above_sma = macro_signals.loc[date, 'spy_above_sma200'] if date in macro_signals.index else 1.0
        spy_mom = macro_signals.loc[date, 'spy_mom_1m'] if date in macro_signals.index else 0.0

        # Count bullish signals
        bullish_count = 0
        if vix_val < 15:
            bullish_count += 1
        if above_sma > 0.5:
            bullish_count += 1
        if spy_mom > 0:
            bullish_count += 1

        # Count bearish signals
        bearish_count = 0
        if vix_val > 25:
            bearish_count += 1
        if above_sma < 0.5:
            bearish_count += 1
        if spy_mom < 0:
            bearish_count += 1

        longs = r['ranked_tickers'][:N_LONG]
        shorts = r['ranked_tickers'][-N_SHORT:]
        long_ret = np.mean([r['sector_returns'].get(t, 0) for t in longs])
        short_ret = np.mean([r['sector_returns'].get(t, 0) for t in shorts])

        # Determine sizing
        if bullish_count >= 2:
            # Bullish regime: full size, tilt long
            period_ret = 1.5 * long_ret - 0.75 * short_ret
        elif bearish_count >= 2:
            # Bearish regime: full size, tilt short
            period_ret = 0.75 * long_ret - 1.5 * short_ret
        else:
            # Neutral/mixed: half size
            period_ret = 0.5 * (long_ret - short_ret)

        returns.append({'date': date, 'return': period_ret, 'spy_ret': r['spy_ret']})
    return returns


def run_variant_H(rankings, macro_signals):
    """Adaptive Concentration: high score dispersion -> top2/bot2, low -> top4/bot4."""
    # Pre-compute score spread percentiles for adaptive thresholds
    all_spreads = []
    for r in rankings:
        sc = np.array(r['scores'])
        if len(sc) > 1:
            all_spreads.append(sc.max() - sc.min())
    spread_p33 = np.percentile(all_spreads, 33) if all_spreads else 0.01
    spread_p67 = np.percentile(all_spreads, 67) if all_spreads else 0.02

    returns = []
    for r in rankings:
        scores = np.array(r['scores'])
        score_spread = scores.max() - scores.min() if len(scores) > 1 else 0

        if score_spread > spread_p67:
            # High dispersion: concentrate in top-2/bottom-2
            n_long = 2
            n_short = 2
        elif score_spread < spread_p33:
            # Low dispersion: diversify to top-4/bottom-4
            n_long = min(4, len(r['ranked_tickers']) // 2)
            n_short = min(4, len(r['ranked_tickers']) // 2)
        else:
            # Normal: standard 3/3
            n_long = N_LONG
            n_short = N_SHORT

        longs = r['ranked_tickers'][:n_long]
        shorts = r['ranked_tickers'][-n_short:]
        long_ret = np.mean([r['sector_returns'].get(t, 0) for t in longs])
        short_ret = np.mean([r['sector_returns'].get(t, 0) for t in shorts])
        period_ret = long_ret - short_ret

        returns.append({'date': r['date'], 'return': period_ret, 'spy_ret': r['spy_ret'],
                        'n_long': n_long, 'n_short': n_short})
    return returns

# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
def compute_metrics(returns_list, label=''):
    """Compute performance metrics from a list of period returns."""
    if not returns_list:
        return {}

    rets = np.array([r['return'] for r in returns_list])
    spy_rets = np.array([r['spy_ret'] for r in returns_list])
    dates = [r['date'] for r in returns_list]

    n = len(rets)
    if n < 2:
        return {}

    # Annualized metrics (monthly periods ~ 12 per year)
    periods_per_year = 252 / REBAL_PERIOD  # ~12
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1)

    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0.0

    # Sortino
    downside_rets = rets[rets < 0]
    downside_std = np.std(downside_rets, ddof=1) if len(downside_rets) > 1 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0.0

    # Win rate
    win_rate = np.mean(rets > 0) * 100

    # Profit factor
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = np.abs(np.sum(rets[rets < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    equity = STARTING_CAPITAL * np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = np.min(drawdown) * 100  # as percentage

    # Total return
    total_ret = (equity[-1] / STARTING_CAPITAL - 1) * 100

    # CAGR
    n_years = n / periods_per_year
    cagr = ((equity[-1] / STARTING_CAPITAL) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # Beta to SPY
    if np.std(spy_rets) > 0:
        beta = np.cov(rets, spy_rets)[0, 1] / np.var(spy_rets)
    else:
        beta = 0.0

    return {
        'label': label,
        'n_periods': n,
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'profit_factor': round(profit_factor, 2),
        'win_rate': round(win_rate, 1),
        'max_dd_pct': round(max_dd, 1),
        'total_return_pct': round(total_ret, 1),
        'cagr_pct': round(cagr, 1),
        'beta_to_spy': round(beta, 3),
        'mean_monthly_ret': round(mean_ret * 100, 2),
        'std_monthly_ret': round(std_ret * 100, 2),
    }

# ---------------------------------------------------------------------------
# 5-Gate Adversarial Validation
# ---------------------------------------------------------------------------
def adversarial_validation(rankings, macro_signals, variant_func, variant_name, n_perms=300):
    """Run 5-gate adversarial validation on a variant."""
    fprint(f"\n[ADVERSARIAL] Running 5-gate validation for {variant_name} ...")
    t0 = time.time()

    actual_returns = variant_func(rankings, macro_signals)
    actual_rets = np.array([r['return'] for r in actual_returns])
    spy_rets = np.array([r['spy_ret'] for r in actual_returns])
    dates = [r['date'] for r in actual_returns]
    n = len(actual_rets)

    if n < 10:
        fprint(f"  [SKIP] Only {n} periods, not enough for validation")
        return {'gates_passed': 0, 'gates_total': 5, 'details': {}}

    periods_per_year = 252 / REBAL_PERIOD
    actual_sharpe = (np.mean(actual_rets) / np.std(actual_rets, ddof=1)) * np.sqrt(periods_per_year) if np.std(actual_rets) > 0 else 0

    results = {}

    # Gate 1: Permutation test (shuffle rankings, recompute returns)
    fprint(f"  Gate 1: Permutation test ({n_perms} trials) ...")
    perm_sharpes = []
    for _ in range(n_perms):
        # Shuffle sector rankings within each rebalance period
        shuffled_rankings = []
        for r in rankings:
            r_copy = r.copy()
            tickers = list(r_copy['ranked_tickers'])
            np.random.shuffle(tickers)
            r_copy['ranked_tickers'] = tickers
            # Keep scores in original order but with shuffled tickers
            r_copy['scores'] = list(np.random.permutation(r_copy['scores']))
            shuffled_rankings.append(r_copy)

        perm_returns = variant_func(shuffled_rankings, macro_signals)
        perm_rets = np.array([r['return'] for r in perm_returns])
        if np.std(perm_rets) > 0:
            perm_sharpe = (np.mean(perm_rets) / np.std(perm_rets, ddof=1)) * np.sqrt(periods_per_year)
        else:
            perm_sharpe = 0
        perm_sharpes.append(perm_sharpe)

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= actual_sharpe)
    gate1_pass = p_value < 0.05
    results['gate1_permutation'] = {
        'pass': gate1_pass,
        'p_value': round(p_value, 4),
        'actual_sharpe': round(actual_sharpe, 2),
        'perm_mean_sharpe': round(np.mean(perm_sharpes), 2),
        'perm_95th': round(np.percentile(perm_sharpes, 95), 2),
    }
    fprint(f"    {'PASS' if gate1_pass else 'FAIL'}: p={p_value:.4f}, actual={actual_sharpe:.2f}, perm_95th={np.percentile(perm_sharpes, 95):.2f}")

    # Gate 2: Regime stability (bull vs bear)
    fprint("  Gate 2: Regime stability ...")
    bull_mask = spy_rets > 0
    bear_mask = spy_rets <= 0

    if bull_mask.sum() > 2 and bear_mask.sum() > 2:
        bull_rets = actual_rets[bull_mask]
        bear_rets = actual_rets[bear_mask]
        bull_sharpe = (np.mean(bull_rets) / np.std(bull_rets, ddof=1)) * np.sqrt(periods_per_year) if np.std(bull_rets) > 0 else 0
        bear_sharpe = (np.mean(bear_rets) / np.std(bear_rets, ddof=1)) * np.sqrt(periods_per_year) if np.std(bear_rets) > 0 else 0
        max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
        regime_diff = abs(bull_sharpe - bear_sharpe) / max_abs if max_abs > 0 else 0
        gate2_pass = regime_diff < 0.50
        results['gate2_regime'] = {
            'pass': gate2_pass,
            'bull_sharpe': round(bull_sharpe, 2),
            'bear_sharpe': round(bear_sharpe, 2),
            'regime_diff_ratio': round(regime_diff, 3),
        }
        fprint(f"    {'PASS' if gate2_pass else 'FAIL'}: bull={bull_sharpe:.2f}, bear={bear_sharpe:.2f}, diff_ratio={regime_diff:.3f}")
    else:
        results['gate2_regime'] = {'pass': False, 'note': 'insufficient data'}
        gate2_pass = False
        fprint("    FAIL: insufficient regime data")

    # Gate 3: Sub-period stability (quarters)
    fprint("  Gate 3: Sub-period stability ...")
    n_quarters = max(4, n // (3 * int(periods_per_year / 4)))
    chunk_size = n // 4
    quarter_sharpes = []
    for q in range(4):
        start_idx = q * chunk_size
        end_idx = start_idx + chunk_size if q < 3 else n
        q_rets = actual_rets[start_idx:end_idx]
        if len(q_rets) > 1 and np.std(q_rets) > 0:
            q_sharpe = (np.mean(q_rets) / np.std(q_rets, ddof=1)) * np.sqrt(periods_per_year)
        else:
            q_sharpe = 0
        quarter_sharpes.append(q_sharpe)

    positive_quarters = sum(1 for s in quarter_sharpes if s > 0)
    gate3_pass = positive_quarters >= 3
    results['gate3_subperiod'] = {
        'pass': gate3_pass,
        'quarter_sharpes': [round(s, 2) for s in quarter_sharpes],
        'positive_quarters': positive_quarters,
    }
    fprint(f"    {'PASS' if gate3_pass else 'FAIL'}: {positive_quarters}/4 quarters positive, sharpes={[round(s,2) for s in quarter_sharpes]}")

    # Gate 4: Outlier removal (drop top/bottom 5%)
    fprint("  Gate 4: Outlier removal ...")
    sorted_rets = np.sort(actual_rets)
    trim = max(1, int(0.05 * n))
    trimmed_rets = sorted_rets[trim:-trim] if trim < n // 2 else sorted_rets
    if len(trimmed_rets) > 1 and np.std(trimmed_rets) > 0:
        trimmed_sharpe = (np.mean(trimmed_rets) / np.std(trimmed_rets, ddof=1)) * np.sqrt(periods_per_year)
    else:
        trimmed_sharpe = 0
    gate4_pass = trimmed_sharpe > 0.5
    results['gate4_outlier'] = {
        'pass': gate4_pass,
        'trimmed_sharpe': round(trimmed_sharpe, 2),
        'n_trimmed': len(trimmed_rets),
    }
    fprint(f"    {'PASS' if gate4_pass else 'FAIL'}: trimmed Sharpe={trimmed_sharpe:.2f}")

    # Gate 5: Yearly consistency
    fprint("  Gate 5: Yearly consistency ...")
    date_arr = pd.to_datetime(dates)
    years = sorted(set(d.year for d in date_arr))
    year_sharpes = []
    for y in years:
        mask = np.array([d.year == y for d in date_arr])
        y_rets = actual_rets[mask]
        if len(y_rets) > 2 and np.std(y_rets) > 0:
            y_sharpe = (np.mean(y_rets) / np.std(y_rets, ddof=1)) * np.sqrt(periods_per_year)
        else:
            y_sharpe = 0
        year_sharpes.append((y, round(y_sharpe, 2)))

    positive_years = sum(1 for _, s in year_sharpes if s > 0)
    gate5_pass = positive_years >= len(years) * 0.6  # at least 60% of years positive
    results['gate5_yearly'] = {
        'pass': gate5_pass,
        'year_sharpes': year_sharpes,
        'positive_years': positive_years,
        'total_years': len(years),
    }
    fprint(f"    {'PASS' if gate5_pass else 'FAIL'}: {positive_years}/{len(years)} years positive")

    gates_passed = sum(1 for g in [gate1_pass, gate2_pass, gate3_pass, gate4_pass, gate5_pass] if g)
    validated = gates_passed >= 4

    elapsed = time.time() - t0
    fprint(f"  VERDICT: {gates_passed}/5 gates passed -> {'VALIDATED' if validated else 'FAILED'} ({elapsed:.1f}s)")

    return {
        'gates_passed': gates_passed,
        'gates_total': 5,
        'validated': validated,
        'details': results,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    fprint("=" * 80)
    fprint("MACRO TIMING OVERLAY v1 — 8 Variant Comparison")
    fprint("=" * 80)
    t_start = time.time()

    # Download data
    close = download_data(start='2018-01-01')

    # Compute features
    features_df = compute_features(close)

    # Compute macro signals
    macro_signals = compute_macro_signals(close)

    # Walk-forward LGBM ranking (runs ONCE, shared by all variants)
    rankings = walk_forward_ranking(features_df, close)

    if len(rankings) < 10:
        fprint("[ERROR] Not enough ranking periods. Exiting.")
        return

    # --- Run all 8 variants ---
    variant_funcs = {
        'A': run_variant_A,
        'B': run_variant_B,
        'C': run_variant_C,
        'D': run_variant_D,
        'E': run_variant_E,
        'F': run_variant_F,
        'G': run_variant_G,
        'H': run_variant_H,
    }

    all_results = {}
    all_returns = {}
    all_adversarial = {}

    for var_key in sorted(variant_funcs.keys()):
        var_name = VARIANT_NAMES[var_key]
        fprint(f"\n{'='*60}")
        fprint(f"Variant {var_key}: {var_name}")
        fprint(f"{'='*60}")

        returns = variant_funcs[var_key](rankings, macro_signals)
        metrics = compute_metrics(returns, label=f"{var_key}: {var_name}")
        all_results[var_key] = metrics
        all_returns[var_key] = returns

        fprint(f"  Sharpe: {metrics.get('sharpe', 'N/A')}")
        fprint(f"  Sortino: {metrics.get('sortino', 'N/A')}")
        fprint(f"  PF: {metrics.get('profit_factor', 'N/A')}")
        fprint(f"  WR: {metrics.get('win_rate', 'N/A')}%")
        fprint(f"  MDD: {metrics.get('max_dd_pct', 'N/A')}%")
        fprint(f"  Total Return: {metrics.get('total_return_pct', 'N/A')}%")
        fprint(f"  CAGR: {metrics.get('cagr_pct', 'N/A')}%")
        fprint(f"  Beta-to-SPY: {metrics.get('beta_to_spy', 'N/A')}")

        # Run adversarial validation
        adv = adversarial_validation(rankings, macro_signals, variant_funcs[var_key], var_name)
        all_adversarial[var_key] = adv

    # --- Summary Table ---
    fprint("\n" + "=" * 100)
    fprint("SUMMARY TABLE")
    fprint("=" * 100)
    header = f"{'Var':<4} {'Name':<32} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR%':>6} {'MDD%':>7} {'TotRet%':>9} {'CAGR%':>7} {'Beta':>6} {'Gates':>6} {'Valid':>6}"
    fprint(header)
    fprint("-" * 100)

    for var_key in sorted(all_results.keys()):
        m = all_results[var_key]
        a = all_adversarial[var_key]
        fprint(f"{var_key:<4} {VARIANT_NAMES[var_key]:<32} {m.get('sharpe',''):>7} {m.get('sortino',''):>8} "
               f"{m.get('profit_factor',''):>6} {m.get('win_rate',''):>6} {m.get('max_dd_pct',''):>7} "
               f"{m.get('total_return_pct',''):>9} {m.get('cagr_pct',''):>7} {m.get('beta_to_spy',''):>6} "
               f"{a.get('gates_passed','')}/{a.get('gates_total','')}"
               f"{'  YES' if a.get('validated', False) else '   NO':>6}")

    # --- Determine best variant ---
    fprint("\n" + "=" * 80)
    best_sharpe_var = max(all_results.keys(), key=lambda k: all_results[k].get('sharpe', 0))
    best_sortino_var = max(all_results.keys(), key=lambda k: all_results[k].get('sortino', 0))

    # Best risk-adjusted with validation
    validated_vars = [k for k in all_results.keys() if all_adversarial[k].get('validated', False)]
    if validated_vars:
        best_validated = max(validated_vars, key=lambda k: all_results[k].get('sharpe', 0))
        fprint(f"BEST VALIDATED VARIANT: {best_validated} ({VARIANT_NAMES[best_validated]}) — Sharpe {all_results[best_validated]['sharpe']}")
    else:
        fprint("WARNING: No variant passed adversarial validation (4/5 gates)")

    fprint(f"Best Sharpe overall: {best_sharpe_var} ({VARIANT_NAMES[best_sharpe_var]}) — {all_results[best_sharpe_var]['sharpe']}")
    fprint(f"Best Sortino overall: {best_sortino_var} ({VARIANT_NAMES[best_sortino_var]}) — {all_results[best_sortino_var]['sortino']}")

    # --- Improvement over baseline ---
    baseline = all_results.get('A', {})
    if baseline:
        fprint(f"\nIMPROVEMENT OVER BASELINE (A):")
        for var_key in sorted(all_results.keys()):
            if var_key == 'A':
                continue
            m = all_results[var_key]
            sharpe_diff = m.get('sharpe', 0) - baseline.get('sharpe', 0)
            sortino_diff = m.get('sortino', 0) - baseline.get('sortino', 0)
            dd_diff = m.get('max_dd_pct', 0) - baseline.get('max_dd_pct', 0)
            fprint(f"  {var_key} ({VARIANT_NAMES[var_key]}): Sharpe {sharpe_diff:+.2f}, "
                   f"Sortino {sortino_diff:+.2f}, MDD {dd_diff:+.1f}%")

    # --- Save results ---
    fprint("\n[SAVE] Saving results ...")

    # Save summary JSON
    summary = {
        'experiment': 'macro_timing_overlay_v1',
        'timestamp': datetime.now().isoformat(),
        'data_range': f"{close.index[0].date()} to {close.index[-1].date()}",
        'n_rebalance_periods': len(rankings),
        'train_window': TRAIN_WINDOW,
        'rebal_period': REBAL_PERIOD,
        'starting_capital': STARTING_CAPITAL,
        'n_features': len(FEATURE_COLS),
        'variants': {},
    }
    for var_key in sorted(all_results.keys()):
        summary['variants'][var_key] = {
            'name': VARIANT_NAMES[var_key],
            'metrics': all_results[var_key],
            'adversarial': all_adversarial[var_key],
        }

    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save equity curves CSV
    equity_curves = {}
    for var_key, returns in all_returns.items():
        rets = [r['return'] for r in returns]
        equity = STARTING_CAPITAL * np.cumprod(1 + np.array(rets))
        dates = [r['date'] for r in returns]
        equity_curves[f'{var_key}_equity'] = pd.Series(equity, index=dates)

    eq_df = pd.DataFrame(equity_curves)
    eq_df.to_csv(OUTPUT_DIR / 'equity_curves.csv')

    # Save period returns CSV
    for var_key, returns in all_returns.items():
        ret_df = pd.DataFrame(returns)
        ret_df.to_csv(OUTPUT_DIR / f'returns_{var_key}.csv', index=False)

    fprint(f"[SAVE] Results saved to {OUTPUT_DIR}/")

    # --- MLflow logging ---
    if USE_MLFLOW:
        fprint("[MLFLOW] Logging to MLflow ...")
        try:
            with mlflow.start_run(run_name="macro_timing_overlay_v1"):
                # Log parameters
                mlflow.log_param("train_window", TRAIN_WINDOW)
                mlflow.log_param("rebal_period", REBAL_PERIOD)
                mlflow.log_param("starting_capital", STARTING_CAPITAL)
                mlflow.log_param("n_features", len(FEATURE_COLS))
                mlflow.log_param("n_rebalance_periods", len(rankings))
                mlflow.log_param("data_start", str(close.index[0].date()))
                mlflow.log_param("data_end", str(close.index[-1].date()))
                mlflow.log_param("n_variants", len(variant_funcs))

                # Log metrics for each variant
                for var_key in sorted(all_results.keys()):
                    m = all_results[var_key]
                    a = all_adversarial[var_key]
                    prefix = f"var_{var_key}_"
                    mlflow.log_metric(f"{prefix}sharpe", m.get('sharpe', 0))
                    mlflow.log_metric(f"{prefix}sortino", m.get('sortino', 0))
                    mlflow.log_metric(f"{prefix}profit_factor", m.get('profit_factor', 0))
                    mlflow.log_metric(f"{prefix}win_rate", m.get('win_rate', 0))
                    mlflow.log_metric(f"{prefix}max_dd_pct", m.get('max_dd_pct', 0))
                    mlflow.log_metric(f"{prefix}total_return_pct", m.get('total_return_pct', 0))
                    mlflow.log_metric(f"{prefix}cagr_pct", m.get('cagr_pct', 0))
                    mlflow.log_metric(f"{prefix}beta_to_spy", m.get('beta_to_spy', 0))
                    mlflow.log_metric(f"{prefix}gates_passed", a.get('gates_passed', 0))
                    mlflow.log_metric(f"{prefix}validated", 1 if a.get('validated', False) else 0)

                # Log artifacts
                mlflow.log_artifact(str(OUTPUT_DIR / 'summary.json'))
                mlflow.log_artifact(str(OUTPUT_DIR / 'equity_curves.csv'))

            fprint("[MLFLOW] Logged successfully.")
        except Exception as e:
            fprint(f"[MLFLOW] Error logging: {e}")

    elapsed_total = time.time() - t_start
    fprint(f"\n{'='*80}")
    fprint(f"COMPLETE. Total runtime: {elapsed_total:.1f}s ({elapsed_total/60:.1f}m)")
    fprint(f"{'='*80}")


if __name__ == '__main__':
    main()
