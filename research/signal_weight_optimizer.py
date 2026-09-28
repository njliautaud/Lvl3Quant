#!/usr/bin/env python3
"""
Signal Weight Optimizer — ML-based optimal signal combination weights
=====================================================================
Uses historical price data (Jan 2026 - Aug 2026) to find which signal
combinations best predict 5-day forward sector ETF returns.

Signals reconstructed from price/volume data:
  1. momentum_20d   — 20-day return momentum
  2. rsi_5d         — 5-day RSI (oversold/overbought)
  3. volume_spike   — Volume surge with directional move
  4. trend_50sma    — Price vs 50-day SMA
  5. breadth_sector — % of sector ETFs above their 20-SMA
  6. vol_regime     — 20d vs 60d volatility (contracting/expanding)
  7. lgbm_rank      — Relative momentum rank across sectors
  8. market_neutral — Sector return minus SPY return (relative strength)
  9. momentum_burst — Short-term (5d) momentum burst
  10. trend_strength — R-squared of 20d linear regression
  11. vix_regime     — VIX level bucket (calm/normal/elevated/crisis)
  12. dispersion     — Cross-sector return dispersion

Train: Jan-Apr 2026 | Test: May-Aug 2026
Model: LightGBM gradient boosted trees
Output: feature importance, optimal confluence, best combos
"""

import json
import warnings
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from datetime import datetime
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC', 'XLY']
MARKET_TICKERS = ['SPY', 'QQQ', 'IWM', 'RSP', 'TLT', 'GLD']
VIX_TICKER = '^VIX'
ALL_TICKERS = MARKET_TICKERS + SECTOR_ETFS + [VIX_TICKER]

# Date ranges — 6 months buffer before for lookback
DATA_START = '2025-06-01'   # Buffer for 200d SMA
DATA_END = '2026-08-17'
TRAIN_START = '2026-01-02'
TRAIN_END = '2026-04-30'
TEST_START = '2026-05-01'
TEST_END = '2026-08-17'

FORWARD_DAYS = 5  # Predict 5-day forward return
OUTPUT_PATH = Path('/home/jupiter/Lvl3Quant/research/signal_weight_optimization.json')

SIGNAL_NAMES = [
    'momentum_20d', 'rsi_5d', 'volume_spike', 'trend_50sma',
    'breadth_sector', 'vol_regime_20v60', 'lgbm_rank',
    'market_neutral', 'momentum_burst_5d', 'trend_strength_r2',
    'vix_regime_score', 'dispersion_score'
]


# ── Data Download ─────────────────────────────────────────────────────────
def download_data():
    """Download OHLCV for all tickers."""
    print("Downloading price data...")
    data = {}
    for t in ALL_TICKERS:
        try:
            df = yf.download(t, start=DATA_START, end=DATA_END, progress=False, auto_adjust=True)
            if len(df) > 50:
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df
                print(f"  {t}: {len(df)} bars")
            else:
                print(f"  {t}: SKIPPED ({len(df)} bars)")
        except Exception as e:
            print(f"  {t}: FAILED ({e})")
    return data


# ── Signal Generators ─────────────────────────────────────────────────────
def calc_momentum_20d(close):
    """20-day return scaled to [-1, +1]."""
    ret = close.pct_change(20)
    return (ret / 0.15).clip(-1, 1)


def calc_rsi(close, period=5):
    """5-day RSI → signal: <30=+1, >70=-1, else linear scale."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Continuous signal instead of binary
    signal = ((50 - rsi) / 50).clip(-1, 1)
    return signal


def calc_volume_spike(close, volume, period=20):
    """Volume ratio × direction → continuous signal."""
    vol_ratio = volume / volume.rolling(period).mean()
    daily_ret = close.pct_change()
    # Scale: vol_ratio > 1.5 and directional
    signal = (vol_ratio - 1.0).clip(0, 2) * np.sign(daily_ret)
    return signal.clip(-1, 1)


def calc_trend_50sma(close):
    """Distance from 50-SMA, normalized."""
    sma50 = close.rolling(50).mean()
    dist = (close - sma50) / sma50
    return (dist / 0.05).clip(-1, 1)


def calc_breadth(sector_closes):
    """% of sector ETFs above their 20-SMA → [-1, +1]."""
    above_sma = pd.DataFrame()
    for t, close in sector_closes.items():
        sma20 = close.rolling(20).mean()
        above_sma[t] = (close > sma20).astype(float)
    pct_above = above_sma.mean(axis=1)
    return (pct_above - 0.5) * 2  # Scale 0-1 → -1 to +1


def calc_vol_regime(close):
    """20d vol vs 60d vol → contracting=+1, expanding=-1."""
    ret = close.pct_change()
    vol20 = ret.rolling(20).std()
    vol60 = ret.rolling(60).std()
    ratio = vol20 / vol60.replace(0, np.nan)
    return (1 - ratio).clip(-1, 1)


def calc_lgbm_rank(sector_closes):
    """Cross-sectional momentum rank. Top → +1, Bottom → -1."""
    ret_20d = pd.DataFrame()
    for t, close in sector_closes.items():
        ret_20d[t] = close.pct_change(20)
    # Rank across sectors for each day
    ranks = ret_20d.rank(axis=1, pct=True)
    return (ranks - 0.5) * 2  # Scale to [-1, +1]


def calc_market_neutral(sector_close, spy_close):
    """Sector return - SPY return (relative strength)."""
    sector_ret = sector_close.pct_change(10)
    spy_ret = spy_close.pct_change(10)
    diff = sector_ret - spy_ret
    return (diff / 0.05).clip(-1, 1)


def calc_momentum_burst(close):
    """5-day momentum burst."""
    ret5 = close.pct_change(5)
    return (ret5 / 0.05).clip(-1, 1)


def calc_trend_strength(close, period=20):
    """R-squared of linear regression on last 20 days."""
    r2_series = pd.Series(np.nan, index=close.index)
    values = close.values
    for i in range(period, len(values)):
        y = values[i - period:i]
        if np.any(np.isnan(y)):
            continue
        x = np.arange(period)
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        direction = np.sign(slope)
        r2_series.iloc[i] = r_value ** 2 * direction
    return r2_series.clip(-1, 1)


def calc_vix_regime(vix_series):
    """VIX level → regime score. Low VIX = bullish = +1."""
    signal = pd.Series(0.0, index=vix_series.index)
    signal[vix_series < 15] = 1.0
    signal[(vix_series >= 15) & (vix_series < 20)] = 0.5
    signal[(vix_series >= 20) & (vix_series < 25)] = -0.25
    signal[(vix_series >= 25) & (vix_series < 30)] = -0.5
    signal[vix_series >= 30] = -1.0
    return signal


def calc_dispersion(sector_closes):
    """Cross-sector return dispersion (std of sector 5d returns)."""
    ret_5d = pd.DataFrame()
    for t, close in sector_closes.items():
        ret_5d[t] = close.pct_change(5)
    disp = ret_5d.std(axis=1)
    # High dispersion = more rotation opportunity = +1
    return ((disp / disp.rolling(63).mean()) - 1).clip(-1, 1)


# ── Build Feature Matrix ─────────────────────────────────────────────────
def build_feature_matrix(data):
    """Build features + target for each sector ETF on each trading day."""
    print("\nBuilding feature matrix...")

    # Get common index
    spy_close = data['SPY']['Close']
    vix_col = VIX_TICKER if VIX_TICKER in data else 'VIX'
    vix_series = data[vix_col]['Close'] if vix_col in data else pd.Series(18.0, index=spy_close.index)

    sector_closes = {}
    for t in SECTOR_ETFS:
        if t in data:
            sector_closes[t] = data[t]['Close']

    # Pre-compute cross-sectional signals
    breadth = calc_breadth(sector_closes)
    lgbm_ranks = calc_lgbm_rank(sector_closes)
    vix_regime = calc_vix_regime(vix_series)
    dispersion = calc_dispersion(sector_closes)

    # SPY regime features
    spy_ret_5d = spy_close.pct_change(5)
    spy_ret_21d = spy_close.pct_change(21)
    spy_above_200sma = (spy_close > spy_close.rolling(200).mean()).astype(float)

    rows = []
    for ticker in SECTOR_ETFS:
        if ticker not in data:
            continue
        df = data[ticker]
        close = df['Close']
        volume = df['Volume']

        # Individual signals
        s_momentum = calc_momentum_20d(close)
        s_rsi = calc_rsi(close)
        s_volume = calc_volume_spike(close, volume)
        s_trend = calc_trend_50sma(close)
        s_vol_regime = calc_vol_regime(close)
        s_mkt_neutral = calc_market_neutral(close, spy_close)
        s_mom_burst = calc_momentum_burst(close)
        s_trend_str = calc_trend_strength(close)

        # Forward return (target)
        fwd_return = close.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS) * 100  # In percent

        for i in range(len(close)):
            date = close.index[i]
            if pd.isna(fwd_return.iloc[i]):
                continue

            # Collect all signal values
            sigs = {
                'momentum_20d': s_momentum.iloc[i] if not pd.isna(s_momentum.iloc[i]) else 0,
                'rsi_5d': s_rsi.iloc[i] if not pd.isna(s_rsi.iloc[i]) else 0,
                'volume_spike': s_volume.iloc[i] if not pd.isna(s_volume.iloc[i]) else 0,
                'trend_50sma': s_trend.iloc[i] if not pd.isna(s_trend.iloc[i]) else 0,
                'breadth_sector': breadth.loc[date] if date in breadth.index and not pd.isna(breadth.loc[date]) else 0,
                'vol_regime_20v60': s_vol_regime.iloc[i] if not pd.isna(s_vol_regime.iloc[i]) else 0,
                'lgbm_rank': lgbm_ranks[ticker].loc[date] if ticker in lgbm_ranks.columns and date in lgbm_ranks.index and not pd.isna(lgbm_ranks[ticker].loc[date]) else 0,
                'market_neutral': s_mkt_neutral.iloc[i] if not pd.isna(s_mkt_neutral.iloc[i]) else 0,
                'momentum_burst_5d': s_mom_burst.iloc[i] if not pd.isna(s_mom_burst.iloc[i]) else 0,
                'trend_strength_r2': s_trend_str.iloc[i] if not pd.isna(s_trend_str.iloc[i]) else 0,
                'vix_regime_score': vix_regime.loc[date] if date in vix_regime.index and not pd.isna(vix_regime.loc[date]) else 0,
                'dispersion_score': dispersion.loc[date] if date in dispersion.index and not pd.isna(dispersion.loc[date]) else 0,
            }

            # Derived features
            bullish_count = sum(1 for v in sigs.values() if v > 0.2)
            bearish_count = sum(1 for v in sigs.values() if v < -0.2)
            confluence = bullish_count - bearish_count
            avg_signal = np.mean(list(sigs.values()))
            max_signal = max(sigs.values())
            min_signal = min(sigs.values())
            signal_std = np.std(list(sigs.values()))

            # Interaction features (pairs of top signals)
            sigs['momentum_x_trend'] = sigs['momentum_20d'] * sigs['trend_50sma']
            sigs['momentum_x_breadth'] = sigs['momentum_20d'] * sigs['breadth_sector']
            sigs['rsi_x_vol_regime'] = sigs['rsi_5d'] * sigs['vol_regime_20v60']
            sigs['trend_x_vix'] = sigs['trend_50sma'] * sigs['vix_regime_score']
            sigs['mkt_neutral_x_momentum'] = sigs['market_neutral'] * sigs['momentum_burst_5d']
            sigs['lgbm_x_trend_str'] = sigs['lgbm_rank'] * sigs['trend_strength_r2']

            # Aggregate features
            sigs['bullish_count'] = bullish_count
            sigs['bearish_count'] = bearish_count
            sigs['confluence_score'] = confluence
            sigs['avg_signal'] = avg_signal
            sigs['max_signal'] = max_signal
            sigs['min_signal'] = min_signal
            sigs['signal_std'] = signal_std

            # Market regime features
            sigs['spy_ret_5d'] = spy_ret_5d.loc[date] * 100 if date in spy_ret_5d.index and not pd.isna(spy_ret_5d.loc[date]) else 0
            sigs['spy_ret_21d'] = spy_ret_21d.loc[date] * 100 if date in spy_ret_21d.index and not pd.isna(spy_ret_21d.loc[date]) else 0
            sigs['spy_above_200sma'] = spy_above_200sma.loc[date] if date in spy_above_200sma.index else 0
            sigs['vix_level'] = float(vix_series.loc[date]) if date in vix_series.index and not pd.isna(vix_series.loc[date]) else 18.0

            # Meta
            sigs['ticker'] = ticker
            sigs['date'] = date
            sigs['fwd_return_5d'] = fwd_return.iloc[i]

            rows.append(sigs)

    df = pd.DataFrame(rows)
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Sectors: {df['ticker'].nunique()}")
    return df


# ── Train LightGBM ───────────────────────────────────────────────────────
def train_model(df):
    """Train LightGBM on train period, evaluate on test period."""
    print("\nTraining LightGBM model...")

    feature_cols = [c for c in df.columns if c not in ['ticker', 'date', 'fwd_return_5d']]

    train_mask = (df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)
    test_mask = (df['date'] >= TEST_START) & (df['date'] <= TEST_END)

    X_train = df.loc[train_mask, feature_cols].astype(float)
    y_train = df.loc[train_mask, 'fwd_return_5d'].astype(float)
    X_test = df.loc[test_mask, feature_cols].astype(float)
    y_test = df.loc[test_mask, 'fwd_return_5d'].astype(float)

    print(f"  Train: {len(X_train)} rows ({TRAIN_START} to {TRAIN_END})")
    print(f"  Test:  {len(X_test)} rows ({TEST_START} to {TEST_END})")

    if len(X_train) < 50 or len(X_test) < 50:
        print("  ERROR: Not enough data for train/test split")
        return None, None, None, None, None

    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # IC (rank correlation)
    train_ic = stats.spearmanr(y_train, y_pred_train)[0]
    test_ic = stats.spearmanr(y_test, y_pred_test)[0]

    print(f"\n  Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}, IC: {train_ic:.4f}")
    print(f"  Test  RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}, IC: {test_ic:.4f}")

    metrics = {
        'train_rmse': round(train_rmse, 4),
        'test_rmse': round(test_rmse, 4),
        'train_r2': round(train_r2, 4),
        'test_r2': round(test_r2, 4),
        'train_ic': round(train_ic, 4),
        'test_ic': round(test_ic, 4),
        'num_rounds': model.best_iteration,
    }

    return model, feature_cols, metrics, df.loc[test_mask], y_pred_test


# ── Feature Importance Analysis ──────────────────────────────────────────
def analyze_importance(model, feature_cols):
    """Extract and rank feature importances."""
    print("\nAnalyzing feature importance...")

    # Gain-based importance
    gain_imp = model.feature_importance(importance_type='gain')
    split_imp = model.feature_importance(importance_type='split')

    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'gain': gain_imp,
        'split': split_imp,
    })
    imp_df['gain_pct'] = imp_df['gain'] / imp_df['gain'].sum() * 100
    imp_df['split_pct'] = imp_df['split'] / imp_df['split'].sum() * 100
    imp_df = imp_df.sort_values('gain_pct', ascending=False)

    print("\n  Top 15 features by gain importance:")
    for _, row in imp_df.head(15).iterrows():
        print(f"    {row['feature']:30s}  gain={row['gain_pct']:6.2f}%  split={row['split_pct']:5.2f}%")

    # Separate core signals from derived/interaction features
    core_signals = [c for c in SIGNAL_NAMES if c in feature_cols]
    core_imp = imp_df[imp_df['feature'].isin(core_signals)].copy()
    core_imp = core_imp.sort_values('gain_pct', ascending=False)

    print("\n  Core signal importance ranking:")
    for i, (_, row) in enumerate(core_imp.iterrows(), 1):
        print(f"    {i:2d}. {row['feature']:30s}  gain={row['gain_pct']:6.2f}%")

    return imp_df, core_imp


# ── Confluence Analysis ──────────────────────────────────────────────────
def analyze_confluence(df_test, y_pred):
    """Test if confluence count predicts better returns."""
    print("\nAnalyzing confluence thresholds...")

    df_test = df_test.copy()
    df_test['prediction'] = y_pred

    results = {}
    for threshold in range(1, 11):
        mask = df_test['bullish_count'] >= threshold
        if mask.sum() < 10:
            continue

        subset = df_test[mask]
        avg_ret = subset['fwd_return_5d'].mean()
        median_ret = subset['fwd_return_5d'].median()
        wr = (subset['fwd_return_5d'] > 0).mean()
        n = len(subset)
        pct_of_total = n / len(df_test) * 100

        # Average prediction
        avg_pred = subset['prediction'].mean()

        # Sharpe-like (mean / std)
        ret_std = subset['fwd_return_5d'].std()
        sharpe_like = avg_ret / ret_std * np.sqrt(252 / 5) if ret_std > 0 else 0

        results[threshold] = {
            'n_signals': n,
            'pct_universe': round(pct_of_total, 1),
            'avg_return_5d_pct': round(avg_ret, 3),
            'median_return_5d_pct': round(median_ret, 3),
            'win_rate': round(wr, 3),
            'sharpe_annualized': round(sharpe_like, 2),
            'avg_prediction': round(avg_pred, 3),
        }

        print(f"    ≥{threshold} bullish signals: n={n:4d} ({pct_of_total:5.1f}%)  "
              f"avgRet={avg_ret:+.3f}%  WR={wr:.1%}  Sharpe={sharpe_like:.2f}")

    # Find optimal threshold (best Sharpe with ≥5% of universe)
    valid = {k: v for k, v in results.items() if v['pct_universe'] >= 5}
    if valid:
        optimal = max(valid.keys(), key=lambda k: valid[k]['sharpe_annualized'])
    else:
        optimal = min(results.keys()) if results else 5

    print(f"\n  OPTIMAL CONFLUENCE THRESHOLD: ≥{optimal} bullish signals")
    return results, optimal


# ── Signal Combination Analysis ──────────────────────────────────────────
def analyze_combinations(df_test, y_pred):
    """Find best 2-signal and 3-signal combinations."""
    print("\nAnalyzing signal combinations...")

    df_test = df_test.copy()
    df_test['prediction'] = y_pred
    core_signals = [c for c in SIGNAL_NAMES if c in df_test.columns]

    def eval_combo(signals, threshold=0.2):
        """When all signals in combo are > threshold, what's avg forward return?"""
        mask = pd.Series(True, index=df_test.index)
        for s in signals:
            mask &= df_test[s] > threshold
        n = mask.sum()
        if n < 10:
            return None
        subset = df_test[mask]
        avg_ret = subset['fwd_return_5d'].mean()
        wr = (subset['fwd_return_5d'] > 0).mean()
        ret_std = subset['fwd_return_5d'].std()
        sharpe = avg_ret / ret_std * np.sqrt(252 / 5) if ret_std > 0 else 0
        return {
            'signals': list(signals),
            'n': int(n),
            'pct_universe': round(n / len(df_test) * 100, 1),
            'avg_return_5d_pct': round(avg_ret, 3),
            'win_rate': round(wr, 3),
            'sharpe_annualized': round(sharpe, 2),
        }

    # 2-signal combos
    print("  2-signal combinations:")
    combo2_results = []
    for combo in itertools.combinations(core_signals, 2):
        result = eval_combo(combo)
        if result:
            combo2_results.append(result)

    combo2_results.sort(key=lambda x: x['sharpe_annualized'], reverse=True)
    for r in combo2_results[:10]:
        print(f"    {'+'.join(r['signals']):50s}  n={r['n']:4d}  "
              f"avgRet={r['avg_return_5d_pct']:+.3f}%  WR={r['win_rate']:.1%}  Sharpe={r['sharpe_annualized']:.2f}")

    # 3-signal combos
    print("\n  3-signal combinations:")
    combo3_results = []
    for combo in itertools.combinations(core_signals, 3):
        result = eval_combo(combo)
        if result:
            combo3_results.append(result)

    combo3_results.sort(key=lambda x: x['sharpe_annualized'], reverse=True)
    for r in combo3_results[:10]:
        print(f"    {'+'.join(r['signals']):60s}  n={r['n']:4d}  "
              f"avgRet={r['avg_return_5d_pct']:+.3f}%  WR={r['win_rate']:.1%}  Sharpe={r['sharpe_annualized']:.2f}")

    return combo2_results[:15], combo3_results[:15]


# ── Per-Sector Analysis ──────────────────────────────────────────────────
def analyze_per_sector(model, df_test, feature_cols):
    """Which signals work best for each sector?"""
    print("\nPer-sector signal preference analysis...")

    sector_prefs = {}
    for ticker in SECTOR_ETFS:
        sector_data = df_test[df_test['ticker'] == ticker]
        if len(sector_data) < 20:
            continue

        X_sector = sector_data[feature_cols].astype(float)
        y_sector = sector_data['fwd_return_5d'].astype(float)

        # Correlation of each signal with forward return
        correlations = {}
        for sig in SIGNAL_NAMES:
            if sig in sector_data.columns:
                corr = sector_data[sig].corr(sector_data['fwd_return_5d'])
                if not pd.isna(corr):
                    correlations[sig] = round(corr, 4)

        # Sort by absolute correlation
        sorted_corrs = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))

        # Best 3 signals for this sector
        top3 = list(sorted_corrs.keys())[:3]

        sector_prefs[ticker] = {
            'signal_correlations': sorted_corrs,
            'best_signals': top3,
            'n_samples': len(sector_data),
            'avg_fwd_return': round(y_sector.mean(), 3),
        }

        print(f"  {ticker}: top signals = {', '.join(top3)}")
        for sig, corr in list(sorted_corrs.items())[:5]:
            print(f"    {sig:30s}  corr={corr:+.4f}")

    return sector_prefs


# ── Directional Analysis ─────────────────────────────────────────────────
def analyze_long_short(df_test, y_pred):
    """Compare long vs short signal effectiveness."""
    print("\nLong vs Short signal analysis...")

    df_test = df_test.copy()
    df_test['prediction'] = y_pred

    # Strong long signals (avg_signal > 0.3)
    long_mask = df_test['avg_signal'] > 0.3
    short_mask = df_test['avg_signal'] < -0.3
    neutral_mask = (~long_mask) & (~short_mask)

    results = {}
    for name, mask in [('strong_long', long_mask), ('strong_short', short_mask), ('neutral', neutral_mask)]:
        if mask.sum() < 10:
            continue
        subset = df_test[mask]
        avg_ret = subset['fwd_return_5d'].mean()
        wr = (subset['fwd_return_5d'] > 0).mean()
        avg_pred = subset['prediction'].mean()
        n = len(subset)
        results[name] = {
            'n': int(n),
            'avg_return_5d_pct': round(avg_ret, 3),
            'win_rate': round(wr, 3),
            'avg_prediction': round(avg_pred, 3),
        }
        print(f"  {name:15s}: n={n:4d}  avgRet={avg_ret:+.3f}%  WR={wr:.1%}  avgPred={avg_pred:+.3f}")

    return results


# ── Model-Derived Optimal Weights ─────────────────────────────────────────
def derive_optimal_weights(imp_df, core_imp):
    """Convert feature importance to recommended signal weights."""
    print("\nDeriving optimal signal weights...")

    # Use gain importance of core signals as weight basis
    total_gain = core_imp['gain_pct'].sum()
    if total_gain == 0:
        total_gain = 1

    weights = {}
    for _, row in core_imp.iterrows():
        # Scale so weights sum to ~1.0
        w = row['gain_pct'] / total_gain
        weights[row['feature']] = round(w, 4)

    # Sort descending
    weights = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))

    print("  Optimal signal weights (sum to 1.0):")
    for sig, w in weights.items():
        bar = '█' * int(w * 50)
        print(f"    {sig:30s}  {w:.4f}  {bar}")

    return weights


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SIGNAL WEIGHT OPTIMIZER")
    print(f"Date range: {TRAIN_START} to {TEST_END}")
    print(f"Train: {TRAIN_START}-{TRAIN_END} | Test: {TEST_START}-{TEST_END}")
    print("=" * 70)

    # Step 1: Download data
    data = download_data()
    if 'SPY' not in data:
        print("FATAL: No SPY data")
        return

    # Step 2: Build feature matrix
    df = build_feature_matrix(data)

    # Step 3: Train model
    model, feature_cols, metrics, df_test, y_pred = train_model(df)
    if model is None:
        print("FATAL: Model training failed")
        return

    # Step 4: Feature importance
    imp_df, core_imp = analyze_importance(model, feature_cols)

    # Step 5: Confluence analysis
    confluence_results, optimal_confluence = analyze_confluence(df_test, y_pred)

    # Step 6: Combination analysis
    combo2, combo3 = analyze_combinations(df_test, y_pred)

    # Step 7: Per-sector analysis
    sector_prefs = analyze_per_sector(model, df_test, feature_cols)

    # Step 8: Long/short analysis
    long_short = analyze_long_short(df_test, y_pred)

    # Step 9: Derive optimal weights
    optimal_weights = derive_optimal_weights(imp_df, core_imp)

    # ── Save Results ─────────────────────────────────────────────────────
    results = {
        'metadata': {
            'run_date': datetime.now().isoformat(),
            'train_period': f'{TRAIN_START} to {TRAIN_END}',
            'test_period': f'{TEST_START} to {TEST_END}',
            'forward_days': FORWARD_DAYS,
            'n_sectors': len(SECTOR_ETFS),
            'n_signals': len(SIGNAL_NAMES),
            'total_features': len(feature_cols),
        },
        'model_performance': metrics,
        'feature_importance_ranking': [
            {
                'rank': i + 1,
                'feature': row['feature'],
                'gain_pct': round(row['gain_pct'], 2),
                'split_pct': round(row['split_pct'], 2),
            }
            for i, (_, row) in enumerate(imp_df.head(25).iterrows())
        ],
        'core_signal_ranking': [
            {
                'rank': i + 1,
                'signal': row['feature'],
                'gain_pct': round(row['gain_pct'], 2),
            }
            for i, (_, row) in enumerate(core_imp.iterrows())
        ],
        'optimal_signal_weights': optimal_weights,
        'confluence_analysis': {
            'thresholds': confluence_results,
            'optimal_threshold': optimal_confluence,
            'current_threshold': 5,
            'recommendation': f'Use >= {optimal_confluence} bullish signals (was 5)',
        },
        'best_2signal_combinations': combo2[:10],
        'best_3signal_combinations': combo3[:10],
        'per_sector_signal_preference': sector_prefs,
        'long_short_analysis': long_short,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {OUTPUT_PATH}")
    print(f"{'=' * 70}")

    # Summary
    print("\n=== KEY FINDINGS ===")
    print(f"  OOS IC: {metrics['test_ic']:.4f}")
    print(f"  OOS R²: {metrics['test_r2']:.4f}")
    print(f"  Optimal confluence: >= {optimal_confluence} signals (was 5)")
    print(f"\n  Top 5 signals by importance:")
    for i, (_, row) in enumerate(core_imp.head(5).iterrows(), 1):
        print(f"    {i}. {row['feature']} ({row['gain_pct']:.1f}%)")
    if combo2:
        print(f"\n  Best 2-signal combo: {'+'.join(combo2[0]['signals'])} (Sharpe={combo2[0]['sharpe_annualized']:.2f})")
    if combo3:
        print(f"  Best 3-signal combo: {'+'.join(combo3[0]['signals'])} (Sharpe={combo3[0]['sharpe_annualized']:.2f})")


if __name__ == '__main__':
    main()
