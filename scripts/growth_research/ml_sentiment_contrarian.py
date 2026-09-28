#!/usr/bin/env python3
"""
Sentiment Contrarian + Fear-Greed Composite
=============================================
INSIGHT: Extreme sentiment readings (fear/greed) are well-documented
contrarian indicators. This strategy builds a composite fear-greed index
from public market data and trades against extremes.

Composite Fear-Greed Features (all from public price data):
  - VIX level + VIX percentile (252d rank)
  - VIX term structure (VIX/VIX3M ratio: contango=complacency, backwardation=fear)
  - Credit spread: HYG-LQD total return differential (widening=fear)
  - Market breadth proxy: IWM vs SPY relative strength (narrow breadth=fragile)
  - Safe haven demand: GLD relative strength vs SPY (rising=fear)
  - Put-call proxy: VIX spike frequency (count of >2% VIX jumps in 20d)
  - Momentum exhaustion: SPY RSI(14) + distance from 200MA
  - Composite z-score → single fear-greed reading

Strategy variants:
  a) Simple contrarian: extreme fear (z < -1.5) → long UPRO, extreme greed (z > 1.5) → defensive SHY
  b) ML-enhanced: LightGBM predicts 5d/21d forward returns from all features, walk-forward

Mandatory gates: 5-day label gap, signal-date permutation (200 shuffles),
10bps costs, R1 regime test, full adversarial suite.

Walk-forward: SLIDING 252d (HC #0). Fixed $100K, NO DCA (HC #713).
Full adversarial: permutation 200x, sub-period 4-block, outlier, R1 regime (HC #705).
"""

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 100_000
TRAIN_WINDOW     = 252          # 1yr sliding window
LABEL_GAP        = 5            # 5-day gap to prevent lookahead
LABEL_HORIZON_5D = 5            # 5-day forward return
LABEL_HORIZON_21D = 21          # 21-day (1mo) forward return
N_PERMUTATIONS   = 200
COST_BPS         = 10           # 10bps round-trip cost
FEAR_THRESHOLD   = -1.5         # z-score for extreme fear → go long
GREED_THRESHOLD  = 1.5          # z-score for extreme greed → go defensive
ML_THRESHOLD     = 0.55         # LightGBM confidence threshold

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_sentiment_contrarian"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Tickers
TICKERS = ['SPY', '^VIX', '^VIX3M', 'HYG', 'LQD', 'TLT', 'GLD', 'QQQ', 'IWM', '^GSPC']
# Trade vehicles
TRADE_LONG  = 'UPRO'   # 3x leveraged S&P long
TRADE_SHORT = 'SHY'    # short-term treasuries (defensive)


def download_data():
    """Download all required data from yfinance."""
    print("=" * 80)
    print("STEP 1: DOWNLOADING DATA")
    print("=" * 80)

    cache = BASE / "data" / "cache" / "sentiment_contrarian_data.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 3600:
        df = pd.read_parquet(cache)
        print(f"  Cached: {df.shape}, {df.index[0].date()} -> {df.index[-1].date()}")
        return df

    all_tickers = TICKERS + [TRADE_LONG, TRADE_SHORT]
    all_tickers = list(set(all_tickers))

    print(f"  Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start='2010-01-01', auto_adjust=True, progress=False)
    closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw

    # Rename index tickers
    rename_map = {'^VIX': 'VIX', '^VIX3M': 'VIX3M', '^GSPC': 'GSPC'}
    closes = closes.rename(columns=rename_map)
    closes = closes.ffill().dropna(thresh=len(closes.columns) - 3)

    cache.parent.mkdir(parents=True, exist_ok=True)
    closes.to_parquet(cache)
    print(f"  Shape: {closes.shape}, {closes.index[0].date()} -> {closes.index[-1].date()}")
    for col in closes.columns:
        valid = closes[col].notna().sum()
        print(f"    {col}: {valid} days")
    return closes


def compute_rsi(series, period=14):
    """RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_fear_greed_features(df):
    """Build composite fear-greed index from market data."""
    print("\n" + "=" * 80)
    print("STEP 2: BUILDING FEAR-GREED FEATURES")
    print("=" * 80)

    feat = pd.DataFrame(index=df.index)

    # --- 1. VIX level + percentile ---
    if 'VIX' in df.columns:
        feat['vix_level'] = df['VIX']
        feat['vix_pct_252d'] = df['VIX'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        print(f"  VIX level: mean={df['VIX'].mean():.1f}, current={df['VIX'].iloc[-1]:.1f}")
    else:
        print("  WARNING: VIX not available")

    # --- 2. VIX term structure ---
    if 'VIX' in df.columns and 'VIX3M' in df.columns:
        feat['vix_term_ratio'] = df['VIX'] / df['VIX3M']
        # > 1 = backwardation (fear), < 1 = contango (complacency)
        feat['vix_term_pct'] = feat['vix_term_ratio'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        print(f"  VIX term structure: mean ratio={feat['vix_term_ratio'].mean():.3f}")
    elif 'VIX' in df.columns:
        # Proxy: use VIX 10d MA vs 60d MA as term structure proxy
        vix_short = df['VIX'].rolling(10).mean()
        vix_long = df['VIX'].rolling(60).mean()
        feat['vix_term_ratio'] = vix_short / vix_long
        feat['vix_term_pct'] = feat['vix_term_ratio'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        print(f"  VIX term structure (proxy): mean ratio={feat['vix_term_ratio'].mean():.3f}")

    # --- 3. Credit spread: HYG vs LQD ---
    if 'HYG' in df.columns and 'LQD' in df.columns:
        hyg_ret = df['HYG'].pct_change()
        lqd_ret = df['LQD'].pct_change()
        # Spread widening = HYG underperforming LQD = fear
        feat['credit_spread_20d'] = (hyg_ret - lqd_ret).rolling(20).sum()
        feat['credit_spread_60d'] = (hyg_ret - lqd_ret).rolling(60).sum()
        # Negative = fear (HYG underperforming)
        print(f"  Credit spread 20d: mean={feat['credit_spread_20d'].mean():.4f}")

    # --- 4. Market breadth proxy: IWM vs SPY ---
    if 'IWM' in df.columns and 'SPY' in df.columns:
        iwm_ret = df['IWM'].pct_change().rolling(20).sum()
        spy_ret = df['SPY'].pct_change().rolling(20).sum()
        feat['breadth_rs_20d'] = iwm_ret - spy_ret
        # Negative = narrow breadth (only large caps working) = fragile
        feat['breadth_rs_60d'] = (df['IWM'].pct_change().rolling(60).sum() -
                                   df['SPY'].pct_change().rolling(60).sum())
        print(f"  Breadth RS 20d: mean={feat['breadth_rs_20d'].mean():.4f}")

    # --- 5. Safe haven demand: GLD vs SPY ---
    if 'GLD' in df.columns and 'SPY' in df.columns:
        gld_ret = df['GLD'].pct_change().rolling(20).sum()
        spy_ret2 = df['SPY'].pct_change().rolling(20).sum()
        feat['gold_rs_20d'] = gld_ret - spy_ret2
        # Positive = gold outperforming = fear/risk-off
        print(f"  Gold RS 20d: mean={feat['gold_rs_20d'].mean():.4f}")

    # --- 6. Put-call proxy: VIX spike frequency ---
    if 'VIX' in df.columns:
        vix_pct_chg = df['VIX'].pct_change()
        # Count days with >2% VIX jump in trailing 20d
        feat['vix_spike_count_20d'] = (vix_pct_chg > 0.02).astype(float).rolling(20).sum()
        # Count days with >5% VIX jump (more extreme)
        feat['vix_spike_count_5pct_20d'] = (vix_pct_chg > 0.05).astype(float).rolling(20).sum()
        print(f"  VIX spike count 20d: mean={feat['vix_spike_count_20d'].mean():.1f}")

    # --- 7. Momentum exhaustion: RSI + distance from 200MA ---
    if 'SPY' in df.columns:
        feat['spy_rsi_14'] = compute_rsi(df['SPY'], 14)
        ma200 = df['SPY'].rolling(200).mean()
        feat['spy_dist_200ma'] = (df['SPY'] / ma200 - 1) * 100  # percent distance
        feat['spy_ret_20d'] = df['SPY'].pct_change().rolling(20).sum() * 100
        feat['spy_ret_60d'] = df['SPY'].pct_change().rolling(60).sum() * 100
        print(f"  SPY RSI(14): mean={feat['spy_rsi_14'].mean():.1f}")
        print(f"  SPY dist from 200MA: mean={feat['spy_dist_200ma'].mean():.1f}%")

    # --- 8. Realized vol ---
    if 'SPY' in df.columns:
        spy_daily_ret = df['SPY'].pct_change()
        feat['realized_vol_20d'] = spy_daily_ret.rolling(20).std() * np.sqrt(252) * 100
        feat['realized_vol_60d'] = spy_daily_ret.rolling(60).std() * np.sqrt(252) * 100
        print(f"  Realized vol 20d: mean={feat['realized_vol_20d'].mean():.1f}%")

    # --- 9. Bond/equity correlation proxy ---
    if 'TLT' in df.columns and 'SPY' in df.columns:
        tlt_ret = df['TLT'].pct_change()
        spy_daily = df['SPY'].pct_change()
        feat['bond_eq_corr_60d'] = tlt_ret.rolling(60).corr(spy_daily)
        # Positive correlation = unusual (risk-off), negative = normal
        print(f"  Bond-equity corr 60d: mean={feat['bond_eq_corr_60d'].mean():.3f}")

    # --- COMPOSITE Z-SCORE ---
    # Components: higher = more greed, lower = more fear
    # Some need to be flipped so that convention is consistent
    z_components = []
    z_names = []

    if 'vix_pct_252d' in feat.columns:
        # High VIX = fear → flip sign for composite (low z = fear)
        z_components.append(-feat['vix_pct_252d'])
        z_names.append('vix_pct')

    if 'vix_term_ratio' in feat.columns:
        # High ratio = backwardation = fear → flip
        z_components.append(-feat['vix_term_ratio'])
        z_names.append('vix_term')

    if 'credit_spread_20d' in feat.columns:
        # Positive = HYG outperforming = greed; negative = fear
        z_components.append(feat['credit_spread_20d'])
        z_names.append('credit')

    if 'breadth_rs_20d' in feat.columns:
        # Positive = small caps leading = broad rally = greed
        z_components.append(feat['breadth_rs_20d'])
        z_names.append('breadth')

    if 'gold_rs_20d' in feat.columns:
        # Positive = gold winning = fear → flip
        z_components.append(-feat['gold_rs_20d'])
        z_names.append('gold_haven')

    if 'vix_spike_count_20d' in feat.columns:
        # More spikes = fear → flip
        z_components.append(-feat['vix_spike_count_20d'])
        z_names.append('vix_spikes')

    if 'spy_rsi_14' in feat.columns:
        # High RSI = greed, low = fear
        z_components.append(feat['spy_rsi_14'])
        z_names.append('rsi')

    if 'spy_dist_200ma' in feat.columns:
        # Far above 200MA = greed
        z_components.append(feat['spy_dist_200ma'])
        z_names.append('ma_dist')

    # Z-score each component using 252d rolling window
    z_scores = pd.DataFrame(index=df.index)
    for comp, name in zip(z_components, z_names):
        mu = comp.rolling(252).mean()
        sigma = comp.rolling(252).std()
        z_scores[name] = (comp - mu) / sigma.replace(0, np.nan)

    # Equal-weight composite
    feat['composite_z'] = z_scores.mean(axis=1)
    feat['composite_z_std'] = z_scores.std(axis=1)  # disagreement among components

    # Count components in fear/greed
    feat['n_fear'] = (z_scores < -1.0).sum(axis=1)
    feat['n_greed'] = (z_scores > 1.0).sum(axis=1)

    print(f"\n  Composite z-score: mean={feat['composite_z'].mean():.3f}, "
          f"std={feat['composite_z'].std():.3f}")
    print(f"  Extreme fear days (z < {FEAR_THRESHOLD}): "
          f"{(feat['composite_z'] < FEAR_THRESHOLD).sum()}")
    print(f"  Extreme greed days (z > {GREED_THRESHOLD}): "
          f"{(feat['composite_z'] > GREED_THRESHOLD).sum()}")

    return feat, z_scores


def build_labels(df, feat):
    """Build forward return labels with 5-day gap."""
    print("\n" + "=" * 80)
    print("STEP 3: BUILDING LABELS (5d gap, 5d/21d horizons)")
    print("=" * 80)

    spy_ret = df['SPY'].pct_change()

    # 5-day forward return (starting AFTER 5-day gap)
    feat['fwd_5d'] = spy_ret.shift(-(LABEL_GAP + LABEL_HORIZON_5D)).rolling(LABEL_HORIZON_5D).sum()
    # Actually: sum of returns from day+6 to day+10
    # More precisely:
    fwd_5d = pd.Series(np.nan, index=df.index)
    fwd_21d = pd.Series(np.nan, index=df.index)
    spy_prices = df['SPY']

    for i in range(len(df) - LABEL_GAP - LABEL_HORIZON_21D):
        idx = df.index[i]
        # 5d forward return starting after gap
        start_price = spy_prices.iloc[i + LABEL_GAP]
        end_5d = spy_prices.iloc[i + LABEL_GAP + LABEL_HORIZON_5D]
        end_21d = spy_prices.iloc[i + LABEL_GAP + LABEL_HORIZON_21D]
        fwd_5d.iloc[i] = (end_5d / start_price) - 1
        fwd_21d.iloc[i] = (end_21d / start_price) - 1

    feat['fwd_5d'] = fwd_5d
    feat['fwd_21d'] = fwd_21d

    valid = feat['fwd_5d'].notna().sum()
    print(f"  Valid 5d labels: {valid}")
    print(f"  Valid 21d labels: {feat['fwd_21d'].notna().sum()}")
    print(f"  5d return: mean={fwd_5d.mean()*100:.3f}%, std={fwd_5d.std()*100:.3f}%")
    print(f"  21d return: mean={fwd_21d.mean()*100:.3f}%, std={fwd_21d.std()*100:.3f}%")

    return feat


def run_simple_contrarian(df, feat):
    """Simple rules-based contrarian: extreme fear → long, extreme greed → defensive."""
    print("\n" + "=" * 80)
    print("STEP 4a: SIMPLE CONTRARIAN BACKTEST")
    print("=" * 80)

    composite = feat['composite_z']
    spy_ret = df['SPY'].pct_change()

    # Trade vehicles
    has_upro = TRADE_LONG in df.columns
    has_shy = TRADE_SHORT in df.columns

    if has_upro:
        upro_ret = df[TRADE_LONG].pct_change()
    else:
        upro_ret = spy_ret * 3  # approximate UPRO as 3x SPY

    if has_shy:
        shy_ret = df[TRADE_SHORT].pct_change()
    else:
        shy_ret = pd.Series(0.001 / 252, index=df.index)  # ~0.1% annual

    # Need warmup for composite z
    valid_start = composite.first_valid_index()
    if valid_start is None:
        print("  ERROR: No valid composite z-scores")
        return pd.Series(dtype=float), {}

    daily_ret = pd.Series(0.0, index=df.index)
    positions = pd.Series('CASH', index=df.index)
    prev_pos = 'CASH'
    cost_per_trade = COST_BPS / 10000

    for i in range(1, len(df)):
        idx = df.index[i]
        z = composite.iloc[i - 1]  # use PREVIOUS day's signal (no lookahead)

        if pd.isna(z):
            daily_ret.iloc[i] = 0.0
            continue

        # Determine position
        if z < FEAR_THRESHOLD:
            pos = 'LONG'
        elif z > GREED_THRESHOLD:
            pos = 'DEFENSIVE'
        else:
            pos = 'CASH'

        # Assign return
        if pos == 'LONG':
            daily_ret.iloc[i] = upro_ret.iloc[i] if not pd.isna(upro_ret.iloc[i]) else 0
        elif pos == 'DEFENSIVE':
            daily_ret.iloc[i] = shy_ret.iloc[i] if not pd.isna(shy_ret.iloc[i]) else 0
        else:
            daily_ret.iloc[i] = 0.0

        # Transaction cost on position change
        if pos != prev_pos and pos != 'CASH' and prev_pos != 'CASH':
            daily_ret.iloc[i] -= cost_per_trade * 2  # exit + entry
        elif pos != prev_pos:
            daily_ret.iloc[i] -= cost_per_trade

        positions.iloc[i] = pos
        prev_pos = pos

    # Trim to valid period
    daily_ret = daily_ret[daily_ret.index >= valid_start]
    daily_ret = daily_ret.iloc[1:]  # skip first day

    # Position stats
    pos_counts = positions[positions.index >= valid_start].value_counts()
    total = len(positions[positions.index >= valid_start])
    print(f"  Position breakdown:")
    for p, c in pos_counts.items():
        print(f"    {p}: {c} days ({c/total*100:.1f}%)")

    n_trades = (positions != positions.shift()).sum()
    print(f"  Total position changes: {n_trades}")

    return daily_ret, {'positions': positions, 'n_trades': int(n_trades)}


def run_ml_enhanced(df, feat):
    """ML-enhanced contrarian: LightGBM predicts forward returns, walk-forward."""
    print("\n" + "=" * 80)
    print("STEP 4b: ML-ENHANCED CONTRARIAN (LightGBM, Walk-Forward)")
    print("=" * 80)

    try:
        import lightgbm as lgb
        use_lgb = True
        print("  Using LightGBM")
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        use_lgb = False
        print("  LightGBM not available, using sklearn GBM")

    # Feature columns (exclude labels and composite)
    exclude_cols = ['fwd_5d', 'fwd_21d', 'composite_z', 'composite_z_std',
                    'n_fear', 'n_greed']
    feature_cols = [c for c in feat.columns if c not in exclude_cols and feat[c].dtype in [np.float64, np.float32, np.int64]]

    # Add composite as feature too
    feature_cols.append('composite_z')
    feature_cols.append('composite_z_std')
    feature_cols.append('n_fear')
    feature_cols.append('n_greed')
    feature_cols = [c for c in feature_cols if c in feat.columns]

    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    # Build dataset
    data = feat[feature_cols + ['fwd_5d', 'fwd_21d']].copy()
    data = data.dropna()
    print(f"  Valid samples: {len(data)}")

    if len(data) < TRAIN_WINDOW + 100:
        print("  ERROR: Not enough data for walk-forward")
        return pd.Series(dtype=float), pd.DataFrame(), {}

    # Binary label: 1 if 5d return > 0, else 0
    data['label_5d'] = (data['fwd_5d'] > 0).astype(int)
    data['label_21d'] = (data['fwd_21d'] > 0).astype(int)

    spy_ret = df['SPY'].pct_change()

    # Walk-forward: sliding 252d train, predict next day
    predictions = []
    actuals_5d = []
    actuals_21d = []
    pred_dates = []
    ml_probs = []

    n = len(data)
    dates = data.index

    print(f"  Walk-forward: {n - TRAIN_WINDOW} prediction days")

    for i in range(TRAIN_WINDOW, n):
        train_start = i - TRAIN_WINDOW
        train_end = i  # exclusive

        X_train = data.iloc[train_start:train_end][feature_cols].values
        y_train = data.iloc[train_start:train_end]['label_5d'].values

        X_test = data.iloc[i:i+1][feature_cols].values

        if np.isnan(X_train).any() or np.isnan(X_test).any():
            continue

        if use_lgb:
            model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, verbose=-1, n_jobs=1,
                random_state=42,
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )

        try:
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[0, 1]  # prob of positive return
        except Exception:
            prob = 0.5

        pred_date = dates[i]
        predictions.append(prob)
        actuals_5d.append(data.iloc[i]['fwd_5d'])
        actuals_21d.append(data.iloc[i]['fwd_21d'])
        pred_dates.append(pred_date)
        ml_probs.append(prob)

    pred_df = pd.DataFrame({
        'date': pred_dates,
        'ml_prob': predictions,
        'fwd_5d': actuals_5d,
        'fwd_21d': actuals_21d,
    }).set_index('date')

    print(f"  Predictions: {len(pred_df)}")
    print(f"  ML prob distribution: mean={pred_df['ml_prob'].mean():.3f}, "
          f"std={pred_df['ml_prob'].std():.3f}")

    # Feature importance (from last model)
    if use_lgb and hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        imp = imp.sort_values(ascending=False)
        print(f"\n  Top features (last fold):")
        for fname, fval in imp.head(10).items():
            print(f"    {fname}: {fval}")

    # Backtest ML strategy
    # Signal: high prob (>0.55) → long SPY, low prob (<0.45) → short/defensive
    daily_ret = pd.Series(0.0, index=pred_df.index)
    positions = pd.Series('CASH', index=pred_df.index)
    prev_pos = 'CASH'
    cost_per_trade = COST_BPS / 10000

    has_upro = TRADE_LONG in df.columns
    has_shy = TRADE_SHORT in df.columns

    upro_ret = df[TRADE_LONG].pct_change() if has_upro else df['SPY'].pct_change() * 3
    shy_ret = df[TRADE_SHORT].pct_change() if has_shy else pd.Series(0.001/252, index=df.index)

    for i in range(1, len(pred_df)):
        idx = pred_df.index[i]
        prob = pred_df['ml_prob'].iloc[i - 1]  # previous day's prediction

        if prob > ML_THRESHOLD:
            pos = 'LONG'
        elif prob < (1 - ML_THRESHOLD):
            pos = 'DEFENSIVE'
        else:
            pos = 'CASH'

        if idx in upro_ret.index and idx in shy_ret.index:
            if pos == 'LONG':
                daily_ret.iloc[i] = upro_ret.loc[idx] if not pd.isna(upro_ret.loc[idx]) else 0
            elif pos == 'DEFENSIVE':
                daily_ret.iloc[i] = shy_ret.loc[idx] if not pd.isna(shy_ret.loc[idx]) else 0

        if pos != prev_pos and pos != 'CASH' and prev_pos != 'CASH':
            daily_ret.iloc[i] -= cost_per_trade * 2
        elif pos != prev_pos:
            daily_ret.iloc[i] -= cost_per_trade

        positions.iloc[i] = pos
        prev_pos = pos

    # Also run SPY-only version (no leverage)
    daily_ret_spy = pd.Series(0.0, index=pred_df.index)
    prev_pos2 = 'CASH'
    for i in range(1, len(pred_df)):
        idx = pred_df.index[i]
        prob = pred_df['ml_prob'].iloc[i - 1]

        if prob > ML_THRESHOLD:
            pos2 = 'LONG'
        elif prob < (1 - ML_THRESHOLD):
            pos2 = 'SHORT'
        else:
            pos2 = 'CASH'

        if idx in spy_ret.index:
            if pos2 == 'LONG':
                daily_ret_spy.iloc[i] = spy_ret.loc[idx] if not pd.isna(spy_ret.loc[idx]) else 0
            elif pos2 == 'SHORT':
                daily_ret_spy.iloc[i] = -spy_ret.loc[idx] if not pd.isna(spy_ret.loc[idx]) else 0

        if pos2 != prev_pos2:
            daily_ret_spy.iloc[i] -= cost_per_trade
        prev_pos2 = pos2

    pos_counts = positions.value_counts()
    total = len(positions)
    print(f"\n  ML Position breakdown:")
    for p, c in pos_counts.items():
        print(f"    {p}: {c} days ({c/total*100:.1f}%)")

    return daily_ret, daily_ret_spy, pred_df, feature_cols


def compute_metrics(ret_series, name=""):
    """Compute risk-adjusted metrics."""
    ret = ret_series.dropna()
    if len(ret) < 60:
        return None

    total_ret = (1 + ret).prod() - 1
    years = len(ret) / 252
    cagr = (1 + total_ret) ** (1 / max(years, 0.1)) - 1

    mu = ret.mean() * 252
    sigma = ret.std() * np.sqrt(252)
    sharpe = mu / sigma if sigma > 0 else 0

    downside = ret[ret < 0].std() * np.sqrt(252)
    sortino = mu / downside if downside > 0 else 0

    cum = (1 + ret).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min() * 100

    calmar = cagr / abs(max_dd / 100) if max_dd != 0 else 0

    win_rate = (ret > 0).mean() * 100
    avg_win = ret[ret > 0].mean() * 100 if (ret > 0).any() else 0
    avg_loss = ret[ret < 0].mean() * 100 if (ret < 0).any() else 0
    pf = abs(ret[ret > 0].sum() / ret[ret < 0].sum()) if ret[ret < 0].sum() != 0 else 999

    if name:
        print(f"  {name}: Sharpe={sharpe:.3f} Sortino={sortino:.3f} CAGR={cagr*100:.1f}% "
              f"MaxDD={max_dd:.1f}% Calmar={calmar:.3f} WR={win_rate:.1f}% PF={pf:.2f}")

    return {
        'name': name, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 1), 'max_dd': round(max_dd, 1),
        'calmar': round(calmar, 3), 'win_rate': round(win_rate, 1),
        'profit_factor': round(pf, 2), 'total_return': round(total_ret * 100, 1),
        'annual_vol': round(sigma * 100, 1), 'years': round(years, 1),
    }


def run_adversarial(ret_series, feat, df, pred_df=None, feature_cols=None, name="Strategy"):
    """Full adversarial validation suite."""
    print("\n" + "=" * 80)
    print(f"ADVERSARIAL VALIDATION: {name}")
    print("=" * 80)

    results = {}
    real = compute_metrics(ret_series)
    if real is None:
        print("  Cannot compute metrics, skipping adversarial")
        return {'summary': {'gates_passed': 0, 'total': 4, 'verdict': 'FAIL'}}
    real_sharpe = real['sharpe']

    # 1. SIGNAL-DATE PERMUTATION TEST (200 shuffles)
    print(f"\n  [1/4] Signal-date permutation test ({N_PERMUTATIONS} shuffles)...")
    perm_sharpes = []
    spy_ret = df['SPY'].pct_change()

    for trial in range(N_PERMUTATIONS):
        # Shuffle the composite z-scores (break date alignment)
        perm_z = feat['composite_z'].copy()
        valid_mask = perm_z.notna()
        valid_vals = perm_z[valid_mask].values.copy()
        np.random.shuffle(valid_vals)
        perm_z[valid_mask] = valid_vals

        # Re-run simple contrarian with shuffled signals
        perm_ret = pd.Series(0.0, index=ret_series.index)
        for i in range(1, len(perm_ret)):
            idx = perm_ret.index[i]
            if idx not in perm_z.index:
                continue
            # Get previous day's z
            loc = perm_z.index.get_loc(idx)
            if loc > 0:
                z = perm_z.iloc[loc - 1]
            else:
                continue

            if pd.isna(z):
                continue

            if z < FEAR_THRESHOLD:
                if idx in spy_ret.index:
                    r = spy_ret.loc[idx]
                    perm_ret.iloc[i] = r * 3 if not pd.isna(r) else 0  # UPRO proxy
            elif z > GREED_THRESHOLD:
                perm_ret.iloc[i] = 0.001 / 252  # SHY proxy

        m = compute_metrics(perm_ret)
        if m:
            perm_sharpes.append(m['sharpe'])

    perm_p = np.mean([s >= real_sharpe for s in perm_sharpes]) if perm_sharpes else 1.0
    perm_pass = perm_p < 0.05
    print(f"    Real Sharpe: {real_sharpe:.3f}")
    print(f"    Perm mean: {np.mean(perm_sharpes):.3f} +/- {np.std(perm_sharpes):.3f}")
    print(f"    p-value: {perm_p:.3f} -> {'PASS' if perm_pass else 'FAIL'}")
    results['permutation'] = {'p_value': round(perm_p, 4), 'pass': bool(perm_pass),
                               'real_sharpe': real_sharpe,
                               'perm_mean': round(float(np.mean(perm_sharpes)), 3)}

    # 2. SUB-PERIOD CONSISTENCY (4 blocks)
    print("\n  [2/4] Sub-period consistency...")
    n = len(ret_series)
    block_size = n // 4
    block_sharpes = []
    for b in range(4):
        start = b * block_size
        end = (b + 1) * block_size if b < 3 else n
        m = compute_metrics(ret_series.iloc[start:end])
        if m:
            block_sharpes.append(m['sharpe'])
            print(f"    Block {b+1}: Sharpe {m['sharpe']:.3f}, CAGR {m['cagr']:.1f}%")

    if block_sharpes and np.mean(block_sharpes) != 0:
        cv = np.std(block_sharpes) / abs(np.mean(block_sharpes))
    else:
        cv = 999
    sub_pass = cv < 0.50
    print(f"    CV: {cv:.3f} -> {'PASS' if sub_pass else 'FAIL'}")
    results['sub_period'] = {'cv': round(cv, 3), 'pass': bool(sub_pass),
                              'block_sharpes': [round(s, 3) for s in block_sharpes]}

    # 3. OUTLIER ROBUSTNESS
    print("\n  [3/4] Outlier robustness...")
    p95 = ret_series.quantile(0.95)
    p05 = ret_series.quantile(0.05)
    trimmed = ret_series[(ret_series > p05) & (ret_series < p95)]
    m_full = compute_metrics(ret_series)
    m_trim = compute_metrics(trimmed)
    if m_full and m_trim and m_full['sharpe'] != 0:
        deg = (m_full['sharpe'] - m_trim['sharpe']) / abs(m_full['sharpe'])
    else:
        deg = 999
    outlier_pass = abs(deg) < 0.30
    print(f"    Full Sharpe: {m_full['sharpe']:.3f}, Trimmed: {m_trim['sharpe'] if m_trim else 'N/A'}")
    print(f"    Degradation: {deg:.1%} -> {'PASS' if outlier_pass else 'FAIL'}")
    results['outlier'] = {'degradation': round(deg, 3), 'pass': bool(outlier_pass)}

    # 4. R1 REGIME TEST
    print("\n  [4/4] R1 regime test...")
    spy_ret_daily = df['SPY'].pct_change()
    spy_20d = spy_ret_daily.rolling(20).sum()

    common_idx = ret_series.index.intersection(spy_20d.dropna().index)
    ret_aligned = ret_series.reindex(common_idx)
    spy_20d_aligned = spy_20d.reindex(common_idx)

    green_mask = spy_20d_aligned > 0
    m_green = compute_metrics(ret_aligned[green_mask], "Green regime")
    m_red = compute_metrics(ret_aligned[~green_mask], "Red regime")

    if m_green and m_red:
        s_g, s_r = m_green['sharpe'], m_red['sharpe']
        gap = abs(s_g - s_r) / max(abs(s_g), abs(s_r), 0.01)
        r1_pass = gap < 0.50
        print(f"    Green: {s_g:.3f}, Red: {s_r:.3f}, Gap: {gap:.3f} -> {'PASS' if r1_pass else 'FAIL'}")
        results['r1_regime'] = {'green_sharpe': s_g, 'red_sharpe': s_r,
                                 'gap': round(gap, 3), 'pass': bool(r1_pass)}
    else:
        r1_pass = False
        results['r1_regime'] = {'pass': False}

    gates = sum([results.get(k, {}).get('pass', False)
                 for k in ['permutation', 'sub_period', 'outlier', 'r1_regime']])
    verdict = 'PASS' if gates >= 3 else 'FAIL'
    results['summary'] = {'gates_passed': gates, 'total': 4, 'verdict': verdict}
    print(f"\n  ADVERSARIAL SUMMARY: {gates}/4 gates -> {verdict}")

    return results


def plot_results(df, feat, simple_ret, ml_ret, ml_spy_ret, pred_df):
    """Generate plots."""
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    fig, axes = plt.subplots(4, 1, figsize=(16, 20))

    # 1. Equity curves
    ax = axes[0]
    spy_ret = df['SPY'].pct_change()
    common_start = simple_ret.index[0]

    spy_aligned = spy_ret.reindex(simple_ret.index).fillna(0)
    cum_simple = (1 + simple_ret).cumprod() * INITIAL_CAPITAL
    cum_spy = (1 + spy_aligned).cumprod() * INITIAL_CAPITAL

    ax.plot(cum_simple.index, cum_simple.values, label='Simple Contrarian', linewidth=2)
    if len(ml_ret) > 0:
        cum_ml = (1 + ml_ret).cumprod() * INITIAL_CAPITAL
        ax.plot(cum_ml.index, cum_ml.values, label='ML Contrarian (UPRO/SHY)', linewidth=2)
    if len(ml_spy_ret) > 0:
        cum_ml_spy = (1 + ml_spy_ret).cumprod() * INITIAL_CAPITAL
        ax.plot(cum_ml_spy.index, cum_ml_spy.values, label='ML Contrarian (SPY L/S)', linewidth=1.5, linestyle='--')
    ax.plot(cum_spy.index, cum_spy.values, label='SPY B&H', alpha=0.7)
    ax.set_title('Sentiment Contrarian - Equity Curves ($100K, No DCA)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Portfolio Value ($)')

    # 2. Composite Fear-Greed Index
    ax = axes[1]
    z = feat['composite_z'].dropna()
    ax.plot(z.index, z.values, linewidth=0.8, color='navy', alpha=0.7)
    ax.axhline(FEAR_THRESHOLD, color='green', linestyle='--', alpha=0.7, label=f'Fear ({FEAR_THRESHOLD})')
    ax.axhline(GREED_THRESHOLD, color='red', linestyle='--', alpha=0.7, label=f'Greed ({GREED_THRESHOLD})')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.fill_between(z.index, z.values, FEAR_THRESHOLD,
                     where=z.values < FEAR_THRESHOLD, alpha=0.3, color='green', label='Extreme Fear')
    ax.fill_between(z.index, z.values, GREED_THRESHOLD,
                     where=z.values > GREED_THRESHOLD, alpha=0.3, color='red', label='Extreme Greed')
    ax.set_title('Composite Fear-Greed Z-Score', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Z-Score')

    # 3. Drawdowns
    ax = axes[2]
    dd_simple = cum_simple / cum_simple.cummax() - 1
    ax.fill_between(dd_simple.index, dd_simple.values, 0, alpha=0.5, color='blue', label='Simple')
    if len(ml_ret) > 0:
        dd_ml = cum_ml / cum_ml.cummax() - 1
        ax.fill_between(dd_ml.index, dd_ml.values, 0, alpha=0.3, color='orange', label='ML')
    ax.set_title('Drawdowns', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Drawdown (%)')

    # 4. ML probability over time
    if pred_df is not None and len(pred_df) > 0:
        ax = axes[3]
        ax.plot(pred_df.index, pred_df['ml_prob'].values, linewidth=0.5, alpha=0.7)
        ax.axhline(ML_THRESHOLD, color='green', linestyle='--', alpha=0.5, label=f'Long ({ML_THRESHOLD})')
        ax.axhline(1 - ML_THRESHOLD, color='red', linestyle='--', alpha=0.5, label=f'Def ({1-ML_THRESHOLD})')
        ax.set_title('ML Predicted Probability (P(5d return > 0))', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Probability')

    plt.tight_layout()
    plt.savefig(OUTPUT / 'equity_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved equity_curves.png")


def main():
    t0 = time.time()
    print("=" * 80)
    print("SENTIMENT CONTRARIAN + FEAR-GREED COMPOSITE")
    print("=" * 80)
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output: {OUTPUT}")

    # Step 1: Download data
    df = download_data()

    # Step 2: Build fear-greed features
    feat, z_scores = build_fear_greed_features(df)

    # Step 3: Build labels
    feat = build_labels(df, feat)

    # Step 4a: Simple contrarian
    simple_ret, simple_info = run_simple_contrarian(df, feat)

    # Step 4b: ML-enhanced
    ml_ret, ml_spy_ret, pred_df, feature_cols = run_ml_enhanced(df, feat)

    # Metrics
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    spy_ret = df['SPY'].pct_change()

    m_simple = compute_metrics(simple_ret, "Simple Contrarian (UPRO/SHY)")
    m_ml = compute_metrics(ml_ret, "ML Contrarian (UPRO/SHY)")
    m_ml_spy = compute_metrics(ml_spy_ret, "ML Contrarian (SPY L/S)")

    # Benchmark
    common_idx = simple_ret.index
    spy_aligned = spy_ret.reindex(common_idx).fillna(0)
    m_spy = compute_metrics(spy_aligned, "SPY Buy & Hold")

    print(f"\n  {'Strategy':<30} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8} {'PF':>6} {'WR':>6}")
    print(f"  {'-'*90}")
    for met in [m_simple, m_ml, m_ml_spy, m_spy]:
        if met:
            print(f"  {met['name']:<30} {met['sharpe']:>8.3f} {met['sortino']:>8.3f} "
                  f"{met['cagr']:>7.1f}% {met['max_dd']:>7.1f}% {met['calmar']:>8.3f} "
                  f"{met['profit_factor']:>5.2f} {met['win_rate']:>5.1f}%")

    # Adversarial: run on both simple and ML
    adv_simple = run_adversarial(simple_ret, feat, df, name="Simple Contrarian")
    adv_ml = run_adversarial(ml_ret, feat, df, pred_df, feature_cols, name="ML Contrarian")

    # Plots
    plot_results(df, feat, simple_ret, ml_ret, ml_spy_ret, pred_df)

    # Save results
    runtime = round(time.time() - t0, 1)
    output = {
        'strategy': 'Sentiment Contrarian + Fear-Greed Composite',
        'run_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_period': f"{df.index[0].date()} to {df.index[-1].date()}",
        'metrics': {
            'simple_contrarian': m_simple,
            'ml_contrarian_upro_shy': m_ml,
            'ml_contrarian_spy_ls': m_ml_spy,
            'spy_benchmark': m_spy,
        },
        'adversarial': {
            'simple': adv_simple,
            'ml': adv_ml,
        },
        'parameters': {
            'train_window': TRAIN_WINDOW,
            'label_gap': LABEL_GAP,
            'label_horizon_5d': LABEL_HORIZON_5D,
            'label_horizon_21d': LABEL_HORIZON_21D,
            'fear_threshold': FEAR_THRESHOLD,
            'greed_threshold': GREED_THRESHOLD,
            'ml_threshold': ML_THRESHOLD,
            'cost_bps': COST_BPS,
            'n_permutations': N_PERMUTATIONS,
            'initial_capital': INITIAL_CAPITAL,
        },
        'composite_description': {
            'components': [
                'VIX level + 252d percentile rank',
                'VIX term structure (VIX/VIX3M or proxy)',
                'Credit spread (HYG-LQD differential)',
                'Market breadth (IWM vs SPY RS)',
                'Safe haven demand (GLD vs SPY RS)',
                'VIX spike frequency (20d count of >2% jumps)',
                'Momentum exhaustion (RSI-14 + dist from 200MA)',
                'Realized vol (20d, 60d)',
                'Bond-equity correlation (60d)',
            ],
            'composite_method': 'Equal-weight z-score of all components (252d rolling)',
        },
        'runtime_seconds': runtime,
    }

    with open(OUTPUT / 'results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    simple_ret.to_csv(OUTPUT / 'daily_returns_simple.csv', header=['return'])
    ml_ret.to_csv(OUTPUT / 'daily_returns_ml.csv', header=['return'])

    # Save composite z-score for analysis
    feat['composite_z'].to_csv(OUTPUT / 'composite_fear_greed.csv', header=['composite_z'])

    print(f"\n{'=' * 80}")
    print(f"COMPLETE in {runtime:.0f}s")
    print(f"{'=' * 80}")
    print(f"\n  Results saved to {OUTPUT}")
    print(f"  Plots: equity_curves.png")

    return output


if __name__ == '__main__':
    main()
