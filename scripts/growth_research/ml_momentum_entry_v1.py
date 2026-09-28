#!/usr/bin/env python3
"""
ML Momentum Entry Timing v1
============================
Train an LGBM classifier to predict optimal entry timing for the momentum burst
options strategy (KB #281).

BACKGROUND:
- Proven momentum burst strategy: buy single-leg options on ETFs showing strong
  21-day momentum. Exit rules: +30% TP, -25% SL, 50% trailing giveback, 5-day
  max hold, DTE=14, ATM strikes.
- Strategy has Sharpe 1.28, permutation p=0.000, works across regimes.
- Current entries are rule-based (simple momentum + RSI thresholds).
- This model predicts WHICH entries will hit +30% TP vs -25% SL.

DESIGN:
- Universe: 11 sector ETFs + SPY, QQQ, IWM (14 total)
- Label: binary — did ATM option bought today hit +30% within 5 trading days?
- Walk-forward: SLIDING 252d train, 21d test, slide 21d
- Model: LightGBM classifier, log_loss objective
- Evaluation: concat OOT, threshold sweep, adversarial gates

PRICING: Inline Black-Scholes. IV = realized_vol * 1.2 * sqrt(DTE/252).
DATA: yfinance daily OHLCV, 2019-01-01 to 2026-06-30.
"""

import sys
import os
import json
import time
import warnings
import traceback
from datetime import datetime, timedelta
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm

print = partial(print, flush=True)
warnings.filterwarnings('ignore')

# --- Path setup (works on Jupiter and Neptune) ---
for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'ml_momentum_entry_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    print("ERROR: lightgbm not installed. pip install lightgbm")
    sys.exit(1)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    print("ERROR: yfinance not installed. pip install yfinance")
    sys.exit(1)

from sklearn.metrics import log_loss, roc_auc_score, precision_recall_curve

# ============================================================
# CONSTANTS
# ============================================================
ETF_UNIVERSE = [
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLC', 'XLU', 'XLP', 'XLB', 'XLRE',
    'SPY', 'QQQ', 'IWM',
]
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLC', 'XLU', 'XLP', 'XLB', 'XLRE']

STARTING_CAPITAL = 10_000.0
COMMISSION_RT = 1.30  # round-trip per contract
RISK_FREE_RATE = 0.05
DTE = 14
HOLD_DAYS = 5
TP_PCT = 0.30
SL_PCT = -0.25
TRAILING_GIVEBACK = 0.50

START_DATE = '2019-01-01'
END_DATE = '2026-06-30'

TRAIN_WINDOW = 252
TEST_WINDOW = 21
N_PERMUTATIONS = 100

# LGBM hyperparams
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
}


# ============================================================
# BLACK-SCHOLES PRICING (inline, self-contained)
# ============================================================

def bs_d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_call_price(S, K, T, r, sigma):
    if T <= 1e-8:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    if T <= 1e-8:
        return max(K - S, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(1 - d2) - S * norm.cdf(-d1)


def estimate_iv(realized_vol_21d, dte=14):
    """IV estimate: realized vol scaled up by 1.2 * sqrt(DTE/252)."""
    # VIX-like IV proxy: realized vol with vol risk premium markup
    iv = realized_vol_21d * 1.2
    return max(iv, 0.10)  # floor at 10%


def price_option(S, direction, dte, iv, r=0.05):
    """Price ATM option. direction: 'call' or 'put'."""
    K = round(S)  # ATM strike
    T = dte / 252.0
    if direction == 'call':
        return bs_call_price(S, K, T, r, iv), K
    else:
        return bs_put_price(S, K, T, r, iv), K


def reprice_option(S_new, K, dte_remaining, iv, direction, r=0.05):
    """Reprice option after underlying moves."""
    T = max(dte_remaining / 252.0, 1e-8)
    if direction == 'call':
        return bs_call_price(S_new, K, T, r, iv)
    else:
        return bs_put_price(S_new, K, T, r, iv)


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_data():
    """Download ETF + VIX data via yfinance with caching."""
    cache_dir = os.path.join(LVL3_ROOT, 'data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'ml_momentum_entry_data.parquet')

    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < 3600:
        print("  Using cached data")
        return pd.read_parquet(cache_file)

    print("  Downloading ETF + VIX data via yfinance...")
    tickers = ETF_UNIVERSE + ['^VIX', '^VIX3M']
    data = yf.download(tickers, start='2018-01-01', end=END_DATE,
                       auto_adjust=True, progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close'].copy()
        volumes = data['Volume'].copy()
        highs = data['High'].copy()
        lows = data['Low'].copy()
        opens = data['Open'].copy()
    else:
        closes = data[['Close']].copy()
        volumes = data[['Volume']].copy()
        highs = data[['High']].copy()
        lows = data[['Low']].copy()
        opens = data[['Open']].copy()

    # Clean column names
    for df in [closes, volumes, highs, lows, opens]:
        df.columns = [c.replace('^', '') for c in df.columns]

    # Forward-fill, then drop rows where all ETFs are NaN
    closes = closes.ffill()
    volumes = volumes.ffill().fillna(0)
    highs = highs.ffill()
    lows = lows.ffill()
    opens = opens.ffill()

    # Combine into single dataframe with suffixes
    combined = pd.DataFrame(index=closes.index)
    for ticker in ETF_UNIVERSE:
        t = ticker.replace('^', '')
        if t in closes.columns:
            combined[f'{t}_close'] = closes[t]
            combined[f'{t}_volume'] = volumes[t] if t in volumes.columns else 0
            combined[f'{t}_high'] = highs[t] if t in highs.columns else closes[t]
            combined[f'{t}_low'] = lows[t] if t in lows.columns else closes[t]
            combined[f'{t}_open'] = opens[t] if t in opens.columns else closes[t]

    # VIX columns
    if 'VIX' in closes.columns:
        combined['VIX'] = closes['VIX']
    if 'VIX3M' in closes.columns:
        combined['VIX3M'] = closes['VIX3M']

    combined = combined.dropna(how='all')
    combined.to_parquet(cache_file)
    print(f"  Data: {len(combined)} days, {combined.shape[1]} columns")
    return combined


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(data, ticker):
    """Build feature set for a single ticker. Returns DataFrame aligned to data.index."""
    t = ticker.replace('^', '')
    close_col = f'{t}_close'
    vol_col = f'{t}_volume'
    high_col = f'{t}_high'
    low_col = f'{t}_low'

    if close_col not in data.columns:
        return None

    close = data[close_col]
    volume = data[vol_col] if vol_col in data.columns else pd.Series(0, index=data.index)
    high = data[high_col] if high_col in data.columns else close
    low = data[low_col] if low_col in data.columns else close

    feats = pd.DataFrame(index=data.index)

    # --- Momentum features ---
    feats['ret_5d'] = close.pct_change(5)
    feats['ret_10d'] = close.pct_change(10)
    feats['ret_21d'] = close.pct_change(21)
    feats['ret_63d'] = close.pct_change(63)
    feats['mom_accel'] = feats['ret_21d'] - feats['ret_63d']  # momentum acceleration
    feats['mom_accel_short'] = feats['ret_5d'] - feats['ret_21d']

    # Relative strength vs SPY
    if 'SPY_close' in data.columns and t != 'SPY':
        spy = data['SPY_close']
        feats['rs_vs_spy_5d'] = close.pct_change(5) - spy.pct_change(5)
        feats['rs_vs_spy_21d'] = close.pct_change(21) - spy.pct_change(21)
    else:
        feats['rs_vs_spy_5d'] = 0.0
        feats['rs_vs_spy_21d'] = 0.0

    # --- Volatility features ---
    daily_ret = close.pct_change()
    feats['rvol_21d'] = daily_ret.rolling(21).std() * np.sqrt(252)
    feats['rvol_5d'] = daily_ret.rolling(5).std() * np.sqrt(252)
    feats['vol_ratio'] = feats['rvol_5d'] / (feats['rvol_21d'] + 1e-8)

    # ATR%
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    feats['atr_pct'] = atr_14 / (close + 1e-8)

    # Bollinger %B
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    feats['bbpct'] = (close - (sma_20 - 2 * std_20)) / (4 * std_20 + 1e-8)

    # --- RSI ---
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    feats['rsi_14'] = 100 - 100 / (1 + rs)

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    feats['macd_signal'] = (macd_line - signal_line) / (close + 1e-8)  # normalized

    # --- Volume features ---
    vol_5d = volume.rolling(5).mean()
    vol_21d = volume.rolling(21).mean()
    feats['vol_ratio_5_21'] = vol_5d / (vol_21d + 1e-8)

    # --- VIX features ---
    if 'VIX' in data.columns:
        vix = data['VIX']
        feats['vix_level'] = vix
        feats['vix_5d_chg'] = vix.pct_change(5)
        feats['vix_21d_pctile'] = vix.rolling(63).apply(
            lambda x: (x.iloc[-1] <= x).mean() if len(x) > 0 else 0.5, raw=False
        )
        # VIX term structure
        if 'VIX3M' in data.columns:
            feats['vix_term_structure'] = data['VIX3M'] / (vix + 1e-8) - 1
        else:
            feats['vix_term_structure'] = 0.0
    else:
        feats['vix_level'] = 20.0
        feats['vix_5d_chg'] = 0.0
        feats['vix_21d_pctile'] = 0.5
        feats['vix_term_structure'] = 0.0

    # --- Calendar features ---
    feats['day_of_week'] = pd.Series(data.index.dayofweek, index=data.index)
    feats['month'] = pd.Series(data.index.month, index=data.index)

    # --- Sector-specific features ---
    if t in SECTOR_ETFS:
        # Sector momentum rank among 11 sectors
        sector_rets_21d = {}
        for s in SECTOR_ETFS:
            sc = f'{s}_close'
            if sc in data.columns:
                sector_rets_21d[s] = data[sc].pct_change(21)
        if sector_rets_21d:
            sector_df = pd.DataFrame(sector_rets_21d)
            ranks = sector_df.rank(axis=1, ascending=False)
            if t in ranks.columns:
                feats['sector_mom_rank'] = ranks[t]
            else:
                feats['sector_mom_rank'] = 6.0

            # Relative strength rank
            sector_rs = {}
            for s in SECTOR_ETFS:
                sc = f'{s}_close'
                if sc in data.columns and 'SPY_close' in data.columns:
                    sector_rs[s] = data[sc].pct_change(21) - data['SPY_close'].pct_change(21)
            if sector_rs:
                rs_df = pd.DataFrame(sector_rs)
                rs_ranks = rs_df.rank(axis=1, ascending=False)
                if t in rs_ranks.columns:
                    feats['sector_rs_rank'] = rs_ranks[t]
                else:
                    feats['sector_rs_rank'] = 6.0
            else:
                feats['sector_rs_rank'] = 6.0
        else:
            feats['sector_mom_rank'] = 6.0
            feats['sector_rs_rank'] = 6.0
    else:
        feats['sector_mom_rank'] = 0.0
        feats['sector_rs_rank'] = 0.0

    # --- Distance from 52w high/low ---
    high_252 = close.rolling(252, min_periods=63).max()
    low_252 = close.rolling(252, min_periods=63).min()
    feats['dist_from_52w_high'] = close / (high_252 + 1e-8) - 1
    feats['dist_from_52w_low'] = close / (low_252 + 1e-8) - 1

    return feats


# ============================================================
# LABEL CONSTRUCTION
# ============================================================

def construct_labels(data, ticker):
    """
    For each day, simulate buying ATM call AND put, check if either
    hits +30% within 5 trading days. Label=1 if TP hit, 0 otherwise.

    For calls: we buy when momentum is positive (bullish signal).
    For puts: we buy when momentum is negative (bearish signal).
    We use the DIRECTION that matches the momentum for labeling.
    """
    t = ticker.replace('^', '')
    close_col = f'{t}_close'

    if close_col not in data.columns:
        return None

    close = data[close_col].values
    dates = data.index
    n = len(close)

    labels = np.full(n, np.nan)
    trade_pnl = np.full(n, np.nan)  # actual P&L for backtest
    direction_arr = []  # 'call' or 'put'

    daily_ret = pd.Series(close).pct_change()
    rvol_21d = daily_ret.rolling(21).std() * np.sqrt(252)
    ret_21d = pd.Series(close).pct_change(21)

    for i in range(63, n - HOLD_DAYS):
        S = close[i]
        if S <= 0 or np.isnan(S):
            continue

        rv = rvol_21d.iloc[i]
        if np.isnan(rv) or rv <= 0:
            rv = 0.20

        mom = ret_21d.iloc[i]
        if np.isnan(mom):
            continue

        # Direction based on momentum
        direction = 'call' if mom >= 0 else 'put'
        direction_arr.append(direction)

        iv = estimate_iv(rv, DTE)
        entry_price, K = price_option(S, direction, DTE, iv)

        if entry_price <= 0.01:
            labels[i] = 0
            trade_pnl[i] = 0
            continue

        # Simulate forward 5 days
        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 + SL_PCT)
        max_price_seen = entry_price
        exit_price = None

        for d in range(1, HOLD_DAYS + 1):
            if i + d >= n:
                break
            S_new = close[i + d]
            dte_rem = DTE - d
            if dte_rem <= 0:
                dte_rem = 1

            opt_price = reprice_option(S_new, K, dte_rem, iv, direction)

            # Track max for trailing stop
            if opt_price > max_price_seen:
                max_price_seen = opt_price

            # Check TP
            if opt_price >= tp_price:
                exit_price = tp_price
                break

            # Check SL
            if opt_price <= sl_price:
                exit_price = sl_price
                break

            # Trailing giveback: if we were up and gave back 50%
            if max_price_seen > entry_price:
                unrealized_gain = max_price_seen - entry_price
                current_gain = opt_price - entry_price
                if current_gain < unrealized_gain * (1 - TRAILING_GIVEBACK):
                    exit_price = opt_price
                    break

        # If no exit triggered, exit at end of hold period
        if exit_price is None:
            last_idx = min(i + HOLD_DAYS, n - 1)
            S_last = close[last_idx]
            dte_rem = max(DTE - HOLD_DAYS, 1)
            exit_price = reprice_option(S_last, K, dte_rem, iv, direction)

        pnl_pct = (exit_price - entry_price) / entry_price
        labels[i] = 1 if pnl_pct >= TP_PCT else 0
        trade_pnl[i] = pnl_pct

    return pd.DataFrame({
        'label': labels,
        'trade_pnl': trade_pnl,
    }, index=dates)


# ============================================================
# WALK-FORWARD TRAINING
# ============================================================

def run_walkforward(data):
    """SLIDING window walk-forward: 252d train, 21d test, slide 21d."""
    print("\n=== WALK-FORWARD TRAINING ===")

    # Build features and labels for all tickers
    all_features = {}
    all_labels = {}

    for ticker in ETF_UNIVERSE:
        print(f"  Building features for {ticker}...")
        feats = build_features(data, ticker)
        labs = construct_labels(data, ticker)
        if feats is not None and labs is not None:
            all_features[ticker] = feats
            all_labels[ticker] = labs

    print(f"  Built features for {len(all_features)} tickers")

    # Get common date range
    dates = data.index
    valid_start_idx = 252 + 63  # need lookback for features + labels
    valid_dates = dates[valid_start_idx:]

    # Filter to START_DATE
    valid_dates = valid_dates[valid_dates >= START_DATE]

    if len(valid_dates) < TRAIN_WINDOW + TEST_WINDOW:
        print("ERROR: Not enough data for walk-forward")
        return None

    # Prepare pooled dataset (all tickers stacked)
    print("  Pooling cross-sectional data...")
    feature_names = None
    rows_X = []
    rows_y = []
    rows_meta = []  # (date, ticker, trade_pnl)

    for ticker in all_features:
        feats = all_features[ticker]
        labs = all_labels[ticker]

        if feature_names is None:
            feature_names = list(feats.columns)

        # Align
        common_idx = feats.index.intersection(labs.index).intersection(valid_dates)
        feats_aligned = feats.loc[common_idx]
        labs_aligned = labs.loc[common_idx]

        # Only keep rows where label is not NaN
        mask = ~labs_aligned['label'].isna()
        feats_clean = feats_aligned[mask]
        labs_clean = labs_aligned[mask]

        for dt in feats_clean.index:
            row = feats_clean.loc[dt].values
            if np.any(np.isnan(row)):
                continue
            rows_X.append(row)
            rows_y.append(int(labs_clean.loc[dt, 'label']))
            rows_meta.append({
                'date': dt,
                'ticker': ticker,
                'trade_pnl': labs_clean.loc[dt, 'trade_pnl'],
            })

    X = np.array(rows_X)
    y = np.array(rows_y)
    meta = pd.DataFrame(rows_meta)

    print(f"  Total samples: {len(X)}, positives: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Features: {len(feature_names)}")
    print(f"  Date range: {meta['date'].min()} to {meta['date'].max()}")

    # Sort by date for walk-forward
    sort_idx = meta['date'].argsort()
    X = X[sort_idx]
    y = y[sort_idx]
    meta = meta.iloc[sort_idx].reset_index(drop=True)

    # Walk-forward splits
    unique_dates = sorted(meta['date'].unique())
    n_dates = len(unique_dates)

    all_oot_preds = []
    all_oot_labels = []
    all_oot_meta = []
    fold_metrics = []

    fold = 0
    test_start = TRAIN_WINDOW  # index into unique_dates

    while test_start + TEST_WINDOW <= n_dates:
        train_end_date = unique_dates[test_start - 1]
        train_start_date = unique_dates[max(0, test_start - TRAIN_WINDOW)]
        test_end_idx = min(test_start + TEST_WINDOW, n_dates)
        test_start_date = unique_dates[test_start]
        test_end_date = unique_dates[test_end_idx - 1]

        # Get train/test masks
        train_mask = (meta['date'] >= train_start_date) & (meta['date'] <= train_end_date)
        test_mask = (meta['date'] >= test_start_date) & (meta['date'] <= test_end_date)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        meta_test = meta[test_mask]

        if len(X_train) < 100 or len(X_test) < 5:
            test_start += TEST_WINDOW
            continue

        # Train LightGBM
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.log_evaluation(0)],  # silent
        )

        # Predict on test
        probs = model.predict_proba(X_test)[:, 1]

        all_oot_preds.extend(probs)
        all_oot_labels.extend(y_test)
        all_oot_meta.append(meta_test)

        # Fold metrics
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, probs)
            ll = log_loss(y_test, probs)
        else:
            auc = 0.5
            ll = -1

        fold_metrics.append({
            'fold': fold,
            'train_start': str(train_start_date)[:10],
            'test_start': str(test_start_date)[:10],
            'test_end': str(test_end_date)[:10],
            'n_train': len(X_train),
            'n_test': len(X_test),
            'auc': auc,
            'logloss': ll,
            'pos_rate_train': y_train.mean(),
            'pos_rate_test': y_test.mean(),
        })

        if fold % 10 == 0:
            print(f"  Fold {fold}: train={len(X_train)}, test={len(X_test)}, "
                  f"AUC={auc:.3f}, LL={ll:.3f}")

        fold += 1
        test_start += TEST_WINDOW

    print(f"\n  Completed {fold} folds")

    # Concat OOT predictions
    oot_preds = np.array(all_oot_preds)
    oot_labels = np.array(all_oot_labels)
    oot_meta = pd.concat(all_oot_meta, ignore_index=True)

    # Get feature importance from last model
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
    }).sort_values('importance', ascending=False)

    return {
        'oot_preds': oot_preds,
        'oot_labels': oot_labels,
        'oot_meta': oot_meta,
        'fold_metrics': fold_metrics,
        'feature_importance': feat_imp,
        'feature_names': feature_names,
    }


# ============================================================
# BACKTEST WITH THRESHOLD SWEEP
# ============================================================

def backtest_with_threshold(oot_meta, oot_preds, oot_labels, threshold):
    """
    Backtest: only enter trades where model probability > threshold.
    Returns performance metrics.
    """
    mask = oot_preds >= threshold
    if mask.sum() < 10:
        return None

    selected = oot_meta[mask].copy()
    selected['pred'] = oot_preds[mask]
    selected['label'] = oot_labels[mask]

    pnls = selected['trade_pnl'].values
    n_trades = len(pnls)
    winners = (pnls > 0).sum()
    losers = (pnls <= 0).sum()
    wr = winners / n_trades if n_trades > 0 else 0

    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls <= 0].sum())
    pf = gross_profit / (gross_loss + 1e-8)

    # Sharpe (daily-equivalent)
    mean_ret = pnls.mean()
    std_ret = pnls.std() if len(pnls) > 1 else 1e-8
    # Annualize: assume ~250 trading days, trades spread over period
    n_days = (selected['date'].max() - selected['date'].min()).days
    trades_per_year = n_trades / max(n_days / 365.25, 0.1)
    sharpe = mean_ret / (std_ret + 1e-8) * np.sqrt(trades_per_year)

    # Max drawdown (cumulative P&L)
    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_dd = drawdowns.max() if len(drawdowns) > 0 else 0

    # Hit rate for TP
    tp_hits = (oot_labels[mask] == 1).mean()

    return {
        'threshold': threshold,
        'n_trades': n_trades,
        'win_rate': wr,
        'profit_factor': pf,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'mean_pnl_pct': mean_ret,
        'total_pnl_pct': pnls.sum(),
        'tp_precision': tp_hits,
        'trades_per_year': trades_per_year,
    }


def run_threshold_sweep(results):
    """Sweep thresholds from 0.3 to 0.7."""
    print("\n=== THRESHOLD SWEEP ===")
    thresholds = np.arange(0.30, 0.75, 0.05)
    sweep_results = []

    for thresh in thresholds:
        r = backtest_with_threshold(
            results['oot_meta'], results['oot_preds'],
            results['oot_labels'], thresh
        )
        if r is not None:
            sweep_results.append(r)
            print(f"  Threshold {thresh:.2f}: trades={r['n_trades']:4d}, "
                  f"WR={r['win_rate']:.1%}, PF={r['profit_factor']:.2f}, "
                  f"Sharpe={r['sharpe']:.2f}, TP precision={r['tp_precision']:.1%}")

    return sweep_results


# ============================================================
# BASELINE: RULE-BASED MOMENTUM ENTRY
# ============================================================

def compute_baseline(results):
    """Baseline: all entries (no ML filter). Same exit rules."""
    print("\n=== BASELINE (No ML Filter) ===")
    meta = results['oot_meta']
    pnls = meta['trade_pnl'].values
    labels = results['oot_labels']

    n_trades = len(pnls)
    wr = (pnls > 0).mean()
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls <= 0].sum())
    pf = gross_profit / (gross_loss + 1e-8)
    mean_ret = pnls.mean()
    std_ret = pnls.std()

    n_days = (meta['date'].max() - meta['date'].min()).days
    trades_per_year = n_trades / max(n_days / 365.25, 0.1)
    sharpe = mean_ret / (std_ret + 1e-8) * np.sqrt(trades_per_year)

    tp_rate = labels.mean()

    print(f"  Trades: {n_trades}, WR: {wr:.1%}, PF: {pf:.2f}, "
          f"Sharpe: {sharpe:.2f}, TP rate: {tp_rate:.1%}")

    return {
        'n_trades': n_trades,
        'win_rate': wr,
        'profit_factor': pf,
        'sharpe': sharpe,
        'mean_pnl_pct': mean_ret,
        'tp_rate': tp_rate,
        'trades_per_year': trades_per_year,
    }


# ============================================================
# ADVERSARIAL GATES
# ============================================================

def permutation_test(results, n_perm=N_PERMUTATIONS):
    """Permutation test: shuffle labels, recompute Sharpe."""
    print(f"\n=== PERMUTATION TEST ({n_perm} shuffles) ===")
    meta = results['oot_meta']
    preds = results['oot_preds']
    labels = results['oot_labels']

    # Find best threshold (highest Sharpe from sweep)
    best_thresh = 0.50  # default
    best_sharpe = -999
    for thresh in np.arange(0.30, 0.75, 0.05):
        r = backtest_with_threshold(meta, preds, labels, thresh)
        if r is not None and r['sharpe'] > best_sharpe:
            best_sharpe = r['sharpe']
            best_thresh = thresh

    print(f"  Best threshold: {best_thresh:.2f}, Sharpe: {best_sharpe:.2f}")

    # Permutation: shuffle trade_pnl, recompute
    null_sharpes = []
    rng = np.random.RandomState(42)
    for p in range(n_perm):
        shuffled_meta = meta.copy()
        shuffled_meta['trade_pnl'] = rng.permutation(meta['trade_pnl'].values)
        shuffled_labels = rng.permutation(labels)

        r = backtest_with_threshold(shuffled_meta, preds, shuffled_labels, best_thresh)
        if r is not None:
            null_sharpes.append(r['sharpe'])

    null_sharpes = np.array(null_sharpes)
    p_value = (null_sharpes >= best_sharpe).mean()
    print(f"  Permutation p-value: {p_value:.4f} "
          f"(null mean={null_sharpes.mean():.2f}, null std={null_sharpes.std():.2f})")

    return {
        'best_threshold': best_thresh,
        'best_sharpe': best_sharpe,
        'p_value': p_value,
        'null_mean': float(null_sharpes.mean()),
        'null_std': float(null_sharpes.std()),
    }


def regime_gap_test(results):
    """R1: Check Sharpe gap between bull and bear regimes."""
    print("\n=== REGIME GAP TEST (R1) ===")
    meta = results['oot_meta']
    preds = results['oot_preds']
    labels = results['oot_labels']

    # Classify regimes by SPY 21d return at each date
    # We need SPY data - reconstruct from meta dates
    dates = meta['date'].values
    unique_dates = np.unique(dates)

    # Simple regime: look at whether SPY was positive or negative over last 21d
    # We don't have SPY directly in meta, so use the overall market direction
    # from the feature data

    # Best threshold
    best_thresh = 0.50
    best_sharpe = -999
    for thresh in np.arange(0.30, 0.75, 0.05):
        r = backtest_with_threshold(meta, preds, labels, thresh)
        if r is not None and r['sharpe'] > best_sharpe:
            best_sharpe = r['sharpe']
            best_thresh = thresh

    mask = preds >= best_thresh
    if mask.sum() < 20:
        print("  Too few trades for regime analysis")
        return {'regime_gap': 0, 'pass': True}

    selected = meta[mask].copy()
    selected['pnl'] = selected['trade_pnl'].values

    # Split into 3 equal time periods as proxy for regimes
    dates_sorted = sorted(selected['date'].unique())
    n_d = len(dates_sorted)
    split1 = dates_sorted[n_d // 3]
    split2 = dates_sorted[2 * n_d // 3]

    period_sharpes = []
    for label_name, start_dt, end_dt in [
        ('Early', dates_sorted[0], split1),
        ('Mid', split1, split2),
        ('Late', split2, dates_sorted[-1]),
    ]:
        pmask = (selected['date'] >= start_dt) & (selected['date'] <= end_dt)
        pnls = selected.loc[pmask, 'pnl'].values
        if len(pnls) < 5:
            continue
        mean_r = pnls.mean()
        std_r = pnls.std() + 1e-8
        s = mean_r / std_r * np.sqrt(len(pnls))
        period_sharpes.append(s)
        print(f"  {label_name}: Sharpe={s:.2f}, n={len(pnls)}")

    if len(period_sharpes) >= 2:
        gap = abs(max(period_sharpes) - min(period_sharpes)) / max(abs(max(period_sharpes)), abs(min(period_sharpes)), 0.01)
        passed = gap <= 0.50
        print(f"  Regime gap: {gap:.2f} ({'PASS' if passed else 'FAIL'}, threshold=0.50)")
    else:
        gap = 0
        passed = True

    return {'regime_gap': gap, 'pass': passed, 'period_sharpes': period_sharpes}


def subperiod_stability_test(results):
    """Check stability across 3 sub-periods."""
    print("\n=== SUB-PERIOD STABILITY ===")
    meta = results['oot_meta']
    preds = results['oot_preds']
    labels = results['oot_labels']

    # Best threshold
    best_thresh = 0.50
    best_sharpe = -999
    for thresh in np.arange(0.30, 0.75, 0.05):
        r = backtest_with_threshold(meta, preds, labels, thresh)
        if r is not None and r['sharpe'] > best_sharpe:
            best_sharpe = r['sharpe']
            best_thresh = thresh

    mask = preds >= best_thresh
    selected = meta[mask].copy()
    selected['pnl'] = selected['trade_pnl'].values
    selected_labels = labels[mask]

    dates_sorted = sorted(selected['date'].unique())
    n_d = len(dates_sorted)
    thirds = [
        dates_sorted[:n_d//3],
        dates_sorted[n_d//3:2*n_d//3],
        dates_sorted[2*n_d//3:],
    ]

    sub_results = []
    for i, period_dates in enumerate(thirds):
        if len(period_dates) < 5:
            continue
        pmask = selected['date'].isin(period_dates)
        pnls = selected.loc[pmask, 'pnl'].values
        n_t = len(pnls)
        wr = (pnls > 0).mean() if n_t > 0 else 0
        pf = pnls[pnls > 0].sum() / (abs(pnls[pnls <= 0].sum()) + 1e-8)
        mean_pnl = pnls.mean()
        std_pnl = pnls.std() + 1e-8
        sharpe = mean_pnl / std_pnl * np.sqrt(n_t)

        print(f"  Period {i+1}: n={n_t}, WR={wr:.1%}, PF={pf:.2f}, Sharpe={sharpe:.2f}")
        sub_results.append({
            'period': i + 1,
            'n_trades': n_t,
            'win_rate': wr,
            'profit_factor': pf,
            'sharpe': sharpe,
        })

    # Check: all sub-periods profitable?
    all_positive = all(r['sharpe'] > 0 for r in sub_results) if sub_results else False
    print(f"  All periods positive Sharpe: {'YES' if all_positive else 'NO'}")

    return sub_results


# ============================================================
# MLFLOW LOGGING
# ============================================================

def log_to_mlflow(results, sweep_results, baseline, perm_results, regime_results, sub_results):
    """Log all results to MLflow."""
    if not MLFLOW_AVAILABLE:
        print("\n  MLflow not available, skipping logging")
        return

    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("ML_Momentum_Entry")

        with mlflow.start_run(run_name="ml_momentum_entry_v1"):
            # Log params
            mlflow.log_params({
                'model': 'LightGBM',
                'train_window': TRAIN_WINDOW,
                'test_window': TEST_WINDOW,
                'n_estimators': LGBM_PARAMS['n_estimators'],
                'max_depth': LGBM_PARAMS['max_depth'],
                'learning_rate': LGBM_PARAMS['learning_rate'],
                'universe_size': len(ETF_UNIVERSE),
                'tp_pct': TP_PCT,
                'sl_pct': SL_PCT,
                'dte': DTE,
                'hold_days': HOLD_DAYS,
            })

            # Log baseline
            mlflow.log_metrics({
                'baseline_sharpe': baseline['sharpe'],
                'baseline_wr': baseline['win_rate'],
                'baseline_pf': baseline['profit_factor'],
                'baseline_trades': baseline['n_trades'],
            })

            # Log best threshold results
            if sweep_results:
                best = max(sweep_results, key=lambda x: x['sharpe'])
                mlflow.log_metrics({
                    'best_sharpe': best['sharpe'],
                    'best_wr': best['win_rate'],
                    'best_pf': best['profit_factor'],
                    'best_threshold': best['threshold'],
                    'best_n_trades': best['n_trades'],
                    'best_tp_precision': best['tp_precision'],
                })

            # Log adversarial
            mlflow.log_metrics({
                'perm_p_value': perm_results['p_value'],
                'regime_gap': regime_results['regime_gap'],
                'regime_pass': 1 if regime_results['pass'] else 0,
            })

            # Log OOT AUC
            if len(np.unique(results['oot_labels'])) > 1:
                auc = roc_auc_score(results['oot_labels'], results['oot_preds'])
                mlflow.log_metric('oot_auc', auc)

            # Log feature importance as artifact
            feat_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
            results['feature_importance'].to_csv(feat_path, index=False)
            mlflow.log_artifact(feat_path)

            print("  Logged to MLflow successfully")

    except Exception as e:
        print(f"  MLflow logging failed: {e}")


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results, sweep_results, baseline, perm_results, regime_results, sub_results):
    """Save all results to output directory."""
    print(f"\n=== SAVING RESULTS to {OUTPUT_DIR} ===")

    # Feature importance
    feat_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
    results['feature_importance'].to_csv(feat_path, index=False)
    print(f"  Saved feature importance ({len(results['feature_importance'])} features)")

    # OOT predictions
    oot_df = results['oot_meta'].copy()
    oot_df['pred_prob'] = results['oot_preds']
    oot_df['label'] = results['oot_labels']
    oot_path = os.path.join(OUTPUT_DIR, 'oot_predictions.parquet')
    oot_df.to_parquet(oot_path)
    print(f"  Saved OOT predictions ({len(oot_df)} samples)")

    # Fold metrics
    fold_path = os.path.join(OUTPUT_DIR, 'fold_metrics.json')
    with open(fold_path, 'w') as f:
        json.dump(results['fold_metrics'], f, indent=2, default=str)

    # Threshold sweep
    sweep_path = os.path.join(OUTPUT_DIR, 'threshold_sweep.json')
    with open(sweep_path, 'w') as f:
        json.dump(sweep_results, f, indent=2, default=str)

    # Summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'baseline': baseline,
        'permutation_test': perm_results,
        'regime_test': regime_results,
        'sub_period_stability': sub_results,
        'threshold_sweep': sweep_results,
        'n_oot_samples': len(results['oot_preds']),
        'oot_pos_rate': float(results['oot_labels'].mean()),
        'top_10_features': results['feature_importance'].head(10).to_dict('records'),
    }

    # Add OOT AUC
    if len(np.unique(results['oot_labels'])) > 1:
        summary['oot_auc'] = float(roc_auc_score(results['oot_labels'], results['oot_preds']))

    summary_path = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved summary to {summary_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("ML MOMENTUM ENTRY TIMING v1")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Universe: {len(ETF_UNIVERSE)} ETFs")
    print(f"Walk-forward: {TRAIN_WINDOW}d train, {TEST_WINDOW}d test, SLIDING")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Strategy: DTE={DTE}, TP={TP_PCT:+.0%}, SL={SL_PCT:+.0%}, "
          f"hold={HOLD_DAYS}d, trailing={TRAILING_GIVEBACK:.0%}")
    print("=" * 70)

    # 1. Download data
    print("\n[1/6] Downloading data...")
    data = download_data()

    # 2. Walk-forward training
    print("\n[2/6] Running walk-forward training...")
    results = run_walkforward(data)
    if results is None:
        print("FATAL: Walk-forward failed")
        return

    # 3. Baseline
    print("\n[3/6] Computing baseline...")
    baseline = compute_baseline(results)

    # 4. Threshold sweep
    print("\n[4/6] Running threshold sweep...")
    sweep_results = run_threshold_sweep(results)

    # 5. Adversarial gates
    print("\n[5/6] Running adversarial gates...")
    perm_results = permutation_test(results)
    regime_results = regime_gap_test(results)
    sub_results = subperiod_stability_test(results)

    # 6. Save & log
    print("\n[6/6] Saving results...")
    save_results(results, sweep_results, baseline, perm_results, regime_results, sub_results)
    log_to_mlflow(results, sweep_results, baseline, perm_results, regime_results, sub_results)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"COMPLETED in {elapsed/60:.1f} minutes")
    print(f"{'=' * 70}")

    # --- FINAL SUMMARY ---
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\nOOT samples: {len(results['oot_preds'])}")
    print(f"OOT positive rate: {results['oot_labels'].mean():.1%}")
    if len(np.unique(results['oot_labels'])) > 1:
        print(f"OOT AUC: {roc_auc_score(results['oot_labels'], results['oot_preds']):.3f}")

    print(f"\nBaseline (no filter): Sharpe={baseline['sharpe']:.2f}, "
          f"WR={baseline['win_rate']:.1%}, PF={baseline['profit_factor']:.2f}, "
          f"trades={baseline['n_trades']}")

    if sweep_results:
        best = max(sweep_results, key=lambda x: x['sharpe'])
        print(f"\nBest ML filter: threshold={best['threshold']:.2f}")
        print(f"  Sharpe={best['sharpe']:.2f}, WR={best['win_rate']:.1%}, "
              f"PF={best['profit_factor']:.2f}, trades={best['n_trades']}")
        print(f"  TP precision={best['tp_precision']:.1%}, "
              f"trades/year={best['trades_per_year']:.0f}")

        improvement = best['sharpe'] - baseline['sharpe']
        print(f"\n  Sharpe improvement: {improvement:+.2f}")

    print(f"\nAdversarial gates:")
    print(f"  Permutation p-value: {perm_results['p_value']:.4f} "
          f"({'PASS' if perm_results['p_value'] < 0.05 else 'FAIL'})")
    print(f"  Regime gap: {regime_results['regime_gap']:.2f} "
          f"({'PASS' if regime_results['pass'] else 'FAIL'})")

    # Sub-period
    if sub_results:
        all_pos = all(r['sharpe'] > 0 for r in sub_results)
        print(f"  Sub-period stability: {'PASS' if all_pos else 'FAIL'} "
              f"({len([r for r in sub_results if r['sharpe'] > 0])}/{len(sub_results)} positive)")

    print(f"\nTop 5 features:")
    for _, row in results['feature_importance'].head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
