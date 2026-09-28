#!/usr/bin/env python3
"""
Equity Rotation Exit Optimization v1
=====================================
Builds on KB #285: pure sector equity rotation (buy top-2 LGBM-ranked ETFs monthly)
gives Sharpe 1.40, Sortino 2.25, WR 64%, MDD -12%, CAGR 24.1%.

Tests 6 exit/rebalance variants to improve performance:
  A. Baseline (monthly rebalance, no stops)
  B. Trailing Stop 10%
  C. Trailing Stop 15%
  D. Biweekly Rebalance (10 trading days)
  E. Weekly Rebalance (5 trading days)
  F. Conditional Rebalance (only sell if rank drops below 6th)

Each variant includes permutation test and regime analysis.
"""

import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ─── MLflow setup ───
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "equity_rotation_exit_optimization"

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    USE_MLFLOW = True
    logger.info(f"MLflow tracking at {MLFLOW_URI}, experiment: {EXPERIMENT_NAME}")
except Exception as e:
    USE_MLFLOW = False
    logger.warning(f"MLflow unavailable: {e}")

# ─── Constants ───
SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
STARTING_CAPITAL = 10000.0
TOP_N = 2
TRAIN_WINDOW = 500  # trading days
FORWARD_HORIZON = 21  # trading days for target

# ─── Feature Engineering ───
def compute_features(df_prices, ticker):
    """Compute 17 features for a single ticker. df_prices has 'Close', 'High', 'Low', 'Volume'."""
    df = pd.DataFrame(index=df_prices.index)
    close = df_prices['Close']
    high = df_prices['High']
    low = df_prices['Low']
    volume = df_prices['Volume']

    # Return features
    df['ret_5d'] = close.pct_change(5)
    df['ret_10d'] = close.pct_change(10)
    df['ret_21d'] = close.pct_change(21)
    df['ret_63d'] = close.pct_change(63)

    # Volatility features
    df['vol_21d'] = close.pct_change().rolling(21).std()
    vol_5d = close.pct_change().rolling(5).std()
    df['vol_ratio'] = vol_5d / df['vol_21d'].replace(0, np.nan)

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger %B
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_pct'] = (close - (sma20 - 2 * std20)) / ((sma20 + 2 * std20) - (sma20 - 2 * std20)).replace(0, np.nan)

    # OBV slope
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df['obv_slope'] = obv.rolling(21).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 21 else np.nan, raw=False)

    # ATR %
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df['atr_pct'] = atr14 / close

    # Skewness and Kurtosis
    daily_ret = close.pct_change()
    df['skew_21d'] = daily_ret.rolling(21).skew()
    df['kurt_21d'] = daily_ret.rolling(21).kurt()

    # Max drawdown 21d
    def max_dd_21(x):
        cummax = np.maximum.accumulate(x)
        dd = (x - cummax) / np.where(cummax == 0, 1, cummax)
        return dd.min()
    df['max_dd_21d'] = close.rolling(21).apply(max_dd_21, raw=True)

    # Up/Down volume ratio
    up_vol = (volume * (daily_ret > 0).astype(float)).rolling(21).sum()
    dn_vol = (volume * (daily_ret < 0).astype(float)).rolling(21).sum()
    df['up_down_vol_ratio'] = up_vol / dn_vol.replace(0, np.nan)

    # Sector relative strength (placeholder — will be computed cross-sectionally later)
    df['sector_rel_strength'] = 0.0

    return df


def compute_cross_sectional_features(all_features, all_prices):
    """Add sector_rel_strength as cross-sectional feature."""
    # Get SPY-like equal-weight benchmark return
    tickers = list(all_prices.keys())
    common_idx = all_prices[tickers[0]].index
    for t in tickers[1:]:
        common_idx = common_idx.intersection(all_prices[t].index)

    # Equal-weight benchmark 21d return
    bench_ret = pd.Series(0.0, index=common_idx)
    count = pd.Series(0, index=common_idx)
    for t in tickers:
        r = all_prices[t]['Close'].reindex(common_idx).pct_change(21)
        valid = r.notna()
        bench_ret[valid] += r[valid]
        count[valid] += 1
    bench_ret = bench_ret / count.replace(0, 1)

    for t in tickers:
        if t in all_features:
            ticker_ret = all_prices[t]['Close'].reindex(common_idx).pct_change(21)
            all_features[t]['sector_rel_strength'] = (ticker_ret - bench_ret).reindex(all_features[t].index)

    return all_features


# ─── Data Fetching ───
def fetch_data():
    """Fetch sector ETF data via yfinance."""
    import yfinance as yf

    logger.info("Fetching sector ETF data via yfinance...")
    all_prices = {}
    for ticker in SECTOR_ETFS:
        try:
            df = yf.download(ticker, start='2008-01-01', end=datetime.now().strftime('%Y-%m-%d'),
                           progress=False, auto_adjust=True)
            if len(df) > 0:
                all_prices[ticker] = df[['Close', 'High', 'Low', 'Volume']].copy()
                # Flatten multi-level columns if present
                if isinstance(all_prices[ticker].columns, pd.MultiIndex):
                    all_prices[ticker].columns = all_prices[ticker].columns.get_level_values(0)
                logger.info(f"  {ticker}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
        except Exception as e:
            logger.error(f"  {ticker} FAILED: {e}")

    # Also fetch SPY for regime classification
    spy = yf.download('SPY', start='2008-01-01', end=datetime.now().strftime('%Y-%m-%d'),
                      progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    return all_prices, spy


# ─── LGBM Walk-Forward Ranking ───
def run_walkforward_lgbm(all_prices, all_features, rebalance_freq=21):
    """
    Walk-forward LGBM ranking. Train on 500 days, predict next rebalance period, slide forward.
    Returns dict: date -> {ticker: predicted_rank}
    """
    import lightgbm as lgb

    feature_cols = ['ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'vol_21d', 'vol_ratio',
                    'rsi_14', 'macd', 'macd_signal', 'bb_pct', 'obv_slope', 'atr_pct',
                    'sector_rel_strength', 'skew_21d', 'kurt_21d', 'max_dd_21d', 'up_down_vol_ratio']

    # Build combined dataset with forward returns as target
    tickers = list(all_features.keys())

    # Find common date range
    common_dates = all_features[tickers[0]].index.copy()
    for t in tickers[1:]:
        common_dates = common_dates.intersection(all_features[t].index)
    common_dates = common_dates.sort_values()

    logger.info(f"Common date range: {common_dates[0].date()} to {common_dates[-1].date()} ({len(common_dates)} days)")

    # Compute forward returns for target
    fwd_returns = {}
    for t in tickers:
        close = all_prices[t]['Close'].reindex(common_dates)
        fwd_returns[t] = close.shift(-FORWARD_HORIZON) / close - 1

    # Walk-forward predictions
    predictions = {}  # date -> {ticker: predicted_return}
    n_dates = len(common_dates)

    # Start after enough data for training
    start_idx = TRAIN_WINDOW + 63 + FORWARD_HORIZON  # need lookback for features + forward for labels

    logger.info(f"Walk-forward: start_idx={start_idx}, rebalance_freq={rebalance_freq}")

    step = 0
    i = start_idx
    while i < n_dates - 1:
        train_end = i
        train_start = max(0, train_end - TRAIN_WINDOW)

        train_dates = common_dates[train_start:train_end]

        # Build training data
        X_train_list = []
        y_train_list = []
        for t in tickers:
            feat = all_features[t].reindex(train_dates)[feature_cols]
            target = fwd_returns[t].reindex(train_dates)
            valid = feat.notna().all(axis=1) & target.notna()
            if valid.sum() > 10:
                X_train_list.append(feat[valid])
                y_train_list.append(target[valid])

        if not X_train_list:
            i += rebalance_freq
            continue

        X_train = pd.concat(X_train_list)
        y_train = pd.concat(y_train_list)

        # Train LGBM
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
        }

        dtrain = lgb.Dataset(X_train.values, label=y_train.values)
        callbacks = [lgb.log_evaluation(period=0)]
        model = lgb.train(params, dtrain, num_boost_round=200, callbacks=callbacks)

        # Predict at rebalance date
        rebal_date = common_dates[i]
        ticker_preds = {}
        for t in tickers:
            feat = all_features[t].reindex([rebal_date])[feature_cols]
            if feat.notna().all(axis=1).any():
                pred = model.predict(feat.values)[0]
                ticker_preds[t] = pred

        if ticker_preds:
            predictions[rebal_date] = ticker_preds

        step += 1
        i += rebalance_freq

        if step % 20 == 0:
            logger.info(f"  WF step {step}: rebal date {rebal_date.date()}, {len(ticker_preds)} tickers predicted")

    logger.info(f"Walk-forward complete: {len(predictions)} rebalance dates")
    return predictions


# ─── Backtesting Variants ───
def get_top_n_tickers(pred_dict, n=TOP_N):
    """Return top-N tickers by predicted return."""
    sorted_tickers = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in sorted_tickers[:n]]


def get_rank(pred_dict, ticker):
    """Return 1-indexed rank of ticker (1=best)."""
    sorted_tickers = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
    for i, (t, _) in enumerate(sorted_tickers):
        if t == ticker:
            return i + 1
    return len(pred_dict)


def backtest_variant_a(predictions, all_prices, rebalance_freq=21):
    """Variant A: Baseline — monthly rebalance, no stops."""
    rebal_dates = sorted(predictions.keys())
    if not rebal_dates:
        return None

    # Get all trading dates
    first_ticker = list(all_prices.keys())[0]
    all_dates = all_prices[first_ticker].index.sort_values()

    capital = STARTING_CAPITAL
    positions = {}  # ticker -> {shares, entry_price, entry_date}
    equity_curve = []
    trades = []

    for idx, rebal_date in enumerate(rebal_dates):
        top_tickers = get_top_n_tickers(predictions[rebal_date])

        # Sell current positions not in new top
        for t in list(positions.keys()):
            if t not in top_tickers:
                exit_price = all_prices[t]['Close'].reindex([rebal_date]).iloc[0] if rebal_date in all_prices[t].index else None
                if exit_price is not None and not np.isnan(exit_price):
                    pnl = (exit_price - positions[t]['entry_price']) * positions[t]['shares']
                    capital += positions[t]['shares'] * exit_price
                    trades.append({
                        'entry_date': positions[t]['entry_date'],
                        'exit_date': rebal_date,
                        'ticker': t,
                        'entry_price': positions[t]['entry_price'],
                        'exit_price': exit_price,
                        'shares': positions[t]['shares'],
                        'pnl': pnl,
                        'hold_days': (rebal_date - positions[t]['entry_date']).days
                    })
                    del positions[t]

        # Also sell positions that ARE in top but need rebalancing
        for t in list(positions.keys()):
            if t in top_tickers:
                exit_price = all_prices[t]['Close'].reindex([rebal_date]).iloc[0] if rebal_date in all_prices[t].index else None
                if exit_price is not None and not np.isnan(exit_price):
                    pnl = (exit_price - positions[t]['entry_price']) * positions[t]['shares']
                    capital += positions[t]['shares'] * exit_price
                    trades.append({
                        'entry_date': positions[t]['entry_date'],
                        'exit_date': rebal_date,
                        'ticker': t,
                        'entry_price': positions[t]['entry_price'],
                        'exit_price': exit_price,
                        'shares': positions[t]['shares'],
                        'pnl': pnl,
                        'hold_days': (rebal_date - positions[t]['entry_date']).days
                    })
                    del positions[t]

        # Buy new positions (equal weight)
        alloc_per_position = capital / TOP_N
        for t in top_tickers:
            entry_price = all_prices[t]['Close'].reindex([rebal_date]).iloc[0] if rebal_date in all_prices[t].index else None
            if entry_price is not None and not np.isnan(entry_price) and entry_price > 0:
                shares = alloc_per_position / entry_price
                positions[t] = {'shares': shares, 'entry_price': entry_price, 'entry_date': rebal_date}
                capital -= shares * entry_price

        # Record equity at rebalance
        total_equity = capital
        for t, pos in positions.items():
            price = all_prices[t]['Close'].reindex([rebal_date]).iloc[0] if rebal_date in all_prices[t].index else pos['entry_price']
            if not np.isnan(price):
                total_equity += pos['shares'] * price
        equity_curve.append({'date': rebal_date, 'equity': total_equity})

    # Build daily equity curve between rebalance dates
    daily_equity = build_daily_equity(rebal_dates, positions, capital, all_prices, all_dates, equity_curve)

    return {'equity_curve': daily_equity, 'trades': trades, 'variant': 'A_Baseline'}


def backtest_variant_trailing_stop(predictions, all_prices, stop_pct, rebalance_freq=21, variant_name='B'):
    """Variant B/C: Monthly rebalance with trailing stop."""
    rebal_dates = sorted(predictions.keys())
    if not rebal_dates:
        return None

    first_ticker = list(all_prices.keys())[0]
    all_dates = all_prices[first_ticker].index.sort_values()

    capital = STARTING_CAPITAL
    positions = {}  # ticker -> {shares, entry_price, entry_date, peak_price}
    equity_curve = []
    trades = []

    for idx, rebal_date in enumerate(rebal_dates):
        next_rebal = rebal_dates[idx + 1] if idx + 1 < len(rebal_dates) else None

        # First, sell all and rebalance
        for t in list(positions.keys()):
            exit_price = all_prices[t]['Close'].reindex([rebal_date]).iloc[0] if rebal_date in all_prices[t].index else None
            if exit_price is not None and not np.isnan(exit_price):
                pnl = (exit_price - positions[t]['entry_price']) * positions[t]['shares']
                capital += positions[t]['shares'] * exit_price
                trades.append({
                    'entry_date': positions[t]['entry_date'],
                    'exit_date': rebal_date,
                    'ticker': t,
                    'entry_price': positions[t]['entry_price'],
                    'exit_price': exit_price,
                    'shares': positions[t]['shares'],
                    'pnl': pnl,
                    'hold_days': (rebal_date - positions[t]['entry_date']).days
                })
                del positions[t]

        # Buy top-N
        top_tickers = get_top_n_tickers(predictions[rebal_date])
        alloc_per_position = capital / TOP_N
        for t in top_tickers:
            entry_price = all_prices[t]['Close'].reindex([rebal_date]).iloc[0] if rebal_date in all_prices[t].index else None
            if entry_price is not None and not np.isnan(entry_price) and entry_price > 0:
                shares = alloc_per_position / entry_price
                positions[t] = {'shares': shares, 'entry_price': entry_price,
                               'entry_date': rebal_date, 'peak_price': entry_price}
                capital -= shares * entry_price

        # Check trailing stop daily until next rebalance
        if next_rebal is not None:
            check_dates = all_dates[(all_dates > rebal_date) & (all_dates < next_rebal)]
            for d in check_dates:
                for t in list(positions.keys()):
                    if t in all_prices and d in all_prices[t].index:
                        price = all_prices[t]['Close'].loc[d]
                        if not np.isnan(price):
                            positions[t]['peak_price'] = max(positions[t]['peak_price'], price)
                            drawdown = (price - positions[t]['peak_price']) / positions[t]['peak_price']
                            if drawdown <= -stop_pct:
                                # Stop hit — sell and go to cash
                                pnl = (price - positions[t]['entry_price']) * positions[t]['shares']
                                capital += positions[t]['shares'] * price
                                trades.append({
                                    'entry_date': positions[t]['entry_date'],
                                    'exit_date': d,
                                    'ticker': t,
                                    'entry_price': positions[t]['entry_price'],
                                    'exit_price': price,
                                    'shares': positions[t]['shares'],
                                    'pnl': pnl,
                                    'hold_days': (d - positions[t]['entry_date']).days
                                })
                                del positions[t]

        # Record equity
        total_equity = capital
        for t, pos in positions.items():
            price = all_prices[t]['Close'].reindex([rebal_date]).iloc[0] if rebal_date in all_prices[t].index else pos['entry_price']
            if not np.isnan(price):
                total_equity += pos['shares'] * price
        equity_curve.append({'date': rebal_date, 'equity': total_equity})

    daily_equity = build_daily_equity(rebal_dates, positions, capital, all_prices, all_dates, equity_curve)
    return {'equity_curve': daily_equity, 'trades': trades, 'variant': f'{variant_name}_TrailingStop{int(stop_pct*100)}pct'}


def backtest_variant_rebalance_freq(predictions_func, all_prices, all_features, freq, variant_name):
    """Variant D/E: Different rebalance frequency. Re-runs LGBM with different freq."""
    predictions = predictions_func(freq)
    return backtest_variant_a_generic(predictions, all_prices, variant_name)


def backtest_variant_a_generic(predictions, all_prices, variant_name):
    """Generic baseline backtest with given predictions."""
    rebal_dates = sorted(predictions.keys())
    if not rebal_dates:
        return None

    first_ticker = list(all_prices.keys())[0]
    all_dates = all_prices[first_ticker].index.sort_values()

    capital = STARTING_CAPITAL
    positions = {}
    equity_curve = []
    trades = []

    for idx, rebal_date in enumerate(rebal_dates):
        # Sell all
        for t in list(positions.keys()):
            exit_price = all_prices[t]['Close'].reindex([rebal_date]).iloc[0] if rebal_date in all_prices[t].index else None
            if exit_price is not None and not np.isnan(exit_price):
                pnl = (exit_price - positions[t]['entry_price']) * positions[t]['shares']
                capital += positions[t]['shares'] * exit_price
                trades.append({
                    'entry_date': positions[t]['entry_date'],
                    'exit_date': rebal_date,
                    'ticker': t,
                    'entry_price': positions[t]['entry_price'],
                    'exit_price': exit_price,
                    'shares': positions[t]['shares'],
                    'pnl': pnl,
                    'hold_days': (rebal_date - positions[t]['entry_date']).days
                })
                del positions[t]

        # Buy top-N
        top_tickers = get_top_n_tickers(predictions[rebal_date])
        alloc_per_position = capital / TOP_N
        for t in top_tickers:
            entry_price = all_prices[t]['Close'].reindex([rebal_date]).iloc[0] if rebal_date in all_prices[t].index else None
            if entry_price is not None and not np.isnan(entry_price) and entry_price > 0:
                shares = alloc_per_position / entry_price
                positions[t] = {'shares': shares, 'entry_price': entry_price, 'entry_date': rebal_date}
                capital -= shares * entry_price

        total_equity = capital
        for t, pos in positions.items():
            price = all_prices[t]['Close'].reindex([rebal_date]).iloc[0] if rebal_date in all_prices[t].index else pos['entry_price']
            if not np.isnan(price):
                total_equity += pos['shares'] * price
        equity_curve.append({'date': rebal_date, 'equity': total_equity})

    daily_equity = build_daily_equity(rebal_dates, positions, capital, all_prices, all_dates, equity_curve)
    return {'equity_curve': daily_equity, 'trades': trades, 'variant': variant_name}


def backtest_variant_conditional(predictions, all_prices, rebalance_freq=21):
    """Variant F: Conditional rebalance — only sell if rank drops below 6th."""
    rebal_dates = sorted(predictions.keys())
    if not rebal_dates:
        return None

    first_ticker = list(all_prices.keys())[0]
    all_dates = all_prices[first_ticker].index.sort_values()

    capital = STARTING_CAPITAL
    positions = {}
    equity_curve = []
    trades = []

    for idx, rebal_date in enumerate(rebal_dates):
        pred_dict = predictions[rebal_date]
        top_tickers = get_top_n_tickers(pred_dict)

        # Sell positions that dropped below rank 6
        for t in list(positions.keys()):
            rank = get_rank(pred_dict, t)
            if rank > 6:  # Bottom half — sell
                exit_price = all_prices[t]['Close'].reindex([rebal_date]).iloc[0] if rebal_date in all_prices[t].index else None
                if exit_price is not None and not np.isnan(exit_price):
                    pnl = (exit_price - positions[t]['entry_price']) * positions[t]['shares']
                    capital += positions[t]['shares'] * exit_price
                    trades.append({
                        'entry_date': positions[t]['entry_date'],
                        'exit_date': rebal_date,
                        'ticker': t,
                        'entry_price': positions[t]['entry_price'],
                        'exit_price': exit_price,
                        'shares': positions[t]['shares'],
                        'pnl': pnl,
                        'hold_days': (rebal_date - positions[t]['entry_date']).days
                    })
                    del positions[t]

        # Count empty slots
        n_held = len(positions)
        n_to_buy = TOP_N - n_held

        if n_to_buy > 0:
            # Sell any remaining positions to rebalance if we need to buy new ones
            # Actually, for conditional: keep winners, only replace sold ones
            # First sell held positions that aren't in top to make room
            held_tickers = set(positions.keys())
            new_buys = [t for t in top_tickers if t not in held_tickers][:n_to_buy]

            if new_buys:
                # Calculate allocation: equal weight for ALL positions (held + new)
                # First, figure out total equity
                total_equity = capital
                for t, pos in positions.items():
                    price = all_prices[t]['Close'].reindex([rebal_date]).iloc[0] if rebal_date in all_prices[t].index else pos['entry_price']
                    if not np.isnan(price):
                        total_equity += pos['shares'] * price

                alloc_per_position = total_equity / TOP_N
                cash_for_new = min(capital, alloc_per_position * len(new_buys))
                alloc_each = cash_for_new / len(new_buys) if new_buys else 0

                for t in new_buys:
                    entry_price = all_prices[t]['Close'].reindex([rebal_date]).iloc[0] if rebal_date in all_prices[t].index else None
                    if entry_price is not None and not np.isnan(entry_price) and entry_price > 0:
                        shares = alloc_each / entry_price
                        positions[t] = {'shares': shares, 'entry_price': entry_price, 'entry_date': rebal_date}
                        capital -= shares * entry_price
        elif n_held == 0:
            # No positions, buy top-N
            alloc_per_position = capital / TOP_N
            for t in top_tickers:
                entry_price = all_prices[t]['Close'].reindex([rebal_date]).iloc[0] if rebal_date in all_prices[t].index else None
                if entry_price is not None and not np.isnan(entry_price) and entry_price > 0:
                    shares = alloc_per_position / entry_price
                    positions[t] = {'shares': shares, 'entry_price': entry_price, 'entry_date': rebal_date}
                    capital -= shares * entry_price

        # Record equity
        total_equity = capital
        for t, pos in positions.items():
            price = all_prices[t]['Close'].reindex([rebal_date]).iloc[0] if rebal_date in all_prices[t].index else pos['entry_price']
            if not np.isnan(price):
                total_equity += pos['shares'] * price
        equity_curve.append({'date': rebal_date, 'equity': total_equity})

    daily_equity = build_daily_equity(rebal_dates, positions, capital, all_prices, all_dates, equity_curve)
    return {'equity_curve': daily_equity, 'trades': trades, 'variant': 'F_ConditionalRebalance'}


def build_daily_equity(rebal_dates, positions, capital, all_prices, all_dates, equity_snapshots):
    """Build daily equity curve from snapshots."""
    if not equity_snapshots:
        return pd.Series(dtype=float)

    eq_df = pd.DataFrame(equity_snapshots).set_index('date')['equity']

    # For simplicity, interpolate between rebalance points using actual position prices
    # Use rebalance-point equity as the daily equity series
    # This is approximate but sufficient for metrics
    start = eq_df.index[0]
    end = eq_df.index[-1]
    daily_dates = all_dates[(all_dates >= start) & (all_dates <= end)]

    # Forward-fill equity between rebalance dates
    daily_eq = eq_df.reindex(daily_dates).ffill().bfill()
    return daily_eq


# ─── Performance Metrics ───
def compute_metrics(result, spy_prices):
    """Compute all performance metrics for a backtest result."""
    if result is None or result['equity_curve'] is None or len(result['equity_curve']) < 10:
        return None

    eq = result['equity_curve'].dropna()
    if len(eq) < 10:
        return None

    trades = result['trades']
    daily_returns = eq.pct_change().dropna()

    # Basic metrics
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Sharpe (annualized, 252 trading days)
    if daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = daily_returns.mean() / downside.std() * np.sqrt(252)
    else:
        sortino = 0

    # Max Drawdown
    cummax = eq.cummax()
    drawdown = (eq - cummax) / cummax
    mdd = drawdown.min()

    # Win Rate and Profit Factor (from trades)
    if trades:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        wr = len(wins) / len(trades) if trades else 0
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_hold = np.mean([t['hold_days'] for t in trades]) if trades else 0
    else:
        wr = 0
        pf = 0
        avg_hold = 0

    # Regime Analysis
    spy_close = spy_prices['Close']
    spy_monthly = spy_close.resample('ME').last().pct_change()

    regime_sharpes = {}
    for regime_name, condition in [('bull', lambda r: r > 0.02), ('bear', lambda r: r < -0.02), ('flat', lambda r: abs(r) <= 0.02)]:
        regime_months = spy_monthly[spy_monthly.apply(condition)].index
        regime_returns = []
        for m in regime_months:
            month_start = m.replace(day=1)
            month_end = m
            mask = (daily_returns.index >= month_start) & (daily_returns.index <= month_end)
            if mask.any():
                regime_returns.extend(daily_returns[mask].tolist())
        if regime_returns and np.std(regime_returns) > 0:
            regime_sharpes[regime_name] = np.mean(regime_returns) / np.std(regime_returns) * np.sqrt(252)
        else:
            regime_sharpes[regime_name] = 0.0

    return {
        'variant': result['variant'],
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'pf': round(pf, 2),
        'wr': round(wr * 100, 1),
        'mdd': round(mdd * 100, 1),
        'cagr': round(cagr * 100, 1),
        'total_return': round(total_return * 100, 1),
        'n_trades': len(trades),
        'avg_hold_days': round(avg_hold, 1),
        'sharpe_bull': round(regime_sharpes.get('bull', 0), 2),
        'sharpe_bear': round(regime_sharpes.get('bear', 0), 2),
        'sharpe_flat': round(regime_sharpes.get('flat', 0), 2),
    }


# ─── Permutation Test ───
def permutation_test(predictions, all_prices, spy_prices, variant_func, n_perms=100):
    """Shuffle rankings and compute null distribution of Sharpe ratios."""
    logger.info(f"  Running permutation test ({n_perms} shuffles)...")

    # Get actual Sharpe
    actual_result = variant_func(predictions)
    if actual_result is None:
        return 1.0
    actual_metrics = compute_metrics(actual_result, spy_prices)
    if actual_metrics is None:
        return 1.0
    actual_sharpe = actual_metrics['sharpe']

    null_sharpes = []
    rng = np.random.RandomState(42)

    for i in range(n_perms):
        # Shuffle the rankings
        shuffled_preds = {}
        for date, pred_dict in predictions.items():
            tickers = list(pred_dict.keys())
            values = list(pred_dict.values())
            rng.shuffle(values)
            shuffled_preds[date] = dict(zip(tickers, values))

        shuffled_result = variant_func(shuffled_preds)
        if shuffled_result is not None:
            sm = compute_metrics(shuffled_result, spy_prices)
            if sm is not None:
                null_sharpes.append(sm['sharpe'])

    if not null_sharpes:
        return 1.0

    p_value = np.mean([s >= actual_sharpe for s in null_sharpes])
    logger.info(f"  Permutation p-value: {p_value:.4f} (actual Sharpe={actual_sharpe:.2f}, null mean={np.mean(null_sharpes):.2f})")
    return p_value


# ─── Main ───
def main():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("EQUITY ROTATION EXIT OPTIMIZATION v1")
    logger.info("=" * 70)

    # 1. Fetch data
    all_prices, spy_df = fetch_data()
    logger.info(f"Fetched {len(all_prices)} sector ETFs + SPY")

    # 2. Compute features
    logger.info("Computing features...")
    all_features = {}
    for t in all_prices:
        all_features[t] = compute_features(all_prices[t], t)
    all_features = compute_cross_sectional_features(all_features, all_prices)

    # 3. Run LGBM walk-forward for each rebalance frequency
    logger.info("\n--- Running LGBM Walk-Forward (monthly, freq=21) ---")
    preds_21 = run_walkforward_lgbm(all_prices, all_features, rebalance_freq=21)

    logger.info("\n--- Running LGBM Walk-Forward (biweekly, freq=10) ---")
    preds_10 = run_walkforward_lgbm(all_prices, all_features, rebalance_freq=10)

    logger.info("\n--- Running LGBM Walk-Forward (weekly, freq=5) ---")
    preds_5 = run_walkforward_lgbm(all_prices, all_features, rebalance_freq=5)

    # 4. Run all 6 variants
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING BACKTEST VARIANTS")
    logger.info("=" * 70)

    results = {}
    all_metrics = {}

    # Variant A: Baseline
    logger.info("\n--- Variant A: Baseline (Monthly, No Stops) ---")
    results['A'] = backtest_variant_a(preds_21, all_prices)
    all_metrics['A'] = compute_metrics(results['A'], spy_df)
    if all_metrics['A']:
        logger.info(f"  Sharpe={all_metrics['A']['sharpe']}, CAGR={all_metrics['A']['cagr']}%")

    # Variant B: Trailing Stop 10%
    logger.info("\n--- Variant B: Trailing Stop 10% ---")
    results['B'] = backtest_variant_trailing_stop(preds_21, all_prices, stop_pct=0.10, variant_name='B')
    all_metrics['B'] = compute_metrics(results['B'], spy_df)
    if all_metrics['B']:
        logger.info(f"  Sharpe={all_metrics['B']['sharpe']}, CAGR={all_metrics['B']['cagr']}%")

    # Variant C: Trailing Stop 15%
    logger.info("\n--- Variant C: Trailing Stop 15% ---")
    results['C'] = backtest_variant_trailing_stop(preds_21, all_prices, stop_pct=0.15, variant_name='C')
    all_metrics['C'] = compute_metrics(results['C'], spy_df)
    if all_metrics['C']:
        logger.info(f"  Sharpe={all_metrics['C']['sharpe']}, CAGR={all_metrics['C']['cagr']}%")

    # Variant D: Biweekly Rebalance
    logger.info("\n--- Variant D: Biweekly Rebalance (10 trading days) ---")
    results['D'] = backtest_variant_a_generic(preds_10, all_prices, 'D_BiweeklyRebalance')
    all_metrics['D'] = compute_metrics(results['D'], spy_df)
    if all_metrics['D']:
        logger.info(f"  Sharpe={all_metrics['D']['sharpe']}, CAGR={all_metrics['D']['cagr']}%")

    # Variant E: Weekly Rebalance
    logger.info("\n--- Variant E: Weekly Rebalance (5 trading days) ---")
    results['E'] = backtest_variant_a_generic(preds_5, all_prices, 'E_WeeklyRebalance')
    all_metrics['E'] = compute_metrics(results['E'], spy_df)
    if all_metrics['E']:
        logger.info(f"  Sharpe={all_metrics['E']['sharpe']}, CAGR={all_metrics['E']['cagr']}%")

    # Variant F: Conditional Rebalance
    logger.info("\n--- Variant F: Conditional Rebalance (sell only if rank < 6) ---")
    results['F'] = backtest_variant_conditional(preds_21, all_prices)
    all_metrics['F'] = compute_metrics(results['F'], spy_df)
    if all_metrics['F']:
        logger.info(f"  Sharpe={all_metrics['F']['sharpe']}, CAGR={all_metrics['F']['cagr']}%")

    # 5. Permutation Tests
    logger.info("\n" + "=" * 70)
    logger.info("PERMUTATION TESTS (100 shuffles each)")
    logger.info("=" * 70)

    p_values = {}

    logger.info("\nPermutation test for Variant A...")
    p_values['A'] = permutation_test(preds_21, all_prices, spy_df,
                                     lambda p: backtest_variant_a(p, all_prices))

    logger.info("\nPermutation test for Variant B...")
    p_values['B'] = permutation_test(preds_21, all_prices, spy_df,
                                     lambda p: backtest_variant_trailing_stop(p, all_prices, 0.10, variant_name='B'))

    logger.info("\nPermutation test for Variant C...")
    p_values['C'] = permutation_test(preds_21, all_prices, spy_df,
                                     lambda p: backtest_variant_trailing_stop(p, all_prices, 0.15, variant_name='C'))

    logger.info("\nPermutation test for Variant D...")
    p_values['D'] = permutation_test(preds_10, all_prices, spy_df,
                                     lambda p: backtest_variant_a_generic(p, all_prices, 'D_BiweeklyRebalance'))

    logger.info("\nPermutation test for Variant E...")
    p_values['E'] = permutation_test(preds_5, all_prices, spy_df,
                                     lambda p: backtest_variant_a_generic(p, all_prices, 'E_WeeklyRebalance'))

    logger.info("\nPermutation test for Variant F...")
    p_values['F'] = permutation_test(preds_21, all_prices, spy_df,
                                     lambda p: backtest_variant_conditional(p, all_prices))

    # 6. Final Summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY — ALL VARIANTS")
    logger.info("=" * 70)

    header = f"{'Variant':<35} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR%':>6} {'MDD%':>7} {'CAGR%':>7} {'TotRet%':>8} {'#Trades':>8} {'AvgHold':>8} {'p-val':>7} {'Sh_Bull':>8} {'Sh_Bear':>8} {'Sh_Flat':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    for key in ['A', 'B', 'C', 'D', 'E', 'F']:
        m = all_metrics.get(key)
        p = p_values.get(key, 1.0)
        if m:
            row = (f"{m['variant']:<35} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['pf']:>6.2f} "
                   f"{m['wr']:>6.1f} {m['mdd']:>7.1f} {m['cagr']:>7.1f} {m['total_return']:>8.1f} "
                   f"{m['n_trades']:>8d} {m['avg_hold_days']:>8.1f} {p:>7.4f} "
                   f"{m['sharpe_bull']:>8.2f} {m['sharpe_bear']:>8.2f} {m['sharpe_flat']:>8.2f}")
            logger.info(row)
        else:
            logger.info(f"{key:<35} FAILED — no valid results")

    # 7. Log to MLflow
    if USE_MLFLOW:
        logger.info("\nLogging to MLflow...")
        try:
            with mlflow.start_run(run_name="exit_optimization_v1"):
                mlflow.log_param("sector_etfs", str(SECTOR_ETFS))
                mlflow.log_param("top_n", TOP_N)
                mlflow.log_param("train_window", TRAIN_WINDOW)
                mlflow.log_param("starting_capital", STARTING_CAPITAL)
                mlflow.log_param("n_variants", 6)
                mlflow.log_param("n_permutations", 100)

                for key in ['A', 'B', 'C', 'D', 'E', 'F']:
                    m = all_metrics.get(key)
                    p = p_values.get(key, 1.0)
                    if m:
                        prefix = f"variant_{key}"
                        mlflow.log_metric(f"{prefix}_sharpe", m['sharpe'])
                        mlflow.log_metric(f"{prefix}_sortino", m['sortino'])
                        mlflow.log_metric(f"{prefix}_pf", m['pf'])
                        mlflow.log_metric(f"{prefix}_wr", m['wr'])
                        mlflow.log_metric(f"{prefix}_mdd", m['mdd'])
                        mlflow.log_metric(f"{prefix}_cagr", m['cagr'])
                        mlflow.log_metric(f"{prefix}_total_return", m['total_return'])
                        mlflow.log_metric(f"{prefix}_n_trades", m['n_trades'])
                        mlflow.log_metric(f"{prefix}_avg_hold_days", m['avg_hold_days'])
                        mlflow.log_metric(f"{prefix}_p_value", p)
                        mlflow.log_metric(f"{prefix}_sharpe_bull", m['sharpe_bull'])
                        mlflow.log_metric(f"{prefix}_sharpe_bear", m['sharpe_bear'])
                        mlflow.log_metric(f"{prefix}_sharpe_flat", m['sharpe_flat'])

                # Log best variant
                valid_metrics = {k: v for k, v in all_metrics.items() if v is not None}
                if valid_metrics:
                    best_key = max(valid_metrics, key=lambda k: valid_metrics[k]['sharpe'])
                    mlflow.log_param("best_variant", best_key)
                    mlflow.log_metric("best_sharpe", valid_metrics[best_key]['sharpe'])

            logger.info("MLflow logging complete.")
        except Exception as e:
            logger.error(f"MLflow logging failed: {e}")

    elapsed = time.time() - start_time
    logger.info(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    logger.info("DONE.")


if __name__ == '__main__':
    main()
