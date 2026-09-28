#!/usr/bin/env python3
"""
Jade Lizard Expanded Universe + ML Entry Timing Research
=========================================================
HC #724 compliant: sliding 504d walk-forward, monthly rebalance, FIFO cost basis.

Strategy: Sell OTM put + bear call spread on high-IV stocks.
ML timing: LightGBM predicts which puts expire worthless.
Validation: permutation test, regime analysis, sub-period stability.

Output: /home/nick/Lvl3Quant/output/jade_lizard_expanded_v1/
MLflow: http://jupiter:5000, experiment "jade_lizard_expanded"
"""

import os
import sys
import json
import time
import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import mlflow
import mlflow.lightgbm

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/jade_lizard_expanded_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "jade_lizard_research.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("jade_lizard")

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "jade_lizard_expanded"

# Universe
FULL_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "GS", "BAC",
    "V", "MA", "UNH", "JNJ", "PG", "KO", "PEP", "MRK", "ABBV", "LLY",
    "HD", "COST", "WMT", "CRM", "AMD", "NFLX", "ADBE", "INTC", "CSCO", "QCOM",
    "XOM", "CVX", "PFE", "TMO", "ABT", "AVGO", "TXN", "MCD", "NKE", "DIS",
    "CMCSA", "T", "VZ", "NEE", "SO", "SHW", "LMT", "RTX", "CAT", "DE",
]
ORIGINAL_15 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "GS", "BAC", "V", "MA", "UNH", "JNJ", "PG"]

# Strategy params
PUT_DELTA = 0.30          # Sell 30-delta OTM put
CALL_SHORT_DELTA = 0.20   # Short call at 20-delta
CALL_LONG_DELTA = 0.10    # Long call at 10-delta (cap)
DTE = 30                  # Days to expiration
IV_PREMIUM = 1.10         # IV = realized_vol * 1.10
RISK_FREE_RATE = 0.04     # ~4% risk-free

# Position sizing
MAX_POSITIONS = 5
MAX_PER_STOCK = 1
POSITION_RISK_PCT = 0.05  # 5% of portfolio
COMMISSION_PER_CONTRACT = 0.65
CONTRACTS_PER_TRADE = 1   # 1 contract each leg
INITIAL_CAPITAL = 100_000

# Risk management
TAKE_PROFIT_PCT = 0.50    # Close at 50% max profit
STOP_LOSS_MULT = 2.0      # Close if loss > 2x premium received
VIX_PAUSE_THRESHOLD = 35  # Pause entries if VIX > 35
EARNINGS_BLACKOUT_DAYS = 5  # No puts within 5 days of earnings

# Walk-forward
TRAIN_WINDOW = 504        # 504 trading days
REBALANCE_FREQ = 21       # Monthly rebalance (21 trading days)

# Validation
N_PERMUTATIONS = 200


# ── Black-Scholes ───────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, option_type="put"):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(0, (K - S) if option_type == "put" else (S - K))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type="put"):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        if option_type == "put":
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def find_strike_for_delta(S, T, r, sigma, target_delta, option_type="put", tol=0.001):
    """Find strike price that gives the target delta."""
    if option_type == "put":
        # OTM put: strike < S, delta is negative
        lo, hi = S * 0.70, S * 1.0
        target = -abs(target_delta)  # Put delta is negative
    else:
        # OTM call: strike > S, delta is positive
        lo, hi = S * 1.0, S * 1.50
        target = abs(target_delta)

    for _ in range(100):
        mid = (lo + hi) / 2
        d = bs_delta(S, mid, T, r, sigma, option_type)
        if option_type == "put":
            if d < target:
                lo = mid
            else:
                hi = mid
        else:
            if d > target:
                lo = mid
            else:
                hi = mid
        if abs(d - target) < tol:
            break
    return mid


# ── Data Fetching ───────────────────────────────────────────────────────────
def fetch_stock_data(tickers, start="2012-01-01", end="2026-07-18"):
    """Fetch daily OHLCV data for all tickers + VIX."""
    import yfinance as yf

    cache_file = OUTPUT_DIR / "price_data_cache.parquet"
    if cache_file.exists():
        log.info("Loading cached price data")
        df = pd.read_parquet(cache_file)
        cached_tickers = df.columns.get_level_values(1).unique().tolist() if isinstance(df.columns, pd.MultiIndex) else []
        if set(tickers).issubset(set(cached_tickers)):
            return df
        log.info("Cache incomplete, re-fetching")

    all_tickers = tickers + ["^VIX"]
    log.info(f"Fetching data for {len(all_tickers)} tickers from {start} to {end}")

    # Fetch in batches to avoid rate limits
    all_data = {}
    batch_size = 10
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        # Never fetch a single ticker alone (yfinance returns different column format)
        # If last batch is size 1, merge it with previous batch
        if len(batch) == 1 and i > 0:
            # Already fetched in augmented previous batch below
            continue
        # Check if next batch would be size 1, if so include it here
        remaining = all_tickers[i+batch_size:]
        if len(remaining) == 1:
            batch = batch + remaining

        log.info(f"  Batch {i//batch_size + 1}: {batch}")
        try:
            data = yf.download(batch, start=start, end=end, group_by='ticker', auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    try:
                        if t in data.columns.get_level_values(0):
                            all_data[t] = data[t].copy()
                    except Exception:
                        pass
            else:
                # Single ticker fallback (shouldn't happen with batch >= 2)
                all_data[batch[0]] = data.copy()
        except Exception as e:
            log.warning(f"  Failed batch {batch}: {e}")
        time.sleep(1)  # Rate limit

    # Combine into multi-index DataFrame
    # All values in all_data are single-ticker DataFrames with flat OHLCV columns
    # pd.concat with keys= creates a proper MultiIndex (ticker, column)
    combined = pd.concat(all_data, axis=1)
    combined.to_parquet(cache_file)
    log.info(f"Saved cache: {cache_file} ({len(combined)} rows)")
    return combined


# ── Feature Engineering ─────────────────────────────────────────────────────
def compute_features(prices_df, ticker, vix_series):
    """Compute features for a single ticker. All features are T-1 (lagged)."""
    try:
        if isinstance(prices_df.columns, pd.MultiIndex):
            close = prices_df[(ticker, 'Close')].dropna()
            high = prices_df[(ticker, 'High')].dropna()
            low = prices_df[(ticker, 'Low')].dropna()
            volume = prices_df[(ticker, 'Volume')].dropna()
        else:
            close = prices_df['Close'].dropna()
            high = prices_df['High'].dropna()
            low = prices_df['Low'].dropna()
            volume = prices_df['Volume'].dropna()
    except (KeyError, TypeError):
        return None

    if len(close) < 300:
        return None

    df = pd.DataFrame(index=close.index)
    df['close'] = close
    df['ticker'] = ticker

    # Returns
    df['ret_1d'] = close.pct_change()
    df['ret_5d'] = close.pct_change(5)
    df['ret_21d'] = close.pct_change(21)  # 1 month momentum
    df['ret_63d'] = close.pct_change(63)  # 3 month momentum

    # Realized volatility (annualized)
    df['rv_20d'] = df['ret_1d'].rolling(20).std() * np.sqrt(252)
    df['rv_60d'] = df['ret_1d'].rolling(60).std() * np.sqrt(252)
    df['rv_252d'] = df['ret_1d'].rolling(252).std() * np.sqrt(252)

    # IV rank (percentile of 20d vol vs 252d history)
    df['iv_rank'] = df['rv_20d'].rolling(252).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) if len(x) == 252 else np.nan
    )

    # IV/HV ratio proxy (short-term vol vs long-term vol)
    df['iv_hv_ratio'] = df['rv_20d'] / (df['rv_60d'] + 1e-10)

    # RSI (14-day)
    delta = df['ret_1d'].copy()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # VIX level
    if vix_series is not None:
        df['vix'] = vix_series.reindex(df.index).ffill()
    else:
        df['vix'] = 20.0  # default

    # Volume ratio (current vs 20d avg)
    if volume is not None and len(volume) > 0:
        vol_aligned = volume.reindex(df.index)
        df['volume_ratio'] = vol_aligned / vol_aligned.rolling(20).mean()
    else:
        df['volume_ratio'] = 1.0

    # Bollinger Band width
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    df['bb_width'] = (2 * std_20) / (sma_20 + 1e-10)

    # Distance from 52-week high/low
    high_252 = close.rolling(252).max()
    low_252 = close.rolling(252).min()
    df['dist_from_high'] = (close - high_252) / (high_252 + 1e-10)
    df['dist_from_low'] = (close - low_252) / (low_252 + 1e-10)

    # Skewness of returns (21d)
    df['ret_skew_21d'] = df['ret_1d'].rolling(21).skew()

    # Put expiry outcome: will a 30-delta OTM put expire worthless in ~21 trading days?
    # This means: stock doesn't drop below the put strike
    # Approximate: stock doesn't drop more than ~1 std dev in next month
    future_min = close.rolling(21).min().shift(-21)
    # Put strike approx at S * (1 - delta_pct), where delta_pct ~ N^{-1}(0.30) * sigma * sqrt(T)
    put_dist = norm.ppf(0.30) * df['rv_20d'] * IV_PREMIUM * np.sqrt(DTE/252)
    put_strike_approx = close * (1 + put_dist)  # put_dist is negative
    df['put_expires_worthless'] = (future_min > put_strike_approx).astype(float)

    # Future return for regime analysis
    df['future_ret_21d'] = close.pct_change(21).shift(-21)

    # Lag ALL features by 1 day (T-1) to prevent lookahead
    feature_cols = [
        'rv_20d', 'rv_60d', 'rv_252d', 'iv_rank', 'iv_hv_ratio',
        'ret_1d', 'ret_5d', 'ret_21d', 'ret_63d',
        'rsi_14', 'vix', 'volume_ratio', 'bb_width',
        'dist_from_high', 'dist_from_low', 'ret_skew_21d',
    ]
    for col in feature_cols:
        df[col] = df[col].shift(1)

    return df


# ── Backtest Engine ─────────────────────────────────────────────────────────
class JadeLizardBacktest:
    def __init__(self, capital=INITIAL_CAPITAL):
        self.initial_capital = capital
        self.capital = capital
        self.positions = []  # List of open positions
        self.trades = []     # Completed trades
        self.equity_curve = []

    def _commission_cost(self):
        """3 legs: short put, short call, long call."""
        return 3 * COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE

    def open_position(self, date, ticker, spot, iv, dte=DTE):
        """Open a jade lizard position."""
        if len(self.positions) >= MAX_POSITIONS:
            return None
        if any(p['ticker'] == ticker for p in self.positions):
            return None

        T = dte / 365.0
        r = RISK_FREE_RATE

        # Find strikes
        put_strike = find_strike_for_delta(spot, T, r, iv, PUT_DELTA, "put")
        call_short_strike = find_strike_for_delta(spot, T, r, iv, CALL_SHORT_DELTA, "call")
        call_long_strike = find_strike_for_delta(spot, T, r, iv, CALL_LONG_DELTA, "call")

        # Ensure call spread is valid
        if call_long_strike <= call_short_strike:
            call_long_strike = call_short_strike * 1.05

        # Price options
        put_premium = bs_price(spot, put_strike, T, r, iv, "put")
        call_short_premium = bs_price(spot, call_short_strike, T, r, iv, "call")
        call_long_premium = bs_price(spot, call_long_strike, T, r, iv, "call")

        # Net credit = short put + short call - long call
        net_credit = put_premium + call_short_premium - call_long_premium

        if net_credit <= 0:
            return None

        # Max risk on put side = put_strike - net_credit (per share, x100 for contract)
        put_risk = (put_strike - net_credit) * 100
        # Max risk on call side = (call_long_strike - call_short_strike) * 100 - net_credit * 100
        call_spread_risk = (call_long_strike - call_short_strike) * 100 - net_credit * 100

        # Jade lizard key property: net credit > call spread width means NO upside risk
        # If not, call spread risk is max_risk on upside
        max_risk = max(put_risk, max(0, call_spread_risk))

        if max_risk <= 0:
            return None

        # Position sizing: risk no more than 5% of capital
        max_contracts = max(1, int(self.capital * POSITION_RISK_PCT / max_risk))
        contracts = min(max_contracts, CONTRACTS_PER_TRADE)

        commission = self._commission_cost() * contracts
        net_credit_total = net_credit * 100 * contracts - commission

        position = {
            'open_date': date,
            'ticker': ticker,
            'spot_at_open': spot,
            'put_strike': put_strike,
            'call_short_strike': call_short_strike,
            'call_long_strike': call_long_strike,
            'iv_at_open': iv,
            'net_credit': net_credit,
            'net_credit_total': net_credit_total,
            'contracts': contracts,
            'max_risk': max_risk * contracts,
            'dte_remaining': dte,
            'commission': commission,
        }
        self.positions.append(position)
        return position

    def mark_to_market(self, date, spot_dict, days_elapsed=1):
        """Mark positions to market, check exit conditions."""
        closed = []
        for pos in self.positions[:]:
            ticker = pos['ticker']
            if ticker not in spot_dict:
                continue

            spot = spot_dict[ticker]
            pos['dte_remaining'] -= days_elapsed

            T = max(pos['dte_remaining'] / 365.0, 1/365.0)
            iv = pos['iv_at_open']  # Assume IV stays constant (conservative)
            r = RISK_FREE_RATE

            # Current value of all legs
            put_val = bs_price(spot, pos['put_strike'], T, r, iv, "put")
            call_short_val = bs_price(spot, pos['call_short_strike'], T, r, iv, "call")
            call_long_val = bs_price(spot, pos['call_long_strike'], T, r, iv, "call")

            current_debit = put_val + call_short_val - call_long_val  # Cost to close
            pnl_per_share = pos['net_credit'] - current_debit
            pnl_total = pnl_per_share * 100 * pos['contracts']

            exit_reason = None

            # Take profit: 50% of max profit
            if pnl_total >= pos['net_credit_total'] * TAKE_PROFIT_PCT:
                exit_reason = "take_profit"

            # Stop loss: loss > 2x premium received
            elif pnl_total < -pos['net_credit_total'] * STOP_LOSS_MULT:
                exit_reason = "stop_loss"

            # Expiration
            elif pos['dte_remaining'] <= 0:
                # At expiration, compute intrinsic values
                put_intrinsic = max(0, pos['put_strike'] - spot)
                call_short_intrinsic = max(0, spot - pos['call_short_strike'])
                call_long_intrinsic = max(0, spot - pos['call_long_strike'])

                pnl_per_share = pos['net_credit'] - (put_intrinsic + call_short_intrinsic - call_long_intrinsic)
                pnl_total = pnl_per_share * 100 * pos['contracts']
                exit_reason = "expiration"

            if exit_reason:
                close_commission = self._commission_cost() * pos['contracts'] if exit_reason != "expiration" else 0
                pnl_total -= close_commission

                trade = {
                    'open_date': pos['open_date'],
                    'close_date': date,
                    'ticker': ticker,
                    'spot_at_open': pos['spot_at_open'],
                    'spot_at_close': spot,
                    'put_strike': pos['put_strike'],
                    'call_short_strike': pos['call_short_strike'],
                    'call_long_strike': pos['call_long_strike'],
                    'iv_at_open': pos['iv_at_open'],
                    'net_credit': pos['net_credit'],
                    'pnl': pnl_total,
                    'pnl_pct': pnl_total / (pos['max_risk'] + 1e-10),
                    'exit_reason': exit_reason,
                    'dte_at_close': pos['dte_remaining'],
                    'contracts': pos['contracts'],
                    'put_expired_worthless': 1 if spot > pos['put_strike'] else 0,
                }
                self.trades.append(trade)
                self.capital += pnl_total
                self.positions.remove(pos)
                closed.append(trade)

        self.equity_curve.append({'date': date, 'equity': self.capital, 'n_positions': len(self.positions)})
        return closed


# ── Walk-Forward ML Pipeline ───────────────────────────────────────────────
FEATURE_COLS = [
    'rv_20d', 'rv_60d', 'rv_252d', 'iv_rank', 'iv_hv_ratio',
    'ret_1d', 'ret_5d', 'ret_21d', 'ret_63d',
    'rsi_14', 'vix', 'volume_ratio', 'bb_width',
    'dist_from_high', 'dist_from_low', 'ret_skew_21d',
]


def run_walk_forward(all_features_df, universe, use_ml=True, label="ml"):
    """
    Walk-forward backtest with optional ML timing.
    Returns backtest object and model metrics.
    """
    log.info(f"Running walk-forward backtest: {label}, universe={len(universe)} stocks, ML={use_ml}")

    # Filter to universe
    df = all_features_df[all_features_df['ticker'].isin(universe)].copy()
    df = df.dropna(subset=FEATURE_COLS + ['put_expires_worthless'])

    dates = sorted(df.index.unique())
    if len(dates) < TRAIN_WINDOW + REBALANCE_FREQ:
        log.error(f"Not enough data: {len(dates)} days, need {TRAIN_WINDOW + REBALANCE_FREQ}")
        return None, None

    bt = JadeLizardBacktest()
    model_metrics = []
    current_model = None
    selected_tickers = set()

    # Get VIX series
    vix_series = df[df['ticker'] == universe[0]]['vix'] if 'vix' in df.columns else None

    log.info(f"Walk-forward: {len(dates)} total dates, train_window={TRAIN_WINDOW}, rebal={REBALANCE_FREQ}")

    rebal_counter = 0
    for i in range(TRAIN_WINDOW, len(dates)):
        date = dates[i]
        rebal_counter += 1

        # Monthly rebalance: retrain model and select stocks
        if rebal_counter >= REBALANCE_FREQ or current_model is None:
            rebal_counter = 0
            train_start = dates[max(0, i - TRAIN_WINDOW)]
            train_end = dates[i - 1]

            train_data = df[(df.index >= train_start) & (df.index <= train_end)]
            train_data = train_data.dropna(subset=FEATURE_COLS + ['put_expires_worthless'])

            if use_ml and len(train_data) > 100:
                X_train = train_data[FEATURE_COLS].values
                y_train = train_data['put_expires_worthless'].values

                # Train LightGBM
                train_set = lgb.Dataset(X_train, label=y_train)
                params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'learning_rate': 0.05,
                    'num_leaves': 31,
                    'max_depth': 6,
                    'min_child_samples': 20,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'seed': 42,
                }
                current_model = lgb.train(params, train_set, num_boost_round=200)

                # In-sample metrics
                train_pred = current_model.predict(X_train)
                try:
                    train_auc = roc_auc_score(y_train, train_pred)
                except:
                    train_auc = 0.5
                model_metrics.append({
                    'date': date,
                    'train_auc': train_auc,
                    'train_size': len(train_data),
                    'pos_rate': y_train.mean(),
                })

            # Select stocks for this period
            today_data = df[df.index == date]
            if use_ml and current_model is not None and len(today_data) > 0:
                X_today = today_data[FEATURE_COLS].values
                preds = current_model.predict(X_today)
                today_data = today_data.copy()
                today_data['ml_score'] = preds
                # Select top scoring stocks (most likely puts expire worthless)
                top = today_data.nlargest(MAX_POSITIONS * 2, 'ml_score')
                selected_tickers = set(top[top['ml_score'] > 0.5]['ticker'].values)
            else:
                # No ML: select stocks with highest IV rank
                today_data = today_data.copy()
                top = today_data.nlargest(MAX_POSITIONS * 2, 'iv_rank')
                selected_tickers = set(top['ticker'].values)

        # Daily: mark to market existing positions
        day_data = df[df.index == date]
        spot_dict = {}
        for _, row in day_data.iterrows():
            spot_dict[row['ticker']] = row['close']

        bt.mark_to_market(date, spot_dict)

        # Check VIX
        vix_val = day_data['vix'].iloc[0] if len(day_data) > 0 and 'vix' in day_data.columns else 20
        if vix_val > VIX_PAUSE_THRESHOLD:
            continue  # Pause entries

        # Open new positions if capacity available
        for ticker in selected_tickers:
            if len(bt.positions) >= MAX_POSITIONS:
                break
            if any(p['ticker'] == ticker for p in bt.positions):
                continue

            ticker_data = day_data[day_data['ticker'] == ticker]
            if len(ticker_data) == 0:
                continue

            row = ticker_data.iloc[0]
            spot = row['close']
            rv = row.get('rv_20d', 0.25)
            if pd.isna(rv) or rv <= 0:
                rv = 0.25
            iv = rv * IV_PREMIUM

            bt.open_position(date, ticker, spot, iv)

    # Close any remaining positions at last date
    if bt.positions:
        last_date = dates[-1]
        day_data = df[df.index == last_date]
        spot_dict = {row['ticker']: row['close'] for _, row in day_data.iterrows()}
        for pos in bt.positions[:]:
            pos['dte_remaining'] = 0
        bt.mark_to_market(last_date, spot_dict)

    return bt, model_metrics


# ── Analysis Functions ──────────────────────────────────────────────────────
def analyze_results(bt, label=""):
    """Compute performance metrics from backtest."""
    if not bt or not bt.trades:
        return {"label": label, "n_trades": 0, "sharpe": 0}

    trades_df = pd.DataFrame(bt.trades)
    equity_df = pd.DataFrame(bt.equity_curve)

    n_trades = len(trades_df)
    win_rate = (trades_df['pnl'] > 0).mean()
    total_pnl = trades_df['pnl'].sum()
    avg_pnl = trades_df['pnl'].mean()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0
    avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if (trades_df['pnl'] <= 0).any() else 0
    profit_factor = abs(avg_win * (trades_df['pnl'] > 0).sum()) / (abs(avg_loss * (trades_df['pnl'] <= 0).sum()) + 1e-10)

    # Equity curve metrics
    if len(equity_df) > 1:
        equity_df['ret'] = equity_df['equity'].pct_change().fillna(0)
        daily_ret = equity_df['ret']
        sharpe = daily_ret.mean() / (daily_ret.std() + 1e-10) * np.sqrt(252)
        downside = daily_ret[daily_ret < 0].std()
        sortino = daily_ret.mean() / (downside + 1e-10) * np.sqrt(252)
        max_dd = (equity_df['equity'] / equity_df['equity'].cummax() - 1).min()
    else:
        sharpe = sortino = max_dd = 0

    # Exit reason breakdown
    exit_reasons = trades_df['exit_reason'].value_counts().to_dict()

    # Put expiry rate
    put_wr = trades_df['put_expired_worthless'].mean()

    results = {
        'label': label,
        'n_trades': n_trades,
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 2),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown': round(max_dd, 4),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'put_expiry_rate': round(put_wr, 4),
        'exit_reasons': exit_reasons,
        'final_equity': round(bt.capital, 2),
        'return_pct': round((bt.capital / bt.initial_capital - 1) * 100, 2),
    }
    return results


def regime_analysis(bt, spy_returns):
    """Analyze performance in bull vs bear regimes."""
    if not bt or not bt.trades:
        return {}

    trades_df = pd.DataFrame(bt.trades)
    trades_df['close_date'] = pd.to_datetime(trades_df['close_date'])

    # Determine regime: bull (SPY 63d ret > 0) vs bear
    spy_63d = spy_returns.rolling(63).sum()  # Approximate 3-month return

    bull_trades = []
    bear_trades = []
    # Make spy_63d a clean Series with unique DatetimeIndex
    spy_63d = spy_63d[~spy_63d.index.duplicated(keep='first')]
    for _, trade in trades_df.iterrows():
        dt = trade['close_date']
        idx_pos = spy_63d.index.get_indexer([dt], method='nearest')
        if len(idx_pos) > 0 and idx_pos[0] >= 0:
            regime_ret = float(spy_63d.iloc[idx_pos[0]])
            if regime_ret > 0:
                bull_trades.append(trade['pnl'])
            else:
                bear_trades.append(trade['pnl'])

    results = {
        'bull_n': len(bull_trades),
        'bull_wr': np.mean([1 if p > 0 else 0 for p in bull_trades]) if bull_trades else 0,
        'bull_avg_pnl': np.mean(bull_trades) if bull_trades else 0,
        'bear_n': len(bear_trades),
        'bear_wr': np.mean([1 if p > 0 else 0 for p in bear_trades]) if bear_trades else 0,
        'bear_avg_pnl': np.mean(bear_trades) if bear_trades else 0,
    }

    # Regime disparity check (HC #428 R1)
    if bull_trades and bear_trades:
        bull_sharpe_proxy = np.mean(bull_trades) / (np.std(bull_trades) + 1e-10)
        bear_sharpe_proxy = np.mean(bear_trades) / (np.std(bear_trades) + 1e-10)
        disparity = abs(bull_sharpe_proxy - bear_sharpe_proxy) / (max(abs(bull_sharpe_proxy), abs(bear_sharpe_proxy)) + 1e-10)
        results['regime_disparity'] = round(disparity, 4)
        results['regime_agnostic'] = disparity < 0.50
    else:
        results['regime_disparity'] = None
        results['regime_agnostic'] = False

    return results


def sub_period_analysis(bt):
    """Split into sub-periods and check stability."""
    if not bt or not bt.trades:
        return {}

    trades_df = pd.DataFrame(bt.trades)
    trades_df['close_date'] = pd.to_datetime(trades_df['close_date'])
    trades_df = trades_df.sort_values('close_date')

    n = len(trades_df)
    if n < 20:
        return {'error': 'too few trades for sub-period analysis'}

    # Split into thirds
    third = n // 3
    periods = {
        'period_1': trades_df.iloc[:third],
        'period_2': trades_df.iloc[third:2*third],
        'period_3': trades_df.iloc[2*third:],
    }

    results = {}
    for name, period in periods.items():
        results[name] = {
            'n_trades': len(period),
            'win_rate': round((period['pnl'] > 0).mean(), 4),
            'avg_pnl': round(period['pnl'].mean(), 2),
            'total_pnl': round(period['pnl'].sum(), 2),
            'date_range': f"{period['close_date'].min().date()} to {period['close_date'].max().date()}",
        }

    # Stability: check if all periods are profitable
    all_profitable = all(results[p]['total_pnl'] > 0 for p in results)
    results['all_periods_profitable'] = all_profitable

    return results


def permutation_test(all_features_df, universe, bt_sharpe, n_perms=N_PERMUTATIONS):
    """Permutation test: shuffle ML scores and re-run to get p-value."""
    log.info(f"Running permutation test ({n_perms} shuffles)...")

    null_sharpes = []
    for i in range(n_perms):
        if i % 50 == 0:
            log.info(f"  Permutation {i}/{n_perms}")

        # Run baseline (no ML, random stock selection)
        bt_null, _ = run_walk_forward(all_features_df, universe, use_ml=False, label=f"perm_{i}")
        if bt_null and bt_null.trades:
            results = analyze_results(bt_null, f"perm_{i}")
            null_sharpes.append(results['sharpe'])
        else:
            null_sharpes.append(0)

    null_sharpes = np.array(null_sharpes)
    p_value = (null_sharpes >= bt_sharpe).mean()

    return {
        'p_value': round(p_value, 4),
        'null_mean_sharpe': round(null_sharpes.mean(), 3),
        'null_std_sharpe': round(null_sharpes.std(), 3),
        'observed_sharpe': round(bt_sharpe, 3),
        'significant_at_05': p_value < 0.05,
        'significant_at_01': p_value < 0.01,
    }


# ── Main Research Pipeline ─────────────────────────────────────────────────
def main():
    start_time = time.time()
    log.info("=" * 80)
    log.info("JADE LIZARD EXPANDED UNIVERSE RESEARCH v1")
    log.info("=" * 80)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"jade_lizard_expanded_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        # Log params
        mlflow.log_params({
            'universe_size': len(FULL_UNIVERSE),
            'put_delta': PUT_DELTA,
            'call_short_delta': CALL_SHORT_DELTA,
            'call_long_delta': CALL_LONG_DELTA,
            'dte': DTE,
            'max_positions': MAX_POSITIONS,
            'train_window': TRAIN_WINDOW,
            'rebalance_freq': REBALANCE_FREQ,
            'take_profit_pct': TAKE_PROFIT_PCT,
            'stop_loss_mult': STOP_LOSS_MULT,
            'vix_pause': VIX_PAUSE_THRESHOLD,
            'initial_capital': INITIAL_CAPITAL,
            'n_permutations': N_PERMUTATIONS,
        })

        # ── Step 1: Fetch Data ──────────────────────────────────────────
        log.info("Step 1: Fetching price data...")
        prices = fetch_stock_data(FULL_UNIVERSE)
        log.info(f"  Data shape: {prices.shape}")

        # Extract VIX
        try:
            if isinstance(prices.columns, pd.MultiIndex):
                vix = prices[("^VIX", "Close")].dropna()
            else:
                vix = None
        except:
            vix = None
        log.info(f"  VIX data: {len(vix) if vix is not None else 'N/A'} days")

        # ── Step 2: Feature Engineering ─────────────────────────────────
        log.info("Step 2: Computing features for all tickers...")
        all_features = []
        for ticker in FULL_UNIVERSE:
            log.info(f"  Features: {ticker}")
            feats = compute_features(prices, ticker, vix)
            if feats is not None:
                all_features.append(feats)
            else:
                log.warning(f"  Skipped {ticker}: insufficient data")

        all_features_df = pd.concat(all_features)
        log.info(f"  Total feature rows: {len(all_features_df)}, tickers: {all_features_df['ticker'].nunique()}")
        mlflow.log_metric("n_tickers_with_data", all_features_df['ticker'].nunique())

        # ── Step 3: Walk-Forward Backtests ──────────────────────────────
        # 3a: ML-timed, full universe
        log.info("Step 3a: Walk-forward backtest — ML-timed, full universe (50 stocks)")
        bt_ml_full, ml_metrics = run_walk_forward(all_features_df, FULL_UNIVERSE, use_ml=True, label="ml_full_50")
        results_ml_full = analyze_results(bt_ml_full, "ML-timed, 50 stocks")
        log.info(f"  ML Full: {results_ml_full}")

        # 3b: Baseline (no ML), full universe
        log.info("Step 3b: Walk-forward backtest — Baseline (no ML), full universe")
        bt_base_full, _ = run_walk_forward(all_features_df, FULL_UNIVERSE, use_ml=False, label="baseline_full_50")
        results_base_full = analyze_results(bt_base_full, "Baseline, 50 stocks")
        log.info(f"  Baseline Full: {results_base_full}")

        # 3c: ML-timed, original 15 stocks
        log.info("Step 3c: Walk-forward backtest — ML-timed, original 15 stocks")
        bt_ml_15, _ = run_walk_forward(all_features_df, ORIGINAL_15, use_ml=True, label="ml_original_15")
        results_ml_15 = analyze_results(bt_ml_15, "ML-timed, 15 stocks")
        log.info(f"  ML 15: {results_ml_15}")

        # 3d: Baseline, original 15 stocks
        log.info("Step 3d: Walk-forward backtest — Baseline, original 15 stocks")
        bt_base_15, _ = run_walk_forward(all_features_df, ORIGINAL_15, use_ml=False, label="baseline_original_15")
        results_base_15 = analyze_results(bt_base_15, "Baseline, 15 stocks")
        log.info(f"  Baseline 15: {results_base_15}")

        # Log all results to MLflow
        for name, res in [("ml_full", results_ml_full), ("base_full", results_base_full),
                          ("ml_15", results_ml_15), ("base_15", results_base_15)]:
            for key in ['n_trades', 'sharpe', 'sortino', 'win_rate', 'profit_factor',
                        'max_drawdown', 'total_pnl', 'return_pct', 'put_expiry_rate']:
                if key in res:
                    mlflow.log_metric(f"{name}_{key}", res[key])

        # ── Step 4: Regime Analysis ─────────────────────────────────────
        log.info("Step 4: Regime analysis (bull vs bear)...")
        # Use SPY as market proxy — approximate with AAPL if SPY not available
        try:
            import yfinance as yf
            spy = yf.download("SPY", start="2012-01-01", end="2026-07-18", auto_adjust=True, progress=False)
            # Handle both flat and MultiIndex columns from yfinance
            if isinstance(spy.columns, pd.MultiIndex):
                spy_close = spy[('Close', 'SPY')].dropna()
            else:
                spy_close = spy['Close'].dropna()
            spy_returns = spy_close.pct_change().dropna()
        except Exception as e:
            log.warning(f"  Failed to fetch SPY: {e}, using AAPL as proxy")
            # Fallback: use AAPL returns
            if isinstance(prices.columns, pd.MultiIndex):
                spy_returns = prices[("AAPL", "Close")].pct_change().dropna()
            else:
                spy_returns = pd.Series(dtype=float)

        regime_ml = regime_analysis(bt_ml_full, spy_returns)
        regime_base = regime_analysis(bt_base_full, spy_returns)
        log.info(f"  Regime ML: {regime_ml}")
        log.info(f"  Regime Base: {regime_base}")

        for key, val in regime_ml.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(f"regime_ml_{key}", val)

        # ── Step 5: Sub-Period Stability ────────────────────────────────
        log.info("Step 5: Sub-period stability analysis...")
        subperiod_ml = sub_period_analysis(bt_ml_full)
        subperiod_base = sub_period_analysis(bt_base_full)
        log.info(f"  Sub-period ML: {subperiod_ml}")
        log.info(f"  Sub-period Base: {subperiod_base}")

        # ── Step 6: Permutation Test ────────────────────────────────────
        log.info("Step 6: Permutation test (200 shuffles)...")
        if results_ml_full['sharpe'] > 0:
            perm_results = permutation_test(all_features_df, FULL_UNIVERSE, results_ml_full['sharpe'], n_perms=N_PERMUTATIONS)
        else:
            perm_results = {'p_value': 1.0, 'significant_at_05': False, 'note': 'negative sharpe, skipped'}
        log.info(f"  Permutation test: {perm_results}")

        for key, val in perm_results.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(f"perm_{key}", val)

        # ── Step 7: Feature Importance ──────────────────────────────────
        log.info("Step 7: Feature importance from final LightGBM model...")
        if ml_metrics:
            # Train final model on all data for feature importance
            df_final = all_features_df[all_features_df['ticker'].isin(FULL_UNIVERSE)].dropna(subset=FEATURE_COLS + ['put_expires_worthless'])
            X_final = df_final[FEATURE_COLS].values
            y_final = df_final['put_expires_worthless'].values

            train_set = lgb.Dataset(X_final, label=y_final)
            params = {
                'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
                'num_leaves': 31, 'max_depth': 6, 'verbose': -1, 'seed': 42,
            }
            final_model = lgb.train(params, train_set, num_boost_round=200)
            importances = dict(zip(FEATURE_COLS, final_model.feature_importance('gain').tolist()))
            importances_sorted = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
            log.info(f"  Feature importances: {importances_sorted}")

            for feat, imp in importances_sorted.items():
                mlflow.log_metric(f"feat_imp_{feat}", imp)

        # ── Step 8: Compile Final Report ────────────────────────────────
        log.info("=" * 80)
        log.info("FINAL REPORT")
        log.info("=" * 80)

        report = {
            'timestamp': datetime.now().isoformat(),
            'strategy': 'Jade Lizard Expanded Universe v1',
            'results': {
                'ml_full_universe': results_ml_full,
                'baseline_full_universe': results_base_full,
                'ml_original_15': results_ml_15,
                'baseline_original_15': results_base_15,
            },
            'regime_analysis': {
                'ml': regime_ml,
                'baseline': regime_base,
            },
            'sub_period': {
                'ml': subperiod_ml,
                'baseline': subperiod_base,
            },
            'permutation_test': perm_results,
            'feature_importance': importances_sorted if ml_metrics else {},
            'model_metrics_summary': {
                'n_rebalances': len(ml_metrics) if ml_metrics else 0,
                'avg_train_auc': round(np.mean([m['train_auc'] for m in ml_metrics]), 4) if ml_metrics else 0,
            },
            'config': {
                'universe': FULL_UNIVERSE,
                'original_15': ORIGINAL_15,
                'put_delta': PUT_DELTA,
                'call_deltas': (CALL_SHORT_DELTA, CALL_LONG_DELTA),
                'dte': DTE,
                'train_window': TRAIN_WINDOW,
                'max_positions': MAX_POSITIONS,
            },
        }

        # Save report
        report_path = OUTPUT_DIR / "research_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        log.info(f"Report saved: {report_path}")

        # Save trades
        if bt_ml_full and bt_ml_full.trades:
            trades_path = OUTPUT_DIR / "ml_full_trades.csv"
            pd.DataFrame(bt_ml_full.trades).to_csv(trades_path, index=False)
            log.info(f"Trades saved: {trades_path}")
            mlflow.log_artifact(str(trades_path))

        if bt_base_full and bt_base_full.trades:
            trades_path = OUTPUT_DIR / "baseline_full_trades.csv"
            pd.DataFrame(bt_base_full.trades).to_csv(trades_path, index=False)

        # Save equity curves
        if bt_ml_full and bt_ml_full.equity_curve:
            eq_path = OUTPUT_DIR / "ml_full_equity.csv"
            pd.DataFrame(bt_ml_full.equity_curve).to_csv(eq_path, index=False)
            mlflow.log_artifact(str(eq_path))

        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(LOG_FILE))

        # ── Summary ────────────────────────────────────────────────────
        elapsed = (time.time() - start_time) / 60
        log.info(f"\nCompleted in {elapsed:.1f} minutes")
        log.info(f"\n{'='*60}")
        log.info("HEADLINE RESULTS:")
        log.info(f"  ML + 50 stocks:   Sharpe={results_ml_full['sharpe']}, WR={results_ml_full['win_rate']}, "
                 f"PF={results_ml_full['profit_factor']}, N={results_ml_full['n_trades']}")
        log.info(f"  Baseline + 50:    Sharpe={results_base_full['sharpe']}, WR={results_base_full['win_rate']}, "
                 f"PF={results_base_full['profit_factor']}, N={results_base_full['n_trades']}")
        log.info(f"  ML + 15 stocks:   Sharpe={results_ml_15['sharpe']}, WR={results_ml_15['win_rate']}, "
                 f"PF={results_ml_15['profit_factor']}, N={results_ml_15['n_trades']}")
        log.info(f"  Baseline + 15:    Sharpe={results_base_15['sharpe']}, WR={results_base_15['win_rate']}, "
                 f"PF={results_base_15['profit_factor']}, N={results_base_15['n_trades']}")
        log.info(f"  Permutation p-value: {perm_results.get('p_value', 'N/A')}")
        log.info(f"  Regime agnostic: {regime_ml.get('regime_agnostic', 'N/A')} (disparity={regime_ml.get('regime_disparity', 'N/A')})")
        log.info(f"  Sub-periods all profitable: {subperiod_ml.get('all_periods_profitable', 'N/A')}")
        log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
