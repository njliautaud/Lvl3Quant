#!/usr/bin/env python3
"""
Sector Equity Rotation Paper Trading Engine
=============================================

Monthly rebalance paper engine. Uses LightGBM trained on sliding 500-day
window to rank 11 sector ETFs, buys top-2 equal weight.

STRATEGY (Backtest KB #285: Sharpe 1.40, Sortino 2.25, WR 64%, MDD -12%, CAGR 24.1%):
  - Train LGBM on sliding 500-day window of sector features
  - Target: forward 21-day return (ret_fwd_21d)
  - Predict rankings for all 11 sector ETFs
  - Buy top-2 ranked, equal weight
  - Rebalance every ~21 trading days (monthly)
  - 15% trailing stop-loss per position
  - Capital: $645 paper (Robinhood, zero commission)

Usage:
  python3 paper_engines/equity_rotation_paper.py
"""

import json
import logging
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)
STATE_DIR = BASE_DIR / 'state'
STATE_DIR.mkdir(exist_ok=True)

STATE_PATH = STATE_DIR / 'equity_rotation_paper_state.json'
LOG_PATH = LOG_DIR / 'equity_rotation_paper.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# --- Strategy Constants ---
CAPITAL_INITIAL = 645.0
TOP_K = 2
REBALANCE_DAYS = 21
TRAILING_STOP_PCT = 0.15  # 15% trailing stop
TRAIN_WINDOW = 500        # sliding window for LGBM training
MIN_HISTORY = 300         # minimum bars needed for features

# 11 sector ETFs
UNIVERSE = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
BENCHMARK = 'SPY'

# LGBM features used for ranking
FEATURE_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d',
    'vol_21d', 'vol_ratio',
    'rsi_14',
    'macd', 'macd_signal',
    'bb_pct',
    'obv_slope',
    'atr_pct',
    'sector_rel_strength',
    'skew_21d', 'kurt_21d', 'max_dd_21d', 'up_down_vol_ratio',
]


def compute_features_series(close: pd.Series, volume: pd.Series, spy_close: pd.Series) -> pd.DataFrame:
    """Compute all features for an entire time series. Returns DataFrame aligned to close index."""
    c = close.copy()
    v = volume.copy().astype(float)
    lr = np.log(c / c.shift(1))

    feats = pd.DataFrame(index=c.index)

    # Momentum returns
    feats['ret_5d'] = c.pct_change(5)
    feats['ret_10d'] = c.pct_change(10)
    feats['ret_21d'] = c.pct_change(21)
    feats['ret_63d'] = c.pct_change(63)

    # Volatility
    feats['vol_21d'] = lr.rolling(21).std() * np.sqrt(252)
    vol_63d = lr.rolling(63).std() * np.sqrt(252)
    feats['vol_ratio'] = feats['vol_21d'] / vol_63d.replace(0, np.nan)

    # RSI 14
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feats['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    feats['macd'] = ema12 - ema26
    feats['macd_signal'] = feats['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Band %
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    feats['bb_pct'] = (c - (sma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)

    # OBV slope
    obv = (np.sign(lr) * v).cumsum()
    feats['obv_slope'] = obv.rolling(21).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 21 else np.nan,
        raw=True
    )

    # ATR %
    high_approx = c * (1 + lr.abs() * 0.5)  # approximate high/low from close
    low_approx = c * (1 - lr.abs() * 0.5)
    tr = pd.concat([
        high_approx - low_approx,
        (high_approx - c.shift(1)).abs(),
        (low_approx - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    feats['atr_pct'] = atr14 / c

    # Sector relative strength vs SPY
    spy_ret_21 = spy_close.pct_change(21).reindex(c.index)
    feats['sector_rel_strength'] = feats['ret_21d'] - spy_ret_21

    # Higher-order stats
    feats['skew_21d'] = lr.rolling(21).skew()
    feats['kurt_21d'] = lr.rolling(21).kurt()

    # Max drawdown 21d
    roll_max_21 = c.rolling(21).max()
    feats['max_dd_21d'] = (c - roll_max_21) / roll_max_21

    # Up/down volume ratio
    up_vol = (v * (lr > 0).astype(float)).rolling(21).sum()
    dn_vol = (v * (lr <= 0).astype(float)).rolling(21).sum()
    feats['up_down_vol_ratio'] = up_vol / dn_vol.replace(0, np.nan)

    return feats


def compute_features_latest(close: pd.Series, volume: pd.Series, spy_close: pd.Series) -> dict:
    """Compute features for the latest bar only. Returns dict."""
    feats_df = compute_features_series(close, volume, spy_close)
    latest = feats_df.iloc[-1]
    return {col: float(latest[col]) if not np.isnan(latest[col]) else np.nan for col in FEATURE_COLS}


def build_training_data(all_data: dict, spy_close: pd.Series, window: int = TRAIN_WINDOW) -> tuple:
    """Build training matrix from all ETF history using sliding window."""
    rows = []
    for ticker, df in all_data.items():
        close = df['Close']
        volume = df['Volume']
        feats_df = compute_features_series(close, volume, spy_close)

        # Target: forward 21-day return
        fwd_ret = close.pct_change(21).shift(-21)
        feats_df['target'] = fwd_ret

        # Use only the training window (skip last 21 rows — no target)
        valid = feats_df.dropna(subset=FEATURE_COLS + ['target'])
        if len(valid) < 50:
            continue

        # Take last `window` rows
        valid = valid.tail(window)
        for _, row in valid.iterrows():
            r = {col: row[col] for col in FEATURE_COLS}
            r['target'] = row['target']
            r['ticker'] = ticker
            rows.append(r)

    if not rows:
        return None, None, None

    df_train = pd.DataFrame(rows)
    X = df_train[FEATURE_COLS].values
    y = df_train['target'].values
    return X, y, df_train


def train_lgbm(X, y):
    """Train LightGBM ranking model on features/targets."""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
    }

    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dtrain],
        callbacks=[lgb.log_evaluation(period=0)],  # suppress output
    )
    return model


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load state: {e}, starting fresh")
    return {
        'capital': CAPITAL_INITIAL,
        'cash': CAPITAL_INITIAL,
        'positions': [],          # list of {ticker, shares, entry_price, entry_date, peak_price}
        'equity_curve': [],       # list of {date, equity}
        'trade_history': [],      # list of {date, action, ticker, shares, price, pnl}
        'rebalance_history': [],  # list of {date, holdings, scores}
        'created': str(datetime.now()),
        'last_run': None,
        'last_rebalance_date': None,
        'days_since_rebalance': 999,
        'total_trades': 0,
        'winning_trades': 0,
    }


def save_state(state: dict):
    state['last_run'] = str(datetime.now())
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def get_current_price(ticker: str) -> float:
    """Get current price for a ticker."""
    import yfinance as yf
    df = yf.download(ticker, period='5d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if len(df) == 0:
        raise ValueError(f"No data for {ticker}")
    return float(df['Close'].iloc[-1])


def sell_position(state: dict, pos: dict, price: float, reason: str = 'rebalance'):
    """Sell a position and update state."""
    pnl = (price - pos['entry_price']) * pos['shares']
    pnl_pct = (price / pos['entry_price'] - 1) * 100
    proceeds = price * pos['shares']
    state['cash'] += proceeds

    state['trade_history'].append({
        'date': str(datetime.now().date()),
        'action': 'SELL',
        'ticker': pos['ticker'],
        'shares': round(pos['shares'], 4),
        'price': round(price, 2),
        'pnl': round(pnl, 2),
        'pnl_pct': round(pnl_pct, 2),
        'reason': reason,
    })

    state['total_trades'] += 1
    if pnl > 0:
        state['winning_trades'] += 1

    log.info(f"  SELL {pos['shares']:.4f} {pos['ticker']} @ ${price:.2f} | "
             f"P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%) | Reason: {reason}")


def buy_position(state: dict, ticker: str, price: float, amount: float):
    """Buy a position and update state."""
    shares = amount / price
    state['cash'] -= amount

    pos = {
        'ticker': ticker,
        'shares': round(shares, 4),
        'entry_price': round(price, 2),
        'entry_date': str(datetime.now().date()),
        'peak_price': round(price, 2),
    }
    state['positions'].append(pos)

    state['trade_history'].append({
        'date': str(datetime.now().date()),
        'action': 'BUY',
        'ticker': ticker,
        'shares': round(shares, 4),
        'price': round(price, 2),
        'amount': round(amount, 2),
    })

    log.info(f"  BUY {shares:.4f} {ticker} @ ${price:.2f} (${amount:.2f})")


def calculate_equity(state: dict) -> float:
    """Calculate total portfolio equity (cash + positions mark-to-market)."""
    equity = state['cash']
    for pos in state['positions']:
        try:
            price = get_current_price(pos['ticker'])
            equity += price * pos['shares']
        except Exception:
            equity += pos['entry_price'] * pos['shares']  # fallback to entry
    return equity


def check_trailing_stops(state: dict) -> list:
    """Check trailing stops, return list of positions to sell."""
    to_sell = []
    for pos in state['positions']:
        try:
            current_price = get_current_price(pos['ticker'])
            # Update peak
            if current_price > pos.get('peak_price', pos['entry_price']):
                pos['peak_price'] = round(current_price, 2)

            peak = pos.get('peak_price', pos['entry_price'])
            drawdown = (peak - current_price) / peak

            if drawdown >= TRAILING_STOP_PCT:
                to_sell.append((pos, current_price))
                log.warning(f"  TRAILING STOP triggered for {pos['ticker']}: "
                            f"peak=${peak:.2f}, current=${current_price:.2f}, "
                            f"drawdown={drawdown*100:.1f}%")
        except Exception as e:
            log.warning(f"  Failed to check stop for {pos['ticker']}: {e}")

    return to_sell


def main():
    import yfinance as yf

    log.info("=" * 60)
    log.info("Sector Equity Rotation Paper Engine — Daily Run")
    log.info("=" * 60)

    if not HAS_LGBM:
        log.error("LightGBM not installed. Cannot run.")
        return

    state = load_state()
    today = datetime.now()

    if today.weekday() >= 5:
        log.info(f"Weekend ({today.strftime('%A')}), skipping")
        return

    # --- Check trailing stops first ---
    if state['positions']:
        log.info("Checking trailing stops...")
        stops_hit = check_trailing_stops(state)
        for pos, price in stops_hit:
            sell_position(state, pos, price, reason='trailing_stop')
            state['positions'].remove(pos)
        if stops_hit:
            log.info(f"  {len(stops_hit)} position(s) stopped out")

    # --- Determine if rebalance needed ---
    state['days_since_rebalance'] = state.get('days_since_rebalance', 999) + 1
    needs_rebalance = (
        state['days_since_rebalance'] >= REBALANCE_DAYS
        or not state['positions']  # no positions (first run or all stopped out)
    )

    if not needs_rebalance:
        # Non-rebalance day: just mark to market
        equity = calculate_equity(state)
        state['equity_curve'].append({
            'date': str(today.date()),
            'equity': round(equity, 2),
        })

        pnl_pct = (equity / CAPITAL_INITIAL - 1) * 100
        holdings_str = ', '.join(p['ticker'] for p in state['positions'])
        log.info(f"Day {state['days_since_rebalance']}/{REBALANCE_DAYS} — "
                 f"Equity: ${equity:.2f} ({pnl_pct:+.1f}%) | Holdings: {holdings_str}")
        save_state(state)
        return

    # === REBALANCE DAY ===
    log.info("=" * 40)
    log.info("REBALANCE DAY — Training model and ranking sectors")
    log.info("=" * 40)

    # Download 3 years of data for all ETFs + SPY
    tickers_to_fetch = UNIVERSE + [BENCHMARK]
    log.info(f"Downloading data for {len(tickers_to_fetch)} tickers...")

    all_data = {}
    for ticker in tickers_to_fetch:
        try:
            df = yf.download(ticker, period='3y', progress=False)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) >= MIN_HISTORY:
                all_data[ticker] = df
                log.info(f"  {ticker}: {len(df)} bars")
            else:
                log.warning(f"  {ticker}: only {len(df)} bars, skipping")
        except Exception as e:
            log.warning(f"  {ticker}: download failed ({e})")

    if BENCHMARK not in all_data:
        log.error("SPY data not available, cannot compute relative strength")
        save_state(state)
        return

    spy_close = all_data[BENCHMARK]['Close']

    # Filter to only sector ETFs for training/prediction
    sector_data = {t: all_data[t] for t in UNIVERSE if t in all_data}
    if len(sector_data) < TOP_K + 1:
        log.error(f"Only {len(sector_data)} ETFs available, need at least {TOP_K + 1}")
        save_state(state)
        return

    # --- Train LGBM model ---
    log.info("Building training data...")
    X_train, y_train, df_train = build_training_data(sector_data, spy_close, window=TRAIN_WINDOW)

    if X_train is None or len(X_train) < 100:
        log.error(f"Insufficient training data: {len(X_train) if X_train is not None else 0} rows")
        save_state(state)
        return

    log.info(f"Training LGBM on {len(X_train)} samples, {len(FEATURE_COLS)} features...")
    model = train_lgbm(X_train, y_train)

    # --- Predict current rankings ---
    log.info("Predicting sector rankings...")
    predictions = {}
    for ticker in sector_data:
        try:
            feats = compute_features_latest(
                sector_data[ticker]['Close'],
                sector_data[ticker]['Volume'],
                spy_close
            )
            # Check for NaN
            feat_vals = [feats.get(col, np.nan) for col in FEATURE_COLS]
            if any(np.isnan(v) for v in feat_vals):
                nan_cols = [col for col, v in zip(FEATURE_COLS, feat_vals) if np.isnan(v)]
                log.warning(f"  {ticker}: NaN features {nan_cols}, skipping")
                continue

            X_pred = np.array(feat_vals).reshape(1, -1)
            pred = model.predict(X_pred)[0]
            predictions[ticker] = {
                'predicted_ret': float(pred),
                'features': {k: round(v, 6) for k, v in feats.items()},
            }
        except Exception as e:
            log.warning(f"  {ticker}: prediction failed ({e})")

    if len(predictions) < TOP_K:
        log.error(f"Only {len(predictions)} predictions, need {TOP_K}")
        save_state(state)
        return

    # Rank by predicted return
    ranked = sorted(predictions.items(), key=lambda x: x[1]['predicted_ret'], reverse=True)

    log.info("\nSector Rankings:")
    for i, (ticker, data) in enumerate(ranked):
        marker = " <-- SELECTED" if i < TOP_K else ""
        log.info(f"  #{i+1} {ticker}: predicted_ret={data['predicted_ret']:.4f}{marker}")

    new_holdings = [t for t, _ in ranked[:TOP_K]]

    # --- Close existing positions ---
    for pos in list(state['positions']):
        try:
            price = get_current_price(pos['ticker'])
        except Exception:
            price = pos['entry_price']
        sell_position(state, pos, price, reason='rebalance')
    state['positions'] = []

    # --- Buy new positions (equal weight) ---
    equity = state['cash']  # after selling everything, equity = cash
    per_position = equity / TOP_K

    log.info(f"\nAllocating ${equity:.2f} across {TOP_K} positions (${per_position:.2f} each)")

    for ticker in new_holdings:
        try:
            price = get_current_price(ticker)
            buy_position(state, ticker, price, per_position)
        except Exception as e:
            log.error(f"  Failed to buy {ticker}: {e}")

    # --- Update state ---
    state['days_since_rebalance'] = 0
    state['last_rebalance_date'] = str(today.date())

    equity = calculate_equity(state)
    state['equity_curve'].append({
        'date': str(today.date()),
        'equity': round(equity, 2),
    })

    state['rebalance_history'].append({
        'date': str(today.date()),
        'holdings': new_holdings,
        'predictions': {t: predictions[t]['predicted_ret'] for t in new_holdings},
        'all_rankings': [(t, d['predicted_ret']) for t, d in ranked],
    })

    # --- Summary ---
    n_rebal = len(state['rebalance_history'])
    pnl_pct = (equity / CAPITAL_INITIAL - 1) * 100
    wr = (state['winning_trades'] / state['total_trades'] * 100) if state['total_trades'] > 0 else 0

    log.info(f"\n{'=' * 40}")
    log.info(f"REBALANCE #{n_rebal} COMPLETE")
    log.info(f"  Holdings: {', '.join(new_holdings)}")
    log.info(f"  Equity: ${equity:.2f} ({pnl_pct:+.1f}% total)")
    log.info(f"  Cash: ${state['cash']:.2f}")
    log.info(f"  Trades: {state['total_trades']} (WR: {wr:.0f}%)")
    log.info(f"  Next rebalance in ~{REBALANCE_DAYS} trading days")
    log.info(f"{'=' * 40}")

    save_state(state)
    log.info("State saved. Done.")


if __name__ == '__main__':
    main()
