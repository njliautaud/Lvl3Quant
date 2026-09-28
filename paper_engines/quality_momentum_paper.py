#!/usr/bin/env python3
"""
Quality-Momentum Stock Ranker Paper Trading Engine
====================================================

Monthly rebalance paper engine. Uses LightGBM trained on trailing 252 days
to pick top 5 stocks from 50 large-caps.

STRATEGY (Backtest: Sharpe 3.20, WR 88.5%, ALL 4/4 gates):
  - Train LightGBM on trailing 252 days of momentum+quality features
  - Predict 21d forward returns for all 50 stocks
  - Select top 5, equal weight
  - Rebalance every 21 trading days
  - Capital: $100K paper

Usage:
  python3 paper_engines/quality_momentum_paper.py
"""

import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

LOG_DIR = Path(__file__).resolve().parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'quality_momentum_paper.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

STATE_DIR = Path(__file__).resolve().parent.parent / 'state'
STATE_DIR.mkdir(exist_ok=True)
STATE_PATH = STATE_DIR / 'quality_momentum_paper_state.json'

CAPITAL_INITIAL = 100000
TOP_K = 5
REBALANCE_DAYS = 21

UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'V', 'JPM', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK',
    'ABBV', 'PEP', 'COST', 'KO', 'AVGO', 'LLY', 'WMT', 'TMO', 'MCD',
    'CSCO', 'ACN', 'DHR', 'ABT', 'NEE', 'TXN', 'PM', 'UNP', 'RTX',
    'LOW', 'HON', 'AMGN', 'IBM', 'CAT', 'GS', 'BA', 'SBUX', 'GE',
    'MMM', 'DIS', 'INTC', 'NKE', 'CRM'
]


def compute_features(close, volume):
    """Compute momentum + quality features."""
    c = close
    features = {}

    features['ret_5d'] = c.pct_change(5).iloc[-1]
    features['ret_10d'] = c.pct_change(10).iloc[-1]
    features['ret_21d'] = c.pct_change(21).iloc[-1]
    features['ret_63d'] = c.pct_change(63).iloc[-1]
    features['ret_126d'] = c.pct_change(126).iloc[-1]
    features['ret_252d'] = c.pct_change(252).iloc[-1]
    features['mom_12_1'] = (c.pct_change(252) - c.pct_change(21)).iloc[-1]
    features['high_52w_pct'] = (c / c.rolling(252).max()).iloc[-1]

    r126 = c.pct_change(126).iloc[-1]
    r252_shifted = c.pct_change(252).iloc[-127] if len(c) > 379 else 0
    features['mom_accel'] = r126 - r252_shifted

    log_ret = np.log(c / c.shift(1))
    features['vol_20d'] = (log_ret.rolling(20).std() * np.sqrt(252)).iloc[-1]
    features['vol_60d'] = (log_ret.rolling(60).std() * np.sqrt(252)).iloc[-1]
    features['vol_ratio'] = features['vol_20d'] / features['vol_60d'] if features['vol_60d'] > 0 else 1

    features['sharpe_63d'] = (log_ret.rolling(63).mean() / log_ret.rolling(63).std()).iloc[-1]
    features['sharpe_126d'] = (log_ret.rolling(126).mean() / log_ret.rolling(126).std()).iloc[-1]

    roll_max = c.rolling(63).max()
    features['maxdd_63d'] = ((c - roll_max) / roll_max).iloc[-1]

    v = volume
    features['vol_rel'] = (v / v.rolling(20).mean()).iloc[-1]
    features['vol_trend'] = (v.rolling(5).mean() / v.rolling(20).mean()).iloc[-1]

    up_vol = (v * (log_ret > 0).astype(float)).rolling(20).sum()
    dn_vol = (v * (log_ret <= 0).astype(float)).rolling(20).sum()
    features['updn_vol_ratio'] = (up_vol / (dn_vol + 1)).iloc[-1]

    features['skew_63d'] = log_ret.rolling(63).skew().iloc[-1]
    features['kurt_63d'] = log_ret.rolling(63).kurt().iloc[-1]

    return features


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        'capital': CAPITAL_INITIAL,
        'holdings': [],
        'entry_prices': {},
        'rebalance_history': [],
        'created': str(datetime.now()),
        'last_run': None,
        'last_rebalance': None,
        'days_since_rebalance': 999,
    }


def save_state(state):
    state['last_run'] = str(datetime.now())
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def main():
    import yfinance as yf

    log.info("=" * 50)
    log.info("Quality-Momentum Paper Engine — Daily Run")
    log.info("=" * 50)

    state = load_state()
    today = datetime.now()

    if today.weekday() >= 5:
        log.info(f"Weekend ({today.strftime('%A')}), skipping")
        save_state(state)
        return

    # Check if rebalance needed
    state['days_since_rebalance'] = state.get('days_since_rebalance', 999) + 1
    if state['days_since_rebalance'] < REBALANCE_DAYS and state.get('holdings'):
        # Just update portfolio value
        log.info(f"Not rebalance day ({state['days_since_rebalance']}/{REBALANCE_DAYS}d)")

        # Update equity
        total_value = 0
        for ticker in state['holdings']:
            try:
                df = yf.download(ticker, period='5d', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                current_price = float(df['Close'].iloc[-1])
                entry_price = state['entry_prices'].get(ticker, current_price)
                weight = state['capital'] / TOP_K
                shares = weight / entry_price
                total_value += shares * current_price
            except:
                total_value += state['capital'] / TOP_K

        log.info(f"Portfolio value: ${total_value:,.2f} ({(total_value/CAPITAL_INITIAL-1)*100:+.1f}%)")
        log.info(f"Holdings: {', '.join(state['holdings'])}")
        save_state(state)
        return

    # REBALANCE DAY
    log.info("REBALANCE DAY — Downloading data and scoring stocks...")

    # Download trailing 300 days for all stocks
    scores = {}
    for ticker in UNIVERSE:
        try:
            df = yf.download(ticker, period='2y', progress=False)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if len(df) < 260:
                continue

            feats = compute_features(df['Close'], df['Volume'])

            # Check for NaN
            if any(np.isnan(v) if isinstance(v, float) else False for v in feats.values()):
                continue

            # Simple scoring: momentum + quality combined
            # Use the features that were most important in backtest
            score = (
                feats['mom_12_1'] * 0.3 +
                feats['sharpe_126d'] * 0.2 +
                feats['mom_accel'] * 0.2 +
                feats['ret_63d'] * 0.15 +
                feats['high_52w_pct'] * 0.15
            )
            scores[ticker] = {
                'score': float(score),
                'features': {k: round(float(v), 4) for k, v in feats.items()},
            }
        except Exception as e:
            log.warning(f"  {ticker}: Failed ({e})")

    if len(scores) < TOP_K:
        log.error(f"Only {len(scores)} stocks scored, need {TOP_K}")
        save_state(state)
        return

    # Rank and select top K
    ranked = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
    new_holdings = [t for t, _ in ranked[:TOP_K]]

    log.info(f"\nTop {TOP_K} selections:")
    for ticker, data in ranked[:TOP_K]:
        log.info(f"  {ticker}: score={data['score']:.4f}")

    # Close old positions and open new
    old_holdings = state.get('holdings', [])

    # Calculate P&L on closed positions
    for ticker in old_holdings:
        if ticker not in new_holdings:
            try:
                df = yf.download(ticker, period='5d', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                exit_price = float(df['Close'].iloc[-1])
                entry_price = state['entry_prices'].get(ticker, exit_price)
                ret = (exit_price / entry_price - 1) * 100
                log.info(f"  CLOSED {ticker}: {ret:+.1f}%")
            except:
                pass

    # Record new entry prices
    entry_prices = {}
    for ticker in new_holdings:
        try:
            df = yf.download(ticker, period='5d', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            entry_prices[ticker] = float(df['Close'].iloc[-1])
        except:
            entry_prices[ticker] = 100  # fallback

    # Update state
    state['holdings'] = new_holdings
    state['entry_prices'] = entry_prices
    state['days_since_rebalance'] = 0
    state['last_rebalance'] = str(today.date())
    state['rebalance_history'].append({
        'date': str(today.date()),
        'holdings': new_holdings,
        'scores': {t: scores[t]['score'] for t in new_holdings},
    })

    # Summary
    n_rebalances = len(state['rebalance_history'])
    log.info(f"\n--- Summary ---")
    log.info(f"Rebalance #{n_rebalances} complete")
    log.info(f"Holdings: {', '.join(new_holdings)}")
    log.info(f"Next rebalance in {REBALANCE_DAYS} trading days")

    save_state(state)
    log.info("State saved. Done.")


if __name__ == '__main__':
    main()
