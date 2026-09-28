#!/usr/bin/env python3
"""
Momentum Growth Rotation Paper Engine
=======================================
Based on Strategy E (RS vol-weighted) from momentum_growth_backtest.py.
Best risk-adjusted of the momentum strategies.

Strategy:
  - Universe: S&P 500 components (fetched dynamically)
  - Score = 6-month return / 20-day volatility (relative strength, vol-adjusted)
  - Select top 5 stocks by RS score
  - Weight by inverse 63-day realized vol (risk parity within winners)
  - Rebalance monthly (~21 trading days)
  - Always fully invested, $10K capital

Usage:
  python3 paper_engines/momentum_growth_paper.py
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

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)
STATE_DIR = BASE_DIR / 'state'
STATE_DIR.mkdir(exist_ok=True)

STATE_PATH = STATE_DIR / 'momentum_growth_paper_state.json'
LOG_PATH = LOG_DIR / 'momentum_growth_paper.log'

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
CAPITAL_INITIAL = 10000.0
TOP_K = 5                 # Hold top 5 momentum stocks
REBALANCE_DAYS = 21       # Monthly rebalance
MOM_LOOKBACK = 126        # 6-month momentum (~126 trading days)
VOL_LOOKBACK_SCORE = 20   # 20-day vol for scoring
VOL_LOOKBACK_WEIGHT = 63  # 63-day vol for weighting
RS_WINDOW = 126           # Relative strength vs SPY
MIN_PRICE = 5.0           # Minimum stock price (avoid penny stocks)
MIN_VOLUME_AVG = 500000   # Minimum 20-day avg volume

BENCHMARK = 'SPY'

# S&P 500 representative sample — top ~100 liquid names covering all sectors
# (Full S&P 500 download is too slow for daily cron; use liquid subset)
UNIVERSE = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AVGO', 'AMD', 'TSM',
    'CRM', 'ORCL', 'ADBE', 'NOW', 'PLTR', 'NFLX', 'INTC', 'CSCO', 'TXN',
    'QCOM', 'AMAT', 'MU', 'PANW', 'CRWD', 'SNOW', 'SHOP',
    # Financials
    'JPM', 'V', 'MA', 'BAC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'C',
    'WFC', 'USB', 'PNC', 'TFC', 'CME',
    # Healthcare
    'LLY', 'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'AMGN',
    'ISRG', 'MDT', 'BMY', 'GILD', 'REGN', 'VRTX',
    # Consumer
    'COST', 'WMT', 'HD', 'MCD', 'SBUX', 'NKE', 'TGT', 'LOW', 'TJX',
    'UBER', 'ABNB', 'BKNG', 'LULU', 'DG', 'ROST',
    # Industrials
    'CAT', 'DE', 'RTX', 'HON', 'GE', 'BA', 'LMT', 'UNP', 'UPS', 'FDX',
    # Energy
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX',
    # Materials
    'LIN', 'APD', 'SHW', 'FCX', 'NUE', 'NEM',
    # Utilities & REITs
    'NEE', 'DUK', 'SO', 'D', 'AEP',
    'PLD', 'AMT', 'CCI', 'SPG', 'O',
    # Communication
    'DIS', 'CMCSA', 'T', 'VZ', 'TMUS',
    # Other
    'BRK-B', 'COIN', 'SQ', 'PYPL', 'MELI',
]


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
        'positions': [],           # list of {ticker, shares, entry_price, entry_date, weight}
        'equity_curve': [],        # list of {date, equity}
        'trade_history': [],       # list of {date, action, ticker, shares, price, pnl}
        'rebalance_history': [],   # list of {date, holdings, scores}
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


def download_universe_data(tickers, period='8mo'):
    """Download close + volume data for the universe."""
    import yfinance as yf
    log.info(f"Downloading data for {len(tickers)} tickers...")

    data = {}
    # Download in batches to avoid rate limits
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        for ticker in batch:
            try:
                df = yf.download(ticker, period=period, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if len(df) >= MOM_LOOKBACK:
                    data[ticker] = df
            except Exception:
                pass

    log.info(f"  Got data for {len(data)}/{len(tickers)} tickers")
    return data


def compute_rs_vol_scores(data: dict, spy_data: pd.DataFrame) -> dict:
    """
    Compute RS vol-weighted scores.
    Score = 6-month return / 20-day volatility (higher = better risk-adjusted momentum)
    Returns dict of {ticker: {score, weight, mom_6m, vol_20d, vol_63d, rs_vs_spy}}
    """
    spy_close = spy_data['Close']
    spy_ret_6m = float((spy_close.iloc[-1] - spy_close.iloc[-MOM_LOOKBACK]) / spy_close.iloc[-MOM_LOOKBACK])

    results = {}
    for ticker, df in data.items():
        close = df['Close']
        volume = df['Volume']

        if len(close) < MOM_LOOKBACK + 5:
            continue

        current_price = float(close.iloc[-1])

        # Filter: minimum price and volume
        if current_price < MIN_PRICE:
            continue
        avg_vol = float(volume.iloc[-20:].mean())
        if avg_vol < MIN_VOLUME_AVG:
            continue

        # 6-month momentum
        mom_6m = (current_price - float(close.iloc[-MOM_LOOKBACK])) / float(close.iloc[-MOM_LOOKBACK])

        # 20-day realized volatility (for scoring)
        daily_ret = close.pct_change().dropna()
        vol_20d = float(daily_ret.iloc[-VOL_LOOKBACK_SCORE:].std() * np.sqrt(252))

        # 63-day realized volatility (for weighting)
        vol_63d = float(daily_ret.iloc[-VOL_LOOKBACK_WEIGHT:].std() * np.sqrt(252))

        # RS vs SPY
        rs_vs_spy = mom_6m - spy_ret_6m

        # Score = momentum / volatility (risk-adjusted momentum)
        if vol_20d > 0.01:  # Avoid division by near-zero vol
            score = mom_6m / vol_20d
        else:
            score = 0.0

        results[ticker] = {
            'score': round(score, 4),
            'mom_6m': round(mom_6m * 100, 2),
            'vol_20d': round(vol_20d * 100, 2),
            'vol_63d': round(vol_63d * 100, 2),
            'rs_vs_spy': round(rs_vs_spy * 100, 2),
            'price': round(current_price, 2),
            'avg_volume': int(avg_vol),
        }

    return results


def compute_inverse_vol_weights(scores: dict, top_tickers: list) -> dict:
    """Compute inverse-vol weights for the selected tickers."""
    vols = {}
    for t in top_tickers:
        vol = scores[t]['vol_63d']
        if vol > 0:
            vols[t] = vol

    if not vols:
        return {t: 1.0 / len(top_tickers) for t in top_tickers}

    inv_vol = {t: 1.0 / v for t, v in vols.items()}
    total = sum(inv_vol.values())
    weights = {t: round(v / total, 4) for t, v in inv_vol.items()}
    return weights


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


def buy_position(state: dict, ticker: str, price: float, amount: float, weight: float):
    """Buy a position and update state."""
    shares = amount / price
    state['cash'] -= amount

    pos = {
        'ticker': ticker,
        'shares': round(shares, 4),
        'entry_price': round(price, 2),
        'entry_date': str(datetime.now().date()),
        'weight': round(weight, 4),
    }
    state['positions'].append(pos)

    state['trade_history'].append({
        'date': str(datetime.now().date()),
        'action': 'BUY',
        'ticker': ticker,
        'shares': round(shares, 4),
        'price': round(price, 2),
        'amount': round(amount, 2),
        'weight': round(weight, 4),
    })

    log.info(f"  BUY {shares:.4f} {ticker} @ ${price:.2f} (${amount:.2f}, wt={weight:.1%})")


def calculate_equity(state: dict) -> float:
    """Calculate total portfolio equity."""
    equity = state['cash']
    for pos in state['positions']:
        try:
            price = get_current_price(pos['ticker'])
            equity += price * pos['shares']
        except Exception:
            equity += pos['entry_price'] * pos['shares']
    return equity


def main():
    import yfinance as yf

    log.info("=" * 60)
    log.info("Momentum Growth Rotation Paper Engine - Daily Run")
    log.info("=" * 60)

    state = load_state()
    today = datetime.now()

    if today.weekday() >= 5:
        log.info(f"Weekend ({today.strftime('%A')}), skipping")
        return

    # --- Determine if rebalance needed ---
    state['days_since_rebalance'] = state.get('days_since_rebalance', 999) + 1
    needs_rebalance = (
        state['days_since_rebalance'] >= REBALANCE_DAYS
        or not state['positions']  # no positions (first run or all sold)
    )

    if not needs_rebalance:
        # Non-rebalance day: just mark to market
        equity = calculate_equity(state)
        state['equity_curve'].append({
            'date': str(today.date()),
            'equity': round(equity, 2),
        })

        pnl_pct = (equity / CAPITAL_INITIAL - 1) * 100
        holdings = ', '.join(f"{p['ticker']}({p['weight']:.0%})" for p in state['positions'])
        log.info(f"Day {state['days_since_rebalance']}/{REBALANCE_DAYS} — "
                 f"Equity: ${equity:.2f} ({pnl_pct:+.1f}%) | Holdings: {holdings}")
        save_state(state)
        return

    # === REBALANCE DAY ===
    log.info("=" * 40)
    log.info("REBALANCE DAY — Computing momentum scores")
    log.info("=" * 40)

    # Download data for universe + benchmark
    all_tickers = list(set(UNIVERSE + [BENCHMARK]))
    data = download_universe_data(all_tickers, period='8mo')

    if BENCHMARK not in data:
        log.error("SPY data not available, cannot compute relative strength")
        save_state(state)
        return

    spy_data = data.pop(BENCHMARK)

    # Compute scores
    scores = compute_rs_vol_scores(data, spy_data)
    if len(scores) < TOP_K:
        log.error(f"Only {len(scores)} scored tickers, need at least {TOP_K}")
        save_state(state)
        return

    # Rank by score (risk-adjusted momentum)
    ranked = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)

    log.info(f"\nTop {min(15, len(ranked))} Momentum Scores:")
    for i, (ticker, info) in enumerate(ranked[:15]):
        marker = " <-- SELECTED" if i < TOP_K else ""
        log.info(f"  #{i+1} {ticker}: score={info['score']:.3f} "
                 f"mom_6m={info['mom_6m']:+.1f}% vol_20d={info['vol_20d']:.1f}% "
                 f"RS={info['rs_vs_spy']:+.1f}%{marker}")

    # Select top K
    top_tickers = [t for t, _ in ranked[:TOP_K]]

    # Compute inverse-vol weights
    weights = compute_inverse_vol_weights(scores, top_tickers)

    log.info(f"\nTarget weights:")
    for t in top_tickers:
        log.info(f"  {t}: {weights[t]:.1%}")

    # --- Close all existing positions ---
    for pos in list(state['positions']):
        try:
            price = get_current_price(pos['ticker'])
        except Exception:
            price = pos['entry_price']
        sell_position(state, pos, price, reason='rebalance')
    state['positions'] = []

    # --- Buy new positions ---
    equity = state['cash']
    log.info(f"\nAllocating ${equity:.2f} across {TOP_K} positions")

    for ticker in top_tickers:
        w = weights.get(ticker, 1.0 / TOP_K)
        amount = equity * w
        if amount < 10:
            continue
        try:
            price = get_current_price(ticker)
            buy_position(state, ticker, price, amount, w)
        except Exception as e:
            log.error(f"  Failed to buy {ticker}: {e}")

    # --- Update state ---
    state['days_since_rebalance'] = 0
    state['last_rebalance_date'] = str(today.date())

    current_equity = calculate_equity(state)
    state['equity_curve'].append({
        'date': str(today.date()),
        'equity': round(current_equity, 2),
    })

    state['rebalance_history'].append({
        'date': str(today.date()),
        'holdings': top_tickers,
        'weights': weights,
        'scores': {t: scores[t]['score'] for t in top_tickers},
        'all_top10': [(t, scores[t]['score']) for t, _ in ranked[:10]],
    })

    # --- Summary ---
    n_rebal = len(state['rebalance_history'])
    pnl_pct = (current_equity / CAPITAL_INITIAL - 1) * 100
    wr = (state['winning_trades'] / state['total_trades'] * 100) if state['total_trades'] > 0 else 0

    log.info(f"\n{'=' * 40}")
    log.info(f"REBALANCE #{n_rebal} COMPLETE")
    log.info(f"  Holdings: {', '.join(top_tickers)}")
    log.info(f"  Equity: ${current_equity:.2f} ({pnl_pct:+.1f}% total)")
    log.info(f"  Cash: ${state['cash']:.2f}")
    log.info(f"  Trades: {state['total_trades']} (WR: {wr:.0f}%)")
    log.info(f"  Next rebalance in ~{REBALANCE_DAYS} trading days")
    log.info(f"{'=' * 40}")

    save_state(state)
    log.info("State saved. Done.")


if __name__ == '__main__':
    main()
