#!/usr/bin/env python3
"""
Flow Reversal 2x Leveraged Paper Engine
=========================================
Based on the Flow Reversal AVO strategy (v25, lockbox Sharpe 3.37).
2x leverage on $10K capital ($20K deployed).
Tracks margin costs at 6% annualized on borrowed portion.

Signal logic:
  - Universe: 18 sector/thematic ETFs
  - Volume z-score (20d lookback) >= 2.0 AND 1d return <= -1%
  - This pattern = institutional accumulation during a dip ("flow reversal")
  - Max 2 concurrent positions, 20% of *deployed* capital per trade
  - Take profit: +3.5%, Trailing stop: -2%, Underwater cut: 1 day at -0.4%
  - Max hold: 3 days

Leverage:
  - 2x = $20K deployed on $10K capital
  - Margin cost: 6% annualized on borrowed $10K, charged daily
  - P&L is 2x the underlying move (minus margin costs)

Usage:
  python3 paper_engines/flow_reversal_2x_paper.py
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

STATE_PATH = STATE_DIR / 'flow_reversal_2x_paper_state.json'
LOG_PATH = LOG_DIR / 'flow_reversal_2x_paper.log'

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
LEVERAGE = 2.0
DEPLOYED_CAPITAL = CAPITAL_INITIAL * LEVERAGE  # $20K
MARGIN_RATE = 0.06  # 6% annualized on borrowed portion
BORROWED = CAPITAL_INITIAL * (LEVERAGE - 1)  # $10K borrowed
DAILY_MARGIN_COST = BORROWED * MARGIN_RATE / 252  # ~$2.38/day when fully deployed

# Signal parameters (from AVO v25)
UNIVERSE = [
    'ARKK', 'IGV', 'ITB', 'KRE', 'KWEB', 'OIH', 'SMH',
    'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP',
    'XLRE', 'XLU', 'XLV', 'XLY',
]
MAX_CONCURRENT = 2
MAX_PER_TRADE_PCT = 0.20  # 20% of deployed capital per trade
MAX_HOLD_DAYS = 3
TAKE_PROFIT = 0.035       # +3.5%
TRAILING_STOP = -0.02     # -2%
UNDERWATER_CUT_DAYS = 1
UNDERWATER_TOLERANCE = -0.004  # -0.4%
VOLUME_LOOKBACK = 20
VOLUME_Z_THRESHOLD = 2.0
PRICE_DIP_THRESHOLD = -0.01  # -1% daily return


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
        'positions': [],
        'equity_curve': [],
        'trade_history': [],
        'created': str(datetime.now()),
        'last_run': None,
        'total_trades': 0,
        'winning_trades': 0,
        'total_margin_cost': 0.0,
        'leverage': LEVERAGE,
    }


def save_state(state: dict):
    state['last_run'] = str(datetime.now())
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def get_price_data(tickers, period='30d'):
    """Download recent price/volume data for tickers."""
    import yfinance as yf
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) >= VOLUME_LOOKBACK + 2:
                data[ticker] = df
        except Exception as e:
            log.warning(f"  {ticker}: download failed ({e})")
    return data


def get_current_price(ticker: str) -> float:
    """Get current price for a ticker."""
    import yfinance as yf
    df = yf.download(ticker, period='5d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if len(df) == 0:
        raise ValueError(f"No data for {ticker}")
    return float(df['Close'].iloc[-1])


def check_flow_reversal_signals(data: dict) -> list:
    """
    Scan universe for flow reversal signals.
    Signal: volume z-score >= threshold AND 1d return <= dip threshold.
    """
    signals = []
    for ticker, df in data.items():
        close = df['Close']
        volume = df['Volume']

        if len(close) < VOLUME_LOOKBACK + 2:
            continue

        # Volume z-score (last bar vs 20d rolling mean/std)
        vol_mean = volume.iloc[-VOLUME_LOOKBACK-1:-1].mean()
        vol_std = volume.iloc[-VOLUME_LOOKBACK-1:-1].std()
        if vol_std == 0 or pd.isna(vol_std):
            continue

        current_vol = float(volume.iloc[-1])
        vol_z = (current_vol - vol_mean) / vol_std

        # 1-day return
        ret_1d = (float(close.iloc[-1]) - float(close.iloc[-2])) / float(close.iloc[-2])

        # Signal: high volume + price dip
        if vol_z >= VOLUME_Z_THRESHOLD and ret_1d <= PRICE_DIP_THRESHOLD:
            score = vol_z * abs(ret_1d)
            signals.append({
                'ticker': ticker,
                'price': float(close.iloc[-1]),
                'vol_z': round(vol_z, 2),
                'ret_1d': round(ret_1d * 100, 2),
                'score': round(score, 4),
            })
            log.info(f"  SIGNAL: {ticker} vol_z={vol_z:.2f} ret_1d={ret_1d*100:.2f}% score={score:.4f}")

    # Sort by score descending
    signals.sort(key=lambda x: x['score'], reverse=True)
    return signals


def manage_positions(state: dict):
    """Check exits for existing positions: TP, trailing stop, underwater cut, max hold."""
    today = datetime.now().date()
    positions_to_close = []

    for pos in state['positions']:
        try:
            current_price = get_current_price(pos['ticker'])
        except Exception as e:
            log.warning(f"  Cannot get price for {pos['ticker']}: {e}")
            continue

        entry_price = pos['entry_price']
        pnl_pct = (current_price - entry_price) / entry_price
        days_held = (today - datetime.strptime(pos['entry_date'], '%Y-%m-%d').date()).days

        # Update peak price for trailing stop
        if current_price > pos.get('peak_price', entry_price):
            pos['peak_price'] = round(current_price, 2)

        peak = pos.get('peak_price', entry_price)
        drawdown_from_peak = (current_price - peak) / peak

        reason = None

        # Take profit
        if pnl_pct >= TAKE_PROFIT:
            reason = f'take_profit ({pnl_pct*100:+.1f}%)'

        # Trailing stop from peak
        elif drawdown_from_peak <= TRAILING_STOP:
            reason = f'trailing_stop (peak=${peak:.2f}, dd={drawdown_from_peak*100:.1f}%)'

        # Underwater cut: held >= 1 day and still down > tolerance
        elif days_held >= UNDERWATER_CUT_DAYS and pnl_pct <= UNDERWATER_TOLERANCE:
            reason = f'underwater_cut (day {days_held}, pnl={pnl_pct*100:.1f}%)'

        # Max hold days
        elif days_held >= MAX_HOLD_DAYS:
            reason = f'max_hold ({days_held} days)'

        if reason:
            positions_to_close.append((pos, current_price, reason))

    # Close positions
    for pos, price, reason in positions_to_close:
        pnl_per_share = price - pos['entry_price']
        # Leveraged P&L: shares already reflect 2x sizing
        pnl = pnl_per_share * pos['shares']
        pnl_pct = (price / pos['entry_price'] - 1) * 100
        proceeds = price * pos['shares']

        state['cash'] += proceeds
        state['positions'].remove(pos)
        state['total_trades'] += 1
        if pnl > 0:
            state['winning_trades'] += 1

        state['trade_history'].append({
            'date': str(today),
            'action': 'SELL',
            'ticker': pos['ticker'],
            'shares': round(pos['shares'], 4),
            'price': round(price, 2),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'reason': reason,
        })

        log.info(f"  SELL {pos['shares']:.4f} {pos['ticker']} @ ${price:.2f} | "
                 f"P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%) | Reason: {reason}")

    return len(positions_to_close)


def charge_daily_margin(state: dict):
    """Charge daily margin cost on borrowed capital for open positions."""
    if not state['positions']:
        return 0.0

    # Charge proportional to how much of the borrowed capital is deployed
    total_position_value = sum(p['entry_price'] * p['shares'] for p in state['positions'])
    # The borrowed fraction is (position_value - equity_in_positions) / borrowed_capacity
    # Simplify: charge on the borrowed portion proportional to deployment
    deployment_ratio = min(total_position_value / DEPLOYED_CAPITAL, 1.0) if DEPLOYED_CAPITAL > 0 else 0
    daily_cost = DAILY_MARGIN_COST * deployment_ratio

    state['cash'] -= daily_cost
    state['total_margin_cost'] = state.get('total_margin_cost', 0) + daily_cost

    if daily_cost > 0.01:
        log.info(f"  Margin cost: ${daily_cost:.2f} (deployment: {deployment_ratio*100:.0f}%)")

    return daily_cost


def calculate_equity(state: dict) -> float:
    """Total equity = cash + market value of positions."""
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
    log.info("Flow Reversal 2x Leveraged Paper Engine - Daily Run")
    log.info("=" * 60)

    state = load_state()
    today = datetime.now()

    if today.weekday() >= 5:
        log.info(f"Weekend ({today.strftime('%A')}), skipping")
        return

    # --- Charge daily margin cost ---
    margin_cost = charge_daily_margin(state)

    # --- Manage existing positions (exits) ---
    n_closed = manage_positions(state)
    if n_closed:
        log.info(f"  Closed {n_closed} position(s)")

    # --- Check for new signals if we have capacity ---
    open_slots = MAX_CONCURRENT - len(state['positions'])
    if open_slots > 0:
        log.info(f"Scanning for flow reversal signals ({open_slots} slot(s) available)...")
        data = get_price_data(UNIVERSE, period='60d')
        signals = check_flow_reversal_signals(data)

        # Filter out tickers we already hold
        held_tickers = {p['ticker'] for p in state['positions']}
        signals = [s for s in signals if s['ticker'] not in held_tickers]

        # Open new positions
        for signal in signals[:open_slots]:
            ticker = signal['ticker']
            price = signal['price']

            # Position size: 20% of deployed capital (2x)
            trade_amount = DEPLOYED_CAPITAL * MAX_PER_TRADE_PCT
            # But can't spend more cash than we have
            trade_amount = min(trade_amount, state['cash'] * 0.95)
            if trade_amount < 50:
                log.info(f"  Insufficient cash for {ticker} (need ${trade_amount:.0f}, have ${state['cash']:.0f})")
                continue

            shares = trade_amount / price
            state['cash'] -= trade_amount

            pos = {
                'ticker': ticker,
                'shares': round(shares, 4),
                'entry_price': round(price, 2),
                'entry_date': str(today.date()),
                'peak_price': round(price, 2),
            }
            state['positions'].append(pos)

            state['trade_history'].append({
                'date': str(today.date()),
                'action': 'BUY',
                'ticker': ticker,
                'shares': round(shares, 4),
                'price': round(price, 2),
                'amount': round(trade_amount, 2),
                'vol_z': signal['vol_z'],
                'ret_1d': signal['ret_1d'],
            })

            log.info(f"  BUY {shares:.4f} {ticker} @ ${price:.2f} (${trade_amount:.2f}) "
                     f"[vol_z={signal['vol_z']}, ret_1d={signal['ret_1d']}%]")
    else:
        log.info(f"At max capacity ({MAX_CONCURRENT} positions), no new entries")

    # --- Mark to market ---
    equity = calculate_equity(state)
    pnl_pct = (equity / CAPITAL_INITIAL - 1) * 100
    leverage_actual = (equity + sum(p['entry_price'] * p['shares'] for p in state['positions']) - state['cash']) / equity if equity > 0 else 0

    state['equity_curve'].append({
        'date': str(today.date()),
        'equity': round(equity, 2),
    })

    # --- Summary ---
    holdings = ', '.join(f"{p['ticker']}" for p in state['positions']) or 'CASH'
    wr = (state['winning_trades'] / state['total_trades'] * 100) if state['total_trades'] > 0 else 0
    margin_total = state.get('total_margin_cost', 0)

    log.info(f"\n{'=' * 40}")
    log.info(f"DAILY SUMMARY")
    log.info(f"  Equity: ${equity:.2f} ({pnl_pct:+.1f}%)")
    log.info(f"  Cash: ${state['cash']:.2f}")
    log.info(f"  Holdings: {holdings}")
    log.info(f"  Positions: {len(state['positions'])}/{MAX_CONCURRENT}")
    log.info(f"  Leverage: {LEVERAGE}x | Margin cost to date: ${margin_total:.2f}")
    log.info(f"  Trades: {state['total_trades']} (WR: {wr:.0f}%)")
    log.info(f"{'=' * 40}")

    save_state(state)
    log.info("State saved. Done.")


if __name__ == '__main__':
    main()
