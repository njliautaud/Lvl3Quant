#!/usr/bin/env python3
"""
Volume Surge (Volume Anomaly E) Paper Trading Engine
=====================================================

Buys sector ETFs when they show 3 consecutive days of volume > 1.5x
their 20-day average volume. Holds for 10 trading days.

STRATEGY (Passed 5/5 validation gates + 4/5 adversarial tests):
  - Universe: 13 sector ETFs (SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLP, XLY, XLB, XLU, XLRE, XLC)
  - Signal: Volume > 1.5x 20-day average for 3 CONSECUTIVE days
  - Entry: Buy at close on day 3 of volume surge
  - Hold: 10 trading days
  - Position size: $200 per position
  - Max 3 simultaneous positions
  - Long only

Usage:
  python3 paper_engines/volume_surge_paper.py
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LOG_DIR = Path(__file__).resolve().parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'volume_surge_paper.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

STATE_DIR = Path(__file__).resolve().parent.parent / 'state'
STATE_DIR.mkdir(exist_ok=True)
STATE_PATH = STATE_DIR / 'volume_surge_paper_state.json'

# Strategy parameters
POSITION_SIZE = 200       # $200 per position
MAX_CONCURRENT = 3        # max 3 simultaneous positions
HOLD_DAYS = 10            # trading days
VOLUME_THRESHOLD = 1.5    # 1.5x 20-day average volume
VOLUME_LOOKBACK = 20      # 20-day volume average
CONSECUTIVE_DAYS = 3      # 3 consecutive days required
DATA_DAYS = 35            # enough history for 20-day avg + 3-day check + buffer

UNIVERSE = [
    'SPY', 'QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
    'XLP', 'XLY', 'XLB', 'XLU', 'XLRE', 'XLC'
]


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        'capital_initial': POSITION_SIZE * MAX_CONCURRENT,
        'capital': POSITION_SIZE * MAX_CONCURRENT,
        'positions': [],
        'closed_trades': [],
        'created': str(datetime.now()),
        'last_run': None,
    }


def save_state(state):
    state['last_run'] = str(datetime.now())
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def check_volume_surge(ticker):
    """Check if ticker has 3 consecutive days of volume > 1.5x 20-day average.

    Returns dict with signal info or None.
    """
    import yfinance as yf

    try:
        df = yf.download(ticker, period=f'{DATA_DAYS}d', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < VOLUME_LOOKBACK + CONSECUTIVE_DAYS:
            log.warning(f"  {ticker}: insufficient data ({len(df)} rows)")
            return None

        volume = df['Volume']
        close = df['Close']

        # Compute 20-day rolling average volume (shifted by 1 to avoid lookahead)
        avg_vol_20 = volume.rolling(VOLUME_LOOKBACK).mean().shift(1)

        # Volume ratio for each day
        vol_ratio = volume / avg_vol_20

        # Check last 3 days for consecutive surge
        recent_ratios = vol_ratio.iloc[-CONSECUTIVE_DAYS:]
        if recent_ratios.isna().any():
            return None

        all_above = all(r >= VOLUME_THRESHOLD for r in recent_ratios.values)

        if all_above:
            current_price = float(close.iloc[-1])
            avg_ratio = float(recent_ratios.mean())
            latest_ratio = float(recent_ratios.iloc[-1])
            avg_volume = float(avg_vol_20.iloc[-1])
            current_volume = float(volume.iloc[-1])

            return {
                'ticker': ticker,
                'current_price': round(current_price, 2),
                'volume_ratio_avg_3d': round(avg_ratio, 2),
                'volume_ratio_latest': round(latest_ratio, 2),
                'avg_volume_20d': int(avg_volume),
                'current_volume': int(current_volume),
                'day1_ratio': round(float(recent_ratios.iloc[0]), 2),
                'day2_ratio': round(float(recent_ratios.iloc[1]), 2),
                'day3_ratio': round(float(recent_ratios.iloc[2]), 2),
            }

    except Exception as e:
        log.warning(f"  {ticker}: Error checking volume ({e})")

    return None


def main():
    import yfinance as yf

    log.info("=" * 50)
    log.info("Volume Surge Paper Engine — Daily Run")
    log.info("=" * 50)

    state = load_state()
    today = datetime.now()

    if today.weekday() >= 5:
        log.info(f"Weekend ({today.strftime('%A')}), skipping")
        save_state(state)
        return

    # ── Process exits first ──
    remaining = []
    for pos in state['positions']:
        entry_date = datetime.strptime(pos['entry_date'], '%Y-%m-%d')
        bdays = int(np.busday_count(entry_date.date(), today.date()))

        if bdays >= HOLD_DAYS:
            # Exit position — fetch current price
            try:
                df = yf.download(pos['ticker'], period='5d', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                exit_price = float(df['Close'].iloc[-1])
            except Exception:
                exit_price = pos['entry_price']

            ret_pct = (exit_price / pos['entry_price'] - 1) * 100
            commission_pct = 0.1  # 10 bps RT for ETFs
            pnl_pct = ret_pct - commission_pct
            pnl_dollar = POSITION_SIZE * pnl_pct / 100

            closed = {
                **pos,
                'exit_date': today.strftime('%Y-%m-%d'),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret_pct, 2),
                'pnl_pct': round(pnl_pct, 2),
                'pnl_dollar': round(pnl_dollar, 2),
                'days_held': bdays,
            }
            state['closed_trades'].append(closed)
            state['capital'] += pnl_dollar

            log.info(f"CLOSED: {pos['ticker']} | {ret_pct:+.1f}% (${pnl_dollar:+.2f}) | "
                     f"{bdays}d held | Entry ${pos['entry_price']} -> Exit ${exit_price:.2f}")
        else:
            remaining.append(pos)

    state['positions'] = remaining

    # ── Scan for new entries ──
    if len(state['positions']) < MAX_CONCURRENT:
        slots = MAX_CONCURRENT - len(state['positions'])
        log.info(f"\nScanning for volume surge entries ({slots} slot(s) open)...")

        # Skip tickers we already hold
        held_tickers = set(p['ticker'] for p in state['positions'])
        # Skip recently closed (within 15 trading days to avoid re-entry overlap)
        recent_closed = set(
            t['ticker'] for t in state['closed_trades']
            if (today - datetime.strptime(t['exit_date'], '%Y-%m-%d')).days < 20
        )
        skip_tickers = held_tickers | recent_closed

        candidates = []
        for ticker in UNIVERSE:
            if ticker in skip_tickers:
                continue
            signal = check_volume_surge(ticker)
            if signal:
                candidates.append(signal)
                log.info(f"  SIGNAL: {ticker} — 3-day vol ratios: "
                         f"{signal['day1_ratio']}x, {signal['day2_ratio']}x, {signal['day3_ratio']}x "
                         f"(avg {signal['volume_ratio_avg_3d']}x) | ${signal['current_price']}")

        # Sort by average 3-day volume ratio (strongest surge first)
        candidates.sort(key=lambda x: x['volume_ratio_avg_3d'], reverse=True)

        # Open positions for top candidates
        for cand in candidates[:slots]:
            pos = {
                'ticker': cand['ticker'],
                'entry_date': today.strftime('%Y-%m-%d'),
                'entry_price': cand['current_price'],
                'position_size': POSITION_SIZE,
                'volume_ratio_at_entry': cand['volume_ratio_avg_3d'],
                'volume_ratio_latest': cand['volume_ratio_latest'],
                'day1_ratio': cand['day1_ratio'],
                'day2_ratio': cand['day2_ratio'],
                'day3_ratio': cand['day3_ratio'],
                'avg_volume_20d': cand['avg_volume_20d'],
                'entry_volume': cand['current_volume'],
                'target_exit_date': str(np.busday_offset(today.date(), HOLD_DAYS)),
            }
            state['positions'].append(pos)
            log.info(f"OPENED: {cand['ticker']} at ${cand['current_price']} | "
                     f"Vol ratio {cand['volume_ratio_avg_3d']}x avg | "
                     f"Size: ${POSITION_SIZE} | Exit target: {pos['target_exit_date']}")

        if not candidates:
            log.info("  No volume surge signals today")

    # ── Summary ──
    n_closed = len(state['closed_trades'])
    n_wins = sum(1 for t in state['closed_trades'] if t.get('pnl_pct', 0) > 0)
    wr = n_wins / n_closed * 100 if n_closed > 0 else 0
    total_pnl = sum(t.get('pnl_dollar', 0) for t in state['closed_trades'])

    log.info(f"\n--- Volume Surge Summary ---")
    log.info(f"Capital: ${state['capital']:,.2f} | Total P&L: ${total_pnl:+.2f}")
    log.info(f"Open: {len(state['positions'])} | Closed: {n_closed} | WR: {wr:.0f}%")

    for pos in state['positions']:
        entry_date = datetime.strptime(pos['entry_date'], '%Y-%m-%d')
        bdays = int(np.busday_count(entry_date.date(), today.date()))

        # Fetch current price for unrealized P&L
        try:
            df = yf.download(pos['ticker'], period='5d', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            cur_price = float(df['Close'].iloc[-1])
            unrealized = (cur_price / pos['entry_price'] - 1) * 100
            price_str = f"${cur_price:.2f} ({unrealized:+.1f}%)"
        except Exception:
            price_str = "N/A"

        log.info(f"  {pos['ticker']}: entry ${pos['entry_price']} on {pos['entry_date']} "
                 f"(day {bdays}/{HOLD_DAYS}) | vol {pos['volume_ratio_at_entry']}x | "
                 f"now {price_str}")

    save_state(state)
    log.info("State saved. Done.")


if __name__ == '__main__':
    main()
