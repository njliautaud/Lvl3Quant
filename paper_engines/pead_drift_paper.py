#!/usr/bin/env python3
"""
PEAD (Post-Earnings Announcement Drift) Paper Trading Engine
==============================================================

Buys stocks that had positive earnings surprises (2%+ gap) once IV rank
drops below 50th percentile. Holds for 40 trading days.

STRATEGY (Backtest: Sharpe 1.18, CAGR 36.8%, WR 63.5%, 3/4 gates):
  - Signal: Stock gapped 2%+ on earnings (positive surprise)
  - Entry: 1 day after earnings, if IV rank < 50th percentile
  - Hold: 40 trading days
  - Max 5 concurrent positions, equal weight
  - Capital: $100K paper
  - Long only (short side = 44% WR, dead)

Usage:
  python3 paper_engines/pead_drift_paper.py
"""

import json
import logging
import warnings
from datetime import datetime, timedelta
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
        logging.FileHandler(LOG_DIR / 'pead_drift_paper.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

STATE_DIR = Path(__file__).resolve().parent.parent / 'state'
STATE_DIR.mkdir(exist_ok=True)
STATE_PATH = STATE_DIR / 'pead_drift_paper_state.json'

CAPITAL_INITIAL = 100000
MAX_CONCURRENT = 5
HOLD_DAYS = 40  # trading days
MIN_GAP_PCT = 2.0
IV_RANK_MAX = 50

# ── Magnitude-Based Position Sizing (HC: PEAD magnitude scaling research) ──
# DOWN gaps drift 2.81% avg vs UP gaps 0.67% (4x more profitable)
# 10-20% gaps have 72% win rate (sweet spot)
# Tiers: 5-10% = base, 10-20% = 2x, 20%+ = 3x
MAGNITUDE_TIERS = {
    'small':  {'min': 2.0, 'max': 10.0, 'position_size': 100, 'multiplier': 1.0, 'label': '2-10% gap'},
    'medium': {'min': 10.0, 'max': 20.0, 'position_size': 150, 'multiplier': 2.0, 'label': '10-20% gap (sweet spot)'},
    'large':  {'min': 20.0, 'max': 999.0, 'position_size': 200, 'multiplier': 3.0, 'label': '20%+ gap'},
}

# DOWN gaps drift more reliably (2.81% avg vs 0.67% for UP)
# Apply 1.5x weight multiplier for down-gap PEAD trades
GAP_DIRECTION_WEIGHT_DOWN = 1.5
GAP_DIRECTION_WEIGHT_UP = 1.0

UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'V', 'JPM', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK',
    'ABBV', 'PEP', 'COST', 'KO', 'AVGO', 'LLY', 'WMT', 'TMO', 'MCD',
    'CSCO', 'ACN', 'DHR', 'ABT', 'NEE', 'TXN', 'PM', 'UNP', 'RTX',
    'LOW', 'HON', 'AMGN', 'IBM', 'CAT', 'GS', 'BA', 'SBUX', 'GE',
    'MMM', 'DIS', 'INTC', 'NKE', 'CRM'
]


def get_magnitude_tier(gap_pct):
    """Determine position sizing tier based on gap magnitude.

    Returns dict with tier info: name, position_size, multiplier, direction_weight.
    DOWN gaps get 1.5x weight (they drift 4x more reliably than UP gaps).
    """
    abs_gap = abs(gap_pct)

    for tier_name, tier in MAGNITUDE_TIERS.items():
        if tier['min'] <= abs_gap < tier['max']:
            direction_weight = GAP_DIRECTION_WEIGHT_DOWN if gap_pct < 0 else GAP_DIRECTION_WEIGHT_UP
            effective_size = tier['position_size'] * direction_weight
            return {
                'tier_name': tier_name,
                'tier_label': tier['label'],
                'base_position_size': tier['position_size'],
                'multiplier': tier['multiplier'],
                'direction': 'DOWN' if gap_pct < 0 else 'UP',
                'direction_weight': direction_weight,
                'effective_position_size': round(effective_size, 2),
            }

    # Default for gaps below minimum (shouldn't happen with MIN_GAP_PCT guard)
    return {
        'tier_name': 'sub_minimum',
        'tier_label': f'{abs_gap:.1f}% gap (below threshold)',
        'base_position_size': 0,
        'multiplier': 0,
        'direction': 'DOWN' if gap_pct < 0 else 'UP',
        'direction_weight': 1.0,
        'effective_position_size': 0,
    }


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        'capital': CAPITAL_INITIAL,
        'positions': [],
        'closed_trades': [],
        'created': str(datetime.now()),
        'last_run': None,
    }


def save_state(state):
    state['last_run'] = str(datetime.now())
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def get_iv_rank(close_series):
    """Compute IV rank proxy: current 20d vol percentile vs trailing 252d."""
    if len(close_series) < 260:
        return None
    log_ret = np.log(close_series / close_series.shift(1)).dropna()
    if len(log_ret) < 252:
        return None
    current_vol = log_ret.iloc[-20:].std() * np.sqrt(252)
    trailing_vols = log_ret.rolling(20).std().iloc[-252:] * np.sqrt(252)
    trailing_vols = trailing_vols.dropna()
    if len(trailing_vols) < 100:
        return None
    return float((trailing_vols < current_vol).mean() * 100)


def check_recent_earnings(ticker):
    """Check if ticker had a 2%+ positive earnings gap in the last 5 trading days."""
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)

        # Get earnings dates
        cal = stock.get_earnings_dates(limit=5)
        if cal is None or len(cal) == 0:
            return None

        # Get recent price data
        hist = stock.history(period='2y', auto_adjust=True)
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        if len(hist) < 260:
            return None

        close = hist['Close']
        today = close.index[-1]

        # Check each recent earnings date
        for idx in cal.index:
            # Handle timezone
            if hasattr(idx, 'tzinfo') and idx.tzinfo is not None:
                edate = pd.Timestamp(idx).tz_convert(None) if hasattr(idx, 'tz_convert') else pd.Timestamp(idx).replace(tzinfo=None)
            else:
                edate = pd.Timestamp(idx)

            # Only look at earnings in the last 5 trading days
            days_ago = (today - edate).days
            if days_ago < 0 or days_ago > 7:
                continue

            # Find the earnings day in our price data
            loc = close.index.searchsorted(edate)
            if loc < 2 or loc >= len(close):
                continue

            pre_price = close.iloc[loc - 1]
            post_price = close.iloc[loc]
            if pre_price <= 0:
                continue

            gap_pct = (post_price / pre_price - 1) * 100

            if gap_pct >= MIN_GAP_PCT:
                # Check IV rank
                iv_rank = get_iv_rank(close)
                if iv_rank is not None and iv_rank <= IV_RANK_MAX:
                    return {
                        'ticker': ticker,
                        'earnings_date': str(edate.date()),
                        'gap_pct': round(gap_pct, 2),
                        'iv_rank': round(iv_rank, 1),
                        'current_price': round(float(close.iloc[-1]), 2),
                    }

    except Exception as e:
        log.warning(f"  {ticker}: Error checking earnings ({e})")

    return None


def main():
    import yfinance as yf

    log.info("=" * 50)
    log.info("PEAD Drift Paper Engine — Daily Run")
    log.info("=" * 50)

    state = load_state()
    today = datetime.now()

    if today.weekday() >= 5:
        log.info(f"Weekend ({today.strftime('%A')}), skipping")
        save_state(state)
        return

    # Process exits first
    remaining = []
    for pos in state['positions']:
        entry_date = datetime.strptime(pos['entry_date'], '%Y-%m-%d')
        bdays = np.busday_count(entry_date.date(), today.date())

        if bdays >= HOLD_DAYS:
            # Exit position
            try:
                df = yf.download(pos['ticker'], period='5d', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                exit_price = float(df['Close'].iloc[-1])
            except:
                exit_price = pos['entry_price']

            ret_pct = (exit_price / pos['entry_price'] - 1) * 100
            commission_pct = 0.1  # 10 bps RT

            closed = {
                **pos,
                'exit_date': today.strftime('%Y-%m-%d'),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret_pct, 2),
                'pnl_pct': round(ret_pct - commission_pct, 2),
                'days_held': bdays,
            }
            state['closed_trades'].append(closed)

            # Use magnitude-scaled position size if available, else fallback to equal-weight
            weight = pos.get('effective_position_size', state['capital'] / MAX_CONCURRENT)
            pnl_dollar = weight * (ret_pct - commission_pct) / 100
            state['capital'] += pnl_dollar

            log.info(f"CLOSED: {pos['ticker']} | {ret_pct:+.1f}% ({bdays}d) | "
                     f"Entry ${pos['entry_price']} → Exit ${exit_price:.2f}")
        else:
            remaining.append(pos)

    state['positions'] = remaining

    # Scan for new entries
    if len(state['positions']) < MAX_CONCURRENT:
        log.info(f"\nScanning for PEAD entries ({MAX_CONCURRENT - len(state['positions'])} slots open)...")

        # Skip tickers we already hold
        held_tickers = set(p['ticker'] for p in state['positions'])
        # Also skip recently closed (within 30 days)
        recent_closed = set(
            t['ticker'] for t in state['closed_trades']
            if (today - datetime.strptime(t['exit_date'], '%Y-%m-%d')).days < 30
        )
        skip_tickers = held_tickers | recent_closed

        candidates = []
        for ticker in UNIVERSE:
            if ticker in skip_tickers:
                continue

            signal = check_recent_earnings(ticker)
            if signal:
                candidates.append(signal)
                log.info(f"  SIGNAL: {ticker} gap +{signal['gap_pct']:.1f}% | "
                         f"IV rank {signal['iv_rank']:.0f}% | ${signal['current_price']}")

        # Sort by gap size (stronger surprise = stronger drift)
        candidates.sort(key=lambda x: x['gap_pct'], reverse=True)

        # Open positions for top candidates with magnitude-based sizing
        slots = MAX_CONCURRENT - len(state['positions'])
        for cand in candidates[:slots]:
            # Determine magnitude tier and position sizing
            tier_info = get_magnitude_tier(cand['gap_pct'])

            pos = {
                'ticker': cand['ticker'],
                'entry_date': today.strftime('%Y-%m-%d'),
                'entry_price': cand['current_price'],
                'earnings_gap': cand['gap_pct'],
                'iv_rank': cand['iv_rank'],
                'earnings_date': cand['earnings_date'],
                # Magnitude scaling fields
                'magnitude_tier': tier_info['tier_name'],
                'magnitude_tier_label': tier_info['tier_label'],
                'base_position_size': tier_info['base_position_size'],
                'magnitude_multiplier': tier_info['multiplier'],
                'gap_direction': tier_info['direction'],
                'direction_weight': tier_info['direction_weight'],
                'effective_position_size': tier_info['effective_position_size'],
            }
            state['positions'].append(pos)
            log.info(f"OPENED: {cand['ticker']} at ${cand['current_price']} | "
                     f"Earnings gap +{cand['gap_pct']:.1f}% ({tier_info['direction']}) | "
                     f"Tier: {tier_info['tier_name']} ({tier_info['tier_label']}) | "
                     f"Size: ${tier_info['effective_position_size']:.0f} "
                     f"(base ${tier_info['base_position_size']} x {tier_info['direction_weight']}x dir weight) | "
                     f"Hold target: {HOLD_DAYS}d")

        if not candidates:
            log.info("  No PEAD signals today")

    # Summary
    total_realized = sum(t.get('pnl_pct', 0) for t in state['closed_trades'])
    n_closed = len(state['closed_trades'])
    n_wins = sum(1 for t in state['closed_trades'] if t.get('pnl_pct', 0) > 0)
    wr = n_wins / n_closed * 100 if n_closed > 0 else 0

    log.info(f"\n--- Summary ---")
    log.info(f"Capital: ${state['capital']:,.2f} ({(state['capital']/CAPITAL_INITIAL-1)*100:+.1f}%)")
    log.info(f"Open: {len(state['positions'])} | Closed: {n_closed} | WR: {wr:.0f}%")

    for pos in state['positions']:
        entry_date = datetime.strptime(pos['entry_date'], '%Y-%m-%d')
        bdays = np.busday_count(entry_date.date(), today.date())
        tier_label = pos.get('magnitude_tier', 'unknown')
        eff_size = pos.get('effective_position_size', 'N/A')
        direction = pos.get('gap_direction', 'UP')
        log.info(f"  {pos['ticker']}: entry ${pos['entry_price']} on {pos['entry_date']} "
                 f"(day {bdays}/{HOLD_DAYS}) | gap +{pos['earnings_gap']:.1f}% {direction} | "
                 f"tier={tier_label} size=${eff_size}")

    save_state(state)
    log.info("State saved. Done.")


if __name__ == '__main__':
    main()
