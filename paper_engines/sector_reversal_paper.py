#!/usr/bin/env python3
"""
Sector-Relative Short-Term Reversal Paper Trading Engine
=========================================================

Buys S&P 500 stocks that drop >5% more than their sector ETF over
5 trading days. Holds for 10 trading days. Half-size when SPY < 200-SMA.

STRATEGY (Passed 5/5 validation gates, adversarial pending):
  - Universe: 100 S&P 500 large-cap stocks
  - Signal: Stock's 5-day return < sector ETF's 5-day return by >5%
  - Entry: Buy at close on signal day
  - Hold: 10 trading days
  - Position size: $130 full / $65 half (SPY < 200-SMA)
  - Max 5 simultaneous positions
  - Regime hedge: half-size in bear

Usage:
  python3 paper_engines/sector_reversal_paper.py
"""

import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

LOG_DIR = Path(__file__).resolve().parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'sector_reversal_paper.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

STATE_DIR = Path(__file__).resolve().parent.parent / 'state'
STATE_DIR.mkdir(exist_ok=True)
STATE_PATH = STATE_DIR / 'sector_reversal_paper_state.json'

# Strategy parameters
POSITION_SIZE_FULL = 130    # $130 per position (full size)
POSITION_SIZE_HALF = 65     # $65 in bear regime
MAX_POSITIONS = 5
HOLD_DAYS = 10
DROP_THRESHOLD = -0.05      # Stock must drop >5% more than sector
LOOKBACK_DAYS = 5           # 5-day return comparison
SMA_PERIOD = 200            # For regime detection

# Universe: 100 S&P 500 large-cap stocks
TICKERS = [
    'AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','BRK-B','UNH','JNJ',
    'JPM','V','PG','XOM','HD','MA','CVX','MRK','ABBV','LLY',
    'PEP','KO','COST','AVGO','TMO','MCD','WMT','CSCO','ACN','ABT',
    'DHR','CRM','NKE','ADBE','TXN','NEE','PM','UNP','RTX','HON',
    'LOW','INTC','UPS','QCOM','BA','AMGN','CAT','IBM','GE','SBUX',
    'INTU','ISRG','BLK','PLD','MDLZ','ADP','GILD','ADI','SYK',
    'DE','LMT','TJX','CB','REGN','MO','CI','SO','DUK','CL',
    'CME','ICE','PGR','SHW','ZTS','BSX','VRTX','FISV','APD','MCK',
    'EL','AON','HUM','EMR','ECL','SLB','ORLY','AIG','WM','PSA',
    'SPG','NSC','F','GM','USB','TFC','PNC','MS','GS','SCHW',
]

# Sector ETF mapping
SECTOR_MAP = {
    'AAPL':'XLK','MSFT':'XLK','AMZN':'XLY','GOOGL':'XLC','META':'XLC',
    'NVDA':'XLK','TSLA':'XLY','BRK-B':'XLF','UNH':'XLV','JNJ':'XLV',
    'JPM':'XLF','V':'XLK','PG':'XLP','XOM':'XLE','HD':'XLY',
    'MA':'XLK','CVX':'XLE','MRK':'XLV','ABBV':'XLV','LLY':'XLV',
    'PEP':'XLP','KO':'XLP','COST':'XLP','AVGO':'XLK','TMO':'XLV',
    'MCD':'XLY','WMT':'XLP','CSCO':'XLK','ACN':'XLK','ABT':'XLV',
    'DHR':'XLV','CRM':'XLK','NKE':'XLY','ADBE':'XLK','TXN':'XLK',
    'NEE':'XLU','PM':'XLP','UNP':'XLI','RTX':'XLI','HON':'XLI',
    'LOW':'XLY','INTC':'XLK','UPS':'XLI','QCOM':'XLK','BA':'XLI',
    'AMGN':'XLV','CAT':'XLI','IBM':'XLK','GE':'XLI','SBUX':'XLY',
    'INTU':'XLK','ISRG':'XLV','BLK':'XLF','PLD':'XLRE','MDLZ':'XLP',
    'ADP':'XLK','GILD':'XLV','ADI':'XLK','SYK':'XLV',
    'DE':'XLI','LMT':'XLI','TJX':'XLY','CB':'XLF','REGN':'XLV',
    'MO':'XLP','CI':'XLV','SO':'XLU','DUK':'XLU','CL':'XLP',
    'CME':'XLF','ICE':'XLF','PGR':'XLF','SHW':'XLB','ZTS':'XLV',
    'BSX':'XLV','VRTX':'XLV','FISV':'XLK','APD':'XLB','MCK':'XLV',
    'EL':'XLP','AON':'XLF','HUM':'XLV','EMR':'XLI','ECL':'XLB',
    'SLB':'XLE','ORLY':'XLY','AIG':'XLF','WM':'XLI','PSA':'XLRE',
    'SPG':'XLRE','NSC':'XLI','F':'XLY','GM':'XLY','USB':'XLF',
    'TFC':'XLF','PNC':'XLF','MS':'XLF','GS':'XLF','SCHW':'XLF',
}

SECTOR_ETFS = list(set(SECTOR_MAP.values()))


def load_state():
    """Load or initialize paper trading state."""
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        'positions': [],
        'closed_trades': [],
        'nav': 645.0,
        'cash': 645.0,
        'created_at': datetime.now().isoformat(),
        'last_scan': None,
    }


def save_state(state):
    """Save paper trading state."""
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def get_regime(spy_data):
    """Determine bull/bear regime based on SPY vs 200-SMA."""
    if len(spy_data) < SMA_PERIOD:
        return 'bull'  # Default to bull if insufficient data
    sma200 = spy_data['Close'].rolling(SMA_PERIOD).mean().iloc[-1]
    current = spy_data['Close'].iloc[-1]
    return 'bull' if current > sma200 else 'bear'


def scan_for_signals():
    """Scan all tickers for sector-relative oversold signals."""
    log.info("Downloading price data...")

    # Download all tickers + sector ETFs + SPY
    all_tickers = list(set(TICKERS + SECTOR_ETFS + ['SPY']))
    end_date = datetime.now()
    start_date = end_date - timedelta(days=300)  # Need enough for 200-SMA

    try:
        data = yf.download(
            all_tickers,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False
        )
    except Exception as e:
        log.error(f"Data download failed: {e}")
        return [], 'bull', {}

    if data.empty:
        log.error("No data returned")
        return [], 'bull', {}

    # Get SPY data for regime
    try:
        spy_close = data['Close']['SPY'] if isinstance(data.columns, pd.MultiIndex) else data['Close']
        spy_df = pd.DataFrame({'Close': spy_close}).dropna()
        regime = get_regime(spy_df)
    except Exception:
        regime = 'bull'

    log.info(f"Regime: {regime}")

    # Scan each ticker for sector-relative oversold
    signals = []
    prices = {}

    for ticker in TICKERS:
        try:
            sector_etf = SECTOR_MAP.get(ticker)
            if not sector_etf:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                stock_close = data['Close'][ticker].dropna()
                sector_close = data['Close'][sector_etf].dropna()
            else:
                continue

            if len(stock_close) < LOOKBACK_DAYS + 1 or len(sector_close) < LOOKBACK_DAYS + 1:
                continue

            # Align dates
            common_dates = stock_close.index.intersection(sector_close.index)
            if len(common_dates) < LOOKBACK_DAYS + 1:
                continue

            stock_close = stock_close.loc[common_dates]
            sector_close = sector_close.loc[common_dates]

            # 5-day returns
            stock_ret = (stock_close.iloc[-1] / stock_close.iloc[-LOOKBACK_DAYS-1]) - 1
            sector_ret = (sector_close.iloc[-1] / sector_close.iloc[-LOOKBACK_DAYS-1]) - 1

            # Relative return (idiosyncratic)
            relative_ret = stock_ret - sector_ret

            current_price = float(stock_close.iloc[-1])
            prices[ticker] = current_price

            if relative_ret < DROP_THRESHOLD:
                signals.append({
                    'ticker': ticker,
                    'price': current_price,
                    'stock_5d_ret': round(float(stock_ret) * 100, 2),
                    'sector_5d_ret': round(float(sector_ret) * 100, 2),
                    'relative_ret': round(float(relative_ret) * 100, 2),
                    'sector_etf': sector_etf,
                })
                log.info(f"  SIGNAL: {ticker} relative return {relative_ret*100:.1f}% "
                        f"(stock {stock_ret*100:.1f}%, sector {sector_ret*100:.1f}%)")
        except Exception as e:
            continue

    # Sort by most oversold (most negative relative return)
    signals.sort(key=lambda x: x['relative_ret'])

    return signals, regime, prices


def close_expired_positions(state, prices):
    """Close positions that have been held for HOLD_DAYS."""
    today = datetime.now()
    still_open = []

    for pos in state['positions']:
        entry_date = datetime.fromisoformat(pos['entry_date'])
        days_held = (today - entry_date).days

        # Approximate trading days (weekdays)
        trading_days = sum(1 for d in range(days_held)
                         if (entry_date + timedelta(days=d)).weekday() < 5)

        if trading_days >= HOLD_DAYS:
            # Close position
            current_price = prices.get(pos['ticker'])
            if current_price is None:
                # Try to get price
                try:
                    tick_data = yf.download(pos['ticker'], period='5d', progress=False, auto_adjust=True)
                    if not tick_data.empty:
                        current_price = float(tick_data['Close'].iloc[-1])
                except Exception:
                    still_open.append(pos)
                    continue

            pnl = (current_price - pos['entry_price']) * pos['shares']
            pnl_pct = (current_price / pos['entry_price'] - 1) * 100

            closed = {
                'ticker': pos['ticker'],
                'entry_date': pos['entry_date'],
                'exit_date': today.isoformat(),
                'entry_price': pos['entry_price'],
                'exit_price': round(current_price, 2),
                'shares': pos['shares'],
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'regime': pos['regime'],
                'hold_days': trading_days,
            }
            state['closed_trades'].append(closed)
            state['cash'] += pos['shares'] * current_price

            log.info(f"  CLOSED: {pos['ticker']} | PnL: ${pnl:.2f} ({pnl_pct:+.1f}%) | "
                    f"Held {trading_days}d | {pos['regime']} regime")
        else:
            still_open.append(pos)

    state['positions'] = still_open


def open_new_positions(state, signals, regime, prices):
    """Open positions on new signals if we have slots."""
    available_slots = MAX_POSITIONS - len(state['positions'])
    if available_slots <= 0:
        log.info(f"No slots available ({len(state['positions'])}/{MAX_POSITIONS} positions)")
        return

    # Don't open positions in tickers we already hold
    held_tickers = {p['ticker'] for p in state['positions']}
    new_signals = [s for s in signals if s['ticker'] not in held_tickers]

    if not new_signals:
        log.info("No new signals (already holding all signal tickers)")
        return

    position_size = POSITION_SIZE_FULL if regime == 'bull' else POSITION_SIZE_HALF

    for signal in new_signals[:available_slots]:
        ticker = signal['ticker']
        price = signal['price']
        shares = max(1, int(position_size / price))
        cost = shares * price

        if cost > state['cash']:
            log.info(f"  SKIP {ticker}: insufficient cash (${state['cash']:.2f} < ${cost:.2f})")
            continue

        pos = {
            'ticker': ticker,
            'entry_date': datetime.now().isoformat(),
            'entry_price': round(price, 2),
            'shares': shares,
            'cost': round(cost, 2),
            'regime': regime,
            'signal': signal,
        }
        state['positions'].append(pos)
        state['cash'] -= cost

        log.info(f"  OPENED: {ticker} | {shares} shares @ ${price:.2f} = ${cost:.2f} | "
                f"Relative ret: {signal['relative_ret']}% | {regime} regime ({'' if regime == 'bull' else 'half '}size)")


def compute_nav(state, prices):
    """Compute current NAV."""
    position_value = 0
    for pos in state['positions']:
        current_price = prices.get(pos['ticker'], pos['entry_price'])
        position_value += current_price * pos['shares']
    state['nav'] = round(state['cash'] + position_value, 2)


def main():
    log.info("=" * 60)
    log.info("SECTOR-RELATIVE REVERSAL PAPER ENGINE — SCAN")
    log.info("=" * 60)

    state = load_state()
    log.info(f"Loaded state: {len(state['positions'])} open positions, "
            f"NAV ${state['nav']:.2f}, Cash ${state['cash']:.2f}")

    # Scan for signals
    signals, regime, prices = scan_for_signals()

    # Close expired positions
    close_expired_positions(state, prices)

    # Open new positions
    if signals:
        log.info(f"Found {len(signals)} sector-relative oversold signals")
        open_new_positions(state, signals, regime, prices)
    else:
        log.info("No sector-relative oversold signals today")

    # Update NAV
    compute_nav(state, prices)

    # Summary
    state['last_scan'] = datetime.now().isoformat()
    total_pnl = sum(t['pnl'] for t in state['closed_trades'])
    n_closed = len(state['closed_trades'])
    win_rate = (sum(1 for t in state['closed_trades'] if t['pnl'] > 0) / n_closed * 100) if n_closed > 0 else 0

    log.info(f"\n--- SUMMARY ---")
    log.info(f"NAV: ${state['nav']:.2f} | Cash: ${state['cash']:.2f}")
    log.info(f"Open positions: {len(state['positions'])}/{MAX_POSITIONS}")
    log.info(f"Closed trades: {n_closed} | Total PnL: ${total_pnl:.2f} | WR: {win_rate:.0f}%")
    log.info(f"Regime: {regime}")

    for pos in state['positions']:
        current = prices.get(pos['ticker'], pos['entry_price'])
        unrealized = (current - pos['entry_price']) * pos['shares']
        log.info(f"  {pos['ticker']}: {pos['shares']} shares @ ${pos['entry_price']:.2f} → ${current:.2f} "
                f"(${unrealized:+.2f})")

    save_state(state)
    log.info("State saved. Done.")


if __name__ == '__main__':
    main()
