#!/usr/bin/env python3
"""
Extreme Idiosyncratic Movers Paper Trading Engine (Variant C — Trend Filter)
=============================================================================

Buy S&P 500 stocks with extreme idiosyncratic moves (>5% vs sector ETF over
5 days) in EITHER direction, but ONLY if above their 50-SMA (trend filter).
Hold 10 trading days. Half-size when SPY < 200-SMA.

STRATEGY (Passed 5/5 gates, adversarial pending):
  Sharpe 1.042, 558 trades, regime gap 0.005, perm p=0.0
  2026-H1 Sharpe +0.95

Usage:
  python3 paper_engines/extreme_idio_paper.py
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
        logging.FileHandler(LOG_DIR / 'extreme_idio_paper.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

STATE_DIR = Path(__file__).resolve().parent.parent / 'state'
STATE_DIR.mkdir(exist_ok=True)
STATE_PATH = STATE_DIR / 'extreme_idio_paper_state.json'

# Strategy parameters
POSITION_SIZE_FULL = 130    # $130 per position (full size, ~5 positions)
POSITION_SIZE_HALF = 65     # $65 in bear regime
MAX_POSITIONS = 5
HOLD_DAYS = 10
MOVE_THRESHOLD = 0.05       # 5% relative move threshold
LOOKBACK_DAYS = 5
SMA_50 = 50                 # Trend filter: above 50-SMA
SMA_200 = 200               # Regime: bull/bear

# Universe
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
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def scan_for_signals():
    """Scan all tickers for extreme idiosyncratic moves + trend filter."""
    log.info("Downloading price data...")
    all_tickers = list(set(TICKERS + SECTOR_ETFS + ['SPY']))
    end_date = datetime.now()
    start_date = end_date - timedelta(days=300)

    try:
        data = yf.download(all_tickers, start=start_date.strftime('%Y-%m-%d'),
                           end=end_date.strftime('%Y-%m-%d'),
                           auto_adjust=True, progress=False)
    except Exception as e:
        log.error(f"Data download failed: {e}")
        return [], 'bull', {}

    if data.empty:
        log.error("No data returned")
        return [], 'bull', {}

    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data

    # Regime
    try:
        spy_close = close['SPY'].dropna()
        sma200 = spy_close.rolling(SMA_200).mean().iloc[-1]
        regime = 'bull' if float(spy_close.iloc[-1]) > float(sma200) else 'bear'
    except:
        regime = 'bull'

    log.info(f"Regime: {regime}")

    signals = []
    prices = {}

    for ticker in TICKERS:
        try:
            sector_etf = SECTOR_MAP.get(ticker)
            if not sector_etf or ticker not in close.columns or sector_etf not in close.columns:
                continue

            stock_close = close[ticker].dropna()
            sector_close = close[sector_etf].dropna()

            if len(stock_close) < max(LOOKBACK_DAYS + 1, SMA_50):
                continue

            common = stock_close.index.intersection(sector_close.index)
            if len(common) < max(LOOKBACK_DAYS + 1, SMA_50):
                continue

            stock_close = stock_close.loc[common]
            sector_close = sector_close.loc[common]

            # 5-day returns
            stock_ret = float(stock_close.iloc[-1] / stock_close.iloc[-LOOKBACK_DAYS-1] - 1)
            sector_ret = float(sector_close.iloc[-1] / sector_close.iloc[-LOOKBACK_DAYS-1] - 1)
            relative_ret = stock_ret - sector_ret

            # Trend filter: stock above 50-SMA
            sma50 = float(stock_close.rolling(SMA_50).mean().iloc[-1])
            current_price = float(stock_close.iloc[-1])
            above_50sma = current_price > sma50

            prices[ticker] = current_price

            # Signal: extreme idiosyncratic move + trend filter
            if abs(relative_ret) > MOVE_THRESHOLD and above_50sma:
                direction = 'UP' if relative_ret > 0 else 'DOWN'
                signals.append({
                    'ticker': ticker,
                    'price': current_price,
                    'stock_5d_ret': round(stock_ret * 100, 2),
                    'sector_5d_ret': round(sector_ret * 100, 2),
                    'relative_ret': round(relative_ret * 100, 2),
                    'abs_relative_ret': round(abs(relative_ret) * 100, 2),
                    'direction': direction,
                    'sector_etf': sector_etf,
                    'sma50': round(sma50, 2),
                    'above_50sma': above_50sma,
                })
                log.info(f"  SIGNAL: {ticker} {direction} {abs(relative_ret)*100:.1f}% vs {sector_etf} "
                        f"(stock {stock_ret*100:.1f}%, sector {sector_ret*100:.1f}%) "
                        f"above 50-SMA: ${current_price:.2f} > ${sma50:.2f}")
        except Exception:
            continue

    # Sort by magnitude of relative move (most extreme first)
    signals.sort(key=lambda x: -x['abs_relative_ret'])

    return signals, regime, prices


def close_expired_positions(state, prices):
    """Close positions held >= HOLD_DAYS."""
    today = datetime.now()
    still_open = []

    for pos in state['positions']:
        entry_date = datetime.fromisoformat(pos['entry_date'])
        days_held = (today - entry_date).days
        trading_days = sum(1 for d in range(days_held)
                         if (entry_date + timedelta(days=d)).weekday() < 5)

        if trading_days >= HOLD_DAYS:
            current_price = prices.get(pos['ticker'])
            if current_price is None:
                try:
                    tick_data = yf.download(pos['ticker'], period='5d', progress=False, auto_adjust=True)
                    if not tick_data.empty:
                        current_price = float(tick_data['Close'].iloc[-1])
                except:
                    still_open.append(pos)
                    continue

            pnl = (current_price - pos['entry_price']) * pos['shares']
            pnl_pct = (current_price / pos['entry_price'] - 1) * 100

            state['closed_trades'].append({
                'ticker': pos['ticker'],
                'entry_date': pos['entry_date'],
                'exit_date': today.isoformat(),
                'entry_price': pos['entry_price'],
                'exit_price': round(current_price, 2),
                'shares': pos['shares'],
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'regime': pos['regime'],
                'direction': pos.get('direction', '?'),
                'hold_days': trading_days,
            })
            state['cash'] += pos['shares'] * current_price
            log.info(f"  CLOSED: {pos['ticker']} {pos.get('direction', '?')} | "
                    f"PnL: ${pnl:.2f} ({pnl_pct:+.1f}%) | Held {trading_days}d")
        else:
            still_open.append(pos)

    state['positions'] = still_open


def open_new_positions(state, signals, regime, prices):
    """Open positions on new signals."""
    available = MAX_POSITIONS - len(state['positions'])
    if available <= 0:
        log.info(f"No slots available ({len(state['positions'])}/{MAX_POSITIONS})")
        return

    held_tickers = {p['ticker'] for p in state['positions']}
    new_signals = [s for s in signals if s['ticker'] not in held_tickers]

    if not new_signals:
        log.info("No new signals after filtering held tickers")
        return

    pos_size = POSITION_SIZE_FULL if regime == 'bull' else POSITION_SIZE_HALF

    for signal in new_signals[:available]:
        ticker = signal['ticker']
        price = signal['price']
        shares = max(1, int(pos_size / price))
        cost = shares * price

        if cost > state['cash']:
            log.info(f"  SKIP {ticker}: insufficient cash (${state['cash']:.2f} < ${cost:.2f})")
            continue

        state['positions'].append({
            'ticker': ticker,
            'entry_date': datetime.now().isoformat(),
            'entry_price': round(price, 2),
            'shares': shares,
            'cost': round(cost, 2),
            'regime': regime,
            'direction': signal['direction'],
            'relative_ret': signal['relative_ret'],
        })
        state['cash'] -= cost
        log.info(f"  OPENED: {ticker} {signal['direction']} | {shares} sh @ ${price:.2f} = ${cost:.2f} | "
                f"Relative: {signal['relative_ret']}% | {regime} regime")


def compute_nav(state, prices):
    position_value = sum(
        prices.get(p['ticker'], p['entry_price']) * p['shares']
        for p in state['positions']
    )
    state['nav'] = round(state['cash'] + position_value, 2)


def main():
    log.info("=" * 60)
    log.info("EXTREME IDIO MOVERS PAPER ENGINE (Variant C) — SCAN")
    log.info("=" * 60)

    state = load_state()
    log.info(f"State: {len(state['positions'])} positions, NAV ${state['nav']:.2f}")

    signals, regime, prices = scan_for_signals()

    close_expired_positions(state, prices)

    if signals:
        log.info(f"Found {len(signals)} extreme idiosyncratic signals")
        open_new_positions(state, signals, regime, prices)
    else:
        log.info("No extreme idiosyncratic signals today")

    compute_nav(state, prices)

    state['last_scan'] = datetime.now().isoformat()
    total_pnl = sum(t['pnl'] for t in state['closed_trades'])
    n_closed = len(state['closed_trades'])
    wr = (sum(1 for t in state['closed_trades'] if t['pnl'] > 0) / n_closed * 100) if n_closed else 0

    log.info(f"\n--- SUMMARY ---")
    log.info(f"NAV: ${state['nav']:.2f} | Cash: ${state['cash']:.2f}")
    log.info(f"Positions: {len(state['positions'])}/{MAX_POSITIONS} | Regime: {regime}")
    log.info(f"Closed: {n_closed} | PnL: ${total_pnl:.2f} | WR: {wr:.0f}%")

    for pos in state['positions']:
        curr = prices.get(pos['ticker'], pos['entry_price'])
        unr = (curr - pos['entry_price']) * pos['shares']
        log.info(f"  {pos['ticker']} {pos.get('direction','?')}: {pos['shares']} sh @ "
                f"${pos['entry_price']:.2f} → ${curr:.2f} (${unr:+.2f})")

    save_state(state)
    log.info("Done.")


if __name__ == '__main__':
    main()
