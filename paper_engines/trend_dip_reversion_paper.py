#!/usr/bin/env python3
"""
Trend Dip Reversion Paper Engine
==================================
LOCKBOX VALIDATED (AVO score 4.49, lockbox Sharpe 0.95):
- Three-mode dip buying in sector ETFs with regime-adaptive behavior
- Mode A (VIX<21): Buy 2%+ dips in uptrends (above SMA50, trend not falling)
- Mode B (VIX 21-25): Same dip logic in mid-vol with trend confirmation
- Mode C (VIX>25): Capitulation bounce on oversold + big dip
- 1-day underwater cut, 4-day max hold, -2.2% trailing stop, +4% TP
- OOS: Sharpe 2.58 geomean, regime gap 0.006, 138 trades across 2022-2025

Paper engine runs daily after close, checks for signals, manages positions.
"""
import json
import os
import sys
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

BASE = Path(__file__).resolve().parents[1]
STATE_PATH = BASE / 'state' / 'trend_dip_reversion_paper_state.json'
LOG_DIR = Path(__file__).resolve().parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'trend_dip_reversion_paper.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Strategy parameters (from AVO v22, score 4.49)
SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP',
               'XLU', 'XLRE', 'XLB']

DIP_THRESHOLD_NORMAL = 0.020    # 2.0% dip in calm markets
VIX_NORMAL_MAX = 21.0
DIP_THRESHOLD_MIDVOL = 0.020    # 2.0% dip in mid-vol
RSI_REQUIRED_MIDVOL = 45
DIP_THRESHOLD_HIGHVOL = 0.025   # 2.5% dip in high vol
VIX_HIGHVOL_MIN = 25.0
RSI_REQUIRED_HIGHVOL = 35

SECTOR_DIP_MULT = {
    'XLE': 1.0, 'XLK': 1.0, 'XLF': 1.0, 'XLV': 0.90,
    'XLI': 1.0, 'XLC': 1.0, 'XLY': 1.0, 'XLP': 0.85,
    'XLU': 0.85, 'XLRE': 0.90, 'XLB': 1.0,
}

TREND_PERIOD = 50
TREND_SLOPE_DAYS = 3
RSI_PERIOD = 14
MAX_SMA_DISTANCE = 0.08

MAX_PER_TRADE = 1500.0
MAX_CONCURRENT = 2
SLIPPAGE_PCT = 0.0001

TRAILING_STOP_PCT = -0.022
TAKE_PROFIT_PCT = 0.040
MAX_HOLD_DAYS = 4
UNDERWATER_CUT_DAYS = 1

START_CAPITAL = 10000.0


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.clip(lower=1e-10)
    return 100 - (100 / (1 + rs))


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        'capital': START_CAPITAL,
        'positions': [],
        'closed_trades': [],
        'created': datetime.now().isoformat(),
    }


def save_state(state):
    state['updated'] = datetime.now().isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def get_market_data():
    """Download recent data for signal generation."""
    try:
        import yfinance as yf
        tickers = SECTOR_ETFS + ['SPY', '^VIX']
        data = yf.download(tickers, period='120d', progress=False)
        close = data['Close']
        return close
    except Exception as e:
        log.error(f"Data download failed: {e}")
        return None


def check_signals(close):
    """Check for entry signals today."""
    if close is None or len(close) < TREND_PERIOD + 5:
        return []

    vix = close['^VIX'].iloc[-1] if '^VIX' in close.columns else 15.0
    is_normal = vix < VIX_NORMAL_MAX
    is_mid = VIX_NORMAL_MAX <= vix < VIX_HIGHVOL_MIN
    is_high = vix >= VIX_HIGHVOL_MIN

    signals = []
    for etf in SECTOR_ETFS:
        if etf not in close.columns:
            continue

        price = close[etf]
        daily_ret = price.pct_change().iloc[-1]

        # Trend filter
        sma = price.rolling(TREND_PERIOD).mean()
        above_trend = price.iloc[-1] > sma.iloc[-1]
        sma_slope = sma.iloc[-1] - sma.iloc[-TREND_SLOPE_DAYS - 1]
        trend_not_falling = sma_slope >= 0

        # SMA distance
        sma_dist = (price.iloc[-1] - sma.iloc[-1]) / sma.iloc[-1]
        near_support = sma_dist < MAX_SMA_DISTANCE

        # RSI
        rsi = compute_rsi(price, RSI_PERIOD).iloc[-1]

        # Sector-specific dip threshold
        mult = SECTOR_DIP_MULT.get(etf, 1.0)

        mode = None
        if is_normal and daily_ret < -(DIP_THRESHOLD_NORMAL * mult) and above_trend and trend_not_falling and near_support:
            mode = 'A'
        elif is_mid and daily_ret < -(DIP_THRESHOLD_MIDVOL * mult) and above_trend and trend_not_falling and near_support:
            mode = 'B'
        elif is_high and daily_ret < -(DIP_THRESHOLD_HIGHVOL * mult) and rsi < RSI_REQUIRED_HIGHVOL:
            mode = 'C'

        if mode:
            signals.append({
                'ticker': etf,
                'mode': mode,
                'dip_pct': daily_ret * 100,
                'vix': vix,
                'rsi': rsi,
                'price': float(price.iloc[-1]),
            })

    return signals


def check_exits(state, close):
    """Check exit conditions for open positions."""
    exits = []
    today = datetime.now().date()

    for i, pos in enumerate(state['positions']):
        ticker = pos['ticker']
        if ticker not in close.columns:
            continue

        current_price = float(close[ticker].iloc[-1])
        entry_price = pos['entry_price']
        high_water = pos.get('high_water', entry_price)

        # Update high water mark
        if current_price > high_water:
            state['positions'][i]['high_water'] = current_price
            high_water = current_price

        entry_date = datetime.fromisoformat(pos['entry_date']).date()
        days_held = np.busday_count(np.datetime64(entry_date), np.datetime64(today))
        pnl_pct = (current_price - entry_price) / entry_price

        reason = None
        if days_held >= MAX_HOLD_DAYS:
            reason = 'max_hold'
        elif pnl_pct >= TAKE_PROFIT_PCT:
            reason = 'take_profit'
        elif high_water > 0:
            dd_from_high = (current_price - high_water) / high_water
            if dd_from_high <= TRAILING_STOP_PCT:
                reason = 'trailing_stop'
        if days_held >= UNDERWATER_CUT_DAYS and pnl_pct < -0.010:
            reason = 'underwater_cut'

        if reason:
            exits.append({
                'index': i,
                'ticker': ticker,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl_pct': pnl_pct,
                'days_held': int(days_held),
                'reason': reason,
                'mode': pos.get('mode', '?'),
            })

    return exits


def run():
    """Main paper engine loop."""
    log.info("=" * 60)
    log.info("Trend Dip Reversion Paper Engine — running")

    state = load_state()
    close = get_market_data()
    if close is None:
        log.error("No market data, skipping")
        save_state(state)
        return state

    # Process exits first
    exits = check_exits(state, close)
    for ex in sorted(exits, key=lambda x: x['index'], reverse=True):
        pos = state['positions'].pop(ex['index'])
        trade = {
            'ticker': ex['ticker'],
            'entry_date': pos['entry_date'],
            'entry_price': ex['entry_price'],
            'exit_date': datetime.now().isoformat(),
            'exit_price': ex['exit_price'],
            'pnl_pct': ex['pnl_pct'],
            'pnl_dollar': ex['pnl_pct'] * pos['size'],
            'days_held': ex['days_held'],
            'reason': ex['reason'],
            'mode': ex['mode'],
        }
        state['closed_trades'].append(trade)
        state['capital'] += pos['size'] + trade['pnl_dollar']
        log.info(f"EXIT {ex['ticker']} mode={ex['mode']} after {ex['days_held']}d: "
                 f"{ex['pnl_pct']*100:+.2f}% (${trade['pnl_dollar']:+.2f}) [{ex['reason']}]")

    # Check for new signals
    n_open = len(state['positions'])
    if n_open < MAX_CONCURRENT:
        signals = check_signals(close)
        open_tickers = {p['ticker'] for p in state['positions']}

        for sig in signals:
            if n_open >= MAX_CONCURRENT:
                break
            if sig['ticker'] in open_tickers:
                continue

            size = min(MAX_PER_TRADE, state['capital'] * 0.3)
            if size < 50:
                log.warning(f"Insufficient capital for {sig['ticker']}")
                continue

            state['positions'].append({
                'ticker': sig['ticker'],
                'entry_date': datetime.now().isoformat(),
                'entry_price': sig['price'],
                'size': size,
                'mode': sig['mode'],
                'high_water': sig['price'],
                'vix_at_entry': sig['vix'],
            })
            state['capital'] -= size
            n_open += 1
            log.info(f"ENTRY {sig['ticker']} mode={sig['mode']} @ ${sig['price']:.2f} "
                     f"(dip={sig['dip_pct']:.1f}%, VIX={sig['vix']:.1f}, RSI={sig['rsi']:.0f}) "
                     f"size=${size:.0f}")

    # Summary
    n_trades = len(state['closed_trades'])
    if n_trades > 0:
        wins = sum(1 for t in state['closed_trades'] if t['pnl_pct'] > 0)
        total_pnl = sum(t['pnl_dollar'] for t in state['closed_trades'])
        log.info(f"Summary: {n_trades} trades, {wins}/{n_trades} wins ({wins/n_trades*100:.0f}% WR), "
                 f"total P&L: ${total_pnl:+.2f}, capital: ${state['capital']:.2f}, "
                 f"open: {len(state['positions'])}")
    else:
        log.info(f"No closed trades yet. Open: {len(state['positions'])}, capital: ${state['capital']:.2f}")

    save_state(state)

    # Write signal state for aggregator
    signal_state = {
        'strategy': 'trend_dip_reversion',
        'timestamp': datetime.now().isoformat(),
        'positions': len(state['positions']),
        'signals': [],
        'capital': state['capital'],
    }
    for pos in state['positions']:
        signal_state['signals'].append({
            'ticker': pos['ticker'],
            'direction': 'long',
            'mode': pos['mode'],
            'confidence': 0.70,
        })
    signal_path = BASE / 'state' / 'trend_dip_reversion_signals.json'
    signal_path.write_text(json.dumps(signal_state, indent=2, default=str))

    return state


if __name__ == '__main__':
    run()
