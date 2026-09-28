#!/usr/bin/env python3
"""
Contrarian Sector Reversion Paper Engine
=========================================
VALIDATED STRATEGY (adversarial-tested):
- When mega-cap stocks gap DOWN >3%, buy sector ETF shares
- Hold 3 trading days
- Mean drift: +0.252%, WR 55.9%, t=7.17, works in bear markets
- Edge over random: +0.156% per trade
- Non-trigger down days are -1.03% = mega-cap trigger is special

Uses fractional shares on RH (zero commission).
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
STATE_PATH = BASE / 'state' / 'contrarian_reversion_paper_state.json'
LOG_DIR = Path(__file__).resolve().parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'contrarian_reversion_paper.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Strategy parameters
MEGA_CAPS = {
    'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'GOOG': 'XLC',
    'META': 'XLC', 'AMZN': 'XLY', 'TSLA': 'XLY', 'JPM': 'XLF',
    'UNH': 'XLV', 'JNJ': 'XLV', 'XOM': 'XLE', 'CVX': 'XLE',
}
GAP_THRESHOLD = -0.03  # mega-cap drops >3%
HOLD_DAYS = 3
SIZE_PCT = 0.30  # 30% of capital per trade
START_CAPITAL = 645.0

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

def check_triggers():
    """Check if any mega-caps gapped down >3% today."""
    try:
        import yfinance as yf
        tickers = list(MEGA_CAPS.keys())
        data = yf.download(tickers, period='5d', progress=False)
        close = data['Close']
        
        if len(close) < 2:
            return []
        
        today_ret = close.iloc[-1] / close.iloc[-2] - 1
        triggers = []
        for ticker, sector in MEGA_CAPS.items():
            if ticker in today_ret.index:
                ret = float(today_ret[ticker])
                if not np.isnan(ret) and ret < GAP_THRESHOLD:
                    triggers.append({
                        'ticker': ticker,
                        'sector': sector,
                        'gap': round(ret * 100, 2),
                        'sector_price': None,  # will be filled from quote
                    })
        return triggers
    except Exception as e:
        log.error(f"Trigger check failed: {e}")
        return []

def check_exits(state):
    """Check if any positions should be closed (held >= HOLD_DAYS)."""
    today = datetime.now()
    to_close = []
    remaining = []
    
    for pos in state['positions']:
        entry_date = datetime.fromisoformat(pos['entry_date'])
        # Count business days
        bdays = np.busday_count(entry_date.date(), today.date())
        if bdays >= HOLD_DAYS:
            to_close.append(pos)
        else:
            remaining.append(pos)
    
    return to_close, remaining

def main():
    log.info("=" * 50)
    log.info("Contrarian Sector Reversion — Daily Check")
    
    state = load_state()
    log.info(f"Capital: ${state['capital']:.2f} | Open: {len(state['positions'])} | "
             f"Closed: {len(state['closed_trades'])}")
    
    # 1. Check exits
    to_close, remaining = check_exits(state)
    for pos in to_close:
        try:
            import yfinance as yf
            data = yf.download(pos['sector'], period='1d', progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(1, axis=1)
            exit_price = float(data['Close'].iloc[-1])
            pnl = pos['shares'] * (exit_price - pos['entry_price'])
            state['capital'] += pos['invested'] + pnl
            state['closed_trades'].append({
                'sector': pos['sector'],
                'trigger': pos['trigger_ticker'],
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl / pos['invested'] * 100, 2),
                'entry_date': pos['entry_date'],
                'exit_date': datetime.now().isoformat(),
            })
            log.info(f"CLOSED {pos['sector']} (triggered by {pos['trigger_ticker']}): "
                     f"${pos['entry_price']:.2f}→${exit_price:.2f} | P&L: ${pnl:.2f} ({pnl/pos['invested']*100:.1f}%)")
        except Exception as e:
            log.error(f"Exit failed for {pos['sector']}: {e}")
            remaining.append(pos)  # keep position if exit fails
    
    state['positions'] = remaining
    
    # 2. Check triggers
    triggers = check_triggers()
    if triggers:
        trigger_strs = [f"{t['ticker']} ({t['gap']:.1f}%)->{t['sector']}" for t in triggers]
        log.info(f"TRIGGERS FOUND: {', '.join(trigger_strs)}")
        
        sectors_already_held = {p['sector'] for p in state['positions']}
        
        for trigger in triggers:
            sector = trigger['sector']
            if sector in sectors_already_held:
                log.info(f"  Skip {sector} — already holding")
                continue
            
            try:
                import yfinance as yf
                data = yf.download(sector, period='1d', progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.droplevel(1, axis=1)
                price = float(data['Close'].iloc[-1])
                
                invest = min(state['capital'] * SIZE_PCT, state['capital'] - 50)
                if invest < 20:
                    log.info(f"  Skip {sector} — insufficient capital")
                    continue
                
                shares = invest / price
                state['capital'] -= invest
                state['positions'].append({
                    'sector': sector,
                    'trigger_ticker': trigger['ticker'],
                    'trigger_gap': trigger['gap'],
                    'entry_price': price,
                    'shares': shares,
                    'invested': invest,
                    'entry_date': datetime.now().isoformat(),
                })
                log.info(f"  OPENED {sector} @ ${price:.2f} | ${invest:.0f} ({shares:.4f} shares) "
                         f"| triggered by {trigger['ticker']} {trigger['gap']:.1f}%")
            except Exception as e:
                log.error(f"  Entry failed for {sector}: {e}")
    else:
        log.info("No triggers today")
    
    # 3. Summary
    total_invested = sum(p['invested'] for p in state['positions'])
    total_equity = state['capital'] + total_invested
    
    if state['closed_trades']:
        total_pnl = sum(t['pnl'] for t in state['closed_trades'])
        wr = len([t for t in state['closed_trades'] if t['pnl'] > 0]) / len(state['closed_trades']) * 100
        log.info(f"SUMMARY: Equity ${total_equity:.2f} | Cash ${state['capital']:.2f} | "
                 f"Invested ${total_invested:.2f} | Total P&L ${total_pnl:.2f} | "
                 f"WR {wr:.0f}% ({len(state['closed_trades'])} trades)")
    else:
        log.info(f"SUMMARY: Equity ${total_equity:.2f} | Cash ${state['capital']:.2f} | "
                 f"Invested ${total_invested:.2f} | No closed trades yet")
    
    save_state(state)
    log.info("State saved")

if __name__ == '__main__':
    main()
