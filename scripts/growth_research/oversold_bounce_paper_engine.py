#!/usr/bin/env python3
"""
Oversold Bounce — Paper Trading Engine
========================================
Monitors 369 stocks for single-day 5%+ drops (the passing v2 config).
When triggered, enters a paper long position and holds for 10 trading days.

Runs every 30min during market hours via PM2.
Logs all trades to JSON for tracking.

Config: drop_1d_5pct_hold10d
  - Entry: stock drops >5% in a single day
  - Action: buy at next-day open (simulated)
  - Exit: close after 10 trading days
  - Max positions: 20 concurrent
  - Sizing: 2% of portfolio per trade
"""

import os, json, time, sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_FILE = ROOT / "data" / "oversold_bounce_paper_state.json"
LOG_FILE = ROOT / "logs" / "oversold_bounce_paper.log"

# Config
STARTING_CAPITAL = 100_000
MAX_POSITIONS = 20
RISK_PER_TRADE_PCT = 0.02  # 2% per trade
DROP_THRESHOLD = -0.05     # 5% single-day drop
HOLD_DAYS = 10             # trading days
SLIPPAGE_PCT = 0.001       # 0.1% per side

# Universe — same 369 stocks from v2 backtest (top ~400 US equities)
UNIVERSE = [
    'AAPL','MSFT','GOOGL','GOOG','AMZN','META','NVDA','TSLA','AVGO','ORCL',
    'CRM','AMD','ADBE','ACN','CSCO','INTC','IBM','TXN','QCOM','NOW',
    'INTU','AMAT','ADI','LRCX','KLAC','SNPS','CDNS','MRVL','FTNT','PANW',
    'CRWD','WDAY','TEAM','ZS','DDOG','HUBS','NET','MDB','SNOW','PLTR',
    'ABNB','DASH','COIN','PYPL','SHOP','U','RBLX','PINS','SNAP',
    'JPM','BAC','WFC','GS','MS','C','BLK','SCHW','AXP','BK',
    'USB','PNC','TFC','COF','CME','ICE','MCO','SPGI','MSCI','FIS',
    'FISV','ADP','NDAQ','MMC','AON','CB','AFL','MET','PRU','ALL',
    'TRV','AIG','HIG','GL','BRO','WRB','CINF','RJF','NTRS',
    'UNH','JNJ','LLY','PFE','MRK','ABBV','ABT','TMO','DHR','BMY',
    'AMGN','MDT','ISRG','ELV','SYK','GILD','VRTX','REGN','BSX','ZBH',
    'BDX','IQV','A','DXCM','IDXX','PODD','ALGN','HOLX','MTD','WAT',
    'HD','MCD','NKE','LOW','SBUX','TJX','BKNG','CMG','MAR','HLT',
    'ORLY','AZO','ROST','DG','DLTR','BBY','POOL','DHI','LEN','PHM',
    'NVR','GPC','GRMN','EBAY','ETSY','LULU','DECK','ON','TPR','RL',
    'PG','KO','PEP','COST','WMT','PM','MO','CL','KMB','GIS',
    'K','SJM','HSY','MNST','STZ','TSN','HRL','CPB','CAG','MKC',
    'CHD','CLX','EL','KHC','KDP','MDLZ','SYY','KR','TGT','WBA',
    'HON','UNP','UPS','CAT','RTX','DE','BA','LMT','GD','NOC',
    'GE','MMM','EMR','ROK','ITW','PH','ETN','IR','CARR','OTIS',
    'AME','DOV','NDSN','SWK','XYL','GNRC','TT','WAB','CSX','NSC',
    'FDX','DAL','UAL','LUV','AAL','JBHT','CHRW','EXPD','ODFL','SAIA',
    'XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO','OXY','DVN',
    'HAL','BKR','FANG','TRGP','WMB',
    'LIN','APD','ECL','SHW','DD','NEM','FCX','NUE','STLD','CF',
    'VMC','MLM','ALB','PPG','DOW','IP','PKG','AVY','EMN',
    'NEE','DUK','SO','D','AEP','SRE','EXC','XEL','WEC',
    'ED','AEE','DTE','CMS','CNP','PNW','EVRG','NI','ATO','OGE',
    'PLD','AMT','CCI','EQIX','PSA','O','SPG','DLR','WELL','AVB',
    'EQR','VTR','IRM','ARE','MAA','UDR','KIM','REG','CPT','HST',
    'DIS','NFLX','CMCSA','CHTR','TMUS','VZ','T','FOX','FOXA','OMC',
    'MTCH','LYV','WBD','EA','TTWO','YELP',
    'RIVN','LCID','SOFI','HOOD','DKNG','PENN','MGM','CZR','WYNN','LVS',
    'MELI','SE','GRAB','BABA','JD','PDD','BIDU','NIO','LI','XPEV',
    'SPOT','ROKU','ZM','DOCU','OKTA','TWLO','PATH','BILL','FOUR','GTLB',
    'RRC','AR','EQT','CHRD','SM','MTDR','MGY',
    'CLF','AA','VALE','RIO','BHP','SCCO','TECK','WPM','GOLD',
    'MOS','FMC','CTVA','AGCO','CNH','FSLR','ENPH','SEDG',
    'RUN','JKS','CSIQ','ARRY','CWEN','AES','ORA',
]


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'capital': STARTING_CAPITAL,
        'open_positions': [],
        'closed_trades': [],
        'created': datetime.now().isoformat(),
        'total_pnl': 0.0,
        'n_wins': 0,
        'n_losses': 0,
    }


def save_state(state):
    state['last_updated'] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def get_quotes(symbols):
    """Get current quotes using yfinance."""
    import yfinance as yf

    quotes = {}
    # Batch download for efficiency
    batch_size = 100
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            data = yf.download(batch, period='5d', progress=False, auto_adjust=True, threads=True)
            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                for sym in batch:
                    try:
                        close_col = data['Close']
                        if sym in close_col.columns:
                            prices = close_col[sym].dropna()
                            if len(prices) >= 2:
                                quotes[sym] = {
                                    'close': float(prices.iloc[-1]),
                                    'prev_close': float(prices.iloc[-2]),
                                    'change_pct': float((prices.iloc[-1] / prices.iloc[-2]) - 1),
                                    'date': str(prices.index[-1].date()),
                                }
                    except:
                        pass
            else:
                # Single ticker
                close = data['Close'].dropna()
                if len(close) >= 2:
                    sym = batch[0]
                    quotes[sym] = {
                        'close': float(close.iloc[-1]),
                        'prev_close': float(close.iloc[-2]),
                        'change_pct': float((close.iloc[-1] / close.iloc[-2]) - 1),
                        'date': str(close.index[-1].date()),
                    }
        except Exception as e:
            log(f"  Quote batch error: {e}")
        time.sleep(0.5)

    return quotes


def check_exits(state, quotes):
    """Close positions that have reached hold period."""
    today = datetime.now().date()
    still_open = []
    newly_closed = []

    for pos in state['open_positions']:
        exit_date = datetime.strptime(pos['exit_target_date'], '%Y-%m-%d').date()

        if today >= exit_date:
            sym = pos['symbol']
            if sym in quotes:
                exit_price = quotes[sym]['close'] * (1 - SLIPPAGE_PCT)
                entry_price = pos['entry_price']
                ret = (exit_price - entry_price) / entry_price
                pnl = pos['size'] * ret

                trade = {
                    **pos,
                    'exit_price': exit_price,
                    'exit_date': str(today),
                    'return_pct': ret * 100,
                    'pnl': pnl,
                    'status': 'CLOSED',
                }
                state['closed_trades'].append(trade)
                state['capital'] += pos['size'] + pnl
                state['total_pnl'] += pnl
                if ret > 0:
                    state['n_wins'] += 1
                else:
                    state['n_losses'] += 1

                newly_closed.append(trade)
                log(f"  EXIT {sym}: {ret*100:+.2f}% (${pnl:+.0f})")
            else:
                still_open.append(pos)  # keep if no quote
        else:
            still_open.append(pos)

    state['open_positions'] = still_open
    return newly_closed


def check_entries(state, quotes):
    """Look for new 5%+ drops to enter."""
    new_entries = []

    # Skip if at max positions
    if len(state['open_positions']) >= MAX_POSITIONS:
        log(f"  At max positions ({MAX_POSITIONS}), skipping entry scan")
        return new_entries

    # Symbols already in open positions
    open_syms = {p['symbol'] for p in state['open_positions']}

    for sym, q in quotes.items():
        if sym in open_syms:
            continue

        if q['change_pct'] <= DROP_THRESHOLD:
            # Signal fired! Enter at simulated next-day open (use current close as proxy)
            entry_price = q['close'] * (1 + SLIPPAGE_PCT)
            size = state['capital'] * RISK_PER_TRADE_PCT

            if size < 100 or state['capital'] < size:
                continue

            # Calculate exit target date (10 trading days from now)
            exit_date = datetime.now().date()
            trading_days_added = 0
            while trading_days_added < HOLD_DAYS:
                exit_date += timedelta(days=1)
                if exit_date.weekday() < 5:  # skip weekends
                    trading_days_added += 1

            pos = {
                'symbol': sym,
                'entry_price': entry_price,
                'entry_date': str(datetime.now().date()),
                'exit_target_date': str(exit_date),
                'size': size,
                'drop_pct': q['change_pct'] * 100,
                'status': 'OPEN',
            }

            state['open_positions'].append(pos)
            state['capital'] -= size
            new_entries.append(pos)
            log(f"  ENTRY {sym}: dropped {q['change_pct']*100:.1f}%, bought at ${entry_price:.2f}, size ${size:.0f}")

            if len(state['open_positions']) >= MAX_POSITIONS:
                break

    return new_entries


def run_scan():
    """Main scan cycle."""
    log("=" * 60)
    log("OVERSOLD BOUNCE PAPER ENGINE — SCAN CYCLE")
    log("=" * 60)

    state = load_state()
    n_open = len(state['open_positions'])
    n_closed = len(state['closed_trades'])
    log(f"State: {n_open} open, {n_closed} closed, capital ${state['capital']:.0f}, total P&L ${state['total_pnl']:.0f}")

    # Get quotes
    log(f"Fetching quotes for {len(UNIVERSE)} stocks...")
    quotes = get_quotes(UNIVERSE)
    log(f"  Got {len(quotes)} quotes")

    # Check exits first
    closed = check_exits(state, quotes)
    if closed:
        log(f"  Closed {len(closed)} positions")

    # Check entries
    entries = check_entries(state, quotes)
    if entries:
        log(f"  Opened {len(entries)} new positions")

    # Summary
    n_open = len(state['open_positions'])
    wr = state['n_wins'] / max(state['n_wins'] + state['n_losses'], 1) * 100
    log(f"After scan: {n_open} open, capital ${state['capital']:.0f}, "
        f"total P&L ${state['total_pnl']:.0f}, WR {wr:.0f}% ({state['n_wins']}W/{state['n_losses']}L)")

    if state['open_positions']:
        log("  Open positions:")
        for p in state['open_positions']:
            sym = p['symbol']
            if sym in quotes:
                cur = quotes[sym]['close']
                ret = (cur - p['entry_price']) / p['entry_price'] * 100
                log(f"    {sym}: entry ${p['entry_price']:.2f} → ${cur:.2f} ({ret:+.1f}%), exit {p['exit_target_date']}")

    save_state(state)
    log("Scan complete.\n")


if __name__ == '__main__':
    run_scan()
