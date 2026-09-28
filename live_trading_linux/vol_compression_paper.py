#!/usr/bin/env python3
"""
Vol Compression Breakout — Paper Trading Engine
=================================================
Monitors S&P 500 + sector ETFs for vol compression events.
When breakout detected, enters paper trade in breakout direction.
Holds for 5 trading days, then exits.

Runs as daemon via PM2. Checks every 30 minutes during market hours.
"""

import os, sys, json, time, signal
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/jupiter/Lvl3Quant')

# ─── Config ───
VOL_LOOKBACK = 21          # 1-month realized vol
VOL_HISTORY = 252          # 1-year for percentile
COMPRESSION_PCT = 10       # Bottom 10% = compressed
BREAKOUT_MULT = 1.5        # Breakout = 1.5x avg abs return
HOLD_DAYS = 5              # Hold for 5 trading days
MAX_POSITIONS = 10         # Max concurrent positions
POSITION_SIZE = 10000      # $10K per position (paper)
CHECK_INTERVAL = 1800      # 30 minutes
COST_BPS = 10              # 10bps round trip assumed cost

STATE_FILE = '/home/jupiter/Lvl3Quant/output/vol_compression_paper/state.json'
TRADES_FILE = '/home/jupiter/Lvl3Quant/output/vol_compression_paper/trades.csv'
LOG_FILE = '/home/jupiter/Lvl3Quant/logs/vol_compression_paper.log'

os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Universe: S&P 500 components + sector ETFs (HC #735 R3: 200-500+ stocks)
# Expanded 2026-07-22 from 113 → 287 symbols (all backtested tickers with vol compression events)
UNIVERSE = [
    # Sector ETFs (no survivorship bias)
    'XLF','XLK','XLE','XLV','XLI','XLC','XLY','XLP','XLU','XLRE','XLB',
    'QQQ','IWM','DIA','SMH','XBI','XOP','KRE','XHB','XRT','XME','GDX',
    # High-liquidity stocks (original 91)
    'AAPL','MSFT','AMZN','NVDA','GOOGL','META','TSLA','JPM','V','JNJ',
    'UNH','XOM','PG','HD','CVX','MRK','ABBV','LLY','PEP','KO','COST',
    'AVGO','TMO','MCD','CSCO','ACN','ABT','WFC','DIS','TXN','PM',
    'AMGN','IBM','COP','LOW','GS','CAT','BA','GE','AMD','ISRG',
    'GILD','BKNG','SYK','ADP','VRTX','TJX','PGR','REGN','SCHW','ZTS',
    'CI','EOG','SLB','MO','BSX','CME','SO','BDX','CL','ANET','ITW',
    'ORCL','CRM','NOW','INTU','ADBE','NFLX','NKE','SBUX','LULU',
    'GD','LMT','NOC','PFE','T','VZ','TMUS','CMCSA','F','GM',
    'FCX','OXY','MPC','VLO','EMR','ALL','D','AFL','FDX','HCA',
    # Expanded S&P 500 (174 additional backtested tickers)
    'ABNB','ADC','ADI','AEP','AFRM','AGCO','AIG','ALGN','AMT','AON','APD','APO','ARES','ASH','ASTE',
    'AWK','AXP','AXTA','AZO','BAH','BIIB','BILL','BLK','BMY','BRK-B','CB','CC','CCI','CDNS','CHTR',
    'CLH','CMG','CMS','CNH','CNP','COIN','CRWD','CTAS','CTSH','CUBE','CW','CWST','DASH','DDOG','DE',
    'DG','DHR','DLR','DLTR','DOCU','DUK','DXCM','ECL','ED','EL','EPRT','EQIX','ES','ETR','ETSY',
    'EVRG','EXC','EXP','EXR','FAST','FE','FOX','FOXA','FTNT','GFL','GIS','GLPI','HII','HOOD','HSY',
    'HUBS','HUM','HUN','HWM','ICE','ILMN','INTC','KEYS','KHC','KKR','KMB','LDOS','LHX','LIN','LRCX',
    'LYFT','MA','MAA','MCK','MDB','MDLZ','MLM','MNST','MRNA','MS','MTD','NDAQ','NEE','NEM','NET',
    'NNN','NWSA','O','OKTA','ON','ORLY','OWL','PANW','PATH','PAYX','PINS','PLD','PNC','PPG','PPL',
    'PSA','PSX','PYPL','RH','RL','ROKU','ROP','RPM','RSG','RTX','RYN','SBAC','SHOP','SHW','SNAP',
    'SNOW','SNPS','SOFI','SPG','SPGI','SRE','TEAM','TFC','TGT','TPR','TROW','TTC','TTD','TXT','U',
    'UBER','UDR','UNP','UPS','USB','VEEV','VICI','VMC','W','WAB','WBD','WCN','WDAY','WEC','WELL',
    'WM','WMB','WMT','WPC','WSM','XEL','YUM','ZM','ZS',
]


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {'positions': [], 'closed_trades': [], 'total_pnl': 0.0}


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def is_market_hours():
    now = datetime.now()
    # Market hours: 9:30 AM - 4:00 PM ET (simple check)
    hour = now.hour
    minute = now.minute
    if now.weekday() >= 5:  # Weekend
        return False
    if hour < 9 or (hour == 9 and minute < 30):
        return False
    if hour >= 16:
        return False
    return True


def get_compression_signals():
    """Check universe for vol compression + breakout signals."""
    import yfinance as yf

    signals = []

    # Download recent data (need ~300 days for vol history + lookback)
    log(f"Scanning {len(UNIVERSE)} symbols for vol compression...")

    batch_size = 50
    for i in range(0, len(UNIVERSE), batch_size):
        batch = UNIVERSE[i:i+batch_size]
        try:
            data = yf.download(batch, period='2y', group_by='ticker',
                             progress=False, threads=True)

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        close = data['Close'].dropna()
                    else:
                        close = data[ticker]['Close'].dropna()

                    if len(close) < VOL_HISTORY + VOL_LOOKBACK + 20:
                        continue

                    returns = close.pct_change().dropna()
                    realized_vol = returns.rolling(VOL_LOOKBACK).std() * np.sqrt(252)
                    avg_abs_ret = returns.abs().rolling(VOL_LOOKBACK).mean()

                    # Vol percentile
                    vol_pctile = realized_vol.rolling(VOL_HISTORY).apply(
                        lambda x: (x.iloc[-1] <= x).mean() * 100
                        if len(x) == VOL_HISTORY else np.nan,
                        raw=False
                    )

                    # Check if currently in compression
                    latest_pctile = vol_pctile.iloc[-1]
                    if pd.isna(latest_pctile) or latest_pctile > COMPRESSION_PCT:
                        continue

                    # Check for breakout today or yesterday
                    latest_ret = returns.iloc[-1]
                    prev_ret = returns.iloc[-2] if len(returns) > 1 else 0
                    threshold = avg_abs_ret.iloc[-2] * BREAKOUT_MULT

                    if pd.isna(threshold) or threshold == 0:
                        continue

                    # Today's breakout
                    if abs(latest_ret) > threshold:
                        direction = 'LONG' if latest_ret > 0 else 'SHORT'
                        signals.append({
                            'ticker': ticker,
                            'direction': direction,
                            'breakout_magnitude': abs(latest_ret),
                            'vol_percentile': latest_pctile,
                            'current_price': float(close.iloc[-1]),
                            'signal_date': str(close.index[-1].date())
                        })
                        log(f"  SIGNAL: {ticker} {direction} (vol pctile={latest_pctile:.0f}, "
                            f"breakout={latest_ret*100:.1f}%, threshold={threshold*100:.1f}%)")

                except Exception as e:
                    pass
        except Exception as e:
            log(f"  Download error for batch: {e}")

    log(f"  Found {len(signals)} signals")
    return signals


def check_exits(state):
    """Check if any positions need to be closed (held for HOLD_DAYS)."""
    import yfinance as yf

    today = datetime.now().date()
    to_close = []

    for pos in state['positions']:
        entry_date = datetime.strptime(pos['entry_date'], '%Y-%m-%d').date()
        # Count trading days (approximate: weekdays)
        days_held = np.busday_count(entry_date, today)

        if days_held >= HOLD_DAYS:
            to_close.append(pos)

    if not to_close:
        return state

    # Get current prices
    tickers = [p['ticker'] for p in to_close]
    try:
        data = yf.download(tickers, period='1d', group_by='ticker',
                          progress=False, threads=True)
        prices = {}
        for t in tickers:
            try:
                if len(tickers) == 1:
                    prices[t] = float(data['Close'].iloc[-1])
                else:
                    prices[t] = float(data[t]['Close'].iloc[-1])
            except:
                pass
    except:
        log("ERROR: Could not get exit prices")
        return state

    for pos in to_close:
        ticker = pos['ticker']
        if ticker not in prices:
            continue

        exit_price = prices[ticker]
        entry_price = pos['entry_price']
        direction_mult = 1 if pos['direction'] == 'LONG' else -1
        pnl_pct = direction_mult * (exit_price / entry_price - 1)
        pnl_pct_net = pnl_pct - (COST_BPS / 10000)  # Deduct costs
        pnl_dollars = pos['position_size'] * pnl_pct_net

        trade = {
            **pos,
            'exit_date': str(today),
            'exit_price': exit_price,
            'pnl_pct': round(pnl_pct * 100, 2),
            'pnl_net_pct': round(pnl_pct_net * 100, 2),
            'pnl_dollars': round(pnl_dollars, 2)
        }

        state['closed_trades'].append(trade)
        state['positions'].remove(pos)
        state['total_pnl'] = round(state['total_pnl'] + pnl_dollars, 2)

        result = '✅' if pnl_dollars > 0 else '❌'
        log(f"  EXIT {result} {ticker} {pos['direction']}: "
            f"entry ${entry_price:.2f} → exit ${exit_price:.2f}, "
            f"PnL {pnl_pct_net*100:.1f}% (${pnl_dollars:.0f})")

    return state


def enter_positions(state, signals):
    """Enter new positions from signals."""
    available_slots = MAX_POSITIONS - len(state['positions'])
    if available_slots <= 0:
        log(f"  Portfolio full ({MAX_POSITIONS} positions), skipping {len(signals)} signals")
        return state

    # Sort by breakout magnitude (strongest first)
    signals.sort(key=lambda x: -x['breakout_magnitude'])

    # Don't enter if already have position in same ticker
    existing_tickers = {p['ticker'] for p in state['positions']}

    entered = 0
    for sig in signals:
        if entered >= available_slots:
            break
        if sig['ticker'] in existing_tickers:
            continue

        pos = {
            'ticker': sig['ticker'],
            'direction': sig['direction'],
            'entry_date': sig['signal_date'],
            'entry_price': sig['current_price'],
            'position_size': POSITION_SIZE,
            'breakout_magnitude': round(sig['breakout_magnitude'] * 100, 2),
            'vol_percentile': round(sig['vol_percentile'], 1)
        }
        state['positions'].append(pos)
        existing_tickers.add(sig['ticker'])
        entered += 1

        log(f"  ENTRY: {sig['ticker']} {sig['direction']} @ ${sig['current_price']:.2f} "
            f"(breakout {sig['breakout_magnitude']*100:.1f}%, vol pctile {sig['vol_percentile']:.0f})")

    return state


def print_portfolio(state):
    """Print current portfolio status."""
    log(f"\n  === PORTFOLIO STATUS ===")
    log(f"  Open positions: {len(state['positions'])}/{MAX_POSITIONS}")
    for p in state['positions']:
        log(f"    {p['ticker']} {p['direction']} @ ${p['entry_price']:.2f} "
            f"(entered {p['entry_date']})")

    n_closed = len(state['closed_trades'])
    if n_closed > 0:
        wins = sum(1 for t in state['closed_trades'] if t['pnl_dollars'] > 0)
        wr = wins / n_closed
        log(f"  Closed trades: {n_closed}, WR: {wr:.0%}")
        log(f"  Total PnL: ${state['total_pnl']:.0f}")
    log(f"  {'='*30}\n")


def run_cycle():
    """Run one check cycle."""
    state = load_state()

    # 1. Check exits
    state = check_exits(state)

    # 2. Get new signals
    signals = get_compression_signals()

    # 3. Enter positions
    if signals:
        state = enter_positions(state, signals)

    # 4. Save and report
    print_portfolio(state)
    save_state(state)

    # Append to CSV
    if state['closed_trades']:
        df = pd.DataFrame(state['closed_trades'])
        df.to_csv(TRADES_FILE, index=False)


def main():
    log("=" * 60)
    log("VOL COMPRESSION BREAKOUT — PAPER TRADING ENGINE")
    log(f"Universe: {len(UNIVERSE)} symbols")
    log(f"Max positions: {MAX_POSITIONS}, Hold: {HOLD_DAYS}d")
    log(f"Position size: ${POSITION_SIZE:,}")
    log("=" * 60)

    # Handle graceful shutdown
    def shutdown(signum, frame):
        log("Shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # Run initial cycle
    run_cycle()

    # Then loop
    while True:
        time.sleep(CHECK_INTERVAL)
        if is_market_hours():
            log("--- Market hours check ---")
            run_cycle()
        else:
            # Check once at 9:35 AM for overnight breakouts
            now = datetime.now()
            if now.hour == 9 and 30 <= now.minute <= 40 and now.weekday() < 5:
                log("--- Pre-market check ---")
                run_cycle()


if __name__ == '__main__':
    main()
