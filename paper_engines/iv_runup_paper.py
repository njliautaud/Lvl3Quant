#!/usr/bin/env python3
"""
Earnings IV Run-Up Paper Trading Engine
=========================================

Paper trades the pre-earnings IV expansion strategy.
Buy ATM straddles 10-15 trading days before earnings, sell 1-2 days before.
Captures IV expansion without taking earnings event risk.

STRATEGY (Backtest: Sharpe 2.60, WR 77%, MDD -2.4%, $645→$3,761):
  - Signal: Earnings date 10-15 trading days away
  - Filter: Low realized vol (below median for that stock)
  - Entry: ATM straddle, DTE = days_to_earnings + 7
  - Exit: 1-2 days before earnings (sell before the event)
  - Fallback exit: +30% TP, -25% SL, trailing stop
  - Capital: $645 paper (mirrors agentic account)
  - Max $200 per position, max 3 concurrent

USAGE:
  python3 iv_runup_paper.py              # normal daily run
  python3 iv_runup_paper.py --check-now  # force scan
  python3 iv_runup_paper.py --summary    # show current state
  python3 iv_runup_paper.py --dry-run    # simulate only

CRON: Run at 9:30 AM ET weekdays (scan for upcoming earnings)
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── Paths ──
BASE = Path(__file__).resolve().parents[1]
LOG_DIR = Path(__file__).resolve().parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)
STATE_DIR = BASE / 'state'
STATE_DIR.mkdir(exist_ok=True)
STATE_PATH = STATE_DIR / 'iv_runup_paper_state.json'
TRADE_LOG = LOG_DIR / 'iv_runup_trades.jsonl'
LOG_FILE = LOG_DIR / 'iv_runup_paper.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ── Flags ──
DRY_RUN = '--dry-run' in sys.argv
CHECK_NOW = '--check-now' in sys.argv
SUMMARY_ONLY = '--summary' in sys.argv

# ==================== STRATEGY CONFIG ====================

STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH', 'DXCM',
    'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT', 'DASH',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL', 'UPST',
    'AFRM', 'U', 'RKLB', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI',
    # Bank earnings Q3 (added 2026-09-25 — JPM/BAC/GS report Oct 13-14)
    'JPM', 'BAC', 'GS', 'WFC', 'C', 'MS',
]

INITIAL_CAPITAL = 391.0  # updated 2026-09-25 to match actual RH agentic account
MAX_POS_COST = 200.0
MAX_CONCURRENT = 3
COMMISSION_PER_LEG = 0.65   # RH options
COMMISSION_RT = 1.30         # round-trip (open + close, per contract)
RISK_FREE_RATE = 0.05
BS_HAIRCUT = 0.85            # calibrate BS to real market

# Entry window: buy when earnings are 10-15 trading days away
ENTRY_WINDOW_MIN = 10  # earliest entry (trading days before earnings)
ENTRY_WINDOW_MAX = 15  # latest entry (trading days before earnings)

# Exit: sell 1-2 days before earnings
EXIT_DAYS_BEFORE = 1   # sell 1 trading day before earnings

# Safety exits
TP_PCT = 0.30           # +30% take profit
SL_PCT = -0.25          # -25% stop loss
TRAILING_ACTIVATE = 0.15
TRAILING_GIVEBACK = 0.50
MAX_HOLD_DAYS = 20      # absolute max


# ==================== BLACK-SCHOLES ====================

def bs_call(S, K, T, r, sigma):
    if T <= 1e-8: return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 1e-8: return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def straddle_price(S, K, T, r, sigma):
    return (bs_call(S, K, T, r, sigma) + bs_put(S, K, T, r, sigma)) * BS_HAIRCUT

def estimate_iv(realized_vol, days_to_earnings):
    """Estimate implied vol given realized vol and days to earnings."""
    if days_to_earnings <= 0:
        mult = 0.8   # post-earnings crush
    elif days_to_earnings <= 1:
        mult = 1.8
    elif days_to_earnings <= 2:
        mult = 1.6
    elif days_to_earnings <= 5:
        mult = 1.4
    elif days_to_earnings <= 10:
        mult = 1.2
    elif days_to_earnings <= 15:
        mult = 1.1
    else:
        mult = 1.0
    return realized_vol * mult


# ==================== STATE ====================

def _default_state():
    return {
        'config_version': 'iv_runup_v1',
        'equity': INITIAL_CAPITAL,
        'cash': INITIAL_CAPITAL,
        'open_positions': [],
        'closed_trades': [],
        'upcoming_earnings': [],  # cached earnings schedule
        'last_scan': None,
        'total_trades': 0,
        'total_pnl': 0.0,
        'wins': 0,
        'losses': 0,
        'created': datetime.now().isoformat(),
    }

def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            state = json.load(f)
        defaults = _default_state()
        for k, v in defaults.items():
            if k not in state:
                state[k] = v
        return state
    return _default_state()

def save_state(state):
    state['last_scan'] = datetime.now().isoformat()
    if not DRY_RUN:
        with open(STATE_PATH, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        log.info("State saved.")
    else:
        log.info("[DRY RUN] State NOT saved.")

def log_trade(record):
    if not DRY_RUN:
        with open(TRADE_LOG, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')


# ==================== EARNINGS SCANNER ====================

def scan_upcoming_earnings():
    """
    Scan universe for stocks with earnings 10-15 trading days away.
    Returns list of dicts with ticker, earnings_date, trading_days_away.
    """
    import yfinance as yf

    today = datetime.now()
    candidates = []

    for ticker in STOCK_UNIVERSE:
        try:
            stock = yf.Ticker(ticker)
            earnings = stock.get_earnings_dates(limit=5)
            if earnings is None or len(earnings) == 0:
                continue

            for earn_date in earnings.index:
                # Handle timezone
                if hasattr(earn_date, 'tzinfo') and earn_date.tzinfo is not None:
                    edate = pd.Timestamp(earn_date).tz_convert(None)
                else:
                    edate = pd.Timestamp(earn_date)

                # Calendar days away
                cal_days = (edate - pd.Timestamp(today.date())).days

                if cal_days < 0:
                    continue  # past earnings

                # Approximate trading days (cal_days * 5/7)
                trading_days = int(cal_days * 5 / 7)

                if ENTRY_WINDOW_MIN <= trading_days <= ENTRY_WINDOW_MAX:
                    candidates.append({
                        'ticker': ticker,
                        'earnings_date': str(edate.date()),
                        'calendar_days': cal_days,
                        'trading_days_approx': trading_days,
                    })
                    log.info(f"  UPCOMING: {ticker} earnings {edate.date()} "
                             f"(~{trading_days} trading days away)")
                    break  # only next earnings per ticker

        except Exception as e:
            log.debug(f"  {ticker}: earnings check failed ({e})")
            continue

    return candidates


def get_stock_data(ticker, period='1y'):
    """Get price data for a ticker."""
    import yfinance as yf
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except:
        return None


def compute_realized_vol(df, window=21):
    """Compute realized volatility over window."""
    if len(df) < window + 1:
        return 0.3
    rets = np.log(df['close'].iloc[-window-1:] / df['close'].iloc[-window-1:].shift(1)).dropna()
    return float(rets.std() * np.sqrt(252))


def is_low_vol(ticker, df):
    """Check if current vol is below historical median (D variant filter)."""
    if len(df) < 252:
        return True  # not enough data, don't filter

    current_vol = compute_realized_vol(df, 21)

    # Compute rolling 21d vol over last year
    close = df['close'].values
    vols = []
    for i in range(21, min(len(close), 252)):
        rets = np.diff(np.log(close[i-21:i+1]))
        v = float(np.std(rets) * np.sqrt(252)) if len(rets) > 5 else 0.3
        vols.append(v)

    if not vols:
        return True

    median_vol = np.median(vols)
    return current_vol <= median_vol


# ==================== POSITION MANAGEMENT ====================

def check_exits(state):
    """Check open positions for exit conditions."""
    if not state['open_positions']:
        return state

    remaining = []
    today = datetime.now()

    for pos in state['open_positions']:
        ticker = pos['ticker']
        entry_date = datetime.fromisoformat(pos['entry_date'])
        days_held = np.busday_count(entry_date.date(), today.date())
        earnings_date = datetime.strptime(pos['earnings_date'], '%Y-%m-%d')
        days_to_earnings = np.busday_count(today.date(), earnings_date.date())

        # Get current price
        df = get_stock_data(ticker, period='5d')
        if df is None or len(df) == 0:
            remaining.append(pos)
            continue

        current_price = float(df['close'].iloc[-1])

        # Estimate current straddle value
        realized_vol = pos.get('realized_vol', 0.3)
        current_iv = estimate_iv(realized_vol, days_to_earnings)
        remaining_dte = max(pos['dte_at_entry'] - days_held, 1)
        T = remaining_dte / 252.0

        current_straddle = straddle_price(
            current_price, pos['strike'], T, RISK_FREE_RATE, current_iv
        )
        current_value = current_straddle * 100 - COMMISSION_RT
        entry_cost = pos['entry_cost']

        pnl_pct = (current_value / entry_cost - 1) if entry_cost > 0 else 0
        peak_pnl = pos.get('peak_pnl_pct', pnl_pct)
        if pnl_pct > peak_pnl:
            pos['peak_pnl_pct'] = pnl_pct
            peak_pnl = pnl_pct

        exit_reason = None

        # PRIMARY EXIT: sell 1-2 days before earnings
        if days_to_earnings <= EXIT_DAYS_BEFORE:
            exit_reason = 'PRE_EARN'

        # Safety exits
        elif pnl_pct >= TP_PCT:
            exit_reason = 'TP'
        elif pnl_pct <= SL_PCT:
            exit_reason = 'SL'
        elif peak_pnl >= TRAILING_ACTIVATE and pnl_pct <= peak_pnl * (1 - TRAILING_GIVEBACK):
            exit_reason = 'TRAIL'
        elif days_held >= MAX_HOLD_DAYS:
            exit_reason = 'TIME'

        if exit_reason:
            pnl_dollar = current_value - entry_cost

            trade_record = {
                **pos,
                'exit_date': today.isoformat(),
                'exit_stock_price': current_price,
                'exit_straddle': round(current_straddle, 2),
                'exit_value': round(current_value, 2),
                'exit_iv': round(current_iv, 4),
                'pnl_pct': round(pnl_pct * 100, 2),
                'pnl_dollar': round(pnl_dollar, 2),
                'days_held': days_held,
                'days_to_earnings_at_exit': days_to_earnings,
                'exit_reason': exit_reason,
            }

            state['closed_trades'].append(trade_record)
            state['cash'] += entry_cost + pnl_dollar
            state['equity'] += pnl_dollar
            state['total_trades'] += 1
            state['total_pnl'] += pnl_dollar

            if pnl_dollar > 0:
                state['wins'] += 1
            else:
                state['losses'] += 1

            log_trade(trade_record)
            log.info(f"  CLOSED: {ticker} straddle ${pos['strike']} | "
                     f"P&L {pnl_pct*100:+.1f}% (${pnl_dollar:+.2f}) | "
                     f"{days_held}d held | {exit_reason}")
        else:
            pos['current_pnl_pct'] = round(pnl_pct * 100, 2)
            pos['current_stock_price'] = current_price
            pos['days_to_earnings'] = days_to_earnings
            remaining.append(pos)

    state['open_positions'] = remaining
    return state


def open_new_positions(state, candidates):
    """Open positions for qualifying earnings candidates."""
    if len(state['open_positions']) >= MAX_CONCURRENT:
        log.info(f"  Max concurrent ({MAX_CONCURRENT}) reached.")
        return state

    held_tickers = {p['ticker'] for p in state['open_positions']}

    for cand in candidates:
        if len(state['open_positions']) >= MAX_CONCURRENT:
            break

        ticker = cand['ticker']
        if ticker in held_tickers:
            continue

        # Get price data
        df = get_stock_data(ticker)
        if df is None or len(df) < 50:
            continue

        # Low-vol filter (D variant — best risk-adjusted)
        if not is_low_vol(ticker, df):
            log.info(f"  SKIP: {ticker} — vol above median (filter)")
            continue

        current_price = float(df['close'].iloc[-1])
        strike = round(current_price)
        realized_vol = compute_realized_vol(df)
        days_to_earn = cand['trading_days_approx']

        # Price the straddle
        entry_iv = estimate_iv(realized_vol, days_to_earn)
        dte = days_to_earn + 7  # extend past earnings for liquidity
        T = dte / 252.0
        straddle = straddle_price(current_price, strike, T, RISK_FREE_RATE, entry_iv)
        entry_cost = straddle * 100 + COMMISSION_RT

        if entry_cost <= 0 or entry_cost > MAX_POS_COST:
            log.info(f"  SKIP: {ticker} — straddle too expensive (${entry_cost:.0f})")
            continue

        if entry_cost > state['cash']:
            log.info(f"  SKIP: {ticker} — insufficient cash (${state['cash']:.0f} < ${entry_cost:.0f})")
            continue

        # Open position
        pos = {
            'ticker': ticker,
            'option_type': 'straddle',
            'strike': strike,
            'dte_at_entry': dte,
            'entry_date': datetime.now().isoformat(),
            'entry_stock_price': current_price,
            'entry_straddle': round(straddle, 2),
            'entry_cost': round(entry_cost, 2),
            'entry_iv': round(entry_iv, 4),
            'realized_vol': round(realized_vol, 4),
            'earnings_date': cand['earnings_date'],
            'trading_days_to_earnings': days_to_earn,
            'peak_pnl_pct': 0,
            'current_pnl_pct': 0,
        }

        state['open_positions'].append(pos)
        state['cash'] -= entry_cost
        held_tickers.add(ticker)

        log.info(f"  OPENED: {ticker} straddle ${strike} @ ${straddle:.2f} | "
                 f"Earnings {cand['earnings_date']} (~{days_to_earn}d) | "
                 f"Cost: ${entry_cost:.2f} | IV: {entry_iv:.1%}")

    return state


# ==================== SUMMARY ====================

def print_summary(state):
    n_closed = state['total_trades']
    wr = state['wins'] / n_closed * 100 if n_closed > 0 else 0

    print(f"\n{'='*60}")
    print(f"  IV Run-Up Paper Engine — Summary")
    print(f"{'='*60}")
    print(f"  Equity:     ${state['equity']:,.2f} ({(state['equity']/INITIAL_CAPITAL-1)*100:+.1f}%)")
    print(f"  Cash:       ${state['cash']:,.2f}")
    print(f"  Open:       {len(state['open_positions'])}")
    print(f"  Closed:     {n_closed} (W: {state['wins']}, L: {state['losses']}, WR: {wr:.0f}%)")
    print(f"  Total P&L:  ${state['total_pnl']:+,.2f}")
    print(f"  Last scan:  {state.get('last_scan', 'never')}")

    if state['open_positions']:
        print(f"\n  Open Positions:")
        for pos in state['open_positions']:
            days = np.busday_count(
                datetime.fromisoformat(pos['entry_date']).date(),
                datetime.now().date()
            )
            dte_left = pos.get('days_to_earnings', '?')
            print(f"    {pos['ticker']} straddle ${pos['strike']} | "
                  f"Earnings {pos['earnings_date']} ({dte_left}d away) | "
                  f"Day {days} | P&L: {pos.get('current_pnl_pct', 0):+.1f}%")

    if state['closed_trades']:
        recent = state['closed_trades'][-5:]
        print(f"\n  Recent Trades:")
        for t in recent:
            print(f"    {t['ticker']} | "
                  f"P&L: {t['pnl_pct']:+.1f}% (${t['pnl_dollar']:+.2f}) | "
                  f"{t['days_held']}d | {t['exit_reason']}")

    print(f"{'='*60}\n")


# ==================== MAIN ====================

def main():
    log.info("=" * 50)
    log.info("IV Run-Up Paper Engine — Daily Run")
    log.info("=" * 50)

    state = load_state()

    if SUMMARY_ONLY:
        print_summary(state)
        return

    now = datetime.now()
    if not CHECK_NOW and now.weekday() >= 5:
        log.info(f"Weekend ({now.strftime('%A')}), skipping.")
        save_state(state)
        return

    log.info(f"Equity: ${state['equity']:.2f}, Open: {len(state['open_positions'])}, "
             f"Cash: ${state['cash']:.2f}")

    # 1. Check exits
    log.info("Checking exits...")
    state = check_exits(state)

    # 2. Scan for upcoming earnings in the entry window
    log.info("Scanning for upcoming earnings (10-15 trading days out)...")
    candidates = scan_upcoming_earnings()

    if candidates:
        log.info(f"  Found {len(candidates)} candidates in entry window")
        state = open_new_positions(state, candidates)
    else:
        log.info("  No earnings in entry window today")

    # Save
    save_state(state)
    print_summary(state)
    log.info("Done.")


if __name__ == '__main__':
    main()
