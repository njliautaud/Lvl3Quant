#!/usr/bin/env python3
"""
Strangle-Selling Paper Trading Engine
======================================

Sells OTM strangles (short OTM put + short OTM call) on a diversified
70-ticker equity universe. First premium-selling strategy to pass
regime-agnostic validation (HC #428 R1).

Strategy (winning config STR_d25_35dte_ba10):
  - Sell 25-delta OTM put + 25-delta OTM call, ~35 DTE
  - Profit take: 50% of net credit received
  - Stop loss: 200% of net credit received (2x premium)
  - Roll at 7 DTE if no exit trigger hit
  - VIX filter: half size when VIX > 30, no entries when VIX > 40
  - Max 15 concurrent positions, max 3% NAV risk per position
  - Max 3 positions per sector (diversification)
  - 24-hour ticker cooldown (HC #701)

Backtest results (27-month walk-forward OOT):
  - Sharpe 4.55, Sortino 13.36, CAGR 76.9%, MaxDD -4.4%
  - R1 regime gap 0.331 (PASS, threshold 0.50)
  - 100% monthly win rate, permutation p=0.00

Cost model:
  - $0.65/contract/leg ($1.30 per strangle open, $1.30 close)
  - 2.5% slippage on premium (min $0.03/share)

Author: Claude (2026-07-15, post strangle backtest validation)
"""

import json
import math
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

# ── HC #702 — Options pricing audit logger ──
try:
    from live_trading_linux.options_pricing_logger import log_option_price
    _HAS_PRICING_LOGGER = True
except ImportError:
    try:
        from options_pricing_logger import log_option_price
        _HAS_PRICING_LOGGER = True
    except ImportError:
        _HAS_PRICING_LOGGER = False

# ── Alpaca real pricing bridge (falls back to BS when market closed) ──
try:
    try:
        from live_trading_linux.alpaca_pricing_bridge import (
            get_premium as _alpaca_get_premium, init_bridge as _init_bridge,
            log_session_summary as _log_pricing_summary,
        )
    except ImportError:
        from alpaca_pricing_bridge import (
            get_premium as _alpaca_get_premium, init_bridge as _init_bridge,
            log_session_summary as _log_pricing_summary,
        )
    _HAS_PRICING_BRIDGE = True
except Exception:
    _HAS_PRICING_BRIDGE = False

# ── Paths ──
ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "live_trading_linux" / "strangle_paper_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
EQUITY_FILE = STATE_DIR / "equity.csv"
TRADES_FILE = STATE_DIR / "trades.jsonl"
NAV_HISTORY_FILE = STATE_DIR / "nav_history.json"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [STRANGLE] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "strangle_paper.log")),
    ],
)
log = logging.getLogger('STRANGLE')

# ══════════════════════════════════════════════════════════════════════
# STRATEGY CONFIGURATION (STR_d25_35dte_ba10 winning config)
# ══════════════════════════════════════════════════════════════════════

STARTING_CAPITAL = 100_000.0
POLL_INTERVAL_SEC = 300  # 5 minutes during market hours

# Strangle parameters
PUT_DELTA_TARGET = 0.25    # 25-delta OTM put
CALL_DELTA_TARGET = 0.25   # 25-delta OTM call
DTE_TARGET = 35            # 35 DTE target
DTE_MIN = 28
DTE_MAX = 42
PROFIT_TAKE_PCT = 0.50     # Close when 50% of premium captured
STOP_LOSS_MULT = 2.0       # Close when loss = 200% of premium received
ROLL_DTE_THRESHOLD = 7     # Roll to next month at 7 DTE
MIN_NET_CREDIT = 0.20      # Min $0.20/share net credit per strangle

# Position sizing
MAX_RISK_PER_POSITION = 0.03  # 3% NAV risk per position
MAX_CONCURRENT = 15            # Max 15 concurrent strangles
MAX_PER_SECTOR = 3             # Max 3 positions per sector (diversification)

# VIX filter
VIX_HALF_SIZE = 30.0       # Reduce size by 50% when VIX > 30
VIX_NO_ENTRY = 40.0        # No new entries when VIX > 40

# Equity curve brake
BRAKE_LOOKBACK_DAYS = 60
BRAKE_THRESHOLD = 0.03     # 3% DD from peak triggers brake
BRAKE_SCALE = 0.25         # Scale exposure to 25% when braking

# Drawdown trigger (fast halt)
DD_TRIGGER_LOOKBACK = 3
DD_TRIGGER_THRESHOLD = -0.05
DD_TRIGGER_ENABLED = True

# Pricing
RISK_FREE = 0.04
COST_PER_CONTRACT = 0.65   # Per contract per leg
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03

# Price filter
MIN_PRICE = 10.0
MAX_PRICE = 500.0

# Volatility filter
SIGMA_MAX_ENTRY = 0.80     # Skip tickers with 20d annualized vol > 80%

# Ticker cooldown (HC #701)
TICKER_COOLDOWN_HOURS = 24

# Persistent loser blacklist
LOSER_BLACKLIST = {"CRSP", "AAL", "CELH"}

# ══════════════════════════════════════════════════════════════════════
# 70-TICKER UNIVERSE (diversified across sectors)
# ══════════════════════════════════════════════════════════════════════

UNIVERSE = {
    # Technology (10)
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
    'AMD': 'Technology', 'INTC': 'Technology', 'TXN': 'Technology',
    'IBM': 'Technology', 'CSCO': 'Technology', 'QCOM': 'Technology',
    'AVGO': 'Technology',
    # Healthcare (8)
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
    'ABBV': 'Healthcare', 'MRK': 'Healthcare', 'GILD': 'Healthcare',
    'CVS': 'Healthcare', 'ABT': 'Healthcare',
    # Financial Services (8)
    'JPM': 'Financial Services', 'BAC': 'Financial Services',
    'GS': 'Financial Services', 'MS': 'Financial Services',
    'AXP': 'Financial Services', 'V': 'Financial Services',
    'MA': 'Financial Services', 'C': 'Financial Services',
    # Consumer Cyclical (7)
    'AMZN': 'Consumer Cyclical', 'TSLA': 'Consumer Cyclical',
    'HD': 'Consumer Cyclical', 'NKE': 'Consumer Cyclical',
    'SBUX': 'Consumer Cyclical', 'WYNN': 'Consumer Cyclical',
    'TGT': 'Consumer Cyclical',
    # Consumer Defensive (5)
    'PG': 'Consumer Defensive', 'KO': 'Consumer Defensive',
    'PEP': 'Consumer Defensive', 'CL': 'Consumer Defensive',
    'WMT': 'Consumer Defensive',
    # Energy (5)
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
    'VLO': 'Energy', 'SLB': 'Energy',
    # Industrials (6)
    'CAT': 'Industrials', 'HON': 'Industrials', 'UNP': 'Industrials',
    'BA': 'Industrials', 'DE': 'Industrials', 'GE': 'Industrials',
    # Communication Services (5)
    'META': 'Communication Services', 'GOOG': 'Communication Services',
    'DIS': 'Communication Services', 'EA': 'Communication Services',
    'VZ': 'Communication Services',
    # Utilities (4)
    'EXC': 'Utilities', 'AEP': 'Utilities', 'DUK': 'Utilities',
    'NEE': 'Utilities',
    # Real Estate (4)
    'DLR': 'Real Estate', 'IRM': 'Real Estate', 'SPG': 'Real Estate',
    'AMT': 'Real Estate',
    # Basic Materials (4)
    'LIN': 'Basic Materials', 'APD': 'Basic Materials',
    'FCX': 'Basic Materials', 'NEM': 'Basic Materials',
    # Misc/Other (4)
    'TMUS': 'Communication Services', 'NFLX': 'Consumer Cyclical',
    'COST': 'Consumer Defensive', 'MCD': 'Consumer Cyclical',
}

# ══════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES PRICING
# ══════════════════════════════════════════════════════════════════════

def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, T, sigma, r=RISK_FREE, kind="put"):
    """Black-Scholes European option price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)
    return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)


def _get_option_price_consistent(ticker, S, K, T, sigma, expiry_date, kind='put'):
    """Get option price using SAME source as entry (Alpaca if available, else BS).
    HC #709 audit fix: entry used Alpaca pricing (IV 0.35-0.47) but close used BS (IV 0.21-0.28),
    creating instant phantom profit on every position. Now both use the same source."""
    if _HAS_PRICING_BRIDGE:
        try:
            price = _alpaca_get_premium(
                ticker, S, K, T, sigma, expiry_date,
                kind=kind, bs_price_fn=bs_price)
            if price and price > 0:
                return price
        except Exception:
            pass
    return bs_price(S, K, T, sigma, kind=kind)


def find_strike(S, sigma, T, delta_target, kind="put"):
    """Binary search for strike at target absolute delta."""
    if T <= 0 or sigma <= 0:
        return S
    if kind == "put":
        lo, hi = S * 0.3, S * 1.0
    else:
        lo, hi = S * 1.0, S * 2.0
    for _ in range(60):
        K = (lo + hi) / 2
        d1 = (math.log(S / K) + (RISK_FREE + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T) + 1e-9)
        if kind == "put":
            delta_abs = _Phi(-d1)
        else:
            delta_abs = _Phi(d1)
        if delta_abs > delta_target:
            if kind == "put":
                hi = K
            else:
                lo = K
        else:
            if kind == "put":
                lo = K
            else:
                hi = K
    return round(K * 2) / 2  # Round to nearest $0.50


def apply_slippage(premium):
    """Reduce credit received by slippage."""
    return max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)


# ══════════════════════════════════════════════════════════════════════
# EARNINGS FILTER
# ══════════════════════════════════════════════════════════════════════

def load_earnings_dates():
    """Load cached earnings dates for the universe."""
    cache_path = ROOT / "wheel_strategy_v1" / "data" / "cache" / "earnings_dates.parquet"
    if not cache_path.exists():
        log.warning("No cached earnings dates found")
        return {}

    df = pd.read_parquet(cache_path)
    lookup = {}
    date_col = "earnings_date" if "earnings_date" in df.columns else "date"
    for ticker, grp in df.groupby("ticker"):
        dates = pd.to_datetime(grp[date_col]).values
        dates = np.sort(dates)
        lookup[ticker] = dates

    log.info(f"Earnings dates loaded for {len(lookup)} tickers")
    return lookup


def has_earnings_soon(ticker, earnings_lookup, check_date=None, dte=DTE_TARGET, buffer=2):
    """Check if ticker has earnings within [today-buffer, today+DTE+buffer]."""
    if ticker not in earnings_lookup:
        return False

    if check_date is None:
        check_date = pd.Timestamp.now()
    else:
        check_date = pd.Timestamp(check_date)

    ed = earnings_lookup[ticker]
    window_start = np.datetime64(check_date) - np.timedelta64(buffer, 'D')
    window_end = np.datetime64(check_date) + np.timedelta64(dte + buffer, 'D')

    idx_s = np.searchsorted(ed, window_start, side='left')
    idx_e = np.searchsorted(ed, window_end, side='right')

    return idx_e > idx_s


# ══════════════════════════════════════════════════════════════════════
# PRICE DATA
# ══════════════════════════════════════════════════════════════════════

def get_live_prices(tickers, batch_size=50):
    """Fetch current prices and 20d realized vol via yfinance."""
    import yfinance as yf
    prices = {}
    sigmas = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, period="60d", auto_adjust=True,
                               threads=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    if t in data["Close"].columns:
                        close = data["Close"][t].dropna()
                        if len(close) > 0:
                            prices[t] = float(close.iloc[-1])
                            log_ret = np.log(close / close.shift(1)).dropna()
                            if len(log_ret) >= 20:
                                sigmas[t] = float(log_ret.tail(20).std() * np.sqrt(252))
                            elif len(log_ret) > 5:
                                sigmas[t] = float(log_ret.std() * np.sqrt(252))
                            sigmas[t] = max(0.05, min(sigmas.get(t, 0.3), 2.0))
            elif len(batch) == 1 and not data.empty:
                t = batch[0]
                close = data["Close"].dropna()
                if len(close) > 0:
                    prices[t] = float(close.iloc[-1])
                    log_ret = np.log(close / close.shift(1)).dropna()
                    sigmas[t] = float(log_ret.tail(20).std() * np.sqrt(252)) if len(log_ret) >= 20 else 0.3
                    sigmas[t] = max(0.05, min(sigmas[t], 2.0))
        except Exception as e:
            log.warning(f"Price batch {i}-{i + batch_size} failed: {e}")
        time.sleep(0.5)

    return prices, sigmas


def get_vix():
    """Fetch current VIX level."""
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        hist = vix.history(period='5d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    return 20.0


# ══════════════════════════════════════════════════════════════════════
# EQUITY CURVE BRAKE
# ══════════════════════════════════════════════════════════════════════

def load_nav_history():
    if NAV_HISTORY_FILE.exists():
        with open(NAV_HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_nav_history(history):
    with open(NAV_HISTORY_FILE, 'w') as f:
        json.dump(history, f)


def compute_brake_scale(nav_history, current_nav):
    """Equity curve brake: scale down if NAV below (1-threshold) of 60d peak."""
    if len(nav_history) < 5:
        return 1.0
    recent = nav_history[-BRAKE_LOOKBACK_DAYS:]
    peak = max(entry["nav"] for entry in recent)
    if current_nav < peak * (1 - BRAKE_THRESHOLD):
        log.info(f"EQUITY BRAKE ACTIVE: NAV ${current_nav:,.0f} < peak ${peak:,.0f} x "
                 f"{1 - BRAKE_THRESHOLD:.0%} = ${peak * (1 - BRAKE_THRESHOLD):,.0f}. "
                 f"Scale -> {BRAKE_SCALE:.0%}")
        return BRAKE_SCALE
    return 1.0


def check_dd_trigger(nav_history):
    """Fast drawdown trigger: halt new entries if 3-day return < -5%."""
    if not DD_TRIGGER_ENABLED or len(nav_history) < DD_TRIGGER_LOOKBACK + 1:
        return False
    recent = nav_history[-(DD_TRIGGER_LOOKBACK + 1):]
    nav_start = recent[0]["nav"]
    nav_end = recent[-1]["nav"]
    if nav_start <= 0:
        return False
    trail_return = (nav_end - nav_start) / nav_start
    if trail_return < DD_TRIGGER_THRESHOLD:
        log.info(f"DD TRIGGER ACTIVE: {DD_TRIGGER_LOOKBACK}d return = "
                 f"{trail_return * 100:.1f}% < {DD_TRIGGER_THRESHOLD * 100:.0f}% threshold. "
                 f"Halting new entries.")
        return True
    return False


# ══════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'cash': STARTING_CAPITAL,
        'positions': [],
        'closed_trades': [],
        'realized_pnl': 0.0,
        'trade_count': 0,
        'start_date': datetime.utcnow().strftime('%Y-%m-%d'),
        'last_check': None,
        'version': 'strangle_v1',
    }


def save_state(state):
    # Keep closed_trades bounded (last 500)
    if len(state.get('closed_trades', [])) > 500:
        state['closed_trades'] = state['closed_trades'][-500:]
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def compute_nav(state, prices, sigmas):
    """Compute NAV = cash + mark-to-market of open strangles.

    Uses BS with entry sigma for MTM stability. The Alpaca API is intermittent
    and causes NAV to oscillate $20-30K between cycles when it returns prices
    on some calls but not others. BS with entry sigma is deterministic.
    Alpaca prices are still used for ENTRY and CLOSE decisions via
    _get_option_price_consistent.
    """
    nav = state['cash']

    for pos in state['positions']:
        ticker = pos['ticker']
        if ticker not in prices:
            continue

        S = prices[ticker]
        # Use entry sigma for stable MTM (not live sigmas which may differ)
        sigma = pos.get('sigma_entry', sigmas.get(ticker, 0.20))
        expiry = pd.Timestamp(pos['expiry'])
        T = max((expiry - pd.Timestamp.now()).days, 0) / 365.0
        contracts = pos.get('contracts', 1)

        # MTM: always BS with entry sigma for stability
        put_val = bs_price(S, pos['short_put_strike'], T, sigma, kind='put')
        call_val = bs_price(S, pos['short_call_strike'], T, sigma, kind='call')

        # We owe the strangle value (short = liability)
        close_cost = (put_val + call_val) * 100 * contracts
        nav -= close_cost

    return nav


def log_equity(nav, realized_pnl):
    """Append to equity CSV."""
    row = f"{datetime.utcnow().isoformat()},{nav:.2f},{realized_pnl:.2f}\n"
    if not EQUITY_FILE.exists():
        with open(EQUITY_FILE, 'w') as f:
            f.write("timestamp,nav,realized_pnl\n")
    with open(EQUITY_FILE, 'a') as f:
        f.write(row)


def log_trade(info):
    """Append trade to JSONL log."""
    with open(TRADES_FILE, 'a') as f:
        f.write(json.dumps(info, default=str) + '\n')


# ══════════════════════════════════════════════════════════════════════
# NAV DROP ALERT
# ══════════════════════════════════════════════════════════════════════

SOD_NAV_FILE = STATE_DIR / "sod_nav.json"
NAV_DROP_ALERT_PCT = 0.025
_nav_drop_alerted_today = set()


def check_nav_drop(nav: float) -> None:
    """Alert if NAV drops >2.5% from start-of-day."""
    today = datetime.utcnow().strftime("%Y-%m-%d")

    sod_data = {}
    if SOD_NAV_FILE.exists():
        try:
            sod_data = json.loads(SOD_NAV_FILE.read_text())
        except Exception:
            pass

    if sod_data.get("date") != today:
        sod_data = {"date": today, "sod_nav": round(nav, 2)}
        SOD_NAV_FILE.write_text(json.dumps(sod_data))
        _nav_drop_alerted_today.clear()
        return

    sod_nav = sod_data["sod_nav"]
    if sod_nav <= 0:
        return

    drop_pct = (sod_nav - nav) / sod_nav
    if drop_pct >= NAV_DROP_ALERT_PCT and today not in _nav_drop_alerted_today:
        _nav_drop_alerted_today.add(today)
        log.warning(f"NAV DROP ALERT: {drop_pct:.1%} intraday "
                    f"(SOD: ${sod_nav:,.0f} -> ${nav:,.0f})")
        try:
            os.system(
                f'node /home/jupiter/teleclaude-main/utils/webhook_notifier.js '
                f'"STRANGLE NAV DROP: {drop_pct:.1%} intraday '
                f'(${sod_nav:,.0f} -> ${nav:,.0f})" 2>/dev/null'
            )
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# EXPIRY LOGIC (monthly — ~35 DTE)
# ══════════════════════════════════════════════════════════════════════

def find_expiry(from_date=None):
    """Find the nearest Friday within DTE_MIN..DTE_MAX of target DTE."""
    if from_date is None:
        from_date = datetime.utcnow()
    best, best_dist = None, 10000
    for d_off in range(DTE_MIN, DTE_MAX + 1):
        cand = from_date + timedelta(days=d_off)
        # Find nearest Friday (weekday 4)
        shift = (4 - cand.weekday()) % 7
        cand_fri = cand + timedelta(days=shift)
        dte = (cand_fri - from_date).days
        if dte < DTE_MIN or dte > DTE_MAX:
            continue
        dist = abs(dte - DTE_TARGET)
        if dist < best_dist:
            best, best_dist = cand_fri, dist
    return best


# ══════════════════════════════════════════════════════════════════════
# POSITION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

def process_positions(state, prices, sigmas):
    """Check existing strangles for profit-take, stop-loss, roll, or expiry."""
    now = pd.Timestamp.now()
    new_positions = []

    # Market hours check for entries/exits
    try:
        import pytz
        et = pytz.timezone('US/Eastern')
        now_et = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(et)
        in_rth = ((now_et.hour == 9 and now_et.minute >= 30) or
                  (10 <= now_et.hour < 16))
    except Exception:
        in_rth = True  # Fail-open

    for pos in state['positions']:
        ticker = pos['ticker']
        if ticker not in prices:
            new_positions.append(pos)
            continue

        S = prices[ticker]
        sigma = sigmas.get(ticker, pos.get('sigma_entry', 0.20))
        expiry = pd.Timestamp(pos['expiry'])
        T_days = (expiry - now).days
        T = max(T_days, 0) / 365.0
        contracts = pos.get('contracts', 1)
        net_credit = pos['net_credit']  # per-share credit received
        total_credit = net_credit * 100 * contracts

        # FIX (HC #709 audit): Use consistent pricing — same source as entry
        expiry_date = expiry.date() if hasattr(expiry, 'date') else expiry
        put_val = _get_option_price_consistent(ticker, S, pos['short_put_strike'], T, sigma, expiry_date, kind='put')
        call_val = _get_option_price_consistent(ticker, S, pos['short_call_strike'], T, sigma, expiry_date, kind='call')
        close_cost_per_share = put_val + call_val
        close_cost = close_cost_per_share * 100 * contracts
        close_fees = 2 * COST_PER_CONTRACT * contracts  # 2 legs to close

        # ── Profit/Loss calculation ──
        # P&L = credit received - cost to close - close fees
        pnl = total_credit - close_cost - close_fees
        profit_pct = pnl / total_credit if total_credit > 0 else 0

        # FIX (HC #709 audit): Anti-churn cooldown — don't close within 24h of open
        entry_dt = pd.Timestamp(pos.get('entry_date', '2020-01-01'))
        hours_held = (now - entry_dt).total_seconds() / 3600
        if hours_held < 24:
            new_positions.append(pos)
            continue

        # ── Profit take (50% of premium captured) ──
        if profit_pct >= PROFIT_TAKE_PCT and in_rth:
            state['cash'] -= (close_cost + close_fees)
            state['realized_pnl'] += pnl
            state['trade_count'] += 1
            log.info(f"PROFIT TAKE {ticker} strangle @ {profit_pct:.0%}. "
                     f"P:{pos['short_put_strike']:.0f} C:{pos['short_call_strike']:.0f} "
                     f"PnL: ${pnl:.2f} ({contracts}c)")
            state['closed_trades'].append({
                'ticker': ticker, 'action': 'profit_take',
                'pnl': round(pnl, 2), 'profit_pct': round(profit_pct, 3),
                'contracts': contracts,
                'close_date': datetime.utcnow().isoformat(),
            })
            log_trade({
                'action': 'profit_take', 'ticker': ticker,
                'short_put': pos['short_put_strike'],
                'short_call': pos['short_call_strike'],
                'pnl': round(pnl, 2), 'profit_pct': round(profit_pct, 3),
                'contracts': contracts, 'time': now.isoformat(),
            })
            continue

        # ── Stop loss (200% of premium = loss exceeds 2x credit) ──
        if profit_pct <= -STOP_LOSS_MULT and in_rth:
            state['cash'] -= (close_cost + close_fees)
            state['realized_pnl'] += pnl
            state['trade_count'] += 1
            log.info(f"STOP LOSS {ticker} strangle @ {profit_pct:.0%} "
                     f"(loss >= {STOP_LOSS_MULT:.0f}x premium). "
                     f"P:{pos['short_put_strike']:.0f} C:{pos['short_call_strike']:.0f} "
                     f"PnL: ${pnl:.2f}")
            state['closed_trades'].append({
                'ticker': ticker, 'action': 'stop_loss',
                'pnl': round(pnl, 2), 'profit_pct': round(profit_pct, 3),
                'contracts': contracts,
                'close_date': datetime.utcnow().isoformat(),
            })
            log_trade({
                'action': 'stop_loss', 'ticker': ticker,
                'short_put': pos['short_put_strike'],
                'short_call': pos['short_call_strike'],
                'pnl': round(pnl, 2), 'profit_pct': round(profit_pct, 3),
                'contracts': contracts, 'time': now.isoformat(),
            })
            continue

        # ── Roll at 7 DTE (if no exit trigger hit) ──
        if T_days <= ROLL_DTE_THRESHOLD and in_rth:
            # Close current position
            state['cash'] -= (close_cost + close_fees)
            state['realized_pnl'] += pnl
            state['trade_count'] += 1
            log.info(f"ROLL CLOSE {ticker} strangle @ {T_days}d DTE. "
                     f"PnL on old leg: ${pnl:.2f}")
            state['closed_trades'].append({
                'ticker': ticker, 'action': 'roll_close',
                'pnl': round(pnl, 2), 'contracts': contracts,
                'close_date': datetime.utcnow().isoformat(),
            })
            log_trade({
                'action': 'roll_close', 'ticker': ticker,
                'short_put': pos['short_put_strike'],
                'short_call': pos['short_call_strike'],
                'pnl': round(pnl, 2), 'contracts': contracts,
                'time': now.isoformat(),
            })

            # Open new strangle at next expiry
            new_expiry = find_expiry()
            if new_expiry is not None:
                T_new = (new_expiry - datetime.utcnow()).days / 365.0
                K_put = find_strike(S, sigma, T_new, PUT_DELTA_TARGET, kind='put')
                K_call = find_strike(S, sigma, T_new, CALL_DELTA_TARGET, kind='call')

                _exp_d = new_expiry.date() if hasattr(new_expiry, 'date') else new_expiry
                if _HAS_PRICING_BRIDGE:
                    prem_put = _alpaca_get_premium(ticker, S, K_put, T_new, sigma, _exp_d, kind='put', bs_price_fn=bs_price)
                    prem_call = _alpaca_get_premium(ticker, S, K_call, T_new, sigma, _exp_d, kind='call', bs_price_fn=bs_price)
                else:
                    prem_put = bs_price(S, K_put, T_new, sigma, kind='put')
                    prem_call = bs_price(S, K_call, T_new, sigma, kind='call')

                prem_put_slip = apply_slippage(prem_put)
                prem_call_slip = apply_slippage(prem_call)
                new_credit_per_share = prem_put_slip + prem_call_slip
                open_fees = 2 * COST_PER_CONTRACT * contracts

                if new_credit_per_share >= MIN_NET_CREDIT:
                    total_new_credit = new_credit_per_share * 100 * contracts - open_fees
                    state['cash'] += total_new_credit

                    new_dte = (new_expiry - datetime.utcnow()).days
                    new_positions.append({
                        'ticker': ticker,
                        'entry_date': datetime.utcnow().isoformat(),
                        'expiry': new_expiry.isoformat(),
                        'short_put_strike': K_put,
                        'short_call_strike': K_call,
                        'net_credit': new_credit_per_share,
                        'contracts': contracts,
                        'spot_entry': round(S, 2),
                        'sigma_entry': round(sigma, 4),
                        'profit_take_price': round(new_credit_per_share * (1 - PROFIT_TAKE_PCT), 4),
                        'stop_loss_price': round(new_credit_per_share * (1 + STOP_LOSS_MULT), 4),
                    })
                    log.info(f"ROLL OPEN {ticker} strangle P:{K_put:.0f} C:{K_call:.0f} "
                             f"({new_dte}d), credit ${total_new_credit:.2f}")
                    log_trade({
                        'action': 'roll_open', 'ticker': ticker,
                        'short_put': K_put, 'short_call': K_call,
                        'net_credit': round(new_credit_per_share, 4),
                        'contracts': contracts, 'dte': new_dte,
                        'time': datetime.utcnow().isoformat(),
                    })

                    # HC #702 pricing log
                    if _HAS_PRICING_LOGGER:
                        for _k, _kind, _bs_val in [(K_put, 'put', prem_put), (K_call, 'call', prem_call)]:
                            log_option_price(
                                engine="strangle", ticker=ticker, spot=S, strike=_k,
                                expiry=str(_exp_d)[:10], option_type=_kind, iv_used=sigma,
                                bs_price=_bs_val, source="bs_or_alpaca", action="roll"
                            )
            continue

        # ── Expiry settlement ──
        if now >= expiry:
            # Calculate intrinsic value at expiry
            put_intrinsic = max(pos['short_put_strike'] - S, 0) * 100 * contracts
            call_intrinsic = max(S - pos['short_call_strike'], 0) * 100 * contracts
            total_loss = put_intrinsic + call_intrinsic
            pnl_expiry = total_credit - total_loss - close_fees

            state['cash'] -= (total_loss + close_fees)
            state['realized_pnl'] += pnl_expiry
            state['trade_count'] += 1

            side = "OTM" if total_loss == 0 else ("PUT" if put_intrinsic > 0 and call_intrinsic == 0
                    else ("CALL" if call_intrinsic > 0 and put_intrinsic == 0 else "BOTH"))
            log.info(f"EXPIRED {ticker} strangle ({side} ITM). "
                     f"P:{pos['short_put_strike']:.0f} C:{pos['short_call_strike']:.0f} "
                     f"S=${S:.2f} PnL: ${pnl_expiry:.2f}")
            state['closed_trades'].append({
                'ticker': ticker, 'action': f'expired_{side.lower()}',
                'pnl': round(pnl_expiry, 2), 'contracts': contracts,
                'close_date': datetime.utcnow().isoformat(),
            })
            log_trade({
                'action': f'expired_{side.lower()}', 'ticker': ticker,
                'short_put': pos['short_put_strike'],
                'short_call': pos['short_call_strike'],
                'pnl': round(pnl_expiry, 2),
                'put_intrinsic': round(put_intrinsic, 2),
                'call_intrinsic': round(call_intrinsic, 2),
                'underlying_at_expiry': round(S, 2),
                'contracts': contracts, 'time': now.isoformat(),
            })
            continue

        # Position survives
        new_positions.append(pos)

    state['positions'] = new_positions


# ══════════════════════════════════════════════════════════════════════
# NEW POSITION ENTRY
# ══════════════════════════════════════════════════════════════════════

def open_new_positions(state, prices, sigmas, vix, earnings_lookup,
                       brake_scale, dd_trigger_active=False):
    """Open new strangle positions on tickers without existing positions."""
    # ── VIX gate ──
    if vix > VIX_NO_ENTRY:
        log.info(f"VIX {vix:.1f} > {VIX_NO_ENTRY} -- no new entries")
        return
    if brake_scale <= 0:
        log.info("EQUITY BRAKE HALT -- no new strangles")
        return
    if dd_trigger_active:
        log.info("DD TRIGGER HALT -- 3-day trailing return below threshold, no new entries")
        return

    # Market hours guard (9:30-16:00 ET)
    try:
        import pytz
        et = pytz.timezone('US/Eastern')
        now_et = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(et)
        if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 30) or now_et.hour >= 16:
            log.info(f"OFF-HOURS ({now_et.strftime('%H:%M ET')}) -- no new entries")
            return
    except Exception:
        pass

    # VIX-based size reduction
    vix_size_mult = 0.50 if vix > VIX_HALF_SIZE else 1.0
    if vix_size_mult < 1.0:
        log.info(f"VIX STRESS: {vix:.1f} > {VIX_HALF_SIZE} -> position size cut to 50%")

    # Active position checks
    active_tickers = {pos['ticker'] for pos in state['positions']}
    if len(state['positions']) >= MAX_CONCURRENT:
        return

    # Sector counts for diversification
    sector_counts = {}
    for pos in state['positions']:
        ticker = pos['ticker']
        sector = UNIVERSE.get(ticker, 'Unknown')
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    # Ticker cooldown (HC #701)
    now_utc = datetime.utcnow()
    cooldown_tickers = set()
    for ct in state.get('closed_trades', []):
        close_str = ct.get('close_date', '')
        if close_str:
            try:
                close_dt = datetime.fromisoformat(close_str.replace('Z', '+00:00').replace('+00:00', ''))
                hours_since = (now_utc - close_dt).total_seconds() / 3600
                if hours_since < TICKER_COOLDOWN_HOURS:
                    cooldown_tickers.add(ct.get('ticker', ''))
            except (ValueError, TypeError):
                pass
    if cooldown_tickers:
        log.info(f"COOLDOWN ({TICKER_COOLDOWN_HOURS}h): {sorted(cooldown_tickers)}")

    nav = compute_nav(state, prices, sigmas)

    # Sort candidates by IV descending (higher IV = more premium)
    candidates = [t for t in UNIVERSE.keys()
                  if t in prices and t not in active_tickers and t not in cooldown_tickers]
    candidates.sort(key=lambda t: sigmas.get(t, 0), reverse=True)

    opened = 0
    for ticker in candidates:
        if len(state['positions']) >= MAX_CONCURRENT:
            break

        if ticker not in prices or ticker not in sigmas:
            continue

        S = prices[ticker]
        sigma = sigmas[ticker]

        # Price filter
        if S < MIN_PRICE or S > MAX_PRICE:
            continue

        # Blacklist
        if ticker in LOSER_BLACKLIST:
            continue

        # Sigma filter
        if sigma > SIGMA_MAX_ENTRY:
            log.debug(f"SIGMA SKIP: {ticker} sigma={sigma:.2f} > {SIGMA_MAX_ENTRY:.2f}")
            continue

        # Sector diversification: max 3 per sector
        sector = UNIVERSE.get(ticker, 'Unknown')
        if sector_counts.get(sector, 0) >= MAX_PER_SECTOR:
            continue

        # Earnings filter
        if has_earnings_soon(ticker, earnings_lookup):
            continue

        # Find expiry
        expiry_date = find_expiry()
        if expiry_date is None:
            continue
        T = (expiry_date - datetime.utcnow()).days / 365.0

        # Find strikes
        K_put = find_strike(S, sigma, T, PUT_DELTA_TARGET, kind='put')
        K_call = find_strike(S, sigma, T, CALL_DELTA_TARGET, kind='call')

        # Sanity: put below spot, call above spot
        if K_put >= S or K_call <= S:
            continue

        # Minimum gap between strikes (avoid strangles too narrow)
        min_gap = max(3.0, S * 0.04)
        if K_call - K_put < min_gap:
            continue

        # Price the options
        _exp_d = expiry_date.date() if hasattr(expiry_date, 'date') else expiry_date
        if _HAS_PRICING_BRIDGE:
            prem_put = _alpaca_get_premium(ticker, S, K_put, T, sigma, _exp_d, kind='put', bs_price_fn=bs_price)
            prem_call = _alpaca_get_premium(ticker, S, K_call, T, sigma, _exp_d, kind='call', bs_price_fn=bs_price)
        else:
            prem_put = bs_price(S, K_put, T, sigma, kind='put')
            prem_call = bs_price(S, K_call, T, sigma, kind='call')

        # Apply slippage
        prem_put_slip = apply_slippage(prem_put)
        prem_call_slip = apply_slippage(prem_call)
        credit_per_share = prem_put_slip + prem_call_slip

        if credit_per_share < MIN_NET_CREDIT:
            continue

        # Position sizing: max 3% NAV risk per position
        # For naked strangle, risk is theoretically unlimited, but we use
        # the stop-loss level as risk: stop triggers at 2x premium loss,
        # so max expected loss = 3x premium (original credit + 2x additional).
        max_loss_per_share = credit_per_share * (1 + STOP_LOSS_MULT)  # 3x premium
        risk_budget = nav * MAX_RISK_PER_POSITION * brake_scale * vix_size_mult
        max_contracts = int(risk_budget / (max_loss_per_share * 100))
        n_contracts = max(min(max_contracts, 5), 1)  # 1 to 5 contracts

        # Commission cost
        open_fees = 2 * COST_PER_CONTRACT * n_contracts  # 2 legs

        # Net credit after slippage and fees
        total_credit = credit_per_share * 100 * n_contracts - open_fees
        if total_credit <= 0:
            continue

        # Execute: receive credit
        state['cash'] += total_credit

        dte = (expiry_date - datetime.utcnow()).days
        position = {
            'ticker': ticker,
            'entry_date': datetime.utcnow().isoformat(),
            'expiry': expiry_date.isoformat(),
            'short_put_strike': K_put,
            'short_call_strike': K_call,
            'net_credit': round(credit_per_share, 4),
            'contracts': n_contracts,
            'spot_entry': round(S, 2),
            'sigma_entry': round(sigma, 4),
            'profit_take_price': round(credit_per_share * (1 - PROFIT_TAKE_PCT), 4),
            'stop_loss_price': round(credit_per_share * (1 + STOP_LOSS_MULT), 4),
        }
        state['positions'].append(position)

        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        opened += 1

        log.info(f"SELL STRANGLE {ticker} P:{K_put:.0f} C:{K_call:.0f} "
                 f"({dte}d, {n_contracts}c), "
                 f"credit ${total_credit:.2f}, "
                 f"PT@${position['profit_take_price']:.2f}/sh, "
                 f"SL@${position['stop_loss_price']:.2f}/sh "
                 f"[{sector}]")
        log_trade({
            'action': 'sell_strangle', 'ticker': ticker,
            'short_put': K_put, 'short_call': K_call,
            'net_credit_per_share': round(credit_per_share, 4),
            'total_credit': round(total_credit, 2),
            'contracts': n_contracts, 'dte': dte,
            'sigma': round(sigma, 3), 'spot': round(S, 2),
            'sector': sector,
            'vix': round(vix, 1),
            'brake_scale': brake_scale,
            'vix_size_mult': vix_size_mult,
            'time': datetime.utcnow().isoformat(),
        })

        # HC #702: Log both legs for pricing audit
        if _HAS_PRICING_LOGGER:
            _exp_str = str(_exp_d)[:10]
            for _k, _kind, _bs_val in [(K_put, 'put', prem_put), (K_call, 'call', prem_call)]:
                log_option_price(
                    engine="strangle", ticker=ticker, spot=S, strike=_k,
                    expiry=_exp_str, option_type=_kind, iv_used=sigma,
                    bs_price=_bs_val, source="bs_or_alpaca", action="entry"
                )

    if opened:
        log.info(f"Opened {opened} new strangles "
                 f"(brake: {brake_scale:.0%}, VIX mult: {vix_size_mult:.0%}, "
                 f"total positions: {len(state['positions'])})")


# ══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════

def run_cycle(earnings_lookup):
    """Run one full check cycle."""
    state = load_state()

    # Fetch prices for active positions + universe
    position_tickers = {pos['ticker'] for pos in state['positions']}
    needed = list(position_tickers | set(UNIVERSE.keys()))
    prices, sigmas = get_live_prices(needed)
    vix = get_vix()

    nav = compute_nav(state, prices, sigmas)

    log.info(f"NAV=${nav:,.0f}, cash=${state['cash']:,.0f}, "
             f"positions={len(state['positions'])}, VIX={vix:.1f}")

    # NAV history for brake
    nav_history = load_nav_history()
    nav_history.append({"date": datetime.utcnow().isoformat(), "nav": round(nav, 2)})
    nav_history = nav_history[-120:]
    save_nav_history(nav_history)

    # Equity curve brake
    brake_scale = compute_brake_scale(nav_history, nav)

    # ── Panic confluence overlay (offensive + defensive) ──
    # Strangles are premium-selling — panic reversal = high IV = fat premiums.
    try:
        from live_trading_linux.panic_confluence_monitor import get_panic_confluence
        panic = get_panic_confluence()
        panic_mult = panic.get('position_multiplier', 1.0)
        panic_mode = panic.get('mode', 'normal')
        if panic_mode == 'offensive':
            brake_scale = brake_scale * panic_mult
            log.info(f"PANIC OFFENSIVE: confluence {panic.get('confluence_score', 0)}/3, "
                     f"multiplier {panic_mult:.2f}x -> effective scale {brake_scale:.0%}")
        elif panic_mode == 'cautious':
            brake_scale = brake_scale * panic_mult  # 0.70x
            log.info(f"PANIC CAUTIOUS: stress building, "
                     f"multiplier {panic_mult:.2f}x -> effective scale {brake_scale:.0%}")
    except Exception as e:
        log.debug(f"Panic confluence check skipped: {e}")

    # Drawdown trigger
    dd_trigger_active = check_dd_trigger(nav_history)

    # Process existing positions (profit-take, stop-loss, roll, expiry)
    process_positions(state, prices, sigmas)

    # Open new positions
    open_new_positions(state, prices, sigmas, vix, earnings_lookup,
                       brake_scale, dd_trigger_active=dd_trigger_active)

    # Final NAV
    nav = compute_nav(state, prices, sigmas)
    check_nav_drop(nav)
    log_equity(nav, state['realized_pnl'])

    state['last_check'] = datetime.utcnow().isoformat()
    save_state(state)

    dd_status = "HALTED" if dd_trigger_active else "normal"
    log.info(f"Cycle end: NAV=${nav:,.0f}, cash=${state['cash']:,.0f}, "
             f"positions={len(state['positions'])}, trades={state['trade_count']}, "
             f"realized=${state['realized_pnl']:.2f}, dd_trigger={dd_status}")


def main():
    log.info("=" * 60)
    log.info("STRANGLE PAPER ENGINE — STR_d25_35dte_ba10")
    log.info(f"  Universe: {len(UNIVERSE)} tickers")
    log.info(f"  Put delta: {PUT_DELTA_TARGET}, Call delta: {CALL_DELTA_TARGET}")
    log.info(f"  DTE: {DTE_TARGET}d, PT: {PROFIT_TAKE_PCT:.0%}, SL: {STOP_LOSS_MULT:.0f}x premium")
    log.info(f"  Roll: at {ROLL_DTE_THRESHOLD} DTE")
    log.info(f"  Max positions: {MAX_CONCURRENT}, max per sector: {MAX_PER_SECTOR}")
    log.info(f"  Risk per position: {MAX_RISK_PER_POSITION:.0%} NAV")
    log.info(f"  VIX gates: half-size > {VIX_HALF_SIZE}, no entry > {VIX_NO_ENTRY}")
    log.info(f"  Brake: {BRAKE_LOOKBACK_DAYS}d/{BRAKE_THRESHOLD:.0%}/{BRAKE_SCALE:.0%}")
    log.info(f"  Cost: ${COST_PER_CONTRACT}/contract/leg x 2 legs, {SLIPPAGE_FRAC:.1%} slippage")
    log.info(f"  Capital: ${STARTING_CAPITAL:,.0f}")
    log.info(f"  Backtest: Sharpe 4.55, Sortino 13.36, CAGR 76.9%, MaxDD -4.4%")
    log.info("=" * 60)

    # Initialize Alpaca real pricing
    if _HAS_PRICING_BRIDGE:
        _init_bridge(engine_name='strangle')
        log.info("Alpaca real pricing bridge initialized")
    else:
        log.info("Alpaca pricing bridge not available, using BS-only pricing")

    earnings_lookup = load_earnings_dates()
    log.info(f"Earnings data loaded for {len(earnings_lookup)} tickers")

    while True:
        try:
            now = datetime.utcnow()
            weekday = now.weekday()
            hour_utc = now.hour + now.minute / 60

            # Market hours: Mon-Fri, 13:30-21:00 UTC (9:30 AM - 4 PM ET)
            if weekday < 5 and 13.5 <= hour_utc <= 21.0:
                run_cycle(earnings_lookup)
            else:
                if now.minute < 5:  # Once per hour off-hours
                    state = load_state()
                    log.info(f"Off-hours: positions={len(state['positions'])}, "
                             f"realized=${state['realized_pnl']:.2f}")

        except Exception as e:
            log.error(f"Cycle error: {e}", exc_info=True)

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == '__main__':
    main()
