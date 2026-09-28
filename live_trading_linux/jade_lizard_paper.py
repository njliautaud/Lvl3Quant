#!/usr/bin/env python3
"""
Jade Lizard Paper Trading Engine
==================================
Sells OTM put + bear call spread on high-IV liquid stocks.
Upgraded to 50-stock universe + IV rank ≥ 70% per rule optimization (HC #728).

Strategy:
  - Sell OTM put (~0.20 delta) + sell call spread (short ~0.25 delta call, long higher call)
  - Credit received MUST exceed call spread width (eliminates upside risk)
  - Entry: IV rank > 70th percentile, DTE 30-50 days, max 5 concurrent
  - Management: 50% profit target, 7 DTE exit, 2-sigma stop
  - Earnings filter: skip tickers with earnings within DTE + 2 day buffer

Rule sweep results (32 variants tested, 2015-2026):
  - ivrank_min_70: Sharpe 0.95, WR 71%, PF 1.48, MaxDD -2.7%, 1364 trades
  - Key finding: 50-stock universe critical (15-stock = negative Sharpe)
  - Simple IV rank filter beats ML timing by huge margin
  - Bear market WR 85% > Bull WR 80% (higher premiums + mean-reversion)

Cost model:
  - $0.65/contract/leg ($1.95 per jade lizard = 3 legs open, $1.95 close)
  - 2.5% slippage on premium (min $0.03/share)

PM2 cron: run daily at 9:35 AM ET (weekdays). One-shot per day.
State: /home/jupiter/Lvl3Quant/data/paper_engines/jade_lizard/

CLI:
    python3 -m live_trading_linux.jade_lizard_paper --run     # normal daily run
    python3 -m live_trading_linux.jade_lizard_paper --status  # print current state
    python3 -m live_trading_linux.jade_lizard_paper --smoke   # smoke test

Author: Claude (2026-07-21, post jade lizard backtest validation)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import logging
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

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

# HC #702 — Options pricing audit logger
try:
    from live_trading_linux.options_pricing_logger import log_option_price
    _HAS_PRICING_LOGGER = True
except ImportError:
    try:
        from options_pricing_logger import log_option_price
        _HAS_PRICING_LOGGER = True
    except ImportError:
        _HAS_PRICING_LOGGER = False

# ── Paths ──
ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "data" / "paper_engines" / "jade_lizard"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
EQUITY_FILE = STATE_DIR / "equity.csv"
TRADES_FILE = STATE_DIR / "trades.jsonl"
NAV_HISTORY_FILE = STATE_DIR / "nav_history.json"
IV_CACHE_FILE = STATE_DIR / "iv_rank_cache.json"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [JADE-LIZ] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_DIR / "jade_lizard_paper.log")),
    ],
)
log = logging.getLogger('JADE-LIZ')


# ══════════════════════════════════════════════════════════════════════
# STRATEGY CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

STARTING_CAPITAL = 100_000.0

# Jade lizard parameters
PUT_DELTA_TARGET = 0.20        # 20-delta OTM put (short)
CALL_DELTA_TARGET = 0.25       # 25-delta OTM call (short)
CALL_SPREAD_WIDTH = 5.0        # $5 call spread width (short call to long call)
DTE_TARGET = 45                # ~45 DTE target
DTE_MIN = 30
DTE_MAX = 50
PROFIT_TAKE_PCT = 0.50         # Close when 50% of premium captured
STOP_SIGMA_MULT = 2.0          # Close if underlying moves > 2 sigma
EXIT_DTE_THRESHOLD = 7         # Close at 7 DTE remaining
MIN_NET_CREDIT = 0.30          # Min $0.30/share total credit
IV_RANK_THRESHOLD = 70.0       # IV rank must be > 70th pctile (rule sweep winner, HC #728)
IV_LOOKBACK_DAYS = 252         # 1 year of IV history for ranking

# Position sizing
MAX_RISK_PER_POSITION = 0.05   # 5% of NAV max risk per position
MAX_CONCURRENT = 5             # Max 5 concurrent (rule sweep: maxpos_10 = disaster)
MAX_PER_TICKER = 1             # Max 1 position per ticker
NAV_MARGIN_CAP = 0.40          # Max 40% of NAV in total margin

# Risk controls
VIX_NO_ENTRY = 40.0            # No new entries when VIX > 40
VIX_HALF_SIZE = 30.0           # Half size when VIX > 30
TICKER_COOLDOWN_HOURS = 48     # 2-day cooldown per ticker after close

# Pricing
RISK_FREE = 0.04
COST_PER_CONTRACT = 0.65       # Per contract per leg
SLIPPAGE_FRAC = 0.025          # 2.5% slippage on premium
SLIPPAGE_MIN = 0.03            # Min slippage per share
NUM_LEGS = 3                   # Jade lizard has 3 legs

# Price filter
MIN_PRICE = 20.0
MAX_PRICE = 600.0

# Equity curve brake
BRAKE_LOOKBACK_DAYS = 60
BRAKE_THRESHOLD = 0.05         # 5% DD from peak triggers brake
BRAKE_SCALE = 0.25

# ══════════════════════════════════════════════════════════════════════
# UNIVERSE — 50 liquid large-cap stocks (rule sweep: 50 >> 15 for diversification)
# Backtest: 15-stock Sharpe -0.67, 50-stock Sharpe 1.03. Diversification critical.
# ══════════════════════════════════════════════════════════════════════

UNIVERSE = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'AMZN': 'ConsDisc',
    'META': 'Tech', 'NVDA': 'Tech', 'TSLA': 'ConsDisc', 'JPM': 'Fin',
    'GS': 'Fin', 'BAC': 'Fin', 'V': 'Fin', 'MA': 'Fin',
    'UNH': 'Health', 'JNJ': 'Health', 'PG': 'ConsStap', 'KO': 'ConsStap',
    'PEP': 'ConsStap', 'MRK': 'Health', 'ABBV': 'Health', 'LLY': 'Health',
    'HD': 'ConsDisc', 'COST': 'ConsStap', 'WMT': 'ConsStap', 'CRM': 'Tech',
    'AMD': 'Tech', 'NFLX': 'Tech', 'ADBE': 'Tech', 'INTC': 'Tech',
    'CSCO': 'Tech', 'QCOM': 'Tech', 'XOM': 'Energy', 'CVX': 'Energy',
    'PFE': 'Health', 'TMO': 'Health', 'ABT': 'Health', 'AVGO': 'Tech',
    'TXN': 'Tech', 'MCD': 'ConsDisc', 'NKE': 'ConsDisc', 'DIS': 'ConsDisc',
    'CMCSA': 'Comm', 'T': 'Comm', 'VZ': 'Comm', 'NEE': 'Util',
    'SO': 'Util', 'SHW': 'Materials', 'LMT': 'Indust', 'RTX': 'Indust',
    'CAT': 'Indust', 'DE': 'Indust',
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


def _get_option_price(ticker, S, K, T, sigma, expiry_date, kind='put'):
    """Get option price using Alpaca if available, else BS fallback.
    HC #709 audit fix: consistent pricing source for entry AND close."""
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


def apply_slippage(premium):
    """Reduce credit received by slippage."""
    return max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)


# ══════════════════════════════════════════════════════════════════════
# IV RANK COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def compute_iv_rank(ticker: str, current_sigma: float, iv_cache: dict) -> float:
    """
    Compute IV percentile rank over trailing 252 days.
    Uses 20-day realized vol as proxy for IV (common for paper engines).
    Returns percentile 0-100.
    """
    import yfinance as yf

    # Check cache (valid for 1 day)
    today_str = datetime.utcnow().strftime('%Y-%m-%d')
    cache_key = f"{ticker}_{today_str}"
    if cache_key in iv_cache:
        return iv_cache[cache_key]

    try:
        df = yf.download(ticker, period="18mo", auto_adjust=True,
                         threads=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) < 60:
            return 50.0  # Default if insufficient data

        # Compute rolling 20-day realized vol (annualized) as IV proxy
        log_ret = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        rolling_vol = log_ret.rolling(20).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) < IV_LOOKBACK_DAYS // 2:
            return 50.0

        # Use last 252 days of rolling vol
        history = rolling_vol.tail(IV_LOOKBACK_DAYS).values
        current_iv = rolling_vol.iloc[-1]

        # Percentile rank: what % of historical IV was below current
        rank = float(np.sum(history < current_iv) / len(history) * 100)

        # Cache it
        iv_cache[cache_key] = round(rank, 1)
        return round(rank, 1)

    except Exception as e:
        log.warning(f"IV rank calc failed for {ticker}: {e}")
        return 50.0


def load_iv_cache() -> dict:
    """Load IV rank cache from disk."""
    if IV_CACHE_FILE.exists():
        try:
            return json.loads(IV_CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_iv_cache(cache: dict):
    """Save IV rank cache to disk."""
    # Prune old entries (keep only today)
    today_str = datetime.utcnow().strftime('%Y-%m-%d')
    pruned = {k: v for k, v in cache.items() if today_str in k}
    IV_CACHE_FILE.write_text(json.dumps(pruned, indent=2))


# ══════════════════════════════════════════════════════════════════════
# EARNINGS FILTER
# ══════════════════════════════════════════════════════════════════════

def load_earnings_dates():
    """Load cached earnings dates for the universe."""
    cache_path = ROOT / "wheel_strategy_v1" / "data" / "cache" / "earnings_dates.parquet"
    if not cache_path.exists():
        log.warning("No cached earnings dates found — using yfinance fallback")
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


def has_earnings_soon(ticker, earnings_lookup, dte=DTE_TARGET, buffer=2):
    """Check if ticker has earnings within [today-buffer, today+DTE+buffer]."""
    if ticker not in earnings_lookup:
        # Fallback: try yfinance calendar
        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)
            cal = tk.calendar
            if cal is not None and not cal.empty:
                if hasattr(cal, 'iloc'):
                    next_date = pd.Timestamp(cal.iloc[0, 0])
                    now = pd.Timestamp.now()
                    days_until = (next_date - now).days
                    return -buffer <= days_until <= dte + buffer
        except Exception:
            pass
        return False

    check_date = pd.Timestamp.now()
    ed = earnings_lookup[ticker]
    window_start = np.datetime64(check_date) - np.timedelta64(buffer, 'D')
    window_end = np.datetime64(check_date) + np.timedelta64(dte + buffer, 'D')
    idx_s = np.searchsorted(ed, window_start, side='left')
    idx_e = np.searchsorted(ed, window_end, side='right')
    return idx_e > idx_s


# ══════════════════════════════════════════════════════════════════════
# PRICE DATA
# ══════════════════════════════════════════════════════════════════════

def get_live_prices(tickers: list) -> Tuple[dict, dict]:
    """Fetch current prices and 20d realized vol via yfinance."""
    import yfinance as yf
    prices = {}
    sigmas = {}

    try:
        data = yf.download(tickers, period="60d", auto_adjust=True,
                           threads=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers:
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
        elif len(tickers) == 1 and not data.empty:
            t = tickers[0]
            close = data["Close"].dropna()
            if len(close) > 0:
                prices[t] = float(close.iloc[-1])
                log_ret = np.log(close / close.shift(1)).dropna()
                sigmas[t] = float(log_ret.tail(20).std() * np.sqrt(252)) if len(log_ret) >= 20 else 0.3
                sigmas[t] = max(0.05, min(sigmas[t], 2.0))
    except Exception as e:
        log.error(f"Price download failed: {e}")

    return prices, sigmas


def get_vix() -> float:
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
# EXPIRY LOGIC (monthly — ~45 DTE)
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
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'cash': STARTING_CAPITAL,
        'positions': [],
        'closed_trades': [],
        'realized_pnl': 0.0,
        'trade_count': 0,
        'wins': 0,
        'losses': 0,
        'start_date': datetime.utcnow().strftime('%Y-%m-%d'),
        'last_run': None,
        'version': 'jade_lizard_v1',
    }


def save_state(state: dict):
    if len(state.get('closed_trades', [])) > 500:
        state['closed_trades'] = state['closed_trades'][-500:]
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def compute_nav(state: dict, prices: dict, sigmas: dict) -> float:
    """Compute NAV = cash + mark-to-market of open jade lizards.
    Uses BS with entry sigma for MTM stability (HC #709)."""
    nav = state['cash']

    for pos in state['positions']:
        ticker = pos['ticker']
        if ticker not in prices:
            continue

        S = prices[ticker]
        sigma = pos.get('sigma_entry', sigmas.get(ticker, 0.20))
        expiry = pd.Timestamp(pos['expiry'])
        T = max((expiry - pd.Timestamp.now()).days, 0) / 365.0
        contracts = pos.get('contracts', 1)

        # Jade lizard has 3 legs:
        # 1. Short put (we owe its value)
        put_val = bs_price(S, pos['short_put_strike'], T, sigma, kind='put')
        # 2. Short call (we owe its value)
        call_val = bs_price(S, pos['short_call_strike'], T, sigma, kind='call')
        # 3. Long call (we own its value)
        long_call_val = bs_price(S, pos['long_call_strike'], T, sigma, kind='call')

        # Net liability = short put + short call - long call
        net_liability = (put_val + call_val - long_call_val) * 100 * contracts
        nav -= net_liability

    return nav


def log_equity(nav: float, realized_pnl: float):
    """Append to equity CSV."""
    row = f"{datetime.utcnow().isoformat()},{nav:.2f},{realized_pnl:.2f}\n"
    if not EQUITY_FILE.exists():
        with open(EQUITY_FILE, 'w') as f:
            f.write("timestamp,nav,realized_pnl\n")
    with open(EQUITY_FILE, 'a') as f:
        f.write(row)


def log_trade(info: dict):
    """Append trade to JSONL log."""
    with open(TRADES_FILE, 'a') as f:
        f.write(json.dumps(info, default=str) + '\n')


# ══════════════════════════════════════════════════════════════════════
# NAV HISTORY & EQUITY CURVE BRAKE
# ══════════════════════════════════════════════════════════════════════

def load_nav_history() -> list:
    if NAV_HISTORY_FILE.exists():
        with open(NAV_HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_nav_history(history: list):
    with open(NAV_HISTORY_FILE, 'w') as f:
        json.dump(history, f)


def compute_brake_scale(nav_history: list, current_nav: float) -> float:
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


# ══════════════════════════════════════════════════════════════════════
# POSITION MANAGEMENT — check exits on existing positions
# ══════════════════════════════════════════════════════════════════════

def process_positions(state: dict, prices: dict, sigmas: dict):
    """Check existing jade lizards for profit-take, stop, DTE exit, expiry."""
    now = pd.Timestamp.now()
    new_positions = []

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
        net_credit = pos['net_credit']  # total per-share credit at entry
        total_credit = net_credit * 100 * contracts

        # Anti-churn: don't close within first 24h
        entry_dt = pd.Timestamp(pos.get('entry_date', '2020-01-01'))
        hours_held = (now - entry_dt).total_seconds() / 3600
        if hours_held < 24:
            new_positions.append(pos)
            continue

        # Current value of the jade lizard (cost to close)
        expiry_date = expiry.date() if hasattr(expiry, 'date') else expiry
        put_val = _get_option_price(ticker, S, pos['short_put_strike'], T, sigma, expiry_date, kind='put')
        call_val = _get_option_price(ticker, S, pos['short_call_strike'], T, sigma, expiry_date, kind='call')
        long_call_val = _get_option_price(ticker, S, pos['long_call_strike'], T, sigma, expiry_date, kind='call')

        # To close: buy back short put + short call, sell long call
        close_cost_per_share = put_val + call_val - long_call_val
        close_cost = close_cost_per_share * 100 * contracts
        close_fees = NUM_LEGS * COST_PER_CONTRACT * contracts

        pnl = total_credit - close_cost - close_fees
        profit_pct = pnl / total_credit if total_credit > 0 else 0

        exit_reason = None

        # ── 1. Profit take (50%) ──
        if profit_pct >= PROFIT_TAKE_PCT:
            exit_reason = 'profit_take'

        # ── 2. DTE exit (7 days remaining) ──
        elif T_days <= EXIT_DTE_THRESHOLD:
            exit_reason = 'dte_exit'

        # ── 3. 2-sigma stop ──
        elif 'spot_entry' in pos and pos.get('sigma_entry', 0) > 0:
            spot_entry = pos['spot_entry']
            sigma_entry = pos['sigma_entry']
            days_held = max((now - entry_dt).days, 1)
            # 2-sigma move over the holding period
            move_threshold = sigma_entry * math.sqrt(days_held / 252.0) * STOP_SIGMA_MULT
            price_move = abs(S - spot_entry) / spot_entry
            if price_move > move_threshold:
                exit_reason = 'sigma_stop'

        # ── 4. Expiry settlement ──
        elif now >= expiry:
            # Calculate intrinsic at expiry
            put_intrinsic = max(pos['short_put_strike'] - S, 0) * 100 * contracts
            short_call_intrinsic = max(S - pos['short_call_strike'], 0) * 100 * contracts
            long_call_intrinsic = max(S - pos['long_call_strike'], 0) * 100 * contracts
            net_assignment = put_intrinsic + short_call_intrinsic - long_call_intrinsic
            pnl = total_credit - net_assignment - close_fees

            state['cash'] -= (net_assignment + close_fees)
            state['realized_pnl'] += pnl
            state['trade_count'] += 1
            if pnl >= 0:
                state['wins'] = state.get('wins', 0) + 1
            else:
                state['losses'] = state.get('losses', 0) + 1

            log.info(f"EXPIRED {ticker} jade lizard. S=${S:.2f} PnL: ${pnl:.2f}")
            state['closed_trades'].append({
                'ticker': ticker, 'action': 'expired',
                'pnl': round(pnl, 2), 'contracts': contracts,
                'close_date': datetime.utcnow().isoformat(),
            })
            log_trade({
                'action': 'expired', 'ticker': ticker,
                'short_put': pos['short_put_strike'],
                'short_call': pos['short_call_strike'],
                'long_call': pos['long_call_strike'],
                'pnl': round(pnl, 2),
                'underlying_at_expiry': round(S, 2),
                'contracts': contracts,
                'time': datetime.utcnow().isoformat(),
            })
            continue

        # Process exit if triggered
        if exit_reason:
            state['cash'] -= (close_cost + close_fees)
            state['realized_pnl'] += pnl
            state['trade_count'] += 1
            if pnl >= 0:
                state['wins'] = state.get('wins', 0) + 1
            else:
                state['losses'] = state.get('losses', 0) + 1

            log.info(f"CLOSE [{exit_reason}] {ticker} jade lizard "
                     f"P:{pos['short_put_strike']:.1f} C:{pos['short_call_strike']:.1f}/"
                     f"{pos['long_call_strike']:.1f} "
                     f"PnL: ${pnl:.2f} ({profit_pct:.0%}) S=${S:.2f}")
            state['closed_trades'].append({
                'ticker': ticker, 'action': exit_reason,
                'pnl': round(pnl, 2), 'contracts': contracts,
                'close_date': datetime.utcnow().isoformat(),
            })
            log_trade({
                'action': exit_reason, 'ticker': ticker,
                'short_put': pos['short_put_strike'],
                'short_call': pos['short_call_strike'],
                'long_call': pos['long_call_strike'],
                'net_credit': round(net_credit, 4),
                'close_cost': round(close_cost_per_share, 4),
                'pnl': round(pnl, 2),
                'profit_pct': round(profit_pct, 4),
                'underlying_at_close': round(S, 2),
                'days_held': max((now - entry_dt).days, 0),
                'contracts': contracts,
                'time': datetime.utcnow().isoformat(),
            })

            # HC #702 pricing log
            if _HAS_PRICING_LOGGER:
                for _k, _kind in [(pos['short_put_strike'], 'put'),
                                  (pos['short_call_strike'], 'call'),
                                  (pos['long_call_strike'], 'call')]:
                    log_option_price(
                        engine="jade_lizard", ticker=ticker, spot=S, strike=_k,
                        expiry=str(expiry_date)[:10], option_type=_kind,
                        iv_used=sigma, bs_price=bs_price(S, _k, T, sigma, kind=_kind),
                        source="bs_or_alpaca", action=f"close_{exit_reason}"
                    )
            continue

        # Position survives
        new_positions.append(pos)

    state['positions'] = new_positions


# ══════════════════════════════════════════════════════════════════════
# NEW POSITION ENTRY
# ══════════════════════════════════════════════════════════════════════

def open_new_positions(state: dict, prices: dict, sigmas: dict,
                       vix: float, earnings_lookup: dict, brake_scale: float,
                       iv_cache: dict):
    """Scan universe and open new jade lizard positions."""

    # ── VIX gate ──
    if vix > VIX_NO_ENTRY:
        log.info(f"VIX {vix:.1f} > {VIX_NO_ENTRY} -- no new entries")
        return
    if brake_scale <= 0:
        log.info("EQUITY BRAKE HALT -- no new jade lizards")
        return

    # VIX-based size reduction
    vix_size_mult = 0.50 if vix > VIX_HALF_SIZE else 1.0
    if vix_size_mult < 1.0:
        log.info(f"VIX STRESS: {vix:.1f} > {VIX_HALF_SIZE} -> position size cut to 50%")

    # Active position checks
    active_tickers = {pos['ticker'] for pos in state['positions']}
    if len(state['positions']) >= MAX_CONCURRENT:
        log.info(f"MAX POSITIONS reached ({MAX_CONCURRENT})")
        return

    # Ticker cooldown
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
            log.debug(f"PRICE SKIP: {ticker} ${S:.2f}")
            continue

        # Earnings filter
        if has_earnings_soon(ticker, earnings_lookup, dte=DTE_TARGET):
            log.info(f"EARNINGS SKIP: {ticker}")
            continue

        # ── IV RANK FILTER (core jade lizard entry condition) ──
        iv_rank = compute_iv_rank(ticker, sigma, iv_cache)
        if iv_rank < IV_RANK_THRESHOLD:
            log.info(f"IV RANK SKIP: {ticker} IVR={iv_rank:.0f}% < {IV_RANK_THRESHOLD:.0f}%")
            continue

        # Find expiry
        expiry_date = find_expiry()
        if expiry_date is None:
            continue
        T = (expiry_date - datetime.utcnow()).days / 365.0

        # ── Find strikes ──
        # 1. Short put: 0.20 delta OTM put
        K_put = find_strike(S, sigma, T, PUT_DELTA_TARGET, kind='put')

        # 2. Short call: 0.25 delta OTM call
        K_call = find_strike(S, sigma, T, CALL_DELTA_TARGET, kind='call')

        # 3. Long call: call spread width above short call
        K_long_call = K_call + CALL_SPREAD_WIDTH
        K_long_call = round(K_long_call * 2) / 2  # Round to nearest $0.50

        # Sanity checks
        if K_put >= S:
            log.debug(f"STRIKE SKIP: {ticker} put strike {K_put} >= spot {S}")
            continue
        if K_call <= S:
            log.debug(f"STRIKE SKIP: {ticker} call strike {K_call} <= spot {S}")
            continue
        if K_long_call <= K_call:
            continue

        # ── Price the legs ──
        expiry_d = expiry_date.date() if hasattr(expiry_date, 'date') else expiry_date
        prem_put = _get_option_price(ticker, S, K_put, T, sigma, expiry_d, kind='put')
        prem_call = _get_option_price(ticker, S, K_call, T, sigma, expiry_d, kind='call')
        prem_long_call = _get_option_price(ticker, S, K_long_call, T, sigma, expiry_d, kind='call')

        # Apply slippage: we receive less on shorts, pay more on long
        prem_put_slip = apply_slippage(prem_put)
        prem_call_slip = apply_slippage(prem_call)
        prem_long_call_cost = prem_long_call * (1 + SLIPPAGE_FRAC)  # We pay more to buy

        # Net credit = short put + short call - long call
        net_credit_per_share = prem_put_slip + prem_call_slip - prem_long_call_cost

        if net_credit_per_share < MIN_NET_CREDIT:
            log.debug(f"CREDIT SKIP: {ticker} net credit ${net_credit_per_share:.2f} < ${MIN_NET_CREDIT}")
            continue

        # ── JADE LIZARD KEY CONDITION: credit > call spread width ──
        # This eliminates upside risk (can't lose on the upside)
        call_spread_width = K_long_call - K_call
        if net_credit_per_share < call_spread_width:
            log.info(f"JADE RULE SKIP: {ticker} credit ${net_credit_per_share:.2f} < "
                     f"call spread width ${call_spread_width:.2f} "
                     f"— upside risk not eliminated")
            continue

        # ── Position sizing ──
        # Max loss on jade lizard = (put strike - 0) * 100 - net credit
        # But practically: max loss on downside = (put_strike * 100 * contracts) - total credit
        # Risk per position = max loss capped at put-strike exposure
        put_margin = K_put * 100  # Per contract, put side margin
        max_contracts_by_risk = max(1, int(
            nav * MAX_RISK_PER_POSITION * brake_scale * vix_size_mult / (put_margin + 1)
        ))
        max_contracts_by_margin = max(1, int(
            nav * NAV_MARGIN_CAP / (put_margin * MAX_CONCURRENT + 1)
        ))
        contracts = min(max_contracts_by_risk, max_contracts_by_margin, 5)  # Cap at 5

        open_fees = NUM_LEGS * COST_PER_CONTRACT * contracts
        total_credit = net_credit_per_share * 100 * contracts - open_fees

        if total_credit <= 0:
            continue

        # ── Open the position ──
        state['cash'] += total_credit
        dte = (expiry_date - datetime.utcnow()).days

        position = {
            'ticker': ticker,
            'entry_date': datetime.utcnow().isoformat(),
            'expiry': expiry_date.isoformat(),
            'short_put_strike': round(K_put, 2),
            'short_call_strike': round(K_call, 2),
            'long_call_strike': round(K_long_call, 2),
            'call_spread_width': round(call_spread_width, 2),
            'net_credit': round(net_credit_per_share, 4),
            'contracts': contracts,
            'spot_entry': round(S, 2),
            'sigma_entry': round(sigma, 4),
            'iv_rank': round(iv_rank, 1),
        }
        state['positions'].append(position)

        log.info(f"OPEN {ticker} jade lizard: "
                 f"P:{K_put:.1f} C:{K_call:.1f}/{K_long_call:.1f} "
                 f"({dte}d DTE) credit ${total_credit:.2f} "
                 f"(${net_credit_per_share:.2f}/sh x {contracts} contracts) "
                 f"IVR={iv_rank:.0f}% sigma={sigma:.2f}")

        log_trade({
            'action': 'open', 'ticker': ticker,
            'short_put': round(K_put, 2),
            'short_call': round(K_call, 2),
            'long_call': round(K_long_call, 2),
            'call_spread_width': round(call_spread_width, 2),
            'net_credit': round(net_credit_per_share, 4),
            'total_credit': round(total_credit, 2),
            'contracts': contracts,
            'spot': round(S, 2),
            'sigma': round(sigma, 4),
            'iv_rank': round(iv_rank, 1),
            'dte': dte,
            'open_fees': round(open_fees, 2),
            'time': datetime.utcnow().isoformat(),
        })

        # HC #702 pricing log
        if _HAS_PRICING_LOGGER:
            for _k, _kind, _prem in [(K_put, 'put', prem_put),
                                     (K_call, 'call', prem_call),
                                     (K_long_call, 'call', prem_long_call)]:
                log_option_price(
                    engine="jade_lizard", ticker=ticker, spot=S, strike=_k,
                    expiry=str(expiry_d)[:10], option_type=_kind, iv_used=sigma,
                    bs_price=bs_price(S, _k, T, sigma, kind=_kind),
                    source="bs_or_alpaca", action="open"
                )

        opened += 1

    if opened > 0:
        log.info(f"Opened {opened} new jade lizard position(s)")
    else:
        log.info("No new jade lizard entries today (all filtered)")


# ══════════════════════════════════════════════════════════════════════
# MAIN RUN CYCLE
# ══════════════════════════════════════════════════════════════════════

def run_daily(earnings_lookup: dict, iv_cache: dict):
    """Run one daily cycle: manage positions, then scan for new entries."""
    log.info("=" * 50)
    log.info("Starting daily jade lizard cycle")

    state = load_state()
    tickers = list(UNIVERSE.keys())

    # Fetch prices
    log.info(f"Fetching prices for {len(tickers)} tickers...")
    prices, sigmas = get_live_prices(tickers)
    log.info(f"Got prices for {len(prices)} tickers")

    if len(prices) == 0:
        log.error("No prices available — aborting cycle")
        return

    # Fetch VIX
    vix = get_vix()
    log.info(f"VIX: {vix:.1f}")

    # NAV history and brake
    nav_history = load_nav_history()
    nav = compute_nav(state, prices, sigmas)
    brake_scale = compute_brake_scale(nav_history, nav)

    # ── 1. Manage existing positions ──
    n_before = len(state['positions'])
    process_positions(state, prices, sigmas)
    n_after = len(state['positions'])
    if n_before != n_after:
        log.info(f"Position management: {n_before} -> {n_after} positions")

    # ── 2. Scan for new entries ──
    open_new_positions(state, prices, sigmas, vix, earnings_lookup, brake_scale, iv_cache)

    # ── 3. Update NAV and save ──
    nav = compute_nav(state, prices, sigmas)
    state['last_run'] = datetime.utcnow().isoformat()

    # Update NAV history
    nav_history.append({
        'date': datetime.utcnow().strftime('%Y-%m-%d'),
        'nav': round(nav, 2),
    })
    if len(nav_history) > 365:
        nav_history = nav_history[-365:]
    save_nav_history(nav_history)

    # Log equity
    log_equity(nav, state['realized_pnl'])

    # Save state
    save_state(state)

    # Save IV cache
    save_iv_cache(iv_cache)

    # ── Compute stats ──
    wins = state.get('wins', 0)
    losses = state.get('losses', 0)
    total_trades = wins + losses
    wr = (wins / total_trades * 100) if total_trades > 0 else 0.0

    log.info(f"Cycle complete: NAV=${nav:,.0f}, cash=${state['cash']:,.0f}, "
             f"positions={len(state['positions'])}, trades={state['trade_count']}, "
             f"realized=${state['realized_pnl']:.2f}, WR={wr:.0f}% ({wins}W/{losses}L), "
             f"VIX={vix:.1f}, brake={brake_scale:.2f}")

    # ── Send NAV summary to webhook (if large move) ──
    if nav_history and len(nav_history) >= 2:
        prev_nav = nav_history[-2]['nav']
        if prev_nav > 0:
            daily_return = (nav - prev_nav) / prev_nav
            if abs(daily_return) > 0.02:  # Alert on > 2% daily move
                try:
                    os.system(
                        f'node /home/jupiter/teleclaude-main/utils/webhook_notifier.js '
                        f'"JADE LIZARD daily move: {daily_return:+.1%} '
                        f'(${prev_nav:,.0f} -> ${nav:,.0f})" 2>/dev/null'
                    )
                except Exception:
                    pass


def print_status():
    """Print current engine status."""
    state = load_state()
    nav_history = load_nav_history()

    print("\n" + "=" * 60)
    print("JADE LIZARD PAPER ENGINE — STATUS")
    print("=" * 60)
    print(f"Version:       {state.get('version', '?')}")
    print(f"Start date:    {state.get('start_date', '?')}")
    print(f"Last run:      {state.get('last_run', 'never')}")
    print(f"Cash:          ${state['cash']:,.2f}")

    if nav_history:
        print(f"Last NAV:      ${nav_history[-1]['nav']:,.2f}")
        if len(nav_history) >= 2:
            daily_ret = (nav_history[-1]['nav'] - nav_history[-2]['nav']) / nav_history[-2]['nav']
            print(f"Daily return:  {daily_ret:+.2%}")

    print(f"Positions:     {len(state['positions'])}")
    print(f"Total trades:  {state['trade_count']}")
    print(f"Realized P&L:  ${state['realized_pnl']:,.2f}")

    wins = state.get('wins', 0)
    losses = state.get('losses', 0)
    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0
    print(f"Win rate:      {wr:.0f}% ({wins}W / {losses}L)")

    if state['positions']:
        print(f"\n{'Ticker':<8} {'Put':>8} {'Call':>8} {'Long C':>8} {'Credit':>8} {'Cts':>4} {'IVR':>5} {'Expiry':>12}")
        print("-" * 72)
        for pos in state['positions']:
            print(f"{pos['ticker']:<8} "
                  f"${pos['short_put_strike']:>7.1f} "
                  f"${pos['short_call_strike']:>7.1f} "
                  f"${pos['long_call_strike']:>7.1f} "
                  f"${pos['net_credit']:>7.2f} "
                  f"{pos.get('contracts', 1):>4} "
                  f"{pos.get('iv_rank', 0):>4.0f}% "
                  f"{pos['expiry'][:10]:>12}")

    print("=" * 60)


def smoke_test():
    """Quick smoke test: fetch prices, compute IV rank, find strikes."""
    print("\n" + "=" * 40)
    print("SMOKE TEST — Jade Lizard Paper Engine")
    print("=" * 40)

    tickers = ['AAPL', 'TSLA', 'META']
    print(f"\nFetching prices for {tickers}...")
    prices, sigmas = get_live_prices(tickers)

    for t in tickers:
        if t not in prices:
            print(f"  {t}: NO PRICE DATA")
            continue
        S = prices[t]
        sigma = sigmas.get(t, 0.30)

        iv_cache = {}
        iv_rank = compute_iv_rank(t, sigma, iv_cache)

        T = DTE_TARGET / 365.0
        K_put = find_strike(S, sigma, T, PUT_DELTA_TARGET, kind='put')
        K_call = find_strike(S, sigma, T, CALL_DELTA_TARGET, kind='call')
        K_long_call = K_call + CALL_SPREAD_WIDTH

        prem_put = bs_price(S, K_put, T, sigma, kind='put')
        prem_call = bs_price(S, K_call, T, sigma, kind='call')
        prem_long = bs_price(S, K_long_call, T, sigma, kind='call')
        net_credit = prem_put + prem_call - prem_long
        call_width = K_long_call - K_call

        jade_ok = "YES" if net_credit >= call_width else "NO"

        print(f"\n  {t}: S=${S:.2f}, sigma={sigma:.2f}, IVR={iv_rank:.0f}%")
        print(f"    Put: K={K_put:.1f} prem=${prem_put:.2f}")
        print(f"    Short call: K={K_call:.1f} prem=${prem_call:.2f}")
        print(f"    Long call:  K={K_long_call:.1f} prem=${prem_long:.2f}")
        print(f"    Net credit: ${net_credit:.2f}, call spread width: ${call_width:.2f}")
        print(f"    Jade rule (credit > width): {jade_ok}")
        print(f"    IV rank threshold ({IV_RANK_THRESHOLD}%): {'PASS' if iv_rank >= IV_RANK_THRESHOLD else 'FAIL'}")

    print("\nSmoke test complete.")


def main():
    parser = argparse.ArgumentParser(description="Jade Lizard Paper Trading Engine")
    parser.add_argument('--run', action='store_true', help='Run daily cycle')
    parser.add_argument('--status', action='store_true', help='Print current status')
    parser.add_argument('--smoke', action='store_true', help='Run smoke test')
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.smoke:
        smoke_test()
        return

    # Default: run daily cycle
    log.info("=" * 60)
    log.info("JADE LIZARD PAPER ENGINE v1")
    log.info(f"  Universe: {len(UNIVERSE)} tickers")
    log.info(f"  Put delta: {PUT_DELTA_TARGET}, Call delta: {CALL_DELTA_TARGET}")
    log.info(f"  Call spread width: ${CALL_SPREAD_WIDTH}")
    log.info(f"  DTE: {DTE_TARGET}d ({DTE_MIN}-{DTE_MAX}), PT: {PROFIT_TAKE_PCT:.0%}")
    log.info(f"  IV rank threshold: {IV_RANK_THRESHOLD}%")
    log.info(f"  Sigma stop: {STOP_SIGMA_MULT}x, DTE exit: {EXIT_DTE_THRESHOLD}d")
    log.info(f"  Max positions: {MAX_CONCURRENT}")
    log.info(f"  VIX gates: half-size > {VIX_HALF_SIZE}, no entry > {VIX_NO_ENTRY}")
    log.info(f"  Cost: ${COST_PER_CONTRACT}/contract/leg x {NUM_LEGS} legs, "
             f"{SLIPPAGE_FRAC:.1%} slippage")
    log.info(f"  Capital: ${STARTING_CAPITAL:,.0f}")
    log.info(f"  Backtest ref: Sharpe 1.77, CAGR 10.2%, WR 70.5%, PF 3.52")
    log.info("=" * 60)

    # Initialize Alpaca real pricing
    if _HAS_PRICING_BRIDGE:
        _init_bridge(engine_name='jade_lizard')
        log.info("Alpaca real pricing bridge initialized")
    else:
        log.info("Alpaca pricing bridge not available, using BS-only pricing")

    earnings_lookup = load_earnings_dates()
    iv_cache = load_iv_cache()

    run_daily(earnings_lookup, iv_cache)

    # Pricing session summary
    if _HAS_PRICING_BRIDGE:
        try:
            _log_pricing_summary()
        except Exception:
            pass

    log.info("Jade lizard daily run complete. Exiting.")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # No args = default to --run (for PM2 cron)
        sys.argv.append('--run')
    main()
