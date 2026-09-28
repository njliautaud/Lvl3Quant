#!/usr/bin/env python3
"""
Earnings Vol Selling Paper Trading Engine
==========================================

Sells 10% OTM strangles 2 trading days before earnings, buys them back the
morning after. Captures the IV crush that happens when earnings uncertainty
resolves.

Strategy (from backtest /output/earnings_vol_selling_v1/):
  - 2 trading days before earnings: sell 10% OTM put + 10% OTM call (strangle)
  - Morning after earnings: buy back both legs
  - 2x premium stop-loss while position is open
  - 1 strangle per stock, max 5 concurrent positions

Backtest results:
  - Sharpe 2.71
  - Passes R1 regime-agnostic test (40+ OOT days, all regimes)
  - Passes permutation test

Best tickers (high WR in backtest):
  NVDA, MSFT, NOW, BA, MCD, BAC, MS, DIS, AAPL, GS, HD, LOW, SBUX, GOOGL, CRM, META

Avoid (net losers in backtest):
  COIN, INTC, DE, CAT, CVX, PG

Cost model:
  - $0.65/contract/leg ($1.30 per strangle open, $1.30 close)
  - 2.5% slippage on premium (min $0.03/share)

PM2 compatible: 5-minute polling cycle, logs to stdout.
State: /home/jupiter/Lvl3Quant/live_trading_linux/earnings_vol_state/state.json

Author: Claude (2026-07-15, post earnings vol backtest validation)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import logging
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf

# ── Pricing bridge (falls back to BS when market closed) ──
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

# ── Pricing audit logger (HC #702) ──
try:
    try:
        from live_trading_linux.options_pricing_logger import log_option_price
    except ImportError:
        from options_pricing_logger import log_option_price
    _HAS_PRICING_LOGGER = True
except ImportError:
    _HAS_PRICING_LOGGER = False

# ── Paths ──
ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "live_trading_linux" / "earnings_vol_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "state.json"
EQUITY_FILE = STATE_DIR / "equity.csv"
TRADES_FILE = STATE_DIR / "trades.jsonl"
NAV_HISTORY_FILE = STATE_DIR / "nav_history.json"
EARNINGS_CACHE_FILE = STATE_DIR / "earnings_cache.json"
SOD_NAV_FILE = STATE_DIR / "sod_nav.json"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [EARN-VOL] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_DIR / "earnings_vol_paper.log")),
    ],
)
log = logging.getLogger('EARN-VOL')

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

STARTING_CAPITAL = 100_000.0
POLL_INTERVAL_SEC = 300  # 5 minutes

# Strangle parameters
OTM_PCT = 0.10             # 10% OTM for both put and call
STOP_LOSS_MULT = 2.0       # Close when loss = 2x premium received
ENTRY_DAYS_BEFORE = 2      # Enter 2 trading days before earnings
EXIT_DAYS_AFTER_MAX = 3    # Force-close if not closed by 3 days after earnings

# Position sizing
MAX_CONCURRENT = 5         # Max 5 concurrent positions
CONTRACTS_PER_STOCK = 1    # 1 strangle per stock

# Pricing
RISK_FREE = 0.04
COST_PER_CONTRACT = 0.65   # Per contract per leg
SLIPPAGE_FRAC = 0.025      # 2.5% slippage on premium
SLIPPAGE_MIN = 0.03        # Min $0.03/share slippage

# NAV drop alert
NAV_DROP_ALERT_PCT = 0.025

# ── Best tickers from backtest ──
UNIVERSE = [
    'NVDA', 'MSFT', 'NOW', 'BA', 'MCD', 'BAC', 'MS', 'DIS',
    'AAPL', 'GS', 'HD', 'LOW', 'SBUX', 'GOOGL', 'CRM', 'META',
]

# ── Avoid list (net losers in backtest) ──
AVOID_TICKERS = {'COIN', 'INTC', 'DE', 'CAT', 'CVX', 'PG'}


# ══════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES PRICING
# ══════════════════════════════════════════════════════════════════════

def _Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, sigma: float,
             r: float = RISK_FREE, kind: str = "put") -> float:
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


def apply_slippage(premium: float) -> float:
    """Reduce credit received by slippage."""
    return max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)


# ══════════════════════════════════════════════════════════════════════
# EARNINGS CALENDAR
# ══════════════════════════════════════════════════════════════════════

def _load_earnings_cache() -> dict:
    """Load cached earnings dates."""
    if EARNINGS_CACHE_FILE.exists():
        try:
            with open(EARNINGS_CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_earnings_cache(cache: dict):
    with open(EARNINGS_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def fetch_earnings_dates_yf(ticker: str) -> List[date]:
    """Fetch upcoming earnings dates from yfinance."""
    dates = set()
    try:
        t = yf.Ticker(ticker)
        # Approach 1: calendar
        try:
            cal = t.calendar
            if cal is not None and isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    if isinstance(ed, (list, tuple)):
                        for d in ed:
                            try:
                                dates.add(pd.Timestamp(d).date())
                            except Exception:
                                pass
                    else:
                        try:
                            dates.add(pd.Timestamp(ed).date())
                        except Exception:
                            pass
        except Exception:
            pass

        # Approach 2: earnings_dates (multiple quarters)
        try:
            edf = t.earnings_dates
            if edf is not None and not edf.empty:
                today = date.today()
                for idx in edf.index:
                    try:
                        d = pd.Timestamp(idx).date()
                        if d >= today - timedelta(days=5):
                            dates.add(d)
                    except Exception:
                        pass
        except Exception:
            pass

    except Exception as e:
        log.warning(f"Error fetching earnings for {ticker}: {e}")

    return sorted(dates)


def refresh_earnings_calendar(force: bool = False) -> Dict[str, List[str]]:
    """
    Refresh earnings calendar for the universe.
    Cache refreshes once per day unless forced.
    Returns dict: ticker -> list of date strings.
    """
    cache = _load_earnings_cache()
    today_str = date.today().isoformat()
    last_refresh = cache.get("_meta", {}).get("refresh_date", "")

    if not force and last_refresh == today_str:
        log.info(f"Earnings calendar: using today's cache "
                 f"({sum(1 for k in cache if k != '_meta')} tickers)")
        return {k: v for k, v in cache.items() if k != "_meta"}

    log.info(f"Refreshing earnings calendar for {len(UNIVERSE)} tickers...")
    new_cache: Dict[str, List[str]] = {}
    errors = 0

    for i, ticker in enumerate(UNIVERSE):
        if ticker in AVOID_TICKERS:
            continue
        try:
            dates = fetch_earnings_dates_yf(ticker)
            new_cache[ticker] = [d.isoformat() for d in dates]
            if dates:
                log.info(f"  {ticker}: earnings dates = "
                         f"{[d.isoformat() for d in dates]}")
        except Exception as e:
            log.warning(f"  {ticker}: fetch failed - {e}")
            if ticker in cache:
                new_cache[ticker] = cache[ticker]
            errors += 1
        time.sleep(0.3)

    new_cache["_meta"] = {"refresh_date": today_str, "errors": errors}
    _save_earnings_cache(new_cache)

    with_dates = sum(1 for k, v in new_cache.items() if k != "_meta" and v)
    log.info(f"Earnings calendar refreshed: {with_dates}/{len(UNIVERSE)} tickers "
             f"have dates ({errors} errors)")
    return {k: v for k, v in new_cache.items() if k != "_meta"}


def get_entry_candidates(earnings_calendar: Dict[str, List[str]],
                         check_date: Optional[date] = None) -> Dict[str, date]:
    """
    Return tickers with earnings in exactly ENTRY_DAYS_BEFORE trading days.
    We use calendar days as proxy (2 calendar days ~ 2 trading days for weekdays).
    Returns dict: ticker -> earnings_date.
    """
    if check_date is None:
        check_date = date.today()

    candidates: Dict[str, date] = {}
    for ticker, date_strs in earnings_calendar.items():
        if ticker in AVOID_TICKERS:
            continue
        for ds in date_strs:
            try:
                ed = date.fromisoformat(ds)
                days_until = (ed - check_date).days
                # Enter 1-3 calendar days before earnings (covers ~2 trading days)
                if 1 <= days_until <= 3:
                    candidates[ticker] = ed
                    break
            except ValueError:
                pass
    return candidates


def should_close_position(pos: dict, check_date: Optional[date] = None) -> bool:
    """
    Check if a position should be closed (morning after earnings).
    Close when: check_date >= earnings_date + 1 (next trading day after earnings).
    """
    if check_date is None:
        check_date = date.today()

    earnings_date_str = pos.get("earnings_date", "")
    if not earnings_date_str:
        return False
    try:
        ed = date.fromisoformat(earnings_date_str)
    except ValueError:
        return False

    days_after = (check_date - ed).days
    # Close morning after earnings (1-3 days after, covering weekends)
    return days_after >= 1


def should_force_close(pos: dict, check_date: Optional[date] = None) -> bool:
    """Force-close if position has been open too long after earnings."""
    if check_date is None:
        check_date = date.today()

    earnings_date_str = pos.get("earnings_date", "")
    if not earnings_date_str:
        return True  # No earnings date = force close
    try:
        ed = date.fromisoformat(earnings_date_str)
    except ValueError:
        return True

    days_after = (check_date - ed).days
    return days_after > EXIT_DAYS_AFTER_MAX


# ══════════════════════════════════════════════════════════════════════
# PRICE DATA
# ══════════════════════════════════════════════════════════════════════

def get_spot_and_vol(ticker: str) -> Tuple[Optional[float], float]:
    """Get current price and 20-day realized vol."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="60d", auto_adjust=True)
        if hist.empty or len(hist) < 5:
            return None, 0.30
        spot = float(hist["Close"].iloc[-1])
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        if len(log_ret) >= 20:
            sigma = float(log_ret.tail(20).std() * np.sqrt(252))
        elif len(log_ret) > 5:
            sigma = float(log_ret.std() * np.sqrt(252))
        else:
            sigma = 0.30
        sigma = max(0.05, min(sigma, 3.0))
        return spot, sigma
    except Exception as e:
        log.warning(f"Price fetch failed for {ticker}: {e}")
        return None, 0.30


def get_batch_prices(tickers: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Fetch prices and sigmas for multiple tickers efficiently."""
    prices = {}
    sigmas = {}
    for i in range(0, len(tickers), 20):
        batch = tickers[i:i+20]
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
                            else:
                                sigmas[t] = 0.30
                            sigmas[t] = max(0.05, min(sigmas[t], 3.0))
            elif len(batch) == 1 and not data.empty:
                t = batch[0]
                close = data["Close"].dropna()
                if len(close) > 0:
                    prices[t] = float(close.iloc[-1])
                    log_ret = np.log(close / close.shift(1)).dropna()
                    sigmas[t] = (float(log_ret.tail(20).std() * np.sqrt(252))
                                 if len(log_ret) >= 20 else 0.30)
                    sigmas[t] = max(0.05, min(sigmas[t], 3.0))
        except Exception as e:
            log.warning(f"Price batch {i}-{i+len(batch)} failed: {e}")
        time.sleep(0.5)
    return prices, sigmas


# ══════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception as e:
            log.error(f"State load error: {e} - using fresh state")
    return {
        'cash': STARTING_CAPITAL,
        'positions': [],
        'closed_trades': [],
        'realized_pnl': 0.0,
        'trade_count': 0,
        'start_date': date.today().isoformat(),
        'last_check': None,
        'version': 'earnings_vol_v1',
    }


def save_state(state: dict):
    if len(state.get('closed_trades', [])) > 500:
        state['closed_trades'] = state['closed_trades'][-500:]
    tmp = STATE_FILE.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(STATE_FILE)


def compute_nav(state: dict, prices: Dict[str, float],
                sigmas: Dict[str, float]) -> float:
    """NAV = cash + mark-to-market of open strangles (short = liability)."""
    nav = state['cash']
    today = date.today()

    for pos in state['positions']:
        ticker = pos['ticker']
        spot = prices.get(ticker)
        if spot is None:
            continue

        sigma = sigmas.get(ticker, pos.get('sigma_entry', 0.25))
        expiry = date.fromisoformat(pos['expiry'])
        T = max((expiry - today).days, 0) / 365.0

        # FIX (HC #709 audit): Use consistent pricing — same source as entry
        expiry_date = expiry
        put_val = _get_option_price_consistent(ticker, spot, pos['short_put_strike'], T, sigma, expiry_date, kind='put')
        call_val = _get_option_price_consistent(ticker, spot, pos['short_call_strike'], T, sigma, expiry_date, kind='call')
        close_cost = (put_val + call_val) * 100 * pos.get('contracts', 1)
        nav -= close_cost

    return nav


def log_equity(nav: float, realized_pnl: float):
    row = f"{datetime.utcnow().isoformat()},{nav:.2f},{realized_pnl:.2f}\n"
    if not EQUITY_FILE.exists():
        with open(EQUITY_FILE, 'w') as f:
            f.write("timestamp,nav,realized_pnl\n")
    with open(EQUITY_FILE, 'a') as f:
        f.write(row)


def log_trade(info: dict):
    with open(TRADES_FILE, 'a') as f:
        f.write(json.dumps(info, default=str) + '\n')


def load_nav_history() -> list:
    if NAV_HISTORY_FILE.exists():
        try:
            with open(NAV_HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_nav_history(history: list):
    with open(NAV_HISTORY_FILE, 'w') as f:
        json.dump(history, f)


_nav_drop_alerted_today: set = set()


def check_nav_drop(nav: float):
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
                f'"EARN-VOL NAV DROP: {drop_pct:.1%} intraday '
                f'(${sod_nav:,.0f} -> ${nav:,.0f})" 2>/dev/null'
            )
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# FIND EXPIRY FOR STRANGLE
# ══════════════════════════════════════════════════════════════════════

def find_expiry_for_earnings(earnings_date: date, entry_date: date) -> date:
    """
    Find a weekly/monthly expiry AFTER the earnings date.
    Target: first Friday on or after earnings_date + 1 day.
    The strangle will be bought back morning after earnings, so the
    expiry just needs to be beyond the earnings date.
    """
    target = earnings_date + timedelta(days=1)
    # Find next Friday >= target
    d = target
    while d.weekday() != 4:  # Friday = 4
        d += timedelta(days=1)
    # Ensure at least 2 days from entry
    if (d - entry_date).days < 2:
        d += timedelta(days=7)
    return d


# ══════════════════════════════════════════════════════════════════════
# POSITION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

def _is_market_hours() -> bool:
    """Check if current time is within RTH (9:30-16:00 ET)."""
    try:
        import pytz
        et = pytz.timezone('US/Eastern')
        now_et = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(et)
        return ((now_et.hour == 9 and now_et.minute >= 30) or
                (10 <= now_et.hour < 16))
    except Exception:
        return True  # Fail-open


def process_positions(state: dict, prices: Dict[str, float],
                      sigmas: Dict[str, float]):
    """
    Check existing positions for:
    1. Morning-after-earnings close (normal exit)
    2. Stop-loss (2x premium)
    3. Force-close (too many days after earnings)
    """
    today = date.today()
    in_rth = _is_market_hours()
    new_positions = []

    for pos in state['positions']:
        ticker = pos['ticker']
        if ticker not in prices:
            new_positions.append(pos)
            continue

        S = prices[ticker]
        sigma = sigmas.get(ticker, pos.get('sigma_entry', 0.25))
        expiry = date.fromisoformat(pos['expiry'])
        T = max((expiry - today).days, 0) / 365.0
        contracts = pos.get('contracts', 1)
        net_credit = pos['net_credit']  # per-share credit received
        total_credit = net_credit * 100 * contracts

        # FIX (HC #709 audit): Use consistent pricing — same source as entry
        expiry_date = expiry
        put_val = _get_option_price_consistent(ticker, S, pos['short_put_strike'], T, sigma, expiry_date, kind='put')
        call_val = _get_option_price_consistent(ticker, S, pos['short_call_strike'], T, sigma, expiry_date, kind='call')

        close_cost_per_share = put_val + call_val
        close_cost = close_cost_per_share * 100 * contracts
        close_fees = 2 * COST_PER_CONTRACT * contracts

        # P&L = credit received - cost to close - fees
        pnl = total_credit - close_cost - close_fees
        profit_pct = pnl / total_credit if total_credit > 0 else 0

        # FIX (HC #709 audit): Anti-churn cooldown — don't close within 24h of open
        from datetime import timedelta as _td
        entry_dt = datetime.fromisoformat(pos.get('entry_date', '2020-01-01'))
        hours_held = (datetime.utcnow() - entry_dt).total_seconds() / 3600
        if hours_held < 24:
            new_positions.append(pos)
            continue

        # ── Stop-loss: 2x premium ──
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
                'earnings_date': pos.get('earnings_date', ''),
                'close_date': datetime.utcnow().isoformat(),
            })
            log_trade({
                'action': 'stop_loss', 'ticker': ticker,
                'short_put': pos['short_put_strike'],
                'short_call': pos['short_call_strike'],
                'pnl': round(pnl, 2), 'profit_pct': round(profit_pct, 3),
                'contracts': contracts,
                'earnings_date': pos.get('earnings_date', ''),
                'spot_at_close': round(S, 2),
                'time': datetime.utcnow().isoformat(),
            })
            continue

        # ── Morning-after-earnings close (normal exit) ──
        if should_close_position(pos, today) and in_rth:
            state['cash'] -= (close_cost + close_fees)
            state['realized_pnl'] += pnl
            state['trade_count'] += 1
            log.info(f"EARNINGS CLOSE {ticker} strangle (post-earnings). "
                     f"P:{pos['short_put_strike']:.0f} C:{pos['short_call_strike']:.0f} "
                     f"PnL: ${pnl:.2f} ({profit_pct:.0%})")
            state['closed_trades'].append({
                'ticker': ticker, 'action': 'earnings_close',
                'pnl': round(pnl, 2), 'profit_pct': round(profit_pct, 3),
                'contracts': contracts,
                'earnings_date': pos.get('earnings_date', ''),
                'close_date': datetime.utcnow().isoformat(),
            })
            log_trade({
                'action': 'earnings_close', 'ticker': ticker,
                'short_put': pos['short_put_strike'],
                'short_call': pos['short_call_strike'],
                'pnl': round(pnl, 2), 'profit_pct': round(profit_pct, 3),
                'contracts': contracts,
                'earnings_date': pos.get('earnings_date', ''),
                'spot_at_close': round(S, 2),
                'spot_at_entry': pos.get('spot_entry', 0),
                'move_pct': round((S - pos.get('spot_entry', S)) / pos.get('spot_entry', S) * 100, 2)
                if pos.get('spot_entry') else 0,
                'time': datetime.utcnow().isoformat(),
            })
            continue

        # ── Force-close (too many days after earnings) ──
        if should_force_close(pos, today) and in_rth:
            state['cash'] -= (close_cost + close_fees)
            state['realized_pnl'] += pnl
            state['trade_count'] += 1
            log.warning(f"FORCE CLOSE {ticker} strangle (>3d after earnings). "
                        f"PnL: ${pnl:.2f}")
            state['closed_trades'].append({
                'ticker': ticker, 'action': 'force_close',
                'pnl': round(pnl, 2), 'contracts': contracts,
                'earnings_date': pos.get('earnings_date', ''),
                'close_date': datetime.utcnow().isoformat(),
            })
            log_trade({
                'action': 'force_close', 'ticker': ticker,
                'pnl': round(pnl, 2), 'contracts': contracts,
                'time': datetime.utcnow().isoformat(),
            })
            continue

        # Position survives
        new_positions.append(pos)

    state['positions'] = new_positions


# ══════════════════════════════════════════════════════════════════════
# NEW POSITION ENTRY
# ══════════════════════════════════════════════════════════════════════

def open_new_positions(state: dict, prices: Dict[str, float],
                       sigmas: Dict[str, float],
                       earnings_calendar: Dict[str, List[str]]):
    """
    Open new strangle positions on tickers with earnings in ~2 trading days.
    """
    if not _is_market_hours():
        log.info("OFF-HOURS - no new entries")
        return

    today = date.today()
    candidates = get_entry_candidates(earnings_calendar, today)

    if not candidates:
        log.info("No earnings candidates in the entry window today")
        return

    log.info(f"Entry candidates: {list(candidates.keys())}")

    active_tickers = {pos['ticker'] for pos in state['positions']}
    if len(state['positions']) >= MAX_CONCURRENT:
        log.info(f"Max positions ({MAX_CONCURRENT}) reached - no new entries")
        return

    nav = compute_nav(state, prices, sigmas)
    opened = 0

    for ticker, earnings_date in candidates.items():
        if len(state['positions']) >= MAX_CONCURRENT:
            break

        if ticker in active_tickers:
            log.info(f"  {ticker}: already have open position - skip")
            continue

        if ticker in AVOID_TICKERS:
            continue

        if ticker not in prices:
            log.info(f"  {ticker}: no price data - skip")
            continue

        S = prices[ticker]
        sigma = sigmas.get(ticker, 0.25)

        if S < 5.0:
            log.info(f"  {ticker}: spot ${S:.2f} too low - skip")
            continue

        # Find strikes: 10% OTM
        K_put = round(S * (1 - OTM_PCT) * 2) / 2   # nearest $0.50
        K_call = round(S * (1 + OTM_PCT) * 2) / 2   # nearest $0.50

        # Sanity
        if K_put >= S or K_call <= S:
            log.warning(f"  {ticker}: bad strikes P:{K_put} C:{K_call} (S={S}) - skip")
            continue

        # Find expiry (first Friday after earnings)
        expiry = find_expiry_for_earnings(earnings_date, today)
        T = max((expiry - today).days, 1) / 365.0

        # Price the legs
        if _HAS_PRICING_BRIDGE:
            try:
                prem_put = _alpaca_get_premium(
                    ticker, S, K_put, T, sigma, expiry,
                    kind='put', bs_price_fn=bs_price)
                prem_call = _alpaca_get_premium(
                    ticker, S, K_call, T, sigma, expiry,
                    kind='call', bs_price_fn=bs_price)
            except Exception:
                prem_put = bs_price(S, K_put, T, sigma, kind='put')
                prem_call = bs_price(S, K_call, T, sigma, kind='call')
        else:
            prem_put = bs_price(S, K_put, T, sigma, kind='put')
            prem_call = bs_price(S, K_call, T, sigma, kind='call')

        # Apply slippage (we receive less when selling)
        prem_put_slip = apply_slippage(prem_put)
        prem_call_slip = apply_slippage(prem_call)
        credit_per_share = prem_put_slip + prem_call_slip

        if credit_per_share < 0.10:
            log.info(f"  {ticker}: credit ${credit_per_share:.3f}/sh too low - skip")
            continue

        n_contracts = CONTRACTS_PER_STOCK
        open_fees = 2 * COST_PER_CONTRACT * n_contracts  # 2 legs
        total_credit = credit_per_share * 100 * n_contracts - open_fees

        if total_credit <= 0:
            log.info(f"  {ticker}: net credit after fees is negative - skip")
            continue

        # Execute: receive credit
        state['cash'] += total_credit

        position = {
            'ticker': ticker,
            'entry_date': today.isoformat(),
            'earnings_date': earnings_date.isoformat(),
            'expiry': expiry.isoformat(),
            'short_put_strike': K_put,
            'short_call_strike': K_call,
            'net_credit': round(credit_per_share, 4),
            'contracts': n_contracts,
            'spot_entry': round(S, 2),
            'sigma_entry': round(sigma, 4),
            'total_credit': round(total_credit, 2),
        }
        state['positions'].append(position)
        active_tickers.add(ticker)
        opened += 1

        log.info(f"SELL STRANGLE {ticker} "
                 f"P:{K_put:.0f} C:{K_call:.0f} "
                 f"(earnings {earnings_date}, expiry {expiry}, {n_contracts}c), "
                 f"credit ${total_credit:.2f}, "
                 f"spot=${S:.2f}, sigma={sigma:.2%}")
        log_trade({
            'action': 'sell_strangle', 'ticker': ticker,
            'short_put': K_put, 'short_call': K_call,
            'net_credit_per_share': round(credit_per_share, 4),
            'total_credit': round(total_credit, 2),
            'contracts': n_contracts,
            'earnings_date': earnings_date.isoformat(),
            'expiry': expiry.isoformat(),
            'sigma': round(sigma, 4),
            'spot': round(S, 2),
            'time': datetime.utcnow().isoformat(),
        })

        # HC #702: pricing audit
        if _HAS_PRICING_LOGGER:
            for _k, _kind, _bs_val in [(K_put, 'put', prem_put),
                                        (K_call, 'call', prem_call)]:
                log_option_price(
                    engine="earnings_vol", ticker=ticker, spot=S,
                    strike=_k, expiry=expiry.isoformat()[:10],
                    option_type=_kind, iv_used=sigma,
                    bs_price=_bs_val, source="bs_or_alpaca",
                    action="entry"
                )

    if opened:
        log.info(f"Opened {opened} new earnings strangles "
                 f"(total positions: {len(state['positions'])})")


# ══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════

def run_cycle(earnings_calendar: Dict[str, List[str]]):
    """Run one full check cycle."""
    state = load_state()

    # Fetch prices for positions + universe
    position_tickers = [pos['ticker'] for pos in state['positions']]
    needed = list(set(position_tickers + UNIVERSE))
    prices, sigmas = get_batch_prices(needed)

    nav = compute_nav(state, prices, sigmas)
    log.info(f"NAV=${nav:,.0f}, cash=${state['cash']:,.0f}, "
             f"positions={len(state['positions'])}, "
             f"realized=${state['realized_pnl']:.2f}")

    # NAV history
    nav_history = load_nav_history()
    nav_history.append({"date": datetime.utcnow().isoformat(),
                        "nav": round(nav, 2)})
    nav_history = nav_history[-120:]
    save_nav_history(nav_history)

    # Process existing positions (stop-loss, earnings close, force close)
    process_positions(state, prices, sigmas)

    # Open new positions for upcoming earnings
    open_new_positions(state, prices, sigmas, earnings_calendar)

    # Final NAV
    nav = compute_nav(state, prices, sigmas)
    check_nav_drop(nav)
    log_equity(nav, state['realized_pnl'])

    state['last_check'] = datetime.utcnow().isoformat()
    save_state(state)

    # Summary
    pos_summary = ", ".join(
        f"{p['ticker']}(earn:{p.get('earnings_date', '?')})"
        for p in state['positions']
    ) or "none"
    log.info(f"Cycle end: NAV=${nav:,.0f}, positions=[{pos_summary}], "
             f"trades={state['trade_count']}, "
             f"realized=${state['realized_pnl']:.2f}")


def main():
    log.info("=" * 60)
    log.info("EARNINGS VOL SELLING PAPER ENGINE")
    log.info(f"  Universe: {len(UNIVERSE)} tickers (best from backtest)")
    log.info(f"  Avoid: {sorted(AVOID_TICKERS)}")
    log.info(f"  Strategy: sell 10% OTM strangle 2d before earnings, "
             f"close morning after")
    log.info(f"  Stop-loss: {STOP_LOSS_MULT:.0f}x premium")
    log.info(f"  Max positions: {MAX_CONCURRENT}, "
             f"{CONTRACTS_PER_STOCK} contract/stock")
    log.info(f"  Cost: ${COST_PER_CONTRACT}/contract/leg, "
             f"{SLIPPAGE_FRAC:.1%} slippage")
    log.info(f"  Capital: ${STARTING_CAPITAL:,.0f}")
    log.info(f"  Backtest: Sharpe 2.71, R1 PASS, permutation PASS")
    log.info("=" * 60)

    # Initialize pricing bridge
    if _HAS_PRICING_BRIDGE:
        _init_bridge(engine_name='earnings-vol')
        log.info("Alpaca real pricing bridge initialized")
    else:
        log.info("Alpaca pricing bridge not available, using BS-only pricing")

    # Refresh earnings calendar on startup
    earnings_calendar = refresh_earnings_calendar()

    cycle_count = 0
    last_calendar_refresh = date.today()

    while True:
        try:
            now = datetime.utcnow()
            weekday = now.weekday()
            hour_utc = now.hour + now.minute / 60

            # Refresh earnings calendar once per day
            if date.today() != last_calendar_refresh:
                earnings_calendar = refresh_earnings_calendar()
                last_calendar_refresh = date.today()

            # Market hours: Mon-Fri, 13:30-21:00 UTC (9:30 AM - 4 PM ET)
            if weekday < 5 and 13.5 <= hour_utc <= 21.0:
                run_cycle(earnings_calendar)
                cycle_count += 1
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
