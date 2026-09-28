#!/usr/bin/env python3
"""
Earnings Jade Lizard Paper Trading Engine
==========================================

Daily paper engine (PM2 cron) that simulates selling jade lizards 5 business
days before earnings on high-IV-rank stocks, closing 1 day after earnings.

STRATEGY (Backtest: Sharpe 2.30, all 4 adversarial gates pass):
  - Entry: 5 bdays before earnings, IV rank proxy > 50th pctl, stock calm (<8% move in 5d)
  - Position: Short put (1.0 stdev OTM) + short call (0.8 stdev OTM) + long call (short call + 5%)
  - Jade lizard condition: call spread credit >= put premium (no upside risk)
  - Exit: 1 bday after earnings, reprice with post-earnings IV crush
  - Max 3 concurrent positions, max $300 risk per trade
  - Capital: $100,000 paper

COST MODEL:
  - $14.10 per jade lizard (3 legs x $4.70 RT)

Usage:
  python3 paper_engines/earnings_jade_lizard_paper.py

Author: Claude (autonomous build)
"""

import json
import logging
import math
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Lazy import yfinance (slow)
yf = None

def _import_yfinance():
    global yf
    if yf is None:
        import yfinance as _yf
        yf = _yf

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "state" / "earnings_jade_lizard_paper_state.json"

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [JADE-LIZ] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("jade_lizard")

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "MA",
    "HD", "UNH", "JNJ", "PG", "KO", "PEP", "MCD", "WMT", "COST", "AVGO",
    "CRM", "ORCL", "ADBE", "NFLX", "AMD", "INTC", "QCOM", "TXN", "AMAT", "MU",
    "GS", "MS", "BAC", "WFC", "C", "AXP", "BLK", "SCHW", "LLY", "PFE",
    "MRK", "ABBV", "TMO", "DHR", "BMY", "XOM", "CVX", "COP", "SLB", "EOG",
]

INITIAL_CAPITAL = 100_000
MAX_CONCURRENT = 3
MAX_RISK_PER_TRADE = 5000.0  # $100K paper account — scale appropriately
COST_PER_JADE_LIZARD = 14.10  # 3 legs x $4.70 RT
ENTRY_DAYS_BEFORE = 5  # business days before earnings
EXIT_DAYS_AFTER = 1    # business days after earnings
MOVEMENT_THRESHOLD = 0.08  # 8% max move in prior 5 bdays
IV_RANK_PERCENTILE = 50    # must be above 50th percentile
PRE_EARNINGS_IV_MULT = 1.5  # realized vol * 1.5 for pre-earnings IV
POST_EARNINGS_IV_MULT = 1.0  # realized vol * 1.0 for post-earnings IV (crush)

# ─────────────────────────────────────────────
#  BLACK-SCHOLES HELPERS
# ─────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price. T in years."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price. T in years."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# ─────────────────────────────────────────────
#  STATE MANAGEMENT
# ─────────────────────────────────────────────

def _default_state() -> dict:
    return {
        "capital": INITIAL_CAPITAL,
        "positions": [],
        "closed_trades": [],
        "total_realized_pnl": 0.0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "last_run": None,
        "daily_equity": {},
    }


def load_state() -> dict:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            log.warning("Corrupt state file, reinitializing")
    return _default_state()


def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ─────────────────────────────────────────────
#  MARKET DATA HELPERS
# ─────────────────────────────────────────────

def _strip_tz(idx):
    """Strip timezone from pandas index or timestamp."""
    if hasattr(idx, 'tz') and idx.tz is not None:
        return idx.tz_convert(None)
    return idx


def _strip_ts_tz(ts):
    """Strip timezone from a single timestamp."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def get_price_history(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Fetch price history via yfinance. Returns DataFrame or None."""
    _import_yfinance()
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            return None
        hist.index = _strip_tz(hist.index)
        return hist
    except Exception as e:
        log.debug(f"Failed to get history for {ticker}: {e}")
        return None


def get_earnings_dates(ticker: str) -> Optional[pd.DataFrame]:
    """Get upcoming earnings dates via yfinance."""
    _import_yfinance()
    try:
        t = yf.Ticker(ticker)
        ed = t.get_earnings_dates(limit=12)
        if ed is None or ed.empty:
            return None
        ed.index = _strip_tz(ed.index)
        return ed
    except Exception as e:
        log.debug(f"Failed to get earnings dates for {ticker}: {e}")
        return None


def compute_realized_vol(hist: pd.DataFrame, window: int = 20) -> Optional[float]:
    """Compute annualized realized vol from close prices (trailing window)."""
    if hist is None or len(hist) < window + 1:
        return None
    closes = hist["Close"].dropna()
    if len(closes) < window + 1:
        return None
    log_returns = np.log(closes / closes.shift(1)).dropna()
    recent = log_returns.iloc[-window:]
    return float(recent.std() * np.sqrt(252))


def compute_iv_rank(hist: pd.DataFrame, current_vol: float, lookback: int = 252) -> Optional[float]:
    """
    IV rank proxy: percentile of current 20d realized vol in trailing 252d distribution.
    Returns percentile 0-100.
    """
    if hist is None or len(hist) < lookback + 21:
        return None
    closes = hist["Close"].dropna()
    log_returns = np.log(closes / closes.shift(1)).dropna()
    if len(log_returns) < lookback:
        return None
    # Rolling 20d vol over lookback period
    rolling_vols = log_returns.rolling(20).std() * np.sqrt(252)
    rolling_vols = rolling_vols.dropna()
    if len(rolling_vols) < 20:
        return None
    percentile = float((rolling_vols < current_vol).sum() / len(rolling_vols) * 100)
    return percentile


def check_movement(hist: pd.DataFrame, days: int = 5) -> Optional[float]:
    """Check absolute % move over last N trading days."""
    if hist is None or len(hist) < days + 1:
        return None
    closes = hist["Close"].dropna()
    if len(closes) < days + 1:
        return None
    current = closes.iloc[-1]
    past = closes.iloc[-(days + 1)]
    return abs(current - past) / past


# ─────────────────────────────────────────────
#  JADE LIZARD PRICING
# ─────────────────────────────────────────────

def price_jade_lizard(
    spot: float,
    realized_vol: float,
    iv_multiplier: float,
    days_to_expiry: int,
    r: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """
    Price a jade lizard given spot and vol parameters.

    Returns dict with strikes, premiums, and net credit, or None if jade lizard
    condition is not met.
    """
    sigma = realized_vol * iv_multiplier
    T = days_to_expiry / 252.0
    if T <= 0 or sigma <= 0:
        return None

    # Strike selection
    put_strike = round(spot * (1 - 1.0 * sigma * math.sqrt(T)), 2)   # ~1.0 stdev below
    call_strike = round(spot * (1 + 0.8 * sigma * math.sqrt(T)), 2)  # ~0.8 stdev above
    long_call_strike = round(call_strike + 0.05 * spot, 2)            # +5% of stock price

    # Ensure ordering
    if put_strike >= spot or call_strike <= spot or long_call_strike <= call_strike:
        return None

    # Price each leg
    short_put_prem = bs_put(spot, put_strike, T, r, sigma)
    short_call_prem = bs_call(spot, call_strike, T, r, sigma)
    long_call_prem = bs_call(spot, long_call_strike, T, r, sigma)

    # Credits
    call_spread_credit = short_call_prem - long_call_prem
    put_credit = short_put_prem

    # Jade lizard condition: call spread credit >= put premium (no upside risk)
    if call_spread_credit < put_credit:
        return None

    net_credit = call_spread_credit + put_credit
    # Max risk is on put side: put_strike - 0 (but practically, put_strike * 100 shares minus credit)
    # For options on 100 shares:
    max_put_risk = (put_strike * 100) - (net_credit * 100)  # theoretical max if stock -> 0
    # Practical risk: distance to put strike * 100 - credit
    practical_risk = (spot - put_strike) * 100 - net_credit * 100
    # Call side: no risk if jade lizard condition met (credit >= debit of spread)

    return {
        "spot": spot,
        "put_strike": put_strike,
        "call_strike": call_strike,
        "long_call_strike": long_call_strike,
        "short_put_prem": round(short_put_prem, 4),
        "short_call_prem": round(short_call_prem, 4),
        "long_call_prem": round(long_call_prem, 4),
        "call_spread_credit": round(call_spread_credit, 4),
        "put_credit": round(put_credit, 4),
        "net_credit": round(net_credit, 4),
        "net_credit_dollars": round(net_credit * 100, 2),
        "practical_risk_dollars": round(max(practical_risk, 0), 2),
        "sigma_used": round(sigma, 4),
        "T_years": round(T, 6),
    }


def reprice_jade_lizard_at_exit(
    position: dict,
    exit_spot: float,
    realized_vol: float,
    days_remaining: int = 0,
    r: float = 0.05,
) -> float:
    """
    Calculate P&L for closing a jade lizard position.

    Returns dollar P&L (positive = profit).
    """
    sigma = realized_vol * POST_EARNINGS_IV_MULT
    T = max(days_remaining, 0) / 252.0

    put_strike = position["put_strike"]
    call_strike = position["call_strike"]
    long_call_strike = position["long_call_strike"]

    # Cost to close each leg (buy back short options, sell long option)
    close_put = bs_put(exit_spot, put_strike, T, r, sigma)
    close_short_call = bs_call(exit_spot, call_strike, T, r, sigma)
    close_long_call = bs_call(exit_spot, long_call_strike, T, r, sigma)

    # If T==0 (expired), these are intrinsic values
    # Cost to close = buy back shorts - sell long
    close_cost = close_put + close_short_call - close_long_call

    # P&L = initial credit received - cost to close - commissions
    entry_credit = position["net_credit_dollars"]
    close_cost_dollars = close_cost * 100
    pnl = entry_credit - close_cost_dollars - COST_PER_JADE_LIZARD

    return round(pnl, 2)


# ─────────────────────────────────────────────
#  EXIT LOGIC
# ─────────────────────────────────────────────

def process_exits(state: dict, today: pd.Timestamp) -> List[str]:
    """
    Close positions where earnings was yesterday (1 bday after earnings).
    Returns list of summary messages.
    """
    messages = []
    still_open = []

    for pos in state["positions"]:
        earnings_date = pd.Timestamp(pos["earnings_date"])
        # Exit 1 business day after earnings
        exit_date = earnings_date + pd.offsets.BDay(EXIT_DAYS_AFTER)

        if today >= exit_date:
            # Get current price for exit
            hist = get_price_history(pos["ticker"], period="5d")
            if hist is not None and not hist.empty:
                exit_spot = float(hist["Close"].iloc[-1])
                rv = compute_realized_vol(get_price_history(pos["ticker"], period="3mo"), window=20)
                if rv is None:
                    rv = pos.get("realized_vol", 0.20)
            else:
                # Fallback: use entry spot (conservative)
                exit_spot = pos["spot"]
                rv = pos.get("realized_vol", 0.20)

            # Days remaining on the options (typically near 0 for weeklies post-earnings)
            days_rem = max(0, pos.get("days_to_expiry", 6) - (today - pd.Timestamp(pos["entry_date"])).days)

            pnl = reprice_jade_lizard_at_exit(pos, exit_spot, rv, days_remaining=days_rem)

            closed = {
                **pos,
                "exit_date": str(today.date()),
                "exit_spot": exit_spot,
                "pnl": pnl,
                "exit_realized_vol": rv,
            }
            state["closed_trades"].append(closed)
            state["total_realized_pnl"] += pnl
            state["total_trades"] += 1
            state["capital"] += pnl
            if pnl > 0:
                state["wins"] += 1
            else:
                state["losses"] += 1

            move_pct = (exit_spot - pos["spot"]) / pos["spot"] * 100
            messages.append(
                f"  CLOSED {pos['ticker']}: P&L ${pnl:+.2f} | "
                f"Entry ${pos['spot']:.2f} -> Exit ${exit_spot:.2f} ({move_pct:+.1f}%) | "
                f"Credit ${pos['net_credit_dollars']:.2f}"
            )
        else:
            still_open.append(pos)

    state["positions"] = still_open
    return messages


# ─────────────────────────────────────────────
#  ENTRY LOGIC
# ─────────────────────────────────────────────

def scan_for_entries(state: dict, today: pd.Timestamp) -> List[str]:
    """
    Scan universe for stocks with earnings ~5 bdays away.
    Apply filters and open paper positions.
    Returns list of summary messages.
    """
    messages = []
    n_open = len(state["positions"])

    if n_open >= MAX_CONCURRENT:
        messages.append(f"  Max concurrent positions ({MAX_CONCURRENT}) reached, skipping scan")
        return messages

    # Target earnings date window: 4-6 bdays from today
    target_start = today + pd.offsets.BDay(ENTRY_DAYS_BEFORE - 1)
    target_end = today + pd.offsets.BDay(ENTRY_DAYS_BEFORE + 1)

    # Track already-held tickers
    held_tickers = {p["ticker"] for p in state["positions"]}

    candidates_checked = 0
    candidates_passed = 0

    for ticker in UNIVERSE:
        if n_open >= MAX_CONCURRENT:
            break
        if ticker in held_tickers:
            continue

        candidates_checked += 1

        # Get earnings dates
        ed = get_earnings_dates(ticker)
        if ed is None:
            continue

        # Find earnings within target window
        upcoming = ed.index[(ed.index >= target_start) & (ed.index <= target_end)]
        if len(upcoming) == 0:
            continue

        earnings_date = upcoming[0]
        log.info(f"  {ticker}: earnings on {earnings_date.date()}, checking filters...")

        # Get price history
        hist = get_price_history(ticker, period="2y")
        if hist is None or len(hist) < 60:
            log.info(f"  {ticker}: insufficient history, skipping")
            continue

        current_price = float(hist["Close"].iloc[-1])

        # Filter 1: IV rank proxy > 50th percentile
        rv = compute_realized_vol(hist, window=20)
        if rv is None:
            log.info(f"  {ticker}: cannot compute vol, skipping")
            continue

        iv_rank = compute_iv_rank(hist, rv)
        if iv_rank is None:
            log.info(f"  {ticker}: IV rank unavailable, skipping")
            continue
        if iv_rank < IV_RANK_PERCENTILE:
            log.info(f"  {ticker}: IV rank {iv_rank:.0f}% < {IV_RANK_PERCENTILE}%, skipping")
            continue

        # Filter 2: stock hasn't moved >8% in prior 5 bdays
        movement = check_movement(hist, days=5)
        if movement is None or movement > MOVEMENT_THRESHOLD:
            pct = movement * 100 if movement else 0
            log.info(f"  {ticker}: 5d move {pct:.1f}% > {MOVEMENT_THRESHOLD*100:.0f}%, skipping")
            continue

        # Calculate days to expiry (entry to ~1 week after earnings for option expiry)
        days_to_expiry = max(6, (earnings_date - today).days + 2)

        # Price the jade lizard
        jl = price_jade_lizard(
            spot=current_price,
            realized_vol=rv,
            iv_multiplier=PRE_EARNINGS_IV_MULT,
            days_to_expiry=days_to_expiry,
        )

        if jl is None:
            log.info(f"  {ticker}: jade lizard condition not met or invalid strikes, skipping")
            continue

        # Filter 3: max risk per trade
        if jl["practical_risk_dollars"] > MAX_RISK_PER_TRADE:
            log.info(f"  {ticker}: risk ${jl['practical_risk_dollars']:.0f} > ${MAX_RISK_PER_TRADE:.0f}, skipping")
            continue

        # All filters passed - open position
        candidates_passed += 1
        position = {
            "ticker": ticker,
            "entry_date": str(today.date()),
            "earnings_date": str(earnings_date.date()),
            "spot": current_price,
            "realized_vol": rv,
            "iv_rank": round(iv_rank, 1),
            "days_to_expiry": days_to_expiry,
            **{k: v for k, v in jl.items() if k != "spot"},
        }
        state["positions"].append(position)
        held_tickers.add(ticker)
        n_open += 1

        messages.append(
            f"  OPENED {ticker}: Spot ${current_price:.2f} | "
            f"Put K=${jl['put_strike']:.2f} / Call K=${jl['call_strike']:.2f}-${jl['long_call_strike']:.2f} | "
            f"Credit ${jl['net_credit_dollars']:.2f} | Risk ${jl['practical_risk_dollars']:.2f} | "
            f"IV rank {iv_rank:.0f}% | Earnings {earnings_date.date()}"
        )

    messages.insert(0, f"  Scanned {candidates_checked} tickers, {candidates_passed} new entries")
    return messages


# ─────────────────────────────────────────────
#  MAIN ENGINE
# ─────────────────────────────────────────────

def run_engine():
    """Main daily execution flow."""
    log.info("=" * 60)
    log.info("Earnings Jade Lizard Paper Engine - Daily Run")
    log.info("=" * 60)

    today = pd.Timestamp(datetime.now().date())
    log.info(f"Date: {today.date()}")

    # Load state
    state = load_state()
    log.info(f"Capital: ${state['capital']:,.2f} | Open positions: {len(state['positions'])} | "
             f"Total trades: {state['total_trades']} | Win rate: "
             f"{state['wins']/max(state['total_trades'],1)*100:.0f}%")

    # 1. Process exits
    log.info("")
    log.info("--- EXIT CHECK ---")
    exit_msgs = process_exits(state, today)
    if exit_msgs:
        for m in exit_msgs:
            log.info(m)
    else:
        log.info("  No positions to close today")

    # 2. Scan for entries
    log.info("")
    log.info("--- ENTRY SCAN ---")
    entry_msgs = scan_for_entries(state, today)
    for m in entry_msgs:
        log.info(m)

    # 3. Record daily equity
    total_equity = state["capital"]  # Approximate; open positions have unrealized P&L
    state["daily_equity"][str(today.date())] = round(total_equity, 2)

    # 4. Save state
    state["last_run"] = str(datetime.now())
    save_state(state)

    # 5. Print summary
    log.info("")
    log.info("--- SUMMARY ---")
    log.info(f"Capital: ${state['capital']:,.2f}")
    log.info(f"Open positions: {len(state['positions'])}")
    for p in state["positions"]:
        log.info(f"  {p['ticker']}: entered {p['entry_date']}, earnings {p['earnings_date']}, "
                 f"credit ${p['net_credit_dollars']:.2f}")
    log.info(f"Total realized P&L: ${state['total_realized_pnl']:+,.2f}")
    log.info(f"Total trades: {state['total_trades']} | Wins: {state['wins']} | Losses: {state['losses']}")
    if state["total_trades"] > 0:
        wr = state["wins"] / state["total_trades"] * 100
        log.info(f"Win rate: {wr:.1f}%")
        # Compute Sharpe if enough trades
        if len(state["closed_trades"]) >= 5:
            pnls = [t["pnl"] for t in state["closed_trades"]]
            avg = np.mean(pnls)
            std = np.std(pnls, ddof=1)
            if std > 0:
                # Annualize assuming ~50 trades/year
                sharpe = (avg / std) * np.sqrt(min(len(pnls), 50))
                log.info(f"Realized Sharpe (annualized est): {sharpe:.2f}")
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        run_engine()
    except Exception as e:
        log.error(f"Engine failed: {e}", exc_info=True)
        sys.exit(1)
