#!/usr/bin/env python3
"""
Pre-Earnings Vol Crush Iron Condor Paper Trading Engine
=======================================================

Daily paper engine that simulates selling iron condors 3 business days before
earnings, closing 1 business day after earnings to capture the vol crush.

STRATEGY (Backtest: Sharpe 1.76, 3/4 gates pass):
  - Entry: 3 bdays before earnings
  - Short put: 1.0 stdev below price (~20-delta)
  - Short call: 0.8 stdev above price (~25-delta)
  - Long put: 5% below short put strike (wing)
  - Long call: 5% above short call strike (wing)
  - IV proxy: realized vol * 1.5 (pre-earnings elevation)
  - Black-Scholes pricing for premiums
  - Filters: IV rank > 50th pctl, stock < 5% move in 5d,
             max 3 concurrent, max $5000 risk per trade
  - Exit: 1 bday after earnings, reprice with IV mult 1.0 (vol crush)
  - If stock beyond short strikes at exit, intrinsic loss applies
  - Cost: $18.80 per IC (4 legs x $4.70 RT)
  - Capital: $100,000 paper

Usage:
  python3 paper_engines/vol_crush_paper.py

Author: Claude (autonomous build)
"""

import json
import logging
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
STATE_FILE = ROOT / "state" / "vol_crush_paper_state.json"

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [VOL-CRUSH-IC] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("vol_crush_ic")

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
MAX_RISK_PER_TRADE = 5_000.0     # dollars
COST_PER_IC = 18.80              # 4 legs x $4.70 RT
ENTRY_DAYS_BEFORE = 3            # business days before earnings
EXIT_DAYS_AFTER = 1              # business days after earnings
MOVEMENT_THRESHOLD = 0.05        # 5% max move in prior 5 trading days
IV_RANK_PERCENTILE = 50          # must be above 50th percentile
PRE_EARNINGS_IV_MULT = 1.5       # realized vol * 1.5 for pre-earnings IV
POST_EARNINGS_IV_MULT = 1.0      # realized vol * 1.0 for post-earnings IV (crush)
WING_WIDTH_PCT = 0.05            # long strikes 5% beyond short strikes
PUT_STDEV = 1.0                  # short put distance in stdevs (~20-delta)
CALL_STDEV = 0.8                 # short call distance in stdevs (~25-delta)
RISK_FREE_RATE = 0.05


# ─────────────────────────────────────────────
#  BLACK-SCHOLES HELPERS
# ─────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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

def _strip_ts_tz(ts):
    """Strip timezone from a single timestamp."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def get_price_history(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Fetch price history via yfinance."""
    _import_yfinance()
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            return None
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)
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
        # Strip tz from each index entry individually (earnings dates can have mixed tz)
        naive_dates = []
        for d in ed.index:
            ed_naive = pd.Timestamp(d).tz_localize(None) if pd.Timestamp(d).tzinfo is not None else pd.Timestamp(d)
            naive_dates.append(ed_naive)
        ed.index = pd.DatetimeIndex(naive_dates)
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
    IV rank proxy: percentile of current 20d realized vol in trailing 252d
    distribution of rolling 20d vols. Returns percentile 0-100.
    """
    if hist is None or len(hist) < lookback + 21:
        return None
    closes = hist["Close"].dropna()
    log_returns = np.log(closes / closes.shift(1)).dropna()
    if len(log_returns) < lookback:
        return None
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
#  IRON CONDOR PRICING
# ─────────────────────────────────────────────

def price_iron_condor(
    spot: float,
    realized_vol: float,
    iv_multiplier: float,
    days_to_expiry: int,
    r: float = RISK_FREE_RATE,
) -> Optional[Dict[str, Any]]:
    """
    Price an iron condor (sell IC) given spot and vol parameters.

    Structure (all OTM):
      Long put (wing) | Short put (1.0 stdev) | --- SPOT --- | Short call (0.8 stdev) | Long call (wing)
      Wings are 5% beyond the respective short strike.

    Returns dict with strikes, premiums, net credit, max risk, or None if invalid.
    """
    sigma = realized_vol * iv_multiplier
    T = days_to_expiry / 252.0
    if T <= 0 or sigma <= 0:
        return None

    sqrt_T = math.sqrt(T)

    # Short strikes (asymmetric: put wider than call per strategy spec)
    short_put = round(spot * (1 - PUT_STDEV * sigma * sqrt_T), 2)
    short_call = round(spot * (1 + CALL_STDEV * sigma * sqrt_T), 2)

    # Long strikes (wings): 5% below/above the short strikes
    long_put = round(short_put * (1 - WING_WIDTH_PCT), 2)
    long_call = round(short_call * (1 + WING_WIDTH_PCT), 2)

    # Validate ordering: LP < SP < spot < SC < LC
    if long_put <= 0 or long_put >= short_put:
        return None
    if short_put >= spot or spot >= short_call:
        return None
    if short_call >= long_call:
        return None

    # Price each leg with elevated IV
    short_put_prem = bs_put(spot, short_put, T, r, sigma)
    long_put_prem = bs_put(spot, long_put, T, r, sigma)
    short_call_prem = bs_call(spot, short_call, T, r, sigma)
    long_call_prem = bs_call(spot, long_call, T, r, sigma)

    # Net credit = (short put - long put) + (short call - long call)
    put_spread_credit = short_put_prem - long_put_prem
    call_spread_credit = short_call_prem - long_call_prem
    net_credit = put_spread_credit + call_spread_credit

    if net_credit <= 0:
        return None

    # Max risk = wider wing width - net credit (per share), times 100 for 1 contract
    put_wing_width = short_put - long_put
    call_wing_width = long_call - short_call
    max_wing_width = max(put_wing_width, call_wing_width)
    max_risk_per_share = max_wing_width - net_credit
    max_risk_dollars = max_risk_per_share * 100

    if max_risk_dollars <= 0:
        max_risk_dollars = 0.01

    return {
        "spot": spot,
        "short_put": short_put,
        "long_put": long_put,
        "short_call": short_call,
        "long_call": long_call,
        "short_put_prem": round(short_put_prem, 4),
        "long_put_prem": round(long_put_prem, 4),
        "short_call_prem": round(short_call_prem, 4),
        "long_call_prem": round(long_call_prem, 4),
        "put_spread_credit": round(put_spread_credit, 4),
        "call_spread_credit": round(call_spread_credit, 4),
        "net_credit": round(net_credit, 4),
        "net_credit_dollars": round(net_credit * 100, 2),
        "put_wing_width": round(put_wing_width, 2),
        "call_wing_width": round(call_wing_width, 2),
        "max_risk_dollars": round(max_risk_dollars, 2),
        "sigma_used": round(sigma, 4),
        "T_years": round(T, 6),
    }


def reprice_iron_condor_at_exit(
    position: dict,
    exit_spot: float,
    realized_vol: float,
    days_remaining: int = 0,
    r: float = RISK_FREE_RATE,
) -> float:
    """
    Calculate P&L for closing an iron condor position at exit.

    Post-earnings: IV crushed to realized vol * 1.0.
    If stock beyond short strikes, intrinsic loss applies (capped by wings).

    Returns dollar P&L (positive = profit).
    """
    sigma = realized_vol * POST_EARNINGS_IV_MULT
    T = max(days_remaining, 0) / 252.0

    short_put = position["short_put"]
    long_put = position["long_put"]
    short_call = position["short_call"]
    long_call = position["long_call"]

    # Cost to close each leg (buy back shorts, sell longs)
    close_short_put = bs_put(exit_spot, short_put, T, r, sigma)
    close_long_put = bs_put(exit_spot, long_put, T, r, sigma)
    close_short_call = bs_call(exit_spot, short_call, T, r, sigma)
    close_long_call = bs_call(exit_spot, long_call, T, r, sigma)

    # Net cost to close = (buy back shorts) - (sell longs)
    close_cost = (close_short_put + close_short_call) - (close_long_put + close_long_call)

    # P&L = initial credit - close cost - commissions
    entry_credit = position["net_credit_dollars"]
    close_cost_dollars = close_cost * 100
    pnl = entry_credit - close_cost_dollars - COST_PER_IC

    # Cap loss at max risk + commissions
    max_loss = -(position["max_risk_dollars"] + COST_PER_IC)
    pnl = max(pnl, max_loss)

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
        exit_date = earnings_date + pd.offsets.BDay(EXIT_DAYS_AFTER)

        if today >= exit_date:
            # Get current price for exit
            hist = get_price_history(pos["ticker"], period="5d")
            if hist is not None and not hist.empty:
                exit_spot = float(hist["Close"].iloc[-1])
                rv_hist = get_price_history(pos["ticker"], period="3mo")
                rv = compute_realized_vol(rv_hist, window=20)
                if rv is None:
                    rv = pos.get("realized_vol", 0.20)
            else:
                # Fallback: use entry spot (conservative — assume no move)
                exit_spot = pos["spot"]
                rv = pos.get("realized_vol", 0.20)

            # Days remaining on options after exit
            entry_date = pd.Timestamp(pos["entry_date"])
            elapsed = (today - entry_date).days
            days_rem = max(0, pos.get("days_to_expiry", 5) - elapsed)

            pnl = reprice_iron_condor_at_exit(pos, exit_spot, rv, days_remaining=days_rem)

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
                f"Credit ${pos['net_credit_dollars']:.2f} | Max risk ${pos['max_risk_dollars']:.2f}"
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
    Scan universe for stocks with earnings ~3 bdays away.
    Apply filters and open paper iron condor positions.
    Returns list of summary messages.
    """
    messages = []
    n_open = len(state["positions"])

    if n_open >= MAX_CONCURRENT:
        messages.append(f"  Max concurrent positions ({MAX_CONCURRENT}) reached, skipping scan")
        return messages

    # Target earnings date window: 2-4 bdays from today (centered on 3)
    target_start = today + pd.offsets.BDay(ENTRY_DAYS_BEFORE - 1)
    target_end = today + pd.offsets.BDay(ENTRY_DAYS_BEFORE + 1)

    # Track already-held tickers
    held_tickers = {p["ticker"] for p in state["positions"]}

    # Skip tickers with recent closed trades (avoid re-entry on same earnings cycle)
    recent_closed = set()
    for t in state["closed_trades"]:
        exit_str = t.get("exit_date", "2000-01-01")
        if (today - pd.Timestamp(exit_str)).days < 30:
            recent_closed.add(t["ticker"])

    candidates_checked = 0
    candidates_passed = 0

    for ticker in UNIVERSE:
        if n_open >= MAX_CONCURRENT:
            break
        if ticker in held_tickers or ticker in recent_closed:
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

        # Get price history (2y for IV rank — needs 252+ bars)
        hist = get_price_history(ticker, period="2y")
        if hist is None or len(hist) < 260:
            log.info(f"  {ticker}: insufficient history ({len(hist) if hist is not None else 0} bars), skipping")
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

        # Filter 2: stock hasn't moved > 5% in prior 5 trading days
        movement = check_movement(hist, days=5)
        if movement is None or movement > MOVEMENT_THRESHOLD:
            pct = movement * 100 if movement else 0
            log.info(f"  {ticker}: 5d move {pct:.1f}% > {MOVEMENT_THRESHOLD * 100:.0f}%, skipping")
            continue

        # Days to expiry: entry to roughly 1 week after earnings for option expiry
        days_to_expiry = max(5, (earnings_date - today).days + 2)

        # Price the iron condor
        ic = price_iron_condor(
            spot=current_price,
            realized_vol=rv,
            iv_multiplier=PRE_EARNINGS_IV_MULT,
            days_to_expiry=days_to_expiry,
        )

        if ic is None:
            log.info(f"  {ticker}: invalid IC pricing (strikes or zero credit), skipping")
            continue

        # Filter 3: max risk per trade
        if ic["max_risk_dollars"] > MAX_RISK_PER_TRADE:
            log.info(f"  {ticker}: risk ${ic['max_risk_dollars']:.0f} > ${MAX_RISK_PER_TRADE:.0f}, skipping")
            continue

        # All filters passed — open position
        candidates_passed += 1
        position = {
            "ticker": ticker,
            "entry_date": str(today.date()),
            "earnings_date": str(earnings_date.date()),
            "spot": current_price,
            "realized_vol": rv,
            "iv_rank": round(iv_rank, 1),
            "days_to_expiry": days_to_expiry,
            **{k: v for k, v in ic.items() if k != "spot"},
        }
        state["positions"].append(position)
        held_tickers.add(ticker)
        n_open += 1

        messages.append(
            f"  OPENED {ticker}: Spot ${current_price:.2f} | "
            f"Puts ${ic['long_put']:.2f}/${ic['short_put']:.2f} | "
            f"Calls ${ic['short_call']:.2f}/${ic['long_call']:.2f} | "
            f"Credit ${ic['net_credit_dollars']:.2f} | Risk ${ic['max_risk_dollars']:.2f} | "
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
    log.info("Pre-Earnings Vol Crush Iron Condor Paper Engine")
    log.info("=" * 60)

    today = pd.Timestamp(datetime.now().date())
    log.info(f"Date: {today.date()}")

    # Load state
    state = load_state()
    log.info(
        f"Capital: ${state['capital']:,.2f} | Open: {len(state['positions'])} | "
        f"Trades: {state['total_trades']} | "
        f"WR: {state['wins'] / max(state['total_trades'], 1) * 100:.0f}%"
    )

    # 1. Process exits (close positions where earnings was yesterday)
    log.info("")
    log.info("--- EXIT CHECK ---")
    exit_msgs = process_exits(state, today)
    if exit_msgs:
        for m in exit_msgs:
            log.info(m)
    else:
        log.info("  No positions to close today")

    # 2. Scan for new entries
    log.info("")
    log.info("--- ENTRY SCAN ---")
    entry_msgs = scan_for_entries(state, today)
    for m in entry_msgs:
        log.info(m)

    # 3. Record daily equity
    total_equity = state["capital"]
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
        days_held = (today - pd.Timestamp(p["entry_date"])).days
        log.info(
            f"  {p['ticker']}: entered {p['entry_date']}, earnings {p['earnings_date']}, "
            f"credit ${p['net_credit_dollars']:.2f}, risk ${p['max_risk_dollars']:.2f}, "
            f"held {days_held}d"
        )
    log.info(f"Total realized P&L: ${state['total_realized_pnl']:+,.2f}")
    log.info(f"Total trades: {state['total_trades']} | Wins: {state['wins']} | Losses: {state['losses']}")

    if state["total_trades"] > 0:
        wr = state["wins"] / state["total_trades"] * 100
        log.info(f"Win rate: {wr:.1f}%")

        if len(state["closed_trades"]) >= 5:
            pnls = [t["pnl"] for t in state["closed_trades"]]
            avg = np.mean(pnls)
            std = np.std(pnls, ddof=1)

            if std > 0:
                # Annualize assuming ~50 trades/year
                n_ann = np.sqrt(min(len(pnls), 50))
                sharpe = (avg / std) * n_ann
                log.info(f"Sharpe (annualized est): {sharpe:.2f}")

                # Sortino
                downside = [p for p in pnls if p < 0]
                if downside:
                    downside_std = np.std(downside, ddof=1)
                    if downside_std > 0:
                        sortino = (avg / downside_std) * n_ann
                        log.info(f"Sortino (annualized est): {sortino:.2f}")

                # Profit factor
                gross_wins = sum(p for p in pnls if p > 0)
                gross_losses = abs(sum(p for p in pnls if p < 0))
                if gross_losses > 0:
                    pf = gross_wins / gross_losses
                    log.info(f"Profit factor: {pf:.2f}")

    log.info("=" * 60)


if __name__ == "__main__":
    try:
        run_engine()
    except Exception as e:
        log.error(f"Engine failed: {e}", exc_info=True)
        sys.exit(1)
