#!/usr/bin/env python3
"""
Earnings IV Crush Paper Engine — Real Quote Ready
===================================================

Small-account ($681) options strategy targeting earnings IV crush.

STRATEGY:
  - Weekly: scan earnings calendar for stocks reporting in 1-3 business days
  - Filter: IV rank > 60th percentile (elevated vol before earnings)
  - Trade: Sell a narrow iron condor (or bull put spread if directional bias)
    Actually for Level 2 (vertical spreads only): sell a put credit spread
    or call credit spread to capture IV crush post-earnings.
  - For small account: sell OTM bull put spreads (defined risk)
    - Short put: ~0.8 stdev below price (~30 delta)
    - Long put: $2.50-$5 below short put
    - Credit target: 30-40% of spread width
    - Max risk per trade: $150 (spread width - credit)
  - Exit: close 1 day after earnings (IV crush realized)
  - If stock drops below short put: loss capped at spread width - credit
  - Stop: close if spread doubles in cost (2x credit received)

PRICING:
  - Paper mode: Black-Scholes estimates flagged as [BS-EST]
  - When execution bridge runs, it verifies with real RH MCP quotes
  - Commission: $0 on Robinhood

CAPITAL: $681 starting
MAX POSITION: $150 risk per trade
MAX CONCURRENT: 3 positions

Usage:
  python3 paper_engines/earnings_iv_crush_real_paper.py
"""

import json
import logging
import math
import os
import sys
import tempfile
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "state" / "earnings_iv_crush_real_state.json"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EARNINGS-IV-CRUSH] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "earnings_iv_crush_real.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("earnings_iv_crush")

# ── Config ──
INITIAL_CAPITAL = 681.0
MAX_RISK_PER_TRADE = 150.0   # max loss per position
MAX_CONCURRENT = 3
ENTRY_DAYS_BEFORE = 3        # scan stocks reporting in 1-3 bdays
EXIT_DAYS_AFTER = 1           # close 1 bday after earnings
IV_RANK_MIN = 0.60            # IV rank percentile threshold
PUT_STDEV = 0.8               # short put distance (stdevs from price)
MIN_CREDIT_PCT = 0.30         # min credit as % of spread width
MAX_CREDIT_PCT = 0.50         # max (too risky if credit is too high)
STOP_MULT = 2.0               # close if cost to close = 2x credit
RISK_FREE_RATE = 0.05
PRE_EARNINGS_IV_MULT = 1.5    # pre-earnings IV inflation
POST_EARNINGS_IV_MULT = 0.85  # post-earnings IV crush
COMMISSION = 0.0              # Robinhood = $0

# Universe: liquid large-caps with active options and frequent earnings surprises
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX",
    "AMD", "CRM", "ORCL", "ADBE", "INTC", "MU", "QCOM", "AVGO",
    "JPM", "GS", "BAC", "WFC", "V", "MA",
    "UNH", "JNJ", "PFE", "LLY", "ABBV", "MRK",
    "XOM", "CVX", "HD", "LOW", "COST", "WMT",
    "DIS", "SBUX", "NKE", "TGT", "FDX", "UPS",
]

yf = None


def _yf():
    global yf
    if yf is None:
        import yfinance as _yf
        yf = _yf
    return yf


# ────────────────────────────────────────────
#  BLACK-SCHOLES
# ────────────────────────────────────────────
def bs_put(S, K, T, r, sigma):
    """BS put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_call(S, K, T, r, sigma):
    """BS call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


# ────────────────────────────────────────────
#  STATE MANAGEMENT
# ────────────────────────────────────────────
def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "capital": INITIAL_CAPITAL,
        "positions": [],
        "closed_trades": [],
        "created": str(datetime.now()),
        "last_run": None,
        "total_credits_received": 0.0,
        "total_pnl": 0.0,
        "strategy": "Earnings IV Crush - Bull Put Spreads (L2 Options, $681 account)",
    }


def save_state(state):
    state["last_run"] = str(datetime.now())
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.rename(STATE_FILE)
    log.info(f"State saved: {len(state['positions'])} open, {len(state['closed_trades'])} closed, capital=${state['capital']:.2f}")


# ────────────────────────────────────────────
#  MARKET DATA
# ────────────────────────────────────────────
def get_stock_data(ticker, days=90):
    """Get price history for a stock."""
    try:
        df = _yf().download(ticker, period=f"{days}d", progress=False)
        if df is None or df.empty or len(df) < 20:
            return None
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        log.warning(f"Failed to get data for {ticker}: {e}")
        return None


def compute_iv_rank(df, window=60):
    """Compute IV rank using realized vol as proxy (annualized)."""
    if len(df) < window + 5:
        return None, None
    returns = df["Close"].pct_change().dropna()
    if len(returns) < window:
        return None, None

    # Current realized vol (20-day)
    current_vol = returns.tail(20).std() * np.sqrt(252)

    # Historical vol range (window days)
    rolling_vol = returns.rolling(20).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()
    if len(rolling_vol) < 10:
        return None, None

    vol_min = rolling_vol.min()
    vol_max = rolling_vol.max()
    if vol_max - vol_min < 0.01:
        return float(current_vol), 0.5

    iv_rank = (current_vol - vol_min) / (vol_max - vol_min)
    return float(current_vol), float(np.clip(iv_rank, 0, 1))


def get_earnings_dates(ticker):
    """Get upcoming earnings dates for a ticker."""
    try:
        tk = _yf().Ticker(ticker)
        cal = tk.calendar
        if cal is None:
            return []
        # yfinance calendar format varies
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.columns:
                dates = cal["Earnings Date"].tolist()
            elif "Earnings Date" in cal.index:
                dates = [cal.loc["Earnings Date"].iloc[0]]
            else:
                return []
        elif isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if not isinstance(dates, list):
                dates = [dates]
        else:
            return []
        # Convert to datetime
        result = []
        for d in dates:
            if isinstance(d, (datetime, pd.Timestamp)):
                result.append(d)
            elif isinstance(d, str):
                try:
                    result.append(pd.Timestamp(d))
                except:
                    pass
        return result
    except Exception as e:
        log.debug(f"No earnings calendar for {ticker}: {e}")
        return []


def find_nearest_strikes(price, spread_width=5.0, stdev_dist=0.8, vol=0.3, days_to_exp=10):
    """Find short put and long put strikes for a bull put spread."""
    # Short put: stdev_dist below current price
    move = price * vol * math.sqrt(days_to_exp / 252) * stdev_dist
    short_put_raw = price - move

    # Round to nearest $1 for stocks > $50, $0.50 for cheaper
    if price > 100:
        short_put = round(short_put_raw)
    elif price > 50:
        short_put = round(short_put_raw)
    else:
        short_put = round(short_put_raw * 2) / 2

    # Long put: $2.50 or $5 below short put (keep spread narrow for small account)
    if price > 100:
        long_put = short_put - 5
        spread = 5.0
    elif price > 50:
        long_put = short_put - 2.5
        spread = 2.5
    else:
        long_put = short_put - 2.5
        spread = 2.5

    return short_put, long_put, spread


def next_monthly_expiry(from_date=None):
    """Find next monthly options expiry (3rd Friday)."""
    if from_date is None:
        from_date = datetime.now()
    # Find 3rd Friday of current month
    year, month = from_date.year, from_date.month
    # Try current month first
    for attempt in range(3):  # try current, next, next+1
        m = month + attempt
        y = year
        if m > 12:
            m -= 12
            y += 1
        # Find 3rd Friday
        first_day = datetime(y, m, 1)
        # Day of week: 0=Mon, 4=Fri
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)
        if third_friday > from_date + timedelta(days=3):  # at least 3 days out
            return third_friday
    return from_date + timedelta(days=30)  # fallback


def next_weekly_expiry(from_date=None):
    """Find next Friday expiry (weekly options)."""
    if from_date is None:
        from_date = datetime.now()
    days_until_friday = (4 - from_date.weekday()) % 7
    if days_until_friday < 2:  # too close, use next week
        days_until_friday += 7
    return from_date + timedelta(days=days_until_friday)


# ────────────────────────────────────────────
#  SCANNING
# ────────────────────────────────────────────
def scan_for_earnings_setups(state):
    """Scan universe for upcoming earnings with elevated IV."""
    today = datetime.now()
    setups = []

    # Skip if already at max positions
    if len(state["positions"]) >= MAX_CONCURRENT:
        log.info(f"At max concurrent positions ({MAX_CONCURRENT}), skipping scan")
        return []

    # Check available capital
    capital_in_use = sum(p.get("max_risk", 0) for p in state["positions"])
    available = state["capital"] - capital_in_use

    log.info(f"Scanning {len(UNIVERSE)} stocks for earnings setups (avail capital: ${available:.2f})")

    for ticker in UNIVERSE:
        try:
            # Get earnings dates
            earnings_dates = get_earnings_dates(ticker)
            if not earnings_dates:
                continue

            # Check if any earnings date is 1-3 business days away
            target_date = None
            for ed in earnings_dates:
                if isinstance(ed, pd.Timestamp):
                    ed = ed.to_pydatetime()
                if hasattr(ed, 'tzinfo') and ed.tzinfo:
                    ed = ed.replace(tzinfo=None)
                days_away = (ed - today).days
                if 1 <= days_away <= 5:  # 1-5 calendar days ≈ 1-3 bdays
                    target_date = ed
                    break

            if target_date is None:
                continue

            # Already have a position in this ticker?
            if any(p["ticker"] == ticker for p in state["positions"]):
                log.info(f"  {ticker}: already have position, skipping")
                continue

            # Get stock data and compute IV rank
            df = get_stock_data(ticker, days=90)
            if df is None:
                continue

            current_price = float(df["Close"].iloc[-1])
            realized_vol, iv_rank = compute_iv_rank(df)

            if realized_vol is None or iv_rank is None:
                continue

            if iv_rank < IV_RANK_MIN:
                log.debug(f"  {ticker}: IV rank {iv_rank:.2f} below threshold {IV_RANK_MIN}")
                continue

            # Calculate spread parameters
            days_to_exp = max((target_date - today).days + 2, 5)  # expire shortly after earnings
            expiry = next_weekly_expiry(target_date)  # weekly expiry after earnings

            short_put, long_put, spread_width = find_nearest_strikes(
                current_price, stdev_dist=PUT_STDEV, vol=realized_vol, days_to_exp=days_to_exp
            )

            # Price the spread using BS (pre-earnings elevated IV)
            pre_iv = realized_vol * PRE_EARNINGS_IV_MULT
            T = days_to_exp / 365.0

            short_put_price = bs_put(current_price, short_put, T, RISK_FREE_RATE, pre_iv)
            long_put_price = bs_put(current_price, long_put, T, RISK_FREE_RATE, pre_iv)
            credit = short_put_price - long_put_price

            # Max risk = spread width - credit (per share, x100 for contract)
            max_risk_per_contract = (spread_width - credit) * 100
            credit_per_contract = credit * 100

            # Filter: credit must be 30-50% of spread width
            credit_pct = credit / spread_width if spread_width > 0 else 0
            if credit_pct < MIN_CREDIT_PCT:
                log.debug(f"  {ticker}: credit {credit_pct:.1%} below min {MIN_CREDIT_PCT:.0%}")
                continue
            if credit_pct > MAX_CREDIT_PCT:
                log.debug(f"  {ticker}: credit {credit_pct:.1%} above max (too risky)")
                continue

            # Position sizing: max risk $150, check available capital
            if max_risk_per_contract > MAX_RISK_PER_TRADE:
                log.debug(f"  {ticker}: risk ${max_risk_per_contract:.0f} exceeds max ${MAX_RISK_PER_TRADE}")
                continue

            if max_risk_per_contract > available:
                log.debug(f"  {ticker}: risk ${max_risk_per_contract:.0f} exceeds available ${available:.0f}")
                continue

            setup = {
                "ticker": ticker,
                "earnings_date": str(target_date.date()),
                "days_to_earnings": (target_date - today).days,
                "current_price": round(current_price, 2),
                "realized_vol": round(realized_vol, 3),
                "iv_rank": round(iv_rank, 3),
                "short_put": short_put,
                "long_put": long_put,
                "spread_width": spread_width,
                "credit": round(credit, 2),
                "credit_per_contract": round(credit_per_contract, 2),
                "max_risk_per_contract": round(max_risk_per_contract, 2),
                "credit_pct": round(credit_pct, 3),
                "expiry": str(expiry.date()),
                "pre_iv": round(pre_iv, 3),
                "pricing_method": "[BS-EST] Black-Scholes estimate — verify with RH quotes before live execution",
            }

            log.info(
                f"  SETUP: {ticker} earnings {target_date.date()} | "
                f"price=${current_price:.0f} IV-rank={iv_rank:.2f} | "
                f"sell {short_put}P / buy {long_put}P | "
                f"credit=${credit_per_contract:.0f} risk=${max_risk_per_contract:.0f} "
                f"[BS-EST]"
            )
            setups.append(setup)

        except Exception as e:
            log.warning(f"  {ticker}: error scanning: {e}")
            continue

    # Sort by IV rank descending (higher IV = better crush potential)
    setups.sort(key=lambda x: x["iv_rank"], reverse=True)
    return setups


# ────────────────────────────────────────────
#  POSITION MANAGEMENT
# ────────────────────────────────────────────
def open_positions(state, setups):
    """Open new positions from setups."""
    opened = 0
    for setup in setups:
        if len(state["positions"]) >= MAX_CONCURRENT:
            break

        capital_in_use = sum(p.get("max_risk", 0) for p in state["positions"])
        available = state["capital"] - capital_in_use
        if setup["max_risk_per_contract"] > available:
            continue

        position = {
            "ticker": setup["ticker"],
            "entry_date": str(datetime.now().date()),
            "earnings_date": setup["earnings_date"],
            "entry_price": setup["current_price"],
            "short_put": setup["short_put"],
            "long_put": setup["long_put"],
            "spread_width": setup["spread_width"],
            "credit_received": setup["credit_per_contract"],
            "max_risk": setup["max_risk_per_contract"],
            "expiry": setup["expiry"],
            "entry_iv": setup["pre_iv"],
            "iv_rank_at_entry": setup["iv_rank"],
            "stop_cost": round(setup["credit_per_contract"] * STOP_MULT, 2),
            "status": "open",
            "pricing_method": setup["pricing_method"],
        }

        state["positions"].append(position)
        state["total_credits_received"] = state.get("total_credits_received", 0) + setup["credit_per_contract"]
        opened += 1
        log.info(
            f"OPENED: {setup['ticker']} bull put spread "
            f"{setup['short_put']}/{setup['long_put']}P "
            f"credit=${setup['credit_per_contract']:.0f} "
            f"max_risk=${setup['max_risk_per_contract']:.0f} "
            f"exp {setup['expiry']}"
        )

    return opened


def check_exits(state):
    """Check if any positions should be closed."""
    today = datetime.now()
    closed = 0
    remaining = []

    for pos in state["positions"]:
        ticker = pos["ticker"]
        should_close = False
        close_reason = ""
        pnl = 0.0

        # Get current price
        df = get_stock_data(ticker, days=10)
        if df is None:
            remaining.append(pos)
            continue

        current_price = float(df["Close"].iloc[-1])
        pos["current_price"] = round(current_price, 2)

        # Check if past earnings date (exit 1 day after)
        earnings_dt = datetime.strptime(pos["earnings_date"], "%Y-%m-%d")
        days_since_earnings = (today - earnings_dt).days

        # Check expiry
        expiry_dt = datetime.strptime(pos["expiry"], "%Y-%m-%d")
        days_to_expiry = (expiry_dt - today).days

        if days_since_earnings >= EXIT_DAYS_AFTER:
            should_close = True
            close_reason = "post-earnings exit (IV crush captured)"

            # Price the spread post-earnings (IV crushed)
            realized_vol, _ = compute_iv_rank(df)
            if realized_vol is None:
                realized_vol = pos["entry_iv"] / PRE_EARNINGS_IV_MULT
            post_iv = realized_vol * POST_EARNINGS_IV_MULT
            T = max(days_to_expiry / 365.0, 1/365.0)

            close_short = bs_put(current_price, pos["short_put"], T, RISK_FREE_RATE, post_iv)
            close_long = bs_put(current_price, pos["long_put"], T, RISK_FREE_RATE, post_iv)
            cost_to_close = (close_short - close_long) * 100

            # If stock below short put, intrinsic loss
            if current_price < pos["short_put"]:
                intrinsic_loss = (pos["short_put"] - max(current_price, pos["long_put"])) * 100
                cost_to_close = max(cost_to_close, intrinsic_loss)

            pnl = pos["credit_received"] - cost_to_close - COMMISSION
            close_reason += f" | cost_to_close=${cost_to_close:.0f}"

        elif days_to_expiry <= 0:
            should_close = True
            close_reason = "expiry reached"
            # At expiry: if OTM, full credit kept; if ITM, spread loss
            if current_price >= pos["short_put"]:
                pnl = pos["credit_received"] - COMMISSION
            else:
                intrinsic = (pos["short_put"] - max(current_price, pos["long_put"])) * 100
                pnl = pos["credit_received"] - intrinsic - COMMISSION

        else:
            # Check stop: if cost to close > 2x credit
            realized_vol, _ = compute_iv_rank(df)
            if realized_vol:
                current_iv = realized_vol * (PRE_EARNINGS_IV_MULT if days_since_earnings < 0 else POST_EARNINGS_IV_MULT)
                T = max(days_to_expiry / 365.0, 1/365.0)
                close_short = bs_put(current_price, pos["short_put"], T, RISK_FREE_RATE, current_iv)
                close_long = bs_put(current_price, pos["long_put"], T, RISK_FREE_RATE, current_iv)
                cost_to_close = (close_short - close_long) * 100

                if cost_to_close >= pos["stop_cost"]:
                    should_close = True
                    close_reason = f"STOP HIT: cost_to_close=${cost_to_close:.0f} >= stop=${pos['stop_cost']:.0f}"
                    pnl = pos["credit_received"] - cost_to_close - COMMISSION

        if should_close:
            trade = {
                "ticker": ticker,
                "entry_date": pos["entry_date"],
                "exit_date": str(today.date()),
                "short_put": pos["short_put"],
                "long_put": pos["long_put"],
                "credit_received": pos["credit_received"],
                "pnl": round(pnl, 2),
                "close_reason": close_reason,
                "entry_price": pos["entry_price"],
                "exit_price": current_price,
                "pricing_method": "[BS-EST]",
            }
            state["closed_trades"].append(trade)
            state["capital"] += pnl
            state["total_pnl"] = state.get("total_pnl", 0) + pnl
            closed += 1
            log.info(
                f"CLOSED: {ticker} {pos['short_put']}/{pos['long_put']}P | "
                f"P&L=${pnl:.2f} | {close_reason}"
            )
        else:
            remaining.append(pos)

    state["positions"] = remaining
    return closed


# ────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────
def run():
    log.info("=" * 60)
    log.info("EARNINGS IV CRUSH PAPER ENGINE — starting daily run")
    log.info("=" * 60)

    state = load_state()

    # 1) Check exits on existing positions
    closed = check_exits(state)
    if closed:
        log.info(f"Closed {closed} positions")

    # 2) Scan for new setups
    setups = scan_for_earnings_setups(state)
    log.info(f"Found {len(setups)} earnings setups")

    # 3) Open best setups
    if setups:
        opened = open_positions(state, setups)
        log.info(f"Opened {opened} new positions")

    # 4) Summary
    capital_in_use = sum(p.get("max_risk", 0) for p in state["positions"])
    total_trades = len(state["closed_trades"])
    winners = sum(1 for t in state["closed_trades"] if t["pnl"] > 0)
    wr = winners / total_trades * 100 if total_trades > 0 else 0

    log.info(f"--- SUMMARY ---")
    log.info(f"Capital: ${state['capital']:.2f} | In use: ${capital_in_use:.2f}")
    log.info(f"Open positions: {len(state['positions'])}")
    log.info(f"Total trades: {total_trades} | Win rate: {wr:.0f}%")
    log.info(f"Total P&L: ${state.get('total_pnl', 0):.2f}")

    for p in state["positions"]:
        log.info(f"  OPEN: {p['ticker']} {p['short_put']}/{p['long_put']}P credit=${p['credit_received']:.0f} exp={p['expiry']}")

    save_state(state)
    log.info("Done.")


if __name__ == "__main__":
    run()
