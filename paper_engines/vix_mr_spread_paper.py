#!/usr/bin/env python3
"""
VIX Mean Reversion Spread Paper Engine
========================================

Small-account ($681) strategy selling put spreads on SPY when VIX spikes.

STRATEGY:
  - When VIX closes above 25: sell a bull put spread on SPY
    (bet that VIX reverts = SPY bounces)
  - Short put: 2-3% below current SPY price (slightly OTM)
  - Long put: $3-5 below short put (defined risk)
  - Expiry: 2-3 weeks out (enough time for VIX mean reversion)
  - Credit target: 30-40% of spread width
  - Take profit: close when spread value drops to 20% of credit received
    (or VIX drops below 18)
  - Stop: close if spread doubles in cost (2x credit)
  - Max risk per trade: $200
  - Max concurrent: 2 positions

WHY IT WORKS:
  - VIX > 25 is historically elevated and mean-reverts ~70% of the time
  - SPY tends to rally after VIX spikes (panic selling overshoots)
  - Selling put spreads = positive theta + positive delta
  - Small account can participate with defined risk

PRICING:
  - Paper mode: Black-Scholes estimates flagged as [BS-EST]
  - Commission: $0 (Robinhood)

CAPITAL: $681 starting
MAX POSITION: $200 risk per trade

Usage:
  python3 paper_engines/vix_mr_spread_paper.py
"""

import json
import logging
import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "state" / "vix_mr_spread_state.json"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VIX-MR] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "vix_mr_spread.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("vix_mr_spread")

# ── Config ──
INITIAL_CAPITAL = 681.0
MAX_RISK_PER_TRADE = 200.0
MAX_CONCURRENT = 2
VIX_ENTRY_THRESHOLD = 25.0    # enter when VIX > 25
VIX_TP_THRESHOLD = 18.0       # take profit when VIX < 18
VIX_EXTREME_THRESHOLD = 35.0  # wait if VIX > 35 (might go higher)
PUT_OTM_PCT = 0.025           # short put 2.5% below SPY
HOLD_DAYS_TARGET = 14         # 2 weeks target hold
MAX_HOLD_DAYS = 21            # 3 weeks max
TP_PCT = 0.80                 # close when 80% of credit captured (spread value = 20% of credit)
STOP_MULT = 2.0               # close if cost to close = 2x credit
RISK_FREE_RATE = 0.05
COMMISSION = 0.0

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
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ────────────────────────────────────────────
#  STATE
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
        "total_pnl": 0.0,
        "vix_signals": [],  # history of VIX readings when we traded
        "strategy": "VIX Mean Reversion - SPY Bull Put Spreads ($681 account)",
    }


def save_state(state):
    state["last_run"] = str(datetime.now())
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.rename(STATE_FILE)


# ────────────────────────────────────────────
#  MARKET DATA
# ────────────────────────────────────────────
def get_vix():
    """Get current VIX level and recent history."""
    try:
        vix = _yf().download("^VIX", period="30d", progress=False)
        if vix is None or vix.empty:
            return None, None
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        current = float(vix["Close"].iloc[-1])
        avg_20d = float(vix["Close"].tail(20).mean())
        return current, avg_20d
    except Exception as e:
        log.warning(f"Failed to get VIX: {e}")
        return None, None


def get_spy_data(days=60):
    """Get SPY price data."""
    try:
        df = _yf().download("SPY", period=f"{days}d", progress=False)
        if df is None or df.empty or len(df) < 20:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        log.warning(f"Failed to get SPY data: {e}")
        return None


def next_monthly_expiry(from_date=None):
    """Find next monthly expiry (3rd Friday) at least 10 days out."""
    if from_date is None:
        from_date = datetime.now()
    for attempt in range(4):
        m = from_date.month + attempt
        y = from_date.year
        while m > 12:
            m -= 12
            y += 1
        first_day = datetime(y, m, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)
        if third_friday >= from_date + timedelta(days=10):
            return third_friday
    return from_date + timedelta(days=30)


def next_friday_expiry(from_date=None, min_days=7):
    """Find next Friday expiry at least min_days out."""
    if from_date is None:
        from_date = datetime.now()
    d = from_date + timedelta(days=min_days)
    days_until_friday = (4 - d.weekday()) % 7
    return d + timedelta(days=days_until_friday)


def round_spy_strike(price):
    """Round to nearest $1 strike for SPY."""
    return round(price)


# ────────────────────────────────────────────
#  SCANNING & ENTRY
# ────────────────────────────────────────────
def check_for_entry(state):
    """Check if VIX is elevated enough to sell SPY put spreads."""
    today = datetime.now()

    if len(state["positions"]) >= MAX_CONCURRENT:
        log.info(f"At max concurrent ({MAX_CONCURRENT}), skipping entry check")
        return None

    capital_in_use = sum(p.get("max_risk", 0) for p in state["positions"])
    available = state["capital"] - capital_in_use

    # Get VIX
    vix_current, vix_avg = get_vix()
    if vix_current is None:
        log.warning("Could not get VIX data")
        return None

    log.info(f"VIX: {vix_current:.2f} (20d avg: {vix_avg:.2f})")

    # Record VIX reading
    state.setdefault("vix_signals", []).append({
        "date": str(today.date()),
        "vix": round(vix_current, 2),
        "vix_avg": round(vix_avg, 2),
        "action": "none",
    })
    # Keep last 100 readings
    state["vix_signals"] = state["vix_signals"][-100:]

    # Entry conditions
    if vix_current < VIX_ENTRY_THRESHOLD:
        log.info(f"VIX {vix_current:.1f} below entry threshold {VIX_ENTRY_THRESHOLD}, no trade")
        return None

    if vix_current > VIX_EXTREME_THRESHOLD:
        log.info(f"VIX {vix_current:.1f} above extreme threshold {VIX_EXTREME_THRESHOLD}, waiting for peak")
        return None

    # Don't enter if we already have a position opened in the last 3 days
    recent_entries = [
        p for p in state["positions"]
        if (today - datetime.strptime(p["entry_date"], "%Y-%m-%d")).days < 3
    ]
    if recent_entries:
        log.info("Already opened a position in last 3 days, waiting")
        return None

    # Get SPY data
    spy_df = get_spy_data(60)
    if spy_df is None:
        return None

    spy_price = float(spy_df["Close"].iloc[-1])

    # Compute SPY realized vol
    returns = spy_df["Close"].pct_change().dropna()
    realized_vol = float(returns.tail(20).std() * np.sqrt(252))
    if realized_vol < 0.05:
        realized_vol = 0.15

    # Use VIX as the implied vol (VIX = SPY 30-day implied vol)
    implied_vol = vix_current / 100.0  # VIX is in percentage points

    # Calculate strikes
    short_put_price = spy_price * (1 - PUT_OTM_PCT)
    short_put = round_spy_strike(short_put_price)

    # Spread width: $3-5 depending on available capital
    if available >= 200:
        spread_width = 5.0
    elif available >= 100:
        spread_width = 3.0
    else:
        spread_width = 2.0

    long_put = short_put - spread_width

    # Expiry: 2-3 weeks out
    expiry = next_friday_expiry(today, min_days=14)
    T = max((expiry - today).days / 365.0, 1/365.0)

    # Price the spread using BS with implied vol (VIX-derived)
    short_put_val = bs_put(spy_price, short_put, T, RISK_FREE_RATE, implied_vol)
    long_put_val = bs_put(spy_price, long_put, T, RISK_FREE_RATE, implied_vol)
    credit = short_put_val - long_put_val
    credit_per_contract = credit * 100

    # Max risk
    max_risk = (spread_width - credit) * 100

    # Validate credit
    credit_pct = credit / spread_width if spread_width > 0 else 0

    log.info(
        f"SPY=${spy_price:.2f} VIX={vix_current:.1f} IV={implied_vol:.3f} | "
        f"short {short_put}P / long {long_put}P | "
        f"credit=${credit_per_contract:.0f} risk=${max_risk:.0f} "
        f"({credit_pct:.0%} of width) exp={expiry.date()}"
    )

    if credit_pct < 0.20:
        log.info(f"Credit too thin ({credit_pct:.0%}), skipping")
        return None

    if max_risk > MAX_RISK_PER_TRADE:
        log.info(f"Risk ${max_risk:.0f} exceeds max ${MAX_RISK_PER_TRADE}, skipping")
        return None

    if max_risk > available:
        log.info(f"Risk ${max_risk:.0f} exceeds available ${available:.0f}, skipping")
        return None

    # Update signal record
    state["vix_signals"][-1]["action"] = "ENTER"

    return {
        "spy_price": round(spy_price, 2),
        "vix_at_entry": round(vix_current, 2),
        "short_put": short_put,
        "long_put": long_put,
        "spread_width": spread_width,
        "credit_per_contract": round(credit_per_contract, 2),
        "max_risk": round(max_risk, 2),
        "credit_pct": round(credit_pct, 3),
        "expiry": str(expiry.date()),
        "days_to_expiry": (expiry - today).days,
        "implied_vol": round(implied_vol, 3),
        "realized_vol": round(realized_vol, 3),
        "pricing_method": "[BS-EST] — VIX used as IV proxy",
    }


def open_position(state, setup):
    """Open a new VIX mean-reversion position."""
    position = {
        "ticker": "SPY",
        "entry_date": str(datetime.now().date()),
        "entry_spy_price": setup["spy_price"],
        "vix_at_entry": setup["vix_at_entry"],
        "short_put": setup["short_put"],
        "long_put": setup["long_put"],
        "spread_width": setup["spread_width"],
        "credit_received": setup["credit_per_contract"],
        "max_risk": setup["max_risk"],
        "expiry": setup["expiry"],
        "entry_iv": setup["implied_vol"],
        "stop_cost": round(setup["credit_per_contract"] * STOP_MULT, 2),
        "tp_credit_remaining": round(setup["credit_per_contract"] * (1 - TP_PCT), 2),
        "pricing_method": setup["pricing_method"],
    }

    state["positions"].append(position)
    log.info(
        f"OPENED: SPY {setup['short_put']}/{setup['long_put']}P bull put spread | "
        f"VIX={setup['vix_at_entry']:.1f} | credit=${setup['credit_per_contract']:.0f} "
        f"risk=${setup['max_risk']:.0f} | exp={setup['expiry']}"
    )


def check_exits(state):
    """Check positions for TP/SL/expiry/VIX exits."""
    today = datetime.now()
    closed = 0
    remaining = []

    vix_current, _ = get_vix()
    spy_df = get_spy_data(30)
    spy_price = float(spy_df["Close"].iloc[-1]) if spy_df is not None else None

    for pos in state["positions"]:
        should_close = False
        close_reason = ""
        pnl = 0.0

        if spy_price is None:
            remaining.append(pos)
            continue

        expiry_dt = datetime.strptime(pos["expiry"], "%Y-%m-%d")
        days_to_expiry = (expiry_dt - today).days
        days_held = (today - datetime.strptime(pos["entry_date"], "%Y-%m-%d")).days

        # Current spread value
        if vix_current is not None:
            current_iv = vix_current / 100.0
        else:
            current_iv = pos["entry_iv"] * 0.85  # assume some crush
        T = max(days_to_expiry / 365.0, 1/365.0)

        short_val = bs_put(spy_price, pos["short_put"], T, RISK_FREE_RATE, current_iv)
        long_val = bs_put(spy_price, pos["long_put"], T, RISK_FREE_RATE, current_iv)
        spread_value = (short_val - long_val) * 100  # cost to close

        # 1) Expiry
        if days_to_expiry <= 0:
            should_close = True
            close_reason = "expiry"
            if spy_price >= pos["short_put"]:
                pnl = pos["credit_received"]
            else:
                intrinsic = (pos["short_put"] - max(spy_price, pos["long_put"])) * 100
                pnl = pos["credit_received"] - intrinsic

        # 2) Max hold days
        elif days_held >= MAX_HOLD_DAYS:
            should_close = True
            close_reason = f"max hold ({MAX_HOLD_DAYS}d)"
            pnl = pos["credit_received"] - spread_value

        # 3) VIX dropped below TP threshold (mean reversion happened)
        elif vix_current is not None and vix_current < VIX_TP_THRESHOLD:
            should_close = True
            close_reason = f"VIX reverted to {vix_current:.1f} (below {VIX_TP_THRESHOLD})"
            pnl = pos["credit_received"] - spread_value

        # 4) TP: spread value dropped to 20% of credit (captured 80% of premium)
        elif spread_value <= pos["tp_credit_remaining"]:
            should_close = True
            close_reason = f"TP: spread=${spread_value:.0f} <= target=${pos['tp_credit_remaining']:.0f}"
            pnl = pos["credit_received"] - spread_value

        # 5) Stop loss: spread cost doubled
        elif spread_value >= pos["stop_cost"]:
            should_close = True
            close_reason = f"STOP: spread=${spread_value:.0f} >= stop=${pos['stop_cost']:.0f}"
            pnl = pos["credit_received"] - spread_value

        if should_close:
            pnl = round(pnl - COMMISSION, 2)
            trade = {
                "ticker": "SPY",
                "entry_date": pos["entry_date"],
                "exit_date": str(today.date()),
                "short_put": pos["short_put"],
                "long_put": pos["long_put"],
                "credit_received": pos["credit_received"],
                "pnl": pnl,
                "return_pct": round(pnl / pos["max_risk"] * 100, 1) if pos["max_risk"] > 0 else 0,
                "close_reason": close_reason,
                "vix_at_entry": pos["vix_at_entry"],
                "vix_at_exit": round(vix_current, 2) if vix_current else None,
                "spy_entry": pos["entry_spy_price"],
                "spy_exit": round(spy_price, 2),
                "days_held": days_held,
                "pricing_method": "[BS-EST]",
            }
            state["closed_trades"].append(trade)
            state["capital"] += pnl
            state["total_pnl"] = state.get("total_pnl", 0) + pnl
            closed += 1
            log.info(
                f"CLOSED: SPY {pos['short_put']}/{pos['long_put']}P | "
                f"P&L=${pnl:.2f} ({trade['return_pct']:.0f}% on risk) | "
                f"VIX {pos['vix_at_entry']:.0f}->{vix_current:.1f if vix_current else '?'} | "
                f"{close_reason}"
            )
        else:
            unrealized = pos["credit_received"] - spread_value
            log.info(
                f"  HOLD: SPY {pos['short_put']}/{pos['long_put']}P | "
                f"unreal=${unrealized:.0f} spread_val=${spread_value:.0f} | "
                f"VIX={vix_current:.1f if vix_current else '?'} | "
                f"{days_to_expiry}d to exp, {days_held}d held"
            )
            remaining.append(pos)

    state["positions"] = remaining
    return closed


# ────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────
def run():
    log.info("=" * 60)
    log.info("VIX MEAN REVERSION SPREAD PAPER ENGINE — daily run")
    log.info("=" * 60)

    state = load_state()

    # 1) Check exits
    closed = check_exits(state)
    if closed:
        log.info(f"Closed {closed} positions")

    # 2) Check for new entry
    setup = check_for_entry(state)
    if setup:
        open_position(state, setup)
    else:
        log.info("No VIX entry signal today")

    # 3) Summary
    capital_in_use = sum(p.get("max_risk", 0) for p in state["positions"])
    total_trades = len(state["closed_trades"])
    winners = sum(1 for t in state["closed_trades"] if t["pnl"] > 0)
    wr = winners / total_trades * 100 if total_trades > 0 else 0
    avg_return = (
        np.mean([t["return_pct"] for t in state["closed_trades"]])
        if total_trades > 0 else 0
    )

    log.info(f"--- SUMMARY ---")
    log.info(f"Capital: ${state['capital']:.2f} | In use: ${capital_in_use:.2f}")
    log.info(f"Open: {len(state['positions'])} | Closed: {total_trades}")
    log.info(f"Win rate: {wr:.0f}% | Avg return on risk: {avg_return:.0f}%")
    log.info(f"Total P&L: ${state.get('total_pnl', 0):.2f}")

    # VIX history
    recent_vix = state.get("vix_signals", [])[-5:]
    if recent_vix:
        log.info("Recent VIX readings: " + ", ".join(
            f"{v['date']}={v['vix']:.1f}({v['action']})" for v in recent_vix
        ))

    save_state(state)
    log.info("Done.")


if __name__ == "__main__":
    run()
