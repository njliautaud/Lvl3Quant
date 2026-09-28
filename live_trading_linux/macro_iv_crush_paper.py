#!/usr/bin/env python3
"""
Macro Event IV Crush Paper Engine
===================================
Validates selling SPY straddles before FOMC/CPI events with REAL prices.

Strategy:
  - Open: 3-5 trading days before FOMC/CPI → sell ATM SPY straddle (call + put)
  - Close: 1 trading day after the event → buy back straddle
  - Edge: IV collapses after macro events are announced

Pricing:
  - Entry/exit: Alpaca real-time quotes via alpaca_options_pricing
  - Fallback: yfinance + Black-Scholes (flagged)

Events:
  - FOMC rate decisions (~8/year)
  - CPI releases (~12/year)
  = ~20 trades/year

Sizing:
  - Max premium spend: 5% NAV per position
  - Max 1 concurrent position (only 1 event at a time usually)
  - Commission: $0 (Robinhood, HC #694)

State: /home/jupiter/Lvl3Quant/data/paper_engines/macro_iv_crush/state.json
PM2: macro-iv-crush-paper
Cron: 9:35 AM ET on weekdays (check if event approaching/just happened)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "data" / "paper_engines" / "macro_iv_crush"
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE  = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"
EQUITY_FILE = STATE_DIR / "equity.csv"
LOG_DIR     = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [MACRO-IV-CRUSH] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_DIR / "macro_iv_crush_paper.log")),
    ],
)
log = logging.getLogger("MACRO-IV-CRUSH")

# ── Configuration ──────────────────────────────────────────────────────────────
STARTING_NAV        = 100_000.0
MAX_PREMIUM_PCT     = 0.05       # max 5% NAV on premium
DAYS_BEFORE_ENTRY   = 3          # sell straddle 3 trading days before event
DAYS_AFTER_EXIT     = 1          # close 1 trading day after event
DTE_TARGET          = 14         # target expiry ~14 DTE at entry
RISK_FREE           = 0.045

# ── Event Calendar ─────────────────────────────────────────────────────────────
# Known upcoming FOMC + CPI dates (extend as needed)
EVENT_CALENDAR: List[Dict] = [
    # 2026
    {"date": "2026-07-15", "type": "CPI"},
    {"date": "2026-07-16", "type": "FOMC_MINUTES"},
    {"date": "2026-07-29", "type": "FOMC"},
    {"date": "2026-08-12", "type": "CPI"},
    {"date": "2026-09-09", "type": "CPI"},
    {"date": "2026-09-16", "type": "FOMC"},
    {"date": "2026-10-14", "type": "CPI"},
    {"date": "2026-10-28", "type": "FOMC"},
    {"date": "2026-11-12", "type": "CPI"},
    {"date": "2026-12-09", "type": "CPI"},
    {"date": "2026-12-16", "type": "FOMC"},
]


def get_upcoming_events(today: date, lookahead_days: int = 7) -> List[Dict]:
    """Return events within the next lookahead_days trading days."""
    upcoming = []
    for ev in EVENT_CALENDAR:
        ev_date = date.fromisoformat(ev["date"])
        delta = (ev_date - today).days
        if 0 <= delta <= lookahead_days:
            upcoming.append({**ev, "date": ev_date, "days_away": delta})
    return sorted(upcoming, key=lambda x: x["date"])


def get_recent_events(today: date, lookback_days: int = 3) -> List[Dict]:
    """Return events in the past lookback_days (need to close positions)."""
    recent = []
    for ev in EVENT_CALENDAR:
        ev_date = date.fromisoformat(ev["date"])
        delta = (today - ev_date).days
        if 1 <= delta <= lookback_days:
            recent.append({**ev, "date": ev_date, "days_ago": delta})
    return sorted(recent, key=lambda x: x["date"])


# ── State Management ───────────────────────────────────────────────────────────

def load_state() -> Dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "nav": STARTING_NAV,
        "positions": [],
        "closed_trades": [],
        "run_count": 0,
        "last_run": None,
    }


def save_state(state: Dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def log_trade(trade: Dict) -> None:
    with open(TRADES_FILE, "a") as f:
        f.write(json.dumps(trade, default=str) + "\n")

    with open(EQUITY_FILE, "a") as f:
        f.write(f"{datetime.now().date()},{trade.get('nav_after', 0)}\n")


# ── Pricing ────────────────────────────────────────────────────────────────────

def get_spy_straddle_price(strike: Optional[float] = None, dte: int = DTE_TARGET) -> Optional[Dict]:
    """
    Get real-time SPY straddle price via Alpaca options pricing.
    Falls back to yfinance + BS.
    """
    try:
        sys.path.insert(0, str(ROOT / "live_trading_linux"))
        from alpaca_options_pricing import get_straddle_quotes
        result = get_straddle_quotes("SPY", dte_target=dte, strike=strike)
        if result:
            return result
    except Exception as e:
        log.warning(f"Alpaca pricing failed: {e}")

    # Fallback: yfinance + Black-Scholes
    return _bs_straddle_fallback(dte)


def _bs_fallback_price(S, K, T, r, sigma, option_type):
    """Black-Scholes option price."""
    from scipy.stats import norm
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    d1 = (math.log(S/K) + (r + sigma**2/2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if option_type == "call":
        return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    else:
        return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def _bs_straddle_fallback(dte: int) -> Optional[Dict]:
    """Compute straddle price using BS + current SPY/VIX from yfinance."""
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        hist = spy.history(period="5d")
        if hist.empty:
            return None
        spot = float(hist["Close"].iloc[-1])

        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="2d")
        sigma = float(vix_hist["Close"].iloc[-1]) / 100 if not vix_hist.empty else 0.18

        T = dte / 365
        K = round(spot)  # ATM

        call_price = _bs_fallback_price(spot, K, T, RISK_FREE, sigma, "call")
        put_price  = _bs_fallback_price(spot, K, T, RISK_FREE, sigma, "put")
        straddle   = call_price + put_price

        return {
            "spot": round(spot, 2),
            "strike": K,
            "dte": dte,
            "call_price": round(call_price, 2),
            "put_price": round(put_price, 2),
            "straddle_price": round(straddle, 2),
            "sigma": round(sigma, 4),
            "source": "BS_fallback",
        }
    except Exception as e:
        log.error(f"BS fallback failed: {e}")
        return None


# ── Core Logic ─────────────────────────────────────────────────────────────────

def open_position(state: Dict, event: Dict, pricing: Dict) -> None:
    """Open a short straddle position."""
    nav = state["nav"]
    straddle_price = pricing["straddle_price"]
    contracts = max(1, int((nav * MAX_PREMIUM_PCT) / (straddle_price * 100)))
    premium_received = straddle_price * contracts * 100

    position = {
        "id": f"{event['type']}_{event['date']}",
        "event_type": event["type"],
        "event_date": str(event["date"]),
        "entry_date": str(date.today()),
        "strike": pricing["strike"],
        "dte_at_entry": pricing["dte"],
        "straddle_price_entry": straddle_price,
        "contracts": contracts,
        "premium_received": round(premium_received, 2),
        "pricing_source": pricing.get("source", "alpaca"),
        "spot_at_entry": pricing.get("spot"),
    }

    state["positions"].append(position)
    log.info(f"OPENED: {position['id']} | {contracts} contracts × ${straddle_price:.2f} | "
             f"premium ${premium_received:.0f} | strike ${position['strike']:.0f}")

    log_trade({
        "action": "open",
        "position": position,
        "nav_after": round(state["nav"], 2),
        "timestamp": datetime.now().isoformat(),
    })


def close_position(state: Dict, pos: Dict, pricing: Dict) -> float:
    """Close a short straddle position. Returns P&L."""
    straddle_price_exit = pricing["straddle_price"]
    contracts = pos["contracts"]

    buyback_cost = straddle_price_exit * contracts * 100
    premium_received = pos["premium_received"]
    pnl = premium_received - buyback_cost

    log.info(f"CLOSED: {pos['id']} | entry ${pos['straddle_price_entry']:.2f} exit ${straddle_price_exit:.2f} | "
             f"IV crush {(pos['straddle_price_entry'] - straddle_price_exit)/pos['straddle_price_entry']*100:.1f}% | "
             f"PnL ${pnl:+.0f}")

    closed = {**pos,
              "exit_date": str(date.today()),
              "straddle_price_exit": straddle_price_exit,
              "buyback_cost": round(buyback_cost, 2),
              "pnl": round(pnl, 2),
              "iv_crush_pct": round((pos["straddle_price_entry"] - straddle_price_exit) /
                                    pos["straddle_price_entry"] * 100, 1),
              "pricing_source_exit": pricing.get("source", "alpaca"),
    }

    state["closed_trades"].append(closed)
    state["nav"] += pnl

    log_trade({
        "action": "close",
        "closed": closed,
        "nav_after": round(state["nav"], 2),
        "timestamp": datetime.now().isoformat(),
    })

    return pnl


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    today = date.today()
    log.info(f"{'='*60}")
    log.info(f"Macro IV Crush Paper Engine — {today}")
    log.info(f"{'='*60}")

    state = load_state()
    state["run_count"] = state.get("run_count", 0) + 1
    state["last_run"] = str(today)

    log.info(f"NAV: ${state['nav']:,.2f} | Open positions: {len(state['positions'])} | "
             f"Closed trades: {len(state['closed_trades'])}")

    # 1. Close any positions where event has passed
    recent_events = get_recent_events(today, lookback_days=DAYS_AFTER_EXIT + 1)
    positions_to_close = []

    for pos in state["positions"]:
        pos_event_date = date.fromisoformat(str(pos["event_date"]))
        days_since_event = (today - pos_event_date).days
        if days_since_event >= DAYS_AFTER_EXIT:
            positions_to_close.append(pos)

    for pos in positions_to_close:
        log.info(f"Closing position {pos['id']} (event was {pos['event_date']})")
        pricing = get_spy_straddle_price(strike=pos.get("strike"), dte=max(5, pos["dte_at_entry"] - 14))
        if pricing is None:
            log.error(f"Cannot get pricing to close {pos['id']} — skipping")
            continue
        pnl = close_position(state, pos, pricing)
        state["positions"].remove(pos)
        log.info(f"Position closed. NAV: ${state['nav']:,.2f} (P&L: ${pnl:+.0f})")

    # 2. Open new positions for upcoming events
    upcoming_events = get_upcoming_events(today, lookahead_days=DAYS_BEFORE_ENTRY + 1)

    # Check which events we don't already have positions for
    open_event_ids = {p["id"] for p in state["positions"]}

    for event in upcoming_events:
        event_id = f"{event['type']}_{event['date']}"
        if event_id in open_event_ids:
            log.info(f"Already have position for {event_id} — skipping")
            continue

        days_away = event.get("days_away", 99)
        if days_away < 2 or days_away > DAYS_BEFORE_ENTRY:
            log.info(f"Event {event_id} is {days_away} days away — not time to enter")
            continue

        log.info(f"Entering position for {event['type']} on {event['date']} ({days_away} days away)")
        pricing = get_spy_straddle_price(dte=DTE_TARGET)
        if pricing is None:
            log.error(f"Cannot get pricing for {event_id} — skipping")
            continue
        if pricing["straddle_price"] <= 0:
            log.error(f"Zero straddle price for {event_id} — skipping")
            continue

        open_position(state, event, pricing)

    # 3. Print status
    log.info(f"\n{'='*40}")
    log.info(f"STATUS | NAV: ${state['nav']:,.2f} | Open: {len(state['positions'])}")
    for pos in state["positions"]:
        log.info(f"  {pos['id']}: strike=${pos['strike']:.0f} contracts={pos['contracts']} "
                 f"premium=${pos['premium_received']:.0f}")

    if state["closed_trades"]:
        total_pnl = sum(t["pnl"] for t in state["closed_trades"])
        win_rate  = sum(1 for t in state["closed_trades"] if t["pnl"] > 0) / len(state["closed_trades"])
        log.info(f"Closed trades: {len(state['closed_trades'])} | Total P&L: ${total_pnl:+,.0f} | "
                 f"Win rate: {win_rate:.1%}")

    save_state(state)
    log.info("Done.")


if __name__ == "__main__":
    run()
