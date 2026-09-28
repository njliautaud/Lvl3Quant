"""
Position State Manager — single authority for position tracking.

Manages:
- active_options.json (open/closed positions, bracket order IDs)
- rh_position_state.json (peak tracking, watchdog state)
- Trade journal (append-only log of all trades)

Every script that reads or writes position data MUST go through this module.
"""

import json
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from lib.constants import (
    ACTIVE_OPTIONS_FILE, POSITION_STATE_FILE, TRADE_JOURNAL_FILE,
    SECTOR_MAX_HOLD_DAYS, DEFAULT_MAX_HOLD_DAYS,
)


def _load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_json(path: Path, data: dict):
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ──────────────────────────────────────────────────────────────
# ACTIVE OPTIONS (primary position tracker)
# ──────────────────────────────────────────────────────────────

def load_active_options() -> dict:
    """Load active_options.json. Creates with defaults if missing."""
    data = _load_json(ACTIVE_OPTIONS_FILE)
    if "positions" not in data:
        data["positions"] = []
    if "closed" not in data:
        data["closed"] = []
    if "equity_positions" not in data:
        data["equity_positions"] = []
    return data


def save_active_options(data: dict):
    data["last_updated"] = datetime.utcnow().isoformat() + "Z"
    _save_json(ACTIVE_OPTIONS_FILE, data)


def get_open_positions() -> list:
    """Return list of open positions.
    Accepts 'open' or 'filled' (entry order filled = position open).
    Excludes closed/expired/cancelled positions.
    """
    INACTIVE = {"closed", "expired", "cancelled"}
    data = load_active_options()
    return [p for p in data["positions"]
            if p.get("status", "").lower() not in INACTIVE]


def add_position(
    ticker: str,
    option_type: str,       # "call" or "put"
    strike: float,
    expiration: str,        # YYYY-MM-DD
    entry_price: float,     # Per-contract option premium
    underlying_entry: float,# Underlying price at entry
    quantity: int = 1,
    strategy: str = "",
    n_sources: int = 0,
    confidence: float = 0.0,
    spread_type: str = "",  # "bull_call_spread", "bear_put_spread", or "" for single-leg
    legs: list = None,      # For spreads: list of leg dicts
    tp_order_id: str = None,
    sl_order_id: str = None,
    notes: str = "",
) -> dict:
    """Add a new position. Returns the position dict."""
    data = load_active_options()

    position = {
        "ticker": ticker,
        "option_type": option_type,
        "strike": strike,
        "expiration": expiration,
        "entry_price": entry_price,
        "entry_date": date.today().isoformat(),
        "entry_time": datetime.now().isoformat(),
        "underlying_entry_price": underlying_entry,
        "quantity": quantity,
        "strategy": strategy,
        "n_sources": n_sources,
        "confidence": confidence,
        "spread_type": spread_type,
        "legs": legs or [],
        "peak_value": entry_price,
        "peak_date": date.today().isoformat(),
        "trailing_active": False,
        "tp_order_id": tp_order_id,
        "sl_order_id": sl_order_id,
        "status": "open",
        "last_check_pnl": 0.0,
        "last_check_time": datetime.now().isoformat(),
        "notes": notes,
    }

    data["positions"].append(position)
    save_active_options(data)

    # Also log to trade journal
    _log_trade_event("ENTRY", position)

    return position


def update_position_brackets(ticker: str, strike: float, tp_order_id: str = None, sl_order_id: str = None):
    """Update bracket order IDs on an open position."""
    data = load_active_options()
    for pos in data["positions"]:
        if (pos.get("ticker") == ticker and
            pos.get("strike") == strike and
            pos.get("status", "").lower() not in {"closed", "expired", "cancelled"}):
            if tp_order_id is not None:
                pos["tp_order_id"] = tp_order_id
            if sl_order_id is not None:
                pos["sl_order_id"] = sl_order_id
            break
    save_active_options(data)


def update_peak(ticker: str, strike: float, current_mark: float) -> bool:
    """Update peak_value if current_mark is new high. Returns True if new peak."""
    data = load_active_options()
    new_peak = False
    for pos in data["positions"]:
        if (pos.get("ticker") == ticker and
            pos.get("strike") == strike and
            pos.get("status", "").lower() not in {"closed", "expired", "cancelled"}):
            if current_mark > pos.get("peak_value", 0):
                pos["peak_value"] = current_mark
                pos["peak_date"] = date.today().isoformat()
                new_peak = True
            entry = pos.get("entry_price", pos.get("entry_debit", current_mark))
            pos["last_check_pnl"] = round((current_mark - entry) / entry * 100, 1) if entry else 0.0
            pos["last_check_time"] = datetime.now().isoformat()
            break
    save_active_options(data)
    return new_peak


def close_position(
    ticker: str,
    strike: float,
    exit_price: float,
    exit_reason: str,
    notes: str = "",
):
    """Move position from open to closed. Compute realized P&L."""
    data = load_active_options()

    for i, pos in enumerate(data["positions"]):
        if (pos.get("ticker") == ticker and
            pos.get("strike") == strike and
            pos.get("status", "").lower() not in {"closed", "expired", "cancelled"}):

            entry = pos["entry_price"]
            qty = pos.get("quantity", 1)
            pnl_pct = round((exit_price - entry) / entry * 100, 1)
            pnl_dollars = round((exit_price - entry) * 100 * qty, 2)  # Options are 100x
            days_held = _trading_days_between(pos["entry_date"], date.today().isoformat())

            pos["status"] = "closed"
            pos["exit_price"] = exit_price
            pos["exit_date"] = date.today().isoformat()
            pos["exit_time"] = datetime.now().isoformat()
            pos["exit_reason"] = exit_reason
            pos["pnl_pct"] = pnl_pct
            pos["pnl_dollars"] = pnl_dollars
            pos["days_held"] = days_held
            if notes:
                pos["notes"] = notes

            # Move to closed array
            data["closed"].insert(0, pos)
            data["positions"].pop(i)

            # Update track record
            _update_track_record(data)

            save_active_options(data)

            # Log to journal
            _log_trade_event("EXIT", pos)

            return pos

    return None  # Position not found


def positions_needing_brackets() -> list:
    """Return open positions that don't have both bracket orders."""
    positions = get_open_positions()
    return [
        p for p in positions
        if not p.get("spread_type")  # Single-leg only
        and (not p.get("tp_order_id") or not p.get("sl_order_id"))
    ]


def set_trailing_active(ticker: str, strike: float):
    """Mark trailing stop as activated for a position."""
    data = load_active_options()
    for pos in data["positions"]:
        if (pos.get("ticker") == ticker and
            pos.get("strike") == strike and
            pos.get("status", "").lower() not in {"closed", "expired", "cancelled"}):
            pos["trailing_active"] = True
            pos["trailing_activated_at"] = datetime.now().isoformat()
            break
    save_active_options(data)


# ──────────────────────────────────────────────────────────────
# POSITION STATE (peak tracking for watchdog)
# ──────────────────────────────────────────────────────────────

def load_position_state() -> dict:
    data = _load_json(POSITION_STATE_FILE)
    if "positions" not in data:
        data["positions"] = {}
    return data


def save_position_state(data: dict):
    _save_json(POSITION_STATE_FILE, data)


def sync_state_from_active():
    """Sync rh_position_state.json from active_options.json (source of truth)."""
    active = load_active_options()
    state = load_position_state()

    INACTIVE = {"closed", "expired", "cancelled"}
    for pos in active["positions"]:
        if pos.get("status", "").lower() in INACTIVE:
            continue
        opt_type = pos.get("option_type", pos.get("type", "call"))
        key = f"{pos['ticker']}_{pos['strike']}_{opt_type}"
        entry_price = pos.get("entry_price", pos.get("entry_debit", 0))
        expiry = pos.get("expiration", pos.get("expiry", ""))
        state["positions"][key] = {
            "symbol": pos["ticker"],
            "strike": pos["strike"],
            "option_type": opt_type,
            "expiration": expiry,
            "entry_price": entry_price,
            "entry_date": pos["entry_date"],
            "underlying_entry_price": pos.get("underlying_entry_price", 0),
            "peak_option_price": pos.get("peak_value", entry_price),
            "peak_value": pos.get("peak_value", entry_price),
            "trailing_stop_active": pos.get("trailing_active", False),
            "tp_order_id": pos.get("tp_order_id"),
            "sl_order_id": pos.get("sl_order_id"),
            "n_sources": pos.get("n_sources", 0),
            "status": "open",
        }

    save_position_state(state)


# ──────────────────────────────────────────────────────────────
# TRADE JOURNAL (append-only)
# ──────────────────────────────────────────────────────────────

def _log_trade_event(event_type: str, position: dict):
    """Append a trade event to the journal."""
    journal = _load_json(TRADE_JOURNAL_FILE)
    if "events" not in journal:
        journal["events"] = []

    journal["events"].append({
        "event": event_type,
        "timestamp": datetime.now().isoformat(),
        "ticker": position.get("ticker"),
        "option_type": position.get("option_type"),
        "strike": position.get("strike"),
        "entry_price": position.get("entry_price"),
        "exit_price": position.get("exit_price"),
        "exit_reason": position.get("exit_reason"),
        "pnl_pct": position.get("pnl_pct"),
        "pnl_dollars": position.get("pnl_dollars"),
        "days_held": position.get("days_held"),
        "n_sources": position.get("n_sources"),
        "strategy": position.get("strategy"),
        "tp_order_id": position.get("tp_order_id"),
        "sl_order_id": position.get("sl_order_id"),
    })

    # Keep last 200 events
    journal["events"] = journal["events"][-200:]
    _save_json(TRADE_JOURNAL_FILE, journal)


def _update_track_record(data: dict):
    """Recompute track record from closed positions."""
    closed = data.get("closed", [])
    wins = sum(1 for t in closed if (t.get("pnl_pct") or 0) > 0)
    losses = sum(1 for t in closed if (t.get("pnl_pct") or 0) <= 0)
    total_pnl = sum(t.get("pnl_dollars", 0) for t in closed)
    wr = round(wins / (wins + losses) * 100, 0) if (wins + losses) > 0 else 0
    data["track_record"] = f"{wins}W / {losses}L (${total_pnl:+.0f} realized). {wr:.0f}% WR."


def get_no_reentry_tickers(days: int = 10) -> dict:
    """Return tickers with recent exits and their cooldown end dates.
    Returns {ticker: {strike: cooldown_end_date, ...}, ...}
    """
    data = load_active_options()
    today = date.today()
    blocked = {}

    for trade in data.get("closed", []):
        exit_date_str = trade.get("exit_date")
        if not exit_date_str:
            continue
        exit_date = datetime.strptime(exit_date_str, "%Y-%m-%d").date()
        days_since = (today - exit_date).days

        ticker = trade.get("ticker", "")
        strike = trade.get("strike", 0)

        # Same ticker+strike: 10-day cooldown (HC #809 R4)
        if days_since < 10:
            if ticker not in blocked:
                blocked[ticker] = {}
            blocked[ticker][str(strike)] = (exit_date + __import__('datetime').timedelta(days=10)).isoformat()

        # Same ticker, different strike: 5-day cooldown
        if days_since < 5:
            if ticker not in blocked:
                blocked[ticker] = {}
            blocked[ticker]["_any_strike"] = (exit_date + __import__('datetime').timedelta(days=5)).isoformat()

    return blocked


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _trading_days_between(start_str: str, end_str: str) -> int:
    """Count weekdays between two dates."""
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    count = 0
    current = start
    while current < end:
        current += __import__('datetime').timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return count
