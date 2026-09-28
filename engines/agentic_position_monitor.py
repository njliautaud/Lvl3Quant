#!/usr/bin/env python3
"""
Agentic Position & Order Monitor
=================================
Autonomous monitor for Robinhood options/equities positions.
Checks that every open position has a corresponding limit sell order,
monitors P&L vs targets/stops, and alerts on thesis violations.

Runs via PM2, checks every 15min during market hours, 60min outside.

PM2 ecosystem entry (add to an ecosystem.config.js):
-------------------------------------------------------
{
  name: "agentic-position-monitor",
  script: "/home/jupiter/Lvl3Quant/engines/agentic_position_monitor.py",
  interpreter: "python3",
  cwd: "/home/jupiter/Lvl3Quant",
  autorestart: true,
  max_restarts: 50,
  restart_delay: 10000,
  watch: false,
  env: {
    PYTHONUNBUFFERED: "1",
  },
  log_file: "/home/jupiter/Lvl3Quant/logs/agentic_position_monitor.log",
  error_file: "/home/jupiter/Lvl3Quant/logs/agentic_position_monitor_err.log",
  out_file: "/home/jupiter/Lvl3Quant/logs/agentic_position_monitor_out.log",
  merge_logs: true,
  time: true,
}
-------------------------------------------------------
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timedelta, date
from pathlib import Path

import pytz

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "agentic_positions.json"

# External position files to sync from
RH_POSITION_STATE_FILE = ROOT / "data" / "rh_position_state.json"
ACTIVE_OPTIONS_FILE = ROOT / "data" / "active_options.json"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "agentic_position_monitor.log"

WEBHOOK_FILE = ROOT.parent / "teleclaude-main" / "API_KEYS.md"

ET = pytz.timezone("US/Eastern")

# ── Timing ─────────────────────────────────────────────────────────────────
MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 9, 30
MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN = 16, 0
CHECK_INTERVAL_MARKET = 15 * 60      # 15 minutes
CHECK_INTERVAL_OFF_HOURS = 60 * 60   # 1 hour
ALERT_COOLDOWN = 4 * 3600            # 4 hours between duplicate alerts
EARNINGS_PROXIMITY_DAYS = 5          # alert if earnings within N biz days

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PositionMonitor] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("agentic_position_monitor")


# ═══════════════════════════════════════════════════════════════════════════
# DISCORD WEBHOOK
# ═══════════════════════════════════════════════════════════════════════════

def get_discord_webhook() -> str | None:
    """Extract Discord webhook URL from API_KEYS.md."""
    try:
        if WEBHOOK_FILE.exists():
            text = WEBHOOK_FILE.read_text()
            for line in text.split("\n"):
                if "discord" in line.lower() and "webhook" in line.lower() and "http" in line:
                    urls = re.findall(r"https://discord\.com/api/webhooks/\S+", line)
                    if urls:
                        return urls[0].strip("`").strip()
    except Exception:
        pass
    # Fallback: environment variable
    return os.environ.get("DISCORD_WEBHOOK_URL")


PENDING_ALERTS_FILE = ROOT / "state" / "pending_alerts.json"


def write_pending_alert(title: str, body: str, conviction: str = "HIGH"):
    """Write alert to pending_alerts.json for alert-router pickup."""
    try:
        existing = []
        if PENDING_ALERTS_FILE.exists():
            try:
                existing = json.loads(PENDING_ALERTS_FILE.read_text())
                if not isinstance(existing, list):
                    existing = []
            except (json.JSONDecodeError, OSError):
                existing = []
        alert = {
            "timestamp": datetime.now(ET).isoformat(),
            "title": title,
            "body": body,
            "conviction": conviction,
            "source": "agentic_position_monitor",
        }
        existing.append(alert)
        if len(existing) > 100:
            existing = existing[-100:]
        tmp = PENDING_ALERTS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(existing, indent=2))
        tmp.rename(PENDING_ALERTS_FILE)
        log.info(f"Alert written to pending_alerts.json: {title}")
    except Exception as e:
        log.error(f"Failed to write pending alert: {e}")


def send_discord(message: str):
    """Send a message to Discord via webhook. Falls back to pending_alerts.json."""
    # Always write to pending alerts for alert-router
    write_pending_alert("Position Monitor Alert", message)

    webhook_url = get_discord_webhook()
    if not webhook_url:
        log.info("No direct webhook — alert routed through pending_alerts.json")
        return

    message = message[:1950]  # Discord limit is 2000
    try:
        data = json.dumps({"content": message}).encode()
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        log.info(f"Discord alert sent ({len(message)} chars)")
    except Exception as e:
        log.error(f"Discord send failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def _default_state() -> dict:
    """
    State schema:
    {
      "positions": {
        "<position_id>": {
          "symbol": "AAPL",
          "type": "option" | "equity",
          "option_details": "AAPL 2026-08-15 250C" | null,
          "quantity": 2,
          "entry_price": 5.40,
          "current_price": null,
          "target_price": 8.00,
          "stop_price": 3.50,
          "entry_date": "2026-07-22",
          "thesis": "Earnings run-up, IV expansion expected",
          "key_signals": ["AAPL > 245 support", "VIX < 18"],
          "signal_status": {},
          "limit_sell_placed": false,
          "limit_sell_order_id": null,
          "status": "OPEN",
          "alerts_sent": {},
          "notes": ""
        }
      },
      "earnings_calendar": {},
      "last_check": null,
      "alert_cooldowns": {},
      "version": "agentic-position-monitor-v1"
    }
    """
    return {
        "positions": {},
        "earnings_calendar": {},
        "last_check": None,
        "alert_cooldowns": {},
        "version": "agentic-position-monitor-v1",
    }


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
            # Ensure required keys exist
            state.setdefault("positions", {})
            state.setdefault("earnings_calendar", {})
            state.setdefault("last_check", None)
            state.setdefault("alert_cooldowns", {})
            return state
        except (json.JSONDecodeError, OSError):
            log.warning("Corrupt state file, starting fresh")
    return _default_state()


def save_state(state: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(STATE_FILE)


# ═══════════════════════════════════════════════════════════════════════════
# MARKET HOURS
# ═══════════════════════════════════════════════════════════════════════════

def is_market_hours() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN, second=0)
    return market_open <= now <= market_close


def is_weekday() -> bool:
    return datetime.now(ET).weekday() < 5


# ═══════════════════════════════════════════════════════════════════════════
# ALERT COOLDOWN
# ═══════════════════════════════════════════════════════════════════════════

def should_alert(state: dict, alert_key: str) -> bool:
    """Check if alert_key is past its cooldown window."""
    cooldowns = state.get("alert_cooldowns", {})
    if alert_key in cooldowns:
        try:
            last = datetime.fromisoformat(cooldowns[alert_key])
            if last.tzinfo is None:
                last = ET.localize(last)
            elapsed = (datetime.now(ET) - last).total_seconds()
            if elapsed < ALERT_COOLDOWN:
                return False
        except (ValueError, TypeError):
            pass
    return True


def mark_alerted(state: dict, alert_key: str):
    state.setdefault("alert_cooldowns", {})[alert_key] = datetime.now(ET).isoformat()


# ═══════════════════════════════════════════════════════════════════════════
# EARNINGS CHECK (PLACEHOLDER)
# ═══════════════════════════════════════════════════════════════════════════

def get_earnings_date(symbol: str, state: dict) -> date | None:
    """
    Return the next known earnings date for symbol, or None.

    Currently reads from state["earnings_calendar"] which can be populated
    manually or by a separate data-fetch job. Format:
      {"AAPL": "2026-07-31", "MSFT": "2026-07-29", ...}

    TODO: Wire in real data source (Robinhood earnings calendar MCP,
    or FMP/Polygon earnings endpoint).
    """
    cal = state.get("earnings_calendar", {})
    date_str = cal.get(symbol)
    if date_str:
        try:
            return date.fromisoformat(date_str)
        except ValueError:
            pass
    return None


def earnings_within_window(symbol: str, state: dict) -> tuple[bool, date | None]:
    """Check if symbol has earnings within EARNINGS_PROXIMITY_DAYS business days."""
    earn_date = get_earnings_date(symbol, state)
    if earn_date is None:
        return False, None

    today = date.today()
    if earn_date < today:
        return False, None  # Past earnings, ignore

    # Count business days between today and earnings
    biz_days = 0
    d = today
    while d < earn_date:
        d += timedelta(days=1)
        if d.weekday() < 5:
            biz_days += 1

    return biz_days <= EARNINGS_PROXIMITY_DAYS, earn_date


# ═══════════════════════════════════════════════════════════════════════════
# CORE MONITOR CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def check_limit_sell_gaps(state: dict) -> list[str]:
    """
    GAP ALERT: Find open positions with no limit sell order placed.
    This was the critical gap the user identified.
    """
    alerts = []
    for pid, pos in state["positions"].items():
        if pos.get("status") != "OPEN":
            continue
        if not pos.get("limit_sell_placed", False):
            alert_key = f"gap_{pid}"
            if should_alert(state, alert_key):
                symbol = pos.get("symbol", "???")
                details = pos.get("option_details", "equity")
                target = pos.get("target_price", "N/A")
                alerts.append(
                    f"**GAP ALERT** -- {symbol} ({details}): "
                    f"NO limit sell order placed! Target: ${target}. "
                    f"Place a limit sell NOW."
                )
                mark_alerted(state, alert_key)
    return alerts


def check_profit_targets(state: dict) -> list[str]:
    """Alert when position reaches 80%+ of profit target."""
    alerts = []
    for pid, pos in state["positions"].items():
        if pos.get("status") != "OPEN":
            continue
        current = pos.get("current_price")
        entry = pos.get("entry_price")
        target = pos.get("target_price")
        if current is None or entry is None or target is None:
            continue

        target_move = target - entry
        if target_move <= 0:
            continue
        current_move = current - entry
        pct_of_target = current_move / target_move

        if pct_of_target >= 0.80:
            alert_key = f"target80_{pid}"
            if should_alert(state, alert_key):
                symbol = pos.get("symbol", "???")
                gain_pct = (current_move / entry) * 100
                alerts.append(
                    f"**TARGET APPROACHING** -- {symbol}: "
                    f"at ${current:.2f} ({gain_pct:+.1f}%), "
                    f"{pct_of_target:.0%} of target ${target:.2f}. "
                    f"Consider taking profits."
                )
                mark_alerted(state, alert_key)
    return alerts


def check_stop_levels(state: dict) -> list[str]:
    """Alert when position hits stop level."""
    alerts = []
    for pid, pos in state["positions"].items():
        if pos.get("status") != "OPEN":
            continue
        current = pos.get("current_price")
        stop = pos.get("stop_price")
        if current is None or stop is None:
            continue

        if current <= stop:
            alert_key = f"stop_{pid}"
            if should_alert(state, alert_key):
                symbol = pos.get("symbol", "???")
                entry = pos.get("entry_price", 0)
                loss_pct = ((current - entry) / entry) * 100 if entry else 0
                alerts.append(
                    f"**STOP HIT** -- {symbol}: "
                    f"at ${current:.2f} ({loss_pct:+.1f}%), "
                    f"stop was ${stop:.2f}. "
                    f"EXIT or reassess thesis NOW."
                )
                mark_alerted(state, alert_key)
    return alerts


def check_thesis_signals(state: dict) -> list[str]:
    """
    Check if any key signals for a position have flipped.
    signal_status stores the last known state of each signal.
    If a signal was True and is now False, alert.

    Note: signal evaluation is currently manual — update signal_status
    in the state file when you check signals. A future version can
    wire in live price checks.
    """
    alerts = []
    for pid, pos in state["positions"].items():
        if pos.get("status") != "OPEN":
            continue
        signal_status = pos.get("signal_status", {})
        for signal_name, is_valid in signal_status.items():
            if is_valid is False:
                alert_key = f"signal_{pid}_{signal_name}"
                if should_alert(state, alert_key):
                    symbol = pos.get("symbol", "???")
                    alerts.append(
                        f"**THESIS SIGNAL REVERSED** -- {symbol}: "
                        f"'{signal_name}' no longer holds. "
                        f"Review position and thesis."
                    )
                    mark_alerted(state, alert_key)
    return alerts


def check_target_ratchet(state: dict) -> list[str]:
    """
    HC #742: If current option price exceeds the limit sell target,
    the target is STALE. Auto-ratchet the target upward and alert.

    New target = current_price rounded UP to nearest $0.10 increment.
    This prevents selling at a stale price after an overnight gap-up.
    """
    import math

    alerts = []
    for pid, pos in state["positions"].items():
        if pos.get("status") != "OPEN":
            continue
        current = pos.get("current_price")
        target = pos.get("target_price")
        if current is None or target is None:
            continue

        if current > target:
            # Target is stale — ratchet up
            # Round current price UP to nearest $0.10
            new_target = math.ceil(current * 10) / 10
            old_target = target
            pos["target_price"] = new_target

            # Mark limit sell as needing replacement
            if pos.get("limit_sell_placed"):
                pos["limit_sell_placed"] = False
                pos["limit_sell_order_id"] = None
                pos["notes"] = (
                    pos.get("notes", "")
                    + f" | AUTO-CANCELED stale limit sell at ${old_target:.2f} — "
                    f"price ${current:.2f} exceeded target."
                )

            symbol = pos.get("symbol", "???")
            details = pos.get("option_details", "")
            alert_key = f"ratchet_{pid}_{new_target:.2f}"
            if should_alert(state, alert_key):
                alerts.append(
                    f"**TARGET RATCHETED** — {symbol} ({details}): "
                    f"price moved to ${current:.2f}, past old target ${old_target:.2f}. "
                    f"New target raised to ${new_target:.2f}. "
                    f"Old limit sell is stale — needs replacement."
                )
                mark_alerted(state, alert_key)
                log.info(
                    f"Target ratcheted for {symbol}: "
                    f"${old_target:.2f} -> ${new_target:.2f} "
                    f"(current: ${current:.2f})"
                )
    return alerts


def is_pre_market() -> bool:
    """Check if we're in the pre-market window (8:00-9:25 ET weekdays)."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    pre_open = now.replace(hour=8, minute=0, second=0)
    pre_close = now.replace(hour=9, minute=25, second=0)
    return pre_open <= now <= pre_close


def check_earnings_proximity(state: dict) -> list[str]:
    """Alert if any held position has earnings approaching."""
    alerts = []
    for pid, pos in state["positions"].items():
        if pos.get("status") != "OPEN":
            continue
        symbol = pos.get("symbol", "???")
        within_window, earn_date = earnings_within_window(symbol, state)
        if within_window and earn_date:
            alert_key = f"earnings_{pid}_{earn_date.isoformat()}"
            if should_alert(state, alert_key):
                biz_days = 0
                d = date.today()
                while d < earn_date:
                    d += timedelta(days=1)
                    if d.weekday() < 5:
                        biz_days += 1
                alerts.append(
                    f"**EARNINGS WARNING** -- {symbol}: "
                    f"reports earnings on {earn_date.strftime('%b %d')} "
                    f"({biz_days} business days away). "
                    f"Review position for IV crush risk."
                )
                mark_alerted(state, alert_key)
    return alerts


def should_skip_entry(symbol: str, state: dict) -> tuple[bool, str]:
    """
    Pre-entry filter: returns (should_skip, reason).
    Call this before adding a new position.
    """
    within_window, earn_date = earnings_within_window(symbol, state)
    if within_window and earn_date:
        return True, (
            f"{symbol} has earnings on {earn_date.strftime('%b %d')} "
            f"(within {EARNINGS_PROXIMITY_DAYS} business days) — SKIP entry"
        )
    return False, ""


# ═══════════════════════════════════════════════════════════════════════════
# TRAILING STOP RATCHETING
# ═══════════════════════════════════════════════════════════════════════════

def ratchet_trailing_stops(state: dict) -> list[str]:
    """
    Auto-ratchet trailing stops based on stock price HWM.

    Trailing stop config lives in position["trailing_stop"]:
      method: "stock_price" (trail on underlying) or "option_price"
      trail_pct: percentage below HWM to set stop (e.g., 2.5 = 2.5%)
      stock_hwm: highest stock price seen
      current_stop: current trailing stop level
      auto_ratchet: bool — if True, automatically move stop up

    Stock price trailing is preferred for directional thesis trades because:
    - Option prices are noisy (theta, IV, gamma effects)
    - The thesis is about stock movement, stop should break when thesis breaks
    - Support/resistance levels are on stock charts, not option chains
    """
    alerts = []
    for pid, pos in state["positions"].items():
        if pos.get("status") != "OPEN":
            continue
        ts = pos.get("trailing_stop")
        if not ts or not ts.get("auto_ratchet", False):
            continue

        current_stock_price = pos.get("current_stock_price")
        if current_stock_price is None:
            continue

        method = ts.get("method", "stock_price")
        trail_pct = ts.get("trail_pct", 2.5)
        old_hwm = ts.get("stock_hwm", 0)
        old_stop = ts.get("current_stop", 0)

        if method == "stock_price" and current_stock_price > old_hwm:
            new_hwm = current_stock_price
            new_stop = round(new_hwm * (1 - trail_pct / 100), 2)

            if new_stop > old_stop:
                ts["stock_hwm"] = new_hwm
                ts["current_stop"] = new_stop
                symbol = pos.get("symbol", "???")
                log.info(
                    f"Trailing stop ratcheted for {symbol}: "
                    f"HWM ${old_hwm:.2f} -> ${new_hwm:.2f}, "
                    f"stop ${old_stop:.2f} -> ${new_stop:.2f}"
                )
                # Only alert on significant moves (>$0.25 stop change)
                if new_stop - old_stop >= 0.25:
                    alert_key = f"trail_ratchet_{pid}"
                    if should_alert(state, alert_key):
                        alerts.append(
                            f"**TRAILING STOP RATCHETED** -- {symbol}: "
                            f"new high ${new_hwm:.2f}, "
                            f"stop moved up ${old_stop:.2f} -> ${new_stop:.2f}"
                        )
                        mark_alerted(state, alert_key)
    return alerts


# ═══════════════════════════════════════════════════════════════════════════
# POSITION SYNC FROM EXTERNAL FILES
# ═══════════════════════════════════════════════════════════════════════════

def _convert_rh_position(pid: str, rh_pos: dict) -> dict | None:
    """Convert a position from rh_position_state.json schema to agentic schema."""
    if rh_pos.get("status") != "OPEN":
        return None

    opt_type = rh_pos.get("option_type", "")
    strike = rh_pos.get("strike", "")
    exp = rh_pos.get("expiration", "")
    symbol = rh_pos.get("symbol", "???")

    option_details = None
    pos_type = "equity"
    if opt_type and strike and exp:
        pos_type = "option"
        type_char = "C" if opt_type.lower() == "call" else "P"
        option_details = f"{symbol} {exp} ${strike}{type_char}"

    entry_price = rh_pos.get("entry_price_per_contract", rh_pos.get("entry_price"))
    quantity = rh_pos.get("contracts", rh_pos.get("quantity", 1))

    # Derive target/stop from percentage-based rules if available
    target_price = None
    stop_price = None
    if entry_price:
        if rh_pos.get("profit_target_pct"):
            target_price = round(entry_price * (1 + rh_pos["profit_target_pct"] / 100), 2)
        if rh_pos.get("stop_loss_pct"):
            stop_price = round(entry_price * (1 - rh_pos["stop_loss_pct"] / 100), 2)

    return {
        "symbol": symbol,
        "type": pos_type,
        "option_details": option_details,
        "quantity": quantity,
        "entry_price": entry_price,
        "current_price": rh_pos.get("last_check_pnl_pct"),  # will be None usually
        "target_price": target_price,
        "stop_price": stop_price,
        "entry_date": rh_pos.get("entry_date", "unknown"),
        "thesis": rh_pos.get("thesis", ""),
        "key_signals": [],
        "signal_status": {},
        "limit_sell_placed": False,
        "limit_sell_order_id": None,
        "status": "OPEN",
        "alerts_sent": {},
        "notes": f"Synced from rh_position_state.json ({pid})",
        "_synced_from": "rh_position_state",
        "_source_id": pid,
    }


def _convert_active_option(ao_pos: dict) -> dict | None:
    """Convert a position from active_options.json schema to agentic schema."""
    status = ao_pos.get("status", "OPEN")
    if status != "OPEN":
        return None

    symbol = ao_pos.get("symbol", ao_pos.get("ticker", "???"))
    opt_type = ao_pos.get("option_type", ao_pos.get("type", ""))
    strike = ao_pos.get("strike", "")
    exp = ao_pos.get("expiration", ao_pos.get("exp", ""))

    option_details = None
    if opt_type and strike and exp:
        type_char = "C" if str(opt_type).lower() == "call" else "P"
        option_details = f"{symbol} {exp} ${strike}{type_char}"

    entry_price = ao_pos.get("entry_price", ao_pos.get("entry_price_per_contract"))
    quantity = ao_pos.get("contracts", ao_pos.get("quantity", 1))

    return {
        "symbol": symbol,
        "type": "option",
        "option_details": option_details,
        "quantity": quantity,
        "entry_price": entry_price,
        "current_price": None,
        "target_price": None,
        "stop_price": None,
        "entry_date": ao_pos.get("entry_date", "unknown"),
        "thesis": ao_pos.get("thesis", ""),
        "key_signals": [],
        "signal_status": {},
        "limit_sell_placed": False,
        "limit_sell_order_id": None,
        "status": "OPEN",
        "alerts_sent": {},
        "notes": f"Synced from active_options.json",
        "_synced_from": "active_options",
    }


def sync_external_positions(state: dict) -> int:
    """
    Merge OPEN positions from rh_position_state.json and active_options.json
    into the agentic state. Returns count of newly added positions.

    Uses symbol+type+strike+expiration as dedup key to avoid duplicates.
    Only adds positions that don't already exist in the agentic state.
    """
    added = 0

    # Build a set of existing position signatures for dedup
    existing_sigs = set()
    for pid, pos in state["positions"].items():
        sig = (
            pos.get("symbol", ""),
            pos.get("type", ""),
            pos.get("option_details", ""),
        )
        existing_sigs.add(sig)

    # ── Sync from rh_position_state.json ──
    try:
        if RH_POSITION_STATE_FILE.exists():
            rh_data = json.loads(RH_POSITION_STATE_FILE.read_text())
            rh_positions = rh_data.get("positions", {})
            for pid, rh_pos in rh_positions.items():
                converted = _convert_rh_position(pid, rh_pos)
                if converted is None:
                    continue
                sig = (
                    converted["symbol"],
                    converted["type"],
                    converted.get("option_details", ""),
                )
                if sig in existing_sigs:
                    continue
                # Use the source pid as agentic pid
                agentic_pid = f"rh_{pid}"
                if agentic_pid not in state["positions"]:
                    state["positions"][agentic_pid] = converted
                    existing_sigs.add(sig)
                    added += 1
                    log.info(
                        f"Synced position from rh_position_state: "
                        f"{converted['symbol']} {converted.get('option_details', '')}"
                    )
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to read {RH_POSITION_STATE_FILE}: {e}")

    # ── Sync from active_options.json ──
    try:
        if ACTIVE_OPTIONS_FILE.exists():
            ao_data = json.loads(ACTIVE_OPTIONS_FILE.read_text())
            ao_positions = ao_data.get("positions", [])
            # Handle both list and dict formats
            if isinstance(ao_positions, dict):
                ao_positions = list(ao_positions.values())
            for i, ao_pos in enumerate(ao_positions):
                converted = _convert_active_option(ao_pos)
                if converted is None:
                    continue
                sig = (
                    converted["symbol"],
                    converted["type"],
                    converted.get("option_details", ""),
                )
                if sig in existing_sigs:
                    continue
                agentic_pid = f"ao_{converted['symbol']}_{i}"
                if agentic_pid not in state["positions"]:
                    state["positions"][agentic_pid] = converted
                    existing_sigs.add(sig)
                    added += 1
                    log.info(
                        f"Synced position from active_options: "
                        f"{converted['symbol']} {converted.get('option_details', '')}"
                    )
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to read {ACTIVE_OPTIONS_FILE}: {e}")

    if added:
        log.info(f"Position sync: {added} new position(s) merged from external files")

    return added


# ═══════════════════════════════════════════════════════════════════════════
# MAIN MONITOR LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_check(state: dict) -> dict:
    """Run all checks, send alerts, update state."""
    now = datetime.now(ET)
    state["last_check"] = now.isoformat()

    # Sync positions from external tracking files before counting
    sync_external_positions(state)

    open_count = sum(
        1 for p in state["positions"].values() if p.get("status") == "OPEN"
    )

    if open_count == 0:
        log.info("No open positions — nothing to monitor")
        save_state(state)
        return state

    log.info(f"Checking {open_count} open position(s)...")

    # Ratchet trailing stops before checking levels
    ratchet_trailing_stops(state)

    # HC #742: Auto-ratchet targets BEFORE other checks
    # (especially important pre-market to catch overnight gaps)
    all_alerts = []
    all_alerts.extend(check_target_ratchet(state))

    # Run all checks
    all_alerts.extend(check_limit_sell_gaps(state))
    all_alerts.extend(check_profit_targets(state))
    all_alerts.extend(check_stop_levels(state))
    all_alerts.extend(check_thesis_signals(state))
    all_alerts.extend(check_earnings_proximity(state))

    # Send alerts
    if all_alerts:
        header = f"**Position Monitor** ({now.strftime('%I:%M %p ET')})\n"
        message = header + "\n".join(all_alerts)
        send_discord(message)
        log.info(f"Sent {len(all_alerts)} alert(s)")
    else:
        log.info("All positions OK — no alerts")

    save_state(state)
    return state


def main():
    log.info("=" * 60)
    log.info("Agentic Position Monitor starting")
    log.info(f"State file: {STATE_FILE}")
    log.info(f"Check interval: {CHECK_INTERVAL_MARKET}s (market) / {CHECK_INTERVAL_OFF_HOURS}s (off-hours)")
    log.info("=" * 60)

    state = load_state()
    save_state(state)  # Ensure state file exists

    while True:
        try:
            state = load_state()  # Re-read each cycle (external edits possible)
            state = run_check(state)
        except Exception as e:
            log.error(f"Check cycle error: {e}")
            log.error(traceback.format_exc())
            # Never crash the daemon
            try:
                send_discord(f"**Position Monitor Error**: {str(e)[:200]}")
            except Exception:
                pass

        # Sleep based on market hours
        # HC #742 R3: Pre-market (8:00-9:25 ET) uses market-hours interval
        # to catch overnight price gaps before open
        if is_market_hours() or is_pre_market():
            interval = CHECK_INTERVAL_MARKET
        else:
            interval = CHECK_INTERVAL_OFF_HOURS

        log.info(f"Sleeping {interval // 60}m (market_hours={is_market_hours()})")
        time.sleep(interval)


if __name__ == "__main__":
    main()
