#!/usr/bin/env python3
"""
RH Trigger Watchdog v2 — uses lib/exit_rules.py for ALL exit logic.

No hardcoded exit params in this file. Everything comes from lib/constants.py.
Polls every 3 min during RTH. If an exit trigger fires or is within 5%,
injects a prompt via autonomy_inject.sh so Claude acts immediately.

Cron: */3 9-15 * * 1-5
"""

import json
import subprocess
import sys
import os
import logging
from datetime import datetime, date, timedelta
from pathlib import Path

# Add parent to path for lib imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.constants import (
    ACTIVE_OPTIONS_FILE, POSITION_STATE_FILE, INJECT_SCRIPT, BASE,
)
from lib.exit_rules import evaluate_exits, Position, MarketData
from lib.position_manager import (
    load_active_options, save_active_options,
    update_peak, set_trailing_active, sync_state_from_active,
)

LOG_FILE = BASE / "logs" / "rh_trigger_watchdog.log"
ALERT_STATE_FILE = BASE / "data" / "rh_trigger_alert_state.json"

os.makedirs(LOG_FILE.parent, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rh_trigger_watchdog")


# ── Price fetching (yfinance — no auth needed) ──

def get_price_yf(ticker: str) -> float | None:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        data = t.history(period="1d")
        if len(data) > 0:
            return float(data["Close"].iloc[-1])
    except Exception as e:
        log.warning(f"yfinance failed for {ticker}: {e}")
    return None


def get_option_price_yf(ticker: str, strike: float, expiry: str, opt_type: str) -> float | None:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiry)
        df = chain.puts if opt_type == "put" else chain.calls
        row = df[abs(df["strike"] - strike) < 0.01]
        if len(row) > 0:
            bid = float(row.iloc[0]["bid"])
            ask = float(row.iloc[0]["ask"])
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            return float(row.iloc[0]["lastPrice"])
    except Exception as e:
        log.warning(f"yfinance option chain failed for {ticker} {strike}{opt_type[0].upper()} {expiry}: {e}")
    return None


def get_vix() -> float:
    try:
        import yfinance as yf
        data = yf.download("^VIX", period="2d", progress=False)
        if data is not None and len(data) > 0:
            return float(data["Close"].values.flatten()[-1])
    except Exception:
        pass
    return 16.0  # Default if unavailable


def get_macro_regime() -> str:
    try:
        macro_path = BASE / "data" / "macro" / "macro_summary.json"
        with open(macro_path) as f:
            macro = json.load(f)
        return macro.get("regime", "RISK_ON")
    except Exception:
        return "RISK_ON"


# ── Alert management ──

def load_alert_state() -> dict:
    try:
        with open(ALERT_STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_alert_state(state: dict):
    with open(ALERT_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def should_alert(key: str, alert_state: dict, cooldown_minutes: int = 30) -> bool:
    last = alert_state.get(key)
    if last is None:
        return True
    last_dt = datetime.fromisoformat(last)
    return (datetime.now() - last_dt).total_seconds() > cooldown_minutes * 60


def inject_alert(message: str):
    log.info(f"INJECTING ALERT: {message[:120]}...")
    try:
        subprocess.run([INJECT_SCRIPT, message], timeout=10, capture_output=True)
    except Exception as e:
        log.error(f"Inject failed: {e}")


# ── Main check loop ──

def check_positions():
    """Check all open positions against exit rules."""
    data = load_active_options()
    # HC #810 fix: positions are saved with status="filled" (order filled),
    # not "open". Monitor everything that isn't explicitly closed/expired.
    INACTIVE = {"closed", "expired", "cancelled"}
    positions = [p for p in data.get("positions", [])
                 if p.get("status", "").lower() not in INACTIVE]

    if not positions:
        log.info("No open positions to monitor.")
        return

    # Get market context once
    vix = get_vix()
    macro_regime = get_macro_regime()
    market_open = datetime.now().replace(hour=9, minute=30, second=0)

    alert_state = load_alert_state()

    for pos in positions:
        ticker = pos.get("ticker", "???")
        strike = pos.get("strike", 0)
        opt_type = (pos.get("option_type") or pos.get("type") or "call").lower()  # records store put/call under "type"
        expiry = pos.get("expiration", pos.get("expiry", ""))
        entry_price = pos.get("entry_price", pos.get("entry_debit", 0))

        # Get current prices
        mark = get_option_price_yf(ticker, strike, expiry, opt_type)
        underlying = get_price_yf(ticker)

        if mark is None:
            log.warning(f"Could not get price for {ticker} {strike}{opt_type[0].upper()} {expiry}. Skipping.")
            continue

        und_str = f"${underlying:.2f}" if underlying else "$0.00"
        log.info(f"{ticker} {strike}{opt_type[0].upper()}: mark=${mark:.2f}, entry=${entry_price:.2f}, "
                 f"underlying={und_str}")

        # Update peak value
        update_peak(ticker, strike, mark)

        # Build position and market data objects for exit engine
        position_obj = Position(
            ticker=ticker,
            option_type=opt_type,
            strike=strike,
            expiration=expiry,
            entry_price=entry_price,
            entry_date=pos.get("entry_date", date.today().isoformat()),
            underlying_entry_price=pos.get("underlying_entry_price", 0),
            quantity=pos.get("quantity", 1),
            peak_value=pos.get("peak_value", entry_price),
            n_sources=pos.get("n_sources", 0),
            tp_order_id=pos.get("tp_order_id"),
            sl_order_id=pos.get("sl_order_id"),
            trailing_active=pos.get("trailing_active", False),
        )

        market_obj = MarketData(
            current_mark=mark,
            underlying_price=underlying or 0,
            vix=vix,
            macro_regime=macro_regime,
            market_open_time=market_open,
        )

        # Evaluate ALL exit rules
        result = evaluate_exits(position_obj, market_obj)

        key = f"{ticker}_{strike}_{opt_type}"

        if result.triggered:
            # EXIT TRIGGERED — always alert (no cooldown)
            msg = (
                f"🚨 EXIT TRIGGERED ({result.rule_name}): {result.reason}\n"
                f"Action: {result.action}. P&L: {result.pnl_pct:+.1f}%.\n"
                f"Execute via review_option_order then place_option_order (sell-to-close at bid)."
            )
            if result.action == "close_half":
                msg += "\nGRADUATED EXIT: Sell HALF only. Hold remainder with -30% hard floor."

            inject_alert(msg)
            alert_state[f"trigger_{key}"] = datetime.now().isoformat()
            log.info(f"  🚨 TRIGGER: {result.rule_name} — {result.reason}")

        elif result.details.get("proximity_warnings"):
            # Near a trigger — alert with cooldown
            alert_key = f"proximity_{key}"
            if should_alert(alert_key, alert_state, cooldown_minutes=30):
                warnings = ", ".join(result.details["proximity_warnings"])
                msg = (
                    f"⚠️ PROXIMITY ALERT: {ticker} {strike}{opt_type[0].upper()} — "
                    f"P&L {result.pnl_pct:+.1f}%. {warnings}. Monitoring."
                )
                inject_alert(msg)
                alert_state[alert_key] = datetime.now().isoformat()
                log.info(f"  ⚠️ PROXIMITY: {warnings}")
        else:
            log.info(f"  ✓ Holding. {result.reason}")

        # Check if trailing stop should activate
        if not position_obj.trailing_active and result.details.get("trailing_active"):
            set_trailing_active(ticker, strike)
            log.info(f"  📈 Trailing stop ACTIVATED")

        # Update last check data
        pos["last_watchdog_check"] = datetime.now().isoformat()
        pos["last_watchdog_pnl_pct"] = round(result.pnl_pct, 1)
        pos["last_watchdog_price"] = round(mark, 2)

    # Save updated data
    save_active_options(data)
    save_alert_state(alert_state)

    # Keep state files in sync
    sync_state_from_active()


def main():
    now = datetime.now()
    log.info(f"=== RH Trigger Watchdog v2 at {now.strftime('%H:%M:%S')} ===")

    # Only run during market hours
    if now.hour < 9 or now.hour >= 16:
        log.info("Outside market hours. Skipping.")
        return

    check_positions()
    log.info("=== Watchdog complete ===\n")


if __name__ == "__main__":
    main()
