#!/usr/bin/env python3
"""
Options Exit Monitor — checks open positions against exit rules every 30 min.
Alerts on take-profit, stop-loss, trailing stop, and max-hold triggers.
Does NOT auto-execute sells (alert-only for now).

Cron: */30 9-15 * * 1-5
"""

import json, logging, sys, os
from datetime import datetime, date
from pathlib import Path
import numpy as np
from scipy.stats import norm

# --- paths ---
BASE = Path("/home/jupiter/Lvl3Quant")
CONFIG = BASE / "data" / "active_options.json"
LOG_FILE = BASE / "logs" / "options_monitor.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("options_monitor")


# ── Black-Scholes for fallback pricing ──────────────────────────────────
def bs_price(S, K, T, r, sigma, opt_type="call"):
    """Simple Black-Scholes European option price."""
    if T <= 0:
        return max(0, (S - K) if opt_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def trading_days_between(d1: date, d2: date) -> int:
    """Rough count of weekdays between two dates."""
    count = 0
    current = d1
    from datetime import timedelta
    while current < d2:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return count


def get_current_value(pos: dict) -> tuple:
    """Get current option value. Returns (price, method, underlying_price)."""
    import yfinance as yf

    ticker = pos["ticker"]
    strike = pos["strike"]
    expiry_str = pos["expiry"]
    opt_type = pos["option_type"]

    # Get underlying price
    stock = yf.Ticker(ticker)
    try:
        info = stock.fast_info
        underlying = info.last_price
    except Exception:
        underlying = stock.history(period="1d")["Close"].iloc[-1]

    if underlying is None or np.isnan(underlying):
        raise ValueError(f"Could not get price for {ticker}")

    # Try to get live option quote from yfinance
    expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    try:
        chain = stock.option_chain(expiry_str)
        opts = chain.puts if opt_type == "put" else chain.calls
        row = opts.loc[(opts["strike"] - strike).abs().idxmin()]
        mid = (row["bid"] + row["ask"]) / 2
        if mid > 0.01:
            return round(mid, 2), "market_mid", round(underlying, 2)
    except Exception as e:
        log.debug(f"Chain lookup failed for {ticker}: {e}")

    # Fallback: Black-Scholes with estimated IV
    T = (expiry_dt - date.today()).days / 365.0
    sigma = 0.30  # default IV estimate
    price = bs_price(underlying, strike, T, 0.05, sigma, opt_type)
    return round(price, 2), "bs_estimate", round(underlying, 2)


def check_exit_rules(pos: dict, current_price: float) -> list:
    """Check all exit rules for a position. Returns list of triggered rules."""
    triggers = []
    entry = pos["entry_price"]
    rules = pos["rules"]
    pnl_pct = ((current_price - entry) / entry) * 100

    # 1. Take profit
    tp = rules.get("tp_pct", 30)
    if pnl_pct >= tp:
        triggers.append(("TAKE_PROFIT", f"+{pnl_pct:.1f}% (threshold: +{tp}%)"))

    # 2. Stop loss
    sl = rules.get("sl_pct", -25)
    if pnl_pct <= sl:
        triggers.append(("STOP_LOSS", f"{pnl_pct:.1f}% (threshold: {sl}%)"))

    # 3. Trailing stop
    trail_activate = rules.get("trail_activate_pct", 15)
    trail_pct = rules.get("trail_pct", 50)
    max_gain = pos.get("max_gain_pct", 0)

    if pnl_pct > max_gain:
        max_gain = pnl_pct  # will be persisted below

    if max_gain >= trail_activate:
        trail_floor = max_gain * (1 - trail_pct / 100)
        if pnl_pct <= trail_floor:
            triggers.append((
                "TRAILING_STOP",
                f"Gain dropped to +{pnl_pct:.1f}% from peak +{max_gain:.1f}% "
                f"(floor: +{trail_floor:.1f}%)",
            ))

    # Update max gain for persistence
    pos["max_gain_pct"] = max_gain

    # 4. Max hold days
    max_hold = rules.get("max_hold_days", 5)
    entry_date = datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()
    days_held = trading_days_between(entry_date, date.today())
    if days_held >= max_hold:
        triggers.append(("MAX_HOLD", f"{days_held} trading days held (max: {max_hold})"))

    return triggers, pnl_pct


def format_alert(pos, triggers, pnl_pct, current_price, underlying, method):
    """Format a human-readable alert string."""
    lines = [
        f"{'='*60}",
        f"EXIT SIGNAL: {pos['ticker']} {pos['strike']}{pos['option_type'][0].upper()} "
        f"exp {pos['expiry']}",
        f"  Entry: ${pos['entry_price']:.2f} | Current: ${current_price:.2f} ({method})",
        f"  Underlying: ${underlying} | P&L: {pnl_pct:+.1f}%",
        f"  Qty: {pos['quantity']}",
    ]
    for trigger_type, detail in triggers:
        lines.append(f"  >> {trigger_type}: {detail}")
    lines.append(f"{'='*60}")
    return "\n".join(lines)


def run():
    log.info("Options exit monitor starting")

    # Market guard — skip on weekends/holidays
    try:
        sys.path.insert(0, str(BASE / "scripts"))
        from market_status import get_market_status
        ms = get_market_status()
        if not ms.get("is_trading_day", True):
            log.info(f"Market closed: {ms.get('reason')}. Skipping.")
            return
    except Exception:
        pass  # Fail-open

    if not CONFIG.exists():
        log.warning(f"No config at {CONFIG} — nothing to monitor")
        return

    positions = json.loads(CONFIG.read_text())
    if not positions:
        log.info("No active positions")
        return

    log.info(f"Checking {len(positions)} position(s)")
    alerts = []

    for pos in positions:
        label = f"{pos['ticker']} {pos['strike']}{pos['option_type'][0].upper()}"
        try:
            current_price, method, underlying = get_current_value(pos)
            triggers, pnl_pct = check_exit_rules(pos, current_price)

            log.info(
                f"{label}: ${current_price:.2f} ({method}) | "
                f"underlying ${underlying} | P&L {pnl_pct:+.1f}%"
            )

            if triggers:
                alert = format_alert(pos, triggers, pnl_pct, current_price, underlying, method)
                alerts.append(alert)
                log.warning(f"EXIT TRIGGERED for {label}")
                print(alert)
            else:
                log.info(f"{label}: No exit signals. Holding.")

        except Exception as e:
            log.error(f"Error checking {label}: {e}")

    # Persist updated max_gain values
    CONFIG.write_text(json.dumps(positions, indent=2))

    if alerts:
        # Write alerts to a pickup file for the Discord bridge
        alert_file = BASE / "logs" / "options_alerts.txt"
        with open(alert_file, "a") as f:
            f.write(f"\n--- {datetime.now().isoformat()} ---\n")
            for a in alerts:
                f.write(a + "\n")
        log.info(f"Wrote {len(alerts)} alert(s) to {alert_file}")

        # HC #784: Inject urgent sell prompt for stop-loss / max-hold triggers
        try:
            import subprocess
            urgent_triggers = [a for a in alerts if "STOP_LOSS" in a or "MAX_HOLD" in a or "TRAILING_STOP" in a]
            if urgent_triggers:
                inject_msg = (
                    f"URGENT_EXIT_TRIGGER: {len(urgent_triggers)} position(s) hit exit rules. "
                    f"Check logs/options_alerts.txt and execute sell orders IMMEDIATELY. "
                    f"Use limit orders per HC #785 (limit slightly below mid for urgency)."
                )
                subprocess.run(
                    ["/bin/bash", str(BASE / "scripts" / "autonomy_inject.sh"), inject_msg],
                    timeout=10
                )
                log.info("Injected urgent exit prompt into autonomy pipeline.")
        except Exception as e:
            log.error(f"Failed to inject exit prompt: {e}")
    else:
        log.info("All positions within bounds. No exits triggered.")

    log.info("Monitor complete")


if __name__ == "__main__":
    run()
