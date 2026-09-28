#!/usr/bin/env python3
"""
After-Hours Equity Capture — buys shares in extended hours when high-conviction
signals fire after market close.

Problem solved: Options don't trade after 4 PM. When a 90%+ signal fires at 4:25 PM,
we can't get exposure until 9:30 AM by which time the stock may have gapped 3-5%.
Solution: Buy a small equity position in RH extended hours (until 8 PM ET) to capture
the overnight gap. At market open, either hold shares or convert to options.

Trigger: Called by signal_flip_detector.py when an after-hours burst is detected.
Also runs via cron at 4:45 PM and 5:15 PM to catch late-firing signals.

Risk management:
- Only triggers on confidence >= 85% (higher bar since shares have no leverage)
- Max allocation: 15% of equity per trade (~$112 at $751 balance)
- Extended hours limit order at ask + small buffer for fill
- Writes to state/afterhours_equity_position.json for morning decision logic
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path("/home/jupiter/Lvl3Quant")
SIGNALS_FILE = BASE / "state" / "agentic_signals.json"
MORNING_QUEUE_FILE = BASE / "state" / "morning_execution_queue.json"
AH_POSITION_FILE = BASE / "state" / "afterhours_equity_position.json"
AH_LOG_FILE = BASE / "logs" / "afterhours_equity.log"
ALERT_FILE = BASE / "state" / "signal_flip_alert.txt"

# Thresholds
MIN_CONFIDENCE_AH = 0.85  # Higher bar for equity (no leverage benefit)
MIN_SOURCES_AH = 4        # Need strong multi-source confirmation
# Tiered allocation: higher confidence = more capital committed
# 85-89% confidence: 15% of equity (~$112 at $751)
# 90%+ confidence: 25% of equity (~$188 at $751) — need this for $170+ ETFs like XLV
ALLOC_NORMAL = 0.15        # 85-89% confidence
ALLOC_HIGH_CONVICTION = 0.25  # 90%+ confidence
ACCOUNT_EQUITY = 751       # Updated dynamically from signals file

# Tickers we've validated as having good burst performance
BURST_POSITIVE_TICKERS = {"XLE", "XLU", "XLK", "XLI", "SMH"}
# Tickers where burst pattern historically loses money
BURST_NEGATIVE_TICKERS = {"XLV", "XLP", "XLY", "XLC", "XLB"}


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(AH_LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def is_extended_hours():
    """Check if we're in RH extended hours window (4:00-8:00 PM ET or 7:00-9:30 AM ET)."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    hour = now_et.hour
    minute = now_et.minute
    weekday = now_et.weekday()

    if weekday >= 5:  # Weekend
        return False

    # After-market: 4:00 PM - 8:00 PM ET
    if 16 <= hour < 20:
        return True
    # Pre-market: 7:00 AM - 9:30 AM ET
    if 7 <= hour < 9 or (hour == 9 and minute < 30):
        return True

    return False


def get_account_equity():
    """Get current account equity from signals file or default."""
    signals = load_json(SIGNALS_FILE)
    equity = signals.get("account_equity", ACCOUNT_EQUITY)
    return max(equity, 100)  # Floor at $100


def find_ah_candidates():
    """Find signals that qualify for after-hours equity capture.

    Checks both the morning queue (overnight flips) and current signals.
    Returns list of candidate dicts with ticker, direction, confidence, max_shares, etc.
    """
    candidates = []
    equity = get_account_equity()
    max_spend_normal = equity * ALLOC_NORMAL
    max_spend_high = equity * ALLOC_HIGH_CONVICTION

    # Check morning queue first (these are pre-screened burst signals)
    morning_queue = load_json(MORNING_QUEUE_FILE)
    if morning_queue and morning_queue.get("status") == "PENDING":
        for flip in morning_queue.get("flips", []):
            ticker = flip.get("ticker", "")
            confidence = flip.get("confidence", 0)
            direction = flip.get("direction", "unknown")
            n_confirming = flip.get("n_confirming", 0)

            if not ticker:
                continue

            # Apply burst ticker boost
            if ticker in BURST_POSITIVE_TICKERS:
                confidence = min(0.95, confidence * 1.10)

            if confidence < MIN_CONFIDENCE_AH:
                log(f"  {ticker}: confidence {confidence:.0%} < {MIN_CONFIDENCE_AH:.0%} — skip")
                continue

            if n_confirming < MIN_SOURCES_AH:
                log(f"  {ticker}: only {n_confirming} sources < {MIN_SOURCES_AH} — skip")
                continue

            # Negative burst tickers blocked UNLESS confidence is extreme (>= 90%)
            # At 90%+ confidence, the signal is strong enough to override historical
            # burst pattern weakness — this is not a moderate-confidence gamble
            if ticker in BURST_NEGATIVE_TICKERS and confidence < 0.90:
                log(f"  {ticker}: in negative burst list at {confidence:.0%} — skip (need 90%+)")
                continue

            if direction != "bull":
                log(f"  {ticker}: direction '{direction}' — can only buy shares long, skip")
                continue

            # Tiered allocation: 90%+ gets more capital
            spend = max_spend_high if confidence >= 0.90 else max_spend_normal

            candidates.append({
                "ticker": ticker,
                "direction": direction,
                "confidence": confidence,
                "n_confirming": n_confirming,
                "max_spend": spend,
                "source": "morning_queue_burst",
            })

    # Also check current signals directly
    signals_data = load_json(SIGNALS_FILE)
    for signal in signals_data.get("signals", []):
        ticker = signal.get("ticker", "")
        confidence = signal.get("confidence_score", 0)
        direction = signal.get("direction", "unknown")
        n_confirming = signal.get("n_confirming", 0)

        if not ticker:
            continue
        # Don't duplicate morning queue candidates
        if any(c["ticker"] == ticker for c in candidates):
            continue

        if confidence < MIN_CONFIDENCE_AH:
            continue
        if n_confirming < MIN_SOURCES_AH:
            continue
        # Same override: negative tickers ok at 90%+ confidence
        if ticker in BURST_NEGATIVE_TICKERS and confidence < 0.90:
            continue
        if direction != "bull":
            continue

        spend = max_spend_high if confidence >= 0.90 else max_spend_normal

        candidates.append({
            "ticker": ticker,
            "direction": direction,
            "confidence": confidence,
            "n_confirming": n_confirming,
            "max_spend": spend,
            "source": "current_signal",
        })

    # Sort by confidence
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates


def get_ticker_price(ticker):
    """Get current price for a ticker. Tries multiple sources.

    NOTE: recommended_strike is the OPTIONS strike (e.g. $60), NOT the stock price
    (e.g. $63.50). Never use it as a price estimate — it caused $60.18 limit orders
    on a $63.50 stock (HC bug fix 2026-08-19).
    """
    # Source 1: signals file last_price field (actual stock price)
    signals = load_json(SIGNALS_FILE)
    for sig in signals.get("signals", []):
        if sig.get("ticker") == ticker:
            last_price = sig.get("last_price", 0)
            if last_price and float(last_price) > 0:
                return float(last_price)
            # Also check close_price as fallback
            close_price = sig.get("close_price", 0)
            if close_price and float(close_price) > 0:
                return float(close_price)

    # Source 2: try yfinance
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass

    # Source 3: try a cached price file
    try:
        price_file = BASE / "state" / "price_cache.json"
        prices = load_json(price_file)
        if ticker in prices:
            return float(prices[ticker])
    except Exception:
        pass

    return 0


def compute_shares_and_limit(ticker, max_spend):
    """Compute number of shares and limit price for extended hours order.

    Returns (shares, limit_price) or (0, 0) if can't compute.
    """
    price_est = get_ticker_price(ticker)
    if price_est <= 0:
        log(f"  Cannot get price for {ticker}")
        return 0, 0

    shares = int(max_spend / price_est)
    if shares < 1:
        log(f"  {ticker} at ${price_est:.2f} too expensive for ${max_spend:.2f} budget")
        return 0, 0

    # Limit price: 0.3% above for fill certainty in extended hours
    limit_price = round(price_est * 1.003, 2)
    return shares, limit_price


def write_ah_instruction(candidate, shares, limit_price):
    """Write the after-hours equity capture instruction for Claude to execute.

    This writes to signal_flip_alert.txt which the cron pipes into autonomy_inject.
    Also writes to afterhours_equity_position.json for morning tracking.
    """
    ticker = candidate["ticker"]
    confidence = candidate["confidence"]
    n_confirming = candidate["n_confirming"]

    instruction = {
        "type": "AFTERHOURS_EQUITY_CAPTURE",
        "ticker": ticker,
        "direction": "buy",
        "shares": shares,
        "limit_price": limit_price,
        "max_spend": round(shares * limit_price, 2),
        "confidence": confidence,
        "n_confirming": n_confirming,
        "source": candidate["source"],
        "timestamp": datetime.now().isoformat(),
        "market_hours": "extended_hours",
        "time_in_force": "gtc",
        "exit_plan": {
            "morning_review": "At 9:30 AM, evaluate: hold shares, convert to options, or sell for gap profit",
            "stop_loss_pct": 3.0,  # Sell if drops 3% from entry
            "take_profit_pct": 5.0,  # Consider selling if up 5%+ at open
        },
        "rationale": (
            f"After-hours equity capture: {ticker} has {confidence:.0%} confidence from "
            f"{n_confirming} sources. Buying {shares} shares at ~${limit_price} to capture "
            f"overnight gap. Options don't trade after hours."
        ),
    }

    # Write instruction file for tracking
    save_json(AH_POSITION_FILE, instruction)

    # Write alert for autonomy_inject to pick up
    alert_text = (
        f"AFTERHOURS_EQUITY_CAPTURE: {ticker} {candidate['direction'].upper()} — "
        f"{confidence:.0%} confidence, {n_confirming} sources. "
        f"BUY {shares} shares at limit ${limit_price} in extended hours. "
        f"Rationale: capture overnight gap that options can't reach. "
        f"Max spend: ${round(shares * limit_price, 2)}. "
        f"Morning plan: review at 9:30 AM for hold/convert/exit."
    )

    try:
        with open(ALERT_FILE, "w") as f:
            f.write(alert_text)
    except OSError as e:
        log(f"WARNING: Could not write alert file: {e}")

    return instruction


def main():
    log("=== After-Hours Equity Capture Check ===")

    if not is_extended_hours():
        log("Not in extended hours window — skipping")
        return

    # Check if we already have an AH position pending (from today or still open)
    existing = load_json(AH_POSITION_FILE)
    if existing and existing.get("type") == "AFTERHOURS_EQUITY_CAPTURE":
        # Check if it was from today
        ts = existing.get("timestamp", "")
        today = datetime.now().strftime("%Y-%m-%d")
        if today in ts:
            log(f"Already have AH capture for {existing.get('ticker')} today — skipping")
            return
        # Also check if the position is still open (not yet reviewed/sold)
        if existing.get("status") not in ("sold", "converted", "closed", None):
            ticker_held = existing.get("ticker", "unknown")
            log(f"Previous AH position {ticker_held} still open — skipping new capture")
            return

    # Find candidates
    candidates = find_ah_candidates()

    if not candidates:
        log("No candidates qualify for after-hours equity capture")
        return

    # Take the best candidate (highest confidence)
    best = candidates[0]
    log(f"Best candidate: {best['ticker']} — {best['confidence']:.0%} conf, "
        f"{best['n_confirming']} sources, source={best['source']}")

    # Compute shares and limit price
    shares, limit_price = compute_shares_and_limit(best["ticker"], best["max_spend"])

    if shares < 1:
        log(f"Cannot compute valid share count for {best['ticker']} "
            f"(max_spend=${best['max_spend']:.2f}) — skipping")
        return

    # Write the instruction
    instruction = write_ah_instruction(best, shares, limit_price)
    log(f"INSTRUCTION WRITTEN: Buy {shares} shares of {best['ticker']} "
        f"at limit ${limit_price} (extended hours)")
    log(f"Max spend: ${round(shares * limit_price, 2)}")
    log(f"Alert file written for autonomy_inject pickup")

    return instruction


if __name__ == "__main__":
    main()
