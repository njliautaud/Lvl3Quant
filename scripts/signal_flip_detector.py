#!/usr/bin/env python3
"""
Signal Flip Detector — detects sudden burst of new confirming sources on a ticker.

Runs every 30 minutes during market hours via cron. Compares current
agentic_signals.json against a prior snapshot to find tickers that suddenly
gained 3+ confirming sources with high confidence and cheap IV.

A "signal flip" = a ticker that wasn't previously a strong signal but now is.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

BASE = Path("/home/jupiter/Lvl3Quant")
SIGNALS_FILE = BASE / "state" / "agentic_signals.json"
PRIOR_FILE = BASE / "state" / "signal_flip_prior.json"
ALERT_FILE = BASE / "state" / "signal_flip_alert.txt"
MORNING_QUEUE_FILE = BASE / "state" / "morning_execution_queue.json"

# Thresholds for a "signal flip"
MIN_CONFIRMING = 3
MIN_CONFIDENCE = 0.75
CHEAP_IV_CLASSES = {"CHEAP", "VERY_CHEAP"}

# A signal is "significantly fewer sources" if prior had fewer than this fraction
SOURCE_JUMP_THRESHOLD = 2  # Must have gained at least 2 new sources vs prior


def load_json(path):
    """Load a JSON file, returning empty dict on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def extract_signal_map(signals_data):
    """Build a dict of ticker -> signal info from agentic_signals.json.

    Looks at both 'signals' (top-tier) and 'below_threshold' lists to catch
    tickers that jump from below-threshold to strong.
    """
    result = {}
    all_signals = []
    all_signals.extend(signals_data.get("signals", []))
    all_signals.extend(signals_data.get("actionable_signals", []))
    all_signals.extend(signals_data.get("below_threshold", []))

    for sig in all_signals:
        ticker = sig.get("ticker", "")
        if not ticker:
            continue
        # Keep the entry with the highest n_confirming if duplicated
        existing = result.get(ticker)
        if existing and existing.get("n_confirming", 0) >= sig.get("n_confirming", 0):
            continue
        result[ticker] = {
            "ticker": ticker,
            "direction": sig.get("direction", "unknown"),
            "confidence_score": sig.get("confidence_score", 0),
            "n_confirming": sig.get("n_confirming", 0),
            "n_conflicting": sig.get("n_conflicting", 0),
            "confirming_sources": sig.get("confirming_sources", []),
            "iv_rank": sig.get("iv_rank"),
            "iv_classification": sig.get("iv_classification", "UNKNOWN"),
            "recommended_option": sig.get("recommended_option", ""),
            "recommended_strike": sig.get("recommended_strike", ""),
            "recommended_expiry": sig.get("recommended_expiry", ""),
            "reason": sig.get("reason", ""),
        }
    return result


def detect_flips(current_map, prior_map):
    """Find tickers that qualify as signal flips.

    A flip occurs when:
    1. n_confirming >= MIN_CONFIRMING
    2. The ticker was NOT in the prior snapshot with 3+ sources
       OR it gained at least SOURCE_JUMP_THRESHOLD new sources
    3. iv_classification is CHEAP or VERY_CHEAP
    4. confidence_score >= MIN_CONFIDENCE
    """
    flips = []

    for ticker, current in current_map.items():
        n_curr = current.get("n_confirming", 0)
        confidence = current.get("confidence_score", 0)
        iv_class = current.get("iv_classification", "UNKNOWN")

        # Gate 1: minimum confirming sources
        if n_curr < MIN_CONFIRMING:
            continue

        # Gate 2: check if this is new or a significant jump
        prior = prior_map.get(ticker, {})
        n_prior = prior.get("n_confirming", 0)

        is_new_strong = n_prior < MIN_CONFIRMING  # wasn't strong before
        is_big_jump = (n_curr - n_prior) >= SOURCE_JUMP_THRESHOLD

        if not (is_new_strong or is_big_jump):
            continue

        # Gate 3: IV should be cheap — but don't block on UNKNOWN/NORMAL
        # (pre-validator re-checks IV with fresh data at execution time)
        BLOCK_IV_CLASSES = {"EXPENSIVE", "VERY_EXPENSIVE"}
        if iv_class in BLOCK_IV_CLASSES:
            continue

        # Gate 4: confidence threshold
        if confidence < MIN_CONFIDENCE:
            continue

        # This is a flip
        new_sources = []
        prior_sources = set(prior.get("confirming_sources", []))
        for src in current.get("confirming_sources", []):
            if src not in prior_sources:
                new_sources.append(src)

        flips.append({
            "ticker": ticker,
            "direction": current.get("direction", "unknown"),
            "confidence": confidence,
            "n_confirming": n_curr,
            "n_prior": n_prior,
            "sources_gained": n_curr - n_prior,
            "new_sources": new_sources,
            "iv_rank": current.get("iv_rank"),
            "iv_classification": iv_class,
            "recommended_option": current.get("recommended_option", ""),
            "recommended_strike": current.get("recommended_strike", ""),
            "recommended_expiry": current.get("recommended_expiry", ""),
            "reason": current.get("reason", ""),
        })

    # Sort by confidence descending
    flips.sort(key=lambda f: f["confidence"], reverse=True)
    return flips


def format_alert(flips, timestamp):
    """Format flips into a human-readable alert string."""
    lines = []
    lines.append(f"SIGNAL FLIP DETECTED at {timestamp}")
    lines.append(f"{len(flips)} ticker(s) with sudden source burst:\n")

    for f in flips:
        lines.append(
            f"  {f['ticker']} {f['direction'].upper()} — "
            f"confidence {f['confidence']:.0%}, "
            f"{f['n_confirming']} sources (was {f['n_prior']}), "
            f"IV {f['iv_classification']} (rank {f['iv_rank']}%)"
        )
        if f["new_sources"]:
            lines.append(f"    New sources: {', '.join(f['new_sources'][:5])}")
        if f["recommended_option"]:
            lines.append(
                f"    Suggested: {f['recommended_option']} "
                f"${f['recommended_strike']} exp {f['recommended_expiry']}"
            )
        lines.append("")

    return "\n".join(lines)


def is_market_open():
    """Check if market is currently in regular trading hours."""
    try:
        sys.path.insert(0, str(BASE / "scripts"))
        from market_status import get_market_status
        status = get_market_status()
        return status.get("is_rth", False) or status.get("session", "") == "regular"
    except Exception:
        # Fallback: check time directly
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        hour = now_et.hour
        weekday = now_et.weekday()
        return weekday < 5 and 9 <= hour < 16


def queue_for_morning(flips, timestamp):
    """Write flips to morning execution queue for next-day processing.

    Called when signal flips are detected after market close. The pre-validator
    reads this queue at 9:30 AM to include in morning execution.
    """
    queue = load_json(MORNING_QUEUE_FILE)
    if not isinstance(queue, dict):
        queue = {}

    queue["queued_at"] = timestamp
    queue["flips"] = flips
    queue["status"] = "PENDING"
    queue["note"] = "After-hours signal flip — queue for morning execution"

    try:
        with open(MORNING_QUEUE_FILE, "w") as f:
            json.dump(queue, f, indent=2)
        print(f"[{timestamp}] Queued {len(flips)} flip(s) for morning execution")
    except OSError as e:
        print(f"[{timestamp}] WARNING: Could not write morning queue: {e}")


def main():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Load current signals
    signals_data = load_json(SIGNALS_FILE)
    if not signals_data or "signals" not in signals_data:
        print(f"[{timestamp}] No signals file or empty — skipping")
        return

    # Load prior snapshot
    prior_data = load_json(PRIOR_FILE)
    prior_map = extract_signal_map(prior_data) if prior_data else {}

    # Build current map
    current_map = extract_signal_map(signals_data)

    # Detect flips
    flips = detect_flips(current_map, prior_map)

    # Save current as new prior snapshot (always, even if no flips)
    try:
        with open(PRIOR_FILE, "w") as f:
            json.dump(signals_data, f, indent=2)
    except OSError as e:
        print(f"[{timestamp}] WARNING: Could not save prior snapshot: {e}")

    # Report results
    if flips:
        market_open = is_market_open()
        alert_text = format_alert(flips, timestamp)

        if market_open:
            # During market hours: write alert for cron inject (existing behavior)
            try:
                with open(ALERT_FILE, "w") as f:
                    f.write(alert_text)
            except OSError as e:
                print(f"[{timestamp}] WARNING: Could not write alert file: {e}")
            print(alert_text)
        else:
            # After hours: queue for morning execution
            queue_for_morning(flips, timestamp)
            print(f"[{timestamp}] AFTER-HOURS FLIP — queued for morning:")
            print(alert_text)
            # Also trigger after-hours equity capture check
            # This buys shares in extended hours to capture overnight gap
            try:
                from afterhours_equity_capture import main as ah_capture
                print(f"[{timestamp}] Running after-hours equity capture check...")
                ah_capture()
            except Exception as e:
                print(f"[{timestamp}] AH equity capture check failed: {e}")
    else:
        print(f"[{timestamp}] No signal flips detected")


if __name__ == "__main__":
    main()
