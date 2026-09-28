#!/usr/bin/env python3
"""
Execution Follow-Up — runs 10 min AFTER the inject.
Checks: did Claude actually execute the trades from execution_ready.json?
If not, fires a LOUDER re-inject.

This is the accountability layer. No more "signal was ready but nothing happened."
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path("/home/jupiter/Lvl3Quant")
READY_FILE = BASE / "state" / "execution_ready.json"
POSITIONS_FILE = BASE / "state" / "agentic_positions.json"
PENDING_FILE = BASE / "state" / "pending_entries.json"
EXEC_LOG = BASE / "state" / "execution_log.json"
INJECT_SCRIPT = BASE / "scripts" / "autonomy_inject.sh"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get_positioned_tickers(positions_data):
    if not positions_data or "positions" not in positions_data:
        return set()
    return {p.get("symbol") for p in positions_data["positions"].values()
            if p.get("status") == "OPEN"}


def inject_message(msg):
    """Fire autonomy inject with a message."""
    try:
        subprocess.run(
            [str(INJECT_SCRIPT), msg],
            capture_output=True, text=True, timeout=10
        )
    except Exception as e:
        print(f"Inject failed: {e}")


def main():
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    current_hour = now.hour

    # Load execution_ready
    ready = load_json(READY_FILE)

    if not ready or ready.get("status") != "EXECUTE":
        print("No pending executions — nothing to follow up on")
        return

    # Check if it's from today
    if ready.get("date") != today_str:
        print(f"Execution ready from {ready.get('date')}, not today — skip")
        return

    trades_ready = ready.get("trades_ready", [])
    if not trades_ready:
        print("No trades in execution_ready — skip")
        return

    # Check what's actually positioned now
    positions = load_json(POSITIONS_FILE)
    positioned_tickers = get_positioned_tickers(positions)

    # Which trades from execution_ready are still NOT executed?
    unexecuted = []
    for trade in trades_ready:
        ticker = trade.get("ticker")
        if ticker not in positioned_tickers:
            unexecuted.append(trade)

    if not unexecuted:
        print("All trades executed ✅")
        # Update execution log
        log = load_json(EXEC_LOG)
        for entry in reversed(log.get("entries", [])):
            if entry.get("date") == today_str and not entry.get("executed"):
                entry["executed"] = True
                entry["execution_time"] = now.strftime("%H:%M:%S")
                break
        with open(EXEC_LOG, "w") as f:
            json.dump(log, f, indent=2)
        return

    # Trades were NOT executed — fire louder inject
    tickers = ", ".join(t["ticker"] for t in unexecuted)
    confidences = ", ".join(f"{t['confidence']:.0%}" for t in unexecuted)

    # Determine urgency based on time
    if current_hour < 10:
        urgency = "MORNING WINDOW"
    elif current_hour < 14:
        urgency = "MIDDAY WINDOW"
    else:
        urgency = "LAST CHANCE — market closes soon"

    msg = (
        f"🚨 EXECUTION OVERDUE ({urgency}) 🚨\n"
        f"{len(unexecuted)} trade(s) passed ALL gates but were NOT executed: {tickers}\n"
        f"Confidences: {confidences}\n"
        f"These signals were validated {ready.get('timestamp', 'earlier')}.\n"
        f"READ state/execution_ready.json and EXECUTE NOW.\n"
        f"DO NOT do research, pulse checks, or anything else first.\n"
        f"EXECUTION IS PRIORITY #1. Report to Discord after placing."
    )

    print(f"⚠️ {len(unexecuted)} trades NOT executed: {tickers}")
    print("Firing re-inject...")
    inject_message(msg)

    # Also write to pending_entries for deferred-entry awareness
    pending = load_json(PENDING_FILE)
    pending["pending"] = [
        {
            "ticker": t["ticker"],
            "direction": t["direction"],
            "option_type": t["option_type"],
            "strike": t["strike"],
            "expiry": t["expiry"],
            "confidence": t["confidence"],
            "deferred_from": ready.get("timestamp"),
            "reason": "NOT EXECUTED on first pass — follow-up re-inject fired"
        }
        for t in unexecuted
    ]
    pending["last_updated"] = now.isoformat()
    pending["note"] = f"Follow-up check at {now.strftime('%H:%M')} — {len(unexecuted)} trades still pending"

    with open(PENDING_FILE, "w") as f:
        json.dump(pending, f, indent=2)


if __name__ == "__main__":
    main()
