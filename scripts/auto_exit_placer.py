#!/usr/bin/env python3
"""
Auto Exit Placer — places GTC limit sell orders for TP on open positions.

This runs ONCE per position after entry to place the take-profit limit sell.
Stop-loss and trailing stops require active monitoring (options_exit_monitor.py)
and inject-driven execution since RH doesn't support native stop orders on options.

For stop-loss triggers: the exit monitor detects the trigger and injects an
URGENT sell prompt into the autonomy pipeline.

Cron: 35 9 * * 1-5 (5 min after market open, weekdays only)
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
BASE = Path("/home/jupiter/Lvl3Quant")
POSITIONS_FILE = BASE / "state" / "agentic_positions.json"
EXIT_LOG = BASE / "logs" / "auto_exit_placer.log"

# Import market status guard
sys.path.insert(0, str(BASE / "scripts"))


def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"[{ts}] {msg}"
    print(line)
    EXIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EXIT_LOG, "a") as f:
        f.write(line + "\n")


def load_positions() -> dict:
    try:
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_positions(data: dict):
    with open(POSITIONS_FILE, "w") as f:
        json.dump(data, f, indent=4)


def generate_exit_instructions() -> list:
    """
    Check all open positions. For any without a limit sell placed,
    generate instructions to place a GTC limit sell at the TP price.

    Returns list of dicts with trade instructions for the autonomy inject.
    """
    from market_status import get_market_status
    ms = get_market_status()
    if not ms.get("is_trading_day", True):
        log(f"Market closed: {ms.get('reason')}. Skipping.")
        return []

    data = load_positions()
    if not data or "positions" not in data:
        log("No positions file or empty.")
        return []

    instructions = []
    for pos_id, pos in data["positions"].items():
        if pos.get("status") != "OPEN":
            continue

        if pos.get("limit_sell_placed", False):
            log(f"{pos_id}: TP limit sell already placed. Skip.")
            continue

        symbol = pos.get("symbol", "???")
        tp_price = pos.get("target_price")
        option_id = pos.get("option_id")
        quantity = pos.get("quantity", 1)

        if not tp_price or not option_id:
            log(f"{pos_id}: Missing target_price or option_id. Skip.")
            continue

        instructions.append({
            "action": "PLACE_TP_LIMIT_SELL",
            "position_id": pos_id,
            "symbol": symbol,
            "option_id": option_id,
            "quantity": quantity,
            "limit_price": tp_price,
            "order_type": "limit",
            "time_in_force": "gtc",
            "reason": f"Auto TP at ${tp_price:.2f} for {pos.get('option_details', symbol)}"
        })
        log(f"{pos_id}: Generated TP sell instruction at ${tp_price:.2f}")

    return instructions


def generate_sl_alert(pos_id: str, pos: dict, current_price: float, pnl_pct: float) -> str:
    """Generate an urgent sell inject message for stop-loss triggers."""
    symbol = pos.get("symbol", "???")
    details = pos.get("option_details", symbol)
    entry = pos.get("entry_price", 0)
    return (
        f"URGENT_EXIT: {details} hit stop-loss. "
        f"Entry ${entry:.2f}, current ${current_price:.2f} ({pnl_pct:+.1f}%). "
        f"Place limit sell at ${current_price * 0.95:.2f} (5% below current to ensure fill). "
        f"Option ID: {pos.get('option_id')}. Quantity: {pos.get('quantity', 1)}. "
        f"DO NOT DELAY — execute immediately per HC #784."
    )


def main():
    log("=== Auto Exit Placer starting ===")
    instructions = generate_exit_instructions()

    if not instructions:
        log("No exit orders to place. Done.")
        return

    # Write instructions for the autonomy inject to pick up
    output_file = BASE / "state" / "pending_exit_orders.json"
    output = {
        "timestamp": datetime.now(ET).isoformat(),
        "instructions": instructions,
        "status": "PENDING"
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    log(f"Wrote {len(instructions)} exit instruction(s) to pending_exit_orders.json")

    # Also inject an execution prompt
    import subprocess
    inject_msg = (
        f"EXIT_ORDERS_READY: {len(instructions)} take-profit limit sell(s) need placement. "
        f"Read state/pending_exit_orders.json and place each order via Robinhood. "
        f"Use limit orders per HC #785. Mark limit_sell_placed=true in agentic_positions.json after each."
    )
    try:
        subprocess.run(
            ["/bin/bash", str(BASE / "scripts" / "autonomy_inject.sh"), inject_msg],
            timeout=10
        )
        log("Injected exit order prompt.")
    except Exception as e:
        log(f"Inject failed: {e}")

    log("=== Auto Exit Placer complete ===")


if __name__ == "__main__":
    main()
