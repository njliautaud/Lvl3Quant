#!/usr/bin/env python3
"""
test_rithmic_connection.py — Verify Rithmic LIVE credentials + connectivity.

Tests:
  1. WebSocket connect to Rithmic gateway
  2. Login to TICKER_PLANT (market data)
  3. Login to ORDER_PLANT (order routing)
  4. Resolve account + trade routes
  5. Subscribe to ESM6 BBO and print 10 ticks
  6. Clean disconnect

Usage:
    # Load .env first
    export $(grep -v '^#' .env | xargs)
    python3 test_rithmic_connection.py
"""

import asyncio
import os
import sys
import pathlib

# Load .env if present
env_path = pathlib.Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Add protobuf path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from rithmic_client import RithmicClient, BBOEvent, TradeEvent

SYMBOL = "ESM6"
EXCHANGE = "CME"
MAX_TICKS = 20
TIMEOUT = 30  # seconds


async def main():
    print("=" * 60)
    print("RITHMIC LIVE CONNECTION TEST")
    print("=" * 60)
    print(f"  System:   {os.environ.get('RITHMIC_SYSTEM', '(not set)')}")
    print(f"  User:     {os.environ.get('RITHMIC_USER', '(not set)')}")
    print(f"  URI:      {os.environ.get('RITHMIC_URI', '(not set)')}")
    print(f"  Symbol:   {SYMBOL} @ {EXCHANGE}")
    print(f"  Max ticks: {MAX_TICKS}")
    print()

    tick_count = 0
    done = asyncio.Event()

    async def on_md(event):
        nonlocal tick_count
        tick_count += 1
        if isinstance(event, BBOEvent):
            print(f"  [{tick_count:3d}] BBO  bid={event.bid_price:.2f}x{event.bid_size}  "
                  f"ask={event.ask_price:.2f}x{event.ask_size}  "
                  f"spread={event.ask_price - event.bid_price:.2f}")
        elif isinstance(event, TradeEvent):
            agg = {1: "BUY", 2: "SELL"}.get(event.aggressor, "UNK")
            print(f"  [{tick_count:3d}] TRADE {event.trade_price:.2f} x{event.trade_size} "
                  f"aggressor={agg}")

        if tick_count >= MAX_TICKS:
            done.set()

    client = RithmicClient()

    try:
        print("[1/5] Connecting to Rithmic gateway...")
        await client.connect()
        print(f"  ✓ Connected! Account: {client.account_id}")
        print(f"  ✓ FCM: {client.fcm_id}, IB: {client.ib_id}")
        print(f"  ✓ Trade route: {client.trade_route}")
        print()

        print(f"[2/5] Subscribing to {SYMBOL} @ {EXCHANGE}...")
        client.set_md_callback(on_md)
        await client.subscribe_md(SYMBOL, EXCHANGE)
        print(f"  ✓ Subscribed. Waiting for {MAX_TICKS} ticks (timeout {TIMEOUT}s)...")
        print()

        print("[3/5] Receiving market data:")
        try:
            await asyncio.wait_for(done.wait(), timeout=TIMEOUT)
        except asyncio.TimeoutError:
            if tick_count > 0:
                print(f"  ⚠ Timeout after {tick_count} ticks (market may be closed)")
            else:
                print(f"  ⚠ No ticks received in {TIMEOUT}s — market likely closed")
                print("  ℹ CME ES futures trade Sun-Fri 17:00-16:00 CT")

        print()
        print(f"[4/5] Connection summary:")
        print(f"  Ticks received: {tick_count}")
        print(f"  Account:        {client.account_id}")
        print(f"  Trade route:    {client.trade_route}")

        print()
        print("[5/5] Disconnecting...")
        await client.disconnect()
        print("  ✓ Clean disconnect")

    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        print(f"  Type: {type(e).__name__}")
        try:
            await client.disconnect()
        except Exception:
            pass
        return 1

    print()
    print("=" * 60)
    if tick_count > 0:
        print("✓ RITHMIC LIVE CONNECTION VERIFIED — ready for trading")
    else:
        print("⚠ CONNECTED but no ticks — verify during market hours")
        print("  ES futures: Sun 17:00 - Fri 16:00 CT")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
