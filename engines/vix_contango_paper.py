#!/usr/bin/env python3
"""
VIX Contango Income Paper Engine
==================================
Paper trades VIX contango premium selling via put spreads.

Strategy:
  - When VIX < 18: "sell" a VIX put spread (buy 15 put, sell 18 put, $300 width)
  - Credit: $0.80 per spread (conservative estimate at VIX<18)
  - Exit: 30 days later OR VIX < 13 (early profit take) OR VIX > 28 (stop loss)
  - Max 1 position at a time
  - Starting capital: $10,000

PM2 cron: "0 20 * * 1-5" (4:00 PM ET, after VIX settles)
State: /home/jupiter/Lvl3Quant/state/vix_contango_paper_state.json
History: /home/jupiter/Lvl3Quant/state/vix_contango_paper_history.csv
"""
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz
import yfinance as yf

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

ET = pytz.timezone("US/Eastern")
STATE_DIR = Path("/home/jupiter/Lvl3Quant/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "vix_contango_paper_state.json"
HISTORY_FILE = STATE_DIR / "vix_contango_paper_history.csv"

# ── Strategy Parameters ──
INITIAL_CAPITAL = 10_000
CREDIT_PER_SPREAD = 80       # $0.80 * 100 multiplier = $80
SPREAD_WIDTH = 300            # $3.00 * 100 = $300 max risk
MAX_RISK = SPREAD_WIDTH - CREDIT_PER_SPREAD  # $220 max loss per spread
VIX_ENTRY_THRESHOLD = 18.0   # Enter when VIX < this
VIX_PROFIT_TAKE = 13.0       # Early exit if VIX drops below this
VIX_STOP_LOSS = 28.0         # Stop loss if VIX spikes above this
HOLD_DAYS = 30               # Max holding period
SHORT_STRIKE = 18            # Sell 18 put
LONG_STRIKE = 15             # Buy 15 put


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "capital": INITIAL_CAPITAL,
        "position_open": False,
        "entry_date": None,
        "entry_vix": None,
        "num_spreads": 0,
        "total_trades": 0,
        "total_pnl": 0,
        "wins": 0,
        "losses": 0,
        "start_date": datetime.now(ET).strftime("%Y-%m-%d"),
        "last_update": None,
    }


def save_state(state: dict):
    state["last_update"] = datetime.now(ET).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2))


def append_history(row: dict):
    df = pd.DataFrame([row])
    if HISTORY_FILE.exists():
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)


def get_vix() -> float | None:
    """Fetch current VIX level."""
    try:
        data = yf.download("^VIX", period="5d", interval="1d", progress=False, timeout=10)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return float(data["Close"].iloc[-1])
    except Exception as e:
        print(f"[ERROR] Failed to fetch VIX: {e}")
        return None


def calculate_pnl(entry_vix: float, exit_vix: float) -> float:
    """
    Calculate P&L per spread based on VIX at expiry.
    Sell 18 put / Buy 15 put.
    - VIX >= 18: keep full credit ($80)
    - VIX 15-18: partial loss = credit - (18 - VIX) * 100
    - VIX <= 15: max loss = -(spread_width - credit) = -$220
    """
    if exit_vix >= SHORT_STRIKE:
        return CREDIT_PER_SPREAD  # Full profit
    elif exit_vix <= LONG_STRIKE:
        return -(SPREAD_WIDTH - CREDIT_PER_SPREAD)  # Max loss
    else:
        intrinsic = (SHORT_STRIKE - exit_vix) * 100
        return CREDIT_PER_SPREAD - intrinsic


def run():
    now = datetime.now(ET)
    print(f"=== VIX Contango Paper Engine — {now.strftime('%Y-%m-%d %H:%M ET')} ===")

    state = load_state()
    vix = get_vix()

    if vix is None:
        print("[ERROR] Could not fetch VIX. Skipping.")
        return

    print(f"Current VIX: {vix:.2f}")
    print(f"Capital: ${state['capital']:,.2f} | Position open: {state['position_open']}")

    if state["position_open"]:
        # Check exit conditions
        entry_date = datetime.fromisoformat(state["entry_date"])
        days_held = (now - entry_date).days
        exit_reason = None

        if days_held >= HOLD_DAYS:
            exit_reason = f"expiry ({days_held}d held)"
        elif vix < VIX_PROFIT_TAKE:
            exit_reason = f"early profit take (VIX={vix:.2f} < {VIX_PROFIT_TAKE})"
        elif vix > VIX_STOP_LOSS:
            exit_reason = f"stop loss (VIX={vix:.2f} > {VIX_STOP_LOSS})"

        if exit_reason:
            pnl_per_spread = calculate_pnl(state["entry_vix"], vix)
            total_pnl = pnl_per_spread * state["num_spreads"]

            state["capital"] += total_pnl
            state["total_pnl"] += total_pnl
            state["total_trades"] += 1
            if total_pnl >= 0:
                state["wins"] += 1
            else:
                state["losses"] += 1

            trade_record = {
                "date": now.strftime("%Y-%m-%d"),
                "action": "CLOSE",
                "reason": exit_reason,
                "entry_date": state["entry_date"],
                "entry_vix": state["entry_vix"],
                "exit_vix": round(vix, 2),
                "days_held": days_held,
                "num_spreads": state["num_spreads"],
                "pnl_per_spread": round(pnl_per_spread, 2),
                "total_pnl": round(total_pnl, 2),
                "capital_after": round(state["capital"], 2),
            }
            append_history(trade_record)

            print(f"  CLOSED: {exit_reason}")
            print(f"  P&L: ${total_pnl:+,.2f} ({state['num_spreads']} spreads x ${pnl_per_spread:+.2f})")
            print(f"  Capital now: ${state['capital']:,.2f}")

            state["position_open"] = False
            state["entry_date"] = None
            state["entry_vix"] = None
            state["num_spreads"] = 0
        else:
            print(f"  Position open {days_held}d, entry VIX={state['entry_vix']:.2f}. No exit trigger.")

    else:
        # Check entry conditions
        if vix < VIX_ENTRY_THRESHOLD:
            # Size: risk no more than 5% of capital per trade
            max_risk_budget = state["capital"] * 0.05
            num_spreads = max(1, int(max_risk_budget / MAX_RISK))

            state["position_open"] = True
            state["entry_date"] = now.isoformat()
            state["entry_vix"] = round(vix, 2)
            state["num_spreads"] = num_spreads

            trade_record = {
                "date": now.strftime("%Y-%m-%d"),
                "action": "OPEN",
                "reason": f"VIX={vix:.2f} < {VIX_ENTRY_THRESHOLD}",
                "entry_date": now.strftime("%Y-%m-%d"),
                "entry_vix": round(vix, 2),
                "exit_vix": "",
                "days_held": 0,
                "num_spreads": num_spreads,
                "pnl_per_spread": 0,
                "total_pnl": 0,
                "capital_after": round(state["capital"], 2),
            }
            append_history(trade_record)

            print(f"  OPENED: Sell {num_spreads}x 18/15 put spread @ VIX {vix:.2f}")
            print(f"  Credit: ${CREDIT_PER_SPREAD * num_spreads:,.2f} | Max risk: ${MAX_RISK * num_spreads:,.2f}")
        else:
            print(f"  No entry: VIX {vix:.2f} >= {VIX_ENTRY_THRESHOLD} threshold")

    # Print summary
    wr = state["wins"] / state["total_trades"] * 100 if state["total_trades"] > 0 else 0
    print(f"\n--- Summary ---")
    print(f"  Total trades: {state['total_trades']} | W/L: {state['wins']}/{state['losses']} | WR: {wr:.0f}%")
    print(f"  Total P&L: ${state['total_pnl']:+,.2f} | Capital: ${state['capital']:,.2f}")

    save_state(state)
    print("Done.")


if __name__ == "__main__":
    run()
