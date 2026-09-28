#!/usr/bin/env python3
"""
Covered Call Income Paper Engine
==================================
Paper trades covered calls on user's Robinhood holdings.

For each held stock:
  - If no CC position open AND stock above 20-day SMA: sell a ~0.30-delta call, ~30 DTE
  - Strike ~ price * 1.05 (approximation for 0.30 delta)
  - Premium estimate: price * ATR_pct * 0.3 * sqrt(30/365), minus 10% bid-ask cost
  - Track total premium collected, assignments, stock P&L

PM2 cron: "0 20 * * 1-5" (4:00 PM ET daily)
State: /home/jupiter/Lvl3Quant/state/covered_call_paper_state.json
History: /home/jupiter/Lvl3Quant/state/covered_call_paper_history.csv
"""
import json
import math
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

ET = pytz.timezone("US/Eastern")
STATE_DIR = Path("/home/jupiter/Lvl3Quant/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "covered_call_paper_state.json"
HISTORY_FILE = STATE_DIR / "covered_call_paper_history.csv"

# ── Holdings (user's RH portfolio) ──
HOLDINGS = {
    "AVAV": 100,
    "CRDO": 100,
    "SKM": 100,
    "NOK": 100,
    "OUST": 100,
    "SEDG": 100,
}

# ── Parameters ──
DTE_TARGET = 30
DELTA_TARGET = 0.30
STRIKE_OFFSET = 0.05       # strike = price * (1 + 0.05) for ~0.30 delta
BID_ASK_COST = 0.10        # 10% bid-ask haircut on premium
ASSIGNMENT_THRESHOLD = 1.0  # assigned if price >= strike at expiry (in-the-money)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "positions": {},       # ticker -> {strike, expiry, premium, shares, entry_price}
        "total_premium": 0,
        "total_assignments": 0,
        "total_trades": 0,
        "assignment_pnl": 0,   # P&L from stock being called away
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


def get_stock_data(ticker: str) -> dict | None:
    """Fetch price, SMA, and ATR for a ticker."""
    try:
        df = yf.download(ticker, period="60d", interval="1d", progress=False, timeout=10)
        if df.empty or len(df) < 20:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        price = float(close.iloc[-1])
        sma20 = float(close.rolling(20).mean().iloc[-1])

        # ATR for premium estimation
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = atr / price

        return {
            "price": price,
            "sma20": sma20,
            "atr": atr,
            "atr_pct": atr_pct,
            "above_sma20": price > sma20,
        }
    except Exception as e:
        print(f"  [WARN] {ticker}: {e}")
        return None


def estimate_premium(price: float, atr_pct: float, dte: int = DTE_TARGET) -> float:
    """
    Rough premium estimate for a ~0.30 delta call.
    premium ~= price * ATR_pct * delta * sqrt(DTE/365)
    Then apply bid-ask haircut.
    """
    raw = price * atr_pct * DELTA_TARGET * math.sqrt(dte / 365)
    return raw * (1 - BID_ASK_COST)


def run():
    now = datetime.now(ET)
    print(f"=== Covered Call Paper Engine — {now.strftime('%Y-%m-%d %H:%M ET')} ===")

    state = load_state()

    print(f"Holdings: {list(HOLDINGS.keys())}")
    print(f"Open CC positions: {list(state['positions'].keys())}")
    print(f"Total premium collected: ${state['total_premium']:,.2f}")

    # ── Check expirations first ──
    expired = []
    for ticker, pos in state["positions"].items():
        expiry = datetime.fromisoformat(pos["expiry"])
        if now >= expiry:
            expired.append(ticker)

    for ticker in expired:
        pos = state["positions"][ticker]
        data = get_stock_data(ticker)
        current_price = data["price"] if data else pos["entry_price"]

        assigned = current_price >= pos["strike"]
        state["total_trades"] += 1

        if assigned:
            # Stock called away at strike
            stock_pnl = (pos["strike"] - pos["entry_price"]) * pos["shares"]
            state["total_assignments"] += 1
            state["assignment_pnl"] += stock_pnl
            action = "ASSIGNED"
            extra_pnl = stock_pnl
            print(f"  {ticker}: ASSIGNED at ${pos['strike']:.2f} (was ${pos['entry_price']:.2f})")
            print(f"    Stock P&L: ${stock_pnl:+,.2f} + premium ${pos['premium']:,.2f}")
        else:
            action = "EXPIRED_OTM"
            extra_pnl = 0
            print(f"  {ticker}: Expired OTM (price ${current_price:.2f} < strike ${pos['strike']:.2f})")
            print(f"    Kept full premium: ${pos['premium']:,.2f}")

        trade_record = {
            "date": now.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "action": action,
            "strike": pos["strike"],
            "expiry": pos["expiry"],
            "entry_price": pos["entry_price"],
            "exit_price": round(current_price, 2),
            "premium": round(pos["premium"], 2),
            "shares": pos["shares"],
            "stock_pnl": round(extra_pnl, 2),
        }
        append_history(trade_record)
        del state["positions"][ticker]

    # ── Check for new entries ──
    for ticker, shares in HOLDINGS.items():
        if ticker in state["positions"]:
            continue  # Already have an open CC

        data = get_stock_data(ticker)
        if data is None:
            continue

        if not data["above_sma20"]:
            print(f"  {ticker}: ${data['price']:.2f} below 20 SMA (${data['sma20']:.2f}), skipping")
            continue

        strike = round(data["price"] * (1 + STRIKE_OFFSET), 2)
        premium_per_share = estimate_premium(data["price"], data["atr_pct"])
        total_premium = premium_per_share * shares
        expiry_date = now + timedelta(days=DTE_TARGET)

        state["positions"][ticker] = {
            "strike": strike,
            "expiry": expiry_date.isoformat(),
            "premium": round(total_premium, 2),
            "premium_per_share": round(premium_per_share, 2),
            "shares": shares,
            "entry_price": round(data["price"], 2),
            "entry_date": now.strftime("%Y-%m-%d"),
        }
        state["total_premium"] += total_premium

        trade_record = {
            "date": now.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "action": "SELL_CC",
            "strike": strike,
            "expiry": expiry_date.strftime("%Y-%m-%d"),
            "entry_price": round(data["price"], 2),
            "exit_price": "",
            "premium": round(total_premium, 2),
            "shares": shares,
            "stock_pnl": 0,
        }
        append_history(trade_record)

        print(f"  {ticker}: SELL CC ${strike:.2f} strike, exp {expiry_date.strftime('%Y-%m-%d')}")
        print(f"    Price: ${data['price']:.2f} | Premium: ${total_premium:.2f} ({premium_per_share:.2f}/sh)")

    # ── Summary ──
    print(f"\n--- Summary ---")
    print(f"  Open positions: {len(state['positions'])}")
    print(f"  Total premium collected: ${state['total_premium']:,.2f}")
    print(f"  Assignments: {state['total_assignments']} | Assignment P&L: ${state['assignment_pnl']:+,.2f}")
    print(f"  Net income: ${state['total_premium'] + state['assignment_pnl']:+,.2f}")

    save_state(state)
    print("Done.")


if __name__ == "__main__":
    run()
