#!/usr/bin/env python3
"""
Insider Momentum Paper Trading Engine
======================================
Downloads prices via yfinance, loads insider data from the feature store,
runs the insider_momentum strategy, and manages paper positions.

Designed to be run once daily (after market close or next morning).
Idempotent: running twice on the same day produces no duplicate trades.
"""

import json
import csv
import os
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ENGINE_DIR = Path(__file__).resolve().parent
STATE_FILE = ENGINE_DIR / "state.json"
TRADES_FILE = ENGINE_DIR / "trades.csv"
INSIDER_DATA_PATH = Path("/home/jupiter/Lvl3Quant/data/feature_store/edgar_form4/insider_daily_v2.parquet")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_state() -> dict:
    """Load paper trading state from disk."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "capital": 10000,
        "open_positions": [],
        "closed_trades": [],
        "last_run": None,
    }


def save_state(state: dict):
    """Persist state to disk."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(row: dict):
    """Append a trade row to the CSV log."""
    fieldnames = ["date", "action", "ticker", "price", "shares", "reason", "pnl"]
    write_header = not TRADES_FILE.exists() or TRADES_FILE.stat().st_size == 0
    with open(TRADES_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def is_market_open_today() -> bool:
    """Check if today is a US equity trading day (weekday + not a major holiday).

    Uses pandas market calendar if available, otherwise falls back to weekday check.
    """
    today = date.today()
    # Weekend check
    if today.weekday() >= 5:
        return False
    # Try pandas_market_calendars for holiday awareness
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_date=today.isoformat(), end_date=today.isoformat()
        )
        return len(schedule) > 0
    except ImportError:
        # Fallback: weekday = open (misses holidays but functional)
        return True


def download_prices() -> pd.DataFrame:
    """Download recent prices for all needed tickers via yfinance."""
    import yfinance as yf

    tickers = ["SPY", "XLU", "XLP", "XLV", "^VIX"]
    # Need enough history for 30-day trend + insider lookback
    data = yf.download(tickers, period="120d", auto_adjust=True, progress=False)

    if data.empty:
        raise RuntimeError("yfinance returned empty data")

    # yfinance returns MultiIndex columns (field, ticker) — extract Close
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data.copy()

    # Rename ^VIX to VIX for convenience
    if "^VIX" in prices.columns:
        prices = prices.rename(columns={"^VIX": "VIX"})

    prices = prices.dropna(how="all")
    return prices


def load_insider_data() -> pd.DataFrame:
    """Load insider trading data from the feature store."""
    if not INSIDER_DATA_PATH.exists():
        print(f"WARNING: Insider data not found at {INSIDER_DATA_PATH}")
        print("  Strategy will run WITHOUT insider gate (all z-scores default to 0)")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(INSIDER_DATA_PATH)
        print(f"  Loaded insider data: {len(df)} rows, columns={list(df.columns)}")
        return df
    except Exception as e:
        print(f"WARNING: Failed to load insider data: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------

def check_exits(state: dict, prices: pd.DataFrame, today_str: str) -> list:
    """Check all open positions for exit conditions. Returns list of closed positions."""
    from strategy import should_exit

    closed = []
    remaining = []

    for pos in state["open_positions"]:
        ticker = pos["ticker"]
        if ticker not in prices.columns:
            print(f"  WARNING: {ticker} not in price data, keeping position")
            remaining.append(pos)
            continue

        current_price = float(prices[ticker].iloc[-1])

        if should_exit(pos, current_price, today_str, portfolio_dd=0.0):
            entry_price = pos.get("entry_price_adj", pos.get("entry_price", current_price))
            shares = pos["shares"]
            pnl = (current_price - entry_price) * shares
            pnl_pct = (current_price - entry_price) / entry_price

            # Determine exit reason
            days_held = np.busday_count(
                np.datetime64(pos["entry_date"], "D"),
                np.datetime64(today_str, "D"),
            )
            if pnl_pct >= 0.035:
                reason = "TAKE_PROFIT"
            elif days_held >= 3:
                reason = "MAX_HOLD"
            elif days_held >= 1 and pnl_pct < 0:
                reason = "UNDERWATER_EXIT"
            else:
                hwm = pos.get("hwm", pos.get("high_water", entry_price))
                dd = (current_price - hwm) / hwm
                if dd <= -0.02:
                    reason = "TRAILING_STOP"
                else:
                    reason = "EXIT"

            trade_record = {
                "ticker": ticker,
                "entry_date": pos["entry_date"],
                "entry_price": entry_price,
                "exit_date": today_str,
                "exit_price": current_price,
                "shares": shares,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct * 100, 2),
                "reason": reason,
            }
            closed.append(trade_record)
            state["closed_trades"].append(trade_record)
            state["capital"] += current_price * shares  # return capital

            log_trade({
                "date": today_str,
                "action": "SELL",
                "ticker": ticker,
                "price": round(current_price, 2),
                "shares": shares,
                "reason": reason,
                "pnl": round(pnl, 2),
            })

            print(f"  EXIT {ticker}: {reason} | price={current_price:.2f} | "
                  f"pnl=${pnl:.2f} ({pnl_pct*100:.1f}%) | held {days_held}d")
        else:
            # Update high water mark
            current_price_val = float(prices[ticker].iloc[-1])
            hwm_key = "hwm" if "hwm" in pos else "high_water"
            if current_price_val > pos.get(hwm_key, pos.get("entry_price", 0)):
                pos[hwm_key] = current_price_val
            remaining.append(pos)

    state["open_positions"] = remaining
    return closed


def check_entries(state: dict, prices: pd.DataFrame, insider_data: pd.DataFrame,
                  today_str: str) -> list:
    """Check for new entry signals. Returns list of new positions."""
    from strategy import generate_signals, TRADEABLE_SECTORS, MAX_PER_TRADE, MAX_CONCURRENT, SLIPPAGE_PCT

    # Don't enter if we already have max positions
    if len(state["open_positions"]) >= MAX_CONCURRENT:
        print("  Max concurrent positions reached, skipping entry scan")
        return []

    # Build price dataframe for the strategy (needs sector ETF columns)
    spy = prices["SPY"] if "SPY" in prices.columns else None
    vix = prices["VIX"] if "VIX" in prices.columns else None

    # Generate signals — strategy expects full price history
    signals = generate_signals(prices, spy, vix, insider_data=insider_data)

    # Check today's signals
    today_dt = pd.Timestamp(today_str)
    if today_dt not in signals.index:
        # Try to find closest date
        closest = signals.index[signals.index <= today_dt]
        if len(closest) == 0:
            print("  No matching date in signals index")
            return []
        today_dt = closest[-1]

    new_positions = []
    for etf in TRADEABLE_SECTORS:
        if len(state["open_positions"]) + len(new_positions) >= MAX_CONCURRENT:
            break

        if etf not in signals.columns:
            continue

        if not signals.loc[today_dt, etf]:
            continue

        # Don't re-enter a ticker we already hold
        held_tickers = {p["ticker"] for p in state["open_positions"]}
        if etf in held_tickers:
            continue

        current_price = float(prices[etf].iloc[-1])
        # Apply slippage
        entry_price = current_price * (1.0 + SLIPPAGE_PCT)
        shares = int(MAX_PER_TRADE / entry_price)

        if shares <= 0:
            continue

        cost = entry_price * shares
        if cost > state["capital"]:
            print(f"  Insufficient capital for {etf}: need ${cost:.2f}, have ${state['capital']:.2f}")
            continue

        pos = {
            "ticker": etf,
            "entry_date": today_str,
            "entry_price": round(entry_price, 4),
            "entry_price_adj": round(entry_price, 4),
            "shares": shares,
            "hwm": round(entry_price, 4),
        }
        new_positions.append(pos)
        state["open_positions"].append(pos)
        state["capital"] -= cost

        log_trade({
            "date": today_str,
            "action": "BUY",
            "ticker": etf,
            "price": round(entry_price, 2),
            "shares": shares,
            "reason": "SIGNAL",
            "pnl": "",
        })

        print(f"  ENTRY {etf}: {shares} shares @ ${entry_price:.2f} = ${cost:.2f}")

    return new_positions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    today_str = date.today().isoformat()
    print(f"\n{'='*60}")
    print(f"Insider Momentum Paper Engine — {today_str}")
    print(f"{'='*60}")

    # Idempotency: skip if already ran today
    state = load_state()
    if state.get("last_run") == today_str:
        print(f"Already ran today ({today_str}). Skipping for idempotency.")
        return

    # Market open check
    if not is_market_open_today():
        print("Market is closed today (weekend/holiday). Skipping.")
        state["last_run"] = today_str
        save_state(state)
        return

    # Download prices
    print("\n[1/4] Downloading prices...")
    try:
        prices = download_prices()
        latest_date = prices.index[-1].strftime("%Y-%m-%d")
        print(f"  Got {len(prices)} days of data, latest: {latest_date}")
        print(f"  Tickers: {list(prices.columns)}")
    except Exception as e:
        print(f"ERROR downloading prices: {e}")
        sys.exit(1)

    # Load insider data
    print("\n[2/4] Loading insider data...")
    insider_data = load_insider_data()

    # Check exits first
    print(f"\n[3/4] Checking exits ({len(state['open_positions'])} open positions)...")
    exits = check_exits(state, prices, today_str)

    # Check entries
    print(f"\n[4/4] Scanning for entry signals...")
    entries = check_entries(state, prices, insider_data, today_str)

    # Update state
    state["last_run"] = today_str
    save_state(state)

    # Summary
    print(f"\n{'='*60}")
    print("DAILY SUMMARY")
    print(f"{'='*60}")
    print(f"  Date:            {today_str}")
    print(f"  Exits today:     {len(exits)}")
    print(f"  Entries today:   {len(entries)}")
    print(f"  Open positions:  {len(state['open_positions'])}")
    print(f"  Capital:         ${state['capital']:.2f}")

    total_closed = len(state["closed_trades"])
    if total_closed > 0:
        pnls = [t["pnl"] for t in state["closed_trades"]]
        total_pnl = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / total_closed * 100
        print(f"  Total trades:    {total_closed}")
        print(f"  Total P&L:       ${total_pnl:.2f}")
        print(f"  Win rate:        {wr:.0f}%")
        print(f"  Avg P&L:         ${total_pnl/total_closed:.2f}")

    for pos in state["open_positions"]:
        ticker = pos["ticker"]
        if ticker in prices.columns:
            cur = float(prices[ticker].iloc[-1])
            ep = pos.get("entry_price_adj", pos.get("entry_price", cur))
            unrealized = (cur - ep) * pos["shares"]
            unrealized_pct = (cur - ep) / ep * 100
            print(f"  HOLDING: {ticker} | {pos['shares']} shares | "
                  f"entry=${ep:.2f} | cur=${cur:.2f} | "
                  f"unrealized=${unrealized:.2f} ({unrealized_pct:+.1f}%)")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
