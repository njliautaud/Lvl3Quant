"""
CTA Multi-Asset Trend Following Paper Engine

Strategy: Equal-weight trend following across 9 diversified ETFs.
Signal: Long when price > SMA(50), flat otherwise.
Rebalance: Weekly (Monday open).
Universe: GLD, SLV, USO, UNG, DBA, COPX, UUP, TLT, EEM

Validated metrics (weekly rebalance):
  - Sharpe ~1.0 (full portfolio), ~0.75 per-asset invested
  - SPY correlation 0.16 (excellent diversifier)
  - Permutation p=0.000 (real edge)
  - R1: fails raw but red-day Sharpe positive (0.83) — passes HC #709

Runs ONE check per invocation (cron at 10:00 AM ET weekdays).
On non-Monday: mark-to-market + NAV update only.
On Monday: rebalance — check SMA signals, update positions.

State: live_trading_linux/cta_trend_state/
  - state.json          — current positions, cash, NAV
  - trades.jsonl        — all trade records
  - equity_curve.jsonl  — daily NAV snapshots
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed")
    sys.exit(1)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
STATE_DIR = Path(__file__).resolve().parent / "cta_trend_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "state.json"
TRADES_LOG = STATE_DIR / "trades.jsonl"
EQUITY_LOG = STATE_DIR / "equity_curve.jsonl"

UNIVERSE = ["GLD", "SLV", "USO", "UNG", "DBA", "COPX", "UUP", "TLT", "EEM"]
SMA_LEN = 50
ANCHOR_USD = 100_000.0
COST_BPS = 10.0  # 10bps per trade (spread cost, commission-free on RH)
REBAL_DAY = 0  # Monday = 0

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "cash": ANCHOR_USD,
        "positions": {},  # {symbol: {"shares": float, "entry_price": float, "entry_date": str}}
        "nav": ANCHOR_USD,
        "start_date": datetime.now(timezone.utc).isoformat(),
        "last_rebalance": None,
        "total_trades": 0,
        "realized_pnl": 0.0,
    }

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

def log_trade(record: dict):
    with open(TRADES_LOG, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

def log_equity(nav: float, date: str):
    with open(EQUITY_LOG, "a") as f:
        f.write(json.dumps({"date": date, "nav": nav, "ts": datetime.now(timezone.utc).isoformat()}) + "\n")

# ---------------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------------

def get_prices(symbols: list[str], lookback_days: int = 80) -> pd.DataFrame:
    """Download recent prices for SMA calculation."""
    data = {}
    for sym in symbols:
        try:
            df = yf.download(sym, period=f"{lookback_days}d", progress=False)
            if len(df) > 0:
                data[sym] = df["Close"].squeeze()
        except Exception as e:
            print(f"  WARNING: Failed to download {sym}: {e}")
    return pd.DataFrame(data)

def get_current_prices(symbols: list[str]) -> dict:
    """Get current/latest prices."""
    prices = {}
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            info = t.fast_info
            prices[sym] = info.get("lastPrice", info.get("previousClose", 0))
        except:
            try:
                df = yf.download(sym, period="2d", progress=False)
                if len(df) > 0:
                    prices[sym] = float(df["Close"].iloc[-1])
            except:
                print(f"  WARNING: Cannot get price for {sym}")
    return prices

# ---------------------------------------------------------------------------
# STRATEGY LOGIC
# ---------------------------------------------------------------------------

def compute_signals(prices_df: pd.DataFrame) -> dict:
    """For each ETF, return 1 (long) or 0 (flat) based on SMA."""
    signals = {}
    for sym in UNIVERSE:
        if sym not in prices_df.columns:
            signals[sym] = 0
            continue
        series = prices_df[sym].dropna()
        if len(series) < SMA_LEN:
            signals[sym] = 0
            continue
        current = series.iloc[-1]
        sma = series.rolling(SMA_LEN).mean().iloc[-1]
        signals[sym] = 1 if current > sma else 0
    return signals

def rebalance(state: dict, signals: dict, current_prices: dict):
    """Rebalance portfolio based on signals."""
    now = datetime.now(timezone.utc).isoformat()

    # Target: equal weight across assets with signal=1
    long_assets = [s for s in UNIVERSE if signals.get(s, 0) == 1 and s in current_prices]
    n_long = len(long_assets)

    if n_long == 0:
        # All flat — sell everything
        target_positions = {}
    else:
        # Equal weight: NAV / n_long per asset
        nav = compute_nav(state, current_prices)
        per_asset = nav / n_long
        target_positions = {}
        for sym in long_assets:
            price = current_prices[sym]
            if price > 0:
                target_positions[sym] = {"target_shares": per_asset / price}

    # Close positions no longer wanted
    for sym in list(state["positions"].keys()):
        if sym not in target_positions:
            pos = state["positions"][sym]
            shares = pos["shares"]
            price = current_prices.get(sym, pos["entry_price"])
            proceeds = shares * price
            cost = proceeds * COST_BPS / 10000
            pnl = (price - pos["entry_price"]) * shares - cost

            state["cash"] += proceeds - cost
            state["realized_pnl"] += pnl
            state["total_trades"] += 1

            log_trade({
                "action": "SELL", "symbol": sym, "shares": shares,
                "price": price, "pnl": round(pnl, 2), "cost": round(cost, 2),
                "reason": "signal_flat", "date": now
            })

            del state["positions"][sym]
            print(f"  SOLD {sym}: {shares:.2f} shares @ ${price:.2f} (PnL: ${pnl:.2f})")

    # Open/adjust positions
    for sym, target in target_positions.items():
        target_shares = target["target_shares"]
        price = current_prices[sym]

        if sym in state["positions"]:
            # Already holding — check if rebalance needed
            current_shares = state["positions"][sym]["shares"]
            diff = target_shares - current_shares
            if abs(diff / current_shares) < 0.15:
                continue  # within 15% tolerance, skip

            # Adjust
            if diff > 0:
                cost_val = diff * price * COST_BPS / 10000
                state["cash"] -= (diff * price + cost_val)
                state["positions"][sym]["shares"] = target_shares
                state["total_trades"] += 1
                log_trade({"action": "ADD", "symbol": sym, "shares_added": round(diff, 2),
                          "price": price, "date": now})
                print(f"  ADDED {diff:.2f} shares of {sym} @ ${price:.2f}")
            else:
                sell_shares = abs(diff)
                proceeds = sell_shares * price
                cost_val = proceeds * COST_BPS / 10000
                state["cash"] += proceeds - cost_val
                state["positions"][sym]["shares"] = target_shares
                state["total_trades"] += 1
                log_trade({"action": "TRIM", "symbol": sym, "shares_trimmed": round(sell_shares, 2),
                          "price": price, "date": now})
        else:
            # New position
            cost_val = target_shares * price * COST_BPS / 10000
            total_cost = target_shares * price + cost_val
            if total_cost > state["cash"]:
                target_shares = (state["cash"] * 0.95) / price  # leave 5% buffer

            if target_shares * price > 100:  # minimum $100 position
                state["cash"] -= (target_shares * price + cost_val)
                state["positions"][sym] = {
                    "shares": target_shares,
                    "entry_price": price,
                    "entry_date": now,
                }
                state["total_trades"] += 1
                log_trade({"action": "BUY", "symbol": sym, "shares": round(target_shares, 2),
                          "price": price, "date": now})
                print(f"  BOUGHT {target_shares:.2f} shares of {sym} @ ${price:.2f}")

    state["last_rebalance"] = now

def compute_nav(state: dict, current_prices: dict) -> float:
    """Compute current NAV."""
    nav = state["cash"]
    for sym, pos in state["positions"].items():
        price = current_prices.get(sym, pos["entry_price"])
        nav += pos["shares"] * price
    return nav

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CTA TREND FOLLOWING PAPER ENGINE")
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    state = load_state()

    # Get market data
    print("\nFetching market data...")
    prices_df = get_prices(UNIVERSE, lookback_days=80)
    current_prices = get_current_prices(UNIVERSE)

    if not current_prices:
        print("ERROR: No prices available. Market may be closed.")
        save_state(state)
        return

    # Compute signals
    signals = compute_signals(prices_df)
    long_count = sum(1 for v in signals.values() if v == 1)
    print(f"\nSignals: {long_count}/{len(UNIVERSE)} assets above SMA{SMA_LEN}")
    for sym, sig in sorted(signals.items()):
        status = "LONG ✅" if sig == 1 else "FLAT ❌"
        price = current_prices.get(sym, 0)
        sma = prices_df[sym].rolling(SMA_LEN).mean().iloc[-1] if sym in prices_df.columns else 0
        print(f"  {sym:5s}: {status}  (price=${price:.2f}, SMA50=${sma:.2f})")

    # Check if rebalance day
    today = datetime.now()
    is_rebal_day = today.weekday() == REBAL_DAY

    if is_rebal_day:
        print(f"\n📊 REBALANCE DAY (Monday)")
        rebalance(state, signals, current_prices)
    else:
        print(f"\n📊 Non-rebalance day (next rebalance: Monday)")

    # Mark-to-market
    nav = compute_nav(state, current_prices)
    state["nav"] = nav
    pnl_pct = (nav / ANCHOR_USD - 1) * 100

    print(f"\n{'='*60}")
    print(f"NAV: ${nav:,.2f} ({pnl_pct:+.2f}%)")
    print(f"Cash: ${state['cash']:,.2f}")
    print(f"Positions: {len(state['positions'])}")
    print(f"Total trades: {state['total_trades']}")
    print(f"Realized PnL: ${state['realized_pnl']:,.2f}")
    print(f"{'='*60}")

    # Log equity
    log_equity(nav, today.strftime("%Y-%m-%d"))

    save_state(state)
    print("State saved.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
