#!/usr/bin/env python3
"""
Trend CTA Momentum Paper Engine — Monthly Rebalance
====================================================

Strategy: Dual momentum (absolute + relative) across 8 diversified ETFs.
  - Universe: SPY, EFA, EEM, TLT, IEF, GLD, DBC, VNQ
  - Signal: 6-month (126 trading day) momentum using T-1 close
  - Absolute momentum filter: only hold assets with positive 6m return
  - Relative momentum ranking: top 3 by 6m return
  - Allocation: equal weight (33.3% each), remainder in SHY
  - Rebalance: first trading day of each month
  - Costs: 10 bps per leg (applied on rebalance)

Validated backtest (2005-2026):
  Sharpe 0.91, CAGR 11.3%, MaxDD -18.2%, SPY corr 0.39
  Permutation p=0.000, sub-period CV=0.15, no lookahead bias (T-1 best)

Modes:
  --rebalance   First-trading-day-of-month check: compute signals, rebalance if needed, log NAV
  --eod         End-of-day mark-to-market: update NAV, log equity curve

State: /home/jupiter/Lvl3Quant/data/paper_engines/trend_cta/
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone, timedelta
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
STATE_DIR = Path("/home/jupiter/Lvl3Quant/data/paper_engines/trend_cta")
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "state.json"
TRADES_LOG = STATE_DIR / "trades.jsonl"
EQUITY_LOG = STATE_DIR / "equity_curve.jsonl"
SIGNAL_LOG = STATE_DIR / "signals.jsonl"

UNIVERSE = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]
CASH_ASSET = "SHY"
ALL_TICKERS = UNIVERSE + [CASH_ASSET]

LOOKBACK_DAYS = 126       # 6 months of trading days
TOP_N = 3                 # top 3 assets by momentum
INITIAL_CAPITAL = 100_000.0
COST_BPS = 10.0           # per leg
REBAL_TOLERANCE = 0.10    # 10% drift tolerance before forced rebalance

# ---------------------------------------------------------------------------
# STATE MANAGEMENT
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "cash": INITIAL_CAPITAL,
        "positions": {},      # {symbol: {"shares": float, "avg_price": float, "entry_date": str}}
        "nav": INITIAL_CAPITAL,
        "start_date": datetime.now(timezone.utc).isoformat(),
        "last_rebalance_date": None,
        "last_eod_date": None,
        "total_trades": 0,
        "realized_pnl": 0.0,
        "target_weights": {},  # current target allocation
        "current_selections": [],  # which assets are selected
        "version": "1.0",
    }


def save_state(state: dict):
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(record: dict):
    record["ts"] = datetime.now(timezone.utc).isoformat()
    with open(TRADES_LOG, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def log_equity(nav: float, date_str: str, positions: dict, cash: float):
    rec = {
        "date": date_str,
        "nav": round(nav, 2),
        "cash": round(cash, 2),
        "n_positions": len(positions),
        "positions": {s: round(p["shares"] * _last_prices.get(s, p["avg_price"]), 2)
                      for s, p in positions.items()} if positions else {},
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(EQUITY_LOG, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def log_signal(date_str: str, momentum_returns: dict, selected: list, target_weights: dict):
    rec = {
        "date": date_str,
        "momentum_returns": {k: round(v, 6) for k, v in momentum_returns.items()},
        "passed_abs_filter": [k for k, v in momentum_returns.items() if v > 0],
        "selected_top3": selected,
        "target_weights": target_weights,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(SIGNAL_LOG, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# Global price cache (set during run)
_last_prices: dict = {}

# ---------------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------------

def download_prices(lookback_buffer: int = 160) -> pd.DataFrame:
    """Download recent close prices for momentum calculation.

    We download extra days as buffer for holidays/weekends to ensure we have
    at least LOOKBACK_DAYS trading days.
    """
    data = {}
    for sym in ALL_TICKERS:
        try:
            df = yf.download(sym, period=f"{lookback_buffer}d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0:
                data[sym] = df["Close"].squeeze()
                print(f"  {sym}: {len(df)} rows")
            else:
                print(f"  WARNING: No data for {sym}")
        except Exception as e:
            print(f"  WARNING: Failed to download {sym}: {e}")
    return pd.DataFrame(data)


def get_current_prices(symbols: list[str]) -> dict:
    """Get latest prices for all symbols."""
    prices = {}
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            info = t.fast_info
            p = info.get("lastPrice", None) or info.get("previousClose", None)
            if p and p > 0:
                prices[sym] = float(p)
                continue
        except Exception:
            pass
        # Fallback to download
        try:
            df = yf.download(sym, period="5d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0:
                prices[sym] = float(df["Close"].iloc[-1])
        except Exception:
            print(f"  WARNING: Cannot get price for {sym}")
    return prices


# ---------------------------------------------------------------------------
# STRATEGY LOGIC
# ---------------------------------------------------------------------------

def compute_momentum_signals(prices_df: pd.DataFrame) -> tuple[dict, list, dict]:
    """Compute 6-month momentum, apply absolute + relative filters.

    Uses T-1 close (second-to-last row) to avoid any look-ahead.

    Returns:
        momentum_returns: {symbol: 6m return}
        selected: list of top-N symbols passing absolute momentum
        target_weights: {symbol: weight}
    """
    # T-1 close = second-to-last available close
    if len(prices_df) < LOOKBACK_DAYS + 2:
        print(f"  WARNING: Only {len(prices_df)} rows, need {LOOKBACK_DAYS + 2}")
        return {}, [], {CASH_ASSET: 1.0}

    # T-1 index (skip most recent day to use T-1)
    t1_idx = -2
    t1_lookback_idx = t1_idx - LOOKBACK_DAYS

    momentum_returns = {}
    for sym in UNIVERSE:
        if sym not in prices_df.columns:
            continue
        series = prices_df[sym].dropna()
        if len(series) < LOOKBACK_DAYS + 2:
            momentum_returns[sym] = -999  # not enough data
            continue

        p_now = series.iloc[t1_idx]
        p_past = series.iloc[t1_lookback_idx]
        if p_past > 0:
            momentum_returns[sym] = (p_now / p_past) - 1.0
        else:
            momentum_returns[sym] = -999

    # Absolute momentum filter: only positive 6m returns
    passed = {s: r for s, r in momentum_returns.items() if r > 0}

    # Relative momentum: rank by 6m return, select top N
    ranked = sorted(passed.items(), key=lambda x: x[1], reverse=True)
    selected = [s for s, _ in ranked[:TOP_N]]

    # Equal weight allocation
    if len(selected) > 0:
        w = 1.0 / TOP_N
        target_weights = {s: w for s in selected}
        allocated = sum(target_weights.values())
        if allocated < 1.0 - 1e-9:
            target_weights[CASH_ASSET] = 1.0 - allocated
    else:
        target_weights = {CASH_ASSET: 1.0}

    return momentum_returns, selected, target_weights


def compute_nav(state: dict, prices: dict) -> float:
    """Current NAV = cash + sum(shares * price)."""
    nav = state["cash"]
    for sym, pos in state["positions"].items():
        price = prices.get(sym, pos["avg_price"])
        nav += pos["shares"] * price
    return nav


def is_first_trading_day_of_month(today: datetime) -> bool:
    """Check if today is the first trading day of the month.

    Simple heuristic: it's the first weekday of the month, or it's a weekday
    early in the month and we haven't rebalanced this month yet.
    """
    if today.weekday() >= 5:  # weekend
        return False
    if today.day <= 3:
        return True
    return False


def needs_monthly_rebalance(state: dict, today: datetime) -> bool:
    """Check if we need to rebalance this month."""
    last_rebal = state.get("last_rebalance_date")
    if last_rebal is None:
        return True  # never rebalanced

    try:
        last_dt = datetime.fromisoformat(last_rebal.replace("Z", "+00:00"))
        # Already rebalanced this month?
        if last_dt.year == today.year and last_dt.month == today.month:
            return False
    except (ValueError, AttributeError):
        pass

    return is_first_trading_day_of_month(today)


# ---------------------------------------------------------------------------
# REBALANCE
# ---------------------------------------------------------------------------

def execute_rebalance(state: dict, target_weights: dict, current_prices: dict,
                      selected: list, momentum_returns: dict):
    """Execute trades to move from current positions to target weights."""
    now_str = datetime.now(timezone.utc).isoformat()
    nav = compute_nav(state, current_prices)

    print(f"\n  TARGET ALLOCATION (NAV=${nav:,.2f}):")
    for sym, w in sorted(target_weights.items(), key=lambda x: -x[1]):
        dollar_target = nav * w
        print(f"    {sym}: {w:.1%} (${dollar_target:,.0f})")

    old_weights = {}
    for sym, pos in state["positions"].items():
        price = current_prices.get(sym, pos["avg_price"])
        old_weights[sym] = (pos["shares"] * price) / nav if nav > 0 else 0

    # --- SELL positions no longer in target (or reduced) ---
    sells_first = []
    buys_second = []

    for sym in list(state["positions"].keys()):
        pos = state["positions"][sym]
        price = current_prices.get(sym, pos["avg_price"])
        current_value = pos["shares"] * price
        target_value = nav * target_weights.get(sym, 0)

        if target_weights.get(sym, 0) == 0:
            # Full exit
            proceeds = current_value
            cost = proceeds * COST_BPS / 10000
            pnl = (price - pos["avg_price"]) * pos["shares"]
            sells_first.append((sym, pos["shares"], price, proceeds, cost, pnl, "EXIT"))
        elif abs(current_value - target_value) / max(nav, 1) > REBAL_TOLERANCE:
            if current_value > target_value:
                # Trim
                trim_value = current_value - target_value
                trim_shares = trim_value / price
                cost = trim_value * COST_BPS / 10000
                pnl = (price - pos["avg_price"]) * trim_shares
                sells_first.append((sym, trim_shares, price, trim_value, cost, pnl, "TRIM"))

    # Execute sells
    for sym, shares, price, proceeds, cost, pnl, action in sells_first:
        if action == "EXIT":
            state["cash"] += proceeds - cost
            state["realized_pnl"] += pnl
            del state["positions"][sym]
        else:  # TRIM
            state["cash"] += proceeds - cost
            state["realized_pnl"] += pnl
            state["positions"][sym]["shares"] -= shares

        state["total_trades"] += 1
        log_trade({
            "action": action, "symbol": sym, "shares": round(shares, 4),
            "price": round(price, 4), "proceeds": round(proceeds, 2),
            "cost": round(cost, 2), "pnl": round(pnl, 2), "date": now_str,
        })
        print(f"    {action} {sym}: {shares:.2f} sh @ ${price:.2f} (PnL: ${pnl:.2f})")

    # --- BUY new positions or add to existing ---
    nav_after_sells = compute_nav(state, current_prices)

    for sym, w in target_weights.items():
        if sym == CASH_ASSET and sym not in state["positions"]:
            # SHY allocation — buy SHY shares
            target_value = nav_after_sells * w
            if target_value < 100:
                continue
        else:
            target_value = nav_after_sells * w

        price = current_prices.get(sym)
        if not price or price <= 0:
            continue

        current_value = 0
        if sym in state["positions"]:
            current_value = state["positions"][sym]["shares"] * price

        diff_value = target_value - current_value
        if diff_value < 50:  # minimum $50 to bother
            continue

        buy_shares = diff_value / price
        cost = diff_value * COST_BPS / 10000
        total_cost = diff_value + cost

        if total_cost > state["cash"]:
            # Scale down to available cash
            available = state["cash"] * 0.98  # keep 2% buffer
            if available < 50:
                continue
            buy_shares = available / price
            diff_value = buy_shares * price
            cost = diff_value * COST_BPS / 10000
            total_cost = diff_value + cost

        state["cash"] -= total_cost

        if sym in state["positions"]:
            # Average up
            old = state["positions"][sym]
            old_value = old["shares"] * old["avg_price"]
            new_value = buy_shares * price
            total_shares = old["shares"] + buy_shares
            state["positions"][sym]["shares"] = total_shares
            state["positions"][sym]["avg_price"] = (old_value + new_value) / total_shares
        else:
            state["positions"][sym] = {
                "shares": buy_shares,
                "avg_price": price,
                "entry_date": now_str,
            }

        state["total_trades"] += 1
        log_trade({
            "action": "BUY", "symbol": sym, "shares": round(buy_shares, 4),
            "price": round(price, 4), "cost": round(cost, 2), "date": now_str,
        })
        print(f"    BUY {sym}: {buy_shares:.2f} sh @ ${price:.2f}")

    state["last_rebalance_date"] = now_str
    state["target_weights"] = {k: round(v, 4) for k, v in target_weights.items()}
    state["current_selections"] = selected


# ---------------------------------------------------------------------------
# MAIN MODES
# ---------------------------------------------------------------------------

def run_rebalance_check():
    """Monthly rebalance check — runs on first trading day of each month at 9:45 AM ET."""
    global _last_prices

    print("=" * 65)
    print("TREND CTA MOMENTUM PAPER ENGINE — REBALANCE CHECK")
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 65)

    state = load_state()
    today = datetime.now()

    # Download historical prices for momentum calc
    print("\nDownloading price history...")
    prices_df = download_prices(lookback_buffer=200)

    if prices_df.empty or len(prices_df) < LOOKBACK_DAYS + 5:
        print("ERROR: Insufficient price data. Aborting.")
        save_state(state)
        return

    # Get current prices for execution
    print("\nGetting current prices...")
    current_prices = get_current_prices(ALL_TICKERS)
    _last_prices = current_prices

    if len(current_prices) < len(UNIVERSE) // 2:
        print("ERROR: Too few prices available. Market may be closed.")
        save_state(state)
        return

    # Compute signals
    momentum_returns, selected, target_weights = compute_momentum_signals(prices_df)

    print(f"\n  6-MONTH MOMENTUM (T-1 close):")
    for sym in sorted(momentum_returns.keys(), key=lambda s: momentum_returns[s], reverse=True):
        ret = momentum_returns[sym]
        status = "PASS" if ret > 0 else "FAIL"
        sel = " << SELECTED" if sym in selected else ""
        print(f"    {sym:5s}: {ret:+.2%}  [{status}]{sel}")

    n_passed = sum(1 for v in momentum_returns.values() if v > 0)
    print(f"\n  Filter: {n_passed}/{len(UNIVERSE)} pass absolute momentum")
    print(f"  Selected: {selected if selected else ['ALL IN SHY']}")

    # Log signal
    log_signal(today.strftime("%Y-%m-%d"), momentum_returns, selected, target_weights)

    # Check if rebalance needed
    if needs_monthly_rebalance(state, today):
        print(f"\n  REBALANCING — first trading day of {today.strftime('%B %Y')}")
        execute_rebalance(state, target_weights, current_prices, selected, momentum_returns)
    else:
        print(f"\n  No rebalance needed (last: {state.get('last_rebalance_date', 'never')})")

    # Mark to market
    nav = compute_nav(state, current_prices)
    state["nav"] = round(nav, 2)
    pnl_pct = (nav / INITIAL_CAPITAL - 1) * 100

    _print_summary(state, nav, pnl_pct, current_prices)

    log_equity(nav, today.strftime("%Y-%m-%d"), state["positions"], state["cash"])
    save_state(state)
    print("\nState saved.")


def run_eod_snapshot():
    """End-of-day NAV update — runs daily at 4:05 PM ET."""
    global _last_prices

    print("=" * 65)
    print("TREND CTA MOMENTUM PAPER ENGINE — EOD SNAPSHOT")
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 65)

    state = load_state()
    today = datetime.now()

    # Skip weekends
    if today.weekday() >= 5:
        print("Weekend — skipping.")
        return

    # Skip if already logged today
    last_eod = state.get("last_eod_date")
    today_str = today.strftime("%Y-%m-%d")
    if last_eod == today_str:
        print(f"Already logged EOD for {today_str}. Skipping.")
        return

    print("\nGetting closing prices...")
    current_prices = get_current_prices(ALL_TICKERS)
    _last_prices = current_prices

    if len(current_prices) < len(UNIVERSE) // 2:
        print("WARNING: Too few prices. Market may be closed. Using last known.")

    nav = compute_nav(state, current_prices)
    state["nav"] = round(nav, 2)
    state["last_eod_date"] = today_str
    pnl_pct = (nav / INITIAL_CAPITAL - 1) * 100

    _print_summary(state, nav, pnl_pct, current_prices)

    log_equity(nav, today_str, state["positions"], state["cash"])
    save_state(state)
    print("\nEOD snapshot saved.")


def _print_summary(state: dict, nav: float, pnl_pct: float, current_prices: dict):
    """Print portfolio summary."""
    print(f"\n{'=' * 65}")
    print(f"  NAV: ${nav:,.2f} ({pnl_pct:+.2f}% from start)")
    print(f"  Cash: ${state['cash']:,.2f} ({state['cash']/nav*100:.1f}%)")
    print(f"  Positions: {len(state['positions'])}")

    if state["positions"]:
        print(f"\n  HOLDINGS:")
        for sym, pos in sorted(state["positions"].items()):
            price = current_prices.get(sym, pos["avg_price"])
            mkt_val = pos["shares"] * price
            unrealized = (price - pos["avg_price"]) * pos["shares"]
            weight = mkt_val / nav * 100 if nav > 0 else 0
            print(f"    {sym:5s}: {pos['shares']:8.2f} sh @ ${price:.2f}"
                  f"  (${mkt_val:,.0f}, {weight:.1f}%, PnL: ${unrealized:+,.0f})")

    print(f"\n  Target: {state.get('current_selections', [])}")
    print(f"  Total trades: {state['total_trades']}")
    print(f"  Realized PnL: ${state['realized_pnl']:,.2f}")
    print(f"  Last rebalance: {state.get('last_rebalance_date', 'never')}")
    print(f"{'=' * 65}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trend CTA Momentum Paper Engine")
    parser.add_argument("--rebalance", action="store_true",
                        help="Run monthly rebalance check (first trading day)")
    parser.add_argument("--eod", action="store_true",
                        help="Run end-of-day NAV snapshot")
    args = parser.parse_args()

    if args.rebalance:
        run_rebalance_check()
    elif args.eod:
        run_eod_snapshot()
    else:
        # Default: rebalance check (backward compatible)
        run_rebalance_check()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
