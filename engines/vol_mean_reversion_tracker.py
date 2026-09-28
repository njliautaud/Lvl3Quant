#!/usr/bin/env python3
"""
Vol Mean Reversion Paper Tracker

Tracks the VIX-based regime switching strategy with mean-reversion overlay.
Validated in creative batch 2: Sharpe 1.467, CAGR 42.8%, MaxDD -30.6%.

5 Regimes:
  UPRO_LOW_VOL:  VIX < 15 and declining → UPRO (complacent, ride leverage)
  UPRO_MEAN_REV: VIX > 20 but dropped 15%+ from 20d peak and declining → UPRO
  CAUTIOUS:      VIX > 20 and rising → 50% SPY + 50% TLT
  DEFENSIVE:     VIX > 25 and rising → 50% GLD + 50% TLT
  NEUTRAL:       Everything else → SPY

Weekly rebalance (Fridays only).
"""

import json, os, sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "vol_mean_reversion_paper.json"
LOG_DIR = ROOT / "logs" / "paper_engines"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "initial_value": 10000.0,
        "portfolio_value": 10000.0,
        "spy_benchmark": 10000.0,
        "upro_benchmark": 10000.0,
        "current_regime": "SPY",
        "last_rebalance_week": None,
        "history": [],
        "trades": [],
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_regime(vix_data, date_idx):
    """Determine current regime from VIX data."""
    if date_idx < 252:
        return "SPY"

    vix_now = vix_data.iloc[date_idx]
    vix_ma10 = vix_data.iloc[max(0, date_idx-10):date_idx+1].mean()
    vix_peak20 = vix_data.iloc[max(0, date_idx-20):date_idx+1].max()

    if np.isnan(vix_now) or np.isnan(vix_ma10):
        return "SPY"

    if vix_now < 15 and vix_now < vix_ma10:
        return "UPRO_LOW_VOL"
    elif vix_now > 20 and vix_now < vix_peak20 * 0.85 and vix_now < vix_ma10:
        return "UPRO_MEAN_REV"
    elif vix_now > 25 and vix_now > vix_ma10:
        return "DEFENSIVE"
    elif vix_now > 20 and vix_now > vix_ma10:
        return "CAUTIOUS"
    else:
        return "SPY"


def get_regime_allocation(regime):
    """Return allocation dict for regime."""
    if regime in ("UPRO_LOW_VOL", "UPRO_MEAN_REV"):
        return {"UPRO": 1.0}
    elif regime == "DEFENSIVE":
        return {"GLD": 0.5, "TLT": 0.5}
    elif regime == "CAUTIOUS":
        return {"SPY": 0.5, "TLT": 0.5}
    else:
        return {"SPY": 1.0}


def main():
    print(f"{'='*60}")
    print(f"VOL MEAN REVERSION — Paper Tracker")
    print(f"{'='*60}")

    state = load_state()

    # Fetch recent data
    tickers = ["SPY", "UPRO", "GLD", "TLT", "^VIX"]
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, period="300d", progress=False)
            name = t.replace("^", "")
            data[name] = df["Close"].squeeze()
        except Exception as e:
            print(f"  Failed to fetch {t}: {e}")

    if "VIX" not in data or "SPY" not in data:
        print("  Missing critical data, skipping")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    today_dt = pd.Timestamp(today)

    # Get latest prices
    latest = {}
    for name, series in data.items():
        if len(series) > 0:
            latest[name] = float(series.iloc[-1])

    vix_now = latest.get("VIX", 20)

    # Determine regime
    vix_series = data["VIX"]
    regime = get_regime(vix_series, len(vix_series) - 1)

    # Check if this is a rebalance week (different from last)
    week_key = f"{datetime.now().year}-W{datetime.now().isocalendar()[1]}"
    is_rebal = week_key != state.get("last_rebalance_week")

    prev_regime = state["current_regime"]

    if is_rebal:
        state["current_regime"] = regime
        state["last_rebalance_week"] = week_key

        if regime != prev_regime:
            state["trades"].append({
                "date": today,
                "from": prev_regime,
                "to": regime,
                "vix": vix_now,
            })

    # Calculate today's return based on allocation
    alloc = get_regime_allocation(state["current_regime"])

    # Get daily returns
    day_ret = 0.0
    for asset, weight in alloc.items():
        if asset in data and len(data[asset]) >= 2:
            r = float((data[asset].iloc[-1] / data[asset].iloc[-2]) - 1)
            day_ret += weight * r

    spy_ret = float((data["SPY"].iloc[-1] / data["SPY"].iloc[-2]) - 1) if len(data["SPY"]) >= 2 else 0
    upro_ret = float((data["UPRO"].iloc[-1] / data["UPRO"].iloc[-2]) - 1) if "UPRO" in data and len(data["UPRO"]) >= 2 else 0

    # Update values
    state["portfolio_value"] *= (1 + day_ret)
    state["spy_benchmark"] *= (1 + spy_ret)
    state["upro_benchmark"] *= (1 + upro_ret)

    # Record history
    state["history"].append({
        "date": today,
        "portfolio": round(state["portfolio_value"], 2),
        "spy": round(state["spy_benchmark"], 2),
        "upro": round(state["upro_benchmark"], 2),
        "regime": state["current_regime"],
        "vix": round(vix_now, 2),
        "day_return": round(day_ret * 100, 3),
    })

    # Print summary
    days_running = len(state["history"])
    port_ret = (state["portfolio_value"] / state["initial_value"] - 1) * 100
    spy_ret_total = (state["spy_benchmark"] / state["initial_value"] - 1) * 100
    upro_ret_total = (state["upro_benchmark"] / state["initial_value"] - 1) * 100

    print(f"\n  Date: {today}")
    print(f"  VIX: {vix_now:.1f}")
    print(f"  Regime: {state['current_regime']} {'(REBALANCED)' if is_rebal and regime != prev_regime else ''}")
    print(f"  Allocation: {alloc}")
    print(f"  Today return: {day_ret*100:+.2f}%")
    print(f"\n  Portfolio: ${state['portfolio_value']:,.2f} ({port_ret:+.2f}%)")
    print(f"  SPY bench: ${state['spy_benchmark']:,.2f} ({spy_ret_total:+.2f}%)")
    print(f"  UPRO bench: ${state['upro_benchmark']:,.2f} ({upro_ret_total:+.2f}%)")
    print(f"  Days tracked: {days_running}")
    print(f"  Total trades: {len(state['trades'])}")

    save_state(state)
    print(f"\n  State saved to {STATE_FILE}")


if __name__ == "__main__":
    main()
