#!/usr/bin/env python3
"""
Vol Harvesting Paper Trader — Term Structure Strategy
=====================================================
Best config from vol_harvesting.py: TS_th0.9_sm1_cap30_cash_VT15

Logic:
  - Compute VIX/VIX3M ratio (term structure)
  - If ratio < 0.9 (strong contango): hold SVXY
  - If ratio >= 0.9 OR VIX > 30: hold cash (money market)
  - Apply vol targeting: scale position to target 15% annualized vol

Performance (backtest):
  - Sharpe 4.13, CAGR 97.2%, MaxDD -6.3%
  - Survived Volmageddon (Feb 2018) with 0% drawdown
  - Permutation p=0.000, WF OOS Sharpe 4.12
  - R1 fails (regime gap 1.23) but MaxDD qualifies under HC #709

PM2 cron: daily at 15:55 ET (near close, before EOD rebalance)
State: /home/jupiter/Lvl3Quant/state/vol_harvest_state.json
History: /home/jupiter/Lvl3Quant/state/vol_harvest_history.csv
"""
import json
import sys
import warnings
from datetime import datetime
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
STATE_FILE = STATE_DIR / "vol_harvest_state.json"
HISTORY_FILE = STATE_DIR / "vol_harvest_history.csv"

# ── Strategy parameters (best WF config) ──
CONTANGO_THRESHOLD = 0.90   # VIX/VIX3M ratio < this = contango
VIX_CAP = 30                # Force cash if VIX > this
VOL_TARGET = 0.15           # Target 15% annualized vol
MAX_LEVERAGE = 1.5          # Max position scaling
INITIAL_CAPITAL = 100_000


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "portfolio_value": INITIAL_CAPITAL,
        "position": "CASH",
        "position_size": 0,  # fraction of portfolio (0-1.5)
        "entry_date": None,
        "entry_price": None,
        "trades": 0,
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


def get_market_data() -> dict:
    """Fetch current VIX, VIX3M, SVXY data."""
    tickers = ["^VIX", "^VIX3M", "SVXY", "SPY"]
    data = {}

    for t in tickers:
        try:
            df = yf.download(t, period="65d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0:
                name = t.replace("^", "")
                data[name] = {
                    "price": float(df["Close"].iloc[-1]),
                    "prev_price": float(df["Close"].iloc[-2]) if len(df) > 1 else None,
                    "prices": df["Close"].values.tolist(),
                }
        except Exception as e:
            print(f"  Warning: failed to fetch {t}: {e}")

    return data


def compute_signal(data: dict) -> dict:
    """Compute term structure signal and vol scaling."""
    vix = data.get("VIX", {}).get("price", 20)
    vix3m = data.get("VIX3M", {}).get("price", 20)
    svxy_price = data.get("SVXY", {}).get("price", 0)

    # Term structure ratio
    ratio = vix / vix3m if vix3m > 0 else 1.0

    # Signal
    in_contango = ratio < CONTANGO_THRESHOLD
    vix_safe = vix <= VIX_CAP
    go_long = in_contango and vix_safe

    # Vol targeting: scale position based on recent SVXY volatility
    svxy_prices = data.get("SVXY", {}).get("prices", [])
    if len(svxy_prices) >= 21:
        rets = np.diff(np.log(svxy_prices[-22:]))  # last 21 daily returns
        realized_vol = np.std(rets) * np.sqrt(252)
        position_size = min(VOL_TARGET / realized_vol, MAX_LEVERAGE) if realized_vol > 0 else 1.0
    else:
        position_size = 1.0

    return {
        "vix": round(vix, 2),
        "vix3m": round(vix3m, 2),
        "ratio": round(ratio, 4),
        "in_contango": in_contango,
        "vix_safe": vix_safe,
        "go_long": go_long,
        "svxy_price": round(svxy_price, 2),
        "realized_vol": round(realized_vol * 100, 1) if 'realized_vol' in dir() else None,
        "position_size": round(position_size, 3),
        "reason": (
            f"LONG SVXY (contango, ratio {ratio:.3f} < {CONTANGO_THRESHOLD})" if go_long
            else f"CASH (backwardation, ratio {ratio:.3f} >= {CONTANGO_THRESHOLD})" if not in_contango
            else f"CASH (VIX {vix:.1f} > cap {VIX_CAP})"
        ),
    }


def run():
    now = datetime.now(ET)
    print(f"\n{'='*60}")
    print(f"Vol Harvest Paper Trader — {now.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*60}")

    state = load_state()
    print(f"Portfolio: ${state['portfolio_value']:,.2f}")
    print(f"Current position: {state['position']} (size: {state['position_size']:.1%})")

    # Get market data
    print("\nFetching market data...")
    data = get_market_data()

    if "SVXY" not in data or "VIX" not in data:
        print("ERROR: Missing critical data (VIX or SVXY). Skipping.")
        return

    # Compute signal
    signal = compute_signal(data)
    print(f"\nSignal:")
    print(f"  VIX: {signal['vix']}, VIX3M: {signal['vix3m']}")
    print(f"  Ratio: {signal['ratio']} (threshold: {CONTANGO_THRESHOLD})")
    print(f"  Contango: {signal['in_contango']}, VIX safe: {signal['vix_safe']}")
    print(f"  Decision: {signal['reason']}")
    if signal.get('realized_vol'):
        print(f"  SVXY realized vol: {signal['realized_vol']}%")
    print(f"  Position size: {signal['position_size']:.1%}")

    # Update portfolio value based on previous day's return
    if state["position"] == "SVXY" and state.get("entry_price"):
        svxy_prev = data["SVXY"].get("prev_price", state["entry_price"])
        svxy_now = data["SVXY"]["price"]
        if svxy_prev and svxy_prev > 0:
            daily_return = (svxy_now / svxy_prev - 1) * state["position_size"]
            state["portfolio_value"] *= (1 + daily_return)

    # Execute signal
    old_position = state["position"]
    if signal["go_long"]:
        state["position"] = "SVXY"
        state["position_size"] = signal["position_size"]
        if old_position != "SVXY":
            state["entry_date"] = now.strftime("%Y-%m-%d")
            state["entry_price"] = signal["svxy_price"]
            state["trades"] += 1
            print(f"\n  >> ENTERING SVXY at ${signal['svxy_price']:.2f} (size: {signal['position_size']:.1%})")
    else:
        state["position"] = "CASH"
        state["position_size"] = 0
        if old_position == "SVXY":
            state["trades"] += 1
            print(f"\n  >> EXITING SVXY at ${signal['svxy_price']:.2f}")
        state["entry_date"] = None
        state["entry_price"] = None

    print(f"\nPortfolio: ${state['portfolio_value']:,.2f}")
    print(f"Position: {state['position']} (trades: {state['trades']})")

    # Save state
    save_state(state)

    # Append to history
    append_history({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "portfolio_value": round(state["portfolio_value"], 2),
        "position": state["position"],
        "position_size": state["position_size"],
        "vix": signal["vix"],
        "vix3m": signal["vix3m"],
        "ratio": signal["ratio"],
        "contango": signal["in_contango"],
        "svxy_price": signal["svxy_price"],
    })

    print(f"\nDone. State saved.")


if __name__ == "__main__":
    run()
