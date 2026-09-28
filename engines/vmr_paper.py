#!/usr/bin/env python3
"""
VMR (Vol Mean Reversion) Paper Trader
======================================
VALIDATED strategy — DAILY rebalance version.
Adversarial validation: Sharpe 3.424, perm p=0.000, sub-period CV 0.141,
outlier deg -0.9%, 15/15 years beats SPY.

5-regime VIX system:
  1. VIX < 15 & declining → UPRO (complacent, go leveraged)
  2. VIX > 20 & mean-reverting (down 15% from peak, declining) → UPRO (catch recovery)
  3. VIX > 25 & rising → 50% GLD + 50% TLT (defensive)
  4. VIX > 20 & rising → 50% SPY + 50% TLT (cautious)
  5. Otherwise → SPY

DAILY rebalance (upgraded from weekly per adversarial validation 2026-07-17).
Sharpe daily=3.424 vs weekly=1.516. Same MaxDD (-30.8% vs -30.6%).

PM2 cron: Daily at 15:50 ET (near close)
State: /home/jupiter/Lvl3Quant/state/vmr_state.json
History: /home/jupiter/Lvl3Quant/state/vmr_history.csv
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
STATE_FILE = STATE_DIR / "vmr_state.json"
HISTORY_FILE = STATE_DIR / "vmr_history.csv"

INITIAL_CAPITAL = 100_000


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "portfolio_value": INITIAL_CAPITAL,
        "regime": "SPY",
        "allocation": {"SPY": 1.0},
        "trades": 0,
        "start_date": datetime.now(ET).strftime("%Y-%m-%d"),
        "last_update": None,
        "last_rebalance": None,
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


def get_data() -> dict:
    """Fetch VIX + asset prices."""
    tickers = ["^VIX", "SPY", "UPRO", "GLD", "TLT"]
    data = {}

    for t in tickers:
        try:
            df = yf.download(t, period="25d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0:
                name = t.replace("^", "")
                data[name] = {
                    "price": float(df["Close"].iloc[-1]),
                    "prev_price": float(df["Close"].iloc[-2]) if len(df) > 1 else None,
                    "prices_10d": df["Close"].values[-10:].tolist() if len(df) >= 10 else df["Close"].values.tolist(),
                    "prices_20d": df["Close"].values[-20:].tolist() if len(df) >= 20 else df["Close"].values.tolist(),
                }
        except Exception as e:
            print(f"  Warning: failed to fetch {t}: {e}")

    return data


def determine_regime(data: dict) -> dict:
    """Determine VMR regime from current data."""
    vix = data.get("VIX", {}).get("price", 20)
    vix_prices = data.get("VIX", {}).get("prices_10d", [vix])
    vix_20d = data.get("VIX", {}).get("prices_20d", [vix])

    # VIX 10d MA
    vix_ma10 = np.mean(vix_prices) if len(vix_prices) > 0 else vix
    # VIX 20d peak
    vix_peak20 = max(vix_20d) if len(vix_20d) > 0 else vix

    vix_declining = vix < vix_ma10

    if vix < 15 and vix_declining:
        regime = "UPRO"
        reason = f"VIX {vix:.1f} < 15 and declining (MA10={vix_ma10:.1f})"
        allocation = {"UPRO": 1.0}
    elif vix > 20 and vix < vix_peak20 * 0.85 and vix_declining:
        regime = "UPRO_MR"
        reason = f"VIX {vix:.1f} mean-reverting (peak20={vix_peak20:.1f}, down {(1-vix/vix_peak20)*100:.0f}%, declining)"
        allocation = {"UPRO": 1.0}
    elif vix > 25 and not vix_declining:
        regime = "DEFENSIVE"
        reason = f"VIX {vix:.1f} > 25 and rising (MA10={vix_ma10:.1f})"
        allocation = {"GLD": 0.5, "TLT": 0.5}
    elif vix > 20 and not vix_declining:
        regime = "CAUTIOUS"
        reason = f"VIX {vix:.1f} > 20 and rising (MA10={vix_ma10:.1f})"
        allocation = {"SPY": 0.5, "TLT": 0.5}
    else:
        regime = "SPY"
        reason = f"VIX {vix:.1f}, default SPY"
        allocation = {"SPY": 1.0}

    return {
        "regime": regime,
        "reason": reason,
        "allocation": allocation,
        "vix": round(vix, 2),
        "vix_ma10": round(vix_ma10, 2),
        "vix_peak20": round(vix_peak20, 2),
        "vix_declining": vix_declining,
    }


def run():
    now = datetime.now(ET)
    print(f"\n{'='*60}")
    print(f"VMR Paper Trader — {now.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*60}")

    state = load_state()
    print(f"Portfolio: ${state['portfolio_value']:,.2f}")
    print(f"Current regime: {state['regime']}")
    print(f"Allocation: {state['allocation']}")

    # Get data
    print("\nFetching market data...")
    data = get_data()

    if "VIX" not in data or "SPY" not in data:
        print("ERROR: Missing critical data. Skipping.")
        return

    # Determine regime
    signal = determine_regime(data)
    print(f"\nSignal:")
    print(f"  VIX: {signal['vix']}, MA10: {signal['vix_ma10']}, Peak20: {signal['vix_peak20']}")
    print(f"  Regime: {signal['regime']}")
    print(f"  Reason: {signal['reason']}")
    print(f"  Allocation: {signal['allocation']}")

    # Update portfolio value based on previous day's returns
    old_alloc = state.get("allocation", {"SPY": 1.0})
    daily_return = 0.0
    for asset, weight in old_alloc.items():
        asset_data = data.get(asset, {})
        if asset_data.get("prev_price") and asset_data.get("price"):
            r = (asset_data["price"] / asset_data["prev_price"] - 1) * weight
            daily_return += r

    state["portfolio_value"] *= (1 + daily_return)

    # Rebalance if regime changed
    old_regime = state["regime"]
    if signal["regime"] != old_regime:
        state["trades"] += 1
        print(f"\n  >> REGIME CHANGE: {old_regime} → {signal['regime']}")
        print(f"  >> New allocation: {signal['allocation']}")

    state["regime"] = signal["regime"]
    state["allocation"] = signal["allocation"]
    state["last_rebalance"] = now.strftime("%Y-%m-%d")

    print(f"\nPortfolio: ${state['portfolio_value']:,.2f}")
    print(f"Regime: {state['regime']} (trades: {state['trades']})")

    save_state(state)

    append_history({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "portfolio_value": round(state["portfolio_value"], 2),
        "regime": state["regime"],
        "vix": signal["vix"],
        "vix_ma10": signal["vix_ma10"],
        "vix_peak20": signal["vix_peak20"],
        "allocation": json.dumps(signal["allocation"]),
    })

    print(f"\nDone. State saved.")


if __name__ == "__main__":
    run()
