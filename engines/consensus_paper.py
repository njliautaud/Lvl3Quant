#!/usr/bin/env python3
"""
GP3+VMR Consensus Paper Trader
===============================
VALIDATED strategy (permutation p=0.000, Sharpe 2.209).

When BOTH Gameplan v3 AND VMR Daily agree on UPRO → hold UPRO.
Otherwise → hold SPY.

This is the most conservative of our 3 validated strategies:
  - Only in UPRO 22% of the time
  - Requires dual confirmation (vol regime + momentum confluence)
  - Sharpe 2.21, CAGR 52%, MaxDD -33.7%

PM2 cron: Daily at 15:55 ET (after VMR and GP3 trackers run)
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
STATE_FILE = STATE_DIR / "consensus_state.json"
HISTORY_FILE = STATE_DIR / "consensus_history.csv"

INITIAL_CAPITAL = 100_000


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "portfolio_value": INITIAL_CAPITAL,
        "position": "SPY",
        "gp3_says_upro": False,
        "vmr_says_upro": False,
        "consensus": False,
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


def compute_rsi(series, window=10):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def get_signals() -> dict:
    """Compute GP3 and VMR signals from current market data."""
    tickers = ["SPY", "UPRO", "^VIX"]
    raw = yf.download(tickers, period="250d", progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw
    prices = prices.rename(columns={"^VIX": "VIX"})
    prices = prices.ffill().dropna(how="all")

    if len(prices) < 200:
        print(f"  WARNING: Only {len(prices)} days of data, need 200+")
        return {"gp3_upro": False, "vmr_upro": False, "details": {}}

    spy = prices["SPY"]
    vix = prices["VIX"]
    spy_ret = spy.pct_change()

    # === VMR Daily Signal ===
    vix_ma10 = vix.rolling(10).mean()
    vix_peak20 = vix.rolling(20).max()

    v = vix.iloc[-1]
    vm10 = vix_ma10.iloc[-1]
    vp20 = vix_peak20.iloc[-1]

    vmr_upro = False
    vmr_regime = "SPY"
    if not (pd.isna(v) or pd.isna(vm10)):
        if v < 15 and v < vm10:
            vmr_upro = True
            vmr_regime = "UPRO"
        elif v > 20 and v < vp20 * 0.85 and v < vm10:
            vmr_upro = True
            vmr_regime = "UPRO_MR"
        elif v > 25 and v > vm10:
            vmr_regime = "DEFENSIVE"
        elif v > 20 and v > vm10:
            vmr_regime = "CAUTIOUS"

    # === GP3 Signal ===
    spy_5d_mom = spy.pct_change(5).iloc[-1]
    spy_rsi10 = compute_rsi(spy, 10).iloc[-1]
    spy_ma200_slope = spy.rolling(200).mean().pct_change(10).iloc[-1]
    ann_vol = spy_ret.iloc[-63:].std() * np.sqrt(252) * 100

    score = 0
    if not pd.isna(spy_5d_mom) and spy_5d_mom > 0: score += 1
    if not pd.isna(spy_rsi10) and spy_rsi10 > 50: score += 1
    if not pd.isna(spy_ma200_slope) and spy_ma200_slope > 0: score += 1

    gp3_upro = False
    if v <= 30 and ann_vol <= 15 and score >= 3:  # Need all 3 for entry (score >= 2.5 rounds to 3)
        gp3_upro = True

    details = {
        "vix": round(float(v), 2),
        "vix_ma10": round(float(vm10), 2),
        "vix_peak20": round(float(vp20), 2),
        "vmr_regime": vmr_regime,
        "ann_vol": round(float(ann_vol), 2),
        "confluence_score": score,
        "spy_5d_mom": round(float(spy_5d_mom) * 100, 2) if not pd.isna(spy_5d_mom) else None,
        "spy_rsi10": round(float(spy_rsi10), 2) if not pd.isna(spy_rsi10) else None,
        "spy_price": round(float(spy.iloc[-1]), 2),
        "upro_price": round(float(prices["UPRO"].iloc[-1]), 2),
    }

    return {"gp3_upro": gp3_upro, "vmr_upro": vmr_upro, "details": details}


def run():
    now = datetime.now(ET)
    print(f"\n{'='*60}")
    print(f"Consensus Paper Trader — {now.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*60}")

    state = load_state()
    print(f"Portfolio: ${state['portfolio_value']:,.2f}")
    print(f"Position: {state['position']}")

    # Get signals
    print("\nFetching signals...")
    signals = get_signals()
    details = signals["details"]

    gp3 = signals["gp3_upro"]
    vmr = signals["vmr_upro"]
    consensus = gp3 and vmr

    print(f"\n  GP3 says UPRO: {gp3} (confluence={details.get('confluence_score')}, vol={details.get('ann_vol')}%)")
    print(f"  VMR says UPRO: {vmr} (regime={details.get('vmr_regime')}, VIX={details.get('vix')})")
    print(f"  CONSENSUS: {'UPRO ✅' if consensus else 'SPY'}")

    # Update portfolio value
    try:
        spy_data = yf.download("SPY", period="3d", progress=False, auto_adjust=True)
        upro_data = yf.download("UPRO", period="3d", progress=False, auto_adjust=True)
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = spy_data.columns.get_level_values(0)
        if isinstance(upro_data.columns, pd.MultiIndex):
            upro_data.columns = upro_data.columns.get_level_values(0)

        if state["position"] == "UPRO" and len(upro_data) >= 2:
            daily_ret = upro_data["Close"].iloc[-1] / upro_data["Close"].iloc[-2] - 1
        elif len(spy_data) >= 2:
            daily_ret = spy_data["Close"].iloc[-1] / spy_data["Close"].iloc[-2] - 1
        else:
            daily_ret = 0

        state["portfolio_value"] *= (1 + daily_ret)
    except Exception as e:
        print(f"  Warning: couldn't update portfolio value: {e}")

    # Check for position change
    old_pos = state["position"]
    new_pos = "UPRO" if consensus else "SPY"

    if new_pos != old_pos:
        state["trades"] += 1
        print(f"\n  >> POSITION CHANGE: {old_pos} → {new_pos}")

    state["position"] = new_pos
    state["gp3_says_upro"] = gp3
    state["vmr_says_upro"] = vmr
    state["consensus"] = consensus

    print(f"\nPortfolio: ${state['portfolio_value']:,.2f}")
    print(f"Position: {state['position']} (trades: {state['trades']})")

    save_state(state)

    append_history({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "portfolio_value": round(state["portfolio_value"], 2),
        "position": state["position"],
        "gp3_upro": gp3,
        "vmr_upro": vmr,
        "consensus": consensus,
        "vix": details.get("vix"),
        "confluence": details.get("confluence_score"),
        "vmr_regime": details.get("vmr_regime"),
    })

    print(f"\nDone. State saved.")


if __name__ == "__main__":
    run()
