#!/usr/bin/env python3
"""
Portfolio Engine v2 — Validated 50/50 VMR + Consensus
=====================================================
Based on portfolio optimization results (2026-07-17):
  - OOS Sharpe 1.01, CAGR 21.6%, MaxDD -27.6%
  - Permutation p=0.004 (timing IS real)
  - 14/17 years positive

Two independent signal sources, equal weight:
  1. VMR Daily (50%): 5-regime VIX system
  2. GP3+VMR Consensus (50%): UPRO only when BOTH GP3 and VMR agree

Combined daily signal → target allocation:
  - Each half independently determines UPRO vs SPY (or GLD/TLT for VMR defensive)
  - Final portfolio = weighted blend of both halves

PM2 cron: Daily at 15:55 ET (near close, after VMR paper runs at 15:50)
State: state/portfolio_v2_state.json
History: state/portfolio_v2_history.csv
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
ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "portfolio_v2_state.json"
HISTORY_FILE = STATE_DIR / "portfolio_v2_history.csv"
ALERTS_FILE = STATE_DIR / "pending_discord_alerts.json"

INITIAL_CAPITAL = 100_000


# ─── Data fetching ───────────────────────────────────────────────────────────

def fetch_data() -> dict:
    """Fetch VIX + asset prices for signal generation."""
    tickers = ["^VIX", "SPY", "UPRO", "GLD", "TLT"]
    result = {}

    for t in tickers:
        try:
            df = yf.download(t, period="250d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0:
                name = t.replace("^", "")
                result[name] = {
                    "price": float(df["Close"].iloc[-1]),
                    "prev_price": float(df["Close"].iloc[-2]) if len(df) > 1 else None,
                    "close_series": df["Close"].values.tolist(),
                }
        except Exception as e:
            print(f"  Warning: failed to fetch {t}: {e}")

    return result


# ─── Signal: VMR Daily (50% weight) ─────────────────────────────────────────

def vmr_signal(data: dict) -> dict:
    """5-regime VIX system. Returns regime and allocation."""
    vix = data.get("VIX", {}).get("price", 20)
    vix_prices = data.get("VIX", {}).get("close_series", [vix])

    # 10d and 20d lookbacks
    vix_10d = vix_prices[-10:] if len(vix_prices) >= 10 else vix_prices
    vix_20d = vix_prices[-20:] if len(vix_prices) >= 20 else vix_prices

    vix_ma10 = np.mean(vix_10d)
    vix_peak20 = max(vix_20d)
    vix_declining = vix < vix_ma10

    if vix < 15 and vix_declining:
        regime = "UPRO"
        alloc = {"UPRO": 1.0}
        reason = f"VIX {vix:.1f} < 15, declining"
    elif vix > 20 and vix < vix_peak20 * 0.85 and vix_declining:
        regime = "UPRO_MR"
        alloc = {"UPRO": 1.0}
        reason = f"VIX {vix:.1f} mean-reverting (peak {vix_peak20:.1f})"
    elif vix > 25 and not vix_declining:
        regime = "DEFENSIVE"
        alloc = {"GLD": 0.5, "TLT": 0.5}
        reason = f"VIX {vix:.1f} > 25, rising"
    elif vix > 20 and not vix_declining:
        regime = "CAUTIOUS"
        alloc = {"SPY": 0.5, "TLT": 0.5}
        reason = f"VIX {vix:.1f} > 20, rising"
    else:
        regime = "SPY"
        alloc = {"SPY": 1.0}
        reason = f"VIX {vix:.1f}, default"

    return {"regime": regime, "allocation": alloc, "reason": reason,
            "vix": round(vix, 2), "vix_ma10": round(vix_ma10, 2)}


# ─── Signal: GP3 Confluence ─────────────────────────────────────────────────

def gp3_signal(data: dict) -> dict:
    """3-timeframe confluence gate. Returns confluence score and UPRO/SPY decision."""
    spy_prices = data.get("SPY", {}).get("close_series", [])
    if len(spy_prices) < 200:
        return {"confluence": 0, "signal": "SPY", "reason": "insufficient data"}

    spy = np.array(spy_prices)
    current = spy[-1]

    # Short-term: 5d momentum > 0 AND RSI(10) > 50
    mom_5d = current / spy[-6] - 1 if len(spy) >= 6 else 0
    # RSI(10) - Wilder's smoothing
    deltas = np.diff(spy[-12:])  # need 11 changes for RSI(10)
    gains = np.maximum(deltas, 0)
    losses = np.abs(np.minimum(deltas, 0))
    avg_gain = np.mean(gains[:10])
    avg_loss = np.mean(losses[:10])
    if len(deltas) > 10:
        avg_gain = (avg_gain * 9 + gains[-1]) / 10
        avg_loss = (avg_loss * 9 + losses[-1]) / 10
    rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100

    short_score = 1.0 if (mom_5d > 0 and rsi > 50) else 0.0

    # Medium-term: SPY > 50d SMA AND 21d vol < 15%
    sma50 = np.mean(spy[-50:])
    log_rets_21d = np.diff(np.log(spy[-22:]))
    vol_21d = np.std(log_rets_21d) * np.sqrt(252) if len(log_rets_21d) >= 20 else 0.20

    medium_score = 1.0 if (current > sma50 and vol_21d < 0.15) else 0.0

    # Long-term: SPY > 200d SMA AND 63d vol declining
    sma200 = np.mean(spy[-200:])
    if len(spy) >= 84:
        vol_63d_now = np.std(np.diff(np.log(spy[-64:]))) * np.sqrt(252)
        vol_63d_prev = np.std(np.diff(np.log(spy[-84:-21]))) * np.sqrt(252)
        vol_declining = vol_63d_now < vol_63d_prev
    else:
        vol_declining = False

    long_score = 1.0 if (current > sma200 and vol_declining) else 0.0

    confluence = short_score + medium_score + long_score

    # GP3 v3: UPRO if confluence >= 2.5 AND vol < 15%
    signal = "UPRO" if (confluence >= 2.5 and vol_21d < 0.15) else "SPY"

    return {
        "confluence": confluence,
        "signal": signal,
        "reason": f"conf={confluence:.1f}, vol={vol_21d:.1%}, RSI={rsi:.0f}, mom5d={mom_5d:.2%}",
        "vol_21d": round(vol_21d, 4),
        "rsi": round(rsi, 1),
    }


# ─── Signal: Consensus (BOTH agree → UPRO, else SPY) ────────────────────────

def consensus_signal(vmr: dict, gp3: dict) -> dict:
    """UPRO only when BOTH VMR and GP3 say UPRO."""
    vmr_says_upro = vmr["regime"] in ("UPRO", "UPRO_MR")
    gp3_says_upro = gp3["signal"] == "UPRO"

    if vmr_says_upro and gp3_says_upro:
        return {"signal": "UPRO", "allocation": {"UPRO": 1.0},
                "reason": "BOTH agree → UPRO"}
    else:
        return {"signal": "SPY", "allocation": {"SPY": 1.0},
                "reason": f"VMR={'UPRO' if vmr_says_upro else 'SPY'}, GP3={'UPRO' if gp3_says_upro else 'SPY'} → SPY"}


# ─── Portfolio blend ─────────────────────────────────────────────────────────

def blend_allocations(vmr_alloc: dict, consensus_alloc: dict,
                      vmr_weight: float = 0.50, cons_weight: float = 0.50) -> dict:
    """Blend VMR (50%) and Consensus (50%) allocations."""
    combined = {}
    for asset, weight in vmr_alloc.items():
        combined[asset] = combined.get(asset, 0) + weight * vmr_weight
    for asset, weight in consensus_alloc.items():
        combined[asset] = combined.get(asset, 0) + weight * cons_weight

    # Clean up tiny weights
    combined = {k: round(v, 4) for k, v in combined.items() if v > 0.001}
    return combined


# ─── State management ────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "portfolio_value": INITIAL_CAPITAL,
        "allocation": {"SPY": 1.0},
        "vmr_regime": "SPY",
        "gp3_confluence": 0,
        "consensus": "SPY",
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


def queue_alert(message: str, channel: str = "system-status"):
    """Queue a Discord alert for Claude to pick up."""
    pending = []
    if ALERTS_FILE.exists():
        try:
            pending = json.loads(ALERTS_FILE.read_text())
        except Exception:
            pending = []

    pending.append({
        "timestamp": datetime.now(ET).isoformat(),
        "channel": channel,
        "message": message,
        "high_conviction": channel == "alerts",
        "sent": False,
    })
    ALERTS_FILE.write_text(json.dumps(pending, indent=2))


# ─── Main run ────────────────────────────────────────────────────────────────

def run():
    now = datetime.now(ET)
    print(f"\n{'='*60}")
    print(f"Portfolio Engine v2 — {now.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*60}")

    state = load_state()
    print(f"Portfolio: ${state['portfolio_value']:,.2f}")
    print(f"Current allocation: {state['allocation']}")

    # Fetch data
    print("\nFetching market data...")
    data = fetch_data()

    if "VIX" not in data or "SPY" not in data:
        print("ERROR: Missing critical data. Skipping.")
        return

    # Generate signals
    vmr = vmr_signal(data)
    gp3 = gp3_signal(data)
    cons = consensus_signal(vmr, gp3)

    print(f"\nSignals:")
    print(f"  VMR: {vmr['regime']} ({vmr['reason']})")
    print(f"  GP3: conf={gp3['confluence']:.1f}, signal={gp3['signal']} ({gp3['reason']})")
    print(f"  Consensus: {cons['signal']} ({cons['reason']})")

    # Blend 50/50
    target = blend_allocations(vmr["allocation"], cons["allocation"])
    print(f"\n  Target allocation (50/50 blend): {target}")

    # Calculate portfolio return from yesterday
    old_alloc = state.get("allocation", {"SPY": 1.0})
    daily_return = 0.0
    for asset, weight in old_alloc.items():
        asset_data = data.get(asset, {})
        if asset_data.get("prev_price") and asset_data.get("price"):
            r = (asset_data["price"] / asset_data["prev_price"] - 1) * weight
            daily_return += r

    state["portfolio_value"] *= (1 + daily_return)

    # Check for allocation change
    old_alloc_str = json.dumps(state.get("allocation", {}), sort_keys=True)
    new_alloc_str = json.dumps(target, sort_keys=True)

    if old_alloc_str != new_alloc_str:
        state["trades"] += 1
        old_regime = state.get("vmr_regime", "?")
        new_regime = vmr["regime"]
        print(f"\n  >> REBALANCE: {state['allocation']} → {target}")

        # Queue alert for significant changes
        if any(target.get(a, 0) > 0 for a in ["UPRO"]):
            queue_alert(
                f"Portfolio v2 entering UPRO: VMR={vmr['regime']}, "
                f"GP3 conf={gp3['confluence']:.1f}. "
                f"New allocation: {target}",
                channel="alerts"
            )

    # Update state
    state["allocation"] = target
    state["vmr_regime"] = vmr["regime"]
    state["gp3_confluence"] = gp3["confluence"]
    state["consensus"] = cons["signal"]

    print(f"\nPortfolio: ${state['portfolio_value']:,.2f} (daily: {daily_return:+.2%})")
    print(f"Regime: VMR={vmr['regime']}, GP3={gp3['signal']}, Cons={cons['signal']}")
    print(f"Allocation: {target}")
    print(f"Trades: {state['trades']}")

    save_state(state)

    append_history({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "portfolio_value": round(state["portfolio_value"], 2),
        "daily_return": round(daily_return, 6),
        "vmr_regime": vmr["regime"],
        "gp3_confluence": gp3["confluence"],
        "consensus": cons["signal"],
        "allocation": json.dumps(target),
        "vix": vmr["vix"],
        "vol_21d": gp3.get("vol_21d", 0),
    })

    print(f"\nDone. State saved.")


if __name__ == "__main__":
    run()
