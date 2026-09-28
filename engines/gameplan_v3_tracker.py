"""
Gameplan v3 Paper Tracker — v2 with Confluence Confirmation Gate.

UPGRADE from v2 (SESSION_STATE entries 427-429):
  v2 rules stay the same (vol tiers, 20/200 MA, Sep hedge).
  NEW: Confluence gate prevents entering UPRO during low-vol breakdown windows.
    - 3-timeframe score: short (5d mom + 10d RSI) + medium (20/50 MA + vol<15%) + long (200d slope + 63d vol trend)
    - Score range 0-3. Entry gate: need ≥ 2.5. Exit gate: drop below 2.0.
    - Hysteresis prevents whipsaw.
    - When vol says UPRO but confluence < threshold → stay in SPY.
    - Vol/MA crisis protection always overrides (unchanged from v2).

Validation:
  v3: Sharpe 2.388 vs v2 baseline 1.812 (+0.576). 13/13 years beats v2.
  Permutation p=0.000. WF 5.95 mean OOS Sharpe. MaxDD -25.2% vs -31.3%.

Runs daily at 16:15 ET via PM2. Tracks BOTH v2 and v3 signals side by side.

Usage:
  python3 gameplan_v3_tracker.py             # daily update
  python3 gameplan_v3_tracker.py --status    # current state
  python3 gameplan_v3_tracker.py --history   # full history
  python3 gameplan_v3_tracker.py --reset     # reset (fresh start)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

ET = pytz.timezone("US/Eastern")

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "gameplan_v3_state.json"
HISTORY_FILE = STATE_DIR / "gameplan_v3_history.csv"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GP-v3] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "gameplan_v3_tracker.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("gameplan_v3")

# === v2 PARAMETERS (unchanged) ===
VOL_FULL_UPRO = 10.0
VOL_OVERNIGHT = 15.0
VOL_CRISIS = 30.0

# === v3 CONFLUENCE GATE PARAMETERS ===
CONFLUENCE_ENTRY = 2.5   # Need score ≥ this to ENTER UPRO
CONFLUENCE_EXIT = 2.0    # Drop below this to EXIT UPRO


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return _default_state()


def _default_state() -> dict:
    return {
        "start_date": None,
        "initial_capital": 500.0,
        "weekly_dca": 100.0,
        # v3 portfolio
        "v3_value": 500.0,
        "v3_holding": None,
        "v3_regime": None,
        "v3_switches": 0,
        "v3_in_upro": False,
        # v2 portfolio (parallel tracking)
        "v2_value": 500.0,
        "v2_holding": None,
        "v2_regime": None,
        "v2_switches": 0,
        # Benchmarks
        "spy_benchmark": 500.0,
        "upro_benchmark": 500.0,
        "total_contributed": 500.0,
        "last_update": None,
        "last_dca_week": None,
        "last_switch_date_v3": None,
        "version": "v3-confluence-gate",
    }


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def append_history(row: dict):
    file_exists = HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size > 0
    fields = ["date", "v3_regime", "v3_holding", "v2_holding", "v3_differs",
              "confluence_score", "vol_21d", "spy_price",
              "v3_value", "v2_value", "spy_benchmark", "upro_benchmark",
              "total_contributed", "v3_daily_ret", "v2_daily_ret",
              "v3_switch", "note"]
    with open(HISTORY_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def fetch_data():
    import yfinance as yf
    tickers = ["SPY", "UPRO", "GLD"]
    data = yf.download(tickers, period="300d", auto_adjust=True,
                       threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data
    if hasattr(closes.columns, "droplevel"):
        try:
            closes.columns = closes.columns.droplevel(1)
        except Exception:
            pass
    return closes.dropna(how="all").dropna(subset=["SPY", "UPRO"])


def fetch_overnight_return():
    import yfinance as yf
    upro = yf.download("UPRO", period="5d", auto_adjust=True, progress=False)
    if len(upro) < 2:
        return 0.0
    prev_close = upro["Close"].iloc[-2]
    today_open = upro["Open"].iloc[-1]
    if isinstance(prev_close, pd.Series):
        prev_close = prev_close.iloc[0]
    if isinstance(today_open, pd.Series):
        today_open = today_open.iloc[0]
    if np.isnan(prev_close) or np.isnan(today_open) or prev_close == 0:
        return 0.0
    return float(today_open / prev_close - 1)


def compute_confluence_score(spy_close):
    """
    Compute 3-timeframe confluence score (0-3).
    All trailing — no look-ahead.
    """
    spy_ret = spy_close.pct_change()

    score = 0.0
    components = {}

    # SHORT: 5d momentum
    mom_5d = spy_close.pct_change(5).iloc[-1]
    components['mom_5d'] = float(mom_5d) if not np.isnan(mom_5d) else 0
    if not np.isnan(mom_5d) and mom_5d > 0:
        score += 0.5

    # SHORT: 10d RSI
    delta = spy_ret.copy()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = gain / loss_s.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    components['rsi_10'] = float(rsi) if not np.isnan(rsi) else 50
    if not np.isnan(rsi) and rsi > 50:
        score += 0.5

    # MEDIUM: 20/50 MA cross
    s20 = spy_close.rolling(20).mean().iloc[-1]
    s50 = spy_close.rolling(50).mean().iloc[-1]
    components['sma20_vs_sma50'] = f"{s20:.0f}/{s50:.0f}"
    if not np.isnan(s20) and not np.isnan(s50) and s20 > s50:
        score += 0.5

    # MEDIUM: vol < 15%
    vol_21d_val = spy_ret.rolling(21).std().iloc[-1] * np.sqrt(252) * 100
    components['vol_21d'] = float(vol_21d_val) if not np.isnan(vol_21d_val) else 15
    if not np.isnan(vol_21d_val) and vol_21d_val < 15:
        score += 0.5

    # LONG: 200d MA slope
    sma200 = spy_close.rolling(200).mean()
    slope = sma200.pct_change(20).iloc[-1]
    components['slope_200d'] = float(slope * 100) if not np.isnan(slope) else 0
    if not np.isnan(slope) and slope > 0:
        score += 0.5

    # LONG: 63d vol trend < 0
    vol_63d = spy_ret.rolling(63).std() * np.sqrt(252) * 100
    vol_trend = (vol_63d - vol_63d.rolling(21).mean()).iloc[-1]
    components['vol_63d_trend'] = float(vol_trend) if not np.isnan(vol_trend) else 0
    if not np.isnan(vol_trend) and vol_trend < 0:
        score += 0.5

    return score, components


def compute_v2_regime(spy_close, today_date):
    """v2 regime (baseline comparison)."""
    spy_ret = spy_close.pct_change()
    vol_21d = spy_ret.rolling(21).std().iloc[-1] * np.sqrt(252) * 100
    sma20 = spy_close.rolling(20).mean().iloc[-1]
    sma200 = spy_close.rolling(200).mean().iloc[-1]

    if np.isnan(vol_21d): vol_21d = 15.0

    if today_date.month == 9:
        return "SPY", "SEPTEMBER_HEDGE", vol_21d

    if not np.isnan(sma20) and not np.isnan(sma200) and sma20 < sma200:
        return "SPY", "BEARISH_TREND", vol_21d

    if vol_21d > VOL_CRISIS:
        return "GLD", "CRISIS", vol_21d
    elif vol_21d > VOL_OVERNIGHT:
        return "SPY", "ELEVATED_VOL", vol_21d
    elif vol_21d > VOL_FULL_UPRO:
        return "UPRO_OVERNIGHT", "MODERATE_VOL", vol_21d
    else:
        return "UPRO", "LOW_VOL", vol_21d


def compute_v3_regime(spy_close, today_date, in_upro):
    """
    v3 regime = v2 + confluence confirmation gate.
    Returns (holding, regime_name, vol_21d, confluence_score, components, new_in_upro).
    """
    spy_ret = spy_close.pct_change()
    vol_21d = spy_ret.rolling(21).std().iloc[-1] * np.sqrt(252) * 100
    sma20 = spy_close.rolling(20).mean().iloc[-1]
    sma200 = spy_close.rolling(200).mean().iloc[-1]

    if np.isnan(vol_21d): vol_21d = 15.0

    # Same overrides as v2
    if today_date.month == 9:
        return "SPY", "SEPTEMBER_HEDGE", vol_21d, None, {}, False

    if not np.isnan(sma20) and not np.isnan(sma200) and sma20 < sma200:
        return "SPY", "BEARISH_TREND", vol_21d, None, {}, False

    if vol_21d > VOL_CRISIS:
        return "GLD", "CRISIS", vol_21d, None, {}, False

    if vol_21d > VOL_OVERNIGHT:
        return "SPY", "ELEVATED_VOL", vol_21d, None, {}, False

    # v2 would say UPRO or UPRO_OVERNIGHT here. Check confluence gate.
    score, components = compute_confluence_score(spy_close)

    if vol_21d > VOL_FULL_UPRO:
        # Vol 10-15%: v2 says UPRO_OVERNIGHT
        # v3: also require confluence gate for overnight
        if in_upro:
            if score < CONFLUENCE_EXIT:
                return "SPY", "CONF_EXIT_ON", vol_21d, score, components, False
            return "UPRO_OVERNIGHT", "MODERATE_VOL_GATED", vol_21d, score, components, True
        else:
            if score >= CONFLUENCE_ENTRY:
                return "UPRO_OVERNIGHT", "MODERATE_VOL_GATED", vol_21d, score, components, True
            return "SPY", "CONF_BLOCKED_ON", vol_21d, score, components, False
    else:
        # Vol < 10%: v2 says full UPRO
        # v3: require confluence gate
        if in_upro:
            if score < CONFLUENCE_EXIT:
                return "SPY", "CONF_EXIT_FULL", vol_21d, score, components, False
            return "UPRO", "LOW_VOL_GATED", vol_21d, score, components, True
        else:
            if score >= CONFLUENCE_ENTRY:
                return "UPRO", "LOW_VOL_GATED", vol_21d, score, components, True
            return "SPY", "CONF_BLOCKED_FULL", vol_21d, score, components, False


def run_daily_update():
    state = load_state()
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    if today.weekday() >= 5:
        log.info("Weekend — skipping")
        return state

    if state["last_update"] == today_str:
        log.info(f"Already updated for {today_str}")
        return state

    log.info(f"=== v3 Daily update for {today_str} ===")

    closes = fetch_data()
    daily_returns = closes.pct_change().iloc[-1]

    if state["start_date"] is None:
        state["start_date"] = today_str

    # Weekly DCA
    week_key = f"{today.year}-W{today.isocalendar()[1]:02d}"
    if state["last_dca_week"] != week_key:
        dca = state["weekly_dca"]
        state["v3_value"] += dca
        state["v2_value"] += dca
        state["spy_benchmark"] += dca
        state["upro_benchmark"] += dca
        state["total_contributed"] += dca
        state["last_dca_week"] = week_key
        log.info(f"DCA: +${dca:.0f} (total: ${state['total_contributed']:.0f})")

    # v2 signal
    v2_holding, v2_regime, vol_21d = compute_v2_regime(closes["SPY"], today)

    # v3 signal
    v3_holding, v3_regime, _, conf_score, conf_comp, new_in_upro = compute_v3_regime(
        closes["SPY"], today, state["v3_in_upro"])

    state["v3_in_upro"] = new_in_upro

    # Switches
    v3_switched = False
    if state["v3_holding"] is not None and state["v3_holding"] != v3_holding:
        v3_switched = True
        state["v3_switches"] += 1
        state["last_switch_date_v3"] = today_str
        log.info(f"v3 SWITCH: {state['v3_holding']} -> {v3_holding}")

    v2_switched = False
    if state["v2_holding"] is not None and state["v2_holding"] != v2_holding:
        v2_switched = True
        state["v2_switches"] += 1

    state["v3_holding"] = v3_holding
    state["v3_regime"] = v3_regime
    state["v2_holding"] = v2_holding
    state["v2_regime"] = v2_regime

    # Apply returns — v3
    v3_ret = 0.0
    if v3_holding == "UPRO_OVERNIGHT":
        v3_ret = fetch_overnight_return()
        state["v3_value"] *= (1 + v3_ret)
    elif v3_holding in daily_returns.index and not np.isnan(daily_returns[v3_holding]):
        v3_ret = daily_returns[v3_holding]
        state["v3_value"] *= (1 + v3_ret)

    # Apply returns — v2
    v2_ret = 0.0
    if v2_holding == "UPRO_OVERNIGHT":
        # Use same overnight return
        v2_ret = v3_ret if v3_holding == "UPRO_OVERNIGHT" else fetch_overnight_return()
        state["v2_value"] *= (1 + v2_ret)
    elif v2_holding in daily_returns.index and not np.isnan(daily_returns[v2_holding]):
        v2_ret = daily_returns[v2_holding]
        state["v2_value"] *= (1 + v2_ret)

    # Benchmarks
    if "SPY" in daily_returns.index and not np.isnan(daily_returns["SPY"]):
        state["spy_benchmark"] *= (1 + daily_returns["SPY"])
    if "UPRO" in daily_returns.index and not np.isnan(daily_returns["UPRO"]):
        state["upro_benchmark"] *= (1 + daily_returns["UPRO"])

    state["last_update"] = today_str

    differs = v3_holding != v2_holding
    note = ""
    if differs:
        note = f"v3≠v2: v3={v3_holding} v2={v2_holding} (score={conf_score:.1f})" if conf_score is not None else f"v3≠v2"

    # Log
    v3_profit = state["v3_value"] - state["total_contributed"]
    v2_profit = state["v2_value"] - state["total_contributed"]
    log.info(f"v3: {v3_regime} -> {v3_holding} (score={conf_score if conf_score else 'N/A'}) | v2: {v2_regime} -> {v2_holding}")
    log.info(f"v3: ${state['v3_value']:.2f} ({'+' if v3_profit>=0 else ''}{v3_profit:.2f}) | v2: ${state['v2_value']:.2f} ({'+' if v2_profit>=0 else ''}{v2_profit:.2f})")
    if differs:
        log.info(f"⚡ v3 DIFFERS from v2: {note}")

    save_state(state)
    append_history({
        "date": today_str,
        "v3_regime": v3_regime,
        "v3_holding": v3_holding,
        "v2_holding": v2_holding,
        "v3_differs": "YES" if differs else "",
        "confluence_score": f"{conf_score:.1f}" if conf_score is not None else "",
        "vol_21d": f"{vol_21d:.1f}",
        "spy_price": f"{closes['SPY'].iloc[-1]:.2f}",
        "v3_value": f"{state['v3_value']:.2f}",
        "v2_value": f"{state['v2_value']:.2f}",
        "spy_benchmark": f"{state['spy_benchmark']:.2f}",
        "upro_benchmark": f"{state['upro_benchmark']:.2f}",
        "total_contributed": f"{state['total_contributed']:.2f}",
        "v3_daily_ret": f"{v3_ret:.4f}",
        "v2_daily_ret": f"{v2_ret:.4f}",
        "v3_switch": "YES" if v3_switched else "",
        "note": note,
    })

    return state


def print_status():
    state = load_state()
    if state["start_date"] is None:
        print("No tracking data yet.")
        return

    v3_profit = state["v3_value"] - state["total_contributed"]
    v2_profit = state["v2_value"] - state["total_contributed"]
    spy_profit = state["spy_benchmark"] - state["total_contributed"]
    upro_profit = state["upro_benchmark"] - state["total_contributed"]
    v3_pct = (v3_profit / state["total_contributed"]) * 100
    v2_pct = (v2_profit / state["total_contributed"]) * 100
    spy_pct = (spy_profit / state["total_contributed"]) * 100

    print(f"\n{'='*60}")
    print(f"  GAMEPLAN v3 Paper Tracker (Confluence-Gated)")
    print(f"  Entry gate ≥{CONFLUENCE_ENTRY} | Exit gate <{CONFLUENCE_EXIT}")
    print(f"{'='*60}")
    print(f"  Started:       {state['start_date']}")
    print(f"  Last update:   {state['last_update']}")
    print(f"  Contributed:   ${state['total_contributed']:,.2f}")
    print(f"")
    print(f"  v3 Portfolio:  ${state['v3_value']:,.2f}  ({'+' if v3_profit>=0 else ''}{v3_profit:,.2f} / {v3_pct:+.1f}%)")
    print(f"  v2 Portfolio:  ${state['v2_value']:,.2f}  ({'+' if v2_profit>=0 else ''}{v2_profit:,.2f} / {v2_pct:+.1f}%)")
    print(f"  SPY benchmark: ${state['spy_benchmark']:,.2f}  ({'+' if spy_profit>=0 else ''}{spy_profit:,.2f} / {spy_pct:+.1f}%)")
    print(f"  UPRO benchmark:${state['upro_benchmark']:,.2f}")
    print(f"")
    print(f"  v3 current:    {state['v3_regime']} -> {state['v3_holding']}")
    print(f"  v2 current:    {state['v2_regime']} -> {state['v2_holding']}")
    print(f"  In UPRO state: {'YES' if state['v3_in_upro'] else 'NO'}")
    print(f"  v3 switches:   {state['v3_switches']} | v2 switches: {state['v2_switches']}")
    print(f"{'='*60}")


def print_history():
    if not HISTORY_FILE.exists():
        print("No history yet.")
        return
    with open(HISTORY_FILE) as f:
        rows = list(csv.DictReader(f))
    print(f"\n{'Date':<12} {'v3Hold':<8} {'v2Hold':<8} {'Diff':>4} {'Score':>5} {'Vol':>5} {'v3$':>10} {'v2$':>10} {'v3Ret':>7}")
    print("-" * 80)
    for r in rows[-30:]:
        diff = " !" if r.get("v3_differs") == "YES" else ""
        print(f"{r['date']:<12} {r['v3_holding']:<8} {r['v2_holding']:<8} {diff:>4} "
              f"{r.get('confluence_score',''):>5} {r['vol_21d']:>5} "
              f"${float(r['v3_value']):>9,.2f} ${float(r['v2_value']):>9,.2f} "
              f"{float(r['v3_daily_ret']):>6.2%}")


def reset_state():
    state = _default_state()
    save_state(state)
    if HISTORY_FILE.exists():
        archive = HISTORY_FILE.with_name("gameplan_v3_history_archive.csv")
        HISTORY_FILE.rename(archive)
    log.info("v3 state reset")
    print("State reset. Ready for v3 tracking.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gameplan v3 Paper Tracker")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        reset_state()
    elif args.status:
        print_status()
    elif args.history:
        print_history()
    else:
        run_daily_update()
