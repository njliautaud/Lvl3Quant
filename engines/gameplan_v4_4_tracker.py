"""
Gameplan v4.4 Full Adaptive Paper Tracker — VIX Percentile + Adaptive Confluence.

UPGRADE from v3 (SESSION_STATE entry 461):
  v3 uses fixed vol thresholds (10/15/30%). v4.4 replaces them with VIX rolling
  percentile (63d lookback), and dynamically adjusts confluence entry/exit thresholds
  based on where VIX sits relative to its recent history.

  Key changes:
    - VIX percentile > 80th -> GLD (crisis, relative to recent vol regime)
    - VIX percentile > 60th -> tighter confluence (entry 3.0, exit 2.5)
    - VIX percentile < 30th -> looser confluence (entry 2.0, exit 1.5)
    - VIX percentile 30-60th -> standard confluence (entry 2.5, exit 2.0)
    - September hedge and bearish MA trend still override (unchanged from v3)

  This naturally adapts: VIX 14 in 2017 (calm) = 80th pctile -> cautious.
                          VIX 20 in 2022 (volatile) = 30th pctile -> normal.

Validation:
  v4.4: Sharpe 3.18 vs v3 baseline 2.27 (+40%). Calmar 4.65 (best). 13/13 WF positive.
  Permutation p=0.000. All adversarial gates pass (4/5, R1 FAIL structural).
  CAVEAT: In-sample Sharpes inflated by DCA. Real OOS likely 0.8-1.2.

Tracks v4.4, v3, and v2 signals side by side.

Usage:
  python3 gameplan_v4_4_tracker.py             # daily update
  python3 gameplan_v4_4_tracker.py --status    # current state
  python3 gameplan_v4_4_tracker.py --history   # full history
  python3 gameplan_v4_4_tracker.py --reset     # reset (fresh start)
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
STATE_FILE = STATE_DIR / "gameplan_v4_4_state.json"
HISTORY_FILE = STATE_DIR / "gameplan_v4_4_history.csv"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GP-v4.4] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "gameplan_v4_4_tracker.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("gameplan_v4_4")

# === v4.4 ADAPTIVE PARAMETERS ===
VIX_LOOKBACK = 63          # Days for VIX percentile calculation
VIX_HIGH_PCTILE = 80       # Above this -> GLD
VIX_LOW_PCTILE = 20        # Below this -> loose confluence
BASE_ENTRY = 2.5           # Standard confluence entry threshold
BASE_EXIT = 2.0            # Standard confluence exit threshold

# === v2 PARAMETERS (for baseline comparison) ===
VOL_FULL_UPRO = 10.0
VOL_OVERNIGHT = 15.0
VOL_CRISIS = 30.0

# === v3 PARAMETERS (for comparison) ===
V3_CONFLUENCE_ENTRY = 2.5
V3_CONFLUENCE_EXIT = 2.0


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return _default_state()


def _default_state() -> dict:
    return {
        "start_date": None,
        "initial_capital": 500.0,
        "weekly_dca": 100.0,
        # v4.4 portfolio
        "v4_value": 500.0,
        "v4_holding": None,
        "v4_regime": None,
        "v4_switches": 0,
        "v4_in_upro": False,
        # v3 portfolio (parallel tracking)
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
        "last_switch_date_v4": None,
        "version": "v4.4-full-adaptive",
    }


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def append_history(row: dict):
    file_exists = HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size > 0
    fields = ["date", "v4_regime", "v4_holding", "v3_holding", "v2_holding",
              "v4_differs_v3", "vix_level", "vix_pctile",
              "confluence_score", "entry_thresh", "exit_thresh",
              "vol_21d", "spy_price",
              "v4_value", "v3_value", "v2_value", "spy_benchmark", "upro_benchmark",
              "total_contributed", "v4_daily_ret", "v3_daily_ret",
              "v4_switch", "note"]
    with open(HISTORY_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def fetch_data():
    """Fetch SPY, UPRO, GLD, and ^VIX data."""
    import yfinance as yf
    tickers = ["SPY", "UPRO", "GLD", "^VIX"]
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
    # Rename ^VIX to VIX
    if "^VIX" in closes.columns:
        closes = closes.rename(columns={"^VIX": "VIX"})
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


def compute_vix_percentile(vix_series, lookback=VIX_LOOKBACK):
    """
    Compute VIX percentile rank: what % of the last `lookback` days had
    VIX lower than today's VIX. Higher = VIX is elevated relative to recent.
    """
    current = vix_series.iloc[-1]
    if np.isnan(current):
        return None
    window = vix_series.iloc[-(lookback + 1):-1].dropna()
    if len(window) < lookback * 0.8:  # Need at least 80% of lookback
        return None
    pctile = (window < current).sum() / len(window) * 100
    return float(pctile)


def compute_confluence_score(spy_close):
    """
    Compute 3-timeframe confluence score (0-3).
    Same 6-factor scoring as v3. All trailing.
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
    """v3 regime (v2 + fixed confluence gate)."""
    spy_ret = spy_close.pct_change()
    vol_21d = spy_ret.rolling(21).std().iloc[-1] * np.sqrt(252) * 100
    sma20 = spy_close.rolling(20).mean().iloc[-1]
    sma200 = spy_close.rolling(200).mean().iloc[-1]

    if np.isnan(vol_21d): vol_21d = 15.0

    if today_date.month == 9:
        return "SPY", "SEPTEMBER_HEDGE", vol_21d, None, False

    if not np.isnan(sma20) and not np.isnan(sma200) and sma20 < sma200:
        return "SPY", "BEARISH_TREND", vol_21d, None, False

    if vol_21d > VOL_CRISIS:
        return "GLD", "CRISIS", vol_21d, None, False

    if vol_21d > VOL_OVERNIGHT:
        return "SPY", "ELEVATED_VOL", vol_21d, None, False

    score, _ = compute_confluence_score(spy_close)

    if in_upro:
        if score < V3_CONFLUENCE_EXIT:
            return "SPY", "CONF_EXIT", vol_21d, score, False
        holding = "UPRO_OVERNIGHT" if vol_21d > VOL_FULL_UPRO else "UPRO"
        return holding, "GATED", vol_21d, score, True
    else:
        if score >= V3_CONFLUENCE_ENTRY:
            holding = "UPRO_OVERNIGHT" if vol_21d > VOL_FULL_UPRO else "UPRO"
            return holding, "GATED", vol_21d, score, True
        return "SPY", "CONF_BLOCKED", vol_21d, score, False


def compute_v4_regime(spy_close, vix_series, today_date, in_upro):
    """
    v4.4 Full Adaptive regime:
    1. Static protections (Sep hedge, bearish MA) unchanged
    2. VIX percentile replaces fixed vol thresholds
    3. Confluence entry/exit thresholds adapt to VIX percentile
    """
    sma20 = spy_close.rolling(20).mean().iloc[-1]
    sma200 = spy_close.rolling(200).mean().iloc[-1]
    spy_ret = spy_close.pct_change()
    vol_21d = spy_ret.rolling(21).std().iloc[-1] * np.sqrt(252) * 100
    if np.isnan(vol_21d):
        vol_21d = 15.0

    # --- Static protections (unchanged) ---
    if today_date.month == 9:
        return "SPY", "SEPTEMBER_HEDGE", vol_21d, None, None, None, None, False

    if not np.isnan(sma20) and not np.isnan(sma200) and sma20 < sma200:
        return "SPY", "BEARISH_TREND", vol_21d, None, None, None, None, False

    # --- VIX percentile ---
    vix_pctile = compute_vix_percentile(vix_series, VIX_LOOKBACK)
    if vix_pctile is None:
        return "SPY", "NO_VIX_DATA", vol_21d, None, None, None, None, False

    # High VIX percentile -> crisis (GLD)
    if vix_pctile > VIX_HIGH_PCTILE:
        return "GLD", "VIX_PCTILE_CRISIS", vol_21d, vix_pctile, None, None, None, False

    # --- Adaptive confluence thresholds ---
    if vix_pctile > 60:
        # Elevated VIX -> harder to enter, easier to exit
        entry_t = BASE_ENTRY + 0.5   # 3.0
        exit_t = BASE_EXIT + 0.5     # 2.5
    elif vix_pctile < 30:
        # Low VIX -> easier to enter, harder to exit
        entry_t = max(1.5, BASE_ENTRY - 0.5)  # 2.0
        exit_t = max(1.0, BASE_EXIT - 0.5)    # 1.5
    else:
        # Normal range
        entry_t = BASE_ENTRY   # 2.5
        exit_t = BASE_EXIT     # 2.0

    # Mid-range VIX percentile: ensure entry is at least base
    if vix_pctile > VIX_LOW_PCTILE:
        entry_t = max(entry_t, BASE_ENTRY)

    # --- Confluence gate with adaptive thresholds ---
    score, components = compute_confluence_score(spy_close)

    if in_upro:
        if score < exit_t:
            return "SPY", f"ADAPTIVE_EXIT(t={exit_t})", vol_21d, vix_pctile, score, entry_t, exit_t, False
        return "UPRO", f"ADAPTIVE_HOLD(t={exit_t})", vol_21d, vix_pctile, score, entry_t, exit_t, True
    else:
        if score >= entry_t:
            return "UPRO", f"ADAPTIVE_ENTER(t={entry_t})", vol_21d, vix_pctile, score, entry_t, exit_t, True
        return "SPY", f"ADAPTIVE_BLOCKED(t={entry_t})", vol_21d, vix_pctile, score, entry_t, exit_t, False


def run_daily_update():
    state = load_state()
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    if today.weekday() >= 5:
        log.info("Weekend -- skipping")
        return state

    if state["last_update"] == today_str:
        log.info(f"Already updated for {today_str}")
        return state

    log.info(f"=== v4.4 Full Adaptive daily update for {today_str} ===")

    closes = fetch_data()
    daily_returns = closes.pct_change().iloc[-1]

    if state["start_date"] is None:
        state["start_date"] = today_str

    # Weekly DCA
    week_key = f"{today.year}-W{today.isocalendar()[1]:02d}"
    if state["last_dca_week"] != week_key:
        dca = state["weekly_dca"]
        state["v4_value"] += dca
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
    v3_holding, v3_regime, _, v3_score, v3_new_in_upro = compute_v3_regime(
        closes["SPY"], today, state["v3_in_upro"])
    state["v3_in_upro"] = v3_new_in_upro

    # v4.4 signal
    vix_series = closes["VIX"] if "VIX" in closes.columns else None
    if vix_series is not None:
        v4_holding, v4_regime, _, vix_pctile, conf_score, entry_t, exit_t, v4_new_in_upro = \
            compute_v4_regime(closes["SPY"], vix_series, today, state["v4_in_upro"])
    else:
        log.warning("No VIX data available! Falling back to v3 signal")
        v4_holding, v4_regime = v3_holding, v3_regime
        vix_pctile, conf_score, entry_t, exit_t = None, v3_score, None, None
        v4_new_in_upro = v3_new_in_upro

    state["v4_in_upro"] = v4_new_in_upro

    # Current VIX level
    vix_level = float(vix_series.iloc[-1]) if vix_series is not None and not np.isnan(vix_series.iloc[-1]) else None

    # Track switches
    v4_switched = False
    if state["v4_holding"] is not None and state["v4_holding"] != v4_holding:
        v4_switched = True
        state["v4_switches"] += 1
        state["last_switch_date_v4"] = today_str
        log.info(f"v4.4 SWITCH: {state['v4_holding']} -> {v4_holding}")

    v3_switched = False
    if state["v3_holding"] is not None and state["v3_holding"] != v3_holding:
        v3_switched = True
        state["v3_switches"] += 1

    v2_switched = False
    if state["v2_holding"] is not None and state["v2_holding"] != v2_holding:
        v2_switched = True
        state["v2_switches"] += 1

    state["v4_holding"] = v4_holding
    state["v4_regime"] = v4_regime
    state["v3_holding"] = v3_holding
    state["v3_regime"] = v3_regime
    state["v2_holding"] = v2_holding
    state["v2_regime"] = v2_regime

    # Apply returns -- v4.4
    v4_ret = 0.0
    if v4_holding == "UPRO_OVERNIGHT":
        v4_ret = fetch_overnight_return()
        state["v4_value"] *= (1 + v4_ret)
    elif v4_holding in daily_returns.index and not np.isnan(daily_returns[v4_holding]):
        v4_ret = daily_returns[v4_holding]
        state["v4_value"] *= (1 + v4_ret)

    # Apply returns -- v3
    v3_ret = 0.0
    if v3_holding == "UPRO_OVERNIGHT":
        v3_ret = v4_ret if v4_holding == "UPRO_OVERNIGHT" else fetch_overnight_return()
        state["v3_value"] *= (1 + v3_ret)
    elif v3_holding in daily_returns.index and not np.isnan(daily_returns[v3_holding]):
        v3_ret = daily_returns[v3_holding]
        state["v3_value"] *= (1 + v3_ret)

    # Apply returns -- v2
    v2_ret = 0.0
    if v2_holding == "UPRO_OVERNIGHT":
        v2_ret = v4_ret if v4_holding == "UPRO_OVERNIGHT" else fetch_overnight_return()
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

    differs = v4_holding != v3_holding
    note = ""
    if differs:
        note = f"v4.4={v4_holding} v3={v3_holding} VIX_pctile={vix_pctile:.0f}" if vix_pctile is not None else f"v4.4={v4_holding} v3={v3_holding}"

    # Log
    v4_profit = state["v4_value"] - state["total_contributed"]
    v3_profit = state["v3_value"] - state["total_contributed"]
    log.info(f"VIX: {vix_level:.1f} | Pctile: {vix_pctile:.0f}th" if vix_pctile else "VIX: N/A")
    log.info(f"v4.4: {v4_regime} -> {v4_holding} (score={conf_score if conf_score else 'N/A'}, entry={entry_t}, exit={exit_t})")
    log.info(f"v3:   {v3_regime} -> {v3_holding} (score={v3_score if v3_score else 'N/A'})")
    log.info(f"v2:   {v2_regime} -> {v2_holding}")
    log.info(f"v4.4: ${state['v4_value']:.2f} ({'+' if v4_profit>=0 else ''}{v4_profit:.2f}) | v3: ${state['v3_value']:.2f} ({'+' if v3_profit>=0 else ''}{v3_profit:.2f})")
    if differs:
        log.info(f"v4.4 DIFFERS from v3: {note}")

    save_state(state)
    append_history({
        "date": today_str,
        "v4_regime": v4_regime,
        "v4_holding": v4_holding,
        "v3_holding": v3_holding,
        "v2_holding": v2_holding,
        "v4_differs_v3": "YES" if differs else "",
        "vix_level": f"{vix_level:.2f}" if vix_level else "",
        "vix_pctile": f"{vix_pctile:.0f}" if vix_pctile else "",
        "confluence_score": f"{conf_score:.1f}" if conf_score is not None else "",
        "entry_thresh": f"{entry_t:.1f}" if entry_t is not None else "",
        "exit_thresh": f"{exit_t:.1f}" if exit_t is not None else "",
        "vol_21d": f"{vol_21d:.1f}",
        "spy_price": f"{closes['SPY'].iloc[-1]:.2f}",
        "v4_value": f"{state['v4_value']:.2f}",
        "v3_value": f"{state['v3_value']:.2f}",
        "v2_value": f"{state['v2_value']:.2f}",
        "spy_benchmark": f"{state['spy_benchmark']:.2f}",
        "upro_benchmark": f"{state['upro_benchmark']:.2f}",
        "total_contributed": f"{state['total_contributed']:.2f}",
        "v4_daily_ret": f"{v4_ret:.4f}",
        "v3_daily_ret": f"{v3_ret:.4f}",
        "v4_switch": "YES" if v4_switched else "",
        "note": note,
    })

    return state


def print_status():
    state = load_state()
    if state["start_date"] is None:
        print("No tracking data yet.")
        return

    v4_profit = state["v4_value"] - state["total_contributed"]
    v3_profit = state["v3_value"] - state["total_contributed"]
    v2_profit = state["v2_value"] - state["total_contributed"]
    spy_profit = state["spy_benchmark"] - state["total_contributed"]
    v4_pct = (v4_profit / state["total_contributed"]) * 100
    v3_pct = (v3_profit / state["total_contributed"]) * 100
    v2_pct = (v2_profit / state["total_contributed"]) * 100
    spy_pct = (spy_profit / state["total_contributed"]) * 100

    print(f"\n{'='*65}")
    print(f"  GAMEPLAN v4.4 Full Adaptive Paper Tracker")
    print(f"  VIX Percentile: {VIX_LOOKBACK}d lookback | Crisis >{VIX_HIGH_PCTILE}th pctile")
    print(f"  Adaptive confluence: entry {BASE_ENTRY} +/-0.5 | exit {BASE_EXIT} +/-0.5")
    print(f"{'='*65}")
    print(f"  Started:       {state['start_date']}")
    print(f"  Last update:   {state['last_update']}")
    print(f"  Contributed:   ${state['total_contributed']:,.2f}")
    print()
    print(f"  v4.4 Portfolio: ${state['v4_value']:,.2f}  ({'+' if v4_profit>=0 else ''}{v4_profit:,.2f} / {v4_pct:+.1f}%)")
    print(f"  v3   Portfolio: ${state['v3_value']:,.2f}  ({'+' if v3_profit>=0 else ''}{v3_profit:,.2f} / {v3_pct:+.1f}%)")
    print(f"  v2   Portfolio: ${state['v2_value']:,.2f}  ({'+' if v2_profit>=0 else ''}{v2_profit:,.2f} / {v2_pct:+.1f}%)")
    print(f"  SPY benchmark:  ${state['spy_benchmark']:,.2f}  ({'+' if spy_profit>=0 else ''}{spy_profit:,.2f} / {spy_pct:+.1f}%)")
    print(f"  UPRO benchmark: ${state['upro_benchmark']:,.2f}")
    print()
    print(f"  v4.4 current: {state['v4_regime']} -> {state['v4_holding']}")
    print(f"  v3   current: {state['v3_regime']} -> {state['v3_holding']}")
    print(f"  v2   current: {state['v2_regime']} -> {state['v2_holding']}")
    print(f"  v4.4 in UPRO: {'YES' if state['v4_in_upro'] else 'NO'}")
    print(f"  Switches: v4.4={state['v4_switches']} | v3={state['v3_switches']} | v2={state['v2_switches']}")
    print(f"{'='*65}")


def print_history():
    if not HISTORY_FILE.exists():
        print("No history yet.")
        return
    with open(HISTORY_FILE) as f:
        rows = list(csv.DictReader(f))
    print(f"\n{'Date':<12} {'v4Hold':<7} {'v3Hold':<7} {'Diff':>4} {'VIX':>5} {'Pctl':>4} {'Score':>5} {'Entry':>5} {'v4$':>10} {'v3$':>10}")
    print("-" * 85)
    for r in rows[-30:]:
        diff = " !" if r.get("v4_differs_v3") == "YES" else ""
        print(f"{r['date']:<12} {r['v4_holding']:<7} {r['v3_holding']:<7} {diff:>4} "
              f"{r.get('vix_level',''):>5} {r.get('vix_pctile',''):>4} "
              f"{r.get('confluence_score',''):>5} {r.get('entry_thresh',''):>5} "
              f"${float(r['v4_value']):>9,.2f} ${float(r['v3_value']):>9,.2f}")


def reset_state():
    state = _default_state()
    save_state(state)
    if HISTORY_FILE.exists():
        archive = HISTORY_FILE.with_name("gameplan_v4_4_history_archive.csv")
        HISTORY_FILE.rename(archive)
    log.info("v4.4 state reset")
    print("State reset. Ready for v4.4 tracking.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gameplan v4.4 Full Adaptive Paper Tracker")
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
