"""
Gameplan v2 FINAL Paper Tracker — Daily signal + paper P&L tracking.

FINAL optimized parameters (walk-forward entry #424 + overnight enhancement #425):
  - Vol < 10%:  hold UPRO full-day
  - Vol 10-15%: hold UPRO overnight-only (buy at close, sell at open next day)
  - Vol 15-30%: hold SPY
  - Vol > 30%:  hold GLD (safe haven)
  - Protection: 20/200 MA crossover (if SPY 20MA < 200MA -> SPY regardless)
  - September:  always SPY regardless of vol
  - Earnings aggression: OFF (no earnings season modifier)
  - Vol measured as 21-day realized vol (annualized)

Runs daily at 16:15 ET via PM2 cron. Records:
  - Today's regime signal and allocation
  - Paper portfolio value (starting $500 + $100/wk DCA)
  - Cumulative performance vs SPY/UPRO benchmarks

Usage:
  python3 gameplan_v2_tracker.py             # daily update
  python3 gameplan_v2_tracker.py --status    # current state
  python3 gameplan_v2_tracker.py --history   # full history table
  python3 gameplan_v2_tracker.py --reset     # reset state (fresh start)
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
STATE_FILE = STATE_DIR / "gameplan_v2_state.json"
HISTORY_FILE = STATE_DIR / "gameplan_v2_history.csv"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GP-v2-FINAL] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "gameplan_v2_tracker.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("gameplan_v2")

# === FINAL OPTIMIZED PARAMETERS ===
VOL_THRESHOLD_FULL_UPRO = 10.0    # vol < 10% -> full-day UPRO
VOL_THRESHOLD_OVERNIGHT = 15.0    # vol 10-15% -> UPRO overnight only
VOL_THRESHOLD_CRISIS = 30.0       # vol > 30% -> GLD
# vol 15-30% -> SPY


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return _default_state()


def _default_state() -> dict:
    return {
        "start_date": None,
        "initial_capital": 500.0,
        "weekly_dca": 100.0,
        "portfolio_value": 500.0,
        "total_contributed": 500.0,
        "current_regime": None,
        "current_holding": None,
        "last_switch_date": None,
        "total_switches": 0,
        "last_update": None,
        "spy_benchmark": 500.0,
        "upro_benchmark": 500.0,
        "last_dca_week": None,
        "version": "v2-final",
    }


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def append_history(row: dict):
    file_exists = HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size > 0
    fields = ["date", "regime", "holding", "vol_21d", "spy_price", "sma20", "sma200",
              "portfolio_value", "spy_benchmark", "upro_benchmark",
              "total_contributed", "daily_return", "switch", "note"]
    with open(HISTORY_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def fetch_data():
    """Fetch current market data for SPY, UPRO, GLD."""
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
    """
    Fetch UPRO overnight return (close-to-open).
    Returns the most recent overnight return as a fraction.
    """
    import yfinance as yf

    upro = yf.download("UPRO", period="5d", auto_adjust=True, progress=False)
    if len(upro) < 2:
        return 0.0

    # Overnight return = today's open / yesterday's close - 1
    prev_close = upro["Close"].iloc[-2]
    today_open = upro["Open"].iloc[-1]

    if isinstance(prev_close, pd.Series):
        prev_close = prev_close.iloc[0]
    if isinstance(today_open, pd.Series):
        today_open = today_open.iloc[0]

    if np.isnan(prev_close) or np.isnan(today_open) or prev_close == 0:
        return 0.0

    return float(today_open / prev_close - 1)


def compute_regime(spy_close, today_date):
    """
    Determine current regime using FINAL v2 optimized rules.

    Tiers:
      vol < 10%:    UPRO full-day
      vol 10-15%:   UPRO overnight-only
      vol 15-30%:   SPY
      vol > 30%:    GLD
      Protection:   20/200 MA crossover -> SPY
      September:    always SPY
      Earnings:     OFF (no modifier)
    """
    spy_ret = spy_close.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    sma20 = spy_close.rolling(20).mean()
    sma200 = spy_close.rolling(200).mean()

    current_vol = vol_21d.iloc[-1]
    current_sma20 = sma20.iloc[-1]
    current_sma200 = sma200.iloc[-1]
    current_spy = spy_close.iloc[-1]

    vol_pct = current_vol * 100 if not np.isnan(current_vol) else 15.0

    # September hedge — always SPY
    if today_date.month == 9:
        regime = "SEPTEMBER_HEDGE"
        holding = "SPY"
        reason = "September hedge rule — always SPY"
        return _make_signal(regime, holding, reason, vol_pct, current_sma20, current_sma200, current_spy)

    # 20/200 MA crossover protection — always SPY if bearish
    protection_active = (not np.isnan(current_sma20) and not np.isnan(current_sma200)
                         and current_sma20 < current_sma200)

    if protection_active:
        regime = "BEARISH_TREND"
        holding = "SPY"
        reason = f"MA protection: SMA20 ({current_sma20:.2f}) < SMA200 ({current_sma200:.2f})"
        return _make_signal(regime, holding, reason, vol_pct, current_sma20, current_sma200, current_spy)

    # Vol-based tiers (FINAL params)
    if vol_pct > VOL_THRESHOLD_CRISIS:
        regime = "CRISIS"
        holding = "GLD"
        reason = f"Vol {vol_pct:.1f}% > {VOL_THRESHOLD_CRISIS}% — safe haven"
    elif vol_pct > VOL_THRESHOLD_OVERNIGHT:
        regime = "ELEVATED_VOL"
        holding = "SPY"
        reason = f"Vol {vol_pct:.1f}% in {VOL_THRESHOLD_OVERNIGHT}-{VOL_THRESHOLD_CRISIS}% range"
    elif vol_pct > VOL_THRESHOLD_FULL_UPRO:
        regime = "MODERATE_VOL"
        holding = "UPRO_OVERNIGHT"
        reason = f"Vol {vol_pct:.1f}% in {VOL_THRESHOLD_FULL_UPRO}-{VOL_THRESHOLD_OVERNIGHT}% — overnight only"
    else:
        regime = "LOW_VOL"
        holding = "UPRO"
        reason = f"Vol {vol_pct:.1f}% < {VOL_THRESHOLD_FULL_UPRO}% — full-day UPRO"

    return _make_signal(regime, holding, reason, vol_pct, current_sma20, current_sma200, current_spy)


def _make_signal(regime, holding, reason, vol_pct, sma20, sma200, spy_price):
    return {
        "regime": regime,
        "holding": holding,
        "reason": reason,
        "vol_21d": float(vol_pct),
        "sma20": float(sma20) if not np.isnan(sma20) else None,
        "sma200": float(sma200) if not np.isnan(sma200) else None,
        "spy_price": float(spy_price),
    }


def run_daily_update():
    """Run the daily portfolio update."""
    state = load_state()
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    # Skip weekends
    if today.weekday() >= 5:
        log.info("Weekend — skipping")
        return state

    # Skip if already updated today
    if state["last_update"] == today_str:
        log.info(f"Already updated for {today_str}")
        return state

    log.info(f"=== Daily update for {today_str} ===")

    # Fetch data
    closes = fetch_data()
    returns = closes.pct_change().iloc[-1]

    # Initialize start date
    if state["start_date"] is None:
        state["start_date"] = today_str

    # Weekly DCA (check by ISO week)
    week_key = f"{today.year}-W{today.isocalendar()[1]:02d}"
    if state["last_dca_week"] != week_key:
        dca = state["weekly_dca"]
        state["portfolio_value"] += dca
        state["spy_benchmark"] += dca
        state["upro_benchmark"] += dca
        state["total_contributed"] += dca
        state["last_dca_week"] = week_key
        log.info(f"DCA: +${dca:.0f} (total contributed: ${state['total_contributed']:.0f})")

    # Compute regime
    signal = compute_regime(closes["SPY"], today)
    holding = signal["holding"]

    # For display/logging, normalize holding name
    holding_display = holding
    note = ""

    # Check for switch
    switched = False
    if state["current_holding"] is not None and state["current_holding"] != holding:
        switched = True
        state["total_switches"] += 1
        state["last_switch_date"] = today_str
        log.info(f"SWITCH: {state['current_holding']} -> {holding} ({signal['reason']})")

    state["current_regime"] = signal["regime"]
    state["current_holding"] = holding

    # Apply returns based on holding type
    daily_ret = 0.0

    if holding == "UPRO_OVERNIGHT":
        # Overnight-only: we only capture the close-to-open return of UPRO
        overnight_ret = fetch_overnight_return()
        daily_ret = overnight_ret
        state["portfolio_value"] *= (1 + daily_ret)
        holding_display = "UPRO(ON)"
        note = f"overnight return: {overnight_ret:.4f}"
        log.info(f"UPRO overnight-only return: {overnight_ret:.4%}")
    elif holding in ["UPRO", "SPY", "GLD"]:
        if holding in returns.index and not np.isnan(returns[holding]):
            daily_ret = returns[holding]
            state["portfolio_value"] *= (1 + daily_ret)
    # else: unknown holding, no return applied

    # Benchmarks always track full-day
    if "SPY" in returns.index and not np.isnan(returns["SPY"]):
        state["spy_benchmark"] *= (1 + returns["SPY"])
    if "UPRO" in returns.index and not np.isnan(returns["UPRO"]):
        state["upro_benchmark"] *= (1 + returns["UPRO"])

    state["last_update"] = today_str

    # Log summary
    profit = state["portfolio_value"] - state["total_contributed"]
    spy_profit = state["spy_benchmark"] - state["total_contributed"]
    log.info(f"Regime: {signal['regime']} -> Holding: {holding_display}")
    log.info(f"Reason: {signal['reason']}")
    log.info(f"Portfolio: ${state['portfolio_value']:.2f} (profit: ${profit:.2f})")
    log.info(f"SPY bench: ${state['spy_benchmark']:.2f} (profit: ${spy_profit:.2f})")
    log.info(f"Daily ret: {daily_ret:.2%} | Switches: {state['total_switches']}")

    # Save
    save_state(state)
    append_history({
        "date": today_str,
        "regime": signal["regime"],
        "holding": holding_display,
        "vol_21d": f"{signal['vol_21d']:.1f}",
        "spy_price": f"{signal['spy_price']:.2f}",
        "sma20": f"{signal['sma20']:.2f}" if signal["sma20"] else "",
        "sma200": f"{signal['sma200']:.2f}" if signal["sma200"] else "",
        "portfolio_value": f"{state['portfolio_value']:.2f}",
        "spy_benchmark": f"{state['spy_benchmark']:.2f}",
        "upro_benchmark": f"{state['upro_benchmark']:.2f}",
        "total_contributed": f"{state['total_contributed']:.2f}",
        "daily_return": f"{daily_ret:.4f}",
        "switch": "YES" if switched else "",
        "note": note,
    })

    return state


def print_status():
    """Print current portfolio status."""
    state = load_state()
    if state["start_date"] is None:
        print("No tracking data yet. Run without --status first.")
        return

    profit = state["portfolio_value"] - state["total_contributed"]
    spy_profit = state["spy_benchmark"] - state["total_contributed"]
    upro_profit = state["upro_benchmark"] - state["total_contributed"]
    pct = (profit / state["total_contributed"]) * 100
    spy_pct = (spy_profit / state["total_contributed"]) * 100

    print(f"\n{'='*55}")
    print(f"  GAMEPLAN v2 FINAL Paper Tracker")
    print(f"  Vol tiers: <10% UPRO | 10-15% UPRO overnight | 15-30% SPY | >30% GLD")
    print(f"{'='*55}")
    print(f"  Started:       {state['start_date']}")
    print(f"  Last update:   {state['last_update']}")
    print(f"  Contributed:   ${state['total_contributed']:,.2f}")
    print(f"")
    print(f"  Portfolio:     ${state['portfolio_value']:,.2f}  ({'+' if profit >= 0 else ''}{profit:,.2f} / {'+' if pct >= 0 else ''}{pct:.1f}%)")
    print(f"  SPY benchmark: ${state['spy_benchmark']:,.2f}  ({'+' if spy_profit >= 0 else ''}{spy_profit:,.2f} / {'+' if spy_pct >= 0 else ''}{spy_pct:.1f}%)")
    print(f"  UPRO benchmark:${state['upro_benchmark']:,.2f}  ({'+' if upro_profit >= 0 else ''}{upro_profit:,.2f})")
    print(f"")
    print(f"  Current:       {state['current_regime']} -> {state['current_holding']}")
    print(f"  Switches:      {state['total_switches']}")
    if state.get('last_switch_date'):
        print(f"  Last switch:   {state['last_switch_date']}")
    print(f"  Version:       {state.get('version', 'unknown')}")
    print(f"{'='*55}")


def print_history():
    """Print full history."""
    if not HISTORY_FILE.exists():
        print("No history yet.")
        return

    with open(HISTORY_FILE) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\n{'Date':<12} {'Regime':<18} {'Hold':<10} {'Vol':>5} {'Value':>10} {'SPY':>10} {'Ret':>7} {'Sw':>3}")
    print("-" * 80)
    for r in rows[-30:]:
        sw = " *" if r.get("switch") == "YES" else ""
        print(f"{r['date']:<12} {r['regime']:<18} {r['holding']:<10} {r['vol_21d']:>5} "
              f"${float(r['portfolio_value']):>9,.2f} ${float(r['spy_benchmark']):>9,.2f} "
              f"{float(r['daily_return']):>6.2%}{sw}")


def reset_state():
    """Reset state for fresh start with new params."""
    state = _default_state()
    save_state(state)
    # Archive old history
    if HISTORY_FILE.exists():
        archive = HISTORY_FILE.with_name("gameplan_v2_history_old_params.csv")
        HISTORY_FILE.rename(archive)
        log.info(f"Archived old history to {archive.name}")
    log.info("State reset for v2-final params")
    print("State reset. Old history archived. Ready for fresh tracking with FINAL params.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gameplan v2 FINAL Paper Tracker")
    parser.add_argument("--status", action="store_true", help="Print current status")
    parser.add_argument("--history", action="store_true", help="Print history")
    parser.add_argument("--reset", action="store_true", help="Reset state for fresh start")
    args = parser.parse_args()

    if args.reset:
        reset_state()
    elif args.status:
        print_status()
    elif args.history:
        print_history()
    else:
        run_daily_update()
