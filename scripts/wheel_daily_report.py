#!/usr/bin/env python3
"""
Wheel Paper Engine Daily Report
================================
Logs daily NAV and metrics for all paper engines to a CSV for trend analysis.
Run once daily via cron (e.g., at 16:30 ET after market close).

Output: output/wheel_paper_dashboard/daily_log.csv
Each row = one engine on one date.
"""
import json
import csv
import os
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "live_trading_linux"
OUT_DIR = ROOT / "output" / "wheel_paper_dashboard"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "daily_log.csv"

ENGINES = {
    "v4_csp":       ("wheel_v4_state",              "CSP 230-name 14DTE"),
    "v5_income":    ("wheel_v5_state",              "CSP 230-name 10DTE"),
    "bps":          ("wheel_bps_state",             "BPS $10 7DTE"),
    "ic":           ("wheel_ic_state",              "Iron Condor 7DTE"),
    "diversified":  ("wheel_diversified_state",     "Diversified 20-name"),
    "spy_base":     ("wheel_paper_state",           "SPY base 37DTE"),
    "balanced":     ("wheel_paper_balanced_state",  "Balanced regime-gated"),
}

STARTING_CAPITAL = 100_000.0

def get_nav(state_dir_name):
    """Get latest NAV from equity.csv (primary) or nav_history/state (fallback)."""
    equity_file = STATE_DIR / state_dir_name / "equity.csv"
    nav_file = STATE_DIR / state_dir_name / "nav_history.json"
    state_file = STATE_DIR / state_dir_name / "state.json"

    # PRIMARY: equity.csv has the proper mark-to-market NAV
    nav = None
    if equity_file.exists():
        try:
            with open(equity_file) as f:
                lines = f.readlines()
            if len(lines) > 1:
                last = lines[-1].strip().split(",")
                nav = float(last[1])
        except Exception:
            pass

    # FALLBACK 1: nav_history.json
    if nav is None and nav_file.exists():
        try:
            with open(nav_file) as f:
                hist = json.load(f)
            if hist:
                nav = hist[-1].get("nav")
        except Exception:
            pass

    # FALLBACK 2: state.json realized + unrealized
    if nav is None and state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
            realized = state.get("realized_pnl", state.get("total_realized", 0)) or 0
            unrealized = state.get("unrealized_pnl", 0) or 0
            nav = STARTING_CAPITAL + realized + unrealized
        except Exception:
            pass

    return nav


def get_max_drawdown(state_dir_name):
    """Compute max drawdown from equity.csv time series."""
    equity_file = STATE_DIR / state_dir_name / "equity.csv"
    if not equity_file.exists():
        return None
    try:
        with open(equity_file) as f:
            lines = f.readlines()[1:]  # skip header
        if len(lines) < 2:
            return 0.0
        navs = [float(line.strip().split(",")[1]) for line in lines if line.strip()]
        peak = navs[0]
        max_dd = 0.0
        for n in navs:
            if n > peak:
                peak = n
            dd = (n - peak) / peak
            if dd < max_dd:
                max_dd = dd
        return round(max_dd * 100, 2)  # as percentage
    except Exception:
        return None


def get_daily_return(state_dir_name):
    """Get today's return (first NAV of day vs latest) from equity.csv."""
    equity_file = STATE_DIR / state_dir_name / "equity.csv"
    if not equity_file.exists():
        return None
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        with open(equity_file) as f:
            lines = f.readlines()[1:]
        today_navs = []
        prev_nav = None
        for line in lines:
            parts = line.strip().split(",")
            ts, nav = parts[0], float(parts[1])
            if today in ts:
                today_navs.append(nav)
            else:
                prev_nav = float(nav)
        if today_navs and prev_nav:
            return round((today_navs[-1] - prev_nav) / prev_nav * 100, 4)
        elif today_navs and len(today_navs) > 1:
            return round((today_navs[-1] - today_navs[0]) / today_navs[0] * 100, 4)
        return 0.0
    except Exception:
        return None


def get_state_metrics(state_dir_name):
    """Extract key metrics from engine state."""
    state_file = STATE_DIR / state_dir_name / "state.json"
    if not state_file.exists():
        return {}

    try:
        with open(state_file) as f:
            state = json.load(f)
    except Exception:
        return {}

    positions = state.get("positions", state.get("open_positions", state.get("spreads", [])))
    return {
        "realized": state.get("realized_pnl", state.get("total_realized", 0)) or 0,
        "n_positions": len(positions) if isinstance(positions, (list, dict)) else 0,
        "n_trades": state.get("trade_count", state.get("total_trades", state.get("n_trades", 0))) or 0,
    }


def main():
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    # Check if CSV exists, create header if not
    write_header = not CSV_PATH.exists()

    rows = []
    for engine_id, (state_dir, description) in ENGINES.items():
        nav = get_nav(state_dir)
        metrics = get_state_metrics(state_dir)

        if nav is None:
            continue

        pnl_pct = (nav - STARTING_CAPITAL) / STARTING_CAPITAL * 100

        max_dd = get_max_drawdown(state_dir)
        daily_ret = get_daily_return(state_dir)

        rows.append({
            "date": date_str,
            "time": time_str,
            "engine": engine_id,
            "description": description,
            "nav": round(nav, 2),
            "pnl_pct": round(pnl_pct, 4),
            "daily_return_pct": daily_ret if daily_ret is not None else 0.0,
            "max_dd_pct": max_dd if max_dd is not None else 0.0,
            "realized": round(metrics.get("realized", 0), 2),
            "n_positions": metrics.get("n_positions", 0),
            "n_trades": metrics.get("n_trades", 0),
        })

    if not rows:
        print("No engine data found")
        return

    fieldnames = ["date", "time", "engine", "description", "nav", "pnl_pct",
                  "daily_return_pct", "max_dd_pct", "realized", "n_positions", "n_trades"]

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print(f"=== Wheel Paper Report {date_str} {time_str} ET ===")
    print(f"{'Engine':<15} {'NAV':>10} {'P&L%':>8} {'Day%':>7} {'MaxDD%':>7} {'Pos':>4} {'Trades':>6}")
    print("-" * 65)
    for r in sorted(rows, key=lambda x: x["pnl_pct"], reverse=True):
        print(f"{r['engine']:<15} ${r['nav']:>9,.0f} {r['pnl_pct']:>+7.2f}% "
              f"{r['daily_return_pct']:>+6.2f}% {r['max_dd_pct']:>+6.2f}% "
              f"{r['n_positions']:>4} {r['n_trades']:>6}")

    print(f"\nLogged {len(rows)} engines to {CSV_PATH}")


if __name__ == "__main__":
    main()
