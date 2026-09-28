#!/usr/bin/env python3
"""
Paper Trading Dashboard
Reads state/history files and prints formatted performance report.

Usage:
    python3 paper_dashboard.py
    python3 paper_dashboard.py --detailed    # Show all closed trades
"""

import json
import csv
import os
import sys
from datetime import datetime
from typing import Optional
import numpy as np

STATE_FILE = "/home/jupiter/Lvl3Quant/data/paper_positions.json"
HISTORY_FILE = "/home/jupiter/Lvl3Quant/data/paper_trade_history.csv"

STRATEGY_LABELS = {
    "multi_tf_dual_signal_l": "Multi-TF Dual Signal L",
    "rsi_divergence_c": "RSI Divergence C",
    "dual_signal_d": "Dual Signal D",
    "bond_yield_signal": "Bond Yield Signal",
    "iv_rv_gap_entry": "IV-RV Gap Entry",
    "liquidity_signal": "Liquidity Signal",
    "consecutive_dip": "Consecutive Dip",
    "earnings_surprise_pead": "Earnings Surprise (PEAD)",
    "sector_etf_rotation": "Sector ETF Rotation",
    "risk_parity": "Risk Parity",
    "factor_rotation": "Factor Rotation",
}

GROUP_MAP = {
    "multi_tf_dual_signal_l": "A",
    "rsi_divergence_c": "A",
    "dual_signal_d": "A",
    "bond_yield_signal": "A",
    "iv_rv_gap_entry": "A",
    "liquidity_signal": "A",
    "consecutive_dip": "A",
    "earnings_surprise_pead": "A",
    "sector_etf_rotation": "B",
    "risk_parity": "B",
    "factor_rotation": "B",
}


def load_state() -> Optional[dict]:
    if not os.path.exists(STATE_FILE):
        return None
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    rows = []
    with open(HISTORY_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_metrics(closed_trades: list) -> dict:
    """Compute performance metrics from closed trades."""
    if not closed_trades:
        return {
            "total_return": 0, "sharpe": None, "win_rate": None,
            "avg_trade": None, "max_dd": None, "trade_count": 0,
            "avg_days": None, "best_trade": None, "worst_trade": None,
            "profit_factor": None,
        }

    pnls = [t.get("pnl", 0) for t in closed_trades]
    n = len(pnls)

    total_return = sum(pnls)
    avg_trade = np.mean(pnls) if pnls else 0
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / n if n > 0 else None

    # Sharpe (annualized, assuming ~1 trade/week)
    sharpe = None
    if n >= 5:
        std = np.std(pnls, ddof=1)
        if std > 0:
            sharpe = (np.mean(pnls) / std) * np.sqrt(52)

    # Max drawdown (cumulative PnL)
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0

    # Avg hold days
    days = [float(t.get("days_held", 0)) for t in closed_trades if t.get("days_held") is not None]
    avg_days = np.mean(days) if days else None

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    return {
        "total_return": round(total_return, 2),
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "win_rate": round(win_rate * 100, 1) if win_rate is not None else None,
        "avg_trade": round(avg_trade, 2) if avg_trade is not None else None,
        "max_dd": round(max_dd, 2),
        "trade_count": n,
        "avg_days": round(avg_days, 1) if avg_days is not None else None,
        "best_trade": round(max(pnls), 2) if pnls else None,
        "worst_trade": round(min(pnls), 2) if pnls else None,
        "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
    }


def fmt(val, fmt_str="", default="---"):
    if val is None:
        return default
    if fmt_str:
        return format(val, fmt_str)
    return str(val)


def print_dashboard(state: dict, detailed: bool = False):
    print()
    print("=" * 90)
    print("  UNIFIED PAPER TRADING DASHBOARD")
    print(f"  Last Run: {state.get('last_run', 'Never')}")
    print(f"  Created:  {state.get('created', 'Unknown')}")
    print("=" * 90)

    positions = state.get("positions", {})
    closed = state.get("closed_trades", {})

    all_strats = list(STRATEGY_LABELS.keys())

    # ── Performance Comparison Table ──
    print("\n  STRATEGY PERFORMANCE COMPARISON")
    print("  " + "-" * 86)
    hdr = f"  {'Strategy':<28s} {'Return':>8s} {'Sharpe':>7s} {'WR%':>6s} {'PF':>6s} {'AvgTrd':>8s} {'MaxDD':>8s} {'Trades':>6s} {'Pos':>4s}"
    print(hdr)
    print("  " + "-" * 86)

    group_a_total = 0
    group_b_total = 0

    for strat_name in all_strats:
        label = STRATEGY_LABELS.get(strat_name, strat_name)
        group = GROUP_MAP.get(strat_name, "?")
        trades = closed.get(strat_name, [])
        active = positions.get(strat_name, [])
        m = compute_metrics(trades)

        ret = fmt(m["total_return"], "+.2f")
        sh = fmt(m["sharpe"], ".2f")
        wr = fmt(m["win_rate"], ".0f")
        pf = fmt(m["profit_factor"], ".2f")
        at = fmt(m["avg_trade"], "+.2f")
        md = fmt(m["max_dd"], ".2f")
        tc = str(m["trade_count"])
        ap = str(len(active))

        # Group separator
        if strat_name == "sector_etf_rotation":
            print("  " + "-" * 86)
            print(f"  {'GROUP A TOTAL':<28s} {group_a_total:>+8.2f}")
            print("  " + "-" * 86)

        print(f"  {label:<28s} ${ret:>7s} {sh:>7s} {wr:>5s}% {pf:>6s} ${at:>7s} ${md:>7s} {tc:>6s} {ap:>4s}")

        if group == "A":
            group_a_total += m["total_return"]
        else:
            group_b_total += m["total_return"]

    print("  " + "-" * 86)
    print(f"  {'GROUP B TOTAL':<28s} ${group_b_total:>+7.2f}")
    print("  " + "-" * 86)
    grand_total = group_a_total + group_b_total
    print(f"  {'GRAND TOTAL':<28s} ${grand_total:>+7.2f}")
    print("  " + "=" * 86)

    # ── Active Positions ──
    any_active = any(len(positions.get(s, [])) > 0 for s in all_strats)
    if any_active:
        print("\n  ACTIVE POSITIONS")
        print("  " + "-" * 70)
        print(f"  {'Strategy':<25s} {'Ticker':>6s} {'Shares':>6s} {'Entry':>8s} {'Date':>12s}")
        print("  " + "-" * 70)
        for strat_name in all_strats:
            for pos in positions.get(strat_name, []):
                label = STRATEGY_LABELS.get(strat_name, strat_name)[:25]
                print(f"  {label:<25s} {pos['ticker']:>6s} {pos['shares']:>6d} ${pos['entry_price']:>7.2f} {pos['entry_date']:>12s}")
        print()
    else:
        print("\n  No active positions.\n")

    # ── Recent Signals ──
    signals = state.get("signals_today", [])
    if signals:
        print("  MOST RECENT SIGNALS")
        print("  " + "-" * 70)
        for s in signals:
            label = STRATEGY_LABELS.get(s["strategy"], s["strategy"])
            opt = " *OPT*" if s.get("options_candidate") else ""
            print(f"  {label:<25s} | {s['ticker']:>5s} | {s['reason']}{opt}")
        print()

    # ── Detailed Trade Log ──
    if detailed:
        print("\n  CLOSED TRADE LOG")
        print("  " + "-" * 90)
        print(f"  {'Strategy':<22s} {'Ticker':>6s} {'Entry':>8s} {'Exit':>8s} {'PnL':>8s} {'Days':>5s} {'Entry Date':>12s} {'Exit Date':>12s}")
        print("  " + "-" * 90)
        for strat_name in all_strats:
            for t in closed.get(strat_name, []):
                label = STRATEGY_LABELS.get(strat_name, strat_name)[:22]
                print(f"  {label:<22s} {t['ticker']:>6s} ${t['entry_price']:>7.2f} ${t.get('exit_price',0):>7.2f} ${t.get('pnl',0):>+7.2f} {t.get('days_held','?'):>5s} {t['entry_date']:>12s} {t.get('exit_date',''):>12s}")
        print()


def main():
    state = load_state()
    if state is None:
        print("\n  No paper trading state found.")
        print(f"  Run the engine first: python3 unified_paper_engine.py")
        print(f"  State file: {STATE_FILE}\n")

        # Initialize empty state so we show the clean template
        state = {
            "created": datetime.now().isoformat(),
            "last_run": "Never",
            "positions": {},
            "closed_trades": {},
            "signals_today": [],
        }

    detailed = "--detailed" in sys.argv
    print_dashboard(state, detailed=detailed)


if __name__ == "__main__":
    main()
