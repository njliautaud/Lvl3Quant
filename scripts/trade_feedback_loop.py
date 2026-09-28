#!/usr/bin/env python3
"""
Trade Feedback Loop (HC #814 R3)
After every closed trade, update signal performance tracking and re-weight.
"""
import json
import os
from datetime import datetime

DATA_DIR = "/home/jupiter/Lvl3Quant/data"
ACTIVE_FILE = os.path.join(DATA_DIR, "active_options.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "signal_performance_tracker.json")
BACKTEST_FILE = os.path.join(DATA_DIR, "signal_backtest_results.json")

def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default or {}

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def analyze_closed_trades():
    """Analyze all closed trades and compute rolling performance metrics."""
    active = load_json(ACTIVE_FILE, {"closed": []})
    closed = active.get("closed", [])

    if not closed:
        print("No closed trades to analyze")
        return

    # Compute rolling metrics
    wins = [t for t in closed if t.get("pnl_pct", 0) > 0]
    losses = [t for t in closed if t.get("pnl_pct", 0) <= 0]

    n_total = len(closed)
    n_wins = len(wins)
    wr = n_wins / n_total if n_total > 0 else 0

    avg_win = sum(t.get("pnl_dollars", 0) for t in wins) / max(n_wins, 1)
    avg_loss = sum(abs(t.get("pnl_dollars", 0)) for t in losses) / max(len(losses), 1)

    total_pnl = sum(t.get("pnl_dollars", 0) for t in closed)

    pf = (sum(t.get("pnl_dollars", 0) for t in wins) /
          max(sum(abs(t.get("pnl_dollars", 0)) for t in losses), 1)) if losses else float('inf')

    # Analyze by exit reason
    exit_reasons = {}
    for t in closed:
        reason = t.get("exit_reason", "unknown")
        if reason not in exit_reasons:
            exit_reasons[reason] = {"n": 0, "pnl": 0, "wins": 0}
        exit_reasons[reason]["n"] += 1
        exit_reasons[reason]["pnl"] += t.get("pnl_dollars", 0)
        if t.get("pnl_pct", 0) > 0:
            exit_reasons[reason]["wins"] += 1

    # Analyze by hold period
    hold_analysis = {"0d": {"n": 0, "wr": 0, "pnl": 0},
                     "1-2d": {"n": 0, "wr": 0, "pnl": 0},
                     "3+d": {"n": 0, "wr": 0, "pnl": 0}}

    for t in closed:
        days = t.get("days_held", 0)
        if days == 0:
            bucket = "0d"
        elif days <= 2:
            bucket = "1-2d"
        else:
            bucket = "3+d"
        hold_analysis[bucket]["n"] += 1
        hold_analysis[bucket]["pnl"] += t.get("pnl_dollars", 0)
        if t.get("pnl_pct", 0) > 0:
            hold_analysis[bucket]["wr"] += 1

    for bucket in hold_analysis:
        n = hold_analysis[bucket]["n"]
        if n > 0:
            hold_analysis[bucket]["wr"] = hold_analysis[bucket]["wr"] / n

    # 5-trade rolling windows
    rolling_5 = []
    for i in range(max(0, n_total - 5), n_total):
        window = closed[max(0, i-4):i+1]
        w_wr = sum(1 for t in window if t.get("pnl_pct", 0) > 0) / len(window)
        w_pnl = sum(t.get("pnl_dollars", 0) for t in window)
        rolling_5.append({"end_idx": i, "wr": w_wr, "pnl": w_pnl})

    # Build feedback report
    report = {
        "updated": datetime.now().isoformat(),
        "total_trades": n_total,
        "win_rate": round(wr, 3),
        "avg_win_dollars": round(avg_win, 2),
        "avg_loss_dollars": round(avg_loss, 2),
        "profit_factor": round(pf, 2),
        "total_pnl_dollars": round(total_pnl, 2),
        "risk_reward_ratio": round(avg_win / max(avg_loss, 1), 2),
        "by_exit_reason": exit_reasons,
        "by_hold_period": hold_analysis,
        "rolling_5_trade": rolling_5[-3:] if rolling_5 else [],
        "lessons": {
            "best_exit": max(exit_reasons.items(), key=lambda x: x[1]["pnl"] / max(x[1]["n"], 1))[0] if exit_reasons else "none",
            "worst_exit": min(exit_reasons.items(), key=lambda x: x[1]["pnl"] / max(x[1]["n"], 1))[0] if exit_reasons else "none",
            "best_hold": max(hold_analysis.items(), key=lambda x: x[1]["wr"])[0] if hold_analysis else "none",
            "gtc_tp_wr": sum(1 for t in closed if "tp" in t.get("exit_reason", "")) / max(sum(1 for t in closed if "tp" in t.get("exit_reason", "")), 1),
        },
        "recommendations": []
    }

    # Generate recommendations
    if report["risk_reward_ratio"] < 1.0:
        report["recommendations"].append("Risk/reward below 1.0 — avg loss exceeds avg win. Tighten SL or widen TP.")

    if hold_analysis.get("0d", {}).get("n", 0) > 0:
        d0_wr = hold_analysis["0d"]["wr"]
        if d0_wr < 0.4:
            report["recommendations"].append(f"Same-day exits have {d0_wr:.0%} WR — avoid same-day trades or require extra confirmation.")

    if hold_analysis.get("3+d", {}).get("n", 0) > 0:
        d3_wr = hold_analysis["3+d"]["wr"]
        if d3_wr > 0.6:
            report["recommendations"].append(f"3+ day holds have {d3_wr:.0%} WR — bias toward longer holds.")

    # GTC TP analysis
    tp_trades = [t for t in closed if "tp" in t.get("exit_reason", "").lower()]
    if tp_trades:
        tp_wr = len([t for t in tp_trades if t.get("pnl_pct", 0) > 0]) / len(tp_trades)
        report["recommendations"].append(f"GTC TP orders: {tp_wr:.0%} WR on {len(tp_trades)} trades — {'keep using' if tp_wr > 0.7 else 'review TP level'}.")

    save_json(FEEDBACK_FILE, report)

    # Print summary
    print(f"\n=== TRADE FEEDBACK LOOP ===")
    print(f"Total trades: {n_total} | WR: {wr:.1%} | PF: {pf:.2f}")
    print(f"Avg win: ${avg_win:.2f} | Avg loss: ${avg_loss:.2f} | R:R = {report['risk_reward_ratio']}")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"\nBy hold period:")
    for bucket, stats in hold_analysis.items():
        if stats["n"] > 0:
            print(f"  {bucket}: {stats['n']} trades, {stats['wr']:.0%} WR, ${stats['pnl']:.0f} P&L")
    print(f"\nRecommendations:")
    for rec in report["recommendations"]:
        print(f"  - {rec}")

    return report


if __name__ == "__main__":
    analyze_closed_trades()
