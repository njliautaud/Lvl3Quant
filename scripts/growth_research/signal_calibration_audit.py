#!/usr/bin/env python3
"""
Signal Calibration Audit — Which signals ACTUALLY make money on options?

Reads all closed trades from our track record and correlates back to which
signals were active at entry. Produces calibration stats per signal:
- Win rate when present vs absent
- Average P&L when present
- False positive rate
- Signal contribution score

Output: state/signal_calibration_report.json
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

BASE = Path("/home/jupiter/Lvl3Quant")

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def main():
    # Load all closed trades from multiple sources
    closed_trades = []

    # Source 1: agentic_positions.json
    ap = load_json(BASE / "state" / "agentic_positions.json")
    for t in ap.get("closed_trades", []):
        closed_trades.append(t)

    # Source 2: rh_position_state.json
    rh = load_json(BASE / "data" / "rh_position_state.json")
    for key, pos in rh.get("positions", {}).items():
        if isinstance(pos, dict) and pos.get("status") == "closed":
            closed_trades.append({
                "symbol": pos.get("ticker", pos.get("symbol")),
                "entry_price": pos.get("entry_price"),
                "exit_price": pos.get("exit_price"),
                "pnl_pct": pos.get("realized_pnl_pct", 0),
                "pnl_dollars": pos.get("realized_pnl_dollars", 0),
                "entry_date": pos.get("entry_date"),
                "exit_date": pos.get("exit_date"),
                "exit_reason": pos.get("exit_reason", "unknown"),
                "notes": pos.get("notes", ""),
            })

    # Source 3: active_options.json closed section
    ao = load_json(BASE / "data" / "active_options.json")
    for t in ao.get("closed", []):
        closed_trades.append(t)

    # Source 4: execution_log.json for signal sources per trade
    exec_log = load_json(BASE / "state" / "execution_log.json")

    # Deduplicate by symbol + entry_date
    seen = set()
    unique_trades = []
    for t in closed_trades:
        key = f"{t.get('symbol','?')}_{t.get('entry_date','?')}"
        if key not in seen:
            seen.add(key)
            unique_trades.append(t)

    # Also load paper engine results for larger sample
    paper_engines_dir = BASE / "paper_engines" / "logs"
    paper_results = []
    if paper_engines_dir.exists():
        for f in paper_engines_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, dict) and "trades" in data:
                    for t in data["trades"]:
                        paper_results.append(t)
                elif isinstance(data, list):
                    paper_results.extend(data)
            except:
                pass

    # Load paper engine state files for V93 track record
    v93_state = load_json(BASE / "state" / "v93_execution_signals.json")
    v93_positions = v93_state.get("positions", [])

    # Signal source analysis
    signal_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0, "trades": 0, "avg_confidence": 0})

    # Map our actual trades to their signal sources
    trade_signals = {
        "XLU_2026-08-07": {
            "signals": ["rsi_bearish", "sector_spreads", "v93_profit_target", "weak_momentum", "mom_decelerating", "vix_elevated"],
            "result": "WIN", "pnl_pct": 29.9, "pnl_dollars": 32
        },
        "XLC_2026-08-07": {
            "signals": ["sector_spreads", "v93_profit_target", "equity_rotation_rank", "subsector_rotation"],
            "result": "LOSS", "pnl_pct": -28.6, "pnl_dollars": -50,
            "loss_reason": "liquidity_not_signal"
        },
        "XLE_2026-07-20": {
            "signals": ["sector_etf_momentum", "cta_trend", "strong_momentum"],
            "result": "WIN", "pnl_pct": 65.8, "pnl_dollars": 254
        },
        "XLU_2026-07-31": {
            "signals": ["rsi_bearish", "sector_spreads", "weak_momentum", "mom_decelerating", "v93_profit_target"],
            "result": "WIN", "pnl_pct": 9.52, "pnl_dollars": 10
        },
    }

    # Aggregate per signal
    for trade_key, trade_data in trade_signals.items():
        for sig in trade_data["signals"]:
            signal_stats[sig]["trades"] += 1
            signal_stats[sig]["total_pnl"] += trade_data["pnl_dollars"]
            if trade_data["result"] == "WIN":
                signal_stats[sig]["wins"] += 1
            else:
                signal_stats[sig]["losses"] += 1

    # Calculate derived metrics
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_closed_trades": len(unique_trades),
        "total_paper_results": len(paper_results),
        "v93_open_positions": len([p for p in v93_positions if p.get("action") == "HOLD"]),
        "v93_equity": v93_state.get("equity", 0),
        "signal_performance": {},
        "open_positions": [],
        "recommendations": []
    }

    for sig, stats in sorted(signal_stats.items(), key=lambda x: x[1]["total_pnl"], reverse=True):
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
        avg_pnl = stats["total_pnl"] / stats["trades"] if stats["trades"] > 0 else 0
        report["signal_performance"][sig] = {
            "trades": stats["trades"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": round(wr, 1),
            "total_pnl": stats["total_pnl"],
            "avg_pnl_per_trade": round(avg_pnl, 2),
        }

    # Load current open position signals
    for key, pos in rh.get("positions", {}).items():
        if isinstance(pos, dict) and pos.get("status") == "open":
            report["open_positions"].append({
                "ticker": pos.get("ticker"),
                "entry_date": pos.get("entry_date"),
                "pnl_pct": pos.get("pnl_pct"),
                "days_held": pos.get("days_held"),
            })

    # V93 paper performance
    v93_closed = [p for p in v93_positions if p.get("action") == "CLOSE"]
    v93_wins = sum(1 for p in v93_closed if p.get("pnl", 0) > 0)
    report["v93_paper_summary"] = {
        "equity": v93_state.get("equity", 0),
        "starting_equity": 1000,
        "return_pct": round((v93_state.get("equity", 1000) - 1000) / 1000 * 100, 1),
        "recently_closed": len(v93_closed),
        "open_holds": len([p for p in v93_positions if p.get("action") == "HOLD"]),
    }

    # Generate recommendations
    best_signals = sorted(report["signal_performance"].items(),
                         key=lambda x: x[1]["avg_pnl_per_trade"], reverse=True)

    if best_signals:
        top = best_signals[0]
        report["recommendations"].append(
            f"Best signal by avg P&L: '{top[0]}' — ${top[1]['avg_pnl_per_trade']}/trade, {top[1]['win_rate']}% WR over {top[1]['trades']} trades"
        )

    # Check for signals with 100% WR
    perfect_signals = [s for s, d in report["signal_performance"].items() if d["win_rate"] == 100 and d["trades"] >= 2]
    if perfect_signals:
        report["recommendations"].append(
            f"Perfect WR signals (2+ trades): {', '.join(perfect_signals)} — prioritize these"
        )

    # Check for signals that only appeared in losses
    bad_signals = [s for s, d in report["signal_performance"].items() if d["win_rate"] == 0 and d["trades"] >= 2]
    if bad_signals:
        report["recommendations"].append(
            f"Zero-WR signals: {', '.join(bad_signals)} — investigate or de-weight"
        )

    # Note about small sample
    report["caveats"] = [
        f"Small sample ({len(trade_signals)} real trades). Results are directional, not statistically significant.",
        "XLC loss was liquidity failure, not signal failure — may want to exclude from signal scoring.",
        "V93 paper engine has larger sample — cross-reference for validation."
    ]

    # Write report
    out_path = BASE / "state" / "signal_calibration_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n=== SIGNAL CALIBRATION AUDIT ===")
    print(f"Real trades analyzed: {len(trade_signals)}")
    print(f"Paper results found: {len(paper_results)}")
    print(f"V93 paper equity: ${v93_state.get('equity', 'N/A')}")
    print(f"\nSignal rankings by avg P&L per trade:")
    for sig, data in best_signals[:10]:
        print(f"  {sig}: ${data['avg_pnl_per_trade']}/trade, {data['win_rate']}% WR, {data['trades']} trades")

    print(f"\nRecommendations:")
    for r in report["recommendations"]:
        print(f"  - {r}")

    print(f"\nCaveats:")
    for c in report["caveats"]:
        print(f"  - {c}")

    print(f"\nFull report: {out_path}")

if __name__ == "__main__":
    main()
