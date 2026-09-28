#!/usr/bin/env python3
"""
Continuous Evolution Check — weekly AVO re-evolution trigger
==============================================================
Runs weekly. Checks ALL paper engines with 5+ trades.
If any strategy's paper win rate falls below its lockbox WR by >15%,
flags it for AVO re-evolution.

Outputs to state/evolution_priorities.json so the autonomy_inject
pulse can pick up the worst performer and trigger an AVO run.

Cron: 0 6 * * 0  (every Sunday 6 AM ET)
"""
import json
import os
from datetime import datetime
from pathlib import Path

import pytz

ET = pytz.timezone("US/Eastern")
BASE = Path("/home/jupiter/Lvl3Quant")

# Lockbox benchmarks (from RUN_HISTORY.md — the bar each strategy must beat)
LOCKBOX_BENCHMARKS = {
    "options_execution_avo": {"sharpe": 2.48, "wr": 0.40, "trades": 70},
    "sentiment_contrarian_avo": {"sharpe": 2.73, "wr": 0.50, "trades": 67},
    "vol_regime_mean_revert": {"sharpe": 1.18, "wr": 0.25, "trades": 16},
    "cross_asset_macro": {"sharpe": 2.49, "wr": 0.45, "trades": 59},
    "vol_compression_avo": {"sharpe": 1.28, "wr": 0.50, "trades": 10},
    "put_call_contrarian": {"sharpe": 3.78, "wr": 0.50, "trades": 52},
    "gold_bond_divergence": {"sharpe": 3.83, "wr": 0.50, "trades": 21},
    "treasury_curve_steepener": {"sharpe": 4.09, "wr": 0.50, "trades": 17},
    "breadth_momentum_regime": {"sharpe": 2.12, "wr": 0.44, "trades": 18},
    "insider_momentum": {"sharpe": 3.33, "wr": 0.44, "trades": 18},
    "flow_reversal_2x": {"sharpe": 1.40, "wr": 0.40, "trades": 15},
    "credit_spread_momentum": {"sharpe": 7.78, "wr": 0.50, "trades": 112},
    "calendar_momentum": {"sharpe": 2.58, "wr": 0.50, "trades": 14},
    "size_rotation": {"sharpe": 2.30, "wr": 0.52, "trades": 21},
    "macro_regime_rotation": {"sharpe": 3.74, "wr": 0.41, "trades": 29},
    "sector_rotation_wf": {"sharpe": 1.00, "wr": 0.38, "trades": 21},
    "trend_dip_reversion": {"sharpe": 0.95, "wr": 0.26, "trades": 19},
    "options_overlay": {"sharpe": None, "wr": 0.50, "trades": None},
}

# AVO target mapping (paper engine name -> AVO target name)
AVO_TARGETS = {
    "options_execution_avo": "options_execution",
    "sentiment_contrarian_avo": "sentiment_contrarian",
    "vol_regime_mean_revert": "vol_regime_mean_revert",
    "cross_asset_macro": "cross_asset_macro",
    "vol_compression_avo": "vol_compression",
    "put_call_contrarian": "put_call_contrarian",
    "gold_bond_divergence": "gold_bond_divergence",
    "treasury_curve_steepener": "treasury_curve_steepener",
    "breadth_momentum_regime": "breadth_momentum_regime",
    "insider_momentum": "insider_momentum",
    "credit_spread_momentum": "credit_spread_momentum",
    "calendar_momentum": "calendar_momentum",
    "size_rotation": "size_rotation",
    "macro_regime_rotation": "macro_regime_rotation",
    "trend_dip_reversion": "trend_dip_reversion",
}


def load_paper_performance():
    """Load performance from all paper engine state files."""
    results = {}
    state_dir = BASE / "paper_engines" / "state"

    for f in state_dir.iterdir():
        if not f.name.endswith("_state.json"):
            continue

        engine_name = f.name.replace("_state.json", "").replace("_paper", "")

        try:
            with open(f) as fh:
                data = json.load(fh)

            trades = data.get("closed_trades", data.get("trades", []))
            if not trades:
                continue

            n = len(trades)
            wins = sum(1 for t in trades
                       if t.get("pnl", t.get("pnl_pct", 0)) > 0)
            wr = wins / n if n > 0 else 0

            total_pnl = sum(t.get("pnl", t.get("pnl_pct", 0)) for t in trades)

            # Get date range
            dates = [t.get("entry_date", t.get("exit_date", "")) for t in trades]
            dates = [d for d in dates if d]

            results[engine_name] = {
                "trades": n,
                "wins": wins,
                "losses": n - wins,
                "wr": round(wr, 3),
                "total_pnl": round(total_pnl, 2),
                "first_trade": min(dates) if dates else None,
                "last_trade": max(dates) if dates else None,
            }
        except Exception as e:
            continue

    return results


def assess_drift():
    """Compare paper performance to lockbox benchmarks."""
    paper = load_paper_performance()
    priorities = []

    for engine, perf in paper.items():
        if perf["trades"] < 5:
            continue  # not enough data

        benchmark = LOCKBOX_BENCHMARKS.get(engine)
        if not benchmark:
            continue

        lockbox_wr = benchmark.get("wr", 0.50)
        paper_wr = perf["wr"]
        wr_gap = lockbox_wr - paper_wr

        avo_target = AVO_TARGETS.get(engine)

        drift_status = "OK"
        urgency = 0

        if wr_gap > 0.25:
            drift_status = "CRITICAL"
            urgency = 3
        elif wr_gap > 0.15:
            drift_status = "WARNING"
            urgency = 2
        elif wr_gap > 0.05:
            drift_status = "WATCH"
            urgency = 1

        # Bonus urgency if ALL trades are losses
        if perf["wins"] == 0 and perf["trades"] >= 3:
            urgency = max(urgency, 3)
            drift_status = "CRITICAL"

        priorities.append({
            "engine": engine,
            "avo_target": avo_target,
            "paper_trades": perf["trades"],
            "paper_wr": perf["wr"],
            "paper_record": f"{perf['wins']}W/{perf['losses']}L",
            "lockbox_wr": lockbox_wr,
            "wr_gap": round(wr_gap, 3),
            "drift_status": drift_status,
            "urgency": urgency,
            "total_pnl": perf["total_pnl"],
            "date_range": f"{perf['first_trade']} to {perf['last_trade']}",
        })

    # Sort by urgency (highest first), then by wr_gap
    priorities.sort(key=lambda x: (-x["urgency"], -x["wr_gap"]))

    return priorities


def run():
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    print(f"[{ts}] Continuous Evolution Check — running...")

    priorities = assess_drift()

    output = {
        "timestamp": ts,
        "total_engines_checked": len(priorities),
        "critical": [p for p in priorities if p["drift_status"] == "CRITICAL"],
        "warning": [p for p in priorities if p["drift_status"] == "WARNING"],
        "watch": [p for p in priorities if p["drift_status"] == "WATCH"],
        "ok": [p for p in priorities if p["drift_status"] == "OK"],
        "recommended_reevolution": [],
    }

    # Top 3 worst performers with AVO targets → recommend for re-evolution
    for p in priorities[:3]:
        if p["avo_target"] and p["urgency"] >= 2:
            output["recommended_reevolution"].append({
                "avo_target": p["avo_target"],
                "reason": f"{p['paper_record']} paper ({p['paper_wr']:.0%} WR vs {p['lockbox_wr']:.0%} lockbox)",
                "urgency": p["drift_status"],
            })

    # Save
    out_file = BASE / "state" / "evolution_priorities.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVOLUTION PRIORITY REPORT — {ts}")
    print(f"{'='*60}")

    for p in priorities:
        status_emoji = {"CRITICAL": "🔴", "WARNING": "🟡", "WATCH": "🟠", "OK": "🟢"}[p["drift_status"]]
        print(f"  {status_emoji} {p['engine']:30s} | {p['paper_record']:8s} "
              f"({p['paper_wr']:.0%} WR) | lockbox {p['lockbox_wr']:.0%} | "
              f"gap {p['wr_gap']:+.0%} | {p['drift_status']}")

    if output["recommended_reevolution"]:
        print(f"\n🔄 RECOMMENDED FOR RE-EVOLUTION:")
        for r in output["recommended_reevolution"]:
            print(f"   → {r['avo_target']}: {r['reason']}")
    else:
        print(f"\n✅ No strategies need immediate re-evolution.")

    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    run()
