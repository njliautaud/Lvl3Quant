#!/usr/bin/env python3
"""
Aggregate Rust fill_sim_cli results from all configs and thresholds.
Run on Jupiter: python3 /home/jupiter/aggregate_rust_results.py
"""
import json
import glob
import os
from pathlib import Path

OUT_BASE = Path("/home/jupiter/lvl3quant/production/results/rust_sim")
CONFIGS = ["h15000_t8_s8", "h15000_t6_s8", "h20000_t8_s7", "h30000_t4_s7"]
THRESHOLDS = ["0.7", "0.5", "0.9"]
NUM_DATES = 27

results_table = []

for cfg in CONFIGS:
    for thresh in THRESHOLDS:
        out_dir = OUT_BASE / f"{cfg}_t{thresh.replace('.', '')}"
        json_files = sorted(out_dir.glob("*.json")) if out_dir.exists() else []

        if not json_files:
            print(f"\n{cfg} thresh={thresh}: NO RESULTS")
            results_table.append({
                "cfg": cfg, "thresh": thresh, "days": 0,
                "total_pnl": 0, "avg_pnl": 0, "total_trades": 0,
                "fill_rate": 0, "win_rate": 0
            })
            continue

        total_pnl = 0.0
        total_trades = 0
        total_filled = 0
        total_posted = 0
        total_wins = 0
        days_ok = 0
        days_with_trades = 0

        for jf in json_files:
            try:
                with open(jf) as f:
                    d = json.load(f)
                s = d["summary"]
                total_pnl += s["total_pnl_dollars"]
                total_trades += s["total_trades"]
                total_filled += s["total_filled"]
                total_posted += s["total_posted"]
                days_ok += 1
                if s["total_trades"] > 0:
                    days_with_trades += 1
                    total_wins += int(s["win_rate"] * s["total_trades"])
            except Exception as e:
                print(f"  ERR {jf.name}: {e}")

        avg_pnl = total_pnl / days_ok if days_ok else 0
        fill_rate = total_filled / total_posted if total_posted else 0
        win_rate = total_wins / total_trades if total_trades else 0

        results_table.append({
            "cfg": cfg, "thresh": thresh,
            "days": days_ok, "days_with_trades": days_with_trades,
            "total_pnl": total_pnl, "avg_pnl": avg_pnl,
            "total_trades": total_trades,
            "fill_rate": fill_rate, "win_rate": win_rate
        })

        print(f"\n{cfg} thresh={thresh}: {days_ok}/{NUM_DATES} days ({days_with_trades} with trades)")
        print(f"  Total PnL:      ${total_pnl:+,.0f}")
        print(f"  Avg PnL/day:    ${avg_pnl:+,.0f}")
        print(f"  Total trades:   {total_trades}")
        print(f"  Fill rate:      {fill_rate*100:.1f}%")
        print(f"  Win rate:       {win_rate*100:.1f}%")

# Summary ranking
print("\n" + "=" * 60)
print("RANKING BY TOTAL PnL (completed combos only):")
print("=" * 60)
ranked = sorted([r for r in results_table if r["days"] > 0],
                key=lambda x: x["total_pnl"], reverse=True)
for i, r in enumerate(ranked, 1):
    print(f"#{i:2d}: {r['cfg']} t{r['thresh']:3s}  "
          f"Total=${r['total_pnl']:+,.0f}  "
          f"Avg/day=${r['avg_pnl']:+,.0f}  "
          f"({r['days']}/{NUM_DATES}d, fill={r['fill_rate']*100:.0f}%, wr={r['win_rate']*100:.0f}%)")
