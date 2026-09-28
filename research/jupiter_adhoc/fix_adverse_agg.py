#!/usr/bin/env python3
"""Fixed aggregator: reads per-date JSON files and produces correct adverse_selection_report.json"""
import json, glob, os
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/adverse_selection_analysis")
REPORT = OUTPUT_DIR / "adverse_selection_report_fixed.json"

# --- Latency impact ---
latency_data = defaultdict(lambda: defaultdict(lambda: {"total_pnl": 0.0, "n_trades": 0, "n_days": 0, "n_wins": 0}))
max_wait_data = defaultdict(lambda: defaultdict(lambda: {"total_pnl": 0.0, "n_trades": 0}))

cards = {"c1": "book_pred", "c5": "raw", "c7": "smooth"}

for f in sorted(OUTPUT_DIR.glob("*_c*_lat*.json")):
    name = f.stem  # e.g. 2026-02-17_c1_lat100
    parts = name.split("_")
    # find card + lat
    card = next((p for p in parts if p.startswith("c") and p[1:].isdigit()), None)
    lat = next((p for p in parts if p.startswith("lat")), None)
    if not card or not lat:
        continue
    try:
        with open(f) as fh:
            r = json.load(fh)
        n = r.get("total_trades") or r.get("summary", {}).get("total_trades") or 0
        pnl = r.get("total_pnl_dollars") or r.get("summary", {}).get("total_pnl_dollars") or 0.0
        wr = r.get("win_rate") or 0.0
        wins = round(n * wr) if n > 0 else 0
        fill_rate = r.get("fill_rate") or 0.0
        avg_pnl = pnl / n if n > 0 else 0.0
        d = latency_data[card][lat]
        d["total_pnl"] += pnl
        d["n_trades"] += n
        d["n_days"] += 1
        d["n_wins"] += wins
        d["fill_rate_sum"] = d.get("fill_rate_sum", 0.0) + fill_rate
    except Exception as e:
        print(f"SKIP {f.name}: {e}")

for f in sorted(OUTPUT_DIR.glob("*_c*_wait_*.json")):
    name = f.stem
    parts = name.split("_")
    card = next((p for p in parts if p.startswith("c") and p[1:].isdigit()), None)
    wait_idx = next((i for i, p in enumerate(parts) if p == "wait"), None)
    if not card or wait_idx is None:
        continue
    wait_label = f"wait_{parts[wait_idx+1]}"
    try:
        with open(f) as fh:
            r = json.load(fh)
        n = r.get("total_trades") or r.get("summary", {}).get("total_trades") or 0
        pnl = r.get("total_pnl_dollars") or r.get("summary", {}).get("total_pnl_dollars") or 0.0
        max_wait_data[card][wait_label]["total_pnl"] += pnl
        max_wait_data[card][wait_label]["n_trades"] += n
    except Exception as e:
        print(f"SKIP {f.name}: {e}")

# --- Build latency summary ---
latency_summary = {}
for card in latency_data:
    latency_summary[card] = {}
    for lat, s in sorted(latency_data[card].items()):
        n = s["n_trades"]
        pnl = s["total_pnl"]
        n_days = s["n_days"]
        wins = s["n_wins"]
        avg_fill = s.get("fill_rate_sum", 0) / n_days if n_days > 0 else 0
        latency_summary[card][lat] = {
            "total_pnl": round(pnl, 2),
            "n_trades": n,
            "n_days": n_days,
            "avg_pnl_per_trade": round(pnl / n, 2) if n > 0 else 0.0,
            "win_rate_pct": round(wins / n * 100, 1) if n > 0 else 0.0,
            "avg_fill_rate_pct": round(avg_fill * 100, 1),
            "pnl_vs_lat0": 0.0,  # fill below
        }
    # compute delta vs lat0
    base = latency_summary[card].get("lat_0ms", {}).get("total_pnl", 0.0)
    for lat in latency_summary[card]:
        latency_summary[card][lat]["pnl_vs_lat0"] = round(latency_summary[card][lat]["total_pnl"] - base, 2)

# --- Build wait summary ---
wait_summary = {}
for card in max_wait_data:
    wait_summary[card] = {}
    for wait, s in sorted(max_wait_data[card].items()):
        n = s["n_trades"]
        pnl = s["total_pnl"]
        wait_summary[card][wait] = {
            "total_pnl": round(pnl, 2),
            "n_trades": n,
            "avg_pnl_per_trade": round(pnl / n, 2) if n > 0 else 0.0,
        }

# --- Print results ---
print("\n=== LATENCY IMPACT (aggregated from per-date JSONs) ===")
for card in sorted(latency_summary):
    print(f"\n  {card}:")
    for lat in sorted(latency_summary[card]):
        s = latency_summary[card][lat]
        print(f"    {lat:12s}: ${s['total_pnl']:>10,.0f} total | {s['n_trades']:5d} trades | "
              f"WR={s['win_rate_pct']:5.1f}% | fill={s['avg_fill_rate_pct']:5.1f}% | "
              f"avg=${s['avg_pnl_per_trade']:>7.2f} | Δ vs 0ms: ${s['pnl_vs_lat0']:>8,.0f}")

print("\n=== MAX WAIT ANALYSIS ===")
for card in sorted(wait_summary):
    print(f"\n  {card}:")
    for wait in sorted(wait_summary[card]):
        s = wait_summary[card][wait]
        print(f"    {wait:12s}: ${s['total_pnl']:>10,.0f} total | {s['n_trades']:5d} trades | avg=${s['avg_pnl_per_trade']:>7.2f}")

# --- Write fixed report ---
report = {
    "timestamp": "2026-04-15T11:54:00Z",
    "source": "fix_adverse_agg.py — reads per-date JSONs, uses correct key names",
    "n_dates_processed": max(s["n_days"] for c in latency_data.values() for s in c.values()) if latency_data else 0,
    "latency_impact": latency_summary,
    "max_wait_analysis": wait_summary,
}
with open(REPORT, "w") as fh:
    json.dump(report, fh, indent=2)
print(f"\nFixed report written to: {REPORT}")
