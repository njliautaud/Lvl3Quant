#!/usr/bin/env python3
"""
Re-aggregate adverse selection results from existing JSON files.
Fixes key mismatch: script used n_trades/total_pnl but fill_sim outputs total_trades/total_pnl_dollars.
"""
import json
import glob
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/adverse_selection_analysis")
REPORT_OUT = OUTPUT_DIR / "adverse_selection_report_fixed.json"

CARDS = ["c1", "c5", "c7"]
LAT_LABELS = ["lat_0ms", "lat_10ms", "lat_50ms", "lat_100ms", "lat_200ms", "lat_500ms"]
WAIT_LABELS = ["wait_5", "wait_10", "wait_20", "wait_50", "wait_100", "wait_inf"]

def read_file(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return None

def parse_lat_file(path):
    """Extract correct fields from a fill_sim output JSON."""
    d = read_file(path)
    if not d:
        return None
    # Try multiple key names
    n = (d.get("total_trades") or d.get("n_trades") or
         d.get("summary", {}).get("total_trades") or d.get("summary", {}).get("n_trades") or 0)
    pnl = (d.get("total_pnl_dollars") or d.get("total_pnl") or
           d.get("summary", {}).get("total_pnl_dollars") or d.get("summary", {}).get("total_pnl") or 0.0)
    wr = d.get("win_rate") or d.get("summary", {}).get("win_rate") or 0.0
    avg_pnl = (pnl / n) if n > 0 else 0.0
    fill_rate = d.get("fill_rate") or d.get("summary", {}).get("fill_rate") or 0.0
    return {"n_trades": n, "total_pnl": round(pnl, 2), "win_rate": round(wr*100 if wr<=1 else wr,1),
            "avg_pnl_per_trade": round(avg_pnl, 2), "fill_rate": round(fill_rate, 4)}

# --- Latency impact ---
lat_results = {c: {l: {"n_trades":0,"total_pnl":0.0,"win_rates":[],"avg_pnls":[],"fill_rates":[]} for l in LAT_LABELS} for c in CARDS}
lat_counts = {c: 0 for c in CARDS}

for f in sorted(OUTPUT_DIR.glob("*_lat*.json")):
    stem = f.stem  # e.g. "2025-12-01_c1_lat0"
    parts = stem.split("_")
    if len(parts) < 3:
        continue
    # Find card
    card = None
    for c in CARDS:
        if f"_{c}_" in stem:
            card = c
            break
    if not card:
        continue
    # Find latency
    lat_part = parts[-1]  # e.g. "lat0" or "lat100"
    lat_label = f"lat_{lat_part[3:]}ms" if lat_part.startswith("lat") else None
    if not lat_label or lat_label not in LAT_LABELS:
        continue

    r = parse_lat_file(f)
    if r:
        lat_results[card][lat_label]["n_trades"] += r["n_trades"]
        lat_results[card][lat_label]["total_pnl"] += r["total_pnl"]
        if r["n_trades"] > 0:
            lat_results[card][lat_label]["win_rates"].append(r["win_rate"])
            lat_results[card][lat_label]["avg_pnls"].append(r["avg_pnl_per_trade"])
            lat_results[card][lat_label]["fill_rates"].append(r["fill_rate"])

# --- Wait analysis ---
wait_results = {c: {l: {"n_trades":0,"total_pnl":0.0} for l in WAIT_LABELS} for c in CARDS}

for f in sorted(OUTPUT_DIR.glob("*_wait_*.json")):
    stem = f.stem
    card = None
    for c in CARDS:
        if f"_{c}_" in stem:
            card = c
            break
    if not card:
        continue
    # Find wait label
    wait_label = None
    for w in WAIT_LABELS:
        if stem.endswith(f"_{w}"):
            wait_label = w
            break
    if not wait_label:
        continue
    r = parse_lat_file(f)
    if r:
        wait_results[card][wait_label]["n_trades"] += r["n_trades"]
        wait_results[card][wait_label]["total_pnl"] += r["total_pnl"]

# --- Compute summary ---
latency_summary = {}
for card in CARDS:
    latency_summary[card] = {}
    for lat_label in LAT_LABELS:
        d = lat_results[card][lat_label]
        n = d["n_trades"]
        pnl = d["total_pnl"]
        avg_wr = sum(d["win_rates"]) / len(d["win_rates"]) if d["win_rates"] else 0.0
        avg_fill = sum(d["fill_rates"]) / len(d["fill_rates"]) if d["fill_rates"] else 0.0
        latency_summary[card][lat_label] = {
            "total_pnl": round(pnl, 2),
            "n_trades": n,
            "avg_pnl_per_trade": round(pnl/n, 2) if n > 0 else 0.0,
            "avg_win_rate_pct": round(avg_wr, 1),
            "avg_fill_rate": round(avg_fill, 4),
            "pnl_vs_0ms": round(pnl - latency_summary.get(card, {}).get("lat_0ms", {}).get("total_pnl", pnl), 2),
        }

wait_summary = {}
for card in CARDS:
    wait_summary[card] = {}
    for wl in WAIT_LABELS:
        d = wait_results[card][wl]
        n = d["n_trades"]
        pnl = d["total_pnl"]
        wait_summary[card][wl] = {
            "total_pnl": round(pnl, 2),
            "n_trades": n,
            "avg_pnl_per_trade": round(pnl/n, 2) if n > 0 else 0.0,
        }

report = {
    "latency_impact": latency_summary,
    "max_wait_analysis": wait_summary,
}

with open(REPORT_OUT, "w") as f:
    json.dump(report, f, indent=2)

# Print summary table
print("\n=== LATENCY IMPACT SUMMARY ===")
print(f"{'Card':<6}  {'Latency':<12}  {'Trades':>7}  {'TotalPnL':>10}  {'AvgPnL':>8}  {'WinRate':>8}  {'FillRate':>9}")
print("-" * 75)
for card in CARDS:
    for lat in LAT_LABELS:
        d = latency_summary[card][lat]
        print(f"{card:<6}  {lat:<12}  {d['n_trades']:>7}  {d['total_pnl']:>10.2f}  {d['avg_pnl_per_trade']:>8.2f}  {d['avg_win_rate_pct']:>7.1f}%  {d['avg_fill_rate']:>9.4f}")

print("\n=== MAX WAIT ANALYSIS ===")
print(f"{'Card':<6}  {'MaxWait':<12}  {'Trades':>7}  {'TotalPnL':>10}  {'AvgPnL':>8}")
print("-" * 55)
for card in CARDS:
    for wl in WAIT_LABELS:
        d = wait_summary[card][wl]
        print(f"{card:<6}  {wl:<12}  {d['n_trades']:>7}  {d['total_pnl']:>10.2f}  {d['avg_pnl_per_trade']:>8.2f}")

print(f"\nFixed report written to: {REPORT_OUT}")
