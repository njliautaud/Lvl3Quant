#!/usr/bin/env python3
"""
Time-of-Day + Exit-Reason Decomposition for Chase Config Results
================================================================
Analyzes the existing chase fill-sim results from sim_20260426_110502/
to identify WHEN losses cluster and which exit reasons dominate.

Goals:
  1. Time-of-day breakdown (30-min buckets) of WR + PnL + MFE/MAE
  2. Exit-reason distribution per model/threshold
  3. MFE-to-fill-latency analysis (does delayed fill kill us?)
  4. Per-day results to see if any day is structurally different
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

SIM_DIR = Path("/home/jupiter/Lvl3Quant/execution/results/fifo_validation/sim_20260426_110502")
OUT_PATH = Path("/home/jupiter/Lvl3Quant/execution/results/fifo_validation/sim_20260426_110502/TOD_DECOMPOSITION.json")
OUT_MD   = Path("/home/jupiter/Lvl3Quant/execution/results/fifo_validation/sim_20260426_110502/TOD_DECOMPOSITION.md")

MODELS = ["mamba_v7", "cnn_mamba_v2"]
THRESHOLDS = [1, 2, 3]
DATES = ["20260302", "20260303", "20260304", "20260305"]


def ns_to_et_minute(ns):
    """Convert ns timestamp to ET minute-of-day (0..1440)."""
    dt_utc = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    # Mar 8 2026 was DST start. All test dates Mar 2-5 are PRE-DST → UTC-5
    et_offset = -5
    et = dt_utc + timedelta(hours=et_offset)
    return et.hour * 60 + et.minute


def load_trades(model, threshold, date):
    p = SIM_DIR / f"{model}_z{threshold}_{date}.json"
    if not p.exists():
        return []
    with open(p) as f:
        d = json.load(f)
    return d.get("trades", []), d


def tod_bucket(minute, bucket_min=30):
    """Bucket minute-of-day to 30-min slot."""
    b = (minute // bucket_min) * bucket_min
    h = b // 60
    m = b % 60
    return f"{h:02d}:{m:02d}"


def analyze(model, threshold):
    """Aggregate trades across all dates for a model/threshold pair."""
    all_trades = []
    daily = {}
    for date in DATES:
        trades, summary = load_trades(model, threshold, date)
        if not trades:
            continue
        for t in trades:
            t["_date"] = date
        all_trades.extend(trades)
        pnls = [t.get("pnl_dollars", 0) for t in trades]
        daily[date] = {
            "n": len(trades),
            "pnl": float(sum(pnls)),
            "wr": float(np.mean([p > 0 for p in pnls])) if pnls else 0,
            "fill_rate": summary.get("fill_rate", 0),
            "n_signals": summary.get("total_signals", 0),
        }

    if not all_trades:
        return None

    # Time-of-day buckets
    tod = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0, "mfe": 0.0, "mae": 0.0, "lat_ms": 0.0})
    for t in all_trades:
        bucket = tod_bucket(ns_to_et_minute(t["fill_time_ns"]))
        tod[bucket]["n"] += 1
        tod[bucket]["pnl"] += t.get("pnl_dollars", 0)
        tod[bucket]["wins"] += int(t.get("pnl_dollars", 0) > 0)
        tod[bucket]["mfe"] += t.get("mfe_ticks", 0)
        tod[bucket]["mae"] += abs(t.get("mae_ticks", 0))
        tod[bucket]["lat_ms"] += t.get("fill_latency_ns", 0) / 1e6

    tod_summary = {}
    for bucket, s in sorted(tod.items()):
        n = s["n"]
        tod_summary[bucket] = {
            "n": n,
            "pnl": round(s["pnl"], 2),
            "wr": round(s["wins"] / n, 3),
            "mfe": round(s["mfe"] / n, 2),
            "mae": round(s["mae"] / n, 2),
            "fill_lat_ms": round(s["lat_ms"] / n, 1),
        }

    # Exit reason breakdown
    exit_reasons = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in all_trades:
        r = t.get("exit_reason", "Unknown")
        exit_reasons[r]["n"] += 1
        exit_reasons[r]["pnl"] += t.get("pnl_dollars", 0)
        exit_reasons[r]["wins"] += int(t.get("pnl_dollars", 0) > 0)

    er_summary = {
        r: {"n": s["n"], "pnl": round(s["pnl"], 2),
            "wr": round(s["wins"]/s["n"], 3), "avg_pnl": round(s["pnl"]/s["n"], 2)}
        for r, s in exit_reasons.items()
    }

    # Fill latency vs PnL
    lats_ms = np.array([t.get("fill_latency_ns", 0) / 1e6 for t in all_trades])
    pnls = np.array([t.get("pnl_dollars", 0) for t in all_trades])
    lat_buckets = []
    for lat_max in [50, 200, 500, 1000, 2000, 5000, 1e9]:
        mask = lats_ms <= lat_max
        if mask.sum() > 0:
            lat_buckets.append({
                "max_ms": lat_max if lat_max < 1e9 else "inf",
                "n": int(mask.sum()),
                "wr": round(float((pnls[mask] > 0).mean()), 3),
                "avg_pnl": round(float(pnls[mask].mean()), 2),
            })

    return {
        "n_trades": len(all_trades),
        "total_pnl": round(float(pnls.sum()), 2),
        "win_rate": round(float((pnls > 0).mean()), 3),
        "daily": daily,
        "tod_30min": tod_summary,
        "exit_reasons": er_summary,
        "fill_latency_buckets": lat_buckets,
    }


def main():
    out = {"metadata": {"sim_dir": str(SIM_DIR), "ts": datetime.now().isoformat()}, "results": {}}

    print(f"{'Config':<25} {'N':>5} {'WR':>6} {'PnL':>12}")
    print("-" * 60)
    for model in MODELS:
        for threshold in THRESHOLDS:
            key = f"{model}_z{threshold}"
            r = analyze(model, threshold)
            if r is None:
                continue
            out["results"][key] = r
            print(f"{key:<25} {r['n_trades']:>5} {r['win_rate']:>5.1%} ${r['total_pnl']:>10,.2f}")

    OUT_PATH.write_text(json.dumps(out, indent=2))

    # Markdown summary — focus on cnn_mamba_v2_z3 (best chase config)
    md_lines = ["# Time-of-Day + Exit Decomposition — Chase Config", ""]
    md_lines.append(f"Source: `{SIM_DIR.name}` ({SIM_DIR})")
    md_lines.append("")

    for key in ["cnn_mamba_v2_z3", "mamba_v7_z3", "cnn_mamba_v2_z2", "mamba_v7_z2"]:
        if key not in out["results"]:
            continue
        r = out["results"][key]
        md_lines.append(f"## {key}")
        md_lines.append(f"- **Total**: n={r['n_trades']}, WR={r['win_rate']:.1%}, PnL=${r['total_pnl']:,.2f}")
        md_lines.append("")
        md_lines.append("### Daily")
        md_lines.append("| Date | N | WR | PnL | FillRate | Signals |")
        md_lines.append("|------|---|----|-----|----------|---------|")
        for date, d in sorted(r["daily"].items()):
            md_lines.append(f"| {date} | {d['n']} | {d['wr']:.1%} | ${d['pnl']:,.2f} | {d['fill_rate']:.1%} | {d['n_signals']} |")
        md_lines.append("")
        md_lines.append("### Time-of-Day (ET, 30-min buckets, only buckets with n>=3)")
        md_lines.append("| ET   | N | WR | PnL | MFE | MAE | FillLat(ms) |")
        md_lines.append("|------|---|----|-----|-----|-----|--------------|")
        for bucket, s in sorted(r["tod_30min"].items()):
            if s["n"] < 3:
                continue
            md_lines.append(f"| {bucket} | {s['n']} | {s['wr']:.1%} | ${s['pnl']:.0f} | {s['mfe']:.1f}t | {s['mae']:.1f}t | {s['fill_lat_ms']:.0f} |")
        md_lines.append("")
        md_lines.append("### Exit Reasons")
        md_lines.append("| Reason | N | WR | Total PnL | Avg PnL |")
        md_lines.append("|--------|---|----|-----------|---------|")
        for reason, s in sorted(r["exit_reasons"].items(), key=lambda kv: -kv[1]["pnl"]):
            md_lines.append(f"| {reason} | {s['n']} | {s['wr']:.1%} | ${s['pnl']:.0f} | ${s['avg_pnl']:.2f} |")
        md_lines.append("")
        md_lines.append("### Fill Latency Buckets")
        md_lines.append("| MaxMs | N | WR | AvgPnl |")
        md_lines.append("|-------|---|----|--------|")
        for b in r["fill_latency_buckets"]:
            md_lines.append(f"| {b['max_ms']} | {b['n']} | {b['wr']:.1%} | ${b['avg_pnl']:.2f} |")
        md_lines.append("")

    OUT_MD.write_text("\n".join(md_lines))
    print(f"\nSaved JSON: {OUT_PATH}")
    print(f"Saved MD: {OUT_MD}")


if __name__ == "__main__":
    main()
