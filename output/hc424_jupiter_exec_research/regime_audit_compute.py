#!/usr/bin/env python3
"""HC #424 regime audit — per-week gross/net FIFO comparison + hourly OOT breakdown."""
import os, sys, json, glob
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np

LABEL_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/hc424_jupiter_exec_research")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ET = timezone(timedelta(hours=-5))  # EST (Feb is standard time, post-DST is EDT but close enough; we'll use offset-aware)

def per_day_stats(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    out = {}
    for side in ("long", "short"):
        f = d[f"tp4sl3_{side}_filled"]
        g = d[f"tp4sl3_{side}_gross_ticks"][f]
        n = d[f"tp4sl3_{side}_net_ticks"][f]
        out[f"{side}_n"] = int(f.sum())
        out[f"{side}_gross_mean"] = float(g.mean()) if f.any() else float("nan")
        out[f"{side}_net_mean"] = float(n.mean()) if f.any() else float("nan")
        out[f"{side}_fill_rate"] = float(f.mean())
    out["n_windows"] = int(len(d["window_k"]))
    return out

def week_of(date_str):
    dt = datetime.strptime(date_str, "%Y%m%d")
    # ISO week
    y, w, _ = dt.isocalendar()
    return f"{y}W{w:02d}"

# Gather all available dates
all_npz = sorted(LABEL_DIR.glob("*_fifo_labels.npz"))
all_dates = [p.name[:8] for p in all_npz]
print(f"Total days available: {len(all_dates)}  range {all_dates[0]} -- {all_dates[-1]}")

# Per-day stats for ALL days
per_day = {}
for p in all_npz:
    d = p.name[:8]
    per_day[d] = per_day_stats(p)
    per_day[d]["week"] = week_of(d)

# Group by week
weeks = {}
for d, s in per_day.items():
    w = s["week"]
    weeks.setdefault(w, []).append(d)

# Per-week aggregated stats (weighted mean across days)
week_stats = {}
for w, days in weeks.items():
    n_long_total = sum(per_day[d]["long_n"] for d in days)
    n_short_total = sum(per_day[d]["short_n"] for d in days)
    if n_long_total == 0 or n_short_total == 0:
        continue
    g_long = sum(per_day[d]["long_gross_mean"] * per_day[d]["long_n"] for d in days) / n_long_total
    g_short = sum(per_day[d]["short_gross_mean"] * per_day[d]["short_n"] for d in days) / n_short_total
    n_long_net = sum(per_day[d]["long_net_mean"] * per_day[d]["long_n"] for d in days) / n_long_total
    n_short_net = sum(per_day[d]["short_net_mean"] * per_day[d]["short_n"] for d in days) / n_short_total
    week_stats[w] = {
        "days": days,
        "n_days": len(days),
        "long_n": n_long_total,
        "short_n": n_short_total,
        "long_gross_mean": g_long,
        "short_gross_mean": g_short,
        "long_net_mean": n_long_net,
        "short_net_mean": n_short_net,
    }

# Print sorted by week
print("\n=== PER-WEEK GROSS FIFO NET (no signal/gating) — tp4sl3 ===")
print(f"{'Week':>8} {'Days':>4} {'L_n':>7} {'L_gross':>8} {'L_net':>8} {'S_n':>7} {'S_gross':>8} {'S_net':>8}  {'dates'}")
for w in sorted(week_stats):
    s = week_stats[w]
    print(f"{w:>8} {s['n_days']:>4} {s['long_n']:>7} {s['long_gross_mean']:>+8.3f} {s['long_net_mean']:>+8.3f} "
          f"{s['short_n']:>7} {s['short_gross_mean']:>+8.3f} {s['short_net_mean']:>+8.3f}  {','.join(s['days'])}")

# OOT week zoom
OOT_DATES = ["20260223","20260224","20260225","20260226","20260227"]
print("\n=== OOT WEEK 20260223-20260227 per-day ===")
print(f"{'Date':>10} {'L_n':>7} {'L_gross':>8} {'L_net':>8} {'S_n':>7} {'S_gross':>8} {'S_net':>8}")
for d in OOT_DATES:
    if d in per_day:
        s = per_day[d]
        print(f"{d:>10} {s['long_n']:>7} {s['long_gross_mean']:>+8.3f} {s['long_net_mean']:>+8.3f} "
              f"{s['short_n']:>7} {s['short_gross_mean']:>+8.3f} {s['short_net_mean']:>+8.3f}")

# Pick 10 random NON-OOT days for comparison + neighboring weeks
import random
random.seed(42)
non_oot = [d for d in all_dates if d not in OOT_DATES]
sample = sorted(random.sample(non_oot, 10))
print("\n=== 10 RANDOM NON-OOT DAYS per-day ===")
print(f"{'Date':>10} {'L_n':>7} {'L_gross':>8} {'L_net':>8} {'S_n':>7} {'S_gross':>8} {'S_net':>8}")
for d in sample:
    s = per_day[d]
    print(f"{d:>10} {s['long_n']:>7} {s['long_gross_mean']:>+8.3f} {s['long_net_mean']:>+8.3f} "
          f"{s['short_n']:>7} {s['short_gross_mean']:>+8.3f} {s['short_net_mean']:>+8.3f}")

# Overall background
all_long_gross, all_long_n = [], []
all_short_gross, all_short_n = [], []
for d, s in per_day.items():
    if d in OOT_DATES: continue
    all_long_gross.append(s["long_gross_mean"]); all_long_n.append(s["long_n"])
    all_short_gross.append(s["short_gross_mean"]); all_short_n.append(s["short_n"])
all_long_gross = np.array(all_long_gross); all_long_n = np.array(all_long_n)
all_short_gross = np.array(all_short_gross); all_short_n = np.array(all_short_n)
bg_long = (all_long_gross * all_long_n).sum() / all_long_n.sum()
bg_short = (all_short_gross * all_short_n).sum() / all_short_n.sum()
oot_long_n_tot = sum(per_day[d]["long_n"] for d in OOT_DATES if d in per_day)
oot_short_n_tot = sum(per_day[d]["short_n"] for d in OOT_DATES if d in per_day)
oot_long_g = sum(per_day[d]["long_gross_mean"]*per_day[d]["long_n"] for d in OOT_DATES if d in per_day) / oot_long_n_tot
oot_short_g = sum(per_day[d]["short_gross_mean"]*per_day[d]["short_n"] for d in OOT_DATES if d in per_day) / oot_short_n_tot
print(f"\n=== Background vs OOT (gross ticks, tp4sl3, fill-weighted) ===")
print(f"Background (all non-OOT, {len(per_day)-5} days): long_gross={bg_long:+.3f}  short_gross={bg_short:+.3f}")
print(f"OOT week (5 days):                    long_gross={oot_long_g:+.3f}  short_gross={oot_short_g:+.3f}")
print(f"Δ (OOT − background): long={oot_long_g-bg_long:+.3f}  short={oot_short_g-bg_short:+.3f}")

# Save JSON
with open(OUT_DIR / "regime_audit_results.json", "w") as f:
    json.dump({
        "per_day": per_day,
        "week_stats": week_stats,
        "oot_dates": OOT_DATES,
        "sample_non_oot": sample,
        "background": {"long_gross": bg_long, "short_gross": bg_short, "n_days": len(per_day)-5},
        "oot": {"long_gross": oot_long_g, "short_gross": oot_short_g},
    }, f, indent=2, default=float)
print(f"\nWrote {OUT_DIR/'regime_audit_results.json'}")

# Hour-by-hour for OOT week — load ts_ns and bucket
print("\n=== OOT WEEK hour-by-hour gross-net (UTC; ES RTH = 14:30-21:00 UTC) ===")
hour_stats = {}
for d in OOT_DATES:
    npz = LABEL_DIR / f"{d}_fifo_labels.npz"
    if not npz.exists(): continue
    data = np.load(npz, allow_pickle=True)
    ts = data["ts_ns"].astype(np.int64)
    hours = (ts // 1_000_000_000 // 3600) % 24
    for side in ("long","short"):
        f = data[f"tp4sl3_{side}_filled"]
        g = data[f"tp4sl3_{side}_gross_ticks"]
        for h in range(24):
            mask = (hours == h) & f
            n = int(mask.sum())
            if n == 0: continue
            hour_stats.setdefault(h, {"long_n":0,"long_sum":0.0,"short_n":0,"short_sum":0.0})
            hour_stats[h][f"{side}_n"] += n
            hour_stats[h][f"{side}_sum"] += float(g[mask].sum())

print(f"{'Hour_UTC':>8} {'L_n':>7} {'L_gross':>8} {'S_n':>7} {'S_gross':>8}")
for h in sorted(hour_stats):
    s = hour_stats[h]
    ln = s["long_n"]; sn = s["short_n"]
    lg = s["long_sum"]/ln if ln else 0
    sg = s["short_sum"]/sn if sn else 0
    print(f"{h:>8} {ln:>7} {lg:>+8.3f} {sn:>7} {sg:>+8.3f}")

with open(OUT_DIR / "regime_audit_hours.json","w") as f:
    json.dump(hour_stats, f, indent=2, default=float)

print("\nDONE")
