#!/usr/bin/env python3
"""
Queue Position Analysis - C1 z>2.0 OOT Results
Tests hypothesis: trades filled at deep queue position (>25) have better
edge than shallow queue (<15).

Deep queue = strong momentum moves that continue past entry.
Shallow queue = possible adverse selection or noise fills.

Data source: z20 conviction sweep results (synced to Jupiter).
"""
import json, os, glob, numpy as np
from datetime import datetime
from collections import defaultdict

Z20_DIR = "/home/jupiter/Lvl3Quant/data/processed/conviction_threshold_sweep/z20"
OUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/queue_position_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# Load z20 results
z20_files = sorted(glob.glob(os.path.join(Z20_DIR, "*.json")))
print("Loading from: %s" % Z20_DIR)
print("Found %d z20 result files" % len(z20_files))

if not z20_files:
    print("ERROR: No z20 files found.")
    print("Sync from Neptune first.")
    import sys; sys.exit(1)

# Collect all trades with queue position
all_trades = []
for fpath in z20_files:
    date_str = os.path.basename(fpath).replace(".json", "")
    try:
        with open(fpath) as f:
            data = json.load(f)
        for trade in data.get("trades", []):
            trade["date"] = date_str
            all_trades.append(trade)
    except Exception as e:
        print("Error reading %s: %s" % (fpath, str(e)))

print("Total trades loaded: %d" % len(all_trades))

# Check queue position distribution
queue_positions = [t.get("queue_position_at_post") for t in all_trades if t.get("queue_position_at_post") is not None]
if queue_positions:
    print("Queue position: min=%.0f max=%.0f mean=%.1f median=%.1f" % (
        min(queue_positions), max(queue_positions),
        float(np.mean(queue_positions)), float(np.median(queue_positions))))
else:
    print("WARNING: No queue position data found in trades")
    print("Sample trade keys: %s" % (list(all_trades[0].keys()) if all_trades else "no trades"))
    import sys; sys.exit(1)

# Bucket analysis
BUCKETS = [
    ("q0-10",  0, 10),
    ("q10-20", 10, 20),
    ("q20-30", 20, 30),
    ("q30-50", 30, 50),
    ("q50+",   50, 10000),
]

print("\n" + "="*60)
print("QUEUE POSITION BUCKET ANALYSIS (C1, z>2.0, TP13/SL20)")
print("="*60)
print("%10s | %8s | %7s | %6s | %8s | %10s" % ("Bucket", "Queue", "Trades", "WR%", "Avg PnL", "Total PnL"))
print("-" * 65)

bucket_results = []
for (label, lo, hi) in BUCKETS:
    bucket_trades = [t for t in all_trades
                     if t.get("queue_position_at_post") is not None
                     and lo <= t["queue_position_at_post"] < hi]
    if not bucket_trades:
        continue
    pnls = [t.get("pnl_dollars", 0) for t in bucket_trades]
    n = len(pnls)
    total = sum(pnls)
    avg = total / n
    wr = 100.0 * sum(1 for p in pnls if p > 0) / n
    mean_d = float(np.mean(pnls))
    down = float(np.std([min(p, 0) for p in pnls]))
    sortino_proxy = mean_d / down if down > 0 else 0

    range_str = "%d-%d" % (lo, hi if hi < 10000 else 999)
    marker = " <<EDGE" if avg > 0 and wr > 55 else ""
    print("%10s | %8s | %7d | %5.1f%% | $%7.2f | $%9.0f%s" % (
        label, range_str, n, wr, avg, total, marker))

    bucket_results.append({
        "label": label, "queue_range": [lo, hi], "n_trades": n,
        "total_pnl": round(total, 2), "avg_pnl": round(avg, 2),
        "win_rate": round(wr, 1), "sortino_proxy": round(sortino_proxy, 3),
    })

# Threshold analysis: edge above/below specific queue levels
print("\n" + "="*60)
print("CUMULATIVE EDGE: queue_position >= threshold")
print("="*60)
print("%12s | %7s | %6s | %8s | %10s | %12s" % ("Threshold", "Trades", "WR%", "Avg PnL", "Total PnL", "% of Total"))
print("-" * 70)

total_n = len(all_trades)
thresholds_results = []
for thresh in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
    filtered = [t for t in all_trades
                if t.get("queue_position_at_post", 0) >= thresh]
    if not filtered:
        continue
    pnls = [t.get("pnl_dollars", 0) for t in filtered]
    n = len(pnls)
    total = sum(pnls)
    avg = total / n
    wr = 100.0 * sum(1 for p in pnls if p > 0) / n
    pct = 100.0 * n / total_n
    marker = " <<" if avg > 0 and pct > 15 else ""
    print("  queue>=%2d | %7d | %5.1f%% | $%7.2f | $%9.0f | %11.1f%%%s" % (
        thresh, n, wr, avg, total, pct, marker))
    thresholds_results.append({
        "min_queue": thresh, "n_trades": n, "pct_of_total": round(pct, 1),
        "total_pnl": round(total, 2), "avg_pnl": round(avg, 2), "win_rate": round(wr, 1)
    })

# Signal strength vs queue position
print("\n" + "="*60)
print("SIGNAL STRENGTH vs QUEUE POSITION")
print("="*60)
valid_pairs = [(abs(t.get("signal_strength", 0)), t.get("queue_position_at_post", 0))
               for t in all_trades
               if t.get("signal_strength") is not None and t.get("queue_position_at_post") is not None]
if valid_pairs:
    strengths = [p[0] for p in valid_pairs]
    positions = [p[1] for p in valid_pairs]
    corr = float(np.corrcoef(strengths, positions)[0, 1])
    print("Correlation(|signal_strength|, queue_position): %.3f" % corr)

    for q_thresh, label in [(0, "all"), (15, "q<15"), (25, "q15-25"), (35, "q>25")]:
        if label == "all":
            subset = [s for s, q in valid_pairs]
        elif label == "q<15":
            subset = [s for s, q in valid_pairs if q < 15]
        elif label == "q15-25":
            subset = [s for s, q in valid_pairs if 15 <= q < 25]
        else:
            subset = [s for s, q in valid_pairs if q >= 25]
        if subset:
            print("  %s (n=%d): mean |z|=%.3f" % (label, len(subset), float(np.mean(subset))))

# Exit reason by queue position
print("\n" + "="*60)
print("EXIT REASON BY QUEUE DEPTH")
print("="*60)
for (label, lo, hi) in [("shallow_q<15", 0, 15), ("medium_q15-30", 15, 30), ("deep_q>30", 30, 10000)]:
    bucket_t = [t for t in all_trades
                if t.get("queue_position_at_post") is not None
                and lo <= t["queue_position_at_post"] < hi]
    if not bucket_t:
        continue
    exits = defaultdict(int)
    for t in bucket_t:
        exits[t.get("exit_reason", "Unknown")] += 1
    pnls = [t.get("pnl_dollars", 0) for t in bucket_t]
    wr = 100.0 * sum(1 for p in pnls if p > 0) / len(pnls)
    print("\n%s (n=%d): exits=%s WR=%.1f%% avg_pnl=$%.2f" % (
        label, len(bucket_t), dict(exits), wr, float(np.mean(pnls))))

# Recommendation
print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
if thresholds_results:
    # Find optimal: best avg_pnl with at least 20% of trades
    eligible = [r for r in thresholds_results if r["pct_of_total"] >= 20]
    if eligible:
        best = max(eligible, key=lambda x: x["avg_pnl"])
        print("Best queue filter (min 20pct of trades): queue >= %d" % best["min_queue"])
        print("  avg_pnl=$%.2f, WR=%.1f%%, %.0f%% of trades" % (
            best["avg_pnl"], best["win_rate"], best["pct_of_total"]))
        if best["min_queue"] > 0:
            print("  RECOMMENDATION: Add min_queue_position=%d filter to C1 config" % best["min_queue"])
        else:
            print("  No queue filter needed - all queue positions profitable")

# Save results
summary = {
    "timestamp": datetime.now().isoformat(),
    "total_trades": len(all_trades),
    "queue_distribution": {
        "min": float(min(queue_positions)), "max": float(max(queue_positions)),
        "mean": round(float(np.mean(queue_positions)), 1),
        "median": round(float(np.median(queue_positions)), 1),
    } if queue_positions else {},
    "bucket_results": bucket_results,
    "threshold_results": thresholds_results,
}
out_file = os.path.join(OUT_DIR, "queue_position_summary.json")
with open(out_file, "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved to: %s" % out_file)
print("DONE")
