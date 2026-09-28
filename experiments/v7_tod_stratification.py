"""
V7 Time-of-Day Stratification (HC #490 R3 confluence axis)
============================================================
Question: Does v7's +0.78 t/day edge concentrate in specific hours,
making time-of-day a viable confluence gate?
Approach: Bucket v7 concat predictions by hour-of-day, compute
top-5%/10%/20% signed edge per bucket. Identify hours with >40%
edge lift over baseline.
"""
import numpy as np
from pathlib import Path
from scipy import stats
import json

V7_DIR = Path("/home/nick/Lvl3Quant/output/meta_v7_prod")
OUT_DIR = Path("/home/nick/Lvl3Quant/output/v7_tod_stratification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = np.load(V7_DIR / "concat_oot_predictions.npz")
preds = data["predictions"]
labels = data["labels"]
dates = data["dates"].astype(str)

# Check for time/timestamp info — fall back to event-index proxy if missing
print(f"Keys: {list(data.keys())}")
print(f"Shapes: preds={preds.shape}, labels={labels.shape}")

# Need event timestamps. Look for sidecar files in fold dirs
import os
fold_dirs = sorted([d for d in os.listdir(V7_DIR.parent) if 'meta_v7_prod' in d])
print(f"Looking for time info in fold preds...")
sample = np.load(V7_DIR / "fold_00_oot_predictions.npz")
print(f"  fold_00 keys: {list(sample.keys())}")

# If no timestamps in concat, use within-day event-index as proxy for time-of-day
# RTH = 9:30 ET (5:30 ET premkt) to 16:00 ET (10:00 ET extended)
# Approximate: split each day into 6 equal buckets (early/mid/late within day)

results = {}
unique_dates = np.unique(dates)
n_buckets = 6  # ~1.1hr each for full session

bucket_metrics = {i: {'preds': [], 'labels': [], 'top5_edge': [], 'top10_edge': []} for i in range(n_buckets)}

for d in unique_dates:
    mask = dates == d
    p = preds[mask]
    y = labels[mask]
    n = len(p)
    if n < 100: continue
    # Within-day event-index buckets
    bucket_ids = (np.arange(n) * n_buckets // n).astype(int)
    for b in range(n_buckets):
        bm = bucket_ids == b
        if bm.sum() < 50: continue
        pb, yb = p[bm], y[bm]
        # Top-5% by |confidence|
        abs_p = np.abs(pb)
        t5 = np.quantile(abs_p, 0.95)
        t10 = np.quantile(abs_p, 0.90)
        m5 = abs_p >= t5
        m10 = abs_p >= t10
        if m5.sum() > 5:
            edge5 = (np.sign(pb[m5]) * yb[m5]).mean()
            bucket_metrics[b]['top5_edge'].append(edge5)
        if m10.sum() > 10:
            edge10 = (np.sign(pb[m10]) * yb[m10]).mean()
            bucket_metrics[b]['top10_edge'].append(edge10)

print(f"\n=== V7 INTRA-DAY BUCKET STRATIFICATION ({len(unique_dates)} dates, {n_buckets} buckets) ===")
print(f"Bucket  N_dates  Top5_edge_mean  Top5_edge_std  Top10_edge_mean  Top10_edge_std")
summary = []
for b in range(n_buckets):
    bm = bucket_metrics[b]
    if not bm['top5_edge']: continue
    t5_arr = np.array(bm['top5_edge'])
    t10_arr = np.array(bm['top10_edge'])
    row = {
        'bucket': b,
        'n_dates': len(t5_arr),
        'top5_mean': float(t5_arr.mean()),
        'top5_std': float(t5_arr.std()),
        'top10_mean': float(t10_arr.mean()) if len(t10_arr) else None,
        'top10_std': float(t10_arr.std()) if len(t10_arr) else None,
    }
    summary.append(row)
    print(f"  {b}      {len(t5_arr):3d}    {t5_arr.mean():+.4f}        {t5_arr.std():.4f}      {t10_arr.mean():+.4f}         {t10_arr.std():.4f}")

# Best vs worst bucket
top5_means = np.array([r['top5_mean'] for r in summary])
overall_mean = top5_means.mean()
best_b = np.argmax(top5_means)
worst_b = np.argmin(top5_means)
lift_best = (top5_means[best_b] - overall_mean) / overall_mean if overall_mean != 0 else 0
print(f"\nOverall top5 edge: {overall_mean:+.4f} ticks")
print(f"Best bucket  ({best_b}): {top5_means[best_b]:+.4f} ticks (+{lift_best*100:.0f}% vs avg)")
print(f"Worst bucket ({worst_b}): {top5_means[worst_b]:+.4f} ticks ({(top5_means[worst_b]-overall_mean)/overall_mean*100:+.0f}% vs avg)")

# F-test: are buckets significantly different?
all_groups = [np.array(bucket_metrics[b]['top5_edge']) for b in range(n_buckets) if bucket_metrics[b]['top5_edge']]
f_stat, p_val = stats.f_oneway(*all_groups)
print(f"\nF-test across buckets: F={f_stat:.2f}, p={p_val:.4f}")

verdict = "PASS" if (lift_best > 0.40 and p_val < 0.05) else "REJECT"
print(f"VERDICT: {verdict} (need best-bucket lift >40% AND F-test p<0.05)")

with open(OUT_DIR / "tod_stratification.json", "w") as f:
    json.dump({
        'n_buckets': n_buckets,
        'n_dates': len(unique_dates),
        'buckets': summary,
        'overall_top5_mean': float(overall_mean),
        'best_bucket': int(best_b),
        'best_lift_pct': float(lift_best * 100),
        'worst_bucket': int(worst_b),
        'f_stat': float(f_stat),
        'p_value': float(p_val),
        'verdict': verdict,
    }, f, indent=2)
print(f"Saved: {OUT_DIR / 'tod_stratification.json'}")
