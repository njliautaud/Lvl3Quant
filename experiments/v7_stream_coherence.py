"""
V7 Stream-Coherence Analysis (HC #486 R3 tree-branch)
======================================================
Tests whether v7's edge comes from coherent prediction streams
(adjacent same-direction signals) vs isolated bursts.
Uses existing meta_v7_prod outputs — no retraining.

Output: per-confidence-bucket coherence stats + edge correlation.
"""
import numpy as np
from pathlib import Path
from scipy import stats
import json
import time

V7_DIR = Path("/home/nick/Lvl3Quant/output/meta_v7_prod")
OUT_DIR = Path("/home/nick/Lvl3Quant/output/v7_stream_coherence")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading v7 prod concat predictions...")
t0 = time.time()
data = np.load(V7_DIR / "concat_oot_predictions.npz")
preds = data["predictions"]
labels = data["labels"]
dates = data["dates"]
print(f"  Loaded {len(preds):,} samples across {len(np.unique(dates)):,} dates in {time.time()-t0:.1f}s")

# Per-date stream-coherence: for each date, compute run-length stats of same-sign predictions
results = {}
unique_dates = np.unique(dates)
all_coh_scores = []
all_edges = []

for d in unique_dates:
    mask = dates == d
    p = preds[mask]
    y = labels[mask]
    if len(p) < 100:
        continue
    # Convert to signs (top 10% confidence = signal)
    abs_p = np.abs(p)
    thresh = np.quantile(abs_p, 0.90)
    signal_mask = abs_p >= thresh
    signs = np.sign(p[signal_mask])
    realized = y[signal_mask]
    if len(signs) < 10:
        continue
    # Run-length analysis: consecutive same-sign predictions
    sign_changes = np.diff(signs) != 0
    n_runs = sign_changes.sum() + 1
    avg_run_len = len(signs) / n_runs if n_runs > 0 else 1.0
    # Edge: signed mean realized (signed by prediction direction)
    signed_edge = (signs * realized).mean()
    results[str(d)] = {
        "n_signals": int(len(signs)),
        "avg_run_len": float(avg_run_len),
        "n_runs": int(n_runs),
        "signed_edge_ticks": float(signed_edge),
    }
    all_coh_scores.append(avg_run_len)
    all_edges.append(signed_edge)

# Cross-day: does coherence predict edge?
coh_arr = np.array(all_coh_scores)
edge_arr = np.array(all_edges)
spear, p_spear = stats.spearmanr(coh_arr, edge_arr)
pearson, p_pearson = stats.pearsonr(coh_arr, edge_arr)

print(f"\n=== V7 STREAM-COHERENCE RESULTS ({len(unique_dates)} dates) ===")
print(f"Avg run length across dates: mean={coh_arr.mean():.2f}, median={np.median(coh_arr):.2f}, std={coh_arr.std():.2f}")
print(f"Per-date signed edge: mean={edge_arr.mean():+.4f} ticks, std={edge_arr.std():.4f}")
print(f"Coherence vs edge: Spearman={spear:+.3f} (p={p_spear:.3f}), Pearson={pearson:+.3f} (p={p_pearson:.3f})")

# Split: high vs low coherence dates
median_coh = np.median(coh_arr)
high_coh_edge = edge_arr[coh_arr >= median_coh].mean()
low_coh_edge = edge_arr[coh_arr < median_coh].mean()
print(f"\nEdge on high-coherence dates (run_len >= {median_coh:.2f}): {high_coh_edge:+.4f} ticks")
print(f"Edge on low-coherence dates (run_len < {median_coh:.2f}):  {low_coh_edge:+.4f} ticks")
print(f"Coherence gate uplift: {high_coh_edge - low_coh_edge:+.4f} ticks")

# VERDICT
verdict = "PASS" if abs(spear) > 0.2 and (high_coh_edge - low_coh_edge) > 0.05 else "REJECT"
print(f"\nVERDICT: {verdict} (need |Spearman|>0.2 AND gate uplift > 0.05 ticks)")

summary = {
    "n_dates": int(len(unique_dates)),
    "mean_run_len": float(coh_arr.mean()),
    "mean_edge_ticks": float(edge_arr.mean()),
    "spearman_coh_vs_edge": float(spear),
    "pearson_coh_vs_edge": float(pearson),
    "high_coh_edge": float(high_coh_edge),
    "low_coh_edge": float(low_coh_edge),
    "gate_uplift_ticks": float(high_coh_edge - low_coh_edge),
    "verdict": verdict,
    "per_date": results,
}
with open(OUT_DIR / "stream_coherence_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'stream_coherence_summary.json'}")
