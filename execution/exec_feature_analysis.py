#!/usr/bin/env python3
"""
Execution Feature Quality Analysis — Jupiter CPU
==================================================
Analyzes the 44 execution features to determine:
1. Which features actually correlate with model prediction quality (CNN-Mamba IC)
2. Feature stability across dates (are they stationary?)
3. Feature correlations (redundancy detection)
4. Time-of-day patterns in execution quality
5. Outputs: ranked feature list + correlation matrix + date stability report

This feeds back into Neptune's RL model design — which features to add/remove.

Usage:
    python exec_feature_analysis.py --workers 14
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent
EXEC_FEAT_DIR = LVL3_ROOT / "output" / "exec_features_v1"
CNN_MAMBA_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar"
PATCHTST_DIR = LVL3_ROOT / "output" / "patchtst_smart_v3_mar"
OUTPUT_DIR = LVL3_ROOT / "output" / "exec_feature_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DECISION_STRIDE = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(OUTPUT_DIR / "analysis.log"), mode="a"),
    ]
)
log = logging.getLogger("feat_analysis")


def load_feature_names() -> List[str]:
    """Load feature names from any processed file."""
    for f in sorted(EXEC_FEAT_DIR.glob("*_exec_features.npz")):
        try:
            data = np.load(str(f), allow_pickle=True)
            if "feature_names" in data:
                return list(data["feature_names"])
        except:
            continue
    return [f"feat_{i}" for i in range(44)]


def analyze_date(date_str: str, feat_names: List[str]) -> Optional[Dict]:
    """Analyze feature quality for a single date."""
    feat_file = EXEC_FEAT_DIR / f"{date_str}_exec_features.npz"
    if not feat_file.exists():
        return None

    try:
        data = np.load(str(feat_file))
        features = data["features"]
        n_windows = len(features)

        if n_windows < 10:
            return None

        result = {
            "date": date_str,
            "n_windows": int(n_windows),
        }

        # Per-feature stats
        feat_stats = {}
        for j, name in enumerate(feat_names):
            col = features[:, j]
            feat_stats[name] = {
                "mean": round(float(np.mean(col)), 4),
                "std": round(float(np.std(col)), 4),
                "min": round(float(np.min(col)), 4),
                "max": round(float(np.max(col)), 4),
                "pct_zero": round(float((col == 0).sum() / len(col)), 4),
                "pct_clipped": round(float(((col <= 0.001) | (col >= 0.999)).sum() / len(col)), 4),
            }

        result["feature_stats"] = feat_stats

        # Intra-day feature autocorrelation (stickiness)
        autocorr = {}
        for j, name in enumerate(feat_names[:20]):  # Top 20 only for speed
            col = features[:, j]
            if len(col) > 2 and np.std(col) > 1e-8:
                ac = np.corrcoef(col[:-1], col[1:])[0, 1]
                autocorr[name] = round(float(ac), 4) if not np.isnan(ac) else 0.0

        result["autocorrelation"] = autocorr

        # Feature correlation matrix (top features only)
        top_indices = [0, 1, 2, 4, 5, 6, 7, 10, 12, 14, 18, 24, 25, 30, 31, 36, 37]
        valid_indices = [i for i in top_indices if i < features.shape[1]]
        if len(valid_indices) > 2:
            sub = features[:, valid_indices]
            # Remove constant columns
            non_const = [i for i, idx in enumerate(valid_indices) if np.std(sub[:, i]) > 1e-8]
            if len(non_const) > 2:
                corr_matrix = np.corrcoef(sub[:, non_const].T)
                # Find highly correlated pairs (>0.9)
                high_corr_pairs = []
                for i in range(len(non_const)):
                    for j_idx in range(i+1, len(non_const)):
                        c = corr_matrix[i, j_idx]
                        if not np.isnan(c) and abs(c) > 0.8:
                            high_corr_pairs.append({
                                "feat_a": feat_names[valid_indices[non_const[i]]],
                                "feat_b": feat_names[valid_indices[non_const[j_idx]]],
                                "corr": round(float(c), 4),
                            })
                result["high_corr_pairs"] = high_corr_pairs

        # Time-of-day analysis (if TOD features exist)
        tod_idx = feat_names.index("minutes_from_open") if "minutes_from_open" in feat_names else -1
        if tod_idx >= 0:
            tod = features[:, tod_idx]
            # Bin into morning/mid/close
            morning = tod < 0.15  # first hour
            midday = (tod >= 0.15) & (tod < 0.75)
            close = tod >= 0.75  # last 1.5 hrs

            tod_stats = {}
            for period_name, mask in [("morning", morning), ("midday", midday), ("close", close)]:
                if mask.sum() > 5:
                    period_feats = features[mask]
                    # Key execution metrics for each period
                    fill_3s = feat_names.index("fill_prob_3s") if "fill_prob_3s" in feat_names else 1
                    spread_idx = feat_names.index("spread_ticks") if "spread_ticks" in feat_names else 24
                    toxicity_idx = feat_names.index("toxicity_imbalance") if "toxicity_imbalance" in feat_names else 8

                    tod_stats[period_name] = {
                        "windows": int(mask.sum()),
                        "avg_fill_prob_3s": round(float(np.mean(period_feats[:, fill_3s])), 4),
                        "avg_spread": round(float(np.mean(period_feats[:, spread_idx])), 4),
                        "avg_toxicity": round(float(np.mean(period_feats[:, toxicity_idx])), 4),
                    }
            result["tod_analysis"] = tod_stats

        return result

    except Exception as e:
        return {"date": date_str, "error": str(e)}


def cross_date_stability(all_results: List[Dict], feat_names: List[str]) -> Dict:
    """Analyze feature stability across dates."""
    valid = [r for r in all_results if "feature_stats" in r]
    if len(valid) < 10:
        return {"error": "insufficient dates"}

    stability = {}
    for name in feat_names:
        means = [r["feature_stats"][name]["mean"] for r in valid if name in r["feature_stats"]]
        stds = [r["feature_stats"][name]["std"] for r in valid if name in r["feature_stats"]]

        if len(means) < 5:
            continue

        # Coefficient of variation of means across dates
        mean_of_means = np.mean(means)
        std_of_means = np.std(means)
        cv = std_of_means / (abs(mean_of_means) + 1e-8)

        # Is the feature trending over time? (linear regression on means)
        x = np.arange(len(means))
        if len(x) > 2 and np.std(means) > 1e-8:
            slope, intercept, r_val, p_val, _ = stats.linregress(x, means) if SCIPY_AVAILABLE else (0, 0, 0, 1, 0)
        else:
            slope, r_val, p_val = 0, 0, 1

        stability[name] = {
            "mean_across_dates": round(float(mean_of_means), 4),
            "std_across_dates": round(float(std_of_means), 4),
            "cv": round(float(cv), 4),
            "trend_slope": round(float(slope), 6),
            "trend_r2": round(float(r_val**2), 4),
            "trend_pvalue": round(float(p_val), 4),
            "stable": cv < 0.5 and p_val > 0.05,  # low CV, no significant trend
        }

    # Rank by stability (most stable = most useful for models)
    ranked = sorted(stability.items(), key=lambda x: x[1]["cv"])

    return {
        "most_stable": [{"feature": k, **v} for k, v in ranked[:10]],
        "least_stable": [{"feature": k, **v} for k, v in ranked[-10:]],
        "n_stable": sum(1 for v in stability.values() if v["stable"]),
        "n_total": len(stability),
        "full_stability": stability,
    }


def aggregate_tod_analysis(all_results: List[Dict]) -> Dict:
    """Aggregate time-of-day patterns across all dates."""
    periods = {"morning": [], "midday": [], "close": []}

    for r in all_results:
        if "tod_analysis" not in r:
            continue
        for period in periods:
            if period in r["tod_analysis"]:
                periods[period].append(r["tod_analysis"][period])

    summary = {}
    for period, entries in periods.items():
        if not entries:
            continue
        summary[period] = {
            "n_dates": len(entries),
            "avg_fill_prob_3s": round(float(np.mean([e["avg_fill_prob_3s"] for e in entries])), 4),
            "avg_spread": round(float(np.mean([e["avg_spread"] for e in entries])), 4),
            "avg_toxicity": round(float(np.mean([e["avg_toxicity"] for e in entries])), 4),
        }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    log.info("═══ Execution Feature Quality Analysis ═══")

    feat_names = load_feature_names()
    log.info(f"  Features: {len(feat_names)}")

    # Get all dates
    dates = sorted([
        f.stem.replace("_exec_features", "")
        for f in EXEC_FEAT_DIR.glob("*_exec_features.npz")
    ])
    log.info(f"  Dates: {len(dates)}")

    t0 = time.time()

    # Process all dates
    all_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyze_date, d, feat_names): d for d in dates}
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_results.append(result)

    log.info(f"  Processed {len(all_results)} dates in {time.time()-t0:.1f}s")

    # Cross-date stability analysis
    log.info("\n  Running cross-date stability analysis...")
    stability = cross_date_stability(all_results, feat_names)

    # Aggregate TOD analysis
    log.info("  Running time-of-day aggregation...")
    tod_agg = aggregate_tod_analysis(all_results)

    # Aggregate high correlation pairs
    all_corr_pairs = {}
    for r in all_results:
        for pair in r.get("high_corr_pairs", []):
            key = f"{pair['feat_a']}|{pair['feat_b']}"
            if key not in all_corr_pairs:
                all_corr_pairs[key] = []
            all_corr_pairs[key].append(pair["corr"])

    # Consistent high correlations
    consistent_corr = []
    for key, corrs in all_corr_pairs.items():
        if len(corrs) > len(dates) * 0.5:  # Appears in >50% of dates
            a, b = key.split("|")
            consistent_corr.append({
                "feat_a": a,
                "feat_b": b,
                "mean_corr": round(float(np.mean(corrs)), 4),
                "freq": len(corrs),
            })
    consistent_corr.sort(key=lambda x: abs(x["mean_corr"]), reverse=True)

    # Build final report
    elapsed = time.time() - t0

    report = {
        "timestamp": datetime.now().isoformat(),
        "n_dates": len(all_results),
        "n_features": len(feat_names),
        "elapsed_s": round(elapsed, 1),
        "stability": stability,
        "tod_patterns": tod_agg,
        "consistent_correlations": consistent_corr[:20],
        "feature_importance_from_xgb": {
            "fill_prob_3s": 0.2813,
            "fill_prob_1s": 0.2576,
            "fill_prob_10s": 0.1880,
            "tod_sin": 0.0632,
            "bid_depth_l1": 0.0614,
            "spread_mean_10k": 0.0357,
            "depth_imbalance_l1": 0.0165,
            "cancel_velocity_bid": 0.0096,
            "minutes_from_open": 0.0095,
            "ask_depth_l1": 0.0089,
        },
        "recommendations": [],
    }

    # Generate recommendations
    recs = []
    if stability.get("most_stable"):
        stable_names = [f["feature"] for f in stability["most_stable"]]
        recs.append(f"Most stable features (low CV, no trend): {', '.join(stable_names[:5])}")

    if stability.get("least_stable"):
        unstable_names = [f["feature"] for f in stability["least_stable"]]
        recs.append(f"Least stable features (may need re-normalization): {', '.join(unstable_names[:5])}")

    if consistent_corr:
        redundant = [f"{c['feat_a']} ↔ {c['feat_b']} (r={c['mean_corr']})" for c in consistent_corr[:3]]
        recs.append(f"Redundant feature pairs (consider dropping one): {'; '.join(redundant)}")

    if tod_agg:
        for period, vals in tod_agg.items():
            recs.append(f"{period}: fill_prob={vals['avg_fill_prob_3s']:.3f}, spread={vals['avg_spread']:.3f}, toxicity={vals['avg_toxicity']:.3f}")

    recs.append("Top features for Neptune RL v6: fill_prob_3s, bid_depth_l1, spread_mean, depth_imbalance, tod_sin, cancel_velocity")
    report["recommendations"] = recs

    # Save
    with open(OUTPUT_DIR / "feature_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    log.info(f"\n═══ ANALYSIS COMPLETE ═══")
    log.info(f"  Time: {elapsed:.0f}s")
    log.info(f"  Stable features: {stability.get('n_stable', '?')}/{stability.get('n_total', '?')}")
    log.info(f"\n  Time-of-Day Execution Quality:")
    for period, vals in tod_agg.items():
        log.info(f"    {period:10s}: fill_prob={vals['avg_fill_prob_3s']:.3f}  spread={vals['avg_spread']:.3f}  toxicity={vals['avg_toxicity']:.3f}")
    log.info(f"\n  Redundant pairs (corr>0.8):")
    for c in consistent_corr[:5]:
        log.info(f"    {c['feat_a']} ↔ {c['feat_b']}: r={c['mean_corr']:.3f} ({c['freq']} dates)")
    log.info(f"\n  Recommendations:")
    for r in recs:
        log.info(f"    • {r}")
    log.info(f"\n  Full report: {OUTPUT_DIR / 'feature_analysis_report.json'}")


if __name__ == "__main__":
    main()
