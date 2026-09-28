#!/usr/bin/env python3
"""
Comprehensive PatchTST smart_v2 Analysis
=========================================
Analyzes all 7 fold prediction files, computing IC/DA/MagCorr at multiple
horizons and confidence tiers. Compares against CNN, Mamba, and LGBM benchmarks.
"""

import numpy as np
import json
import os
import sys
from scipy.stats import spearmanr
from datetime import datetime


BASE_DIR = "/home/jupiter/Lvl3Quant/output/patchtst_sliding60d_smart_v2"
OUTPUT_JSON = os.path.join(BASE_DIR, "patchtst_smart_v2_analysis.json")

# Benchmarks
BENCHMARKS = {
    "CNN_baseline": {"IC_10s": 0.132, "IC_1s": None, "DA_All": None},
    "Mamba_v7_March": {"IC_10s": 0.069, "IC_1s": 0.207, "DA_All": None},
    "LGBM_smart_v3": {"IC_10s": None, "IC_1s": None, "DA_All": 0.547, "DA_Top1pct": 0.688},
}

HORIZON_NAMES = ["1s", "5s", "10s"]
CONFIDENCE_TIERS = {
    "All": 1.0,
    "Top50pct": 0.50,
    "Top25pct": 0.25,
    "Top10pct": 0.10,
    "Top5pct": 0.05,
    "Top1pct": 0.01,
}


def compute_ic(preds, labels):
    """Spearman rank correlation."""
    if len(preds) < 10:
        return float("nan")
    corr, pval = spearmanr(preds, labels)
    return float(corr)


def compute_da(preds, labels):
    """Directional accuracy: fraction where sign(pred) == sign(label)."""
    if len(preds) < 10:
        return float("nan")
    # Exclude zero predictions and labels for cleaner DA
    mask = (preds != 0) & (labels != 0)
    if mask.sum() < 10:
        return float("nan")
    return float(np.mean(np.sign(preds[mask]) == np.sign(labels[mask])))


def compute_magcorr(preds, labels):
    """Magnitude correlation: Pearson correlation of abs values."""
    if len(preds) < 10:
        return float("nan")
    abs_p = np.abs(preds)
    abs_l = np.abs(labels)
    if abs_p.std() == 0 or abs_l.std() == 0:
        return float("nan")
    return float(np.corrcoef(abs_p, abs_l)[0, 1])


def analyze_at_confidence_tier(preds, labels, tier_fraction):
    """Filter to top N% by absolute prediction magnitude, then compute metrics."""
    n = len(preds)
    if n < 10:
        return {"ic": float("nan"), "da": float("nan"), "magcorr": float("nan"), "n_samples": 0}

    if tier_fraction >= 1.0:
        subset_preds = preds
        subset_labels = labels
    else:
        abs_preds = np.abs(preds)
        threshold = np.percentile(abs_preds, 100 * (1 - tier_fraction))
        mask = abs_preds >= threshold
        subset_preds = preds[mask]
        subset_labels = labels[mask]

    return {
        "ic": compute_ic(subset_preds, subset_labels),
        "da": compute_da(subset_preds, subset_labels),
        "magcorr": compute_magcorr(subset_preds, subset_labels),
        "n_samples": int(len(subset_preds)),
    }


def analyze_fold(fold_path, fold_idx):
    """Full analysis of a single fold."""
    data = np.load(fold_path)
    preds = data["predictions"]  # (n, 3)
    labels = data["labels"]      # (n, 3)
    n_samples = preds.shape[0]

    # Pre-computed ICs from the training script
    precomputed = {
        "ic_1s": float(data["ic_1s"]),
        "ic_5s": float(data["ic_5s"]),
        "ic_10s": float(data["ic_10s"]),
    }

    oot_file = str(data["oot_files"][0]) if "oot_files" in data else "unknown"

    fold_result = {
        "fold": fold_idx,
        "n_samples": n_samples,
        "oot_file": oot_file,
        "precomputed_ic": precomputed,
        "is_suspect": n_samples < 5000,
        "horizons": {},
    }

    for h_idx, h_name in enumerate(HORIZON_NAMES):
        h_preds = preds[:, h_idx]
        h_labels = labels[:, h_idx]

        horizon_result = {
            "pred_stats": {
                "mean": float(np.mean(h_preds)),
                "std": float(np.std(h_preds)),
                "min": float(np.min(h_preds)),
                "max": float(np.max(h_preds)),
                "pct_zero": float(np.mean(h_preds == 0)),
            },
            "label_stats": {
                "mean": float(np.mean(h_labels)),
                "std": float(np.std(h_labels)),
                "min": float(np.min(h_labels)),
                "max": float(np.max(h_labels)),
            },
            "tiers": {},
        }

        for tier_name, tier_frac in CONFIDENCE_TIERS.items():
            horizon_result["tiers"][tier_name] = analyze_at_confidence_tier(
                h_preds, h_labels, tier_frac
            )

        fold_result["horizons"][h_name] = horizon_result

    data.close()
    return fold_result


def compute_concat_metrics(fold_results):
    """Concatenate all non-suspect folds and compute aggregate metrics."""
    concat = {h: {"preds": [], "labels": []} for h in HORIZON_NAMES}
    concat_all = {h: {"preds": [], "labels": []} for h in HORIZON_NAMES}

    for fr in fold_results:
        data = np.load(os.path.join(BASE_DIR, f"fold_{fr['fold']:02d}_oot_predictions.npz"))
        for h_idx, h_name in enumerate(HORIZON_NAMES):
            concat_all[h_name]["preds"].append(data["predictions"][:, h_idx])
            concat_all[h_name]["labels"].append(data["labels"][:, h_idx])
            if not fr["is_suspect"]:
                concat[h_name]["preds"].append(data["predictions"][:, h_idx])
                concat[h_name]["labels"].append(data["labels"][:, h_idx])
        data.close()

    results = {"excluding_suspect": {}, "all_folds": {}}

    for label, src in [("excluding_suspect", concat), ("all_folds", concat_all)]:
        for h_name in HORIZON_NAMES:
            all_p = np.concatenate(src[h_name]["preds"])
            all_l = np.concatenate(src[h_name]["labels"])

            h_result = {"n_samples": int(len(all_p)), "tiers": {}}
            for tier_name, tier_frac in CONFIDENCE_TIERS.items():
                h_result["tiers"][tier_name] = analyze_at_confidence_tier(all_p, all_l, tier_frac)

            results[label][h_name] = h_result

    return results


def assess_signal_decay(fold_results):
    """Check if IC degrades across newer folds."""
    valid_folds = [fr for fr in fold_results if not fr["is_suspect"]]
    decay = {}
    for h_name in HORIZON_NAMES:
        ics = []
        for fr in valid_folds:
            ic_val = fr["horizons"][h_name]["tiers"]["All"]["ic"]
            ics.append({"fold": fr["fold"], "ic": ic_val, "n_samples": fr["n_samples"]})

        if len(ics) >= 3:
            fold_nums = [x["fold"] for x in ics]
            ic_vals = [x["ic"] for x in ics]
            corr, _ = spearmanr(fold_nums, ic_vals)
            trend = "improving" if corr > 0.3 else "degrading" if corr < -0.3 else "stable"
        else:
            trend = "insufficient_data"
            corr = float("nan")

        decay[h_name] = {
            "per_fold": ics,
            "trend_correlation": float(corr) if not np.isnan(corr) else None,
            "trend": trend,
        }

    return decay


def benchmark_comparison(concat_metrics):
    """Compare concat metrics against benchmarks."""
    exc = concat_metrics["excluding_suspect"]
    comparisons = {}

    # CNN baseline comparison
    patchtst_ic10 = exc["10s"]["tiers"]["All"]["ic"]
    cnn_ic10 = BENCHMARKS["CNN_baseline"]["IC_10s"]
    comparisons["vs_CNN_baseline"] = {
        "metric": "IC_10s",
        "patchtst": patchtst_ic10,
        "benchmark": cnn_ic10,
        "delta": patchtst_ic10 - cnn_ic10,
        "pct_delta": (patchtst_ic10 - cnn_ic10) / cnn_ic10 * 100,
        "verdict": "BEATS" if patchtst_ic10 > cnn_ic10 else "LOSES",
    }

    # Mamba comparison
    patchtst_ic1 = exc["1s"]["tiers"]["All"]["ic"]
    mamba_ic1 = BENCHMARKS["Mamba_v7_March"]["IC_1s"]
    mamba_ic10 = BENCHMARKS["Mamba_v7_March"]["IC_10s"]
    comparisons["vs_Mamba_IC1s"] = {
        "metric": "IC_1s",
        "patchtst": patchtst_ic1,
        "benchmark": mamba_ic1,
        "delta": patchtst_ic1 - mamba_ic1,
        "verdict": "BEATS" if patchtst_ic1 > mamba_ic1 else "LOSES",
    }
    comparisons["vs_Mamba_IC10s"] = {
        "metric": "IC_10s",
        "patchtst": patchtst_ic10,
        "benchmark": mamba_ic10,
        "delta": patchtst_ic10 - mamba_ic10,
        "verdict": "BEATS" if patchtst_ic10 > mamba_ic10 else "LOSES",
    }

    # LGBM DA comparison
    patchtst_da_all = exc["10s"]["tiers"]["All"]["da"]
    lgbm_da_all = BENCHMARKS["LGBM_smart_v3"]["DA_All"]
    comparisons["vs_LGBM_DA_All"] = {
        "metric": "DA_All_10s",
        "patchtst": patchtst_da_all,
        "benchmark": lgbm_da_all,
        "delta": patchtst_da_all - lgbm_da_all,
        "verdict": "BEATS" if patchtst_da_all > lgbm_da_all else "LOSES",
    }

    patchtst_da_top1 = exc["10s"]["tiers"]["Top1pct"]["da"]
    lgbm_da_top1 = BENCHMARKS["LGBM_smart_v3"]["DA_Top1pct"]
    comparisons["vs_LGBM_DA_Top1pct"] = {
        "metric": "DA_Top1pct_10s",
        "patchtst": patchtst_da_top1,
        "benchmark": lgbm_da_top1,
        "delta": patchtst_da_top1 - lgbm_da_top1,
        "verdict": "BEATS" if patchtst_da_top1 > lgbm_da_top1 else "LOSES",
    }

    return comparisons


def generate_verdict(concat_metrics, decay, comparisons, fold_results):
    """Generate final verdict on whether to continue training."""
    exc = concat_metrics["excluding_suspect"]

    strengths = []
    weaknesses = []

    # Check IC performance
    ic10 = exc["10s"]["tiers"]["All"]["ic"]
    ic1 = exc["1s"]["tiers"]["All"]["ic"]
    ic5 = exc["5s"]["tiers"]["All"]["ic"]

    if ic10 > 0.10:
        strengths.append(f"Strong IC_10s={ic10:.4f} (>0.10 threshold)")
    elif ic10 > 0.05:
        strengths.append(f"Moderate IC_10s={ic10:.4f}")
    else:
        weaknesses.append(f"Weak IC_10s={ic10:.4f}")

    if ic1 > 0.20:
        strengths.append(f"Excellent IC_1s={ic1:.4f} (>0.20)")

    # Check confidence tier scaling
    da_all = exc["10s"]["tiers"]["All"]["da"]
    da_top10 = exc["10s"]["tiers"]["Top10pct"]["da"]
    da_top1 = exc["10s"]["tiers"]["Top1pct"]["da"]

    if da_top10 > da_all + 0.02:
        strengths.append(f"DA improves with confidence: All={da_all:.3f} -> Top10%={da_top10:.3f} -> Top1%={da_top1:.3f}")
    else:
        weaknesses.append(f"DA does NOT improve with confidence filtering")

    # Check decay
    if decay["10s"]["trend"] == "improving":
        strengths.append(f"IC_10s IMPROVING across folds (trend corr={decay['10s']['trend_correlation']:.3f})")
    elif decay["10s"]["trend"] == "degrading":
        weaknesses.append(f"IC_10s DEGRADING across folds (trend corr={decay['10s']['trend_correlation']:.3f})")
    else:
        strengths.append(f"IC_10s stable across folds")

    # Check benchmark comparisons
    for name, comp in comparisons.items():
        if comp["verdict"] == "BEATS":
            strengths.append(f"{name}: {comp['metric']}={comp['patchtst']:.4f} BEATS {comp['benchmark']:.4f}")
        else:
            weaknesses.append(f"{name}: {comp['metric']}={comp['patchtst']:.4f} LOSES to {comp['benchmark']:.4f}")

    # Suspect folds
    suspect = [fr for fr in fold_results if fr["is_suspect"]]
    if suspect:
        weaknesses.append(f"{len(suspect)} suspect folds with <5000 samples (folds {[f['fold'] for f in suspect]})")

    # Final recommendation
    beats_cnn = comparisons["vs_CNN_baseline"]["verdict"] == "BEATS"
    beats_mamba = comparisons["vs_Mamba_IC10s"]["verdict"] == "BEATS"
    improving = decay["10s"]["trend"] in ("improving", "stable")

    if beats_cnn and improving:
        recommendation = "STRONG CONTINUE - Beats CNN baseline and signal is stable/improving"
    elif beats_mamba and improving:
        recommendation = "CONTINUE - Beats Mamba, improving trend, but hasn't surpassed CNN yet"
    elif improving and ic10 > 0.08:
        recommendation = "CAUTIOUS CONTINUE - Promising but needs more folds to prove itself"
    else:
        recommendation = "STOP - Not competitive enough to justify GPU hours"

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendation": recommendation,
        "continue_training": "STOP" not in recommendation,
    }


def main():
    print("=" * 70)
    print("PatchTST smart_v2 Comprehensive Analysis")
    print("=" * 70)

    # 1. Analyze each fold
    fold_results = []
    for fold_idx in range(7):
        path = os.path.join(BASE_DIR, f"fold_{fold_idx:02d}_oot_predictions.npz")
        if not os.path.exists(path):
            print(f"  MISSING: fold_{fold_idx:02d}")
            continue
        fr = analyze_fold(path, fold_idx)
        fold_results.append(fr)
        flag = " *** SUSPECT (tiny)" if fr["is_suspect"] else ""
        ic10 = fr["horizons"]["10s"]["tiers"]["All"]["ic"]
        ic1 = fr["horizons"]["1s"]["tiers"]["All"]["ic"]
        da10 = fr["horizons"]["10s"]["tiers"]["All"]["da"]
        print(f"  Fold {fold_idx}: n={fr['n_samples']:>6d}  IC_1s={ic1:.4f}  IC_10s={ic10:.4f}  DA_10s={da10:.3f}  [{fr['oot_file']}]{flag}")

    # 2. Concat metrics
    print("\nComputing concatenated metrics...")
    concat_metrics = compute_concat_metrics(fold_results)

    exc = concat_metrics["excluding_suspect"]
    print(f"\nCONCAT (excl. suspect folds):")
    for h in HORIZON_NAMES:
        ic = exc[h]["tiers"]["All"]["ic"]
        da = exc[h]["tiers"]["All"]["da"]
        mc = exc[h]["tiers"]["All"]["magcorr"]
        n = exc[h]["n_samples"]
        print(f"  {h:>3s}: IC={ic:.4f}  DA={da:.3f}  MagCorr={mc:.4f}  n={n}")

    # 3. Signal decay
    print("\nSignal decay analysis...")
    decay = assess_signal_decay(fold_results)
    for h in HORIZON_NAMES:
        print(f"  {h}: trend={decay[h]['trend']} (corr={decay[h]['trend_correlation']})")

    # 4. Confidence tier analysis
    print("\nConfidence tier analysis (10s horizon, concat excl. suspect):")
    for tier_name in CONFIDENCE_TIERS:
        t = exc["10s"]["tiers"][tier_name]
        print(f"  {tier_name:>10s}: IC={t['ic']:.4f}  DA={t['da']:.3f}  MagCorr={t['magcorr']:.4f}  n={t['n_samples']}")

    # 5. Benchmark comparisons
    print("\nBenchmark comparisons:")
    comparisons = benchmark_comparison(concat_metrics)
    for name, comp in comparisons.items():
        print(f"  {name}: PatchTST {comp['metric']}={comp['patchtst']:.4f} vs {comp['benchmark']:.4f} -> {comp['verdict']} (delta={comp['delta']:+.4f})")

    # 6. Verdict
    verdict = generate_verdict(concat_metrics, decay, comparisons, fold_results)
    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict['recommendation']}")
    print(f"{'='*70}")
    print(f"\nStrengths:")
    for s in verdict["strengths"]:
        print(f"  + {s}")
    print(f"\nWeaknesses:")
    for w in verdict["weaknesses"]:
        print(f"  - {w}")

    # 7. Save JSON
    analysis = {
        "model": "PatchTST_smart_v2",
        "analysis_date": datetime.now().isoformat(),
        "data_dir": BASE_DIR,
        "fold_results": fold_results,
        "concat_metrics": concat_metrics,
        "signal_decay": decay,
        "benchmark_comparisons": comparisons,
        "verdict": verdict,
        "benchmarks_used": BENCHMARKS,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\nAnalysis saved to: {OUTPUT_JSON}")
    return analysis


if __name__ == "__main__":
    main()
