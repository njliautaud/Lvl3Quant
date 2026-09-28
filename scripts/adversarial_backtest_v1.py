#!/usr/bin/env python3
"""
Adversarial Backtest v1 — Meta-Model Edge Validation

Tests whether the production meta-model's edge is real or a statistical artifact
by comparing real predictions against various null distributions.

Null hypotheses tested:
  1. Shuffled meta-scores (within-date permutation)
  2. Shuffled timestamps (within-date time reorder)
  3. Random meta-model (uniform [0,1] replacement)
  4. Reversed meta-model (negated scores — should show negative edge)

Designed for: Razer (Windows, RTX 3070) or Jupiter (Linux). CPU-only.
"""

import sys
import json
import time
import platform
from pathlib import Path
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — auto-detect OS
# ---------------------------------------------------------------------------
if platform.system() == "Windows":
    BASE = Path(r"C:\Users\claude\Lvl3Quant")
else:
    BASE = Path("/home/jupiter/Lvl3Quant")

SHORTS_DIR = BASE / "output" / "meta_production_v1"
LONGS_DIR = BASE / "output" / "meta_production_longs_v1"
OUTPUT_DIR = BASE / "output" / "adversarial_backtest_v1"

N_PERMUTATIONS = 1000
TOP_PCT_FILTER = 3  # top 3% filter for key metric
PROGRESS_INTERVAL = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_predictions(model_dir: Path):
    """Load concat predictions and reconstruct per-date groupings."""
    npz_path = model_dir / "concat_predictions.npz"
    results_path = model_dir / "results.json"

    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found.")
        return None, None, None

    if not results_path.exists():
        print(f"ERROR: {results_path} not found.")
        return None, None, None

    # Load predictions
    data = np.load(npz_path, allow_pickle=True)
    available_keys = list(data.keys())
    print(f"NPZ keys: {available_keys}")

    if "predictions" not in data or "actuals" not in data:
        print(f"ERROR: Expected keys 'predictions' and 'actuals', got {available_keys}")
        print("Cannot proceed — update script to match npz structure.")
        sys.exit(1)

    meta_scores = data["predictions"].astype(np.float64)
    actuals = data["actuals"].astype(np.float64)
    print(f"Loaded {len(meta_scores)} trades. Scores range: [{meta_scores.min():.4f}, {meta_scores.max():.4f}]")
    print(f"Actuals range: [{actuals.min():.4f}, {actuals.max():.4f}]")

    # Load per-fold info to reconstruct date assignments
    with open(results_path) as f:
        results = json.load(f)

    per_fold = results.get("per_fold", [])
    if not per_fold:
        print("ERROR: No per_fold info in results.json — cannot reconstruct dates.")
        sys.exit(1)

    # Build date array from fold sizes (folds are concatenated in order)
    dates = []
    for fold in sorted(per_fold, key=lambda x: x["fold"]):
        n = fold["n_test"]
        d = int(fold["date"])
        dates.extend([d] * n)

    dates = np.array(dates)
    if len(dates) != len(meta_scores):
        print(f"WARNING: date array length ({len(dates)}) != predictions length ({len(meta_scores)})")
        print("Attempting to proceed with min length...")
        n = min(len(dates), len(meta_scores))
        dates = dates[:n]
        meta_scores = meta_scores[:n]
        actuals = actuals[:n]

    unique_dates = np.unique(dates)
    print(f"Date range: {unique_dates[0]} to {unique_dates[-1]} ({len(unique_dates)} dates)")

    return meta_scores, actuals, dates


def compute_metrics(scores, actuals, dates, top_pct=TOP_PCT_FILTER):
    """
    Compute key metrics for a given set of scores:
      - concat_corr: Pearson correlation between scores and actuals
      - top_N_gross: mean actual PnL (ticks) for top N% by score
      - top_N_wr: win rate for top N%
      - top_N_pf: profit factor for top N%
      - top_N_n: number of trades in top N%
    """
    n = len(scores)
    if n < 10:
        return {"concat_corr": 0.0, "top_gross": 0.0, "top_wr": 0.0, "top_pf": 0.0, "top_n": 0}

    # Concat correlation
    std_s = np.std(scores)
    std_a = np.std(actuals)
    if std_s < 1e-12 or std_a < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(scores, actuals)[0, 1])

    # Top N% filter
    threshold_idx = max(1, int(n * (1 - top_pct / 100)))
    sorted_indices = np.argsort(scores)
    top_indices = sorted_indices[threshold_idx:]
    top_actuals = actuals[top_indices]

    top_n = len(top_actuals)
    top_gross = float(np.mean(top_actuals)) if top_n > 0 else 0.0
    top_wr = float(np.mean(top_actuals > 0) * 100) if top_n > 0 else 0.0

    wins = np.sum(top_actuals[top_actuals > 0])
    losses = np.abs(np.sum(top_actuals[top_actuals < 0]))
    top_pf = float(wins / losses) if losses > 1e-12 else (99.0 if wins > 0 else 0.0)

    return {
        "concat_corr": corr,
        "top_gross": top_gross,
        "top_wr": top_wr,
        "top_pf": top_pf,
        "top_n": top_n,
    }


def permute_within_dates(arr, dates, rng):
    """Permute values within each date group independently."""
    result = arr.copy()
    for d in np.unique(dates):
        mask = dates == d
        idx = np.where(mask)[0]
        result[idx] = rng.permutation(result[idx])
    return result


# ---------------------------------------------------------------------------
# Null hypothesis tests
# ---------------------------------------------------------------------------

def run_shuffled_scores(meta_scores, actuals, dates, n_perm, rng):
    """Test 1: Randomly permute meta-scores within each date."""
    print(f"\n--- Test 1: Shuffled Meta-Scores ({n_perm} permutations) ---")
    results = []
    for i in range(n_perm):
        shuffled = permute_within_dates(meta_scores, dates, rng)
        m = compute_metrics(shuffled, actuals, dates)
        results.append(m)
        if (i + 1) % PROGRESS_INTERVAL == 0:
            print(f"  [{i+1}/{n_perm}] mean_corr={np.mean([r['concat_corr'] for r in results]):.5f}")
    return results


def run_shuffled_timestamps(meta_scores, actuals, dates, n_perm, rng):
    """Test 2: Randomly permute time ordering within each date."""
    print(f"\n--- Test 2: Shuffled Timestamps ({n_perm} permutations) ---")
    results = []
    for i in range(n_perm):
        # Shuffle actuals within each date (equivalent to reordering time)
        shuffled_actuals = permute_within_dates(actuals, dates, rng)
        m = compute_metrics(meta_scores, shuffled_actuals, dates)
        results.append(m)
        if (i + 1) % PROGRESS_INTERVAL == 0:
            print(f"  [{i+1}/{n_perm}] mean_corr={np.mean([r['concat_corr'] for r in results]):.5f}")
    return results


def run_random_model(meta_scores, actuals, dates, n_perm, rng):
    """Test 3: Replace meta-scores with uniform random [0, 1]."""
    print(f"\n--- Test 3: Random Meta-Model ({n_perm} permutations) ---")
    n = len(meta_scores)
    results = []
    for i in range(n_perm):
        random_scores = rng.uniform(0, 1, size=n)
        m = compute_metrics(random_scores, actuals, dates)
        results.append(m)
        if (i + 1) % PROGRESS_INTERVAL == 0:
            print(f"  [{i+1}/{n_perm}] mean_corr={np.mean([r['concat_corr'] for r in results]):.5f}")
    return results


def run_reversed_model(meta_scores, actuals, dates):
    """Test 4: Negate meta-scores (single run, no permutation needed)."""
    print("\n--- Test 4: Reversed Meta-Model (negated scores) ---")
    reversed_scores = -meta_scores
    m = compute_metrics(reversed_scores, actuals, dates)
    return m


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_null_stats(null_results, real_metrics, metric_name):
    """Compute p-value and effect size for a given metric."""
    null_values = np.array([r[metric_name] for r in null_results])
    real_value = real_metrics[metric_name]

    mean_null = float(np.mean(null_values))
    std_null = float(np.std(null_values))
    p_value = float(np.mean(null_values >= real_value))
    effect_size = float((real_value - mean_null) / std_null) if std_null > 1e-12 else float("inf")

    return {
        "real": real_value,
        "null_mean": mean_null,
        "null_std": std_null,
        "null_p5": float(np.percentile(null_values, 5)),
        "null_p95": float(np.percentile(null_values, 95)),
        "null_max": float(np.max(null_values)),
        "p_value": p_value,
        "effect_size_z": effect_size,
    }


def format_table(rows, headers):
    """Simple ASCII table printer."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-+-".join("-" * w for w in col_widths)

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ADVERSARIAL BACKTEST v1 — Meta-Model Edge Validation")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Base path: {BASE}")
    print("=" * 70)

    # Load shorts data
    print("\nLoading shorts (meta_production_v1)...")
    meta_scores, actuals, dates = load_predictions(SHORTS_DIR)
    if meta_scores is None:
        print("FATAL: Could not load predictions. Exiting.")
        sys.exit(1)

    # Real model metrics
    print("\nComputing real model metrics...")
    real_metrics = compute_metrics(meta_scores, actuals, dates)
    print(f"  concat_corr  = {real_metrics['concat_corr']:.5f}")
    print(f"  top {TOP_PCT_FILTER}% gross = {real_metrics['top_gross']:.4f} ticks")
    print(f"  top {TOP_PCT_FILTER}% WR    = {real_metrics['top_wr']:.1f}%")
    print(f"  top {TOP_PCT_FILTER}% PF    = {real_metrics['top_pf']:.3f}")
    print(f"  top {TOP_PCT_FILTER}% n     = {real_metrics['top_n']}")

    rng = np.random.default_rng(seed=42)
    t0 = time.time()

    # Test 1: Shuffled meta-scores
    null_shuffled = run_shuffled_scores(meta_scores, actuals, dates, N_PERMUTATIONS, rng)

    # Test 2: Shuffled timestamps
    null_timestamps = run_shuffled_timestamps(meta_scores, actuals, dates, N_PERMUTATIONS, rng)

    # Test 3: Random model
    null_random = run_random_model(meta_scores, actuals, dates, N_PERMUTATIONS, rng)

    # Test 4: Reversed model
    reversed_metrics = run_reversed_model(meta_scores, actuals, dates)

    elapsed = time.time() - t0
    print(f"\nAll permutation tests completed in {elapsed:.1f}s")

    # ---------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    all_results = {}

    for test_name, null_data in [
        ("shuffled_scores", null_shuffled),
        ("shuffled_timestamps", null_timestamps),
        ("random_model", null_random),
    ]:
        all_results[test_name] = {}
        print(f"\n--- {test_name} ---")
        for metric in ["concat_corr", "top_gross", "top_wr", "top_pf"]:
            stats = compute_null_stats(null_data, real_metrics, metric)
            all_results[test_name][metric] = stats
            sig = "***" if stats["p_value"] < 0.001 else ("**" if stats["p_value"] < 0.01 else ("*" if stats["p_value"] < 0.05 else ""))
            print(f"  {metric:>12s}: real={stats['real']:.4f}  null={stats['null_mean']:.4f}±{stats['null_std']:.4f}  "
                  f"p={stats['p_value']:.4f} {sig}  z={stats['effect_size_z']:.2f}  "
                  f"null_range=[{stats['null_p5']:.4f}, {stats['null_p95']:.4f}]")

    # Reversed model comparison
    all_results["reversed_model"] = {}
    print(f"\n--- reversed_model ---")
    for metric in ["concat_corr", "top_gross", "top_wr", "top_pf"]:
        rv = reversed_metrics[metric]
        rr = real_metrics[metric]
        delta = rr - rv
        all_results["reversed_model"][metric] = {
            "real": rr,
            "reversed": rv,
            "delta": delta,
        }
        direction = "GOOD (real > reversed)" if delta > 0 else "BAD (reversed beats real!)"
        print(f"  {metric:>12s}: real={rr:.4f}  reversed={rv:.4f}  delta={delta:.4f}  {direction}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"SUMMARY: Top {TOP_PCT_FILTER}% Gross Ticks — p-values")
    print("=" * 70)

    headers = ["Test", "Real", "Null Mean", "Null Max", "p-value", "Z-score", "Verdict"]
    rows = []
    for test_name in ["shuffled_scores", "shuffled_timestamps", "random_model"]:
        s = all_results[test_name]["top_gross"]
        p = s["p_value"]
        verdict = "REAL EDGE" if p < 0.01 else ("Marginal" if p < 0.05 else "NOT SIGNIFICANT")
        rows.append([
            test_name,
            f"{s['real']:.4f}",
            f"{s['null_mean']:.4f}",
            f"{s['null_max']:.4f}",
            f"{p:.4f}",
            f"{s['effect_size_z']:.2f}",
            verdict,
        ])

    # Reversed
    rev_gross = all_results["reversed_model"]["top_gross"]
    rev_verdict = "REAL EDGE" if rev_gross["delta"] > 0.2 else ("Marginal" if rev_gross["delta"] > 0 else "FAILED")
    rows.append([
        "reversed_model",
        f"{rev_gross['real']:.4f}",
        f"{rev_gross['reversed']:.4f}",
        "N/A",
        "N/A",
        f"delta={rev_gross['delta']:.4f}",
        rev_verdict,
    ])

    format_table(rows, headers)

    # Overall verdict
    pvals = [all_results[t]["top_gross"]["p_value"] for t in ["shuffled_scores", "shuffled_timestamps", "random_model"]]
    all_sig = all(p < 0.01 for p in pvals)
    reversed_ok = rev_gross["delta"] > 0

    print("\n" + "=" * 70)
    if all_sig and reversed_ok:
        verdict = "PASS — Edge appears REAL across all null tests (p < 0.01) and reversed model shows degradation."
    elif all(p < 0.05 for p in pvals) and reversed_ok:
        verdict = "MARGINAL PASS — Edge is statistically significant (p < 0.05) but not highly significant."
    else:
        failing = [t for t, p in zip(["shuffled_scores", "shuffled_timestamps", "random_model"], pvals) if p >= 0.05]
        verdict = f"FAIL — Edge NOT significant for: {', '.join(failing)}. Possible statistical artifact."
        if not reversed_ok:
            verdict += " ALSO: Reversed model beats real model — strong evidence of NO edge."

    print(f"OVERALL VERDICT: {verdict}")
    print("=" * 70)

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "results.json"

    save_data = {
        "timestamp": datetime.now().isoformat(),
        "model_dir": str(SHORTS_DIR),
        "n_permutations": N_PERMUTATIONS,
        "top_pct_filter": TOP_PCT_FILTER,
        "n_trades": len(meta_scores),
        "n_dates": len(np.unique(dates)),
        "date_range": [int(np.min(dates)), int(np.max(dates))],
        "elapsed_seconds": round(elapsed, 1),
        "real_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in real_metrics.items()},
        "tests": {},
        "verdict": verdict,
    }

    for test_name in ["shuffled_scores", "shuffled_timestamps", "random_model", "reversed_model"]:
        save_data["tests"][test_name] = {}
        for metric in all_results[test_name]:
            entry = all_results[test_name][metric]
            save_data["tests"][test_name][metric] = {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in entry.items()
            }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
