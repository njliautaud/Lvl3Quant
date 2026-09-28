"""
confidence_bands.py — Bootstrap confidence intervals for model evaluation metrics.

Provides IC, Directional Accuracy, and Sortino with 95% CI via bootstrap
resampling. Supports tier-based filtering by prediction magnitude and
multi-horizon evaluation from concat NPZ files.

Usage:
    from confidence_bands import evaluate_predictions_with_ci, format_results_with_ci

    # From arrays
    results = compute_metrics_with_ci(predictions, labels)
    print(format_results_with_ci(results, horizon="10s"))

    # From NPZ file
    report = evaluate_predictions_with_ci("path/to/concat_oot_predictions.npz")
    print(report)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HORIZONS = ["1s", "5s", "10s"]

# Tier definitions: (label, quantile threshold on |prediction|)
# e.g. top 10% means we keep the 10% of samples with largest |pred|
TIERS = [
    ("All",     0.0),
    ("Top 50%", 0.50),
    ("Top 25%", 0.75),
    ("Top 10%", 0.90),
    ("Top 5%",  0.95),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetricCI:
    """Point estimate with bootstrap confidence interval."""
    point: float
    lower: float
    upper: float

    def __repr__(self) -> str:
        return f"{self.point:.4f} [{self.lower:.4f}, {self.upper:.4f}]"


@dataclass
class TierResult:
    """Metrics for a single prediction-magnitude tier."""
    tier_name: str
    n_samples: int
    ic: MetricCI
    da: MetricCI
    sortino: Optional[MetricCI] = None


@dataclass
class HorizonResult:
    """Full evaluation for one horizon across all tiers."""
    horizon: str
    tiers: List[TierResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def _spearman_ic(preds: np.ndarray, labels: np.ndarray) -> float:
    """Spearman rank correlation. Returns NaN if insufficient data."""
    valid = ~(np.isnan(preds) | np.isnan(labels))
    if valid.sum() < 20:
        return float("nan")
    rho, _ = spearmanr(preds[valid], labels[valid])
    return float(rho)


def _directional_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of samples where sign(pred) == sign(label). Excludes zeros."""
    valid = ~(np.isnan(preds) | np.isnan(labels))
    p, l = preds[valid], labels[valid]
    mask = (p != 0) & (l != 0)
    if mask.sum() < 20:
        return float("nan")
    return float(np.mean(np.sign(p[mask]) == np.sign(l[mask])))


def _sortino_ratio(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Sortino-style ratio: mean(labels where pred > 0) / downside_std.
    Treats predictions as a signal: go long when pred > 0, measure
    returns (labels) on those trades. Downside deviation uses only
    negative returns.
    """
    valid = ~(np.isnan(preds) | np.isnan(labels))
    p, l = preds[valid], labels[valid]
    long_returns = l[p > 0]
    if len(long_returns) < 20:
        return float("nan")
    mean_ret = np.mean(long_returns)
    downside = long_returns[long_returns < 0]
    if len(downside) < 5:
        # Very few negative returns — Sortino is effectively infinite
        return float("nan")
    downside_std = np.std(downside)
    if downside_std < 1e-12:
        return float("nan")
    return float(mean_ret / downside_std)


# ---------------------------------------------------------------------------
# Bootstrap engine
# ---------------------------------------------------------------------------

def _bootstrap_metric(
    preds: np.ndarray,
    labels: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> MetricCI:
    """
    Bootstrap a metric function over (preds, labels) pairs.

    Returns the point estimate (on full data) plus percentile CI.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    point = metric_fn(preds, labels)
    n = len(preds)

    if np.isnan(point) or n < 40:
        return MetricCI(point=point, lower=float("nan"), upper=float("nan"))

    boot_values = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_values[i] = metric_fn(preds[idx], labels[idx])

    # Drop NaN bootstrap samples (can happen with small tiers)
    boot_valid = boot_values[~np.isnan(boot_values)]
    if len(boot_valid) < n_resamples * 0.5:
        # Too many failed bootstrap samples — CI unreliable
        return MetricCI(point=point, lower=float("nan"), upper=float("nan"))

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_valid, 100 * alpha))
    upper = float(np.percentile(boot_valid, 100 * (1 - alpha)))
    return MetricCI(point=point, lower=lower, upper=upper)


# ---------------------------------------------------------------------------
# Tier-based evaluation
# ---------------------------------------------------------------------------

def _filter_tier(
    preds: np.ndarray, labels: np.ndarray, quantile_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only samples where |pred| >= the given quantile of |pred|."""
    if quantile_threshold <= 0.0:
        return preds, labels
    abs_preds = np.abs(preds)
    cutoff = np.nanquantile(abs_preds, quantile_threshold)
    mask = abs_preds >= cutoff
    return preds[mask], labels[mask]


def compute_metrics_with_ci(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    include_sortino: bool = True,
    tiers: Optional[List[Tuple[str, float]]] = None,
    seed: int = 42,
) -> List[TierResult]:
    """
    Compute IC, DA, and optionally Sortino with bootstrap CI for each tier.

    Parameters
    ----------
    predictions : 1D array of model predictions
    labels : 1D array of ground truth labels (e.g. forward returns)
    n_resamples : number of bootstrap iterations (default 1000)
    ci : confidence level (default 0.95 for 95% CI)
    include_sortino : whether to compute Sortino ratio
    tiers : list of (name, quantile_threshold) pairs, or None for defaults
    seed : random seed for reproducibility

    Returns
    -------
    List of TierResult, one per tier.
    """
    if tiers is None:
        tiers = TIERS

    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()

    # Remove rows where both are NaN
    valid = ~(np.isnan(predictions) | np.isnan(labels))
    predictions = predictions[valid]
    labels = labels[valid]

    rng = np.random.default_rng(seed)
    results = []

    for tier_name, q_thresh in tiers:
        p, l = _filter_tier(predictions, labels, q_thresh)
        n = len(p)

        ic = _bootstrap_metric(p, l, _spearman_ic, n_resamples, ci, rng)
        da = _bootstrap_metric(p, l, _directional_accuracy, n_resamples, ci, rng)

        sortino = None
        if include_sortino:
            sortino = _bootstrap_metric(p, l, _sortino_ratio, n_resamples, ci, rng)

        results.append(TierResult(
            tier_name=tier_name,
            n_samples=n,
            ic=ic,
            da=da,
            sortino=sortino,
        ))

    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_ci(m: MetricCI, pct: bool = False) -> str:
    """Format a MetricCI as a string. If pct=True, multiply by 100 and add %."""
    if np.isnan(m.point):
        return "NaN"
    if pct:
        if np.isnan(m.lower):
            return f"{m.point * 100:.1f}%"
        return f"{m.point * 100:.1f}% [{m.lower * 100:.1f}%, {m.upper * 100:.1f}%]"
    else:
        if np.isnan(m.lower):
            return f"{m.point:.4f}"
        return f"{m.point:.4f} [{m.lower:.4f}, {m.upper:.4f}]"


def format_results_with_ci(
    tier_results: List[TierResult],
    horizon: str = "10s",
    include_sortino: bool = True,
) -> str:
    """
    Format tier results into a readable multi-line string for logging.

    Example output:
        IC_10s = 0.1320 [0.0980, 0.1660] (95% CI) | DA = 53.3% [51.2%, 55.4%]
        Top 10% (n=4200): IC = 0.2450 [0.1980, 0.2920] | DA = 62.1% [58.3%, 65.9%]
    """
    lines = []
    for tr in tier_results:
        parts = []

        # IC
        ic_label = f"IC_{horizon}" if tr.tier_name == "All" else "IC"
        parts.append(f"{ic_label} = {_fmt_ci(tr.ic)}")

        # DA
        parts.append(f"DA = {_fmt_ci(tr.da, pct=True)}")

        # Sortino
        if include_sortino and tr.sortino is not None:
            parts.append(f"Sortino = {_fmt_ci(tr.sortino)}")

        metric_str = " | ".join(parts)

        if tr.tier_name == "All":
            ci_note = " (95% CI)" if not np.isnan(tr.ic.lower) else ""
            lines.append(f"{metric_str}{ci_note}  [n={tr.n_samples:,}]")
        else:
            lines.append(f"  {tr.tier_name} (n={tr.n_samples:,}): {metric_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# NPZ-based evaluation (standalone entry point)
# ---------------------------------------------------------------------------

def evaluate_predictions_with_ci(
    npz_path: Union[str, Path],
    horizons: Optional[List[str]] = None,
    n_resamples: int = 1000,
    include_sortino: bool = True,
) -> str:
    """
    Load a concat_oot_predictions.npz and produce a full CI report.

    Expected NPZ keys: preds_{horizon}, labels_{horizon} for each horizon.

    Parameters
    ----------
    npz_path : path to .npz file
    horizons : list of horizons to evaluate (default: ["1s", "5s", "10s"])
    n_resamples : bootstrap iterations
    include_sortino : include Sortino ratio

    Returns
    -------
    Formatted multi-line report string.
    """
    if horizons is None:
        horizons = HORIZONS

    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path)
    available_keys = list(data.keys())
    logger.info(f"Loading {npz_path.name} — keys: {available_keys}")

    sections = []
    sections.append(f"{'=' * 70}")
    sections.append(f"CONFIDENCE BAND REPORT: {npz_path.name}")
    sections.append(f"{'=' * 70}")

    for h in horizons:
        preds_key = f"preds_{h}"
        labels_key = f"labels_{h}"

        if preds_key not in data or labels_key not in data:
            logger.warning(f"Skipping horizon {h}: missing keys in NPZ")
            continue

        preds = data[preds_key]
        labels = data[labels_key]

        sections.append(f"\n--- Horizon: {h} ---")

        tier_results = compute_metrics_with_ci(
            preds, labels,
            n_resamples=n_resamples,
            include_sortino=include_sortino,
        )
        sections.append(format_results_with_ci(
            tier_results, horizon=h, include_sortino=include_sortino,
        ))

    sections.append(f"\n{'=' * 70}")
    sections.append(f"Bootstrap: {n_resamples} resamples, 95% CI")
    sections.append(f"{'=' * 70}")

    report = "\n".join(sections)
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Compute bootstrap confidence intervals for model predictions."
    )
    parser.add_argument(
        "npz_path",
        help="Path to concat_oot_predictions.npz",
    )
    parser.add_argument(
        "--horizons", nargs="+", default=HORIZONS,
        help="Horizons to evaluate (default: 1s 5s 10s)",
    )
    parser.add_argument(
        "--n-resamples", type=int, default=1000,
        help="Number of bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--no-sortino", action="store_true",
        help="Skip Sortino ratio computation",
    )
    args = parser.parse_args()

    try:
        report = evaluate_predictions_with_ci(
            args.npz_path,
            horizons=args.horizons,
            n_resamples=args.n_resamples,
            include_sortino=not args.no_sortino,
        )
        print(report)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
