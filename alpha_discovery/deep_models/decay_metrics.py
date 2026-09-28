#!/usr/bin/env python3
"""
Comprehensive decay metrics — implements the FULL Model Evaluation Checklist (HC #44/63).

Computes all metrics from raw predictions + labels arrays:
- IC (Spearman) per horizon
- Directional Accuracy per horizon
- Magnitude IC (|pred| vs |true|) per horizon
- Conditional IC at top-10%, top-5%, top-1%, top-0.5% confidence bands
- Per-decile hit rate table
- Signal autocorrelation (prediction stickiness)
- Long/Short signal distribution and per-direction IC
- Coverage (events passing each threshold)

All functions take (preds, labels) numpy arrays and return dicts.
"""

import numpy as np
import scipy.stats


def compute_ic(preds, labels):
    """Spearman rank IC."""
    mask = np.isfinite(preds) & np.isfinite(labels)
    if mask.sum() < 10:
        return 0.0
    return float(scipy.stats.spearmanr(preds[mask], labels[mask]).statistic)


def compute_pearson_ic(preds, labels):
    """Pearson (linear) IC."""
    mask = np.isfinite(preds) & np.isfinite(labels)
    if mask.sum() < 10:
        return 0.0
    r, _ = scipy.stats.pearsonr(preds[mask], labels[mask])
    return float(r)


def compute_directional_accuracy(preds, labels):
    """Fraction of correct sign predictions (excluding zero labels)."""
    mask = np.isfinite(preds) & np.isfinite(labels) & (labels != 0) & (preds != 0)
    if mask.sum() < 10:
        return 0.5
    return float(np.mean(np.sign(preds[mask]) == np.sign(labels[mask])))


def compute_magnitude_ic(preds, labels):
    """Magnitude IC: correlation between |pred| and |label|.
    Tests whether higher conviction predictions correspond to larger moves."""
    mask = np.isfinite(preds) & np.isfinite(labels)
    if mask.sum() < 10:
        return 0.0
    return float(scipy.stats.spearmanr(np.abs(preds[mask]), np.abs(labels[mask])).statistic)


def compute_conditional_ic(preds, labels, quantile):
    """IC computed only on the top-quantile of |pred| (high-conviction signals)."""
    mask = np.isfinite(preds) & np.isfinite(labels)
    p, l = preds[mask], labels[mask]
    if len(p) < 100:
        return 0.0
    threshold = np.quantile(np.abs(p), 1.0 - quantile)
    sel = np.abs(p) >= threshold
    if sel.sum() < 20:
        return 0.0
    return float(scipy.stats.spearmanr(p[sel], l[sel]).statistic)


def compute_conditional_da(preds, labels, quantile):
    """Directional accuracy at top-quantile of |pred|."""
    mask = np.isfinite(preds) & np.isfinite(labels) & (labels != 0) & (preds != 0)
    p, l = preds[mask], labels[mask]
    if len(p) < 100:
        return 0.5
    threshold = np.quantile(np.abs(p), 1.0 - quantile)
    sel = np.abs(p) >= threshold
    if sel.sum() < 10:
        return 0.5
    return float(np.mean(np.sign(p[sel]) == np.sign(l[sel])))


def compute_decile_hit_rates(preds, labels):
    """Per-decile hit rate table. Decile 10 = highest |pred|, Decile 1 = lowest."""
    mask = np.isfinite(preds) & np.isfinite(labels) & (labels != 0) & (preds != 0)
    p, l = preds[mask], labels[mask]
    if len(p) < 100:
        return {}

    abs_p = np.abs(p)
    decile_edges = np.percentile(abs_p, np.arange(0, 110, 10))

    result = {}
    for d in range(10):
        lo = decile_edges[d]
        hi = decile_edges[d + 1]
        if d == 9:
            sel = abs_p >= lo
        else:
            sel = (abs_p >= lo) & (abs_p < hi)
        if sel.sum() < 5:
            continue
        hr = float(np.mean(np.sign(p[sel]) == np.sign(l[sel])))
        avg_pred = float(np.mean(np.abs(p[sel])))
        avg_label = float(np.mean(np.abs(l[sel])))
        result[f"decile_{d+1}"] = {
            "hit_rate": round(hr, 4),
            "count": int(sel.sum()),
            "avg_abs_pred": round(avg_pred, 4),
            "avg_abs_label": round(avg_label, 4),
        }
    return result


def compute_signal_autocorrelation(preds, max_lag=10):
    """Signal autocorrelation — measures how 'sticky' predictions are.
    High autocorrelation = predictions don't change much event-to-event."""
    mask = np.isfinite(preds)
    p = preds[mask]
    if len(p) < 100:
        return {}

    p_centered = p - np.mean(p)
    var = np.var(p_centered)
    if var < 1e-12:
        return {}

    result = {}
    for lag in [1, 5, 10, 50, 100]:
        if lag >= len(p):
            continue
        autocorr = np.mean(p_centered[:-lag] * p_centered[lag:]) / var
        result[f"lag_{lag}"] = round(float(autocorr), 4)
    return result


def compute_long_short_analysis(preds, labels):
    """Long vs Short signal analysis — bias, count, IC per direction."""
    mask = np.isfinite(preds) & np.isfinite(labels) & (preds != 0)
    p, l = preds[mask], labels[mask]
    if len(p) < 20:
        return {}

    long_mask = p > 0
    short_mask = p < 0
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    n_total = n_long + n_short

    result = {
        "n_long": n_long,
        "n_short": n_short,
        "pct_long": round(n_long / n_total * 100, 1) if n_total > 0 else 0,
        "pct_short": round(n_short / n_total * 100, 1) if n_total > 0 else 0,
    }

    if n_long >= 10:
        result["long_ic"] = compute_ic(p[long_mask], l[long_mask])
        result["long_da"] = compute_directional_accuracy(p[long_mask], l[long_mask])
        result["long_avg_pred"] = round(float(np.mean(p[long_mask])), 4)
        result["long_avg_label"] = round(float(np.mean(l[long_mask])), 4)

    if n_short >= 10:
        result["short_ic"] = compute_ic(p[short_mask], l[short_mask])
        result["short_da"] = compute_directional_accuracy(p[short_mask], l[short_mask])
        result["short_avg_pred"] = round(float(np.mean(p[short_mask])), 4)
        result["short_avg_label"] = round(float(np.mean(l[short_mask])), 4)

    return result


def compute_coverage(preds, thresholds_zscore=None):
    """Coverage: how many events pass each z-score threshold.
    Reports count and percentage at each threshold."""
    mask = np.isfinite(preds)
    p = preds[mask]
    if len(p) < 100:
        return {}

    if thresholds_zscore is None:
        # Compute thresholds from data percentiles
        thresholds_zscore = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Z-score the predictions
    mu, std = np.mean(p), np.std(p)
    if std < 1e-12:
        return {}
    z = np.abs((p - mu) / std)

    result = {}
    for t in thresholds_zscore:
        passing = int((z >= t).sum())
        result[f"z>={t:.1f}"] = {
            "count": passing,
            "pct": round(passing / len(p) * 100, 2),
        }
    return result


def compute_all_metrics(preds, labels, horizon_name="10s"):
    """Compute the FULL Model Evaluation Checklist for one (preds, labels) pair.
    Returns a comprehensive dict with all metrics."""

    result = {
        "horizon": horizon_name,
        "n_predictions": int(len(preds)),
        "n_valid": int(np.isfinite(preds).sum() & np.isfinite(labels).sum()),

        # Core metrics
        "IC": compute_ic(preds, labels),
        "pearson_IC": compute_pearson_ic(preds, labels),
        "DA": compute_directional_accuracy(preds, labels),
        "magnitude_IC": compute_magnitude_ic(preds, labels),

        # Confidence-conditional IC
        "condIC_top10pct": compute_conditional_ic(preds, labels, 0.10),
        "condIC_top5pct": compute_conditional_ic(preds, labels, 0.05),
        "condIC_top1pct": compute_conditional_ic(preds, labels, 0.01),
        "condIC_top0.5pct": compute_conditional_ic(preds, labels, 0.005),

        # Confidence-conditional DA
        "condDA_top10pct": compute_conditional_da(preds, labels, 0.10),
        "condDA_top5pct": compute_conditional_da(preds, labels, 0.05),
        "condDA_top1pct": compute_conditional_da(preds, labels, 0.01),
        "condDA_top0.5pct": compute_conditional_da(preds, labels, 0.005),

        # Per-decile analysis
        "decile_hit_rates": compute_decile_hit_rates(preds, labels),

        # Signal properties
        "signal_autocorrelation": compute_signal_autocorrelation(preds),
        "long_short_analysis": compute_long_short_analysis(preds, labels),
        "coverage": compute_coverage(preds),

        # Prediction distribution stats
        "pred_mean": round(float(np.nanmean(preds)), 6),
        "pred_std": round(float(np.nanstd(preds)), 6),
        "pred_skew": round(float(scipy.stats.skew(preds[np.isfinite(preds)])), 4) if np.isfinite(preds).sum() > 10 else 0,
        "label_std": round(float(np.nanstd(labels)), 6),
    }

    return result


def format_metrics_table(metrics_by_horizon, model_name="", date_str="", days_since=0):
    """Format metrics into a readable multi-line string."""
    lines = []
    lines.append(f"  {model_name} | {date_str} (day +{days_since})")
    lines.append(f"  {'Horizon':<6} {'IC':>7} {'DA':>6} {'MagIC':>7} "
                 f"{'Top10%':>7} {'Top5%':>7} {'Top1%':>7} {'Top0.5%':>7} "
                 f"{'DA@1%':>7} {'L/S%':>7} {'AutoC1':>7}")
    lines.append("  " + "-" * 95)

    for h_name, m in sorted(metrics_by_horizon.items()):
        ls = m.get("long_short_analysis", {})
        pct_long = ls.get("pct_long", 50)
        ac = m.get("signal_autocorrelation", {})
        ac1 = ac.get("lag_1", 0)

        lines.append(
            f"  {h_name:<6} {m['IC']:>+7.4f} {m['DA']:>6.3f} {m['magnitude_IC']:>+7.4f} "
            f"{m['condIC_top10pct']:>+7.4f} {m['condIC_top5pct']:>+7.4f} "
            f"{m['condIC_top1pct']:>+7.4f} {m['condIC_top0.5pct']:>+7.4f} "
            f"{m['condDA_top1pct']:>6.3f} {pct_long:>5.1f}%L {ac1:>+7.4f}"
        )
    return "\n".join(lines)
