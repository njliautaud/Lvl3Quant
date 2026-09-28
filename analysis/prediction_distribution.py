"""
Prediction Distribution Analysis — Width, uncertainty, and magnitude distribution

Analyzes:
- Prediction magnitude distribution (width of predictions)
- Calibration: do larger predictions = larger actual moves?
- Prediction uncertainty (if model outputs confidence/variance)
- Percentile-based performance (top decile IC vs bottom decile)

This informs threshold selection: what magnitude is "actionable"?
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from scipy.stats import pearsonr, spearmanr

from prediction_loader import PredictionData

logger = logging.getLogger(__name__)


def analyze_prediction_magnitude(pred_data: PredictionData) -> Dict:
    """
    Analyze distribution of prediction magnitudes.

    Returns:
        Dict with statistics:
        - mean, median, std of absolute predictions
        - percentiles (5, 10, 25, 50, 75, 90, 95)
        - skewness, kurtosis
        - zero_rate (fraction of zero predictions)
    """
    preds = pred_data.predictions
    preds_abs = np.abs(preds)

    stats = {
        'mean': np.mean(preds),
        'median': np.median(preds),
        'std': np.std(preds),
        'mean_abs': np.mean(preds_abs),
        'median_abs': np.median(preds_abs),
        'percentiles': {
            'p5': np.percentile(preds_abs, 5),
            'p10': np.percentile(preds_abs, 10),
            'p25': np.percentile(preds_abs, 25),
            'p50': np.percentile(preds_abs, 50),
            'p75': np.percentile(preds_abs, 75),
            'p90': np.percentile(preds_abs, 90),
            'p95': np.percentile(preds_abs, 95),
        },
        'zero_rate': np.mean(preds == 0),
        'skewness': float(np.mean((preds - np.mean(preds))**3) / np.std(preds)**3) if np.std(preds) > 0 else 0,
        'kurtosis': float(np.mean((preds - np.mean(preds))**4) / np.std(preds)**4) if np.std(preds) > 0 else 0,
    }

    logger.info(f"Prediction magnitude: mean_abs={stats['mean_abs']:.3f}, median_abs={stats['median_abs']:.3f}")

    return stats


def calibration_analysis(pred_data: PredictionData, num_bins: int = 10) -> Dict:
    """
    Analyze prediction calibration: do larger predictions = larger actual moves?

    Args:
        pred_data: PredictionData object
        num_bins: Number of magnitude bins to analyze

    Returns:
        Dict with calibration metrics:
        - bin_edges: Magnitude bin edges
        - mean_pred_by_bin: Mean prediction magnitude per bin
        - mean_label_by_bin: Mean actual return per bin
        - ic_by_bin: IC within each bin
        - calibration_slope: Slope of mean_label vs mean_pred
        - rank_correlation: Spearman correlation (magnitude vs |actual|)
    """
    preds = pred_data.predictions
    labels = pred_data.labels
    preds_abs = np.abs(preds)
    labels_abs = np.abs(labels)

    # Remove zeros
    mask = preds_abs > 0
    preds_abs = preds_abs[mask]
    labels_abs = labels_abs[mask]
    preds = preds[mask]
    labels = labels[mask]

    if len(preds) < num_bins * 10:
        logger.warning(f"Not enough data for {num_bins} bins")
        num_bins = max(2, len(preds) // 10)

    # Bin by prediction magnitude
    bin_edges = np.percentile(preds_abs, np.linspace(0, 100, num_bins + 1))
    bin_indices = np.digitize(preds_abs, bin_edges[1:-1])

    mean_pred_by_bin = []
    mean_label_by_bin = []
    ic_by_bin = []

    for b in range(num_bins):
        bin_mask = bin_indices == b
        if np.sum(bin_mask) < 10:
            mean_pred_by_bin.append(np.nan)
            mean_label_by_bin.append(np.nan)
            ic_by_bin.append(np.nan)
            continue

        bin_preds = preds[bin_mask]
        bin_labels = labels[bin_mask]

        mean_pred_by_bin.append(np.mean(np.abs(bin_preds)))
        mean_label_by_bin.append(np.mean(np.abs(bin_labels)))

        # IC within bin
        if np.std(bin_preds) > 0 and np.std(bin_labels) > 0:
            ic, _ = pearsonr(bin_preds, bin_labels)
            ic_by_bin.append(ic)
        else:
            ic_by_bin.append(np.nan)

    # Calibration slope (should be close to 1 for well-calibrated predictions)
    valid_bins = ~np.isnan(mean_pred_by_bin)
    if np.sum(valid_bins) >= 2:
        slope, intercept = np.polyfit(
            np.array(mean_pred_by_bin)[valid_bins],
            np.array(mean_label_by_bin)[valid_bins],
            1
        )
    else:
        slope, intercept = np.nan, np.nan

    # Rank correlation (magnitude ordering)
    rank_corr, _ = spearmanr(preds_abs, labels_abs)

    logger.info(f"Calibration: slope={slope:.3f}, rank_corr={rank_corr:.3f}")

    return {
        'bin_edges': bin_edges,
        'mean_pred_by_bin': np.array(mean_pred_by_bin),
        'mean_label_by_bin': np.array(mean_label_by_bin),
        'ic_by_bin': np.array(ic_by_bin),
        'calibration_slope': slope,
        'calibration_intercept': intercept,
        'rank_correlation': rank_corr,
    }


def percentile_ic_analysis(pred_data: PredictionData, percentiles: List[int] = [10, 20, 30, 40, 50]) -> Dict:
    """
    Analyze IC for top/bottom percentiles of predictions.

    Args:
        pred_data: PredictionData object
        percentiles: List of top percentiles to analyze (e.g., [10, 20] = top 10%, top 20%)

    Returns:
        Dict mapping percentile to IC:
        - top_N_ic: IC for top N% by magnitude
        - bottom_N_ic: IC for bottom N% by magnitude
        - top_N_sharpe: Sharpe ratio for top N%
    """
    preds = pred_data.predictions
    labels = pred_data.labels
    preds_abs = np.abs(preds)

    results = {}

    for pct in percentiles:
        # Top percentile (highest magnitude)
        threshold_top = np.percentile(preds_abs, 100 - pct)
        top_mask = preds_abs >= threshold_top

        if np.sum(top_mask) >= 10:
            top_preds = preds[top_mask]
            top_labels = labels[top_mask]

            if np.std(top_preds) > 0 and np.std(top_labels) > 0:
                ic_top, _ = pearsonr(top_preds, top_labels)
            else:
                ic_top = np.nan

            # Sharpe of returns (assuming predictions = position sizing)
            top_returns = top_preds * top_labels  # Directional P&L
            sharpe_top = np.mean(top_returns) / np.std(top_returns) if np.std(top_returns) > 0 else 0
        else:
            ic_top = np.nan
            sharpe_top = np.nan

        # Bottom percentile (lowest magnitude)
        threshold_bottom = np.percentile(preds_abs, pct)
        bottom_mask = preds_abs <= threshold_bottom

        if np.sum(bottom_mask) >= 10:
            bottom_preds = preds[bottom_mask]
            bottom_labels = labels[bottom_mask]

            if np.std(bottom_preds) > 0 and np.std(bottom_labels) > 0:
                ic_bottom, _ = pearsonr(bottom_preds, bottom_labels)
            else:
                ic_bottom = np.nan
        else:
            ic_bottom = np.nan

        results[f'top_{pct}'] = {'ic': ic_top, 'sharpe': sharpe_top, 'count': np.sum(top_mask)}
        results[f'bottom_{pct}'] = {'ic': ic_bottom, 'count': np.sum(bottom_mask)}

        logger.info(f"Top {pct}%: IC={ic_top:.4f}, Sharpe={sharpe_top:.3f}, N={np.sum(top_mask)}")

    return results


def threshold_sweep_analysis(pred_data: PredictionData, thresholds: Optional[List[float]] = None) -> Dict:
    """
    Sweep different magnitude thresholds to find optimal actionable threshold.

    Args:
        pred_data: PredictionData object
        thresholds: List of absolute magnitude thresholds to test (default: percentiles)

    Returns:
        Dict with results:
        - thresholds: Array of thresholds tested
        - ic_by_threshold: IC above each threshold
        - sharpe_by_threshold: Sharpe ratio above each threshold
        - hit_rate_by_threshold: Win rate above each threshold
        - trade_count_by_threshold: Number of trades above each threshold
    """
    preds = pred_data.predictions
    labels = pred_data.labels
    preds_abs = np.abs(preds)

    if thresholds is None:
        # Use percentiles as thresholds
        thresholds = np.percentile(preds_abs, np.linspace(0, 95, 20))

    ic_by_threshold = []
    sharpe_by_threshold = []
    hit_rate_by_threshold = []
    trade_count_by_threshold = []

    for thresh in thresholds:
        mask = preds_abs >= thresh

        if np.sum(mask) < 10:
            ic_by_threshold.append(np.nan)
            sharpe_by_threshold.append(np.nan)
            hit_rate_by_threshold.append(np.nan)
            trade_count_by_threshold.append(0)
            continue

        thresh_preds = preds[mask]
        thresh_labels = labels[mask]

        # IC
        if np.std(thresh_preds) > 0 and np.std(thresh_labels) > 0:
            ic, _ = pearsonr(thresh_preds, thresh_labels)
        else:
            ic = np.nan

        # Sharpe
        thresh_returns = thresh_preds * thresh_labels
        sharpe = np.mean(thresh_returns) / np.std(thresh_returns) if np.std(thresh_returns) > 0 else 0

        # Hit rate
        hit_rate = np.mean((thresh_preds * thresh_labels) > 0)

        ic_by_threshold.append(ic)
        sharpe_by_threshold.append(sharpe)
        hit_rate_by_threshold.append(hit_rate)
        trade_count_by_threshold.append(np.sum(mask))

    # Find optimal threshold (max Sharpe)
    optimal_idx = np.nanargmax(sharpe_by_threshold)
    optimal_threshold = thresholds[optimal_idx]
    optimal_sharpe = sharpe_by_threshold[optimal_idx]

    logger.info(f"Optimal threshold: {optimal_threshold:.3f} (Sharpe={optimal_sharpe:.3f})")

    return {
        'thresholds': np.array(thresholds),
        'ic_by_threshold': np.array(ic_by_threshold),
        'sharpe_by_threshold': np.array(sharpe_by_threshold),
        'hit_rate_by_threshold': np.array(hit_rate_by_threshold),
        'trade_count_by_threshold': np.array(trade_count_by_threshold),
        'optimal_threshold': optimal_threshold,
        'optimal_sharpe': optimal_sharpe,
    }


def plot_prediction_histogram(pred_data: PredictionData, save_path: Optional[Path] = None):
    """
    Plot prediction magnitude histogram.

    Args:
        pred_data: PredictionData object
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    preds = pred_data.predictions

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw predictions (with sign)
    ax1.hist(preds, bins=100, color='blue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(preds), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(preds):.3f}')
    ax1.axvline(np.median(preds), color='green', linestyle='--', linewidth=2, label=f'Median={np.median(preds):.3f}')
    ax1.set_xlabel('Prediction (ticks)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Prediction Distribution (Signed)', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Absolute predictions (magnitude)
    preds_abs = np.abs(preds)
    ax2.hist(preds_abs, bins=100, color='orange', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(preds_abs), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(preds_abs):.3f}')
    ax2.axvline(np.median(preds_abs), color='green', linestyle='--', linewidth=2, label=f'Median={np.median(preds_abs):.3f}')
    ax2.set_xlabel('|Prediction| (ticks)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Prediction Magnitude Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved prediction histogram to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_calibration(calibration: Dict, save_path: Optional[Path] = None):
    """
    Plot calibration curve: predicted magnitude vs actual magnitude.

    Args:
        calibration: Dict from calibration_analysis()
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    mean_pred = calibration['mean_pred_by_bin']
    mean_label = calibration['mean_label_by_bin']

    # Remove NaNs
    valid = ~(np.isnan(mean_pred) | np.isnan(mean_label))
    mean_pred = mean_pred[valid]
    mean_label = mean_label[valid]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(mean_pred, mean_label, s=100, alpha=0.6)

    # Perfect calibration line
    lim = max(mean_pred.max(), mean_label.max())
    ax.plot([0, lim], [0, lim], 'k--', linewidth=2, label='Perfect Calibration')

    # Fitted line
    slope = calibration['calibration_slope']
    intercept = calibration['calibration_intercept']
    if not np.isnan(slope):
        x_fit = np.array([0, lim])
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

    ax.set_xlabel('Mean Predicted Magnitude (ticks)', fontsize=12)
    ax.set_ylabel('Mean Actual Magnitude (ticks)', fontsize=12)
    ax.set_title('Prediction Calibration', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved calibration plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_threshold_sweep(sweep_results: Dict, save_path: Optional[Path] = None):
    """
    Plot IC, Sharpe, and trade count vs threshold.

    Args:
        sweep_results: Dict from threshold_sweep_analysis()
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    thresholds = sweep_results['thresholds']
    ic = sweep_results['ic_by_threshold']
    sharpe = sweep_results['sharpe_by_threshold']
    trade_count = sweep_results['trade_count_by_threshold']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # IC and Sharpe
    ax1_twin = ax1.twinx()
    ax1.plot(thresholds, ic, 'b-', marker='o', linewidth=2, label='IC')
    ax1_twin.plot(thresholds, sharpe, 'r-', marker='s', linewidth=2, label='Sharpe')

    # Mark optimal
    optimal_thresh = sweep_results['optimal_threshold']
    ax1.axvline(optimal_thresh, color='green', linestyle='--', alpha=0.5, label=f'Optimal: {optimal_thresh:.3f}')

    ax1.set_xlabel('Magnitude Threshold (ticks)', fontsize=12)
    ax1.set_ylabel('IC', fontsize=12, color='b')
    ax1_twin.set_ylabel('Sharpe Ratio', fontsize=12, color='r')
    ax1.set_title('IC and Sharpe vs Threshold', fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    # Trade count
    ax2.plot(thresholds, trade_count, 'g-', marker='o', linewidth=2)
    ax2.axvline(optimal_thresh, color='green', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Magnitude Threshold (ticks)', fontsize=12)
    ax2.set_ylabel('Trade Count', fontsize=12)
    ax2.set_title('Trade Count vs Threshold', fontsize=14)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved threshold sweep plot to {save_path}")
    else:
        plt.show()

    plt.close()
