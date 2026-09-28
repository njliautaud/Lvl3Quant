"""
Time Decay Analysis — How IC decays over time after prediction

Measures:
- IC at different horizons (1s, 2s, 5s, 10s, 30s, 60s)
- Half-life of prediction signal
- Optimal holding period
- Prediction stability over time

This informs execution strategy: how quickly to act on signal, how long to hold.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from scipy.stats import pearsonr
from dataclasses import dataclass

from prediction_loader import PredictionData

logger = logging.getLogger(__name__)


@dataclass
class TimeDecayMetrics:
    """Container for time decay analysis results."""
    horizons_ns: np.ndarray  # Horizons in nanoseconds
    horizons_sec: np.ndarray  # Horizons in seconds
    ic_by_horizon: np.ndarray  # IC at each horizon
    pvalue_by_horizon: np.ndarray  # P-values
    half_life_sec: Optional[float] = None  # Half-life of signal
    optimal_horizon_sec: Optional[float] = None  # Horizon with best IC


def calculate_time_decay(
    pred_data: PredictionData,
    tick_data: np.ndarray,
    tick_timestamps: np.ndarray,
    horizons_sec: List[float] = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60]
) -> TimeDecayMetrics:
    """
    Calculate IC decay over multiple time horizons.

    Args:
        pred_data: PredictionData object with predictions
        tick_data: (M,) array of mid-prices (all ticks)
        tick_timestamps: (M,) array of tick timestamps (nanoseconds)
        horizons_sec: List of horizons to evaluate (in seconds)

    Returns:
        TimeDecayMetrics object

    Note: For each prediction, calculates IC between prediction and actual
    price change at each horizon.
    """
    if pred_data.timestamps is None:
        raise ValueError("pred_data must have timestamps for time decay analysis")

    N = len(pred_data)
    horizons_ns = np.array([h * 1e9 for h in horizons_sec], dtype=np.int64)
    ic_by_horizon = np.zeros(len(horizons_ns))
    pvalue_by_horizon = np.zeros(len(horizons_ns))

    for h_idx, horizon_ns in enumerate(horizons_ns):
        # Calculate actual returns at this horizon
        actual_returns = np.zeros(N)
        valid_mask = np.zeros(N, dtype=bool)

        for i in range(N):
            pred_ts = pred_data.timestamps[i]
            target_ts = pred_ts + horizon_ns

            # Find entry and exit ticks
            entry_idx = np.searchsorted(tick_timestamps, pred_ts, side='left')
            exit_idx = np.searchsorted(tick_timestamps, target_ts, side='left')

            if entry_idx >= len(tick_data) or exit_idx >= len(tick_data):
                continue

            entry_price = tick_data[entry_idx]
            exit_price = tick_data[exit_idx]

            actual_returns[i] = exit_price - entry_price
            valid_mask[i] = True

        # Calculate IC at this horizon
        if np.sum(valid_mask) < 10:
            # Not enough valid data
            ic_by_horizon[h_idx] = np.nan
            pvalue_by_horizon[h_idx] = np.nan
            continue

        valid_preds = pred_data.predictions[valid_mask]
        valid_returns = actual_returns[valid_mask]

        if np.std(valid_preds) > 0 and np.std(valid_returns) > 0:
            ic, pval = pearsonr(valid_preds, valid_returns)
            ic_by_horizon[h_idx] = ic
            pvalue_by_horizon[h_idx] = pval
        else:
            ic_by_horizon[h_idx] = np.nan
            pvalue_by_horizon[h_idx] = np.nan

    # Calculate half-life (horizon where IC drops to 50% of peak)
    peak_ic = np.nanmax(ic_by_horizon)
    if peak_ic > 0:
        half_life_ic = peak_ic * 0.5
        # Find first horizon where IC drops below half-life
        above_half = ic_by_horizon >= half_life_ic
        if np.any(~above_half):
            half_life_idx = np.where(~above_half)[0][0]
            half_life_sec = horizons_sec[half_life_idx]
        else:
            half_life_sec = horizons_sec[-1]  # Signal persists beyond measured range
    else:
        half_life_sec = None

    # Find optimal horizon (max IC)
    optimal_idx = np.nanargmax(ic_by_horizon)
    optimal_horizon_sec = horizons_sec[optimal_idx]

    logger.info(f"Time decay analysis: peak IC={peak_ic:.4f} at {optimal_horizon_sec}s, half-life={half_life_sec}s")

    return TimeDecayMetrics(
        horizons_ns=horizons_ns,
        horizons_sec=np.array(horizons_sec),
        ic_by_horizon=ic_by_horizon,
        pvalue_by_horizon=pvalue_by_horizon,
        half_life_sec=half_life_sec,
        optimal_horizon_sec=optimal_horizon_sec
    )


def calculate_time_decay_from_labels(
    predictions: np.ndarray,
    labels_by_horizon: Dict[str, np.ndarray]
) -> TimeDecayMetrics:
    """
    Calculate time decay from pre-computed labels at multiple horizons.

    Args:
        predictions: (N,) array of predictions
        labels_by_horizon: Dict mapping horizon (e.g., "1s", "5s", "10s") to label arrays

    Returns:
        TimeDecayMetrics object

    Note: This is a simplified version when you have pre-computed labels
    at standard horizons (e.g., from .npz files with labels_1s, labels_5s, etc.)
    """
    # Parse horizons
    horizon_map = {}
    for key, labels in labels_by_horizon.items():
        # Extract seconds from key (e.g., "1s" -> 1.0)
        if 's' in key:
            sec = float(key.replace('s', ''))
            horizon_map[sec] = labels

    horizons_sec = sorted(horizon_map.keys())
    horizons_ns = np.array([h * 1e9 for h in horizons_sec], dtype=np.int64)

    ic_by_horizon = np.zeros(len(horizons_sec))
    pvalue_by_horizon = np.zeros(len(horizons_sec))

    for i, sec in enumerate(horizons_sec):
        labels = horizon_map[sec]

        if len(predictions) != len(labels):
            logger.warning(f"Prediction/label length mismatch at horizon {sec}s")
            ic_by_horizon[i] = np.nan
            pvalue_by_horizon[i] = np.nan
            continue

        # Remove NaNs
        valid_mask = ~(np.isnan(predictions) | np.isnan(labels))
        valid_preds = predictions[valid_mask]
        valid_labels = labels[valid_mask]

        if len(valid_preds) < 10:
            ic_by_horizon[i] = np.nan
            pvalue_by_horizon[i] = np.nan
            continue

        if np.std(valid_preds) > 0 and np.std(valid_labels) > 0:
            ic, pval = pearsonr(valid_preds, valid_labels)
            ic_by_horizon[i] = ic
            pvalue_by_horizon[i] = pval
        else:
            ic_by_horizon[i] = np.nan
            pvalue_by_horizon[i] = np.nan

    # Calculate half-life
    peak_ic = np.nanmax(ic_by_horizon)
    if peak_ic > 0:
        half_life_ic = peak_ic * 0.5
        above_half = ic_by_horizon >= half_life_ic
        if np.any(~above_half):
            half_life_idx = np.where(~above_half)[0][0]
            half_life_sec = horizons_sec[half_life_idx]
        else:
            half_life_sec = horizons_sec[-1]
    else:
        half_life_sec = None

    # Find optimal horizon
    optimal_idx = np.nanargmax(ic_by_horizon)
    optimal_horizon_sec = horizons_sec[optimal_idx]

    logger.info(f"Time decay: peak IC={peak_ic:.4f} at {optimal_horizon_sec}s, half-life={half_life_sec}s")

    return TimeDecayMetrics(
        horizons_ns=horizons_ns,
        horizons_sec=np.array(horizons_sec),
        ic_by_horizon=ic_by_horizon,
        pvalue_by_horizon=pvalue_by_horizon,
        half_life_sec=half_life_sec,
        optimal_horizon_sec=optimal_horizon_sec
    )


def plot_time_decay(
    decay_metrics: TimeDecayMetrics,
    save_path: Optional[Path] = None
):
    """
    Plot IC decay curve over time.

    Args:
        decay_metrics: TimeDecayMetrics object
        save_path: Path to save figure (optional)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot IC vs horizon
    ax.plot(decay_metrics.horizons_sec, decay_metrics.ic_by_horizon,
            marker='o', linewidth=2, markersize=6)

    # Mark peak IC
    peak_idx = np.nanargmax(decay_metrics.ic_by_horizon)
    peak_horizon = decay_metrics.horizons_sec[peak_idx]
    peak_ic = decay_metrics.ic_by_horizon[peak_idx]

    ax.axvline(peak_horizon, color='green', linestyle='--', alpha=0.5,
               label=f'Peak: {peak_ic:.4f} @ {peak_horizon}s')

    # Mark half-life
    if decay_metrics.half_life_sec is not None:
        ax.axvline(decay_metrics.half_life_sec, color='red', linestyle='--', alpha=0.5,
                   label=f'Half-life: {decay_metrics.half_life_sec:.1f}s')

    ax.set_xlabel('Horizon (seconds)', fontsize=12)
    ax.set_ylabel('Information Coefficient (IC)', fontsize=12)
    ax.set_title('IC Time Decay Analysis', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved time decay plot to {save_path}")
    else:
        plt.show()

    plt.close()


def compare_time_decay(
    decay_metrics_list: List[Tuple[str, TimeDecayMetrics]],
    save_path: Optional[Path] = None
):
    """
    Compare time decay curves for multiple models.

    Args:
        decay_metrics_list: List of (model_name, TimeDecayMetrics) tuples
        save_path: Path to save figure (optional)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))

    for model_name, metrics in decay_metrics_list:
        ax.plot(metrics.horizons_sec, metrics.ic_by_horizon,
                marker='o', linewidth=2, markersize=5, label=model_name)

    ax.set_xlabel('Horizon (seconds)', fontsize=12)
    ax.set_ylabel('Information Coefficient (IC)', fontsize=12)
    ax.set_title('Model Comparison: IC Time Decay', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved time decay comparison to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_prediction_stability(
    pred_data: PredictionData,
    tick_data: np.ndarray,
    tick_timestamps: np.ndarray,
    horizon_ns: int = 10_000_000_000,
    window_ns: int = 1_000_000_000
) -> Dict:
    """
    Analyze prediction stability: does the signal remain consistent or reverse?

    Args:
        pred_data: PredictionData object
        tick_data: (M,) array of mid-prices
        tick_timestamps: (M,) array of timestamps (ns)
        horizon_ns: Target horizon (default 10s)
        window_ns: Window size for intermediate checks (default 1s)

    Returns:
        Dict with stability metrics:
        - reversal_rate: fraction of predictions that reverse sign
        - mean_correlation: average correlation between intermediate and final returns
        - stability_score: composite stability metric
    """
    if pred_data.timestamps is None:
        raise ValueError("pred_data must have timestamps")

    N = len(pred_data)
    reversals = 0
    correlations = []

    num_windows = int(horizon_ns / window_ns)

    for i in range(N):
        pred_ts = pred_data.timestamps[i]
        pred_sign = np.sign(pred_data.predictions[i])

        if pred_sign == 0:
            continue

        # Track returns at intermediate windows
        intermediate_returns = []
        for w in range(1, num_windows + 1):
            target_ts = pred_ts + w * window_ns
            entry_idx = np.searchsorted(tick_timestamps, pred_ts, side='left')
            exit_idx = np.searchsorted(tick_timestamps, target_ts, side='left')

            if entry_idx >= len(tick_data) or exit_idx >= len(tick_data):
                break

            ret = tick_data[exit_idx] - tick_data[entry_idx]
            intermediate_returns.append(ret)

        if len(intermediate_returns) < 2:
            continue

        # Check if sign reversed
        final_sign = np.sign(intermediate_returns[-1])
        if final_sign != 0 and final_sign != pred_sign:
            reversals += 1

        # Correlation between intermediate and final
        intermediate_returns = np.array(intermediate_returns)
        if np.std(intermediate_returns) > 0:
            corr = np.corrcoef(np.arange(len(intermediate_returns)), intermediate_returns)[0, 1]
            correlations.append(corr)

    reversal_rate = reversals / N if N > 0 else 0
    mean_correlation = np.mean(correlations) if len(correlations) > 0 else 0
    stability_score = (1 - reversal_rate) * (1 + mean_correlation) / 2

    logger.info(f"Stability analysis: reversal_rate={reversal_rate:.3f}, mean_corr={mean_correlation:.3f}")

    return {
        'reversal_rate': reversal_rate,
        'mean_correlation': mean_correlation,
        'stability_score': stability_score,
        'num_samples': N
    }
