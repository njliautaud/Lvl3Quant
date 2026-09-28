"""
Regime Analysis — IC by volatility regime, time of day, market conditions

Analyzes:
- IC by volatility regime (low, medium, high vol)
- IC by time of day (market open, mid-day, close)
- IC by spread regime (tight vs wide spreads)
- IC by trend regime (trending vs mean-reverting)
- Conditional performance analysis

This identifies when the model works best and when to avoid trading.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from scipy.stats import pearsonr
from datetime import datetime

from prediction_loader import PredictionData

logger = logging.getLogger(__name__)


def volatility_regime_analysis(
    pred_data: PredictionData,
    tick_data: np.ndarray,
    tick_timestamps: np.ndarray,
    lookback_ns: int = 300_000_000_000,  # 5 minutes
    num_regimes: int = 3
) -> Dict:
    """
    Analyze IC by volatility regime.

    Args:
        pred_data: PredictionData object
        tick_data: (M,) array of mid-prices
        tick_timestamps: (M,) array of timestamps (ns)
        lookback_ns: Lookback period for volatility calculation
        num_regimes: Number of volatility regimes (default: 3 = low/med/high)

    Returns:
        Dict with regime statistics:
        - regime_thresholds: Volatility thresholds for each regime
        - ic_by_regime: IC within each regime
        - count_by_regime: Number of predictions in each regime
        - sharpe_by_regime: Sharpe ratio by regime
    """
    if pred_data.timestamps is None:
        raise ValueError("pred_data must have timestamps for regime analysis")

    N = len(pred_data)

    # Calculate realized volatility at each prediction time
    realized_vol = np.zeros(N)

    for i in range(N):
        pred_ts = pred_data.timestamps[i]
        lookback_start = pred_ts - lookback_ns

        # Get ticks in lookback window
        mask = (tick_timestamps >= lookback_start) & (tick_timestamps < pred_ts)
        window_prices = tick_data[mask]

        if len(window_prices) >= 10:
            # Realized vol = std of returns
            returns = np.diff(window_prices)
            realized_vol[i] = np.std(returns)
        else:
            realized_vol[i] = np.nan

    # Remove NaN vols
    valid_mask = ~np.isnan(realized_vol)
    realized_vol = realized_vol[valid_mask]
    valid_preds = pred_data.predictions[valid_mask]
    valid_labels = pred_data.labels[valid_mask]

    # Define regime thresholds (percentiles)
    regime_percentiles = np.linspace(0, 100, num_regimes + 1)
    regime_thresholds = np.percentile(realized_vol, regime_percentiles)

    # Assign each prediction to a regime
    regime_assignments = np.digitize(realized_vol, regime_thresholds[1:-1])

    # Calculate IC by regime
    ic_by_regime = []
    count_by_regime = []
    sharpe_by_regime = []

    for r in range(num_regimes):
        regime_mask = regime_assignments == r

        if np.sum(regime_mask) < 10:
            ic_by_regime.append(np.nan)
            sharpe_by_regime.append(np.nan)
            count_by_regime.append(0)
            continue

        regime_preds = valid_preds[regime_mask]
        regime_labels = valid_labels[regime_mask]

        # IC
        if np.std(regime_preds) > 0 and np.std(regime_labels) > 0:
            ic, _ = pearsonr(regime_preds, regime_labels)
        else:
            ic = np.nan

        # Sharpe
        regime_returns = regime_preds * regime_labels
        sharpe = np.mean(regime_returns) / np.std(regime_returns) if np.std(regime_returns) > 0 else 0

        ic_by_regime.append(ic)
        sharpe_by_regime.append(sharpe)
        count_by_regime.append(np.sum(regime_mask))

        logger.info(f"Regime {r} (vol {regime_thresholds[r]:.3f}-{regime_thresholds[r+1]:.3f}): IC={ic:.4f}, N={np.sum(regime_mask)}")

    return {
        'regime_thresholds': regime_thresholds,
        'ic_by_regime': np.array(ic_by_regime),
        'sharpe_by_regime': np.array(sharpe_by_regime),
        'count_by_regime': np.array(count_by_regime),
        'realized_vol': realized_vol,
        'regime_assignments': regime_assignments
    }


def time_of_day_analysis(pred_data: PredictionData, hour_bins: int = 6) -> Dict:
    """
    Analyze IC by time of day.

    Args:
        pred_data: PredictionData object
        hour_bins: Number of hour bins (default: 6 = 4-hour bins for 24h)

    Returns:
        Dict with time-of-day statistics:
        - hour_edges: Hour bin edges
        - ic_by_hour: IC within each hour bin
        - count_by_hour: Number of predictions per bin
    """
    if pred_data.timestamps is None:
        raise ValueError("pred_data must have timestamps for time-of-day analysis")

    # Convert nanoseconds to datetime
    timestamps_sec = pred_data.timestamps / 1e9
    hours = np.array([datetime.fromtimestamp(ts).hour for ts in timestamps_sec])

    # Define hour bins
    hour_edges = np.linspace(0, 24, hour_bins + 1)
    hour_assignments = np.digitize(hours, hour_edges[1:-1])

    # Calculate IC by hour bin
    ic_by_hour = []
    count_by_hour = []
    sharpe_by_hour = []

    for h in range(hour_bins):
        hour_mask = hour_assignments == h

        if np.sum(hour_mask) < 10:
            ic_by_hour.append(np.nan)
            sharpe_by_hour.append(np.nan)
            count_by_hour.append(0)
            continue

        hour_preds = pred_data.predictions[hour_mask]
        hour_labels = pred_data.labels[hour_mask]

        # IC
        if np.std(hour_preds) > 0 and np.std(hour_labels) > 0:
            ic, _ = pearsonr(hour_preds, hour_labels)
        else:
            ic = np.nan

        # Sharpe
        hour_returns = hour_preds * hour_labels
        sharpe = np.mean(hour_returns) / np.std(hour_returns) if np.std(hour_returns) > 0 else 0

        ic_by_hour.append(ic)
        sharpe_by_hour.append(sharpe)
        count_by_hour.append(np.sum(hour_mask))

        logger.info(f"Hour bin {h} ({hour_edges[h]:.0f}-{hour_edges[h+1]:.0f}): IC={ic:.4f}, N={np.sum(hour_mask)}")

    return {
        'hour_edges': hour_edges,
        'ic_by_hour': np.array(ic_by_hour),
        'sharpe_by_hour': np.array(sharpe_by_hour),
        'count_by_hour': np.array(count_by_hour),
        'hour_assignments': hour_assignments
    }


def spread_regime_analysis(
    pred_data: PredictionData,
    spread_data: np.ndarray,
    spread_timestamps: np.ndarray,
    num_regimes: int = 3
) -> Dict:
    """
    Analyze IC by bid-ask spread regime.

    Args:
        pred_data: PredictionData object
        spread_data: (M,) array of bid-ask spreads (in ticks)
        spread_timestamps: (M,) array of timestamps (ns)
        num_regimes: Number of spread regimes

    Returns:
        Dict with spread regime statistics
    """
    if pred_data.timestamps is None:
        raise ValueError("pred_data must have timestamps")

    N = len(pred_data)
    pred_spreads = np.zeros(N)

    # Match spreads to predictions
    for i in range(N):
        pred_ts = pred_data.timestamps[i]
        idx = np.searchsorted(spread_timestamps, pred_ts, side='left')

        if idx < len(spread_data):
            pred_spreads[i] = spread_data[idx]
        else:
            pred_spreads[i] = np.nan

    # Remove NaNs
    valid_mask = ~np.isnan(pred_spreads)
    pred_spreads = pred_spreads[valid_mask]
    valid_preds = pred_data.predictions[valid_mask]
    valid_labels = pred_data.labels[valid_mask]

    # Define spread regimes (percentiles)
    regime_percentiles = np.linspace(0, 100, num_regimes + 1)
    regime_thresholds = np.percentile(pred_spreads, regime_percentiles)

    regime_assignments = np.digitize(pred_spreads, regime_thresholds[1:-1])

    # Calculate IC by regime
    ic_by_regime = []
    count_by_regime = []

    for r in range(num_regimes):
        regime_mask = regime_assignments == r

        if np.sum(regime_mask) < 10:
            ic_by_regime.append(np.nan)
            count_by_regime.append(0)
            continue

        regime_preds = valid_preds[regime_mask]
        regime_labels = valid_labels[regime_mask]

        if np.std(regime_preds) > 0 and np.std(regime_labels) > 0:
            ic, _ = pearsonr(regime_preds, regime_labels)
        else:
            ic = np.nan

        ic_by_regime.append(ic)
        count_by_regime.append(np.sum(regime_mask))

        logger.info(f"Spread regime {r} ({regime_thresholds[r]:.1f}-{regime_thresholds[r+1]:.1f} ticks): IC={ic:.4f}")

    return {
        'regime_thresholds': regime_thresholds,
        'ic_by_regime': np.array(ic_by_regime),
        'count_by_regime': np.array(count_by_regime),
    }


def trend_regime_analysis(
    pred_data: PredictionData,
    tick_data: np.ndarray,
    tick_timestamps: np.ndarray,
    lookback_ns: int = 300_000_000_000,  # 5 minutes
    num_regimes: int = 3
) -> Dict:
    """
    Analyze IC by trend regime (trending vs mean-reverting).

    Args:
        pred_data: PredictionData object
        tick_data: (M,) array of mid-prices
        tick_timestamps: (M,) array of timestamps (ns)
        lookback_ns: Lookback period for trend calculation
        num_regimes: Number of trend regimes

    Returns:
        Dict with trend regime statistics

    Trend measure: autocorrelation of returns over lookback period
    High autocorr = trending, low/negative = mean-reverting
    """
    if pred_data.timestamps is None:
        raise ValueError("pred_data must have timestamps")

    N = len(pred_data)
    trend_metric = np.zeros(N)

    for i in range(N):
        pred_ts = pred_data.timestamps[i]
        lookback_start = pred_ts - lookback_ns

        # Get ticks in lookback window
        mask = (tick_timestamps >= lookback_start) & (tick_timestamps < pred_ts)
        window_prices = tick_data[mask]

        if len(window_prices) >= 20:
            # Calculate autocorrelation of returns
            returns = np.diff(window_prices)
            if len(returns) > 10 and np.std(returns) > 0:
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                trend_metric[i] = autocorr
            else:
                trend_metric[i] = np.nan
        else:
            trend_metric[i] = np.nan

    # Remove NaNs
    valid_mask = ~np.isnan(trend_metric)
    trend_metric = trend_metric[valid_mask]
    valid_preds = pred_data.predictions[valid_mask]
    valid_labels = pred_data.labels[valid_mask]

    # Define trend regimes
    regime_percentiles = np.linspace(0, 100, num_regimes + 1)
    regime_thresholds = np.percentile(trend_metric, regime_percentiles)

    regime_assignments = np.digitize(trend_metric, regime_thresholds[1:-1])

    # Calculate IC by regime
    ic_by_regime = []
    count_by_regime = []

    regime_names = ['Mean-reverting', 'Neutral', 'Trending'] if num_regimes == 3 else [f'Regime_{r}' for r in range(num_regimes)]

    for r in range(num_regimes):
        regime_mask = regime_assignments == r

        if np.sum(regime_mask) < 10:
            ic_by_regime.append(np.nan)
            count_by_regime.append(0)
            continue

        regime_preds = valid_preds[regime_mask]
        regime_labels = valid_labels[regime_mask]

        if np.std(regime_preds) > 0 and np.std(regime_labels) > 0:
            ic, _ = pearsonr(regime_preds, regime_labels)
        else:
            ic = np.nan

        ic_by_regime.append(ic)
        count_by_regime.append(np.sum(regime_mask))

        logger.info(f"{regime_names[r]} (autocorr {regime_thresholds[r]:.3f}-{regime_thresholds[r+1]:.3f}): IC={ic:.4f}")

    return {
        'regime_thresholds': regime_thresholds,
        'regime_names': regime_names,
        'ic_by_regime': np.array(ic_by_regime),
        'count_by_regime': np.array(count_by_regime),
        'trend_metric': trend_metric,
        'regime_assignments': regime_assignments
    }


def plot_regime_comparison(
    regime_results: Dict,
    regime_type: str,
    save_path: Optional[Path] = None
):
    """
    Plot IC comparison across regimes.

    Args:
        regime_results: Dict from regime analysis function
        regime_type: Type of regime ("volatility", "time_of_day", etc.)
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    if regime_type == 'time_of_day':
        x_labels = [f"{int(regime_results['hour_edges'][i])}-{int(regime_results['hour_edges'][i+1])}h"
                    for i in range(len(regime_results['ic_by_hour']))]
        ic = regime_results['ic_by_hour']
        counts = regime_results['count_by_hour']
    elif regime_type == 'volatility':
        x_labels = [f"R{i}" for i in range(len(regime_results['ic_by_regime']))]
        ic = regime_results['ic_by_regime']
        counts = regime_results['count_by_regime']
    elif regime_type == 'trend':
        x_labels = regime_results.get('regime_names', [f"R{i}" for i in range(len(regime_results['ic_by_regime']))])
        ic = regime_results['ic_by_regime']
        counts = regime_results['count_by_regime']
    else:
        x_labels = [f"R{i}" for i in range(len(regime_results['ic_by_regime']))]
        ic = regime_results['ic_by_regime']
        counts = regime_results['count_by_regime']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # IC by regime
    x_pos = np.arange(len(x_labels))
    colors = ['green' if i > 0 else 'red' for i in ic]

    ax1.bar(x_pos, ic, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.set_ylabel('IC', fontsize=12)
    ax1.set_title(f'IC by {regime_type.replace("_", " ").title()} Regime', fontsize=14)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.grid(alpha=0.3)

    # Count by regime
    ax2.bar(x_pos, counts, color='blue', alpha=0.7, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Sample Count by Regime', fontsize=14)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved regime comparison plot to {save_path}")
    else:
        plt.show()

    plt.close()


def multi_regime_conditional_analysis(
    pred_data: PredictionData,
    regimes: Dict[str, np.ndarray]
) -> Dict:
    """
    Analyze IC conditioned on multiple regime variables simultaneously.

    Args:
        pred_data: PredictionData object
        regimes: Dict mapping regime name to (N,) array of regime assignments

    Returns:
        Dict with conditional IC statistics for each regime combination

    Example:
        regimes = {
            'volatility': vol_regime_assignments,
            'time_of_day': hour_assignments,
            'spread': spread_regime_assignments
        }
    """
    # Combine regimes into unique combinations
    N = len(pred_data)

    # Create combined regime index
    regime_keys = sorted(regimes.keys())
    regime_arrays = [regimes[k] for k in regime_keys]

    # Generate unique combinations
    from itertools import product

    num_regimes_per_var = [int(np.max(r)) + 1 for r in regime_arrays]
    all_combinations = list(product(*[range(n) for n in num_regimes_per_var]))

    results = {}

    for combo in all_combinations:
        # Build mask for this combination
        mask = np.ones(N, dtype=bool)
        combo_name = []

        for i, (key, regime_val) in enumerate(zip(regime_keys, combo)):
            mask &= regime_arrays[i] == regime_val
            combo_name.append(f"{key}={regime_val}")

        combo_name = ", ".join(combo_name)

        if np.sum(mask) < 10:
            continue

        combo_preds = pred_data.predictions[mask]
        combo_labels = pred_data.labels[mask]

        if np.std(combo_preds) > 0 and np.std(combo_labels) > 0:
            ic, _ = pearsonr(combo_preds, combo_labels)
        else:
            ic = np.nan

        results[combo_name] = {
            'ic': ic,
            'count': np.sum(mask),
            'regime_combo': combo
        }

        logger.info(f"{combo_name}: IC={ic:.4f}, N={np.sum(mask)}")

    return results
