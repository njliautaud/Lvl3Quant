"""
MFE/MAE Analysis — Maximum Favorable/Adverse Excursion Analysis

Analyzes how far price moves in favor (MFE) and against (MAE) each prediction
before reaching the target timeframe. Critical for setting stops and profit targets.

Key metrics:
- MFE: Maximum favorable excursion (best price before target)
- MAE: Maximum adverse excursion (worst price before target)
- MFE/MAE ratio: Risk-reward ratio
- Win rate at different thresholds
- Optimal stop/target levels

This requires tick-by-tick or high-resolution price data to track path-dependent moves.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from prediction_loader import PredictionData

logger = logging.getLogger(__name__)


@dataclass
class ExcursionMetrics:
    """Container for MFE/MAE metrics."""
    mfe: np.ndarray  # Maximum favorable excursion per prediction
    mae: np.ndarray  # Maximum adverse excursion per prediction
    final_pnl: np.ndarray  # Final P&L at target time
    hit_target: np.ndarray  # Boolean: did price reach predicted direction target
    time_to_mfe: Optional[np.ndarray] = None  # Time to reach MFE (if available)
    time_to_mae: Optional[np.ndarray] = None  # Time to reach MAE (if available)


def calculate_mfe_mae_from_price_path(
    predictions: np.ndarray,
    price_paths: np.ndarray,
    target_horizon_idx: int = -1
) -> ExcursionMetrics:
    """
    Calculate MFE/MAE from full price paths.

    Args:
        predictions: (N,) array of predicted price changes
        price_paths: (N, T) array of price paths from entry to target
                     Each row is cumulative price change from entry
        target_horizon_idx: Index of target time (-1 = last)

    Returns:
        ExcursionMetrics object

    Note: price_paths should be in ticks or basis points (same units as predictions)
    """
    N = len(predictions)
    assert price_paths.shape[0] == N, "Price paths must match prediction count"

    mfe = np.zeros(N)
    mae = np.zeros(N)
    time_to_mfe = np.zeros(N, dtype=np.int32)
    time_to_mae = np.zeros(N, dtype=np.int32)

    for i in range(N):
        path = price_paths[i]
        pred_sign = np.sign(predictions[i])

        if pred_sign == 0:
            # No directional prediction
            mfe[i] = 0
            mae[i] = 0
            continue

        # Favorable = move in predicted direction
        # Adverse = move against predicted direction
        if pred_sign > 0:
            # Long prediction: favor = positive, adverse = negative
            mfe[i] = np.max(path)
            mae[i] = np.min(path)
            time_to_mfe[i] = np.argmax(path)
            time_to_mae[i] = np.argmin(path)
        else:
            # Short prediction: favor = negative, adverse = positive
            mfe[i] = -np.min(path)
            mae[i] = -np.max(path)
            time_to_mfe[i] = np.argmin(path)
            time_to_mae[i] = np.argmax(path)

    # Final P&L at target
    final_pnl = price_paths[:, target_horizon_idx] * np.sign(predictions)

    # Did we hit target?
    hit_target = final_pnl > 0

    return ExcursionMetrics(
        mfe=mfe,
        mae=mae,
        final_pnl=final_pnl,
        hit_target=hit_target,
        time_to_mfe=time_to_mfe,
        time_to_mae=time_to_mae
    )


def calculate_mfe_mae_simple(
    pred_data: PredictionData,
    tick_data: np.ndarray,
    tick_timestamps: np.ndarray,
    horizon_ns: int = 10_000_000_000  # 10 seconds default
) -> ExcursionMetrics:
    """
    Calculate MFE/MAE from tick data (simpler interface).

    Args:
        pred_data: PredictionData object
        tick_data: (M,) array of mid-prices (all ticks)
        tick_timestamps: (M,) array of tick timestamps (nanoseconds)
        horizon_ns: Target horizon in nanoseconds

    Returns:
        ExcursionMetrics object

    Note: This assumes pred_data.timestamps aligns with tick_timestamps.
    For each prediction, finds the corresponding tick and tracks path to target.
    """
    if pred_data.timestamps is None:
        raise ValueError("pred_data must have timestamps for tick-level analysis")

    N = len(pred_data)
    price_paths = []

    for i in range(N):
        pred_ts = pred_data.timestamps[i]
        target_ts = pred_ts + horizon_ns

        # Find ticks in range [pred_ts, target_ts]
        mask = (tick_timestamps >= pred_ts) & (tick_timestamps <= target_ts)
        path_ticks = tick_data[mask]

        if len(path_ticks) < 2:
            # Not enough data — skip or use zeros
            price_paths.append(np.zeros(100))  # Dummy path
            continue

        # Convert to cumulative price change from entry
        entry_price = path_ticks[0]
        path = path_ticks - entry_price

        # Pad/truncate to fixed length for array storage
        # Alternative: store variable-length paths in list
        path = np.pad(path, (0, max(0, 100 - len(path))), mode='edge')[:100]
        price_paths.append(path)

    price_paths = np.array(price_paths)

    return calculate_mfe_mae_from_price_path(
        pred_data.predictions,
        price_paths,
        target_horizon_idx=-1
    )


def analyze_mfe_mae(excursion: ExcursionMetrics) -> Dict:
    """
    Analyze MFE/MAE metrics and return summary statistics.

    Returns:
        Dict with summary metrics:
        - mean_mfe, median_mfe, std_mfe
        - mean_mae, median_mae, std_mae
        - mfe_mae_ratio (mean)
        - win_rate
        - avg_win, avg_loss
        - expectancy
    """
    # Remove zero predictions (no directional bias)
    mask = (excursion.mfe != 0) | (excursion.mae != 0)

    mfe = excursion.mfe[mask]
    mae = excursion.mae[mask]
    final_pnl = excursion.final_pnl[mask]
    hit_target = excursion.hit_target[mask]

    # Basic stats
    stats = {
        'count': len(mfe),
        'mean_mfe': np.mean(mfe),
        'median_mfe': np.median(mfe),
        'std_mfe': np.std(mfe),
        'mean_mae': np.mean(mae),
        'median_mae': np.median(mae),
        'std_mae': np.std(mae),
        'mfe_mae_ratio': np.mean(mfe) / np.mean(np.abs(mae)) if np.mean(np.abs(mae)) > 0 else np.inf,
    }

    # Win/loss stats
    winners = final_pnl > 0
    losers = final_pnl < 0

    stats['win_rate'] = np.mean(winners)
    stats['avg_win'] = np.mean(final_pnl[winners]) if np.any(winners) else 0
    stats['avg_loss'] = np.mean(final_pnl[losers]) if np.any(losers) else 0
    stats['expectancy'] = stats['win_rate'] * stats['avg_win'] + (1 - stats['win_rate']) * stats['avg_loss']

    # Time to MFE/MAE (if available)
    if excursion.time_to_mfe is not None:
        stats['mean_time_to_mfe'] = np.mean(excursion.time_to_mfe[mask])
        stats['median_time_to_mfe'] = np.median(excursion.time_to_mfe[mask])
    if excursion.time_to_mae is not None:
        stats['mean_time_to_mae'] = np.mean(excursion.time_to_mae[mask])
        stats['median_time_to_mae'] = np.median(excursion.time_to_mae[mask])

    return stats


def optimal_stop_target(
    excursion: ExcursionMetrics,
    risk_reward_ratio: float = 2.0
) -> Tuple[float, float]:
    """
    Find optimal stop and target levels based on MFE/MAE distribution.

    Args:
        excursion: ExcursionMetrics object
        risk_reward_ratio: Desired risk/reward ratio (target/stop)

    Returns:
        (optimal_stop, optimal_target) in same units as predictions

    Strategy: Maximize expectancy over different stop/target combinations.
    """
    # Remove zero predictions
    mask = (excursion.mfe != 0) | (excursion.mae != 0)
    mfe = excursion.mfe[mask]
    mae = excursion.mae[mask]
    final_pnl = excursion.final_pnl[mask]

    # Grid search over stop levels
    # Stop is placed at worst MAE percentile
    # Target is placed at best MFE percentile
    mae_abs = np.abs(mae)

    best_expectancy = -np.inf
    best_stop = 0
    best_target = 0

    # Try different stop percentiles
    for stop_pct in range(10, 91, 5):
        stop_level = np.percentile(mae_abs, stop_pct)
        target_level = stop_level * risk_reward_ratio

        # Simulate outcomes
        stopped_out = mae_abs >= stop_level
        hit_target = mfe >= target_level

        # P&L simulation
        pnl = np.zeros(len(mfe))
        pnl[stopped_out] = -stop_level  # Stop loss hit
        pnl[hit_target & ~stopped_out] = target_level  # Target hit first
        pnl[~stopped_out & ~hit_target] = final_pnl[~stopped_out & ~hit_target]  # Neither hit

        expectancy = np.mean(pnl)

        if expectancy > best_expectancy:
            best_expectancy = expectancy
            best_stop = stop_level
            best_target = target_level

    logger.info(f"Optimal stop={best_stop:.3f}, target={best_target:.3f}, expectancy={best_expectancy:.3f}")

    return best_stop, best_target


def plot_mfe_mae_scatter(
    excursion: ExcursionMetrics,
    save_path: Optional[Path] = None
):
    """
    Plot MFE vs MAE scatter to visualize risk/reward distribution.

    Args:
        excursion: ExcursionMetrics object
        save_path: Path to save figure (optional)
    """
    import matplotlib.pyplot as plt

    # Remove zeros
    mask = (excursion.mfe != 0) | (excursion.mae != 0)
    mfe = excursion.mfe[mask]
    mae = excursion.mae[mask]
    hit_target = excursion.hit_target[mask]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by win/loss
    colors = ['green' if ht else 'red' for ht in hit_target]

    ax.scatter(mae, mfe, c=colors, alpha=0.3, s=10)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    # Add diagonal line (MFE = -MAE)
    lim = max(np.abs(ax.get_xlim()).max(), np.abs(ax.get_ylim()).max())
    ax.plot([-lim, lim], [lim, -lim], 'k--', alpha=0.3, label='MFE = -MAE')

    ax.set_xlabel('MAE (Maximum Adverse Excursion)', fontsize=12)
    ax.set_ylabel('MFE (Maximum Favorable Excursion)', fontsize=12)
    ax.set_title('MFE vs MAE Scatter', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved MFE/MAE scatter to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_mfe_mae_histograms(
    excursion: ExcursionMetrics,
    save_path: Optional[Path] = None
):
    """
    Plot MFE and MAE histograms side-by-side.

    Args:
        excursion: ExcursionMetrics object
        save_path: Path to save figure (optional)
    """
    import matplotlib.pyplot as plt

    # Remove zeros
    mask = (excursion.mfe != 0) | (excursion.mae != 0)
    mfe = excursion.mfe[mask]
    mae = excursion.mae[mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # MFE histogram
    ax1.hist(mfe, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(mfe), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(mfe):.2f}')
    ax1.axvline(np.median(mfe), color='blue', linestyle='--', linewidth=2, label=f'Median={np.median(mfe):.2f}')
    ax1.set_xlabel('MFE (ticks)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Maximum Favorable Excursion', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # MAE histogram (absolute value)
    mae_abs = np.abs(mae)
    ax2.hist(mae_abs, bins=50, color='red', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(mae_abs), color='green', linestyle='--', linewidth=2, label=f'Mean={np.mean(mae_abs):.2f}')
    ax2.axvline(np.median(mae_abs), color='blue', linestyle='--', linewidth=2, label=f'Median={np.median(mae_abs):.2f}')
    ax2.set_xlabel('|MAE| (ticks)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Maximum Adverse Excursion', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved MFE/MAE histograms to {save_path}")
    else:
        plt.show()

    plt.close()
