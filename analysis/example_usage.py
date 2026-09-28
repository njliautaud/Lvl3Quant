"""
Example usage of the prediction analysis framework.

Demonstrates how to load predictions and generate exploitability reports.
"""

import sys
import logging
from pathlib import Path

# Add analysis dir to path
sys.path.insert(0, str(Path(__file__).parent))

from prediction_loader import load_npz_predictions, load_model_predictions
from prediction_distribution import (
    analyze_prediction_magnitude,
    calibration_analysis,
    threshold_sweep_analysis,
    percentile_ic_analysis
)
from time_decay_analysis import calculate_time_decay_from_labels
from exploitability_report import generate_exploitability_report, compare_models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_analysis():
    """Example 1: Basic analysis of a single prediction file."""
    logger.info("="*80)
    logger.info("EXAMPLE 1: Basic Analysis")
    logger.info("="*80)

    # Path to prediction file
    pred_file = Path("/home/jupiter/Lvl3Quant/data/cnn_s76_concat_oot_predictions.npz")

    if not pred_file.exists():
        logger.warning(f"Prediction file not found: {pred_file}")
        return

    # Load predictions
    pred_data = load_npz_predictions(pred_file, timeframe="10s")
    logger.info(f"Loaded: {pred_data}")

    # Analyze prediction magnitude
    mag_stats = analyze_prediction_magnitude(pred_data)
    logger.info(f"Mean magnitude: {mag_stats['mean_abs']:.3f} ticks")
    logger.info(f"Median magnitude: {mag_stats['median_abs']:.3f} ticks")
    logger.info(f"90th percentile: {mag_stats['percentiles']['p90']:.3f} ticks")

    # Calibration analysis
    calib = calibration_analysis(pred_data, num_bins=10)
    logger.info(f"Calibration slope: {calib['calibration_slope']:.3f} (should be ~1.0)")
    logger.info(f"Rank correlation: {calib['rank_correlation']:.3f}")

    # Threshold sweep
    sweep = threshold_sweep_analysis(pred_data)
    logger.info(f"Optimal threshold: {sweep['optimal_threshold']:.3f} ticks")
    logger.info(f"Sharpe at optimal: {sweep['optimal_sharpe']:.3f}")

    # Percentile IC
    pct_ic = percentile_ic_analysis(pred_data, percentiles=[10, 20, 30])
    for pct in [10, 20, 30]:
        if f'top_{pct}' in pct_ic:
            ic = pct_ic[f'top_{pct}']['ic']
            sharpe = pct_ic[f'top_{pct}']['sharpe']
            logger.info(f"Top {pct}%: IC={ic:.4f}, Sharpe={sharpe:.3f}")


def example_2_time_decay():
    """Example 2: Time decay analysis using multiple horizons."""
    logger.info("="*80)
    logger.info("EXAMPLE 2: Time Decay Analysis")
    logger.info("="*80)

    pred_file = Path("/home/jupiter/Lvl3Quant/data/cnn_s76_concat_oot_predictions.npz")

    if not pred_file.exists():
        logger.warning(f"Prediction file not found: {pred_file}")
        return

    # Load raw data to get all horizons
    import numpy as np
    data = np.load(pred_file)

    # Extract predictions and labels at multiple horizons
    predictions = data['preds_10s']  # Use 10s predictions as baseline

    labels_by_horizon = {
        '1s': data['labels_1s'],
        '5s': data['labels_5s'],
        '10s': data['labels_10s']
    }

    # Calculate time decay
    decay_metrics = calculate_time_decay_from_labels(predictions, labels_by_horizon)

    logger.info(f"IC by horizon:")
    for horizon, ic in zip(decay_metrics.horizons_sec, decay_metrics.ic_by_horizon):
        logger.info(f"  {horizon:.1f}s: IC={ic:.4f}")

    logger.info(f"Peak IC: {np.max(decay_metrics.ic_by_horizon):.4f} at {decay_metrics.optimal_horizon_sec:.1f}s")

    if decay_metrics.half_life_sec:
        logger.info(f"Signal half-life: {decay_metrics.half_life_sec:.1f}s")


def example_3_full_report():
    """Example 3: Generate complete exploitability report."""
    logger.info("="*80)
    logger.info("EXAMPLE 3: Complete Exploitability Report")
    logger.info("="*80)

    pred_file = Path("/home/jupiter/Lvl3Quant/data/cnn_s76_concat_oot_predictions.npz")

    if not pred_file.exists():
        logger.warning(f"Prediction file not found: {pred_file}")
        return

    # Generate report (includes plots)
    output_dir = Path("/home/jupiter/Lvl3Quant/analysis/example_reports")
    output_dir.mkdir(exist_ok=True, parents=True)

    report = generate_exploitability_report(
        pred_file=pred_file,
        timeframe="10s",
        output_dir=output_dir,
        include_plots=True
    )

    # Report is automatically printed and saved
    logger.info(f"Report saved to {output_dir}")


def example_4_compare_models():
    """Example 4: Compare multiple models."""
    logger.info("="*80)
    logger.info("EXAMPLE 4: Model Comparison")
    logger.info("="*80)

    # Find all concat prediction files
    data_dir = Path("/home/jupiter/Lvl3Quant/data")
    pred_files = list(data_dir.glob("*_concat_oot_predictions.npz"))

    if not pred_files:
        logger.warning(f"No prediction files found in {data_dir}")
        return

    logger.info(f"Found {len(pred_files)} prediction files")

    # Generate reports for all
    output_dir = Path("/home/jupiter/Lvl3Quant/analysis/comparison_reports")
    output_dir.mkdir(exist_ok=True, parents=True)

    reports = compare_models(
        pred_files=pred_files,
        timeframe="10s",
        output_dir=output_dir
    )

    # Print comparison table
    logger.info("\nModel Comparison:")
    logger.info(f"{'Model':<20} {'IC':>8} {'Sharpe@Opt':>12} {'Opt Thresh':>12} {'Dir Acc':>10}")
    logger.info("-" * 70)

    for model_name, report in reports.items():
        ic = report.ic_metrics.get('ic', 0)
        sharpe = report.threshold_sweep.get('optimal_sharpe', 0)
        thresh = report.threshold_sweep.get('optimal_threshold', 0)
        dir_acc = report.ic_metrics.get('directional_accuracy', 0)

        logger.info(f"{model_name:<20} {ic:>8.4f} {sharpe:>12.3f} {thresh:>12.3f} {dir_acc:>10.3f}")


def example_5_fold_level_analysis():
    """Example 5: Analyze individual folds (not just concat)."""
    logger.info("="*80)
    logger.info("EXAMPLE 5: Fold-Level Analysis")
    logger.info("="*80)

    data_dir = Path("/home/jupiter/Lvl3Quant/data")

    # Find fold-level prediction files
    fold_files = sorted(data_dir.glob("cnn_s76_fold_*_oot_predictions.npz"))

    if not fold_files:
        logger.warning("No fold-level prediction files found")
        return

    logger.info(f"Found {len(fold_files)} fold files")

    # Analyze each fold
    import numpy as np

    fold_ics = []
    fold_sharpes = []

    for fold_file in fold_files:
        pred_data = load_npz_predictions(fold_file, timeframe="10s")

        # Quick IC calculation
        from scipy.stats import pearsonr
        preds = pred_data.predictions
        labels = pred_data.labels

        valid_mask = ~(np.isnan(preds) | np.isnan(labels))
        if np.sum(valid_mask) < 10:
            continue

        ic, _ = pearsonr(preds[valid_mask], labels[valid_mask])

        # Quick Sharpe
        returns = preds[valid_mask] * labels[valid_mask]
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        fold_num = pred_data.metadata['fold']
        logger.info(f"Fold {fold_num}: IC={ic:.4f}, Sharpe={sharpe:.3f}, N={np.sum(valid_mask):,}")

        fold_ics.append(ic)
        fold_sharpes.append(sharpe)

    if fold_ics:
        logger.info(f"\nFold Statistics:")
        logger.info(f"  Mean IC: {np.mean(fold_ics):.4f} ± {np.std(fold_ics):.4f}")
        logger.info(f"  Mean Sharpe: {np.mean(fold_sharpes):.3f} ± {np.std(fold_sharpes):.3f}")
        logger.info(f"  IC stability (CV): {np.std(fold_ics) / np.mean(fold_ics):.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Example usage of prediction analysis framework")
    parser.add_argument("--example", type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                       help="Which example to run (0 = all)")
    args = parser.parse_args()

    examples = {
        1: example_1_basic_analysis,
        2: example_2_time_decay,
        3: example_3_full_report,
        4: example_4_compare_models,
        5: example_5_fold_level_analysis,
    }

    if args.example == 0:
        # Run all examples
        for i in sorted(examples.keys()):
            try:
                examples[i]()
                print("\n")
            except Exception as e:
                logger.error(f"Example {i} failed: {e}", exc_info=True)
    else:
        # Run specific example
        examples[args.example]()
