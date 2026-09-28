#!/usr/bin/env python3
"""
Batch Analysis Tool — Generate exploitability reports for all models

Usage:
    python batch_analyze.py --data-dir /path/to/data --output-dir /path/to/reports
    python batch_analyze.py --model cnn_s76  # Analyze specific model only
    python batch_analyze.py --fold-level     # Analyze individual folds (not concat)
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import List, Optional
import json

# Add analysis dir to path
sys.path.insert(0, str(Path(__file__).parent))

from exploitability_report import generate_exploitability_report, compare_models
from prediction_loader import load_npz_predictions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def find_prediction_files(
    data_dir: Path,
    model_pattern: Optional[str] = None,
    concat_only: bool = True
) -> List[Path]:
    """
    Find prediction files in data directory.

    Args:
        data_dir: Directory containing .npz prediction files
        model_pattern: Optional model name filter (e.g., "cnn", "transformer")
        concat_only: If True, only find concat files (default)

    Returns:
        List of prediction file paths
    """
    if concat_only:
        pattern = "*_concat_oot_predictions.npz"
    else:
        pattern = "*_predictions.npz"

    pred_files = sorted(data_dir.glob(pattern))

    if model_pattern:
        pred_files = [f for f in pred_files if model_pattern.lower() in f.stem.lower()]

    logger.info(f"Found {len(pred_files)} prediction files matching pattern")

    return pred_files


def batch_analyze(
    data_dir: Path,
    output_dir: Path,
    timeframe: str = "10s",
    model_pattern: Optional[str] = None,
    concat_only: bool = True,
    include_plots: bool = True
):
    """
    Generate exploitability reports for all models in data directory.

    Args:
        data_dir: Directory containing prediction .npz files
        output_dir: Output directory for reports
        timeframe: Timeframe to analyze
        model_pattern: Optional model name filter
        concat_only: If True, only analyze concat files
        include_plots: Whether to generate plots
    """
    # Find prediction files
    pred_files = find_prediction_files(data_dir, model_pattern, concat_only)

    if not pred_files:
        logger.warning(f"No prediction files found in {data_dir}")
        return

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate reports for each model
    reports = {}
    summary_data = []

    for i, pred_file in enumerate(pred_files, 1):
        logger.info(f"[{i}/{len(pred_files)}] Processing {pred_file.name}...")

        try:
            report = generate_exploitability_report(
                pred_file=pred_file,
                timeframe=timeframe,
                output_dir=output_dir,
                include_plots=include_plots
            )

            model_name = report.metadata['model']
            reports[model_name] = report

            # Collect summary data (convert numpy types to Python types)
            summary_data.append({
                'model': str(model_name),
                'fold': str(report.metadata.get('fold', 'unknown')),
                'ic': float(report.ic_metrics.get('ic', 0)),
                'rank_ic': float(report.ic_metrics.get('rank_ic', 0)),
                'directional_accuracy': float(report.ic_metrics.get('directional_accuracy', 0)),
                'optimal_threshold': float(report.threshold_sweep.get('optimal_threshold', 0)),
                'optimal_sharpe': float(report.threshold_sweep.get('optimal_sharpe', 0)),
                'num_predictions': int(report.ic_metrics.get('num_predictions', 0)),
                'top_10_ic': float(report.percentile_ic.get('top_10', {}).get('ic', 0) if report.percentile_ic else 0),
                'top_10_sharpe': float(report.percentile_ic.get('top_10', {}).get('sharpe', 0) if report.percentile_ic else 0),
            })

        except Exception as e:
            logger.error(f"Failed to process {pred_file.name}: {e}", exc_info=True)
            continue

    # Save summary table
    if summary_data:
        summary_path = output_dir / "summary_table.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        logger.info(f"Saved summary table to {summary_path}")

        # Print comparison table
        print("\n" + "="*120)
        print("BATCH ANALYSIS SUMMARY")
        print("="*120)
        print(f"{'Model':<30} {'IC':>8} {'Rank IC':>10} {'Dir Acc':>10} {'Opt Thresh':>12} {'Opt Sharpe':>12} {'Top 10% IC':>12}")
        print("-"*120)

        # Sort by IC descending
        summary_data.sort(key=lambda x: x['ic'], reverse=True)

        for row in summary_data:
            print(f"{row['model']:<30} {row['ic']:>8.4f} {row['rank_ic']:>10.4f} "
                  f"{row['directional_accuracy']:>10.3f} {row['optimal_threshold']:>12.3f} "
                  f"{row['optimal_sharpe']:>12.3f} {row['top_10_ic']:>12.4f}")

        print("="*120 + "\n")

        # Identify best models
        best_ic = max(summary_data, key=lambda x: x['ic'])
        best_sharpe = max(summary_data, key=lambda x: x['optimal_sharpe'])

        print("BEST MODELS:")
        print(f"  Highest IC: {best_ic['model']} (IC={best_ic['ic']:.4f})")
        print(f"  Highest Sharpe: {best_sharpe['model']} (Sharpe={best_sharpe['optimal_sharpe']:.3f})")
        print()

    logger.info(f"Batch analysis complete. Reports saved to {output_dir}")


def analyze_fold_progression(
    data_dir: Path,
    model_name: str,
    timeframe: str = "10s"
):
    """
    Analyze performance progression across walk-forward folds.

    Args:
        data_dir: Directory containing prediction files
        model_name: Model name pattern (e.g., "cnn_s76")
        timeframe: Timeframe to analyze
    """
    # Find fold-level files for this model
    fold_files = sorted(data_dir.glob(f"{model_name}_fold_*_oot_predictions.npz"))

    if not fold_files:
        logger.warning(f"No fold files found for model {model_name}")
        return

    logger.info(f"Found {len(fold_files)} folds for {model_name}")

    # Analyze each fold
    import numpy as np
    from scipy.stats import pearsonr

    fold_results = []

    for fold_file in fold_files:
        pred_data = load_npz_predictions(fold_file, timeframe=timeframe)

        # Quick metrics
        preds = pred_data.predictions
        labels = pred_data.labels

        valid_mask = ~(np.isnan(preds) | np.isnan(labels))
        if np.sum(valid_mask) < 10:
            continue

        preds = preds[valid_mask]
        labels = labels[valid_mask]

        ic, pval = pearsonr(preds, labels)
        returns = preds * labels
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        fold_num = pred_data.metadata['fold']

        fold_results.append({
            'fold': fold_num,
            'ic': ic,
            'sharpe': sharpe,
            'num_predictions': len(preds)
        })

        logger.info(f"Fold {fold_num}: IC={ic:.4f}, Sharpe={sharpe:.3f}, N={len(preds):,}")

    if not fold_results:
        logger.warning("No valid fold results")
        return

    # Print summary
    ics = [r['ic'] for r in fold_results]
    sharpes = [r['sharpe'] for r in fold_results]

    print("\n" + "="*80)
    print(f"FOLD PROGRESSION ANALYSIS: {model_name}")
    print("="*80)
    print(f"\nIC Statistics:")
    print(f"  Mean: {np.mean(ics):.4f} ± {np.std(ics):.4f}")
    print(f"  Median: {np.median(ics):.4f}")
    print(f"  Range: [{np.min(ics):.4f}, {np.max(ics):.4f}]")
    print(f"  Stability (CV): {np.std(ics) / np.mean(ics):.3f}")

    print(f"\nSharpe Statistics:")
    print(f"  Mean: {np.mean(sharpes):.3f} ± {np.std(sharpes):.3f}")
    print(f"  Median: {np.median(sharpes):.3f}")
    print(f"  Range: [{np.min(sharpes):.3f}, {np.max(sharpes):.3f}]")

    # Check for degradation trend
    fold_nums = [r['fold'] for r in fold_results]
    if len(fold_nums) > 3:
        # Simple linear regression on fold number vs IC
        from scipy.stats import linregress
        if all(isinstance(fn, int) for fn in fold_nums):
            slope, intercept, r_value, p_value, std_err = linregress(fold_nums, ics)

            if p_value < 0.05:
                if slope > 0:
                    print(f"\nTrend: IMPROVING over time (slope={slope:.5f}, p={p_value:.3f})")
                else:
                    print(f"\nTrend: DEGRADING over time (slope={slope:.5f}, p={p_value:.3f})")
            else:
                print(f"\nTrend: STABLE over time (no significant trend)")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Batch analysis of prediction files")

    parser.add_argument("--data-dir", type=str,
                       default="/home/jupiter/Lvl3Quant/data",
                       help="Directory containing prediction .npz files")

    parser.add_argument("--output-dir", type=str,
                       default="/home/jupiter/Lvl3Quant/analysis/batch_reports",
                       help="Output directory for reports")

    parser.add_argument("--timeframe", type=str, default="10s",
                       help="Timeframe to analyze (1s, 5s, 10s, 30s)")

    parser.add_argument("--model", type=str, default=None,
                       help="Filter for specific model name (e.g., 'cnn', 'transformer')")

    parser.add_argument("--fold-level", action="store_true",
                       help="Analyze individual folds (not just concat)")

    parser.add_argument("--fold-progression", type=str, default=None,
                       help="Analyze fold progression for specific model (e.g., 'cnn_s76')")

    parser.add_argument("--no-plots", action="store_true",
                       help="Skip plot generation (faster)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # Run fold progression analysis if requested
    if args.fold_progression:
        analyze_fold_progression(data_dir, args.fold_progression, args.timeframe)
        return

    # Run batch analysis
    batch_analyze(
        data_dir=data_dir,
        output_dir=output_dir,
        timeframe=args.timeframe,
        model_pattern=args.model,
        concat_only=not args.fold_level,
        include_plots=not args.no_plots
    )


if __name__ == "__main__":
    main()
