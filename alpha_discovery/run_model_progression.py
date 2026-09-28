"""
Model Progression Runner — Compare LightGBM vs Temporal Models.

Runs the full model progression pipeline:
1. Load data from snapshot caches
2. Run LightGBM baseline (for comparison)
3. Run Temporal CNN
4. Run Sequence LSTM
5. Run MicroTransformer
6. Compare all models
7. Report to Discord

Usage:
    python alpha_discovery/run_model_progression.py [options]

Options:
    --horizon HORIZON    Target horizon: 3s, 5s, 10s, 30s, 1m, 5m (default: 3s)
    --target TARGET      Target type: return, volatility (default: return)
    --seq-len N          Sequence length for temporal models (default: 50)
    --model-size SIZE    Model size: small, medium, large (default: small)
    --skip-lgbm          Skip LightGBM baseline (use cached results)
    --models MODELS      Comma-separated models to test (default: cnn,lstm,transformer)
    --n-days N           Limit to N days of data (for quick testing)
    --no-discord         Disable Discord notifications
    --quick              Quick mode: fewer epochs, smaller models
"""

import gc
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

# Add project root
LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner
from alpha_discovery.temporal_trainer import TemporalWalkForwardEvaluator, format_comparison_report
from alpha_discovery.temporal_models import count_parameters, create_model

# Results directory
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("model_progression")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                RESULTS_DIR / f"model_progression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                encoding='utf-8',
            ),
        ],
    )


def send_discord(msg: str):
    """Send message to Discord if available."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "teleclaude-main"))
        from lib.discord import send_message
        send_message(msg)
    except Exception:
        pass


def run_lgbm_baseline(
    scanner: MBOAlphaScanner,
    targets: dict,
    horizon: str,
    target_type: str,
) -> dict:
    """Run LightGBM baseline for comparison."""
    logger.info(f"\n{'#'*60}")
    logger.info(f"PHASE 1: LightGBM Baseline — {horizon}_{target_type}")
    logger.info(f"{'#'*60}")

    if horizon not in targets:
        return {'error': f'Horizon {horizon} not available'}
    if target_type not in targets[horizon]:
        return {'error': f'Target {target_type} not available for {horizon}'}

    target = targets[horizon][target_type]

    t0 = time.time()
    result = scanner.walk_forward_evaluate(
        target=target,
        target_name=target_type,
        horizon_name=horizon,
        min_train_days=3,
    )
    result['elapsed_sec'] = time.time() - t0

    if 'error' not in result:
        logger.info(f"LightGBM: IC={result['ic']:.4f} ICIR={result['icir']:.2f} "
                     f"t={result['tstat']:.2f} [{result['elapsed_sec']:.0f}s]")
    else:
        logger.info(f"LightGBM: ERROR: {result['error']}")

    return result


def run_temporal_models(
    scanner: MBOAlphaScanner,
    targets: dict,
    horizon: str,
    target_type: str,
    models_to_test: list,
    model_size: str = 'small',
    seq_len: int = 50,
    stride: int = 10,
    n_epochs: int = 15,
    lr: float = 1e-3,
    quick: bool = False,
) -> list:
    """Run temporal model progression."""
    logger.info(f"\n{'#'*60}")
    logger.info(f"PHASE 2: Temporal Models — {horizon}_{target_type}")
    logger.info(f"{'#'*60}")

    if horizon not in targets:
        return [{'error': f'Horizon {horizon} not available', 'model_type': m}
                for m in models_to_test]
    if target_type not in targets[horizon]:
        return [{'error': f'Target {target_type} not available for {horizon}',
                 'model_type': m} for m in models_to_test]

    target = targets[horizon][target_type]

    # Quick mode overrides
    if quick:
        n_epochs = 8
        model_size = 'small'
        stride = max(stride, 20)
        logger.info("Quick mode: reduced epochs, small models, wider stride")

    # Create evaluator
    evaluator = TemporalWalkForwardEvaluator(
        features=scanner.features,
        mid_prices=scanner.mid_prices,
        day_boundaries=scanner.day_boundaries,
        seq_len=seq_len,
        stride=stride,
    )

    # Log model sizes
    for mt in models_to_test:
        model = create_model(mt, scanner.features.shape[1], 1, model_size)
        n_params = count_parameters(model)
        logger.info(f"  {mt} ({model_size}): {n_params:,} parameters")
        del model

    results = []
    for mt in models_to_test:
        logger.info(f"\n--- Running {mt} ({model_size}) ---")

        result = evaluator.evaluate_model(
            model_type=mt,
            target=target,
            target_name=target_type,
            horizon_name=horizon,
            model_size=model_size,
            n_epochs=n_epochs,
            lr=lr,
            min_train_days=5,
        )
        results.append(result)

        # Discord progress update
        if 'error' not in result:
            status = "PASS" if result['passed'] else "FAIL"
            send_discord(
                f"`{mt} ({model_size})` [{status}]\n"
                f"  IC={result['ic']:.4f} t={result['tstat']:.2f} "
                f"HR={result['hit_rate']:.1%} ({result['n_params']:,} params, "
                f"{result['total_train_time']:.0f}s)"
            )

        gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description='Model Progression Runner')
    parser.add_argument('--horizon', default='3s',
                        help='Target horizon (default: 3s)')
    parser.add_argument('--target', default='return',
                        help='Target type (default: return)')
    parser.add_argument('--seq-len', type=int, default=50,
                        help='Sequence length for temporal models (default: 50)')
    parser.add_argument('--stride', type=int, default=10,
                        help='Window stride (default: 10)')
    parser.add_argument('--model-size', default='small',
                        choices=['small', 'medium', 'large'],
                        help='Model size (default: small)')
    parser.add_argument('--skip-lgbm', action='store_true',
                        help='Skip LightGBM baseline')
    parser.add_argument('--models', default='cnn,lstm,transformer',
                        help='Comma-separated models to test')
    parser.add_argument('--n-days', type=int, default=None,
                        help='Limit to N days')
    parser.add_argument('--no-discord', action='store_true',
                        help='Disable Discord')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (fewer epochs)')
    parser.add_argument('--n-epochs', type=int, default=15,
                        help='Training epochs per fold (default: 15)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--multi-horizon', action='store_true',
                        help='Test across multiple horizons')
    args = parser.parse_args()

    setup_logging()
    logger.info("=" * 70)
    logger.info("MODEL PROGRESSION — LightGBM vs Temporal Models")
    logger.info("=" * 70)
    logger.info(f"Horizon: {args.horizon}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Seq len: {args.seq_len}")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Models: {args.models}")

    models_to_test = [m.strip() for m in args.models.split(',')]

    # Determine horizons to test
    if args.multi_horizon:
        horizons = ['3s', '5s', '10s', '30s', '1m']
    else:
        horizons = [args.horizon]

    # ================================================================
    # Load data
    # ================================================================
    logger.info("\nLoading data from snapshot caches...")
    t0 = time.time()

    scanner = MBOAlphaScanner()

    # Add 3s horizon if not present
    if '3s' not in scanner.horizons:
        scanner.horizons['3s'] = 3
    if '10s' not in scanner.horizons:
        scanner.horizons['10s'] = 10

    load_info = scanner.load_from_cache(n_days=args.n_days)

    logger.info(f"Data loaded in {time.time() - t0:.1f}s")
    logger.info(f"  {load_info['n_snapshots']:,} snapshots, {load_info['n_days']} days, "
                 f"{load_info['n_features']} features")

    if not args.no_discord:
        send_discord(
            f"**Model Progression Started**\n"
            f"Data: {load_info['n_days']} days, {load_info['n_snapshots']:,} bars\n"
            f"Horizons: {', '.join(horizons)}\n"
            f"Models: {', '.join(models_to_test)} ({args.model_size})\n"
            f"Seq len: {args.seq_len}, Stride: {args.stride}"
        )

    # ================================================================
    # Compute targets
    # ================================================================
    logger.info("\nComputing targets...")
    targets = scanner.compute_targets()

    all_results = {}

    for horizon in horizons:
        logger.info(f"\n{'*'*70}")
        logger.info(f"HORIZON: {horizon}")
        logger.info(f"{'*'*70}")

        # ================================================================
        # LightGBM baseline
        # ================================================================
        if not args.skip_lgbm:
            lgbm_result = run_lgbm_baseline(
                scanner, targets, horizon, args.target,
            )
        else:
            # Load from previous results
            lgbm_result = {'error': 'skipped', 'ic': 0, 'icir': 0, 'tstat': 0}
            logger.info("Skipping LightGBM baseline (--skip-lgbm)")

        # ================================================================
        # Temporal models
        # ================================================================
        temporal_results = run_temporal_models(
            scanner=scanner,
            targets=targets,
            horizon=horizon,
            target_type=args.target,
            models_to_test=models_to_test,
            model_size=args.model_size,
            seq_len=args.seq_len,
            stride=args.stride,
            n_epochs=args.n_epochs,
            lr=args.lr,
            quick=args.quick,
        )

        # ================================================================
        # Comparison report
        # ================================================================
        report = format_comparison_report(lgbm_result, temporal_results)
        logger.info(report)

        all_results[horizon] = {
            'lgbm': lgbm_result,
            'temporal': temporal_results,
            'report': report,
        }

        # Discord report
        if not args.no_discord:
            # Send condensed version
            lines = [f"**Model Progression — {horizon}_{args.target}**\n"]

            if 'error' not in lgbm_result:
                lines.append(
                    f"`LightGBM`: IC={lgbm_result['ic']:.4f} "
                    f"t={lgbm_result['tstat']:.2f}"
                )

            for r in temporal_results:
                if 'error' in r:
                    lines.append(f"`{r.get('model_type', '?')}`: ERROR")
                else:
                    icon = "**PASS**" if r['passed'] else "---"
                    lines.append(
                        f"`{r['model_type']} ({r['model_size']})` [{icon}]: "
                        f"IC={r['ic']:.4f} t={r['tstat']:.2f} "
                        f"({r['n_params']:,} params)"
                    )

            # Verdict
            lgbm_ic = lgbm_result.get('ic', 0) if 'error' not in lgbm_result else 0
            best = max(
                (r for r in temporal_results if 'error' not in r),
                key=lambda x: abs(x['ic']),
                default=None,
            )
            if best and abs(best['ic']) > abs(lgbm_ic) * 1.1:
                lines.append(
                    f"\nVerdict: {best['model_type']} WINS "
                    f"(+{(abs(best['ic'])/max(abs(lgbm_ic),0.001)-1)*100:.1f}% over LightGBM)"
                )
            elif best and abs(best['ic']) > abs(lgbm_ic):
                lines.append(f"\nVerdict: Marginal improvement, need more data")
            else:
                lines.append(f"\nVerdict: LightGBM baseline holds")

            send_discord("\n".join(lines))

    # ================================================================
    # Save all results
    # ================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = RESULTS_DIR / f"model_progression_{timestamp}.json"

    # Make results JSON-serializable
    serializable = {}
    for hz, hz_results in all_results.items():
        serializable[hz] = {
            'lgbm': _make_serializable(hz_results['lgbm']),
            'temporal': [_make_serializable(r) for r in hz_results['temporal']],
        }

    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'horizons': horizons,
                'target': args.target,
                'seq_len': args.seq_len,
                'stride': args.stride,
                'model_size': args.model_size,
                'models': models_to_test,
                'n_days': load_info['n_days'],
                'n_snapshots': load_info['n_snapshots'],
            },
            'results': serializable,
        }, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")

    # ================================================================
    # Final multi-horizon summary (if applicable)
    # ================================================================
    if len(horizons) > 1:
        logger.info(f"\n{'='*70}")
        logger.info("MULTI-HORIZON SUMMARY")
        logger.info(f"{'='*70}")

        lines = [
            f"\n{'Horizon':<10s} {'LightGBM':>10s} ",
        ]
        for mt in models_to_test:
            lines[0] += f" {mt:>12s}"

        for hz in horizons:
            hz_res = all_results.get(hz, {})
            lgbm_ic = hz_res.get('lgbm', {}).get('ic', 0) if 'error' not in hz_res.get('lgbm', {}) else 0
            line = f"{hz:<10s} {lgbm_ic:>10.4f} "
            for mt in models_to_test:
                mt_result = next(
                    (r for r in hz_res.get('temporal', [])
                     if r.get('model_type') == mt and 'error' not in r),
                    None,
                )
                if mt_result:
                    line += f" {mt_result['ic']:>12.4f}"
                else:
                    line += f" {'N/A':>12s}"
            lines.append(line)

        summary = "\n".join(lines)
        logger.info(summary)

        if not args.no_discord:
            send_discord(f"**Multi-Horizon Summary**\n```\n{summary}\n```")

    logger.info("\nModel progression complete!")


def _make_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


if __name__ == '__main__':
    main()
