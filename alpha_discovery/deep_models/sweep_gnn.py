#!/usr/bin/env python3
"""
GNN Hyperparameter Sweep — Systematic search for optimal GNN configuration.

Tests combinations of hidden_dim, num_layers, epochs, and optionally GAT.
Each config gets a full 10-fold walk-forward evaluation.

Results saved to: results/gnn_sweep_YYYYMMDD_HHMMSS.json
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Sweep grid
CONFIGS = []

# Main sweep: hidden × layers (all with 10 epochs)
for hidden, layers in product([64, 128, 256], [2, 3, 4]):
    CONFIGS.append({
        'hidden': hidden,
        'layers': layers,
        'epochs': 10,
        'use_gat': False,
        'label': f'gcn_h{hidden}_l{layers}_e10',
    })

# Add GAT variant for best GCN hidden/layer combos
for hidden in [128, 256]:
    CONFIGS.append({
        'hidden': hidden,
        'layers': 2,
        'epochs': 10,
        'use_gat': True,
        'label': f'gat_h{hidden}_l2_e10',
    })

# Total: 9 GCN + 2 GAT = 11 configs


def run_config(cfg, folds=10, batch_size=512, subsample=3):
    """Run a single GNN config and return results."""
    cmd = [
        sys.executable, '-u', '-m', 'alpha_discovery.deep_models.train_gnn',
        '--folds', str(folds),
        '--epochs', str(cfg['epochs']),
        '--hidden', str(cfg['hidden']),
        '--layers', str(cfg['layers']),
        '--batch-size', str(batch_size),
        '--subsample-train', str(subsample),
    ]
    if cfg.get('use_gat'):
        cmd.extend(['--use-gat', '--gat-heads', '4'])

    print(f"\n{'='*70}")
    print(f"CONFIG: {cfg['label']}")
    print(f"  hidden={cfg['hidden']}, layers={cfg['layers']}, "
          f"epochs={cfg['epochs']}, gat={cfg.get('use_gat', False)}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),  # Lvl3Quant root
            capture_output=True,
            text=True,
            timeout=18000,  # 5hr max per config
        )
        elapsed = time.time() - start

        # Parse results from stdout
        lines = result.stdout.strip().split('\n')
        mean_ic = None
        icir = None
        pct_positive = None
        fold_ics = []

        for line in lines:
            if 'Mean IC:' in line:
                try:
                    mean_ic = float(line.split('Mean IC:')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'ICIR:' in line and 'ICIR' in line:
                try:
                    icir = float(line.split('ICIR:')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'Pct positive:' in line:
                try:
                    pct_positive = float(line.split('Pct positive:')[1].strip().rstrip('%'))
                except (ValueError, IndexError):
                    pass
            elif 'Per-fold ICs:' in line:
                try:
                    ics_str = line.split('Per-fold ICs:')[1].strip()
                    fold_ics = [float(x.strip("' []")) for x in ics_str.strip('[]').split(',')]
                except (ValueError, IndexError):
                    pass

        return {
            'config': cfg,
            'mean_ic': mean_ic,
            'icir': icir,
            'pct_positive': pct_positive,
            'fold_ics': fold_ics,
            'elapsed_s': round(elapsed, 1),
            'returncode': result.returncode,
            'error': result.stderr[-500:] if result.returncode != 0 else None,
        }

    except subprocess.TimeoutExpired:
        return {
            'config': cfg,
            'mean_ic': None,
            'icir': None,
            'error': 'TIMEOUT (5hr)',
            'elapsed_s': 18000,
        }
    except Exception as e:
        return {
            'config': cfg,
            'mean_ic': None,
            'icir': None,
            'error': str(e),
            'elapsed_s': time.time() - start,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='GNN Hyperparameter Sweep')
    parser.add_argument('--folds', type=int, default=10, help='Folds per config')
    parser.add_argument('--quick', action='store_true', help='Quick mode: 3 folds, 5 epochs')
    args = parser.parse_args()

    folds = 3 if args.quick else args.folds

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'gnn_sweep_{timestamp}.json'

    print(f"GNN Hyperparameter Sweep")
    print(f"  Configs: {len(CONFIGS)}")
    print(f"  Folds per config: {folds}")
    print(f"  Output: {out_file}")
    print()

    all_results = []
    sweep_start = time.time()

    for i, cfg in enumerate(CONFIGS):
        if args.quick:
            cfg = {**cfg, 'epochs': 5}

        print(f"\n[{i+1}/{len(CONFIGS)}] Starting {cfg['label']}...")
        result = run_config(cfg, folds=folds)
        all_results.append(result)

        ic_str = f"{result['mean_ic']:+.4f}" if result.get('mean_ic') is not None else 'FAILED'
        icir_str = f"{result.get('icir', 0):.2f}" if result.get('icir') is not None else 'N/A'
        print(f"\n  -> {cfg['label']}: IC={ic_str}, ICIR={icir_str}, time={result['elapsed_s']:.0f}s")

        # Save intermediate results
        sweep_data = {
            'timestamp': timestamp,
            'folds_per_config': folds,
            'total_configs': len(CONFIGS),
            'completed': len(all_results),
            'elapsed_s': round(time.time() - sweep_start, 1),
            'results': all_results,
        }
        with open(out_file, 'w') as f:
            json.dump(sweep_data, f, indent=2)

    # Final summary
    total_time = time.time() - sweep_start
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE: {len(all_results)} configs in {total_time/60:.1f} min")
    print(f"{'='*70}")

    # Sort by mean IC
    valid = [r for r in all_results if r.get('mean_ic') is not None]
    valid.sort(key=lambda x: x['mean_ic'], reverse=True)

    print(f"\n{'Label':30s} | {'Mean IC':>10} | {'ICIR':>6} | {'%Pos':>5} | {'Time':>6}")
    print('-' * 70)
    for r in valid:
        label = r['config']['label']
        ic = f"{r['mean_ic']:+.4f}"
        icir = f"{r.get('icir', 0):.2f}"
        pct = f"{r.get('pct_positive', 0):.0f}%"
        tm = f"{r['elapsed_s']/60:.1f}m"
        print(f"  {label:30s} | {ic:>10} | {icir:>6} | {pct:>5} | {tm:>6}")

    if valid:
        best = valid[0]
        print(f"\nBEST: {best['config']['label']} — IC={best['mean_ic']:+.4f}, ICIR={best.get('icir', 0):.2f}")
        print(f"CNN baseline: IC=+0.130 (for comparison)")

    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    main()
