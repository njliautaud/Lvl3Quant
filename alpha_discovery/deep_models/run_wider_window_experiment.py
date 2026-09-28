"""
Wider Window Experiment — BookSpatialCNN with larger temporal context.

Tests window_size=50 (5 seconds) and optionally window_size=100 (10 seconds).

Architecture: EXACT same WiderBookSpatialCNN as proven Mar 18 run
  - spatial_channels = (64, 128, 256, 512)
  - temporal_channels = 512
  - ~12.6M parameters
  - dropout = 0.15

Window mode: EXPANDING (no --max-train-days cap)
  Each fold trains on ALL available prior data.

Hypothesis:
  Current 20-bar window (2s) captures most short-term dynamics.
  If 50-bar (5s) improves IC → longer temporal context is exploitable.
  If IC stays same → 2s is sufficient.

Memory note:
  window_size=50 means 2.5x more data per sample (vs w=20).
  batch_size reduced to 256 (vs 512 default) to fit in 3090 VRAM.
  float16 tensor storage is active (OOM fix from 2026-03-26).

Usage:
  python run_wider_window_experiment.py --window-size 50
  python run_wider_window_experiment.py --window-size 100

  Or run both sequentially:
  python run_wider_window_experiment.py --all

Outputs:
  results/wider_window_50/   (window_size=50)
  results/wider_window_100/  (window_size=100)
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Thread / device config ───────────────────────────────────────────────────
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import book_spatial_cnn
import train_walkforward as twf


# ── Wider architecture patch (same as proven Mar 18 / run_wider_cnn_expanding_wf.py) ──
OriginalCNN = book_spatial_cnn.BookSpatialCNN

class WiderBookSpatialCNN(OriginalCNN):
    """2x wider BookSpatialCNN — spatial=(64,128,256,512), temporal=512, ~12.6M params."""
    def __init__(self, **kwargs):
        kwargs['spatial_channels'] = (64, 128, 256, 512)
        kwargs['temporal_channels'] = 512
        kwargs['dropout'] = 0.15
        super().__init__(**kwargs)

# Replace globally so train_walkforward sees the wider architecture
book_spatial_cnn.BookSpatialCNN = WiderBookSpatialCNN
twf.BookSpatialCNN = WiderBookSpatialCNN


# ── Per-fold MLflow metrics patch ────────────────────────────────────────────
_orig_json_dump = json.dump
_mlflow_run = None
_mlflow_last_fold_logged = [0]

def _tagged_json_dump(obj, fp, **kwargs):
    if isinstance(obj, dict) and 'completed_folds' in obj and 'fold_ics' in obj:
        obj['wider_window_experiment'] = True
        obj['window_mode'] = 'expanding'
        if MLFLOW_AVAILABLE and _mlflow_run is not None:
            try:
                fold_num = obj['completed_folds']
                fold_ics = obj['fold_ics']
                mean_ic  = obj.get('mean_ic', 0.0)
                if fold_ics and fold_num > _mlflow_last_fold_logged[0]:
                    mlflow.log_metrics(
                        {'fold_ic': float(fold_ics[-1]), 'mean_ic': float(mean_ic)},
                        step=fold_num,
                    )
                    _mlflow_last_fold_logged[0] = fold_num
            except Exception:
                pass
    return _orig_json_dump(obj, fp, **kwargs)

json.dump = _tagged_json_dump


def run_window_experiment(window_size: int) -> dict:
    """
    Run one walk-forward experiment with the given window_size.
    Returns the results dict from train_walkforward.main().
    """
    global _mlflow_run, _mlflow_last_fold_logged

    _mlflow_last_fold_logged[0] = 0

    # ── Output directory (isolated per window size) ───────────────────────────
    results_dir = SCRIPT_DIR / 'results' / f'wider_window_{window_size}'
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Batch size: scale down to avoid VRAM OOM with larger windows ──────────
    # window=20 → batch=256 works fine on 3090.
    # window=50 → 2.5x more data per sample. Keep 256 (float16 storage helps).
    # window=100 → 5x data. Drop to 128 to be safe.
    batch_size = 256 if window_size <= 50 else 128

    print()
    print("=" * 65)
    print(f"WIDER WINDOW EXPERIMENT — window_size={window_size}")
    print(f"  Architecture:  WiderBookSpatialCNN (64,128,256,512) + t=512")
    print(f"  Params:        ~12.6M")
    print(f"  Window:        {window_size} bars = {window_size / 10:.1f} seconds")
    print(f"  Batch size:    {batch_size}")
    print(f"  Window mode:   EXPANDING (no cap)")
    print(f"  Output dir:    {results_dir}")
    print("=" * 65)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://localhost:5000')
            mlflow.set_experiment('CNN_Training')
            _mlflow_run = mlflow.start_run(
                run_name=f'wider_window_{window_size}_expanding_wf'
            )
            mlflow.log_params({
                'model':             'WiderBookSpatialCNN',
                'spatial_channels':  '64,128,256,512',
                'temporal_channels': 512,
                'dropout':           0.15,
                'window_size':       window_size,
                'window_seconds':    window_size / 10,
                'epochs':            2,
                'batch_size':        batch_size,
                'subsample_train':   10,
                'window_mode':       'expanding',
                'warm_start':        True,
                'n_params_approx':   '12.6M',
            })
            print(f"MLflow run started: {_mlflow_run.info.run_id}")
        except Exception as e:
            print(f"MLflow init failed (non-fatal): {e}")
            _mlflow_run = None

    # ── Checkpoint resume ─────────────────────────────────────────────────────
    resume_fold = 0
    # Look for any valid checkpoint in this results dir
    ckpts = sorted(results_dir.glob('checkpoint_wider_window_*.json'))
    ckpts += sorted(results_dir.glob('checkpoint_book_*.json'))
    for ckpt in ckpts:
        try:
            with open(ckpt) as f:
                ckpt_data = json.load(f)
            n_folds = ckpt_data.get('completed_folds', 0)
            if n_folds > resume_fold:
                resume_fold = n_folds
                print(f"Found checkpoint: {ckpt.name} with {n_folds} folds completed")
        except Exception:
            pass

    print(f"Resuming from fold {resume_fold} (EXPANDING window)")
    print()

    # ── Build sys.argv for train_walkforward.main() ───────────────────────────
    sys.argv = [
        'run_wider_window_experiment.py',
        '--model',           'book',
        '--epochs',          '2',
        '--batch-size',      str(batch_size),
        '--subsample-train', '10',
        '--device',          'cuda',
        '--warm-start',
        '--window-size',     str(window_size),
        '--output-dir',      str(results_dir),
    ]
    if resume_fold > 0:
        sys.argv += ['--start-fold', str(resume_fold)]
        print(f"Resuming from fold {resume_fold} via --start-fold")

    # ── Run ───────────────────────────────────────────────────────────────────
    wf_results = None
    try:
        wf_results = twf.main()
    finally:
        # Log aggregate metrics to MLflow
        if MLFLOW_AVAILABLE and _mlflow_run is not None:
            try:
                if wf_results and isinstance(wf_results, dict) and 'agg_ic' in wf_results:
                    mlflow.log_metrics({
                        'agg_ic':    float(wf_results.get('agg_ic', 0.0)),
                        'agg_ic_std': float(wf_results.get('agg_ic_std', 0.0)),
                        'agg_icir':  float(wf_results.get('agg_icir', 0.0)),
                        'concat_ic': float(wf_results.get('concat_ic', 0.0)),
                        'n_folds':   float(wf_results.get('n_folds', 0)),
                    })
                mlflow.end_run()
                print("MLflow run ended.")
            except Exception as e:
                print(f"MLflow end_run failed (non-fatal): {e}")
                try:
                    mlflow.end_run()
                except Exception:
                    pass

        # Rename book-named files to wider_window-named for consistent tagging
        tag = f'wider_window_{window_size}'
        for ckpt in results_dir.glob('checkpoint_book_*.json'):
            new_name = ckpt.name.replace('checkpoint_book_', f'checkpoint_{tag}_')
            ckpt.rename(ckpt.parent / new_name)
            print(f"Renamed checkpoint: {new_name}")
        for pred in results_dir.glob('oos_predictions_book_*.npz'):
            new_name = pred.name.replace('oos_predictions_book_', f'oos_predictions_{tag}_')
            if not (pred.parent / new_name).exists():
                pred.rename(pred.parent / new_name)
                print(f"Renamed: {new_name}")
        for pred in results_dir.glob('ckpt_preds_book_*.npz'):
            new_name = pred.name.replace('ckpt_preds_book_', f'ckpt_preds_{tag}_')
            if not (pred.parent / new_name).exists():
                pred.rename(pred.parent / new_name)
                print(f"Renamed: {new_name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if wf_results and isinstance(wf_results, dict):
        print()
        print("=" * 65)
        print(f"EXPERIMENT COMPLETE — window_size={window_size} ({window_size/10:.1f}s)")
        print(f"  Folds completed:  {wf_results.get('n_folds', '?')}")
        print(f"  Mean fold IC:     {wf_results.get('agg_ic', 0):.4f} ± {wf_results.get('agg_ic_std', 0):.4f}")
        print(f"  IC-IR:            {wf_results.get('agg_icir', 0):.4f}")
        print(f"  Concat IC:        {wf_results.get('concat_ic', 0):.4f}")
        print(f"  Results dir:      {results_dir}")
        print("=" * 65)

    return wf_results or {}


def main():
    parser = argparse.ArgumentParser(
        description='BookSpatialCNN wider window experiment (window_size=50 or 100)'
    )
    parser.add_argument(
        '--window-size', type=int, default=50,
        help='Window size in bars (default: 50 = 5 seconds). Use 100 for 10s.'
    )
    parser.add_argument(
        '--all', action='store_true', default=False,
        help='Run both window_size=50 AND window_size=100 sequentially.'
    )
    args = parser.parse_args()

    if args.all:
        print("Running both window_size=50 and window_size=100 sequentially...")
        results_50  = run_window_experiment(50)
        results_100 = run_window_experiment(100)

        # Comparison summary
        print()
        print("=" * 65)
        print("WINDOW SIZE COMPARISON")
        print(f"  Baseline (w=20, 2s):  IC ~0.261 (from wider_cnn results)")
        print(f"  w=50  (5s):  mean IC={results_50.get('agg_ic', 0):.4f}  concat={results_50.get('concat_ic', 0):.4f}")
        print(f"  w=100 (10s): mean IC={results_100.get('agg_ic', 0):.4f}  concat={results_100.get('concat_ic', 0):.4f}")
        print("=" * 65)
    else:
        run_window_experiment(args.window_size)


if __name__ == '__main__':
    main()
