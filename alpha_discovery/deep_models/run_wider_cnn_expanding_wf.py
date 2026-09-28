"""
Wider CNN WF with EXPANDING WINDOW on Neptune GPU.

Architecture: BookSpatialCNN wider (64, 128, 256, 512) + temporal=512
  ~12.6M params (matches Mar 18 run)

Window mode: EXPANDING (no --max-train-days cap)
  Each fold trains on ALL available prior data, not just last 15 days.
  This grows the training set from 5 days at fold 1 to 130+ days at later folds.

Training is automatically resumed from the latest checkpoint in results/wider_cnn/.

Safety:
  OMP_NUM_THREADS=16, MKL_NUM_THREADS=16, CUDA_VISIBLE_DEVICES=0
  These are set by launch_wider_cnn_expanding.py before calling this script.
"""
import sys
import os
import json
import time
from pathlib import Path

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Thread limits (also set in launcher env, belt+suspenders)
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

# Add project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import book_spatial_cnn
import train_walkforward as twf

# ── Monkey-patch to wider architecture ──────────────────────────────────────
OriginalCNN = book_spatial_cnn.BookSpatialCNN

class WiderBookSpatialCNN(OriginalCNN):
    """2x wider BookSpatialCNN — same as the Mar 18 run (~12.6M params)."""
    def __init__(self, **kwargs):
        kwargs['spatial_channels'] = (64, 128, 256, 512)
        kwargs['temporal_channels'] = 512
        kwargs['dropout'] = 0.15
        super().__init__(**kwargs)

# Replace globally
book_spatial_cnn.BookSpatialCNN = WiderBookSpatialCNN
twf.BookSpatialCNN = WiderBookSpatialCNN

# ── MLflow run setup ─────────────────────────────────────────────────────────
_mlflow_run = None
if MLFLOW_AVAILABLE:
    try:
        mlflow.set_tracking_uri('http://localhost:5000')
        mlflow.set_experiment('CNN_Training')
        _mlflow_run = mlflow.start_run(run_name='wider_cnn_expanding_wf')
        mlflow.log_params({
            'model':             'WiderBookSpatialCNN',
            'spatial_channels':  '64,128,256,512',
            'temporal_channels': 512,
            'dropout':           0.15,
            'epochs':            2,
            'batch_size':        256,
            'subsample_train':   10,
            'window_mode':       'expanding',
            'warm_start':        True,
            'n_params_approx':   '12.6M',
        })
        print(f"MLflow run started: {_mlflow_run.info.run_id}")
    except Exception as _e:
        print(f"MLflow init failed (non-fatal): {_e}")
        _mlflow_run = None

# ── Tag checkpoints with wider_cnn marker + emit per-fold MLflow metrics ─────
_orig_json_dump = json.dump
_mlflow_last_fold_logged = [0]  # mutable container to track last fold logged

def _tagged_json_dump(obj, fp, **kwargs):
    if isinstance(obj, dict) and 'completed_folds' in obj and 'fold_ics' in obj:
        obj['wider_cnn'] = True
        obj['window_mode'] = 'expanding'
        # Log per-fold MLflow metrics on each checkpoint save
        if MLFLOW_AVAILABLE and _mlflow_run is not None:
            try:
                fold_num  = obj['completed_folds']            # 1-indexed count
                fold_ics  = obj['fold_ics']
                mean_ic   = obj.get('mean_ic', 0.0)
                if fold_ics and fold_num > _mlflow_last_fold_logged[0]:
                    fold_ic = float(fold_ics[-1])
                    mlflow.log_metrics(
                        {
                            'fold_ic':  fold_ic,
                            'mean_ic':  float(mean_ic),
                        },
                        step=fold_num,
                    )
                    _mlflow_last_fold_logged[0] = fold_num
            except Exception:
                pass
    return _orig_json_dump(obj, fp, **kwargs)
json.dump = _tagged_json_dump

# ── Results directory (isolated from standard CNN) ──────────────────────────
RESULTS_DIR = SCRIPT_DIR / 'results' / 'wider_cnn'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("WIDER CNN — EXPANDING WINDOW EXPERIMENT")
print("  Architecture: spatial=(64,128,256,512) temporal=512")
print("  Params: ~12.6M")
print("  Window: EXPANDING (no cap) — float16 dataset storage fixes OOM")
print("  Memory fix: book_tensors stored float16 (50% RAM vs float32)")
print("  Settings: epochs=2, batch=256, subsample=10, warm_start=True")
print("=" * 60)

# ── Find resume fold ────────────────────────────────────────────────────────
_resume_fold = 0
_wider_ckpts = sorted(RESULTS_DIR.glob('checkpoint_wider_cnn_*.json'))
_wider_ckpts += sorted(RESULTS_DIR.glob('checkpoint_book_*.json'))
for _wc in _wider_ckpts:
    try:
        with open(_wc) as _wf:
            _wd = json.load(_wf)
        _nf = _wd.get('completed_folds', 0)
        if _nf > _resume_fold:
            _resume_fold = _nf
            print(f"Found checkpoint: {_wc.name} with {_nf} folds")
    except Exception:
        pass

print(f"Resuming from fold {_resume_fold} (EXPANDING window, no cap)")
print()

# ── Build sys.argv for train_walkforward.main() ─────────────────────────────
# OOM FIX (2026-03-26): book_tensors now stored as float16 in BarDataset.
# - 80-day expanding window was 6.0 GB float32, OOMing at fold 76.
# - float16 storage = 3.0 GB — fits in 13.3 GB available RAM.
# - Cast back to float32 in __getitem__ (no persistent RAM cost).
# No --max-train-days cap needed. True expanding window restored.
sys.argv = [
    'run_wider_cnn_expanding_wf.py',
    '--model', 'book',
    '--epochs', '2',
    '--batch-size', '256',
    '--subsample-train', '10',
    '--device', 'cuda',
    '--warm-start',
    '--output-dir', str(RESULTS_DIR),
]
if _resume_fold > 0:
    sys.argv += ['--start-fold', str(_resume_fold)]
    print(f"Resuming from fold {_resume_fold} via --start-fold")

_wf_results = None
try:
    _wf_results = twf.main()
finally:
    # Log aggregate metrics to MLflow before ending the run
    if MLFLOW_AVAILABLE and _mlflow_run is not None:
        try:
            if _wf_results and isinstance(_wf_results, dict) and 'agg_ic' in _wf_results:
                mlflow.log_metrics({
                    'agg_ic':      float(_wf_results.get('agg_ic', 0.0)),
                    'agg_ic_std':  float(_wf_results.get('agg_ic_std', 0.0)),
                    'agg_icir':    float(_wf_results.get('agg_icir', 0.0)),
                    'concat_ic':   float(_wf_results.get('concat_ic', 0.0)),
                    'n_folds':     float(_wf_results.get('n_folds', 0)),
                })
            mlflow.end_run()
            print("MLflow run ended.")
        except Exception as _e:
            print(f"MLflow end_run failed (non-fatal): {_e}")
            try:
                mlflow.end_run()
            except Exception:
                pass

    # Rename book-named files to wider_cnn-named (consistent tagging)
    for ckpt in RESULTS_DIR.glob('checkpoint_book_*.json'):
        wider_name = ckpt.name.replace('checkpoint_book_', 'checkpoint_wider_cnn_')
        ckpt.rename(ckpt.parent / wider_name)
        print(f"Renamed: {wider_name}")
    for pred in RESULTS_DIR.glob('oos_predictions_book_*.npz'):
        wider_name = pred.name.replace('oos_predictions_book_', 'oos_predictions_wider_cnn_')
        if not (pred.parent / wider_name).exists():
            pred.rename(pred.parent / wider_name)
            print(f"Renamed: {wider_name}")
    for pred in RESULTS_DIR.glob('ckpt_preds_book_*.npz'):
        wider_name = pred.name.replace('ckpt_preds_book_', 'ckpt_preds_wider_cnn_')
        if not (pred.parent / wider_name).exists():
            pred.rename(pred.parent / wider_name)
            print(f"Renamed: {wider_name}")
