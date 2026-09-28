"""
dispatch_v34_fold0.py — v3.4 dual-trunk CNN-Mamba walk-forward dispatch driver.

Clones the v3.3 `run_weekly_wf_v33` pattern but swaps in v3.4 classes
(CNNMambaV34DualTrunk + SmartV34DualTrunkDataset + collate_v34 + train_one_fold_v34).

Defaults are tuned for the v3.4 fold-0 falsification gate:
  * --n-folds 1 → run ONLY fold-0 (the gate-test fold)
  * --warmstart-ckpt → v3.2 long_context intra-ckpt (v3.3 didn't save .pt weights)

Authorized by HC #365 (v3.4 sole priority) + HC #366 (autonomous lean-and-execute).
HC #307D-compliant new tooling under scripts/v3_4_research/.

Usage (Neptune):
  PYTHONPATH=/home/nick/Lvl3Quant python3 scripts/v3_4_research/dispatch_v34_fold0.py \\
      --device cuda --n-folds 1 \\
      --warmstart-ckpt /home/nick/Lvl3Quant/output/cnn_mamba_v3_2_long_context/fold_00_intra_ckpt.pt
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Path bootstrap so this script can be run from anywhere
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import (  # noqa: E402
    SmartV32Dataset,
    build_weekly_fold_schedule,
    date_from_path,
    WINDOW_SIZE_T1,
    WINDOW_SIZE_T2,
    WINDOW_SIZE_T3,
    STRIDE,
    BATCH_SIZE,
    EPOCHS_PER_FOLD,
    WF_TRAIN_DAYS,
    DEFAULT_DATA_DIR,
    DEFAULT_FIFO_LABEL_DIR,
    DEFAULT_ALPHA_LABEL_DIR,
    DEFAULT_PT_PRED_DIR,
    DEFAULT_TIER2_PARQUET_ROOT,
    DEFAULT_TIER3_PARQUET_ROOT,
)
from alpha_discovery.deep_models.train_cnn_mamba_v3_4 import (  # noqa: E402
    CNNMambaV34DualTrunk,
    SmartV34DualTrunkDataset,
    collate_v34,
    train_one_fold_v34,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_BOOK_FEATURES_DIR,
    MLFLOW_EXPERIMENT,
    ALL_HEAD_NAMES,
)

try:
    import mlflow  # noqa: E402
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


PROJECT_ROOT = Path("/home/jupiter/Lvl3Quant")
V32_WARMSTART_DEFAULT = str(
    PROJECT_ROOT / "output" / "cnn_mamba_v3_2_long_context" / "fold_00_intra_ckpt.pt"
)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")


def _count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def discover_aligned_dates(data_dir: Path, book_dir: Path):
    """Intersect event NPZ dates with book NPZ dates → only fully-aligned dates train."""
    event_npz = sorted(data_dir.glob("*_mbo_events.npz"))
    event_dates = set()
    for p in event_npz:
        d = date_from_path(p)
        if d:
            event_dates.add(d)
    book_dates = set()
    for p in book_dir.glob("*_book_features.npz"):
        stem = p.stem.split("_")[0]
        if len(stem) == 8 and stem.isdigit():
            book_dates.add(stem)
    return sorted(event_dates & book_dates)


def main():
    p = argparse.ArgumentParser(description="v3.4 dual-trunk WF dispatch (default: fold-0 only)")
    p.add_argument("--n-folds", type=int, default=1, help="default 1 = fold-0 only for falsification gate")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmstart-ckpt", default=V32_WARMSTART_DEFAULT,
                   help="path to v3.2 long_context fold_00_intra_ckpt.pt")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--book-features-dir", default=DEFAULT_BOOK_FEATURES_DIR)
    p.add_argument("--fifo-label-dir", default=str(DEFAULT_FIFO_LABEL_DIR))
    p.add_argument("--alpha-label-dir", default=str(DEFAULT_ALPHA_LABEL_DIR))
    p.add_argument("--pt-pred-dir", default=str(DEFAULT_PT_PRED_DIR))
    p.add_argument("--tier2-parquet-root", default=str(DEFAULT_TIER2_PARQUET_ROOT))
    p.add_argument("--tier3-parquet-root", default=str(DEFAULT_TIER3_PARQUET_ROOT))
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print(">>> dispatch_v34: CUDA unavailable — falling back to CPU", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    use_amp = (device.type == "cuda") and not args.no_amp

    print(f">>> dispatch_v34: device={device} amp={use_amp} output_dir={output_dir}", flush=True)
    print(f">>> dispatch_v34: warmstart={args.warmstart_ckpt}", flush=True)

    data_dir = Path(args.data_dir)
    book_dir = Path(args.book_features_dir)
    aligned = discover_aligned_dates(data_dir, book_dir)
    if not aligned:
        print(f">>> dispatch_v34: ERROR — no aligned book+event dates found", flush=True)
        return 1
    print(f">>> dispatch_v34: aligned book+event dates = {len(aligned)} "
          f"({aligned[0]} → {aligned[-1]})", flush=True)

    folds = build_weekly_fold_schedule(aligned, n_folds=args.n_folds, train_days=WF_TRAIN_DAYS)
    print(f">>> dispatch_v34: built {len(folds)} fold(s)", flush=True)
    for f in folds:
        print(f"  Fold {f['fold']}: train {f['train_start']}→{f['train_end']} "
              f"({len(f['train_dates'])}d) | "
              f"OOT {f['oot_start']}→{f['oot_end']} ({len(f['oot_dates'])}d)", flush=True)
    with open(output_dir / "fold_schedule.json", "w") as fh:
        json.dump(folds, fh, indent=2, default=str)

    # MLflow init
    mlflow_run = None
    if MLFLOW_AVAILABLE and not args.no_mlflow:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            gpu_name = (torch.cuda.get_device_name(0)
                        if device.type == "cuda" else "cpu")
            mlflow_run = mlflow.start_run(
                run_name=f"v3.4_fold0_{time.strftime('%Y%m%d_%H%M')}_{socket.gethostname()}",
                tags={
                    "model_family": "cnn_mamba_v3.4_dual_trunk",
                    "version": "3.4",
                    "purpose": "fold-0_falsification_gate",
                    "loss_kind": "uncertainty_weighted_kendall2018",
                },
            )
            mlflow.log_params({
                "model": "CNNMambaV34DualTrunk",
                "trainer": "v3.4_dual_trunk_uncertainty_weighted",
                "window_t1": WINDOW_SIZE_T1,
                "window_t2": WINDOW_SIZE_T2,
                "window_t3": WINDOW_SIZE_T3,
                "stride": STRIDE,
                "batch_size": BATCH_SIZE,
                "epochs_per_fold": EPOCHS_PER_FOLD,
                "warmstart_ckpt": args.warmstart_ckpt,
                "n_folds_planned": len(folds),
                "wf_train_days": WF_TRAIN_DAYS,
                "node": socket.gethostname(),
                "gpu": gpu_name,
                "mixed_precision": "bf16" if use_amp else "none",
                "n_heads": len(ALL_HEAD_NAMES),
            })
            print(f">>> dispatch_v34: MLflow run started: {mlflow_run.info.run_id}", flush=True)
        except Exception as e:
            print(f">>> dispatch_v34: MLflow init failed ({e}) — continuing without tracking", flush=True)
            mlflow_run = None

    try:
        for f_info in folds:
            fold_idx = f_info["fold"]
            print("=" * 60, flush=True)
            print(f"FOLD {fold_idx} | train {f_info['train_start']}→{f_info['train_end']} "
                  f"| OOT {f_info['oot_start']}→{f_info['oot_end']}", flush=True)
            print("=" * 60, flush=True)

            # Build v3.2 inner datasets (events only)
            train_inner = SmartV32Dataset(
                data_dir=data_dir,
                fifo_label_dir=Path(args.fifo_label_dir),
                alpha_label_dir=Path(args.alpha_label_dir),
                pt_pred_dir=Path(args.pt_pred_dir),
                tier2_parquet_root=Path(args.tier2_parquet_root),
                tier3_parquet_root=Path(args.tier3_parquet_root),
                dates=f_info["train_dates"],
                window_t1=WINDOW_SIZE_T1,
                window_t2=WINDOW_SIZE_T2,
                window_t3=WINDOW_SIZE_T3,
                stride=STRIDE,
                feature_stats=None,
                cache_size=1,
                require_alpha_labels=False,
            )
            feature_stats = train_inner.get_feature_stats()
            np.savez(
                output_dir / f"fold_{fold_idx:02d}_feature_stats.npz",
                mean_t1=feature_stats["mean_t1"], std_t1=feature_stats["std_t1"],
            )
            oot_inner = SmartV32Dataset(
                data_dir=data_dir,
                fifo_label_dir=Path(args.fifo_label_dir),
                alpha_label_dir=Path(args.alpha_label_dir),
                pt_pred_dir=Path(args.pt_pred_dir),
                tier2_parquet_root=Path(args.tier2_parquet_root),
                tier3_parquet_root=Path(args.tier3_parquet_root),
                dates=f_info["oot_dates"],
                window_t1=WINDOW_SIZE_T1,
                window_t2=WINDOW_SIZE_T2,
                window_t3=WINDOW_SIZE_T3,
                stride=STRIDE,
                feature_stats=feature_stats,
                cache_size=1,
                require_alpha_labels=False,
            )

            # Wrap with v3.4 book-loading dataset
            train_ds = SmartV34DualTrunkDataset(
                train_inner, book_features_dir=str(book_dir)
            )
            oot_ds = SmartV34DualTrunkDataset(
                oot_inner, book_features_dir=str(book_dir)
            )

            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                pin_memory=True, drop_last=True, collate_fn=collate_v34,
                persistent_workers=False, prefetch_factor=None,
            )
            oot_loader = DataLoader(
                oot_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0,
                pin_memory=True, collate_fn=collate_v34,
                persistent_workers=False, prefetch_factor=None,
            )

            model = CNNMambaV34DualTrunk().to(device)
            if fold_idx == 0:
                n_params = _count_params(model)
                print(f"Model parameters: {n_params:,}", flush=True)
                if MLFLOW_AVAILABLE and mlflow_run is not None:
                    try:
                        mlflow.log_params({"model_params": n_params})
                    except Exception:
                        pass

            warm_stats = model.load_v33_warmstart(args.warmstart_ckpt, device)
            if MLFLOW_AVAILABLE and mlflow_run is not None and fold_idx == 0:
                try:
                    mlflow.log_params({
                        "warmstart_loaded_tensors": warm_stats["loaded"],
                        "warmstart_partial_tensors": warm_stats["partial"],
                        "warmstart_skipped": warm_stats["skipped"],
                        "warmstart_random_init_params": warm_stats["random_init"],
                    })
                except Exception:
                    pass

            total_steps = EPOCHS_PER_FOLD * len(train_loader)
            print(f">>> dispatch_v34: fold {fold_idx} train_batches={len(train_loader)} "
                  f"oot_batches={len(oot_loader)} total_steps={total_steps}", flush=True)

            result = train_one_fold_v34(
                model, train_loader, oot_loader,
                fold_idx, output_dir, mlflow_run, device,
                total_train_steps=total_steps, use_amp=use_amp,
            )

            if result.get("falsification_killed"):
                print(f">>> dispatch_v34: FALSIFICATION KILL at fold {fold_idx}: "
                      f"{result['falsification_reason']}", flush=True)
                if MLFLOW_AVAILABLE and mlflow_run is not None:
                    try:
                        mlflow.set_tag("run_outcome", "falsification_killed")
                    except Exception:
                        pass
                break

            print(f">>> dispatch_v34: fold {fold_idx} complete | "
                  f"best_val_loss={result.get('best_val_loss'):.4f}", flush=True)

    finally:
        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass

    print(">>> dispatch_v34: DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
