"""
CNN-Mamba v2 Warm-Start Walk-Forward Test (HC #86)

PURPOSE: Walk-forward with warm start from existing fold_10 weights.
Test: train on sliding 60d window ending Mar 31, warm start from fold_10_best.pt,
      OOT = April dates. Then analyze April IC to see if walk-forward resolves decay.

This is a WRAPPER — it calls train_cnn_mamba.py's main logic with warm-start injection.

Usage:
    python warmstart_cnn_mamba.py \
        --warm-start-weights /path/to/fold_10_best.pt \
        --data-dir /path/to/mbo_events \
        --output-dir /path/to/output \
        --start-oot-date 20260401 \
        --window-days 60

Walk-forward increments: trains fold with OOT=Apr1, then OOT=Apr7, etc.
"""

import os
import sys
import gc
import time
import logging
import argparse
import socket
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import scipy.stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Import from the main training script
sys.path.insert(0, str(Path(__file__).parent))
from train_cnn_mamba import (
    CNNMamba, MboEventDataset, LazyMboEventDataset, FileSequentialSampler,
    compute_ic, run_oot_inference, count_parameters,
    HORIZONS, WINDOW_SIZE, STRIDE, BATCH_SIZE,
    MAMBA_D_MODEL, MAMBA_D_STATE, MAMBA_N_LAYERS,
    MAMBA_DROPOUT, MAMBA_DT_RANK, MAMBA_D_CONV,
    CNN_CHANNELS, CNN_KERNEL, CNN_LAYERS,
    LR, EPOCHS_PER_FOLD, WARMUP_STEPS, GRAD_CLIP,
    N_TOTAL_FEATURES, SKIP_NORMALIZE,
)

# MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def _detect_mlflow_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    try:
        s = socket.create_connection(("localhost", 5000), timeout=2)
        s.close()
        return "http://localhost:5000"
    except Exception:
        pass
    return "http://neptune-win:5000"


def get_sorted_npz_files(data_dir: Path) -> List[Path]:
    """Get all NPZ files sorted by date."""
    npz_files = sorted(data_dir.glob("*.npz"))
    # Filter out files with all-NaN labels
    valid = []
    for f in npz_files:
        try:
            with np.load(f, allow_pickle=True) as d:
                if "labels_1s" in d:
                    labels = d["labels_1s"]
                    if not np.all(np.isnan(labels)):
                        valid.append(f)
                else:
                    valid.append(f)
        except Exception:
            pass
    return valid


def extract_date(f: Path) -> str:
    """Extract YYYYMMDD date from NPZ filename."""
    name = f.stem
    # Try common patterns
    for part in name.split("_"):
        if len(part) == 8 and part.isdigit():
            return part
    # Fallback: last 8 digits
    digits = "".join(c for c in name if c.isdigit())
    return digits[-8:] if len(digits) >= 8 else name


def train_one_fold_warmstart(
    model, train_loader, oot_loader,
    fold_idx, output_dir, device, total_steps,
    use_amp=True, lr=None,
):
    """Train one fold — identical to train_cnn_mamba but with lower LR for warm-start."""
    lr = lr or LR
    # Use lower LR for warm-start (fine-tuning, not from scratch)
    warmstart_lr = lr * 0.3  # 30% of base LR for fine-tuning

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=warmstart_lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=warmstart_lr, total_steps=max(total_steps, 1),
        pct_start=0.1, anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()

        for events, labels in train_loader:
            events = events.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                preds = model(events)
                loss = F.mse_loss(preds, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

            if n_batches % 100 == 0:
                elapsed = time.time() - epoch_start
                logger.info(f"  Batch {n_batches}/{len(train_loader)} | Loss: {epoch_loss/n_batches:.4f} | {elapsed:.0f}s")

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for events, labels in oot_loader:
                events = events.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with amp_ctx:
                    preds = model(events)
                    loss = F.mse_loss(preds, labels)
                val_loss += loss.item() * events.size(0)
                val_n += events.size(0)
        val_loss /= max(val_n, 1)

        logger.info(
            f"  Epoch {epoch+1}/{EPOCHS_PER_FOLD} | "
            f"Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {time.time()-epoch_start:.0f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            ckpt = {
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
                "fold": fold_idx,
                "warm_started": True,
            }
            torch.save(ckpt, output_dir / f"fold_{fold_idx:02d}_best.pt")
            logger.info(f"  ★ New best val_loss: {val_loss:.6f} (epoch {epoch+1})")

    logger.info(f"  Best epoch: {best_epoch+1}, best val_loss: {best_val_loss:.6f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="CNN-Mamba v2 Warm-Start Walk-Forward")
    parser.add_argument("--warm-start-weights", type=str, required=True,
                        help="Path to existing fold weights (e.g., fold_10_best.pt)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to MBO event NPZ files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for walk-forward results")
    parser.add_argument("--start-oot-date", type=str, default="20260401",
                        help="First OOT date (YYYYMMDD)")
    parser.add_argument("--window-days", type=int, default=60,
                        help="Sliding window size in days")
    parser.add_argument("--step-days", type=int, default=5,
                        help="Walk-forward step size in trading days")
    parser.add_argument("--n-steps", type=int, default=5,
                        help="Number of walk-forward steps")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Epochs per fold (fewer for warm-start)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # File logging
    fh = logging.FileHandler(output_dir / "warmstart_wf.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Override epochs for warm-start (fewer needed)
    global EPOCHS_PER_FOLD
    EPOCHS_PER_FOLD = args.epochs

    # Load all NPZ files
    data_dir = Path(args.data_dir)
    all_files = get_sorted_npz_files(data_dir)
    logger.info(f"Found {len(all_files)} valid NPZ files in {data_dir}")

    # Map dates to files
    date_to_file = {}
    for f in all_files:
        d = extract_date(f)
        date_to_file[d] = f
    dates_sorted = sorted(date_to_file.keys())
    logger.info(f"Date range: {dates_sorted[0]} to {dates_sorted[-1]}")

    # Load warm-start weights
    warm_weights_path = Path(args.warm_start_weights)
    if not warm_weights_path.exists():
        logger.error(f"Warm-start weights not found: {warm_weights_path}")
        return
    warm_ckpt = torch.load(warm_weights_path, map_location="cpu", weights_only=False)
    logger.info(f"Loaded warm-start weights from {warm_weights_path}")
    logger.info(f"  Original val_loss: {warm_ckpt.get('val_loss', 'N/A')}")

    # Find start OOT index
    oot_date = args.start_oot_date
    if oot_date not in dates_sorted:
        # Find closest date >= start_oot_date
        candidates = [d for d in dates_sorted if d >= oot_date]
        if not candidates:
            logger.error(f"No dates >= {oot_date} in data")
            return
        oot_date = candidates[0]
        logger.info(f"Adjusted start OOT date to {oot_date} (closest available)")

    # MLflow
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(_detect_mlflow_uri())
        mlflow.set_experiment("cnn_mamba_v2_warmstart_wf")
        mlflow_run = mlflow.start_run(
            run_name=f"warmstart_wf_{time.strftime('%Y%m%d_%H%M')}"
        )
        mlflow.log_params({
            "warm_start_weights": str(warm_weights_path),
            "start_oot_date": args.start_oot_date,
            "window_days": args.window_days,
            "step_days": args.step_days,
            "n_steps": args.n_steps,
            "epochs_per_fold": args.epochs,
            "lr_scale": 0.3,
            "model": "CNNMamba_v2_warmstart",
        })

    use_amp = device.type == "cuda"
    all_results = []

    # Walk-forward loop
    for step in range(args.n_steps):
        oot_idx = dates_sorted.index(oot_date) if oot_date in dates_sorted else -1
        if oot_idx < 0 or oot_idx >= len(dates_sorted):
            logger.info(f"No more OOT dates available after step {step}")
            break

        # Training window: window_days before OOT
        train_end_idx = oot_idx
        train_start_idx = max(0, train_end_idx - args.window_days)
        train_files = [date_to_file[dates_sorted[i]] for i in range(train_start_idx, train_end_idx)]

        # OOT: step_days after train end
        oot_end_idx = min(len(dates_sorted), oot_idx + args.step_days)
        oot_files = [date_to_file[dates_sorted[i]] for i in range(oot_idx, oot_end_idx)]

        if not train_files or not oot_files:
            logger.warning(f"Step {step}: empty train or OOT set, stopping")
            break

        train_dates = [extract_date(f) for f in train_files]
        oot_dates = [extract_date(f) for f in oot_files]

        logger.info(f"\n{'='*70}")
        logger.info(f"WARM-START FOLD {step} | Train: {len(train_files)} days "
                     f"({train_dates[0]}->{train_dates[-1]}) | "
                     f"OOT: {len(oot_files)} days ({oot_dates[0]}->{oot_dates[-1]})")
        logger.info(f"{'='*70}")

        # Build datasets
        logger.info("Building train dataset...")
        TrainDSClass = LazyMboEventDataset if len(train_files) > 50 else MboEventDataset
        train_ds = TrainDSClass(
            train_files, window_size=WINDOW_SIZE, stride=STRIDE,
            normalize_features=not SKIP_NORMALIZE,
        )
        feature_stats = train_ds.get_feature_stats()

        logger.info("Building OOT dataset...")
        oot_ds = MboEventDataset(
            oot_files, window_size=WINDOW_SIZE, stride=STRIDE,
            feature_stats=feature_stats,
            normalize_features=not SKIP_NORMALIZE,
        )

        # Use FileSequentialSampler for LazyMboEventDataset (prevents OOM from random access)
        _train_sampler = FileSequentialSampler(train_ds) if isinstance(train_ds, LazyMboEventDataset) else None
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE,
            shuffle=(_train_sampler is None),
            sampler=_train_sampler,
            num_workers=0, pin_memory=True, drop_last=True,
        )
        oot_loader = DataLoader(
            oot_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        # Create model and load warm-start weights
        model = CNNMamba(
            d_model=MAMBA_D_MODEL, d_state=MAMBA_D_STATE,
            n_layers=MAMBA_N_LAYERS, dt_rank=MAMBA_DT_RANK,
            d_conv=MAMBA_D_CONV, dropout=MAMBA_DROPOUT,
            n_targets=len(HORIZONS),
            cnn_channels=CNN_CHANNELS, cnn_kernel=CNN_KERNEL,
            cnn_layers=CNN_LAYERS,
        ).to(device)

        # WARM START: load weights from previous checkpoint
        state_dict = warm_ckpt["model_state"]
        loaded, skipped = 0, 0
        model_state = model.state_dict()
        for key in state_dict:
            if key in model_state and state_dict[key].shape == model_state[key].shape:
                model_state[key] = state_dict[key]
                loaded += 1
            else:
                skipped += 1
                if key in model_state:
                    logger.warning(f"  Shape mismatch: {key} "
                                   f"({state_dict[key].shape} vs {model_state[key].shape})")
        model.load_state_dict(model_state)
        logger.info(f"Warm-start: loaded {loaded}/{loaded+skipped} tensors "
                     f"({skipped} skipped due to shape mismatch)")

        if step == 0:
            n_params = count_parameters(model)
            logger.info(f"Model parameters: {n_params:,}")

        # Train
        total_steps = EPOCHS_PER_FOLD * len(train_loader)
        train_one_fold_warmstart(
            model, train_loader, oot_loader,
            step, output_dir, device, total_steps,
            use_amp=use_amp,
        )

        # Reload best checkpoint for OOT inference
        ckpt_path = output_dir / f"fold_{step:02d}_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            logger.info(f"Loaded best checkpoint (val_loss={ckpt['val_loss']:.6f})")
            # Update warm_ckpt for NEXT fold (cascading warm-start)
            warm_ckpt = ckpt

        # OOT inference
        logger.info("Running OOT inference...")
        oot_preds, oot_labels, oot_embeds = run_oot_inference(
            model, oot_loader, device, use_amp=use_amp, extract_embeddings=True,
        )

        # Compute IC per horizon
        fold_ics = {}
        for i, h in enumerate(HORIZONS):
            ic = compute_ic(oot_preds[:, i], oot_labels[:, i])
            fold_ics[h] = ic

        ic_str = " | ".join(f"{h}: {fold_ics[h]:.4f}" for h in HORIZONS)
        logger.info(f"Fold {step} OOT IC | {ic_str}")

        # Save predictions
        np.savez_compressed(
            output_dir / f"fold_{step:02d}_oot_predictions.npz",
            predictions=oot_preds,
            labels=oot_labels,
            dates=np.array(oot_dates),
        )

        # Save feature stats
        np.savez_compressed(
            output_dir / f"fold_{step:02d}_feature_stats.npz",
            **feature_stats,
        )

        result = {
            "step": step,
            "train_dates": f"{train_dates[0]}-{train_dates[-1]}",
            "oot_dates": oot_dates,
            "n_train_days": len(train_files),
            "n_oot_days": len(oot_files),
            "ics": fold_ics,
            "n_samples": len(oot_preds),
        }
        all_results.append(result)

        if MLFLOW_AVAILABLE and mlflow_run:
            for h in HORIZONS:
                mlflow.log_metric(f"oot_ic_{h}", fold_ics[h], step=step)
            mlflow.log_metric("n_oot_samples", len(oot_preds), step=step)

        # Advance OOT date
        next_oot_idx = oot_idx + args.step_days
        if next_oot_idx < len(dates_sorted):
            oot_date = dates_sorted[next_oot_idx]
        else:
            logger.info("Reached end of available dates")
            break

        # Cleanup
        del model, train_ds, oot_ds, train_loader, oot_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"WARM-START WALK-FORWARD COMPLETE — {len(all_results)} steps")
    logger.info(f"{'='*70}")
    for r in all_results:
        ic_str = " | ".join(f"{h}: {r['ics'][h]:.4f}" for h in HORIZONS)
        logger.info(f"  Step {r['step']}: OOT {r['oot_dates']} | {ic_str} | n={r['n_samples']}")

    if MLFLOW_AVAILABLE and mlflow_run:
        # Log summary
        for h in HORIZONS:
            ics = [r["ics"][h] for r in all_results]
            mlflow.log_metric(f"mean_ic_{h}", np.mean(ics))
            mlflow.log_metric(f"std_ic_{h}", np.std(ics))
        mlflow.end_run()

    logger.info("Done!")


if __name__ == "__main__":
    main()
