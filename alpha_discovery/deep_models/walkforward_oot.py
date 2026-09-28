"""
walkforward_oot.py — Walk-Forward CNN Training on OOT Period

Continues the walk-forward BookSpatialCNN training from the IS period
(100 days, Jul-Nov 2025) into the OOT period (68 days, Dec 2025 - Mar 2026).

Protocol: Expanding window. For OOT day d:
  - Train on ALL IS days (100) + OOT days 0..d-2 (1-day purge gap)
  - Predict on OOT day d
  - Save weights + predictions

Usage:
    python walkforward_oot.py [--device cuda] [--epochs 3] [--batch-size 512]
"""

import argparse
import gc
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ─── Root / path setup ────────────────────────────────────────────────────────
FILE_DIR = Path(__file__).parent.resolve()
ROOT = FILE_DIR.parent.parent  # Lvl3Quant/

# ─── Import BookSpatialCNN ─────────────────────────────────────────────────────
sys.path.insert(0, str(FILE_DIR))
from book_spatial_cnn import BookSpatialCNN  # noqa: E402

# ─── Logging ──────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("walkforward_oot")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─── Target computation ────────────────────────────────────────────────────────
def compute_mfe_net(mid_prices: np.ndarray, horizon: int = 100) -> np.ndarray:
    """
    Max Favorable Excursion (net) — same convention as train_oot_predictions.py.

    For each bar i, look forward `horizon` bars and compute:
        mfe_long  = max(mid[i+1..i+horizon]) - mid[i]
        mfe_short = mid[i] - min(mid[i+1..i+horizon])
        mfe_net   = mfe_long - mfe_short   (positive = bullish edge)

    Last `horizon` bars of the array are set to NaN (boundary).
    """
    n = len(mid_prices)
    targets = np.full(n, np.nan, dtype=np.float32)

    if n <= horizon:
        return targets

    mid = mid_prices.astype(np.float64)

    for i in range(n - horizon):
        future = mid[i + 1: i + 1 + horizon]
        mfe_long = np.max(future) - mid[i]
        mfe_short = mid[i] - np.min(future)
        targets[i] = float(mfe_long - mfe_short)

    return targets.astype(np.float32)


# ─── Dataset ───────────────────────────────────────────────────────────────────
class BookDataset(Dataset):
    """
    Flat dataset built from pre-concatenated tensors + targets.
    Only valid (non-NaN) samples are included.
    """

    def __init__(self, book_tensors: np.ndarray, targets: np.ndarray,
                 subsample: int = 1):
        """
        book_tensors : (N, window_size, num_levels, num_features)  float32
        targets      : (N,)  float32, may contain NaN
        subsample    : keep every Nth sample
        """
        assert len(book_tensors) == len(targets), "tensor/target length mismatch"

        # Apply subsample first, then filter NaN
        idx = np.arange(0, len(targets), subsample)
        t_sub = targets[idx]
        valid = np.isfinite(t_sub)
        idx = idx[valid]

        self.X = book_tensors[idx]   # (M, window_size, num_levels, num_features)
        self.y = targets[idx]        # (M,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i])   # (window_size, num_levels, num_features)
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return x, y


# ─── Feature transform ─────────────────────────────────────────────────────────
def apply_feature_transform(bt: np.ndarray) -> np.ndarray:
    """
    Apply log1p to volume/size channels (1, 2, 3). Channel 0 is price, no transform.
    bt shape: (N, window_size, num_features)  — note: NOT including num_levels dim here.

    Actually shape from .npz is (N, 20, 4): (bars, levels, features).
    We reshape to (N, window_size=20, num_levels=20, num_features=4) via the caller.

    This function receives (N, levels, features) and transforms channels 1,2,3.
    """
    bt = bt.astype(np.float32)
    bt[:, :, 1] = np.log1p(np.abs(bt[:, :, 1])) * np.sign(bt[:, :, 1])
    bt[:, :, 2] = np.log1p(np.abs(bt[:, :, 2])) * np.sign(bt[:, :, 2])
    bt[:, :, 3] = np.log1p(np.abs(bt[:, :, 3])) * np.sign(bt[:, :, 3])
    return bt


# ─── Day loading ───────────────────────────────────────────────────────────────
def load_day(npz_path: Path, horizon: int = 100,
             window_size: int = 20) -> dict | None:
    """
    Load a single .npz file and return a dict with:
        date      : str  (YYYY-MM-DD)
        bt        : np.ndarray (N, window_size, num_levels, num_features) float32
                    where num_levels=20, num_features=4
        targets   : np.ndarray (N,) float32 (NaN for last `horizon` bars)
        mid       : np.ndarray (N,) float64
        n         : int
    Returns None on error.
    """
    try:
        npz = np.load(npz_path, allow_pickle=False)
        bt_raw = npz["book_tensors"].astype(np.float32)  # (N, 20, 4) expected
        mid = npz["mid_prices"].astype(np.float64)
        n = len(mid)

        if n < window_size + horizon + 10:
            return None  # too short

        # book_tensors shape from file: (N, num_levels, num_features) = (N, 20, 4)
        # We want (N, window_size, num_levels, num_features).
        # The CNN uses a sliding window of `window_size` consecutive bars.
        # Build windows: shape (N - window_size + 1, window_size, num_levels, num_features)
        #
        # Apply feature transform BEFORE windowing (cheaper).
        bt_raw = apply_feature_transform(bt_raw)  # (N, 20, 4)

        num_levels = bt_raw.shape[1]
        num_features = bt_raw.shape[2]

        # Sliding window via stride tricks — zero-copy view
        # Result shape: (n_windows, window_size, num_levels, num_features)
        n_windows = n - window_size + 1
        shape = (n_windows, window_size, num_levels, num_features)
        strides = (bt_raw.strides[0],
                   bt_raw.strides[0],
                   bt_raw.strides[1],
                   bt_raw.strides[2])
        bt_windows = np.lib.stride_tricks.as_strided(bt_raw, shape=shape,
                                                     strides=strides)
        # as_strided is a VIEW — make a copy so we own the memory after npz is freed
        bt_windows = bt_windows.copy()

        # Targets align with the LAST bar of each window
        # window i covers bars [i, i+window_size-1], label = bar i+window_size-1
        targets_all = compute_mfe_net(mid, horizon=horizon)  # (N,)
        targets_windowed = targets_all[window_size - 1:]     # (n_windows,)

        # Mid prices aligned with window end bar
        mid_windowed = mid[window_size - 1:]

        date_str = npz_path.name.split("_book_tensors")[0]

        return {
            "date": date_str,
            "bt": bt_windows,          # (n_windows, ws, nl, nf)
            "targets": targets_windowed,
            "mid": mid_windowed.astype(np.float32),
            "n": n_windows,
        }

    except Exception as e:
        return None


# ─── Training ─────────────────────────────────────────────────────────────────
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    use_amp: bool,
                    scaler,
                    logger: logging.Logger,
                    epoch: int) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                pred = model(xb).squeeze(-1)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 200 == 0:
            logger.debug(f"  Epoch {epoch} batch {batch_idx}/{len(loader)} "
                         f"loss={loss.item():.4f}")

    return total_loss / max(n_batches, 1)


def compute_ic(preds: np.ndarray, targets: np.ndarray) -> float:
    """Pearson IC, ignoring NaN."""
    mask = np.isfinite(preds) & np.isfinite(targets)
    if mask.sum() < 10:
        return float("nan")
    p = preds[mask]
    t = targets[mask]
    if p.std() < 1e-8 or t.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Walk-Forward CNN OOT training")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs per fold (default: 3)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--subsample-train", type=int, default=3,
                        help="Use every Nth training bar (default: 3)")
    parser.add_argument("--is-dir", type=Path,
                        default=ROOT / "data/processed/dl_book_cache",
                        help="IS book cache directory")
    parser.add_argument("--oot-dir", type=Path,
                        default=ROOT / "data/processed/dl_book_cache_oot",
                        help="OOT book cache directory")
    parser.add_argument("--output-dir", type=Path,
                        default=FILE_DIR / "results",
                        help="Output directory for predictions + logs")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=FILE_DIR / "checkpoints",
                        help="Directory for model checkpoints")
    parser.add_argument("--min-train-days", type=int, default=5,
                        help="Minimum training days before predicting (default: 5)")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save checkpoint every N folds (default: 1)")
    parser.add_argument("--horizon", type=int, default=100,
                        help="MFE horizon in bars (default: 100)")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--num-levels", type=int, default=20)
    parser.add_argument("--num-features", type=int, default=4)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.output_dir / f"walkforward_oot_{timestamp}.log"
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("Walk-Forward CNN OOT Training")
    logger.info(f"Timestamp   : {timestamp}")
    logger.info(f"IS dir      : {args.is_dir}")
    logger.info(f"OOT dir     : {args.oot_dir}")
    logger.info(f"Output dir  : {args.output_dir}")
    logger.info(f"Checkpoint  : {args.checkpoint_dir}")
    logger.info(f"Epochs/fold : {args.epochs}")
    logger.info(f"Batch size  : {args.batch_size}")
    logger.info(f"Subsample   : {args.subsample_train}")
    logger.info(f"Horizon     : {args.horizon}")
    logger.info(f"Window size : {args.window_size}")
    logger.info("=" * 70)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        use_amp = False
        if args.device == "cuda":
            logger.warning("CUDA not available, falling back to CPU")
        logger.info(f"Using CPU ({torch.get_num_threads()} threads)")

    # ── Gather file lists ─────────────────────────────────────────────────────
    is_files = sorted(args.is_dir.glob("*_book_tensors.npz"))
    oot_files = sorted(args.oot_dir.glob("*_book_tensors.npz"))

    if len(is_files) == 0:
        logger.error(f"No IS files found in {args.is_dir}")
        sys.exit(1)
    if len(oot_files) == 0:
        logger.error(f"No OOT files found in {args.oot_dir}")
        sys.exit(1)

    logger.info(f"IS files : {len(is_files)} days")
    logger.info(f"OOT files: {len(oot_files)} days")

    # ── Load IS data (day by day, kept as list) ───────────────────────────────
    logger.info("Loading IS data...")
    t0 = time.time()
    is_days = []
    for f in is_files:
        day = load_day(f, horizon=args.horizon, window_size=args.window_size)
        if day is None:
            logger.warning(f"  Skipped IS file (too short or error): {f.name}")
            continue
        is_days.append(day)
        logger.debug(f"  Loaded IS {day['date']}: {day['n']} windows")

    logger.info(f"IS load complete: {len(is_days)} days in {time.time()-t0:.1f}s")

    # ── Load OOT data (day by day, kept as list) ──────────────────────────────
    logger.info("Loading OOT data...")
    t0 = time.time()
    oot_days = []
    for f in oot_files:
        day = load_day(f, horizon=args.horizon, window_size=args.window_size)
        if day is None:
            logger.warning(f"  Skipped OOT file (too short or error): {f.name}")
            continue
        oot_days.append(day)
        logger.debug(f"  Loaded OOT {day['date']}: {day['n']} windows")

    logger.info(f"OOT load complete: {len(oot_days)} days in {time.time()-t0:.1f}s")

    if len(oot_days) == 0:
        logger.error("No valid OOT days loaded. Exiting.")
        sys.exit(1)

    # ── Model factory ─────────────────────────────────────────────────────────
    def make_model() -> BookSpatialCNN:
        m = BookSpatialCNN(
            window_size=args.window_size,
            num_levels=args.num_levels,
            num_features=args.num_features,
            spatial_channels=(32, 64, 128, 256),
            temporal_channels=256,
            dropout=0.1,
            num_classes=1,
        )
        return m.to(device)

    # ── AMP scaler ────────────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    criterion = nn.HuberLoss(delta=1.0)

    # ── Walk-forward loop ──────────────────────────────────────────────────────
    all_predictions = {}   # date -> {'preds': np.ndarray, 'mid': np.ndarray}
    fold_ics = []

    logger.info("=" * 70)
    logger.info("Starting Walk-Forward OOT Loop")
    logger.info("=" * 70)

    session_start = time.time()

    for fold_idx, oot_day in enumerate(oot_days):
        oot_date = oot_day["date"]
        fold_start = time.time()

        # Training data = IS days + OOT days 0..fold_idx-2 (1-day purge gap)
        oot_train_days = oot_days[:max(0, fold_idx - 1)]
        train_days = is_days + oot_train_days
        n_train_days = len(train_days)

        logger.info(f"\nFold {fold_idx+1}/{len(oot_days)} — OOT date: {oot_date}")
        logger.info(f"  Training on {len(is_days)} IS + {len(oot_train_days)} OOT "
                    f"= {n_train_days} days total")

        if n_train_days < args.min_train_days:
            logger.info(f"  Skipping — only {n_train_days} training days "
                        f"(min={args.min_train_days})")
            continue

        # ── Build training arrays (concatenate selected days) ─────────────────
        bt_list = [d["bt"] for d in train_days]
        tgt_list = [d["targets"] for d in train_days]

        bt_train = np.concatenate(bt_list, axis=0)
        tgt_train = np.concatenate(tgt_list, axis=0)

        n_total = len(tgt_train)
        n_valid = int(np.isfinite(tgt_train).sum())
        logger.info(f"  Training samples: {n_total} total, "
                    f"{n_valid} valid (after NaN filter)")

        # ── Dataset + DataLoader ──────────────────────────────────────────────
        dataset = BookDataset(bt_train, tgt_train,
                              subsample=args.subsample_train)
        del bt_train, tgt_train  # free before training
        gc.collect()

        n_ds = len(dataset)
        logger.info(f"  Dataset size after subsample+NaN filter: {n_ds}")

        if n_ds < 100:
            logger.warning(f"  Too few training samples ({n_ds}), skipping fold")
            del dataset
            gc.collect()
            continue

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        # ── Model + optimizer ─────────────────────────────────────────────────
        model = make_model()
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=3e-4, weight_decay=0.01)
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model parameters: {param_count:,}")

        # ── Training epochs ───────────────────────────────────────────────────
        for ep in range(1, args.epochs + 1):
            ep_start = time.time()
            loss = train_one_epoch(
                model, loader, optimizer, criterion, device,
                use_amp, scaler, logger, ep
            )
            logger.info(f"  Epoch {ep}/{args.epochs} — "
                        f"loss={loss:.4f} ({time.time()-ep_start:.1f}s)")

        del loader, dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (fold_idx + 1) % args.save_every == 0:
            ckpt_path = (args.checkpoint_dir /
                         f"cnn_oot_fold{fold_idx+1}_{oot_date}.pt")
            torch.save({
                "fold": fold_idx + 1,
                "date": oot_date,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epochs": args.epochs,
                "n_train_days": n_train_days,
            }, ckpt_path)
            logger.info(f"  Checkpoint saved: {ckpt_path.name}")

        # ── Inference on OOT day ──────────────────────────────────────────────
        model.eval()
        bt_oot = oot_day["bt"]     # (N, ws, nl, nf)
        mid_oot = oot_day["mid"]   # (N,)
        targets_oot = oot_day["targets"]  # (N,)

        preds_list = []
        with torch.no_grad():
            for i in range(0, len(bt_oot), args.batch_size):
                chunk = torch.from_numpy(
                    bt_oot[i: i + args.batch_size]).to(device)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        out = model(chunk).squeeze(-1)
                else:
                    out = model(chunk).squeeze(-1)
                preds_list.append(out.cpu().float().numpy())

        preds = np.concatenate(preds_list, axis=0).astype(np.float32)

        # IC
        ic = compute_ic(preds, targets_oot)
        fold_ics.append(ic)
        logger.info(f"  OOT day IC = {ic:.4f} | "
                    f"fold time = {time.time()-fold_start:.1f}s")

        # Store predictions
        all_predictions[oot_date] = {
            "preds": preds,
            "mid": mid_oot,
        }

        # Cleanup
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Walk-Forward OOT Complete")
    logger.info(f"Total wall time: {(time.time()-session_start)/60:.1f} min")
    logger.info(f"Folds completed: {len(fold_ics)}")

    if fold_ics:
        arr = np.array([x for x in fold_ics if np.isfinite(x)])
        logger.info(f"Mean IC (finite): {arr.mean():.4f}")
        logger.info(f"Std  IC         : {arr.std():.4f}")
        logger.info(f"Pct positive    : {(arr > 0).mean()*100:.1f}%")
        logger.info(f"Min / Max IC    : {arr.min():.4f} / {arr.max():.4f}")

        # ICIR
        if arr.std() > 1e-8:
            logger.info(f"ICIR            : {arr.mean()/arr.std():.3f}")
    else:
        logger.warning("No folds completed — no predictions saved.")

    # ── Save predictions ───────────────────────────────────────────────────────
    if all_predictions:
        pred_path = (args.output_dir /
                     f"oos_predictions_book_oot_{timestamp}.npz")
        save_dict = {}
        for date, v in all_predictions.items():
            save_dict[f"{date}_preds"] = v["preds"]
            save_dict[f"{date}_mid"] = v["mid"]
        np.savez_compressed(pred_path, **save_dict)
        logger.info(f"\nPredictions saved: {pred_path}")
        logger.info(f"  Keys: {len(all_predictions)} dates "
                    f"({list(all_predictions.keys())[0]} .. "
                    f"{list(all_predictions.keys())[-1]})")
    else:
        logger.warning("No predictions to save.")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
