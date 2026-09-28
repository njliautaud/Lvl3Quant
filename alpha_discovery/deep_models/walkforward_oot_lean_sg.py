"""
walkforward_oot_lean_sg.py — GPU-optimized Walk-Forward CNN OOT Training
                              with Savitzky-Golay LOB smoothing experiment

V2+SG: Adds optional Savitzky-Golay smoothing of LOB features (depth, num_orders,
queue_age — features 1-3) after log1p transforms. Feature 0 (price_relative) is
intentionally NOT smoothed to preserve price microstructure.

SG smoothing applies along the time axis (axis=0) using 'nearest' boundary mode.
Disabled by default (--sg-window 0).

Usage:
    # Baseline (no SG)
    python walkforward_oot_lean_sg.py --device cuda --epochs 3 --subsample-train 5 --end-fold 3

    # SG window=7, polyorder=2
    python walkforward_oot_lean_sg.py --device cuda --epochs 3 --subsample-train 5 --sg-window 7 --sg-polyorder 2 --end-fold 3

    # SG window=11, polyorder=2
    python walkforward_oot_lean_sg.py --device cuda --epochs 3 --subsample-train 5 --sg-window 11 --sg-polyorder 2 --end-fold 3

    # SG window=11, polyorder=3
    python walkforward_oot_lean_sg.py --device cuda --epochs 3 --subsample-train 5 --sg-window 11 --sg-polyorder 3 --end-fold 3
"""

import argparse
import gc
import json
import logging
import sys
import time
import psutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import savgol_filter

FILE_DIR = Path(__file__).parent.resolve()
ROOT = FILE_DIR.parent.parent

sys.path.insert(0, str(FILE_DIR))
from book_spatial_cnn import BookSpatialCNN


def setup_logging(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("wf_lean_sg")
    logger.setLevel(logging.DEBUG)
    # Remove existing handlers to avoid duplicates on resume
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def compute_mfe_net(mid, horizon=100):
    n = len(mid)
    targets = np.full(n, np.nan, dtype=np.float32)
    if n <= horizon:
        return targets
    mid64 = mid.astype(np.float64)
    for i in range(n - horizon):
        future = mid64[i + 1: i + 1 + horizon]
        targets[i] = float(np.max(future) - mid64[i] - (mid64[i] - np.min(future)))
    return targets


def load_day_raw(npz_path, horizon=100, window_size=20, sg_window=0, sg_polyorder=2):
    """Load raw book tensors WITHOUT windowing. Returns ~70MB per day.

    If sg_window > 0, applies Savitzky-Golay smoothing to features 1-3
    (depth, num_orders, queue_age) along the time axis after log1p transforms.
    Feature 0 (price_relative) is intentionally NOT smoothed.
    """
    try:
        npz = np.load(npz_path, allow_pickle=False)
        bt_raw = npz["book_tensors"].astype(np.float32)
        mid = npz["mid_prices"].astype(np.float64)
        npz.close()
        n = len(mid)
        if n < window_size + horizon + 10:
            return None
        bt_raw[:, :, 1] = np.log1p(np.abs(bt_raw[:, :, 1])) * np.sign(bt_raw[:, :, 1])
        bt_raw[:, :, 2] = np.log1p(np.abs(bt_raw[:, :, 2])) * np.sign(bt_raw[:, :, 2])
        bt_raw[:, :, 3] = np.log1p(np.abs(bt_raw[:, :, 3])) * np.sign(bt_raw[:, :, 3])

        # Savitzky-Golay smoothing on features 1-3 (time axis)
        # bt_raw shape: (N, num_levels=20, num_features=4)
        # axis=0 smooths along time dimension N
        if sg_window > 0:
            # window_length must be odd and > polyorder
            wl = sg_window if sg_window % 2 == 1 else sg_window + 1
            if wl > sg_polyorder and n >= wl:
                bt_raw[:, :, 1:] = savgol_filter(
                    bt_raw[:, :, 1:],
                    window_length=wl,
                    polyorder=sg_polyorder,
                    axis=0,
                    mode='nearest'
                )

        targets = compute_mfe_net(mid, horizon)
        date_str = npz_path.name.split("_book_tensors")[0]
        return {"date": date_str, "bt": bt_raw, "targets": targets, "mid": mid, "n": n}
    except Exception:
        return None


def train_one_epoch_gpu(model, days, optimizer, criterion, device, use_amp, scaler,
                        logger, epoch, window_size=20, subsample=5, batch_size=1024):
    """GPU-native training: unfold on GPU, no DataLoader overhead."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_samples = 0

    # Shuffle day order each epoch
    day_indices = np.random.permutation(len(days))

    for di_idx, di in enumerate(day_indices):
        day = days[di]
        bt = day["bt"]  # (N, 20, 4)
        targets = day["targets"]  # (N,)
        n = day["n"]

        if n < window_size + 10:
            continue

        # Upload to GPU and unfold: (N, 20, 4) -> windows (N-ws+1, ws, 20, 4)
        bt_gpu = torch.from_numpy(bt).to(device)  # (N, 20, 4)
        # unfold along dim 0: (N-ws+1, 20, 4, ws) then permute
        windows = bt_gpu.unfold(0, window_size, 1)  # (N-ws+1, 20, 4, ws)
        windows = windows.permute(0, 3, 1, 2).contiguous()  # (N-ws+1, ws, 20, 4)

        # Targets for each window (aligned to window end)
        tgt = torch.from_numpy(targets[window_size - 1:]).to(device)  # (N-ws+1,)

        # Valid mask (finite targets)
        valid = torch.isfinite(tgt)
        windows = windows[valid]
        tgt = tgt[valid]

        if len(windows) < 10:
            del bt_gpu, windows, tgt
            continue

        # Subsample
        if subsample > 1:
            indices = torch.arange(0, len(windows), subsample, device=device)
            windows = windows[indices]
            tgt = tgt[indices]

        # Shuffle within day
        perm = torch.randperm(len(windows), device=device)
        windows = windows[perm]
        tgt = tgt[perm]

        total_samples += len(windows)

        # Mini-batch training
        for bstart in range(0, len(windows), batch_size):
            bend = min(bstart + batch_size, len(windows))
            xb = windows[bstart:bend]
            yb = tgt[bstart:bend]

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
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

        del bt_gpu, windows, tgt
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (di_idx + 1) % 20 == 0:
            logger.debug(f"  E{epoch} day {di_idx+1}/{len(days)} "
                         f"batches={n_batches} loss={total_loss/max(n_batches,1):.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    logger.debug(f"  E{epoch} total: {total_samples:,} samples, {n_batches} batches")
    return avg_loss


def compute_ic(preds, targets):
    mask = np.isfinite(preds) & np.isfinite(targets)
    if mask.sum() < 10:
        return float("nan")
    p, t = preds[mask], targets[mask]
    if p.std() < 1e-8 or t.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def compute_fold_stats(preds, targets, mid, date_str, ws):
    """Compute comprehensive per-fold statistics."""
    valid = slice(ws - 1, None)
    p = preds[valid]
    t = targets[valid]
    m = mid[valid]

    mask = np.isfinite(p) & np.isfinite(t)
    p_valid, t_valid = p[mask], t[mask]

    stats = {
        "date": date_str,
        "n_bars": int(len(mid)),
        "n_valid": int(mask.sum()),
        "ic": float(np.corrcoef(p_valid, t_valid)[0, 1]) if len(p_valid) > 10 else float("nan"),
        "pred_mean": float(p_valid.mean()),
        "pred_std": float(p_valid.std()),
        "pred_min": float(p_valid.min()),
        "pred_max": float(p_valid.max()),
        "target_mean": float(t_valid.mean()),
        "target_std": float(t_valid.std()),
        "mid_open": float(m[0]),
        "mid_close": float(m[-1]),
        "mid_range_ticks": float((m.max() - m.min()) / 0.25),
    }

    # Time-of-day IC breakdown
    n_valid_bars = len(p)
    if n_valid_bars > 100:
        half = n_valid_bars // 2
        p_am, t_am = p[:half], t[:half]
        p_pm, t_pm = p[half:], t[half:]
        mask_am = np.isfinite(p_am) & np.isfinite(t_am)
        mask_pm = np.isfinite(p_pm) & np.isfinite(t_pm)
        stats["ic_morning"] = float(np.corrcoef(p_am[mask_am], t_am[mask_am])[0, 1]) if mask_am.sum() > 10 else float("nan")
        stats["ic_afternoon"] = float(np.corrcoef(p_pm[mask_pm], t_pm[mask_pm])[0, 1]) if mask_pm.sum() > 10 else float("nan")

        q1, q2, q3 = n_valid_bars // 4, n_valid_bars // 2, 3 * n_valid_bars // 4
        for label, s, e in [("q1", 0, q1), ("q2", q1, q2), ("q3", q2, q3), ("q4", q3, n_valid_bars)]:
            pq, tq = p[s:e], t[s:e]
            mq = np.isfinite(pq) & np.isfinite(tq)
            stats[f"ic_{label}"] = float(np.corrcoef(pq[mq], tq[mq])[0, 1]) if mq.sum() > 10 else float("nan")

    try:
        from datetime import datetime as dt
        d = dt.strptime(date_str, "%Y-%m-%d")
        stats["day_of_week"] = d.strftime("%A")
        stats["month"] = d.strftime("%Y-%m")
        stats["week_number"] = d.isocalendar()[1]
    except Exception:
        pass

    mid_returns = np.diff(m) / 0.25
    stats["realized_vol_ticks"] = float(np.std(mid_returns))
    stats["abs_return_ticks"] = float(np.abs(m[-1] - m[0]) / 0.25)

    return stats


def main():
    parser = argparse.ArgumentParser(description="GPU-optimized Walk-Forward CNN OOT with SG smoothing")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--subsample-train", type=int, default=5)
    parser.add_argument("--is-dir", type=Path, default=ROOT / "data/processed/dl_book_cache")
    parser.add_argument("--oot-dir", type=Path, default=ROOT / "data/processed/dl_book_cache_oot")
    parser.add_argument("--output-dir", type=Path, default=FILE_DIR / "results")
    parser.add_argument("--checkpoint-dir", type=Path, default=FILE_DIR / "checkpoints")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--resume", action="store_true", help="Resume from last completed fold")
    parser.add_argument("--start-fold", type=int, default=0, help="Starting fold index (0-based)")
    parser.add_argument("--end-fold", type=int, default=None, help="Ending fold index (exclusive)")
    parser.add_argument("--warm-start", action="store_true",
                        help="Load previous fold's weights instead of training from scratch")
    # Savitzky-Golay smoothing args
    parser.add_argument("--sg-window", type=int, default=0,
                        help="Savitzky-Golay window length (0=disabled, must be odd or will be rounded up)")
    parser.add_argument("--sg-polyorder", type=int, default=2,
                        help="Savitzky-Golay polynomial order (default 2)")
    args = parser.parse_args()

    # Determine SG config label for output naming
    if args.sg_window > 0:
        wl = args.sg_window if args.sg_window % 2 == 1 else args.sg_window + 1
        sg_label = f"sg{wl}p{args.sg_polyorder}"
    else:
        sg_label = "sg_off"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use SG-specific output/checkpoint dirs to avoid collisions with baseline
    sg_output_dir = args.output_dir / sg_label
    sg_checkpoint_dir = args.checkpoint_dir / sg_label
    sg_output_dir.mkdir(parents=True, exist_ok=True)
    sg_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state_file = sg_output_dir / "walkforward_state.json"
    incr_path = sg_output_dir / "oot_wf_predictions_incremental.npz"
    stats_file = sg_output_dir / "walkforward_fold_stats.json"

    completed_folds = set()
    all_fold_stats = []
    if args.resume and state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        completed_folds = set(state.get("completed_dates", []))
        all_fold_stats = state.get("fold_stats", [])

    log_path = sg_output_dir / f"walkforward_oot_lean_sg_{sg_label}_{timestamp}.log"
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("GPU-Optimized Walk-Forward CNN OOT Training (V2 + SG Smoothing)")
    logger.info(f"IS dir      : {args.is_dir}")
    logger.info(f"OOT dir     : {args.oot_dir}")
    logger.info(f"Epochs/fold : {args.epochs}")
    logger.info(f"Subsample   : {args.subsample_train}")
    logger.info(f"Batch size  : {args.batch_size}")
    logger.info(f"SG config   : window={args.sg_window} polyorder={args.sg_polyorder} "
                f"(label={sg_label})")
    if args.sg_window > 0:
        logger.info(f"  Smoothing features 1-3 (depth, num_orders, queue_age) along time axis")
        logger.info(f"  Feature 0 (price_relative) NOT smoothed")
    else:
        logger.info(f"  SG smoothing DISABLED (baseline mode)")
    logger.info(f"RAM free    : {psutil.virtual_memory().available/1e9:.1f}GB")
    if completed_folds:
        logger.info(f"RESUMING    : {len(completed_folds)} folds already done")
    logger.info("=" * 70)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    else:
        device = torch.device("cpu")
        use_amp = False
        logger.info(f"Using CPU ({torch.get_num_threads()} threads)")

    is_files = sorted(args.is_dir.glob("*_book_tensors.npz"))
    oot_files = sorted(args.oot_dir.glob("*_book_tensors.npz"))
    logger.info(f"IS files : {len(is_files)} days")
    logger.info(f"OOT files: {len(oot_files)} days")

    if not is_files or not oot_files:
        logger.error("Missing IS or OOT files!")
        sys.exit(1)

    logger.info("Loading IS data (raw, no windowing)...")
    t0 = time.time()
    is_days = []
    for fi, f in enumerate(is_files):
        day = load_day_raw(f, horizon=args.horizon, window_size=args.window_size,
                           sg_window=args.sg_window, sg_polyorder=args.sg_polyorder)
        if day is None:
            continue
        is_days.append(day)
        if (fi + 1) % 20 == 0:
            ram = psutil.virtual_memory().available / 1e9
            logger.info(f"  Loaded {fi+1}/{len(is_files)} IS days, RAM: {ram:.1f}GB free")
    logger.info(f"IS loaded: {len(is_days)} days in {time.time()-t0:.1f}s, "
                f"RAM: {psutil.virtual_memory().available/1e9:.1f}GB free")

    logger.info("Loading OOT data (raw, no windowing)...")
    t0 = time.time()
    oot_days = []
    for fi, f in enumerate(oot_files):
        day = load_day_raw(f, horizon=args.horizon, window_size=args.window_size,
                           sg_window=args.sg_window, sg_polyorder=args.sg_polyorder)
        if day is None:
            continue
        oot_days.append(day)
        if (fi + 1) % 20 == 0:
            ram = psutil.virtual_memory().available / 1e9
            logger.info(f"  Loaded {fi+1}/{len(oot_files)} OOT days, RAM: {ram:.1f}GB free")
    logger.info(f"OOT loaded: {len(oot_days)} days in {time.time()-t0:.1f}s, "
                f"RAM: {psutil.virtual_memory().available/1e9:.1f}GB free")

    if not oot_days:
        logger.error("No valid OOT days!")
        sys.exit(1)

    criterion = nn.HuberLoss(delta=1.0)

    all_predictions = {}
    if args.resume and incr_path.exists():
        existing = np.load(str(incr_path), allow_pickle=True)
        for k in existing.files:
            all_predictions[k] = existing[k]
        existing.close()
        logger.info(f"Loaded {len(all_predictions)//2} existing prediction days")

    fold_ics = []
    session_start = time.time()

    # Fold range for parallel execution across machines
    start_fold = args.start_fold
    end_fold = args.end_fold if args.end_fold is not None else len(oot_days)
    end_fold = min(end_fold, len(oot_days))

    logger.info("=" * 70)
    logger.info(f"Starting Walk-Forward: folds {start_fold+1}-{end_fold}/{len(oot_days)} "
                f"({'warm-start' if args.warm_start else 'fresh model'}, GPU-native unfold)")
    logger.info("=" * 70)

    prev_ckpt_path = None  # For warm-start chain

    for fold_idx, oot_day in enumerate(oot_days):
        # Skip folds outside our assigned range
        if fold_idx < start_fold or fold_idx >= end_fold:
            # Still track prev checkpoint for warm-start
            ckpt_candidate = sg_checkpoint_dir / f"cnn_wf_fold{fold_idx+1}_{oot_day['date']}.pt"
            if ckpt_candidate.exists():
                prev_ckpt_path = ckpt_candidate
            continue

        oot_date = oot_day["date"]

        if oot_date in completed_folds:
            for s in all_fold_stats:
                if s["date"] == oot_date:
                    fold_ics.append(s["ic"])
                    break
            # Track checkpoint for warm-start chain
            ckpt_candidate = sg_checkpoint_dir / f"cnn_wf_fold{fold_idx+1}_{oot_date}.pt"
            if ckpt_candidate.exists():
                prev_ckpt_path = ckpt_candidate
            logger.info(f"Fold {fold_idx+1}/{len(oot_days)} — {oot_date} [SKIPPED, already done]")
            continue

        fold_start = time.time()

        # Training: IS + OOT[0..fold_idx-2] (1-day purge gap)
        oot_train = oot_days[:max(0, fold_idx - 1)]
        train_days = is_days + oot_train

        logger.info(f"\nFold {fold_idx+1}/{len(oot_days)} — {oot_date}")
        logger.info(f"  Train: {len(is_days)} IS + {len(oot_train)} OOT = {len(train_days)} days")

        if len(train_days) < 5:
            continue

        # Model setup: warm-start from previous fold or fresh
        model = BookSpatialCNN(
            window_size=args.window_size, num_levels=20, num_features=4,
            spatial_channels=(32, 64, 128, 256), temporal_channels=256,
            dropout=0.1, num_classes=1,
        ).to(device)

        if args.warm_start and prev_ckpt_path and prev_ckpt_path.exists():
            state = torch.load(prev_ckpt_path, map_location=device, weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
            logger.info(f"  Warm-started from {prev_ckpt_path.name}")
            # Lower LR for fine-tuning on warm-start
            lr = 1e-4
        else:
            lr = 3e-4

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        # Train with GPU-native unfold
        epoch_losses = []
        for ep in range(1, args.epochs + 1):
            ep_start = time.time()
            loss = train_one_epoch_gpu(
                model, train_days, optimizer, criterion, device, use_amp, scaler,
                logger, ep, window_size=args.window_size,
                subsample=args.subsample_train, batch_size=args.batch_size
            )
            epoch_losses.append(loss)
            ep_time = time.time() - ep_start
            logger.info(f"  Epoch {ep}/{args.epochs} loss={loss:.4f} ({ep_time:.1f}s)")

        # Save checkpoint with full metadata
        ckpt_path = sg_checkpoint_dir / f"cnn_wf_fold{fold_idx+1}_{oot_date}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "fold_idx": fold_idx,
            "date": oot_date,
            "train_days_count": len(train_days),
            "is_days_count": len(is_days),
            "oot_days_used": len(oot_train),
            "epochs": args.epochs,
            "epoch_losses": epoch_losses,
            "warm_started": args.warm_start and prev_ckpt_path is not None,
            "timestamp": datetime.now().isoformat(),
            "sg_window": args.sg_window,
            "sg_polyorder": args.sg_polyorder,
            "sg_label": sg_label,
        }, ckpt_path)
        prev_ckpt_path = ckpt_path  # Chain for next fold's warm-start
        logger.info(f"  Checkpoint saved: {ckpt_path.name}")

        # Inference on OOT day (also GPU-native unfold)
        model.eval()
        bt = oot_day["bt"]
        n = oot_day["n"]
        ws = args.window_size
        preds = np.zeros(n, dtype=np.float32)

        with torch.no_grad():
            # Process in chunks to avoid VRAM overflow on large days
            chunk_size = 5000  # bars per chunk
            for cstart in range(0, n, chunk_size):
                cend = min(cstart + chunk_size, n)
                # Need ws-1 extra bars before chunk for windowing
                slice_start = max(0, cstart - ws + 1)
                bt_chunk = torch.from_numpy(bt[slice_start:cend]).to(device)

                if len(bt_chunk) < ws:
                    continue

                # Unfold: (M, 20, 4) -> (M-ws+1, 20, 4, ws) -> (M-ws+1, ws, 20, 4)
                windows = bt_chunk.unfold(0, ws, 1).permute(0, 3, 1, 2).contiguous()

                # Predict in mini-batches
                for bstart in range(0, len(windows), args.batch_size):
                    bend = min(bstart + args.batch_size, len(windows))
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            out = model(windows[bstart:bend]).squeeze(-1)
                    else:
                        out = model(windows[bstart:bend]).squeeze(-1)

                    # Map back to global indices
                    g_start = slice_start + ws - 1 + bstart
                    g_end = slice_start + ws - 1 + bend
                    preds[g_start:g_end] = out.cpu().float().numpy()

                del bt_chunk, windows
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # Compute comprehensive stats
        targets_oot = oot_day["targets"]
        fold_stats = compute_fold_stats(preds, targets_oot, oot_day["mid"], oot_date, ws)
        fold_stats["fold_idx"] = fold_idx
        fold_stats["train_days"] = len(train_days)
        fold_stats["epoch_losses"] = epoch_losses
        fold_stats["fold_time_seconds"] = time.time() - fold_start
        fold_stats["checkpoint"] = ckpt_path.name
        fold_stats["sg_window"] = args.sg_window
        fold_stats["sg_polyorder"] = args.sg_polyorder
        fold_stats["sg_label"] = sg_label

        ic = fold_stats["ic"]
        fold_ics.append(ic)
        all_fold_stats.append(fold_stats)

        fold_time = time.time() - fold_start
        ram = psutil.virtual_memory().available / 1e9
        logger.info(f"  IC={ic:.4f} | {fold_time:.1f}s | RAM: {ram:.1f}GB free")
        if "ic_morning" in fold_stats:
            logger.info(f"  IC morning={fold_stats['ic_morning']:.4f} afternoon={fold_stats['ic_afternoon']:.4f}")
        logger.info(f"  Pred range=[{fold_stats['pred_min']:.4f}, {fold_stats['pred_max']:.4f}] "
                     f"RVol={fold_stats['realized_vol_ticks']:.2f}t")

        # Elapsed time estimate
        elapsed = time.time() - session_start
        done = len(fold_ics)
        remaining = len(oot_days) - fold_idx - 1
        if done > 0:
            eta_min = (elapsed / done) * remaining / 60
            logger.info(f"  Progress: {done}/{len(oot_days)} folds, ETA: {eta_min:.0f} min")

        # Save predictions incrementally
        all_predictions[f"{oot_date}_preds"] = preds
        all_predictions[f"{oot_date}_mid"] = oot_day["mid"]
        np.savez_compressed(str(incr_path), **all_predictions)

        # Save state (crash-resilient)
        completed_folds.add(oot_date)
        state = {
            "completed_dates": list(completed_folds),
            "fold_stats": all_fold_stats,
            "last_updated": datetime.now().isoformat(),
            "config": {
                "epochs": args.epochs,
                "subsample": args.subsample_train,
                "batch_size": args.batch_size,
                "horizon": args.horizon,
                "window_size": args.window_size,
                "sg_window": args.sg_window,
                "sg_polyorder": args.sg_polyorder,
                "sg_label": sg_label,
            }
        }
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)
        with open(stats_file, "w") as f:
            json.dump(all_fold_stats, f, indent=2, default=str)

        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # ==================== Summary ====================
    logger.info("\n" + "=" * 70)
    logger.info(f"Walk-Forward OOT Complete [{sg_label}]")
    logger.info(f"Total time: {(time.time()-session_start)/60:.1f} min")
    logger.info(f"Folds: {len(fold_ics)}")

    if fold_ics:
        arr = np.array([x for x in fold_ics if np.isfinite(x)])
        logger.info(f"Mean IC:      {arr.mean():.4f}")
        logger.info(f"Std IC:       {arr.std():.4f}")
        logger.info(f"Pct positive: {(arr > 0).mean()*100:.1f}%")
        logger.info(f"Range:        [{arr.min():.4f}, {arr.max():.4f}]")
        if arr.std() > 1e-8:
            logger.info(f"ICIR:         {arr.mean()/arr.std():.3f}")

        # Monthly IC breakdown
        monthly = {}
        for s in all_fold_stats:
            m = s.get("month", "unknown")
            if np.isfinite(s["ic"]):
                monthly.setdefault(m, []).append(s["ic"])
        logger.info("\nMonthly IC Breakdown:")
        for m in sorted(monthly):
            vals = np.array(monthly[m])
            logger.info(f"  {m}: mean={vals.mean():.4f} std={vals.std():.4f} n={len(vals)}")

        # Day-of-week IC breakdown
        dow = {}
        for s in all_fold_stats:
            d = s.get("day_of_week", "unknown")
            if np.isfinite(s["ic"]):
                dow.setdefault(d, []).append(s["ic"])
        logger.info("\nDay-of-Week IC Breakdown:")
        for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            if d in dow:
                vals = np.array(dow[d])
                logger.info(f"  {d}: mean={vals.mean():.4f} n={len(vals)}")

        # Morning vs Afternoon
        am_ics = [s.get("ic_morning", float("nan")) for s in all_fold_stats]
        pm_ics = [s.get("ic_afternoon", float("nan")) for s in all_fold_stats]
        am_arr = np.array([x for x in am_ics if np.isfinite(x)])
        pm_arr = np.array([x for x in pm_ics if np.isfinite(x)])
        if len(am_arr) > 0:
            logger.info(f"\nMorning IC:   {am_arr.mean():.4f} (n={len(am_arr)})")
        if len(pm_arr) > 0:
            logger.info(f"Afternoon IC: {pm_arr.mean():.4f} (n={len(pm_arr)})")

        # Vol regime breakdown
        vol_ics = {}
        for s in all_fold_stats:
            rv = s.get("realized_vol_ticks", 0)
            if rv < 0.5:
                regime = "low_vol"
            elif rv < 1.0:
                regime = "med_vol"
            else:
                regime = "high_vol"
            if np.isfinite(s["ic"]):
                vol_ics.setdefault(regime, []).append(s["ic"])
        logger.info("\nVol Regime IC Breakdown:")
        for regime in ["low_vol", "med_vol", "high_vol"]:
            if regime in vol_ics:
                vals = np.array(vol_ics[regime])
                logger.info(f"  {regime}: mean={vals.mean():.4f} n={len(vals)}")

    # Save final predictions
    if all_predictions:
        pred_path = sg_output_dir / f"oos_predictions_book_oot_wf_{sg_label}_{timestamp}.npz"
        np.savez_compressed(str(pred_path), **all_predictions)
        logger.info(f"\nPredictions saved: {pred_path}")

    # Save final comprehensive stats
    arr_for_summary = np.array([x for x in fold_ics if np.isfinite(x)]) if fold_ics else np.array([])
    final_stats = {
        "summary": {
            "total_folds": len(fold_ics),
            "mean_ic": float(arr_for_summary.mean()) if len(arr_for_summary) > 0 else None,
            "std_ic": float(arr_for_summary.std()) if len(arr_for_summary) > 0 else None,
            "icir": float(arr_for_summary.mean() / arr_for_summary.std()) if len(arr_for_summary) > 0 and arr_for_summary.std() > 1e-8 else None,
            "pct_positive": float((arr_for_summary > 0).mean() * 100) if len(arr_for_summary) > 0 else None,
            "total_time_minutes": (time.time() - session_start) / 60,
            "device": str(device),
            "timestamp": timestamp,
        },
        "config": {
            "epochs": args.epochs,
            "subsample_train": args.subsample_train,
            "batch_size": args.batch_size,
            "horizon": args.horizon,
            "window_size": args.window_size,
            "is_days": len(is_days),
            "oot_days": len(oot_days),
            "sg_window": args.sg_window,
            "sg_polyorder": args.sg_polyorder,
            "sg_label": sg_label,
        },
        "fold_details": all_fold_stats,
    }
    final_stats_path = sg_output_dir / f"walkforward_stats_{sg_label}_{timestamp}.json"
    with open(final_stats_path, "w") as f:
        json.dump(final_stats, f, indent=2, default=str)
    logger.info(f"Stats saved: {final_stats_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
