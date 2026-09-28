#!/usr/bin/env python3
"""
Event Mamba + Book30 Features - Memory-Efficient Version

Key fix: Streaming data loader (doesn't load all events into RAM)

Features: 15 event + 30 book = 45 total
Post-Jan 2026 data (Feb onwards)

Usage:
  python train_mamba_book_v2.py --d-model 64 --n-folds 3 --window 1000
"""

import os, sys, time, logging, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.stats
import socket

try:
    import mlflow
    MLFLOW_AVAILABLE = False
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path("/home/nick/Lvl3Quant")
EVENT_DIR = ROOT / "data" / "processed" / "mbo_events_feat"
BOOK_DIR = ROOT / "data" / "processed" / "mbo_book_features"
OUT_DIR = ROOT / "alpha_discovery" / "results" / "mamba_book"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = "EventDriven_Mamba_BookFusion"


class StreamingFusionDataset(Dataset):
    """Memory-efficient: loads files on-demand, creates windows on-the-fly."""

    def __init__(self, date_list, window_size=1000, stride=500, horizon="10s"):
        self.date_list = date_list
        self.window_size = window_size
        self.stride = stride
        self.horizon_key = f"labels_{horizon}"

        # Pre-compute file offsets for windowing
        self.file_windows = []
        for date_str in date_list:
            event_file = EVENT_DIR / f"{date_str}_mbo_events.npz"
            book_file = BOOK_DIR / f"{date_str}_book_features.npz"

            if not event_file.exists() or not book_file.exists():
                logger.warning(f"Missing {date_str}, skipping")
                continue

            # Just get length without loading full data
            with np.load(event_file, mmap_mode='r') as e:
                n_events = len(e['events'])

            # Create window indices for this file
            for start_idx in range(0, n_events - window_size, stride):
                self.file_windows.append((date_str, start_idx))

        logger.info(f"Dataset: {len(date_list)} files, {len(self.file_windows)} windows")

    def __len__(self):
        return len(self.file_windows)

    def __getitem__(self, idx):
        date_str, start_idx = self.file_windows[idx]
        end_idx = start_idx + self.window_size

        # Load only the needed window (memory-mapped)
        event_file = EVENT_DIR / f"{date_str}_mbo_events.npz"
        book_file = BOOK_DIR / f"{date_str}_book_features.npz"

        e = np.load(event_file, mmap_mode='r')
        b = np.load(book_file, mmap_mode='r')

        event_feat = e['events'][start_idx:end_idx].astype(np.float32)  # (W, 15)
        book_feat = b['features'][start_idx:end_idx].astype(np.float32)  # (W, 30)
        labels = b[self.horizon_key][start_idx:end_idx].astype(np.float32)  # (W,)

        # Concatenate
        fused = np.concatenate([event_feat, book_feat], axis=1)  # (W, 45)

        return torch.from_numpy(fused).float(), torch.from_numpy(labels).float()


class MambaBlock(nn.Module):
    """Simplified Mamba SSM block."""
    def __init__(self, d_model, d_state, d_conv, dt_rank, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=d_conv, padding=d_conv-1, groups=d_model)
        self.ssm_param = nn.Linear(d_model, d_state * 2 + dt_rank)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dt_proj = nn.Linear(dt_rank, d_model)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        B, L, D = x.shape
        x_proj = self.in_proj(x)
        x_main, x_gate = x_proj.chunk(2, dim=-1)
        x_conv = self.conv1d(x_main.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = nn.functional.silu(x_conv)
        y = x_conv * nn.functional.silu(x_gate)
        y = self.out_proj(y)
        return self.dropout(y) + x


class MambaModel(nn.Module):
    def __init__(self, n_features=45, d_model=64, d_state=64, n_layers=2, d_conv=4, dt_rank=16, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, dt_rank, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x).squeeze(-1)


def train_fold(model, train_loader, val_loader, device, epochs=3, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_ic = -999

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x.to(device))
                val_preds.append(pred.cpu().numpy())
                val_labels.append(y.numpy())

        val_preds = np.concatenate(val_preds).flatten()
        val_labels = np.concatenate(val_labels).flatten()
        ic = scipy.stats.spearmanr(val_preds, val_labels).correlation
        best_ic = max(best_ic, ic)

        logger.info(f"Epoch {epoch+1}/{epochs}: loss={train_loss/len(train_loader):.4f} ic={ic:.4f}")

    return best_ic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--d-state', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--window', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-folds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    # Get post-Jan 2026 files
    all_files = sorted([f.stem[:8] for f in BOOK_DIR.glob("2026*_book_features.npz")])
    post_jan = [f for f in all_files if f >= "20260201"]
    logger.info(f"Post-Jan 2026: {len(post_jan)} days ({post_jan[0]}..{post_jan[-1]})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow.start_run(run_name=f"Mamba_Book45_{time.strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params({
            "model": "MambaBook45",
            "n_features": 45,
            "d_model": args.d_model,
            "d_state": args.d_state,
            "n_layers": args.n_layers,
            "window_size": args.window,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_folds": args.n_folds,
            "epochs_per_fold": args.epochs,
            "node": socket.gethostname(),
            "n_files": len(post_jan),
        })

    # Walk-forward training
    fold_size = max(len(post_jan) // (args.n_folds + 1), 5)

    for fold in range(args.n_folds):
        logger.info(f"\n{'='*60}\nFOLD {fold+1}/{args.n_folds}\n{'='*60}")

        train_end = fold_size * (fold + 1)
        val_start = train_end
        val_end = min(val_start + fold_size, len(post_jan))

        if val_end >= len(post_jan):
            break

        train_dates = post_jan[:train_end]
        val_dates = post_jan[val_start:val_end]
        logger.info(f"Train: {train_dates[0]}..{train_dates[-1]} ({len(train_dates)}d)")
        logger.info(f"Val: {val_dates[0]}..{val_dates[-1]} ({len(val_dates)}d)")

        train_ds = StreamingFusionDataset(train_dates, window_size=args.window, stride=args.window//2)
        val_ds = StreamingFusionDataset(val_dates, window_size=args.window, stride=args.window//2)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        model = MambaModel(
            n_features=45,
            d_model=args.d_model,
            d_state=args.d_state,
            n_layers=args.n_layers,
            d_conv=4,
            dt_rank=16,
            dropout=0.1,
        ).to(device)

        logger.info(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

        ic = train_fold(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)

        if MLFLOW_AVAILABLE:
            mlflow.log_metric(f"fold{fold:02d}_ic_10s", ic)

        logger.info(f"Fold {fold+1} complete: IC={ic:.4f}")

    if MLFLOW_AVAILABLE:
        mlflow.end_run()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
