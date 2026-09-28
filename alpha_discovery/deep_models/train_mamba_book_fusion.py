#!/usr/bin/env python3
"""
Event Mamba + Book Features Fusion Training

Combines:
  - Event features (15-dim from mbo_events_feat)
  - Book features (30-dim from mbo_book_features)
  = 45 total features per event

Training on post-January 2026 data (Feb 1 onwards).

Architecture: Same Mamba SSM as train_event_mamba.py but with 45-dim input.

Usage:
  # Small model, 3 folds, increased context
  MAMBA_D_MODEL=64 MAMBA_D_STATE=64 MAMBA_N_LAYERS=2 \\
  EVENT_WINDOW_SIZE=1000 EVENT_N_FOLDS=3 \\
  python train_mamba_book_fusion.py
"""

import os, sys, time, logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.stats
import socket

# MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Paths
ROOT = Path("/home/jupiter/Lvl3Quant")
EVENT_DIR = ROOT / "data" / "processed" / "mbo_events_feat"
BOOK_DIR = ROOT / "data" / "processed" / "mbo_book_features"
OUT_DIR = ROOT / "alpha_discovery" / "results" / "mamba_book_fusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MLflow
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = "EventDriven_Mamba_BookFusion"

# Hyperparameters
D_MODEL = int(os.environ.get("MAMBA_D_MODEL", 64))
D_STATE = int(os.environ.get("MAMBA_D_STATE", 64))
N_LAYERS = int(os.environ.get("MAMBA_N_LAYERS", 2))
DROPOUT = float(os.environ.get("MAMBA_DROPOUT", 0.1))
DT_RANK = int(os.environ.get("MAMBA_DT_RANK", 16))
D_CONV = int(os.environ.get("MAMBA_D_CONV", 4))

WINDOW_SIZE = int(os.environ.get("EVENT_WINDOW_SIZE", 1000))
STRIDE = int(os.environ.get("EVENT_STRIDE", WINDOW_SIZE // 2))
BATCH_SIZE = int(os.environ.get("EVENT_BATCH_SIZE", 32))
LR = float(os.environ.get("EVENT_LR", 3e-4))
EPOCHS_PER_FOLD = int(os.environ.get("EVENT_EPOCHS", 3))
N_FOLDS = int(os.environ.get("EVENT_N_FOLDS", 3))
WARMUP_STEPS = int(os.environ.get("EVENT_WARMUP", 300))
GRAD_CLIP = float(os.environ.get("EVENT_GRAD_CLIP", 1.0))

N_EVENT_FEATURES = 15
N_BOOK_FEATURES = 30
N_TOTAL_FEATURES = N_EVENT_FEATURES + N_BOOK_FEATURES  # 45
HORIZONS = ["1s", "5s", "10s"]


class FusionDataset(Dataset):
    """Loads aligned event + book features and creates windowed samples."""

    def __init__(self, date_list, window_size=1000, stride=500, horizon="10s"):
        self.window_size = window_size
        self.stride = stride
        self.horizon_key = f"labels_{horizon}"

        all_features = []
        all_labels = []

        for date_str in date_list:
            event_file = EVENT_DIR / f"{date_str}_mbo_events.npz"
            book_file = BOOK_DIR / f"{date_str}_book_features.npz"

            if not event_file.exists() or not book_file.exists():
                logger.warning(f"Missing file for {date_str}, skipping")
                continue

            e = np.load(event_file, allow_pickle=True)
            b = np.load(book_file, allow_pickle=True)

            event_feat = e["events"].astype(np.float32)  # (N, 15)
            book_feat = b["features"].astype(np.float32)  # (N, 30)
            labels = b[self.horizon_key].astype(np.float32)  # (N,)

            # Verify alignment
            assert len(event_feat) == len(book_feat) == len(labels), \
                f"Length mismatch for {date_str}: {len(event_feat)} vs {len(book_feat)} vs {len(labels)}"

            # Concatenate features
            fused = np.concatenate([event_feat, book_feat], axis=1)  # (N, 45)

            all_features.append(fused)
            all_labels.append(labels)

        self.features = np.concatenate(all_features, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)

        # Create windows
        self.windows = []
        for i in range(0, len(self.features) - window_size, stride):
            self.windows.append(i)

        logger.info(f"Dataset: {len(self.features)} events, {len(self.windows)} windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start = self.windows[idx]
        end = start + self.window_size

        x = torch.from_numpy(self.features[start:end]).float()  # (W, 45)
        y = torch.from_numpy(self.labels[start:end]).float()    # (W,)

        return x, y


class MambaBlock(nn.Module):
    """Simplified Mamba SSM block."""

    def __init__(self, d_model, d_state, d_conv, dt_rank, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=d_conv, padding=d_conv-1, groups=d_model)
        self.ssm_param = nn.Linear(d_model, d_state * 3 + dt_rank)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.dt_proj = nn.Linear(dt_rank, d_model)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape

        x_proj = self.in_proj(x)  # (B, L, 2D)
        x_main, x_gate = x_proj.chunk(2, dim=-1)  # (B, L, D) each

        # Conv1d expects (B, D, L)
        x_conv = self.conv1d(x_main.transpose(1, 2))[:, :, :L].transpose(1, 2)  # (B, L, D)
        x_conv = nn.functional.silu(x_conv)

        # SSM parameters
        ssm_out = self.ssm_param(x_conv)  # (B, L, 3*d_state + dt_rank)
        B_ssm, C_ssm, dt_ssm = ssm_out.split([self.d_state, self.d_state, ssm_out.shape[-1] - 2*self.d_state], dim=-1)
        dt = nn.functional.softplus(self.dt_proj(dt_ssm))  # (B, L, D)

        # Simplified SSM computation (state-space scan)
        A = -torch.exp(self.A_log)  # (D, N)

        # Discretize: A_bar = exp(A * dt)
        # Simple approximation for efficiency
        y = x_conv  # Simplified - in full Mamba would do selective scan

        # Gating
        y = y * nn.functional.silu(x_gate)

        y = self.out_proj(y)
        y = self.dropout(y)

        return y + x  # Residual


class EventMambaBookFusion(nn.Module):
    """Mamba model with 45-dim input (event + book features)."""

    def __init__(self, n_features=45, d_model=64, d_state=64, n_layers=2, d_conv=4, dt_rank=16, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, dt_rank, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, L, 45)
        x = self.input_proj(x)  # (B, L, D)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        out = self.head(x).squeeze(-1)  # (B, L)

        return out


def train_fold(model, train_loader, val_loader, device, epochs=3):
    """Train one fold."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                pred = model(x)
                val_preds.append(pred.cpu().numpy())
                val_labels.append(y.numpy())

        val_preds = np.concatenate(val_preds, axis=0).flatten()
        val_labels = np.concatenate(val_labels, axis=0).flatten()

        ic = scipy.stats.spearmanr(val_preds, val_labels).correlation

        logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss/len(train_loader):.4f} val_ic={ic:.4f}")

    return ic


def main():
    # Get post-Jan 2026 files (Feb onwards)
    all_files = sorted([f.stem[:8] for f in BOOK_DIR.glob("2026*_book_features.npz")])
    post_jan_files = [f for f in all_files if f >= "20260201"]

    logger.info(f"Found {len(post_jan_files)} post-Jan 2026 files: {post_jan_files[0]}..{post_jan_files[-1]}")

    if len(post_jan_files) < 10:
        logger.error("Not enough data for training")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        run_name = f"Mamba_Book45_{time.strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)

        mlflow.log_params({
            "model": "MambaBookFusion",
            "n_features": N_TOTAL_FEATURES,
            "d_model": D_MODEL,
            "d_state": D_STATE,
            "n_layers": N_LAYERS,
            "window_size": WINDOW_SIZE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs_per_fold": EPOCHS_PER_FOLD,
            "n_folds": N_FOLDS,
            "n_files": len(post_jan_files),
            "node": socket.gethostname(),
        })

    # Walk-forward training
    fold_size = max(len(post_jan_files) // (N_FOLDS + 1), 5)

    for fold in range(N_FOLDS):
        logger.info(f"\n{'='*60}\nFOLD {fold+1}/{N_FOLDS}\n{'='*60}")

        # Expanding window
        train_end = fold_size * (fold + 1)
        val_start = train_end
        val_end = min(val_start + fold_size, len(post_jan_files))

        if val_end >= len(post_jan_files):
            break

        train_dates = post_jan_files[:train_end]
        val_dates = post_jan_files[val_start:val_end]

        logger.info(f"Train: {train_dates[0]}..{train_dates[-1]} ({len(train_dates)} days)")
        logger.info(f"Val: {val_dates[0]}..{val_dates[-1]} ({len(val_dates)} days)")

        train_ds = FusionDataset(train_dates, window_size=WINDOW_SIZE, stride=STRIDE, horizon="10s")
        val_ds = FusionDataset(val_dates, window_size=WINDOW_SIZE, stride=STRIDE, horizon="10s")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        model = EventMambaBookFusion(
            n_features=N_TOTAL_FEATURES,
            d_model=D_MODEL,
            d_state=D_STATE,
            n_layers=N_LAYERS,
            d_conv=D_CONV,
            dt_rank=DT_RANK,
            dropout=DROPOUT,
        ).to(device)

        logger.info(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        ic = train_fold(model, train_loader, val_loader, device, epochs=EPOCHS_PER_FOLD)

        if MLFLOW_AVAILABLE:
            mlflow.log_metric(f"fold{fold:02d}_val_ic_10s", ic)

        logger.info(f"Fold {fold+1} complete: IC_10s={ic:.4f}")

    if MLFLOW_AVAILABLE:
        mlflow.end_run()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
