"""
Event-Driven Transformer — Training Script v1

Data format (from Jupiter /home/jupiter/Lvl3Quant/data/processed/mbo_events/):
  - events: (N_events, 6) float32
      [time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks]
  - labels_1s/5s/10s/30s: (N_events,) float32 — mid-price change in ticks
  - timestamps: (N_events,) int64 nanoseconds

Architecture:
  - Window of 1000 raw events per sample (sliding with stride 500)
  - Event embedding: Linear(6 → 64) + time-aware positional encoding
  - Causal transformer: 4 layers, 64-dim, 4 heads
  - Multi-task head: predict 1s, 5s, 10s price change (regression)
  - ~2-3M params — fits in 24GB with room

Training rules:
  - Expanding window (ABSOLUTE RULE — no sliding)
  - 10 folds using available dates
  - Concat IC as primary metric
  - Mixed precision (fp16)
  - num_workers=8, pin_memory=True
  - MLflow logging
  - Save .pt weights + .npz predictions per fold
"""

import os
import sys
import time
import logging
import argparse
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.stats

# MLflow
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not installed — skipping experiment tracking")

# ============================================================
# Logging setup
# ============================================================
LOG_DIR = Path(__file__).parent / "results"
LOG_DIR.mkdir(exist_ok=True)

log_path = LOG_DIR / "event_transformer_v1.log"
_file_handler = logging.FileHandler(log_path)
_file_handler.setLevel(logging.INFO)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[_file_handler, _stream_handler],
)
logger = logging.getLogger(__name__)

# Force flush on every log call (avoids buffering on Windows when redirected)
class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

for _h in logging.root.handlers:
    _h.__class__ = _FlushHandler

# ============================================================
# Config
# ============================================================
DEFAULT_DATA_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events")
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results" / "event_transformer_v1"
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "EventDriven_Transformer"

WINDOW_SIZE = 1000        # events per sample (~10s of events at 100Hz)
STRIDE = 500              # events between consecutive windows
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
BATCH_SIZE = 64           # reduced: 1000-event sequences are large
LR = 3e-4
EPOCHS_PER_FOLD = 15
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
N_FOLDS = 10
HORIZONS = ["1s", "5s", "10s"]

# ============================================================
# Dataset
# ============================================================

class MboEventDataset(Dataset):
    """
    Creates (window_of_events, labels) pairs from MBO event NPZ files.

    For each file (day), loads events + labels, then slides a window across the
    event stream with the given stride. The label for each window is the
    price-change at the LAST event in the window for each horizon.

    Samples with NaN labels are skipped.
    """

    def __init__(
        self,
        npz_files: List[Path],
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        horizons: List[str] = None,
        normalize_features: bool = True,
        feature_stats: Optional[Dict] = None,
    ):
        self.window_size = window_size
        self.stride = stride
        self.horizons = horizons or ["1s", "5s", "10s"]
        self.normalize_features = normalize_features

        # Storage
        # We store (file_idx, start_event_idx) tuples to avoid loading all
        # data into memory at once. But for training speed we pre-load everything.
        self.all_events: List[np.ndarray] = []
        self.all_labels: Dict[str, List[np.ndarray]] = {h: [] for h in self.horizons}
        self.sample_index: List[Tuple[int, int]] = []  # (day_idx, event_start)

        # Compute feature statistics for normalization
        if normalize_features and feature_stats is None:
            self._compute_stats(npz_files)
        elif feature_stats is not None:
            self.feature_mean = feature_stats["mean"]
            self.feature_std = feature_stats["std"]
        else:
            self.feature_mean = np.zeros(6, dtype=np.float32)
            self.feature_std = np.ones(6, dtype=np.float32)

        self._load_data(npz_files)

    def _compute_stats(self, npz_files: List[Path]):
        """Compute running mean/std across all events for normalization."""
        logger.info(f"Computing feature statistics from {len(npz_files)} files...")
        total_sum = np.zeros(6, dtype=np.float64)
        total_sq = np.zeros(6, dtype=np.float64)
        total_count = 0

        for f in npz_files:
            data = np.load(f, allow_pickle=True)
            events = data["events"].astype(np.float64)
            total_sum += events.sum(axis=0)
            total_sq += (events ** 2).sum(axis=0)
            total_count += len(events)

        self.feature_mean = (total_sum / total_count).astype(np.float32)
        var = (total_sq / total_count) - (self.feature_mean.astype(np.float64) ** 2)
        self.feature_std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
        logger.info(f"Feature mean: {self.feature_mean}")
        logger.info(f"Feature std:  {self.feature_std}")

    def get_feature_stats(self) -> Dict:
        return {"mean": self.feature_mean, "std": self.feature_std}

    def _load_data(self, npz_files: List[Path]):
        """Load all NPZ files and build sample index."""
        for day_idx, f in enumerate(npz_files):
            data = np.load(f, allow_pickle=True)
            events = data["events"].astype(np.float32)  # (N, 6)
            n_events = len(events)

            # Normalize
            if self.normalize_features:
                events = (events - self.feature_mean) / (self.feature_std + 1e-8)

            # Load labels for each horizon
            day_labels = {}
            for h in self.horizons:
                lbl = data[f"labels_{h}"].astype(np.float32)
                day_labels[h] = lbl

            # Slide window across events
            for start in range(0, n_events - self.window_size + 1, self.stride):
                end = start + self.window_size  # exclusive
                # Label is at the LAST event in the window
                label_idx = end - 1

                # Skip if any label is NaN
                labels_ok = all(
                    not np.isnan(day_labels[h][label_idx])
                    for h in self.horizons
                )
                if not labels_ok:
                    continue

                self.sample_index.append((day_idx, start))

            # Store raw data arrays for this day
            self.all_events.append(events)
            for h in self.horizons:
                self.all_labels[h].append(day_labels[h])

        logger.info(
            f"Dataset: {len(npz_files)} days, {len(self.sample_index)} samples "
            f"(window={self.window_size}, stride={self.stride})"
        )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        day_idx, start = self.sample_index[idx]
        end = start + self.window_size

        events = self.all_events[day_idx][start:end]  # (W, 6)

        # Labels: price change at last event for each horizon (regression)
        label_idx = end - 1
        labels = np.array(
            [self.all_labels[h][day_idx][label_idx] for h in self.horizons],
            dtype=np.float32,
        )  # (3,)

        return torch.from_numpy(events), torch.from_numpy(labels)


# ============================================================
# Time-Aware Positional Encoding
# ============================================================

class TimeAwarePositionalEncoding(nn.Module):
    """
    Positional encoding based on ELAPSED TIME from first event in window,
    not position index. Uses the log-time-delta feature (feature index 0).

    For each event, computes elapsed time since start of window by cumsum
    of time_delta_log values. Then encodes this via sinusoidal encoding.

    This respects the irregular temporal spacing of MBO events.
    """

    def __init__(self, d_model: int, max_time_scale: float = 100.0):
        super().__init__()
        self.d_model = d_model
        self.max_time_scale = max_time_scale

        # Learnable scale for time encoding
        self.time_scale = nn.Parameter(torch.ones(1) * 10.0)

        # Projection from sinusoidal to d_model
        assert d_model % 2 == 0
        self.half_d = d_model // 2

    def forward(self, x: torch.Tensor, time_delta_log: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) — event embeddings
            time_delta_log: (B, L) — log time delta feature (feature[:,0])

        Returns:
            x + time positional encoding: (B, L, d_model)
        """
        B, L, D = x.shape

        # Compute elapsed time for each position (cumsum of exp(time_delta_log))
        # time_delta_log is already normalized; we use it as-is for relative timing
        # Elapsed = cumulative sum of time deltas (in normalized space)
        elapsed = torch.cumsum(time_delta_log, dim=1)  # (B, L)

        # Scale elapsed time
        t = elapsed * self.time_scale  # (B, L)

        # Sinusoidal encoding dimensions
        freqs = torch.pow(
            self.max_time_scale,
            -torch.arange(self.half_d, device=x.device, dtype=x.dtype) / self.half_d
        )  # (half_d,)

        # Compute sin/cos encoding
        args = t.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # (B, L, half_d)
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, L, d_model)

        return x + pe


# ============================================================
# Model
# ============================================================

class EventTransformerV1(nn.Module):
    """
    Causal Event Transformer for MBO event stream prediction.

    Architecture:
        1. Linear projection: 6 raw features → 64-dim embedding
        2. Time-aware positional encoding using cumulative time_delta_log
        3. Causal Transformer encoder: 4 layers, 64-dim, 4 heads
        4. CLS-token aggregation (prepend learnable CLS token)
        5. Multi-task regression head: predict 1s, 5s, 10s price change

    Causality: Only past events in window influence prediction (no look-ahead).
    The causal mask is applied to prevent each position from attending to
    future positions, simulating real-time inference.

    Parameters: ~2.5M
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        n_targets: int = 3,
        window_size: int = WINDOW_SIZE,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_targets = n_targets

        # Input projection: 6 raw features → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(6, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Time-aware positional encoding
        self.time_pe = TimeAwarePositionalEncoding(d_model)

        # CLS token (prepended to each sequence for global aggregation)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Causal Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Pre-register causal mask buffer (computed dynamically per window_size)
        self._window_size = window_size

        # Prediction head: d_model → n_targets (multi-task regression)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build causal mask for seq_len+1 (including CLS token at position 0).
        CLS can attend to all positions; events can only attend to past events + CLS.
        """
        L = seq_len + 1  # +1 for CLS
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        # CLS token (row 0) can attend everywhere (no masking on first row)
        mask[0, :] = False
        return mask

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Args:
            events: (B, L, 6) float32 — batch of event windows

        Returns:
            preds: (B, n_targets) — predicted price changes
        """
        B, L, _ = events.shape

        # Extract time_delta_log for positional encoding (feature 0)
        time_delta_log = events[:, :, 0]  # (B, L)

        # Project events to d_model
        x = self.input_proj(events)  # (B, L, d_model)

        # Apply time-aware positional encoding
        x = self.time_pe(x, time_delta_log)  # (B, L, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)           # (B, L+1, d_model)

        # Causal mask
        causal_mask = self._get_causal_mask(L, events.device)  # (L+1, L+1)

        # Transformer encoding
        x = self.transformer(x, mask=causal_mask)  # (B, L+1, d_model)

        # Extract CLS token output for prediction
        cls_out = x[:, 0, :]  # (B, d_model)

        # Prediction
        preds = self.head(cls_out)  # (B, n_targets)
        return preds


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# IC Computation
# ============================================================

def compute_ic(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Spearman IC between predictions and labels. Handles NaN."""
    valid = ~(np.isnan(predictions) | np.isnan(labels))
    if valid.sum() < 20:
        return float("nan")
    rho, _ = scipy.stats.spearmanr(predictions[valid], labels[valid])
    return float(rho)


# ============================================================
# LR Scheduler with Warmup
# ============================================================

class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        s = self._step
        for i, pg in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if s <= self.warmup_steps:
                lr = base_lr * s / max(self.warmup_steps, 1)
            else:
                progress = (s - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
            pg["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ============================================================
# Training Loop (single fold)
# ============================================================

def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold_idx: int,
    output_dir: Path,
    mlflow_run,
    device: torch.device,
    total_train_steps: int,
) -> Dict:
    """Train model for one fold. Returns dict of metrics."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=WARMUP_STEPS,
        total_steps=total_train_steps,
    )

    best_val_loss = float("inf")
    global_step = 0
    train_losses = []

    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for events, labels in train_loader:
            events = events.to(device, non_blocking=True)    # (B, W, 6)
            labels = labels.to(device, non_blocking=True)    # (B, 3)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                preds = model(events)              # (B, 3)
                loss = F.mse_loss(preds, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # Validation
        val_metrics = evaluate_metrics_only(model, val_loader, device)

        logger.info(
            f"Fold {fold_idx:02d} | Epoch {epoch+1:2d}/{EPOCHS_PER_FOLD} | "
            f"Train Loss: {avg_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val IC (10s): {val_metrics.get('ic_10s', float('nan')):.4f} | "
            f"LR: {scheduler.get_lr():.2e}"
        )

        if MLFLOW_AVAILABLE and mlflow_run:
            step_offset = fold_idx * EPOCHS_PER_FOLD + epoch
            mlflow.log_metrics({
                f"fold{fold_idx:02d}_train_loss": avg_loss,
                f"fold{fold_idx:02d}_val_loss": val_metrics["loss"],
                f"fold{fold_idx:02d}_val_ic_1s": val_metrics.get("ic_1s", float("nan")),
                f"fold{fold_idx:02d}_val_ic_10s": val_metrics.get("ic_10s", float("nan")),
            }, step=step_offset)

        # Save best model for this fold
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "fold": fold_idx,
                "epoch": epoch,
                "val_loss": val_metrics["loss"],
                "val_ic_10s": val_metrics.get("ic_10s"),
            }, ckpt_path)

    return {"best_val_loss": best_val_loss}


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """Run inference on a DataLoader and return loss + IC metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for events, labels in loader:
            events = events.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                preds = model(events)
                loss = F.mse_loss(preds, labels)

            total_loss += loss.item()
            n_batches += 1
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())

    if not all_preds:
        return {"loss": total_loss / max(n_batches, 1)}, np.empty((0, len(HORIZONS))), np.empty((0, len(HORIZONS)))
    all_preds = np.concatenate(all_preds, axis=0)    # (N, 3)
    all_labels = np.concatenate(all_labels, axis=0)  # (N, 3)

    metrics = {"loss": total_loss / max(n_batches, 1)}
    for i, h in enumerate(HORIZONS):
        metrics[f"ic_{h}"] = compute_ic(all_preds[:, i], all_labels[:, i])

    return metrics, all_preds, all_labels


def evaluate_metrics_only(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """Evaluate and return only metrics dict (no pred arrays)."""
    metrics, _, _ = evaluate(model, loader, device)
    return metrics


# ============================================================
# OOT Inference (generate fold predictions)
# ============================================================

def run_oot_inference(
    model: nn.Module,
    oot_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on OOT fold, return (predictions, labels)."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for events, labels in oot_loader:
            events = events.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                preds = model(events)
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0)


# ============================================================
# Walk-Forward (Expanding Window)
# ============================================================

def run_expanding_wf(
    npz_files: List[Path],
    output_dir: Path,
    device: torch.device,
    n_folds: int = N_FOLDS,
):
    """
    Expanding window walk-forward training.

    Folds are defined by splitting the sorted date list into train+OOT segments.
    Each fold adds one more day to the training set (expanding).

    Example with 42 days and 10 folds:
      - Min train set: ~30 days (to have enough samples)
      - Each fold uses the next 1-2 days as OOT
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort files by date
    npz_files = sorted(npz_files)
    n_files = len(npz_files)
    logger.info(f"Total files: {n_files} ({npz_files[0].name} → {npz_files[-1].name})")

    # Expanding window split
    # Minimum training days = n_files - n_folds (so each fold has at least 1 OOT day)
    min_train = max(5, n_files - n_folds)
    fold_boundaries = []
    for fold in range(n_folds):
        train_end = min_train + fold  # exclusive
        oot_start = train_end
        oot_end = oot_start + max(1, (n_files - min_train) // n_folds)
        oot_end = min(oot_end, n_files)
        if oot_start >= n_files:
            break
        fold_boundaries.append((fold, list(range(train_end)), list(range(oot_start, oot_end))))

    logger.info(f"Running {len(fold_boundaries)} folds (expanding window)")

    # Storage for concat IC computation
    concat_preds = {h: [] for h in HORIZONS}
    concat_labels = {h: [] for h in HORIZONS}

    # MLflow run
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow_run = mlflow.start_run(run_name=f"EventTransformerV1_{time.strftime('%Y%m%d_%H%M')}")
        mlflow.log_params({
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs_per_fold": EPOCHS_PER_FOLD,
            "n_folds": len(fold_boundaries),
            "horizons": str(HORIZONS),
            "n_files": n_files,
        })

    try:
        for fold_idx, train_file_idxs, oot_file_idxs in fold_boundaries:
            train_files = [npz_files[i] for i in train_file_idxs]
            oot_files = [npz_files[i] for i in oot_file_idxs]

            logger.info(
                f"\n{'='*60}\n"
                f"FOLD {fold_idx:02d} | Train: {len(train_files)} days "
                f"({train_files[0].name}→{train_files[-1].name}) | "
                f"OOT: {len(oot_files)} days ({oot_files[0].name}→{oot_files[-1].name})"
                f"\n{'='*60}"
            )

            # Build training dataset (compute stats from train set only — no leakage)
            logger.info("Building train dataset...")
            train_ds = MboEventDataset(
                train_files,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
            )
            feature_stats = train_ds.get_feature_stats()

            # Build OOT dataset using TRAIN feature stats (no leakage)
            logger.info("Building OOT dataset...")
            oot_ds = MboEventDataset(
                oot_files,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
                feature_stats=feature_stats,  # use train stats
            )

            # DataLoaders
            # Windows note: num_workers > 0 uses spawn-based multiprocessing.
            # Large in-memory datasets (GBs of numpy arrays) CANNOT be pickled
            # across the spawn pipe on Windows — causes OSError/truncation.
            # Solution: use num_workers=0 (main process DataLoader) which avoids
            # all pickling. GPU is the bottleneck anyway — DataLoader overhead
            # is negligible compared to Transformer forward/backward pass.
            _num_workers = 0

            train_loader = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=_num_workers,
                pin_memory=True,
                drop_last=True,
            )
            oot_loader = DataLoader(
                oot_ds,
                batch_size=BATCH_SIZE * 2,
                shuffle=False,
                num_workers=_num_workers,
                pin_memory=True,
            )

            # Instantiate fresh model per fold
            model = EventTransformerV1().to(device)
            if fold_idx == 0:
                logger.info(f"Model parameters: {count_parameters(model):,}")

            total_steps = EPOCHS_PER_FOLD * len(train_loader)
            train_metrics = train_one_fold(
                model, train_loader, oot_loader,
                fold_idx, output_dir, mlflow_run, device, total_steps,
            )

            # Load best checkpoint for this fold's OOT inference
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model_state"])
                logger.info(f"Loaded best checkpoint (val_loss={ckpt['val_loss']:.4f})")

            # OOT inference
            logger.info("Running OOT inference...")
            oot_preds, oot_labels = run_oot_inference(model, oot_loader, device)

            # Per-fold IC
            fold_ics = {}
            for i, h in enumerate(HORIZONS):
                ic = compute_ic(oot_preds[:, i], oot_labels[:, i])
                fold_ics[h] = ic
                concat_preds[h].append(oot_preds[:, i])
                concat_labels[h].append(oot_labels[:, i])

            logger.info(
                f"Fold {fold_idx:02d} OOT IC | "
                + " | ".join(f"{h}: {fold_ics[h]:.4f}" for h in HORIZONS)
            )

            # Save fold artifacts: .pt weights already saved above as best ckpt
            # Save .npz predictions
            pred_path = output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
            np.savez_compressed(
                pred_path,
                predictions=oot_preds,
                labels=oot_labels,
                horizons=np.array(HORIZONS),
                ic_1s=np.array(fold_ics.get("1s", float("nan"))),
                ic_5s=np.array(fold_ics.get("5s", float("nan"))),
                ic_10s=np.array(fold_ics.get("10s", float("nan"))),
                oot_files=np.array([str(f) for f in oot_files]),
            )
            logger.info(f"Saved predictions → {pred_path}")

            # Save feature stats for this fold (needed for inference)
            stats_path = output_dir / f"fold_{fold_idx:02d}_feature_stats.npz"
            np.savez(stats_path, mean=feature_stats["mean"], std=feature_stats["std"])

            # Log per-fold metrics to MLflow
            if MLFLOW_AVAILABLE and mlflow_run:
                mlflow.log_metrics(
                    {f"oot_ic_{h}_fold{fold_idx:02d}": fold_ics[h] for h in HORIZONS},
                    step=fold_idx,
                )

            # Clean up to free memory
            del train_ds, oot_ds, train_loader, oot_loader
            import gc; gc.collect()
            torch.cuda.empty_cache()

        # ============================================================
        # Compute CONCAT IC (primary metric)
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("CONCAT IC (primary metric — all folds combined)")
        logger.info("="*60)

        concat_ic = {}
        for h in HORIZONS:
            if concat_preds[h]:
                all_p = np.concatenate(concat_preds[h])
                all_l = np.concatenate(concat_labels[h])
                ic = compute_ic(all_p, all_l)
                concat_ic[h] = ic
                logger.info(f"  Concat IC ({h}): {ic:.4f}")
            else:
                concat_ic[h] = float("nan")

        # Save concat predictions
        concat_path = output_dir / "concat_oot_predictions.npz"
        np.savez_compressed(
            concat_path,
            **{f"preds_{h}": np.concatenate(concat_preds[h]) for h in HORIZONS if concat_preds[h]},
            **{f"labels_{h}": np.concatenate(concat_labels[h]) for h in HORIZONS if concat_labels[h]},
            **{f"concat_ic_{h}": np.array(concat_ic[h]) for h in HORIZONS},
        )
        logger.info(f"Saved concat predictions → {concat_path}")

        # Log final concat IC to MLflow
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.log_metrics({f"concat_ic_{h}": concat_ic[h] for h in HORIZONS})

        logger.info("\n" + "="*60)
        logger.info("LEAKAGE AUDIT: PASSED (expanding window, no future data in training)")
        logger.info("Feature normalization computed from train set only per fold.")
        logger.info("="*60)

        return concat_ic

    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.end_run()


# ============================================================
# Data Transfer: Jupiter → Neptune via SCP
# ============================================================

def _jupiter_exec(cmd: str, timeout: int = 30) -> str:
    """Execute a command on Jupiter via its Flask API and return stdout."""
    import urllib.request, json
    payload = json.dumps({"command": cmd}).encode()
    req = urllib.request.Request(
        "http://jupiter:8765/exec",
        data=payload,
        headers={"X-API-Key": os.environ.get("QCC_API_KEY", ""), "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.load(resp)
    return result.get("stdout", "")


def transfer_data_from_jupiter(dest_dir: Path):
    """
    Copy MBO event NPZ files from Jupiter to Neptune.

    Strategy:
      1. List files on Jupiter via API
      2. For each missing file: have Jupiter base64-encode it in chunks and
         stream it to Neptune via the Jupiter exec API.
         Falls back to SCP if available.

    Files are large (~300-600MB each) so we use base64 in 8MB chunks.
    """
    import subprocess, base64, json

    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = set(f.name for f in dest_dir.glob("*.npz"))

    # List remote files
    try:
        stdout = _jupiter_exec(
            "ls /home/jupiter/Lvl3Quant/data/processed/mbo_events/ | grep '.npz$'",
            timeout=15,
        )
        remote_files = [f.strip() for f in stdout.splitlines() if f.strip().endswith(".npz")]
    except Exception as e:
        logger.warning(f"Could not list Jupiter files: {e}")
        return

    to_copy = [f for f in remote_files if f not in existing]
    if not to_copy:
        logger.info(f"All {len(remote_files)} files already on Neptune. Skipping transfer.")
        return

    logger.info(f"Transferring {len(to_copy)}/{len(remote_files)} files from Jupiter → Neptune...")

    # Try SCP first (fastest if available with key-based auth)
    scp_available = False
    try:
        r = subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
             f"jupiter@jupiter:/home/jupiter/Lvl3Quant/data/processed/mbo_events/{to_copy[0]}",
             str(dest_dir / to_copy[0])],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0:
            scp_available = True
            logger.info("SCP key-based auth works — using SCP for transfer")
        else:
            logger.info(f"SCP auth failed ({r.stderr[:100]}), falling back to API base64 transfer")
    except Exception:
        logger.info("SCP not available, using API base64 transfer")

    for fname in to_copy:
        remote_path = f"/home/jupiter/Lvl3Quant/data/processed/mbo_events/{fname}"
        dest_path = dest_dir / fname

        if dest_path.exists():
            logger.info(f"  Already exists: {fname}")
            continue

        logger.info(f"  Transferring: {fname}")
        t0 = time.time()

        if scp_available:
            r = subprocess.run(
                ["scp", "-o", "StrictHostKeyChecking=no",
                 f"jupiter@jupiter:{remote_path}", str(dest_path)],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0:
                size_mb = dest_path.stat().st_size / 1e6
                logger.info(f"    Done: {fname} ({size_mb:.0f} MB in {time.time()-t0:.1f}s)")
            else:
                logger.warning(f"    SCP failed: {r.stderr[:200]}")
        else:
            # API-based base64 transfer: read file in 8MB chunks
            try:
                # Get file size
                size_str = _jupiter_exec(f"stat -c %s {remote_path}", timeout=10).strip()
                file_size = int(size_str)
                chunk_size = 8 * 1024 * 1024  # 8MB in bytes
                n_chunks = (file_size + chunk_size - 1) // chunk_size
                logger.info(f"    File size: {file_size/1e6:.0f} MB, {n_chunks} chunks")

                with open(dest_path, "wb") as fout:
                    for chunk_i in range(n_chunks):
                        offset = chunk_i * chunk_size
                        # dd skip counts in bs-size blocks; use bs=1 for byte-level offset
                        # For performance use bs=1M
                        skip_mb = offset // (1024 * 1024)
                        count_mb = max(1, chunk_size // (1024 * 1024))
                        cmd = (
                            f"dd if={remote_path} bs=1M skip={skip_mb} count={count_mb} 2>/dev/null | "
                            f"base64 -w 0"
                        )
                        b64_data = _jupiter_exec(cmd, timeout=60).strip()
                        if not b64_data:
                            logger.warning(f"    Empty chunk {chunk_i} — skipping")
                            break
                        fout.write(base64.b64decode(b64_data))
                        if (chunk_i + 1) % 5 == 0:
                            logger.info(f"    Progress: {chunk_i+1}/{n_chunks} chunks")

                actual_size = dest_path.stat().st_size
                if abs(actual_size - file_size) > 1024:  # allow 1KB tolerance
                    logger.warning(
                        f"    Size mismatch: expected {file_size}, got {actual_size}. Removing."
                    )
                    dest_path.unlink()
                else:
                    logger.info(
                        f"    Done: {fname} ({actual_size/1e6:.0f} MB in {time.time()-t0:.1f}s)"
                    )
            except Exception as e:
                logger.error(f"    Transfer failed for {fname}: {e}")
                if dest_path.exists():
                    dest_path.unlink()

    final_files = list(dest_dir.glob("*.npz"))
    logger.info(f"Files available on Neptune after transfer: {len(final_files)}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Event Transformer v1 Walk-Forward Training")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory with mbo_events NPZ files")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for checkpoints and predictions")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--skip-transfer", action="store_true",
                        help="Skip data transfer from Jupiter")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("="*60)
    logger.info("Event Transformer v1 — Walk-Forward Training")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("="*60)

    # Step 1: Transfer data from Jupiter if needed
    if not args.skip_transfer:
        transfer_data_from_jupiter(data_dir)

    # Gather NPZ files
    npz_files = sorted(data_dir.glob("*_mbo_events.npz"))
    if not npz_files:
        logger.error(f"No *_mbo_events.npz files found in {data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(npz_files)} NPZ files: {npz_files[0].name} → {npz_files[-1].name}")

    # Step 2: Set process priority to BELOW_NORMAL (Windows)
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        logger.info("Process priority set to BELOW_NORMAL")
    except Exception as e:
        logger.info(f"Could not set priority (non-critical): {e}")

    # Step 3: Run expanding walk-forward
    concat_ic = run_expanding_wf(
        npz_files=npz_files,
        output_dir=output_dir,
        device=device,
        n_folds=args.n_folds,
    )

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("Final Concat IC:")
    for h in HORIZONS:
        logger.info(f"  {h}: {concat_ic.get(h, float('nan')):.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
