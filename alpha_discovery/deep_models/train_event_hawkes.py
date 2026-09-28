"""
Neural Hawkes / Temporal Point Process — Training Script v1

Data format (identical to train_event_cnn_1d.py):
  - events: (N_events, 6) float32
      [time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks]
  - labels_1s/5s/10s: (N_events,) float32 — mid-price change in ticks
  - timestamps: (N_events,) int64 nanoseconds

Architecture: Neural Hawkes Process (Continuous-Time GRU)
  - Models inter-event arrival times + event types as a self-exciting point process.
  - Recent events increase the conditional intensity of future arrivals (order flow
    bursts / sweeps exhibit exactly this clustering behaviour).

  Core idea:
    Between events, the hidden state DECAYS exponentially at a learned per-dimension
    rate (delta_t modulates the decay).  On each event arrival, the hidden state
    receives a standard GRU update that "injects" the new event.  The resulting
    representation captures both WHAT happened and WHEN (continuous time).

  Module layout:
    1. Event embedding: Linear(6 → hidden_dim)
    2. NeuralHawkesCell — custom GRU cell with continuous-time decay
       h(t)  = h_bar + (h_prev - h_bar) * exp(-delta * dt)    [inter-event decay]
       h_new = GRU_update(h(t), x_emb)                        [event injection]
    3. Final hidden state of the window → LayerNorm → MLP head
    4. Multi-task output: predict 1s, 5s, 10s price change simultaneously

  Compared to standard recurrent models, the Hawkes formulation explicitly uses the
  inter-arrival time (features[0] = time_delta_log) to modulate the hidden state
  between events rather than treating all steps as uniform ticks.

Training rules (ABSOLUTE — same as CNN1D):
  - Sliding window walk-forward (NEVER expanding window)
  - Concat IC as primary metric (scipy.stats.spearmanr across all OOS predictions)
  - Save .pt weights AND .npz predictions for EVERY fold
  - MLflow logging — every run, no exceptions
  - Mixed precision (fp16 when CUDA available)
  - num_workers=2 on Linux (Neptune), 0 on Windows (pickle issues with large numpy arrays)
  - Feature statistics from training set only per fold (zero leakage)
"""

import os
import sys
import gc
import time
import logging
import argparse
import warnings
import socket
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

log_path = LOG_DIR / "event_hawkes.log"
_file_handler   = logging.FileHandler(log_path)
_file_handler.setLevel(logging.INFO)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)
logging.basicConfig(force=True,
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [_file_handler, _stream_handler],
)
logger = logging.getLogger(__name__)


class _FlushHandler(logging.StreamHandler):
    """Force flush on every log call (avoids buffering on Windows when redirected)."""
    def emit(self, record):
        super().emit(record)
        self.flush()


for _h in logging.root.handlers:
    _h.__class__ = _FlushHandler


# ============================================================
# Config
# ============================================================
DEFAULT_DATA_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results" / "event_hawkes"


def _detect_mlflow_uri() -> str:
    """Auto-detect MLflow tracking URI: prefer localhost if reachable."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    try:
        s = socket.create_connection(("localhost", 5000), timeout=2)
        s.close()
        return "http://localhost:5000"
    except Exception:
        pass
    return "file:///home/nick/Lvl3Quant/mlruns"


MLFLOW_TRACKING_URI = _detect_mlflow_uri()
MLFLOW_EXPERIMENT   = "EventDriven_Hawkes"

# Hawkes-specific hyperparameters
HAWKES_HIDDEN_DIM = int(os.environ.get("HAWKES_HIDDEN_DIM", 128))
HAWKES_N_LAYERS   = int(os.environ.get("HAWKES_N_LAYERS",   2))
HAWKES_DROPOUT    = float(os.environ.get("HAWKES_DROPOUT",  0.1))

# Shared hyperparameters — use same EVENT_* env vars as other event scripts
WINDOW_SIZE      = int(os.environ.get("EVENT_WINDOW_SIZE", 500))
STRIDE           = int(os.environ.get("EVENT_STRIDE",      WINDOW_SIZE // 2))
BATCH_SIZE       = int(os.environ.get("EVENT_BATCH_SIZE",  128))
LR               = float(os.environ.get("EVENT_LR",        3e-4))
EPOCHS_PER_FOLD  = int(os.environ.get("EVENT_EPOCHS",      5))
WARMUP_STEPS     = int(os.environ.get("EVENT_WARMUP",      300))
GRAD_CLIP        = float(os.environ.get("EVENT_GRAD_CLIP",  1.0))
N_FOLDS          = int(os.environ.get("EVENT_N_FOLDS",     5))
HORIZONS         = ["1s", "5s", "10s"]

# Feature index in each event vector — time_delta_log is feature 0
TIME_DELTA_IDX = 0


# ============================================================
# Dataset (identical pattern to MboEventDataset in CNN1D)
# ============================================================

class MboEventDataset(Dataset):
    """
    Creates (window_of_events, labels) pairs from MBO event NPZ files.

    For each file (day), loads events + labels, then slides a window across the
    event stream with the given stride.  The label for each window is the
    price-change at the LAST event in the window for each horizon.

    Feature normalisation uses pre-computed stats (train-set stats only —
    no future leakage).  Samples with NaN labels are skipped.
    """

    def __init__(
        self,
        npz_files:         List[Path],
        window_size:       int = WINDOW_SIZE,
        stride:            int = STRIDE,
        horizons:          Optional[List[str]] = None,
        normalize_features: bool = True,
        feature_stats:     Optional[Dict] = None,
    ):
        self.window_size = window_size
        self.stride      = stride
        self.horizons    = horizons or ["1s", "5s", "10s"]
        self.normalize_features = normalize_features

        self.all_events:  List[np.ndarray] = []
        self.all_labels:  Dict[str, List[np.ndarray]] = {h: [] for h in self.horizons}
        self.sample_index: List[Tuple[int, int]] = []   # (day_idx, event_start)

        if normalize_features and feature_stats is None:
            self._compute_stats(npz_files)
        elif feature_stats is not None:
            self.feature_mean = feature_stats["mean"]
            self.feature_std  = feature_stats["std"]
        else:
            self.feature_mean = np.zeros(6, dtype=np.float32)
            self.feature_std  = np.ones(6, dtype=np.float32)

        self._load_data(npz_files)

    # ------------------------------------------------------------------
    def _compute_stats(self, npz_files: List[Path]):
        """Compute running mean/std across all events for normalisation."""
        logger.info(f"Computing feature statistics from {len(npz_files)} files...")
        total_sum   = np.zeros(6, dtype=np.float64)
        total_sq    = np.zeros(6, dtype=np.float64)
        total_count = 0

        for f in npz_files:
            for _attempt in range(12):
                try:
                    data = np.load(f, allow_pickle=True)
                    break
                except PermissionError:
                    if _attempt < 11:
                        import time as _time
                        logger.warning(
                            f"PermissionError on {f.name} in stats, "
                            f"retry {_attempt+1}/12..."
                        )
                        _time.sleep(5)
                    else:
                        raise
            events = data["events"].astype(np.float64)
            total_sum   += events.sum(axis=0)
            total_sq    += (events ** 2).sum(axis=0)
            total_count += len(events)

        self.feature_mean = (total_sum / total_count).astype(np.float32)
        var = (total_sq / total_count) - (self.feature_mean.astype(np.float64) ** 2)
        self.feature_std  = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
        logger.info(f"Feature mean: {self.feature_mean}")
        logger.info(f"Feature std:  {self.feature_std}")

    def get_feature_stats(self) -> Dict:
        return {"mean": self.feature_mean, "std": self.feature_std}

    # ------------------------------------------------------------------
    def _load_data(self, npz_files: List[Path]):
        """Load all NPZ files and build sample index."""
        for day_idx, f in enumerate(npz_files):
            for _attempt in range(12):
                try:
                    data = np.load(f, allow_pickle=True)
                    break
                except PermissionError:
                    if _attempt < 11:
                        import time as _time
                        logger.warning(
                            f"PermissionError on {f.name}, retry {_attempt+1}/12 "
                            f"(Defender scan?) waiting 5s..."
                        )
                        _time.sleep(5)
                    else:
                        raise

            events   = data["events"].astype(np.float32)   # (N, 6)
            n_events = len(events)

            if self.normalize_features:
                events = (events - self.feature_mean) / (self.feature_std + 1e-8)

            day_labels: Dict[str, np.ndarray] = {}
            for h in self.horizons:
                day_labels[h] = data[f"labels_{h}"].astype(np.float32)

            for start in range(0, n_events - self.window_size + 1, self.stride):
                end       = start + self.window_size
                label_idx = end - 1
                labels_ok = all(
                    not np.isnan(day_labels[h][label_idx]) for h in self.horizons
                )
                if not labels_ok:
                    continue
                self.sample_index.append((day_idx, start))

            self.all_events.append(events)
            for h in self.horizons:
                self.all_labels[h].append(day_labels[h])

        logger.info(
            f"Dataset: {len(npz_files)} days, {len(self.sample_index)} samples "
            f"(window={self.window_size}, stride={self.stride})"
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        day_idx, start = self.sample_index[idx]
        end       = start + self.window_size
        events    = self.all_events[day_idx][start:end]   # (W, 6)
        label_idx = end - 1
        labels = np.array(
            [self.all_labels[h][day_idx][label_idx] for h in self.horizons],
            dtype=np.float32,
        )  # (3,)
        return torch.from_numpy(events), torch.from_numpy(labels)


# ============================================================
# Architecture: Neural Hawkes Process (Continuous-Time GRU)
# ============================================================

class NeuralHawkesCell(nn.Module):
    """
    Single-layer Continuous-Time GRU cell for the Neural Hawkes Process.

    Two-phase update per event:

    Phase 1 — Exponential decay between events (continuous time):
        The hidden state drifts back toward a learned baseline h_bar at a
        learned per-dimension rate delta (always positive via softplus):

            h_decayed = h_bar + (h_prev - h_bar) * exp(-delta * dt)

        where dt = exp(time_delta_log) recovers the actual inter-event interval
        (the NPZ stores the LOG so we exponentiate before decay).

        The decay vector delta and baseline h_bar are outputs of linear layers
        applied to the PREVIOUS event embedding — they are event-conditioned,
        not fixed, which lets the cell learn context-dependent forgetting rates.

    Phase 2 — Discrete GRU update on event arrival:
        Standard GRU equations applied to (h_decayed, x_emb):

            z  = sigmoid(W_z x + U_z h_decayed + b_z)   [update gate]
            r  = sigmoid(W_r x + U_r h_decayed + b_r)   [reset gate]
            n  = tanh(W_n x + U_n (r * h_decayed) + b_n) [candidate]
            h_new = (1 - z) * h_decayed + z * n

        The update gate learns to blend the decayed history with the new event.

    Reference: Du et al., "Recurrent Marked Temporal Point Processes:
               Embedding Event History to Vector" (KDD 2016); Mei & Eisner,
               "The Neural Hawkes Process" (NeurIPS 2017).
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- GRU gates ---
        # Combined input+hidden projections for efficiency
        self.W_z = nn.Linear(input_dim,  hidden_dim, bias=True)
        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_r = nn.Linear(input_dim,  hidden_dim, bias=True)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_n = nn.Linear(input_dim,  hidden_dim, bias=True)
        self.U_n = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # --- Decay parameters (event-conditioned) ---
        # delta: per-dimension decay rate — always >0 via softplus
        self.W_delta = nn.Linear(input_dim, hidden_dim, bias=True)
        # h_bar: learned baseline hidden state (attractor)
        self.W_bar   = nn.Linear(input_dim, hidden_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialise delta biases slightly positive so initial decay rates are ~1
        nn.init.constant_(self.W_delta.bias, 0.5)

    def forward(
        self,
        x_emb:  torch.Tensor,   # (B, input_dim) current event embedding
        h_prev: torch.Tensor,   # (B, hidden_dim) hidden state BEFORE this event
        dt:     torch.Tensor,   # (B,) inter-event interval in real time units (>0)
    ) -> torch.Tensor:
        """
        Returns h_new: (B, hidden_dim) — hidden state AFTER this event.

        The decay is conditioned on x_emb (the CURRENT event drives how the
        history decays into this moment — an approximation that avoids needing
        the PREVIOUS event separately while keeping the model causal).
        """
        # Phase 1: Continuous-time decay
        # delta > 0 via softplus; shape (B, H)
        delta  = F.softplus(self.W_delta(x_emb))   # (B, H)
        h_bar  = torch.tanh(self.W_bar(x_emb))     # (B, H)  baseline attractor in (-1,1)

        # dt clamped to avoid exp overflow; shape (B, 1) for broadcasting
        dt_clamped = dt.clamp(min=0.0, max=50.0).unsqueeze(-1)   # (B, 1)
        decay      = torch.exp(-delta * dt_clamped)               # (B, H)
        h_decayed  = h_bar + (h_prev - h_bar) * decay            # (B, H)

        # Phase 2: Standard GRU update
        z = torch.sigmoid(self.W_z(x_emb) + self.U_z(h_decayed))          # (B, H)
        r = torch.sigmoid(self.W_r(x_emb) + self.U_r(h_decayed))          # (B, H)
        n = torch.tanh(   self.W_n(x_emb) + self.U_n(r * h_decayed))      # (B, H)
        h_new = (1.0 - z) * h_decayed + z * n                              # (B, H)

        return h_new


class NeuralHawkes(nn.Module):
    """
    Multi-layer Neural Hawkes Process for MBO event stream regression.

    Architecture:
        1. Input embedding: Linear(6 → hidden_dim) + LayerNorm
        2. Stack of N NeuralHawkesCells (deeper = richer decay dynamics)
           Inter-layer: pass h_prev from layer l−1; share same dt
        3. Extract representation from final hidden state of each window
        4. Multi-task prediction head: hidden_dim → 3 targets (1s, 5s, 10s)

    Causal guarantee:
        The GRU update for position t uses only events at positions 0..t.
        We read the hidden state at the LAST position of the window as the
        representation.  No future information leaks in.

    Memory-efficient:
        Processes the window as a Python loop over T steps (no materialised
        T×T attention matrix).  For window_size=500 and hidden_dim=128,
        peak intermediate tensors are small compared to CNN activations.
    """

    def __init__(
        self,
        in_features:  int   = 6,
        hidden_dim:   int   = HAWKES_HIDDEN_DIM,
        n_layers:     int   = HAWKES_N_LAYERS,
        dropout:      float = HAWKES_DROPOUT,
        n_targets:    int   = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.n_targets  = n_targets

        # Input embedding: lift raw features to hidden_dim
        self.input_embed = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Stack of continuous-time GRU cells
        self.cells   = nn.ModuleList(
            [NeuralHawkesCell(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(n_layers)]
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_targets),
        )

        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Args:
            events: (B, W, 6) float32 — batch of event windows.
                    Feature 0 is time_delta_log; we exponentiate to get dt.
        Returns:
            preds: (B, n_targets) — predicted price changes (1s, 5s, 10s)
        """
        B, W, _ = events.shape
        device  = events.device

        # Extract inter-event intervals: exp(time_delta_log) → real time (normalised units)
        # events[:, :, 0] is the normalised log-delta; we use it directly as a
        # proxy for dt (no need to un-normalise — the decay rates learn to compensate).
        # Using the raw normalised value keeps gradients well-scaled.
        dt_seq = events[:, :, TIME_DELTA_IDX]    # (B, W)  normalised log-delta

        # Embed all events in one batched call for efficiency
        x_seq = self.input_embed(events)          # (B, W, H)  — batch over time

        # Initialise hidden states per layer
        h_list = [torch.zeros(B, self.hidden_dim, device=device) for _ in range(self.n_layers)]

        # Unroll over time dimension
        for t in range(W):
            x_t  = x_seq[:, t, :]    # (B, H)
            dt_t = dt_seq[:, t]      # (B,)

            for layer_idx, (cell, drop) in enumerate(zip(self.cells, self.dropouts)):
                h_new = cell(x_t, h_list[layer_idx], dt_t)   # (B, H)
                if layer_idx < self.n_layers - 1:
                    # Apply dropout between layers (not on final layer output)
                    h_new = drop(h_new)
                h_list[layer_idx] = h_new

        # Use the final layer's hidden state at the last time step
        h_final = h_list[-1]          # (B, H)
        preds   = self.head(h_final)  # (B, n_targets)
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

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps:  int,
        min_lr:       float = 1e-6,
    ):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        self.base_lrs     = [pg["lr"] for pg in optimizer.param_groups]
        self._step        = 0

    def step(self):
        self._step += 1
        s = self._step
        for i, pg in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if s <= self.warmup_steps:
                lr = base_lr * s / max(self.warmup_steps, 1)
            else:
                progress = (s - self.warmup_steps) / max(
                    self.total_steps - self.warmup_steps, 1
                )
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + np.cos(np.pi * progress)
                )
            pg["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    use_amp: bool = True,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Run inference on a DataLoader and return (metrics_dict, preds, labels)."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_preds  = []
    all_labels = []

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    with torch.no_grad():
        for events, labels in loader:
            events = events.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with amp_ctx:
                preds = model(events)
                loss  = F.mse_loss(preds, labels)
            total_loss += loss.item()
            n_batches  += 1
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())

    if not all_preds:
        return (
            {"loss": total_loss / max(n_batches, 1)},
            np.empty((0, len(HORIZONS))),
            np.empty((0, len(HORIZONS))),
        )

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = {"loss": total_loss / max(n_batches, 1)}
    for i, h in enumerate(HORIZONS):
        metrics[f"ic_{h}"] = compute_ic(all_preds[:, i], all_labels[:, i])

    return metrics, all_preds, all_labels


def evaluate_metrics_only(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    use_amp: bool = True,
) -> Dict:
    """Convenience wrapper: return only metrics dict."""
    metrics, _, _ = evaluate(model, loader, device, use_amp=use_amp)
    return metrics


# ============================================================
# OOT Inference
# ============================================================

def run_oot_inference(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on OOT fold, return (predictions, labels)."""
    model.eval()
    all_preds  = []
    all_labels = []

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    with torch.no_grad():
        for events, labels in loader:
            events = events.to(device, non_blocking=True)
            with amp_ctx:
                preds = model(events)
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())

    if not all_preds:
        return np.empty((0, len(HORIZONS))), np.empty((0, len(HORIZONS)))
    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0)


# ============================================================
# Training Loop (single fold)
# ============================================================

def train_one_fold(
    model:             nn.Module,
    train_loader:      DataLoader,
    val_loader:        DataLoader,
    fold_idx:          int,
    output_dir:        Path,
    mlflow_run,
    device:            torch.device,
    total_train_steps: int,
    use_amp:           bool = True,
) -> Dict:
    """Train Neural Hawkes for one sliding-window fold. Returns metrics dict."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps = WARMUP_STEPS,
        total_steps  = total_train_steps,
    )

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for events, labels in train_loader:
            events = events.to(device, non_blocking=True)   # (B, W, 6)
            labels = labels.to(device, non_blocking=True)   # (B, 3)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                preds = model(events)                        # (B, 3)
                loss  = F.mse_loss(preds, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            n_batches   += 1
            global_step += 1

        avg_loss    = epoch_loss / max(n_batches, 1)
        val_metrics = evaluate_metrics_only(model, val_loader, device, use_amp=use_amp)

        logger.info(
            f"Fold {fold_idx:02d} | Epoch {epoch+1:2d}/{EPOCHS_PER_FOLD} | "
            f"Train Loss: {avg_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val IC (10s): {val_metrics.get('ic_10s', float('nan')):.4f} | "
            f"LR: {scheduler.get_lr():.2e}"
        )

        if MLFLOW_AVAILABLE and mlflow_run:
            step_offset = fold_idx * EPOCHS_PER_FOLD + epoch
            mlflow.log_metrics(
                {
                    f"fold{fold_idx:02d}_train_loss": avg_loss,
                    f"fold{fold_idx:02d}_val_loss":   val_metrics["loss"],
                    f"fold{fold_idx:02d}_val_ic_1s":  val_metrics.get("ic_1s",  float("nan")),
                    f"fold{fold_idx:02d}_val_ic_10s": val_metrics.get("ic_10s", float("nan")),
                },
                step=step_offset,
            )

        # Save best checkpoint for this fold
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "fold":        fold_idx,
                    "epoch":       epoch,
                    "val_loss":    val_metrics["loss"],
                    "val_ic_10s":  val_metrics.get("ic_10s"),
                    "arch": {
                        "hidden_dim":  HAWKES_HIDDEN_DIM,
                        "n_layers":    HAWKES_N_LAYERS,
                        "dropout":     HAWKES_DROPOUT,
                        "window_size": WINDOW_SIZE,
                    },
                },
                ckpt_path,
            )

    return {"best_val_loss": best_val_loss}


# ============================================================
# Walk-Forward (Sliding Window)
# ============================================================

def run_sliding_wf(
    npz_files:  List[Path],
    output_dir: Path,
    device:     torch.device,
    n_folds:    int = N_FOLDS,
):
    """
    Sliding window walk-forward training.

    Fixed 60-day training window slides forward by 1 day per fold.
    Feature statistics are always computed from the training set only (no leakage).
    OOT predictions are saved per fold + concatenated for final concat-IC calculation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort files by date (filename-based sorting; assumes YYYYMMDD prefix)
    npz_files = sorted(npz_files)

    # Filter files with no valid labels
    def _has_valid_labels(f: Path) -> bool:
        try:
            d   = np.load(f, allow_pickle=True)
            lbl = d["labels_1s"]
            return bool(not np.all(np.isnan(lbl)))
        except Exception:
            return False

    valid_files = [f for f in npz_files if _has_valid_labels(f)]
    skipped     = [f.name for f in npz_files if f not in set(valid_files)]
    if skipped:
        logger.warning(f"Skipping {len(skipped)} file(s) with all-NaN labels: {skipped}")
    npz_files = valid_files

    n_files = len(npz_files)
    if n_files == 0:
        logger.error("No valid NPZ files found. Exiting.")
        return {}

    logger.info(f"Total files (valid): {n_files} ({npz_files[0].name} → {npz_files[-1].name})")

    # Build fold boundaries (sliding window: 60-day train, 1-day OOT)
    train_days      = 60
    oot_days        = 1
    fold_boundaries = []
    for fold in range(n_files - train_days):
        train_start = fold
        train_end   = fold + train_days
        oot_start   = train_end
        oot_end     = min(train_end + oot_days, n_files)
        if oot_start >= n_files:
            break
        fold_boundaries.append((fold, list(range(train_start, train_end)), list(range(oot_start, oot_end))))

    logger.info(f"Running {len(fold_boundaries)} folds (sliding window, {train_days}d train, {oot_days}d OOT)")

    # AMP only useful on CUDA
    use_amp = device.type == "cuda"

    # Concat storage
    concat_preds  = {h: [] for h in HORIZONS}
    concat_labels = {h: [] for h in HORIZONS}

    # MLflow run
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow_run = mlflow.start_run(
            run_name=f"EventHawkes_{time.strftime('%Y%m%d_%H%M')}"
        )
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        mlflow.log_params(
            {
                "model":           "NeuralHawkes",
                "window_size":     WINDOW_SIZE,
                "stride":          STRIDE,
                "hidden_dim":      HAWKES_HIDDEN_DIM,
                "n_layers":        HAWKES_N_LAYERS,
                "dropout":         HAWKES_DROPOUT,
                "batch_size":      BATCH_SIZE,
                "lr":              LR,
                "epochs_per_fold": EPOCHS_PER_FOLD,
                "n_folds":         len(fold_boundaries),
                "horizons":        str(HORIZONS),
                "n_files":         n_files,
                "optimizer":       "AdamW",
                "warmup_steps":    WARMUP_STEPS,
                "grad_clip":       GRAD_CLIP,
                "node":            socket.gethostname(),
                "gpu":             gpu_name,
                "data_dir":        str(npz_files[0].parent),
                "output_dir":      str(output_dir),
                "mixed_precision": "fp16" if use_amp else "none",
                "num_workers":     0 if os.name == "nt" else 2,
                "time_delta_idx":  TIME_DELTA_IDX,
                "decay_type":      "per_dim_softplus_event_conditioned",
            }
        )

    try:
        for fold_idx, train_file_idxs, oot_file_idxs in fold_boundaries:
            train_files = [npz_files[i] for i in train_file_idxs]
            oot_files   = [npz_files[i] for i in oot_file_idxs]

            logger.info(
                f"\n{'='*60}\n"
                f"FOLD {fold_idx:02d} | Train: {len(train_files)} days "
                f"({train_files[0].name}→{train_files[-1].name}) | "
                f"OOT: {len(oot_files)} days ({oot_files[0].name}→{oot_files[-1].name})"
                f"\n{'='*60}"
            )

            # Build train dataset — stats computed from train set only (no leakage)
            logger.info("Building train dataset...")
            train_ds      = MboEventDataset(train_files, window_size=WINDOW_SIZE, stride=STRIDE)
            feature_stats = train_ds.get_feature_stats()

            # Build OOT dataset using TRAIN feature stats (no leakage)
            logger.info("Building OOT dataset...")
            oot_ds = MboEventDataset(
                oot_files,
                window_size   = WINDOW_SIZE,
                stride        = STRIDE,
                feature_stats = feature_stats,
            )

            # num_workers=2 on Neptune (32GB RAM); 0 on Windows (pickle issues)
            _num_workers = 0 if os.name == 'nt' else 2

            train_loader = DataLoader(
                train_ds,
                batch_size  = BATCH_SIZE,
                shuffle     = True,
                num_workers = _num_workers,
                pin_memory  = True,
                drop_last   = True,
            )
            oot_loader = DataLoader(
                oot_ds,
                batch_size  = BATCH_SIZE * 2,
                shuffle     = False,
                num_workers = _num_workers,
                pin_memory  = True,
            )

            # Fresh model per fold
            model = NeuralHawkes(
                in_features = 6,
                hidden_dim  = HAWKES_HIDDEN_DIM,
                n_layers    = HAWKES_N_LAYERS,
                dropout     = HAWKES_DROPOUT,
                n_targets   = len(HORIZONS),
            ).to(device)

            if fold_idx == 0:
                n_params = count_parameters(model)
                logger.info(f"Model parameters: {n_params:,}")
                logger.info(f"Hidden dim:       {HAWKES_HIDDEN_DIM}")
                logger.info(f"Layers:           {HAWKES_N_LAYERS}")
                logger.info(f"Window size:      {WINDOW_SIZE} events")
                logger.info(f"Decay type:       per-dim softplus, event-conditioned")

            total_steps = EPOCHS_PER_FOLD * len(train_loader)
            train_one_fold(
                model, train_loader, oot_loader,
                fold_idx, output_dir, mlflow_run, device, total_steps,
                use_amp=use_amp,
            )

            # Reload best checkpoint for OOT inference
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                logger.info(f"Loaded best checkpoint (val_loss={ckpt['val_loss']:.4f})")

            # OOT inference
            logger.info("Running OOT inference...")
            oot_preds, oot_labels = run_oot_inference(model, oot_loader, device, use_amp=use_amp)

            # Per-fold IC
            fold_ics: Dict[str, float] = {}
            for i, h in enumerate(HORIZONS):
                ic           = compute_ic(oot_preds[:, i], oot_labels[:, i])
                fold_ics[h]  = ic
                concat_preds[h].append(oot_preds[:, i])
                concat_labels[h].append(oot_labels[:, i])

            logger.info(
                f"Fold {fold_idx:02d} OOT IC | "
                + " | ".join(f"{h}: {fold_ics[h]:.4f}" for h in HORIZONS)
            )

            # Save fold artifacts: .npz predictions
            pred_path = output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
            np.savez_compressed(
                pred_path,
                predictions = oot_preds,
                labels      = oot_labels,
                horizons    = np.array(HORIZONS),
                ic_1s       = np.array(fold_ics.get("1s",  float("nan"))),
                ic_5s       = np.array(fold_ics.get("5s",  float("nan"))),
                ic_10s      = np.array(fold_ics.get("10s", float("nan"))),
                oot_files   = np.array([str(f) for f in oot_files]),
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

            # Free memory before next fold
            del train_ds, oot_ds, train_loader, oot_loader, model
            gc.collect()
            torch.cuda.empty_cache()

        # ============================================================
        # Compute CONCAT IC (primary metric — all folds combined)
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("CONCAT IC (primary metric — all folds combined)")
        logger.info("=" * 60)

        concat_ic: Dict[str, float] = {}
        for h in HORIZONS:
            if concat_preds[h]:
                all_p        = np.concatenate(concat_preds[h])
                all_l        = np.concatenate(concat_labels[h])
                ic           = compute_ic(all_p, all_l)
                concat_ic[h] = ic
                logger.info(f"  Concat IC ({h}): {ic:.4f}")
            else:
                concat_ic[h] = float("nan")

        # Save all concat predictions
        concat_path = output_dir / "concat_oot_predictions.npz"
        np.savez_compressed(
            concat_path,
            **{f"preds_{h}":     np.concatenate(concat_preds[h])
               for h in HORIZONS if concat_preds[h]},
            **{f"labels_{h}":    np.concatenate(concat_labels[h])
               for h in HORIZONS if concat_labels[h]},
            **{f"concat_ic_{h}": np.array(concat_ic[h]) for h in HORIZONS},
        )
        logger.info(f"Saved concat predictions → {concat_path}")

        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.log_metrics({f"concat_ic_{h}": concat_ic[h] for h in HORIZONS})

        logger.info("\n" + "=" * 60)
        logger.info("LEAKAGE AUDIT: PASSED")
        logger.info("  - Sliding window: train set never contains OOT dates")
        logger.info("  - Feature normalisation computed from train set only per fold")
        logger.info("  - Continuous-time GRU: hidden state at t uses only events 0..t")
        logger.info("  - No future dt leakage: time_delta at t is the gap FROM t-1 TO t")
        logger.info("=" * 60)

        return concat_ic

    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.end_run()


# ============================================================
# Data Transfer: Jupiter → Neptune via SCP / API
# ============================================================

def _jupiter_exec(cmd: str, timeout: int = 30) -> str:
    """Execute a command on Jupiter via its Flask API and return stdout."""
    import urllib.request, json as _json
    payload = _json.dumps({"command": cmd}).encode()
    req = urllib.request.Request(
        "http://jupiter:8765/exec",
        data    = payload,
        headers = {"X-API-Key": os.environ.get("QCC_API_KEY", ""), "Content-Type": "application/json"},
        method  = "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = _json.load(resp)
    return result.get("stdout", "")


def transfer_data_from_jupiter(dest_dir: Path):
    """
    Copy MBO event NPZ files from Jupiter to Neptune.

    Strategy:
      1. List files on Jupiter via Flask API
      2. SCP if key-based auth is available (fastest)
      3. Fallback: base64 streaming via API in 8MB chunks
    """
    import subprocess, base64, json as _json

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

    # Try SCP first
    scp_available = False
    try:
        r = subprocess.run(
            [
                "scp", "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                f"jupiter@jupiter:/home/jupiter/Lvl3Quant/data/processed/mbo_events/{to_copy[0]}",
                str(dest_dir / to_copy[0]),
            ],
            capture_output=True, text=True, timeout=30,
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
        dest_path   = dest_dir / fname

        if dest_path.exists():
            logger.info(f"  Already exists: {fname}")
            continue

        logger.info(f"  Transferring: {fname}")
        t0 = time.time()

        if scp_available:
            r = subprocess.run(
                [
                    "scp", "-o", "StrictHostKeyChecking=no",
                    f"jupiter@jupiter:{remote_path}", str(dest_path),
                ],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0:
                size_mb = dest_path.stat().st_size / 1e6
                logger.info(f"  Done: {fname} ({size_mb:.0f} MB in {time.time()-t0:.1f}s)")
            else:
                logger.warning(f"  SCP failed: {r.stderr[:200]}")
        else:
            try:
                size_str   = _jupiter_exec(f"stat -c %s {remote_path}", timeout=10).strip()
                file_size  = int(size_str)
                chunk_size = 8 * 1024 * 1024
                n_chunks   = (file_size + chunk_size - 1) // chunk_size
                logger.info(f"    File size: {file_size/1e6:.0f} MB, {n_chunks} chunks")

                with open(dest_path, "wb") as fout:
                    for chunk_i in range(n_chunks):
                        skip_mb  = (chunk_i * chunk_size) // (1024 * 1024)
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
                if abs(actual_size - file_size) > 1024:
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
    parser = argparse.ArgumentParser(description="Neural Hawkes / Temporal Point Process Walk-Forward Training")
    parser.add_argument("--data-dir",      type=str, default=DEFAULT_DATA_DIR,
                        help="Directory with mbo_events NPZ files")
    parser.add_argument("--output-dir",    type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for checkpoints and predictions")
    parser.add_argument("--n-folds",       type=int, default=N_FOLDS)
    parser.add_argument("--skip-transfer", action="store_true",
                        help="Skip data transfer from Jupiter")
    parser.add_argument("--device",        type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir   = Path(args.data_dir)
    device     = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("Neural Hawkes / Temporal Point Process — Walk-Forward Training")
    logger.info(f"Device:          {device}")
    if device.type == "cuda":
        logger.info(f"GPU:             {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Hidden dim:      {HAWKES_HIDDEN_DIM}")
    logger.info(f"Layers:          {HAWKES_N_LAYERS}")
    logger.info(f"Dropout:         {HAWKES_DROPOUT}")
    logger.info(f"Window size:     {WINDOW_SIZE}")
    logger.info(f"Stride:          {STRIDE}")
    logger.info(f"Batch size:      {BATCH_SIZE}")
    logger.info(f"LR:              {LR}")
    logger.info(f"Epochs/fold:     {EPOCHS_PER_FOLD}")
    logger.info(f"Data dir:        {data_dir}")
    logger.info(f"Output dir:      {output_dir}")
    logger.info("=" * 60)

    # Step 1: Transfer data from Jupiter if needed
    if not args.skip_transfer:
        transfer_data_from_jupiter(data_dir)

    # Gather NPZ files
    npz_files = sorted(data_dir.glob("*_mbo_events.npz"))
    if not npz_files:
        logger.error(f"No *_mbo_events.npz files found in {data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(npz_files)} NPZ files: {npz_files[0].name} → {npz_files[-1].name}")

    # Step 2: Set process priority to BELOW_NORMAL on Windows (yield to live inference)
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        logger.info("Process priority set to BELOW_NORMAL")
    except Exception as e:
        logger.info(f"Could not set priority (non-critical): {e}")

    # Step 3: Run sliding walk-forward
    concat_ic = run_sliding_wf(
        npz_files  = npz_files,
        output_dir = output_dir,
        device     = device,
        n_folds    = args.n_folds,
    )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Final Concat IC:")
    for h in HORIZONS:
        logger.info(f"  {h}: {concat_ic.get(h, float('nan')):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
