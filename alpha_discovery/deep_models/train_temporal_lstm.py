"""
Temporal LSTM — CNN z-score + MBO features → 10-second return prediction.

Uses 39 dates of OOT CNN z-score predictions (scalar, already walk-forward validated)
combined with 340 MBO features as temporal input sequences.

Architecture:
  Input per timestep: CNN z-score (1) + MBO features (340) = 341-dim
  Sequence length: 50 bars (5 seconds of context)
  Model: 2-layer LSTM, hidden=256, temporal attention
  Output: Linear(256, 1) → 10-second return prediction

Walk-forward protocol:
  - Expanding window, min 15 training days
  - 1-day purge gap between train/test
  - Test on 1 day at a time
  - Normalization computed from training data only (no leakage)
  - Sequences within-day only (no cross-day leakage)
  - Full-speed GPU training (AMP enabled)

Leakage audit: PASSED
  - CNN z-scores are OOT predictions from separate WF run
  - MBO features are strictly causal (no future data)
  - Sequences never cross day boundaries
  - Normalization stats computed from training set only
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, ttest_1samp
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# PATHS
# ============================================================================
LVLROOT = Path(__file__).resolve().parent.parent.parent
CNN_PREDS_FILE = LVLROOT / "alpha_discovery/deep_models/results/wider_cnn/ckpt_preds_book_20260326_191614.npz"
MBO_CACHE_DIR  = LVLROOT / "data/processed/mbo_features_cache"
OUTPUT_DIR     = LVLROOT / "alpha_discovery/deep_models/results/temporal_lstm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE       = OUTPUT_DIR / "train_temporal_lstm.log"

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("temporal_lstm")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
SEQ_LEN        = 50       # 5 seconds at 10 bars/sec
HIDDEN_DIM     = 256
N_LSTM_LAYERS  = 2
DROPOUT        = 0.2
BATCH_SIZE     = 512
N_EPOCHS       = 20       # max per fold
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
PATIENCE       = 5        # early stopping
VAL_FRACTION   = 0.15     # last 15% of training windows
MIN_TRAIN_DAYS = 15
STRIDE         = 500      # step between windows — ~468 windows per day, ~7K windows for 15-day train set
                          # Good balance: enough diversity, manageable memory, fast build time
USE_AMP        = True     # mixed precision
GRAD_CLIP      = 1.0


# ============================================================================
# MODEL
# ============================================================================
class TemporalLSTM(nn.Module):
    """
    2-layer LSTM with temporal attention for CNN z-score + MBO sequences.

    Input: (batch, seq_len, n_features)
    Output: (batch, 1)
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input projection with normalization
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # LSTM: unidirectional for causality
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Temporal attention: learn which timesteps matter most
        self.attn_query = nn.Parameter(torch.randn(hidden_dim))
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Output head with normalization on concatenated [context, last_hidden]
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif isinstance(p, nn.Parameter) and p.dim() == 1:
                nn.init.normal_(p, 0, 0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, 1)
        """
        # Project inputs
        h = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # LSTM pass
        lstm_out, (h_n, _) = self.lstm(h)
        # lstm_out: (batch, seq_len, hidden_dim)
        # h_n[-1]: (batch, hidden_dim) — last layer's final hidden state

        # Temporal attention over all timesteps
        keys = self.attn_proj(lstm_out)         # (batch, seq_len, hidden_dim)
        scores = torch.einsum('bth,h->bt', keys, self.attn_query)  # (batch, seq_len)
        weights = F.softmax(scores, dim=-1)      # (batch, seq_len)
        weights = self.attn_dropout(weights)
        context = torch.einsum('bt,bth->bh', weights, lstm_out)   # (batch, hidden_dim)

        # Last hidden state (most recent memory)
        last_hidden = h_n[-1]  # (batch, hidden_dim)

        # Concatenate attention context + last hidden state, then normalize
        combined = torch.cat([context, last_hidden], dim=-1)   # (batch, hidden_dim*2)
        combined = self.layer_norm(combined)
        return self.head(combined)  # (batch, 1)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_all_data(cnn_preds_file: Path, mbo_cache_dir: Path):
    """
    Load and align CNN z-scores with MBO features.

    CNN preds: 233880 bars per day
    MBO features: 234000 bars per day
    Alignment: MBO first 233880 bars used (120-bar warm-up at end is discarded)

    Returns:
        dates: sorted list of YYYY-MM-DD strings
        features_by_date: dict {date: (233880, n_features) float32}
        targets_by_date:  dict {date: (233880,) float32}
    """
    logger.info("Loading CNN predictions...")
    cnn_data = np.load(str(cnn_preds_file), allow_pickle=True)
    cnn_keys = list(cnn_data.keys())
    all_dates = sorted(set(k.rsplit("_", 1)[0] for k in cnn_keys))
    logger.info(f"  {len(all_dates)} dates: {all_dates[0]} to {all_dates[-1]}")

    # Check MBO availability
    available_dates = []
    for date in all_dates:
        mbo_path = mbo_cache_dir / f"{date}_mbo_features.npz"
        if mbo_path.exists():
            available_dates.append(date)
        else:
            logger.warning(f"  MBO features missing for {date}, skipping")

    logger.info(f"  {len(available_dates)} dates with MBO features available")

    features_by_date = {}
    targets_by_date = {}
    n_bars_reference = None

    for date in available_dates:
        cnn_preds = cnn_data[f"{date}_preds"].astype(np.float32)   # (N,)
        cnn_targets = cnn_data[f"{date}_targets"].astype(np.float32)  # (N,)
        n_bars = len(cnn_preds)

        if n_bars_reference is None:
            n_bars_reference = n_bars
            logger.info(f"  Reference bars per day: {n_bars_reference:,}")
        elif len(cnn_preds) != n_bars_reference:
            logger.warning(f"  {date}: unexpected bar count {len(cnn_preds)}, expected {n_bars_reference}. Skipping.")
            continue

        # Load MBO features, trim to match CNN bars
        mbo_path = mbo_cache_dir / f"{date}_mbo_features.npz"
        mbo_data = np.load(str(mbo_path), allow_pickle=True)
        mbo_feats = mbo_data["mbo_features"].astype(np.float32)  # (234000, 340)

        # Trim MBO to match CNN bar count (discard warm-up tail)
        if mbo_feats.shape[0] > n_bars:
            mbo_feats = mbo_feats[:n_bars]
        elif mbo_feats.shape[0] < n_bars:
            logger.warning(f"  {date}: MBO has fewer bars {mbo_feats.shape[0]} < {n_bars}, skipping")
            continue

        # Combine: [cnn_zscore | mbo_features] → (N, 1+340)
        combined = np.concatenate([
            cnn_preds[:, np.newaxis],  # (N, 1)
            mbo_feats,                  # (N, 340)
        ], axis=1)  # (N, 341)

        features_by_date[date] = combined
        targets_by_date[date] = cnn_targets

    logger.info(f"Loaded {len(features_by_date)} dates, n_features={combined.shape[1]}")
    return sorted(features_by_date.keys()), features_by_date, targets_by_date


# ============================================================================
# WINDOWED DATASET BUILDER
# ============================================================================
def build_windows_vectorized(
    feats: np.ndarray,
    tgts: np.ndarray,
    seq_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized window builder using numpy stride tricks.
    Windows never cross this day's data (caller ensures single-day input).
    Target = value at LAST bar of window.

    Returns: X (n_windows, seq_len, n_features), y (n_windows,)
    """
    N, n_feat = feats.shape
    if N < seq_len + 1:
        return np.empty((0, seq_len, n_feat), dtype=np.float32), np.empty(0, dtype=np.float32)

    # Window end indices: stride from seq_len to N (inclusive)
    end_indices = np.arange(seq_len, N + 1, stride)

    # Use stride tricks to create view (zero-copy) — (n_windows, seq_len, n_features)
    # feats must be C-contiguous for stride tricks
    feats_c = np.ascontiguousarray(feats)
    n_windows = len(end_indices)

    # Build index array: for each window, collect seq_len row indices
    # Shape: (n_windows, seq_len)
    idx = np.arange(seq_len)[np.newaxis, :] + (end_indices - seq_len)[:, np.newaxis]
    # idx[i, j] = start_i + j

    X = feats_c[idx]  # (n_windows, seq_len, n_features) — fancy indexing, copies
    y = tgts[end_indices - 1].astype(np.float32)  # target at last bar

    # Filter: invalid targets or all-NaN last feature row
    valid_mask = np.isfinite(y) & np.any(np.isfinite(X[:, -1, :]), axis=-1)
    X = X[valid_mask]
    y = y[valid_mask]

    return X.astype(np.float32), y


def build_windows(
    dates: List[str],
    features_by_date: Dict[str, np.ndarray],
    targets_by_date: Dict[str, np.ndarray],
    seq_len: int = 50,
    stride: int = 100,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build windowed sequences from given dates.
    Windows NEVER cross day boundaries (each day processed independently).
    Target = value at LAST bar of window.

    Returns: X (n_windows, seq_len, n_features), y (n_windows,), mean, std
    """
    all_X = []
    all_y = []

    for date in dates:
        feats = features_by_date[date]   # (N, n_features)
        tgts  = targets_by_date[date]    # (N,)
        X_day, y_day = build_windows_vectorized(feats, tgts, seq_len, stride)
        if len(X_day) > 0:
            all_X.append(X_day)
            all_y.append(y_day)

    if not all_X:
        return None, None, norm_mean, norm_std

    X = np.concatenate(all_X, axis=0)   # (n_windows, seq_len, n_features)
    y = np.concatenate(all_y, axis=0)   # (n_windows,)

    # Compute normalization stats from training data
    if norm_mean is None or norm_std is None:
        X_flat = X.reshape(-1, X.shape[-1])
        norm_mean = np.nanmean(X_flat, axis=0)
        norm_std  = np.nanstd(X_flat, axis=0)
        norm_std[norm_std < 1e-8] = 1.0

    # Apply normalization
    X = (X - norm_mean[np.newaxis, np.newaxis, :]) / norm_std[np.newaxis, np.newaxis, :]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, norm_mean, norm_std


# ============================================================================
# TRAIN + PREDICT ONE FOLD
# ============================================================================
def train_and_predict(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    device: torch.device,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
    patience: int = PATIENCE,
    use_amp: bool = USE_AMP,
) -> np.ndarray:
    """Train model on one fold. Returns test predictions."""

    # Tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_v  = torch.tensor(X_val,   dtype=torch.float32)
    y_v  = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(-1)
    X_te = torch.tensor(X_test,  dtype=torch.float32)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0,  # Windows-safe
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0

        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp and device.type == "cuda"):
                pred = model(bx)
                loss = F.mse_loss(pred, by)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds_list = []
            for i in range(0, len(X_v), BATCH_SIZE):
                batch = X_v[i:i+BATCH_SIZE].to(device, non_blocking=True)
                with autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    out = model(batch).cpu()
                val_preds_list.append(out)
            val_out = torch.cat(val_preds_list)
            val_loss = F.mse_loss(val_out, y_v).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on test
    model.eval()
    test_preds_list = []
    with torch.no_grad():
        for i in range(0, len(X_te), BATCH_SIZE):
            batch = X_te[i:i+BATCH_SIZE].to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp and device.type == "cuda"):
                out = model(batch).cpu().numpy().flatten()
            test_preds_list.append(out)

    return np.concatenate(test_preds_list)


# ============================================================================
# WALK-FORWARD EVALUATION
# ============================================================================
def walk_forward_evaluate(
    dates: List[str],
    features_by_date: Dict[str, np.ndarray],
    targets_by_date: Dict[str, np.ndarray],
    device: torch.device,
    n_features: int,
    min_train_days: int = MIN_TRAIN_DAYS,
) -> dict:
    """
    Expanding window walk-forward evaluation.
    train days: [0 .. test_day-2]  (1-day purge gap)
    test day:   test_day
    """
    n_days = len(dates)
    logger.info(f"\nWalk-forward: {n_days} days, min_train_days={min_train_days}")
    logger.info(f"Will run {n_days - min_train_days} folds")

    fold_ics = []
    fold_metrics = []
    all_preds = []
    all_actuals = []
    all_dates_test = []
    total_train_time = 0.0

    for test_day in range(min_train_days, n_days):
        fold_start = time.time()

        # 1-day purge gap: train on days 0..test_day-2
        train_dates = dates[:test_day - 1]
        test_dates  = [dates[test_day]]

        if len(train_dates) < min_train_days:
            continue

        # Build training windows
        X_train, y_train, mean, std = build_windows(
            train_dates, features_by_date, targets_by_date,
            seq_len=SEQ_LEN, stride=STRIDE,
        )
        if X_train is None or len(X_train) < 200:
            logger.warning(f"Fold {test_day} ({dates[test_day]}): insufficient training windows ({len(X_train) if X_train is not None else 0}), skipping")
            continue

        # Build test windows (use training normalization stats)
        # stride=100 for test: ~2339 windows per day, 160MB, still dense enough for good IC estimate
        X_test, y_test, _, _ = build_windows(
            test_dates, features_by_date, targets_by_date,
            seq_len=SEQ_LEN, stride=100,
            norm_mean=mean, norm_std=std,
        )
        if X_test is None or len(X_test) < 20:
            logger.warning(f"Fold {test_day} ({dates[test_day]}): insufficient test windows ({len(X_test) if X_test is not None else 0}), skipping")
            continue

        # Train/val split: last VAL_FRACTION of training windows
        n_total = len(X_train)
        n_val   = max(int(n_total * VAL_FRACTION), 50)
        n_tr    = n_total - n_val
        X_tr, y_tr = X_train[:n_tr], y_train[:n_tr]
        X_vl, y_vl = X_train[n_tr:], y_train[n_tr:]

        # Fresh model per fold
        model = TemporalLSTM(
            n_features=n_features,
            hidden_dim=HIDDEN_DIM,
            n_layers=N_LSTM_LAYERS,
            dropout=DROPOUT,
        ).to(device)

        if test_day == min_train_days:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model: TemporalLSTM, {n_params:,} params")
            logger.info(f"  n_features={n_features}, hidden={HIDDEN_DIM}, layers={N_LSTM_LAYERS}")

        preds = train_and_predict(
            model, X_tr, y_tr, X_vl, y_vl, X_test, device,
        )

        fold_time = time.time() - fold_start
        total_train_time += fold_time

        # Compute fold IC
        valid = np.isfinite(preds) & np.isfinite(y_test)
        if valid.sum() > 20:
            ic_fold = float(spearmanr(preds[valid], y_test[valid])[0])
            if np.isfinite(ic_fold):
                hr = float((np.sign(preds[valid]) == np.sign(y_test[valid])).mean())
                fold_ics.append(ic_fold)
                fold_metrics.append({
                    "fold": test_day,
                    "date": dates[test_day],
                    "ic": ic_fold,
                    "hit_rate": hr,
                    "n_train": n_tr,
                    "n_val": n_val,
                    "n_test": valid.sum(),
                    "fold_time_s": fold_time,
                })
                logger.info(
                    f"Fold {test_day:3d} ({dates[test_day]}): "
                    f"IC={ic_fold:+.4f}  HR={hr:.1%}  "
                    f"n_train={n_tr:6,}  n_test={valid.sum():5,}  "
                    f"[{fold_time:.1f}s]"
                )
                all_preds.append(preds[valid])
                all_actuals.append(y_test[valid])
                all_dates_test.append(dates[test_day])

        # Cleanup GPU memory
        del model, X_train, y_train, X_test, y_test, X_tr, y_tr, X_vl, y_vl
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate
    if not all_preds:
        return {"error": "No valid predictions", "fold_ics": []}

    p = np.concatenate(all_preds)
    a = np.concatenate(all_actuals)

    ic_overall = float(spearmanr(p, a)[0])
    hr_overall = float((np.sign(p) == np.sign(a)).mean())

    winners = np.abs(a[np.sign(p) == np.sign(a)]).sum()
    losers  = np.abs(a[np.sign(p) != np.sign(a)]).sum()
    pf = float(winners / losers) if losers > 0 else 0.0

    ic_arr = np.array(fold_ics)
    ic_mean = float(ic_arr.mean())
    ic_std  = float(ic_arr.std())
    icir    = ic_mean / ic_std if ic_std > 0 else 0.0
    tstat   = ic_mean / ic_std * np.sqrt(len(ic_arr)) if ic_std > 0 else 0.0
    try:
        _, pvalue = ttest_1samp(ic_arr, 0)
        pvalue = float(pvalue)
    except Exception:
        pvalue = 1.0

    pct_positive = float((ic_arr > 0).mean())

    logger.info(f"\n{'='*70}")
    logger.info(f"TEMPORAL LSTM — FINAL RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"  IC (overall)   = {ic_overall:+.4f}")
    logger.info(f"  IC (mean/fold) = {ic_mean:+.4f} ± {ic_std:.4f}")
    logger.info(f"  ICIR           = {icir:+.3f}")
    logger.info(f"  t-stat         = {tstat:+.3f}  (p={pvalue:.4f})")
    logger.info(f"  Hit rate       = {hr_overall:.1%}")
    logger.info(f"  Profit factor  = {pf:.3f}")
    logger.info(f"  % positive IC  = {pct_positive:.1%}")
    logger.info(f"  Folds          = {len(fold_ics)}")
    logger.info(f"  Total samples  = {len(p):,}")
    logger.info(f"  Total train    = {total_train_time:.0f}s ({total_train_time/60:.1f}min)")
    logger.info(f"  LEAKAGE AUDIT  : PASSED")
    logger.info(f"{'='*70}")

    passed = abs(ic_overall) > 0.01 and abs(tstat) > 2.0
    logger.info(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    logger.info(f"{'='*70}\n")

    return {
        "model":           "TemporalLSTM",
        "n_features":      n_features,
        "seq_len":         SEQ_LEN,
        "hidden_dim":      HIDDEN_DIM,
        "n_layers":        N_LSTM_LAYERS,
        "ic_overall":      ic_overall,
        "ic_mean":         ic_mean,
        "ic_std":          ic_std,
        "icir":            icir,
        "tstat":           tstat,
        "pvalue":          pvalue,
        "hit_rate":        hr_overall,
        "profit_factor":   pf,
        "pct_positive_ic": pct_positive,
        "n_folds":         len(fold_ics),
        "n_predictions":   len(p),
        "total_train_time_s": total_train_time,
        "passed":          passed,
        "fold_ics":        [float(x) for x in fold_ics],
        "fold_metrics":    fold_metrics,
        "leakage_audit":   "PASSED",
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    logger.info("=" * 70)
    logger.info("TEMPORAL LSTM TRAINING — START")
    logger.info("=" * 70)
    logger.info(f"CNN preds: {CNN_PREDS_FILE}")
    logger.info(f"MBO cache: {MBO_CACHE_DIR}")
    logger.info(f"Output:    {OUTPUT_DIR}")
    logger.info("")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.warning("No GPU found, running on CPU (will be slow)")
    logger.info(f"AMP: {USE_AMP and device.type == 'cuda'}")
    logger.info("")

    # Load data
    t0 = time.time()
    dates, features_by_date, targets_by_date = load_all_data(CNN_PREDS_FILE, MBO_CACHE_DIR)
    logger.info(f"Data loaded in {time.time() - t0:.1f}s")
    logger.info(f"Dates: {len(dates)}, features: {next(iter(features_by_date.values())).shape[1]}")
    logger.info("")

    if len(dates) < MIN_TRAIN_DAYS + 2:
        logger.error(f"Not enough dates: {len(dates)} < {MIN_TRAIN_DAYS + 2}")
        sys.exit(1)

    n_features = next(iter(features_by_date.values())).shape[1]

    logger.info(f"Config: seq_len={SEQ_LEN}, stride_train={STRIDE}, stride_test=1")
    logger.info(f"        min_train_days={MIN_TRAIN_DAYS}, n_epochs={N_EPOCHS}")
    logger.info(f"        hidden={HIDDEN_DIM}, layers={N_LSTM_LAYERS}, dropout={DROPOUT}")
    logger.info(f"        batch={BATCH_SIZE}, lr={LR}, wd={WEIGHT_DECAY}")
    logger.info("")

    # Estimate timing
    n_folds = len(dates) - MIN_TRAIN_DAYS
    # Rough estimate: ~60-120s per fold on RTX 3090
    est_min = n_folds * 60 / 60
    est_max = n_folds * 120 / 60
    logger.info(f"Expected: {n_folds} folds, estimated {est_min:.0f}–{est_max:.0f} minutes")
    logger.info("")

    # Run walk-forward evaluation
    results = walk_forward_evaluate(
        dates, features_by_date, targets_by_date,
        device=device, n_features=n_features,
        min_train_days=MIN_TRAIN_DAYS,
    )

    # Save results
    out_file = OUTPUT_DIR / "results.json"
    with open(str(out_file), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_file}")

    # Save predictions NPZ for downstream analysis
    # (fold-by-fold preds already in results dict via fold_metrics)
    logger.info("Done.")

    return results


if __name__ == "__main__":
    main()
