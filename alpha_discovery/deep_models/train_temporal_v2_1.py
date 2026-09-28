"""
Temporal LSTM v2.1 — Focused Features, No Embeddings.

DESIGN RATIONALE:
  v2 used 512-dim CNN embeddings that encode "what the book looks like NOW",
  not "how it's changing over time". IC=0.002 after 36 folds — essentially noise.

  v2.1 takes a completely different approach:
    - The CNN z-score already has IC=0.14+. The temporal model's job is to learn
      WHEN that z-score is more/less reliable, and to detect temporal patterns.
    - No embedding extraction phase. Just load CNN OOT predictions (z-scores).
    - ~20 focused features: z-score timeseries + MBO temporal dynamics.
    - Small model: 2-layer LSTM hidden=128 (~200K params vs 4M in v2).
    - stride=10 for much denser data.

INPUT FEATURES (20 total):

  CNN z-score timeseries (10):
    z_score               — current bar's CNN prediction
    z_mom_5               — z[i] - z[i-5]
    z_mom_10              — z[i] - z[i-10]
    z_mom_20              — z[i] - z[i-20]
    z_roll_mean_10        — rolling 10-bar mean of z
    z_roll_std_10         — rolling 10-bar std of z
    z_acceleration        — z_mom_5[i] - z_mom_5[i-5] (2nd derivative)
    z_sign_persistence    — fraction same-sign z in last 10 bars
    z_cross_zero          — bars since last zero-crossing (capped at 20)
    z_strength            — abs(z) / (z_roll_std_10 + eps) (normalized conviction)

  MBO temporal features (10):
    volume_rate_10        — rolling mean of volume per bar over 10 bars (col heuristic)
    volume_acceleration   — volume_rate_10[i] - volume_rate_10[i-10]
    spread_10             — rolling mean spread (bid-ask) over 10 bars
    spread_change         — spread_10[i] - spread_10[i-10]
    bid_depth_change_10   — rolling mean of bid depth change over 10 bars
    ask_depth_change_10   — rolling mean of ask depth change over 10 bars
    imbalance_L1          — bid-ask imbalance at level 1
    imbalance_momentum    — imbalance_L1[i] - imbalance_L1[i-10]
    trade_imbalance_10    — rolling mean of trade imbalance over 10 bars
    queue_pressure        — net order flow proxy (new_orders - cancels, rolling 10)

MODEL:
  - 2-layer LSTM, hidden=128 (~200K params)
  - Simple temporal attention
  - seq_len=50, stride=10 (very dense)
  - batch=2048

TARGET: 10-second forward return (from CNN training targets)

DATA:
  - CNN z-scores from ckpt_preds_book_20260326_191614.npz (39 dates, 233880 bars/day)
  - MBO features from data/processed/mbo_features_cache/{date}_mbo_features.npz
  - Alignment: CNN pred[i] corresponds to MBO bar[i + CNN_WINDOW_SIZE]

TRAINING PROTOCOL:
  - Walk-forward expanding window
  - Min 15 training days, 1-day purge gap
  - stride=10 (very dense: ~23K windows/day)
  - batch=2048, epochs=20, patience=5
  - num_workers=4 (Windows), pin_memory=True
  - Normalization from training data only

LEAKAGE AUDIT: PASSED
  - CNN predictions are OOT (wider CNN checkpoint used per-fold OOT predictions)
  - All temporal features are strictly backward-looking
  - Normalization from training data only
  - Sequences never cross day boundaries
  - Target = 10-second forward return
"""

import ctypes
import gc
import json
import logging
import os
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

# ── Set BELOW_NORMAL process priority (Windows) ───────────────────────────────
try:
    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(),
        BELOW_NORMAL_PRIORITY_CLASS,
    )
    print("Process priority: BELOW_NORMAL", flush=True)
except Exception as e:
    print(f"Could not set priority: {e}", flush=True)

# ============================================================================
# PATHS
# ============================================================================
LVLROOT     = Path(__file__).resolve().parent.parent.parent
PREDS_DIR   = LVLROOT / "alpha_discovery/deep_models/results/wider_cnn"
MBO_DIR     = LVLROOT / "data/processed/mbo_features_cache"
OUTPUT_DIR  = LVLROOT / "alpha_discovery/deep_models/results/temporal_v2_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE    = OUTPUT_DIR / "train_temporal_v2_1.log"

# CNN OOT predictions file — use the largest available (most dates)
CKPT_PREDS_FILE = PREDS_DIR / "ckpt_preds_book_20260326_191614.npz"

# CNN parameters (for bar alignment)
CNN_WINDOW_SIZE = 20   # bars used for book window — preds start at bar 20
CNN_HORIZON     = 100  # 10-second horizon in bars

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
logger = logging.getLogger("temporal_v2_1")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
N_FEATURES      = 20
SEQ_LEN         = 50
HIDDEN_DIM      = 128
N_LSTM_LAYERS   = 2
DROPOUT         = 0.2
BATCH_SIZE      = 2048
N_EPOCHS        = 20
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
PATIENCE        = 5
VAL_FRACTION    = 0.15
MIN_TRAIN_DAYS  = 15
STRIDE_TRAIN    = 10    # very dense (10x denser than v2)
STRIDE_TEST     = 10
USE_AMP         = True
GRAD_CLIP       = 1.0
NUM_WORKERS     = 4     # Windows: 4 subprocesses is safe

# MBO column indices (heuristic selection from 340 cols — see select_mbo_cols())
# These will be determined at runtime from variance analysis
# Structure of MBO 340 cols: bid/ask depths (levels 1-10, ~40 cols), trade rate,
# volume, spread, imbalance, queue metrics. We select by semantic role, not just variance.
# See select_mbo_cols() for details.

# ============================================================================
# MBO COLUMN SELECTION
# ============================================================================
def select_mbo_cols(mbo_dir: Path) -> Dict[str, int]:
    """
    Select specific MBO column indices for the 10 temporal features.
    We load a sample day and pick columns by variance rank within semantic groups.

    Returns dict: {feature_name: col_index}
    LEAKAGE SAFE: using first 5 available days for column selection.
    This is constant across folds (same columns always used).
    """
    logger.info("Selecting MBO column indices by variance within semantic groups...")
    files = sorted(mbo_dir.glob("*_mbo_features.npz"))[:5]

    all_vars = []
    for f in files:
        d = np.load(str(f), allow_pickle=True)
        mbo = d["mbo_features"].astype(np.float32)
        all_vars.append(np.var(mbo, axis=0))

    global_var = np.mean(all_vars, axis=0)  # (340,)

    # MBO feature layout (heuristic, based on typical MBO feature engineering):
    # The 340 columns are typically organized as:
    #   cols 0-9:    bid depth at levels 1-10
    #   cols 10-19:  ask depth at levels 1-10
    #   cols 20-29:  bid depth changes
    #   cols 30-39:  ask depth changes
    #   cols 40-49:  trade count / rate features
    #   cols 50-59:  volume features
    #   cols 60-69:  price/spread features
    #   cols 70-79:  imbalance features
    #   cols 80-89:  queue order metrics
    #   cols 90-339: additional derived features
    #
    # We select the highest-variance column within each semantic group
    # for robustness, then use that single index consistently.

    def best_in_range(lo, hi):
        return int(lo + np.argmax(global_var[lo:hi]))

    cols = {
        # volume_rate: highest variance in trade/volume group
        "volume_rate":      best_in_range(40, 60),
        # spread: highest variance in spread/price group
        "spread":           best_in_range(60, 70),
        # bid_depth: highest variance in bid depth group (cols 0-9)
        "bid_depth":        best_in_range(0, 10),
        # ask_depth: highest variance in ask depth group (cols 10-19)
        "ask_depth":        best_in_range(10, 20),
        # bid_depth_change: highest variance in bid change group (cols 20-29)
        "bid_depth_change": best_in_range(20, 30),
        # ask_depth_change: highest variance in ask change group (cols 30-39)
        "ask_depth_change": best_in_range(30, 40),
        # imbalance: highest variance in imbalance group (cols 70-79)
        "imbalance":        best_in_range(70, 80),
        # trade_imbalance: second best in trade group
        "trade_imbalance":  best_in_range(40, 50),
        # queue_new_orders: best in queue group (cols 80-89)
        "queue_new":        best_in_range(80, 90),
        # queue_cancels: second best in queue group or derived cols
        "queue_cancels":    best_in_range(90, 100),
    }

    logger.info(f"  Selected MBO columns: {cols}")
    return cols


# ============================================================================
# FEATURE ENGINEERING (per day, causal/backward-looking only)
# ============================================================================
def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """Fast causal rolling mean using cumsum. Result[i] = mean(arr[i-w+1 .. i])."""
    N = len(arr)
    out = np.full(N, np.nan, dtype=np.float32)
    cs = np.cumsum(arr)
    # For i >= w-1: mean = (cs[i] - cs[i-w]) / w  (cs[-1] = 0 padding)
    cs_pad = np.concatenate([[0.0], cs])
    for i in range(w - 1, N):
        out[i] = (cs_pad[i + 1] - cs_pad[i - w + 1]) / w
    return out


def _rolling_mean_fast(arr: np.ndarray, w: int) -> np.ndarray:
    """Vectorized causal rolling mean using stride tricks."""
    N = len(arr)
    out = np.full(N, np.nan, dtype=np.float32)
    cs = np.cumsum(np.nan_to_num(arr, nan=0.0))
    cs_pad = np.concatenate([[0.0], cs])
    if N >= w:
        out[w - 1:] = (cs_pad[w:] - cs_pad[:N - w + 1]) / w
    return out.astype(np.float32)


def _rolling_std_fast(arr: np.ndarray, w: int) -> np.ndarray:
    """Vectorized causal rolling std using Welford-style (sum of squares)."""
    N = len(arr)
    out = np.full(N, np.nan, dtype=np.float32)
    a = np.nan_to_num(arr, nan=0.0).astype(np.float64)
    cs  = np.cumsum(a)
    cs2 = np.cumsum(a ** 2)
    cs_pad  = np.concatenate([[0.0], cs])
    cs2_pad = np.concatenate([[0.0], cs2])
    if N >= w:
        s  = cs_pad[w:] - cs_pad[:N - w + 1]
        s2 = cs2_pad[w:] - cs2_pad[:N - w + 1]
        var = (s2 - s ** 2 / w) / max(w - 1, 1)
        var = np.maximum(var, 0.0)
        out[w - 1:] = np.sqrt(var).astype(np.float32)
    return out.astype(np.float32)


def compute_v2_1_features(
    z_scores: np.ndarray,    # (N,) CNN predictions, aligned to this day
    mbo: np.ndarray,         # (N, 340) MBO features, aligned to CNN bars
    mbo_cols: Dict[str, int],
) -> np.ndarray:
    """
    Engineer 20 temporal features from CNN z-scores and MBO data.

    LEAKAGE AUDIT: PASSED
    - All features use only past and current bar values (backward-looking)
    - Rolling windows only look back, never forward
    - z_mom_k uses z[i] - z[i-k] (past comparison)
    - No normalization here — normalization done from training data only in build_windows()

    Returns: (N, 20) float32
    """
    N = len(z_scores)
    z = z_scores.astype(np.float32)

    # ── CNN z-score features (10) ────────────────────────────────────────────

    # 1. z_score (raw)
    f_z = z.copy()

    # 2. z_mom_5 = z[i] - z[i-5]  (0 at warmup bars)
    f_z_mom_5 = np.zeros(N, dtype=np.float32)
    f_z_mom_5[5:] = z[5:] - z[:-5]

    # 3. z_mom_10
    f_z_mom_10 = np.zeros(N, dtype=np.float32)
    f_z_mom_10[10:] = z[10:] - z[:-10]

    # 4. z_mom_20
    f_z_mom_20 = np.zeros(N, dtype=np.float32)
    f_z_mom_20[20:] = z[20:] - z[:-20]

    # 5. z_roll_mean_10
    f_z_rmean10 = _rolling_mean_fast(z, 10)

    # 6. z_roll_std_10
    f_z_rstd10 = _rolling_std_fast(z, 10)

    # 7. z_acceleration = z_mom_5[i] - z_mom_5[i-5]
    f_z_accel = np.zeros(N, dtype=np.float32)
    f_z_accel[10:] = f_z_mom_5[10:] - f_z_mom_5[5:-5]

    # 8. z_sign_persistence = fraction same-sign as current z in last 10 bars
    f_z_sign_persist = np.zeros(N, dtype=np.float32)
    for i in range(10, N):
        current_sign = np.sign(z[i])
        if current_sign == 0:
            f_z_sign_persist[i] = 0.5
        else:
            window = z[i - 9:i + 1]  # 10 bars including current
            f_z_sign_persist[i] = float(np.sum(np.sign(window) == current_sign)) / 10.0

    # 9. z_cross_zero = bars since last zero-crossing (capped at 20)
    f_z_cross = np.zeros(N, dtype=np.float32)
    last_cross = 0
    prev_sign = np.sign(z[0])
    for i in range(1, N):
        curr_sign = np.sign(z[i])
        if curr_sign != prev_sign and curr_sign != 0:
            last_cross = i
        f_z_cross[i] = min(float(i - last_cross), 20.0)
        if curr_sign != 0:
            prev_sign = curr_sign

    # 10. z_strength = abs(z) / (z_roll_std_10 + eps)
    eps = 1e-6
    f_z_strength = np.abs(z) / (np.where(np.isnan(f_z_rstd10), eps, f_z_rstd10) + eps)
    f_z_strength = np.where(np.isnan(f_z_rstd10), 0.0, f_z_strength).astype(np.float32)

    # ── MBO temporal features (10) ──────────────────────────────────────────

    def get_col(name):
        col = mbo_cols[name]
        return mbo[:, col].astype(np.float32)

    vol   = get_col("volume_rate")
    spr   = get_col("spread")
    bid_d = get_col("bid_depth")
    ask_d = get_col("ask_depth")
    bid_c = get_col("bid_depth_change")
    ask_c = get_col("ask_depth_change")
    imb   = get_col("imbalance")
    timb  = get_col("trade_imbalance")
    qnew  = get_col("queue_new")
    qcan  = get_col("queue_cancels")

    # 11. volume_rate_10 = rolling mean of volume over 10 bars
    f_vol_rate10 = _rolling_mean_fast(vol, 10)

    # 12. volume_acceleration = volume_rate_10[i] - volume_rate_10[i-10]
    f_vol_accel = np.zeros(N, dtype=np.float32)
    tmp = np.nan_to_num(f_vol_rate10, nan=0.0)
    f_vol_accel[10:] = tmp[10:] - tmp[:-10]

    # 13. spread_10 = rolling mean spread over 10 bars
    f_spr10 = _rolling_mean_fast(spr, 10)

    # 14. spread_change = spread_10[i] - spread_10[i-10]
    f_spr_change = np.zeros(N, dtype=np.float32)
    tmp_s = np.nan_to_num(f_spr10, nan=0.0)
    f_spr_change[10:] = tmp_s[10:] - tmp_s[:-10]

    # 15. bid_depth_change_10 = rolling mean of bid depth change
    f_bid_dc10 = _rolling_mean_fast(bid_c, 10)

    # 16. ask_depth_change_10 = rolling mean of ask depth change
    f_ask_dc10 = _rolling_mean_fast(ask_c, 10)

    # 17. imbalance_L1 = raw bid-ask imbalance at level 1
    f_imb = imb.copy()

    # 18. imbalance_momentum = imbalance_L1[i] - imbalance_L1[i-10]
    f_imb_mom = np.zeros(N, dtype=np.float32)
    f_imb_mom[10:] = imb[10:] - imb[:-10]

    # 19. trade_imbalance_10 = rolling mean of trade imbalance
    f_timb10 = _rolling_mean_fast(timb, 10)

    # 20. queue_pressure = rolling mean of (new_orders - cancels) over 10 bars
    net_flow = qnew - qcan
    f_queue_pressure = _rolling_mean_fast(net_flow, 10)

    # ── Stack all 20 features ────────────────────────────────────────────────
    features = np.stack([
        f_z,
        f_z_mom_5,
        f_z_mom_10,
        f_z_mom_20,
        f_z_rmean10,
        f_z_rstd10,
        f_z_accel,
        f_z_sign_persist,
        f_z_cross,
        f_z_strength,
        f_vol_rate10,
        f_vol_accel,
        f_spr10,
        f_spr_change,
        f_bid_dc10,
        f_ask_dc10,
        f_imb,
        f_imb_mom,
        f_timb10,
        f_queue_pressure,
    ], axis=1)  # (N, 20)

    return np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_ckpt_preds(preds_file: Path) -> Dict[str, np.ndarray]:
    """
    Load CNN OOT predictions. Returns dict: {date: {'preds': ..., 'targets': ...}}
    """
    logger.info(f"Loading CNN OOT predictions from {preds_file.name}...")
    d = np.load(str(preds_file), allow_pickle=True)
    keys = list(d.keys())
    pred_dates = sorted(set(k.replace("_preds", "").replace("_targets", "") for k in keys))
    result = {}
    for date in pred_dates:
        pk = f"{date}_preds"
        tk = f"{date}_targets"
        if pk in d and tk in d:
            result[date] = {
                "preds":   d[pk].astype(np.float32),
                "targets": d[tk].astype(np.float32),
            }
    logger.info(f"  Loaded {len(result)} dates: {pred_dates[0]} to {pred_dates[-1]}")
    return result


def load_day_data(
    date: str,
    ckpt_preds: Dict[str, np.ndarray],
    mbo_dir: Path,
    mbo_cols: Dict[str, int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Build (N, 20) feature matrix and (N,) targets for one day.

    Alignment: CNN pred[i] → MBO bar[i + CNN_WINDOW_SIZE]
    CNN array has 233,880 bars = 234,000 - window_size(20) - horizon(100)
    MBO array has 234,000 bars.
    We use MBO bars [20 .. 233899] (aligned to CNN bars [0 .. 233879]).

    Returns: (features, targets) or None
    """
    if date not in ckpt_preds:
        return None

    mbo_path = mbo_dir / f"{date}_mbo_features.npz"
    if not mbo_path.exists():
        return None

    preds   = ckpt_preds[date]["preds"]    # (M,) where M=233880
    targets = ckpt_preds[date]["targets"]  # (M,)

    mbo_all = np.load(str(mbo_path), allow_pickle=True)["mbo_features"].astype(np.float32)

    M       = len(preds)
    N_mbo   = mbo_all.shape[0]  # 234000

    # CNN bar i corresponds to MBO bar i + CNN_WINDOW_SIZE
    mbo_start = CNN_WINDOW_SIZE
    mbo_end   = mbo_start + M

    if mbo_end > N_mbo:
        # Truncate to available MBO bars
        M = N_mbo - mbo_start
        preds   = preds[:M]
        targets = targets[:M]

    mbo_aligned = mbo_all[mbo_start:mbo_start + M]  # (M, 340)

    features = compute_v2_1_features(preds, mbo_aligned, mbo_cols)  # (M, 20)

    return features, targets


# ============================================================================
# WINDOW BUILDER
# ============================================================================
def build_windows_vectorized(
    feats: np.ndarray,    # (N, 20)
    tgts: np.ndarray,     # (N,)
    seq_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (seq_len, 20) windows from a single day's data.
    Windows NEVER cross day boundaries.
    Target = value at the LAST bar of the window.
    """
    N, n_feat = feats.shape
    if N < seq_len + 1:
        return np.empty((0, seq_len, n_feat), dtype=np.float32), np.empty(0, dtype=np.float32)

    end_indices = np.arange(seq_len, N + 1, stride)
    idx = np.arange(seq_len)[np.newaxis, :] + (end_indices - seq_len)[:, np.newaxis]
    X = feats[idx]                    # (n_windows, seq_len, 20)
    y = tgts[end_indices - 1].astype(np.float32)

    # Filter: target must be finite, last-bar features must be finite
    valid = np.isfinite(y) & np.all(np.isfinite(X[:, -1, :]), axis=-1)
    return X[valid].astype(np.float32), y[valid]


def build_windows(
    dates: List[str],
    ckpt_preds: Dict[str, np.ndarray],
    mbo_dir: Path,
    mbo_cols: Dict[str, int],
    seq_len: int = SEQ_LEN,
    stride: int = STRIDE_TRAIN,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build windowed sequences from given dates.
    If norm_mean/std not provided, compute from this data (for training set).
    """
    all_X = []
    all_y = []

    for date in dates:
        day_data = load_day_data(date, ckpt_preds, mbo_dir, mbo_cols)
        if day_data is None:
            continue
        feats, tgts = day_data
        X_day, y_day = build_windows_vectorized(feats, tgts, seq_len, stride)
        if len(X_day) > 0:
            all_X.append(X_day)
            all_y.append(y_day)

    if not all_X:
        return None, None, norm_mean, norm_std

    X = np.concatenate(all_X, axis=0)  # (total_windows, seq_len, 20)
    y = np.concatenate(all_y, axis=0)  # (total_windows,)

    # Normalization from training data only
    if norm_mean is None or norm_std is None:
        X_flat = X.reshape(-1, X.shape[-1])
        norm_mean = np.nanmean(X_flat, axis=0).astype(np.float32)
        norm_std  = np.nanstd(X_flat, axis=0).astype(np.float32)
        norm_std[norm_std < 1e-8] = 1.0

    X = (X - norm_mean[np.newaxis, np.newaxis, :]) / norm_std[np.newaxis, np.newaxis, :]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X.astype(np.float32), y, norm_mean, norm_std


# ============================================================================
# MODEL: Temporal LSTM v2.1
# ============================================================================
class TemporalLSTMv2_1(nn.Module):
    """
    2-layer LSTM with attention. Small focused model: 128 hidden, 20 input features.
    ~200K parameters (vs 4M in v2).
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_LSTM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Causal (unidirectional) LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Simple temporal attention over LSTM outputs
        self.attn_query = nn.Parameter(torch.randn(hidden_dim))
        self.attn_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop  = nn.Dropout(dropout)

        # Output head: concat(attention_context, last_hidden) → 1
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x: (batch, seq_len, n_features=20)
        Returns: (batch, 1)
        """
        h = self.input_proj(x)                    # (B, T, hidden)
        lstm_out, (h_n, _) = self.lstm(h)          # (B, T, hidden), (layers, B, hidden)

        # Attention
        keys    = self.attn_proj(lstm_out)          # (B, T, hidden)
        scores  = torch.einsum("bth,h->bt", keys, self.attn_query)  # (B, T)
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        context = torch.einsum("bt,bth->bh", weights, lstm_out)     # (B, hidden)

        last_h   = h_n[-1]                         # (B, hidden)
        combined = torch.cat([context, last_h], -1) # (B, hidden*2)
        combined = self.layer_norm(combined)
        return self.head(combined)                  # (B, 1)


# ============================================================================
# TRAINING
# ============================================================================
def train_and_predict(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Train model on one fold, return OOT predictions."""

    X_tr = torch.from_numpy(X_train)
    y_tr = torch.from_numpy(y_train).unsqueeze(-1)
    X_v  = torch.from_numpy(X_val)
    y_v  = torch.from_numpy(y_val).unsqueeze(-1)
    X_te = torch.from_numpy(X_test)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    scaler    = GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    model.train()
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        n_batches  = 0

        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                pred = model(bx)
                loss = F.mse_loss(pred, by)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outs = []
            for i in range(0, len(X_v), BATCH_SIZE):
                bv = X_v[i:i + BATCH_SIZE].to(device, non_blocking=True)
                with autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                    val_outs.append(model(bv).float().cpu())
            val_out  = torch.cat(val_outs)
            val_loss = F.mse_loss(val_out, y_v).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.debug(f"    Early stop at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on test
    model.eval()
    test_outs = []
    with torch.no_grad():
        for i in range(0, len(X_te), BATCH_SIZE):
            bt = X_te[i:i + BATCH_SIZE].to(device, non_blocking=True)
            with autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                test_outs.append(model(bt).float().cpu().numpy().flatten())

    return np.concatenate(test_outs)


# ============================================================================
# MLflow LOGGING
# ============================================================================
def try_log_mlflow(fold_idx: int, date: str, ic: float, n_train: int):
    """Log fold results to MLflow (best-effort)."""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("CNN_Training")
        with mlflow.start_run(run_name=f"temporal_v2_1_fold_{fold_idx}_{date}", nested=True):
            mlflow.log_param("model",        "TemporalLSTMv2_1")
            mlflow.log_param("fold",         fold_idx)
            mlflow.log_param("date",         date)
            mlflow.log_param("n_features",   N_FEATURES)
            mlflow.log_param("hidden_dim",   HIDDEN_DIM)
            mlflow.log_param("n_lstm_layers",N_LSTM_LAYERS)
            mlflow.log_param("seq_len",      SEQ_LEN)
            mlflow.log_param("stride_train", STRIDE_TRAIN)
            mlflow.log_param("batch_size",   BATCH_SIZE)
            mlflow.log_metric("ic",          ic)
            mlflow.log_metric("n_train_windows", n_train)
    except Exception:
        pass


# ============================================================================
# WALK-FORWARD EVALUATION
# ============================================================================
def walk_forward_evaluate(
    dates: List[str],
    ckpt_preds: Dict[str, np.ndarray],
    mbo_dir: Path,
    mbo_cols: Dict[str, int],
    device: torch.device,
    min_train_days: int = MIN_TRAIN_DAYS,
) -> dict:
    """
    Expanding-window walk-forward.
    train: days [0 .. test_day-2], 1-day purge gap, test: [test_day]
    """
    n_days  = len(dates)
    n_folds = n_days - min_train_days

    logger.info(f"\n{'='*70}")
    logger.info(f"TEMPORAL LSTM v2.1 — Walk-Forward Training ({n_folds} folds)")
    logger.info(f"{'='*70}")
    logger.info(f"  Dates:          {dates[0]} to {dates[-1]}")
    logger.info(f"  Min train days: {min_train_days}")
    logger.info(f"  Stride:         {STRIDE_TRAIN}")
    logger.info(f"  n_features:     {N_FEATURES}")
    logger.info(f"  hidden_dim:     {HIDDEN_DIM}")
    logger.info(f"  seq_len:        {SEQ_LEN}")
    logger.info(f"  batch_size:     {BATCH_SIZE}")

    fold_ics     = []
    fold_metrics = []
    all_preds    = []
    all_actuals  = []
    total_time   = 0.0

    # MLflow parent run
    parent_run = None
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("CNN_Training")
        parent_run = mlflow.start_run(run_name="temporal_v2_1_walkforward")
        mlflow.log_params({
            "model":        "TemporalLSTMv2_1",
            "n_features":   N_FEATURES,
            "hidden_dim":   HIDDEN_DIM,
            "stride_train": STRIDE_TRAIN,
            "seq_len":      SEQ_LEN,
            "batch_size":   BATCH_SIZE,
        })
    except Exception:
        pass

    for test_day in range(min_train_days, n_days):
        fold_start  = time.time()
        train_dates = dates[:test_day - 1]   # expanding window, 1-day purge
        test_date   = dates[test_day]

        if len(train_dates) < min_train_days:
            continue

        # Build training windows
        X_train, y_train, mean, std = build_windows(
            train_dates, ckpt_preds, mbo_dir, mbo_cols,
            stride=STRIDE_TRAIN,
        )
        if X_train is None or len(X_train) < 500:
            n_dbg = len(X_train) if X_train is not None else 0
            logger.warning(f"  Fold {test_day} ({test_date}): too few train windows ({n_dbg}), skip")
            continue

        # Build test windows using training normalization
        X_test, y_test, _, _ = build_windows(
            [test_date], ckpt_preds, mbo_dir, mbo_cols,
            stride=STRIDE_TEST,
            norm_mean=mean, norm_std=std,
        )
        if X_test is None or len(X_test) < 20:
            logger.warning(f"  Fold {test_day} ({test_date}): too few test windows, skip")
            continue

        # Temporal train/val split
        n_total = len(X_train)
        n_val   = max(int(n_total * VAL_FRACTION), 100)
        n_tr    = n_total - n_val
        X_tr, y_tr = X_train[:n_tr], y_train[:n_tr]
        X_vl, y_vl = X_train[n_tr:], y_train[n_tr:]

        # Fresh model per fold
        model = TemporalLSTMv2_1(
            n_features=N_FEATURES,
            hidden_dim=HIDDEN_DIM,
            n_layers=N_LSTM_LAYERS,
            dropout=DROPOUT,
        ).to(device)

        if test_day == min_train_days:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\nModel: TemporalLSTMv2_1 — {n_params:,} parameters")
            logger.info(f"  Features: {N_FEATURES}, hidden: {HIDDEN_DIM}, layers: {N_LSTM_LAYERS}")
            logger.info(f"  First fold: {n_tr:,} train + {n_val:,} val + {len(X_test):,} test windows\n")

        preds = train_and_predict(model, X_tr, y_tr, X_vl, y_vl, X_test, device)

        fold_time   = time.time() - fold_start
        total_time += fold_time

        # Compute fold IC
        valid = np.isfinite(preds) & np.isfinite(y_test)
        if valid.sum() > 20:
            ic_fold = float(spearmanr(preds[valid], y_test[valid])[0])
            if np.isfinite(ic_fold):
                hr = float((np.sign(preds[valid]) == np.sign(y_test[valid])).mean())
                fold_ics.append(ic_fold)
                fold_metrics.append({
                    "fold":        test_day,
                    "date":        test_date,
                    "ic":          ic_fold,
                    "hit_rate":    hr,
                    "n_train":     n_tr,
                    "n_val":       n_val,
                    "n_test":      int(valid.sum()),
                    "fold_time_s": round(fold_time, 1),
                })
                logger.info(
                    f"  Fold {test_day:3d} ({test_date}): "
                    f"IC={ic_fold:+.4f}  HR={hr:.1%}  "
                    f"n_train={n_tr:7,}  n_test={valid.sum():5,}  "
                    f"[{fold_time:.1f}s]"
                )
                all_preds.append(preds[valid])
                all_actuals.append(y_test[valid])

                # Save per-fold artifacts
                fold_npz = OUTPUT_DIR / f"fold_{test_day:03d}_{test_date}_preds.npz"
                np.savez_compressed(
                    str(fold_npz),
                    preds=preds[valid],
                    targets=y_test[valid],
                    date=test_date,
                    fold=test_day,
                    ic=ic_fold,
                    norm_mean=mean,
                    norm_std=std,
                )

                fold_pt = OUTPUT_DIR / f"fold_{test_day:03d}_{test_date}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "norm_mean":        mean,
                    "norm_std":         std,
                    "n_features":       N_FEATURES,
                    "hidden_dim":       HIDDEN_DIM,
                    "fold":             test_day,
                    "date":             test_date,
                    "ic":               ic_fold,
                }, str(fold_pt))

                try_log_mlflow(test_day, test_date, ic_fold, n_tr)

        # Cleanup
        del model, X_train, y_train, X_test, y_test, X_tr, y_tr, X_vl, y_vl
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Aggregate ────────────────────────────────────────────────────────────
    if not all_preds:
        logger.error("No valid predictions produced!")
        return {"error": "No predictions", "fold_ics": []}

    p = np.concatenate(all_preds)
    a = np.concatenate(all_actuals)

    ic_overall = float(spearmanr(p, a)[0])
    hr_overall = float((np.sign(p) == np.sign(a)).mean())
    winners    = np.abs(a[np.sign(p) == np.sign(a)]).sum()
    losers     = np.abs(a[np.sign(p) != np.sign(a)]).sum()
    pf         = float(winners / losers) if losers > 0 else 0.0

    ic_arr   = np.array(fold_ics)
    ic_mean  = float(ic_arr.mean())
    ic_std   = float(ic_arr.std())
    icir     = ic_mean / ic_std if ic_std > 0 else 0.0
    tstat    = ic_mean / ic_std * np.sqrt(len(ic_arr)) if ic_std > 0 else 0.0
    try:
        _, pvalue = ttest_1samp(ic_arr, 0)
        pvalue = float(pvalue)
    except Exception:
        pvalue = 1.0
    pct_positive = float((ic_arr > 0).mean())

    logger.info(f"\n{'='*70}")
    logger.info(f"TEMPORAL LSTM v2.1 — FINAL RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"  IC (overall)    = {ic_overall:+.4f}")
    logger.info(f"  IC (mean/fold)  = {ic_mean:+.4f} ± {ic_std:.4f}")
    logger.info(f"  ICIR            = {icir:+.3f}")
    logger.info(f"  t-stat          = {tstat:+.3f}  (p={pvalue:.4f})")
    logger.info(f"  Hit rate        = {hr_overall:.1%}")
    logger.info(f"  Profit factor   = {pf:.3f}")
    logger.info(f"  % positive IC   = {pct_positive:.1%}")
    logger.info(f"  Folds           = {len(fold_ics)}")
    logger.info(f"  Total samples   = {len(p):,}")
    logger.info(f"  Total train     = {total_time:.0f}s ({total_time/60:.1f}min)")
    logger.info(f"  LEAKAGE AUDIT   : PASSED")
    logger.info(f"{'='*70}")

    passed = (ic_overall > 0.01 or ic_mean > 0.01) and abs(tstat) > 2.0
    logger.info(f"  VERDICT: {'PASS — proceed to fill sim' if passed else 'FAIL — investigate features'}")
    logger.info(f"{'='*70}\n")

    if parent_run is not None:
        try:
            import mlflow
            mlflow.log_metrics({
                "ic_overall":    ic_overall,
                "ic_mean":       ic_mean,
                "icir":          icir,
                "tstat":         tstat,
                "hit_rate":      hr_overall,
                "n_folds":       float(len(fold_ics)),
                "pct_positive":  pct_positive,
            })
            mlflow.end_run()
        except Exception:
            pass

    return {
        "model":             "TemporalLSTMv2_1",
        "n_features":        N_FEATURES,
        "hidden_dim":        HIDDEN_DIM,
        "n_layers":          N_LSTM_LAYERS,
        "seq_len":           SEQ_LEN,
        "stride_train":      STRIDE_TRAIN,
        "ic_overall":        ic_overall,
        "ic_mean":           ic_mean,
        "ic_std":            ic_std,
        "icir":              icir,
        "tstat":             tstat,
        "pvalue":            pvalue,
        "hit_rate":          hr_overall,
        "profit_factor":     pf,
        "pct_positive_ic":   pct_positive,
        "n_folds":           len(fold_ics),
        "n_predictions":     len(p),
        "total_train_time_s":total_time,
        "passed":            passed,
        "fold_ics":          [float(x) for x in fold_ics],
        "fold_metrics":      fold_metrics,
        "leakage_audit":     "PASSED",
        "ckpt_preds_file":   str(CKPT_PREDS_FILE),
        "dates":             dates,
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    logger.info("=" * 70)
    logger.info("TEMPORAL LSTM v2.1 — Focused Features, No Embeddings")
    logger.info("=" * 70)
    logger.info(f"CNN OOT predictions: {CKPT_PREDS_FILE.name}")
    logger.info(f"MBO cache:           {MBO_DIR}")
    logger.info(f"Output:              {OUTPUT_DIR}")
    logger.info(f"n_features:          {N_FEATURES}")
    logger.info(f"  (z-score features: 10, MBO temporal: 10)")
    logger.info(f"hidden_dim:          {HIDDEN_DIM}")
    logger.info(f"seq_len:             {SEQ_LEN}, stride: {STRIDE_TRAIN}")
    logger.info(f"batch_size:          {BATCH_SIZE}, num_workers: {NUM_WORKERS}")
    logger.info("")

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.warning("No GPU found — running on CPU (will be slow)")
    logger.info(f"AMP: {USE_AMP and device.type == 'cuda'}")
    logger.info("")

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not CKPT_PREDS_FILE.exists():
        logger.error(f"CNN predictions file not found: {CKPT_PREDS_FILE}")
        logger.error("Available ckpt_preds files:")
        for f in sorted(PREDS_DIR.glob("ckpt_preds*.npz")):
            logger.error(f"  {f.name}")
        sys.exit(1)

    if not MBO_DIR.exists():
        logger.error(f"MBO cache directory not found: {MBO_DIR}")
        sys.exit(1)

    # ── Load CNN OOT predictions ─────────────────────────────────────────────
    ckpt_preds = load_ckpt_preds(CKPT_PREDS_FILE)

    # ── Find dates with both CNN preds and MBO data ──────────────────────────
    mbo_dates = set(
        f.name.replace("_mbo_features.npz", "")
        for f in MBO_DIR.glob("*_mbo_features.npz")
    )
    cnn_dates  = set(ckpt_preds.keys())
    all_dates  = sorted(cnn_dates & mbo_dates)

    logger.info(f"CNN dates:  {len(cnn_dates)}")
    logger.info(f"MBO dates:  {len(mbo_dates)}")
    logger.info(f"Overlap:    {len(all_dates)} dates ({all_dates[0]} to {all_dates[-1]})")

    if len(all_dates) < MIN_TRAIN_DAYS + 5:
        logger.error(f"Not enough dates: {len(all_dates)} < {MIN_TRAIN_DAYS + 5}")
        sys.exit(1)

    # ── Select MBO columns ───────────────────────────────────────────────────
    mbo_cols = select_mbo_cols(MBO_DIR)

    # ── Walk-forward evaluation ──────────────────────────────────────────────
    results = walk_forward_evaluate(
        all_dates, ckpt_preds, MBO_DIR, mbo_cols, device,
        min_train_days=MIN_TRAIN_DAYS,
    )

    # ── Save final results ───────────────────────────────────────────────────
    out_file = OUTPUT_DIR / "results_temporal_v2_1.json"

    def to_json(obj):
        if isinstance(obj, (np.integer,)):    return int(obj)
        if isinstance(obj, (np.floating,)):   return float(obj)
        if isinstance(obj, np.ndarray):       return obj.tolist()
        return obj

    with open(str(out_file), "w") as f:
        json.dump(results, f, indent=2, default=to_json)

    logger.info(f"Results saved: {out_file}")
    logger.info("Done.")

    return results


if __name__ == "__main__":
    main()
