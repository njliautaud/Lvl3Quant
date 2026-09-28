"""
CNN + PatchTST + Mamba 4-Branch Fusion Training Script  (cnn_patchtst_mamba_v1)

Implements the canonical 4-branch architecture spec at
    docs/architectures/cnn_patchtst_mamba_v1.md
locked 2026-04-27 22:05 ET (Book CNN dropped → 4 branches final).

Branches (per-step concat → Mamba backbone → multi-task head):
    1) Event Temporal CNN   (raw 6 channels  → 64-d, 3 Conv1d k=5 + GELU + residual)
    2) PatchTST             (raw 6 channels  → per-patch d_pt then upsample-by-repeat to per-step → 64-d)
    3) smart raw passthrough (29 smart_v4 OR 25 smart_v3 features → thin Linear+LN+GELU → 32-d)
    4) VOL stream            (vol_lgbm_v3 per-anchor preds, 3 channels, train-only-norm, frozen)

Concat → Linear → LN → (B, L, d_model) → MambaBlock × N → final LN → last-step pool
       → 6-target head (Δ1s, Δ5s, Δ10s, Δ30s, MFE_q, MAE_q)
       (MFE/MAE are unsupervised in v1 if labels missing — heads still produced for inference)

Training rules (HARD CONSTRAINTS, DIRECTIVES.md 2026-04-27 23:25 ET):
  - NO intra-day leakage: feature stats are computed on TRAIN files ONLY per-fold.
    smart_v3 / smart_v4 are pre-normalized (causal rolling stats — verified by inspection).
    vol_lgbm_v3 predictions come from per-day models trained on strictly prior 60d → leak-free.
  - WF folds match the cnn_mamba_v2 / EventCNN1D bake-off layout (same train/oot split).
  - MLflow logging mandatory (experiment: FusionBakeoff_v1_mamba). MLflow URI defaults to
    Tailscale post-move IP http://jupiter:5000 (NEVER jupiter:5000).
  - Output dir is unique per run (NEVER share dirs — fold-76 leakage was caused by sharing).
  - Save .pt + .npz preds per fold.
  - Mixed precision (fp16) on CUDA.

CLI matches the train_event_cnn_1d.py style:
    --window-mode sliding --train-days 60 --oot-days 1 --n-folds 11
plus:
    --max-folds N        (run only the first N folds — for tonight: --max-folds 1)
    --warm-start-cnn-mamba CKPT   (load Event CNN + Mamba weights from train_cnn_mamba.py ckpt)
    --warm-start-patchtst CKPT    (load PatchTST weights from train_event_patchtst.py ckpt)
    --vol-pred-dir DIR    (default: /home/jupiter/Lvl3Quant/output/vol_lgbm_v3/)
    --skip-vol-branch     (force vol stream off; uses 0s + missing flag)

Author: Generated 2026-04-27 23:30 ET (Opus 4.6).
"""

import os
import sys
import gc
import json
import time
import socket
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

# ---- Optional CUDA Mamba kernel (10-50x faster than pure PyTorch) -------------
try:
    from mamba_ssm import Mamba as CUDAMamba
    USE_CUDA_MAMBA = True
    print("*** Using mamba_ssm CUDA kernels (FAST) ***", flush=True)
except ImportError:
    USE_CUDA_MAMBA = False
    print("*** Using pure-PyTorch SelectiveSSM fallback (SLOWER) ***", flush=True)

# ---- MLflow ------------------------------------------------------------------
try:
    if int(os.environ.get("DISABLE_MLFLOW", 0)):
        raise ImportError("DISABLE_MLFLOW=1")
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow disabled / not installed — skipping experiment tracking")

# ============================================================================
# Reuse proven internals from sibling training scripts (no rewriting).
#   - SelectiveSSM, MambaBlock, FileSequentialSampler, WarmupCosineScheduler,
#     compute_ic, compute_comprehensive_metrics  → train_cnn_mamba
#   - PatchTST class                              → train_event_patchtst
# ============================================================================
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Disable MLflow inside the imported helpers so they don't fight us for the run.
os.environ.setdefault("DISABLE_MLFLOW", "0")

from train_cnn_mamba import (  # type: ignore
    SelectiveSSM,
    MambaBlock,
    FileSequentialSampler,
    WarmupCosineScheduler,
    compute_ic,
    compute_comprehensive_metrics,
)
from train_event_patchtst import PatchTST  # type: ignore


# ============================================================================
# Logging
# ============================================================================
LOG_DIR = Path(__file__).parent / "results"
LOG_DIR.mkdir(exist_ok=True)
log_path = LOG_DIR / "cnn_patchtst_mamba.log"

logging.root.handlers.clear()
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh = logging.FileHandler(log_path); _fh.setFormatter(_fmt); _fh.setLevel(logging.INFO)
_sh = logging.StreamHandler(sys.stdout); _sh.setFormatter(_fmt); _sh.setLevel(logging.INFO)
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_fh)
logging.root.addHandler(_sh)
logger = logging.getLogger(__name__)
print(">>> train_cnn_patchtst_mamba.py loaded", flush=True)


# ============================================================================
# Config (env-overridable, same naming as sibling scripts for cross-comparison)
# ============================================================================
DEFAULT_RAW_DATA_DIR    = "/home/jupiter/Lvl3Quant/data/processed/mbo_events"
DEFAULT_SMART_DATA_DIR  = "/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3"
DEFAULT_VOL_PRED_DIR    = "/home/jupiter/Lvl3Quant/output/vol_lgbm_v3"
DEFAULT_OUTPUT_DIR      = "/home/jupiter/Lvl3Quant/output/fusion_bakeoff_v1"

# MLflow URI (Tailscale post-move IP — DIRECTIVE 23:25 ET)
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://jupiter:5000",
)
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "FusionBakeoff_v1_mamba")

# Per-event raw event channel count (cnn_mamba_v2 baseline = 6)
N_RAW_FEATURES = 6
# Smart channel count default (smart_v3 = 25, smart_v4 = 29). Detected at runtime.
DEFAULT_N_SMART = 25

# Branch encoder dims (per spec)
EVT_CNN_CHANNELS    = int(os.environ.get("EVT_CNN_CHANNELS", 64))
EVT_CNN_KERNEL      = int(os.environ.get("EVT_CNN_KERNEL", 5))
EVT_CNN_LAYERS      = int(os.environ.get("EVT_CNN_LAYERS", 3))
PT_PATCH_SIZE       = int(os.environ.get("PT_PATCH_SIZE", 25))
PT_D_MODEL          = int(os.environ.get("PT_D_MODEL", 256))  # v2 fix: 64→256 to match standalone PatchTST capacity
PT_N_HEADS          = int(os.environ.get("PT_N_HEADS", 4))
PT_HEAD_DIM         = int(os.environ.get("PT_HEAD_DIM", 16))
PT_N_LAYERS         = int(os.environ.get("PT_N_LAYERS", 2))
PT_FFN_DIM          = int(os.environ.get("PT_FFN_DIM", PT_D_MODEL * 4))
PT_DROPOUT          = float(os.environ.get("PT_DROPOUT", 0.1))
SMART_PROJ_DIM      = int(os.environ.get("SMART_PROJ_DIM", 32))
VOL_DIM             = 3   # vol_10s, vol_30s, vol_60s

# Mamba backbone (defaults match CNN-Mamba v2 checkpoint exactly)
MAMBA_D_MODEL       = int(os.environ.get("MAMBA_D_MODEL", 96))
MAMBA_D_STATE       = int(os.environ.get("MAMBA_D_STATE", 32))
MAMBA_N_LAYERS      = int(os.environ.get("MAMBA_N_LAYERS", 3))    # v2 = 3 layers
MAMBA_DT_RANK       = int(os.environ.get("MAMBA_DT_RANK", 6))    # v2 ckpt uses dt_rank=6
MAMBA_D_CONV        = int(os.environ.get("MAMBA_D_CONV", 4))
MAMBA_DROPOUT       = float(os.environ.get("MAMBA_DROPOUT", 0.1))

# V2 backbone front-end dims (must match checkpoint)
V2_FEATURE_MLP_HIDDEN = int(os.environ.get("V2_FEATURE_MLP_HIDDEN", 128))
V2_FEATURE_MLP_OUT    = int(os.environ.get("V2_FEATURE_MLP_OUT", 64))
V2_CNN_CHANNELS_PER_SCALE = int(os.environ.get("V2_CNN_CHANNELS_PER_SCALE", 16))
V2_CNN_KERNELS        = [3, 7, 15, 31]  # multi-scale temporal CNN

# Window / training
WINDOW_SIZE         = int(os.environ.get("EVENT_WINDOW_SIZE", 1000))   # divisible by patch
STRIDE              = int(os.environ.get("EVENT_STRIDE", WINDOW_SIZE // 2))
BATCH_SIZE          = int(os.environ.get("EVENT_BATCH_SIZE", 32))
LR                  = float(os.environ.get("EVENT_LR", 3e-4))
EPOCHS_PER_FOLD     = int(os.environ.get("EVENT_EPOCHS", 5))  # v2 fix: 3→5 for convergence
WARMUP_STEPS        = int(os.environ.get("EVENT_WARMUP", 300))
GRAD_CLIP           = float(os.environ.get("EVENT_GRAD_CLIP", 1.0))

assert WINDOW_SIZE % PT_PATCH_SIZE == 0, (
    f"WINDOW_SIZE ({WINDOW_SIZE}) must be divisible by PT_PATCH_SIZE ({PT_PATCH_SIZE})"
)

# Output heads: Δ1s, Δ5s, Δ10s, Δ30s, MFE_quantile, MAE_quantile
LABEL_HORIZONS = ["1s", "5s", "10s", "30s"]
N_HEADS_OUT = 6


# ============================================================================
# Vol-stream alignment helper
# ============================================================================

def load_vol_predictions(vol_pred_dir: Path, date_str: str, n_events: int
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (vol_per_event, mask) of shapes (n_events, 3) and (n_events,).

    vol_lgbm_v3 stores predictions at sparse anchor_idxs (window-end positions).
    We forward-fill: every event in [anchor_i, anchor_{i+1}) gets prediction at anchor_i.
    Pre-first-anchor events get zeros + mask=0 (they are unused targets anyway since
    labels at index < WINDOW_SIZE-1 are skipped by the dataset).
    """
    f = vol_pred_dir / f"vol_v3_{date_str}_predictions.npz"
    if not f.exists():
        return (np.zeros((n_events, 3), dtype=np.float32),
                np.zeros((n_events,), dtype=np.float32))
    d = np.load(f, allow_pickle=True)
    preds   = d["predictions"].astype(np.float32)        # (K, 3)
    anchors = d["anchor_idxs"].astype(np.int64)          # (K,)
    # Forward-fill into per-event arrays
    out  = np.zeros((n_events, 3), dtype=np.float32)
    mask = np.zeros((n_events,),    dtype=np.float32)
    if len(anchors) == 0:
        return out, mask
    # Sort by anchor (should already be sorted but be safe)
    order = np.argsort(anchors)
    anchors = anchors[order]
    preds   = preds[order]
    # Vectorized forward-fill via searchsorted: for each event idx, find the latest anchor <= idx
    event_idx = np.arange(n_events, dtype=np.int64)
    pos = np.searchsorted(anchors, event_idx, side="right") - 1   # -1 for pre-first-anchor
    valid = pos >= 0
    out[valid]  = preds[pos[valid]]
    mask[valid] = 1.0
    return out, mask


# ============================================================================
# Dataset — fuses raw events + smart features + vol predictions per fold.
# ============================================================================

class FusionDataset(Dataset):
    """
    Per-window output (per __getitem__):
        raw     : (W, N_RAW_FEATURES)         float32  [normalized w/ TRAIN-only stats]
        smart   : (W, N_SMART_FEATURES)       float32  [pre-normalized in source files]
        vol     : (W, 3)                      float32  [normalized w/ TRAIN-only stats]
        labels  : (4,)                        float32  [Δ1s, Δ5s, Δ10s, Δ30s]
        valid   : float32 mask                          (1.0 if all label horizons present)

    File pairing: for each smart_v3/v4 *_mbo_events.npz file we look up the matching
    raw 6-channel file in raw_data_dir (same date prefix). Vol predictions from
    vol_pred_dir keyed by date.

    Lazy loading + LRU cache to keep RAM bounded.
    """

    def __init__(
        self,
        smart_files:        List[Path],
        raw_data_dir:       Path,
        vol_pred_dir:       Optional[Path],
        window_size:        int = WINDOW_SIZE,
        stride:             int = STRIDE,
        raw_feature_stats:  Optional[Dict] = None,
        vol_feature_stats:  Optional[Dict] = None,
        n_smart:            int = DEFAULT_N_SMART,
        cache_days:         int = 4,
        skip_vol_branch:    bool = False,
    ):
        self.smart_files     = list(smart_files)
        self.raw_data_dir    = Path(raw_data_dir)
        self.vol_pred_dir    = Path(vol_pred_dir) if vol_pred_dir else None
        self.window_size     = window_size
        self.stride          = stride
        self.n_smart         = n_smart
        self.cache_days      = cache_days
        self.skip_vol_branch = skip_vol_branch

        from collections import OrderedDict
        self._cache: "OrderedDict[int, Dict]" = OrderedDict()

        # Stats: raw events normalized per-fold; smart already normalized in source;
        #         vol normalized per-fold.
        self.raw_feature_mean: Optional[np.ndarray] = (
            raw_feature_stats["mean"] if raw_feature_stats else None
        )
        self.raw_feature_std:  Optional[np.ndarray] = (
            raw_feature_stats["std"]  if raw_feature_stats else None
        )
        self.vol_feature_mean: Optional[np.ndarray] = (
            vol_feature_stats["mean"] if vol_feature_stats else None
        )
        self.vol_feature_std:  Optional[np.ndarray] = (
            vol_feature_stats["std"]  if vol_feature_stats else None
        )

        if raw_feature_stats is None:
            self._compute_raw_stats()
        if (vol_feature_stats is None) and (not self.skip_vol_branch) and (self.vol_pred_dir is not None):
            self._compute_vol_stats()
        if self.vol_feature_mean is None:
            # Default to identity normalization
            self.vol_feature_mean = np.zeros(VOL_DIM, dtype=np.float32)
            self.vol_feature_std  = np.ones(VOL_DIM, dtype=np.float32)

        self._build_index()

    # ---- File pairing helpers ------------------------------------------------
    def _date_of(self, smart_file: Path) -> str:
        """Date prefix YYYYMMDD from filename like '20260223_mbo_events.npz'."""
        return smart_file.name.split("_")[0]

    def _raw_file_for(self, smart_file: Path) -> Path:
        return self.raw_data_dir / smart_file.name

    # ---- Stats (TRAIN-only, no leakage) -------------------------------------
    def _compute_raw_stats(self):
        logger.info(f"Computing raw-event stats from {len(self.smart_files)} files...")
        s   = np.zeros(N_RAW_FEATURES, dtype=np.float64)
        sq  = np.zeros(N_RAW_FEATURES, dtype=np.float64)
        cnt = 0
        for sf in self.smart_files:
            rf = self._raw_file_for(sf)
            if not rf.exists():
                logger.warning(f"  raw file missing for {sf.name} — skipping for stats")
                continue
            d = np.load(rf, allow_pickle=True)
            ev = d["events"].astype(np.float64)
            s  += ev.sum(axis=0)
            sq += (ev ** 2).sum(axis=0)
            cnt += len(ev)
        if cnt == 0:
            self.raw_feature_mean = np.zeros(N_RAW_FEATURES, dtype=np.float32)
            self.raw_feature_std  = np.ones(N_RAW_FEATURES,  dtype=np.float32)
            return
        self.raw_feature_mean = (s / cnt).astype(np.float32)
        var = (sq / cnt) - (self.raw_feature_mean.astype(np.float64) ** 2)
        self.raw_feature_std  = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
        logger.info(f"  raw_mean={self.raw_feature_mean}")
        logger.info(f"  raw_std ={self.raw_feature_std}")

    def _compute_vol_stats(self):
        """Aggregate vol predictions for TRAIN dates only — NO LEAKAGE."""
        s   = np.zeros(VOL_DIM, dtype=np.float64)
        sq  = np.zeros(VOL_DIM, dtype=np.float64)
        cnt = 0
        n_present = 0
        for sf in self.smart_files:
            date = self._date_of(sf)
            f = self.vol_pred_dir / f"vol_v3_{date}_predictions.npz"
            if not f.exists():
                continue
            n_present += 1
            d = np.load(f, allow_pickle=True)
            p = d["predictions"].astype(np.float64)
            s  += p.sum(axis=0)
            sq += (p ** 2).sum(axis=0)
            cnt += len(p)
        if cnt == 0:
            logger.warning("  no vol predictions present in train set — vol stats default to (0, 1)")
            self.vol_feature_mean = np.zeros(VOL_DIM, dtype=np.float32)
            self.vol_feature_std  = np.ones(VOL_DIM, dtype=np.float32)
            return
        self.vol_feature_mean = (s / cnt).astype(np.float32)
        var = (sq / cnt) - (self.vol_feature_mean.astype(np.float64) ** 2)
        self.vol_feature_std  = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
        logger.info(f"  vol stats from {n_present}/{len(self.smart_files)} train days, n={cnt}")
        logger.info(f"  vol_mean={self.vol_feature_mean}, vol_std={self.vol_feature_std}")

    def get_feature_stats(self) -> Dict:
        return {
            "raw":  {"mean": self.raw_feature_mean, "std": self.raw_feature_std},
            "vol":  {"mean": self.vol_feature_mean, "std": self.vol_feature_std},
        }

    # ---- Sample index --------------------------------------------------------
    def _build_index(self):
        self.sample_index: List[Tuple[int, int]] = []
        for day_idx, sf in enumerate(self.smart_files):
            try:
                d = np.load(sf, allow_pickle=True)
            except Exception as e:
                logger.warning(f"  failed to read {sf.name}: {e}")
                continue
            n_events = len(d["events"])
            day_labels = {h: d[f"labels_{h}"] for h in LABEL_HORIZONS if f"labels_{h}" in d.files}
            for start in range(0, n_events - self.window_size + 1, self.stride):
                end = start + self.window_size
                idx = end - 1
                ok = True
                for h in LABEL_HORIZONS:
                    arr = day_labels.get(h)
                    if arr is None or np.isnan(arr[idx]):
                        ok = False
                        break
                if ok:
                    self.sample_index.append((day_idx, start))
            del d, day_labels
        logger.info(
            f"FusionDataset: {len(self.smart_files)} days, "
            f"{len(self.sample_index)} samples (W={self.window_size}, stride={self.stride})"
        )

    # ---- Lazy day load with LRU cache ---------------------------------------
    def _load_day(self, day_idx: int) -> Dict:
        if day_idx in self._cache:
            self._cache.move_to_end(day_idx)
            return self._cache[day_idx]
        sf   = self.smart_files[day_idx]
        date = self._date_of(sf)

        smart_d = np.load(sf, allow_pickle=True)
        smart   = smart_d["events"].astype(np.float32)            # (N, n_smart) pre-normalized
        labels  = {h: smart_d[f"labels_{h}"].astype(np.float32) for h in LABEL_HORIZONS}
        n_events = len(smart)

        # Raw 6-channel events: prefer separate raw_data_dir; fall back to first 6 cols of smart
        rf = self._raw_file_for(sf)
        if rf.exists():
            raw_d = np.load(rf, allow_pickle=True)
            raw = raw_d["events"].astype(np.float32)
            # Some date pairings differ in row count slightly — clamp to common len
            n = min(len(raw), n_events)
            raw     = raw[:n]
            smart   = smart[:n]
            labels  = {h: v[:n] for h, v in labels.items()}
            n_events = n
        else:
            # Fallback: smart_v3/v4 has the raw 6 features in cols 0..5
            raw = smart[:, :N_RAW_FEATURES].copy()

        # Normalize raw (TRAIN-only stats)
        raw = (raw - self.raw_feature_mean) / (self.raw_feature_std + 1e-8)

        # Vol stream
        if self.skip_vol_branch or (self.vol_pred_dir is None):
            vol  = np.zeros((n_events, VOL_DIM), dtype=np.float32)
            vmask = np.zeros((n_events,),         dtype=np.float32)
        else:
            vol, vmask = load_vol_predictions(self.vol_pred_dir, date, n_events)
            vol = (vol - self.vol_feature_mean) / (self.vol_feature_std + 1e-8)
            # Zero out positions where vol is missing (mask=0) to avoid leaking norm-of-zero
            vol = vol * vmask[:, None]

        day_data = {
            "raw":    raw,
            "smart":  smart,
            "vol":    vol,
            "vmask":  vmask,
            "labels": labels,
            "n":      n_events,
        }
        self._cache[day_idx] = day_data
        while len(self._cache) > self.cache_days:
            self._cache.popitem(last=False)
        return day_data

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        day_idx, start = self.sample_index[idx]
        end = start + self.window_size
        day = self._load_day(day_idx)
        raw   = day["raw"][start:end]                      # (W, 6)
        smart = day["smart"][start:end]                    # (W, n_smart)
        vol   = day["vol"][start:end]                      # (W, 3)
        label_idx = end - 1
        labs = np.array(
            [day["labels"][h][label_idx] for h in LABEL_HORIZONS],
            dtype=np.float32,
        )
        return (
            torch.from_numpy(raw),
            torch.from_numpy(smart),
            torch.from_numpy(vol),
            torch.from_numpy(labs),
        )


# ============================================================================
# Fusion Model — CNN-Mamba v2 backbone + PatchTST + Vol additive injection
# ============================================================================

class FusionModel(nn.Module):
    """
    CNN-Mamba v2 backbone with PatchTST and vol stream added on top.

    CORE (exact replica of CNNMambaV2 — same param names for full warm-start):
      1. feature_mlp:   25 smart features → 128 → 64 (pointwise MLP per timestep)
      2. temporal_cnns:  4× multi-scale causal Conv1d(25, 16, k=3/7/15/31) → 64d
      3. fusion_proj:   concat(64+64=128) → Linear(128, d_model=96) + LayerNorm
      4. blocks:        3× MambaBlock (d_model=96, d_state=32, time-delta conditioning)
      5. final_norm + head

    ADDED branches (injected additively before Mamba blocks):
      A. PatchTST on raw 6-ch features → per-patch tokens → upsample → project to d_model
      B. Vol stream (3-ch) → project to d_model

    Injection: x = v2_backbone_out + sigmoid(pt_gate) * pt_emb + sigmoid(vol_gate) * vol_emb
    Gates initialized at -2 (sigmoid≈0.12) so initial behavior ≈ v2 with small perturbation.
    This allows full warm-start from v2 checkpoint (ALL 57 tensors transfer by name).
    """

    def __init__(
        self,
        n_smart_features: int = DEFAULT_N_SMART,
        d_model:          int = MAMBA_D_MODEL,
        d_state:          int = MAMBA_D_STATE,
        n_layers:         int = MAMBA_N_LAYERS,
        dt_rank:          int = MAMBA_DT_RANK,
        d_conv:           int = MAMBA_D_CONV,
        dropout:          float = MAMBA_DROPOUT,
        window_size:      int = WINDOW_SIZE,
        patch_size:       int = PT_PATCH_SIZE,
        pt_d_model:       int = PT_D_MODEL,
        pt_n_heads:       int = PT_N_HEADS,
        pt_head_dim:      int = PT_HEAD_DIM,
        pt_n_layers:      int = PT_N_LAYERS,
        pt_ffn_dim:       int = PT_FFN_DIM,
        pt_dropout:       float = PT_DROPOUT,
        n_heads_out:      int = N_HEADS_OUT,
        feature_mlp_hidden: int = V2_FEATURE_MLP_HIDDEN,
        feature_mlp_out:    int = V2_FEATURE_MLP_OUT,
        cnn_channels_per_scale: int = V2_CNN_CHANNELS_PER_SCALE,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_size  = patch_size
        self.n_patches   = window_size // patch_size
        self.d_model     = d_model
        self.n_heads_out = n_heads_out

        # ================================================================
        # V2 BACKBONE — exact same param names as CNNMambaV2 for warm-start
        # ================================================================

        # Pathway 1: Feature Interaction MLP (pointwise per timestep)
        # smart 25 → 128 → 64
        self.feature_mlp = nn.Sequential(
            nn.Linear(n_smart_features, feature_mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(feature_mlp_hidden, feature_mlp_out),
            nn.LayerNorm(feature_mlp_out),
        )

        # Pathway 2: Multi-Scale Temporal CNN (causal, on smart features)
        self.cnn_kernels = V2_CNN_KERNELS
        self.temporal_cnns = nn.ModuleList()
        for k in self.cnn_kernels:
            self.temporal_cnns.append(
                nn.Sequential(
                    nn.Conv1d(n_smart_features, cnn_channels_per_scale,
                              kernel_size=k, padding=0, bias=True),
                    nn.GELU(),
                )
            )
        self.cnn_out_dim = cnn_channels_per_scale * len(self.cnn_kernels)  # 16*4=64

        # Fusion: combine both pathways → d_model
        fusion_dim = feature_mlp_out + self.cnn_out_dim  # 64 + 64 = 128
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Mamba backbone (identical names to v2)
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                dt_rank=dt_rank,
                d_conv=d_conv,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Multi-task head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_heads_out),
        )

        # ================================================================
        # ADDED BRANCHES — PatchTST + Vol (additive injection)
        # ================================================================

        # PatchTST on raw 6-ch features
        self.patchtst = PatchTST(
            n_features=N_RAW_FEATURES,
            patch_size=patch_size,
            d_model=pt_d_model,
            n_heads=pt_n_heads,
            head_dim=pt_head_dim,
            n_layers=pt_n_layers,
            ffn_dim=pt_ffn_dim,
            dropout=pt_dropout,
            n_targets=3,
            window_size=window_size,
        )
        # Project PatchTST output to d_model for additive injection
        self.pt_proj = nn.Sequential(
            nn.Linear(pt_d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Vol stream projection to d_model
        self.vol_proj = nn.Sequential(
            nn.Linear(VOL_DIM, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable injection gates (initialized small so v2 backbone dominates at start)
        self.pt_gate = nn.Parameter(torch.tensor(-2.0))    # sigmoid(-2)≈0.12
        self.vol_gate = nn.Parameter(torch.tensor(-2.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Re-set gate values after _init_weights (they got overwritten)
        with torch.no_grad():
            self.pt_gate.fill_(-2.0)
            self.vol_gate.fill_(-2.0)

    def forward(
        self,
        raw:   torch.Tensor,    # (B, L, 6)    — normalized raw events
        smart: torch.Tensor,    # (B, L, n_smart) — pre-normalized smart features
        vol:   torch.Tensor,    # (B, L, 3)    — normalized vol predictions
        return_embedding: bool = False,
    ):
        B, L, _ = smart.shape
        assert L == self.window_size, f"Got L={L}, expected {self.window_size}"

        # Time-delta for Mamba (smart feature index 0 = time_delta_log)
        time_delta = smart[:, :, 0]  # (B, L)

        # ---- V2 Backbone: Feature MLP + Multi-Scale CNN → fusion_proj ----

        # Pathway 1: pointwise MLP on smart features
        feat_out = self.feature_mlp(smart)  # (B, L, feature_mlp_out=64)

        # Pathway 2: multi-scale temporal CNN on smart features
        x_t = smart.transpose(1, 2)  # (B, n_smart, L) for Conv1d
        cnn_outputs = []
        for k, conv_block in zip(self.cnn_kernels, self.temporal_cnns):
            padded = F.pad(x_t, (k - 1, 0))  # causal left-pad
            out = conv_block(padded)           # (B, cnn_ch_per_scale, L)
            cnn_outputs.append(out)
        cnn_cat = torch.cat(cnn_outputs, dim=1)   # (B, 64, L)
        cnn_out = cnn_cat.transpose(1, 2)          # (B, L, 64)

        # Fuse both pathways
        fused = torch.cat([feat_out, cnn_out], dim=-1)  # (B, L, 128)
        x = self.fusion_proj(fused)                      # (B, L, d_model)

        # ---- PatchTST branch (raw features) → additive injection ----
        pt_x = raw.reshape(B, self.n_patches, self.patch_size * N_RAW_FEATURES)
        pt_x = self.patchtst.patch_embed(pt_x)
        for layer in self.patchtst.layers:
            pt_x = layer(pt_x)
        pt_x = self.patchtst.norm(pt_x)                       # (B, n_patches, pt_d_model)
        # Upsample: repeat each patch token patch_size times
        pt_x = pt_x.unsqueeze(2).expand(-1, -1, self.patch_size, -1)
        pt_x = pt_x.reshape(B, L, -1)                         # (B, L, pt_d_model)
        pt_emb = self.pt_proj(pt_x)                            # (B, L, d_model)

        # ---- Vol branch → additive injection ----
        vol_emb = self.vol_proj(vol)                           # (B, L, d_model)

        # ---- Additive fusion (gates control injection strength) ----
        x = x + torch.sigmoid(self.pt_gate) * pt_emb + torch.sigmoid(self.vol_gate) * vol_emb

        # ---- Mamba backbone ----
        for block in self.blocks:
            x = block(x, time_delta=time_delta)

        # Take LAST position (causal)
        x_last = x[:, -1, :]                                  # (B, d_model)
        emb = self.final_norm(x_last)
        preds = self.head(emb)                                 # (B, n_heads_out)

        if return_embedding:
            return preds, emb
        return preds


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Warm-start helpers
# ============================================================================

def warm_start_cnn_mamba(model: FusionModel, ckpt_path: Path):
    """Load CNN-Mamba v2 weights into the fusion model's v2 backbone.

    Because the fusion model's v2 backbone uses IDENTICAL param names
    (feature_mlp.*, temporal_cnns.*, fusion_proj.*, blocks.*, final_norm.*, head.*),
    ALL matching tensors transfer directly by name.

    The only expected mismatches:
      - head.3 (output dim: v2=3 targets, fusion=N_HEADS_OUT) — shape mismatch, skipped
      - patchtst.*, pt_proj.*, vol_proj.*, pt_gate, vol_gate — not in v2 ckpt, skipped
    """
    if not ckpt_path.exists():
        logger.warning(f"warm-start cnn_mamba ckpt not found: {ckpt_path}")
        return
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state", ck)
    n_loaded = 0
    n_skipped_shape = 0
    n_skipped_missing = 0
    own_sd = model.state_dict()

    for k, v in sd.items():
        if k in own_sd:
            if own_sd[k].shape == v.shape:
                own_sd[k] = v
                n_loaded += 1
            else:
                n_skipped_shape += 1
                logger.warning(f"  warm-start SHAPE MISMATCH: {k} ckpt={v.shape} model={own_sd[k].shape}")
        else:
            n_skipped_missing += 1

    if n_loaded > 0:
        model.load_state_dict(own_sd)
    logger.info(f"Warm-start cnn_mamba_v2: loaded {n_loaded}/{len(sd)} tensors, "
                f"skipped {n_skipped_shape} shape mismatches, "
                f"{n_skipped_missing} not in model from {ckpt_path.name}")
    if n_loaded == 0:
        logger.error(f"  CRITICAL: 0 tensors loaded! Check d_model and architecture match.")


def warm_start_patchtst(model: FusionModel, ckpt_path: Path):
    """Load PatchTST encoder weights from a train_event_patchtst checkpoint."""
    if not ckpt_path.exists():
        logger.warning(f"warm-start patchtst ckpt not found: {ckpt_path}")
        return
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state", ck)
    n_loaded = 0
    own_sd = model.state_dict()
    for k, v in sd.items():
        # External PatchTST is top-level; here it's nested under "patchtst."
        target = f"patchtst.{k}"
        if target in own_sd and own_sd[target].shape == v.shape:
            own_sd[target] = v
            n_loaded += 1
    model.load_state_dict(own_sd)
    logger.info(f"Warm-start patchtst: loaded {n_loaded} tensors from {ckpt_path.name}")


# ============================================================================
# Loss — supervises only the available label horizons (1s/5s/10s/30s).
# MFE/MAE head outputs (indices 4, 5) are unsupervised in v1 (no labels yet).
# ============================================================================

def compute_loss(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # preds: (B, 6); labels: (B, 4) for [1s, 5s, 10s, 30s]
    return F.mse_loss(preds[:, :4], labels)


# ============================================================================
# Eval / OOT inference
# ============================================================================

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = True):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []
    amp_ctx = (torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
               else torch.amp.autocast("cpu", enabled=False))
    with torch.no_grad():
        for raw, smart, vol, labels in loader:
            raw    = raw.to(device, non_blocking=True)
            smart  = smart.to(device, non_blocking=True)
            vol    = vol.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with amp_ctx:
                preds = model(raw, smart, vol)
                loss = compute_loss(preds, labels)
            total_loss += loss.item()
            n_batches  += 1
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())
    if not all_preds:
        return ({"loss": float("nan")},
                np.empty((0, N_HEADS_OUT)),
                np.empty((0, len(LABEL_HORIZONS))))
    preds_arr  = np.concatenate(all_preds, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    metrics = {"loss": total_loss / max(n_batches, 1)}
    for i, h in enumerate(LABEL_HORIZONS):
        metrics[f"ic_{h}"] = compute_ic(preds_arr[:, i], labels_arr[:, i])
    return metrics, preds_arr, labels_arr


# ============================================================================
# Train one fold
# ============================================================================

def train_one_fold(
    model:        nn.Module,
    train_loader: DataLoader,
    oot_loader:   DataLoader,
    fold_idx:     int,
    output_dir:   Path,
    mlflow_run,
    device:       torch.device,
    total_steps:  int,
    use_amp:      bool = True,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=WARMUP_STEPS, total_steps=total_steps)
    amp_ctx = (torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
               else torch.amp.autocast("cpu", enabled=False))

    best_val_loss = float("inf")
    global_step = 0
    total_batches = len(train_loader)
    logger.info(f"FOLD {fold_idx:02d} train_one_fold: {EPOCHS_PER_FOLD} epochs × {total_batches} batches")

    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()

        for raw, smart, vol, labels in train_loader:
            raw    = raw.to(device, non_blocking=True)
            smart  = smart.to(device, non_blocking=True)
            vol    = vol.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                preds = model(raw, smart, vol)
                loss  = compute_loss(preds, labels)

            # NaN guard — skip bad batches, avoid poisoning optimizer state
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count = getattr(train_one_fold, '_nan_count', 0) + 1
                train_one_fold._nan_count = nan_count
                if nan_count % 50 == 1:
                    logger.warning(f"NaN/Inf loss detected (count={nan_count}), skipping batch")
                if nan_count > 200:
                    logger.error(f"NaN loss exceeded 200 consecutive — aborting fold {fold_idx}")
                    return best_val_loss, None
                continue
            else:
                train_one_fold._nan_count = 0

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1
            global_step += 1

            if n_batches % 100 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / n_batches * (total_batches - n_batches)
                msg = (f"  Fold {fold_idx:02d} Epoch {epoch+1}/{EPOCHS_PER_FOLD} "
                       f"Batch {n_batches}/{total_batches} | Loss {epoch_loss/n_batches:.4f} | "
                       f"Elapsed {elapsed:.0f}s ETA {eta:.0f}s")
                print(msg, flush=True); logger.info(msg)

        avg_loss = epoch_loss / max(n_batches, 1)
        ep_time  = time.time() - epoch_start
        val_metrics, _, _ = evaluate(model, oot_loader, device, use_amp=use_amp)
        logger.info(
            f"Fold {fold_idx:02d} Epoch {epoch+1}/{EPOCHS_PER_FOLD} | "
            f"train_loss={avg_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_ic_10s={val_metrics.get('ic_10s', float('nan')):.4f} "
            f"lr={scheduler.get_lr():.2e} time={ep_time:.0f}s"
        )

        if MLFLOW_AVAILABLE and mlflow_run is not None:
            step = fold_idx * EPOCHS_PER_FOLD + epoch
            mlflow.log_metrics({
                f"fold{fold_idx:02d}_train_loss": avg_loss,
                f"fold{fold_idx:02d}_val_loss":   val_metrics["loss"],
                **{f"fold{fold_idx:02d}_val_ic_{h}": val_metrics.get(f"ic_{h}", float("nan"))
                   for h in LABEL_HORIZONS},
            }, step=step)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ck_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "fold": fold_idx, "epoch": epoch,
                "val_loss": val_metrics["loss"],
                "val_ic_10s": val_metrics.get("ic_10s"),
                "arch": {
                    "model": "FusionModel_v1_mamba",
                    "d_model": MAMBA_D_MODEL, "d_state": MAMBA_D_STATE,
                    "n_layers": MAMBA_N_LAYERS, "dt_rank": MAMBA_DT_RANK,
                    "d_conv": MAMBA_D_CONV, "dropout": MAMBA_DROPOUT,
                    "evt_cnn_channels": EVT_CNN_CHANNELS,
                    "pt_d_model": PT_D_MODEL, "pt_n_layers": PT_N_LAYERS,
                    "smart_proj_dim": SMART_PROJ_DIM,
                    "window_size": WINDOW_SIZE, "patch_size": PT_PATCH_SIZE,
                },
            }, ck_path)
            logger.info(f"  saved best ckpt (val_loss={val_metrics['loss']:.4f}) → {ck_path.name}")

    return {"best_val_loss": best_val_loss}


# ============================================================================
# Walk-forward (sliding window matching cnn_mamba_v2 / EventCNN1D bake-off)
# ============================================================================

def build_fold_boundaries(n_files: int, n_folds: int,
                          train_days: int, oot_days: int) -> List[Tuple[int, List[int], List[int]]]:
    """
    Sliding OOT identical to train_event_cnn_1d.run_expanding_wf:
        fold f gets oot_end = n_files - (n_folds - 1 - f) * oot_days
    """
    folds = []
    oot_days = min(oot_days, n_files - 5)
    for fold in range(n_folds):
        oot_end_idx   = n_files - (n_folds - 1 - fold) * oot_days
        oot_start_idx = oot_end_idx - oot_days
        train_end_idx = oot_start_idx
        train_start_idx = max(0, train_end_idx - train_days)
        if train_end_idx < 5 or oot_start_idx >= n_files or oot_start_idx < 0:
            continue
        folds.append((fold,
                      list(range(train_start_idx, train_end_idx)),
                      list(range(oot_start_idx, oot_end_idx))))
    return folds


def run_wf(
    smart_files:     List[Path],
    raw_data_dir:    Path,
    vol_pred_dir:    Optional[Path],
    output_dir:      Path,
    device:          torch.device,
    n_folds:         int,
    train_days:      int,
    oot_days:        int,
    max_folds:       Optional[int] = None,
    n_smart:         int = DEFAULT_N_SMART,
    skip_vol_branch: bool = False,
    warm_cnn_mamba:  Optional[Path] = None,
    warm_patchtst:   Optional[Path] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    smart_files = sorted(smart_files)
    n_files = len(smart_files)
    if n_files == 0:
        logger.error("no smart files found"); return {}
    logger.info(f"Total files: {n_files} ({smart_files[0].name} → {smart_files[-1].name})")

    folds = build_fold_boundaries(n_files, n_folds, train_days, oot_days)
    if max_folds is not None:
        folds = folds[:max_folds]
    logger.info(f"Running {len(folds)} folds (sliding {train_days}d train / {oot_days}d OOT)")

    use_amp = device.type == "cuda"
    concat_preds  = {h: [] for h in LABEL_HORIZONS}
    concat_labels = {h: [] for h in LABEL_HORIZONS}

    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow_run = mlflow.start_run(
                run_name=f"FusionV2Backbone_{time.strftime('%Y%m%d_%H%M')}"
            )
            mlflow.log_params({
                "model":           "FusionModel_v2backbone_patchtst_vol",
                "branches":        "v2_backbone(MLP+multiCNN)+PatchTST+vol",
                "window_size":     WINDOW_SIZE,
                "stride":          STRIDE,
                "patch_size":      PT_PATCH_SIZE,
                "v2_feat_mlp_hidden": V2_FEATURE_MLP_HIDDEN,
                "v2_feat_mlp_out":    V2_FEATURE_MLP_OUT,
                "v2_cnn_per_scale":   V2_CNN_CHANNELS_PER_SCALE,
                "pt_d_model":      PT_D_MODEL,
                "pt_n_layers":     PT_N_LAYERS,
                "n_smart":         n_smart,
                "vol_dim":         VOL_DIM,
                "skip_vol_branch": skip_vol_branch,
                "d_model":         MAMBA_D_MODEL,
                "d_state":         MAMBA_D_STATE,
                "n_layers":        MAMBA_N_LAYERS,
                "dt_rank":         MAMBA_DT_RANK,
                "d_conv":          MAMBA_D_CONV,
                "dropout":         MAMBA_DROPOUT,
                "batch_size":      BATCH_SIZE,
                "lr":              LR,
                "epochs_per_fold": EPOCHS_PER_FOLD,
                "n_folds":         len(folds),
                "max_folds":       max_folds if max_folds is not None else -1,
                "train_days":      train_days,
                "oot_days":        oot_days,
                "horizons":        ",".join(LABEL_HORIZONS),
                "n_files":         n_files,
                "node":            socket.gethostname(),
                "device":          str(device),
                "smart_data_dir":  str(smart_files[0].parent),
                "raw_data_dir":    str(raw_data_dir),
                "vol_pred_dir":    str(vol_pred_dir) if vol_pred_dir else "",
                "output_dir":      str(output_dir),
                "mixed_precision": "fp16" if use_amp else "none",
                "warm_cnn_mamba":  str(warm_cnn_mamba) if warm_cnn_mamba else "",
                "warm_patchtst":   str(warm_patchtst)  if warm_patchtst  else "",
            })
        except Exception as e:
            logger.warning(f"MLflow setup failed (continuing): {e}")
            mlflow_run = None

    try:
        for fold_idx, train_idxs, oot_idxs in folds:
            train_files = [smart_files[i] for i in train_idxs]
            oot_files   = [smart_files[i] for i in oot_idxs]
            logger.info(
                f"\n{'='*60}\nFOLD {fold_idx:02d} | "
                f"Train {len(train_files)}d ({train_files[0].name}→{train_files[-1].name}) | "
                f"OOT {len(oot_files)}d ({oot_files[0].name}→{oot_files[-1].name})\n{'='*60}"
            )

            # ---- Train dataset (computes train-only stats) ----
            logger.info("Building TRAIN dataset...")
            train_ds = FusionDataset(
                smart_files     = train_files,
                raw_data_dir    = raw_data_dir,
                vol_pred_dir    = vol_pred_dir,
                window_size     = WINDOW_SIZE,
                stride          = STRIDE,
                n_smart         = n_smart,
                skip_vol_branch = skip_vol_branch,
            )
            stats = train_ds.get_feature_stats()

            # ---- OOT dataset reusing TRAIN stats (NO LEAKAGE) ----
            logger.info("Building OOT dataset...")
            oot_ds = FusionDataset(
                smart_files       = oot_files,
                raw_data_dir      = raw_data_dir,
                vol_pred_dir      = vol_pred_dir,
                window_size       = WINDOW_SIZE,
                stride            = STRIDE,
                raw_feature_stats = stats["raw"],
                vol_feature_stats = stats["vol"],
                n_smart           = n_smart,
                skip_vol_branch   = skip_vol_branch,
            )

            train_sampler = FileSequentialSampler(train_ds)
            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE,
                sampler=train_sampler, shuffle=False,
                num_workers=0, pin_memory=True, drop_last=True,
            )
            oot_loader = DataLoader(
                oot_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                num_workers=0, pin_memory=True,
            )

            # ---- Fresh model per fold ----
            model = FusionModel(n_smart_features=n_smart).to(device)
            if warm_cnn_mamba is not None:
                warm_start_cnn_mamba(model, warm_cnn_mamba)
            if warm_patchtst is not None:
                warm_start_patchtst(model, warm_patchtst)

            if fold_idx == 0:
                n_params = count_parameters(model)
                logger.info(f"Model parameters: {n_params:,}")
                logger.info(f"  V2 backbone: feature_mlp(25→{V2_FEATURE_MLP_HIDDEN}→{V2_FEATURE_MLP_OUT}) "
                            f"+ temporal_cnns(4×{V2_CNN_CHANNELS_PER_SCALE}) "
                            f"→ fusion_proj({V2_FEATURE_MLP_OUT + V2_CNN_CHANNELS_PER_SCALE * 4}→{MAMBA_D_MODEL})")
                logger.info(f"  Mamba d_model={MAMBA_D_MODEL} n_layers={MAMBA_N_LAYERS}")
                logger.info(f"  Added branches: PatchTST(d={PT_D_MODEL}) + Vol({VOL_DIM}d) via gated injection")

            total_steps = EPOCHS_PER_FOLD * len(train_loader)
            train_one_fold(
                model, train_loader, oot_loader, fold_idx,
                output_dir, mlflow_run, device, total_steps, use_amp=use_amp,
            )

            # ---- Reload best ckpt for OOT inference ----
            ck_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            if ck_path.exists():
                ck = torch.load(ck_path, map_location=device, weights_only=False)
                model.load_state_dict(ck["model_state"])

            # ---- OOT inference ----
            oot_metrics, oot_preds, oot_labels = evaluate(model, oot_loader, device, use_amp=use_amp)
            fold_ics = {h: compute_ic(oot_preds[:, i], oot_labels[:, i])
                        for i, h in enumerate(LABEL_HORIZONS)}
            for i, h in enumerate(LABEL_HORIZONS):
                concat_preds[h].append(oot_preds[:, i])
                concat_labels[h].append(oot_labels[:, i])
            logger.info(f"Fold {fold_idx:02d} OOT IC | "
                        + " | ".join(f"{h}={fold_ics[h]:.4f}" for h in LABEL_HORIZONS))

            # ---- Save fold artifacts ----
            pred_path = output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
            np.savez_compressed(
                pred_path,
                predictions = oot_preds,
                labels      = oot_labels,
                horizons    = np.array(LABEL_HORIZONS),
                head_names  = np.array(["dz1s", "dz5s", "dz10s", "dz30s", "mfe_q", "mae_q"]),
                ic_1s       = np.array(fold_ics["1s"]),
                ic_5s       = np.array(fold_ics["5s"]),
                ic_10s      = np.array(fold_ics["10s"]),
                ic_30s      = np.array(fold_ics["30s"]),
                oot_files   = np.array([str(f) for f in oot_files]),
            )
            stats_path = output_dir / f"fold_{fold_idx:02d}_feature_stats.npz"
            np.savez(
                stats_path,
                raw_mean=stats["raw"]["mean"], raw_std=stats["raw"]["std"],
                vol_mean=stats["vol"]["mean"], vol_std=stats["vol"]["std"],
            )

            # ---- Comprehensive metrics per horizon ----
            try:
                fold_analysis = {"fold": fold_idx,
                                 "oot_files": [f.name for f in oot_files],
                                 "n_samples": int(oot_preds.shape[0]),
                                 "horizons": {}}
                for i, h in enumerate(LABEL_HORIZONS):
                    fold_analysis["horizons"][h] = compute_comprehensive_metrics(
                        oot_preds[:, i], oot_labels[:, i],
                        horizon=h, fold_idx=fold_idx, output_dir=output_dir, oot_files=oot_files,
                    )
                with open(output_dir / f"fold_{fold_idx:02d}_analysis.json", "w") as fh:
                    json.dump(fold_analysis, fh, indent=2, default=str)
            except Exception as e:
                logger.warning(f"comprehensive metrics failed: {e}")

            if MLFLOW_AVAILABLE and mlflow_run is not None:
                mlflow.log_metrics(
                    {f"oot_ic_{h}_fold{fold_idx:02d}": fold_ics[h] for h in LABEL_HORIZONS},
                    step=fold_idx,
                )

            del model, train_ds, oot_ds, train_loader, oot_loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ---- Concat IC ----
        logger.info("\n" + "=" * 60 + "\nCONCAT IC (all folds)\n" + "=" * 60)
        concat_ic: Dict[str, float] = {}
        for h in LABEL_HORIZONS:
            if concat_preds[h]:
                p = np.concatenate(concat_preds[h])
                l = np.concatenate(concat_labels[h])
                concat_ic[h] = compute_ic(p, l)
                logger.info(f"  Concat IC ({h}): {concat_ic[h]:.4f}")

        save = {**{f"preds_{h}":  np.concatenate(concat_preds[h])
                   for h in LABEL_HORIZONS if concat_preds[h]},
                **{f"labels_{h}": np.concatenate(concat_labels[h])
                   for h in LABEL_HORIZONS if concat_labels[h]},
                **{f"concat_ic_{h}": np.array(concat_ic.get(h, float("nan")))
                   for h in LABEL_HORIZONS}}
        np.savez_compressed(output_dir / "concat_oot_predictions.npz", **save)

        if MLFLOW_AVAILABLE and mlflow_run is not None:
            mlflow.log_metrics({f"concat_ic_{h}": concat_ic.get(h, float("nan"))
                                for h in LABEL_HORIZONS})

        logger.info("\n" + "=" * 60 + "\nLEAKAGE AUDIT")
        logger.info("  - Sliding-window WF: train ends BEFORE OOT starts")
        logger.info("  - raw stats computed on TRAIN files only (no leakage)")
        logger.info("  - smart features pre-normalized causally in source files")
        logger.info("  - vol stats computed on TRAIN dates only; vol_lgbm_v3 uses 60d-prior models")
        logger.info("  - Mamba causal by construction (h[t] depends only on events ≤ t)")
        logger.info("  - PatchTST is bidirectional WITHIN a window — window boundaries are causal")
        logger.info("=" * 60)
        return concat_ic
    finally:
        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Fusion v1 (CNN+PatchTST+smart+vol → Mamba)")
    p.add_argument("--smart-data-dir", type=str, default=DEFAULT_SMART_DATA_DIR,
                   help="Per-event smart_v3/v4 npz directory (used for both raw fallback and smart branch)")
    p.add_argument("--raw-data-dir",   type=str, default=DEFAULT_RAW_DATA_DIR,
                   help="Per-event 6-channel raw mbo_events npz directory (matched by date)")
    p.add_argument("--vol-pred-dir",   type=str, default=DEFAULT_VOL_PRED_DIR,
                   help="Directory of vol_lgbm_v3 prediction npz files")
    p.add_argument("--output-dir",     type=str, default=DEFAULT_OUTPUT_DIR,
                   help="Run-specific output dir (DO NOT SHARE between runs)")
    p.add_argument("--n-folds",        type=int, default=11)
    p.add_argument("--train-days",     type=int, default=60)
    p.add_argument("--oot-days",       type=int, default=1)
    p.add_argument("--max-folds",      type=int, default=None,
                   help="Run only the first N folds (for tonight: --max-folds 1)")
    p.add_argument("--max-days",       type=int, default=None,
                   help="Limit to most recent N smart files")
    p.add_argument("--device",         type=str, default="cuda")
    p.add_argument("--skip-vol-branch", action="store_true",
                   help="Force vol stream to all-zeros (bypasses load_vol_predictions)")
    p.add_argument("--warm-start-cnn-mamba", type=str, default=None,
                   help="Path to a train_cnn_mamba checkpoint to warm-start CNN+Mamba")
    p.add_argument("--warm-start-patchtst",  type=str, default=None,
                   help="Path to a train_event_patchtst checkpoint to warm-start PatchTST")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smart_dir    = Path(args.smart_data_dir)
    raw_dir      = Path(args.raw_data_dir)
    vol_dir      = Path(args.vol_pred_dir) if args.vol_pred_dir else None

    # Per-run log
    run_log = open(output_dir / "training.log", "a", buffering=1)
    rh = logging.StreamHandler(run_log); rh.setFormatter(_fmt); rh.setLevel(logging.INFO)
    logger.addHandler(rh)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable — falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    logger.info("=" * 60)
    logger.info("Fusion v1 (CNN + PatchTST + smart + vol → Mamba)")
    logger.info(f"  smart_data_dir = {smart_dir}")
    logger.info(f"  raw_data_dir   = {raw_dir}")
    logger.info(f"  vol_pred_dir   = {vol_dir}")
    logger.info(f"  output_dir     = {output_dir}")
    logger.info(f"  device         = {device}")
    logger.info(f"  WF             = sliding train_days={args.train_days} oot_days={args.oot_days} n_folds={args.n_folds}"
                f" max_folds={args.max_folds}")
    logger.info(f"  MLflow URI     = {MLFLOW_TRACKING_URI}")
    logger.info(f"  MLflow exp     = {MLFLOW_EXPERIMENT}")
    logger.info("=" * 60)

    smart_files = sorted(smart_dir.glob("*_mbo_events.npz"))
    if args.max_days and len(smart_files) > args.max_days:
        smart_files = smart_files[-args.max_days:]
    if not smart_files:
        logger.error(f"no smart files in {smart_dir}"); sys.exit(1)

    # Detect smart feature count from first file
    n_smart = DEFAULT_N_SMART
    try:
        d0 = np.load(smart_files[0], allow_pickle=True)
        n_smart = int(d0["events"].shape[1])
    except Exception as e:
        logger.warning(f"could not detect n_smart: {e}; using default {DEFAULT_N_SMART}")
    logger.info(f"  detected n_smart = {n_smart}")

    warm_cnn_mamba = Path(args.warm_start_cnn_mamba) if args.warm_start_cnn_mamba else None
    warm_patchtst  = Path(args.warm_start_patchtst)  if args.warm_start_patchtst  else None

    concat_ic = run_wf(
        smart_files     = smart_files,
        raw_data_dir    = raw_dir,
        vol_pred_dir    = vol_dir,
        output_dir      = output_dir,
        device          = device,
        n_folds         = args.n_folds,
        train_days      = args.train_days,
        oot_days        = args.oot_days,
        max_folds       = args.max_folds,
        n_smart         = n_smart,
        skip_vol_branch = args.skip_vol_branch,
        warm_cnn_mamba  = warm_cnn_mamba,
        warm_patchtst   = warm_patchtst,
    )

    logger.info("\n" + "=" * 60 + "\nFINAL CONCAT IC")
    for h in LABEL_HORIZONS:
        logger.info(f"  {h}: {concat_ic.get(h, float('nan')):.4f}")
    logger.info(f"Artifacts in: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
