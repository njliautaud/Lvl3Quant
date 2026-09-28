#!/usr/bin/env python3
"""
Model Decay Test V2 — Mar 16 → Apr 29, 2026 (ALL unseen data post-training)

Tests CNN-Mamba v2, PatchTST, AND LGBM Vol models on all available post-training dates.
- v1 had a bug: April dates used old smart_v3 files with broken labels (IC=0).
  Those files have been reprocessed as of 2026-04-30 10:07 AM.
- v1 was missing 19 dates (Mar 30 → Apr 20).
- v1 did not include LGBM Vol.

This v2 tests all 39 available dates and adds LGBM Vol as a third model.

CPU inference on Jupiter (no GPU).
"""

import os
import sys
import time
import json
import pickle
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

warnings.filterwarnings("ignore")

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR_SMART = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
DATA_DIR_RAW   = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")

# ALL post-training dates (trained up to ~Mar 15, 2026)
# 39 dates covering ~45 calendar days
TEST_DATES = [
    # Week 1: Mar 16-20 (days 1-5)
    "20260316", "20260317", "20260318", "20260319", "20260320",
    # Week 2: Mar 22-27 (days 7-12)
    "20260322", "20260323", "20260324", "20260325", "20260326", "20260327",
    # Week 3: Mar 29-31 (days 14-16)
    "20260329", "20260330", "20260331",
    # Week 4: Apr 1-3 (days 17-19)
    "20260401", "20260402", "20260403",
    # Week 4-5: Apr 5-10 (days 21-26)
    "20260405", "20260406", "20260407", "20260408", "20260409", "20260410",
    # Week 5-6: Apr 12-17 (days 28-33)
    "20260412", "20260413", "20260414", "20260415", "20260416", "20260417",
    # Week 6: Apr 19-20 (days 35-36)
    "20260419", "20260420",
    # Week 6-7: Apr 21-24 (days 37-40)
    "20260421", "20260422", "20260423", "20260424",
    # Week 7: Apr 26-29 (days 42-45)
    "20260426", "20260427", "20260428", "20260429",
]

# CNN-Mamba v2 config
CNNMAMBA_WEIGHTS = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar/fold_10_best.pt")
CNNMAMBA_STATS   = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar/fold_09_feature_stats.npz")
CNNMAMBA_WINDOW  = 3000
CNNMAMBA_STRIDE  = 3000  # Non-overlapping (CPU is too slow for dense stride)

# PatchTST config
PATCHTST_WEIGHTS = Path("/home/jupiter/Lvl3Quant/output/patchtst_razer_weights/fold_17_best.pt")
PATCHTST_WINDOW  = 500
PATCHTST_STRIDE  = 500  # Non-overlapping for speed

# LGBM Vol config
LGBM_MODEL_DIR = Path("/home/jupiter/Lvl3Quant/models/lgbm_60_5_fold35")
LGBM_HORIZONS  = ["1s", "5s", "10s", "30s"]  # LGBM has 30s too

# Baselines from training period
BASELINE_IC = {
    "CNN-Mamba v2": {"IC_1s": 0.040, "IC_5s": 0.065, "IC_10s": 0.085},
    "PatchTST":     {"IC_1s": 0.035, "IC_5s": 0.055, "IC_10s": 0.075},
    "LGBM Vol":     {"IC_1s": 0.122, "IC_5s": 0.055, "IC_10s": 0.041},  # from fold_meta.json
}

DEVICE = torch.device("cpu")
MAX_WINDOWS_PER_DAY = 500  # Cap for CPU feasibility


# =============================================================================
# Model definitions — CNN-Mamba v2 (standalone, matching saved state dict)
# =============================================================================

class SelectiveSSM(nn.Module):
    """Selective State Space Model — exact replica from train_cnn_mamba.py."""

    def __init__(self, d_model: int, d_state: int = 32, dt_rank: int = 6, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.d_inner = d_model * 2

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=0, groups=self.d_inner, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.time_decay_rate = nn.Parameter(torch.ones(self.d_inner, 1) * 0.1)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, time_delta: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_conv = x_branch.transpose(1, 2).contiguous()
        x_conv = F.pad(x_conv, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.transpose(1, 2).contiguous()
        x_branch = F.silu(x_conv)

        x_proj = self.x_proj(x_branch)
        dt_x = x_proj[:, :, :self.dt_rank]
        B_sel = x_proj[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C_sel = x_proj[:, :, self.dt_rank + self.d_state:]

        dt = F.softplus(self.dt_proj(dt_x))
        A = -torch.exp(self.A_log)

        # Sequential scan
        y = self._scan(x_branch, dt, A, B_sel, C_sel, self.D, time_delta)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _scan(self, x, dt, A, B, C, D, time_delta=None):
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        A_expanded = A.unsqueeze(0).unsqueeze(0)
        dt_expanded = dt.unsqueeze(-1)
        dA = torch.exp(A_expanded * dt_expanded)

        if time_delta is not None:
            td = time_delta.unsqueeze(-1).unsqueeze(-1)
            dr = F.softplus(self.time_decay_rate).unsqueeze(0).unsqueeze(0)
            time_decay = torch.exp(-dr * td.abs())
            dA = dA * time_decay

        dB = B.unsqueeze(2) * dt_expanded
        dBx = dB * x.unsqueeze(-1)

        CHUNK = 256
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t_start in range(0, seq_len, CHUNK):
            t_end = min(t_start + CHUNK, seq_len)
            chunk_outputs = []
            for t in range(t_start, t_end):
                h = dA[:, t] * h + dBx[:, t]
                y_t = torch.einsum("bn,bdn->bd", C[:, t], h)
                chunk_outputs.append(y_t)
            outputs.extend(chunk_outputs)
        return torch.stack(outputs, dim=1) + x * D.unsqueeze(0).unsqueeze(0)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 32, dt_rank: int = 6,
                 d_conv: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model=d_model, d_state=d_state, dt_rank=dt_rank, d_conv=d_conv)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_delta=None):
        residual = x
        x = self.norm(x)
        x = self.ssm(x, time_delta=time_delta)
        x = self.dropout(x)
        return x + residual


class CNNMambaV2(nn.Module):
    """CNN-Mamba v2: Feature MLP + Multi-Scale Temporal CNN + Mamba backbone."""

    CNN_KERNELS = [3, 7, 15, 31]

    def __init__(
        self,
        n_smart_features: int = 25,
        d_model: int = 96,
        d_state: int = 32,
        n_layers: int = 3,
        dt_rank: int = 6,
        d_conv: int = 4,
        dropout: float = 0.1,
        n_targets: int = 3,
        feature_mlp_hidden: int = 128,
        feature_mlp_out: int = 64,
        cnn_channels_per_scale: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_targets = n_targets

        self.feature_mlp = nn.Sequential(
            nn.Linear(n_smart_features, feature_mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(feature_mlp_hidden, feature_mlp_out),
            nn.LayerNorm(feature_mlp_out),
        )

        self.cnn_kernels = self.CNN_KERNELS
        self.temporal_cnns = nn.ModuleList()
        for k in self.cnn_kernels:
            self.temporal_cnns.append(
                nn.Sequential(
                    nn.Conv1d(n_smart_features, cnn_channels_per_scale,
                              kernel_size=k, padding=0, bias=True),
                    nn.GELU(),
                )
            )
        self.cnn_out_dim = cnn_channels_per_scale * len(self.cnn_kernels)

        fusion_dim = feature_mlp_out + self.cnn_out_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, dt_rank=dt_rank,
                       d_conv=d_conv, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )

    def forward(self, smart: torch.Tensor):
        B, L, _ = smart.shape
        time_delta = smart[:, :, 0]

        feat_out = self.feature_mlp(smart)

        x_t = smart.transpose(1, 2)
        cnn_outputs = []
        for k, conv_block in zip(self.cnn_kernels, self.temporal_cnns):
            padded = F.pad(x_t, (k - 1, 0))
            out = conv_block(padded)
            cnn_outputs.append(out)
        cnn_cat = torch.cat(cnn_outputs, dim=1)
        cnn_out = cnn_cat.transpose(1, 2)

        fused = torch.cat([feat_out, cnn_out], dim=-1)
        x = self.fusion_proj(fused)

        for block in self.blocks:
            x = block(x, time_delta=time_delta)

        x = self.final_norm(x)
        last_hidden = x[:, -1, :]
        preds = self.head(last_hidden)
        return preds


# =============================================================================
# PatchTST — import from existing code
# =============================================================================

os.environ["TST_FEATURE_SET"] = "smart_v3"
os.environ["SKIP_NORMALIZE"] = "1"
os.environ["DISABLE_MLFLOW"] = "1"

_DEEP_MODELS_DIR = str(Path(__file__).resolve().parent)
if _DEEP_MODELS_DIR not in sys.path:
    sys.path.insert(0, _DEEP_MODELS_DIR)

import logging
logging.disable(logging.INFO)
from train_event_patchtst import PatchTST, ALiBiAttention, TransformerBlock
logging.disable(logging.NOTSET)


# =============================================================================
# LGBM — Feature computation (from streaming_features.py _compute_derived_reference)
# =============================================================================

def compute_lgbm_features(ev: np.ndarray) -> np.ndarray:
    """
    Compute the 21-dim features expected by the LGBM model.
    Input: (N, 6) raw MBO events [time_delta, event_type, side, price, qty, spread]
    Output: (N, 21) = [6 raw + 15 derived]

    This is the exact same logic as _compute_derived_reference in streaming_features.py.
    """
    N = len(ev)
    td = ev[:, 0].astype('f4')
    et = ev[:, 1].astype('f4')
    side = ev[:, 2].astype('f4')
    price = ev[:, 3].astype('f4')
    qty = ev[:, 4].astype('f4')
    sprd = ev[:, 5].astype('f4')

    W20 = np.ones(20, 'f4') / 20
    W50 = np.ones(50, 'f4') / 50
    W100 = np.ones(100, 'f4') / 100
    W200 = np.ones(200, 'f4') / 200
    W500 = np.ones(500, 'f4') / 500

    is_t = (et == 3).astype('f4')
    is_c = (et == 1).astype('f4')
    is_a = (et == 0).astype('f4')
    is_bid = (side == 0).astype('f4')
    is_ask = (side == 1).astype('f4')

    bf = is_t * is_ask * qty
    sf = is_t * is_bid * qty
    ofi = bf - sf

    rofi  = np.convolve(ofi, W100, mode='full')[:N]
    casym = np.convolve(is_c * is_bid - is_c * is_ask, W100, mode='full')[:N]
    dens  = np.convolve(np.where(td > 1e-9, 1. / (td + 1e-6), 1.).astype('f4'), W50, mode='full')[:N]

    pdiff = np.zeros(N, 'f4')
    pdiff[20:] = price[20:] - price[:-20]
    pmom  = np.convolve(pdiff, W20, mode='full')[:N]
    qpmom = np.convolve(qty * np.sign(pdiff), W20, mode='full')[:N]

    cr    = np.cumsum(ofi).astype('f4')
    crma  = np.convolve(cr, W500, mode='full')[:N]
    crstd = np.sqrt(np.maximum(np.convolve(cr ** 2, W500, mode='full')[:N] - crma ** 2, 1e-8))
    cd    = (cr - crma) / (crstd + 1e-6)

    rd5   = np.convolve(ofi, W500, mode='full')[:N]
    os20  = np.convolve(ofi, W20, mode='full')[:N]
    crate = np.convolve(is_c, W100, mode='full')[:N]
    trate = np.convolve(is_t, W50, mode='full')[:N]
    aasym = np.convolve(is_a * is_bid - is_a * is_ask, W100, mode='full')[:N]

    sdiff = np.zeros(N, 'f4')
    sdiff[1:] = sprd[1:] - sprd[:-1]
    svel  = np.convolve(sdiff, W50, mode='full')[:N]

    baq = is_a * is_bid * qty
    aaq = is_a * is_ask * qty
    qai = (np.convolve(baq - aaq, W100, mode='full')[:N]
           / (np.convolve(baq + aaq, W100, mode='full')[:N] + 1e-6))

    psm = np.convolve(np.sign(pdiff), W200, mode='full')[:N]

    br_ma = np.convolve(is_t * is_bid, W20, mode='full')[:N]
    ar_ma = np.convolve(is_t * is_ask, W20, mode='full')[:N]
    br = br_ma * is_a * is_bid
    ar = ar_ma * is_a * is_ask
    fr = np.convolve(br + ar, W20, mode='full')[:N]

    d = np.stack([rofi, casym, dens, pmom, qpmom, cd, rd5, os20,
                  crate, trate, aasym, svel, qai, psm, fr], axis=1)
    return np.concatenate([ev, d], axis=1).astype('f4')


# =============================================================================
# Metric computations
# =============================================================================

def compute_ic(preds: np.ndarray, labels: np.ndarray) -> float:
    """Spearman rank IC."""
    if len(preds) < 10:
        return 0.0
    mask = np.isfinite(preds) & np.isfinite(labels)
    if mask.sum() < 10:
        return 0.0
    return float(scipy.stats.spearmanr(preds[mask], labels[mask]).statistic)


def compute_directional_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of correct sign predictions."""
    mask = np.isfinite(preds) & np.isfinite(labels) & (labels != 0)
    if mask.sum() < 10:
        return 0.5
    return float(np.mean(np.sign(preds[mask]) == np.sign(labels[mask])))


def compute_conditional_ic(preds: np.ndarray, labels: np.ndarray, quantile: float) -> float:
    """IC computed only on the top-quantile of |pred| (high-conviction signals)."""
    mask = np.isfinite(preds) & np.isfinite(labels)
    p, l = preds[mask], labels[mask]
    if len(p) < 100:
        return 0.0
    threshold = np.quantile(np.abs(p), 1.0 - quantile)
    sel = np.abs(p) >= threshold
    if sel.sum() < 20:
        return 0.0
    return float(scipy.stats.spearmanr(p[sel], l[sel]).statistic)


# =============================================================================
# Data loading and inference
# =============================================================================

def load_day_data_smart(date_str: str) -> Dict[str, np.ndarray]:
    """Load a single day's smart_v3 NPZ file (for CNN-Mamba and PatchTST)."""
    fpath = DATA_DIR_SMART / f"{date_str}_mbo_events.npz"
    if not fpath.exists():
        raise FileNotFoundError(f"Data file not found: {fpath}")
    d = np.load(fpath)
    return {
        "events": d["events"],
        "labels_1s": d["labels_1s"],
        "labels_5s": d["labels_5s"],
        "labels_10s": d["labels_10s"],
    }


def load_day_data_raw(date_str: str) -> Dict[str, np.ndarray]:
    """Load a single day's raw 6-column MBO events NPZ (for LGBM)."""
    fpath = DATA_DIR_RAW / f"{date_str}_mbo_events.npz"
    if not fpath.exists():
        raise FileNotFoundError(f"Data file not found: {fpath}")
    d = np.load(fpath)
    result = {
        "events": d["events"],
        "labels_1s": d["labels_1s"],
        "labels_5s": d["labels_5s"],
        "labels_10s": d["labels_10s"],
    }
    if "labels_30s" in d:
        result["labels_30s"] = d["labels_30s"]
    return result


def run_sliding_window_inference(
    model: nn.Module,
    events: np.ndarray,
    window_size: int,
    stride: int,
    batch_size: int = 4,
) -> tuple:
    """
    Run sliding window inference for deep models.
    Returns: (N_events, n_targets) array of predictions, valid_mask
    """
    N, F = events.shape
    n_targets = 3  # 1s, 5s, 10s

    if N < window_size:
        print(f"    WARNING: Only {N} events, need {window_size}. Padding with zeros.")
        padded = np.zeros((window_size, F), dtype=np.float32)
        padded[-N:] = events
        events = padded
        N = window_size

    starts = list(range(0, N - window_size + 1, stride))
    if not starts:
        starts = [0]

    if len(starts) > MAX_WINDOWS_PER_DAY:
        step = len(starts) // MAX_WINDOWS_PER_DAY
        starts = starts[::step][:MAX_WINDOWS_PER_DAY]
        print(f"    Capped to {len(starts)} windows (uniform sample)")

    pred_sum = np.zeros((N, n_targets), dtype=np.float64)
    pred_count = np.zeros(N, dtype=np.float64)

    model.eval()
    total_windows = len(starts)
    with torch.no_grad():
        for batch_idx, batch_start in enumerate(range(0, len(starts), batch_size)):
            batch_starts = starts[batch_start:batch_start + batch_size]
            windows = []
            for s in batch_starts:
                windows.append(events[s:s + window_size])
            batch_tensor = torch.tensor(np.array(windows), dtype=torch.float32, device=DEVICE)

            out = model(batch_tensor)
            if isinstance(out, tuple):
                out = out[0]
            preds_np = out.cpu().numpy()

            for i, s in enumerate(batch_starts):
                end_idx = s + window_size - 1
                pred_sum[end_idx] += preds_np[i]
                pred_count[end_idx] += 1

            done = min(batch_start + batch_size, total_windows)
            if done % 50 == 0 or done == total_windows:
                print(f"    Progress: {done}/{total_windows} windows", end="\r")

    valid = pred_count > 0
    result = np.zeros((N, n_targets), dtype=np.float32)
    result[valid] = (pred_sum[valid] / pred_count[valid, np.newaxis]).astype(np.float32)

    return result, valid


def run_lgbm_inference(
    lgbm_models: Dict[str, object],
    events_raw: np.ndarray,
) -> tuple:
    """
    Run LGBM inference on raw 6-column events.
    Computes derived features then runs each horizon model.
    Returns: (N, n_horizons) predictions, valid_mask
    """
    N = len(events_raw)
    print(f"    Computing LGBM features for {N:,} events...")
    t0 = time.time()
    features_21 = compute_lgbm_features(events_raw)  # (N, 21)
    print(f"    Features computed in {time.time() - t0:.1f}s")

    horizons = list(lgbm_models.keys())
    n_horizons = len(horizons)
    preds = np.zeros((N, n_horizons), dtype=np.float32)

    t0 = time.time()
    for hi, hz in enumerate(horizons):
        model = lgbm_models[hz]
        preds[:, hi] = model.predict(features_21).astype(np.float32)
    print(f"    LGBM predict: {time.time() - t0:.1f}s")

    # All events get predictions (no windowing needed)
    valid = np.ones(N, dtype=bool)
    return preds, valid


def load_cnn_mamba_v2() -> nn.Module:
    """Load CNN-Mamba v2 model from checkpoint."""
    print("Loading CNN-Mamba v2 weights...")
    ckpt = torch.load(str(CNNMAMBA_WEIGHTS), map_location=DEVICE, weights_only=False)
    arch = ckpt.get("arch", {})

    state = ckpt["model_state"]
    d_state = arch.get("d_state", 32)
    key = "blocks.0.ssm.x_proj.weight"
    if key in state:
        out_dim = state[key].shape[0]
        dt_rank = out_dim - 2 * d_state
    else:
        dt_rank = 6

    model = CNNMambaV2(
        d_model=arch.get("d_model", 96),
        d_state=d_state,
        n_layers=arch.get("n_layers", 3),
        dt_rank=dt_rank,
        d_conv=arch.get("d_conv", 4),
        dropout=arch.get("dropout", 0.1),
        n_targets=3,
        feature_mlp_hidden=arch.get("feature_mlp_hidden", 128),
        feature_mlp_out=arch.get("feature_mlp_out", 64),
        cnn_channels_per_scale=arch.get("cnn_channels_per_scale", 16),
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  WARN unexpected keys: {unexpected}")
    if missing:
        print(f"  Missing keys (defaulted): {missing}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded fold={ckpt.get('fold')} epoch={ckpt.get('epoch')} "
          f"val_ic_10s={ckpt.get('val_ic_10s', 'N/A'):.4f}")
    print(f"  dt_rank={dt_rank} params={n_params:,}")
    model.to(DEVICE)
    model.eval()
    return model


def load_patchtst() -> nn.Module:
    """Load PatchTST model from checkpoint."""
    print("Loading PatchTST weights...")
    ckpt = torch.load(str(PATCHTST_WEIGHTS), map_location=DEVICE, weights_only=False)
    arch = ckpt.get("arch", {})

    model = PatchTST(
        n_features=arch.get("n_features", 25),
        patch_size=arch.get("patch_size", 25),
        d_model=arch.get("d_model", 256),
        n_heads=arch.get("n_heads", 4),
        head_dim=arch.get("head_dim", 64),
        n_layers=arch.get("n_layers", 4),
        ffn_dim=arch.get("ffn_dim", 1024),
        dropout=arch.get("dropout", 0.1),
        n_targets=3,
        window_size=arch.get("window_size", 500),
    )

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if unexpected:
        print(f"  WARN unexpected keys: {unexpected}")
    if missing:
        print(f"  Missing keys (defaulted): {missing}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded fold={ckpt.get('fold')} epoch={ckpt.get('epoch')} "
          f"val_ic_10s={ckpt.get('val_ic_10s', 'N/A'):.4f}")
    print(f"  params={n_params:,}")
    model.to(DEVICE)
    model.eval()
    return model


def load_lgbm_models() -> Dict[str, object]:
    """Load all LGBM horizon models from pickle files."""
    print("Loading LGBM Vol models...")
    models = {}
    for hz in LGBM_HORIZONS:
        pkl_path = LGBM_MODEL_DIR / f"labels_{hz}_lgbm.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                models[hz] = pickle.load(f)
            print(f"  Loaded {hz} model: {models[hz].n_features_in_} features, "
                  f"n_estimators={models[hz].n_estimators_}")
        else:
            print(f"  WARNING: {pkl_path} not found, skipping {hz}")
    return models


# =============================================================================
# Main test
# =============================================================================

def get_days_since_training(date_str: str) -> int:
    """Days since training cutoff (Mar 15, 2026)."""
    d = datetime.strptime(date_str, "%Y%m%d")
    cutoff = datetime(2026, 3, 15)
    return (d - cutoff).days


def get_week_group(date_str: str) -> str:
    """Group dates into weekly buckets for decay curve."""
    days = get_days_since_training(date_str)
    if days <= 7:
        return "Week1 (d1-7)"
    elif days <= 14:
        return "Week2 (d8-14)"
    elif days <= 21:
        return "Week3 (d15-21)"
    elif days <= 28:
        return "Week4 (d22-28)"
    elif days <= 35:
        return "Week5 (d29-35)"
    elif days <= 42:
        return "Week6 (d36-42)"
    else:
        return "Week7+ (d43+)"


def run_decay_test():
    start_time = time.time()
    print("=" * 80)
    print(f"MODEL DECAY TEST V2 — Mar 16 → Apr 29, 2026 ({len(TEST_DATES)} unseen dates)")
    print("Models trained on data up to ~Mar 15, 2026 (1-45 day gap)")
    print("Models: CNN-Mamba v2, PatchTST, LGBM Vol (1s/5s/10s/30s)")
    print("=" * 80)
    print()

    # Load all models
    cnn_mamba = load_cnn_mamba_v2()
    patchtst = load_patchtst()
    lgbm_models = load_lgbm_models()
    print()

    # Deep models: use smart_v3 data, predict 3 horizons (1s, 5s, 10s)
    deep_models = {
        "CNN-Mamba v2": (cnn_mamba, CNNMAMBA_WINDOW, CNNMAMBA_STRIDE, 4),
        "PatchTST":     (patchtst, PATCHTST_WINDOW, PATCHTST_STRIDE, 32),
    }

    # Horizons for reporting
    DEEP_HORIZONS = ["1s", "5s", "10s"]

    # Results storage
    results = defaultdict(dict)
    raw_data = defaultdict(lambda: defaultdict(lambda: {
        "preds": defaultdict(list),
        "labels": defaultdict(list),
    }))

    # =========================================================================
    # Phase 1: Deep models (CNN-Mamba v2 + PatchTST)
    # =========================================================================
    for model_name, (model, window, stride, bsz) in deep_models.items():
        print(f"\n{'='*60}")
        print(f"  {model_name}  (window={window}, stride={stride})")
        print(f"{'='*60}")

        all_preds = {h: [] for h in DEEP_HORIZONS}
        all_labels = {h: [] for h in DEEP_HORIZONS}

        for date_str in TEST_DATES:
            days_since = get_days_since_training(date_str)
            week_group = get_week_group(date_str)
            print(f"\n  Date: {date_str} (day +{days_since}, {week_group})")
            t0 = time.time()

            try:
                data = load_day_data_smart(date_str)
            except FileNotFoundError as e:
                print(f"    SKIP: {e}")
                continue

            events = data["events"]
            print(f"    Events: {len(events):,}")

            preds, valid_mask = run_sliding_window_inference(
                model, events, window, stride, batch_size=bsz
            )
            elapsed = time.time() - t0
            n_valid = valid_mask.sum()
            print(f"    Inference: {elapsed:.1f}s, {n_valid:,} predictions")

            date_results = {"days_since_training": days_since, "week_group": week_group}
            for hi, horizon in enumerate(DEEP_HORIZONS):
                label_key = f"labels_{horizon}"
                labels = data[label_key][:len(preds)]
                p = preds[valid_mask, hi]
                l = labels[valid_mask]

                ic = compute_ic(p, l)
                da = compute_directional_accuracy(p, l)
                cond_ic_5 = compute_conditional_ic(p, l, 0.05)
                cond_ic_1 = compute_conditional_ic(p, l, 0.01)

                date_results[f"IC_{horizon}"] = ic
                date_results[f"DA_{horizon}"] = da
                date_results[f"CondIC5%_{horizon}"] = cond_ic_5
                date_results[f"CondIC1%_{horizon}"] = cond_ic_1

                all_preds[horizon].append(p)
                all_labels[horizon].append(l)

                raw_data[model_name][week_group]["preds"][horizon].append(p)
                raw_data[model_name][week_group]["labels"][horizon].append(l)

                print(f"    {horizon}: IC={ic:+.4f}  DA={da:.3f}  "
                      f"Top5%IC={cond_ic_5:+.4f}  Top1%IC={cond_ic_1:+.4f}")

            results[model_name][date_str] = date_results

        # Concat IC across all dates
        print(f"\n  --- Concat (all dates) ---")
        concat_results = {}
        for horizon in DEEP_HORIZONS:
            if not all_preds[horizon]:
                continue
            p_all = np.concatenate(all_preds[horizon])
            l_all = np.concatenate(all_labels[horizon])
            ic = compute_ic(p_all, l_all)
            da = compute_directional_accuracy(p_all, l_all)
            cond_ic_5 = compute_conditional_ic(p_all, l_all, 0.05)
            cond_ic_1 = compute_conditional_ic(p_all, l_all, 0.01)
            concat_results[f"IC_{horizon}"] = ic
            concat_results[f"DA_{horizon}"] = da
            concat_results[f"CondIC5%_{horizon}"] = cond_ic_5
            concat_results[f"CondIC1%_{horizon}"] = cond_ic_1
            baseline_ic = BASELINE_IC[model_name][f"IC_{horizon}"]
            decay = ic - baseline_ic
            decay_pct = (decay / abs(baseline_ic)) * 100 if baseline_ic != 0 else 0
            print(f"  {horizon}: IC={ic:+.4f} (baseline={baseline_ic:+.4f}, "
                  f"delta={decay:+.4f} [{decay_pct:+.1f}%])  DA={da:.3f}  "
                  f"Top5%IC={cond_ic_5:+.4f}  Top1%IC={cond_ic_1:+.4f}")
        results[model_name]["concat"] = concat_results

    # =========================================================================
    # Phase 2: LGBM Vol
    # =========================================================================
    model_name = "LGBM Vol"
    lgbm_horizons_available = list(lgbm_models.keys())
    print(f"\n{'='*60}")
    print(f"  {model_name}  (per-event, horizons: {lgbm_horizons_available})")
    print(f"{'='*60}")

    all_preds_lgbm = {h: [] for h in lgbm_horizons_available}
    all_labels_lgbm = {h: [] for h in lgbm_horizons_available}

    for date_str in TEST_DATES:
        days_since = get_days_since_training(date_str)
        week_group = get_week_group(date_str)
        print(f"\n  Date: {date_str} (day +{days_since}, {week_group})")
        t0 = time.time()

        try:
            data = load_day_data_raw(date_str)
        except FileNotFoundError as e:
            print(f"    SKIP: {e}")
            continue

        events_raw = data["events"]
        print(f"    Raw events: {len(events_raw):,} (shape {events_raw.shape})")

        preds, valid_mask = run_lgbm_inference(lgbm_models, events_raw)
        elapsed = time.time() - t0
        n_valid = valid_mask.sum()
        print(f"    Total inference: {elapsed:.1f}s, {n_valid:,} predictions")

        date_results = {"days_since_training": days_since, "week_group": week_group}
        for hi, horizon in enumerate(lgbm_horizons_available):
            label_key = f"labels_{horizon}"
            if label_key not in data:
                print(f"    {horizon}: SKIP (no labels)")
                continue
            labels = data[label_key][:len(preds)]
            p = preds[valid_mask, hi]
            l = labels[valid_mask]

            # Filter NaN labels
            finite_mask = np.isfinite(l)
            p = p[finite_mask]
            l = l[finite_mask]

            ic = compute_ic(p, l)
            da = compute_directional_accuracy(p, l)
            cond_ic_5 = compute_conditional_ic(p, l, 0.05)
            cond_ic_1 = compute_conditional_ic(p, l, 0.01)

            date_results[f"IC_{horizon}"] = ic
            date_results[f"DA_{horizon}"] = da
            date_results[f"CondIC5%_{horizon}"] = cond_ic_5
            date_results[f"CondIC1%_{horizon}"] = cond_ic_1

            all_preds_lgbm[horizon].append(p)
            all_labels_lgbm[horizon].append(l)

            raw_data[model_name][week_group]["preds"][horizon].append(p)
            raw_data[model_name][week_group]["labels"][horizon].append(l)

            print(f"    {horizon}: IC={ic:+.4f}  DA={da:.3f}  "
                  f"Top5%IC={cond_ic_5:+.4f}  Top1%IC={cond_ic_1:+.4f}")

        results[model_name][date_str] = date_results

    # Concat IC for LGBM
    print(f"\n  --- Concat (all dates) ---")
    concat_results = {}
    for horizon in lgbm_horizons_available:
        if not all_preds_lgbm[horizon]:
            continue
        p_all = np.concatenate(all_preds_lgbm[horizon])
        l_all = np.concatenate(all_labels_lgbm[horizon])
        ic = compute_ic(p_all, l_all)
        da = compute_directional_accuracy(p_all, l_all)
        cond_ic_5 = compute_conditional_ic(p_all, l_all, 0.05)
        cond_ic_1 = compute_conditional_ic(p_all, l_all, 0.01)
        concat_results[f"IC_{horizon}"] = ic
        concat_results[f"DA_{horizon}"] = da
        concat_results[f"CondIC5%_{horizon}"] = cond_ic_5
        concat_results[f"CondIC1%_{horizon}"] = cond_ic_1

        baseline_key = f"IC_{horizon}"
        if baseline_key in BASELINE_IC.get(model_name, {}):
            baseline_ic = BASELINE_IC[model_name][baseline_key]
            decay = ic - baseline_ic
            decay_pct = (decay / abs(baseline_ic)) * 100 if baseline_ic != 0 else 0
            print(f"  {horizon}: IC={ic:+.4f} (baseline={baseline_ic:+.4f}, "
                  f"delta={decay:+.4f} [{decay_pct:+.1f}%])  DA={da:.3f}  "
                  f"Top5%IC={cond_ic_5:+.4f}  Top1%IC={cond_ic_1:+.4f}")
        else:
            print(f"  {horizon}: IC={ic:+.4f}  DA={da:.3f}  "
                  f"Top5%IC={cond_ic_5:+.4f}  Top1%IC={cond_ic_1:+.4f}")
    results[model_name]["concat"] = concat_results

    # =========================================================================
    # Weekly decay curve
    # =========================================================================
    all_model_names = list(deep_models.keys()) + ["LGBM Vol"]

    print("\n")
    print("=" * 110)
    print("WEEKLY DECAY CURVE (IC_10s)")
    print("=" * 110)

    week_order = ["Week1 (d1-7)", "Week2 (d8-14)", "Week3 (d15-21)",
                  "Week4 (d22-28)", "Week5 (d29-35)", "Week6 (d36-42)", "Week7+ (d43+)"]

    header = f"{'Model':<16} {'Week':<20} {'IC_10s':>8} {'DA_10s':>7} {'Top5%IC':>8} {'n_preds':>10} {'vs_baseline':>12}"
    print(header)
    print("-" * 110)

    weekly_results = defaultdict(dict)
    for mn in all_model_names:
        for week in week_order:
            if week not in raw_data[mn]:
                continue
            wd = raw_data[mn][week]
            if not wd["preds"]["10s"]:
                continue
            p = np.concatenate(wd["preds"]["10s"])
            l = np.concatenate(wd["labels"]["10s"])
            ic = compute_ic(p, l)
            da = compute_directional_accuracy(p, l)
            cond_ic_5 = compute_conditional_ic(p, l, 0.05)
            bl = BASELINE_IC.get(mn, {}).get("IC_10s", 0)
            decay_pct = ((ic - bl) / abs(bl)) * 100 if bl != 0 else 0
            print(f"{mn:<16} {week:<20} {ic:>+8.4f} {da:>7.3f} {cond_ic_5:>+8.4f} {len(p):>10,} {decay_pct:>+11.0f}%")
            weekly_results[mn][week] = {"IC_10s": ic, "DA_10s": da, "n_preds": int(len(p)), "decay_pct": decay_pct}
        print("-" * 110)

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n")
    print("=" * 110)
    print("PER-DATE SUMMARY TABLE")
    print("=" * 110)

    header = f"{'Model':<16} {'Date':<10} {'Day+':>5} {'IC_1s':>8} {'IC_5s':>8} {'IC_10s':>8} {'DA_10s':>7} {'Top5%IC_10s':>12}"
    print(header)
    print("-" * 110)

    for mn in all_model_names:
        for date_str in TEST_DATES:
            if date_str not in results[mn]:
                continue
            r = results[mn][date_str]
            days = r.get("days_since_training", "?")
            ic_1s = r.get("IC_1s", 0)
            ic_5s = r.get("IC_5s", 0)
            ic_10s = r.get("IC_10s", 0)
            da_10s = r.get("DA_10s", 0)
            cond = r.get("CondIC5%_10s", 0)
            print(f"{mn:<16} {date_str:<10} {days:>5} "
                  f"{ic_1s:>+8.4f} {ic_5s:>+8.4f} {ic_10s:>+8.4f} "
                  f"{da_10s:>7.3f} {cond:>+12.4f}")
        # Concat row
        if "concat" in results[mn]:
            r = results[mn]["concat"]
            print(f"{mn:<16} {'ALL':<10} {'':>5} "
                  f"{r.get('IC_1s',0):>+8.4f} {r.get('IC_5s',0):>+8.4f} {r.get('IC_10s',0):>+8.4f} "
                  f"{r.get('DA_10s',0):>7.3f} {r.get('CondIC5%_10s', 0):>+12.4f}")
        # Baseline row
        bl = BASELINE_IC.get(mn, {})
        if bl:
            print(f"{mn:<16} {'BASELINE':<10} {'':>5} "
                  f"{bl.get('IC_1s',0):>+8.4f} {bl.get('IC_5s',0):>+8.4f} {bl.get('IC_10s',0):>+8.4f} "
                  f"{'---':>7} {'---':>12}")
        print("-" * 110)

    # LGBM extra: 30s horizon summary
    if "30s" in lgbm_horizons_available:
        print("\nLGBM Vol — 30s Horizon (extra):")
        print(f"{'Date':<10} {'Day+':>5} {'IC_30s':>8} {'DA_30s':>7} {'Top5%IC':>8}")
        print("-" * 50)
        for date_str in TEST_DATES:
            if date_str in results["LGBM Vol"]:
                r = results["LGBM Vol"][date_str]
                if "IC_30s" in r:
                    print(f"{date_str:<10} {r['days_since_training']:>5} "
                          f"{r['IC_30s']:>+8.4f} {r.get('DA_30s', 0):>7.3f} "
                          f"{r.get('CondIC5%_30s', 0):>+8.4f}")
        if "concat" in results["LGBM Vol"] and "IC_30s" in results["LGBM Vol"]["concat"]:
            r = results["LGBM Vol"]["concat"]
            print(f"{'ALL':<10} {'':>5} {r['IC_30s']:>+8.4f} {r.get('DA_30s',0):>7.3f} "
                  f"{r.get('CondIC5%_30s',0):>+8.4f}")

    # Decay verdict
    print("\nDECAY VERDICT:")
    for mn in all_model_names:
        if "concat" not in results[mn]:
            continue
        concat = results[mn]["concat"]
        bl_10s = BASELINE_IC.get(mn, {}).get("IC_10s", 0)
        ic_10s = concat.get("IC_10s", 0)
        if bl_10s == 0:
            print(f"  {mn:<16}: concat IC_10s={ic_10s:+.4f} (no baseline)")
            continue
        decay_pct = ((ic_10s - bl_10s) / abs(bl_10s)) * 100

        if ic_10s <= 0:
            verdict = "DEAD (IC <= 0)"
        elif decay_pct < -50:
            verdict = f"SEVERE DECAY ({decay_pct:+.0f}%)"
        elif decay_pct < -25:
            verdict = f"MODERATE DECAY ({decay_pct:+.0f}%)"
        elif decay_pct < -10:
            verdict = f"MILD DECAY ({decay_pct:+.0f}%)"
        elif decay_pct < 10:
            verdict = f"STABLE ({decay_pct:+.0f}%)"
        else:
            verdict = f"IMPROVED ({decay_pct:+.0f}%)"

        print(f"  {mn:<16}: concat IC_10s={ic_10s:+.4f} vs baseline={bl_10s:+.4f} -> {verdict}")

    # Retrain recommendation
    print("\nRETRAIN CADENCE RECOMMENDATION:")
    for mn in all_model_names:
        weekly_ics = []
        for week in week_order:
            if week in weekly_results.get(mn, {}):
                wr = weekly_results[mn][week]
                weekly_ics.append((week, wr["IC_10s"], wr["decay_pct"]))

        if len(weekly_ics) >= 2:
            last_ic = weekly_ics[-1][1]
            last_decay = weekly_ics[-1][2]
            if last_ic <= 0:
                print(f"  {mn}: Model DEAD by {weekly_ics[-1][0]} -> retrain IMMEDIATELY, weekly cadence")
            elif last_decay < -50:
                print(f"  {mn}: Severe decay by {weekly_ics[-1][0]} -> retrain every 1-2 weeks")
            elif last_decay < -25:
                print(f"  {mn}: Moderate decay by {weekly_ics[-1][0]} -> retrain every 2-3 weeks")
            elif last_decay < -10:
                print(f"  {mn}: Mild decay by {weekly_ics[-1][0]} -> retrain monthly")
            else:
                print(f"  {mn}: Stable through {weekly_ics[-1][0]} -> retrain monthly or less")

    # Save results to JSON
    out_path = Path("/home/jupiter/Lvl3Quant/output/decay_analysis_v2_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {}
    for mn in all_model_names:
        save_data[mn] = {
            "per_date": {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                            for kk, vv in v.items()}
                        for k, v in results[mn].items()},
            "weekly": {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                          for kk, vv in v.items()}
                      for k, v in weekly_results.get(mn, {}).items()},
            "baseline": BASELINE_IC.get(mn, {}),
        }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time/3600:.1f} hours ({total_time:.0f}s)")
    print("\nDone.")


if __name__ == "__main__":
    run_decay_test()
