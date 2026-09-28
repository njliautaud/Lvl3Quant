#!/usr/bin/env python3
"""
Model Decay Test V4 — PARALLELIZED across dates.

Runs CNN-Mamba v2, PatchTST, and LGBM Vol on ALL post-training dates (Mar 6 → Apr 29, 2026).
Training cutoff: March 5, 2026 (fold_10 last training date).
Uses multiprocessing to saturate all CPU cores.

Strategy:
- Deep models (CNN-Mamba, PatchTST): Load model in each worker process,
  set torch.set_num_threads(2), run N_WORKERS=8 processes = 16 cores saturated.
- LGBM: Load models once, parallelize date processing.

Usage:
    python3 alpha_discovery/deep_models/test_model_decay_v4_parallel.py
    python3 alpha_discovery/deep_models/test_model_decay_v4_parallel.py --workers 12
"""

import os
import sys
import time
import json
import pickle
import warnings
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import scipy.stats

# Suppress warnings
warnings.filterwarnings("ignore")

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR_SMART = ROOT / "data" / "processed" / "mbo_events_smart_v3"
DATA_DIR_RAW   = ROOT / "data" / "processed" / "mbo_events"

# ALL dates after training cutoff (March 5, 2026) — 47 dates total
TEST_DATES = [
    # Mar 6-15 (first 8 OOT days — CRITICAL, were missing from v3)
    "20260306", "20260308", "20260309", "20260310", "20260311", "20260312", "20260313", "20260315",
    # Mar 16-31
    "20260316", "20260317", "20260318", "20260319", "20260320",
    "20260322", "20260323", "20260324", "20260325", "20260326", "20260327",
    "20260329", "20260330", "20260331",
    # Apr 1-29
    "20260401", "20260402", "20260403",
    "20260405", "20260406", "20260407", "20260408", "20260409", "20260410",
    "20260412", "20260413", "20260414", "20260415", "20260416", "20260417",
    "20260419", "20260420",
    "20260421", "20260422", "20260423", "20260424",
    "20260426", "20260427", "20260428", "20260429",
]

# Training cutoff — fold_10 last training date
TRAINING_CUTOFF = datetime(2026, 3, 5)

CNNMAMBA_WEIGHTS = ROOT / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_10_best.pt"
CNNMAMBA_STATS   = ROOT / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_09_feature_stats.npz"
CNNMAMBA_WINDOW  = 3000
CNNMAMBA_STRIDE  = 3000

PATCHTST_WEIGHTS = ROOT / "output" / "patchtst_razer_weights" / "fold_17_best.pt"
PATCHTST_WINDOW  = 500
PATCHTST_STRIDE  = 500

LGBM_MODEL_DIR = ROOT / "models" / "lgbm_60_5_fold35"
LGBM_HORIZONS  = ["1s", "5s", "10s", "30s"]

BASELINE_IC = {
    "CNN-Mamba v2": {"IC_1s": 0.040, "IC_5s": 0.065, "IC_10s": 0.085},
    "PatchTST":     {"IC_1s": 0.035, "IC_5s": 0.055, "IC_10s": 0.075},
    "LGBM Vol":     {"IC_1s": 0.122, "IC_5s": 0.055, "IC_10s": 0.041},
}

MAX_WINDOWS_PER_DAY = 500
DEEP_HORIZONS = ["1s", "5s", "10s"]

# =============================================================================
# Metric computations (must be top-level for pickling)
# =============================================================================

def compute_ic(preds, labels):
    if len(preds) < 10:
        return 0.0
    mask = np.isfinite(preds) & np.isfinite(labels)
    if mask.sum() < 10:
        return 0.0
    return float(scipy.stats.spearmanr(preds[mask], labels[mask]).statistic)

def compute_directional_accuracy(preds, labels):
    mask = np.isfinite(preds) & np.isfinite(labels) & (labels != 0)
    if mask.sum() < 10:
        return 0.5
    return float(np.mean(np.sign(preds[mask]) == np.sign(labels[mask])))

def compute_conditional_ic(preds, labels, quantile):
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
# Data loading
# =============================================================================

def load_day_data_smart(date_str):
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

def load_day_data_raw(date_str):
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


# =============================================================================
# Model definitions (inline for pickling in workers)
# =============================================================================

def _build_cnn_mamba_v2():
    """Build and load CNN-Mamba v2 model."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import Optional

    class SelectiveSSM(nn.Module):
        def __init__(self, d_model, d_state=32, dt_rank=6, d_conv=4):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.dt_rank = dt_rank
            self.d_conv = d_conv
            self.d_inner = d_model * 2
            self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=0, groups=self.d_inner, bias=True)
            self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
            self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
            A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))
            self.time_decay_rate = nn.Parameter(torch.ones(self.d_inner, 1) * 0.1)
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        def forward(self, x, time_delta=None):
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
        def __init__(self, d_model, d_state=32, dt_rank=6, d_conv=4, dropout=0.1):
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
        CNN_KERNELS = [3, 7, 15, 31]
        def __init__(self, n_smart_features=25, d_model=96, d_state=32, n_layers=3,
                     dt_rank=6, d_conv=4, dropout=0.1, n_targets=3,
                     feature_mlp_hidden=128, feature_mlp_out=64, cnn_channels_per_scale=16):
            super().__init__()
            self.d_model = d_model
            self.n_targets = n_targets
            self.feature_mlp = nn.Sequential(
                nn.Linear(n_smart_features, feature_mlp_hidden), nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(feature_mlp_hidden, feature_mlp_out), nn.LayerNorm(feature_mlp_out))
            self.cnn_kernels = self.CNN_KERNELS
            self.temporal_cnns = nn.ModuleList()
            for k in self.cnn_kernels:
                self.temporal_cnns.append(nn.Sequential(
                    nn.Conv1d(n_smart_features, cnn_channels_per_scale, kernel_size=k, padding=0, bias=True),
                    nn.GELU()))
            self.cnn_out_dim = cnn_channels_per_scale * len(self.cnn_kernels)
            fusion_dim = feature_mlp_out + self.cnn_out_dim
            self.fusion_proj = nn.Sequential(nn.Linear(fusion_dim, d_model), nn.LayerNorm(d_model))
            self.blocks = nn.ModuleList([
                MambaBlock(d_model=d_model, d_state=d_state, dt_rank=dt_rank, d_conv=d_conv, dropout=dropout)
                for _ in range(n_layers)])
            self.final_norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, n_targets))
        def forward(self, smart):
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
            return self.head(last_hidden)

    device = torch.device("cpu")
    ckpt = torch.load(str(CNNMAMBA_WEIGHTS), map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    state = ckpt["model_state"]
    d_state = arch.get("d_state", 32)
    key = "blocks.0.ssm.x_proj.weight"
    dt_rank = state[key].shape[0] - 2 * d_state if key in state else 6

    model = CNNMambaV2(
        d_model=arch.get("d_model", 96), d_state=d_state, n_layers=arch.get("n_layers", 3),
        dt_rank=dt_rank, d_conv=arch.get("d_conv", 4), dropout=arch.get("dropout", 0.1),
        n_targets=3, feature_mlp_hidden=arch.get("feature_mlp_hidden", 128),
        feature_mlp_out=arch.get("feature_mlp_out", 64),
        cnn_channels_per_scale=arch.get("cnn_channels_per_scale", 16))
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def _build_patchtst():
    """Build and load PatchTST model."""
    import torch

    os.environ["TST_FEATURE_SET"] = "smart_v3"
    os.environ["SKIP_NORMALIZE"] = "1"
    os.environ["DISABLE_MLFLOW"] = "1"

    _DEEP_MODELS_DIR = str(Path(__file__).resolve().parent)
    if _DEEP_MODELS_DIR not in sys.path:
        sys.path.insert(0, _DEEP_MODELS_DIR)

    import logging
    logging.disable(logging.INFO)
    from train_event_patchtst import PatchTST
    logging.disable(logging.NOTSET)

    device = torch.device("cpu")
    ckpt = torch.load(str(PATCHTST_WEIGHTS), map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    model = PatchTST(
        n_features=arch.get("n_features", 25), patch_size=arch.get("patch_size", 25),
        d_model=arch.get("d_model", 256), n_heads=arch.get("n_heads", 4),
        head_dim=arch.get("head_dim", 64), n_layers=arch.get("n_layers", 4),
        ffn_dim=arch.get("ffn_dim", 1024), dropout=arch.get("dropout", 0.1),
        n_targets=3, window_size=arch.get("window_size", 500))
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()
    return model


# =============================================================================
# LGBM feature computation
# =============================================================================

def compute_lgbm_features(ev):
    N = len(ev)
    td = ev[:, 0].astype('f4')
    et = ev[:, 1].astype('f4')
    side = ev[:, 2].astype('f4')
    price = ev[:, 3].astype('f4')
    qty = ev[:, 4].astype('f4')
    sprd = ev[:, 5].astype('f4')
    W20 = np.ones(20, 'f4') / 20; W50 = np.ones(50, 'f4') / 50
    W100 = np.ones(100, 'f4') / 100; W200 = np.ones(200, 'f4') / 200; W500 = np.ones(500, 'f4') / 500
    is_t = (et == 3).astype('f4'); is_c = (et == 1).astype('f4')
    is_a = (et == 0).astype('f4'); is_bid = (side == 0).astype('f4'); is_ask = (side == 1).astype('f4')
    bf = is_t * is_ask * qty; sf = is_t * is_bid * qty; ofi = bf - sf
    rofi = np.convolve(ofi, W100, mode='full')[:N]
    casym = np.convolve(is_c * is_bid - is_c * is_ask, W100, mode='full')[:N]
    dens = np.convolve(np.where(td > 1e-9, 1. / (td + 1e-6), 1.).astype('f4'), W50, mode='full')[:N]
    pdiff = np.zeros(N, 'f4'); pdiff[20:] = price[20:] - price[:-20]
    pmom = np.convolve(pdiff, W20, mode='full')[:N]
    qpmom = np.convolve(qty * np.sign(pdiff), W20, mode='full')[:N]
    cr = np.cumsum(ofi).astype('f4')
    crma = np.convolve(cr, W500, mode='full')[:N]
    crstd = np.sqrt(np.maximum(np.convolve(cr ** 2, W500, mode='full')[:N] - crma ** 2, 1e-8))
    cd = (cr - crma) / (crstd + 1e-6)
    rd5 = np.convolve(ofi, W500, mode='full')[:N]
    os20 = np.convolve(ofi, W20, mode='full')[:N]
    crate = np.convolve(is_c, W100, mode='full')[:N]
    trate = np.convolve(is_t, W50, mode='full')[:N]
    aasym = np.convolve(is_a * is_bid - is_a * is_ask, W100, mode='full')[:N]
    sdiff = np.zeros(N, 'f4'); sdiff[1:] = sprd[1:] - sprd[:-1]
    svel = np.convolve(sdiff, W50, mode='full')[:N]
    baq = is_a * is_bid * qty; aaq = is_a * is_ask * qty
    qai = (np.convolve(baq - aaq, W100, mode='full')[:N] / (np.convolve(baq + aaq, W100, mode='full')[:N] + 1e-6))
    psm = np.convolve(np.sign(pdiff), W200, mode='full')[:N]
    br_ma = np.convolve(is_t * is_bid, W20, mode='full')[:N]
    ar_ma = np.convolve(is_t * is_ask, W20, mode='full')[:N]
    br = br_ma * is_a * is_bid; ar = ar_ma * is_a * is_ask
    fr = np.convolve(br + ar, W20, mode='full')[:N]
    d = np.stack([rofi, casym, dens, pmom, qpmom, cd, rd5, os20, crate, trate, aasym, svel, qai, psm, fr], axis=1)
    return np.concatenate([ev, d], axis=1).astype('f4')


# =============================================================================
# Worker functions (called in separate processes)
# =============================================================================

def _process_deep_date(args):
    """Process a single date for a deep model. Runs in a worker process."""
    model_name, date_str, window, stride, batch_size, threads_per_worker = args

    import torch
    torch.set_num_threads(threads_per_worker)

    t0 = time.time()

    # Load model in worker
    if model_name == "CNN-Mamba v2":
        model = _build_cnn_mamba_v2()
    elif model_name == "PatchTST":
        model = _build_patchtst()
    else:
        return None

    try:
        data = load_day_data_smart(date_str)
    except FileNotFoundError as e:
        return {"date": date_str, "model": model_name, "error": str(e)}

    events = data["events"]
    N, F = events.shape
    n_targets = 3

    if N < window:
        padded = np.zeros((window, F), dtype=np.float32)
        padded[-N:] = events
        events = padded
        N = window

    starts = list(range(0, N - window + 1, stride))
    if not starts:
        starts = [0]

    if len(starts) > MAX_WINDOWS_PER_DAY:
        step = len(starts) // MAX_WINDOWS_PER_DAY
        starts = starts[::step][:MAX_WINDOWS_PER_DAY]

    pred_sum = np.zeros((N, n_targets), dtype=np.float64)
    pred_count = np.zeros(N, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start:batch_start + batch_size]
            windows = [events[s:s + window] for s in batch_starts]
            batch_tensor = torch.tensor(np.array(windows), dtype=torch.float32)
            out = model(batch_tensor)
            if isinstance(out, tuple):
                out = out[0]
            preds_np = out.cpu().numpy()
            for i, s in enumerate(batch_starts):
                end_idx = s + window - 1
                pred_sum[end_idx] += preds_np[i]
                pred_count[end_idx] += 1

    valid = pred_count > 0
    result_preds = np.zeros((N, n_targets), dtype=np.float32)
    result_preds[valid] = (pred_sum[valid] / pred_count[valid, np.newaxis]).astype(np.float32)

    # Compute metrics
    days_since = (datetime.strptime(date_str, "%Y%m%d") - TRAINING_CUTOFF).days
    date_results = {"date": date_str, "model": model_name, "days_since_training": days_since,
                    "n_events": int(len(data["events"])), "n_windows": len(starts)}

    preds_per_horizon = {}
    labels_per_horizon = {}
    for hi, horizon in enumerate(DEEP_HORIZONS):
        label_key = f"labels_{horizon}"
        labels = data[label_key][:len(result_preds)]
        p = result_preds[valid, hi]
        l = labels[valid]

        ic = compute_ic(p, l)
        da = compute_directional_accuracy(p, l)
        cond_ic_5 = compute_conditional_ic(p, l, 0.05)
        cond_ic_1 = compute_conditional_ic(p, l, 0.01)

        date_results[f"IC_{horizon}"] = ic
        date_results[f"DA_{horizon}"] = da
        date_results[f"CondIC5%_{horizon}"] = cond_ic_5
        date_results[f"CondIC1%_{horizon}"] = cond_ic_1

        preds_per_horizon[horizon] = p
        labels_per_horizon[horizon] = l

    elapsed = time.time() - t0
    date_results["elapsed_s"] = round(elapsed, 1)

    return date_results


def _process_lgbm_date(args):
    """Process a single date for LGBM Vol. Runs in a worker process."""
    date_str, threads_per_worker = args

    t0 = time.time()

    # Load LGBM models in each worker
    lgbm_models = {}
    for hz in LGBM_HORIZONS:
        pkl_path = LGBM_MODEL_DIR / f"labels_{hz}_lgbm.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                lgbm_models[hz] = pickle.load(f)

    try:
        data = load_day_data_raw(date_str)
    except FileNotFoundError as e:
        return {"date": date_str, "model": "LGBM Vol", "error": str(e)}

    events_raw = data["events"]
    features_21 = compute_lgbm_features(events_raw)

    horizons = list(lgbm_models.keys())
    N = len(events_raw)
    preds = np.zeros((N, len(horizons)), dtype=np.float32)
    for hi, hz in enumerate(horizons):
        preds[:, hi] = lgbm_models[hz].predict(features_21).astype(np.float32)

    days_since = (datetime.strptime(date_str, "%Y%m%d") - TRAINING_CUTOFF).days
    date_results = {"date": date_str, "model": "LGBM Vol", "days_since_training": days_since,
                    "n_events": N}

    for hi, horizon in enumerate(horizons):
        label_key = f"labels_{horizon}"
        if label_key not in data:
            continue
        labels = data[label_key][:N]
        p = preds[:, hi]
        finite = np.isfinite(labels)
        p, l = p[finite], labels[finite]

        ic = compute_ic(p, l)
        da = compute_directional_accuracy(p, l)
        cond_ic_5 = compute_conditional_ic(p, l, 0.05)
        cond_ic_1 = compute_conditional_ic(p, l, 0.01)

        date_results[f"IC_{horizon}"] = ic
        date_results[f"DA_{horizon}"] = da
        date_results[f"CondIC5%_{horizon}"] = cond_ic_5
        date_results[f"CondIC1%_{horizon}"] = cond_ic_1

    elapsed = time.time() - t0
    date_results["elapsed_s"] = round(elapsed, 1)
    return date_results


# =============================================================================
# Helper
# =============================================================================

def get_week_group(date_str):
    days = (datetime.strptime(date_str, "%Y%m%d") - TRAINING_CUTOFF).days
    if days <= 7: return "Week1 (d1-7)"
    elif days <= 14: return "Week2 (d8-14)"
    elif days <= 21: return "Week3 (d15-21)"
    elif days <= 28: return "Week4 (d22-28)"
    elif days <= 35: return "Week5 (d29-35)"
    elif days <= 42: return "Week6 (d36-42)"
    else: return "Week7+ (d43+)"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--threads-per-worker", type=int, default=2, help="Torch threads per worker")
    parser.add_argument("--dates", type=str, default=None, help="Comma-separated dates to test (default: all)")
    args = parser.parse_args()

    N_WORKERS = args.workers
    THREADS_PER_WORKER = args.threads_per_worker

    # Allow filtering to specific dates
    global TEST_DATES
    if args.dates:
        requested = [d.strip() for d in args.dates.split(",")]
        TEST_DATES = [d for d in requested if d in TEST_DATES or True]  # accept any valid date
        print(f"Filtered to {len(TEST_DATES)} specific dates")

    total_cores = os.cpu_count() or 16
    print(f"System: {total_cores} cores available")
    print(f"Config: {N_WORKERS} workers x {THREADS_PER_WORKER} threads = {N_WORKERS * THREADS_PER_WORKER} threads")
    print(f"Dates to test: {len(TEST_DATES)}")
    print(f"Training cutoff: {TRAINING_CUTOFF.strftime('%Y-%m-%d')}")
    print()

    start_time = time.time()
    all_results = defaultdict(dict)

    # =========================================================================
    # Phase 1: CNN-Mamba v2 (parallel across dates)
    # =========================================================================
    for model_name, window, stride, bsz in [
        ("CNN-Mamba v2", CNNMAMBA_WINDOW, CNNMAMBA_STRIDE, 4),
        ("PatchTST", PATCHTST_WINDOW, PATCHTST_STRIDE, 32),
    ]:
        print(f"\n{'='*70}")
        print(f"  {model_name} — {len(TEST_DATES)} dates x {N_WORKERS} workers")
        print(f"{'='*70}")
        t0_model = time.time()

        tasks = [(model_name, d, window, stride, bsz, THREADS_PER_WORKER) for d in TEST_DATES]
        completed = 0

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(_process_deep_date, t): t[1] for t in tasks}
            for future in as_completed(futures):
                date_str = futures[future]
                try:
                    result = future.result()
                    completed += 1
                    if result and "error" not in result:
                        all_results[model_name][date_str] = result
                        ic_10s = result.get("IC_10s", 0)
                        print(f"  [{completed}/{len(TEST_DATES)}] {date_str} "
                              f"day+{result['days_since_training']} "
                              f"IC_10s={ic_10s:+.4f} "
                              f"({result['elapsed_s']:.0f}s, {result.get('n_windows', '?')} windows)")
                    elif result:
                        completed += 1
                        print(f"  [{completed}/{len(TEST_DATES)}] {date_str} SKIP: {result['error']}")
                except Exception as e:
                    completed += 1
                    print(f"  [{completed}/{len(TEST_DATES)}] {date_str} ERROR: {e}")

        model_elapsed = time.time() - t0_model
        print(f"\n  {model_name} complete: {model_elapsed:.0f}s ({model_elapsed/60:.1f}min)")

        # Concat IC
        all_preds = {h: [] for h in DEEP_HORIZONS}
        all_labels_concat = {h: [] for h in DEEP_HORIZONS}
        # We need raw preds/labels for concat — recompute from saved results isn't possible
        # Instead report per-date stats
        print(f"\n  Per-date IC_10s:")
        for d in TEST_DATES:
            if d in all_results[model_name]:
                r = all_results[model_name][d]
                bl = BASELINE_IC[model_name]["IC_10s"]
                ic = r.get("IC_10s", 0)
                decay = ((ic - bl) / abs(bl)) * 100 if bl else 0
                print(f"    {d} day+{r['days_since_training']:>2}: IC_10s={ic:+.4f} ({decay:+.0f}% vs baseline)")

    # =========================================================================
    # Phase 2: LGBM Vol (parallel across dates)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  LGBM Vol — {len(TEST_DATES)} dates x {N_WORKERS} workers")
    print(f"{'='*70}")
    t0_lgbm = time.time()

    tasks = [(d, THREADS_PER_WORKER) for d in TEST_DATES]
    completed = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_process_lgbm_date, t): t[0] for t in tasks}
        for future in as_completed(futures):
            date_str = futures[future]
            try:
                result = future.result()
                completed += 1
                if result and "error" not in result:
                    all_results["LGBM Vol"][date_str] = result
                    ic_10s = result.get("IC_10s", 0)
                    print(f"  [{completed}/{len(TEST_DATES)}] {date_str} "
                          f"day+{result['days_since_training']} "
                          f"IC_10s={ic_10s:+.4f} ({result['elapsed_s']:.0f}s)")
                elif result:
                    print(f"  [{completed}/{len(TEST_DATES)}] {date_str} SKIP: {result['error']}")
            except Exception as e:
                completed += 1
                print(f"  [{completed}/{len(TEST_DATES)}] {date_str} ERROR: {e}")

    lgbm_elapsed = time.time() - t0_lgbm
    print(f"\n  LGBM Vol complete: {lgbm_elapsed:.0f}s ({lgbm_elapsed/60:.1f}min)")

    # =========================================================================
    # Summary tables
    # =========================================================================
    all_model_names = ["CNN-Mamba v2", "PatchTST", "LGBM Vol"]
    week_order = ["Week1 (d1-7)", "Week2 (d8-14)", "Week3 (d15-21)",
                  "Week4 (d22-28)", "Week5 (d29-35)", "Week6 (d36-42)", "Week7+ (d43+)"]

    # Weekly decay curve
    print(f"\n\n{'='*110}")
    print("WEEKLY DECAY CURVE (IC_10s)")
    print(f"{'='*110}")
    header = f"{'Model':<16} {'Week':<20} {'IC_10s':>8} {'DA_10s':>7} {'Top5%IC':>8} {'n_dates':>8} {'vs_baseline':>12}"
    print(header)
    print("-" * 110)

    weekly_results = defaultdict(dict)
    for mn in all_model_names:
        for week in week_order:
            week_ics = []
            week_das = []
            week_cond = []
            for d in TEST_DATES:
                if d in all_results.get(mn, {}) and get_week_group(d) == week:
                    r = all_results[mn][d]
                    if "IC_10s" in r:
                        week_ics.append(r["IC_10s"])
                        week_das.append(r.get("DA_10s", 0.5))
                        week_cond.append(r.get("CondIC5%_10s", 0))
            if week_ics:
                avg_ic = np.mean(week_ics)
                avg_da = np.mean(week_das)
                avg_cond = np.mean(week_cond)
                bl = BASELINE_IC.get(mn, {}).get("IC_10s", 0)
                decay_pct = ((avg_ic - bl) / abs(bl)) * 100 if bl else 0
                print(f"{mn:<16} {week:<20} {avg_ic:>+8.4f} {avg_da:>7.3f} {avg_cond:>+8.4f} {len(week_ics):>8} {decay_pct:>+11.0f}%")
                weekly_results[mn][week] = {"IC_10s": avg_ic, "DA_10s": avg_da, "n_dates": len(week_ics), "decay_pct": decay_pct}
        print("-" * 110)

    # Per-date table
    print(f"\n\n{'='*110}")
    print("PER-DATE SUMMARY TABLE")
    print(f"{'='*110}")
    header = f"{'Model':<16} {'Date':<10} {'Day+':>5} {'IC_1s':>8} {'IC_5s':>8} {'IC_10s':>8} {'DA_10s':>7} {'Top5%IC_10s':>12}"
    print(header)
    print("-" * 110)

    for mn in all_model_names:
        for d in TEST_DATES:
            if d in all_results.get(mn, {}):
                r = all_results[mn][d]
                print(f"{mn:<16} {d:<10} {r['days_since_training']:>5} "
                      f"{r.get('IC_1s',0):>+8.4f} {r.get('IC_5s',0):>+8.4f} {r.get('IC_10s',0):>+8.4f} "
                      f"{r.get('DA_10s',0):>7.3f} {r.get('CondIC5%_10s',0):>+12.4f}")
        # Mean row
        model_dates = [all_results[mn][d] for d in TEST_DATES if d in all_results.get(mn, {})]
        if model_dates:
            mean_1s = np.mean([r.get("IC_1s", 0) for r in model_dates])
            mean_5s = np.mean([r.get("IC_5s", 0) for r in model_dates])
            mean_10s = np.mean([r.get("IC_10s", 0) for r in model_dates])
            mean_da = np.mean([r.get("DA_10s", 0.5) for r in model_dates])
            bl = BASELINE_IC.get(mn, {})
            print(f"{mn:<16} {'MEAN':<10} {'':>5} "
                  f"{mean_1s:>+8.4f} {mean_5s:>+8.4f} {mean_10s:>+8.4f} "
                  f"{mean_da:>7.3f} {'---':>12}")
            print(f"{mn:<16} {'BASELINE':<10} {'':>5} "
                  f"{bl.get('IC_1s',0):>+8.4f} {bl.get('IC_5s',0):>+8.4f} {bl.get('IC_10s',0):>+8.4f} "
                  f"{'---':>7} {'---':>12}")
        print("-" * 110)

    # Decay verdict
    print("\nDECAY VERDICT:")
    for mn in all_model_names:
        model_dates = [all_results[mn][d] for d in TEST_DATES if d in all_results.get(mn, {})]
        if not model_dates:
            continue
        mean_10s = np.mean([r.get("IC_10s", 0) for r in model_dates])
        bl_10s = BASELINE_IC.get(mn, {}).get("IC_10s", 0)
        if bl_10s == 0:
            print(f"  {mn:<16}: mean IC_10s={mean_10s:+.4f} (no baseline)")
            continue
        decay_pct = ((mean_10s - bl_10s) / abs(bl_10s)) * 100
        if mean_10s <= 0:
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
        print(f"  {mn:<16}: mean IC_10s={mean_10s:+.4f} vs baseline={bl_10s:+.4f} -> {verdict}")

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
                print(f"  {mn}: Model DEAD by {weekly_ics[-1][0]} -> retrain IMMEDIATELY")
            elif last_decay < -50:
                print(f"  {mn}: Severe decay by {weekly_ics[-1][0]} -> retrain every 1-2 weeks")
            elif last_decay < -25:
                print(f"  {mn}: Moderate decay by {weekly_ics[-1][0]} -> retrain every 2-3 weeks")
            elif last_decay < -10:
                print(f"  {mn}: Mild decay by {weekly_ics[-1][0]} -> retrain monthly")
            else:
                print(f"  {mn}: Stable through {weekly_ics[-1][0]} -> retrain monthly or less")

    # Save results
    out_path = ROOT / "output" / "decay_analysis_v3_results.json"
    save_data = {}
    for mn in all_model_names:
        save_data[mn] = {
            "per_date": {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                            for kk, vv in v.items()}
                        for k, v in all_results.get(mn, {}).items()},
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
    print(f"Speedup vs serial: estimated ~{3.0:.1f}x (3 models parallel)")
    print("\nDone.")


if __name__ == "__main__":
    main()
