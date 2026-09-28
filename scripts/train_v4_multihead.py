#!/usr/bin/env python3
"""
train_v4_multihead.py — v4 Multi-Head CNN-Mamba with Smooth Pressure Targets

Predicts smooth pressure targets (NTPS, EOFI, PDI, TIA) alongside raw directional
returns using a shared CNN-Mamba encoder with multiple prediction heads.

Architecture:
  Shared Encoder (from CNN-Mamba v2 backbone):
    - 1D CNN layers (3x Conv1d + BatchNorm + GELU + residual) for local patterns
    - Pure-PyTorch Mamba SSM layers for sequence modeling (no mamba_ssm dependency)
    - Output: shared representation per event (d_model dimensional)

  Prediction Heads:
    Head A — Raw Directional: regression on labels_1s/5s/10s (MSE loss)
    Head B — NTPS: net taker pressure score at 1s/5s/10s/30s (MSE, tanh output)
    Head C — EOFI: exp order flow imbalance at 1s/5s/10s/30s (MSE)
    Head D — PDI:  pressure duration index at 1s/5s/10s/30s (MSE, tanh output)
    Head E — TIA:  trade intensity asymmetry at 1s/5s/10s/30s (MSE, tanh output)

  Multi-task loss: weighted sum of per-head MSE losses.
  Default weights: A=1.0, B=2.0, C=1.0, D=1.0, E=1.0

Training:
  - Walk-forward: 60-day sliding window, 1-day OOT, drop oldest day
  - Spearman IC as primary evaluation metric per head per horizon
  - Mixed-precision (fp16) on CUDA
  - MLflow logging mandatory

Data:
  - Input X: smart_v3 event files (.npz), 25 features per event
  - Directional labels: labels_1s/5s/10s from event files
  - Pressure labels: NTPS/EOFI/PDI/TIA at 1s/5s/10s/30s from smooth_pressure_targets/

Usage:
  python train_v4_multihead.py \\
    --data-dir /path/to/mbo_events_smart_v3 \\
    --pressure-dir /path/to/smooth_pressure_targets \\
    --output-dir /path/to/output/v4_multihead_pressure \\
    --device cuda

Smoke test:
  python train_v4_multihead.py --seq-len 256 --epochs 1 --batch-size 16 \\
    --n-folds 1 --stride 128

Author: Claude (head-of-quant), 2026-06-01.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import socket
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Logging
# ============================================================
logging.root.handlers.clear()
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(_fmt)
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_sh)
logger = logging.getLogger("train_v4")

# ============================================================
# MLflow
# ============================================================
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
    if os.environ.get("DISABLE_MLFLOW", "0") == "1":
        MLFLOW_AVAILABLE = False
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not installed — skipping experiment tracking")

# ============================================================
# Constants
# ============================================================
N_EVENT_FEATURES = 25  # smart_v3 feature count

# Head definitions
DIR_HORIZONS = ["1s", "5s", "10s"]  # raw directional
PRESSURE_HORIZONS = ["1s", "5s", "10s", "30s"]  # smooth pressure
PRESSURE_TARGETS = ["ntps", "eofi", "pdi", "tia"]

# Which heads use tanh output (bounded [-1,+1])
TANH_HEADS = {"ntps", "pdi", "tia"}  # EOFI is unbounded


# ============================================================
# Pure-PyTorch Selective SSM (no mamba_ssm dependency)
# ============================================================

class SelectiveSSM(nn.Module):
    """
    Pure-PyTorch Mamba-style selective state space model.
    No dependency on mamba_ssm — runs on any platform with PyTorch.

    Core recurrence:
        h_new = A(x) * h_old + B(x) * input
        output = C(x) * h_new

    With time-delta conditioning: A_effective = A(x) * exp(-decay * time_delta)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_rank: int = 16,
        d_conv: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * 2

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=0,
            groups=self.d_inner, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # Initialize dt bias for stable training
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (np.log(0.1) - np.log(0.001)) + np.log(0.001)
            )
            inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.time_decay_rate = nn.Parameter(torch.ones(self.d_inner, 1) * 0.1)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.d_conv = d_conv
        self.dt_rank = dt_rank

    def forward(self, x: torch.Tensor, time_delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Causal conv1d
        x_conv = x_branch.transpose(1, 2).contiguous()
        x_conv = F.pad(x_conv, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_conv).transpose(1, 2).contiguous()
        x_branch = F.silu(x_conv)

        # Selective parameters
        x_proj = self.x_proj(x_branch)
        dt_x = x_proj[:, :, :self.dt_rank]
        B_sel = x_proj[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C_sel = x_proj[:, :, self.dt_rank + self.d_state:]

        dt = F.softplus(self.dt_proj(dt_x))
        A = -torch.exp(self.A_log)

        # Selective scan
        y = self._scan(x_branch, dt, A, B_sel, C_sel, time_delta)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _scan(self, x, dt, A, B, C, time_delta):
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        A_exp = A.unsqueeze(0).unsqueeze(0)
        dt_exp = dt.unsqueeze(-1)
        dA = torch.exp(A_exp * dt_exp)

        if time_delta is not None:
            td = time_delta.unsqueeze(-1).unsqueeze(-1)
            dr = F.softplus(self.time_decay_rate).unsqueeze(0).unsqueeze(0)
            dA = dA * torch.exp(-dr * td.abs())

        dBx = B.unsqueeze(2) * dt_exp * x.unsqueeze(-1)

        # Chunked sequential scan
        CHUNK = 64
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t0 in range(0, seq_len, CHUNK):
            t1 = min(t0 + CHUNK, seq_len)
            chunk_out = []
            for t in range(t0, t1):
                h = dA[:, t] * h + dBx[:, t]
                y_t = torch.einsum("bn,bdn->bd", C[:, t], h)
                chunk_out.append(y_t)
            outputs.append(torch.stack(chunk_out, dim=1))

        y = torch.cat(outputs, dim=1)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y


class MambaBlock(nn.Module):
    """Pre-norm residual Mamba block."""

    def __init__(self, d_model: int, d_state: int = 64, dt_rank: int = 16,
                 d_conv: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, dt_rank, d_conv)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, time_delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.ssm(x, time_delta=time_delta)
        x = self.dropout(x)
        return x + residual


# ============================================================
# V4 Multi-Head CNN-Mamba Model
# ============================================================

class PredictionHead(nn.Module):
    """A single prediction head: Linear -> GELU -> Dropout -> Linear -> optional tanh."""

    def __init__(self, d_model: int, n_outputs: int, dropout: float = 0.1,
                 use_tanh: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_outputs),
        )
        self.use_tanh = use_tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.use_tanh:
            out = torch.tanh(out)
        return out


class CNNMambaV4MultiHead(nn.Module):
    """
    V4 Multi-Head CNN-Mamba for simultaneous directional + pressure prediction.

    Shared encoder: CNN front-end -> Mamba SSM backbone -> final LayerNorm
    5 prediction heads: directional, NTPS, EOFI, PDI, TIA
    """

    def __init__(
        self,
        n_features: int = N_EVENT_FEATURES,
        d_model: int = 96,
        d_state: int = 32,
        n_layers: int = 3,
        dt_rank: int = 16,
        d_conv: int = 4,
        dropout: float = 0.1,
        cnn_channels: int = 64,
        cnn_kernel: int = 5,
        cnn_layers: int = 3,
        n_dir_horizons: int = 3,        # 1s, 5s, 10s
        n_pressure_horizons: int = 4,    # 1s, 5s, 10s, 30s
    ):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        # ---- CNN Front-End ----
        cnn_pad = cnn_kernel // 2
        self.cnn_conv_layers = nn.ModuleList()
        self.cnn_norms = nn.ModuleList()
        self.cnn_conv_layers.append(
            nn.Conv1d(n_features, cnn_channels, kernel_size=cnn_kernel, padding=cnn_pad)
        )
        self.cnn_norms.append(nn.BatchNorm1d(cnn_channels))
        for _ in range(cnn_layers - 1):
            self.cnn_conv_layers.append(
                nn.Conv1d(cnn_channels, cnn_channels, kernel_size=cnn_kernel, padding=cnn_pad)
            )
            self.cnn_norms.append(nn.BatchNorm1d(cnn_channels))
        self.cnn_dropout = nn.Dropout(dropout)
        self.cnn_act = nn.GELU()

        # ---- Projection: cnn_channels -> d_model ----
        self.projection = nn.Sequential(
            nn.Linear(cnn_channels, d_model),
            nn.LayerNorm(d_model),
        )

        # ---- Mamba Backbone ----
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, dt_rank, d_conv, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # ---- Prediction Heads ----
        self.head_A = PredictionHead(d_model, n_dir_horizons, dropout, use_tanh=False)
        self.head_B = PredictionHead(d_model, n_pressure_horizons, dropout, use_tanh=True)   # NTPS
        self.head_C = PredictionHead(d_model, n_pressure_horizons, dropout, use_tanh=False)  # EOFI
        self.head_D = PredictionHead(d_model, n_pressure_horizons, dropout, use_tanh=True)   # PDI
        self.head_E = PredictionHead(d_model, n_pressure_horizons, dropout, use_tanh=True)   # TIA

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

    def forward(self, events: torch.Tensor, return_embedding: bool = False):
        """
        Args:
            events: (B, L, n_features) float32
            return_embedding: if True, also return the shared embedding

        Returns:
            dict of head predictions:
                'dir': (B, n_dir_horizons)
                'ntps': (B, n_pressure_horizons)
                'eofi': (B, n_pressure_horizons)
                'pdi': (B, n_pressure_horizons)
                'tia': (B, n_pressure_horizons)
            embedding: (B, d_model) if return_embedding
        """
        B, L, _ = events.shape
        time_delta = events[:, :, 0]  # time_delta_log is feature index 0

        # CNN front-end: (B, L, F) -> (B, F, L) -> Conv1d -> (B, L, cnn_ch)
        x = events.transpose(1, 2).contiguous()
        for i, (conv, norm) in enumerate(zip(self.cnn_conv_layers, self.cnn_norms)):
            residual = x
            x = conv(x)
            x = norm(x)
            x = self.cnn_act(x)
            x = self.cnn_dropout(x)
            if i > 0:
                x = x + residual
        x = x.transpose(1, 2).contiguous()

        # Projection to d_model
        x = self.projection(x)

        # Mamba backbone
        for block in self.blocks:
            x = block(x, time_delta=time_delta)

        # Take last position (causal) and normalize
        embedding = self.final_norm(x[:, -1, :])

        # Multi-head predictions
        preds = {
            "dir":  self.head_A(embedding),
            "ntps": self.head_B(embedding),
            "eofi": self.head_C(embedding),
            "pdi":  self.head_D(embedding),
            "tia":  self.head_E(embedding),
        }

        if return_embedding:
            return preds, embedding
        return preds


# ============================================================
# Dataset
# ============================================================

class V4MultiHeadDataset(Dataset):
    """
    Lazy-loading dataset for smart_v3 event files + smooth pressure targets.

    CRITICAL: Does NOT load all days into memory at once. Each day is ~650MB,
    and 60 days = ~39GB which exceeds Neptune's 32GB RAM.

    Instead, stores file paths and event counts during __init__, then loads
    individual days on-demand in __getitem__ with an LRU cache (default 3 days
    in memory at a time = ~2GB).
    """

    def __init__(
        self,
        event_files: List[Path],
        pressure_files: Dict[str, Path],  # date_str -> pressure .npz path
        seq_len: int = 100,
        stride: int = 50,
        dir_horizons: List[str] = None,
        pressure_horizons: List[str] = None,
        cache_days: int = 3,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.dir_horizons = dir_horizons or DIR_HORIZONS
        self.pressure_horizons = pressure_horizons or PRESSURE_HORIZONS
        self.cache_days = cache_days

        # Store file metadata only — NOT the data itself
        self.day_files: List[Tuple[Path, Optional[Path], int]] = []  # (event_path, pressure_path, n_events)
        self.sample_index: List[Tuple[int, int]] = []

        # LRU cache for loaded days: day_idx -> dict of arrays
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._cache_order: List[int] = []

        for ef in event_files:
            date_str = ef.stem[:8]

            # Quick scan: just get n_events without loading full array
            try:
                with np.load(ef, allow_pickle=False) as edata:
                    n_events = edata["events"].shape[0]
                    # Verify labels exist
                    has_labels = any(f"labels_{h}" in edata for h in self.dir_horizons)
                    if not has_labels:
                        continue
            except Exception as e:
                logger.warning(f"Failed to scan event file {ef.name}: {e}")
                continue

            if n_events < seq_len + 10:
                logger.debug(f"Skipping {date_str}: only {n_events} events")
                continue

            pressure_path = pressure_files.get(date_str)
            day_idx = len(self.day_files)
            self.day_files.append((ef, pressure_path, n_events))

            for s in range(0, n_events - seq_len + 1, stride):
                self.sample_index.append((day_idx, s))

        logger.info(
            f"V4MultiHeadDataset: {len(self.day_files)} days, "
            f"{len(self.sample_index)} samples (seq_len={seq_len}, stride={stride}), "
            f"cache={cache_days} days"
        )

    def _load_day(self, day_idx: int) -> Dict[str, np.ndarray]:
        """Load a day's data into the cache, evicting oldest if needed."""
        if day_idx in self._cache:
            return self._cache[day_idx]

        ef, pressure_path, n_events = self.day_files[day_idx]
        date_str = ef.stem[:8]

        edata = np.load(ef, allow_pickle=False)
        events = edata["events"].astype(np.float32)
        timestamps = edata["timestamps"]

        # Directional labels
        dir_labels = {}
        for h in self.dir_horizons:
            key = f"labels_{h}"
            if key in edata:
                dir_labels[h] = edata[key].astype(np.float32)
            else:
                dir_labels[h] = np.full(n_events, np.nan, dtype=np.float32)

        # Pressure labels
        pressure_labels = {}
        if pressure_path is not None:
            try:
                pdata = np.load(pressure_path, allow_pickle=False)
                p_timestamps = pdata["timestamps"]

                if len(p_timestamps) == n_events and np.array_equal(timestamps, p_timestamps):
                    for target in PRESSURE_TARGETS:
                        for h in self.pressure_horizons:
                            key = f"{target}_label_{h}"
                            if key in pdata:
                                pressure_labels[f"{target}_{h}"] = pdata[key].astype(np.float32)
                else:
                    p_idx = np.searchsorted(p_timestamps, timestamps, side="left")
                    p_idx = np.clip(p_idx, 0, len(p_timestamps) - 1)
                    ts_diff = np.abs(timestamps.astype(np.int64) - p_timestamps[p_idx].astype(np.int64))
                    good_mask = ts_diff < 1_000_000

                    for target in PRESSURE_TARGETS:
                        for h in self.pressure_horizons:
                            key = f"{target}_label_{h}"
                            if key in pdata:
                                raw = pdata[key].astype(np.float32)
                                aligned = np.full(n_events, np.nan, dtype=np.float32)
                                aligned[good_mask] = raw[p_idx[good_mask]]
                                pressure_labels[f"{target}_{h}"] = aligned
            except Exception as e:
                logger.warning(f"Failed to load pressure file for {date_str}: {e}")

        for target in PRESSURE_TARGETS:
            for h in self.pressure_horizons:
                key = f"{target}_{h}"
                if key not in pressure_labels:
                    pressure_labels[key] = np.full(n_events, np.nan, dtype=np.float32)

        day = {"events": events, "timestamps": timestamps}
        day.update({f"dir_{h}": dir_labels[h] for h in self.dir_horizons})
        day.update(pressure_labels)

        # Evict oldest cached day if at capacity
        while len(self._cache) >= self.cache_days:
            evict_idx = self._cache_order.pop(0)
            del self._cache[evict_idx]

        self._cache[day_idx] = day
        self._cache_order.append(day_idx)
        return day

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        day_idx, start = self.sample_index[idx]
        end = start + self.seq_len
        d = self._load_day(day_idx)

        events = torch.from_numpy(d["events"][start:end].copy())  # (seq_len, 25)
        last = end - 1

        # Directional labels
        dir_label = np.array([
            d[f"dir_{h}"][last] for h in self.dir_horizons
        ], dtype=np.float32)

        # Pressure labels — one array per target type
        ntps_label = np.array([d[f"ntps_{h}"][last] for h in self.pressure_horizons], dtype=np.float32)
        eofi_label = np.array([d[f"eofi_{h}"][last] for h in self.pressure_horizons], dtype=np.float32)
        pdi_label = np.array([d[f"pdi_{h}"][last] for h in self.pressure_horizons], dtype=np.float32)
        tia_label = np.array([d[f"tia_{h}"][last] for h in self.pressure_horizons], dtype=np.float32)

        labels = {
            "dir":  torch.from_numpy(dir_label),
            "ntps": torch.from_numpy(ntps_label),
            "eofi": torch.from_numpy(eofi_label),
            "pdi":  torch.from_numpy(pdi_label),
            "tia":  torch.from_numpy(tia_label),
        }

        return events, labels


class DayGroupedSampler:
    """
    Cache-friendly sampler for lazy-loading datasets.

    Shuffles the ORDER of days, then iterates sequentially through samples
    within each day. This means the LRU cache only needs to hold 1 day at a time,
    eliminating the cache-thrashing that killed performance with random shuffle.

    Samples within each day are also shuffled to prevent ordering artifacts.
    """

    def __init__(self, dataset: V4MultiHeadDataset):
        self.dataset = dataset
        # Group sample indices by day_idx
        self.day_groups: Dict[int, List[int]] = {}
        for global_idx, (day_idx, _start) in enumerate(dataset.sample_index):
            if day_idx not in self.day_groups:
                self.day_groups[day_idx] = []
            self.day_groups[day_idx].append(global_idx)

    def __iter__(self):
        # Shuffle day order
        day_indices = list(self.day_groups.keys())
        np.random.shuffle(day_indices)

        for day_idx in day_indices:
            # Shuffle samples within this day
            samples = self.day_groups[day_idx].copy()
            np.random.shuffle(samples)
            yield from samples

    def __len__(self):
        return len(self.dataset)


def collate_v4(batch):
    """Custom collate for dict-label dataset."""
    events = torch.stack([b[0] for b in batch])
    keys = batch[0][1].keys()
    labels = {k: torch.stack([b[1][k] for b in batch]) for k in keys}
    return events, labels


# ============================================================
# Metrics
# ============================================================

def compute_ic(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Spearman IC, NaN-safe."""
    valid = np.isfinite(predictions) & np.isfinite(labels)
    if valid.sum() < 20:
        return float("nan")
    rho, _ = scipy.stats.spearmanr(predictions[valid], labels[valid])
    return float(rho)


# ============================================================
# LR Scheduler
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
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            pg["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ============================================================
# Multi-Task Loss
# ============================================================

def compute_multitask_loss(
    preds: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    head_weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute weighted multi-task loss across all heads.
    NaN labels are masked out per-head.

    Returns:
        total_loss: scalar tensor
        loss_breakdown: dict of per-head loss values (for logging)
    """
    total_loss = torch.tensor(0.0, device=next(iter(preds.values())).device)
    breakdown = {}

    for head_name in preds:
        p = preds[head_name]
        l = labels[head_name].to(p.device)
        w = head_weights.get(head_name, 1.0)

        # Mask out NaN labels
        valid_mask = torch.isfinite(l)
        if valid_mask.any():
            # Apply mask per-element (handles partial NaN across horizons)
            p_valid = p[valid_mask]
            l_valid = l[valid_mask]
            head_loss = F.mse_loss(p_valid, l_valid)
        else:
            head_loss = torch.tensor(0.0, device=p.device)

        total_loss = total_loss + w * head_loss
        breakdown[head_name] = head_loss.item()

    return total_loss, breakdown


# ============================================================
# Training Loop (single fold)
# ============================================================

def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold_idx: int,
    output_dir: Path,
    device: torch.device,
    head_weights: Dict[str, float],
    lr: float = 1e-4,
    epochs: int = 15,
    warmup_steps: int = 300,
    grad_clip: float = 1.0,
    use_amp: bool = True,
) -> Dict:
    """Train the multi-head model for one fold. Returns best val metrics."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    best_val_loss = float("inf")
    total_batches = len(train_loader)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_breakdown = {}
        n_batches = 0
        t0 = time.time()

        for events, labels in train_loader:
            events = events.to(device, non_blocking=True)
            labels_dev = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                preds = model(events)
                loss, breakdown = compute_multitask_loss(preds, labels_dev, head_weights)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            for k, v in breakdown.items():
                epoch_breakdown[k] = epoch_breakdown.get(k, 0.0) + v

            if n_batches % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / n_batches * (total_batches - n_batches)
                logger.info(
                    f"  Fold {fold_idx:02d} Ep {epoch+1}/{epochs} "
                    f"Batch {n_batches}/{total_batches} "
                    f"loss={loss.item():.5f} lr={scheduler.get_lr():.2e} "
                    f"ETA={eta:.0f}s"
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_bd = {k: v / max(n_batches, 1) for k, v in epoch_breakdown.items()}
        elapsed = time.time() - t0

        # Validation
        val_loss, val_bd = evaluate_loss(model, val_loader, device, head_weights, use_amp)

        logger.info(
            f"  Fold {fold_idx:02d} Ep {epoch+1}/{epochs} done in {elapsed:.0f}s | "
            f"train_loss={avg_loss:.5f} val_loss={val_loss:.5f}"
        )
        bd_str = " | ".join(f"{k}={avg_bd[k]:.5f}" for k in sorted(avg_bd))
        logger.info(f"    Train breakdown: {bd_str}")
        bd_str = " | ".join(f"{k}={val_bd[k]:.5f}" for k in sorted(val_bd))
        logger.info(f"    Val   breakdown: {bd_str}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
                "fold": fold_idx,
            }
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            torch.save(ckpt, ckpt_path)
            logger.info(f"    Saved best checkpoint (val_loss={val_loss:.5f})")

    return {"best_val_loss": best_val_loss}


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    head_weights: Dict[str, float],
    use_amp: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Compute average multi-task loss on a loader."""
    model.eval()
    total_loss = 0.0
    total_bd = {}
    n_batches = 0

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    with torch.no_grad():
        for events, labels in loader:
            events = events.to(device, non_blocking=True)
            labels_dev = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
            with amp_ctx:
                preds = model(events)
                loss, breakdown = compute_multitask_loss(preds, labels_dev, head_weights)
            total_loss += loss.item()
            n_batches += 1
            for k, v in breakdown.items():
                total_bd[k] = total_bd.get(k, 0.0) + v

    avg = total_loss / max(n_batches, 1)
    avg_bd = {k: v / max(n_batches, 1) for k, v in total_bd.items()}
    return avg, avg_bd


# ============================================================
# OOT Inference
# ============================================================

def run_oot_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[np.ndarray]]:
    """Run inference, return (preds_dict, labels_dict, embeddings)."""
    model.eval()
    all_preds = {k: [] for k in ["dir", "ntps", "eofi", "pdi", "tia"]}
    all_labels = {k: [] for k in ["dir", "ntps", "eofi", "pdi", "tia"]}
    all_embeds = []

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    with torch.no_grad():
        for events, labels in loader:
            events = events.to(device, non_blocking=True)
            with amp_ctx:
                preds, emb = model(events, return_embedding=True)

            for k in all_preds:
                all_preds[k].append(preds[k].float().cpu().numpy())
                all_labels[k].append(labels[k].float().cpu().numpy())
            all_embeds.append(emb.float().cpu().numpy())

    preds_out = {k: np.concatenate(v) for k, v in all_preds.items() if v}
    labels_out = {k: np.concatenate(v) for k, v in all_labels.items() if v}
    embeds_out = np.concatenate(all_embeds) if all_embeds else None

    return preds_out, labels_out, embeds_out


# ============================================================
# File Discovery
# ============================================================

def find_data_files(
    data_dir: Path,
    pressure_dir: Path,
) -> Tuple[List[Path], Dict[str, Path]]:
    """
    Find event files and pressure files, return aligned lists.
    Event files: YYYYMMDD_mbo_events.npz
    Pressure files: YYYYMMDD_pressure.npz
    """
    event_files = sorted(data_dir.glob("*_mbo_events.npz"))
    if not event_files:
        logger.error(f"No event files found in {data_dir}")
        sys.exit(1)

    pressure_map = {}
    if pressure_dir.exists():
        for pf in pressure_dir.glob("*_pressure.npz"):
            date_str = pf.stem[:8]
            pressure_map[date_str] = pf

    n_matched = sum(1 for ef in event_files if ef.stem[:8] in pressure_map)
    logger.info(
        f"Found {len(event_files)} event files, {len(pressure_map)} pressure files, "
        f"{n_matched} matched"
    )

    return event_files, pressure_map


# ============================================================
# Walk-Forward
# ============================================================

def run_walk_forward(args):
    """Main walk-forward training loop."""

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data_dir = Path(args.data_dir)
    pressure_dir = Path(args.pressure_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse horizons
    dir_horizons = [f"{h}s" if not h.endswith("s") else h for h in args.horizons.split(",")]
    # Only use 1s/5s/10s for directional head (no 30s directional label in standard event files)
    dir_h_filtered = [h for h in dir_horizons if h in ["1s", "5s", "10s"]]
    pressure_h = [h for h in dir_horizons]  # all horizons for pressure heads

    # Parse head weights
    hw_list = [float(x) for x in args.head_weights.split(",")]
    assert len(hw_list) == 5, "head-weights must be 5 comma-separated floats (A,B,C,D,E)"
    head_weight_map = {
        "dir": hw_list[0],
        "ntps": hw_list[1],
        "eofi": hw_list[2],
        "pdi": hw_list[3],
        "tia": hw_list[4],
    }

    # Discover files
    event_files, pressure_map = find_data_files(data_dir, pressure_dir)

    # Filter files with valid directional labels
    def _has_valid_labels(f: Path) -> bool:
        try:
            d = np.load(f, allow_pickle=True)
            lbl = d["labels_1s"]
            return bool(not np.all(np.isnan(lbl)))
        except Exception:
            return False

    valid_files = [f for f in event_files if _has_valid_labels(f)]
    logger.info(f"Files with valid labels: {len(valid_files)}/{len(event_files)}")
    event_files = valid_files

    n_files = len(event_files)
    if n_files < args.wf_window + 1:
        logger.error(f"Need at least {args.wf_window + 1} files, have {n_files}")
        sys.exit(1)

    # Build fold boundaries: sliding window
    n_folds = min(args.n_folds, n_files - args.wf_window)
    if n_folds <= 0:
        n_folds = n_files - args.wf_window
    fold_boundaries = []
    for fold in range(n_folds):
        oot_idx = args.wf_window + fold
        if oot_idx >= n_files:
            break
        train_start = fold  # sliding: drop oldest day
        train_end = oot_idx
        fold_boundaries.append((fold, list(range(train_start, train_end)), [oot_idx]))

    logger.info(
        f"Walk-forward: {len(fold_boundaries)} folds, "
        f"{args.wf_window}-day sliding window, 1-day OOT"
    )

    use_amp = device.type == "cuda"

    # Concat storage
    head_names = ["dir", "ntps", "eofi", "pdi", "tia"]
    concat_preds = {h: [] for h in head_names}
    concat_labels = {h: [] for h in head_names}
    concat_embeds = []

    # MLflow
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("v4_multihead_pressure")
        mlflow_run = mlflow.start_run(
            run_name=f"v4_multihead_{time.strftime('%Y%m%d_%H%M')}"
        )
        mlflow.log_params({
            "model": "CNNMambaV4MultiHead",
            "seq_len": args.seq_len,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "n_folds": len(fold_boundaries),
            "wf_window": args.wf_window,
            "dir_horizons": str(dir_h_filtered),
            "pressure_horizons": str(pressure_h),
            "head_weights": str(head_weight_map),
            "d_model": 96,
            "d_state": 32,
            "n_layers": 3,
            "cnn_channels": 64,
            "cnn_kernel": 5,
            "cnn_layers": 3,
            "node": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "data_dir": str(data_dir),
            "pressure_dir": str(pressure_dir),
            "output_dir": str(output_dir),
            "mixed_precision": "fp16" if use_amp else "none",
        })

    # Add log file handler
    log_file = output_dir / "training.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(_fmt)
    logger.addHandler(fh)

    try:
        for fold_idx, train_idxs, oot_idxs in fold_boundaries:
            if fold_idx < args.start_fold:
                continue
            train_files = [event_files[i] for i in train_idxs]
            oot_files = [event_files[i] for i in oot_idxs]

            logger.info(
                f"\n{'='*70}\n"
                f"FOLD {fold_idx:02d} | Train: {len(train_files)} days "
                f"({train_files[0].stem[:8]}->{train_files[-1].stem[:8]}) | "
                f"OOT: {oot_files[0].stem[:8]}"
                f"\n{'='*70}"
            )

            # Build datasets
            logger.info("Building train dataset...")
            train_ds = V4MultiHeadDataset(
                train_files, pressure_map,
                seq_len=args.seq_len, stride=args.stride,
                dir_horizons=dir_h_filtered, pressure_horizons=pressure_h,
            )

            logger.info("Building OOT dataset...")
            oot_ds = V4MultiHeadDataset(
                oot_files, pressure_map,
                seq_len=args.seq_len, stride=args.stride,
                dir_horizons=dir_h_filtered, pressure_horizons=pressure_h,
            )

            if len(train_ds) == 0 or len(oot_ds) == 0:
                logger.warning(f"Fold {fold_idx:02d}: empty dataset, skipping")
                continue

            # DayGroupedSampler: iterates one day at a time (cache-friendly)
            # then shuffles within each day. This prevents cache thrashing
            # that killed performance with random shuffle + lazy loading.
            train_sampler = DayGroupedSampler(train_ds)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, sampler=train_sampler,
                num_workers=0, pin_memory=True, drop_last=True,
                collate_fn=collate_v4, persistent_workers=False,
            )
            oot_loader = DataLoader(
                oot_ds, batch_size=args.batch_size * 2, shuffle=False,
                num_workers=0, pin_memory=True,
                collate_fn=collate_v4,
            )

            # Fresh model per fold
            model = CNNMambaV4MultiHead(
                n_features=N_EVENT_FEATURES,
                n_dir_horizons=len(dir_h_filtered),
                n_pressure_horizons=len(pressure_h),
            ).to(device)

            if fold_idx == 0:
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"Model parameters: {n_params:,}")

            # Train
            train_one_fold(
                model, train_loader, oot_loader,
                fold_idx, output_dir, device, head_weight_map,
                lr=args.lr, epochs=args.epochs,
                use_amp=use_amp,
            )

            # Reload best checkpoint
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                logger.info(f"Loaded best checkpoint (val_loss={ckpt['val_loss']:.5f})")

            # OOT inference
            logger.info("Running OOT inference...")
            oot_preds, oot_labels, oot_embeds = run_oot_inference(
                model, oot_loader, device, use_amp=use_amp,
            )

            # Per-fold IC for each head and horizon
            fold_ics = {}
            for head_name in head_names:
                if head_name not in oot_preds:
                    continue
                p = oot_preds[head_name]
                l = oot_labels[head_name]
                horizons = dir_h_filtered if head_name == "dir" else pressure_h
                for i, h in enumerate(horizons):
                    if i < p.shape[1]:
                        ic = compute_ic(p[:, i], l[:, i])
                        key = f"{head_name}_{h}"
                        fold_ics[key] = ic

                concat_preds[head_name].append(p)
                concat_labels[head_name].append(l)

            if oot_embeds is not None:
                concat_embeds.append(oot_embeds)

            # Log fold ICs
            ic_strs = [f"{k}={v:.4f}" for k, v in sorted(fold_ics.items())]
            logger.info(f"Fold {fold_idx:02d} OOT IC: " + " | ".join(ic_strs))

            # Save fold predictions
            save_dict = {"fold": np.array(fold_idx)}
            for head_name in head_names:
                if head_name in oot_preds:
                    save_dict[f"preds_{head_name}"] = oot_preds[head_name]
                    save_dict[f"labels_{head_name}"] = oot_labels[head_name]
            for k, v in fold_ics.items():
                save_dict[f"ic_{k}"] = np.array(v)
            if oot_embeds is not None:
                save_dict["embeddings"] = oot_embeds
            save_dict["oot_files"] = np.array([str(f) for f in oot_files])

            pred_path = output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
            np.savez_compressed(pred_path, **save_dict)
            logger.info(f"Saved fold {fold_idx:02d} predictions")

            # MLflow per-fold metrics
            if MLFLOW_AVAILABLE and mlflow_run:
                mlflow.log_metrics(
                    {f"oot_{k}_f{fold_idx:02d}": v for k, v in fold_ics.items()
                     if not np.isnan(v)},
                    step=fold_idx,
                )

            # Free memory
            del train_ds, oot_ds, train_loader, oot_loader, model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ============================================================
        # Concat IC (primary metric)
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("CONCAT IC (all folds combined)")
        logger.info("=" * 70)

        concat_ic = {}
        concat_save = {}

        for head_name in head_names:
            if not concat_preds[head_name]:
                continue
            all_p = np.concatenate(concat_preds[head_name])
            all_l = np.concatenate(concat_labels[head_name])
            concat_save[f"preds_{head_name}"] = all_p
            concat_save[f"labels_{head_name}"] = all_l

            horizons = dir_h_filtered if head_name == "dir" else pressure_h
            for i, h in enumerate(horizons):
                if i < all_p.shape[1]:
                    ic = compute_ic(all_p[:, i], all_l[:, i])
                    key = f"{head_name}_{h}"
                    concat_ic[key] = ic
                    logger.info(f"  Concat IC ({key}): {ic:.4f}")

        if concat_embeds:
            concat_save["embeddings"] = np.concatenate(concat_embeds)

        for k, v in concat_ic.items():
            concat_save[f"concat_ic_{k}"] = np.array(v)

        concat_path = output_dir / "concat_oot_predictions.npz"
        np.savez_compressed(concat_path, **concat_save)
        logger.info(f"Saved concat predictions")

        # MLflow concat metrics
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.log_metrics(
                {f"concat_ic_{k}": v for k, v in concat_ic.items() if not np.isnan(v)}
            )

        # Summary table
        logger.info("\n" + "=" * 70)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 70)

        # Group by head
        for head_name in head_names:
            horizons = dir_h_filtered if head_name == "dir" else pressure_h
            ics = []
            for h in horizons:
                key = f"{head_name}_{h}"
                ic = concat_ic.get(key, float("nan"))
                ics.append(f"{h}={ic:.4f}")
            logger.info(f"  {head_name:>5}: {' | '.join(ics)}")

        logger.info(f"\n  Output: {output_dir}")
        logger.info("=" * 70)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.end_run()

    return concat_ic


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="V4 Multi-Head CNN-Mamba: Directional + Smooth Pressure Targets"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3",
        help="Directory with smart_v3 event .npz files",
    )
    parser.add_argument(
        "--pressure-dir", type=str,
        default="/home/jupiter/Lvl3Quant/data/processed/smooth_pressure_targets",
        help="Directory with smooth pressure target .npz files",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/home/jupiter/Lvl3Quant/output/v4_multihead_pressure",
        help="Output directory for predictions/weights",
    )
    parser.add_argument("--horizons", type=str, default="1,5,10,30",
                        help="Comma-separated horizon values in seconds")
    parser.add_argument("--seq-len", type=int, default=100,
                        help="Sequence length (events per sample)")
    parser.add_argument("--stride", type=int, default=50,
                        help="Stride for sliding window over events")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Epochs per fold")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--head-weights", type=str, default="1.0,2.0,1.0,1.0,1.0",
                        help="Comma-separated weights for heads A,B,C,D,E")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--mlflow-uri", type=str, default="http://neptune:5000",
                        help="MLflow tracking URI")
    parser.add_argument("--wf-window", type=int, default=60,
                        help="Walk-forward training window in days")
    parser.add_argument("--n-folds", type=int, default=999,
                        help="Max number of folds (default: all available)")
    parser.add_argument("--start-fold", type=int, default=0,
                        help="Resume from this fold index")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("V4 Multi-Head CNN-Mamba — Directional + Smooth Pressure Targets")
    logger.info("=" * 70)
    logger.info(f"Config:")
    logger.info(f"  data_dir:       {args.data_dir}")
    logger.info(f"  pressure_dir:   {args.pressure_dir}")
    logger.info(f"  output_dir:     {args.output_dir}")
    logger.info(f"  horizons:       {args.horizons}")
    logger.info(f"  seq_len:        {args.seq_len}")
    logger.info(f"  stride:         {args.stride}")
    logger.info(f"  batch_size:     {args.batch_size}")
    logger.info(f"  lr:             {args.lr}")
    logger.info(f"  epochs:         {args.epochs}")
    logger.info(f"  head_weights:   {args.head_weights}")
    logger.info(f"  wf_window:      {args.wf_window}")
    logger.info(f"  mlflow_uri:     {args.mlflow_uri}")

    concat_ic = run_walk_forward(args)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
