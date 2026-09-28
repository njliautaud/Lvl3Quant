#!/usr/bin/env python3
"""
mamba_inference.py — CPU-compatible Mamba inference for live trading.

Supports two architectures (auto-detected from checkpoint):
    1. CNN-Mamba v2 (production): feature_mlp + temporal_cnns + fusion_proj + Mamba backbone
    2. Legacy Mamba v7: input_proj + Mamba backbone (EventMambaCPU)

Key design:
    - Auto-detects architecture from checkpoint state_dict keys
    - Infers all architecture params from weight shapes (dt_rank, mlp_hidden, etc.)
    - Pure PyTorch forward pass (no CUDA required)
    - strict=False for load_state_dict (3 missing time_decay_rate keys expected)

Usage:
    engine = MambaInferenceEngine(
        weights_path="output/cnn_mamba_v2_smart_v3_mar/fold_10_best.pt",
        stats_path="output/cnn_mamba_v2_smart_v3_mar/fold_09_feature_stats.npz",
    )
    # Feed raw smart_v3 features (25 features per event)
    predictions = engine.predict(feature_window)  # (1000, 25) -> (3,) [1s, 5s, 10s]
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, Dict


# ── Selective SSM (CPU-compatible, matches CUDA Mamba parameter shapes) ──

class CUDACompatSSM(nn.Module):
    """
    Selective SSM that matches mamba_ssm.Mamba parameter shapes exactly.
    Runs on CPU using chunked sequential scan.
    """

    def __init__(self, d_model: int, d_state: int = 32, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        # Match CUDA Mamba's dt_rank computation
        self.dt_rank = math.ceil(d_model / 16)

        # Input projection: d_model -> 2 * d_inner
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise conv1d (causal)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=0,
            groups=self.d_inner, bias=True,
        )

        # Selective projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # SSM parameters
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            out: (B, L, d_model)
        """
        B, L, _ = x.shape

        # Input projection -> x_branch and z (gate)
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal conv1d
        x_conv = x_branch.transpose(1, 2).contiguous()  # (B, d_inner, L)
        x_conv = F.pad(x_conv, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.transpose(1, 2).contiguous()  # (B, L, d_inner)
        x_branch = F.silu(x_conv)

        # Selective parameters
        x_proj = self.x_proj(x_branch)  # (B, L, dt_rank + 2*d_state)
        dt_x = x_proj[:, :, :self.dt_rank]
        B_sel = x_proj[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C_sel = x_proj[:, :, self.dt_rank + self.d_state:]

        # Delta
        dt = F.softplus(self.dt_proj(dt_x))  # (B, L, d_inner)

        # A from log space
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Selective scan (chunked sequential for CPU)
        y = self._scan(x_branch, dt, A, B_sel, C_sel)

        # Skip connection
        y = y + x_branch * self.D.unsqueeze(0).unsqueeze(0)

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)

    def _scan(self, x, dt, A, B, C):
        """
        Selective scan with GPU-optimized chunked approach.
        On GPU: uses larger chunks with batched matmul to minimize kernel launches.
        On CPU: falls back to smaller chunks.
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        is_cuda = x.is_cuda

        A_exp = A.unsqueeze(0).unsqueeze(0)  # (1, 1, D, N)
        dt_exp = dt.unsqueeze(-1)  # (B, L, D, 1)
        dA = torch.exp(A_exp * dt_exp)  # (B, L, D, N)
        dBx = B.unsqueeze(2) * dt_exp * x.unsqueeze(-1)  # (B, L, D, N)

        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        if is_cuda:
            # GPU path: process all timesteps, collect outputs in pre-allocated tensor
            y_all = torch.empty(batch, seq_len, d_inner, device=x.device, dtype=x.dtype)
            for t in range(seq_len):
                h = dA[:, t] * h + dBx[:, t]
                # (B, N) x (B, D, N) -> (B, D) via batched dot
                y_all[:, t] = (C[:, t].unsqueeze(1) * h).sum(-1)
            return y_all
        else:
            # CPU path: chunked with list accumulation (original)
            CHUNK = 64
            outputs = []
            for t_start in range(0, seq_len, CHUNK):
                t_end = min(t_start + CHUNK, seq_len)
                chunk_outs = []
                for t in range(t_start, t_end):
                    h = dA[:, t] * h + dBx[:, t]
                    y_t = torch.einsum("bn,bdn->bd", C[:, t], h)
                    chunk_outs.append(y_t)
                outputs.append(torch.stack(chunk_outs, dim=1))
            return torch.cat(outputs, dim=1)


class MambaBlockCPU(nn.Module):
    """MambaBlock compatible with CUDA-trained weights."""

    def __init__(self, d_model: int, d_state: int = 32, d_conv: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = CUDACompatSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        return x + residual


class EventMambaCPU(nn.Module):
    """
    CPU-compatible EventMamba that loads CUDA-trained weights.
    Matches the exact architecture from train_event_mamba_cuda.py.
    """

    def __init__(self, n_features: int = 25, d_model: int = 96, d_state: int = 32,
                 n_layers: int = 3, d_conv: int = 4, dropout: float = 0.1, n_targets: int = 3):
        super().__init__()
        self.d_model = d_model

        # Input projection: n_features -> d_model
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlockCPU(d_model=d_model, d_state=d_state, d_conv=d_conv, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Args:
            events: (B, L, n_features) normalized feature windows
        Returns:
            preds: (B, n_targets) predicted price changes [1s, 5s, 10s]
        """
        x = self.input_proj(events)
        for block in self.blocks:
            x = block(x)
        x_last = x[:, -1, :]
        embedding = self.final_norm(x_last)
        return self.head(embedding)


# ── CNN-Mamba v2 Architecture (production) ──

class SelectiveSSMv2(nn.Module):
    """Mamba SSM with time-delta gating — matches blocks.N.ssm.* keys in CNN-Mamba v2."""

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

        # Causal conv1d
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

        A_exp = A.unsqueeze(0).unsqueeze(0)
        dt_exp = dt.unsqueeze(-1)
        dA_all = torch.exp(A_exp * dt_exp)

        if time_delta is not None:
            td = time_delta.unsqueeze(-1).unsqueeze(-1)
            dr = F.softplus(self.time_decay_rate).unsqueeze(0).unsqueeze(0)
            dA_all = dA_all * torch.exp(-dr * td.abs())

        dB_all = B.unsqueeze(2) * dt_exp
        dBx_all = dB_all * x.unsqueeze(-1)

        y = torch.empty(batch, seq_len, d_inner, device=x.device, dtype=x.dtype)
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        D_term = D.unsqueeze(0).unsqueeze(0)

        for t in range(seq_len):
            h = dA_all[:, t] * h + dBx_all[:, t]
            y[:, t] = (h * C[:, t].unsqueeze(1)).sum(-1)

        return y + x * D_term


class MambaBlockV2(nn.Module):
    def __init__(self, d_model, d_state=32, dt_rank=6, d_conv=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSMv2(d_model, d_state, dt_rank, d_conv)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_delta=None):
        residual = x
        x = self.norm(x)
        x = self.ssm(x, time_delta=time_delta)
        x = self.dropout(x)
        return x + residual


class CNNMambaV2(nn.Module):
    """
    CNN-Mamba v2 — Feature MLP + Multi-Scale Temporal CNN + Mamba backbone.
    Production architecture matching fold_10_best.pt.
    """
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

        # Pathway 1: Feature Interaction MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(n_smart_features, feature_mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(feature_mlp_hidden, feature_mlp_out),
            nn.LayerNorm(feature_mlp_out),
        )

        # Pathway 2: Multi-Scale Temporal CNN
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
            MambaBlockV2(d_model, d_state, dt_rank, d_conv, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )

    def forward(self, smart: torch.Tensor, return_embedding: bool = False):
        B, L, _ = smart.shape
        time_delta = smart[:, :, 0]  # feature 0 = time_delta_log

        # Pathway 1: pointwise MLP at each timestep
        feat_out = self.feature_mlp(smart)

        # Pathway 2: multi-scale temporal CNN
        x_t = smart.transpose(1, 2)
        cnn_outputs = []
        for k, conv_block in zip(self.cnn_kernels, self.temporal_cnns):
            padded = F.pad(x_t, (k - 1, 0))
            cnn_outputs.append(conv_block(padded))
        cnn_cat = torch.cat(cnn_outputs, dim=1).transpose(1, 2)

        # Fuse and project
        fused = torch.cat([feat_out, cnn_cat], dim=-1)
        x = self.fusion_proj(fused)

        # Mamba blocks
        for block in self.blocks:
            x = block(x, time_delta=time_delta)

        x = self.final_norm(x)
        embedding = x[:, -1, :]
        preds = self.head(embedding)

        if return_embedding:
            return preds, embedding
        return preds


def _load_cnn_mamba_v2(ckpt: dict, device: torch.device) -> CNNMambaV2:
    """Build and load a CNNMambaV2 model from checkpoint, inferring all arch params from weights."""
    state = ckpt["model_state"]
    arch = ckpt.get("arch", {})

    # Infer architecture params from weight shapes
    d_state = state["blocks.0.ssm.A_log"].shape[1]
    x_proj = state["blocks.0.ssm.x_proj.weight"]
    dt_rank = x_proj.shape[0] - 2 * d_state

    n_features = state["feature_mlp.0.weight"].shape[1]
    d_model = arch.get("d_model", 96)
    n_layers = sum(1 for k in state if k.endswith(".ssm.A_log"))
    d_conv = arch.get("d_conv", 4)
    dropout = arch.get("dropout", 0.1)

    mlp_hidden = state["feature_mlp.0.weight"].shape[0]
    mlp_out = state["feature_mlp.3.weight"].shape[0]
    cnn_ch = state["temporal_cnns.0.0.weight"].shape[0]

    print(f"Loading CNN-Mamba v2 model:")
    print(f"  Architecture: d_model={d_model}, d_state={d_state}, dt_rank={dt_rank}, "
          f"n_layers={n_layers}, mlp_hidden={mlp_hidden}, mlp_out={mlp_out}, cnn_ch={cnn_ch}")
    print(f"  Fold: {ckpt.get('fold')}, Val IC_10s: {ckpt.get('val_ic_10s', 0):.4f}")

    model = CNNMambaV2(
        n_smart_features=n_features,
        d_model=d_model, d_state=d_state, n_layers=n_layers,
        dt_rank=dt_rank, d_conv=d_conv, dropout=dropout,
        feature_mlp_hidden=mlp_hidden, feature_mlp_out=mlp_out,
        cnn_channels_per_scale=cnn_ch,
    )

    # strict=False: fold_10_best.pt was trained before time_decay_rate was added
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys (expected for older ckpt): {missing}")
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    model.to(device)
    model.eval()
    return model


def _load_legacy_mamba_v7(ckpt: dict, device: torch.device) -> EventMambaCPU:
    """Build and load a legacy EventMambaCPU model from checkpoint."""
    arch = ckpt["arch"]
    print(f"Loading Legacy Mamba v7 model:")
    print(f"  Architecture: d_model={arch['d_model']}, d_state={arch['d_state']}, "
          f"n_layers={arch['n_layers']}, window={arch['window_size']}")
    print(f"  Fold: {ckpt['fold']}, Val IC_10s: {ckpt.get('val_ic_10s', 'N/A')}")

    model = EventMambaCPU(
        n_features=25,
        d_model=arch["d_model"],
        d_state=arch["d_state"],
        n_layers=arch["n_layers"],
        d_conv=arch.get("d_conv", 4),
        dropout=arch.get("dropout", 0.1),
        n_targets=3,
    )
    model.load_state_dict(ckpt["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    model.to(device)
    model.eval()
    return model


# ── Inference Engine ──

class MambaInferenceEngine:
    """
    Complete Mamba inference engine for live trading.

    Handles:
    - Model loading from CUDA-trained checkpoint
    - Feature normalization using saved per-fold stats
    - Sliding window management
    - Confidence tier classification
    - Multi-horizon prediction output
    """

    def __init__(
        self,
        weights_path: str,
        stats_path: str,
        window_size: int = None,
        stride: int = 500,
        device: str = "cpu",
    ):
        self.stride = stride
        self.device = torch.device(device)

        # Load checkpoint
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        arch = ckpt.get("arch", {})
        state = ckpt["model_state"]

        # Auto-detect architecture from state_dict keys
        state_keys = list(state.keys())
        has_feature_mlp = any(k.startswith("feature_mlp") for k in state_keys)
        has_input_proj = any(k.startswith("input_proj") for k in state_keys)

        if has_feature_mlp:
            # CNN-Mamba v2 (production)
            self.arch_type = "cnn_mamba_v2"
            self.model = _load_cnn_mamba_v2(ckpt, self.device)
            # Default window_size=1000 for CNN-Mamba v2
            self.window_size = window_size or arch.get("window_size", 1000)
        elif has_input_proj:
            # Legacy Mamba v7
            self.arch_type = "mamba_v7"
            self.model = _load_legacy_mamba_v7(ckpt, self.device)
            self.window_size = window_size or arch.get("window_size", 1000)
        else:
            raise ValueError(
                f"Cannot detect architecture from checkpoint keys. "
                f"First 10 keys: {state_keys[:10]}"
            )

        # Try torch.compile for GPU inference (PyTorch 2.0+)
        if self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print(f"  torch.compile enabled (reduce-overhead mode)")
            except Exception as e:
                print(f"  torch.compile unavailable: {e}")

        print(f"  Architecture type: {self.arch_type}")
        print(f"  Window size: {self.window_size}")
        print(f"  Device: {self.device}")

        # Load feature stats for normalization
        stats = np.load(stats_path)
        self.feat_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=self.device)
        self.feat_std = torch.tensor(stats["std"], dtype=torch.float32, device=self.device)
        # Avoid division by zero
        self.feat_std = torch.clamp(self.feat_std, min=1e-8)
        print(f"  Feature stats loaded: {len(self.feat_mean)} features")

        # Event buffer for streaming
        self.event_buffer = []
        self.prediction_count = 0

        # Confidence thresholds (will be calibrated from historical predictions)
        self.confidence_thresholds = {
            "Top5%": None,
            "Top1%": None,
            "Top0.5%": None,
            "Top0.1%": None,
        }

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features using saved per-fold statistics."""
        return (features - self.feat_mean) / self.feat_std

    @torch.no_grad()
    def predict(self, feature_window: np.ndarray) -> Dict:
        """
        Run inference on a single feature window.

        Args:
            feature_window: (window_size, n_features) raw features

        Returns:
            dict with:
                predictions: [pred_1s, pred_5s, pred_10s]
                confidence_1s: absolute value of 1s prediction
                direction: +1 (LONG) or -1 (SHORT)
                tier: confidence tier string or None
        """
        # Convert and normalize
        x = torch.tensor(feature_window, dtype=torch.float32, device=self.device)
        x = self.normalize(x)
        x = x.unsqueeze(0)  # (1, L, F)

        # Forward pass
        preds = self.model(x)  # (1, 3)
        preds = preds.squeeze(0).cpu().numpy()  # (3,)

        pred_1s, pred_5s, pred_10s = preds
        confidence = abs(pred_1s)
        direction = 1 if pred_1s > 0 else -1

        # Classify confidence tier
        tier = None
        for tier_name, threshold in self.confidence_thresholds.items():
            if threshold is not None and confidence >= threshold:
                tier = tier_name

        return {
            "predictions": preds.tolist(),
            "pred_1s": float(pred_1s),
            "pred_5s": float(pred_5s),
            "pred_10s": float(pred_10s),
            "confidence_1s": float(confidence),
            "direction": direction,
            "tier": tier,
        }

    def add_event(self, features: np.ndarray) -> Optional[Dict]:
        """
        Add a single event to the buffer. Returns prediction when stride is reached.

        Args:
            features: (n_features,) raw features for one event

        Returns:
            Prediction dict if stride reached, else None
        """
        self.event_buffer.append(features)

        # Check if we have enough events for a prediction
        if len(self.event_buffer) >= self.window_size:
            self.prediction_count += 1

            # Only predict every `stride` events after the first window
            if self.prediction_count == 1 or (len(self.event_buffer) - self.window_size) % self.stride == 0:
                window = np.array(self.event_buffer[-self.window_size:])
                result = self.predict(window)
                result["event_count"] = len(self.event_buffer)
                result["prediction_number"] = self.prediction_count

                # Trim buffer to avoid unbounded growth (keep 2x window)
                if len(self.event_buffer) > self.window_size * 2:
                    self.event_buffer = self.event_buffer[-self.window_size:]

                return result

        return None

    def calibrate_thresholds(self, predictions_path: str):
        """
        Calibrate confidence thresholds from historical predictions.

        Args:
            predictions_path: path to concat_oot_predictions.npz or fold NPZ
        """
        data = np.load(predictions_path, allow_pickle=True)
        # Handle different NPZ key formats
        if "predictions" in data:
            preds_1s = data["predictions"][:, 0]
        elif "preds_1s" in data:
            preds_1s = data["preds_1s"]
        else:
            raise KeyError(f"No prediction keys found. Available: {list(data.keys())}")
        confidences = np.abs(preds_1s)  # 1s magnitude

        self.confidence_thresholds = {
            "Top5%": float(np.percentile(confidences, 95)),
            "Top1%": float(np.percentile(confidences, 99)),
            "Top0.5%": float(np.percentile(confidences, 99.5)),
            "Top0.1%": float(np.percentile(confidences, 99.9)),
        }
        print(f"Calibrated confidence thresholds:")
        for tier, thresh in self.confidence_thresholds.items():
            print(f"  {tier}: >= {thresh:.4f}")

    def benchmark_speed(self, n_runs: int = 100):
        """Benchmark inference speed on CPU."""
        import time
        dummy = np.random.randn(self.window_size, 25).astype(np.float32)

        # Warmup
        for _ in range(5):
            self.predict(dummy)

        start = time.time()
        for _ in range(n_runs):
            self.predict(dummy)
        elapsed = time.time() - start

        ms_per_pred = (elapsed / n_runs) * 1000
        print(f"Inference speed: {ms_per_pred:.1f} ms/prediction ({1000/ms_per_pred:.0f} predictions/sec)")
        return ms_per_pred


# ── Test ──

if __name__ == "__main__":
    import sys

    LVL3 = Path("/home/jupiter/Lvl3Quant")
    WEIGHTS = LVL3 / "output/mamba_v7_tiny_smart_v3_mar_apr/fold_15_best.pt"
    STATS = LVL3 / "output/mamba_v7_tiny_smart_v3_mar_apr/fold_15_feature_stats.npz"
    PREDS = LVL3 / "output/mamba_v7_tiny_smart_v3_mar_apr/concat_oot_predictions.npz"

    if not WEIGHTS.exists():
        print(f"ERROR: Weights not found at {WEIGHTS}")
        sys.exit(1)

    engine = MambaInferenceEngine(
        weights_path=str(WEIGHTS),
        stats_path=str(STATS),
    )

    # Calibrate thresholds from historical predictions
    if PREDS.exists():
        engine.calibrate_thresholds(str(PREDS))

    # Benchmark
    engine.benchmark_speed(n_runs=50)

    # Test with random data
    print("\nTest prediction (random data):")
    dummy = np.random.randn(1000, 25).astype(np.float32)
    result = engine.predict(dummy)
    print(f"  1s: {result['pred_1s']:.4f}, 5s: {result['pred_5s']:.4f}, 10s: {result['pred_10s']:.4f}")
    print(f"  Direction: {'LONG' if result['direction'] == 1 else 'SHORT'}")
    print(f"  Confidence: {result['confidence_1s']:.4f}, Tier: {result['tier']}")

    # Test with real data if available
    test_data_path = LVL3 / "data/processed/mbo_events_smart_v3/20260313_mbo_events.npz"
    if test_data_path.exists():
        print(f"\nTest with real data ({test_data_path.name}):")
        src = np.load(test_data_path)
        features = src["events"][:1000]  # First window (key is 'events' in smart_v3)
        result = engine.predict(features)
        print(f"  1s: {result['pred_1s']:.4f}, 5s: {result['pred_5s']:.4f}, 10s: {result['pred_10s']:.4f}")
        print(f"  Direction: {'LONG' if result['direction'] == 1 else 'SHORT'}")
        print(f"  Confidence: {result['confidence_1s']:.4f}, Tier: {result['tier']}")
