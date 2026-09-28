"""
batch_inference_cpu.py
======================
Run CNN-Mamba v2 and PatchTST inference on CPU for dates not yet covered by
existing prediction NPZ files.

CNN-Mamba v2 architecture (fold_10_best.pt):
  - Feature MLP (25->128->64) + Multi-Scale Temporal CNN (kernels 3,7,15,31)
  - Fusion projection -> d_model=96
  - 3x Mamba blocks with time-delta aware SSM
  - Head: Linear(96,96) -> GELU -> Linear(96,3)
  window_size=1000, stride=500

PatchTST architecture (fold_15_best.pt):
  - Patch embedding (25 features, patch_size=25, d_model=256)
  - 4x Transformer blocks with ALiBi attention
  - Mean pool -> head
  window_size=500, stride=250

Output format (matches existing fold_NN_oot_predictions.npz):
  {predictions:(N,3), labels:(N,3), horizons:(3,), ic_1s, ic_5s, ic_10s,
   oot_files:(1,), embeddings:(N, d_model)}

Usage:
  python batch_inference_cpu.py [--model {cnn,ptst,both}] [--batch N]
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_default_root = Path("/home/jupiter/Lvl3Quant")
if not _default_root.exists() and Path("/home/nick/Lvl3Quant").exists():
    _default_root = Path("/home/nick/Lvl3Quant")
REPO_ROOT = Path(os.environ.get("LVL3_ROOT", str(_default_root)))
MBO_DIR   = REPO_ROOT / "data/processed/mbo_events_smart_v3"

CNN_WEIGHTS  = REPO_ROOT / "output/cnn_mamba_v2_smart_v3_mar/fold_10_best.pt"
CNN_OUT_DIR  = REPO_ROOT / "output/cnn_mamba_v2_smart_v3_mar"

# Use fold_15 (best IC=0.0632 across all PatchTST folds)
PTST_WEIGHTS = REPO_ROOT / "output/patchtst_razer_weights/fold_15_best.pt"
PTST_OUT_DIR = REPO_ROOT / "output/patchtst_smart_v3_mar"

# CNN-Mamba windowing (verified: w=1000, s=500 -> 37,189 samples on 20260305)
CNN_WINDOW = 1000
CNN_STRIDE = 500

# PatchTST windowing (from training config default)
PTST_WINDOW = 500
PTST_STRIDE = 250

HORIZONS   = ["1s", "5s", "10s"]
BATCH_SIZE = 256  # CPU batch

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_inference")


# ─────────────────────────────────────────────────────────────────────────────
# CNN-Mamba v2 Model — matches fold_10_best.pt
# Architecture: Feature MLP + Multi-Scale Temporal CNN + Mamba backbone
# Source: alpha_discovery/deep_models/test_model_decay.py:CNNMambaV2
# ─────────────────────────────────────────────────────────────────────────────

class SelectiveSSMv2(nn.Module):
    """Mamba SSM with time-delta gating — matches blocks.N.ssm.* keys in fold_10."""

    CNN_KERNELS = None  # not used here

    def __init__(self, d_model: int, d_state: int = 32, dt_rank: int = 6, d_conv: int = 4):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.dt_rank  = dt_rank
        self.d_conv   = d_conv
        self.d_inner  = d_model * 2

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=0, groups=self.d_inner, bias=True,
        )
        self.x_proj   = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj  = nn.Linear(dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log         = nn.Parameter(torch.log(A))
        self.D             = nn.Parameter(torch.ones(self.d_inner))
        self.time_decay_rate = nn.Parameter(torch.ones(self.d_inner, 1) * 0.1)
        self.out_proj      = nn.Linear(self.d_inner, d_model, bias=False)

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
        dt_x   = x_proj[:, :, :self.dt_rank]
        B_sel  = x_proj[:, :, self.dt_rank : self.dt_rank + self.d_state]
        C_sel  = x_proj[:, :, self.dt_rank + self.d_state:]

        dt = F.softplus(self.dt_proj(dt_x))
        A  = -torch.exp(self.A_log)

        y = self._scan(x_branch, dt, A, B_sel, C_sel, self.D, time_delta)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _scan(self, x, dt, A, B, C, D, time_delta=None):
        """
        Last-step-only scan optimized for inference.

        The model only uses h[:, -1, :] (last position) for prediction, so we:
        1. Only compute y at the final timestep (skip all intermediate y_t)
        2. Pre-compute all dA and dBx vectorized before the loop
        3. Keep only h (B, d_inner, d_state) updated each step — no output list

        The output tensor is filled with zeros for non-final positions and
        the actual SSM output only at position L-1. The gate+out_proj operations
        in forward() use only h[:,-1,:] implicitly through the residual structure,
        but we must return a full (B,L,d_inner) tensor for the gate multiplication.

        To avoid computing y at every step: use y = zeros, set y[:,-1] = last output,
        then the gate operation y * silu(z) is mostly zeroed. BUT: the residual
        `x + self.drop(self.ssm(self.norm(x)))` uses ALL positions (for next block).

        Therefore we must compute y_t at all positions but can skip stack():
        use a pre-allocated output buffer instead.
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Pre-compute all discretized transitions (vectorized, single pass)
        A_exp   = A.unsqueeze(0).unsqueeze(0)   # (1,1,d_inner,d_state)
        dt_exp  = dt.unsqueeze(-1)              # (B,L,d_inner,1)
        dA_all  = torch.exp(A_exp * dt_exp)     # (B,L,d_inner,d_state)

        if time_delta is not None:
            td = time_delta.unsqueeze(-1).unsqueeze(-1)
            dr = F.softplus(self.time_decay_rate).unsqueeze(0).unsqueeze(0)
            dA_all = dA_all * torch.exp(-dr * td.abs())

        dB_all  = B.unsqueeze(2) * dt_exp       # (B,L,d_inner,d_state)
        dBx_all = dB_all * x.unsqueeze(-1)      # (B,L,d_inner,d_state)

        # Sequential scan with pre-allocated output buffer (avoids list+stack)
        y = torch.empty(batch, seq_len, d_inner, device=x.device, dtype=x.dtype)
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        D_term = D.unsqueeze(0).unsqueeze(0)  # (1,1,d_inner)

        for t in range(seq_len):
            h = dA_all[:, t] * h + dBx_all[:, t]          # (B, d_inner, d_state)
            y[:, t] = (h * C[:, t].unsqueeze(1)).sum(-1)   # (B, d_inner)

        return y + x * D_term


class MambaBlockV2(nn.Module):
    def __init__(self, d_model, d_state=32, dt_rank=6, d_conv=4, dropout=0.1):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.ssm     = SelectiveSSMv2(d_model, d_state, dt_rank, d_conv)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_delta=None):
        residual = x
        x = self.norm(x)
        x = self.ssm(x, time_delta=time_delta)
        x = self.dropout(x)
        return x + residual


class CNNMambaV2(nn.Module):
    """
    CNN-Mamba v2 — architecture matching fold_10_best.pt.
    Feature MLP + Multi-Scale Temporal CNN + Mamba backbone.
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
        self.d_model  = d_model
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
        """
        Args:
            smart: (B, L, n_smart_features) pre-normalized smart_v3 features
        Returns:
            preds: (B, n_targets)
            embedding: (B, d_model) [if return_embedding=True]
        """
        B, L, _ = smart.shape
        time_delta = smart[:, :, 0]  # feature 0 = time_delta_log

        # Pathway 1: pointwise MLP at each timestep
        feat_out = self.feature_mlp(smart)  # (B, L, feature_mlp_out)

        # Pathway 2: multi-scale temporal CNN
        x_t = smart.transpose(1, 2)  # (B, F, L)
        cnn_outputs = []
        for k, conv_block in zip(self.cnn_kernels, self.temporal_cnns):
            padded = F.pad(x_t, (k - 1, 0))
            cnn_outputs.append(conv_block(padded))  # (B, cnn_ch, L)
        cnn_cat = torch.cat(cnn_outputs, dim=1).transpose(1, 2)  # (B, L, cnn_out_dim)

        # Fuse and project
        fused = torch.cat([feat_out, cnn_cat], dim=-1)
        x = self.fusion_proj(fused)  # (B, L, d_model)

        # Mamba blocks
        for block in self.blocks:
            x = block(x, time_delta=time_delta)

        x = self.final_norm(x)
        embedding = x[:, -1, :]  # (B, d_model)
        preds = self.head(embedding)

        if return_embedding:
            return preds, embedding
        return preds


def load_cnn_mamba(path: str) -> CNNMambaV2:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    arch  = ckpt.get("arch", {})

    # Infer dt_rank from x_proj weight shape (arch dict stores incorrect value)
    d_state  = state["blocks.0.ssm.A_log"].shape[1]
    x_proj   = state["blocks.0.ssm.x_proj.weight"]
    dt_rank  = x_proj.shape[0] - 2 * d_state

    n_features = state["feature_mlp.0.weight"].shape[1]
    d_model    = arch.get("d_model", 96)
    n_layers   = sum(1 for k in state if k.endswith(".ssm.A_log"))
    d_conv     = arch.get("d_conv", 4)
    dropout    = arch.get("dropout", 0.1)

    # Infer feature_mlp_hidden, feature_mlp_out, cnn_channels_per_scale
    mlp_hidden = state["feature_mlp.0.weight"].shape[0]  # 128
    mlp_out    = state["feature_mlp.3.weight"].shape[0]  # 64
    # cnn_channels_per_scale: each temporal_cnn has out channels
    cnn_ch     = state["temporal_cnns.0.0.weight"].shape[0]  # 16

    log.info(f"CNN-Mamba arch: n_features={n_features}, d_model={d_model}, "
             f"d_state={d_state}, dt_rank={dt_rank}, n_layers={n_layers}, "
             f"mlp_hidden={mlp_hidden}, mlp_out={mlp_out}, cnn_ch={cnn_ch}, "
             f"fold={ckpt.get('fold')}, val_ic_10s={ckpt.get('val_ic_10s', 0):.4f}")

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
        log.info(f"CNN-Mamba missing keys (expected for older ckpt): {missing}")
    if unexpected:
        log.warning(f"CNN-Mamba unexpected keys: {unexpected}")

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"CNN-Mamba loaded: {n_params:,} params")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# PatchTST Model
# ─────────────────────────────────────────────────────────────────────────────

class ALiBiAttention(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, dropout=0.1):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = head_dim
        self.scale    = head_dim ** -0.5
        self.qkv      = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)
        self.register_buffer("alibi_slopes", self._get_alibi_slopes(n_heads))

    @staticmethod
    def _get_alibi_slopes(n_heads):
        ratio  = 2 ** (-8.0 / n_heads)
        return torch.tensor([ratio ** (i + 1) for i in range(n_heads)], dtype=torch.float32)

    def _alibi_bias(self, L, device):
        pos  = torch.arange(L, dtype=torch.float32, device=device)
        dist = pos.unsqueeze(0) - pos.unsqueeze(1)
        return dist.unsqueeze(0) * self.alibi_slopes.view(-1, 1, 1)

    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self._alibi_bias(L, x.device)
        attn = self.drop(torch.softmax(attn, dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, L, self.n_heads * self.head_dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = ALiBiAttention(d_model, n_heads, head_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PatchTST(nn.Module):
    def __init__(self, n_features=25, patch_size=25, d_model=256, n_heads=4,
                 head_dim=64, n_layers=4, ffn_dim=1024, dropout=0.1,
                 n_targets=3, window_size=500):
        super().__init__()
        self.d_model   = d_model
        self.patch_size = patch_size
        self.n_patches  = window_size // patch_size

        patch_dim = patch_size * n_features
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, head_dim, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_targets),
        )

    def forward(self, x, return_embedding=False):
        B, W, F = x.shape
        x = x.reshape(B, self.n_patches, self.patch_size * F)
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        embedding = self.norm(x).mean(dim=1)
        preds = self.head(embedding)
        if return_embedding:
            return preds, embedding
        return preds


def load_patchtst(path: str) -> PatchTST:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    arch  = ckpt.get("arch", {})

    n_features  = arch.get("n_features", 25)
    patch_size  = arch.get("patch_size", 25)
    d_model     = arch.get("d_model", 256)
    n_heads     = arch.get("n_heads", 4)
    head_dim    = arch.get("head_dim", 64)
    n_layers    = arch.get("n_layers", 4)
    ffn_dim     = arch.get("ffn_dim", 1024)
    dropout     = arch.get("dropout", 0.1)
    window_size = arch.get("window_size", 500)

    log.info(f"PatchTST arch: n_features={n_features}, patch_size={patch_size}, "
             f"d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, "
             f"window_size={window_size}, fold={ckpt.get('fold')}, "
             f"val_ic_10s={ckpt.get('val_ic_10s', 0):.4f}")

    model = PatchTST(
        n_features=n_features, patch_size=patch_size, d_model=d_model,
        n_heads=n_heads, head_dim=head_dim, n_layers=n_layers,
        ffn_dim=ffn_dim, dropout=dropout, n_targets=3, window_size=window_size,
    )
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        log.warning(f"PatchTST missing keys: {missing}")
    if unexpected:
        log.warning(f"PatchTST unexpected keys: {unexpected}")

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"PatchTST loaded: {n_params:,} params")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Prediction discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_covered_dates(out_dir: Path) -> set:
    """Return set of YYYYMMDD dates already covered by fold_*_oot_predictions.npz."""
    covered = set()
    for npz in sorted(out_dir.glob("fold_*_oot_predictions.npz")):
        try:
            d = np.load(npz, allow_pickle=True)
            if "oot_files" in d:
                oot  = d["oot_files"].tolist()
                fn   = str(oot[0]).replace("\\", "/").split("/")[-1] if oot else ""
                date = fn.split("_")[0]
                if len(date) == 8 and date.isdigit():
                    covered.add(date)
        except Exception:
            pass
    return covered


def next_fold_idx(out_dir: Path) -> int:
    """Return next available fold index."""
    existing = sorted(out_dir.glob("fold_*_oot_predictions.npz"))
    if not existing:
        return 0
    last = existing[-1].name
    try:
        return int(last.split("_")[1]) + 1
    except Exception:
        return len(existing)


# ─────────────────────────────────────────────────────────────────────────────
# Inference kernel
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference_on_day(
    model: nn.Module,
    events: np.ndarray,
    labels_1s: np.ndarray,
    labels_5s: np.ndarray,
    labels_10s: np.ndarray,
    window_size: int,
    stride: int,
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window inference over one day.

    Returns:
        predictions : (N, 3) float32
        labels      : (N, 3) float32
        embeddings  : (N, d_model) float32
    """
    n_events = len(events)
    valid_starts = [
        s for s in range(0, n_events - window_size + 1, stride)
        if (not np.isnan(labels_1s[s + window_size - 1])
            and not np.isnan(labels_5s[s + window_size - 1])
            and not np.isnan(labels_10s[s + window_size - 1]))
    ]

    N = len(valid_starts)
    d_model = model.d_model
    if N == 0:
        return (np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.float32),
                np.zeros((0, d_model), np.float32))

    all_preds  = []
    all_embeds = []
    all_labels = []

    events_t = torch.from_numpy(events)

    for i in range(0, N, batch_size):
        batch_starts = valid_starts[i : i + batch_size]
        batch = torch.stack([events_t[s : s + window_size] for s in batch_starts])  # (B, W, F)

        preds, embeds = model(batch, return_embedding=True)
        all_preds.append(preds.numpy())
        all_embeds.append(embeds.numpy())

        all_labels.append(np.array([
            [labels_1s[s + window_size - 1],
             labels_5s[s + window_size - 1],
             labels_10s[s + window_size - 1]]
            for s in batch_starts
        ], dtype=np.float32))

        if (i // batch_size) % 100 == 0 and i > 0:
            pct = i / N * 100
            log.info(f"    {i:,}/{N:,} ({pct:.0f}%)")

    return (np.concatenate(all_preds, 0),
            np.concatenate(all_labels, 0),
            np.concatenate(all_embeds, 0))


def compute_ic(predictions: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    ics = []
    for col in range(3):
        valid = ~(np.isnan(predictions[:, col]) | np.isnan(labels[:, col]))
        if valid.sum() < 10:
            ics.append(float("nan"))
        else:
            ic, _ = scipy.stats.spearmanr(predictions[valid, col], labels[valid, col])
            ics.append(float(ic))
    return tuple(ics)


def save_predictions(
    out_dir: Path,
    fold_idx: int,
    mbo_path: Path,
    predictions: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray,
    ic_1s: float,
    ic_5s: float,
    ic_10s: float,
):
    out_path = out_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
    np.savez_compressed(
        out_path,
        predictions = predictions.astype(np.float32),
        labels      = labels.astype(np.float32),
        horizons    = np.array(HORIZONS),
        ic_1s       = np.float64(ic_1s),
        ic_5s       = np.float64(ic_5s),
        ic_10s      = np.float64(ic_10s),
        oot_files   = np.array([str(mbo_path)]),
        embeddings  = embeddings.astype(np.float32),
    )
    log.info(f"  Saved → {out_path.name}  "
             f"(N={len(predictions):,}, IC_1s={ic_1s:.4f}, IC_5s={ic_5s:.4f}, "
             f"IC_10s={ic_10s:.4f})")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Worker function for parallel processing
# ─────────────────────────────────────────────────────────────────────────────

def _worker_process_date(args_tuple):
    """
    Worker function for multiprocessing.Pool.
    Each worker loads its own model copy (no shared state between processes).
    """
    (date, mbo_path_str, cnn_weights_str, ptst_weights_str,
     do_cnn, do_ptst, cnn_out_dir_str, ptst_out_dir_str,
     cnn_fold_idx, ptst_fold_idx, batch_size) = args_tuple

    # Suppress logging from child processes (output goes to parent)
    import logging
    logger = logging.getLogger(f"worker.{date}")
    logger.setLevel(logging.WARNING)

    mbo_path     = Path(mbo_path_str)
    cnn_out_dir  = Path(cnn_out_dir_str)
    ptst_out_dir = Path(ptst_out_dir_str)

    t0 = time.time()
    results = {"date": date, "cnn_ok": False, "ptst_ok": False, "error": None}

    try:
        data      = np.load(mbo_path, allow_pickle=True)
        events    = data["events"].astype(np.float32)
        labels_1s = data["labels_1s"].astype(np.float32)
        labels_5s = data["labels_5s"].astype(np.float32)
        labels_10s = data["labels_10s"].astype(np.float32)

        if do_cnn and cnn_weights_str:
            cnn_model = load_cnn_mamba(cnn_weights_str)
            cnn_model.eval()
            with torch.no_grad():
                preds, lbls, embeds = run_inference_on_day(
                    cnn_model, events, labels_1s, labels_5s, labels_10s,
                    CNN_WINDOW, CNN_STRIDE, batch_size,
                )
            ic_1s, ic_5s, ic_10s = compute_ic(preds, lbls)
            save_predictions(cnn_out_dir, cnn_fold_idx, mbo_path,
                             preds, lbls, embeds, ic_1s, ic_5s, ic_10s)
            results["cnn_ok"] = True
            results["cnn_n"]  = len(preds)
            results["cnn_ic"] = (ic_1s, ic_5s, ic_10s)

        if do_ptst and ptst_weights_str:
            ptst_model = load_patchtst(ptst_weights_str)
            ptst_model.eval()
            with torch.no_grad():
                preds, lbls, embeds = run_inference_on_day(
                    ptst_model, events, labels_1s, labels_5s, labels_10s,
                    PTST_WINDOW, PTST_STRIDE, batch_size,
                )
            ic_1s, ic_5s, ic_10s = compute_ic(preds, lbls)
            save_predictions(ptst_out_dir, ptst_fold_idx, mbo_path,
                             preds, lbls, embeds, ic_1s, ic_5s, ic_10s)
            results["ptst_ok"] = True
            results["ptst_n"]  = len(preds)
            results["ptst_ic"] = (ic_1s, ic_5s, ic_10s)

    except Exception as e:
        results["error"] = str(e)

    results["elapsed"] = time.time() - t0
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["cnn", "ptst", "both"], default="both")
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers (default 8; each loads model independently)")
    args = parser.parse_args()

    target_start = "20260316"
    target_end   = "20260429"

    all_mbo = sorted(MBO_DIR.glob("*_mbo_events.npz"))
    target_files = [f for f in all_mbo if target_start <= f.name.split("_")[0] <= target_end]
    target_dates = [f.name.split("_")[0] for f in target_files]

    log.info(f"Target range: {target_start} – {target_end}  ({len(target_dates)} dates)")
    if not target_files:
        log.error("No MBO files found in target range.")
        sys.exit(1)

    run_cnn  = args.model in ("cnn", "both")
    run_ptst = args.model in ("ptst", "both")

    if run_cnn and not CNN_WEIGHTS.exists():
        log.error(f"CNN weights not found: {CNN_WEIGHTS}")
        run_cnn = False
    if run_ptst and not PTST_WEIGHTS.exists():
        log.error(f"PatchTST weights not found: {PTST_WEIGHTS}")
        run_ptst = False

    if not run_cnn and not run_ptst:
        log.error("No models available.")
        sys.exit(1)

    # ── Find uncovered dates ──────────────────────────────────────────────────
    CNN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    PTST_OUT_DIR.mkdir(parents=True, exist_ok=True)

    cnn_covered  = get_covered_dates(CNN_OUT_DIR)  if run_cnn  else set()
    ptst_covered = get_covered_dates(PTST_OUT_DIR) if run_ptst else set()
    cnn_needed   = sorted(d for d in target_dates if d not in cnn_covered)  if run_cnn  else []
    ptst_needed  = sorted(d for d in target_dates if d not in ptst_covered) if run_ptst else []

    log.info(f"CNN-Mamba  covered: {len(cnn_covered)} | to generate: {len(cnn_needed)}")
    log.info(f"PatchTST   covered: {len(ptst_covered)} | to generate: {len(ptst_needed)}")

    all_needed = sorted(set(cnn_needed) | set(ptst_needed))
    if not all_needed:
        log.info("All target dates already have predictions. Nothing to do.")
        return

    cnn_fold_base  = next_fold_idx(CNN_OUT_DIR)
    ptst_fold_base = next_fold_idx(PTST_OUT_DIR)
    log.info(f"Next CNN fold idx: {cnn_fold_base}  |  Next PTST fold idx: {ptst_fold_base}")

    # Build work list with pre-assigned fold indices (deterministic ordering)
    work = []
    cnn_offset = ptst_offset = 0
    for date in all_needed:
        do_cnn  = run_cnn  and date in cnn_needed
        do_ptst = run_ptst and date in ptst_needed
        cnn_fi  = cnn_fold_base  + cnn_offset  if do_cnn  else -1
        ptst_fi = ptst_fold_base + ptst_offset if do_ptst else -1

        work.append((
            date,
            str(MBO_DIR / f"{date}_mbo_events.npz"),
            str(CNN_WEIGHTS)  if do_cnn  else "",
            str(PTST_WEIGHTS) if do_ptst else "",
            do_cnn, do_ptst,
            str(CNN_OUT_DIR), str(PTST_OUT_DIR),
            cnn_fi, ptst_fi,
            args.batch,
        ))
        if do_cnn:  cnn_offset  += 1
        if do_ptst: ptst_offset += 1

    log.info(f"\nLaunching {len(work)} date jobs with {args.workers} parallel workers...")
    log.info(f"  CNN-Mamba dates: {len(cnn_needed)}")
    log.info(f"  PatchTST  dates: {len(ptst_needed)}")

    t_total = time.time()
    cnn_written = ptst_written = 0
    errors = []

    if args.workers > 1:
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(_worker_process_date, work):
                date = result["date"]
                elapsed = result.get("elapsed", 0)
                if result.get("error"):
                    log.error(f"  {date}: FAILED — {result['error']}")
                    errors.append(date)
                    continue

                cnn_info = (f"CNN IC_10s={result['cnn_ic'][2]:.4f} N={result['cnn_n']:,}"
                            if result.get("cnn_ok") else "CNN skip")
                ptst_info = (f"PTST IC_10s={result['ptst_ic'][2]:.4f} N={result['ptst_n']:,}"
                             if result.get("ptst_ok") else "PTST skip")
                log.info(f"  {date}: done in {elapsed:.0f}s  |  {cnn_info}  |  {ptst_info}")
                if result.get("cnn_ok"):  cnn_written  += 1
                if result.get("ptst_ok"): ptst_written += 1
    else:
        for w in work:
            result = _worker_process_date(w)
            date = result["date"]
            elapsed = result.get("elapsed", 0)
            if result.get("error"):
                log.error(f"  {date}: FAILED — {result['error']}")
                errors.append(date)
            else:
                cnn_info = (f"CNN IC_10s={result['cnn_ic'][2]:.4f} N={result['cnn_n']:,}"
                            if result.get("cnn_ok") else "CNN skip")
                ptst_info = (f"PTST IC_10s={result['ptst_ic'][2]:.4f} N={result['ptst_n']:,}"
                             if result.get("ptst_ok") else "PTST skip")
                log.info(f"  {date}: done in {elapsed:.0f}s  |  {cnn_info}  |  {ptst_info}")
                if result.get("cnn_ok"):  cnn_written  += 1
                if result.get("ptst_ok"): ptst_written += 1

    log.info(f"\n{'='*60}")
    log.info(f"Batch inference complete.  Total: {(time.time()-t_total)/60:.1f}min")
    log.info(f"  CNN-Mamba files written: {cnn_written}")
    log.info(f"  PatchTST  files written: {ptst_written}")
    if errors:
        log.warning(f"  Failed dates: {errors}")


if __name__ == "__main__":
    main()
