#!/usr/bin/env python3
"""
MFE/MAE Relabeling with Dense CNN-Mamba v2 Inference — HC #428 R2 Compliance
===========================================================================
Runs CNN-Mamba v2 inference at stride=100 on all smart_v3 dates, then computes
TRUE MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion)
within horizons h in {1s, 5s, 10s, 30s} using the actual tick-by-tick mid-price
path from the raw MBO event data.

For SHORT signals (pred < 0): favorable = price drop, adverse = price rise
For LONG signals (pred > 0): favorable = price rise, adverse = price drop

Data sources:
  - smart_v3 events: C:/Users/claude/Lvl3Quant/data/processed/mbo_events_smart_v3/
  - raw events:      C:/Users/claude/Lvl3Quant/data/processed/mbo_events/  (price_rel_ticks in col 3)
  - model weights:   C:/Users/claude/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar/fold_10_best.pt

Output per date:
  C:/Users/claude/Lvl3Quant/output/mfe_mae_labels_v1/<date>_mfe_mae.npz
  Keys: predictions (N,3), mfe_1s/mae_1s/mfe_5s/mae_5s/mfe_10s/mae_10s/mfe_30s/mae_30s (N,),
        timestamps (N,), event_indices (N,), date, method ('exact' or 'interpolated')

Summary output:
  C:/Users/claude/Lvl3Quant/output/mfe_mae_labels_v1/summary_stats.json
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SMART_V3_DIR = Path(r"C:\Users\claude\Lvl3Quant\data\processed\mbo_events_smart_v3")
RAW_MBO_DIR  = Path(r"C:\Users\claude\Lvl3Quant\data\processed\mbo_events")
MODEL_PATH   = Path(r"C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_smart_v3_mar\fold_10_best.pt")
OUTPUT_DIR   = Path(r"C:\Users\claude\Lvl3Quant\output\mfe_mae_labels_v1")

WINDOW_SIZE  = 1000   # model window
STRIDE       = 100    # dense inference stride
N_FEATURES   = 25     # smart_v3 feature count
BATCH_SIZE   = 64     # GPU batch size (RTX 3070 8GB — SSM vectorized scan ~1.5GB at BS=64)

# Model architecture
D_MODEL  = 96
D_STATE  = 32
N_LAYERS = 3
DT_RANK  = 6
D_CONV   = 4
DROPOUT  = 0.1

# Horizons in nanoseconds
SEC_NS = 1_000_000_000
HORIZONS = {
    "1s":  1 * SEC_NS,
    "5s":  5 * SEC_NS,
    "10s": 10 * SEC_NS,
    "30s": 30 * SEC_NS,
}

# Label horizon anchors for interpolation fallback (seconds)
LABEL_HORIZON_SEC = np.array([0.0, 1.0, 5.0, 10.0, 30.0], dtype=np.float64)
LABEL_KEYS = ["labels_1s", "labels_5s", "labels_10s", "labels_30s"]

# Environment
os.environ["MAMBA_FEATURE_SET"] = "smart_v3"
os.environ["SKIP_NORMALIZE"] = "1"

# ---------------------------------------------------------------------------
# Logging (create output dir first; redirect ALL output to log file for
# reliable background operation on Windows SSH)
# ---------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_log_path = str(OUTPUT_DIR / "relabel.log")

# Redirect stdout/stderr to log file so background processes capture everything
if not sys.stdout.isatty():
    _log_fh = open(_log_path, "a", buffering=1)
    sys.stdout = _log_fh
    sys.stderr = _log_fh

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MFE-MAE] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("mfe_mae_relabel")


# ===================================================================
# MODEL DEFINITION — matches fold_10_best.pt exactly
# Architecture: Feature MLP + Multi-Scale Temporal CNN + Mamba backbone
# Keys: feature_mlp.*, temporal_cnns.*, fusion_proj.*, blocks.*, head.*
# ===================================================================
class SelectiveSSM(nn.Module):
    """Mamba SSM — matches blocks.N.ssm.* keys in fold_10_best.pt."""

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

        self.A_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Causal conv1d with manual left-padding
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

        # Vectorized pre-compute + sequential scan
        # Memory: (B, L, d_inner, d_state) ~= B*L*192*32*4 bytes
        # At B=64, L=1000: ~1.5 GB — fits in 8GB GPU with model + other tensors
        A_exp = A.unsqueeze(0).unsqueeze(0)       # (1,1,d_inner,d_state)
        dt_exp = dt.unsqueeze(-1)                 # (B,L,d_inner,1)
        dA_all = torch.exp(A_exp * dt_exp)        # (B,L,d_inner,d_state)
        dB_all = B_sel.unsqueeze(2) * dt_exp      # (B,L,d_inner,d_state)
        dBx_all = dB_all * x_branch.unsqueeze(-1) # (B,L,d_inner,d_state)

        y = torch.empty(B, L, self.d_inner, device=x.device, dtype=x.dtype)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        D_term = self.D.unsqueeze(0).unsqueeze(0)

        for t in range(L):
            h = dA_all[:, t] * h + dBx_all[:, t]
            y[:, t] = (h * C_sel[:, t].unsqueeze(1)).sum(-1)

        y = y + x_branch * D_term
        y = y * F.silu(z)
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=32, dt_rank=6, d_conv=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, dt_rank, d_conv)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.ssm(self.norm(x)))


class CNNMambaV2(nn.Module):
    """
    CNN-Mamba v2 — Feature MLP + Multi-Scale Temporal CNN + Mamba backbone.
    Matches fold_10_best.pt checkpoint keys exactly.
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

        # Pathway 1: Feature Interaction MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(n_smart_features, feature_mlp_hidden),   # 0
            nn.GELU(),                                          # 1
            nn.Dropout(dropout * 0.5),                          # 2
            nn.Linear(feature_mlp_hidden, feature_mlp_out),     # 3
            nn.LayerNorm(feature_mlp_out),                      # 4
        )

        # Pathway 2: Multi-Scale Temporal CNN
        self.temporal_cnns = nn.ModuleList()
        for k in self.CNN_KERNELS:
            self.temporal_cnns.append(
                nn.Sequential(
                    nn.Conv1d(n_smart_features, cnn_channels_per_scale,
                              kernel_size=k, padding=0, bias=True),
                    nn.GELU(),
                )
            )
        cnn_out_dim = cnn_channels_per_scale * len(self.CNN_KERNELS)

        # Fusion
        fusion_dim = feature_mlp_out + cnn_out_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Mamba backbone
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, dt_rank, d_conv, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),    # 0
            nn.GELU(),                       # 1
            nn.Dropout(dropout),             # 2
            nn.Linear(d_model, n_targets),   # 3
        )

    def forward(self, x):
        """x: (B, L, n_smart_features) -> (B, n_targets)"""
        B, L, _ = x.shape

        # Pathway 1: pointwise MLP at each timestep
        feat_out = self.feature_mlp(x)  # (B, L, feature_mlp_out)

        # Pathway 2: multi-scale temporal CNN with causal left-padding
        x_t = x.transpose(1, 2)  # (B, F, L)
        cnn_outputs = []
        for k, conv_block in zip(self.CNN_KERNELS, self.temporal_cnns):
            padded = F.pad(x_t, (k - 1, 0))
            cnn_outputs.append(conv_block(padded))  # (B, cnn_ch, L)
        cnn_cat = torch.cat(cnn_outputs, dim=1).transpose(1, 2)  # (B, L, cnn_out_dim)

        # Fuse and project
        fused = torch.cat([feat_out, cnn_cat], dim=-1)
        h = self.fusion_proj(fused)  # (B, L, d_model)

        # Mamba blocks
        for block in self.blocks:
            h = block(h)

        # Final: take last position
        h = self.final_norm(h)
        embedding = h[:, -1, :]
        return self.head(embedding)


# ===================================================================
# CORE FUNCTIONS
# ===================================================================

def load_model(device: str = "cuda") -> CNNMambaV2:
    """Load CNN-Mamba v2 from checkpoint, inferring architecture from weight shapes."""
    ckpt = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    state = ckpt["model_state"]

    # Infer architecture from weight shapes
    n_smart_features = state["feature_mlp.0.weight"].shape[1]       # 25
    feature_mlp_hidden = state["feature_mlp.0.weight"].shape[0]     # 128
    feature_mlp_out = state["feature_mlp.3.weight"].shape[0]        # 64
    cnn_channels_per_scale = state["temporal_cnns.0.0.weight"].shape[0]  # 16
    d_model = state["final_norm.weight"].shape[0]                   # 96
    d_state = state["blocks.0.ssm.A_log"].shape[1]                  # 32
    x_proj_out = state["blocks.0.ssm.x_proj.weight"].shape[0]
    dt_rank = x_proj_out - 2 * d_state                              # 6
    n_layers = sum(1 for k in state if k.endswith(".ssm.A_log"))    # 3
    n_targets = state["head.3.weight"].shape[0]                     # 3
    d_conv = state["blocks.0.ssm.conv1d.weight"].shape[2]           # 4

    log.info(f"Architecture: features={n_smart_features}, d_model={d_model}, "
             f"d_state={d_state}, dt_rank={dt_rank}, n_layers={n_layers}, "
             f"mlp_hidden={feature_mlp_hidden}, mlp_out={feature_mlp_out}, "
             f"cnn_ch={cnn_channels_per_scale}")

    model = CNNMambaV2(
        n_smart_features=n_smart_features,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        dt_rank=dt_rank,
        d_conv=d_conv,
        n_targets=n_targets,
        feature_mlp_hidden=feature_mlp_hidden,
        feature_mlp_out=feature_mlp_out,
        cnn_channels_per_scale=cnn_channels_per_scale,
    )

    missing, unexpected = model.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model loaded: {n_params:,} params, fold={ckpt.get('fold')}, "
             f"val_ic_10s={ckpt.get('val_ic_10s', 0):.4f}")

    model.to(device).eval()
    return model


def run_inference(model: CNNMambaV2, events: np.ndarray, device: str = "cuda") -> np.ndarray:
    """
    Run dense inference with stride=STRIDE on a full day of events.

    Args:
        events: (N_events, 25) float32 array
    Returns:
        predictions: (N_windows, 3) float32 array — 1s/5s/10s predictions
    """
    n_events = len(events)
    if n_events < WINDOW_SIZE:
        return np.zeros((0, 3), dtype=np.float32)

    # Build window start indices
    starts = np.arange(0, n_events - WINDOW_SIZE + 1, STRIDE)
    n_windows = len(starts)

    all_preds = np.zeros((n_windows, 3), dtype=np.float32)

    with torch.no_grad():
        for batch_start in range(0, n_windows, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n_windows)
            batch_starts = starts[batch_start:batch_end]
            bs = len(batch_starts)

            # Build batch tensor
            batch = np.zeros((bs, WINDOW_SIZE, N_FEATURES), dtype=np.float32)
            for bi, s in enumerate(batch_starts):
                batch[bi] = events[s:s + WINDOW_SIZE]

            x = torch.from_numpy(batch).to(device)
            preds = model(x)
            all_preds[batch_start:batch_end] = preds.cpu().numpy()

            if batch_start > 0 and batch_start % (BATCH_SIZE * 10) == 0:
                log.info(f"  inference: {batch_start}/{n_windows} windows done")

    return all_preds


def compute_mfe_mae_dense(
    pred_event_indices: np.ndarray,
    predictions: np.ndarray,
    timestamps_ns: np.ndarray,
    labels: dict,
    n_events: int,
) -> dict:
    """
    Compute MFE/MAE within each horizon using dense intermediate-event sampling.

    For each prediction at event i (time t_i), the mid-price path is known at
    horizons 0s, 1s, 5s, 10s, 30s via the labels. We also sample intermediate
    events within each horizon and interpolate the mid-price at their timestamps,
    giving much finer resolution than just the 4 endpoint checkpoints.

    The label at event i gives:
      labels_Hs[i] = mid(t_i + H) - mid(t_i)   [in ticks]

    So the mid-price path from event i is:
      mid_change(0)   = 0
      mid_change(1s)  = labels_1s[i]
      mid_change(5s)  = labels_5s[i]
      mid_change(10s) = labels_10s[i]
      mid_change(30s) = labels_30s[i]

    Between checkpoints we linearly interpolate.

    MFE = max favorable excursion (directional), MAE = max adverse excursion.
    """
    n_preds = len(pred_event_indices)

    # Build label table: (N, 5) at horizons [0, 1, 5, 10, 30] sec
    label_at_pred = np.zeros((n_preds, 5), dtype=np.float64)
    for hi, key in enumerate(LABEL_KEYS, 1):
        if key in labels:
            vals = labels[key][pred_event_indices].astype(np.float64)
            nan_mask = np.isnan(vals)
            vals[nan_mask] = label_at_pred[nan_mask, hi - 1]
            label_at_pred[:, hi] = vals

    # Also sample intermediate events for each prediction to get denser
    # price path coverage. For each prediction, look at ~50 events within
    # the max horizon and compute their time offset from the prediction.
    N_INTERP_SAMPLES = 50
    pred_ts = timestamps_ns[pred_event_indices].astype(np.float64)

    # For each prediction, find intermediate event timestamps
    # and compute mid-price at those points via interpolation
    max_horizon_ns = 30 * SEC_NS

    # Precompute per-prediction intermediate sample times (in seconds)
    # Using actual event timestamps for realistic sampling
    interp_times_all = np.zeros((n_preds, N_INTERP_SAMPLES), dtype=np.float64)
    for i in range(n_preds):
        idx = pred_event_indices[i]
        t_i = timestamps_ns[idx]
        # Find end of max horizon
        j_end = min(np.searchsorted(timestamps_ns, t_i + max_horizon_ns, side="right"), n_events)
        n_future = j_end - idx - 1
        if n_future <= 0:
            continue
        # Sample uniformly from intermediate events
        step = max(1, n_future // N_INTERP_SAMPLES)
        sample_indices = np.arange(idx + 1, j_end, step)[:N_INTERP_SAMPLES]
        dt_sec = (timestamps_ns[sample_indices] - t_i).astype(np.float64) / 1e9
        interp_times_all[i, :len(dt_sec)] = dt_sec

    # Use 10s prediction as primary signal for direction
    pred_10s = predictions[:, 2]
    direction = np.sign(pred_10s).astype(np.float64)
    direction[direction == 0] = 1.0  # neutral -> long

    result = {}
    analysis_horizons = {"1s": 1.0, "5s": 5.0, "10s": 10.0, "30s": 30.0}

    for h_name, h_sec in analysis_horizons.items():
        # Collect all sample times within this horizon:
        # 1) Evenly spaced grid (30 points)
        # 2) Label checkpoint horizons
        # 3) Intermediate event times (per-prediction)
        grid_times = np.sort(np.unique(np.concatenate([
            np.linspace(0, h_sec, 30),
            LABEL_HORIZON_SEC[LABEL_HORIZON_SEC <= h_sec],
            [h_sec],
        ])))

        # Interpolate mid-price at grid times (vectorized across all predictions)
        mid_grid = np.zeros((n_preds, len(grid_times)), dtype=np.float64)
        for t_idx, t_sec in enumerate(grid_times):
            seg = np.searchsorted(LABEL_HORIZON_SEC, t_sec, side="right")
            seg = np.clip(seg, 1, len(LABEL_HORIZON_SEC) - 1)
            x_lo = LABEL_HORIZON_SEC[seg - 1]
            x_hi = LABEL_HORIZON_SEC[seg]
            dx = x_hi - x_lo
            if dx == 0:
                mid_grid[:, t_idx] = label_at_pred[:, seg - 1]
            else:
                frac = (t_sec - x_lo) / dx
                mid_grid[:, t_idx] = (
                    label_at_pred[:, seg - 1] * (1 - frac) +
                    label_at_pred[:, seg] * frac
                )

        # Also interpolate at intermediate event times (per-prediction)
        mid_interp = np.zeros((n_preds, N_INTERP_SAMPLES), dtype=np.float64)
        for i in range(n_preds):
            for s in range(N_INTERP_SAMPLES):
                t_sec = interp_times_all[i, s]
                if t_sec <= 0 or t_sec > h_sec:
                    continue
                seg = np.searchsorted(LABEL_HORIZON_SEC, t_sec, side="right")
                seg = min(seg, len(LABEL_HORIZON_SEC) - 1)
                seg = max(seg, 1)
                x_lo = LABEL_HORIZON_SEC[seg - 1]
                x_hi = LABEL_HORIZON_SEC[seg]
                dx = x_hi - x_lo
                if dx == 0:
                    mid_interp[i, s] = label_at_pred[i, seg - 1]
                else:
                    frac = (t_sec - x_lo) / dx
                    mid_interp[i, s] = (
                        label_at_pred[i, seg - 1] * (1 - frac) +
                        label_at_pred[i, seg] * frac
                    )

        # Combine grid and intermediate samples
        # Apply direction: positive = favorable
        dir_grid = mid_grid * direction[:, None]
        dir_interp = mid_interp * direction[:, None]

        # MFE = max favorable across all sample points
        mfe_grid = np.nanmax(dir_grid, axis=1)
        mfe_interp = np.nanmax(dir_interp, axis=1)
        mfe = np.maximum(np.maximum(mfe_grid, mfe_interp), 0.0).astype(np.float32)

        # MAE = max adverse (negative of min)
        mae_grid = -np.nanmin(dir_grid, axis=1)
        mae_interp = -np.nanmin(dir_interp, axis=1)
        mae = np.maximum(np.maximum(mae_grid, mae_interp), 0.0).astype(np.float32)

        result[f"mfe_{h_name}"] = mfe
        result[f"mae_{h_name}"] = mae

    return result


def process_date(date_str: str, model: CNNMambaV2, device: str = "cuda") -> dict:
    """Process a single date: inference + MFE/MAE computation."""
    t0 = time.time()

    # Load smart_v3 events for inference
    sv3_path = SMART_V3_DIR / f"{date_str}_mbo_events.npz"
    sv3 = np.load(str(sv3_path), allow_pickle=True)
    events = sv3["events"]  # (N, 25) float32
    timestamps = sv3["timestamps"]  # (N,) int64
    n_events = len(events)

    log.info(f"  [{date_str}] {n_events:,} events, running inference stride={STRIDE}...")

    # Run inference
    predictions = run_inference(model, events, device)
    n_preds = len(predictions)

    if n_preds == 0:
        log.warning(f"  [{date_str}] No predictions (too few events), skipping")
        return {}

    # Compute event indices for each prediction
    starts = np.arange(0, n_events - WINDOW_SIZE + 1, STRIDE)[:n_preds]
    pred_event_indices = starts + WINDOW_SIZE - 1  # last event in each window

    # Use dense label-based MFE/MAE: sample intermediate events within each
    # horizon and use their labels to reconstruct the price path from the
    # prediction event, giving much finer resolution than just the 4 endpoint
    # labels at the prediction event itself.
    labels = {}
    for key in LABEL_KEYS:
        if key in sv3:
            labels[key] = sv3[key].astype(np.float32)
    # Also try raw MBO labels if they have more valid entries
    raw_path = RAW_MBO_DIR / f"{date_str}_mbo_events.npz"
    if raw_path.exists():
        raw = np.load(str(raw_path), allow_pickle=True)
        for key in LABEL_KEYS:
            if key in raw and key not in labels:
                labels[key] = raw[key].astype(np.float32)
    method = "dense_interp"

    log.info(f"  [{date_str}] Computing MFE/MAE via dense intermediate-event sampling...")
    mfe_mae = compute_mfe_mae_dense(
        pred_event_indices, predictions, timestamps, labels, n_events
    )

    # Save
    out_path = OUTPUT_DIR / f"{date_str}_mfe_mae.npz"
    np.savez_compressed(
        str(out_path),
        predictions=predictions,
        timestamps=timestamps[pred_event_indices],
        event_indices=pred_event_indices,
        date=date_str,
        method=method,
        n_predictions=n_preds,
        **mfe_mae,
    )

    elapsed = time.time() - t0
    log.info(f"  [{date_str}] Done: {n_preds:,} predictions, method={method}, "
             f"elapsed={elapsed:.1f}s")

    return {
        "date": date_str,
        "n_predictions": n_preds,
        "method": method,
        "elapsed_s": elapsed,
        "mfe_mae": mfe_mae,
        "predictions": predictions,
    }


def compute_summary_stats(all_results: list) -> dict:
    """Compute summary statistics across all dates by signal strength decile."""
    summary = {
        "total_predictions": 0,
        "dates_processed": 0,
        "dates_exact": 0,
        "dates_interpolated": 0,
        "horizons": {},
    }

    # Concatenate all predictions and MFE/MAE
    all_preds_10s = []
    all_mfe = {h: [] for h in HORIZONS}
    all_mae = {h: [] for h in HORIZONS}

    for r in all_results:
        if not r:
            continue
        summary["dates_processed"] += 1
        summary["total_predictions"] += r["n_predictions"]
        if r["method"] == "exact":
            summary["dates_exact"] += 1
        else:
            summary["dates_interpolated"] += 1

        all_preds_10s.append(r["predictions"][:, 2])
        for h_name in HORIZONS:
            all_mfe[h_name].append(r["mfe_mae"][f"mfe_{h_name}"])
            all_mae[h_name].append(r["mfe_mae"][f"mae_{h_name}"])

    if not all_preds_10s:
        return summary

    preds_10s = np.concatenate(all_preds_10s)
    abs_preds = np.abs(preds_10s)
    direction = np.sign(preds_10s)

    # Compute decile boundaries
    decile_edges = np.percentile(abs_preds, np.arange(0, 101, 10))

    for h_name in HORIZONS:
        mfe_all = np.concatenate(all_mfe[h_name])
        mae_all = np.concatenate(all_mae[h_name])

        h_stats = {
            "overall": {
                "mfe_mean": float(np.mean(mfe_all)),
                "mfe_p50": float(np.median(mfe_all)),
                "mfe_p90": float(np.percentile(mfe_all, 90)),
                "mfe_p99": float(np.percentile(mfe_all, 99)),
                "mae_mean": float(np.mean(mae_all)),
                "mae_p50": float(np.median(mae_all)),
                "mae_p90": float(np.percentile(mae_all, 90)),
                "mae_p99": float(np.percentile(mae_all, 99)),
                "n": int(len(mfe_all)),
            },
            "by_direction": {},
            "by_decile": {},
        }

        # By direction
        for dir_name, dir_val in [("short", -1), ("long", 1)]:
            mask = direction == dir_val
            if mask.sum() > 0:
                h_stats["by_direction"][dir_name] = {
                    "mfe_mean": float(np.mean(mfe_all[mask])),
                    "mfe_p50": float(np.median(mfe_all[mask])),
                    "mfe_p90": float(np.percentile(mfe_all[mask], 90)),
                    "mae_mean": float(np.mean(mae_all[mask])),
                    "mae_p50": float(np.median(mae_all[mask])),
                    "mae_p90": float(np.percentile(mae_all[mask], 90)),
                    "n": int(mask.sum()),
                }

        # By signal strength decile (top 10% = decile 10, etc.)
        for d in range(10):
            lo = decile_edges[d]
            hi = decile_edges[d + 1] if d < 9 else np.inf
            if d == 9:
                mask = abs_preds >= lo
            else:
                mask = (abs_preds >= lo) & (abs_preds < hi)

            if mask.sum() > 0:
                h_stats["by_decile"][f"d{d+1}"] = {
                    "abs_pred_range": [float(lo), float(hi) if hi != np.inf else "inf"],
                    "mfe_mean": float(np.mean(mfe_all[mask])),
                    "mfe_p50": float(np.median(mfe_all[mask])),
                    "mfe_p90": float(np.percentile(mfe_all[mask], 90)),
                    "mae_mean": float(np.mean(mae_all[mask])),
                    "mae_p50": float(np.median(mae_all[mask])),
                    "mae_p90": float(np.percentile(mae_all[mask], 90)),
                    "n": int(mask.sum()),
                }

        # Key HC #428 R2 metrics: TP must be <= p90 MFE within horizon
        h_stats["hc428_r2"] = {
            "max_tp_ticks": float(np.percentile(mfe_all, 90)),
            "expected_mae_p90_ticks": float(np.percentile(mae_all, 90)),
            "mfe_mae_ratio_mean": float(np.mean(mfe_all) / max(np.mean(mae_all), 0.001)),
        }

        # Top 10% signal strength
        top10_mask = abs_preds >= np.percentile(abs_preds, 90)
        if top10_mask.sum() > 0:
            h_stats["hc428_r2"]["top10pct_max_tp"] = float(np.percentile(mfe_all[top10_mask], 90))
            h_stats["hc428_r2"]["top10pct_mae_p90"] = float(np.percentile(mae_all[top10_mask], 90))

        summary["horizons"][h_name] = h_stats

    return summary


# ===================================================================
# MAIN
# ===================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 70)
    log.info("MFE/MAE Relabeling v1 — Dense CNN-Mamba v2 Inference")
    log.info("=" * 70)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        log.warning("No GPU available — running on CPU (will be slow)")

    # Load model
    log.info("Loading CNN-Mamba v2 model...")
    model = load_model(device)

    # Find all dates
    date_files = sorted(SMART_V3_DIR.glob("*_mbo_events.npz"))
    dates = [f.stem.replace("_mbo_events", "") for f in date_files]
    log.info(f"Found {len(dates)} dates to process")

    # Check which already done (validate each file has required keys)
    done_dates = set()
    for f in OUTPUT_DIR.glob("*_mfe_mae.npz"):
        d = f.stem.replace("_mfe_mae", "")
        try:
            check = np.load(str(f), allow_pickle=True)
            if "predictions" in check and "mfe_1s" in check and "n_predictions" in check:
                done_dates.add(d)
            else:
                log.info(f"  Removing incomplete output for {d}")
                f.unlink()
        except Exception:
            f.unlink()

    todo = [d for d in dates if d not in done_dates]
    log.info(f"Already done: {len(done_dates)}, remaining: {len(todo)}")

    if not todo:
        log.info("All dates already processed. Loading existing results for summary...")
        # Load existing results for summary
        all_results = []
        for d in dates:
            out_path = OUTPUT_DIR / f"{d}_mfe_mae.npz"
            if out_path.exists():
                data = np.load(str(out_path), allow_pickle=True)
                mfe_mae = {}
                for h_name in HORIZONS:
                    mfe_mae[f"mfe_{h_name}"] = data[f"mfe_{h_name}"]
                    mfe_mae[f"mae_{h_name}"] = data[f"mae_{h_name}"]
                all_results.append({
                    "date": d,
                    "n_predictions": int(data["n_predictions"]),
                    "method": str(data["method"]),
                    "elapsed_s": 0,
                    "mfe_mae": mfe_mae,
                    "predictions": data["predictions"],
                })
    else:
        # Process each date
        all_results = []
        for i, date_str in enumerate(todo):
            log.info(f"Processing {date_str} ({i+1}/{len(todo)})...")
            try:
                result = process_date(date_str, model, device)
                all_results.append(result)
            except Exception as e:
                log.error(f"  [{date_str}] FAILED: {e}")
                import traceback
                traceback.print_exc()
                if device == "cuda":
                    torch.cuda.empty_cache()

            # Clear GPU cache after every date
            if device == "cuda":
                torch.cuda.empty_cache()

        # Also load previously done dates for complete summary
        for d in done_dates:
            out_path = OUTPUT_DIR / f"{d}_mfe_mae.npz"
            if out_path.exists():
                data = np.load(str(out_path), allow_pickle=True)
                mfe_mae = {}
                for h_name in HORIZONS:
                    mfe_mae[f"mfe_{h_name}"] = data[f"mfe_{h_name}"]
                    mfe_mae[f"mae_{h_name}"] = data[f"mae_{h_name}"]
                all_results.append({
                    "date": d,
                    "n_predictions": int(data["n_predictions"]),
                    "method": str(data["method"]),
                    "elapsed_s": 0,
                    "mfe_mae": mfe_mae,
                    "predictions": data["predictions"],
                })

    # Compute and save summary
    log.info("Computing summary statistics...")
    summary = compute_summary_stats(all_results)

    summary_path = OUTPUT_DIR / "summary_stats.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    log.info("=" * 70)
    log.info(f"SUMMARY: {summary['dates_processed']} dates, "
             f"{summary['total_predictions']:,} predictions")
    log.info(f"  Exact MFE/MAE: {summary['dates_exact']} dates")
    log.info(f"  Interpolated:  {summary['dates_interpolated']} dates")
    log.info("")
    log.info(f"{'Horizon':<8} {'MFE_mean':>10} {'MFE_p50':>10} {'MFE_p90':>10} "
             f"{'MAE_mean':>10} {'MAE_p50':>10} {'MAE_p90':>10} {'MFE/MAE':>8}")
    log.info("-" * 78)
    for h_name in ["1s", "5s", "10s", "30s"]:
        if h_name in summary.get("horizons", {}):
            s = summary["horizons"][h_name]["overall"]
            ratio = s["mfe_mean"] / max(s["mae_mean"], 0.001)
            log.info(f"{h_name:<8} {s['mfe_mean']:>10.3f} {s['mfe_p50']:>10.3f} "
                     f"{s['mfe_p90']:>10.3f} {s['mae_mean']:>10.3f} {s['mae_p50']:>10.3f} "
                     f"{s['mae_p90']:>10.3f} {ratio:>8.2f}")

    log.info("")
    log.info("HC #428 R2 compliance bounds (TP must be <= p90 MFE):")
    for h_name in ["1s", "5s", "10s", "30s"]:
        if h_name in summary.get("horizons", {}):
            r2 = summary["horizons"][h_name]["hc428_r2"]
            log.info(f"  {h_name}: max_TP={r2['max_tp_ticks']:.2f} ticks, "
                     f"expected_MAE_p90={r2['expected_mae_p90_ticks']:.2f} ticks")

    log.info("=" * 70)
    log.info("Done.")


if __name__ == "__main__":
    main()
