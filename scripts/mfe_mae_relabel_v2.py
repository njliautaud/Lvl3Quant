#!/usr/bin/env python3
"""
MFE/MAE Target Relabeling v2 — TRUE tick-by-tick price path
=============================================================
For each CNN-Mamba v2 prediction point, compute Maximum Favorable Excursion
(MFE) and Maximum Adverse Excursion (MAE) within {1s, 5s, 10s, 30s} horizons
using the ACTUAL cumulative mid-price path from raw MBO event data.

Unlike v1 (which interpolated between 4 label checkpoints), v2 walks the
real event-by-event price path via cumsum(events[:, 3]) and uses searchsorted
to bound horizon windows — giving exact MFE/MAE at full tick resolution.

Data source: smart_v3 .npz files (events, timestamps)
Model:       CNN-Mamba v2 (fold_10_best.pt) — 3x Conv1d + residual + Mamba
Output:      Per-date .npz with predictions, MFE/MAE/net per horizon

Runs on Windows (Razer, RTX 3070 8GB) with PyTorch CUDA.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["PYTHONUNBUFFERED"] = "1"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SMART_V3_DIR = Path(r"C:\Users\claude\Lvl3Quant\data\processed\mbo_events_smart_v3")
MODEL_PATH   = Path(r"C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_smart_v3_mar\fold_10_best.pt")
OUTPUT_DIR   = Path(r"C:\Users\claude\Lvl3Quant\output\mfe_mae_labels_v2")

WINDOW_SIZE  = 1000
STRIDE       = 250    # every 250 events (~62ms at typical event rate)
N_FEATURES   = 25
BATCH_SIZE   = 128    # GPU inference batch size
CHUNK_SIZE   = 10000  # MFE/MAE computation chunk size

# Model architecture constants
D_MODEL  = 96
D_STATE  = 32
N_LAYERS = 3
DT_RANK  = 6
D_CONV   = 4
DROPOUT  = 0.1
# Multi-scale CNN config embedded in model class (kernels 3,7,15,31, 16ch each)

# Horizons in nanoseconds
SEC_NS = 1_000_000_000
HORIZONS = {
    "1s":  1 * SEC_NS,
    "5s":  5 * SEC_NS,
    "10s": 10 * SEC_NS,
    "30s": 30 * SEC_NS,
}


# ===================================================================
# MODEL DEFINITION — matches fold_10_best.pt (canonical from
# live_trading/cnn_mamba_v2_model.py: 3x Conv1d + residual + Mamba)
# ===================================================================

class SelectiveSSM(nn.Module):
    """Mamba SSM block — pure PyTorch selective scan (no mamba_ssm CUDA kernels)."""

    def __init__(self, d_model: int = 96, d_state: int = 32,
                 dt_rank: int = 6, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        d_inner = d_model * 2  # 192

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv,
                                padding=d_conv - 1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        self.A_log = nn.Parameter(torch.zeros(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        """x: (B, L, d_model) -> (B, L, d_model)"""
        B, L, _ = x.shape
        d_inner = self.d_model * 2

        # Project and split into x_branch + gate
        xz = self.in_proj(x)                       # (B, L, 2*d_inner)
        x_branch, z = xz.split(d_inner, dim=-1)

        # Causal conv1d (padding=d_conv-1 on left, truncate right)
        x_conv = x_branch.transpose(1, 2)           # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]      # truncate to causal
        x_conv = x_conv.transpose(1, 2)
        x_branch = F.silu(x_conv)

        # SSM parameters from input
        x_dbl = self.x_proj(x_branch)
        dt, B_param, C_param = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))            # (B, L, d_inner)

        # Discretize: A_bar = exp(A * dt)
        A = -torch.exp(self.A_log)                   # (d_inner, d_state)

        # Vectorized pre-computation for the sequential scan
        # Memory: (B, L, d_inner, d_state) — at B=128, L=1000: ~3 GB
        # For RTX 3070 8GB this is tight; we split into sub-batches if needed
        A_exp = A.unsqueeze(0).unsqueeze(0)          # (1, 1, d_inner, d_state)
        dt_exp = dt.unsqueeze(-1)                    # (B, L, d_inner, 1)
        dA = torch.exp(A_exp * dt_exp)               # (B, L, d_inner, d_state)
        dB = B_param.unsqueeze(2) * dt_exp           # (B, L, d_inner, d_state)
        dBx = dB * x_branch.unsqueeze(-1)            # (B, L, d_inner, d_state)

        # Sequential scan
        y = torch.empty(B, L, d_inner, device=x.device, dtype=x.dtype)
        h = torch.zeros(B, d_inner, self.d_state, device=x.device, dtype=x.dtype)

        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            y[:, t] = (h * C_param[:, t].unsqueeze(1)).sum(-1)

        # Skip connection + gate
        y = y + x_branch * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """Pre-norm residual Mamba block."""

    def __init__(self, d_model=96, d_state=32, dt_rank=6, d_conv=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, dt_rank, d_conv)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.ssm(self.norm(x)))


class CNNMambaV2(nn.Module):
    """
    CNN-Mamba v2 — Feature MLP + Multi-Scale Temporal CNN + Mamba backbone.

    Architecture from fold_10_best.pt checkpoint:
      feature_mlp: Linear(25,128) → ReLU → Dropout → Linear(128,64) → LayerNorm(64)
      temporal_cnns: 4× Conv1d(25, 16, k) for k ∈ {3,7,15,31} → concat → 64 channels
      fusion_proj: Linear(128, 96) → LayerNorm(96)   [128 = 64 MLP + 64 CNN]
      blocks.{0,1,2}: MambaBlock (pre-norm + SSM)
      final_norm: LayerNorm(96)
      head: Linear(96,96) → GELU → Dropout → Linear(96,3)
    """

    def __init__(self, input_dim=25, d_model=96, d_state=32, n_layers=3,
                 dt_rank=6, d_conv=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Feature MLP: 25 → 128 → 64 + LayerNorm
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),      # 0
            nn.ReLU(),                       # 1
            nn.Dropout(dropout),             # 2
            nn.Linear(128, 64),              # 3
            nn.LayerNorm(64),                # 4
        )

        # Multi-scale temporal CNNs: 4 parallel Conv1d(25, 16, k)
        self.temporal_cnns = nn.ModuleList([
            nn.Sequential(nn.Conv1d(input_dim, 16, k, padding=k // 2))
            for k in [3, 7, 15, 31]
        ])

        # Fusion: concat(mlp_64, cnn_64) = 128 → d_model + LayerNorm
        self.fusion_proj = nn.Sequential(
            nn.Linear(128, d_model),         # 0
            nn.LayerNorm(d_model),           # 1
        )

        # Mamba backbone
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, dt_rank, d_conv, dropout)
            for _ in range(n_layers)
        ])

        # Final norm + prediction head
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),    # 0
            nn.GELU(),                       # 1
            nn.Dropout(dropout),             # 2
            nn.Linear(d_model, 3),           # 3
        )

    def forward(self, x):
        """
        x: (B, L, input_dim) — sequence of MBO events
        Returns: (B, 3) predictions for [1s, 5s, 10s] horizons
        """
        B, L, D = x.shape

        # Feature MLP (applied per-timestep)
        mlp_out = self.feature_mlp(x)                # (B, L, 64)

        # Multi-scale temporal CNN
        x_t = x.transpose(1, 2)                      # (B, input_dim, L)
        cnn_outs = []
        for cnn in self.temporal_cnns:
            out = cnn(x_t)                            # (B, 16, L')
            out = out[:, :, :L]                       # truncate to L (causal padding)
            cnn_outs.append(out)
        cnn_cat = torch.cat(cnn_outs, dim=1)          # (B, 64, L)
        cnn_cat = F.relu(cnn_cat)
        cnn_out = cnn_cat.transpose(1, 2)             # (B, L, 64)

        # Fusion
        fused = torch.cat([mlp_out, cnn_out], dim=-1) # (B, L, 128)
        h = self.fusion_proj(fused)                    # (B, L, d_model)

        # Mamba blocks
        for block in self.blocks:
            h = block(h)

        # Take last position, normalize, predict
        h = self.final_norm(h[:, -1, :])              # (B, d_model)
        return self.head(h)                           # (B, 3)


# ===================================================================
# MODEL LOADING
# ===================================================================

def load_model(device: str = "cuda") -> CNNMambaV2:
    """Load CNN-Mamba v2 from fold_10_best.pt, inferring architecture from weights."""
    if not MODEL_PATH.exists():
        print(f"ERROR: Model checkpoint not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading checkpoint: {MODEL_PATH}")
    ckpt = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)

    # The checkpoint stores state under 'model_state' key
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        # Assume the checkpoint IS the state dict
        state = ckpt

    # Diagnostic: print first 20 keys to verify architecture
    print(f"Checkpoint keys ({len(state)} total):")
    for i, k in enumerate(sorted(state.keys())):
        if i < 25:
            print(f"  {k}: {tuple(state[k].shape)}")
    if len(state) > 25:
        print(f"  ... ({len(state) - 25} more)")

    # Detect architecture from keys
    has_feature_mlp = "feature_mlp.0.weight" in state

    if not has_feature_mlp:
        print("\nERROR: Checkpoint does not match expected CNN-Mamba v2 architecture.")
        print("       Expected key 'feature_mlp.0.weight' not found.")
        print(f"       Available keys: {list(state.keys())[:10]}...")
        sys.exit(1)

    # Infer architecture from weight shapes
    input_dim = state["feature_mlp.0.weight"].shape[1]
    d_model = state["final_norm.weight"].shape[0]
    d_state = state["blocks.0.ssm.A_log"].shape[1]
    x_proj_out = state["blocks.0.ssm.x_proj.weight"].shape[0]
    dt_rank = x_proj_out - 2 * d_state
    n_layers = sum(1 for k in state if k.endswith(".ssm.A_log"))
    d_conv = state["blocks.0.ssm.conv1d.weight"].shape[2]

    print(f"\nInferred architecture: input={input_dim}, "
          f"d_model={d_model}, d_state={d_state}, dt_rank={dt_rank}, "
          f"n_layers={n_layers}, d_conv={d_conv}")

    model = CNNMambaV2(
        input_dim=input_dim, d_model=d_model, d_state=d_state,
        n_layers=n_layers, dt_rank=dt_rank, d_conv=d_conv,
    )

    # Strict load — architecture must match exactly
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"\nERROR: State dict mismatch: {e}")
        print("\nExpected keys:")
        for k, v in sorted(model.state_dict().items()):
            print(f"  {k}: {tuple(v.shape)}")
        print("\nCheckpoint keys:")
        for k, v in sorted(state.items()):
            print(f"  {k}: {tuple(v.shape)}")
        sys.exit(1)

    n_params = sum(p.numel() for p in model.parameters())
    fold = ckpt.get("fold", "?")
    val_ic = ckpt.get("val_ic_10s", 0)
    print(f"Model loaded: {n_params:,} params, fold={fold}, val_ic_10s={val_ic:.4f}")

    model.to(device).eval()
    return model


# ===================================================================
# GPU INFERENCE
# ===================================================================

def run_inference(model: CNNMambaV2, events: np.ndarray,
                  device: str = "cuda") -> tuple:
    """
    Run inference at stride=STRIDE across a full day of events.

    Returns:
        predictions: (M, 3) float32 — model output [1s, 5s, 10s]
        event_indices: (M,) int64 — the event index each prediction maps to
                       (last event in each window)
    """
    n_events = len(events)
    if n_events < WINDOW_SIZE:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=np.int64))

    # Window start indices: 0, STRIDE, 2*STRIDE, ...
    starts = np.arange(0, n_events - WINDOW_SIZE + 1, STRIDE)
    n_windows = len(starts)
    # Each prediction corresponds to the LAST event in its window
    pred_indices = starts + WINDOW_SIZE - 1

    all_preds = np.zeros((n_windows, 3), dtype=np.float32)

    # Process in GPU batches — use smaller effective batch to stay within
    # 8GB VRAM (SSM scan allocates ~B*L*d_inner*d_state*4 bytes internally)
    # At B=128, L=1000, d_inner=192, d_state=32: ~3 GB for scan tensors
    # Plus model params + activations ~1.5 GB => ~4.5 GB total, fits in 8 GB
    effective_bs = BATCH_SIZE

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        for batch_start in range(0, n_windows, effective_bs):
            batch_end = min(batch_start + effective_bs, n_windows)
            batch_starts = starts[batch_start:batch_end]
            bs = len(batch_starts)

            # Build batch: (bs, WINDOW_SIZE, N_FEATURES)
            batch = np.zeros((bs, WINDOW_SIZE, N_FEATURES), dtype=np.float32)
            for bi, s in enumerate(batch_starts):
                batch[bi] = events[s:s + WINDOW_SIZE]

            x = torch.from_numpy(batch).to(device)
            preds = model(x)
            all_preds[batch_start:batch_end] = preds.cpu().numpy()

    return all_preds, pred_indices


# ===================================================================
# MFE/MAE COMPUTATION — TRUE TICK-BY-TICK
# ===================================================================

def compute_mfe_mae(
    pred_indices: np.ndarray,
    predictions: np.ndarray,
    timestamps_ns: np.ndarray,
    mid_price: np.ndarray,
) -> dict:
    """
    Compute true MFE/MAE/net from the actual tick-by-tick mid-price path.

    For each prediction at event index i with direction d:
      - Find event range [i, j) where timestamps[j-1] <= timestamps[i] + horizon
      - price_path = mid_price[i:j] - mid_price[i]  (relative to entry)
      - LONG  (d > 0): MFE = max(path), MAE = -min(path)  [both >= 0]
      - SHORT (d < 0): MFE = -min(path), MAE = max(path)  [both >= 0]
      - net = mid_price[j-1] - mid_price[i]  (signed, at horizon end)

    Vectorized approach: use searchsorted to find horizon endpoints,
    then process in chunks using cumulative max/min over slices.
    """
    n_preds = len(pred_indices)
    n_events = len(mid_price)

    # Use 10s head as primary direction signal
    pred_10s = predictions[:, 2]
    direction = np.sign(pred_10s).astype(np.float64)
    direction[direction == 0] = 1.0  # neutral -> long

    # Timestamps at each prediction point
    pred_ts = timestamps_ns[pred_indices]

    result = {}

    for h_name, h_ns in HORIZONS.items():
        mfe = np.zeros(n_preds, dtype=np.float32)
        mae = np.zeros(n_preds, dtype=np.float32)
        net = np.zeros(n_preds, dtype=np.float32)

        # Find horizon endpoint for each prediction using searchsorted
        # j_end[k] = first event index AFTER timestamps[pred_indices[k]] + h_ns
        end_ts = pred_ts + h_ns
        j_ends = np.searchsorted(timestamps_ns, end_ts, side="right")
        # Clip to valid range
        j_ends = np.minimum(j_ends, n_events)

        # Process in chunks to limit memory usage
        for chunk_start in range(0, n_preds, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n_preds)

            for k in range(chunk_start, chunk_end):
                i = pred_indices[k]
                j = j_ends[k]

                if j <= i + 1:
                    # No events in horizon (or only the entry event itself)
                    continue

                # Price path relative to entry (in ticks)
                path = mid_price[i:j] - mid_price[i]

                path_max = path.max()
                path_min = path.min()

                if direction[k] > 0:
                    # LONG: favorable = price up, adverse = price down
                    mfe[k] = max(path_max, 0.0)
                    mae[k] = max(-path_min, 0.0)
                else:
                    # SHORT: favorable = price down, adverse = price up
                    mfe[k] = max(-path_min, 0.0)
                    mae[k] = max(path_max, 0.0)

                # Net move at horizon end (signed, in direction of trade)
                end_move = mid_price[j - 1] - mid_price[i]
                net[k] = end_move * direction[k]

        result[f"mfe_{h_name}"] = mfe
        result[f"mae_{h_name}"] = mae
        result[f"net_{h_name}"] = net

    return result


# ===================================================================
# PER-DATE PROCESSING
# ===================================================================

def process_date(date_str: str, model: CNNMambaV2,
                 device: str = "cuda") -> dict:
    """Process one date: load data, run inference, compute MFE/MAE, save."""
    t0 = time.time()

    # Load smart_v3 events
    sv3_path = SMART_V3_DIR / f"{date_str}_mbo_events.npz"
    sv3 = np.load(str(sv3_path), allow_pickle=True)
    events = sv3["events"]           # (N, 25) float32
    timestamps = sv3["timestamps"]   # (N,) int64
    n_events = len(events)

    # Compute cumulative mid-price path from price_rel_ticks (column 3)
    mid_price = np.cumsum(events[:, 3]).astype(np.float64)

    print(f"  [{date_str}] {n_events:,} events, "
          f"price range: {mid_price.min():.1f} to {mid_price.max():.1f} ticks")

    # Run GPU inference
    t_inf = time.time()
    predictions, pred_indices = run_inference(model, events, device)
    n_preds = len(predictions)
    inf_elapsed = time.time() - t_inf

    if n_preds == 0:
        print(f"  [{date_str}] No predictions (too few events), skipping")
        return {}

    print(f"  [{date_str}] {n_preds:,} predictions, "
          f"inference: {inf_elapsed:.1f}s")

    # Compute TRUE MFE/MAE from tick-by-tick price path
    t_mfe = time.time()
    mfe_mae = compute_mfe_mae(pred_indices, predictions, timestamps, mid_price)
    mfe_elapsed = time.time() - t_mfe

    # Save output
    out_path = OUTPUT_DIR / f"{date_str}_mfe_mae.npz"
    np.savez_compressed(
        str(out_path),
        predictions=predictions.astype(np.float32),
        event_indices=pred_indices.astype(np.int64),
        timestamps_ns=timestamps[pred_indices].astype(np.int64),
        date=date_str,
        **{k: v.astype(np.float32) for k, v in mfe_mae.items()},
    )

    elapsed = time.time() - t0

    # Print per-horizon summary
    for h_name in ["1s", "5s", "10s", "30s"]:
        m = mfe_mae[f"mfe_{h_name}"]
        a = mfe_mae[f"mae_{h_name}"]
        n = mfe_mae[f"net_{h_name}"]
        print(f"    {h_name:>3s}: MFE mean={m.mean():.3f} p90={np.percentile(m, 90):.3f}  "
              f"MAE mean={a.mean():.3f} p90={np.percentile(a, 90):.3f}  "
              f"net mean={n.mean():.3f}")

    print(f"  [{date_str}] Done: {n_preds:,} preds, "
          f"inf={inf_elapsed:.1f}s mfe={mfe_elapsed:.1f}s total={elapsed:.1f}s")

    return {
        "date": date_str,
        "n_predictions": n_preds,
        "elapsed_s": elapsed,
        "mfe_mae": mfe_mae,
        "predictions": predictions,
    }


# ===================================================================
# SUMMARY STATISTICS
# ===================================================================

def compute_summary(all_results: list) -> dict:
    """Compute cross-date summary stats with HC #428 R2 compliance bounds."""
    summary = {
        "total_predictions": 0,
        "dates_processed": 0,
        "per_date": [],
        "horizons": {},
    }

    all_preds_10s = []
    all_mfe = {h: [] for h in HORIZONS}
    all_mae = {h: [] for h in HORIZONS}
    all_net = {h: [] for h in HORIZONS}

    for r in all_results:
        if not r:
            continue
        summary["dates_processed"] += 1
        summary["total_predictions"] += r["n_predictions"]

        # Per-date mini-summary
        date_stats = {"date": r["date"], "n_predictions": r["n_predictions"]}
        for h_name in HORIZONS:
            m = r["mfe_mae"][f"mfe_{h_name}"]
            a = r["mfe_mae"][f"mae_{h_name}"]
            date_stats[f"mfe_{h_name}_mean"] = float(np.mean(m))
            date_stats[f"mae_{h_name}_mean"] = float(np.mean(a))
        summary["per_date"].append(date_stats)

        all_preds_10s.append(r["predictions"][:, 2])
        for h_name in HORIZONS:
            all_mfe[h_name].append(r["mfe_mae"][f"mfe_{h_name}"])
            all_mae[h_name].append(r["mfe_mae"][f"mae_{h_name}"])
            all_net[h_name].append(r["mfe_mae"][f"net_{h_name}"])

    if not all_preds_10s:
        return summary

    preds_10s = np.concatenate(all_preds_10s)
    abs_preds = np.abs(preds_10s)
    direction = np.sign(preds_10s)

    for h_name in HORIZONS:
        mfe_all = np.concatenate(all_mfe[h_name])
        mae_all = np.concatenate(all_mae[h_name])
        net_all = np.concatenate(all_net[h_name])

        def _stats(arr):
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "n": int(len(arr)),
            }

        h_stats = {
            "overall": {
                "mfe": _stats(mfe_all),
                "mae": _stats(mae_all),
                "net": _stats(net_all),
            },
            "by_direction": {},
            "by_confidence_decile": {},
        }

        # By direction (long vs short)
        for dir_name, dir_val in [("short", -1), ("long", 1)]:
            mask = direction == dir_val
            if mask.sum() > 0:
                h_stats["by_direction"][dir_name] = {
                    "mfe": _stats(mfe_all[mask]),
                    "mae": _stats(mae_all[mask]),
                    "net": _stats(net_all[mask]),
                }

        # By signal strength decile
        decile_edges = np.percentile(abs_preds, np.arange(0, 101, 10))
        for d in range(10):
            lo = decile_edges[d]
            hi = decile_edges[d + 1] if d < 9 else np.inf
            mask = (abs_preds >= lo) & (abs_preds < hi) if d < 9 else (abs_preds >= lo)
            if mask.sum() > 0:
                h_stats["by_confidence_decile"][f"d{d+1}"] = {
                    "abs_pred_range": [float(lo),
                                       float(hi) if hi != np.inf else "inf"],
                    "mfe": _stats(mfe_all[mask]),
                    "mae": _stats(mae_all[mask]),
                    "net": _stats(net_all[mask]),
                }

        # HC #428 R2 compliance: TP must be <= p90 MFE within horizon
        top10_mask = abs_preds >= np.percentile(abs_preds, 90)
        h_stats["hc428_r2"] = {
            "max_tp_ticks_overall": float(np.percentile(mfe_all, 90)),
            "max_tp_ticks_top10pct": float(
                np.percentile(mfe_all[top10_mask], 90)
            ) if top10_mask.sum() > 0 else None,
            "expected_mae_p90": float(np.percentile(mae_all, 90)),
            "mfe_mae_ratio": float(
                np.mean(mfe_all) / max(np.mean(mae_all), 0.001)
            ),
        }

        summary["horizons"][h_name] = h_stats

    return summary


# ===================================================================
# MAIN
# ===================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MFE/MAE Target Relabeling v2 — TRUE Tick-by-Tick Price Path")
    print("=" * 72)

    # CUDA setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No GPU — running on CPU (slow)")

    # Load model
    print("\nLoading CNN-Mamba v2 model...")
    model = load_model(device)
    print()

    # Find all dates to process
    if not SMART_V3_DIR.exists():
        print(f"ERROR: Data directory not found: {SMART_V3_DIR}")
        sys.exit(1)

    date_files = sorted(SMART_V3_DIR.glob("*_mbo_events.npz"))
    all_dates = [f.stem.replace("_mbo_events", "") for f in date_files]
    print(f"Found {len(all_dates)} dates in smart_v3 directory")

    # Skip already completed dates (validate required keys exist)
    done_dates = set()
    required_keys = {"predictions", "event_indices", "timestamps_ns",
                     "mfe_1s", "mae_1s", "net_1s",
                     "mfe_5s", "mae_5s", "net_5s",
                     "mfe_10s", "mae_10s", "net_10s",
                     "mfe_30s", "mae_30s", "net_30s"}

    for f in OUTPUT_DIR.glob("*_mfe_mae.npz"):
        d = f.stem.replace("_mfe_mae", "")
        try:
            check = np.load(str(f), allow_pickle=True)
            if required_keys.issubset(set(check.files)):
                done_dates.add(d)
            else:
                missing = required_keys - set(check.files)
                print(f"  Removing incomplete {d} (missing: {missing})")
                f.unlink()
        except Exception:
            print(f"  Removing corrupt {d}")
            f.unlink()

    todo = [d for d in all_dates if d not in done_dates]
    print(f"Already done: {len(done_dates)}, remaining: {len(todo)}")

    # Process remaining dates
    new_results = []
    for i, date_str in enumerate(todo):
        print(f"\nProcessing {date_str} ({i+1}/{len(todo)})...")
        try:
            result = process_date(date_str, model, device)
            new_results.append(result)
        except Exception as e:
            print(f"  [{date_str}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            if device == "cuda":
                torch.cuda.empty_cache()

        # Clear GPU cache between dates
        if device == "cuda":
            torch.cuda.empty_cache()

    # Load all completed dates (including previously done) for summary
    print("\nLoading all completed dates for summary...")
    all_results = []
    for d in all_dates:
        out_path = OUTPUT_DIR / f"{d}_mfe_mae.npz"
        if not out_path.exists():
            continue
        try:
            data = np.load(str(out_path), allow_pickle=True)
            mfe_mae = {}
            for h_name in HORIZONS:
                mfe_mae[f"mfe_{h_name}"] = data[f"mfe_{h_name}"]
                mfe_mae[f"mae_{h_name}"] = data[f"mae_{h_name}"]
                mfe_mae[f"net_{h_name}"] = data[f"net_{h_name}"]
            all_results.append({
                "date": d,
                "n_predictions": len(data["predictions"]),
                "elapsed_s": 0,
                "mfe_mae": mfe_mae,
                "predictions": data["predictions"],
            })
        except Exception as e:
            print(f"  Warning: could not load {d}: {e}")

    # Compute and save summary
    print("\nComputing summary statistics...")
    summary = compute_summary(all_results)

    summary_path = OUTPUT_DIR / "summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")

    # Print formatted summary table
    print("\n" + "=" * 72)
    print(f"SUMMARY: {summary['dates_processed']} dates, "
          f"{summary['total_predictions']:,} predictions")
    print("=" * 72)
    print(f"{'Horizon':<8} {'MFE_mean':>9} {'MFE_p50':>9} {'MFE_p90':>9} "
          f"{'MAE_mean':>9} {'MAE_p50':>9} {'MAE_p90':>9} "
          f"{'Net_mean':>9} {'MFE/MAE':>8}")
    print("-" * 82)

    for h_name in ["1s", "5s", "10s", "30s"]:
        if h_name not in summary.get("horizons", {}):
            continue
        s = summary["horizons"][h_name]["overall"]
        ratio = s["mfe"]["mean"] / max(s["mae"]["mean"], 0.001)
        print(f"{h_name:<8} "
              f"{s['mfe']['mean']:>9.3f} {s['mfe']['median']:>9.3f} "
              f"{s['mfe']['p90']:>9.3f} "
              f"{s['mae']['mean']:>9.3f} {s['mae']['median']:>9.3f} "
              f"{s['mae']['p90']:>9.3f} "
              f"{s['net']['mean']:>9.3f} {ratio:>8.2f}")

    print("\nHC #428 R2 compliance bounds (TP must be <= p90 MFE within horizon):")
    for h_name in ["1s", "5s", "10s", "30s"]:
        if h_name not in summary.get("horizons", {}):
            continue
        r2 = summary["horizons"][h_name]["hc428_r2"]
        print(f"  {h_name}: max_TP={r2['max_tp_ticks_overall']:.2f} ticks, "
              f"top10% max_TP={r2.get('max_tp_ticks_top10pct', 'N/A')}, "
              f"MAE_p90={r2['expected_mae_p90']:.2f} ticks, "
              f"MFE/MAE={r2['mfe_mae_ratio']:.2f}")

    print("\n" + "=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
