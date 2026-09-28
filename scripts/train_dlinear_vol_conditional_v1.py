#!/usr/bin/env python3
"""
DLinear Vol-Conditional Quantile v1.

PURPOSE
-------
Extend the proven DLinear quantile model (W=500, pinball loss, IC_P50=0.21/0.14/0.10
at 1s/5s/10s) with realized-volatility conditioning. The hypothesis: different vol
regimes require different prediction surfaces. A vol embedding lets the model learn
regime-specific quantile estimates.

ARCHITECTURE
------------
  1. DLinear trunk (same as proven quantile): trend + seasonal decomposition,
     W=500 window, trend branch + seasonal branch -> 128-dim each -> 256-dim concat.
  2. Vol conditioning branch: compute realized vol from last 50 events' price_rel_ticks
     std + qty_log std, bucket into 5 quantiles (within training window), embed as
     16-dim vector.
  3. Concatenate DLinear trunk (256) + vol embedding (16) = 272-dim.
  4. Head: 272 -> GELU -> Linear(272, 128) -> GELU -> Linear(128, 9).
     Output = 3 quantiles (P10/P50/P90) x 3 horizons (1s/5s/10s) = 9 outputs.
  5. Pinball (quantile) loss.
  ~3.5M params total.

DATA
----
  NPZ files: data/processed/mbo_events_smart_v3/{YYYYMMDD}_mbo_events.npz
  25 features (smart_v3), labels_1s/5s/10s.
  Feature columns used for vol:
    col 3: price_rel_ticks (mid price change proxy)
    col 4: qty_log (signed trade quantity proxy)

TRAINING
--------
  Sliding walk-forward: 10-day train, 1-day OOT, drop oldest.
  Pinball loss, AdamW, lr=1e-3, 3 epochs, batch 512.

METRICS: per-fold Spearman IC for P50 at each horizon, concat IC across all OOT folds.

TARGET: Razer (Windows, RTX 3070 8GB VRAM). num_workers=0 for DataLoader.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import mlflow
    HAVE_MLFLOW = True
except Exception:
    HAVE_MLFLOW = False

# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────
REPO = Path(os.environ.get("LVL3_ROOT", r"C:\Users\claude\Lvl3Quant"))
MBO_DIR = REPO / "data" / "processed" / "mbo_events_smart_v3"

WINDOW = 500
TRAIN_STRIDE = 25
OOT_STRIDE = 5
BATCH = 512
N_FEAT = 25
HORIZONS = ["1s", "5s", "10s"]
N_HORIZONS = len(HORIZONS)
QUANTILES = [0.10, 0.50, 0.90]
N_QUANTILES = len(QUANTILES)
N_TRAIN_DAYS = 10
N_EPOCHS = 3
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MLFLOW_URI = "http://jupiter:5000"
MLFLOW_EXP = "hc488_dlinear_vol_conditional_v1"

# Vol conditioning config
VOL_LOOKBACK = 50       # events to compute realized vol
VOL_N_BUCKETS = 5       # number of vol regime buckets
VOL_EMBED_DIM = 16      # embedding dimension for vol bucket
VOL_PRICE_COL = 3       # price_rel_ticks column index
VOL_QTY_COL = 4         # qty_log column index

# Progress printing
PRINT_EVERY = 100       # print every N batches


# ────────────────────────────────────────────────────────────────────────────
# Volatility feature computation
# ────────────────────────────────────────────────────────────────────────────
def compute_vol_features(events: np.ndarray, lookback: int = VOL_LOOKBACK) -> np.ndarray:
    """
    Compute per-event realized volatility from rolling std of price_rel_ticks
    and qty_log over last `lookback` events.

    Returns (N,) float32 array: combined vol = sqrt(price_std^2 + qty_std^2).
    """
    N = events.shape[0]
    price = events[:, VOL_PRICE_COL].astype(np.float64)
    qty = events[:, VOL_QTY_COL].astype(np.float64)

    # Rolling std via cumsum trick (causal, no future leak)
    def rolling_std(sig, W):
        cs = np.concatenate(([0.0], np.cumsum(sig)))
        cs2 = np.concatenate(([0.0], np.cumsum(sig ** 2)))
        idx = np.arange(N, dtype=np.int64)
        start = np.maximum(0, idx - W + 1)
        end = idx + 1
        counts = (end - start).astype(np.float64)
        counts = np.maximum(counts, 2.0)  # need at least 2 for std
        sums = cs[end] - cs[start]
        sums2 = cs2[end] - cs2[start]
        means = sums / counts
        var = np.maximum(sums2 / counts - means ** 2, 0.0)
        return np.sqrt(var).astype(np.float32)

    price_std = rolling_std(price, lookback)
    qty_std = rolling_std(qty, lookback)
    combined_vol = np.sqrt(price_std.astype(np.float64) ** 2 +
                           qty_std.astype(np.float64) ** 2).astype(np.float32)
    return combined_vol


def compute_vol_buckets_train(vol_values: np.ndarray, n_buckets: int = VOL_N_BUCKETS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Percentile-rank vol values within training set, bucket into n_buckets.
    Returns (bucket_ids as int64, percentile_edges as float32).
    """
    # Compute percentile edges from non-NaN values
    valid = vol_values[~np.isnan(vol_values)]
    if len(valid) == 0:
        return np.zeros(len(vol_values), dtype=np.int64), np.zeros(n_buckets + 1, dtype=np.float32)

    edges = np.percentile(valid, np.linspace(0, 100, n_buckets + 1))
    edges = edges.astype(np.float32)
    # Ensure monotonicity (nudge duplicates)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-8

    # Bucket assignment: np.digitize gives 1-based, subtract 1, clip to [0, n_buckets-1]
    buckets = np.digitize(vol_values, edges[1:-1]).astype(np.int64)
    buckets = np.clip(buckets, 0, n_buckets - 1)
    return buckets, edges


def apply_vol_buckets(vol_values: np.ndarray, edges: np.ndarray, n_buckets: int = VOL_N_BUCKETS) -> np.ndarray:
    """Apply pre-computed percentile edges to new vol values."""
    buckets = np.digitize(vol_values, edges[1:-1]).astype(np.int64)
    buckets = np.clip(buckets, 0, n_buckets - 1)
    return buckets


# ────────────────────────────────────────────────────────────────────────────
# Model — DLinear trunk + vol conditioning
# ────────────────────────────────────────────────────────────────────────────
class DLinearVolConditionalQuantile(nn.Module):
    """
    DLinear with volatility conditioning for quantile prediction.

    Trunk: trend/seasonal decomposition via moving average.
    Vol branch: embedding of vol regime bucket (5 buckets -> 16-dim).
    Head: concat(trunk_256, vol_embed_16) -> 272 -> GELU -> 128 -> GELU -> n_h * n_q.
    """
    def __init__(self, window: int = WINDOW, n_feat: int = N_FEAT,
                 n_horizons: int = N_HORIZONS, n_quantiles: int = N_QUANTILES,
                 n_vol_buckets: int = VOL_N_BUCKETS, vol_embed_dim: int = VOL_EMBED_DIM,
                 kernel: int = 25):
        super().__init__()
        self.window = window
        self.n_h = n_horizons
        self.n_q = n_quantiles
        self.n_out = n_horizons * n_quantiles

        # DLinear trunk: trend + seasonal decomposition
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=kernel // 2)
        self.lin_trend = nn.Linear(window * n_feat, 128)
        self.lin_season = nn.Linear(window * n_feat, 128)

        # Vol conditioning branch
        self.vol_embed = nn.Embedding(n_vol_buckets, vol_embed_dim)

        # Head: 256 (trunk) + vol_embed_dim -> output
        trunk_dim = 256  # 128 trend + 128 seasonal
        fused_dim = trunk_dim + vol_embed_dim  # 272
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.n_out),
        )

    def forward(self, x: torch.Tensor, vol_bucket: torch.Tensor) -> torch.Tensor:
        """
        x: (B, W, F)
        vol_bucket: (B,) int64 — vol regime bucket [0, n_vol_buckets)
        Returns: (B, n_h, n_q)
        """
        B, W, F_ = x.shape

        # DLinear trunk
        x_t = x.transpose(1, 2)  # (B, F, W)
        trend = self.avg(x_t)
        if trend.shape[-1] > W:
            trend = trend[..., :W]
        season = x_t - trend
        flat_t = trend.flatten(1)  # (B, W*F)
        flat_s = season.flatten(1)
        trunk_h = torch.cat([self.lin_trend(flat_t), self.lin_season(flat_s)], dim=1)  # (B, 256)

        # Vol conditioning
        vol_h = self.vol_embed(vol_bucket)  # (B, vol_embed_dim)

        # Fuse and predict
        fused = torch.cat([trunk_h, vol_h], dim=1)  # (B, 272)
        out = self.head(fused)  # (B, n_h * n_q)
        return out.view(B, self.n_h, self.n_q)  # (B, n_h, n_q)


# ────────────────────────────────────────────────────────────────────────────
# Pinball loss
# ────────────────────────────────────────────────────────────────────────────
def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                 quantiles: List[float]) -> torch.Tensor:
    """
    y_pred: (B, n_h, n_q)
    y_true: (B, n_h) — broadcast to match n_q
    quantiles: list of tau values [0.1, 0.5, 0.9]
    """
    taus = torch.tensor(quantiles, device=y_pred.device, dtype=y_pred.dtype)  # (n_q,)
    y_true_exp = y_true.unsqueeze(-1)  # (B, n_h, 1)
    error = y_true_exp - y_pred  # (B, n_h, n_q)
    loss = torch.max(taus * error, (taus - 1.0) * error)  # (B, n_h, n_q)
    return loss.mean()


# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────
def list_dates(mbo_dir: Path) -> List[str]:
    files = sorted(mbo_dir.glob("*_mbo_events.npz"))
    return [f.name.split("_")[0] for f in files]


def load_day(mbo_dir: Path, date: str):
    """Returns (events, labels_1s, labels_5s, labels_10s)."""
    f = mbo_dir / f"{date}_mbo_events.npz"
    d = np.load(f, allow_pickle=True)
    ev = d["events"].astype(np.float32)
    l1 = d["labels_1s"].astype(np.float32)
    l5 = d["labels_5s"].astype(np.float32)
    l10 = d["labels_10s"].astype(np.float32)
    return ev, l1, l5, l10


def valid_starts(n: int, window: int, stride: int,
                 l1: np.ndarray, l5: np.ndarray, l10: np.ndarray) -> np.ndarray:
    """Get valid window start indices where no target is NaN at window end."""
    starts = np.arange(0, n - window + 1, stride)
    end_idx = starts + window - 1
    mask = ~(np.isnan(l1[end_idx]) | np.isnan(l5[end_idx]) | np.isnan(l10[end_idx]))
    return starts[mask]


def compute_feature_stats(mbo_dir: Path, train_dates: List[str],
                          sample_frac: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/std of features from training dates for normalization."""
    mu = np.zeros(N_FEAT, dtype=np.float64)
    sq = np.zeros(N_FEAT, dtype=np.float64)
    n_total = 0
    for date in train_dates:
        try:
            ev, *_ = load_day(mbo_dir, date)
        except Exception:
            continue
        n_take = max(int(len(ev) * sample_frac), min(1000, len(ev)))
        idx = np.random.choice(len(ev), n_take, replace=False)
        sub = ev[idx]
        mu += sub.sum(0)
        sq += (sub ** 2).sum(0)
        n_total += len(sub)
    mu /= n_total
    var = sq / n_total - mu ** 2
    sd = np.sqrt(np.clip(var, 1e-12, None))
    return mu.astype(np.float32), sd.astype(np.float32)


def compute_vol_edges_from_train(mbo_dir: Path, train_dates: List[str],
                                 sample_frac: float = 0.1) -> np.ndarray:
    """Compute vol percentile edges from training data (used for bucketing)."""
    all_vols = []
    for date in train_dates:
        try:
            ev, *_ = load_day(mbo_dir, date)
        except Exception:
            continue
        vol = compute_vol_features(ev)
        # Sample to keep memory reasonable
        n_take = max(int(len(vol) * sample_frac), min(5000, len(vol)))
        idx = np.random.choice(len(vol), n_take, replace=False)
        all_vols.append(vol[idx])
    all_vols = np.concatenate(all_vols)
    _, edges = compute_vol_buckets_train(all_vols, VOL_N_BUCKETS)
    return edges


# ────────────────────────────────────────────────────────────────────────────
# Train one fold
# ────────────────────────────────────────────────────────────────────────────
def train_one_fold(model: nn.Module, mbo_dir: Path, train_dates: List[str],
                   mu: np.ndarray, sd: np.ndarray, vol_edges: np.ndarray,
                   out_dir: Path, fold_idx: int,
                   log: logging.Logger) -> nn.Module:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    for ep in range(N_EPOCHS):
        ep_loss = 0.0
        n_batches = 0
        t_ep_start = time.time()

        for date in train_dates:
            try:
                ev, l1, l5, l10 = load_day(mbo_dir, date)
            except Exception as e:
                log.warning(f"  load fail {date}: {e}")
                continue

            starts = valid_starts(len(ev), WINDOW, TRAIN_STRIDE, l1, l5, l10)
            if len(starts) < BATCH:
                continue

            # Precompute vol features + buckets for this day
            vol = compute_vol_features(ev)
            vol_buckets = apply_vol_buckets(vol, vol_edges, VOL_N_BUCKETS)

            perm = np.random.permutation(len(starts))
            starts = starts[perm]

            for i in range(0, len(starts), BATCH):
                bs = starts[i:i + BATCH]
                if len(bs) < 2:
                    continue

                # Extract windows
                xb = np.stack([ev[s:s + WINDOW] for s in bs])
                xb = (xb - mu) / sd

                # Extract vol buckets at window end position
                end_idx = bs + WINDOW - 1
                vb = vol_buckets[end_idx]

                # Extract targets: (B, n_h)
                yb = np.stack([l1[end_idx], l5[end_idx], l10[end_idx]], axis=1)

                xb_t = torch.from_numpy(xb).to(DEVICE, non_blocking=True)
                vb_t = torch.from_numpy(vb).to(DEVICE, non_blocking=True)
                yb_t = torch.from_numpy(yb).to(DEVICE, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                pred = model(xb_t, vb_t)  # (B, n_h, n_q)
                loss = pinball_loss(pred, yb_t, QUANTILES)
                loss.backward()
                opt.step()

                ep_loss += float(loss.item())
                n_batches += 1

                if n_batches % PRINT_EVERY == 0:
                    elapsed = time.time() - t_ep_start
                    avg_loss = ep_loss / n_batches
                    log.info(f"    ep {ep+1}/{N_EPOCHS} batch {n_batches} "
                             f"loss={avg_loss:.6f} elapsed={elapsed:.0f}s")

        avg_ep_loss = ep_loss / max(n_batches, 1)
        log.info(f"    epoch {ep+1}/{N_EPOCHS} avg_pinball={avg_ep_loss:.6f} "
                 f"batches={n_batches} time={time.time()-t_ep_start:.0f}s")

        if HAVE_MLFLOW:
            try:
                mlflow.log_metric("train_pinball", avg_ep_loss, step=ep)
            except Exception:
                pass

        # Intra-fold checkpoint after each epoch
        ckpt_path = out_dir / f"fold_{fold_idx:02d}_epoch_{ep+1}.pt"
        torch.save({
            "epoch": ep + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "loss": avg_ep_loss,
        }, ckpt_path)

    return model


# ────────────────────────────────────────────────────────────────────────────
# Predict one OOT day
# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_day(model: nn.Module, mbo_dir: Path, date: str,
                mu: np.ndarray, sd: np.ndarray, vol_edges: np.ndarray):
    model.eval()
    ev, l1, l5, l10 = load_day(mbo_dir, date)
    starts = valid_starts(len(ev), WINDOW, OOT_STRIDE, l1, l5, l10)
    if len(starts) == 0:
        return None, None, None

    # Precompute vol
    vol = compute_vol_features(ev)
    vol_buckets = apply_vol_buckets(vol, vol_edges, VOL_N_BUCKETS)

    preds_chunks, labs_chunks = [], []
    for i in range(0, len(starts), BATCH):
        bs = starts[i:i + BATCH]
        xb = np.stack([ev[s:s + WINDOW] for s in bs])
        xb = (xb - mu) / sd
        end_idx = bs + WINDOW - 1
        vb = vol_buckets[end_idx]
        yb = np.stack([l1[end_idx], l5[end_idx], l10[end_idx]], axis=1)  # (B, n_h)

        xb_t = torch.from_numpy(xb).to(DEVICE, non_blocking=True)
        vb_t = torch.from_numpy(vb).to(DEVICE, non_blocking=True)

        pred = model(xb_t, vb_t).cpu().numpy()  # (B, n_h, n_q)
        preds_chunks.append(pred)
        labs_chunks.append(yb)

    P = np.concatenate(preds_chunks, 0)  # (N, n_h, n_q)
    Y = np.concatenate(labs_chunks, 0)   # (N, n_h)

    from scipy.stats import spearmanr

    # Compute IC for P50 (quantile index 1) at each horizon
    ic_p50 = []
    for hi in range(N_HORIZONS):
        p = P[:, hi, 1]  # P50 predictions for horizon hi
        y = Y[:, hi]
        valid = ~(np.isnan(p) | np.isnan(y))
        if valid.sum() >= 50:
            r, _ = spearmanr(p[valid], y[valid])
            ic_p50.append(float(r))
        else:
            ic_p50.append(float("nan"))

    # Compute IC for P10 and P90 as well (diagnostic)
    ic_p10 = []
    ic_p90 = []
    for hi in range(N_HORIZONS):
        y = Y[:, hi]
        for qi, ic_list in [(0, ic_p10), (2, ic_p90)]:
            p = P[:, hi, qi]
            valid = ~(np.isnan(p) | np.isnan(y))
            if valid.sum() >= 50:
                r, _ = spearmanr(p[valid], y[valid])
                ic_list.append(float(r))
            else:
                ic_list.append(float("nan"))

    # Quantile coverage: fraction of labels below P10 (should be ~10%) and below P90 (~90%)
    coverage_p10, coverage_p90, width_p10_p90 = [], [], []
    for hi in range(N_HORIZONS):
        y = Y[:, hi]
        valid = ~np.isnan(y)
        if valid.sum() >= 50:
            coverage_p10.append(float(np.mean(y[valid] < P[valid, hi, 0])))
            coverage_p90.append(float(np.mean(y[valid] < P[valid, hi, 2])))
            width_p10_p90.append(float(np.mean(P[valid, hi, 2] - P[valid, hi, 0])))
        else:
            coverage_p10.append(float("nan"))
            coverage_p90.append(float("nan"))
            width_p10_p90.append(float("nan"))

    diag = dict(
        ic_p50=ic_p50,
        ic_p10=ic_p10,
        ic_p90=ic_p90,
        coverage_p10=coverage_p10,
        coverage_p90=coverage_p90,
        width_p10_p90=width_p10_p90,
    )
    return P, Y, diag


# ────────────────────────────────────────────────────────────────────────────
# Concat IC: concatenate all OOT predictions across folds
# ────────────────────────────────────────────────────────────────────────────
def compute_concat_ic(out_dir: Path, log: logging.Logger):
    """Load all fold predictions and compute concat IC across all OOT days."""
    from scipy.stats import spearmanr

    all_preds = []
    all_labels = []
    fold_files = sorted(out_dir.glob("fold_*_preds.npz"))
    for f in fold_files:
        d = np.load(f, allow_pickle=True)
        all_preds.append(d["preds"])    # (N, n_h, n_q)
        all_labels.append(d["labels"])  # (N, n_h)

    if not all_preds:
        log.warning("No fold predictions found for concat IC")
        return

    P = np.concatenate(all_preds, 0)
    Y = np.concatenate(all_labels, 0)
    log.info(f"\n=== CONCAT IC ({len(fold_files)} folds, {len(P):,} samples) ===")

    for hi, hname in enumerate(HORIZONS):
        p50 = P[:, hi, 1]
        y = Y[:, hi]
        valid = ~(np.isnan(p50) | np.isnan(y))
        if valid.sum() >= 50:
            r, _ = spearmanr(p50[valid], y[valid])
            log.info(f"  {hname}: concat_IC_P50 = {r:.4f}  (N={valid.sum():,})")
        else:
            log.info(f"  {hname}: insufficient valid samples")

    if HAVE_MLFLOW:
        try:
            for hi, hname in enumerate(HORIZONS):
                p50 = P[:, hi, 1]
                y = Y[:, hi]
                valid = ~(np.isnan(p50) | np.isnan(y))
                if valid.sum() >= 50:
                    r, _ = spearmanr(p50[valid], y[valid])
                    mlflow.log_metric(f"concat_ic_p50_{hname}", float(r))
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="DLinear Vol-Conditional Quantile v1")
    ap.add_argument("--output-dir", type=str, required=True,
                    help="Output directory for predictions and checkpoints")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Override MBO data directory (default: REPO/data/processed/mbo_events_smart_v3)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mbo_dir = Path(args.data_dir) if args.data_dir else MBO_DIR

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(out_dir / "run.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("dlinear_vol_cond")

    log.info(f"Device: {DEVICE}")
    log.info(f"Output: {out_dir}")
    log.info(f"Data: {mbo_dir}")
    log.info(f"Window={WINDOW}, Batch={BATCH}, LR={LR}, Epochs={N_EPOCHS}")
    log.info(f"Vol: lookback={VOL_LOOKBACK}, buckets={VOL_N_BUCKETS}, embed_dim={VOL_EMBED_DIM}")
    log.info(f"Quantiles: {QUANTILES}, Horizons: {HORIZONS}")

    # MLflow setup
    mlflow_parent_run = None
    if HAVE_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(MLFLOW_EXP)
            mlflow_parent_run = mlflow.start_run(run_name="vol_conditional_v1_walkforward")
            mlflow.log_params(dict(
                window=WINDOW, train_stride=TRAIN_STRIDE, oot_stride=OOT_STRIDE,
                batch=BATCH, n_epochs=N_EPOCHS, lr=LR,
                n_train_days=N_TRAIN_DAYS, n_feat=N_FEAT,
                horizons=str(HORIZONS), quantiles=str(QUANTILES),
                vol_lookback=VOL_LOOKBACK, vol_n_buckets=VOL_N_BUCKETS,
                vol_embed_dim=VOL_EMBED_DIM,
                model="DLinearVolConditionalQuantile",
                device=DEVICE,
            ))
            log.info(f"MLflow: {MLFLOW_URI} experiment={MLFLOW_EXP}")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e}")

    # List available dates
    dates = list_dates(mbo_dir)
    log.info(f"Available dates: {len(dates)} (first={dates[0]}, last={dates[-1]})")
    if len(dates) < N_TRAIN_DAYS + 1:
        log.error("Not enough dates for walk-forward")
        sys.exit(2)

    # Build sliding walk-forward folds
    folds = [(dates[i - N_TRAIN_DAYS:i], dates[i])
             for i in range(N_TRAIN_DAYS, len(dates))]
    log.info(f"Total folds: {len(folds)}")

    # Check for completed folds (resume support)
    completed = set()
    for f in out_dir.glob("fold_*_preds.npz"):
        try:
            d = np.load(f, allow_pickle=True)
            completed.add(str(d["date"]))
        except Exception:
            pass
    if completed:
        log.info(f"Resume: {len(completed)} folds already done")

    t_start = time.time()
    n_params_logged = False

    for fi, (train_dates, test_date) in enumerate(folds, start=1):
        log.info(f"\n=== FOLD {fi}/{len(folds)}  test={test_date}  "
                 f"train={train_dates[0]}..{train_dates[-1]} ===")

        if test_date in completed:
            log.info("  already done, skip")
            continue

        np.random.seed(42 + fi)
        torch.manual_seed(42 + fi)

        # Compute feature normalization stats from training data
        mu, sd = compute_feature_stats(mbo_dir, train_dates)
        log.info(f"  feature stats: |mu|_mean={np.abs(mu).mean():.3f}, sd_mean={sd.mean():.3f}")

        # Compute vol percentile edges from training data
        vol_edges = compute_vol_edges_from_train(mbo_dir, train_dates)
        log.info(f"  vol edges: {vol_edges}")

        # Start MLflow child run for this fold
        mlflow_fold_run = None
        if HAVE_MLFLOW:
            try:
                mlflow_fold_run = mlflow.start_run(
                    run_name=f"fold_{fi:02d}_{test_date}",
                    nested=True,
                )
                mlflow.log_params(dict(
                    fold=fi, test_date=test_date,
                    train_start=train_dates[0], train_end=train_dates[-1],
                ))
            except Exception as e:
                log.warning(f"mlflow fold start failed: {e}")

        # Create model
        t_fold = time.time()
        model = DLinearVolConditionalQuantile().to(DEVICE)
        if not n_params_logged:
            n_params = sum(p.numel() for p in model.parameters())
            log.info(f"  n_params: {n_params / 1e6:.3f}M")
            if HAVE_MLFLOW:
                try:
                    mlflow.log_metric("n_params_M", n_params / 1e6)
                except Exception:
                    pass
            n_params_logged = True

        # Train
        model = train_one_fold(model, mbo_dir, train_dates, mu, sd, vol_edges,
                               out_dir, fi, log)

        # Save model weights
        model_path = out_dir / f"fold_{fi:02d}_model.pt"
        torch.save(model.state_dict(), model_path)

        # Predict on OOT day
        P, Y, diag = predict_day(model, mbo_dir, test_date, mu, sd, vol_edges)
        if P is None or len(P) == 0:
            log.warning(f"  fold {fi} test={test_date}: no predictions — skip")
            if HAVE_MLFLOW and mlflow_fold_run is not None:
                try:
                    mlflow.set_tag("status", "no_predictions")
                    mlflow.end_run()
                except Exception:
                    pass
            continue

        # Save predictions
        pred_path = out_dir / f"fold_{fi:02d}_preds.npz"
        np.savez(
            pred_path,
            preds=P.astype(np.float32),     # (N, n_h, n_q)
            labels=Y.astype(np.float32),     # (N, n_h)
            pred_p10_1s=P[:, 0, 0].astype(np.float32),
            pred_p50_1s=P[:, 0, 1].astype(np.float32),
            pred_p90_1s=P[:, 0, 2].astype(np.float32),
            pred_p10_5s=P[:, 1, 0].astype(np.float32),
            pred_p50_5s=P[:, 1, 1].astype(np.float32),
            pred_p90_5s=P[:, 1, 2].astype(np.float32),
            pred_p10_10s=P[:, 2, 0].astype(np.float32),
            pred_p50_10s=P[:, 2, 1].astype(np.float32),
            pred_p90_10s=P[:, 2, 2].astype(np.float32),
            date=test_date,
            horizons=np.array(HORIZONS),
            quantiles=np.array(QUANTILES, dtype=np.float32),
            vol_edges=vol_edges,
            ic_p50=np.array(diag["ic_p50"], dtype=np.float32),
            coverage_p10=np.array(diag["coverage_p10"], dtype=np.float32),
            coverage_p90=np.array(diag["coverage_p90"], dtype=np.float32),
            width_p10_p90=np.array(diag["width_p10_p90"], dtype=np.float32),
        )

        # Log results
        log.info(f"  fold {fi} test={test_date}  N={len(P):,}  "
                 f"time={time.time()-t_fold:.0f}s")
        for hi, hname in enumerate(HORIZONS):
            log.info(f"    {hname}: IC_P50={diag['ic_p50'][hi]:.4f}  "
                     f"IC_P10={diag['ic_p10'][hi]:.4f}  IC_P90={diag['ic_p90'][hi]:.4f}  "
                     f"cov_P10={diag['coverage_p10'][hi]:.3f}  "
                     f"cov_P90={diag['coverage_p90'][hi]:.3f}  "
                     f"width={diag['width_p10_p90'][hi]:.4f}")

        if HAVE_MLFLOW and mlflow_fold_run is not None:
            try:
                for hi, hname in enumerate(HORIZONS):
                    mlflow.log_metric(f"ic_p50_{hname}", diag["ic_p50"][hi])
                    mlflow.log_metric(f"ic_p10_{hname}", diag["ic_p10"][hi])
                    mlflow.log_metric(f"ic_p90_{hname}", diag["ic_p90"][hi])
                    mlflow.log_metric(f"coverage_p10_{hname}", diag["coverage_p10"][hi])
                    mlflow.log_metric(f"coverage_p90_{hname}", diag["coverage_p90"][hi])
                    mlflow.log_metric(f"width_p10_p90_{hname}", diag["width_p10_p90"][hi])
                mlflow.end_run()
            except Exception as e:
                log.warning(f"mlflow log/end failed: {e}")

        # Clean up intra-fold epoch checkpoints to save disk
        for ep_ckpt in out_dir.glob(f"fold_{fi:02d}_epoch_*.pt"):
            try:
                ep_ckpt.unlink()
            except Exception:
                pass

    # Compute concat IC across all folds
    compute_concat_ic(out_dir, log)

    elapsed_min = (time.time() - t_start) / 60.0
    log.info(f"\n=== ALL DONE in {elapsed_min:.1f} min ===")

    # End parent MLflow run
    if HAVE_MLFLOW and mlflow_parent_run is not None:
        try:
            mlflow.log_metric("total_minutes", elapsed_min)
            mlflow.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    main()
