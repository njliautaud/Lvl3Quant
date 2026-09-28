#!/usr/bin/env python3
"""
HC #488 — Multi-Window DLinear Quantile v1.

PURPOSE
-------
The single-window DLinear quantile (W=500) produces IC_P50 ~0.26/0.20/0.14 at
1s/5s/10s — matching CNN-Mamba v2. Problem: passive fills get adversely selected.
We need STRONGER signal to overcome execution costs.

IDEA: Use three temporal windows simultaneously:
  W=100: fast microstructure (queue flips, immediate pressure)
  W=300: medium dynamics (trend formation)
  W=500: full context (proven window)

Each window captures different dynamics. A fusion head learns to combine them.

ARCHITECTURE (MultiWindowDLinearQuantile):
  - Three DLinear trunks (one per window), each producing 128-dim output
  - Concatenate all trunk outputs -> 384-dim -> GELU -> Linear(384,128) -> GELU
    -> Linear(128, n_h * n_q) where n_h=3 horizons, n_q=3 quantiles
  - Same pinball loss as single-window quantile
  - ~9.6M params total

DATA: NPZ files at data/processed/mbo_events_smart_v3/{YYYYMMDD}_mbo_events.npz
  25 features, labels_1s/5s/10s.

WINDOWING: For each sample at position `start` (indexing the W=500 window):
  - W=500: events[start : start+500]
  - W=300: events[start+200 : start+500]
  - W=100: events[start+400 : start+500]
  All three windows end at the same event, so the label is at start+499.

TRAINING: Sliding 10-day train / 1-day OOT walk-forward. Pinball loss.
OUTPUT: Per-fold NPZ with P10/P50/P90 per horizon, IC diagnostics.
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path
from typing import List, Tuple

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
OUT_DIR = REPO / "output" / "hc488_dlinear_multiwindow_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT = OUT_DIR / "intra_ckpt.pt"

WINDOWS = [100, 300, 500]          # multi-window sizes
MAX_WINDOW = max(WINDOWS)          # 500 — governs sample extraction
TRAIN_STRIDE = 25
OOT_STRIDE = 5
BATCH = 256
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
MLFLOW_EXP = "hc488_dlinear_multiwindow_v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(OUT_DIR / "run.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("hc488_mw")


# ────────────────────────────────────────────────────────────────────────────
# Model — Multi-Window DLinear Quantile
# ────────────────────────────────────────────────────────────────────────────
class DLinearTrunk(nn.Module):
    """Single DLinear trunk: trend/seasonal decomp -> 128-dim embedding."""
    def __init__(self, window: int, n_feat: int = N_FEAT, kernel: int = 25):
        super().__init__()
        self.window = window
        k = min(kernel, window)  # kernel can't exceed window
        self.avg = nn.AvgPool1d(kernel_size=k, stride=1, padding=k // 2)
        self.lin_trend = nn.Linear(window * n_feat, 128)
        self.lin_season = nn.Linear(window * n_feat, 128)

    def forward(self, x):  # x: (B, W, F)
        B, W, F_ = x.shape
        x_t = x.transpose(1, 2)  # (B, F, W)
        trend = self.avg(x_t)
        if trend.shape[-1] > W:
            trend = trend[..., :W]
        season = x_t - trend
        flat_t = trend.flatten(1)
        flat_s = season.flatten(1)
        # Concatenate trend and seasonal linear projections -> 128-dim each
        h = torch.cat([self.lin_trend(flat_t), self.lin_season(flat_s)], dim=1)
        return h  # (B, 256) — but we'll only use 128 per branch below


class MultiWindowDLinearQuantile(nn.Module):
    """
    Three DLinear trunks (W=100, W=300, W=500), each producing 128-dim.
    Fusion head: concat(3 x 128) = 384 -> GELU -> 128 -> GELU -> n_h * n_q.
    """
    def __init__(self, windows=None, n_feat: int = N_FEAT,
                 n_horizons: int = N_HORIZONS, n_quantiles: int = N_QUANTILES,
                 kernel: int = 25):
        super().__init__()
        if windows is None:
            windows = WINDOWS
        self.windows = windows
        self.n_h = n_horizons
        self.n_q = n_quantiles

        # One trunk per window. Each trunk outputs 256-dim (128 trend + 128 season)
        # We project each trunk's 256-dim to 128-dim for the fusion.
        self.trunks = nn.ModuleList()
        self.trunk_projectors = nn.ModuleList()
        for w in windows:
            self.trunks.append(DLinearTrunk(window=w, n_feat=n_feat, kernel=kernel))
            self.trunk_projectors.append(nn.Sequential(
                nn.GELU(),
                nn.Linear(256, 128),
            ))

        # Fusion head: 128 * n_trunks -> n_h * n_q
        trunk_out_dim = 128 * len(windows)  # 384
        self.fusion = nn.Sequential(
            nn.GELU(),
            nn.Linear(trunk_out_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_horizons * n_quantiles),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: list of 3 tensors, each (B, W_i, F) for W_i in [100, 300, 500]
        Returns:
            (B, n_h * n_q) quantile predictions
        """
        trunk_outs = []
        for trunk, proj, x in zip(self.trunks, self.trunk_projectors, inputs):
            h = trunk(x)      # (B, 256)
            h = proj(h)       # (B, 128)
            trunk_outs.append(h)
        fused = torch.cat(trunk_outs, dim=1)  # (B, 384)
        return self.fusion(fused)              # (B, n_h * n_q)


# ────────────────────────────────────────────────────────────────────────────
# Pinball Loss
# ────────────────────────────────────────────────────────────────────────────
class PinballLoss(nn.Module):
    """Quantile (pinball) loss for multiple horizons and quantiles."""
    def __init__(self, quantiles=None, n_horizons: int = N_HORIZONS):
        super().__init__()
        if quantiles is None:
            quantiles = QUANTILES
        # Register as buffer so it moves with .to(device)
        self.register_buffer(
            "tau",
            torch.tensor(quantiles, dtype=torch.float32)
                 .unsqueeze(0).unsqueeze(0)   # (1, 1, n_q)
        )
        self.n_h = n_horizons
        self.n_q = len(quantiles)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   (B, n_h * n_q) — flattened quantile predictions
        target: (B, n_h)       — actual labels per horizon
        """
        B = pred.shape[0]
        pred = pred.view(B, self.n_h, self.n_q)       # (B, n_h, n_q)
        target = target.unsqueeze(2)                    # (B, n_h, 1)
        error = target - pred                           # (B, n_h, n_q)
        loss = torch.where(
            error >= 0,
            self.tau * error,
            (self.tau - 1.0) * error,
        )
        return loss.mean()


# ────────────────────────────────────────────────────────────────────────────
# Data
# ────────────────────────────────────────────────────────────────────────────
def list_dates() -> List[str]:
    files = sorted(MBO_DIR.glob("*_mbo_events.npz"))
    return [f.name.split("_")[0] for f in files]


def load_day(date: str):
    f = MBO_DIR / f"{date}_mbo_events.npz"
    d = np.load(f, allow_pickle=True)
    ev = d["events"].astype(np.float32)
    l1 = d["labels_1s"].astype(np.float32)
    l5 = d["labels_5s"].astype(np.float32)
    l10 = d["labels_10s"].astype(np.float32)
    return ev, l1, l5, l10


def valid_starts(n: int, window: int, stride: int,
                 l1: np.ndarray, l5: np.ndarray, l10: np.ndarray) -> np.ndarray:
    """Return valid start indices for the max window, where labels are non-NaN."""
    starts = np.arange(0, n - window + 1, stride)
    end_idx = starts + window - 1
    mask = ~(np.isnan(l1[end_idx]) | np.isnan(l5[end_idx]) | np.isnan(l10[end_idx]))
    return starts[mask]


def extract_multi_windows(ev: np.ndarray, starts: np.ndarray,
                          windows: List[int], max_window: int) -> List[np.ndarray]:
    """
    For each start position (indexing the max_window block), extract sub-windows
    that all end at start + max_window - 1.

    Returns list of arrays, one per window size. Each is (N, W_i, F).
    """
    results = []
    for w in windows:
        offset = max_window - w  # how many events to skip from start
        chunks = np.stack([ev[s + offset: s + offset + w] for s in starts])
        results.append(chunks)
    return results


# ────────────────────────────────────────────────────────────────────────────
# Feature normalization
# ────────────────────────────────────────────────────────────────────────────
def compute_feature_stats(train_dates: List[str], sample_frac: float = 0.05):
    mu = np.zeros(N_FEAT, dtype=np.float64)
    sd = np.zeros(N_FEAT, dtype=np.float64)
    n_total = 0
    for date in train_dates:
        try:
            ev, *_ = load_day(date)
        except Exception:
            continue
        n_take = max(int(len(ev) * sample_frac), min(1000, len(ev)))
        idx = np.random.choice(len(ev), n_take, replace=False)
        sub = ev[idx]
        mu += sub.sum(0)
        sd += (sub ** 2).sum(0)
        n_total += len(sub)
    mu /= max(n_total, 1)
    var = sd / max(n_total, 1) - mu ** 2
    sd = np.sqrt(np.clip(var, 1e-12, None))
    return mu.astype(np.float32), sd.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Train
# ────────────────────────────────────────────────────────────────────────────
def precompute_day_windows(ev: np.ndarray, starts: np.ndarray,
                           mu: np.ndarray, sd: np.ndarray,
                           l1: np.ndarray, l5: np.ndarray,
                           l10: np.ndarray) -> Tuple:
    """Pre-extract + normalize all windows and labels for a day.
    Returns (list_of_window_arrays, labels) ready for batching."""
    mw_all = extract_multi_windows(ev, starts, WINDOWS, MAX_WINDOW)
    # Normalize in-place
    mw_normed = [(w - mu) / sd for w in mw_all]
    end = starts + MAX_WINDOW - 1
    yb = np.stack([l1[end], l5[end], l10[end]], axis=1).astype(np.float32)
    return mw_normed, yb


def train_one_fold(model: nn.Module, train_dates: List[str],
                   mu: np.ndarray, sd: np.ndarray, n_epochs: int = N_EPOCHS):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = PinballLoss().to(DEVICE)

    # Pre-load all days once to avoid repeated NPZ reads
    day_cache = []
    for date in train_dates:
        try:
            ev, l1, l5, l10 = load_day(date)
        except Exception as e:
            log.warning(f"  load fail {date}: {e}")
            continue
        starts = valid_starts(len(ev), MAX_WINDOW, TRAIN_STRIDE, l1, l5, l10)
        if len(starts) < BATCH:
            continue
        mw_normed, yb = precompute_day_windows(ev, starts, mu, sd, l1, l5, l10)
        day_cache.append((starts, mw_normed, yb))
    log.info(f"    pre-loaded {len(day_cache)} days into memory")

    for ep in range(n_epochs):
        ep_loss, n_batches = 0.0, 0
        for starts, mw_normed, yb in day_cache:
            perm = np.random.permutation(len(starts))

            for i in range(0, len(perm), BATCH):
                idx = perm[i:i + BATCH]
                if len(idx) < 2:
                    continue

                mw_tensors = [torch.from_numpy(w[idx]).to(DEVICE, non_blocking=True)
                              for w in mw_normed]
                yb_t = torch.from_numpy(yb[idx]).to(DEVICE, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                pred = model(mw_tensors)  # (B, n_h * n_q)
                loss = criterion(pred, yb_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss.item())
                n_batches += 1

        avg_loss = ep_loss / max(n_batches, 1)
        log.info(f"    epoch {ep+1}/{n_epochs} avg_pinball={avg_loss:.6f} "
                 f"batches={n_batches}")
        if HAVE_MLFLOW:
            try:
                mlflow.log_metric("train_pinball", avg_loss, step=ep)
            except Exception:
                pass
    return model


# ────────────────────────────────────────────────────────────────────────────
# Predict
# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_day(model: nn.Module, date: str,
                mu: np.ndarray, sd: np.ndarray):
    """
    Returns:
        P: (N, n_h, n_q) quantile predictions
        Y: (N, n_h) actual labels
        diag: dict with IC, coverage, width metrics
    """
    model.eval()
    ev, l1, l5, l10 = load_day(date)
    starts = valid_starts(len(ev), MAX_WINDOW, OOT_STRIDE, l1, l5, l10)
    if len(starts) == 0:
        return None, None, None

    preds_chunks, labs_chunks = [], []
    for i in range(0, len(starts), BATCH):
        bs = starts[i:i + BATCH]
        mw = extract_multi_windows(ev, bs, WINDOWS, MAX_WINDOW)
        mw_normed = [(w - mu) / sd for w in mw]
        mw_tensors = [torch.from_numpy(w).to(DEVICE, non_blocking=True)
                      for w in mw_normed]

        end = bs + MAX_WINDOW - 1
        yb = np.stack([l1[end], l5[end], l10[end]], axis=1)

        out = model(mw_tensors).cpu().numpy()  # (B, n_h * n_q)
        preds_chunks.append(out)
        labs_chunks.append(yb)

    P_flat = np.concatenate(preds_chunks, 0)  # (N, n_h * n_q)
    Y = np.concatenate(labs_chunks, 0)         # (N, n_h)
    N_samples = P_flat.shape[0]
    P = P_flat.reshape(N_samples, N_HORIZONS, N_QUANTILES)  # (N, n_h, n_q)

    # Diagnostics
    from scipy.stats import spearmanr
    diag = {}
    for hi, hname in enumerate(HORIZONS):
        p50 = P[:, hi, 1]  # median quantile
        p10 = P[:, hi, 0]
        p90 = P[:, hi, 2]
        y = Y[:, hi]
        v = ~(np.isnan(p50) | np.isnan(y))

        # IC on P50 (median prediction)
        if v.sum() >= 50:
            ic, _ = spearmanr(p50[v], y[v])
        else:
            ic = float("nan")

        # Coverage: fraction of y in [P10, P90]
        width = p90 - p10
        in_interval = (y >= p10) & (y <= p90)
        coverage = float(in_interval[v].mean()) if v.sum() > 0 else float("nan")

        # Mean width
        mean_width = float(width[v].mean()) if v.sum() > 0 else float("nan")

        # IC of width vs |y| — does the model know when it's uncertain?
        abs_y = np.abs(y)
        if v.sum() >= 50:
            ic_width, _ = spearmanr(width[v], abs_y[v])
        else:
            ic_width = float("nan")

        diag[hname] = {
            "ic_p50": float(ic) if not np.isnan(ic) else 0.0,
            "coverage_80": float(coverage),
            "mean_width": float(mean_width),
            "ic_width_vs_abs_y": float(ic_width) if not np.isnan(ic_width) else 0.0,
        }

    return P, Y, diag


# ────────────────────────────────────────────────────────────────────────────
# Main — sliding walk-forward
# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="Resume from intra_ckpt.pt (skip completed folds)")
    ap.add_argument("--max-folds", type=int, default=23,
                    help="Maximum number of folds to run")
    args = ap.parse_args()

    log.info(f"Device: {DEVICE}")
    log.info(f"Windows: {WINDOWS}")
    log.info(f"torch.get_num_threads() = {torch.get_num_threads()}")

    if HAVE_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(MLFLOW_EXP)
            log.info(f"MLflow: {MLFLOW_URI} experiment={MLFLOW_EXP}")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e}")

    dates = list_dates()
    log.info(f"Available dates: {len(dates)} (first={dates[0]}, last={dates[-1]})")
    if len(dates) < N_TRAIN_DAYS + 1:
        log.error("Not enough dates for walk-forward")
        sys.exit(2)

    # Build folds: sliding 10-day train, 1-day OOT
    folds = [(dates[i - N_TRAIN_DAYS:i], dates[i])
             for i in range(N_TRAIN_DAYS, len(dates))]
    if args.max_folds < len(folds):
        folds = folds[:args.max_folds]
    log.info(f"Folds: {len(folds)} (test dates: {[f[1] for f in folds]})")

    # Resume support
    completed = set()
    if args.resume and CKPT.exists():
        try:
            cdata = torch.load(CKPT, map_location="cpu", weights_only=False)
            completed = set(cdata.get("completed", []))
            log.info(f"Resume: {len(completed)} folds already done")
        except Exception as e:
            log.warning(f"Resume load failed: {e}")

    # Aggregate IC tracker for concat IC computation
    all_p50_preds = {h: [] for h in HORIZONS}
    all_labels = {h: [] for h in HORIZONS}

    t_start = time.time()
    n_done = 0

    for fi, (train_dates, test_date) in enumerate(folds, start=1):
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fi}/{len(folds)}  test={test_date}  "
                 f"train={train_dates[0]}..{train_dates[-1]}")
        log.info(f"{'='*60}")

        if test_date in completed:
            log.info("  Already done (resume), skipping")
            # Still load predictions for concat IC if they exist
            pred_path = OUT_DIR / f"fold_{fi:02d}_preds.npz"
            if pred_path.exists():
                try:
                    saved = np.load(pred_path, allow_pickle=True)
                    for hi, hname in enumerate(HORIZONS):
                        all_p50_preds[hname].append(saved[f"p50_{hname}"])
                        all_labels[hname].append(saved[f"labels_{hname}"])
                except Exception:
                    pass
            continue

        np.random.seed(42 + fi)
        torch.manual_seed(42 + fi)

        # Feature normalization from train dates
        mu, sd = compute_feature_stats(train_dates)
        log.info(f"  Feature stats: |mu|_mean={np.abs(mu).mean():.4f}, "
                 f"sd_mean={sd.mean():.4f}")

        # MLflow run
        mlflow_run = None
        if HAVE_MLFLOW:
            try:
                mlflow_run = mlflow.start_run(
                    run_name=f"fold_{fi:02d}_{test_date}")
                mlflow.log_params(dict(
                    fold=fi, test_date=test_date,
                    windows=str(WINDOWS), train_stride=TRAIN_STRIDE,
                    oot_stride=OOT_STRIDE, batch=BATCH,
                    n_epochs=N_EPOCHS, lr=LR, n_train_days=N_TRAIN_DAYS,
                    horizons=str(HORIZONS), quantiles=str(QUANTILES),
                    model="MultiWindowDLinearQuantile",
                ))
            except Exception as e:
                log.warning(f"MLflow start_run failed: {e}")

        # Build and train model
        t_m = time.time()
        model = MultiWindowDLinearQuantile().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        log.info(f"  Model params: {n_params:,} ({n_params/1e6:.2f}M)")
        model = train_one_fold(model, train_dates, mu, sd)

        # Predict OOT
        P, Y, diag = predict_day(model, test_date, mu, sd)
        if P is None or len(P) == 0:
            log.warning(f"  Fold {fi} test={test_date}: no predictions")
            completed.add(test_date)
            torch.save({"completed": list(completed)}, CKPT)
            if HAVE_MLFLOW and mlflow_run is not None:
                try:
                    mlflow.set_tag("status", "no_predictions")
                    mlflow.end_run()
                except Exception:
                    pass
            continue

        # Save per-fold predictions with P10/P50/P90 split arrays
        save_dict = dict(date=test_date, horizons=np.array(HORIZONS))
        for hi, hname in enumerate(HORIZONS):
            save_dict[f"p10_{hname}"] = P[:, hi, 0].astype(np.float32)
            save_dict[f"p50_{hname}"] = P[:, hi, 1].astype(np.float32)
            save_dict[f"p90_{hname}"] = P[:, hi, 2].astype(np.float32)
            save_dict[f"labels_{hname}"] = Y[:, hi].astype(np.float32)
            # Track for concat IC
            all_p50_preds[hname].append(P[:, hi, 1])
            all_labels[hname].append(Y[:, hi])

        out_path = OUT_DIR / f"fold_{fi:02d}_preds.npz"
        np.savez(out_path, **save_dict)

        # Save model weights
        torch.save(model.state_dict(), OUT_DIR / f"fold_{fi:02d}_model.pt")

        # Log diagnostics
        log.info(f"  Fold {fi} test={test_date}  N={len(P):,}  "
                 f"time={time.time()-t_m:.1f}s")
        for hname in HORIZONS:
            d = diag[hname]
            log.info(f"    {hname}: IC_P50={d['ic_p50']:.4f}  "
                     f"coverage_80={d['coverage_80']:.3f}  "
                     f"mean_width={d['mean_width']:.5f}  "
                     f"IC_width_vs_|y|={d['ic_width_vs_abs_y']:.4f}")

        # MLflow metrics
        if HAVE_MLFLOW and mlflow_run is not None:
            try:
                for hname in HORIZONS:
                    d = diag[hname]
                    mlflow.log_metric(f"ic_p50_{hname}", d["ic_p50"])
                    mlflow.log_metric(f"coverage_80_{hname}", d["coverage_80"])
                    mlflow.log_metric(f"mean_width_{hname}", d["mean_width"])
                    mlflow.log_metric(f"ic_width_vs_abs_y_{hname}",
                                      d["ic_width_vs_abs_y"])
                mlflow.log_metric("n_samples", len(P))
                mlflow.end_run()
            except Exception as e:
                log.warning(f"MLflow log/end failed: {e}")

        completed.add(test_date)
        torch.save({"completed": list(completed)}, CKPT)
        n_done += 1

    # ── Concat IC across all folds ─────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"CONCAT IC (all {n_done} new + {len(completed)-n_done} resumed folds)")
    log.info(f"{'='*60}")

    from scipy.stats import spearmanr
    concat_ic = {}
    for hname in HORIZONS:
        if len(all_p50_preds[hname]) == 0:
            concat_ic[hname] = float("nan")
            continue
        all_p = np.concatenate(all_p50_preds[hname])
        all_y = np.concatenate(all_labels[hname])
        v = ~(np.isnan(all_p) | np.isnan(all_y))
        if v.sum() >= 50:
            ic, _ = spearmanr(all_p[v], all_y[v])
            concat_ic[hname] = float(ic)
        else:
            concat_ic[hname] = float("nan")
        log.info(f"  {hname}: concat_IC_P50 = {concat_ic[hname]:.4f}  "
                 f"(N={v.sum():,})")

    # Save summary
    summary = {
        "concat_ic": concat_ic,
        "n_folds": len(completed),
        "windows": WINDOWS,
        "total_time_min": (time.time() - t_start) / 60,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Log concat IC to MLflow as a separate summary run
    if HAVE_MLFLOW:
        try:
            with mlflow.start_run(run_name="concat_summary"):
                for hname in HORIZONS:
                    mlflow.log_metric(f"concat_ic_p50_{hname}",
                                      concat_ic.get(hname, 0.0))
                mlflow.log_params(dict(
                    windows=str(WINDOWS),
                    n_folds=len(completed),
                    model="MultiWindowDLinearQuantile",
                ))
        except Exception:
            pass

    log.info(f"\nALL DONE in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
