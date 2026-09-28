#!/usr/bin/env python3
"""
DLinear predicting FIFO-realized net ticks directly.

EXECUTION AXIS experiment (HC #488 R2 rotation from dead meta-classifier axis).

KEY INSIGHT: All proxy-label models (log_ret) show good IC but die in FIFO replay
because passive fills are adversely selected. This model skips the proxy entirely
and learns to predict FIFO-REALIZED net_ticks from microstructure features.

Labels come from fifo_label_generator_v3.py canonical replay engine.
- tp4sl3_short_net_ticks: FIFO net P&L for short at TP=4/SL=3 (0 if unfilled)
- tp4sl3_long_net_ticks: same for long side

Architecture: same DLinear trunk (trend/seasonal decomp) as HC #488 quantile v1.
Training: sliding 10-day train / 1-day OOT. Loss = MSE on FIFO net_ticks.
Window alignment: FIFO labels at stride=250, window=3000 -> decision event at
  k*250 + 2999. Feature window is last 500 events ending at decision event.

If the model can predict FIFO net_ticks with positive IC, it learns BOTH
fill probability and conditional profitability in one shot.

Inputs: mbo_events_smart_v3 (features) + mbo_events_smart_v3_fifo_labels (labels)
Output: per-fold OOT predictions + IC metrics
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

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
FIFO_DIR = REPO / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
OUT_DIR = REPO / "output" / "dlinear_fifo_truth_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_WINDOW = 500       # feature window size (same as DLinear quantile)
FIFO_WINDOW = 3000      # FIFO replay window size
FIFO_STRIDE = 250       # FIFO replay stride
TRAIN_STRIDE = 1        # use every FIFO window for training (they're already sparse)
BATCH = 256
N_FEAT = 25
N_TRAIN_DAYS = 10
N_EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MLFLOW_URI = "http://jupiter:5000"
MLFLOW_EXP = "dlinear_fifo_truth_v1"

# Which FIFO label configs to predict
LABEL_CONFIGS = [
    "tp4sl3_short_net_ticks",
    "tp4sl3_long_net_ticks",
    "tp8sl5_short_net_ticks",
    "tp8sl5_long_net_ticks",
]
N_TARGETS = len(LABEL_CONFIGS)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(OUT_DIR / "run.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("fifo_truth")


# ────────────────────────────────────────────────────────────────────────────
# Model — same DLinear trunk, output = N_TARGETS FIFO net_ticks predictions
# ────────────────────────────────────────────────────────────────────────────
class DLinearFIFO(nn.Module):
    def __init__(self, window=FEAT_WINDOW, n_feat=N_FEAT,
                 n_targets=N_TARGETS, kernel=25):
        super().__init__()
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=kernel // 2)
        self.lin_trend = nn.Linear(window * n_feat, 128)
        self.lin_season = nn.Linear(window * n_feat, 128)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, n_targets),
        )

    def forward(self, x):  # x: (B, W, F)
        B, W, F_ = x.shape
        x_t = x.transpose(1, 2)
        trend = self.avg(x_t)
        if trend.shape[-1] > W:
            trend = trend[..., :W]
        season = x_t - trend
        flat_t = trend.flatten(1)
        flat_s = season.flatten(1)
        h = torch.cat([self.lin_trend(flat_t), self.lin_season(flat_s)], dim=1)
        return self.head(h)  # (B, n_targets)


# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────
def list_paired_dates() -> List[str]:
    """Return dates that have BOTH smart_v3 events AND FIFO labels."""
    ev_dates = {f.name.split("_")[0] for f in MBO_DIR.glob("*_mbo_events.npz")}
    fi_dates = {f.name.split("_")[0] for f in FIFO_DIR.glob("*_fifo_labels.npz")}
    both = sorted(ev_dates & fi_dates)
    return both


def load_day_aligned(date: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load features + FIFO labels for one day, aligned at FIFO decision points.

    Returns:
        X: (N, FEAT_WINDOW, N_FEAT) feature windows
        Y: (N, N_TARGETS) FIFO net_ticks labels
    """
    ev_path = MBO_DIR / f"{date}_mbo_events.npz"
    fi_path = FIFO_DIR / f"{date}_fifo_labels.npz"

    ev_data = np.load(ev_path, allow_pickle=True)
    fi_data = np.load(fi_path, allow_pickle=True)

    events = ev_data["events"].astype(np.float32)
    n_events = len(events)

    window_ks = fi_data["window_k"]

    # Build labels array
    labels = np.stack([fi_data[cfg].astype(np.float32) for cfg in LABEL_CONFIGS], axis=1)  # (N_fifo, N_TARGETS)

    # Compute decision event index for each FIFO window
    # Decision point = end of the 3000-event window = k * FIFO_STRIDE + FIFO_WINDOW - 1
    decision_idx = window_ks * FIFO_STRIDE + FIFO_WINDOW - 1

    # Feature window: last FEAT_WINDOW events ending at decision point
    feat_start = decision_idx - FEAT_WINDOW + 1

    # Filter: need feat_start >= 0 and decision_idx < n_events
    valid = (feat_start >= 0) & (decision_idx < n_events)

    if valid.sum() == 0:
        return None

    feat_start = feat_start[valid]
    labels = labels[valid]

    # Build feature windows
    X = np.stack([events[s:s + FEAT_WINDOW] for s in feat_start])  # (N, W, F)

    # Filter NaN labels
    nan_mask = np.any(np.isnan(labels), axis=1)
    if nan_mask.all():
        return None
    X = X[~nan_mask]
    labels = labels[~nan_mask]

    return X, labels


def compute_normalization(dates: List[str]):
    """Compute mean/std from training dates (per-feature, collapsed over window)."""
    running_sum = np.zeros(N_FEAT, dtype=np.float64)
    running_sq = np.zeros(N_FEAT, dtype=np.float64)
    n = 0
    for date in dates:
        try:
            ev_data = np.load(MBO_DIR / f"{date}_mbo_events.npz", allow_pickle=True)
            events = ev_data["events"].astype(np.float64)
            running_sum += events.sum(axis=0)
            running_sq += (events ** 2).sum(axis=0)
            n += len(events)
        except Exception as e:
            log.warning(f"  norm skip {date}: {e}")
            continue
    mu = running_sum / max(n, 1)
    var = running_sq / max(n, 1) - mu ** 2
    sd = np.sqrt(np.maximum(var, 1e-12))
    return mu.astype(np.float32), sd.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Train
# ────────────────────────────────────────────────────────────────────────────
def train_one_fold(model, train_dates, mu, sd, n_epochs=N_EPOCHS):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    mse = nn.MSELoss()

    for ep in range(n_epochs):
        ep_loss, n_batches = 0.0, 0
        np.random.shuffle(train_dates)  # shuffle date order each epoch
        for date in train_dates:
            try:
                result = load_day_aligned(date)
                if result is None:
                    continue
                X, Y = result
            except Exception as e:
                log.warning(f"  load fail {date}: {e}")
                continue

            if len(X) < BATCH:
                continue

            # Normalize features
            X = (X - mu) / sd

            # Shuffle
            perm = np.random.permutation(len(X))
            X = X[perm]
            Y = Y[perm]

            for i in range(0, len(X), BATCH):
                xb = X[i:i + BATCH]
                yb = Y[i:i + BATCH]
                if len(xb) < 2:
                    continue

                xb_t = torch.from_numpy(xb).to(DEVICE, non_blocking=True)
                yb_t = torch.from_numpy(yb).to(DEVICE, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                pred = model(xb_t)
                loss = mse(pred, yb_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss.item())
                n_batches += 1

        sched.step()
        avg_loss = ep_loss / max(n_batches, 1)
        log.info(f"    epoch {ep+1}/{n_epochs} avg_mse={avg_loss:.6f} batches={n_batches}")
        if HAVE_MLFLOW:
            try:
                mlflow.log_metric("train_mse", avg_loss, step=ep)
            except Exception:
                pass
    return model


@torch.no_grad()
def predict_day(model, date, mu, sd):
    """Predict FIFO net_ticks for one OOT day."""
    model.eval()
    result = load_day_aligned(date)
    if result is None:
        return None, None
    X, Y = result
    X = (X - mu) / sd

    preds_chunks = []
    for i in range(0, len(X), BATCH):
        xb = X[i:i + BATCH]
        xb_t = torch.from_numpy(xb).to(DEVICE, non_blocking=True)
        preds_chunks.append(model(xb_t).cpu().numpy())

    P = np.concatenate(preds_chunks, 0)  # (N, N_TARGETS)
    return P, Y


def compute_metrics(P, Y, fold_idx, date):
    """Compute IC (Spearman) and profitability metrics per label config."""
    metrics = {}
    for i, cfg in enumerate(LABEL_CONFIGS):
        p, y = P[:, i], Y[:, i]

        # IC on all events
        ic_all, _ = spearmanr(p, y)

        # IC on filled events only (y != 0)
        filled = y != 0
        if filled.sum() > 10:
            ic_filled, _ = spearmanr(p[filled], y[filled])
        else:
            ic_filled = float('nan')

        # Fill rate in data
        fill_rate = filled.mean()

        # Mean actual net_ticks for top 10% predicted events
        top10_thr = np.percentile(p, 90) if "long" in cfg else np.percentile(p, 10)
        if "long" in cfg:
            top10 = p >= top10_thr
        else:
            top10 = p <= top10_thr  # for short, lower prediction = more short-confident

        top10_actual = y[top10].mean() if top10.sum() > 0 else float('nan')

        # Mean actual for bottom 10% (opposite side)
        if "long" in cfg:
            bot10 = p <= np.percentile(p, 10)
        else:
            bot10 = p >= np.percentile(p, 90)
        bot10_actual = y[bot10].mean() if bot10.sum() > 0 else float('nan')

        metrics[cfg] = {
            "ic_all": float(ic_all) if not np.isnan(ic_all) else 0.0,
            "ic_filled": float(ic_filled) if not np.isnan(ic_filled) else 0.0,
            "fill_rate": float(fill_rate),
            "n_events": int(len(y)),
            "n_filled": int(filled.sum()),
            "top10_actual_net_ticks": float(top10_actual),
            "bot10_actual_net_ticks": float(bot10_actual),
        }

        log.info(f"    {cfg}: IC_all={metrics[cfg]['ic_all']:.4f} "
                 f"IC_filled={metrics[cfg]['ic_filled']:.4f} "
                 f"fill_rate={metrics[cfg]['fill_rate']:.3f} "
                 f"top10_net={metrics[cfg]['top10_actual_net_ticks']:.3f}")

    return metrics


# ────────────────────────────────────────────────────────────────────────────
# Walk-forward main loop
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-folds", type=int, default=999)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target-only", nargs="*", default=None,
                        help="Only test these dates")
    args = parser.parse_args()

    dates = list_paired_dates()
    log.info(f"Found {len(dates)} paired dates (events + FIFO labels)")

    if len(dates) < N_TRAIN_DAYS + 1:
        log.error(f"Need at least {N_TRAIN_DAYS + 1} dates, got {len(dates)}")
        return

    # MLflow
    if HAVE_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(MLFLOW_EXP)
        except Exception as e:
            log.warning(f"MLflow setup fail: {e}")

    # Determine folds to skip (resume mode)
    done_folds = set()
    if args.resume:
        for p in OUT_DIR.glob("fold_*_preds.npz"):
            try:
                idx = int(p.name.split("_")[1])
                done_folds.add(idx)
            except Exception:
                pass
        log.info(f"Resume mode: skipping {len(done_folds)} completed folds")

    all_metrics = []
    fold_count = 0

    for i in range(N_TRAIN_DAYS, len(dates)):
        if fold_count >= args.max_folds:
            break

        test_date = dates[i]
        if args.target_only and test_date not in args.target_only:
            continue

        fold_idx = i - N_TRAIN_DAYS
        if fold_idx in done_folds:
            log.info(f"Fold {fold_idx} ({test_date}) already done, skipping")
            continue

        train_dates = dates[i - N_TRAIN_DAYS:i]

        log.info(f"{'='*60}")
        log.info(f"FOLD {fold_idx} | train {train_dates[0]}->{train_dates[-1]} | OOT {test_date}")
        log.info(f"{'='*60}")

        # MLflow run
        mlflow_run = None
        if HAVE_MLFLOW:
            try:
                mlflow_run = mlflow.start_run(run_name=f"fold_{fold_idx}_{test_date}")
                mlflow.log_params({
                    "fold_idx": fold_idx,
                    "test_date": test_date,
                    "n_train_days": N_TRAIN_DAYS,
                    "n_epochs": N_EPOCHS,
                    "feat_window": FEAT_WINDOW,
                    "lr": LR,
                    "label_configs": ",".join(LABEL_CONFIGS),
                })
            except Exception:
                pass

        # Normalization from train dates
        log.info("  Computing feature normalization...")
        mu, sd = compute_normalization(train_dates)

        # Train
        model = DLinearFIFO().to(DEVICE)
        log.info(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
        log.info(f"  Training on {len(train_dates)} dates...")
        model = train_one_fold(model, list(train_dates), mu, sd)

        # Predict OOT
        log.info(f"  Predicting OOT {test_date}...")
        P, Y = predict_day(model, test_date, mu, sd)

        if P is None:
            log.warning(f"  No valid data for {test_date}, skipping")
            if mlflow_run:
                try: mlflow.end_run()
                except: pass
            continue

        # Metrics
        metrics = compute_metrics(P, Y, fold_idx, test_date)
        metrics["fold_idx"] = fold_idx
        metrics["test_date"] = test_date
        all_metrics.append(metrics)

        # Log to MLflow
        if HAVE_MLFLOW:
            try:
                for cfg in LABEL_CONFIGS:
                    for k, v in metrics[cfg].items():
                        mlflow.log_metric(f"{cfg}_{k}", v)
            except Exception:
                pass

        # Save predictions
        save_dict = {
            "predictions": P,
            "labels": Y,
            "label_configs": np.array(LABEL_CONFIGS),
            "test_date": np.array(test_date),
            "mu": mu,
            "sd": sd,
        }
        pred_path = OUT_DIR / f"fold_{fold_idx:02d}_preds.npz"
        np.savez_compressed(pred_path, **save_dict)
        log.info(f"  Saved predictions to {pred_path.name}")

        # Save model
        torch.save(model.state_dict(), OUT_DIR / f"fold_{fold_idx:02d}_model.pt")

        if mlflow_run:
            try: mlflow.end_run()
            except: pass

        fold_count += 1

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"SUMMARY — {fold_count} folds completed")
    log.info(f"{'='*60}")

    if all_metrics:
        summary = {}
        for cfg in LABEL_CONFIGS:
            ics = [m[cfg]["ic_all"] for m in all_metrics if cfg in m]
            ics_filled = [m[cfg]["ic_filled"] for m in all_metrics if cfg in m]
            top10s = [m[cfg]["top10_actual_net_ticks"] for m in all_metrics if cfg in m]

            summary[cfg] = {
                "mean_ic_all": float(np.nanmean(ics)),
                "mean_ic_filled": float(np.nanmean(ics_filled)),
                "mean_top10_net": float(np.nanmean(top10s)),
                "n_folds": len(ics),
            }
            log.info(f"  {cfg}: IC_all={summary[cfg]['mean_ic_all']:.4f} "
                     f"IC_filled={summary[cfg]['mean_ic_filled']:.4f} "
                     f"top10_net={summary[cfg]['mean_top10_net']:.3f} "
                     f"({summary[cfg]['n_folds']} folds)")

        # Save summary
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump({"folds": all_metrics, "summary": summary}, f, indent=2, default=str)
        log.info(f"  Summary saved to summary.json")


if __name__ == "__main__":
    main()
