#!/usr/bin/env python3
"""
Confluence-Aware Meta-Model v1 — MLP predicting FIFO-realized net ticks.

EXECUTION RESEARCH: Learns confluence patterns from 25 smart_v3 base features
plus engineered interaction/ratio/magnitude features (~55 total).

Architecture: MLP 256->128->64->1, BatchNorm, Dropout 0.2, ReLU.
Walk-forward: 10-day sliding train, 1-day OOT (only ~30 paired dates available).
Target: FIFO-realized net P&L in ticks (tp4sl3_short_net_ticks primary).
MLflow logging mandatory. Saves .npz predictions per fold.

Designed for RTX 3070 8GB (Windows). num_workers=0, pin_memory=False.
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "5"
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
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
OUT_DIR = REPO / "output" / "confluence_meta_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_WINDOW = 500       # feature window size
FIFO_WINDOW = 3000      # FIFO replay window size
FIFO_STRIDE = 250       # FIFO replay stride
N_BASE_FEAT = 25
BATCH = 256
N_TRAIN_DAYS = 10       # sliding window (30 paired dates -> 20 OOT folds)
N_EPOCHS = 8
LR = 5e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MLFLOW_URI = "http://jupiter:5000"
MLFLOW_EXP = "confluence_meta_v1"

# FIFO label configs to predict (multi-target)
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
log = logging.getLogger("confluence_meta")


# ────────────────────────────────────────────────────────────────────────────
# Confluence Feature Engineering
# ────────────────────────────────────────────────────────────────────────────
# The 25 smart_v3 features (indices):
# 0: ofi_1s, 1: ofi_5s, 2: ofi_10s, 3: ofi_30s
# 4: trade_tape_velocity, 5: trade_tape_imbalance
# 6: microprice_delta, 7: spread_bps
# 8: queue_imbalance_l1, 9: queue_imbalance_l2, 10: queue_imbalance_l3
# 11: book_depth_imbalance, 12: book_pressure_ratio
# 13: vol_1s, 14: vol_5s, 15: vol_10s, 16: vol_30s
# 17: return_1s, 18: return_5s, 19: return_10s, 20: return_30s
# 21: event_rate_1s, 22: event_rate_5s
# 23: large_trade_flag, 24: sweep_flag

# Top predictive features (from IC analysis): ofi_1s(0), trade_tape_velocity(4),
# microprice_delta(6), queue_imbalance_l1(8), book_pressure_ratio(12)
TOP5_IDX = [0, 4, 6, 8, 12]

# Pairwise product indices for top 5 (10 pairs)
PAIR_INDICES = []
for i in range(len(TOP5_IDX)):
    for j in range(i + 1, len(TOP5_IDX)):
        PAIR_INDICES.append((TOP5_IDX[i], TOP5_IDX[j]))

# Ratio features: momentum acceleration
RATIO_PAIRS = [
    (0, 1),   # ofi_1s / ofi_5s (short-term momentum vs medium)
    (1, 2),   # ofi_5s / ofi_10s
    (13, 14), # vol_1s / vol_5s (vol acceleration)
    (17, 18), # return_1s / return_5s
    (21, 22), # event_rate_1s / event_rate_5s
]

# Absolute value features (magnitude regardless of direction)
ABS_INDICES = [0, 4, 6, 8, 17]  # ofi_1s, tape_vel, microprice_delta, queue_imb, ret_1s

# Triplet products of top 3
TRIPLET_INDICES = [
    (0, 4, 6),   # ofi_1s * tape_vel * microprice_delta
    (0, 8, 12),  # ofi_1s * queue_imb * book_pressure
    (4, 6, 8),   # tape_vel * microprice_delta * queue_imb
]


def engineer_confluence_features(X: np.ndarray) -> np.ndarray:
    """
    Given X of shape (N, W, 25), compute confluence features at the LAST
    timestep of each window (decision point). Returns (N, N_CONFLUENCE).

    We aggregate the window via mean of last 50 events for smoothing.
    """
    # Use last 50 events for feature aggregation (smoothed decision-point features)
    tail = X[:, -50:, :]  # (N, 50, 25)
    base = tail.mean(axis=1)  # (N, 25) — smoothed features at decision point

    parts = [base]  # start with 25 base features

    # Pairwise products of top 5 (10 features)
    for i, j in PAIR_INDICES:
        parts.append((base[:, i] * base[:, j]).reshape(-1, 1))

    # Ratios (5 features) — clipped to avoid explosion
    for i, j in RATIO_PAIRS:
        denom = base[:, j].copy()
        denom[np.abs(denom) < 1e-8] = 1e-8  # avoid div by zero
        ratio = np.clip(base[:, i] / denom, -10, 10)
        parts.append(ratio.reshape(-1, 1))

    # Absolute values (5 features)
    for idx in ABS_INDICES:
        parts.append(np.abs(base[:, idx]).reshape(-1, 1))

    # Triplet products (3 features)
    for i, j, k in TRIPLET_INDICES:
        parts.append((base[:, i] * base[:, j] * base[:, k]).reshape(-1, 1))

    # Cross-timeframe OFI spread: ofi_1s - ofi_10s (1 feature)
    parts.append((base[:, 0] - base[:, 2]).reshape(-1, 1))

    # Queue-book confluence: queue_imb_l1 * book_depth_imbalance (1 feature)
    parts.append((base[:, 8] * base[:, 11]).reshape(-1, 1))

    # Vol-normalized OFI: ofi_1s / vol_1s (1 feature)
    vol1 = base[:, 13].copy()
    vol1[np.abs(vol1) < 1e-8] = 1e-8
    parts.append(np.clip(base[:, 0] / vol1, -10, 10).reshape(-1, 1))

    # Total: 25 + 10 + 5 + 5 + 3 + 1 + 1 + 1 = 51 features
    result = np.concatenate(parts, axis=1).astype(np.float32)
    return result


N_TOTAL_FEAT = 51  # 25 base + 26 engineered


# ────────────────────────────────────────────────────────────────────────────
# Model — MLP 256->128->64->1 with BatchNorm, Dropout, ReLU
# ────────────────────────────────────────────────────────────────────────────
class ConfluenceMLP(nn.Module):
    def __init__(self, in_dim=N_TOTAL_FEAT, n_targets=N_TARGETS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, n_targets),
        )

    def forward(self, x):  # x: (B, N_TOTAL_FEAT)
        return self.net(x)  # (B, N_TARGETS)


# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────
def list_paired_dates() -> List[str]:
    """Return dates that have BOTH smart_v3 events AND FIFO labels."""
    ev_dates = {f.name.split("_")[0] for f in MBO_DIR.glob("*_mbo_events.npz")}
    fi_dates = {f.name.split("_")[0] for f in FIFO_DIR.glob("*_fifo_labels.npz")}
    both = sorted(ev_dates & fi_dates)
    return both


def load_day_confluence(date: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load features + FIFO labels for one day, engineer confluence features.

    Returns:
        X: (N, N_TOTAL_FEAT) confluence features at each FIFO decision point
        Y: (N, N_TARGETS) FIFO net_ticks labels
    """
    ev_path = MBO_DIR / f"{date}_mbo_events.npz"
    fi_path = FIFO_DIR / f"{date}_fifo_labels.npz"

    if not ev_path.exists() or not fi_path.exists():
        return None

    ev_data = np.load(ev_path, allow_pickle=True)
    fi_data = np.load(fi_path, allow_pickle=True)

    events = ev_data["events"].astype(np.float32)
    n_events = len(events)

    window_ks = fi_data["window_k"]

    # Build labels array
    labels = np.stack(
        [fi_data[cfg].astype(np.float32) for cfg in LABEL_CONFIGS], axis=1
    )  # (N_fifo, N_TARGETS)

    # Decision event index for each FIFO window
    decision_idx = window_ks * FIFO_STRIDE + FIFO_WINDOW - 1

    # Feature window: last FEAT_WINDOW events ending at decision point
    feat_start = decision_idx - FEAT_WINDOW + 1

    # Filter valid indices
    valid = (feat_start >= 0) & (decision_idx < n_events)
    if valid.sum() == 0:
        return None

    feat_start_valid = feat_start[valid]
    decision_idx_valid = decision_idx[valid]
    labels = labels[valid]

    # Build raw feature windows (N, FEAT_WINDOW, 25)
    X_raw = np.stack([
        events[s:s + FEAT_WINDOW] for s in feat_start_valid
    ])

    # Engineer confluence features -> (N, N_TOTAL_FEAT)
    X = engineer_confluence_features(X_raw)

    # Filter NaN labels
    nan_mask = np.any(np.isnan(labels), axis=1)
    if nan_mask.all():
        return None
    X = X[~nan_mask]
    labels = labels[~nan_mask]

    # Also filter NaN/inf in features
    bad = np.any(~np.isfinite(X), axis=1)
    if bad.all():
        return None
    X = X[~bad]
    labels = labels[~bad]

    return X, labels


def compute_normalization(dates: List[str]):
    """Compute mean/std from training dates for the confluence features."""
    all_feats = []
    for date in dates:
        try:
            result = load_day_confluence(date)
            if result is None:
                continue
            X, _ = result
            all_feats.append(X)
        except Exception as e:
            log.warning(f"  norm skip {date}: {e}")
            continue

    if not all_feats:
        return np.zeros(N_TOTAL_FEAT, dtype=np.float32), np.ones(N_TOTAL_FEAT, dtype=np.float32)

    all_X = np.concatenate(all_feats, axis=0)
    mu = np.nanmean(all_X, axis=0).astype(np.float32)
    sd = np.nanstd(all_X, axis=0).astype(np.float32)
    sd[sd < 1e-8] = 1.0  # avoid div by zero
    return mu, sd


# ────────────────────────────────────────────────────────────────────────────
# Train
# ────────────────────────────────────────────────────────────────────────────
def train_one_fold(model, train_dates, mu, sd, n_epochs=N_EPOCHS):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loss_fn = nn.HuberLoss(delta=2.0)  # robust to outlier P&L values

    mu_t = torch.from_numpy(mu).to(DEVICE)
    sd_t = torch.from_numpy(sd).to(DEVICE)

    for ep in range(n_epochs):
        ep_loss, n_batches = 0.0, 0
        train_dates_shuffled = list(train_dates)
        np.random.shuffle(train_dates_shuffled)

        for date in train_dates_shuffled:
            try:
                result = load_day_confluence(date)
                if result is None:
                    continue
                X, Y = result
            except Exception as e:
                log.warning(f"  load fail {date}: {e}")
                continue

            if len(X) < 4:
                continue

            # Shuffle
            perm = np.random.permutation(len(X))
            X = X[perm]
            Y = Y[perm]

            for i in range(0, len(X), BATCH):
                xb = X[i:i + BATCH]
                yb = Y[i:i + BATCH]
                if len(xb) < 2:
                    continue

                xb_t = torch.from_numpy(xb).to(DEVICE)
                yb_t = torch.from_numpy(yb).to(DEVICE)

                # Normalize
                xb_t = (xb_t - mu_t) / sd_t

                opt.zero_grad(set_to_none=True)
                pred = model(xb_t)
                loss = loss_fn(pred, yb_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss.item())
                n_batches += 1

        sched.step()
        avg_loss = ep_loss / max(n_batches, 1)
        if (ep + 1) % 2 == 0 or ep == 0:
            log.info(f"    epoch {ep+1}/{n_epochs} avg_loss={avg_loss:.6f} batches={n_batches}")

    return model


@torch.no_grad()
def predict_day(model, date, mu, sd):
    """Predict FIFO net_ticks for one OOT day."""
    model.eval()
    result = load_day_confluence(date)
    if result is None:
        return None, None
    X, Y = result

    mu_t = torch.from_numpy(mu).to(DEVICE)
    sd_t = torch.from_numpy(sd).to(DEVICE)

    preds_chunks = []
    for i in range(0, len(X), BATCH):
        xb = X[i:i + BATCH]
        xb_t = torch.from_numpy(xb).to(DEVICE)
        xb_t = (xb_t - mu_t) / sd_t
        preds_chunks.append(model(xb_t).cpu().numpy())

    P = np.concatenate(preds_chunks, 0)
    return P, Y


def compute_metrics(P, Y, fold_idx, date):
    """Compute IC (Spearman) and profitability metrics per label config."""
    metrics = {}
    for i, cfg in enumerate(LABEL_CONFIGS):
        p, y = P[:, i], Y[:, i]

        # IC on all events
        if len(p) > 2 and np.std(p) > 1e-12 and np.std(y) > 1e-12:
            ic_all, _ = spearmanr(p, y)
        else:
            ic_all = 0.0

        # IC on filled events only (y != 0)
        filled = y != 0
        if filled.sum() > 10:
            p_f, y_f = p[filled], y[filled]
            if np.std(p_f) > 1e-12 and np.std(y_f) > 1e-12:
                ic_filled, _ = spearmanr(p_f, y_f)
            else:
                ic_filled = 0.0
        else:
            ic_filled = float('nan')

        fill_rate = float(filled.mean())

        # Top/bottom decile analysis
        if len(p) > 20:
            top10_thr = np.percentile(p, 90)
            bot10_thr = np.percentile(p, 10)
            top10_mask = p >= top10_thr
            bot10_mask = p <= bot10_thr
            top10_actual = float(y[top10_mask].mean()) if top10_mask.sum() > 0 else float('nan')
            bot10_actual = float(y[bot10_mask].mean()) if bot10_mask.sum() > 0 else float('nan')
        else:
            top10_actual = float('nan')
            bot10_actual = float('nan')

        metrics[cfg] = {
            "ic_all": float(ic_all) if not np.isnan(ic_all) else 0.0,
            "ic_filled": float(ic_filled) if not np.isnan(ic_filled) else 0.0,
            "fill_rate": fill_rate,
            "n_events": int(len(y)),
            "n_filled": int(filled.sum()),
            "top10_actual_net_ticks": top10_actual,
            "bot10_actual_net_ticks": bot10_actual,
        }

        log.info(f"    {cfg}: IC_all={metrics[cfg]['ic_all']:.4f} "
                 f"IC_filled={metrics[cfg]['ic_filled']:.4f} "
                 f"fill={metrics[cfg]['fill_rate']:.3f} "
                 f"top10={top10_actual:.3f} bot10={bot10_actual:.3f}")

    return metrics


# ────────────────────────────────────────────────────────────────────────────
# Concat IC computation (across all OOT folds)
# ────────────────────────────────────────────────────────────────────────────
def compute_concat_ic(all_preds, all_labels):
    """Compute IC by concatenating all OOT predictions (the REAL metric)."""
    P = np.concatenate(all_preds, axis=0)
    Y = np.concatenate(all_labels, axis=0)

    log.info(f"\n  CONCAT IC ({len(P)} total OOT events):")
    concat_metrics = {}
    for i, cfg in enumerate(LABEL_CONFIGS):
        p, y = P[:, i], Y[:, i]
        if np.std(p) > 1e-12 and np.std(y) > 1e-12:
            ic, _ = spearmanr(p, y)
        else:
            ic = 0.0

        filled = y != 0
        if filled.sum() > 10:
            ic_f, _ = spearmanr(p[filled], y[filled])
        else:
            ic_f = 0.0

        concat_metrics[cfg] = {
            "concat_ic_all": float(ic),
            "concat_ic_filled": float(ic_f),
            "n_total": int(len(y)),
            "n_filled": int(filled.sum()),
        }
        log.info(f"    {cfg}: concat_IC_all={ic:.4f} concat_IC_filled={ic_f:.4f} "
                 f"n={len(y)} filled={filled.sum()}")

    return concat_metrics


# ────────────────────────────────────────────────────────────────────────────
# Walk-forward main loop
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-folds", type=int, default=999)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dates = list_paired_dates()
    log.info(f"Found {len(dates)} paired dates (events + FIFO labels)")
    log.info(f"Date range: {dates[0]} -> {dates[-1]}")
    log.info(f"Train window: {N_TRAIN_DAYS} days, yields {max(0, len(dates)-N_TRAIN_DAYS)} OOT folds")
    log.info(f"Total features: {N_TOTAL_FEAT} (25 base + 26 confluence)")
    log.info(f"Device: {DEVICE}")

    if len(dates) < N_TRAIN_DAYS + 1:
        log.error(f"Need at least {N_TRAIN_DAYS + 1} dates, got {len(dates)}")
        return

    # MLflow setup — non-blocking, training proceeds even if MLflow is down
    mlflow_ok = False
    if HAVE_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(MLFLOW_EXP)
            mlflow_ok = True
            log.info(f"MLflow connected: {MLFLOW_URI} / {MLFLOW_EXP}")
        except Exception as e:
            log.warning(f"MLflow setup fail (proceeding without): {e}")

    # Resume check
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
    all_preds = []
    all_labels = []
    fold_count = 0

    for i in range(N_TRAIN_DAYS, len(dates)):
        if fold_count >= args.max_folds:
            break

        test_date = dates[i]
        fold_idx = i - N_TRAIN_DAYS

        if fold_idx in done_folds:
            # Load existing predictions for concat IC
            try:
                saved = np.load(OUT_DIR / f"fold_{fold_idx:02d}_preds.npz", allow_pickle=True)
                all_preds.append(saved["predictions"])
                all_labels.append(saved["labels"])
                log.info(f"Fold {fold_idx} ({test_date}) loaded from cache")
                fold_count += 1
            except Exception:
                pass
            continue

        train_dates = dates[i - N_TRAIN_DAYS:i]

        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold_idx} | train {train_dates[0]}->{train_dates[-1]} | OOT {test_date}")
        log.info(f"{'='*60}")

        t0 = time.time()

        # MLflow run
        mlflow_run = None
        if mlflow_ok:
            try:
                mlflow_run = mlflow.start_run(run_name=f"fold_{fold_idx}_{test_date}")
                mlflow.log_params({
                    "fold_idx": fold_idx,
                    "test_date": test_date,
                    "n_train_days": N_TRAIN_DAYS,
                    "n_epochs": N_EPOCHS,
                    "feat_window": FEAT_WINDOW,
                    "n_total_feat": N_TOTAL_FEAT,
                    "lr": LR,
                    "architecture": "MLP_256_128_64",
                    "label_configs": ",".join(LABEL_CONFIGS),
                })
            except Exception:
                pass

        # Normalization from train dates
        log.info("  Computing confluence feature normalization...")
        mu, sd = compute_normalization(train_dates)

        # Train
        model = ConfluenceMLP().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        log.info(f"  Model params: {n_params:,}")
        log.info(f"  Training on {len(train_dates)} dates, {N_EPOCHS} epochs...")
        model = train_one_fold(model, train_dates, mu, sd)

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
        metrics["train_time_s"] = round(time.time() - t0, 1)
        all_metrics.append(metrics)
        all_preds.append(P)
        all_labels.append(Y)

        # Log to MLflow
        if mlflow_ok:
            try:
                for cfg in LABEL_CONFIGS:
                    for k, v in metrics[cfg].items():
                        if isinstance(v, (int, float)) and not np.isnan(v):
                            mlflow.log_metric(f"{cfg}_{k}", v)
                mlflow.log_metric("fold_train_time_s", metrics["train_time_s"])
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
        log.info(f"  Saved predictions -> fold_{fold_idx:02d}_preds.npz")

        # Save model weights
        torch.save(model.state_dict(), OUT_DIR / f"fold_{fold_idx:02d}_model.pt")

        if mlflow_run:
            try: mlflow.end_run()
            except: pass

        fold_count += 1
        log.info(f"  Fold {fold_idx} done in {metrics['train_time_s']}s")

    # ── Summary ──────────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"SUMMARY - {fold_count} folds completed")
    log.info(f"{'='*60}")

    if all_metrics:
        # Per-fold summary
        summary = {}
        for cfg in LABEL_CONFIGS:
            ics = [m[cfg]["ic_all"] for m in all_metrics if cfg in m]
            ics_filled = [m[cfg]["ic_filled"] for m in all_metrics if cfg in m]
            top10s = [m[cfg]["top10_actual_net_ticks"] for m in all_metrics
                      if cfg in m and not np.isnan(m[cfg]["top10_actual_net_ticks"])]

            summary[cfg] = {
                "mean_ic_all": float(np.nanmean(ics)),
                "std_ic_all": float(np.nanstd(ics)),
                "mean_ic_filled": float(np.nanmean(ics_filled)),
                "mean_top10_net": float(np.nanmean(top10s)) if top10s else float('nan'),
                "n_folds": len(ics),
            }
            log.info(f"  {cfg}: IC_all={summary[cfg]['mean_ic_all']:.4f} "
                     f"(+/-{summary[cfg]['std_ic_all']:.4f}) "
                     f"IC_filled={summary[cfg]['mean_ic_filled']:.4f} "
                     f"top10_net={summary[cfg]['mean_top10_net']:.3f} "
                     f"({summary[cfg]['n_folds']} folds)")

        # Concat IC (the real metric)
        if all_preds:
            concat_metrics = compute_concat_ic(all_preds, all_labels)
            summary["concat_ic"] = concat_metrics

            # Log concat IC to MLflow as a final parent run
            if mlflow_ok:
                try:
                    with mlflow.start_run(run_name="concat_summary"):
                        mlflow.log_params({
                            "n_folds": fold_count,
                            "n_train_days": N_TRAIN_DAYS,
                            "n_total_feat": N_TOTAL_FEAT,
                            "architecture": "MLP_256_128_64",
                        })
                        for cfg, cm in concat_metrics.items():
                            for k, v in cm.items():
                                if isinstance(v, (int, float)):
                                    mlflow.log_metric(f"concat_{cfg}_{k}", v)
                except Exception:
                    pass

        # Save full summary
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump({"folds": all_metrics, "summary": summary}, f, indent=2, default=str)
        log.info(f"Summary saved to summary.json")


if __name__ == "__main__":
    main()
