#!/usr/bin/env python3
"""
HC #488 — DLinear MSE baseline (apples-to-apples vs quantile/pinball v1).

PURPOSE
-------
The quantile-DLinear (Razer, 2026-05-22) produced IC_P50 ≈ 0.26/0.20/0.14 at
1s/5s/10s on 20260427-28. Earlier baseline DLinear on the same data reported
~IC 0.13 at 1s. The leakage audit returned CLEAN.

This script reproduces the EXACT same setup as train_dlinear_quantile_v1.py
EXCEPT: output dim = n_h*1 instead of n_h*3, loss = MSE instead of pinball.
Goal: confirm the IC lift comes from the pinball objective rather than any
incidental difference (data, dates, optimizer, train days).

CHANGES vs quantile script:
  - output head: Linear(256, n_h*1)
  - loss: MSE
  - eval: Spearman(pred, label) per horizon  (called "ic_pred" for clarity)
  - no quantile / coverage / width logging
  - REPO path = Jupiter (no Razer's C:\\ path), DEVICE = cpu

Outputs: /home/jupiter/Lvl3Quant/output/hc488_dlinear_mse_baseline_mar_v1/
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

try:
    import mlflow
    HAVE_MLFLOW = True
except Exception:
    HAVE_MLFLOW = False

# ────────────────────────────────────────────────────────────────────────────
# Config — IDENTICAL to quantile script except DEVICE and paths
# ────────────────────────────────────────────────────────────────────────────
REPO = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
MBO_DIR = REPO / "data" / "processed" / "mbo_events_smart_v3"
OUT_DIR = REPO / "output" / "hc488_dlinear_mse_baseline_mar_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT = OUT_DIR / "intra_ckpt.pt"

WINDOW = 500
TRAIN_STRIDE = 25
OOT_STRIDE = 5
BATCH = 256
N_FEAT = 25
HORIZONS = ["1s", "5s", "10s"]
N_HORIZONS = len(HORIZONS)
N_TRAIN_DAYS = 10
N_EPOCHS = 3
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MLFLOW_URI = "http://jupiter:5000"
MLFLOW_EXP = "hc488_dlinear_mse_baseline_mar_v1"

# Apples-to-apples target: limit to test dates that overlap with quantile run.
# Quantile fold tests were 20260427, 20260428, 20260429.
# With N_TRAIN_DAYS=10, train windows are the 10 prior trading days.
TARGET_TEST_DATES = {"20260312", "20260313", "20260315", "20260316", "20260317"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(OUT_DIR / "run.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("hc488_mse")


# ────────────────────────────────────────────────────────────────────────────
# Model — same trunk, head dim = n_h*1
# ────────────────────────────────────────────────────────────────────────────
class DLinearMSE(nn.Module):
    def __init__(self, window=WINDOW, n_feat=N_FEAT,
                 n_horizons=N_HORIZONS, kernel=25):
        super().__init__()
        self.n_h = n_horizons
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=kernel // 2)
        self.lin_trend = nn.Linear(window * n_feat, 128)
        self.lin_season = nn.Linear(window * n_feat, 128)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(256, n_horizons),  # n_h * 1 point predictions
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
        return self.head(h)  # (B, n_h)


# ────────────────────────────────────────────────────────────────────────────
# Data — IDENTICAL to quantile script
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


def valid_starts(n, window, stride, l1, l5, l10):
    starts = np.arange(0, n - window + 1, stride)
    end_idx = starts + window - 1
    mask = ~(np.isnan(l1[end_idx]) | np.isnan(l5[end_idx]) | np.isnan(l10[end_idx]))
    return starts[mask]


# ────────────────────────────────────────────────────────────────────────────
# Train/eval
# ────────────────────────────────────────────────────────────────────────────
def train_one_fold(model, train_dates, mu, sd, n_epochs=N_EPOCHS):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    mse = nn.MSELoss()
    for ep in range(n_epochs):
        ep_loss, n_batches = 0.0, 0
        for date in train_dates:
            try:
                ev, l1, l5, l10 = load_day(date)
            except Exception as e:
                log.warning(f"  load fail {date}: {e}")
                continue
            starts = valid_starts(len(ev), WINDOW, TRAIN_STRIDE, l1, l5, l10)
            if len(starts) < BATCH:
                continue
            perm = np.random.permutation(len(starts))
            starts = starts[perm]
            for i in range(0, len(starts), BATCH):
                bs = starts[i:i + BATCH]
                if len(bs) < 2:
                    continue
                xb = np.stack([ev[s:s + WINDOW] for s in bs])
                xb = (xb - mu) / sd
                end = bs + WINDOW - 1
                yb = np.stack([l1[end], l5[end], l10[end]], axis=1)  # (B, n_h)
                xb_t = torch.from_numpy(xb).to(DEVICE, non_blocking=True)
                yb_t = torch.from_numpy(yb).to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred = model(xb_t)
                loss = mse(pred, yb_t)
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                n_batches += 1
        log.info(f"    epoch {ep+1}/{n_epochs} avg_mse={ep_loss/max(n_batches,1):.6f} batches={n_batches}")
        if HAVE_MLFLOW:
            try:
                mlflow.log_metric("train_mse", ep_loss / max(n_batches, 1), step=ep)
            except Exception:
                pass
    return model


@torch.no_grad()
def predict_day(model, date, mu, sd):
    model.eval()
    ev, l1, l5, l10 = load_day(date)
    starts = valid_starts(len(ev), WINDOW, OOT_STRIDE, l1, l5, l10)
    if len(starts) == 0:
        return None, None, None
    preds_chunks, labs_chunks = [], []
    for i in range(0, len(starts), BATCH):
        bs = starts[i:i + BATCH]
        xb = np.stack([ev[s:s + WINDOW] for s in bs])
        xb = (xb - mu) / sd
        end = bs + WINDOW - 1
        yb = np.stack([l1[end], l5[end], l10[end]], axis=1)
        xb_t = torch.from_numpy(xb).to(DEVICE, non_blocking=True)
        preds_chunks.append(model(xb_t).cpu().numpy())  # (B, n_h)
        labs_chunks.append(yb)
    P = np.concatenate(preds_chunks, 0)  # (N, n_h)
    Y = np.concatenate(labs_chunks, 0)   # (N, n_h)

    from scipy.stats import spearmanr, pearsonr
    ic_s, ic_p = [], []
    for c in range(N_HORIZONS):
        p = P[:, c]; y = Y[:, c]
        v = ~(np.isnan(p) | np.isnan(y))
        if v.sum() >= 50:
            r_s, _ = spearmanr(p[v], y[v])
            r_p, _ = pearsonr(p[v], y[v])
            ic_s.append(float(r_s)); ic_p.append(float(r_p))
        else:
            ic_s.append(float("nan")); ic_p.append(float("nan"))
    diag = dict(ic_spearman=ic_s, ic_pearson=ic_p)
    return P, Y, diag


def compute_feature_stats(train_dates, sample_frac=0.05):
    mu = np.zeros(N_FEAT, dtype=np.float64)
    sd = np.zeros(N_FEAT, dtype=np.float64)
    n_total = 0
    for date in train_dates:
        try:
            ev, *_ = load_day(date)
        except Exception:
            continue
        n_take = int(len(ev) * sample_frac)
        if n_take < 1000:
            n_take = min(1000, len(ev))
        idx = np.random.choice(len(ev), n_take, replace=False)
        sub = ev[idx]
        mu += sub.sum(0)
        sd += (sub ** 2).sum(0)
        n_total += len(sub)
    mu /= n_total
    var = sd / n_total - mu ** 2
    sd = np.sqrt(np.clip(var, 1e-12, None))
    return mu.astype(np.float32), sd.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--target-only", action="store_true",
                    help="Only run folds whose test date is in TARGET_TEST_DATES")
    args = ap.parse_args()

    log.info(f"Device: {DEVICE}")
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
        log.error("not enough dates")
        sys.exit(2)

    folds = [(dates[i - N_TRAIN_DAYS:i], dates[i])
             for i in range(N_TRAIN_DAYS, len(dates))]
    if args.target_only:
        folds = [f for f in folds if f[1] in TARGET_TEST_DATES]
    log.info(f"Folds (after target filter): {len(folds)} "
             f"({[f[1] for f in folds]})")

    completed = set()
    if args.resume and CKPT.exists():
        try:
            cdata = torch.load(CKPT, map_location="cpu", weights_only=False)
            completed = set(cdata.get("completed", []))
            log.info(f"Resume: {len(completed)} folds already done")
        except Exception as e:
            log.warning(f"resume load failed: {e}")

    t_start = time.time()
    for fi, (train_dates, test_date) in enumerate(folds, start=1):
        log.info(f"\n=== FOLD {fi}/{len(folds)}  test={test_date}  "
                 f"train={train_dates[0]}..{train_dates[-1]} ===")
        if test_date in completed:
            log.info("  already done, skip")
            continue
        np.random.seed(42 + fi)
        torch.manual_seed(42 + fi)
        mu, sd = compute_feature_stats(train_dates)
        log.info(f"  feature stats: |mu|_mean={np.abs(mu).mean():.3f}, "
                 f"sd_mean={sd.mean():.3f}")

        mlflow_run = None
        if HAVE_MLFLOW:
            try:
                mlflow_run = mlflow.start_run(run_name=f"fold_{fi:02d}_{test_date}")
                mlflow.log_params(dict(
                    fold=fi, test_date=test_date,
                    window=WINDOW, train_stride=TRAIN_STRIDE,
                    oot_stride=OOT_STRIDE, batch=BATCH,
                    n_epochs=N_EPOCHS, lr=LR, n_train_days=N_TRAIN_DAYS,
                    horizons=str(HORIZONS), model="DLinearMSE",
                ))
            except Exception as e:
                log.warning(f"mlflow start_run failed: {e}")

        t_m = time.time()
        model = DLinearMSE().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        log.info(f"  n_params: {n_params/1e6:.3f}M")
        model = train_one_fold(model, train_dates, mu, sd)

        P, Y, diag = predict_day(model, test_date, mu, sd)
        if P is None or len(P) == 0:
            log.warning(f"  fold {fi} test={test_date}: predict_day returned empty — marking complete and continuing")
            completed.add(test_date)
            torch.save({"completed": list(completed)}, CKPT)
            if HAVE_MLFLOW and mlflow_run is not None:
                try:
                    mlflow.set_tag("status", "no_predictions")
                    mlflow.end_run()
                except Exception:
                    pass
            continue
        out_path = OUT_DIR / f"fold_{fi:02d}_preds.npz"
        np.savez(
            out_path,
            preds=P.astype(np.float32),    # (N, n_h)
            labels=Y.astype(np.float32),
            pred_1s=P[:, 0].astype(np.float32),
            pred_5s=P[:, 1].astype(np.float32),
            pred_10s=P[:, 2].astype(np.float32),
            date=test_date,
            horizons=np.array(HORIZONS),
            ic_spearman=np.array(diag["ic_spearman"], dtype=np.float32),
            ic_pearson=np.array(diag["ic_pearson"], dtype=np.float32),
        )
        log.info(f"  fold {fi} test={test_date}")
        for hi, hname in enumerate(HORIZONS):
            log.info(f"    {hname}: IC_spear={diag['ic_spearman'][hi]:.4f}  "
                     f"IC_pear={diag['ic_pearson'][hi]:.4f}")
        log.info(f"  saved {out_path.name}  N={len(P):,}  "
                 f"in {time.time()-t_m:.1f}s")

        if HAVE_MLFLOW and mlflow_run is not None:
            try:
                for hi, hname in enumerate(HORIZONS):
                    mlflow.log_metric(f"ic_spearman_{hname}",
                                      diag["ic_spearman"][hi])
                    mlflow.log_metric(f"ic_pearson_{hname}",
                                      diag["ic_pearson"][hi])
                mlflow.end_run()
            except Exception as e:
                log.warning(f"mlflow log/end failed: {e}")

        completed.add(test_date)
        torch.save({"completed": list(completed)}, CKPT)

    log.info(f"\n=== ALL DONE in {(time.time()-t_start)/60:.1f} min ===")


if __name__ == "__main__":
    main()
