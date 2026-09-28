#!/usr/bin/env python3
"""
Fill-Probability v2.1 — COMBINED features (HC #497 R4)
=======================================================
Combines:
  (a) 41 microstructure features from /output/exec_features_v1 (v1 XGB inputs:
      queue velocity, depth, cancel rate, spread, OFI, etc.)
  (b) 42 signal-aware features from experiments/adverse_selection_v1.py
      (signal direction, signal magnitude, MAE/MFE proxies, etc.)
  -> 83-dim input to DualHeadMLP (83 -> 128 -> 64 -> 2 heads)

Heads identical to v2:
  head1 = sigmoid(P_fill)
  head2 = regression(adverse_cost in ticks at 5s)

Loss / optimizer / WF identical to v2 (BCE+1.0*MSE, AdamW 1e-3, sliding 60d/5d).

MLflow experiment: fill_prob_v21_adverse_combined
Author: Claude (v2 regressed because it dropped the v1 microstructure feature
        set. v2.1 keeps them AND adds the signal features.)
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LVL3_ROOT = Path("/home/nick/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT / "experiments"))
sys.path.insert(0, str(LVL3_ROOT / "scripts"))
import adverse_selection_v1 as v1  # noqa: E402
# Reuse v2's extract_features_v2 to get same signal-side features+targets
import train_fill_prob_v2_adverse as v2  # noqa: E402

EXEC_DIR = LVL3_ROOT / "output" / "exec_features_v1"
OUTPUT_DIR = LVL3_ROOT / "output" / "fill_prob_v21_combined"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "training.log"

logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [FILL_V21] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(str(LOG_FILE), mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fill_v21")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "fill_prob_v21_adverse_combined"

N_SIGNAL_FEATURES = v1.N_FEATURES  # 42
N_MICRO_FEATURES = 41
N_FEATURES = N_SIGNAL_FEATURES + N_MICRO_FEATURES  # 83

FILL_THRESHOLD_TICKS = v2.FILL_THRESHOLD_TICKS


def load_micro_features(date_str: str):
    """Return (micro_feat_array Nwx41, n_events, decision_stride) for date or None."""
    p = EXEC_DIR / f"{date_str}_exec_features.npz"
    if not p.exists():
        return None
    d = np.load(str(p), allow_pickle=True)
    feats = d["features"]  # (n_windows, 44)
    names = [str(n) for n in d["feature_names"]]
    # Drop the 3 fill_prob_* outputs (they ARE the v1 XGB outputs -> potential leakage)
    keep_idx = [i for i, n in enumerate(names) if not n.startswith("fill_prob_")]
    assert len(keep_idx) == 41, f"expected 41 micro feats, got {len(keep_idx)}"
    micro = feats[:, keep_idx].astype(np.float32)
    return {
        "micro": micro,
        "n_events": int(d["n_events"]),
        "decision_stride": int(d["decision_stride"]),
        "n_windows": int(d["n_windows"]),
    }


def align_micro_to_signal(signal_meta: np.ndarray, micro_info: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each signal-stride row (meta[:,1] = event index), find the most-recent
    completed exec window. Exec windows cover event range [k*stride, (k+1)*stride).
    Signal at event e -> use exec row k = e // stride - 1 (last fully-observed window)
    to avoid lookahead. Rows with e < stride (no complete exec window yet) are dropped.

    Returns (micro_aligned (N_keep, 41), keep_mask (N_signal,))
    """
    micro = micro_info["micro"]
    stride = micro_info["decision_stride"]
    n_windows = micro_info["n_windows"]

    e_idx = signal_meta[:, 1].astype(np.int64)
    # last completed exec window for event e: floor(e/stride) - 1
    exec_row = e_idx // stride - 1
    keep = (exec_row >= 0) & (exec_row < n_windows)
    aligned = np.zeros((int(keep.sum()), micro.shape[1]), dtype=np.float32)
    if keep.any():
        aligned = micro[exec_row[keep]]
    return aligned, keep


def extract_features_combined(mbo_path: Path, predictions: np.ndarray, labels: np.ndarray,
                              date_str: str):
    """Produce (features 83-dim, targets dict, meta)."""
    sig_feats, targets, meta = v2.extract_features_v2(mbo_path, predictions, labels)
    n_sig = len(sig_feats)
    if n_sig == 0:
        return sig_feats, targets, meta, 0, 0

    micro_info = load_micro_features(date_str)
    if micro_info is None:
        return np.zeros((0, N_FEATURES), dtype=np.float32), {
            k: np.zeros(0, dtype=np.float32) for k in targets
        }, np.zeros((0, meta.shape[1] if meta.ndim == 2 else 3), dtype=np.float32), n_sig, 0

    micro_aligned, keep = align_micro_to_signal(meta, micro_info)
    n_keep = int(keep.sum())
    if n_keep == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32), {
            k: np.zeros(0, dtype=np.float32) for k in targets
        }, np.zeros((0, meta.shape[1]), dtype=np.float32), n_sig, 0

    # Concatenate: [signal 42] + [micro 41] = 83
    combined = np.concatenate(
        [sig_feats[keep].astype(np.float32), micro_aligned.astype(np.float32)], axis=1
    )
    # NaN/Inf safety on the combined matrix
    np.nan_to_num(combined, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    new_targets = {k: v[keep] for k, v in targets.items()}
    new_meta = meta[keep]
    return combined, new_targets, new_meta, n_sig, n_keep


def load_all_dates_combined() -> List[dict]:
    pred_index = v1.build_date_pred_index()
    mbo_files = {
        Path(f).stem.replace("_mbo_events", ""): f
        for f in sorted(v1.MBO_DIR.glob("*_mbo_events.npz"))
    }
    log.info(f"Found {len(pred_index)} pred dates, {len(mbo_files)} MBO files, {len(list(EXEC_DIR.glob('*_exec_features.npz')))} exec files")
    all_dates = []
    total_sig = 0
    total_keep = 0
    for date_str in sorted(pred_index.keys()):
        if date_str not in mbo_files:
            continue
        try:
            pred_data = np.load(str(pred_index[date_str]), allow_pickle=True)
            predictions = pred_data["predictions"]
            labels = pred_data["labels"]
            feats, targets, meta, n_sig, n_keep = extract_features_combined(
                Path(mbo_files[date_str]), predictions, labels, date_str
            )
            total_sig += n_sig
            total_keep += n_keep
            if len(feats) < 10:
                continue
            all_dates.append({
                "date": date_str,
                "features": feats,
                "targets": targets,
                "meta": meta,
                "n_samples": len(feats),
            })
            if len(all_dates) % 20 == 0:
                log.info(f"  {len(all_dates)} dates loaded ({date_str}: sig={n_sig:,} keep={n_keep:,})")
        except Exception as e:
            log.error(f"  {date_str}: {e}")
            traceback.print_exc()
            continue
        gc.collect()
    n_total = sum(d["n_samples"] for d in all_dates)
    n_fill = sum(d["targets"]["fill_label"].sum() for d in all_dates)
    fill_rate = n_fill / n_total if n_total else 0
    drop_pct = 1.0 - (total_keep / max(total_sig, 1))
    log.info(
        f"Total: {len(all_dates)} dates, {n_total:,} samples, fill_rate={fill_rate:.1%}  "
        f"(signal rows dropped due to no exec coverage: {drop_pct:.2%})"
    )
    return all_dates


# Dual-head MLP same as v2 but input dim = 83
class DualHeadMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...] = (128, 64), dropout: float = 0.2):
        super().__init__()
        layers = []
        d_prev = in_dim
        for h in hidden:
            layers += [nn.Linear(d_prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            d_prev = h
        self.trunk = nn.Sequential(*layers)
        self.head_fill = nn.Linear(d_prev, 1)
        self.head_adv = nn.Linear(d_prev, 1)

    def forward(self, x):
        z = self.trunk(x)
        return self.head_fill(z).squeeze(-1), self.head_adv(z).squeeze(-1)


def train_fold(
    train_X, train_yf, train_ya,
    eval_X, eval_yf, eval_ya,
    epochs: int = 20,
    batch: int = 4096,
    lr: float = 1e-3,
    alpha: float = 1.0,
    patience: int = 4,
    fold_idx: int = 0,
):
    model = DualHeadMLP(N_FEATURES).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tX = torch.from_numpy(train_X).float().to(DEVICE)
    tYf = torch.from_numpy(train_yf).float().to(DEVICE)
    tYa = torch.from_numpy(train_ya).float().to(DEVICE)
    eX = torch.from_numpy(eval_X).float().to(DEVICE)
    eYf = torch.from_numpy(eval_yf).float().to(DEVICE)
    eYa = torch.from_numpy(eval_ya).float().to(DEVICE)

    pos = float(train_yf.sum())
    neg = float(len(train_yf) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = nn.MSELoss()

    n_train = len(train_X)
    best_val = float("inf")
    best_state = None
    stale = 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        tot_bce = 0.0; tot_mse = 0.0; nb = 0
        t0 = time.time()
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            xb = tX[idx]; yfb = tYf[idx]; yab = tYa[idx]
            logit_f, pred_a = model(xb)
            l_bce = bce(logit_f, yfb)
            l_mse = mse(pred_a, yab)
            loss = l_bce + alpha * l_mse
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_bce += l_bce.item(); tot_mse += l_mse.item(); nb += 1
        sched.step()
        tr_bce = tot_bce / max(nb, 1); tr_mse = tot_mse / max(nb, 1)

        model.eval()
        with torch.no_grad():
            vlogit_f, vpred_a = model(eX)
            v_bce = bce(vlogit_f, eYf).item()
            v_mse = mse(vpred_a, eYa).item()
            v_loss = v_bce + alpha * v_mse
            v_pfill = torch.sigmoid(vlogit_f).cpu().numpy()
        try:
            from sklearn.metrics import roc_auc_score
            v_auc = float(roc_auc_score(eval_yf, v_pfill)) if eval_yf.sum() > 0 and eval_yf.sum() < len(eval_yf) else float("nan")
        except Exception:
            v_auc = float("nan")
        dt = time.time() - t0
        log.info(
            f"  fold{fold_idx} ep{ep:02d} dt={dt:.1f}s  "
            f"tr_bce={tr_bce:.4f} tr_mse={tr_mse:.4f}  "
            f"val_bce={v_bce:.4f} val_mse={v_mse:.4f} val_auc={v_auc:.4f} val_loss={v_loss:.4f}"
        )
        try:
            import mlflow
            mlflow.log_metric(f"fold{fold_idx}_tr_bce", tr_bce, step=ep)
            mlflow.log_metric(f"fold{fold_idx}_tr_mse", tr_mse, step=ep)
            mlflow.log_metric(f"fold{fold_idx}_val_bce", v_bce, step=ep)
            mlflow.log_metric(f"fold{fold_idx}_val_mse", v_mse, step=ep)
            mlflow.log_metric(f"fold{fold_idx}_val_auc", v_auc, step=ep)
            mlflow.log_metric(f"fold{fold_idx}_val_loss", v_loss, step=ep)
        except Exception:
            pass
        if v_loss < best_val - 1e-5:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log.info(f"  fold{fold_idx} early stop at ep{ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vlogit_f, vpred_a = model(eX)
        v_pfill = torch.sigmoid(vlogit_f).cpu().numpy()
        v_adv = vpred_a.cpu().numpy()
    return model, {
        "best_val_loss": best_val,
        "n_train": n_train,
        "n_eval": len(eval_X),
        "fill_rate_train": float(train_yf.mean()),
        "fill_rate_eval": float(eval_yf.mean()),
        "adverse_mean_eval": float(eval_ya.mean()),
        "adverse_std_eval": float(eval_ya.std()),
    }, v_pfill, v_adv


def walk_forward(all_dates, train_window: int = 60, eval_window: int = 5, max_folds: int = None):
    n = len(all_dates)
    fold_metrics = []
    fold_idx = 0
    start = 0
    while start + train_window + eval_window <= n:
        if max_folds is not None and fold_idx >= max_folds:
            break
        train_slice = all_dates[start:start + train_window]
        eval_slice = all_dates[start + train_window:start + train_window + eval_window]
        train_X = np.concatenate([d["features"] for d in train_slice])
        train_yf = np.concatenate([d["targets"]["fill_label"] for d in train_slice])
        train_ya = np.concatenate([d["targets"]["adverse_cost"] for d in train_slice])
        eval_X = np.concatenate([d["features"] for d in eval_slice])
        eval_yf = np.concatenate([d["targets"]["fill_label"] for d in eval_slice])
        eval_ya = np.concatenate([d["targets"]["adverse_cost"] for d in eval_slice])

        for arr in (train_X, eval_X):
            np.nan_to_num(arr, copy=False)
        log.info(
            f"FOLD {fold_idx}: train {train_slice[0]['date']}..{train_slice[-1]['date']} "
            f"({len(train_X):,}) eval {eval_slice[0]['date']}..{eval_slice[-1]['date']} ({len(eval_X):,})"
        )
        model, m, vpf, vad = train_fold(
            train_X, train_yf, train_ya, eval_X, eval_yf, eval_ya, fold_idx=fold_idx
        )
        m["fold"] = fold_idx
        m["train_start"] = train_slice[0]["date"]
        m["train_end"] = train_slice[-1]["date"]
        m["eval_start"] = eval_slice[0]["date"]
        m["eval_end"] = eval_slice[-1]["date"]
        fold_metrics.append(m)
        torch.save(model.state_dict(), str(OUTPUT_DIR / f"fold_{fold_idx:03d}.pt"))
        np.savez(
            str(OUTPUT_DIR / f"fold_{fold_idx:03d}_oot.npz"),
            p_fill=vpf, pred_adverse=vad,
            y_fill=eval_yf, y_adverse=eval_ya,
        )
        start += eval_window
        fold_idx += 1
        gc.collect()
        torch.cuda.empty_cache()
    return fold_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--train-window", type=int, default=60)
    ap.add_argument("--eval-window", type=int, default=5)
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()

    log.info(f"Device: {DEVICE}")
    log.info(f"CUDA: {torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    log.info(f"Features: {N_FEATURES} (signal={N_SIGNAL_FEATURES} + micro={N_MICRO_FEATURES})")

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    run = mlflow.start_run(run_name=f"fill_v21_combined_{time.strftime('%Y%m%d_%H%M')}")
    mlflow.log_params({
        "n_features": N_FEATURES,
        "n_signal_features": N_SIGNAL_FEATURES,
        "n_micro_features": N_MICRO_FEATURES,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "alpha": args.alpha,
        "train_window_days": args.train_window,
        "eval_window_days": args.eval_window,
        "fill_threshold_ticks": FILL_THRESHOLD_TICKS,
        "fill_horizon": "1s",
        "adverse_horizon": "5s",
        "model": "DualHeadMLP_128_64",
        "hc": "497_R4",
        "version": "v21_combined",
    })
    log.info(f"MLflow run id: {run.info.run_id}")
    log.info(f"MLflow tracking uri: {MLFLOW_URI}")

    t0 = time.time()
    all_dates = load_all_dates_combined()
    log.info(f"Loaded {len(all_dates)} dates in {time.time()-t0:.1f}s")
    if len(all_dates) < args.train_window + args.eval_window:
        log.error(f"Not enough dates ({len(all_dates)}) for WF (need {args.train_window+args.eval_window})")
        mlflow.end_run(status="FAILED")
        sys.exit(2)

    fold_metrics = walk_forward(
        all_dates,
        train_window=args.train_window,
        eval_window=args.eval_window,
        max_folds=args.max_folds,
    )

    all_pfill, all_yfill, all_padv, all_yadv = [], [], [], []
    for fm in fold_metrics:
        d = np.load(str(OUTPUT_DIR / f"fold_{fm['fold']:03d}_oot.npz"))
        all_pfill.append(d["p_fill"]); all_yfill.append(d["y_fill"])
        all_padv.append(d["pred_adverse"]); all_yadv.append(d["y_adverse"])
    if all_pfill:
        pfill_c = np.concatenate(all_pfill); yfill_c = np.concatenate(all_yfill)
        padv_c = np.concatenate(all_padv); yadv_c = np.concatenate(all_yadv)
        try:
            from sklearn.metrics import roc_auc_score, brier_score_loss
            concat_auc = float(roc_auc_score(yfill_c, pfill_c))
            concat_brier = float(brier_score_loss(yfill_c, pfill_c))
        except Exception:
            concat_auc = float("nan"); concat_brier = float("nan")
        concat_mse = float(np.mean((padv_c - yadv_c) ** 2))
        log.info(f"CONCAT n={len(pfill_c):,}  fill_AUC={concat_auc:.4f}  fill_Brier={concat_brier:.4f}  adv_MSE={concat_mse:.4f}")
        mlflow.log_metric("concat_fill_auc", concat_auc)
        mlflow.log_metric("concat_fill_brier", concat_brier)
        mlflow.log_metric("concat_adverse_mse", concat_mse)
        mlflow.log_metric("n_folds", len(fold_metrics))
        mlflow.log_metric("n_total_oot", len(pfill_c))

    summary = {
        "hc": "497_R4",
        "model": "DualHeadMLP_128_64",
        "n_features": N_FEATURES,
        "n_signal_features": N_SIGNAL_FEATURES,
        "n_micro_features": N_MICRO_FEATURES,
        "fill_threshold_ticks": FILL_THRESHOLD_TICKS,
        "fill_horizon": "1s",
        "adverse_horizon": "5s",
        "loss": "BCE(fill) + alpha*MSE(adverse)",
        "alpha": args.alpha,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "train_window_days": args.train_window,
        "eval_window_days": args.eval_window,
        "n_folds": len(fold_metrics),
        "fold_metrics": fold_metrics,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Wrote {OUTPUT_DIR/'summary.json'}")
    mlflow.log_artifact(str(OUTPUT_DIR / "summary.json"))
    mlflow.end_run()
    log.info("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"FATAL: {e}")
        traceback.print_exc()
        try:
            import mlflow
            mlflow.end_run(status="FAILED")
        except Exception:
            pass
        sys.exit(1)
