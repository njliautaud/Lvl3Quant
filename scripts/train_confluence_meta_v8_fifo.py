"""
Confluence Meta-Model v8 (FIFO-aware)
=====================================
Trains an MLP meta-model that takes CNN-Mamba v2 + PatchTST predictions as inputs
and predicts realized FIFO net P&L in ticks.

Walk-forward: 15-day sliding train window, 1-day OOT prediction.
Target: directional FIFO net ticks (long if cm_1s > 0, short if cm_1s < 0).

Designed for: Razer (RTX 3070, Windows, 8GB VRAM)
"""

import os
import sys
import json
import time
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr

# ── Paths ──────────────────────────────────────────────────────────────────────
CM_DIR = Path("C:/Users/claude/Lvl3Quant/output/cnn_mamba_v2_bulk_oot_v2")
PTST_DIR = Path("C:/Users/claude/Lvl3Quant/output/patchtst_bulk_oot")
FIFO_DIR = Path("C:/Users/claude/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels")
OUT_DIR = Path("C:/Users/claude/Lvl3Quant/output/confluence_meta_v8_fifo")

# ── Hyperparameters ────────────────────────────────────────────────────────────
TRAIN_WINDOW = 15       # sliding window size in dates
EPOCHS = 30
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
PATIENCE = 5            # early stopping patience on val loss
LR_PATIENCE = 3         # ReduceLROnPlateau patience
CM_THRESHOLD = 0.01     # skip rows where |cm_1s| < this
TP_SL_KEY = "tp4sl3"    # which FIFO label set to use

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Model ──────────────────────────────────────────────────────────────────────
class ConfluenceMetaMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Data Loading ───────────────────────────────────────────────────────────────
def discover_dates():
    """Find all dates where CM + FIFO exist. Flag which also have PatchTST."""
    cm_dates = set()
    for f in CM_DIR.glob("*_predictions.npz"):
        d = f.stem.replace("_predictions", "")
        cm_dates.add(d)

    fifo_dates = set()
    for f in FIFO_DIR.glob("*_fifo_labels.npz"):
        d = f.stem.replace("_fifo_labels", "")
        fifo_dates.add(d)

    ptst_dates = set()
    for f in PTST_DIR.glob("*_predictions.npz"):
        d = f.stem.replace("_predictions", "")
        ptst_dates.add(d)

    usable = sorted(cm_dates & fifo_dates)
    has_ptst = {d: d in ptst_dates for d in usable}

    print(f"Discovered dates: {len(usable)} CM+FIFO, "
          f"{sum(has_ptst.values())} also have PatchTST")
    return usable, has_ptst


def load_day(date: str, has_ptst: bool):
    """Load one day's data, build features and target. Returns (X, y) or None."""
    # Load CM predictions
    cm_path = CM_DIR / f"{date}_predictions.npz"
    cm = np.load(cm_path)
    cm_preds = cm["predictions"]  # (N, 3)

    # Load FIFO labels
    fifo_path = FIFO_DIR / f"{date}_fifo_labels.npz"
    fifo = np.load(fifo_path)
    fifo_wk = fifo["window_k"]
    long_net = fifo[f"{TP_SL_KEY}_long_net_ticks"]
    short_net = fifo[f"{TP_SL_KEY}_short_net_ticks"]
    long_filled = fifo[f"{TP_SL_KEY}_long_filled"]
    short_filled = fifo[f"{TP_SL_KEY}_short_filled"]

    # Load PatchTST if available
    if has_ptst:
        ptst_path = PTST_DIR / f"{date}_predictions.npz"
        ptst = np.load(ptst_path)
        ptst_preds = ptst["predictions"]  # (N, 3)
    else:
        ptst_preds = None

    # Align by truncating to min length
    n = min(len(cm_preds), len(fifo_wk))
    if ptst_preds is not None:
        n = min(n, len(ptst_preds))
        ptst_preds = ptst_preds[:n]

    cm_preds = cm_preds[:n]
    long_net = long_net[:n]
    short_net = short_net[:n]
    long_filled = long_filled[:n]
    short_filled = short_filled[:n]

    # If no PatchTST, zero-fill
    if ptst_preds is None:
        ptst_preds = np.zeros_like(cm_preds)

    # ── Feature Engineering ────────────────────────────────────────────────
    cm_1s, cm_5s, cm_10s = cm_preds[:, 0], cm_preds[:, 1], cm_preds[:, 2]
    pt_1s, pt_5s, pt_10s = ptst_preds[:, 0], ptst_preds[:, 1], ptst_preds[:, 2]

    features = []

    # 1. Raw predictions (6)
    features.extend([cm_1s, cm_5s, cm_10s, pt_1s, pt_5s, pt_10s])

    # 2. Agreement features (3) — sign agreement per horizon
    features.append((np.sign(cm_1s) == np.sign(pt_1s)).astype(np.float32))
    features.append((np.sign(cm_5s) == np.sign(pt_5s)).astype(np.float32))
    features.append((np.sign(cm_10s) == np.sign(pt_10s)).astype(np.float32))

    # 3. Magnitude features (6)
    features.extend([np.abs(cm_1s), np.abs(cm_5s), np.abs(cm_10s),
                      np.abs(pt_1s), np.abs(pt_5s), np.abs(pt_10s)])

    # 4. Cross-horizon agreement (2) — all horizons same sign within model
    cm_all_same = ((np.sign(cm_1s) == np.sign(cm_5s)) &
                   (np.sign(cm_5s) == np.sign(cm_10s))).astype(np.float32)
    pt_all_same = ((np.sign(pt_1s) == np.sign(pt_5s)) &
                   (np.sign(pt_5s) == np.sign(pt_10s))).astype(np.float32)
    features.extend([cm_all_same, pt_all_same])

    # 5. Pair products (9) — interaction terms
    for cm_h in [cm_1s, cm_5s, cm_10s]:
        for pt_h in [pt_1s, pt_5s, pt_10s]:
            features.append(cm_h * pt_h)

    # 6. Spread features (6)
    features.append(cm_5s - cm_1s)
    features.append(cm_10s - cm_5s)
    features.append(pt_5s - pt_1s)
    features.append(pt_10s - pt_5s)
    features.append(cm_1s - pt_1s)
    features.append(cm_5s - pt_5s)

    # 7. has_ptst indicator (1)
    features.append(np.full(n, float(has_ptst), dtype=np.float32))

    X = np.column_stack(features).astype(np.float32)  # (N, 33)

    # ── Target: directional FIFO net ticks ─────────────────────────────────
    # Long signal → long net ticks; short signal → short net ticks
    target = np.zeros(n, dtype=np.float32)
    is_long = cm_1s > CM_THRESHOLD
    is_short = cm_1s < -CM_THRESHOLD
    is_valid = is_long | is_short

    target[is_long & long_filled] = long_net[is_long & long_filled]
    target[is_long & ~long_filled] = 0.0
    target[is_short & short_filled] = short_net[is_short & short_filled]
    target[is_short & ~short_filled] = 0.0

    # Filter to valid rows only
    X = X[is_valid]
    target = target[is_valid]

    # Also return raw cm/ptst for saving
    cm_out = cm_preds[is_valid]
    pt_out = ptst_preds[is_valid]

    return X, target, cm_out, pt_out


# ── Training Utilities ─────────────────────────────────────────────────────────
def compute_ic(pred, actual):
    """Pearson correlation (IC) between predicted and actual."""
    if len(pred) < 5:
        return 0.0
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 5:
        return 0.0
    r, _ = pearsonr(pred[mask], actual[mask])
    return r if np.isfinite(r) else 0.0


def compute_top_pct_mean(pred, actual, pct=0.20):
    """Mean actual net ticks for the top `pct` fraction of predictions."""
    if len(pred) < 10:
        return 0.0
    threshold = np.percentile(pred, 100 * (1 - pct))
    mask = pred >= threshold
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(actual[mask]))


def train_one_fold(model, X_train, y_train, X_val, y_val):
    """Train model for one fold with early stopping. Returns best val loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=0.5)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)

    val_X_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
            train_n += len(xb)

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X_t)
            val_loss = criterion(val_pred, val_y_t).item()

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_loss


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Confluence Meta-Model v8 (FIFO-aware)")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover available dates
    dates, has_ptst = discover_dates()
    if len(dates) < TRAIN_WINDOW + 1:
        print(f"ERROR: Need at least {TRAIN_WINDOW + 1} dates, found {len(dates)}")
        sys.exit(1)

    # Pre-load all days
    print(f"\nLoading {len(dates)} days of data...")
    day_data = {}
    for d in dates:
        try:
            result = load_day(d, has_ptst[d])
            if result is not None and len(result[0]) > 0:
                day_data[d] = result
                print(f"  {d}: {len(result[0]):>7,} rows, "
                      f"target mean={result[1].mean():+.4f}, "
                      f"ptst={'Y' if has_ptst[d] else 'N'}")
            else:
                print(f"  {d}: SKIPPED (no valid rows)")
        except Exception as e:
            print(f"  {d}: FAILED ({e})")

    valid_dates = [d for d in dates if d in day_data]
    print(f"\nUsable dates: {len(valid_dates)}")

    n_features = day_data[valid_dates[0]][0].shape[1]
    print(f"Feature count: {n_features}")

    # ── Walk-forward ───────────────────────────────────────────────────────
    n_folds = len(valid_dates) - TRAIN_WINDOW
    print(f"Walk-forward folds: {n_folds} "
          f"(dates {TRAIN_WINDOW} through {len(valid_dates) - 1} are OOT)")
    print("-" * 70)

    all_pred = []
    all_actual = []
    all_cm = []
    all_ptst = []
    all_dates_oot = []
    fold_results = []

    for fold_i in range(n_folds):
        train_dates = valid_dates[fold_i: fold_i + TRAIN_WINDOW]
        oot_date = valid_dates[fold_i + TRAIN_WINDOW]

        if oot_date not in day_data:
            continue

        # Build train set (last 2 days as validation)
        val_split = max(1, 2)
        pure_train_dates = train_dates[:-val_split]
        val_dates_list = train_dates[-val_split:]

        X_train_parts, y_train_parts = [], []
        for d in pure_train_dates:
            if d in day_data:
                X_train_parts.append(day_data[d][0])
                y_train_parts.append(day_data[d][1])

        X_val_parts, y_val_parts = [], []
        for d in val_dates_list:
            if d in day_data:
                X_val_parts.append(day_data[d][0])
                y_val_parts.append(day_data[d][1])

        if not X_train_parts or not X_val_parts:
            continue

        X_train = np.concatenate(X_train_parts)
        y_train = np.concatenate(y_train_parts)
        X_val = np.concatenate(X_val_parts)
        y_val = np.concatenate(y_val_parts)

        # Standardize features (fit on train only)
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1.0  # avoid div by zero for constant features
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

        # Standardize OOT
        X_oot, y_oot, cm_oot, pt_oot = day_data[oot_date]
        X_oot_norm = (X_oot - mean) / std

        # Train model
        model = ConfluenceMetaMLP(n_features).to(device)
        best_val_loss = train_one_fold(model, X_train, y_train, X_val, y_val)

        # Predict OOT
        model.eval()
        with torch.no_grad():
            X_oot_t = torch.tensor(X_oot_norm, dtype=torch.float32).to(device)
            pred_oot = model(X_oot_t).cpu().numpy()

        # Metrics
        ic = compute_ic(pred_oot, y_oot)
        top20_mean = compute_top_pct_mean(pred_oot, y_oot, 0.20)
        bot20_mean = compute_top_pct_mean(-pred_oot, y_oot, 0.20)  # bottom 20% (worst predicted)

        fold_result = {
            "fold": fold_i,
            "oot_date": oot_date,
            "n_train": len(X_train),
            "n_oot": len(X_oot),
            "val_loss": float(best_val_loss),
            "oot_ic": float(ic),
            "top20_mean_net_ticks": float(top20_mean),
            "bot20_mean_net_ticks": float(bot20_mean),
            "oot_mean_actual": float(y_oot.mean()),
            "oot_std_actual": float(y_oot.std()),
        }
        fold_results.append(fold_result)

        print(f"Fold {fold_i:3d} | {oot_date} | "
              f"n={len(X_oot):>6,} | "
              f"IC={ic:+.4f} | "
              f"top20={top20_mean:+.3f}t | "
              f"bot20={bot20_mean:+.3f}t | "
              f"val_loss={best_val_loss:.4f}")

        # Save per-fold predictions
        np.savez_compressed(
            OUT_DIR / f"fold_{fold_i:03d}_predictions.npz",
            predicted_net_ticks=pred_oot.astype(np.float32),
            actual_net_ticks=y_oot.astype(np.float32),
            cm_preds=cm_oot.astype(np.float32),
            ptst_preds=pt_oot.astype(np.float32),
            date=np.array(oot_date),
        )

        # Accumulate for concat metrics
        all_pred.append(pred_oot)
        all_actual.append(y_oot)
        all_cm.append(cm_oot)
        all_ptst.append(pt_oot)
        all_dates_oot.append(oot_date)

    # ── Concat OOT Summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CONCAT OOT RESULTS")
    print("=" * 70)

    if not all_pred:
        print("ERROR: No folds completed successfully")
        sys.exit(1)

    concat_pred = np.concatenate(all_pred)
    concat_actual = np.concatenate(all_actual)

    concat_ic = compute_ic(concat_pred, concat_actual)
    concat_top20 = compute_top_pct_mean(concat_pred, concat_actual, 0.20)
    concat_top10 = compute_top_pct_mean(concat_pred, concat_actual, 0.10)
    concat_top5 = compute_top_pct_mean(concat_pred, concat_actual, 0.05)
    concat_bot20 = compute_top_pct_mean(-concat_pred, concat_actual, 0.20)

    print(f"Total OOT rows:     {len(concat_pred):,}")
    print(f"OOT dates:          {len(all_dates_oot)}")
    print(f"Concat IC:          {concat_ic:+.4f}")
    print(f"Mean actual ticks:  {concat_actual.mean():+.4f}")
    print()
    print("Filter performance (mean actual net ticks):")
    print(f"  Top  5% predicted: {concat_top5:+.4f} ticks")
    print(f"  Top 10% predicted: {concat_top10:+.4f} ticks")
    print(f"  Top 20% predicted: {concat_top20:+.4f} ticks")
    print(f"  Bot 20% predicted: {concat_bot20:+.4f} ticks")

    # Sharpe-like estimate: mean/std of per-day mean net ticks for top 20%
    daily_top20 = []
    offset = 0
    for i, d in enumerate(all_dates_oot):
        n_d = len(all_pred[i])
        day_pred = concat_pred[offset:offset + n_d]
        day_actual = concat_actual[offset:offset + n_d]
        day_top20 = compute_top_pct_mean(day_pred, day_actual, 0.20)
        daily_top20.append(day_top20)
        offset += n_d

    daily_top20 = np.array(daily_top20)
    if daily_top20.std() > 0:
        sharpe_est = daily_top20.mean() / daily_top20.std() * np.sqrt(252)
    else:
        sharpe_est = 0.0

    pf_days = (daily_top20 > 0).sum()
    print(f"\nDaily top-20% Sharpe estimate: {sharpe_est:+.2f} (annualized)")
    print(f"Profitable days (top 20%):     {pf_days}/{len(daily_top20)} "
          f"({100 * pf_days / len(daily_top20):.0f}%)")
    print(f"Mean daily top-20% net ticks:  {daily_top20.mean():+.4f}")

    # Per-fold IC distribution
    ics = [r["oot_ic"] for r in fold_results]
    print(f"\nPer-fold IC: mean={np.mean(ics):+.4f}, "
          f"median={np.median(ics):+.4f}, "
          f"std={np.std(ics):.4f}, "
          f"positive={sum(1 for x in ics if x > 0)}/{len(ics)}")

    # ── Save Summary ──────────────────────────────────────────────────────
    summary = {
        "model": "confluence_meta_v8_fifo",
        "architecture": "MLP 33->256->128->64->1",
        "target": f"{TP_SL_KEY}_directional_net_ticks",
        "train_window": TRAIN_WINDOW,
        "n_folds": len(fold_results),
        "n_features": n_features,
        "total_oot_rows": int(len(concat_pred)),
        "concat_ic": float(concat_ic),
        "top5_mean_net_ticks": float(concat_top5),
        "top10_mean_net_ticks": float(concat_top10),
        "top20_mean_net_ticks": float(concat_top20),
        "bot20_mean_net_ticks": float(concat_bot20),
        "sharpe_estimate_annualized": float(sharpe_est),
        "profitable_days_top20": f"{pf_days}/{len(daily_top20)}",
        "mean_daily_top20_net_ticks": float(daily_top20.mean()),
        "per_fold_ic_mean": float(np.mean(ics)),
        "per_fold_ic_median": float(np.median(ics)),
        "per_fold_ic_positive_rate": float(sum(1 for x in ics if x > 0) / len(ics)),
        "oot_dates": all_dates_oot,
        "fold_results": fold_results,
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = OUT_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Also save concat predictions
    np.savez_compressed(
        OUT_DIR / "concat_oot_predictions.npz",
        predicted_net_ticks=concat_pred.astype(np.float32),
        actual_net_ticks=concat_actual.astype(np.float32),
        dates=np.array(all_dates_oot),
    )

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
