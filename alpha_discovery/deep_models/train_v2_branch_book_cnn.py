#!/usr/bin/env python3
"""
Triple Fusion v2 — Book CNN diversifying branch.

Goal: produce OOT predictions PIXEL-ALIGNED to cnn_mamba_v2_smart_v3_mar/fold_NN
so the meta-learner can stack streams without re-aligning anything.

Inputs per event (after book→event alignment by timestamp searchsorted):
  - smart_v3 events: (T, 25)   — included at low weight to give book context
  - book_shape:      (T, 20)   — 10 levels per side, near-price (--depth flag)
  - book_dynamics:   (T, 10)   — book change rates / pressure features

Architecture:
  Per window (W=1000) → 1D CNN over time (kernel=5 dilated 1/2/4) → AvgPool
  → 3 small MLP heads (1s / 5s / 10s) — same horizon set as cnn_mamba_v2.

Output (per OOT fold):
  fold_NN_oot_predictions.npz with keys:
    predictions: (n_windows, 3) float32
    labels:      (n_windows, 3) float32
    horizons:    ['1s','5s','10s']
    oot_files:   list of source npz paths
    embeddings:  (n_windows, 64) float32   [for downstream stacking]

Walk-forward fold layout MIRRORS cnn_mamba_v2_smart_v3_mar:
  fold_05 → 2026-03-01 (skip if zero RTH events)
  fold_06 → 2026-03-02
  fold_07 → 2026-03-03
  fold_08 → 2026-03-04
  fold_09 → 2026-03-05
  ... (extend as user purchases more dates)

Usage:
  python train_v2_branch_book_cnn.py \\
      --depth 10 \\
      --folds 5,6,7,8,9 \\
      --epochs 4 \\
      --output-dir /home/nick/Lvl3Quant/output/v2_book_cnn_depth10
"""
import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("v2_book_cnn")

# ---------------- Path autodetect (local Jupiter vs Neptune) ----------------
HOST = socket.gethostname().lower()
if "neptune" in HOST or os.path.exists("/home/nick/Lvl3Quant"):
    ROOT = Path("/home/nick/Lvl3Quant")
else:
    ROOT = Path("/home/jupiter/Lvl3Quant")

EVENT_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3"
BOOK_DIR  = ROOT / "data" / "processed" / "mbo_book_normalized"
BOOK_CACHE = ROOT / "data" / "processed" / "book_aligned_cache"  # uncompressed .npy
DEFAULT_OUT = ROOT / "output" / "v2_book_cnn"

# ---------------- WF fold map (matches cnn_mamba_v2_smart_v3_mar) -----------
# Each fold = (train_start, train_end, oot_date) inclusive yyyymmdd strings.
# Train = expanding window from 20260101 up to (oot_date - 1 trading day).
# These are the OOT dates we care about for v2 stacking parity:
FOLD_MAP = {
    5:  "20260301",
    6:  "20260302",
    7:  "20260303",
    8:  "20260304",
    9:  "20260305",
    10: "20260306",
    11: "20260308",
    12: "20260309",
    13: "20260310",
    14: "20260311",
    15: "20260312",
    16: "20260313",
    17: "20260315",
}

WINDOW = 1000
STRIDE = 500
HORIZONS = ("1s", "5s", "10s")

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT = "TripleFusion_v2_BookCNN"
MLFLOW_DISABLED = (MLFLOW_URI == "" or os.environ.get("DISABLE_MLFLOW") == "1")

# ----------------------------------------------------------------------------
# Data alignment helpers
# ----------------------------------------------------------------------------

def load_event_day(date_str):
    p = EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not p.exists():
        return None
    d = np.load(p, mmap_mode="r")
    return {
        "events": d["events"],            # (N_e, 25)
        "ts":     d["timestamps"],        # (N_e,) int64 ns
        "y1":     d["labels_1s"],
        "y5":     d["labels_5s"],
        "y10":    d["labels_10s"],
    }


def load_book_day(date_str, depth):
    """Subset book_shape to first `depth` levels per side (assumes 10 per side)."""
    p = BOOK_DIR / f"{date_str}_book_norm.npz"
    if not p.exists():
        return None
    d = np.load(p, mmap_mode="r")
    bs = d["book_shape"]      # (N_b, 20)
    bd = d["book_dynamics"]   # (N_b, 10)
    bts = d["timestamps"]     # (N_b,) int64
    if depth >= 10:
        bs_sub = bs
    else:
        # Layout assumed: [bid_lvl_1..10, ask_lvl_1..10] → take first `depth` of each.
        bs_sub = np.concatenate([bs[:, :depth], bs[:, 10:10 + depth]], axis=1)
    return {"shape": bs_sub, "dyn": bd, "ts": bts}


def align_book_to_events(event_ts, book):
    """For each event timestamp, take the most recent book row at-or-before it."""
    idx = np.searchsorted(book["ts"], event_ts, side="right") - 1
    idx = np.clip(idx, 0, len(book["ts"]) - 1)
    return book["shape"][idx], book["dyn"][idx]


def build_features_for_day(date_str, depth):
    ev = load_event_day(date_str)
    if ev is None:
        return None
    bk = load_book_day(date_str, depth=depth)
    if bk is None:
        return None
    bs_aligned, bd_aligned = align_book_to_events(np.asarray(ev["ts"]), bk)
    feats = np.concatenate(
        [np.asarray(ev["events"]).astype(np.float32),
         bs_aligned.astype(np.float32),
         bd_aligned.astype(np.float32)],
        axis=1,
    )  # (N_e, 25 + 2*depth + 10)
    labels = np.stack([np.asarray(ev["y1"]),
                       np.asarray(ev["y5"]),
                       np.asarray(ev["y10"])], axis=1).astype(np.float32)
    return feats, labels, np.asarray(ev["ts"])


# ----------------------------------------------------------------------------
# Window dataset
# ----------------------------------------------------------------------------

class WindowDataset(Dataset):
    """Streaming windowed dataset.

    Holds only mmap views per date (cheap) and a list of window indices.
    Per-date book→event alignment is computed once (small int array per day).
    Windows materialize lazily in __getitem__.
    """

    def __init__(self, dates, depth, window=WINDOW, stride=STRIDE):
        self.window = window
        self.depth = depth
        self.feature_dim = None
        self.day_data = {}   # date -> dict(events_mmap, ev_y, bs_aligned, bd_aligned)
        self.windows = []    # list of (date, start_idx)

        for d in dates:
            ev_path = EVENT_DIR / f"{d}_mbo_events.npz"
            bs_cache = BOOK_CACHE / d / "book_shape.npy"
            bd_cache = BOOK_CACHE / d / "book_dyn.npy"
            if not ev_path.exists():
                log.warning(f"skip {d}: missing event data")
                continue
            if not (bs_cache.exists() and bd_cache.exists()):
                log.warning(f"skip {d}: missing aligned book cache (run precompute_book_aligned.py)")
                continue
            ev = np.load(ev_path, mmap_mode="r")
            bs = np.load(bs_cache, mmap_mode="r")   # (N, 20) float32 mmap
            bd = np.load(bd_cache, mmap_mode="r")   # (N, 10) float32 mmap

            n = ev["events"].shape[0]
            if bs.shape[0] != n or bd.shape[0] != n:
                log.warning(f"skip {d}: cache row mismatch (ev={n} bs={bs.shape[0]} bd={bd.shape[0]})")
                continue

            n_bs = bs.shape[1]
            bs_dim = (2 * depth) if depth < 10 else n_bs
            f_dim = ev["events"].shape[1] + bs_dim + bd.shape[1]
            if self.feature_dim is None:
                self.feature_dim = f_dim
            elif f_dim != self.feature_dim:
                log.warning(f"skip {d}: feature_dim mismatch {f_dim} vs {self.feature_dim}")
                continue

            self.day_data[d] = {
                "events":     ev["events"],          # mmap (N, 25)
                "y1":         ev["labels_1s"],
                "y5":         ev["labels_5s"],
                "y10":        ev["labels_10s"],
                "book_shape": bs,                    # mmap (N, 20) uncompressed
                "book_dyn":   bd,                    # mmap (N, 10) uncompressed
                "n":          n,
            }
            for s in range(0, n - window + 1, stride):
                self.windows.append((d, s))

        log.info(f"WindowDataset: {len(dates)} dates → {len(self.windows)} windows, F={self.feature_dim}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        date, s = self.windows[i]
        e = s + self.window
        D = self.day_data[date]
        ev_slice = np.asarray(D["events"][s:e]).astype(np.float32, copy=False)
        bs_full  = np.asarray(D["book_shape"][s:e])
        if self.depth < 10:
            bs_slice = np.concatenate([bs_full[:, :self.depth],
                                        bs_full[:, 10:10 + self.depth]], axis=1)
        else:
            bs_slice = bs_full
        bd_slice = np.asarray(D["book_dyn"][s:e])
        X = np.concatenate([ev_slice, bs_slice, bd_slice], axis=1)
        # Guard against NaN/Inf and extreme outliers (e.g. raw price levels).
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        np.clip(X, -1e6, 1e6, out=X)
        y = np.array([D["y1"][e - 1], D["y5"][e - 1], D["y10"][e - 1]], dtype=np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=10.0, neginf=-10.0)
        np.clip(y, -10.0, 10.0, out=y)
        return torch.from_numpy(X), torch.from_numpy(y)


# ----------------------------------------------------------------------------
# Model — small dilated 1D CNN
# ----------------------------------------------------------------------------

class DilatedCNNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm(self.conv(x)))
        h = self.drop(h)
        return h + self.proj(x)


class BookCNN(nn.Module):
    def __init__(self, n_features, n_horizons=3, hidden=64, dropout=0.1):
        super().__init__()
        # LayerNorm over features handles raw/unnormalized inputs.
        self.input_norm = nn.LayerNorm(n_features)
        self.input_proj = nn.Conv1d(n_features, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            DilatedCNNBlock(hidden, hidden, dilation=1, dropout=dropout),
            DilatedCNNBlock(hidden, hidden, dilation=2, dropout=dropout),
            DilatedCNNBlock(hidden, hidden, dilation=4, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                          nn.Dropout(dropout), nn.Linear(hidden, 1))
            for _ in range(n_horizons)
        ])
        self.embed_dim = hidden

    def forward(self, x):
        # x: (B, W, F) — LN over feature dim (last), then transpose for Conv1d.
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        h = self.input_proj(x)
        h = self.blocks(h)
        emb = self.pool(h).squeeze(-1)
        preds = torch.cat([head(emb) for head in self.heads], dim=1)
        return preds, emb


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    P, Y, E = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            p, e = model(x)
            P.append(p.cpu().numpy())
            Y.append(y.numpy())
            E.append(e.cpu().numpy())
    P = np.concatenate(P) if P else np.zeros((0, 3), dtype=np.float32)
    Y = np.concatenate(Y) if Y else np.zeros((0, 3), dtype=np.float32)
    E = np.concatenate(E) if E else np.zeros((0, 64), dtype=np.float32)
    ics = []
    for h in range(3):
        if P.shape[0] > 5:
            ic = spearmanr(P[:, h], Y[:, h]).correlation
            ics.append(0.0 if (ic is None or np.isnan(ic)) else float(ic))
        else:
            ics.append(0.0)
    return P, Y, E, ics


def train_one_fold(args, fold_id, oot_date, train_dates, device):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"--- FOLD {fold_id:02d} | OOT {oot_date} | train {train_dates[0]}..{train_dates[-1]} ({len(train_dates)}d) ---")

    train_ds = WindowDataset(train_dates, depth=args.depth, stride=args.train_stride)
    oot_ds   = WindowDataset([oot_date], depth=args.depth, stride=STRIDE)
    if len(train_ds) == 0 or len(oot_ds) == 0:
        log.warning(f"fold {fold_id:02d}: empty dataset, skipping")
        return None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    oot_loader = DataLoader(oot_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = BookCNN(n_features=train_ds.feature_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    crit = nn.MSELoss()

    best_ic10 = -999
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        nb = 0
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            p, _ = model(x)
            loss = crit(p, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        train_loss = ep_loss / max(nb, 1)
        P, Y, E, ics = evaluate(model, oot_loader, device)
        log.info(f"  epoch {ep+1}/{args.epochs} loss={train_loss:.5f} ic_1s={ics[0]:+.4f} ic_5s={ics[1]:+.4f} ic_10s={ics[2]:+.4f} ({time.time()-t0:.0f}s)")
        if ics[2] > best_ic10:
            best_ic10 = ics[2]
            np.savez(
                out_dir / f"fold_{fold_id:02d}_oot_predictions.npz",
                predictions=P,
                labels=Y,
                horizons=np.array(list(HORIZONS)),
                oot_files=np.array([str(EVENT_DIR / f"{oot_date}_mbo_events.npz")]),
                embeddings=E,
            )
        if MLFLOW_AVAILABLE and not MLFLOW_DISABLED:
            mlflow.log_metric(f"fold{fold_id:02d}_ic_1s",  ics[0], step=ep)
            mlflow.log_metric(f"fold{fold_id:02d}_ic_5s",  ics[1], step=ep)
            mlflow.log_metric(f"fold{fold_id:02d}_ic_10s", ics[2], step=ep)
            mlflow.log_metric(f"fold{fold_id:02d}_loss",   train_loss, step=ep)

    # Save fold analysis for parity with cnn_mamba_v2 layout
    analysis = {
        "fold": fold_id,
        "oot_date": oot_date,
        "best_ic_10s": best_ic10,
        "n_train_dates": len(train_dates),
        "feature_dim": int(train_ds.feature_dim),
        "depth": args.depth,
    }
    (out_dir / f"fold_{fold_id:02d}_analysis.json").write_text(json.dumps(analysis, indent=2))
    return best_ic10


def get_train_dates(oot_date, lookback_start="20260101", max_days=None):
    """Trading dates in EVENT_DIR strictly before oot_date and ≥ lookback_start.
    If max_days is set, keep only the MOST RECENT max_days dates (60-day WF window)."""
    files = sorted(EVENT_DIR.glob("*_mbo_events.npz"))
    dates = [f.name[:8] for f in files]
    dates = [d for d in dates if (lookback_start <= d < oot_date)]
    if max_days is not None and len(dates) > max_days:
        dates = dates[-max_days:]
    return dates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=10, help="levels per side (5/10)")
    p.add_argument("--folds", type=str, default="5,6,7,8,9", help="comma-separated fold ids")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lookback-start", type=str, default="20260101")
    p.add_argument("--max-train-days", type=int, default=60,
                   help="Cap training to the most recent N trading days (60 = WF default).")
    p.add_argument("--train-stride", type=int, default=1000,
                   help="Stride for training windows (larger = fewer windows). Eval always uses 500.")
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"host={HOST} device={device} ROOT={ROOT}")
    log.info(f"depth={args.depth} folds={args.folds} epochs={args.epochs} hidden={args.hidden}")

    if MLFLOW_AVAILABLE and not MLFLOW_DISABLED:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow.start_run(run_name=f"v2_book_cnn_depth{args.depth}_{time.strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params({
                "branch": "book_cnn",
                "depth": args.depth,
                "window": WINDOW,
                "stride": STRIDE,
                "epochs_per_fold": args.epochs,
                "batch_size": args.batch_size,
                "hidden": args.hidden,
                "lr": args.lr,
                "host": HOST,
                "lookback_start": args.lookback_start,
                "folds": args.folds,
            })
        except Exception as e:
            log.warning(f"MLflow init failed: {e}")

    fold_ids = [int(x.strip()) for x in args.folds.split(",")]
    summary = {}
    for fid in fold_ids:
        oot = FOLD_MAP.get(fid)
        if oot is None:
            log.warning(f"fold {fid} not in FOLD_MAP, skipping")
            continue
        train_dates = get_train_dates(oot, lookback_start=args.lookback_start,
                                       max_days=args.max_train_days)
        if not train_dates:
            log.warning(f"fold {fid}: no train dates < {oot}, skipping")
            continue
        ic = train_one_fold(args, fid, oot, train_dates, device)
        summary[fid] = ic

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "summary.json").write_text(json.dumps({
        "best_ic_10s_per_fold": summary,
        "args": vars(args),
        "host": HOST,
    }, indent=2))

    if MLFLOW_AVAILABLE and not MLFLOW_DISABLED:
        try:
            mlflow.end_run()
        except Exception:
            pass

    # Concat IC across folds (the only IC that matters)
    try:
        Ps, Ys = [], []
        for fid in summary.keys():
            f = Path(args.output_dir) / f"fold_{fid:02d}_oot_predictions.npz"
            if f.exists():
                d = np.load(f)
                Ps.append(d["predictions"]); Ys.append(d["labels"])
        if Ps:
            P = np.concatenate(Ps); Y = np.concatenate(Ys)
            for h, name in enumerate(HORIZONS):
                ic = spearmanr(P[:, h], Y[:, h]).correlation
                log.info(f"CONCAT IC {name}: {ic:+.4f} (n={P.shape[0]})")
    except Exception as e:
        log.warning(f"concat IC compute failed: {e}")

    log.info("v2 book CNN training complete.")


if __name__ == "__main__":
    sys.exit(main())
