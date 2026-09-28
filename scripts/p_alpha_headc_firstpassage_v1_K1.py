#!/usr/bin/env python3
"""
p_alpha_headc_firstpassage_v1.py — P-Alpha provisional first-pass (HC #497 head-C)

PURPOSE
  Bounded provisional evidence on whether the cumulative-K-tick first-passage
  CLASSIFICATION paradigm has edge BEFORE the full v3.5 multi-head training
  (which is blocked ~20h on Jupiter label build).

  Single-head BCE classifier on smart_v3 25-feature input. Cumulative ±K-tick
  first-passage label with K=2, h=5s (aligned with microstructure gate best cell).

LABEL
  For each strided window-end event at time t with cumulative signed price
  walk c(τ) reconstructed from labels_1s diffs over τ∈(t, t+h]:
    label = 1   if c first crosses +K within h
    label = 0   if c first crosses -K within h
    label = NaN if neither boundary touched within h (DROPPED)

REUSES
  - alpha_discovery/deep_models/train_cnn_mamba_v2.py  (CNNMambaV2 backbone)
  - alpha_discovery/deep_models/fifo_market_replay.py  (FIFOReplayEngine)
  - scripts/build_continuation_labels.py               (first-passage logic ported)

OUTPUT DIR  /home/nick/Lvl3Quant/output/p_alpha_headc_v1/
  per-fold:  fold_{i}_oot_{date}.npz  (preds, labels, ts_ns)
             fold_{i}_model.pt
  summary:   summary.json            (aggregate IC/AUC/baseline + FIFO grade)
             per_day_metrics.csv
             fifo_gate_results.csv

MLFLOW EXPERIMENT  p_alpha_headc_firstpassage_v1

WALL-CAP  90 min (HC #428-bounded). Gracefully skips remaining folds if exceeded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import logging
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

# ── Paths / config ──────────────────────────────────────────────────────────
LVL3_ROOT = Path("/home/nick/Lvl3Quant")
SMART_V3_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
OUT_DIR = LVL3_ROOT / "output" / "p_alpha_headc_v1_K1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = LVL3_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Microstructure OOT date list (the 35 dates used by the standalone-REJECT grade).
MICRO_LEADERBOARD = LVL3_ROOT / "output" / "p_microstructure_fifo_v1" / "best_cell_per_day.csv"

# Hyperparameters — kept small to fit the 90-min wall-cap on RTX 3090.
WINDOW_SIZE = 3000          # same as v2 production
STRIDE = 2000               # 8x sparser than v2 production (250) → much faster training
BATCH_SIZE = 128
LR = 3e-4
EPOCHS_PER_FOLD = 2         # provisional — 2 epochs is enough to detect any edge
WF_WINDOW_DAYS = 60         # SLIDING (HC #0)
HORIZON_S = 5.0             # head-C horizon
K_TICKS = 1.0               # head-C K
DROP_NO_TOUCH = True        # NaN events whose path didn't reach ±K within h
GRAD_CLIP = 1.0
WARMUP_STEPS = 200

WALL_CAP_MIN_DEFAULT = 90.0

MLFLOW_EXPERIMENT = "p_alpha_headc_firstpassage_v1_K1"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

# ── Logging ─────────────────────────────────────────────────────────────────
ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"p_alpha_headc_v1_{ts_now}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("p_alpha_headc")

# ── Import v2 backbone (smart_v3 feature mode) ──────────────────────────────
# Must set the feature-set env BEFORE importing.
os.environ["MAMBA_FEATURE_SET"] = "smart_v3"
sys.path.insert(0, str(LVL3_ROOT / "alpha_discovery" / "deep_models"))
import train_cnn_mamba_v2 as v2  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402

# ── Process-wide label cache (sliding folds reuse the same day labels) ──────
_GLOBAL_LABEL_CACHE: Dict = {}


# ── First-passage label (ported from build_continuation_labels.py) ──────────
def _horizon_end_idx(ts_ns: np.ndarray, h_ns: int) -> np.ndarray:
    """For each i return smallest j > i with ts_ns[j] >= ts_ns[i]+h_ns; else N."""
    n = len(ts_ns)
    out = np.empty(n, dtype=np.int64)
    j = 0
    for i in range(n):
        if j <= i:
            j = i + 1
        target = ts_ns[i] + h_ns
        while j < n and ts_ns[j] < target:
            j += 1
        out[i] = j
    return out


def first_passage_label_at_positions(
    ts_ns: np.ndarray,
    labels_1s: np.ndarray,
    positions: np.ndarray,
    K: float,
    horizon_s: float,
) -> np.ndarray:
    """First-passage binary label at given event positions only.

    Returns float32 array with values {0.0, 1.0, NaN}. NaN = neither ±K touched.
    Cumulative move reconstructed via diff(labels_1s) per-event increments
    (same approximation as build_continuation_labels.head_c_first_passage_K).
    """
    incr = np.diff(labels_1s, prepend=labels_1s[0])
    incr = np.where(np.isnan(incr), 0.0, incr).astype(np.float32)

    h_ns = int(horizon_s * 1e9)
    h_end = _horizon_end_idx(ts_ns, h_ns)

    out = np.full(len(positions), np.nan, dtype=np.float32)
    for idx_out, i in enumerate(positions):
        end = h_end[i]
        if end <= i + 1:
            continue
        cum = np.cumsum(incr[i + 1: end])
        pos_hit = np.flatnonzero(cum >= K)
        neg_hit = np.flatnonzero(cum <= -K)
        fp = int(pos_hit[0]) if len(pos_hit) else (1 << 62)
        fn = int(neg_hit[0]) if len(neg_hit) else (1 << 62)
        if fp == (1 << 62) and fn == (1 << 62):
            continue  # NaN
        out[idx_out] = 1.0 if fp < fn else 0.0
    return out


# ── Dataset: lazy-loaded windows with on-the-fly first-passage labels ───────
class HeadCDataset(Dataset):
    """Streams smart_v3 windows, computes first-passage label at last position.

    Caches the most recent N days in RAM (events + ts + labels_1s + label cache).
    """

    def __init__(
        self,
        npz_files: List[Path],
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        feature_stats: Optional[Dict] = None,
        cache_size: int = 4,
    ):
        self.npz_files = list(npz_files)
        self.window_size = window_size
        self.stride = stride
        self.cache_size = cache_size

        if feature_stats is None:
            self._compute_stats()
        else:
            self.feature_mean = feature_stats["mean"]
            self.feature_std = feature_stats["std"]

        self._cache: Dict[int, Dict] = {}
        self._cache_order: List[int] = []
        self._build_index()

    def _compute_stats(self):
        log.info(f"Computing smart_v3 normalize stats over {len(self.npz_files)} files…")
        n_feat = 25
        tot = np.zeros(n_feat, dtype=np.float64)
        totsq = np.zeros(n_feat, dtype=np.float64)
        cnt = 0
        for f in self.npz_files:
            with np.load(f, allow_pickle=False) as z:
                ev = z["events"].astype(np.float64)
            tot += ev.sum(axis=0)
            totsq += (ev ** 2).sum(axis=0)
            cnt += len(ev)
        self.feature_mean = (tot / cnt).astype(np.float32)
        var = (totsq / cnt) - (self.feature_mean.astype(np.float64) ** 2)
        self.feature_std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)

    def get_feature_stats(self) -> Dict:
        return {"mean": self.feature_mean, "std": self.feature_std}

    def _build_index(self):
        self.sample_index: List[Tuple[int, int]] = []  # (day_idx, label_pos)
        # Pre-compute labels per day so we can drop NaN at index time.
        for day_idx, f in enumerate(self.npz_files):
            cache_key = (str(f), self.window_size, self.stride, K_TICKS, HORIZON_S)
            cached = _GLOBAL_LABEL_CACHE.get(cache_key)
            if cached is not None:
                label_positions, labels = cached
            else:
                with np.load(f, allow_pickle=False) as z:
                    ts = z["timestamps"]
                    l1 = z["labels_1s"]
                    n_events = len(ts)
                label_positions = np.arange(
                    self.window_size - 1,
                    n_events,
                    self.stride,
                    dtype=np.int64,
                )
                if len(label_positions) == 0:
                    _GLOBAL_LABEL_CACHE[cache_key] = (label_positions, np.array([], dtype=np.float32))
                    continue
                labels = first_passage_label_at_positions(
                    ts, l1, label_positions, K_TICKS, HORIZON_S,
                )
                _GLOBAL_LABEL_CACHE[cache_key] = (label_positions, labels)
            for pos, lbl in zip(label_positions, labels):
                if DROP_NO_TOUCH and np.isnan(lbl):
                    continue
                self.sample_index.append((day_idx, int(pos)))
            # also cache labels for fast __getitem__
            self._cache_labels_for_day(day_idx, label_positions, labels)
        log.info(
            f"HeadCDataset: {len(self.npz_files)} days, {len(self.sample_index)} valid samples "
            f"(W={self.window_size}, S={self.stride}, K={K_TICKS}, h={HORIZON_S}s)"
        )

    def _cache_labels_for_day(self, day_idx, positions, labels):
        # Store sparse position→label dict
        if not hasattr(self, "_day_label_map"):
            self._day_label_map: Dict[int, Dict[int, float]] = {}
        self._day_label_map[day_idx] = {int(p): float(l) for p, l in zip(positions, labels)}

    def _load_day(self, day_idx: int) -> Dict:
        if day_idx in self._cache:
            self._cache_order.remove(day_idx)
            self._cache_order.append(day_idx)
            return self._cache[day_idx]

        with np.load(self.npz_files[day_idx], allow_pickle=False) as z:
            events = z["events"].astype(np.float32)
            ts = z["timestamps"].copy()
        events = (events - self.feature_mean) / (self.feature_std + 1e-8)
        d = {"events": events, "ts": ts}

        self._cache[day_idx] = d
        self._cache_order.append(day_idx)
        while len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            del self._cache[old]
        return d

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        day_idx, label_pos = self.sample_index[idx]
        start = label_pos - self.window_size + 1
        end = label_pos + 1
        d = self._load_day(day_idx)
        ev = d["events"][start:end]                # (W, 25)
        ts = int(d["ts"][label_pos])
        lbl = self._day_label_map[day_idx][label_pos]
        return (
            torch.from_numpy(ev),
            torch.tensor(lbl, dtype=torch.float32),
            torch.tensor(ts, dtype=torch.int64),
            torch.tensor(day_idx, dtype=torch.int64),
            torch.tensor(label_pos, dtype=torch.int64),
        )


def _collate(batch):
    ev = torch.stack([b[0] for b in batch], dim=0)
    lb = torch.stack([b[1] for b in batch], dim=0)
    ts = torch.stack([b[2] for b in batch], dim=0)
    di = torch.stack([b[3] for b in batch], dim=0)
    po = torch.stack([b[4] for b in batch], dim=0)
    return ev, lb, ts, di, po


# ── Single-head model: CNNMambaV2 backbone + 1 logit ────────────────────────
class HeadCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = v2.CNNMambaV2(n_targets=1)
        # n_targets=1 backbone head outputs 1 scalar — perfect for BCEWithLogits.

    def forward(self, ev):
        logits = self.backbone(ev)  # (B, 1)
        return logits.squeeze(-1)


# ── OOT discovery ───────────────────────────────────────────────────────────
def discover_2026_dates() -> List[str]:
    files = sorted(SMART_V3_DIR.glob("2026*_mbo_events.npz"))
    return [f.name.split("_")[0] for f in files]


def discover_oot_dates_from_micro() -> List[str]:
    """Return the 35 OOT dates the microstructure FIFO grade used."""
    if not MICRO_LEADERBOARD.exists():
        return []
    import csv
    dates = set()
    with open(MICRO_LEADERBOARD) as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            dates.add(row["date"])
    return sorted(dates)


# ── IC / AUC / accuracy ─────────────────────────────────────────────────────
def spearman_ic(p: np.ndarray, y: np.ndarray) -> float:
    valid = ~(np.isnan(p) | np.isnan(y))
    if valid.sum() < 20:
        return float("nan")
    from scipy.stats import spearmanr
    r, _ = spearmanr(p[valid], y[valid])
    return float(r) if not np.isnan(r) else 0.0


def roc_auc(p: np.ndarray, y: np.ndarray) -> float:
    valid = ~(np.isnan(p) | np.isnan(y))
    if valid.sum() < 20:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y[valid], p[valid]))
    except Exception:
        return float("nan")


# ── Training one fold ───────────────────────────────────────────────────────
def train_fold(
    fold_idx: int,
    oot_date: str,
    train_files: List[Path],
    oot_file: Path,
    device: torch.device,
    wall_deadline: float,
    mlflow_run=None,
    shared_feat_stats: Optional[Dict] = None,
) -> Optional[Dict]:
    log.info(f"=== Fold {fold_idx} | OOT {oot_date} | train_days={len(train_files)} ===")

    train_ds = HeadCDataset(train_files, feature_stats=shared_feat_stats)
    feat_stats = train_ds.get_feature_stats()
    oot_ds = HeadCDataset([oot_file], feature_stats=feat_stats)

    if len(train_ds) == 0 or len(oot_ds) == 0:
        log.warning(f"  Fold {fold_idx}: empty dataset (train={len(train_ds)} oot={len(oot_ds)}) — skip")
        return None

    # Class balance check
    train_labels = np.array([train_ds._day_label_map[d][p] for (d, p) in train_ds.sample_index])
    base_rate = float(train_labels.mean())
    log.info(f"  Train class balance: pos_rate={base_rate:.3f} n={len(train_labels)}")

    # PerDaySampler keeps the LRU day cache hot (shuffles day order, sequential
    # within day). num_workers=0 to avoid fork-OOM on 32GB Neptune.
    train_sampler = v2.FileSequentialSampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=0, pin_memory=True, collate_fn=_collate, drop_last=True,
    )
    oot_loader = DataLoader(
        oot_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=_collate,
    )

    model = HeadCModel().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = v2.WarmupCosineScheduler(optim, warmup_steps=WARMUP_STEPS,
                                     total_steps=len(train_loader) * EPOCHS_PER_FOLD)
    scaler = torch.amp.GradScaler("cuda")
    bce = nn.BCEWithLogitsLoss()

    step = 0
    t_fold_start = time.time()
    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for ev, lb, _ts, _di, _po in train_loader:
            if time.time() > wall_deadline:
                log.warning(f"  Wall deadline during train epoch {epoch} step {step}")
                break
            ev = ev.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(ev)
                loss = bce(logits, lb)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optim)
            scaler.update()
            sched.step()
            step += 1
            ep_loss += float(loss.item())
            n_batches += 1
        log.info(f"  epoch {epoch}: train_loss={ep_loss/max(n_batches,1):.4f} ({n_batches} batches)")
        if time.time() > wall_deadline:
            break

    # OOT inference
    model.eval()
    preds_all, labels_all, ts_all = [], [], []
    with torch.no_grad():
        for ev, lb, ts, _di, _po in oot_loader:
            ev = ev.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(ev)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds_all.append(probs)
            labels_all.append(lb.numpy())
            ts_all.append(ts.numpy())
    preds = np.concatenate(preds_all).astype(np.float32)
    labels = np.concatenate(labels_all).astype(np.float32)
    ts_ns = np.concatenate(ts_all).astype(np.int64)

    ic = spearman_ic(preds, labels)
    auc = roc_auc(preds, labels)
    pos_rate_oot = float(labels.mean()) if len(labels) else float("nan")
    acc_majority = max(pos_rate_oot, 1 - pos_rate_oot)
    # Threshold @ 0.5 accuracy
    acc05 = float(((preds >= 0.5).astype(np.float32) == labels).mean()) if len(labels) else float("nan")

    res = {
        "fold": fold_idx,
        "oot_date": oot_date,
        "n_train_days": len(train_files),
        "n_train_samples": len(train_ds),
        "n_oot_samples": int(len(preds)),
        "train_pos_rate": base_rate,
        "oot_pos_rate": pos_rate_oot,
        "ic_spearman": ic,
        "auc": auc,
        "acc_majority_baseline": acc_majority,
        "acc_at_0.5": acc05,
        "fold_wall_sec": time.time() - t_fold_start,
    }
    log.info(f"  Fold {fold_idx} OOT {oot_date}: n={len(preds)} IC={ic:.4f} AUC={auc:.4f} "
             f"acc@0.5={acc05:.3f} (baseline={acc_majority:.3f}) "
             f"posrate={pos_rate_oot:.3f} t={res['fold_wall_sec']:.1f}s")

    # Persist per-fold predictions
    pred_path = OUT_DIR / f"fold_{fold_idx:02d}_oot_{oot_date}.npz"
    np.savez_compressed(pred_path,
                        predictions=preds, labels=labels, ts_ns=ts_ns,
                        oot_date=np.array(oot_date))
    log.info(f"  Saved preds to {pred_path.name}")

    if mlflow_run is not None:
        import mlflow
        mlflow.log_metrics({
            f"fold{fold_idx:02d}_ic": ic if not np.isnan(ic) else 0.0,
            f"fold{fold_idx:02d}_auc": auc if not np.isnan(auc) else 0.0,
            f"fold{fold_idx:02d}_acc05": acc05 if not np.isnan(acc05) else 0.0,
            f"fold{fold_idx:02d}_n_oot": float(len(preds)),
            f"fold{fold_idx:02d}_posrate": pos_rate_oot if not np.isnan(pos_rate_oot) else 0.0,
        })

    # Cleanup
    del model, train_ds, oot_ds, train_loader, oot_loader
    torch.cuda.empty_cache()
    return res


# ── Simple FIFO grade (lightweight wrapper around canonical engine) ─────────
def run_simple_fifo_grade(fold_results: List[Dict], wall_deadline: float) -> Dict:
    """Apply head-C gate (prob >= 0.7 for long, <= 0.3 for short) and route
    through canonical FIFOReplayEngine. NOT a microstructure composition —
    that requires xgb gate heads which we skip here (handled by Pass 2 if time).

    Per HC #428 R2: TP/SL bounded by horizon. h=5s, so TP=SL=K_TICKS (=2 ticks)
    and hold≤1.5*h=7.5s, cancel≤h=5s.
    """
    # Lazy import of canonical FIFO engine
    fifo_path = LVL3_ROOT / "alpha_discovery" / "deep_models" / "fifo_market_replay.py"
    spec = importlib.util.spec_from_file_location("fifo_market_replay", str(fifo_path))
    fmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fmod)
    FIFOReplayEngine = fmod.FIFOReplayEngine

    TP_TICKS = K_TICKS  # ≤ p90 realized MFE within h, K is also the label boundary
    SL_TICKS = K_TICKS
    CANCEL_NS = int(HORIZON_S * 1e9)
    HOLD_NS = int(1.5 * HORIZON_S * 1e9)
    TAU_LONG = 0.70
    TAU_SHORT = 0.30

    per_day_rows = []
    for fr in fold_results:
        if time.time() > wall_deadline:
            log.warning("  FIFO grade: wall deadline reached")
            break
        date = fr["oot_date"]
        pred_npz = OUT_DIR / f"fold_{fr['fold']:02d}_oot_{date}.npz"
        if not pred_npz.exists():
            continue
        with np.load(pred_npz, allow_pickle=False) as z:
            preds = z["predictions"]
            ts_ns = z["ts_ns"]

        # Build signals: long if prob>=TAU_LONG, short if prob<=TAU_SHORT
        long_mask = preds >= TAU_LONG
        short_mask = preds <= TAU_SHORT
        sig_idx = np.where(long_mask | short_mask)[0]
        if len(sig_idx) == 0:
            per_day_rows.append({
                "date": date, "n_signals": 0, "n_trades": 0,
                "net_ticks_per_trade": float("nan"),
                "sum_net_ticks": 0.0, "win_rate": float("nan"),
                "sharpe": float("nan"), "pf": float("nan"),
            })
            continue

        signals = []
        for i in sig_idx:
            signals.append({
                "ts_ns": int(ts_ns[i]),
                "direction": "long" if preds[i] >= TAU_LONG else "short",
                "strength": float(abs(preds[i] - 0.5) * 2.0),
            })

        try:
            eng = FIFOReplayEngine(date=date, cancel_after_ns=CANCEL_NS, max_hold_ns=HOLD_NS)
            trades = eng.simulate(signals=signals, tp_ticks=TP_TICKS,
                                  sl_ticks=SL_TICKS, order_type="limit")
        except Exception as e:
            log.warning(f"  FIFO {date}: engine error: {e}")
            per_day_rows.append({"date": date, "n_signals": len(signals),
                                 "n_trades": 0, "error": str(e)[:80]})
            continue

        if not trades:
            per_day_rows.append({
                "date": date, "n_signals": len(signals), "n_trades": 0,
                "net_ticks_per_trade": float("nan"),
                "sum_net_ticks": 0.0, "win_rate": float("nan"),
                "sharpe": float("nan"), "pf": float("nan"),
            })
            continue

        # trades is a list of dataclass or dict — try both
        net_ticks = []
        for t in trades:
            if hasattr(t, "pnl_ticks_net"):
                net_ticks.append(float(t.pnl_ticks_net))
            elif isinstance(t, dict) and "pnl_ticks_net" in t:
                net_ticks.append(float(t["pnl_ticks_net"]))
        net_ticks = np.array(net_ticks, dtype=np.float64)
        if len(net_ticks) == 0:
            per_day_rows.append({"date": date, "n_signals": len(signals), "n_trades": 0})
            continue

        wins = net_ticks[net_ticks > 0]
        losses = net_ticks[net_ticks < 0]
        wr = float((net_ticks > 0).mean())
        pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
        std = float(net_ticks.std(ddof=1)) if len(net_ticks) > 1 else float("nan")
        sh = float(net_ticks.mean() / std) if std and std > 0 else float("nan")
        per_day_rows.append({
            "date": date,
            "n_signals": len(signals),
            "n_trades": len(net_ticks),
            "net_ticks_per_trade": float(net_ticks.mean()),
            "sum_net_ticks": float(net_ticks.sum()),
            "win_rate": wr,
            "pf": pf,
            "sharpe": sh,
        })
        log.info(f"  FIFO {date}: n_sig={len(signals)} n_trades={len(net_ticks)} "
                 f"ntpt={net_ticks.mean():+.3f} WR={wr:.3f} PF={pf:.2f} SH={sh:+.2f}")

    # Aggregate
    import csv
    csv_path = OUT_DIR / "fifo_gate_results.csv"
    if per_day_rows:
        keys = sorted({k for r in per_day_rows for k in r.keys()})
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in per_day_rows:
                w.writerow(r)

    valid = [r for r in per_day_rows if r.get("n_trades", 0) > 0
             and not (isinstance(r.get("net_ticks_per_trade"), float)
                      and np.isnan(r["net_ticks_per_trade"]))]
    if not valid:
        return {"status": "no_trades", "n_days": len(per_day_rows)}

    ntpts = np.array([r["net_ticks_per_trade"] for r in valid])
    n_trades = np.array([r["n_trades"] for r in valid])
    pos_days = int((ntpts > 0).sum())

    summary = {
        "n_days_with_trades": len(valid),
        "n_days_total": len(per_day_rows),
        "n_days_positive": pos_days,
        "pct_days_positive": pos_days / len(valid),
        "mean_ntpt": float(ntpts.mean()),
        "median_ntpt": float(np.median(ntpts)),
        "mean_trades_per_day": float(n_trades.mean()),
        "commission_floor_ticks": 0.376,
        "viable_per_hc494_r1": bool(
            ntpts.mean() > 0.376 and len(valid) >= 30 and n_trades.mean() >= 5
        ),
        "csv": str(csv_path),
    }
    return summary


# ── Main ─────────────────────────────────────────────────────────────────────
def build_folds(all_2026_dates: List[str], oot_dates: List[str], wf_days: int):
    """Sliding window: for each oot_date d, train = up to wf_days dates strictly before d."""
    all_set = set(all_2026_dates)
    folds = []
    sorted_all = sorted(all_2026_dates)
    for i, oot in enumerate(oot_dates):
        if oot not in all_set:
            continue
        train = [d for d in sorted_all if d < oot][-wf_days:]
        if len(train) < 30:
            log.warning(f"  Skipping OOT {oot}: only {len(train)} train days available (<30)")
            continue
        folds.append({
            "fold_idx": i,
            "oot_date": oot,
            "train_dates": train,
        })
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-cap-min", type=float, default=WALL_CAP_MIN_DEFAULT)
    ap.add_argument("--max-folds", type=int, default=0, help="0 = all OOT folds")
    ap.add_argument("--smoke", action="store_true", help="1 fold, 1 epoch, tiny")
    args = ap.parse_args()

    t0 = time.time()
    wall_deadline = t0 + args.wall_cap_min * 60

    log.info(f"=== p_alpha_headc_firstpassage_v1 START === wall_cap={args.wall_cap_min}min")
    log.info(f"Log file: {LOG_FILE}")
    log.info(f"Output dir: {OUT_DIR}")

    if args.smoke:
        global EPOCHS_PER_FOLD, STRIDE
        EPOCHS_PER_FOLD = 1

    # Device
    if not torch.cuda.is_available():
        log.error("CUDA not available — abort")
        sys.exit(1)
    device = torch.device("cuda")
    log.info(f"Device: {torch.cuda.get_device_name(0)}  "
             f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # Dates
    all_2026 = discover_2026_dates()
    log.info(f"All 2026 smart_v3 dates available: {len(all_2026)}")
    oot_dates = discover_oot_dates_from_micro()
    if not oot_dates:
        log.error("No microstructure OOT dates found — falling back to last 35 2026 dates")
        oot_dates = all_2026[-35:]
    log.info(f"OOT dates from microstructure grade: {len(oot_dates)} "
             f"({oot_dates[0]}..{oot_dates[-1]})")

    folds = build_folds(all_2026, oot_dates, WF_WINDOW_DAYS)
    log.info(f"Built {len(folds)} folds")
    if args.smoke:
        folds = folds[:1]
    if args.max_folds > 0:
        folds = folds[:args.max_folds]
    log.info(f"Will train {len(folds)} folds")

    # MLflow
    mlflow_run = None
    mlflow_run_id = None
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow_run = mlflow.start_run(run_name=f"headc_firstpass_{ts_now}")
        mlflow_run_id = mlflow_run.info.run_id
        mlflow.log_params({
            "K_ticks": K_TICKS,
            "horizon_s": HORIZON_S,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs_per_fold": EPOCHS_PER_FOLD,
            "wf_window_days": WF_WINDOW_DAYS,
            "n_folds_planned": len(folds),
            "wall_cap_min": args.wall_cap_min,
            "feature_set": "smart_v3",
            "model": "CNNMambaV2 single-head BCE",
        })
        log.info(f"MLflow run started: {mlflow_run_id}")
    except Exception as e:
        log.warning(f"MLflow init failed: {e} — continuing without tracking")

    # Pre-compute feature stats ONCE on a representative subset (last 20 days
    # of available training data). Reused for all folds — smart_v3 is already
    # smart-normalized by the preprocessor, so per-fold drift is negligible.
    shared_feat_stats = None
    if folds:
        ref_dates = folds[-1]["train_dates"][-20:]  # most recent 20 train days
        ref_files = [SMART_V3_DIR / f"{d}_mbo_events.npz" for d in ref_dates if (SMART_V3_DIR / f"{d}_mbo_events.npz").exists()]
        if ref_files:
            log.info(f"Computing shared feature stats over {len(ref_files)} reference days…")
            t_stats = time.time()
            n_feat = 25
            tot = np.zeros(n_feat, dtype=np.float64)
            totsq = np.zeros(n_feat, dtype=np.float64)
            cnt = 0
            for f in ref_files:
                with np.load(f, allow_pickle=False) as z:
                    ev = z["events"].astype(np.float64)
                tot += ev.sum(axis=0)
                totsq += (ev ** 2).sum(axis=0)
                cnt += len(ev)
            fmean = (tot / cnt).astype(np.float32)
            var = (totsq / cnt) - (fmean.astype(np.float64) ** 2)
            fstd = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
            shared_feat_stats = {"mean": fmean, "std": fstd}
            log.info(f"Shared stats computed in {time.time()-t_stats:.1f}s")

    # Train all folds
    fold_results: List[Dict] = []
    for spec in folds:
        if time.time() > wall_deadline:
            log.warning(f"Wall deadline reached before fold {spec['fold_idx']} — stopping training")
            break
        train_files = [SMART_V3_DIR / f"{d}_mbo_events.npz" for d in spec["train_dates"]]
        oot_file = SMART_V3_DIR / f"{spec['oot_date']}_mbo_events.npz"
        train_files = [p for p in train_files if p.exists()]
        if not oot_file.exists():
            log.warning(f"  OOT file missing: {oot_file.name}")
            continue
        # RESUME (added 2026-06-11): skip folds whose predictions already exist,
        # so watcher relaunches continue past the wall cap instead of redoing folds 0-N.
        pred_path = OUT_DIR / f"fold_{spec[fold_idx]:02d}_oot_{spec[oot_date]}.npz"
        if pred_path.exists():
            try:
                _z = np.load(pred_path)
                _p = _z["predictions"].astype(np.float32)
                _l = _z["labels"].astype(np.float32)
                _pos = float(_l.mean()) if len(_l) else float("nan")
                fold_results.append({
                    "fold": spec["fold_idx"], "oot_date": spec["oot_date"],
                    "n_train_days": len(train_files), "n_train_samples": -1,
                    "n_oot_samples": int(len(_p)), "train_pos_rate": float("nan"),
                    "oot_pos_rate": _pos,
                    "ic_spearman": spearman_ic(_p, _l), "auc": roc_auc(_p, _l),
                    "acc_majority_baseline": max(_pos, 1 - _pos),
                    "acc_at_0.5": float(((_p >= 0.5).astype(np.float32) == _l).mean()) if len(_l) else float("nan"),
                    "fold_wall_sec": 0.0,
                })
                log.info(f"  [RESUME] Fold {spec[fold_idx]} OOT {spec[oot_date]}: existing preds found - skipping retrain")
                continue
            except Exception as _e:
                log.warning(f"  [RESUME] failed to load {pred_path.name} ({_e}) - retraining fold")
        try:
            res = train_fold(
                fold_idx=spec["fold_idx"],
                oot_date=spec["oot_date"],
                train_files=train_files,
                oot_file=oot_file,
                device=device,
                wall_deadline=wall_deadline,
                mlflow_run=mlflow_run,
                shared_feat_stats=shared_feat_stats,
            )
            if res is not None:
                fold_results.append(res)
        except Exception as e:
            log.exception(f"  Fold {spec['fold_idx']} OOT {spec['oot_date']} failed: {e}")
            continue

    # Aggregate
    if fold_results:
        ics = [r["ic_spearman"] for r in fold_results if not np.isnan(r["ic_spearman"])]
        aucs = [r["auc"] for r in fold_results if not np.isnan(r["auc"])]
        accs = [r["acc_at_0.5"] for r in fold_results if not np.isnan(r["acc_at_0.5"])]
        baseline = [r["acc_majority_baseline"] for r in fold_results
                    if not np.isnan(r["acc_majority_baseline"])]
        agg = {
            "n_folds_completed": len(fold_results),
            "mean_ic": float(np.mean(ics)) if ics else float("nan"),
            "median_ic": float(np.median(ics)) if ics else float("nan"),
            "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "mean_acc05": float(np.mean(accs)) if accs else float("nan"),
            "mean_baseline_acc": float(np.mean(baseline)) if baseline else float("nan"),
            "lift_over_baseline": (float(np.mean(accs)) - float(np.mean(baseline)))
                                  if accs and baseline else float("nan"),
        }
        log.info(f"=== AGGREGATE === {agg}")
    else:
        agg = {"n_folds_completed": 0}
        log.error("No folds completed")

    # Per-day metrics CSV
    import csv
    per_day_csv = OUT_DIR / "per_day_metrics.csv"
    if fold_results:
        keys = list(fold_results[0].keys())
        with open(per_day_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in fold_results:
                w.writerow(r)

    # FIFO grade
    fifo_grade = {"status": "skipped", "reason": "no folds completed"}
    if fold_results and time.time() < wall_deadline:
        log.info("=== Running canonical FIFO grade ===")
        try:
            fifo_grade = run_simple_fifo_grade(fold_results, wall_deadline)
            log.info(f"FIFO grade: {fifo_grade}")
        except Exception as e:
            log.exception(f"FIFO grade failed: {e}")
            fifo_grade = {"status": "error", "error": str(e)[:200]}
    elif fold_results:
        fifo_grade = {"status": "skipped", "reason": "wall deadline reached"}

    # Summary
    summary = {
        "experiment": MLFLOW_EXPERIMENT,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "ts_started": ts_now,
        "ts_finished": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "wall_sec": time.time() - t0,
        "K_ticks": K_TICKS,
        "horizon_s": HORIZON_S,
        "model": "CNNMambaV2 single-head BCE first-passage",
        "n_folds_planned": len(folds),
        "fold_aggregate": agg,
        "fifo_gate": fifo_grade,
        "log_file": str(LOG_FILE),
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"Summary written to {summary_path}")

    if mlflow_run is not None:
        try:
            import mlflow
            for k, v in agg.items():
                if isinstance(v, (int, float)) and not np.isnan(v):
                    mlflow.log_metric(f"agg_{k}", float(v))
            if isinstance(fifo_grade, dict):
                for k, v in fifo_grade.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        mlflow.log_metric(f"fifo_{k}", float(v))
            mlflow.log_artifact(str(summary_path))
            if per_day_csv.exists():
                mlflow.log_artifact(str(per_day_csv))
            fifo_csv = OUT_DIR / "fifo_gate_results.csv"
            if fifo_csv.exists():
                mlflow.log_artifact(str(fifo_csv))
            mlflow.end_run()
        except Exception as e:
            log.warning(f"MLflow finalize failed: {e}")

    log.info(f"=== DONE wall_sec={time.time()-t0:.1f} ===")
    return summary


if __name__ == "__main__":
    main()
