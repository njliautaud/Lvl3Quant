"""
train_vol_lgbm_v2.py — Volume Profile + Order Flow LGBM v2 for 10s prediction.

Key improvements over v1:
  - Real volume profile: POC, VWAP, value area width, price-vs-VWAP
  - Multi-scale OFI: full / half / quarter window (captures flow acceleration)
  - Absorption signal: large passive counter-flow against aggressive orders
  - Large-order OFI: institutional flow separated from retail noise
  - VWAP trend: is value drifting up or down within the window?
  - Microstructure regime: density + spread interaction
  - Fixed duplicate (cum_delta was identical to OFI in v1)
  - W=1000 events (aligned with CNN/Mamba, more temporal context)
  - Stronger regularization to fight fold decay (DART + reg_lambda/alpha)
  - Model registry logging (consistent with CNN/Mamba)

Features (48 total):
  GROUP 1 — Raw stats multi-scale         18
  GROUP 2 — Volume profile                 9
  GROUP 3 — Order flow (multi-scale)      10
  GROUP 4 — Microstructure                 8
  GROUP 5 — Interaction features           3

Env vars:
  LGBM_WINDOW=1000
  LGBM_STRIDE=500
  LGBM_TRAIN_DAYS=60
  LGBM_TEST_DAYS=15
  LGBM_N_FOLDS=9
  LGBM_N_ESTIMATORS=600
  LGBM_LR=0.02
  LGBM_NUM_LEAVES=63
  LGBM_MAX_DEPTH=7
  LGBM_MIN_CHILD=100
  LGBM_SUBSAMPLE=0.7
  LGBM_COLSAMPLE=0.7
  LGBM_N_JOBS=8
  MLFLOW_TRACKING_URI=http://neptune-win:5002
  MLFLOW_EXPERIMENT=VolLGBM_10s
"""

import os, sys, time, logging, socket, argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import scipy.stats

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW     = int(os.environ.get("LGBM_WINDOW",        1000))
STRIDE     = int(os.environ.get("LGBM_STRIDE",         500))
TRAIN_DAYS = int(os.environ.get("LGBM_TRAIN_DAYS",      60))
TEST_DAYS  = int(os.environ.get("LGBM_TEST_DAYS",       15))
N_FOLDS    = int(os.environ.get("LGBM_N_FOLDS",          9))
N_EST      = int(os.environ.get("LGBM_N_ESTIMATORS",   600))
LGBM_LR    = float(os.environ.get("LGBM_LR",           0.02))
NUM_LEAVES = int(os.environ.get("LGBM_NUM_LEAVES",       63))
MAX_DEPTH  = int(os.environ.get("LGBM_MAX_DEPTH",         7))
MIN_CHILD  = int(os.environ.get("LGBM_MIN_CHILD",       100))
SUBSAMPLE  = float(os.environ.get("LGBM_SUBSAMPLE",     0.7))
COLSAMPLE  = float(os.environ.get("LGBM_COLSAMPLE",     0.7))
N_JOBS     = int(os.environ.get("LGBM_N_JOBS",           10))

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://neptune-win:5002")
MLFLOW_EXP = os.environ.get("MLFLOW_EXPERIMENT",   "VolLGBM_10s")
DATA_DIR   = Path(os.environ.get("LGBM_DATA_DIR",
    "/home/jupiter/Lvl3Quant/data/processed/mbo_events"))

N_RAW = 6  # raw MBO features per event

# ── Logging ───────────────────────────────────────────────────────────────────
class _FH(logging.StreamHandler):
    def emit(self, r):
        super().emit(r)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[_FH(sys.stdout)],
)
log = logging.getLogger(__name__)

if os.environ.get("DISABLE_MLFLOW", "0") == "1":
    MLFLOW_OK = False
    log.info("MLflow disabled via DISABLE_MLFLOW=1")
else:
    try:
        import mlflow
        MLFLOW_OK = True
    except ImportError:
        MLFLOW_OK = False
        log.warning("mlflow not found — logging disabled")

try:
    import lightgbm as lgb
except ImportError:
    log.error("lightgbm not installed! Run: pip install lightgbm")
    sys.exit(1)


# ── Column index constants ─────────────────────────────────────────────────────
# Raw MBO columns:
# 0: time_delta_log   1: event_type   2: side   3: price_rel_ticks
# 4: qty_log          5: spread_ticks
COL_TIME   = 0
COL_TYPE   = 1
COL_SIDE   = 2    # 1=buy, -1=sell, 0=neutral
COL_PRICE  = 3    # price in relative ticks
COL_QTY    = 4    # log1p(qty)
COL_SPREAD = 5    # spread in ticks

TYPE_CANCEL = 2   # MBO cancel event type id


# ── Feature names ─────────────────────────────────────────────────────────────
def feature_names() -> List[str]:
    names = []
    # GROUP 1: raw stats (18)
    for i in range(N_RAW): names.append(f"raw_mean_{i}")
    for i in range(N_RAW): names.append(f"raw_std_{i}")
    for i in range(N_RAW): names.append(f"raw_mean_q4_{i}")  # last-quarter mean
    # GROUP 2: volume profile (9)
    names += ["poc_dist_ticks", "price_vs_vwap", "val_area_width",
              "price_in_va", "vol_above_poc_ratio", "poc_strength",
              "price_range_ticks", "vwap_trend", "vol_dist_skew"]
    # GROUP 3: order flow (10)
    names += ["ofi_full", "ofi_half", "ofi_q4", "ofi_accel",
              "cancel_asym", "cancel_ratio",
              "cum_delta_norm", "large_order_ofi",
              "absorption", "ofi_x_poc"]
    # GROUP 4: microstructure (8)
    names += ["event_density", "event_density_q4", "density_accel",
              "price_mom_full", "price_mom_q4",
              "price_volatility", "spread_mean", "spread_trend"]
    # GROUP 5: interactions (3)
    names += ["ofi_in_value_area", "density_x_spread", "momentum_vs_flow"]
    return names

N_FEATURES = 48


# ── Vectorised feature extraction ─────────────────────────────────────────────
VOL_BINS = 20   # price histogram bins for volume profile


def compute_features_batch(windows: np.ndarray) -> np.ndarray:
    """
    Vectorised feature extraction.
    windows : (N, W, 6)  float32
    returns : (N, 48)    float32
    """
    N, W, _ = windows.shape
    out = np.empty((N, N_FEATURES), dtype=np.float32)

    q4_start = max(0, W * 3 // 4)   # last 25% of window
    q2_start = max(0, W // 2)        # last 50%

    # ── Raw stats ────────────────────────────────────────────────────────────
    out[:, :6]   = windows.mean(axis=1)
    out[:, 6:12] = windows.std(axis=1) + 1e-8
    out[:, 12:18] = windows[:, q4_start:, :].mean(axis=1)

    # ── Helpers ──────────────────────────────────────────────────────────────
    prices  = windows[:, :, COL_PRICE]              # (N, W)
    qty_raw = np.exp(windows[:, :, COL_QTY]) - 1    # (N, W) — undo log1p
    qty_raw = np.maximum(qty_raw, 0)
    sides   = windows[:, :, COL_SIDE]               # (N, W)

    total_qty  = qty_raw.sum(axis=1) + 1e-8         # (N,)
    last_price = prices[:, -1]                       # (N,)

    # ── GROUP 2: Volume Profile ───────────────────────────────────────────────

    # VWAP — volume-weighted average price
    vwap = (prices * qty_raw).sum(axis=1) / total_qty  # (N,)

    # Volume-weighted variance → value area width approximation
    vwap_sq   = ((prices ** 2) * qty_raw).sum(axis=1) / total_qty
    vol_var   = np.maximum(vwap_sq - vwap ** 2, 0)
    vol_std   = np.sqrt(vol_var)                     # (N,) ≈ value area half-width

    # POC via price histogram (VOL_BINS buckets per sample)
    p_min   = prices.min(axis=1)                     # (N,)
    p_max   = prices.max(axis=1)                     # (N,)
    p_range = np.maximum(p_max - p_min, 1e-6)        # (N,)

    # Bin each event into its price bucket (0..VOL_BINS-1)
    p_norm  = ((prices - p_min[:, None]) / p_range[:, None] * (VOL_BINS - 1))
    p_bin   = np.clip(p_norm.astype(np.int32), 0, VOL_BINS - 1)  # (N, W)

    # Volume per bin: vol_hist[n, b] = sum of qty_raw in bin b for sample n
    vol_hist = np.zeros((N, VOL_BINS), dtype=np.float32)
    for b in range(VOL_BINS):
        mask = (p_bin == b)               # (N, W)
        vol_hist[:, b] = (qty_raw * mask).sum(axis=1)

    # POC bucket and reconstructed price
    poc_bin   = np.argmax(vol_hist, axis=1)          # (N,)
    poc_price = p_min + (poc_bin / (VOL_BINS - 1)) * p_range  # (N,)
    poc_dist  = last_price - poc_price                # (N,) ticks from POC

    # Value area (±1 vol_std around VWAP)
    val_area_width = vol_std * 2                      # (N,)
    price_in_va    = ((last_price >= vwap - vol_std) &
                      (last_price <= vwap + vol_std)).astype(np.float32)

    # Fraction of volume above POC
    above_poc    = (prices > poc_price[:, None])
    vol_above    = (qty_raw * above_poc).sum(axis=1) / total_qty  # (N,)

    # POC strength = fraction of total volume at peak bin
    poc_strength = vol_hist[np.arange(N), poc_bin] / total_qty   # (N,)

    # VWAP trend: VWAP of last quarter vs first quarter
    tq = qty_raw[:, q4_start:]
    tp = prices[:, q4_start:]
    vwap_q4  = (tp * tq).sum(axis=1) / (tq.sum(axis=1) + 1e-8)

    eq = qty_raw[:, :q4_start] if q4_start > 0 else qty_raw[:, :1]
    ep = prices[:, :q4_start]  if q4_start > 0 else prices[:, :1]
    vwap_q1  = (ep * eq).sum(axis=1) / (eq.sum(axis=1) + 1e-8)
    vwap_trend = vwap_q4 - vwap_q1

    # Volume distribution skew (signed: positive = more vol above POC)
    vol_dist_skew = vol_above - (1.0 - vol_above)    # in [-1, 1]

    out[:, 18] = poc_dist
    out[:, 19] = last_price - vwap
    out[:, 20] = val_area_width
    out[:, 21] = price_in_va
    out[:, 22] = vol_above
    out[:, 23] = poc_strength
    out[:, 24] = p_range
    out[:, 25] = vwap_trend
    out[:, 26] = vol_dist_skew

    # ── GROUP 3: Order Flow ───────────────────────────────────────────────────

    # OFI at multiple scales
    sv_full = sides * qty_raw
    ofi_full = sv_full.sum(axis=1)                              # (N,)
    ofi_half = sv_full[:, q2_start:].sum(axis=1)               # last 50%
    ofi_q4_  = sv_full[:, q4_start:].sum(axis=1)               # last 25%
    # Acceleration: is recent flow stronger than expected from full?
    ofi_accel = ofi_q4_ * 4 - ofi_full                         # positive = accelerating

    # Cancel asymmetry
    is_cancel   = (windows[:, :, COL_TYPE] == TYPE_CANCEL)     # (N, W)
    cancel_buy  = (is_cancel & (sides > 0)).sum(axis=1).astype(np.float32)
    cancel_sell = (is_cancel & (sides < 0)).sum(axis=1).astype(np.float32)
    cancel_tot  = cancel_buy + cancel_sell + 1e-8
    cancel_asym = (cancel_buy - cancel_sell) / cancel_tot
    cancel_ratio = cancel_tot / (W + 1e-8)

    # Normalised cumulative delta (different from OFI — normalised by volume)
    cum_delta_norm = ofi_full / total_qty

    # Large-order OFI: OFI computed only on events with qty > 75th percentile
    qty75 = np.percentile(qty_raw, 75, axis=1, keepdims=True)  # (N, 1)
    large = (qty_raw > qty75)                                   # (N, W)
    large_ofi = (sv_full * large).sum(axis=1) / (total_qty + 1e-8)

    # Absorption: large passive orders in opposite direction to recent aggressor
    # Proxy: large qty_raw orders that are opposite sign to ofi_q4_
    ofi_sign = np.sign(ofi_q4_[:, None])                       # (N, 1)
    passive   = (sides * ofi_sign < 0).astype(np.float32)      # opposite direction
    absorption = (qty_raw * passive * large).sum(axis=1) / (total_qty + 1e-8)

    # OFI × POC direction (is flow pushing price toward or away from POC?)
    ofi_x_poc = ofi_full * np.sign(poc_dist + 1e-8)

    out[:, 27] = ofi_full
    out[:, 28] = ofi_half
    out[:, 29] = ofi_q4_
    out[:, 30] = ofi_accel
    out[:, 31] = cancel_asym
    out[:, 32] = cancel_ratio
    out[:, 33] = cum_delta_norm
    out[:, 34] = large_ofi
    out[:, 35] = absorption
    out[:, 36] = ofi_x_poc

    # ── GROUP 4: Microstructure ───────────────────────────────────────────────

    time_deltas    = np.exp(windows[:, :, COL_TIME]) - 1       # (N, W)
    mean_td_full   = time_deltas.mean(axis=1) + 1e-6
    mean_td_q4     = time_deltas[:, q4_start:].mean(axis=1) + 1e-6
    mean_td_q1_end = time_deltas[:, :q4_start].mean(axis=1) + 1e-6 if q4_start > 0 else mean_td_full

    event_density      = 1.0 / mean_td_full
    event_density_q4   = 1.0 / mean_td_q4
    density_accel      = event_density_q4 - event_density       # positive = speeding up

    price_mom_full = last_price - prices[:, 0]
    price_mom_q4   = last_price - prices[:, q4_start]
    price_vol      = prices.std(axis=1) + 1e-8

    spreads = windows[:, :, COL_SPREAD]
    spread_mean  = spreads.mean(axis=1)
    spread_half  = max(1, W // 2)
    spread_trend = spreads[:, spread_half:].mean(axis=1) - spreads[:, :spread_half].mean(axis=1)

    out[:, 37] = event_density
    out[:, 38] = event_density_q4
    out[:, 39] = density_accel
    out[:, 40] = price_mom_full
    out[:, 41] = price_mom_q4
    out[:, 42] = price_vol
    out[:, 43] = spread_mean
    out[:, 44] = spread_trend

    # ── GROUP 5: Interaction features ────────────────────────────────────────

    # OFI only when price is inside the value area (structural support/resistance)
    out[:, 45] = ofi_full * price_in_va

    # Microstructure stress: high event density + wide spread = stressed market
    out[:, 46] = event_density * spread_mean

    # Momentum vs flow: price moved opposite to flow = potential reversal
    out[:, 47] = price_mom_full * -np.sign(ofi_full + 1e-8)

    return out


# ── Dataset builder ───────────────────────────────────────────────────────────
_CHUNK = 25_000   # windows per batch (larger W = smaller chunk to cap RAM)


def build_xy(files: List[Path], stats: Dict, window: int, stride: int
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) from file list. All stats from train fold only (no leakage)."""
    mean = stats["mean"]
    std  = stats["std"]
    all_X, all_y = [], []

    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            ev   = (data["features"] if "features" in data else data["events"]).astype(np.float32)
            labs = data["labels_10s"].astype(np.float32)
        except Exception:
            continue

        ev_norm = (ev - mean) / (std + 1e-8)
        n_ev    = len(ev_norm)
        if n_ev < window:
            continue

        starts     = np.arange(0, n_ev - window + 1, stride, dtype=np.int32)
        label_idxs = starts + window - 1
        valid      = (label_idxs < len(labs)) & ~np.isnan(labs[label_idxs])
        starts     = starts[valid]
        label_idxs = label_idxs[valid]
        if len(starts) == 0:
            continue

        col_idx = np.arange(window, dtype=np.int32)
        for ci in range(0, len(starts), _CHUNK):
            s   = starts[ci : ci + _CHUNK]
            li  = label_idxs[ci : ci + _CHUNK]
            idx = s[:, None] + col_idx[None, :]  # (chunk, W)
            W_batch = ev_norm[idx]               # (chunk, W, 6)
            all_X.append(compute_features_batch(W_batch))
            all_y.append(labs[li])

    if not all_X:
        return np.empty((0, N_FEATURES), np.float32), np.empty(0, np.float32)
    return np.concatenate(all_X), np.concatenate(all_y)


def compute_stats(files: List[Path]) -> Dict:
    s, sq, n = np.zeros(N_RAW, np.float64), np.zeros(N_RAW, np.float64), 0
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            ev   = (data["features"] if "features" in data else data["events"]).astype(np.float64)
            s += ev.sum(0); sq += (ev**2).sum(0); n += len(ev)
        except Exception:
            pass
    if n == 0:
        return {"mean": np.zeros(N_RAW, np.float32), "std": np.ones(N_RAW, np.float32)}
    mean = (s / n).astype(np.float32)
    std  = np.sqrt(np.maximum(sq/n - (s/n)**2, 1e-8)).astype(np.float32)
    return {"mean": mean, "std": std}


def ic(preds: np.ndarray, labels: np.ndarray) -> float:
    mask = ~(np.isnan(preds) | np.isnan(labels))
    if mask.sum() < 20: return float("nan")
    return float(scipy.stats.spearmanr(preds[mask], labels[mask])[0])


# ── Walk-forward runner ───────────────────────────────────────────────────────
def run_wf(files: List[Path], output_dir: Path, run_name: str):
    n = len(files)
    feat_names = feature_names()

    # Sliding window folds (fixed-size training window)
    folds = []
    min_train = max(TRAIN_DAYS, 30)  # minimum 30 days to start
    test_start = min_train
    while test_start + TEST_DAYS <= n:
        train_start = max(0, test_start - TRAIN_DAYS)  # SLIDING: fixed window
        train_idx = list(range(train_start, test_start))
        test_idx  = list(range(test_start, min(test_start + TEST_DAYS, n)))
        folds.append((len(folds), train_idx, test_idx))
        test_start += TEST_DAYS

    if not folds:
        log.error(f"Not enough files ({n}) for min_train={min_train} + TEST_DAYS={TEST_DAYS}")
        return

    log.info(f"Folds: {len(folds)} | Train: SLIDING ({TRAIN_DAYS}d window) | Test: {TEST_DAYS}d each")
    log.info(f"Config: W={WINDOW} S={STRIDE} n_est={N_EST} lr={LGBM_LR} "
             f"leaves={NUM_LEAVES} depth={MAX_DEPTH} min_child={MIN_CHILD} "
             f"sub={SUBSAMPLE} col={COLSAMPLE} n_features={N_FEATURES}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # MLflow
    mlrun = None
    if MLFLOW_OK:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXP)
        mlrun = mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            "window": WINDOW, "stride": STRIDE, "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS, "n_folds": len(folds), "n_files": n,
            "n_estimators": N_EST, "lr": LGBM_LR, "num_leaves": NUM_LEAVES,
            "max_depth": MAX_DEPTH, "min_child_samples": MIN_CHILD,
            "subsample": SUBSAMPLE, "colsample_bytree": COLSAMPLE,
            "n_features": N_FEATURES, "node": socket.gethostname(),
            "version": "v2",
        })

    all_preds, all_labels = [], []
    feat_importance_sum = np.zeros(N_FEATURES)
    best_fold_ic, best_fold_idx = -np.inf, 0
    best_model = None

    for fold_idx, train_idx, test_idx in folds:
        t_files   = [files[i] for i in train_idx]
        oot_files = [files[i] for i in test_idx]

        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold_idx:02d} | Train: {t_files[0].name}..{t_files[-1].name} "
                 f"({len(t_files)}d) | Test: {oot_files[0].name}..{oot_files[-1].name} "
                 f"({len(oot_files)}d)")

        stats = compute_stats(t_files)
        log.info(f"  mean: {stats['mean'].round(3)} | std: {stats['std'].round(3)}")

        t0 = time.time()
        X_tr, y_tr = build_xy(t_files,   stats, WINDOW, STRIDE)
        X_ot, y_ot = build_xy(oot_files, stats, WINDOW, STRIDE)
        log.info(f"  Features: {time.time()-t0:.1f}s | Train={len(X_tr):,} | OOT={len(X_ot):,}")

        if len(X_tr) == 0 or len(X_ot) == 0:
            log.warning(f"  Fold {fold_idx}: empty, skipping")
            continue

        tr_mask = ~np.isnan(y_tr); X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
        ot_mask = ~np.isnan(y_ot); X_ot, y_ot = X_ot[ot_mask], y_ot[ot_mask]

        # ── LGBM training ─────────────────────────────────────────────────────
        # DART boosting drops trees randomly → robust to regime shift, less decay.
        # Strong L2 regularization + depth limit to prevent overfitting early folds.
        params = {
            "boosting_type":     "dart",    # randomly drops trees → less decay
            "objective":         "regression",
            "metric":            "rmse",
            "n_estimators":      N_EST,
            "learning_rate":     LGBM_LR,
            "num_leaves":        NUM_LEAVES,
            "max_depth":         MAX_DEPTH,
            "min_child_samples": MIN_CHILD,
            "subsample":         SUBSAMPLE,
            "subsample_freq":    1,
            "colsample_bytree":  COLSAMPLE,
            "reg_alpha":         0.3,        # L1
            "reg_lambda":        2.0,        # L2
            "min_split_gain":    0.01,       # no trivial splits
            "drop_rate":         0.1,        # DART drop rate
            "skip_drop":         0.5,        # skip dropping 50% of iters
            "n_jobs":            N_JOBS,
            "verbose":           -1,
            "random_state":      42 + fold_idx,
        }

        t1 = time.time()
        model = lgb.LGBMRegressor(**params)
        # Note: DART doesn't support early stopping reliably — use fixed n_est
        model.fit(X_tr, y_tr,
                  eval_set=[(X_ot, y_ot)],
                  callbacks=[lgb.log_evaluation(period=100)])
        train_secs = time.time() - t1

        fold_preds  = model.predict(X_ot)
        fold_ic_val = ic(fold_preds, y_ot)
        train_ic    = ic(model.predict(X_tr), y_tr)

        log.info(f"  Trained {train_secs:.1f}s | TrainIC={train_ic:.4f} | OOT IC_10s={fold_ic_val:.4f}")

        # Track best fold for model registry
        if fold_ic_val > best_fold_ic:
            best_fold_ic  = fold_ic_val
            best_fold_idx = fold_idx
            best_model    = model

        feat_importance_sum += model.feature_importances_

        if MLFLOW_OK and mlrun:
            mlflow.log_metrics({
                f"fold{fold_idx:02d}_oot_ic":   fold_ic_val,
                f"fold{fold_idx:02d}_train_ic": train_ic,
            }, step=fold_idx)

        np.savez(output_dir / f"fold{fold_idx:02d}_preds.npz",
                 preds=fold_preds, labels=y_ot)
        model.booster_.save_model(str(output_dir / f"fold{fold_idx:02d}_model.txt"))

        all_preds.append(fold_preds)
        all_labels.append(y_ot)

    # ── Final reporting ────────────────────────────────────────────────────────
    if all_preds:
        concat_p  = np.concatenate(all_preds)
        concat_l  = np.concatenate(all_labels)
        concat_ic_val = ic(concat_p, concat_l)

        log.info(f"\n{'='*60}")
        log.info(f"CONCAT IC_10s = {concat_ic_val:.4f}  ({len(concat_p):,} samples)")

        # Feature importance table
        imp_order = np.argsort(feat_importance_sum)[::-1]
        log.info("Top-15 features by importance:")
        for rank, fi in enumerate(imp_order[:15]):
            log.info(f"  {rank+1:>2}. {feat_names[fi]:<30} {feat_importance_sum[fi]:.0f}")

        np.save(str(output_dir / "feat_importance.npy"), feat_importance_sum)

        if MLFLOW_OK and mlrun:
            mlflow.log_metric("concat_ic_10s", concat_ic_val)
            mlflow.log_params({"concat_ic_10s": concat_ic_val})
            # Log top-10 feature importances as metrics
            for rank, fi in enumerate(imp_order[:10]):
                mlflow.log_metric(f"imp_{feat_names[fi]}", float(feat_importance_sum[fi]))

    # ── Model registry ────────────────────────────────────────────────────────
    if MLFLOW_OK and mlrun and best_model is not None:
        try:
            import mlflow.lightgbm as mlflow_lgbm
            mlflow_lgbm.log_model(best_model.booster_, artifact_path="model")
            log.info(f"  Logged best model (fold {best_fold_idx:02d}, IC={best_fold_ic:.4f})")
            model_name = f"VolLGBM_10s_W{WINDOW}_v2"
            run_id = mlrun.info.run_id
            reg = mlflow.register_model(f"runs:/{run_id}/model", model_name)
            log.info(f"  Registered as '{model_name}' v{reg.version}")
        except Exception as e:
            log.warning(f"  Model registry failed (non-fatal): {e}")

    if MLFLOW_OK and mlrun:
        mlflow.end_run()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Vol Profile + Orderflow LGBM v2")
    ap.add_argument("--output-dir", default="/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results/vol_lgbm_v2")
    ap.add_argument("--data-dir",   default=str(DATA_DIR))
    ap.add_argument("--run-name",   default=None)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    data_dir   = Path(args.data_dir)

    files = sorted(data_dir.glob("*.npz"))
    if not files:
        log.error(f"No NPZ files in {data_dir}"); sys.exit(1)
    log.info(f"Found {len(files)} files: {files[0].name} .. {files[-1].name}")

    valid = []
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            if "labels_10s" in d and not np.all(np.isnan(d["labels_10s"])):
                valid.append(f)
        except Exception:
            pass
    log.info(f"Valid (have labels_10s): {len(valid)}")
    files = valid

    log.info(f"Config: W={WINDOW} S={STRIDE} n_est={N_EST} lr={LGBM_LR} "
             f"leaves={NUM_LEAVES} depth={MAX_DEPTH} min_child={MIN_CHILD} "
             f"n_jobs={N_JOBS} features={N_FEATURES}")
    log.info(f"Walk-forward: EXPANDING train / {TEST_DAYS}d test")

    run_name = args.run_name or f"VolLGBM_v2_W{WINDOW}_{time.strftime('%H%M')}"
    run_wf(files, output_dir, run_name=run_name)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        log.error(f"FATAL: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)
