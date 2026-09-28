"""
train_vol_lgbm_v3.py — "ANY-confluence" volatility predictor.

Per DIRECTIVES.md (2026-04-27 19:32 ET entry):
  - Sliding 60-day train window (NOT expanding) — explicit user override of the
    "NEVER sliding window" rule, scoped to vol/auxiliary LGBMs only.
  - Exponential decay sample weights, half-life=15d (recent days weighted up).
  - Multi-horizon (10s/30s/60s) realized vol prediction = mean|ret| over horizon.
  - Per-event predictions saved as .npz aligned to cnn_mamba_v2 fold OOT dates,
    so existing confluence sims ingest without re-running the deep model.
  - Generic output: usable as gate, sizing, regime tag, or any other confluence.

This is fresh code composed from understood patterns. CPU-only (Jupiter).

Usage:
  python train_vol_lgbm_v3.py                     # full 10-fold run
  python train_vol_lgbm_v3.py --folds 0           # smoke test fold 0 only
  python train_vol_lgbm_v3.py --n-estimators 100  # quick run

Env vars (optional): VOL_LGBM_DATA_DIR, MLFLOW_TRACKING_URI
"""

import os, sys, time, argparse, json, pickle, logging, socket
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np
import scipy.stats

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("VOL_LGBM_DATA_DIR",
    "/home/jupiter/Lvl3Quant/data/processed/mbo_events"))

# Test dates aligned to cnn_mamba_v2 fold OOT (so .npz outputs merge directly
# into existing confluence sims).
TEST_DATES = [
    "20260223", "20260224", "20260225", "20260226", "20260227",
    "20260301", "20260302", "20260303", "20260304", "20260305",
]

WINDOW = 1000           # events per feature window
STRIDE = 500            # event stride (matches cnn_mamba_v2)
TRAIN_DAYS = 60         # sliding window length
HALF_LIFE_DAYS = 15.0   # exp decay on training samples
HORIZONS_S = [10.0, 30.0, 60.0]   # vol prediction horizons in seconds

# MBO column indices
COL_TIME = 0; COL_TYPE = 1; COL_SIDE = 2
COL_PRICE = 3; COL_QTY = 4; COL_SPREAD = 5
TYPE_TRADE = 3   # action 'T'

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/vol_lgbm_v3")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXP = "VolLGBM_v3_AnyConfluence"

# ── Feature names (28 features, slim by design) ───────────────────────────────
FEATURE_NAMES = [
    # Group A: realized vol of past window @ multiple scales (5)
    "rv_full", "rv_half", "rv_q4", "rv_q1", "rv_accel",
    # Group B: spread / range (4)
    "spread_mean", "spread_q4", "spread_trend", "price_range_ticks",
    # Group C: order flow (5)
    "ofi_full", "ofi_half", "ofi_q4", "ofi_accel", "cum_delta_norm",
    # Group D: event density (4)
    "evt_density_full", "evt_density_q4", "density_accel", "trade_share",
    # Group E: volume (4)
    "vol_total_log", "vol_q4_share", "vol_std_log", "vol_skew",
    # Group F: cancel intensity (3)
    "cancel_share", "cancel_asym", "cancel_q4_share",
    # Group G: micro interactions (3)
    "spread_x_density", "rv_x_spread", "ofi_abs_norm",
]
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 28


def compute_features(windows: np.ndarray) -> np.ndarray:
    """
    Vectorised feature extraction.
    windows : (N, W, 6) float32  — already mean/std normalised? NO, raw events.
              We accept raw events here; normalisation handled per-fold by caller.
    returns : (N, 28) float32
    """
    N, W, _ = windows.shape
    out = np.empty((N, N_FEATURES), dtype=np.float32)

    q4_start = W * 3 // 4
    half_start = W // 2

    prices  = windows[:, :, COL_PRICE]
    qty_log = windows[:, :, COL_QTY]
    qty     = np.maximum(np.exp(qty_log) - 1.0, 0.0)
    sides   = windows[:, :, COL_SIDE]
    spreads = windows[:, :, COL_SPREAD]
    types   = windows[:, :, COL_TYPE]
    tdelta  = np.maximum(np.exp(windows[:, :, COL_TIME]) - 1.0, 0.0)  # seconds

    # --- A: realized vol on past returns -----------------------------------
    # ret = price[t] - price[t-1]   (in ticks, since price_rel_ticks)
    rets   = np.diff(prices, axis=1)        # (N, W-1)
    abs_r  = np.abs(rets)
    rv_full = abs_r.mean(axis=1)
    rv_half = abs_r[:, half_start:].mean(axis=1)
    rv_q4   = abs_r[:, q4_start:].mean(axis=1)
    rv_q1   = abs_r[:, :half_start].mean(axis=1)
    rv_accel = rv_q4 * 4.0 - rv_full        # >0 = vol speeding up

    out[:, 0] = rv_full;   out[:, 1] = rv_half
    out[:, 2] = rv_q4;     out[:, 3] = rv_q1
    out[:, 4] = rv_accel

    # --- B: spread / range --------------------------------------------------
    sp_mean = spreads.mean(axis=1)
    sp_q4   = spreads[:, q4_start:].mean(axis=1)
    sp_trend = spreads[:, half_start:].mean(axis=1) - spreads[:, :half_start].mean(axis=1)
    p_range = prices.max(axis=1) - prices.min(axis=1)
    out[:, 5] = sp_mean;   out[:, 6] = sp_q4
    out[:, 7] = sp_trend;  out[:, 8] = p_range

    # --- C: order flow ------------------------------------------------------
    # side_id encoding: 0=B(buy), 1=A(sell/ask side), 2=N. Convert to signed:
    sgn = np.where(sides == 0, 1.0, np.where(sides == 1, -1.0, 0.0)).astype(np.float32)
    sv = sgn * qty
    ofi_full = sv.sum(axis=1)
    ofi_half = sv[:, half_start:].sum(axis=1)
    ofi_q4_  = sv[:, q4_start:].sum(axis=1)
    ofi_accel = ofi_q4_ * 4.0 - ofi_full
    total_qty = qty.sum(axis=1) + 1e-8
    cum_delta_norm = ofi_full / total_qty

    out[:, 9]  = ofi_full;  out[:, 10] = ofi_half
    out[:, 11] = ofi_q4_;   out[:, 12] = ofi_accel
    out[:, 13] = cum_delta_norm

    # --- D: event density ---------------------------------------------------
    sec_full = tdelta.sum(axis=1) + 1e-6
    sec_q4   = tdelta[:, q4_start:].sum(axis=1) + 1e-6
    evt_d_full = float(W) / sec_full
    evt_d_q4   = float(W - q4_start) / sec_q4
    dens_accel = evt_d_q4 - evt_d_full
    is_trade   = (types == TYPE_TRADE).astype(np.float32)
    trade_share = is_trade.mean(axis=1)
    out[:, 14] = evt_d_full;  out[:, 15] = evt_d_q4
    out[:, 16] = dens_accel;  out[:, 17] = trade_share

    # --- E: volume ----------------------------------------------------------
    vol_total = total_qty
    vol_q4 = qty[:, q4_start:].sum(axis=1)
    vol_q4_share = vol_q4 / vol_total
    vol_std = qty.std(axis=1)
    # signed skew proxy: (mean - median) / (std+eps)
    vol_med = np.median(qty, axis=1)
    vol_skew = (qty.mean(axis=1) - vol_med) / (vol_std + 1e-8)
    out[:, 18] = np.log1p(vol_total)
    out[:, 19] = vol_q4_share
    out[:, 20] = np.log1p(vol_std)
    out[:, 21] = vol_skew

    # --- F: cancel intensity ------------------------------------------------
    # action_encoding C=1
    is_cancel = (types == 1).astype(np.float32)
    cancel_share = is_cancel.mean(axis=1)
    # asymmetry: cancels on bid vs ask
    cb = (is_cancel * (sides == 0)).sum(axis=1)
    ca = (is_cancel * (sides == 1)).sum(axis=1)
    cancel_asym = (cb - ca) / (cb + ca + 1e-6)
    cancel_q4 = is_cancel[:, q4_start:].mean(axis=1)
    out[:, 22] = cancel_share
    out[:, 23] = cancel_asym
    out[:, 24] = cancel_q4

    # --- G: interactions ----------------------------------------------------
    out[:, 25] = sp_mean * evt_d_full        # stress
    out[:, 26] = rv_full * sp_mean           # vol amplified by wide spread
    out[:, 27] = np.abs(ofi_full) / total_qty  # absolute imbalance

    return out


def compute_realized_vol_targets(timestamps_ns: np.ndarray,
                                 prices_ticks: np.ndarray,
                                 anchor_idxs: np.ndarray,
                                 horizons_s: List[float]
                                 ) -> np.ndarray:
    """
    For each anchor index, look forward in time and compute mean|return| over
    each horizon window in seconds. Returns (n_anchors, n_horizons) float32.

    timestamps_ns : (N,) int64 nanoseconds
    prices_ticks  : (N,) float, price in ticks
    anchor_idxs   : (n_anchors,) — indices into the day's events at which to anchor
    horizons_s    : list of seconds, e.g. [10, 30, 60]

    Method:
        For each anchor, find the index range up to anchor_ts + h*1e9 ns. Compute
        mean|diff(price)| within that range. If <5 events in horizon → NaN.
    """
    n = len(anchor_idxs)
    H = len(horizons_s)
    out = np.full((n, H), np.nan, dtype=np.float32)

    if n == 0:
        return out

    # Fast path: searchsorted on the timestamps array (assumed monotonic
    # nondecreasing, which MBO events are by construction).
    for h_i, h in enumerate(horizons_s):
        h_ns = np.int64(h * 1_000_000_000)
        anchor_ts = timestamps_ns[anchor_idxs]
        end_ts = anchor_ts + h_ns
        end_idxs = np.searchsorted(timestamps_ns, end_ts, side="right")

        for k in range(n):
            i0 = anchor_idxs[k]
            i1 = end_idxs[k]
            if i1 - i0 < 6:
                continue  # too few events
            seg = prices_ticks[i0:i1]
            d = np.diff(seg)
            if d.size == 0:
                continue
            out[k, h_i] = np.float32(np.abs(d).mean())

    return out


def list_dates_in_range(start_date: str, end_date: str, files: List[Path]) -> List[Path]:
    """Return sorted file list for files with date stem in [start, end] inclusive."""
    sd = start_date; ed = end_date
    return sorted([f for f in files if sd <= f.stem[:8] <= ed])


def load_day(fp: Path) -> Optional[dict]:
    """Load a day file. Returns dict with events, timestamps, or None."""
    try:
        d = np.load(fp, allow_pickle=True)
        ev = d["events"].astype(np.float32)
        ts = d["timestamps"].astype(np.int64)
        if "metadata" in d.files:
            meta_str = str(d["metadata"][0]) if d["metadata"].ndim > 0 else str(d["metadata"])
        else:
            meta_str = ""
        return {"events": ev, "timestamps": ts, "metadata": meta_str, "path": fp}
    except Exception as e:
        log.warning(f"Failed to load {fp.name}: {e}")
        return None


def build_xy_for_day(day: dict, normalise_stats: Optional[dict] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For one day file, build:
      X : (n_anchors, 28) features
      Y : (n_anchors, 3)  realized vol targets at 3 horizons
      anchor_idxs : (n_anchors,) — indices into events array
    Anchor positions = window-end indices (i.e. starts + WINDOW - 1) at stride STRIDE.
    """
    ev = day["events"]; ts = day["timestamps"]
    n_ev = len(ev)
    if n_ev < WINDOW + 100:
        return np.empty((0, N_FEATURES), np.float32), np.empty((0, 3), np.float32), np.empty(0, np.int64)

    starts = np.arange(0, n_ev - WINDOW + 1, STRIDE, dtype=np.int64)
    anchor_idxs = starts + (WINDOW - 1)

    # Feature windows in chunks (to bound RAM)
    CHUNK = 8192
    Xs = []
    col_idx = np.arange(WINDOW, dtype=np.int64)
    for ci in range(0, len(starts), CHUNK):
        sub = starts[ci:ci+CHUNK]
        # gather windows: (chunk, W, 6)
        idx = sub[:, None] + col_idx[None, :]
        w = ev[idx]
        feat = compute_features(w)
        Xs.append(feat)
    X = np.vstack(Xs).astype(np.float32)

    # Targets: realised vol forward from anchor
    prices_ticks = ev[:, COL_PRICE]
    Y = compute_realized_vol_targets(ts, prices_ticks, anchor_idxs, HORIZONS_S)

    # Apply optional normalization (per-fold train-only stats)
    if normalise_stats is not None:
        m = normalise_stats["mean"]; s = normalise_stats["std"] + 1e-8
        X = (X - m) / s
        X = np.clip(X, -8.0, 8.0).astype(np.float32)

    return X, Y, anchor_idxs


def fold_train(test_date: str, all_files: List[Path], args, mlflow_run=None) -> dict:
    """
    Train one sliding-window fold.
    Test = test_date (1 day). Train = 60 trading days BEFORE test_date.
    Returns dict with results, also saves predictions to OUT_DIR.
    """
    log.info(f"=" * 75)
    log.info(f"FOLD test_date={test_date}")

    # Pick up to TRAIN_DAYS files strictly before test_date
    avail_train = [f for f in all_files if f.stem[:8] < test_date]
    avail_train = avail_train[-TRAIN_DAYS:]   # most recent 60 trading days
    test_files = [f for f in all_files if f.stem[:8] == test_date]
    if not test_files:
        log.warning(f"  test date {test_date} missing — skip")
        return {"status": "skipped_no_test"}
    if len(avail_train) < 30:
        log.warning(f"  insufficient train data ({len(avail_train)}d) — skip")
        return {"status": "skipped_no_train"}

    log.info(f"  train: {avail_train[0].stem[:8]}..{avail_train[-1].stem[:8]} ({len(avail_train)}d)")
    log.info(f"  test : {test_date}")

    # ---- Load training data (raw features, no norm yet) ----
    Xtr_list, Ytr_list, sample_dates = [], [], []
    test_anchor = datetime.strptime(test_date, "%Y%m%d")
    for f in avail_train:
        day = load_day(f)
        if day is None: continue
        X, Y, _ = build_xy_for_day(day, normalise_stats=None)
        if X.size == 0: continue
        Xtr_list.append(X); Ytr_list.append(Y)
        d = datetime.strptime(f.stem[:8], "%Y%m%d")
        sample_dates.extend([d] * len(X))
    if not Xtr_list:
        log.warning("  no train data extracted"); return {"status": "skipped_empty_train"}

    Xtr = np.vstack(Xtr_list); Ytr = np.vstack(Ytr_list)
    sample_dates = np.array(sample_dates)
    log.info(f"  train samples: X={Xtr.shape}, Y={Ytr.shape}")

    # ---- Per-fold normalization stats (TRAIN ONLY → no leakage) ----
    Xtr_mean = np.nanmean(Xtr, axis=0)
    Xtr_std  = np.nanstd(Xtr, axis=0) + 1e-8
    Xtr_norm = np.clip((Xtr - Xtr_mean) / Xtr_std, -8.0, 8.0).astype(np.float32)

    # ---- Decay weights (exp half-life) ----
    days_old = np.array([(test_anchor - d).days for d in sample_dates], dtype=np.float64)
    weights = np.exp(-days_old * np.log(2.0) / args.half_life)
    # Normalise so mean weight = 1 (LightGBM will scale by N internally if needed)
    weights = (weights / weights.mean()).astype(np.float32)
    log.info(f"  decay weights: min={weights.min():.3f} max={weights.max():.3f} half_life={args.half_life}d")

    # ---- Filter NaN ----
    valid_mask = ~(np.isnan(Ytr).any(axis=1) | np.isnan(Xtr_norm).any(axis=1))
    Xtr_norm = Xtr_norm[valid_mask]; Ytr = Ytr[valid_mask]; weights = weights[valid_mask]
    log.info(f"  after NaN filter: n={len(Xtr_norm)}")

    # ---- Train one LGBM per horizon ----
    import lightgbm as lgb
    models = []
    horizon_ic = {}
    for h_i, hs in enumerate(HORIZONS_S):
        y_h = Ytr[:, h_i]
        m = lgb.LGBMRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.lr,
            num_leaves=63,
            min_child_samples=100,
            max_depth=8,
            subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            n_jobs=args.n_jobs,
            random_state=42,
            verbose=-1,
        )
        t0 = time.time()
        m.fit(Xtr_norm, y_h, sample_weight=weights, feature_name=FEATURE_NAMES)
        dt = time.time() - t0
        log.info(f"  fit horizon {hs:.0f}s : {dt:.1f}s")
        models.append(m)

    # ---- Predict on test day ----
    test_day = load_day(test_files[0])
    if test_day is None:
        log.error("  test day load failed"); return {"status": "test_load_failed"}
    Xte, Yte, anchor_idxs_te = build_xy_for_day(test_day, normalise_stats={"mean": Xtr_mean, "std": Xtr_std})
    log.info(f"  test samples: X={Xte.shape}, Y={Yte.shape}")

    # Drop test rows where ALL horizons are NaN (no fwd data)
    keep = ~np.isnan(Yte).all(axis=1)
    Xte = Xte[keep]; Yte = Yte[keep]; anchor_idxs_te = anchor_idxs_te[keep]

    preds = np.full_like(Yte, np.nan, dtype=np.float32)
    for h_i, m in enumerate(models):
        preds[:, h_i] = m.predict(Xte).astype(np.float32)

    # ---- Metrics: Spearman IC per horizon, plus quintile-stratified IC ----
    metrics = {"test_date": test_date, "n_test": int(len(Yte))}
    for h_i, hs in enumerate(HORIZONS_S):
        p = preds[:, h_i]; y = Yte[:, h_i]
        valid = ~np.isnan(p) & ~np.isnan(y)
        if valid.sum() < 50:
            metrics[f"ic_{int(hs)}s"] = float("nan"); continue
        ic = scipy.stats.spearmanr(p[valid], y[valid]).statistic
        metrics[f"ic_{int(hs)}s"] = float(ic)

        # Quintile-stratified IC (top-quintile by |pred|)
        ap = np.abs(p[valid])
        thr = np.percentile(ap, 80)
        top = ap >= thr
        if top.sum() >= 30:
            pv, yv = p[valid][top], y[valid][top]
            ic_top = scipy.stats.spearmanr(pv, yv).statistic
            metrics[f"ic_q5_{int(hs)}s"] = float(ic_top)
        log.info(f"  horizon {int(hs):2d}s  IC={ic:+.4f}  Q5_IC={metrics.get(f'ic_q5_{int(hs)}s', float('nan')):+.4f}")

    # ---- Save predictions npz ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = OUT_DIR / f"vol_v3_{test_date}_predictions.npz"
    np.savez_compressed(
        out_npz,
        predictions=preds,                              # (n, 3) float32
        labels=Yte,                                     # (n, 3) float32
        anchor_idxs=anchor_idxs_te.astype(np.int64),    # (n,) into day events
        horizons_s=np.array(HORIZONS_S, dtype=np.float32),
        feature_names=np.array(FEATURE_NAMES),
        train_mean=Xtr_mean.astype(np.float32),
        train_std=Xtr_std.astype(np.float32),
        test_date=np.array([test_date]),
        train_dates=np.array([f.stem[:8] for f in avail_train]),
        half_life_days=np.float32(args.half_life),
    )
    log.info(f"  saved {out_npz}")

    # Pickle models
    out_pkl = OUT_DIR / f"vol_v3_{test_date}_models.pkl"
    with open(out_pkl, "wb") as fh:
        pickle.dump({"models": models, "feature_names": FEATURE_NAMES,
                     "train_mean": Xtr_mean, "train_std": Xtr_std,
                     "horizons_s": HORIZONS_S}, fh)

    # Save metrics json
    out_json = OUT_DIR / f"vol_v3_{test_date}_metrics.json"
    with open(out_json, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # MLflow logging
    if mlflow_run is not None:
        try:
            import mlflow
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                    mlflow.log_metric(k, v)
        except Exception as e:
            log.warning(f"  mlflow log_metric failed: {e}")

    metrics["status"] = "ok"
    return metrics


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="all",
                    help="'all' or comma-separated indices into TEST_DATES")
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--half-life", type=float, default=HALF_LIFE_DAYS)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve folds
    if args.folds == "all":
        test_dates = TEST_DATES
    else:
        idxs = [int(x) for x in args.folds.split(",")]
        test_dates = [TEST_DATES[i] for i in idxs]
    log.info(f"VOL LGBM v3 — {len(test_dates)} test dates: {test_dates}")
    log.info(f"  TRAIN_DAYS={TRAIN_DAYS} HALF_LIFE={args.half_life}d HORIZONS={HORIZONS_S}")
    log.info(f"  n_estimators={args.n_estimators} lr={args.lr} n_jobs={args.n_jobs}")
    log.info(f"  data_dir={DATA_DIR}")
    log.info(f"  out_dir={OUT_DIR}")

    files = sorted([f for f in DATA_DIR.glob("*_mbo_events.npz")])
    log.info(f"  available days: {len(files)} ({files[0].stem[:8]}..{files[-1].stem[:8]})")

    # MLflow setup
    mlflow_run = None
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXP)
        run_name = f"vol_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}_h{int(args.half_life)}d"
        mlflow_run = mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            "train_days": TRAIN_DAYS,
            "half_life_days": args.half_life,
            "window": WINDOW, "stride": STRIDE,
            "horizons_s": ",".join(str(int(h)) for h in HORIZONS_S),
            "n_estimators": args.n_estimators,
            "lr": args.lr,
            "host": socket.gethostname(),
            "n_features": N_FEATURES,
            "test_dates": ",".join(test_dates),
        })
        log.info(f"  MLflow run: {mlflow_run.info.run_id}  exp={MLFLOW_EXP}")
    except Exception as e:
        log.warning(f"  MLflow disabled: {e}")
        mlflow_run = None

    # Run each fold
    all_results = []
    t_start = time.time()
    for td in test_dates:
        try:
            res = fold_train(td, files, args, mlflow_run=mlflow_run)
            all_results.append(res)
        except Exception as e:
            log.exception(f"fold {td} failed: {e}")
            all_results.append({"status": "error", "test_date": td, "err": str(e)})

    # Summary
    total_dt = time.time() - t_start
    log.info("=" * 75)
    log.info(f"DONE — {len(test_dates)} folds in {total_dt/60:.1f} min")
    ic10 = [r.get("ic_10s") for r in all_results if r.get("status") == "ok"]
    ic30 = [r.get("ic_30s") for r in all_results if r.get("status") == "ok"]
    ic60 = [r.get("ic_60s") for r in all_results if r.get("status") == "ok"]
    if ic10:
        log.info(f"  IC_10s mean={np.nanmean(ic10):+.4f}  min={np.nanmin(ic10):+.4f}  max={np.nanmax(ic10):+.4f}")
        log.info(f"  IC_30s mean={np.nanmean(ic30):+.4f}")
        log.info(f"  IC_60s mean={np.nanmean(ic60):+.4f}")
        if mlflow_run is not None:
            try:
                import mlflow
                mlflow.log_metric("avg_ic_10s", float(np.nanmean(ic10)))
                mlflow.log_metric("avg_ic_30s", float(np.nanmean(ic30)))
                mlflow.log_metric("avg_ic_60s", float(np.nanmean(ic60)))
            except Exception:
                pass

    # Save run summary
    summary = {
        "n_folds_ok": int(sum(r.get("status") == "ok" for r in all_results)),
        "n_folds_total": len(all_results),
        "wall_time_min": total_dt / 60.0,
        "config": {"train_days": TRAIN_DAYS, "half_life": args.half_life,
                   "n_estimators": args.n_estimators, "lr": args.lr},
        "folds": all_results,
    }
    summary_path = OUT_DIR / f"vol_v3_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as fh:
        # Strip non-JSON-able if any; floats already fine
        def _clean(o):
            if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, list): return [_clean(v) for v in o]
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.integer,)):  return int(o)
            return o
        json.dump(_clean(summary), fh, indent=2)
    log.info(f"  summary: {summary_path}")

    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_artifact(str(summary_path))
            mlflow.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    main()
