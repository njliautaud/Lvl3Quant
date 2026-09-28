"""
Adaptive Exit Model for ES Futures — Lvl3Quant

Concept: Instead of fixed hold periods, continuously monitor order flow while in
a trade. If adverse signals appear (P(loss) > threshold), EXIT IMMEDIATELY at
market. This cuts losers fast while letting winners run.

Memory-efficient design: loads features day-by-day from disk rather than
all 70 days at once (which would require 22 GB RAM).

Pipeline:
  Step 1: Enumerate feature cache files. Load mid_prices for all N days (tiny).
  Step 2: Walk-forward entry model training — load rolling window of N days,
          train LightGBM direction+magnitude, predict one test day, discard.
  Step 3: Find entry bars per day using saved predictions + mid_prices.
  Step 4: For each test day, build monitoring dataset from entries in that day
          (load features for that day only), run adverse flow detector.
  Step 5: Backtest fixed-hold vs adaptive exit vs flip exit on same entries.
  Metrics: PnL, Sharpe, PF, avg winner/loser, % losers cut early.

Usage:
    python alpha_discovery/adaptive_exit.py
    python alpha_discovery/adaptive_exit.py --n-days 70 --quick
    python alpha_discovery/adaptive_exit.py --n-days 35
"""

import gc
import sys
import time
import json
import logging
import argparse
import platform
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging — write to both file and stdout
# ---------------------------------------------------------------------------
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = RESULTS_DIR / f"adaptive_exit_{_ts}.log"
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in _root.handlers[:]:
    _root.removeHandler(_h)
_fmt = logging.Formatter('%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
_fh = logging.FileHandler(str(_log_file), mode='w')
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_root.addHandler(_fh)
_root.addHandler(_sh)
logger = logging.getLogger("adaptive_exit")

# ---------------------------------------------------------------------------
# Constants (ES futures, 100ms bars)
# ---------------------------------------------------------------------------
TICK_SIZE        = 0.25
TICK_VALUE       = 12.50
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE   # 0.24 ticks
HALF_TICK        = TICK_SIZE / 2                # 0.125
BARS_PER_SEC     = 10                           # 100ms bars

# Entry gate (matches magnitude_gated_sim.py defaults)
MAG_GATE_TICKS   = 2.0
SIGNAL_QUANTILE  = 0.90   # top 10% direction signal
MIN_BARS_BETWEEN = 50     # 5s spacing between signals

# Cross-platform paths
if platform.system() == 'Windows':
    DEFAULT_FEAT_CACHE = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
    DEFAULT_SNAP_CACHE = LVL3_ROOT / "data" / "processed" / "medium_snapshots_cache"
else:
    _home = Path.home() / "lvl3quant"
    DEFAULT_FEAT_CACHE = _home / "data" / "processed" / "mbo_features_cache"
    DEFAULT_SNAP_CACHE = _home / "data" / "processed" / "medium_snapshots_cache"


# ============================================================================
# Day registry — enumerate all available days without loading features
# ============================================================================

@dataclass
class DayRecord:
    date_str:  str
    feat_path: Path    # *_mbo_features.npz
    snap_path: Path    # *_snapshots.npz  (for mid_prices)
    n_bars:    int


def build_day_registry(n_days: Optional[int] = None) -> List[DayRecord]:
    """
    Scan both caches, match by date, load only mid_prices (tiny) to
    get n_bars. Returns sorted list of DayRecord.
    """
    feat_dir = DEFAULT_FEAT_CACHE
    snap_dir = DEFAULT_SNAP_CACHE

    feat_files = sorted(feat_dir.glob("*_mbo_features.npz"))
    snap_files = sorted(snap_dir.glob("????-??-??_snapshots.npz"))

    snap_by_date = {f.name[:10]: f for f in snap_files}

    records = []
    for ff in feat_files:
        date_str = ff.name[:10]
        if date_str not in snap_by_date:
            continue
        sf = snap_by_date[date_str]
        # Load only mid_prices (small)
        try:
            d = np.load(str(sf), allow_pickle=True)
            mp = d['mid_prices']
            d.close()
            n = len(mp)
            del mp
        except Exception as exc:
            logger.warning(f"  Could not load {sf.name}: {exc}")
            continue
        if n < 100:
            continue
        records.append(DayRecord(date_str=date_str, feat_path=ff,
                                 snap_path=sf, n_bars=n))

    records = records[:n_days] if n_days else records
    logger.info(f"Day registry: {len(records)} days "
                f"({records[0].date_str} to {records[-1].date_str})")
    return records


# ============================================================================
# Per-day loaders
# ============================================================================

def load_day_mid(rec: DayRecord) -> np.ndarray:
    """Load mid_prices for one day (float32, shape (n_bars,))."""
    d = np.load(str(rec.snap_path), allow_pickle=True)
    mp = d['mid_prices'].astype(np.float32)
    d.close()
    return mp


def load_day_features(rec: DayRecord) -> np.ndarray:
    """Load 340-feature array for one day (float32, shape (n_bars, 340))."""
    d = np.load(str(rec.feat_path))
    feats = d['mbo_features'].astype(np.float32)
    d.close()
    return feats


def load_window_features_and_mids(
    records: List[DayRecord],
    day_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Load features + mid_prices for a list of day indices (into records).
    Returns: (X float32, mids float32, boundaries list)
    """
    all_X = []
    all_m = []
    boundaries = [0]
    offset = 0
    for di in day_indices:
        rec = records[di]
        X = load_day_features(rec)
        m = load_day_mid(rec)
        assert len(X) == len(m), f"Shape mismatch on {rec.date_str}: X={len(X)} m={len(m)}"
        all_X.append(X)
        all_m.append(m)
        offset += len(m)
        boundaries.append(offset)
    Xraw = np.concatenate(all_X)
    np.clip(Xraw, -60000, 60000, out=Xraw)
    np.nan_to_num(Xraw, copy=False, nan=0.0, posinf=60000, neginf=-60000)
    X_cat = Xraw.astype(np.float16)
    del Xraw
    m_cat = np.concatenate(all_m)
    del all_X, all_m
    gc.collect()
    return X_cat, m_cat, boundaries


# ============================================================================
# Target computation (respects day boundaries)
# ============================================================================

def direction_target_for_window(mids: np.ndarray, hz_bars: int,
                                 boundaries: List[int]) -> np.ndarray:
    N = len(mids)
    n_days = len(boundaries) - 1
    tgt = np.full(N, np.nan, dtype=np.float32)
    for d in range(n_days):
        s, e = boundaries[d], boundaries[d + 1]
        dl = e - s
        if dl <= hz_bars:
            continue
        tgt[s:s + dl - hz_bars] = (
            mids[s + hz_bars:e] - mids[s:s + dl - hz_bars]
        ) / TICK_SIZE
    return tgt


def magnitude_target_for_window(mids: np.ndarray, hz_bars: int,
                                  boundaries: List[int]) -> np.ndarray:
    N = len(mids)
    n_days = len(boundaries) - 1
    tgt = np.full(N, np.nan, dtype=np.float32)
    for d in range(n_days):
        s, e = boundaries[d], boundaries[d + 1]
        dl = e - s
        if dl <= hz_bars:
            continue
        tgt[s:s + dl - hz_bars] = np.abs(
            mids[s + hz_bars:e] - mids[s:s + dl - hz_bars]
        ) / TICK_SIZE
    return tgt


# ============================================================================
# LightGBM training helpers
# ============================================================================

LGBM_BASE_PARAMS = dict(
    n_estimators=200,       # 200 vs 300: faster, IC barely changes
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_samples=100,
    verbose=-1,
    n_jobs=4,
    device='cpu',
    max_bin=63,
    force_row_wise=True,
)

MAX_TRAIN_ROWS = 400_000


def _fit_regressor(X_tr, y_tr, X_val, y_val):
    import lightgbm as lgb
    params = {**LGBM_BASE_PARAMS, 'objective': 'regression', 'metric': 'rmse'}
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


def _fit_binary(X_tr, y_tr, X_val, y_val):
    import lightgbm as lgb
    params = {**LGBM_BASE_PARAMS,
              'objective': 'binary', 'metric': 'auc',
              'min_child_samples': 50, 'is_unbalance': True}
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


def _subsample(X, y, rng, n=MAX_TRAIN_ROWS):
    valid = np.where(np.isfinite(y))[0]
    if len(valid) > n:
        valid = np.sort(rng.choice(valid, n, replace=False))
    return X[valid].astype(np.float32), y[valid]


# ============================================================================
# Entry data structures
# ============================================================================

@dataclass
class EntryBar:
    day_idx:      int    # index into records[]
    fill_bar_local: int  # bar offset within that day
    direction:    int    # +1 long, -1 short
    entry_price:  float
    dir_pred:     float
    mag_pred:     float


@dataclass
class TradeResult:
    entry_idx:   int
    day_idx:     int
    direction:   int
    bars_held:   int
    net_ticks:   float
    net_dollars: float
    exit_type:   str


# ============================================================================
# Step 2: Walk-forward entry model — day-by-day, rolling window
# ============================================================================

def train_entry_models_walk_forward(
    records: List[DayRecord],
    hz_bars: int,
    min_train_days: int = 5,
    max_train_days: int = 15,
    label: str = '',
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Walk-forward entry model training. For each test day:
      - Load rolling window of [max(0, test-max_train)..(test-2)] days  (1-day purge)
      - Train direction + magnitude models
      - Predict on test day features
      - Discard everything

    Returns:
        dir_preds_by_day  : list of float32 arrays, one per day (NaN where no pred)
        mag_preds_by_day  : same
    """
    n_days = len(records)
    dir_preds_by_day: List[Optional[np.ndarray]] = [None] * n_days
    mag_preds_by_day: List[Optional[np.ndarray]] = [None] * n_days

    t0 = time.time()
    n_folds = 0
    dir_ics: List[float] = []
    mag_ics: List[float] = []

    for test_day in range(min_train_days, n_days):
        train_end = test_day - 2       # 1-day purge gap
        train_start = max(0, test_day - max_train_days)
        if train_end < train_start:
            continue

        # Load training window
        train_days = list(range(train_start, train_end + 1))
        try:
            X_tr_raw, m_tr, b_tr = load_window_features_and_mids(records, train_days)
        except Exception as exc:
            logger.warning(f"  [{label}] train load failed day {test_day}: {exc}")
            continue

        y_dir_tr = direction_target_for_window(m_tr, hz_bars, b_tr)
        y_mag_tr = magnitude_target_for_window(m_tr, hz_bars, b_tr)
        del m_tr, b_tr
        gc.collect()

        # Load test day
        try:
            X_te_raw = load_day_features(records[test_day])
            m_te = load_day_mid(records[test_day])
        except Exception as exc:
            logger.warning(f"  [{label}] test load failed day {test_day}: {exc}")
            del X_tr_raw, y_dir_tr, y_mag_tr
            gc.collect()
            continue

        y_dir_te = direction_target_for_window(
            m_te, hz_bars, [0, len(m_te)])
        y_mag_te = magnitude_target_for_window(
            m_te, hz_bars, [0, len(m_te)])

        rng = np.random.default_rng(seed=test_day)

        # --- Direction model ---
        X_tr_d, y_tr_d = _subsample(X_tr_raw, y_dir_tr, rng)
        split = int(len(X_tr_d) * 0.85)
        te_valid = np.isfinite(y_dir_te)
        X_te_d = X_te_raw[te_valid].astype(np.float32)
        y_te_d = y_dir_te[te_valid]

        dp_day = np.full(len(m_te), np.nan, dtype=np.float32)
        if split > 100 and te_valid.sum() > 50:
            try:
                m_dir = _fit_regressor(X_tr_d[:split], y_tr_d[:split],
                                       X_tr_d[split:], y_tr_d[split:])
                p_dir = m_dir.predict(X_te_d).astype(np.float32)
                dp_day[te_valid] = p_dir
                if len(p_dir) > 10:
                    ic = float(spearmanr(p_dir, y_te_d)[0])
                    if np.isfinite(ic):
                        dir_ics.append(ic)
                del m_dir, p_dir
            except Exception as exc:
                logger.warning(f"  [{label}] dir model failed day {test_day}: {exc}")

        del X_tr_d, y_tr_d, X_te_d, y_te_d
        gc.collect()

        # --- Magnitude model ---
        X_tr_m, y_tr_m = _subsample(X_tr_raw, y_mag_tr, rng)
        X_te_m = X_te_raw.astype(np.float32)   # mag target valid same places as dir
        y_te_m = y_mag_te

        mp_day = np.full(len(m_te), np.nan, dtype=np.float32)
        mag_valid = np.isfinite(y_te_m)
        if split > 100 and mag_valid.sum() > 50:
            X_te_m_valid = X_te_m[mag_valid].astype(np.float32)
            y_te_m_valid = y_te_m[mag_valid]
            try:
                m_mag = _fit_regressor(X_tr_m[:split], y_tr_m[:split],
                                       X_tr_m[split:], y_tr_m[split:])
                p_mag = m_mag.predict(X_te_m_valid).astype(np.float32)
                mp_day[mag_valid] = p_mag
                if len(p_mag) > 10:
                    ic = float(spearmanr(p_mag, y_te_m_valid)[0])
                    if np.isfinite(ic):
                        mag_ics.append(ic)
                del m_mag, p_mag
            except Exception as exc:
                logger.warning(f"  [{label}] mag model failed day {test_day}: {exc}")
            del X_te_m_valid, y_te_m_valid

        del X_tr_raw, y_dir_tr, y_mag_tr, X_tr_m, y_tr_m, X_te_raw, X_te_m, y_mag_te
        del m_te
        gc.collect()

        dir_preds_by_day[test_day] = dp_day
        mag_preds_by_day[test_day] = mp_day

        n_folds += 1
        if n_folds % 5 == 0:
            d_ic = np.mean(dir_ics) if dir_ics else 0
            m_ic = np.mean(mag_ics) if mag_ics else 0
            elapsed = time.time() - t0
            logger.info(
                f"  [{label}] fold {n_folds} (day {test_day} / {n_days}): "
                f"dir_IC={d_ic:.4f} mag_IC={m_ic:.4f} [{elapsed:.0f}s]"
            )
            sys.stdout.flush()

    d_ic_mean = float(np.mean(dir_ics)) if dir_ics else 0.0
    m_ic_mean = float(np.mean(mag_ics)) if mag_ics else 0.0
    logger.info(
        f"  [{label}] DONE: {n_folds} folds, "
        f"dir_IC={d_ic_mean:.4f} mag_IC={m_ic_mean:.4f} "
        f"[{time.time()-t0:.0f}s]"
    )
    return dir_preds_by_day, mag_preds_by_day


# ============================================================================
# Step 3: Find entries per day (uses saved predictions + mid_prices)
# ============================================================================

def find_entries_for_day(
    day_idx:         int,
    mid_prices:      np.ndarray,        # full day mid prices
    dir_preds:       np.ndarray,        # full day direction predictions
    mag_preds:       np.ndarray,        # full day magnitude predictions
    signal_threshold: float,
    fill_horizon_bars: int = 100,
    min_bars_between:  int = MIN_BARS_BETWEEN,
) -> List[EntryBar]:
    """Find all filled entries on one day."""
    N = len(mid_prices)
    has_pred = np.isfinite(dir_preds) & np.isfinite(mag_preds)
    entries = []
    next_allowed = 0

    for i in range(N):
        if i < next_allowed:
            continue
        if not has_pred[i]:
            continue
        if mag_preds[i] < MAG_GATE_TICKS:
            continue
        if abs(dir_preds[i]) < signal_threshold:
            continue

        direction = 1 if dir_preds[i] > 0 else -1
        post = i + 1
        if post >= N:
            continue

        post_mid = mid_prices[post]
        entry_limit = post_mid - HALF_TICK if direction == 1 else post_mid + HALF_TICK

        fill_end = min(post + fill_horizon_bars, N)
        for j in range(post + 1, fill_end):
            m = mid_prices[j]
            if direction == 1 and m <= entry_limit:
                entries.append(EntryBar(
                    day_idx=day_idx, fill_bar_local=j,
                    direction=direction, entry_price=float(entry_limit),
                    dir_pred=float(dir_preds[i]), mag_pred=float(mag_preds[i]),
                ))
                break
            elif direction == -1 and m >= entry_limit:
                entries.append(EntryBar(
                    day_idx=day_idx, fill_bar_local=j,
                    direction=direction, entry_price=float(entry_limit),
                    dir_pred=float(dir_preds[i]), mag_pred=float(mag_preds[i]),
                ))
                break
        next_allowed = i + min_bars_between

    return entries


# ============================================================================
# Step 4: Monitoring dataset + adverse flow detector
# ============================================================================

def build_monitoring_rows(
    entries:      List[EntryBar],
    features_day: np.ndarray,    # (n_bars, 340) for this day
    mid_prices:   np.ndarray,    # (n_bars,)
    hz_bars:      int,
    loss_thresh:  float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Build one monitoring sample per (entry, bar_held).
    Features = [340 original | unrealized_pnl | bars_held_norm | max_adverse | direction]
    Label = 1 if final PnL (at timeout) < -loss_thresh ticks.
    Returns X (M,344), y (M,), entry_local_ids (M,)
    """
    N = len(mid_prices)
    n_base = features_day.shape[1]
    n_extra = 4
    total_cols = n_base + n_extra

    rows_X = []
    rows_y = []
    rows_eid = []

    for local_eid, entry in enumerate(entries):
        fill = entry.fill_bar_local
        ep = entry.entry_price
        direction = entry.direction
        exit_end = min(fill + hz_bars, N)

        # Final PnL at timeout (market exit)
        final_mid = mid_prices[exit_end - 1]
        final_pnl = (final_mid - ep) / TICK_SIZE * direction - COMMISSION_TICKS
        label = 1.0 if final_pnl < -loss_thresh else 0.0

        max_adverse = 0.0
        for b in range(fill + 1, exit_end):
            cur_mid = mid_prices[b]
            unrealized = (cur_mid - ep) / TICK_SIZE * direction
            adverse_now = -unrealized
            if adverse_now > max_adverse:
                max_adverse = adverse_now

            bars_held = b - fill
            base = features_day[b].astype(np.float32)
            extras = np.array([
                float(unrealized),
                float(bars_held) / max(hz_bars, 1),
                float(max_adverse),
                float(direction),
            ], dtype=np.float32)

            row = np.empty(total_cols, dtype=np.float32)
            row[:n_base] = base
            row[n_base:] = extras
            rows_X.append(row)
            rows_y.append(label)
            rows_eid.append(local_eid)

    if not rows_X:
        return (np.empty((0, total_cols), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                [])

    return np.stack(rows_X), np.array(rows_y, dtype=np.float32), rows_eid


# ============================================================================
# Step 5: Walk-forward adverse detector (day-by-day)
# ============================================================================

def train_adverse_detector_wf(
    records:       List[DayRecord],
    entries_all:   List[EntryBar],
    mid_by_day:    List[np.ndarray],
    hz_bars:       int,
    min_train_days: int = 6,
    max_train_days: int = 20,
    loss_thresh:   float = 0.5,
    label: str = 'adverse',
) -> List[Optional[np.ndarray]]:
    """
    Walk-forward adverse flow detector.
    Training: monitoring rows from entries in days [train_start..test_day-2]
    (1-day purge: skip test_day-1)
    Prediction: entries on test_day
    Returns p_loss_by_entry: list len=len(entries_all), each element is a
    float32 array of P(loss) at each monitoring bar, or None if no prediction.
    """
    import lightgbm as lgb

    n_days = len(records)

    # Group entries by day
    entries_by_day: Dict[int, List[Tuple[int, EntryBar]]] = {}
    for eid, entry in enumerate(entries_all):
        d = entry.day_idx
        entries_by_day.setdefault(d, []).append((eid, entry))

    p_loss_by_entry: List[Optional[np.ndarray]] = [None] * len(entries_all)

    t0 = time.time()
    n_folds = 0
    aucs: List[float] = []

    for test_day in range(min_train_days, n_days):
        # Purge gap: skip test_day-1
        train_start = max(0, test_day - max_train_days)
        train_end = test_day - 2
        if train_end < train_start:
            continue

        test_entries_info = entries_by_day.get(test_day, [])
        if not test_entries_info:
            continue

        # ----- Build training monitoring dataset -----
        train_X_parts = []
        train_y_parts = []
        for d in range(train_start, train_end + 1):
            if d not in entries_by_day:
                continue
            day_entries = [e for _, e in entries_by_day[d]]
            if not day_entries:
                continue
            try:
                feats_d = load_day_features(records[d])
            except Exception:
                continue
            Xd, yd, _ = build_monitoring_rows(
                day_entries, feats_d, mid_by_day[d], hz_bars, loss_thresh)
            del feats_d
            gc.collect()
            if len(Xd) > 0:
                train_X_parts.append(Xd)
                train_y_parts.append(yd)

        if not train_X_parts:
            continue

        X_tr = np.concatenate(train_X_parts)
        y_tr = np.concatenate(train_y_parts)
        del train_X_parts, train_y_parts
        gc.collect()

        if len(X_tr) < 200 or y_tr.sum() < 10 or (1 - y_tr).sum() < 10:
            del X_tr, y_tr
            gc.collect()
            continue

        # Subsample
        rng = np.random.default_rng(seed=test_day)
        MAX_MON = 200_000
        if len(X_tr) > MAX_MON:
            idx = np.sort(rng.choice(len(X_tr), MAX_MON, replace=False))
            X_tr = X_tr[idx]
            y_tr = y_tr[idx]

        # ----- Build test monitoring dataset -----
        test_entries_list = [e for _, e in test_entries_info]
        try:
            feats_test = load_day_features(records[test_day])
        except Exception as exc:
            logger.warning(f"  [{label}] test feat load failed day {test_day}: {exc}")
            del X_tr, y_tr
            gc.collect()
            continue

        X_te, y_te, entry_ids_te = build_monitoring_rows(
            test_entries_list, feats_test, mid_by_day[test_day], hz_bars, loss_thresh)
        del feats_test
        gc.collect()

        if len(X_te) < 5:
            del X_tr, y_tr, X_te, y_te
            gc.collect()
            continue

        # ----- Train -----
        split = int(len(X_tr) * 0.85)
        try:
            model = _fit_binary(X_tr[:split], y_tr[:split],
                                X_tr[split:], y_tr[split:])
            p_loss = model.predict_proba(X_te)[:, 1].astype(np.float32)
            del model
        except Exception as exc:
            logger.warning(f"  [{label}] fold {test_day} failed: {exc}")
            del X_tr, y_tr, X_te, y_te
            gc.collect()
            continue

        del X_tr, y_tr
        gc.collect()

        # AUC
        if len(y_te) > 20 and y_te.sum() > 2 and (1-y_te).sum() > 2:
            try:
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(y_te, p_loss))
                aucs.append(auc)
            except Exception:
                pass

        # Map predictions back
        entry_ids_arr = np.array(entry_ids_te, dtype=np.int32)
        for local_eid, (global_eid, _) in enumerate(test_entries_info):
            mask = entry_ids_arr == local_eid
            if mask.sum() == 0:
                continue
            p_loss_by_entry[global_eid] = p_loss[mask]

        del X_te, y_te, p_loss, entry_ids_arr
        gc.collect()

        n_folds += 1
        if n_folds % 5 == 0:
            auc_str = f"AUC={np.mean(aucs):.4f}" if aucs else "AUC=N/A"
            logger.info(f"  [{label}] fold {n_folds} (day {test_day}/{n_days}): "
                        f"{auc_str} [{time.time()-t0:.0f}s]")
            sys.stdout.flush()

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    logger.info(f"  [{label}] DONE: {n_folds} folds, mean AUC={mean_auc:.4f} "
                f"[{time.time()-t0:.0f}s]")
    return p_loss_by_entry


# ============================================================================
# Step 6: Backtests — fixed / adaptive / flip
# ============================================================================

def _pnl_at_bar(entry: EntryBar, exit_bar_local: int, mid_prices: np.ndarray,
                is_market_exit: bool) -> TradeResult:
    N = len(mid_prices)
    eb = min(exit_bar_local, N - 1)
    exit_mid = mid_prices[eb]
    ep = entry.entry_price
    direction = entry.direction
    dir_pnl = (exit_mid - ep) / TICK_SIZE * direction
    exit_edge = -0.5 if is_market_exit else 0.5
    net_ticks = dir_pnl + exit_edge - COMMISSION_TICKS
    net_dollars = net_ticks * TICK_VALUE
    return TradeResult(
        entry_idx=0, day_idx=entry.day_idx, direction=direction,
        bars_held=eb - entry.fill_bar_local,
        net_ticks=net_ticks, net_dollars=net_dollars, exit_type='',
    )


def backtest_fixed(
    entries:      List[EntryBar],
    mid_by_day:   List[np.ndarray],
    hz_bars:      int,
) -> List[TradeResult]:
    results = []
    for i, e in enumerate(entries):
        N = len(mid_by_day[e.day_idx])
        exit_bar = min(e.fill_bar_local + hz_bars, N - 1)
        r = _pnl_at_bar(e, exit_bar, mid_by_day[e.day_idx], is_market_exit=True)
        r.entry_idx = i
        r.exit_type = 'timeout'
        results.append(r)
    return results


def backtest_adaptive(
    entries:           List[EntryBar],
    p_loss_by_entry:   List[Optional[np.ndarray]],
    mid_by_day:        List[np.ndarray],
    hz_bars:           int,
    threshold:         float = 0.70,
    take_profit_ticks: Optional[float] = None,
) -> List[TradeResult]:
    results = []
    n_early = 0
    n_timeout = 0
    n_nosig = 0

    for i, entry in enumerate(entries):
        mids = mid_by_day[entry.day_idx]
        N = len(mids)
        fill = entry.fill_bar_local
        ep = entry.entry_price
        direction = entry.direction
        exit_end = min(fill + hz_bars, N)
        p_arr = p_loss_by_entry[i]

        if p_arr is None:
            # No detector available — fall back to fixed hold
            exit_bar = exit_end - 1
            r = _pnl_at_bar(entry, exit_bar, mids, is_market_exit=True)
            r.entry_idx = i; r.exit_type = 'timeout_no_signal'
            results.append(r); n_nosig += 1; continue

        exited_early = False
        for b_local, b_global in enumerate(range(fill + 1, exit_end)):
            if b_local >= len(p_arr):
                break
            unrealized = (mids[b_global] - ep) / TICK_SIZE * direction

            if take_profit_ticks is not None and unrealized >= take_profit_ticks:
                r = _pnl_at_bar(entry, b_global, mids, is_market_exit=True)
                r.entry_idx = i; r.exit_type = 'take_profit'
                results.append(r); exited_early = True; n_early += 1; break

            if p_arr[b_local] > threshold:
                r = _pnl_at_bar(entry, b_global, mids, is_market_exit=True)
                r.entry_idx = i; r.exit_type = 'adaptive_exit'
                results.append(r); exited_early = True; n_early += 1; break

        if not exited_early:
            exit_bar = exit_end - 1
            r = _pnl_at_bar(entry, exit_bar, mids, is_market_exit=True)
            r.entry_idx = i; r.exit_type = 'timeout'
            results.append(r); n_timeout += 1

    logger.info(f"    Adaptive (t={threshold}): early={n_early} "
                f"({n_early/max(len(entries),1):.1%}), timeout={n_timeout}, "
                f"no_signal={n_nosig}")
    return results


def backtest_flip(
    entries:          List[EntryBar],
    dir_preds_by_day: List[Optional[np.ndarray]],
    mid_by_day:       List[np.ndarray],
    hz_bars:          int,
) -> List[TradeResult]:
    results = []
    n_flip = 0
    n_timeout = 0

    for i, entry in enumerate(entries):
        mids = mid_by_day[entry.day_idx]
        N = len(mids)
        fill = entry.fill_bar_local
        direction = entry.direction
        exit_end = min(fill + hz_bars, N)
        dp = dir_preds_by_day[entry.day_idx]

        exited_flip = False
        if dp is not None:
            for b_global in range(fill + 1, exit_end):
                if not np.isfinite(dp[b_global]):
                    continue
                current_dir = 1 if dp[b_global] > 0 else -1
                if current_dir != direction:
                    r = _pnl_at_bar(entry, b_global, mids, is_market_exit=True)
                    r.entry_idx = i; r.exit_type = 'flip_exit'
                    results.append(r); exited_flip = True; n_flip += 1; break

        if not exited_flip:
            exit_bar = exit_end - 1
            r = _pnl_at_bar(entry, exit_bar, mids, is_market_exit=True)
            r.entry_idx = i; r.exit_type = 'timeout'
            results.append(r); n_timeout += 1

    logger.info(f"    Flip exit: flipped={n_flip} "
                f"({n_flip/max(len(entries),1):.1%}), timeout={n_timeout}")
    return results


# ============================================================================
# Step 7: Metrics
# ============================================================================

def compute_metrics(trades: List[TradeResult], n_days: int, label: str) -> Dict:
    if not trades:
        logger.info(f"  [{label}] No trades.")
        return {'label': label, 'n_trades': 0}

    net_t = np.array([t.net_ticks for t in trades])
    net_d = np.array([t.net_dollars for t in trades])

    day_pnl: Dict[int, float] = {}
    for t in trades:
        day_pnl[t.day_idx] = day_pnl.get(t.day_idx, 0.0) + t.net_dollars
    day_arr = np.array(list(day_pnl.values()))

    sharpe = 0.0
    if len(day_arr) > 2 and day_arr.std() > 0:
        sharpe = float(day_arr.mean() / day_arr.std() * np.sqrt(252))

    gross_w = float(net_t[net_t > 0].sum()) if (net_t > 0).any() else 0.0
    gross_l = float(abs(net_t[net_t < 0].sum())) if (net_t < 0).any() else 0.001
    pf = gross_w / gross_l

    cum = np.cumsum(net_d)
    max_dd = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) > 0 else 0.0

    winners = net_t[net_t > 0]
    losers  = net_t[net_t < 0]

    exit_counts: Dict[str, int] = {}
    for t in trades:
        exit_counts[t.exit_type] = exit_counts.get(t.exit_type, 0) + 1

    early_exits = [t for t in trades if t.exit_type in ('adaptive_exit', 'flip_exit')]
    early_losing = sum(1 for t in early_exits if t.net_ticks < 0)

    return {
        'label':               label,
        'n_trades':            len(trades),
        'trades_per_day':      len(trades) / max(n_days, 1),
        'total_pnl_ticks':     float(net_t.sum()),
        'total_pnl_dollars':   float(net_d.sum()),
        'mean_pnl_ticks':      float(net_t.mean()),
        'mean_pnl_dollars':    float(net_d.mean()),
        'median_pnl_ticks':    float(np.median(net_t)),
        'win_rate':            float((net_t > 0).mean()),
        'profit_factor':       pf,
        'sharpe_annualized':   sharpe,
        'max_drawdown_dollars': max_dd,
        'avg_winner_ticks':    float(winners.mean()) if len(winners) > 0 else 0.0,
        'avg_loser_ticks':     float(losers.mean()) if len(losers) > 0 else 0.0,
        'n_winners':           int((net_t > 0).sum()),
        'n_losers':            int((net_t < 0).sum()),
        'n_positive_days':     int((day_arr > 0).sum()),
        'n_negative_days':     int((day_arr <= 0).sum()),
        'day_pnl_mean':        float(day_arr.mean()) if len(day_arr) > 0 else 0.0,
        'day_pnl_std':         float(day_arr.std()) if len(day_arr) > 1 else 0.0,
        'n_early_exits':       len(early_exits),
        'pct_early_exits':     len(early_exits) / max(len(trades), 1),
        'early_exit_losing':   early_losing,
        'exit_type_counts':    exit_counts,
    }


def print_comparison(metrics_list: List[Dict]) -> None:
    sep = "=" * 110
    logger.info(f"\n{sep}")
    logger.info("RESULTS COMPARISON — Adaptive Exit Model")
    logger.info(sep)

    hdr = (f"{'Strategy':<38} {'N':>5} {'TotalPnL':>10} {'Sharpe':>7} "
           f"{'PF':>6} {'WR':>7} {'AvgW':>7} {'AvgL':>7} {'EarlyExit%':>12}")
    logger.info(hdr)
    logger.info("-" * 110)

    for m in metrics_list:
        if not m or m.get('n_trades', 0) == 0:
            continue
        row = (
            f"{m['label']:<38} "
            f"{m['n_trades']:>5,} "
            f"${m['total_pnl_dollars']:>+9,.0f} "
            f"{m['sharpe_annualized']:>7.2f} "
            f"{m['profit_factor']:>6.2f} "
            f"{m['win_rate']:>7.1%} "
            f"{m['avg_winner_ticks']:>+7.2f}t "
            f"{m['avg_loser_ticks']:>+7.2f}t "
            f"{m['pct_early_exits']:>12.1%}"
        )
        logger.info(row)
    logger.info(sep)

    for m in metrics_list:
        if not m or m.get('n_trades', 0) == 0:
            continue
        logger.info(f"\n--- {m['label']} ---")
        logger.info(f"  Trades: {m['n_trades']:,} ({m['trades_per_day']:.1f}/day)")
        logger.info(f"  Total PnL: ${m['total_pnl_dollars']:+,.0f}  "
                    f"({m['total_pnl_ticks']:+.1f}t)")
        logger.info(f"  Mean/trade: ${m['mean_pnl_dollars']:+.2f}  "
                    f"({m['mean_pnl_ticks']:+.4f}t)")
        logger.info(f"  Sharpe: {m['sharpe_annualized']:.3f}  PF: {m['profit_factor']:.3f}")
        logger.info(f"  WinRate: {m['win_rate']:.1%}  "
                    f"Avg winner: {m['avg_winner_ticks']:+.3f}t  "
                    f"Avg loser: {m['avg_loser_ticks']:+.3f}t")
        logger.info(f"  MaxDD: ${m['max_drawdown_dollars']:,.0f}")
        logger.info(f"  Days: {m['n_positive_days']} pos / "
                    f"{m['n_negative_days']} neg  "
                    f"(${m['day_pnl_mean']:+.0f}/day avg, std=${m['day_pnl_std']:.0f})")
        if m['n_early_exits'] > 0:
            logger.info(f"  Early exits: {m['n_early_exits']} "
                        f"({m['pct_early_exits']:.1%}) — "
                        f"{m['early_exit_losing']} were losers cut early")
        logger.info(f"  Exit breakdown: {m['exit_type_counts']}")


# ============================================================================
# Main
# ============================================================================

def run(args):
    t_total = time.time()
    logger.info("=" * 70)
    logger.info("ADAPTIVE EXIT MODEL — Lvl3Quant ES Futures")
    logger.info(f"n_days={args.n_days}  quick={args.quick}")
    logger.info(f"Memory-efficient: loads features day-by-day, no full concat")
    logger.info("=" * 70)

    # --- Step 1: Build day registry ---
    logger.info("\nStep 1: Enumerating days and loading mid_prices...")
    records = build_day_registry(n_days=args.n_days)
    n_days = len(records)

    # Pre-load all mid_prices (tiny: 70 days x 234k bars x 4 bytes = 65 MB)
    logger.info("  Loading all mid_prices into RAM (~65 MB for 70 days)...")
    mid_by_day: List[np.ndarray] = []
    for rec in records:
        mid_by_day.append(load_day_mid(rec))
    logger.info(f"  Mid prices loaded: {sum(len(m) for m in mid_by_day):,} bars total")

    horizons = {'10s': 100, '30s': 300}
    if args.quick:
        horizons = {'10s': 100}

    all_metrics = []

    for hz_name, hz_bars in horizons.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"HORIZON: {hz_name} ({hz_bars} bars = {hz_bars/BARS_PER_SEC:.0f}s)")
        logger.info(f"{'='*70}")

        # --- Step 2: Walk-forward entry models ---
        logger.info(f"\nStep 2: Walk-forward entry models ({hz_name})...")
        dir_preds_by_day, mag_preds_by_day = train_entry_models_walk_forward(
            records, hz_bars,
            min_train_days=5,
            max_train_days=15,  # 15 days rolling: ~4.5 GB per fold, faster than 20
            label=f'entry_{hz_name}',
        )

        # Compute signal threshold from all valid direction predictions
        all_abs_dir = []
        for dp in dir_preds_by_day:
            if dp is not None:
                valid = dp[np.isfinite(dp)]
                if len(valid) > 0:
                    all_abs_dir.append(np.abs(valid))
        if not all_abs_dir:
            logger.warning(f"No valid direction predictions for {hz_name}, skipping.")
            continue
        abs_concat = np.concatenate(all_abs_dir)
        signal_threshold = float(np.percentile(abs_concat, SIGNAL_QUANTILE * 100))
        logger.info(f"  Signal threshold (top {(1-SIGNAL_QUANTILE)*100:.0f}%): "
                    f"{signal_threshold:.6f}")
        del all_abs_dir, abs_concat
        gc.collect()

        # --- Step 3: Find entries for each day ---
        logger.info(f"\nStep 3: Finding entry bars ({hz_name})...")
        entries_all: List[EntryBar] = []
        for d in range(n_days):
            dp = dir_preds_by_day[d]
            mp = mag_preds_by_day[d]
            if dp is None or mp is None:
                continue
            day_entries = find_entries_for_day(
                day_idx=d,
                mid_prices=mid_by_day[d],
                dir_preds=dp,
                mag_preds=mp,
                signal_threshold=signal_threshold,
                fill_horizon_bars=100,
                min_bars_between=MIN_BARS_BETWEEN,
            )
            entries_all.extend(day_entries)

        logger.info(f"  Total entries found: {len(entries_all):,} "
                    f"({len(entries_all)/max(n_days,1):.1f}/day)")

        if len(entries_all) < 10:
            logger.warning(f"  Too few entries for {hz_name}, skipping.")
            continue

        # --- Step 4: Fixed hold baseline ---
        logger.info(f"\nStep 4a: Backtest — FIXED HOLD ({hz_name}, {hz_bars} bars)...")
        fixed_trades = backtest_fixed(entries_all, mid_by_day, hz_bars)
        m_fixed = compute_metrics(fixed_trades, n_days, f'fixed_hold_{hz_name}')
        all_metrics.append(m_fixed)
        logger.info(f"  Fixed hold: ${m_fixed['total_pnl_dollars']:+,.0f}  "
                    f"Sharpe={m_fixed['sharpe_annualized']:.3f}  "
                    f"WR={m_fixed['win_rate']:.1%}  PF={m_fixed['profit_factor']:.3f}")

        # --- Step 5: Flip exit ---
        logger.info(f"\nStep 4b: Backtest — FLIP EXIT ({hz_name})...")
        flip_trades = backtest_flip(entries_all, dir_preds_by_day, mid_by_day, hz_bars)
        m_flip = compute_metrics(flip_trades, n_days, f'flip_exit_{hz_name}')
        all_metrics.append(m_flip)
        logger.info(f"  Flip exit: ${m_flip['total_pnl_dollars']:+,.0f}  "
                    f"Sharpe={m_flip['sharpe_annualized']:.3f}  "
                    f"WR={m_flip['win_rate']:.1%}  PF={m_flip['profit_factor']:.3f}")

        # --- Step 6: Train adverse flow detector ---
        logger.info(f"\nStep 5: Training adverse flow detector ({hz_name})...")
        p_loss_by_entry = train_adverse_detector_wf(
            records=records,
            entries_all=entries_all,
            mid_by_day=mid_by_day,
            hz_bars=hz_bars,
            min_train_days=6,
            max_train_days=15,
            loss_thresh=0.5,
            label=f'adverse_{hz_name}',
        )

        has_pred = sum(1 for p in p_loss_by_entry if p is not None)
        logger.info(f"  Adverse detector: {has_pred}/{len(entries_all)} entries have predictions")

        # --- Step 7: Adaptive exit at multiple thresholds ---
        thresholds = [0.55, 0.60, 0.65, 0.70] if not args.quick else [0.55, 0.65]
        for thresh in thresholds:
            logger.info(f"\nStep 6: Backtest — ADAPTIVE EXIT ({hz_name}, thresh={thresh})...")
            adaptive_trades = backtest_adaptive(
                entries=entries_all,
                p_loss_by_entry=p_loss_by_entry,
                mid_by_day=mid_by_day,
                hz_bars=hz_bars,
                threshold=thresh,
                take_profit_ticks=None,
            )
            m_adap = compute_metrics(
                adaptive_trades, n_days,
                f'adaptive_{hz_name}_t{thresh:.2f}')
            all_metrics.append(m_adap)
            logger.info(f"  Adaptive (t={thresh}): "
                        f"${m_adap['total_pnl_dollars']:+,.0f}  "
                        f"Sharpe={m_adap['sharpe_annualized']:.3f}  "
                        f"WR={m_adap['win_rate']:.1%}  "
                        f"PF={m_adap['profit_factor']:.3f}  "
                        f"early={m_adap['pct_early_exits']:.1%}")

        # Release per-horizon data
        del dir_preds_by_day, mag_preds_by_day, entries_all, p_loss_by_entry
        del fixed_trades, flip_trades
        gc.collect()

    # --- Final comparison ---
    print_comparison(all_metrics)

    # Save JSON
    out_path = RESULTS_DIR / f"adaptive_exit_{_ts}.json"
    try:
        with open(str(out_path), 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        logger.info(f"\nResults saved: {out_path}")
    except Exception as exc:
        logger.warning(f"Could not save JSON: {exc}")

    elapsed = time.time() - t_total
    logger.info(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    logger.info(f"Log: {_log_file}")
    return all_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive Exit Model for ES Futures")
    p.add_argument('--n-days', type=int, default=70,
                   help="Number of trading days (default: 70)")
    p.add_argument('--quick', action='store_true',
                   help="Quick mode: 10s horizon only, single threshold")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
