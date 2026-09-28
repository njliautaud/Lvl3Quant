#!/usr/bin/env python3
"""
Vol Signal Writer — Trains an intraday vol LightGBM model on recent MBO data
and writes a fresh prediction to vol_signal.json for the VolModel to consume.

This bridges the Lvl3Quant vol model (IC=0.644 at 30min, IC=0.674 at 1s) to
the prediction markets scanner. Because the walk-forward protocol never saved
model artifacts, this script retrains a model on the last N_TRAIN_DAYS of MBO
data and predicts the current intraday vol level.

Two modes:
  1. SINGLE:   Train once, write signal, exit. Used by cron / task scheduler.
  2. DAEMON:   Loop every REFRESH_INTERVAL seconds during market hours.

Usage:
  python vol_signal_writer.py               # single-shot prediction
  python vol_signal_writer.py --daemon      # loop every 5 minutes
  python vol_signal_writer.py --horizon 30min --train-days 30

The signal file is read by VolModel._read_signal_file() in kalshi_client.py.
Signal format:
  {
    "timestamp": "2026-03-04T10:30:00.123456",
    "annualized_vol": 0.213,           # annualized rvol predicted, decimal
    "raw_prediction_pct": 21.3,        # annualized rvol %, float
    "z_score": 0.41,                   # z-score vs baseline mean/std
    "horizon": "30min",                # which horizon this applies to
    "model_ic": 0.644,                 # walk-forward IC for this horizon
    "n_train_days": 30,                # how many days were used
    "trailing_rvol_pct": 19.5,        # trailing realized vol (naive baseline)
    "confidence": 0.68                 # confidence [0-1], based on IC
  }

Environment variables:
  LVL3QUANT_PATH   Path to the Lvl3Quant repo root (default: ../Lvl3Quant)
  VOL_SIGNAL_FILE  Override output path (default: data/vol_signal.json next to this file)
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_SIGNAL_FILE = DATA_DIR / "vol_signal.json"
SIGNAL_FILE = Path(os.environ.get("VOL_SIGNAL_FILE", str(DEFAULT_SIGNAL_FILE)))

LVL3_ROOT = Path(os.environ.get(
    "LVL3QUANT_PATH",
    str(THIS_DIR.parents[1] / ".." / "Lvl3Quant")
)).resolve()
MBO_CACHE_DIR = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"

# Add Lvl3Quant root to path so we can import mbo_features
if str(LVL3_ROOT) not in sys.path:
    sys.path.insert(0, str(LVL3_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [vol_writer] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("vol_writer")

# ---------------------------------------------------------------------------
# Constants (must match intraday_vol_prediction.py exactly)
# ---------------------------------------------------------------------------
EXCLUDE_FEATURES = [0, 3, 8, 9]   # mid, microprice, best_bid, best_ask
BARS_PER_SECOND = 10               # 100ms bars → 10 bars/second
SAMPLE_INTERVAL = 100              # sample every 10 seconds

# Horizon definitions: name → horizon_bars
HORIZONS = {
    "1s":    100,                        # 1 second = 10 bars
    "30min": 1800 * BARS_PER_SECOND,     # 18,000 bars
    "1hr":   3600 * BARS_PER_SECOND,     # 36,000 bars
    "2hr":   7200 * BARS_PER_SECOND,     # 72,000 bars
    "4hr":   14400 * BARS_PER_SECOND,    # 144,000 bars
}

# Walk-forward calibration (IC, std) from validated results.
# These are updated by _load_calibration_from_results() if result files exist.
CALIBRATION_DEFAULTS = {
    "1s":    {"mean_ic": 0.674, "std_ic": 0.118},
    "30min": {"mean_ic": 0.644, "std_ic": 0.256},
    "1hr":   {"mean_ic": 0.568, "std_ic": 0.294},
    "2hr":   {"mean_ic": 0.440, "std_ic": 0.421},
    "4hr":   {"mean_ic": 0.406, "std_ic": 0.474},
}

# Long-run SPX annualized rvol statistics (used for z-score conversion)
RVOL_MEAN_PCT = 18.0   # annualized %
RVOL_STD_PCT  = 8.0    # vol-of-vol

# LightGBM hyperparams (same as intraday_vol_prediction.py)
LGB_PARAMS = {
    "objective":         "regression",
    "metric":            "mse",
    "learning_rate":     0.05,
    "num_leaves":        63,
    "max_depth":         6,
    "min_child_samples": 200,
    "subsample":         0.7,
    "colsample_bytree":  0.7,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "verbose":           -1,
    "n_jobs":            -1,
    "seed":              42,
}

MIN_TRAIN_DAYS = 15
N_BOOST_ROUNDS = 100


# ---------------------------------------------------------------------------
# Data helpers (identical logic to intraday_vol_prediction.py)
# ---------------------------------------------------------------------------

def load_day(fpath: Path):
    """Load one day's MBO features. Returns (features_N, mid_prices_N)."""
    data = np.load(str(fpath))
    raw = data["mbo_features"]
    mid = raw[:, 0].copy()

    # Forward-fill NaN
    mask = np.isnan(mid)
    if mask.any():
        first_valid = int(np.argmax(~mask))
        if first_valid > 0:
            mid[:first_valid] = mid[first_valid]
        for i in range(1, len(mid)):
            if np.isnan(mid[i]):
                mid[i] = mid[i - 1]

    keep = [i for i in range(raw.shape[1]) if i not in EXCLUDE_FEATURES]
    features = raw[:, keep].astype(np.float32)
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    del raw
    return features, mid


def compute_forward_rvol(mid: np.ndarray, horizon_bars: int) -> np.ndarray:
    """Compute forward annualized realized vol for each 10s sample bar.

    Returns an array of length N (same as mid), with NaN where not computable.
    Units: annualized percent (e.g. 20.0 means 20% annualized rvol).
    """
    N = len(mid)
    step = BARS_PER_SECOND   # 1 second = 10 bars

    prices = mid[::step]
    n_prices = len(prices)
    returns = np.diff(np.log(prices))
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    horizon_seconds = horizon_bars // BARS_PER_SECOND
    n_ret = len(returns)

    if n_ret < horizon_seconds:
        return np.full(N, np.nan)

    r2 = returns ** 2
    cumsum_r2 = np.concatenate(([0.0], np.cumsum(r2)))

    n_valid = n_ret - horizon_seconds + 1
    sum_r2 = (
        cumsum_r2[horizon_seconds : horizon_seconds + n_valid]
        - cumsum_r2[:n_valid]
    )
    seconds_per_year = 23400 * 252
    rvol = np.sqrt(sum_r2 / horizon_seconds) * np.sqrt(seconds_per_year) * 100.0

    fwd_rvol = np.full(N, np.nan)
    for i in range(min(n_valid, N // step)):
        bar_idx = i * step
        if bar_idx < N:
            fwd_rvol[bar_idx] = rvol[i]

    return fwd_rvol


def compute_trailing_rvol(mid: np.ndarray, lookback_bars: int) -> float:
    """Compute a single trailing realized vol estimate using the last lookback_bars.

    Used as the naive baseline. Returns annualized percent.
    """
    if len(mid) < lookback_bars + 2:
        return RVOL_MEAN_PCT

    prices = mid[-lookback_bars::BARS_PER_SECOND]
    if len(prices) < 2:
        return RVOL_MEAN_PCT

    returns = np.diff(np.log(prices))
    returns = np.nan_to_num(returns, nan=0.0)

    if len(returns) < 2:
        return RVOL_MEAN_PCT

    seconds_per_year = 23400 * 252
    rvol_pct = float(np.std(returns) * np.sqrt(seconds_per_year) * 100.0)
    return max(3.0, min(100.0, rvol_pct))


def prepare_day_data(fpath: Path, horizon_bars: int):
    """Load one day and return sampled (X, y) arrays ready for LightGBM."""
    features, mid = load_day(fpath)
    fwd_vol = compute_forward_rvol(mid, horizon_bars)

    indices = np.arange(0, len(features), SAMPLE_INTERVAL)
    X = features[indices]
    y = fwd_vol[indices]

    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & (y > 0)
    X = X[valid].astype(np.float32)
    y = y[valid].astype(np.float32)

    del features, mid, fwd_vol
    gc.collect()

    return X, y


# ---------------------------------------------------------------------------
# Calibration loader
# ---------------------------------------------------------------------------

def load_calibration():
    """Load walk-forward IC stats from Lvl3Quant result files if available."""
    cal = {k: dict(v) for k, v in CALIBRATION_DEFAULTS.items()}

    results_dir = LVL3_ROOT / "alpha_discovery" / "results"
    if not results_dir.exists():
        return cal

    # Multi-horizon results
    intraday_files = sorted(results_dir.glob("intraday_vol_pred_*.json"))
    if intraday_files:
        try:
            with open(intraday_files[-1]) as f:
                data = json.load(f)
            for hz, hz_data in data.items():
                if hz in cal:
                    cal[hz]["mean_ic"] = hz_data["mean_ic"]
                    cal[hz]["std_ic"]  = hz_data["std_ic"]
            log.info(f"Loaded multi-horizon calibration from {intraday_files[-1].name}")
        except Exception as e:
            log.warning(f"Could not load multi-horizon calibration: {e}")

    # 1s rvol results
    rvol_files = sorted(results_dir.glob("vol_pred_rvol_1s_*.json"))
    if rvol_files:
        try:
            with open(rvol_files[-1]) as f:
                data = json.load(f)
            cal["1s"]["mean_ic"] = data["mean_ic"]
            cal["1s"]["std_ic"]  = data["std_ic"]
            log.info(f"Loaded 1s calibration: IC={data['mean_ic']:.3f}")
        except Exception as e:
            log.warning(f"Could not load 1s rvol calibration: {e}")

    return cal


# ---------------------------------------------------------------------------
# Core: train model and make prediction
# ---------------------------------------------------------------------------

def train_and_predict(
    horizon: str = "30min",
    n_train_days: int = 30,
    verbose: bool = True,
) -> dict:
    """Train a LightGBM vol model on recent MBO data and return a prediction.

    The model is trained on the last n_train_days days of MBO features,
    predicting forward realized vol at the given horizon. Prediction is
    made using the most recent available features (last hour of most recent day).

    Args:
        horizon:      One of "1s", "30min", "1hr", "2hr", "4hr".
        n_train_days: How many past trading days to train on.
        verbose:      Whether to log training progress.

    Returns:
        dict with:
          raw_prediction_pct   float  — model predicted annualized rvol %
          annualized_vol       float  — same but in decimal (÷100)
          z_score              float  — z-score vs long-run mean/std
          trailing_rvol_pct    float  — naive trailing rvol baseline
          horizon              str
          model_ic             float  — walk-forward IC for this horizon
          n_train_days         int
          confidence           float
          timestamp            str
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise RuntimeError(
            "LightGBM is not installed. Run: pip install lightgbm"
        )

    if horizon not in HORIZONS:
        raise ValueError(f"Unknown horizon: {horizon}. Choose from {list(HORIZONS.keys())}")

    horizon_bars = HORIZONS[horizon]
    cal = load_calibration()
    mean_ic = cal[horizon]["mean_ic"]

    # Discover MBO files
    if not MBO_CACHE_DIR.exists():
        raise FileNotFoundError(
            f"MBO cache directory not found: {MBO_CACHE_DIR}\n"
            f"Set LVL3QUANT_PATH env var to the Lvl3Quant repo root."
        )

    all_files = sorted(MBO_CACHE_DIR.glob("*_mbo_features.npz"))
    if not all_files:
        raise FileNotFoundError(f"No MBO files found in {MBO_CACHE_DIR}")

    # Use the last n_train_days+1 files: last n_train_days for training,
    # the most recent file for prediction.
    use_files = all_files[-(n_train_days + 1):]
    if len(use_files) < 2:
        raise RuntimeError(
            f"Need at least 2 MBO files (got {len(use_files)}). "
            f"Reduce --train-days or wait for more data."
        )

    train_files = use_files[:-1]
    pred_file   = use_files[-1]

    log.info(
        f"horizon={horizon}  train_days={len(train_files)}  "
        f"pred_date={pred_file.stem.replace('_mbo_features','')}"
    )

    # Load training data
    day_data = []
    for i, fpath in enumerate(train_files):
        date_str = fpath.stem.replace("_mbo_features", "")
        try:
            X, y = prepare_day_data(fpath, horizon_bars)
            if len(y) >= 20:
                day_data.append({"date": date_str, "X": X, "y": y})
        except Exception as e:
            log.warning(f"  Skipping {date_str}: {e}")

    if len(day_data) < MIN_TRAIN_DAYS:
        raise RuntimeError(
            f"Only {len(day_data)} valid training days (need {MIN_TRAIN_DAYS}). "
            f"Increase --train-days."
        )

    if verbose:
        log.info(f"  Training on {len(day_data)} days")

    X_train = np.vstack([d["X"] for d in day_data])
    y_train = np.concatenate([d["y"] for d in day_data])

    # Cap training size to avoid memory issues (same as intraday_vol_prediction.py)
    if len(X_train) > 200_000:
        step = len(X_train) // 200_000
        X_train = X_train[::step]
        y_train = y_train[::step]

    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=N_BOOST_ROUNDS)

    del X_train, y_train, dtrain, day_data
    gc.collect()

    # Load prediction features: last hour of the most recent day
    # (= the most recently observed intraday features)
    pred_features, pred_mid = load_day(pred_file)

    # Take the last SAMPLE_INTERVAL * 360 bars = last 1 hour (3600s / 10s per sample)
    # or whatever is available. Average the predictions for stability.
    n_samples = min(360, len(pred_features) // SAMPLE_INTERVAL)
    if n_samples < 1:
        n_samples = 1
    start_idx = max(0, len(pred_features) - n_samples * SAMPLE_INTERVAL)
    indices = np.arange(start_idx, len(pred_features), SAMPLE_INTERVAL)
    X_pred = pred_features[indices].astype(np.float32)
    np.nan_to_num(X_pred, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Also compute trailing rvol from the prediction day for the naive baseline
    trailing_rvol_pct = compute_trailing_rvol(pred_mid, horizon_bars)

    preds = model.predict(X_pred)

    # Use the median prediction (more robust than mean against outlier bars)
    raw_pred_pct = float(np.median(preds))
    # Clamp: 3% to 100% annualized
    raw_pred_pct = max(3.0, min(100.0, raw_pred_pct))

    # Convert to z-score vs long-run baseline
    z_score = (raw_pred_pct - RVOL_MEAN_PCT) / RVOL_STD_PCT if RVOL_STD_PCT > 0 else 0.0

    # Confidence: based on IC and number of predictions
    confidence = float(min(1.0, mean_ic * (1.0 + 0.1 * min(len(day_data), 30) / 30)))

    result = {
        "timestamp":           datetime.now().isoformat(),
        "horizon":             horizon,
        "annualized_vol":      float(raw_pred_pct / 100.0),
        "raw_prediction_pct":  float(raw_pred_pct),
        "z_score":             float(z_score),
        "trailing_rvol_pct":   float(trailing_rvol_pct),
        "model_ic":            float(mean_ic),
        "n_train_days":        len(day_data),
        "confidence":          float(confidence),
        "pred_date":           pred_file.stem.replace("_mbo_features", ""),
        "n_pred_samples":      int(len(X_pred)),
        "pred_pct_p25":        float(np.percentile(preds, 25)),
        "pred_pct_p75":        float(np.percentile(preds, 75)),
    }

    del pred_features, pred_mid, X_pred, model
    gc.collect()

    return result


# ---------------------------------------------------------------------------
# Model artifact persistence
# ---------------------------------------------------------------------------

def _train_and_predict_with_model(
    horizon: str = "30min",
    n_train_days: int = 30,
):
    """Like train_and_predict() but also returns the trained lgb.Booster object.

    Returns (result_dict, lgb.Booster). The caller is responsible for freeing.
    This exists separately from train_and_predict() so the normal path can
    stay memory-lean (deletes the model before returning).
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise RuntimeError("LightGBM is not installed. Run: pip install lightgbm")

    if horizon not in HORIZONS:
        raise ValueError(f"Unknown horizon: {horizon}")

    horizon_bars = HORIZONS[horizon]
    cal = load_calibration()
    mean_ic = cal[horizon]["mean_ic"]

    if not MBO_CACHE_DIR.exists():
        raise FileNotFoundError(f"MBO cache directory not found: {MBO_CACHE_DIR}")

    all_files = sorted(MBO_CACHE_DIR.glob("*_mbo_features.npz"))
    if not all_files:
        raise FileNotFoundError(f"No MBO files found in {MBO_CACHE_DIR}")

    use_files = all_files[-(n_train_days + 1):]
    if len(use_files) < 2:
        raise RuntimeError(f"Need at least 2 MBO files (got {len(use_files)})")

    train_files = use_files[:-1]
    pred_file   = use_files[-1]

    day_data = []
    for fpath in train_files:
        date_str = fpath.stem.replace("_mbo_features", "")
        try:
            X, y = prepare_day_data(fpath, horizon_bars)
            if len(y) >= 20:
                day_data.append({"date": date_str, "X": X, "y": y})
        except Exception as e:
            log.warning(f"  Skipping {date_str}: {e}")

    if len(day_data) < MIN_TRAIN_DAYS:
        raise RuntimeError(f"Only {len(day_data)} valid training days (need {MIN_TRAIN_DAYS})")

    X_train = np.vstack([d["X"] for d in day_data])
    y_train = np.concatenate([d["y"] for d in day_data])

    if len(X_train) > 200_000:
        step = len(X_train) // 200_000
        X_train = X_train[::step]
        y_train = y_train[::step]

    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=N_BOOST_ROUNDS)

    # Prediction
    pred_features, pred_mid = load_day(pred_file)
    n_samples = min(360, len(pred_features) // SAMPLE_INTERVAL)
    if n_samples < 1:
        n_samples = 1
    start_idx = max(0, len(pred_features) - n_samples * SAMPLE_INTERVAL)
    indices = np.arange(start_idx, len(pred_features), SAMPLE_INTERVAL)
    X_pred = pred_features[indices].astype(np.float32)
    np.nan_to_num(X_pred, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    trailing_rvol_pct = compute_trailing_rvol(pred_mid, horizon_bars)

    preds = model.predict(X_pred)
    raw_pred_pct = float(np.median(preds))
    raw_pred_pct = max(3.0, min(100.0, raw_pred_pct))
    z_score = (raw_pred_pct - RVOL_MEAN_PCT) / RVOL_STD_PCT if RVOL_STD_PCT > 0 else 0.0
    confidence = float(min(1.0, mean_ic * (1.0 + 0.1 * min(len(day_data), 30) / 30)))

    result = {
        "timestamp":           datetime.now().isoformat(),
        "horizon":             horizon,
        "annualized_vol":      float(raw_pred_pct / 100.0),
        "raw_prediction_pct":  float(raw_pred_pct),
        "z_score":             float(z_score),
        "trailing_rvol_pct":   float(trailing_rvol_pct),
        "model_ic":            float(mean_ic),
        "n_train_days":        len(day_data),
        "confidence":          float(confidence),
        "pred_date":           pred_file.stem.replace("_mbo_features", ""),
        "n_pred_samples":      int(len(X_pred)),
        "pred_pct_p25":        float(np.percentile(preds, 25)),
        "pred_pct_p75":        float(np.percentile(preds, 75)),
    }

    del X_train, y_train, dtrain, day_data, pred_features, pred_mid, X_pred
    gc.collect()

    return result, model


def _save_model_artifact(model, horizon: str, models_dir: Path = None) -> Path:
    """Save a trained LightGBM model to disk for later reloading."""
    if models_dir is None:
        models_dir = THIS_DIR / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"vol_model_{horizon}_{ts}.txt"
    model.save_model(str(model_path))
    log.info(f"Model artifact saved → {model_path}")

    # Prune old artifacts: keep only the 3 most recent per horizon
    existing = sorted(models_dir.glob(f"vol_model_{horizon}_*.txt"))
    if len(existing) > 3:
        for old in existing[:-3]:
            try:
                old.unlink()
                log.debug(f"Pruned old model: {old.name}")
            except Exception:
                pass

    return model_path


# ---------------------------------------------------------------------------
# Signal file writer
# ---------------------------------------------------------------------------

def write_signal(result: dict, signal_file: Path = None):
    """Write the prediction dict to the signal JSON file."""
    path = signal_file or SIGNAL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(
        f"Signal written → {path}  "
        f"vol={result['raw_prediction_pct']:.1f}%  "
        f"z={result['z_score']:+.2f}  "
        f"horizon={result['horizon']}  "
        f"confidence={result['confidence']:.2f}"
    )


def is_market_hours() -> bool:
    """Return True if it's currently a US equities market day/time (ET, rough check)."""
    # Rough check: Mon-Fri, 09:30-16:00 ET.
    # In production, use pytz or zoneinfo for exact ET timezone.
    now = datetime.now()
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    # Approximate market hours in local time (not accounting for DST precisely)
    # Just allow 09:00-16:30 range to give margin
    hour = now.hour
    return 9 <= hour <= 16


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_once(
    horizon: str = "30min",
    n_train_days: int = 30,
    signal_file: Path = None,
    save_model: bool = False,
    models_dir: Path = None,
) -> dict | None:
    """Run one prediction cycle and write the signal file.

    Args:
        horizon:      Prediction horizon.
        n_train_days: Training window size.
        signal_file:  Override output path for the signal JSON.
        save_model:   If True, also serialize the trained LightGBM model
                      to models_dir for fast reloading by VolModel.
        models_dir:   Where to save the model artifact. Defaults to
                      data/models/ next to this file.
    """
    log.info(f"Running vol prediction: horizon={horizon}, n_train_days={n_train_days}")
    try:
        if save_model:
            result, model = _train_and_predict_with_model(
                horizon=horizon, n_train_days=n_train_days
            )
        else:
            result = train_and_predict(horizon=horizon, n_train_days=n_train_days)
            model = None

        write_signal(result, signal_file=signal_file)

        if save_model and model is not None:
            _save_model_artifact(model, horizon, models_dir)

        return result
    except FileNotFoundError as e:
        log.error(f"MBO data not found: {e}")
        _write_fallback_signal(signal_file)
        return None
    except RuntimeError as e:
        log.error(f"Training failed: {e}")
        _write_fallback_signal(signal_file)
        return None
    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        _write_fallback_signal(signal_file)
        return None


def _write_fallback_signal(signal_file: Path = None):
    """Write a fallback signal indicating the model could not run."""
    path = signal_file or SIGNAL_FILE
    # Don't overwrite a recent valid signal with a fallback
    if path.exists():
        try:
            with open(path) as f:
                existing = json.load(f)
            ts = existing.get("timestamp")
            if ts:
                age = (datetime.now() - datetime.fromisoformat(ts)).total_seconds()
                if age < 3600:  # keep valid signal up to 1 hour old
                    log.info(f"Keeping existing signal (age={age:.0f}s)")
                    return
        except Exception:
            pass

    fallback = {
        "timestamp":          datetime.now().isoformat(),
        "horizon":            "30min",
        "annualized_vol":     0.18,
        "raw_prediction_pct": 18.0,
        "z_score":            0.0,
        "trailing_rvol_pct":  18.0,
        "model_ic":           0.0,
        "n_train_days":       0,
        "confidence":         0.1,
        "source":             "FALLBACK_NO_DATA",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(fallback, f, indent=2)
    log.warning(f"Fallback signal written to {path}")


def run_daemon(
    horizon: str = "30min",
    n_train_days: int = 30,
    refresh_interval: int = 300,  # 5 minutes default
    signal_file: Path = None,
    save_model: bool = False,
    models_dir: Path = None,
):
    """Run as a daemon, refreshing the signal every refresh_interval seconds.

    Only trains during market hours. On the first invocation it trains
    immediately regardless of market hours (useful for testing).
    """
    log.info(
        f"Starting vol signal daemon: horizon={horizon}, "
        f"refresh={refresh_interval}s, train_days={n_train_days}, "
        f"save_model={save_model}"
    )

    first_run = True
    run_count = 0
    while True:
        if first_run or is_market_hours():
            run_once(
                horizon=horizon,
                n_train_days=n_train_days,
                signal_file=signal_file,
                # Only save model artifact on first run and every 6th run (30min cycle)
                save_model=save_model and (run_count % 6 == 0),
                models_dir=models_dir,
            )
            first_run = False
            run_count += 1
        else:
            log.debug("Outside market hours, skipping prediction")

        log.info(f"Sleeping {refresh_interval}s until next refresh...")
        time.sleep(refresh_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a vol LightGBM model and write a prediction to vol_signal.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default="30min",
        choices=list(HORIZONS.keys()),
        help="Prediction horizon for realized vol target",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=30,
        help="Number of recent trading days to train on",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as a daemon, refreshing every --interval seconds",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Refresh interval in seconds (daemon mode only)",
    )
    parser.add_argument(
        "--signal-file",
        type=str,
        default=None,
        help="Override signal file path",
    )
    parser.add_argument(
        "--mbo-dir",
        type=str,
        default=None,
        help="Override MBO cache directory path",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Also serialize the trained LightGBM model to data/models/ for reloading by VolModel",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Override directory for saved model artifacts (default: data/models/ next to this file)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log verbosity",
    )
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    signal_file = Path(args.signal_file) if args.signal_file else None
    models_dir  = Path(args.models_dir) if args.models_dir else None

    if args.mbo_dir:
        global MBO_CACHE_DIR
        MBO_CACHE_DIR = Path(args.mbo_dir)

    if args.daemon:
        run_daemon(
            horizon=args.horizon,
            n_train_days=args.train_days,
            refresh_interval=args.interval,
            signal_file=signal_file,
            save_model=args.save_model,
            models_dir=models_dir,
        )
    else:
        result = run_once(
            horizon=args.horizon,
            n_train_days=args.train_days,
            signal_file=signal_file,
            save_model=args.save_model,
            models_dir=models_dir,
        )
        if result:
            print(json.dumps(result, indent=2))
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
