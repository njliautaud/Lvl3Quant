"""
Kalshi API Client for Prediction Markets Trading

Wraps the kalshi-python SDK with:
- Demo/live mode switching
- SPX bracket contract discovery and pricing
- Vol model integration for fair value calculation
- Order management with risk limits
- Position tracking and P&L

Uses demo API by default. Set KALSHI_MODE=live for real trading.

PRICING FORMAT (post March 5, 2026):
  All prices are in dollars (0.0000 - 1.0000) with subpenny precision.
  The Kalshi API returns `_dollars` fields as FixedPointDollars strings (e.g., "0.5600").
  Legacy integer-cent fields (yes_bid, yes_ask, no_bid, no_ask) have been removed.
  Internal dict keys remain `yes_bid`, `yes_ask`, etc. but values are always floats in [0, 1].
"""

import math
import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

try:
    from kalshi_python import KalshiClient, Configuration
    from kalshi_python import EventsApi, MarketsApi, PortfolioApi
    HAS_SDK = True
except ImportError:
    HAS_SDK = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

logger = logging.getLogger("kalshi_client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# Paths to vol model outputs (Lvl3Quant project)
LVL3_ROOT = Path(os.environ.get(
    "LVL3QUANT_PATH",
    str(Path(__file__).resolve().parents[2] / ".." / "Lvl3Quant")
)).resolve()
VOL_RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
MBO_CACHE_DIR = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"

# Directory where vol_signal_writer.py saves serialized LightGBM model artifacts
# (written by VolModel.save_model() or vol_signal_writer's --save-model flag)
VOL_MODELS_DIR = Path(__file__).parent / "data" / "models"

# Config
DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"
LIVE_BASE = "https://trading-api.kalshi.com/trade-api/v2"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Fee constants (SPX/NDX have halved fees)
FEE_COEFFICIENT_STANDARD = 0.07
FEE_COEFFICIENT_SPX = 0.035
MAKER_FEE_COEFFICIENT = 0.0175


def calculate_taker_fee(contracts: int, price: float, is_spx: bool = True) -> float:
    """Calculate taker fee in dollars. Price is 0-1."""
    coeff = FEE_COEFFICIENT_SPX if is_spx else FEE_COEFFICIENT_STANDARD
    fee_cents = np.ceil(coeff * contracts * price * (1 - price) * 100) / 100
    return fee_cents


def calculate_maker_fee(contracts: int, price: float) -> float:
    """Calculate maker fee in dollars. Price is 0-1."""
    fee_cents = np.ceil(MAKER_FEE_COEFFICIENT * contracts * price * (1 - price) * 100) / 100
    return fee_cents


class VolModel:
    """Interface to our volatility prediction model.

    Walk-forward validated performance:
      - 1s (10 MBO bars):   IC=0.674, t=52.8, 100% positive folds
      - 30min (18K bars):   IC=0.644, t=23.2, 97.6% positive folds
      - 1hr (36K bars):     IC=0.568, t=17.8, 95.3% positive folds
      - 2hr (72K bars):     IC=0.440, t=9.6,  81.2% positive folds
      - 4hr (144K bars):    IC=0.406, t=7.8,  83.3% positive folds

    The model predicts realized vol RANK (Spearman IC), not absolute levels.
    To get a usable vol forecast we:
      1. Load walk-forward calibration stats (IC, std per horizon)
      2. Accept either:
         a) A direct z-score from the LightGBM model (if running live)
         b) A VIX-based baseline adjusted by model confidence
      3. Convert to annualized vol for bracket pricing

    Integration modes:
      - LIVE:     LightGBM model loaded, receives MBO features, outputs predictions
      - SIGNAL:   External process writes predictions to a JSON file, we read it
      - FALLBACK: No model available, uses VIX-implied vol (degraded accuracy)
    """

    # Walk-forward calibration constants from validated results
    # Source: Lvl3Quant/alpha_discovery/results/intraday_vol_pred_20260228_003217.json
    # and vol_pred_rvol_1s_20260227_200923.json
    CALIBRATION = {
        "1s":    {"mean_ic": 0.674, "std_ic": 0.118, "t_stat": 52.8, "pct_positive": 100.0, "n_folds": 85},
        "30min": {"mean_ic": 0.644, "std_ic": 0.256, "t_stat": 23.2, "pct_positive": 97.6, "n_folds": 85},
        "1hr":   {"mean_ic": 0.568, "std_ic": 0.294, "t_stat": 17.8, "pct_positive": 95.3, "n_folds": 85},
        "2hr":   {"mean_ic": 0.440, "std_ic": 0.421, "t_stat": 9.6,  "pct_positive": 81.2, "n_folds": 85},
        "4hr":   {"mean_ic": 0.406, "std_ic": 0.474, "t_stat": 7.8,  "pct_positive": 83.3, "n_folds": 84},
    }

    # Typical ES/SPX realized vol statistics (annualized %)
    # Used to convert z-scores to absolute vol levels
    RVOL_BASELINE = {
        "mean_annual_pct": 18.0,   # Long-run SPX annualized vol
        "std_annual_pct": 8.0,     # Vol-of-vol (how much annualized vol varies)
        "30min_mean_annual_pct": 20.0,  # Intraday tends slightly higher
        "30min_std_annual_pct": 10.0,
    }

    # Signal file path (written by external model process)
    SIGNAL_FILE = Path(__file__).parent / "data" / "vol_signal.json"

    def __init__(self, model_path: str = None, signal_file: str = None,
                 auto_load: bool = True):
        """Initialize VolModel.

        Args:
            model_path: Path to a saved LightGBM model file (.txt or .bin).
                        If None and auto_load=True, scans VOL_MODELS_DIR for the
                        most recent saved model artifact.
            signal_file: Path to JSON file with live predictions from external process.
                         If None, uses default SIGNAL_FILE path.
            auto_load:  If True (default), automatically load the most recently saved
                        LightGBM model artifact from VOL_MODELS_DIR when no explicit
                        model_path is given.
        """
        self.last_prediction = None
        self.last_update = None
        self._override_vol = None
        self._lgb_model = None
        self._lgb_horizon = None   # which horizon the loaded model targets
        self._mode = "FALLBACK"
        self._calibration_loaded = False
        self._fold_ics = {}  # Full fold IC arrays per horizon

        if signal_file:
            self.SIGNAL_FILE = Path(signal_file)

        # Try to load calibration data from results files
        self._load_calibration()

        # Load LightGBM model: explicit path takes priority, then auto-discovery
        if model_path and HAS_LGB:
            self._load_lgb_model(model_path)
        elif auto_load and HAS_LGB:
            self._auto_load_model()

    def _load_calibration(self):
        """Load full walk-forward calibration from Lvl3Quant results files."""
        # Try loading the multi-horizon results
        intraday_results = sorted(VOL_RESULTS_DIR.glob("intraday_vol_pred_*.json")) if VOL_RESULTS_DIR.exists() else []
        if intraday_results:
            try:
                with open(intraday_results[-1]) as f:
                    data = json.load(f)
                for hz_name, hz_data in data.items():
                    if hz_name in self.CALIBRATION:
                        self.CALIBRATION[hz_name] = {
                            "mean_ic": hz_data["mean_ic"],
                            "std_ic": hz_data["std_ic"],
                            "t_stat": hz_data["t_stat"],
                            "pct_positive": hz_data["pct_positive"],
                            "n_folds": hz_data["n_folds"],
                        }
                        self._fold_ics[hz_name] = hz_data.get("fold_ics", [])
                self._calibration_loaded = True
                logger.info(f"Vol calibration loaded from {intraday_results[-1].name}: "
                           f"{list(data.keys())}")
            except Exception as e:
                logger.warning(f"Could not load intraday vol calibration: {e}")

        # Also try loading 1s rvol results
        rvol_results = sorted(VOL_RESULTS_DIR.glob("vol_pred_rvol_1s_*.json")) if VOL_RESULTS_DIR.exists() else []
        if rvol_results:
            try:
                with open(rvol_results[-1]) as f:
                    data = json.load(f)
                self.CALIBRATION["1s"] = {
                    "mean_ic": data["mean_ic"],
                    "std_ic": data["std_ic"],
                    "t_stat": data["t_stat"],
                    "pct_positive": data["pct_positive"],
                    "n_folds": data["n_folds"],
                }
                self._fold_ics["1s"] = data.get("fold_ics", [])
                self._calibration_loaded = True
                logger.info(f"1s rvol calibration loaded: IC={data['mean_ic']:.3f}")
            except Exception as e:
                logger.warning(f"Could not load 1s rvol calibration: {e}")

    def _load_lgb_model(self, model_path: str):
        """Load a saved LightGBM model for live predictions."""
        if not HAS_LGB:
            logger.warning("LightGBM not installed, cannot load model")
            return
        try:
            self._lgb_model = lgb.Booster(model_file=model_path)
            self._mode = "LIVE"
            # Try to infer the horizon from the filename, e.g. vol_model_30min_*.txt
            stem = Path(model_path).stem
            for hz in ["1s", "30min", "1hr", "2hr", "4hr"]:
                if hz in stem:
                    self._lgb_horizon = hz
                    break
            logger.info(f"LightGBM vol model loaded from {model_path} (horizon={self._lgb_horizon})")
        except Exception as e:
            logger.warning(f"Could not load LightGBM model from {model_path}: {e}")

    def _auto_load_model(self):
        """Scan VOL_MODELS_DIR for the most recently saved LightGBM model artifact.

        Looks for files matching vol_model_*.txt (written by save_model()).
        Selects the most recently modified file. This is a best-effort load;
        if no files exist it falls back to SIGNAL or VIX modes silently.
        """
        if not HAS_LGB:
            return
        if not VOL_MODELS_DIR.exists():
            return
        candidates = sorted(VOL_MODELS_DIR.glob("vol_model_*.txt"))
        if not candidates:
            logger.debug(f"No saved vol model artifacts in {VOL_MODELS_DIR}")
            return
        # Most recently modified
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        age_hours = (time.time() - latest.stat().st_mtime) / 3600
        if age_hours > 24:
            logger.info(
                f"Most recent vol model artifact is {age_hours:.1f}h old "
                f"({latest.name}) — consider retraining with vol_signal_writer.py"
            )
        self._load_lgb_model(str(latest))

    def save_model(self, horizon: str = "30min", n_train_days: int = 30) -> str | None:
        """Train a fresh LightGBM model on recent MBO data and save it to VOL_MODELS_DIR.

        Trains on the last n_train_days of MBO data and serializes the model so
        it can be reloaded at startup via _auto_load_model(). Also writes the
        signal file and sets this instance to LIVE mode.

        Args:
            horizon:      Which horizon to train for (default "30min").
            n_train_days: Number of recent trading days to train on.

        Returns:
            Path to the saved model file, or None on failure.
        """
        if not HAS_LGB:
            logger.error("LightGBM not installed, cannot save model")
            return None
        try:
            import importlib.util
            _writer_path = Path(__file__).parent / "vol_signal_writer.py"
            spec = importlib.util.spec_from_file_location("vol_signal_writer", _writer_path)
            vsw = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vsw)

            # Train once and retain the model object (single training pass)
            result, model = vsw._train_and_predict_with_model(
                horizon=horizon, n_train_days=n_train_days
            )

            # Save model artifact
            model_path = vsw._save_model_artifact(model, horizon, VOL_MODELS_DIR)

            # Update this instance
            self._lgb_model = model
            self._lgb_horizon = horizon
            self._mode = "LIVE"
            logger.info(f"Vol model trained and saved to {model_path}")

            # Also write the signal file so the prediction is immediately available
            vsw.write_signal(result, signal_file=self.SIGNAL_FILE)

            return str(model_path)

        except Exception as e:
            logger.exception(f"save_model() failed: {e}")
            return None

    def train_and_write_signal(self, horizon: str = "30min", n_train_days: int = 30) -> dict | None:
        """Convenience: train a fresh vol model and write vol_signal.json.

        Calls vol_signal_writer.run_once() directly. Returns the prediction
        dict, or None on failure. This is equivalent to running:
            python vol_signal_writer.py --horizon 30min --train-days 30

        Use this to manually trigger a signal refresh from Python code.
        """
        try:
            import importlib.util
            _writer_path = Path(__file__).parent / "vol_signal_writer.py"
            spec = importlib.util.spec_from_file_location("vol_signal_writer", _writer_path)
            vsw = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vsw)
            return vsw.run_once(horizon=horizon, n_train_days=n_train_days,
                                signal_file=self.SIGNAL_FILE)
        except Exception as e:
            logger.exception(f"train_and_write_signal() failed: {e}")
            return None

    # Maximum age (seconds) before a signal file is considered stale.
    # At 30min intraday vol prediction horizon, vol changes slowly enough that
    # a signal up to 30 minutes old is still useful. Beyond 2 hours, reject.
    SIGNAL_MAX_AGE_SECONDS = 1800   # 30 minutes: use signal at degraded confidence
    SIGNAL_HARD_REJECT_SECONDS = 7200  # 2 hours: too stale, treat as no signal

    def _read_signal_file(self) -> dict | None:
        """Read predictions from external signal file.

        Staleness handling:
          - age < 30 min:  use at full confidence
          - age 30-120 min: use but lower confidence (stale_factor applied)
          - age > 120 min:  reject (return None, fall through to VIX/fallback)
        """
        if not self.SIGNAL_FILE.exists():
            return None
        try:
            with open(self.SIGNAL_FILE) as f:
                signal = json.load(f)

            ts = signal.get("timestamp")
            if ts:
                age = (datetime.now() - datetime.fromisoformat(ts)).total_seconds()
                if age > self.SIGNAL_HARD_REJECT_SECONDS:
                    logger.warning(
                        f"Vol signal is {age/3600:.1f}h old (hard reject). "
                        f"Run vol_signal_writer.py to refresh."
                    )
                    return None
                elif age > self.SIGNAL_MAX_AGE_SECONDS:
                    # Degrade confidence proportionally from 1.0 at 30min to 0.2 at 2hr
                    stale_factor = max(
                        0.2,
                        1.0 - 0.8 * (age - self.SIGNAL_MAX_AGE_SECONDS)
                        / (self.SIGNAL_HARD_REJECT_SECONDS - self.SIGNAL_MAX_AGE_SECONDS),
                    )
                    signal["_stale_factor"] = stale_factor
                    signal["_age_seconds"] = age
                    logger.info(
                        f"Vol signal is {age/60:.0f}min old (stale factor={stale_factor:.2f})"
                    )
                else:
                    signal["_stale_factor"] = 1.0
                    signal["_age_seconds"] = age

            return signal
        except Exception as e:
            logger.warning(f"Could not read vol signal file: {e}")
            return None

    def _select_horizon(self, hours_to_close: float) -> str:
        """Select the best calibration horizon for the given time to close."""
        if hours_to_close <= 0.08:     # < 5 min
            return "1s"
        elif hours_to_close <= 0.75:   # < 45 min
            return "30min"
        elif hours_to_close <= 1.5:    # < 90 min
            return "1hr"
        elif hours_to_close <= 3.0:    # < 3 hr
            return "2hr"
        else:
            return "4hr"

    def _z_to_annualized_vol(self, z_score: float, horizon: str = "30min") -> float:
        """Convert a model z-score prediction to annualized vol (decimal).

        The model predicts realized vol rank (Spearman). A z-score of +1
        means the model predicts vol 1 std above the mean for this horizon.

        We convert using the IC-weighted mapping:
            predicted_vol = mean_vol + IC * z_score * vol_of_vol

        The IC dampens the z-score because rank correlation < 1 means
        our predictions are noisy.
        """
        cal = self.CALIBRATION.get(horizon, self.CALIBRATION["30min"])
        ic = cal["mean_ic"]

        mean_vol = self.RVOL_BASELINE["30min_mean_annual_pct"] / 100.0
        vol_of_vol = self.RVOL_BASELINE["30min_std_annual_pct"] / 100.0

        predicted_vol = mean_vol + ic * z_score * vol_of_vol
        # Clamp to reasonable range (5% to 80% annualized)
        predicted_vol = max(0.05, min(0.80, predicted_vol))
        return predicted_vol

    def predict_vol_from_features(self, features: np.ndarray) -> dict | None:
        """Run the LightGBM model on raw MBO features.

        Args:
            features: 1D array of MBO feature values (336 features after
                      excluding price levels). Same feature set as the
                      walk-forward training in intraday_vol_prediction.py.

        Returns:
            dict with raw prediction and z-score, or None if model not loaded.
        """
        if self._lgb_model is None:
            return None

        try:
            raw_pred = self._lgb_model.predict(features.reshape(1, -1))[0]
            # The model predicts absolute rvol in annualized % (matching training target).
            # Convert to z-score relative to baseline.
            mean_vol_pct = self.RVOL_BASELINE["30min_mean_annual_pct"]
            std_vol_pct = self.RVOL_BASELINE["30min_std_annual_pct"]
            z_score = (raw_pred - mean_vol_pct) / std_vol_pct if std_vol_pct > 0 else 0.0

            return {
                "raw_prediction_pct": float(raw_pred),
                "z_score": float(z_score),
                "annualized_vol": float(raw_pred / 100.0),
            }
        except Exception as e:
            logger.error(f"LightGBM prediction failed: {e}")
            return None

    def get_current_vol_estimate(self, current_spx: float = None,
                                  hours_to_close: float = None,
                                  vix_level: float = None,
                                  mbo_features: np.ndarray = None) -> dict:
        """Get the best available vol estimate right now.

        Tries sources in priority order:
          1. Live LightGBM model (if loaded and features provided)
          2. External signal file (if fresh)
          3. VIX-based estimate with model IC confidence
          4. Hardcoded fallback (18% annualized)

        Args:
            current_spx: Current SPX/ES price (for move calculations)
            hours_to_close: Hours until market close (for horizon selection)
            vix_level: Current VIX level (if available, improves fallback)
            mbo_features: Raw MBO features for live model prediction

        Returns:
            dict with:
              - annualized_vol: float (decimal, e.g., 0.18 for 18%)
              - vol_30min_forecast_pct: float (annualized % for 30min horizon)
              - confidence: float (0-1, based on model IC)
              - source: str ("LIVE", "SIGNAL", "VIX_ADJUSTED", "FALLBACK")
              - horizon: str (which calibration horizon was used)
              - model_ic: float (walk-forward IC for this horizon)
              - timestamp: str (ISO format)
        """
        if hours_to_close is None:
            now = datetime.now()
            close_time = now.replace(hour=16, minute=0, second=0)
            hours_to_close = max(0.01, (close_time - now).total_seconds() / 3600)

        horizon = self._select_horizon(hours_to_close)
        cal = self.CALIBRATION.get(horizon, self.CALIBRATION["30min"])

        # Override takes priority
        if self._override_vol is not None:
            result = self._build_result(
                annualized_vol=self._override_vol,
                source="OVERRIDE",
                horizon=horizon,
                cal=cal,
                current_spx=current_spx,
                hours_to_close=hours_to_close,
            )
            self.last_prediction = result
            self.last_update = datetime.now()
            return result

        # 1. Try live model
        if self._lgb_model is not None and mbo_features is not None:
            pred = self.predict_vol_from_features(mbo_features)
            if pred is not None:
                result = self._build_result(
                    annualized_vol=pred["annualized_vol"],
                    source="LIVE",
                    horizon=horizon,
                    cal=cal,
                    current_spx=current_spx,
                    hours_to_close=hours_to_close,
                    z_score=pred["z_score"],
                )
                self.last_prediction = result
                self.last_update = datetime.now()
                self._mode = "LIVE"
                return result

        # 2. Try signal file (written by vol_signal_writer.py)
        signal = self._read_signal_file()
        if signal is not None:
            z_score = signal.get("z_score", 0.0)
            annualized_vol = signal.get("annualized_vol")
            if annualized_vol is None:
                annualized_vol = self._z_to_annualized_vol(z_score, horizon)

            # Use signal's horizon calibration if it matches what we need
            signal_horizon = signal.get("horizon", horizon)
            if signal_horizon in self.CALIBRATION:
                signal_cal = self.CALIBRATION[signal_horizon]
            else:
                signal_cal = cal

            result = self._build_result(
                annualized_vol=annualized_vol,
                source="SIGNAL",
                horizon=signal_horizon,
                cal=signal_cal,
                current_spx=current_spx,
                hours_to_close=hours_to_close,
                z_score=z_score,
            )

            # Apply staleness penalty to confidence
            stale_factor = signal.get("_stale_factor", 1.0)
            result["confidence"] = float(result["confidence"] * stale_factor)
            if stale_factor < 1.0:
                result["signal_stale_factor"] = stale_factor
                result["signal_age_seconds"] = signal.get("_age_seconds")

            # Attach extra signal metadata when available
            for key in ("trailing_rvol_pct", "n_train_days", "pred_date",
                        "pred_pct_p25", "pred_pct_p75"):
                if key in signal:
                    result[key] = signal[key]

            self.last_prediction = result
            self.last_update = datetime.now()
            self._mode = "SIGNAL"
            return result

        # 3. VIX-based estimate
        if vix_level is not None:
            # VIX is annualized implied vol in %. Convert to decimal.
            # VIX typically overstates realized vol (VRP ~ 2-5 pts).
            # Apply small VRP adjustment.
            vrp_adjustment = 0.02  # 2 percentage points
            base_vol = max(0.05, (vix_level / 100.0) - vrp_adjustment)

            result = self._build_result(
                annualized_vol=base_vol,
                source="VIX_ADJUSTED",
                horizon=horizon,
                cal=cal,
                current_spx=current_spx,
                hours_to_close=hours_to_close,
            )
            self.last_prediction = result
            self.last_update = datetime.now()
            self._mode = "FALLBACK"
            return result

        # 4. Pure fallback
        fallback_vol = 0.18
        result = self._build_result(
            annualized_vol=fallback_vol,
            source="FALLBACK",
            horizon=horizon,
            cal=cal,
            current_spx=current_spx,
            hours_to_close=hours_to_close,
        )
        self.last_prediction = result
        self.last_update = datetime.now()
        self._mode = "FALLBACK"
        return result

    def _build_result(self, annualized_vol: float, source: str, horizon: str,
                      cal: dict, current_spx: float = None,
                      hours_to_close: float = None, z_score: float = None) -> dict:
        """Build a standardized vol prediction result dict."""
        if hours_to_close is None:
            hours_to_close = 1.0
        if hours_to_close <= 0:
            hours_to_close = 0.01

        time_fraction = hours_to_close / (252 * 6.5)
        expected_move_1sd = (current_spx or 5900) * annualized_vol * np.sqrt(time_fraction)
        spx = current_spx or 5900

        # Confidence is based on model IC and source quality
        if source == "LIVE":
            confidence = min(1.0, cal["mean_ic"] * 1.2)  # Slight boost for live
        elif source == "SIGNAL":
            confidence = cal["mean_ic"]
        elif source == "VIX_ADJUSTED":
            confidence = 0.5  # VIX is decent but not model-grade
        elif source == "OVERRIDE":
            confidence = 1.0
        else:
            confidence = 0.3  # Fallback is a guess

        result = {
            "annualized_vol": float(annualized_vol),
            "vol_30min_forecast_pct": float(annualized_vol * 100),
            "hours_to_close": hours_to_close,
            "expected_move_1sd": float(expected_move_1sd),
            "1sd_range": (float(spx - expected_move_1sd), float(spx + expected_move_1sd)),
            "2sd_range": (float(spx - 2 * expected_move_1sd), float(spx + 2 * expected_move_1sd)),
            "confidence": float(confidence),
            "source": source,
            "horizon": horizon,
            "model_ic": float(cal["mean_ic"]),
            "model_t_stat": float(cal["t_stat"]),
            "model_pct_positive_folds": float(cal["pct_positive"]),
            "timestamp": datetime.now().isoformat(),
        }
        if z_score is not None:
            result["z_score"] = float(z_score)
        return result

    def predict_vol(self, current_spx: float, hours_to_close: float,
                    vix_level: float = None) -> dict:
        """Get current vol prediction. Backward-compatible interface.

        Args:
            current_spx: current SPX price
            hours_to_close: hours until market close
            vix_level: optional current VIX level for better fallback

        Returns:
            dict with 'annualized_vol', 'expected_move_1sd', '1sd_range', '2sd_range',
            plus new fields: 'confidence', 'source', 'horizon', 'model_ic'
        """
        return self.get_current_vol_estimate(
            current_spx=current_spx,
            hours_to_close=hours_to_close,
            vix_level=vix_level,
        )

    def set_vol_override(self, annualized_vol: float):
        """Override with a specific vol value (for backtesting or manual input).

        Pass None to clear the override.
        """
        self._override_vol = annualized_vol
        self.last_prediction = None  # Force recalculation

    def refresh(self, vix_level: float = None, current_spx: float = None):
        """Refresh predictions by re-reading signal file and recalculating.

        Call this periodically (e.g., every 30 seconds) during live trading.
        """
        if current_spx is None:
            current_spx = 5900  # Placeholder
        self.get_current_vol_estimate(
            current_spx=current_spx,
            vix_level=vix_level,
        )

    def write_signal(self, annualized_vol: float = None, z_score: float = None,
                     raw_prediction_pct: float = None):
        """Write a vol prediction to the signal file (for external model processes).

        Call this from the model training/inference pipeline to provide
        live predictions to the trading system.

        Args:
            annualized_vol: Predicted annualized vol as decimal (e.g., 0.22)
            z_score: Model z-score (if annualized_vol not provided, converted via calibration)
            raw_prediction_pct: Raw model output in % (annualized rvol %)
        """
        signal = {"timestamp": datetime.now().isoformat()}
        if annualized_vol is not None:
            signal["annualized_vol"] = annualized_vol
        if z_score is not None:
            signal["z_score"] = z_score
        if raw_prediction_pct is not None:
            signal["raw_prediction_pct"] = raw_prediction_pct
            signal["annualized_vol"] = raw_prediction_pct / 100.0

        self.SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.SIGNAL_FILE, "w") as f:
            json.dump(signal, f, indent=2)
        logger.info(f"Vol signal written: {signal}")

    def get_calibration_summary(self) -> dict:
        """Return a summary of model calibration and connection status."""
        signal_age = None
        signal_vol = None
        if self.SIGNAL_FILE.exists():
            try:
                with open(self.SIGNAL_FILE) as f:
                    sig = json.load(f)
                ts = sig.get("timestamp")
                if ts:
                    signal_age = (datetime.now() - datetime.fromisoformat(ts)).total_seconds()
                signal_vol = sig.get("raw_prediction_pct")
            except Exception:
                pass

        # Check for saved model artifacts
        saved_models = []
        if VOL_MODELS_DIR.exists():
            saved_models = [p.name for p in sorted(VOL_MODELS_DIR.glob("vol_model_*.txt"))]

        return {
            "calibration": self.CALIBRATION,
            "calibration_loaded_from_file": self._calibration_loaded,
            "mode": self._mode,
            "lgb_model_loaded": self._lgb_model is not None,
            "lgb_model_horizon": self._lgb_horizon,
            "signal_file": str(self.SIGNAL_FILE),
            "signal_file_exists": self.SIGNAL_FILE.exists(),
            "signal_age_seconds": signal_age,
            "signal_vol_pct": signal_vol,
            "signal_stale": signal_age is not None and signal_age > self.SIGNAL_MAX_AGE_SECONDS,
            "saved_model_artifacts": saved_models,
            "vol_models_dir": str(VOL_MODELS_DIR),
            "mbo_cache_dir": str(MBO_CACHE_DIR),
            "mbo_cache_exists": MBO_CACHE_DIR.exists(),
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


class BracketPricer:
    """Prices SPX bracket contracts using vol predictions.

    Uses Student-t distribution (df=4) by default for fat tails.
    SPX returns are leptokurtic -- normal distribution underprices tail brackets by ~80%.
    df=4 is more conservative than df=5 and better matches empirical SPX
    intraday kurtosis (excess kurtosis ~5-8 for intraday).
    """

    def __init__(self, vol_model: VolModel = None, use_fat_tails: bool = True, df: float = 4.0):
        self.vol_model = vol_model or VolModel()
        self.use_fat_tails = use_fat_tails
        self.df = df  # Degrees of freedom for Student-t (lower = fatter tails)

    def bracket_fair_value(
        self, floor: float, cap: float, current_spx: float,
        predicted_vol: float, hours_to_close: float
    ) -> float:
        """
        Calculate fair value of a bracket contract [floor, cap].
        Returns probability SPX closes within the range (0-1).

        Uses Student-t distribution for fat tails when use_fat_tails=True.
        Student-t with df=5 matches empirical SPX return kurtosis (~4-6 excess).
        """
        if not HAS_SCIPY:
            return self._approx_bracket_value(floor, cap, current_spx, predicted_vol, hours_to_close)

        time_fraction = hours_to_close / (252 * 6.5)
        sigma = current_spx * predicted_vol * np.sqrt(time_fraction)

        if sigma <= 0:
            return 1.0 if floor <= current_spx <= cap else 0.0

        # Scale sigma for Student-t to match normal variance
        # Student-t with df degrees of freedom has variance = df/(df-2)
        # Scale down so total variance matches the vol prediction
        if self.use_fat_tails and self.df > 2:
            scale = sigma * np.sqrt((self.df - 2) / self.df)
            dist = scipy_stats.t(df=self.df, loc=current_spx, scale=scale)
        else:
            dist = scipy_stats.norm(loc=current_spx, scale=sigma)

        # Handle unbounded brackets
        if floor <= 0 or floor == float("-inf"):
            return float(dist.cdf(cap))
        if cap >= current_spx * 2 or cap == float("inf"):
            return float(1.0 - dist.cdf(floor))

        prob = dist.cdf(cap) - dist.cdf(floor)
        return max(0.0, min(1.0, float(prob)))

    def _approx_bracket_value(self, floor, cap, current_spx, predicted_vol, hours_to_close):
        """Approximation without scipy."""
        time_fraction = hours_to_close / (252 * 6.5)
        sigma = current_spx * predicted_vol * np.sqrt(time_fraction)
        if sigma <= 0:
            return 1.0 if floor <= current_spx <= cap else 0.0
        # Simple linear approximation in the middle
        width = cap - floor
        distance = abs(current_spx - (floor + cap) / 2)
        return max(0.0, min(1.0, (width / (2 * sigma)) * np.exp(-0.5 * (distance / sigma) ** 2)))

    def price_all_brackets(
        self, brackets: list, current_spx: float,
        predicted_vol: float, hours_to_close: float
    ) -> list:
        """
        Price all brackets in a chain.

        Args:
            brackets: list of dicts with 'ticker', 'floor', 'cap', 'yes_bid', 'yes_ask',
                       'no_bid', 'no_ask'. All prices are floats in dollars (0.0 - 1.0).
            current_spx: current SPX price
            predicted_vol: annualized vol prediction
            hours_to_close: hours until 4pm ET

        Returns:
            list of dicts with fair_value, edge, recommended action
        """
        results = []
        for b in brackets:
            fv = self.bracket_fair_value(
                b["floor"], b["cap"], current_spx, predicted_vol, hours_to_close
            )

            # Calculate edges
            yes_ask = b.get("yes_ask", 1.0)
            no_ask = b.get("no_ask", 1.0)
            yes_bid = b.get("yes_bid", 0.0)
            no_bid = b.get("no_bid", 0.0)

            buy_yes_edge = fv - yes_ask if yes_ask > 0 else 0
            buy_no_edge = (1 - fv) - no_ask if no_ask > 0 else 0
            sell_yes_edge = yes_bid - fv if yes_bid > 0 else 0
            sell_no_edge = no_bid - (1 - fv) if no_bid > 0 else 0

            best_edge = max(buy_yes_edge, buy_no_edge, sell_yes_edge, sell_no_edge)
            if best_edge == buy_yes_edge and best_edge > 0:
                action = "BUY_YES"
                entry_price = yes_ask
            elif best_edge == buy_no_edge and best_edge > 0:
                action = "BUY_NO"
                entry_price = no_ask
            elif best_edge == sell_yes_edge and best_edge > 0:
                action = "SELL_YES"
                entry_price = yes_bid
            elif best_edge == sell_no_edge and best_edge > 0:
                action = "SELL_NO"
                entry_price = no_bid
            else:
                action = "NO_EDGE"
                entry_price = 0

            # Account for fees
            fee = calculate_taker_fee(1, entry_price, is_spx=True)
            net_edge = best_edge - fee if best_edge > 0 else 0

            results.append({
                "ticker": b["ticker"],
                "floor": b["floor"],
                "cap": b["cap"],
                "fair_value": round(fv, 4),
                "yes_ask": yes_ask,
                "no_ask": no_ask,
                "yes_bid": yes_bid,
                "no_bid": no_bid,
                "edge": round(best_edge, 4),
                "net_edge_after_fees": round(net_edge, 4),
                "action": action,
                "entry_price": entry_price,
                "fee": round(fee, 4),
            })

        return sorted(results, key=lambda x: x["net_edge_after_fees"], reverse=True)

    def find_mispricings(
        self, brackets: list, current_spx: float,
        hours_to_close: float, vix_level: float = None,
    ) -> dict:
        """Compare model-priced vs market-priced brackets to find mispricings.

        Uses the enhanced VolModel to get a calibrated vol forecast, then
        prices all brackets and compares to market mid prices.

        Args:
            brackets: list of bracket dicts from Kalshi API
            current_spx: current SPX price
            hours_to_close: hours until market close
            vix_level: optional VIX level for fallback vol estimate

        Returns:
            dict with:
              - vol_estimate: the vol model's output
              - brackets: list of bracket analysis dicts
              - summary: overall mispricing statistics
              - top_opportunities: best 5 trades by net edge
        """
        # Get vol estimate from enhanced model
        vol_estimate = self.vol_model.get_current_vol_estimate(
            current_spx=current_spx,
            hours_to_close=hours_to_close,
            vix_level=vix_level,
        )
        predicted_vol = vol_estimate["annualized_vol"]

        # Price all brackets with our model
        priced = self.price_all_brackets(brackets, current_spx, predicted_vol, hours_to_close)

        # Compute market-implied vol for each bracket (reverse-engineer from market mid)
        analysis = []
        for p in priced:
            yes_mid = (p["yes_ask"] + p["yes_bid"]) / 2 if (p["yes_ask"] + p["yes_bid"]) > 0 else 0
            market_prob = yes_mid  # Market-implied probability

            model_prob = p["fair_value"]
            mispricing = model_prob - market_prob  # Positive = market underprices

            # Confidence-weighted edge: scale by model confidence
            confidence = vol_estimate["confidence"]
            adjusted_edge = p["net_edge_after_fees"] * confidence

            analysis.append({
                **p,
                "market_mid": round(yes_mid, 4),
                "market_prob": round(market_prob, 4),
                "model_prob": round(model_prob, 4),
                "mispricing": round(mispricing, 4),
                "confidence": round(confidence, 3),
                "adjusted_edge": round(adjusted_edge, 4),
                "vol_source": vol_estimate["source"],
            })

        # Summary stats
        mispricings = [a["mispricing"] for a in analysis if a["market_prob"] > 0]
        top_opps = sorted(analysis, key=lambda x: x["adjusted_edge"], reverse=True)[:5]

        summary = {
            "n_brackets": len(analysis),
            "predicted_vol": round(predicted_vol, 4),
            "vol_source": vol_estimate["source"],
            "vol_confidence": round(vol_estimate["confidence"], 3),
            "vol_horizon": vol_estimate["horizon"],
            "model_ic": vol_estimate["model_ic"],
            "mean_abs_mispricing": round(np.mean(np.abs(mispricings)), 4) if mispricings else 0,
            "max_mispricing": round(max(mispricings, key=abs), 4) if mispricings else 0,
            "n_positive_edge": sum(1 for a in analysis if a["adjusted_edge"] > 0),
            "total_edge_available": round(sum(a["adjusted_edge"] for a in analysis if a["adjusted_edge"] > 0), 4),
        }

        return {
            "vol_estimate": vol_estimate,
            "brackets": analysis,
            "summary": summary,
            "top_opportunities": top_opps,
        }


class KalshiTrader:
    """
    Main trading class for Kalshi prediction markets.

    Handles:
    - API connection (demo/live)
    - SPX bracket discovery
    - Fair value calculation via vol model
    - Order placement with risk limits
    - Position tracking
    """

    def __init__(self, mode: str = "demo", api_key: str = None, private_key_path: str = None,
                 vol_model_path: str = None, vol_signal_file: str = None):
        self.mode = mode
        self.base_url = DEMO_BASE if mode == "demo" else LIVE_BASE
        self.api_key = api_key or os.environ.get("KALSHI_API_KEY")
        self.private_key_path = private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH")

        self.vol_model = VolModel(model_path=vol_model_path, signal_file=vol_signal_file)
        self.pricer = BracketPricer(self.vol_model, df=4.0)

        # Risk limits
        self.max_position_per_bracket = 100  # contracts
        self.max_total_exposure = 5000  # dollars
        self.max_single_trade = 500  # dollars
        self.min_edge_threshold = 0.02  # 2 cents minimum edge
        self.min_net_edge = 0.005  # 0.5 cents after fees

        # State
        self.positions = {}
        self.orders = {}
        self.trade_log = []
        self.pnl_history = []

        # SDK client
        self.client = None
        if HAS_SDK and self.api_key:
            self._init_client()

        vol_info = self.vol_model.get_calibration_summary()
        logger.info(f"KalshiTrader initialized in {mode} mode. SDK: {HAS_SDK}, "
                     f"API key: {'set' if self.api_key else 'NOT SET'}, "
                     f"Vol model: {vol_info['mode']} (calibration loaded: {vol_info['calibration_loaded_from_file']})")

    def _init_client(self):
        """Initialize the kalshi-python SDK client."""
        try:
            config = Configuration()
            config.host = self.base_url
            self.client = KalshiClient(configuration=config)
            self.client.set_kalshi_auth(self.api_key, self.private_key_path)
            self.events_api = EventsApi(self.client)
            self.markets_api = MarketsApi(self.client)
            self.portfolio_api = PortfolioApi(self.client)
            logger.info("Kalshi SDK client initialized")
        except Exception as e:
            logger.warning(f"SDK init failed: {e}. Will use REST fallback.")
            self.client = None

    @staticmethod
    def _parse_dollars_field(obj, dollars_field: str, legacy_field: str = None) -> float:
        """
        Extract a price from Kalshi API response, preferring the _dollars field.

        Post March 5, 2026 the legacy integer-cent fields are removed.
        The _dollars fields are FixedPointDollars strings (e.g., "0.5600").

        Args:
            obj: Market response object (SDK model or dict)
            dollars_field: e.g. 'yes_bid_dollars'
            legacy_field: e.g. 'yes_bid' (fallback, may be None after deprecation)

        Returns:
            float in [0, 1] or 0.0 if unavailable
        """
        # Try _dollars field first (string like "0.5600")
        if hasattr(obj, dollars_field):
            val = getattr(obj, dollars_field, None)
        elif isinstance(obj, dict):
            val = obj.get(dollars_field)
        else:
            val = None

        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass

        # Fallback to legacy cents field (integer 0-100) — removed after March 5 2026
        if legacy_field:
            if hasattr(obj, legacy_field):
                cents = getattr(obj, legacy_field, None)
            elif isinstance(obj, dict):
                cents = obj.get(legacy_field)
            else:
                cents = None

            if cents is not None:
                try:
                    c = float(cents)
                    # Legacy fields were integers 0-100 (cents); convert to dollars
                    return c / 100.0 if c > 1.0 else c
                except (TypeError, ValueError):
                    pass

        return 0.0

    def get_spx_brackets(self, date: str = None) -> list:
        """
        Get all SPX bracket contracts for a given date.

        Args:
            date: 'YYYY-MM-DD' format. Defaults to today.

        Returns:
            list of bracket dicts with ticker, floor, cap, prices (all prices in dollars 0-1)
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Convert date to Kalshi event ticker format: INXD-YYMMMDD
        dt = datetime.strptime(date, "%Y-%m-%d")
        event_ticker = f"INXD-{dt.strftime('%y%b%d').upper()}"

        if self.client:
            try:
                event = self.client.get_event(event_ticker=event_ticker)
                brackets = []
                for market_ticker in event.markets:
                    market = self.client.get_market(ticker=market_ticker)
                    # Parse floor and cap from ticker or subtitle
                    floor, cap = self._parse_bracket(market)
                    brackets.append({
                        "ticker": market_ticker,
                        "floor": floor,
                        "cap": cap,
                        # Primary: _dollars fields (FixedPointDollars strings, subpenny)
                        # Fallback: legacy integer-cent fields (removed after March 5 2026)
                        "yes_bid": self._parse_dollars_field(market, "yes_bid_dollars", "yes_bid"),
                        "yes_ask": self._parse_dollars_field(market, "yes_ask_dollars", "yes_ask"),
                        "no_bid": self._parse_dollars_field(market, "no_bid_dollars", "no_bid"),
                        "no_ask": self._parse_dollars_field(market, "no_ask_dollars", "no_ask"),
                        "volume": market.volume,
                        "open_interest": market.open_interest,
                        "status": market.status,
                    })
                return brackets
            except Exception as e:
                logger.error(f"SDK bracket fetch failed: {e}")

        # Fallback: REST API
        return self._rest_get_brackets(event_ticker)

    def _parse_bracket(self, market) -> tuple:
        """Parse floor and cap from market subtitle like '5800 to 5825'."""
        try:
            subtitle = market.subtitle or ""
            if "or above" in subtitle.lower():
                val = float(subtitle.split()[0].replace(",", ""))
                return (val, float("inf"))
            elif "or below" in subtitle.lower():
                val = float(subtitle.split()[0].replace(",", ""))
                return (float("-inf"), val)
            elif " to " in subtitle:
                parts = subtitle.split(" to ")
                return (float(parts[0].replace(",", "")), float(parts[1].replace(",", "")))
        except Exception as e:
            logger.warning(f"Could not parse bracket from '{market.subtitle}': {e}")

        # Fallback: parse from ticker
        try:
            parts = market.ticker.split("-B")[1].split("-")
            return (float(parts[0]), float(parts[1]))
        except:
            return (0, 0)

    def _rest_get_brackets(self, event_ticker: str) -> list:
        """REST API fallback for fetching brackets.

        Parses _dollars fields from the JSON response and normalizes to
        internal dict format with float prices in [0, 1].
        """
        import urllib.request
        url = f"{self.base_url}/events/{event_ticker}"
        try:
            req = urllib.request.Request(url)
            if self.api_key:
                # TODO: add RSA-PSS auth headers
                pass
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                raw_markets = data.get("markets", [])
                brackets = []
                for m in raw_markets:
                    brackets.append({
                        "ticker": m.get("ticker", ""),
                        "floor": m.get("floor_strike", 0),
                        "cap": m.get("cap_strike", 0),
                        "yes_bid": self._parse_dollars_field(m, "yes_bid_dollars", "yes_bid"),
                        "yes_ask": self._parse_dollars_field(m, "yes_ask_dollars", "yes_ask"),
                        "no_bid": self._parse_dollars_field(m, "no_bid_dollars", "no_bid"),
                        "no_ask": self._parse_dollars_field(m, "no_ask_dollars", "no_ask"),
                        "volume": m.get("volume"),
                        "open_interest": m.get("open_interest"),
                        "status": m.get("status"),
                    })
                return brackets
        except Exception as e:
            logger.error(f"REST bracket fetch failed: {e}")
            return []

    def scan_for_opportunities(self, current_spx: float, predicted_vol: float = None,
                               hours_to_close: float = None, vix_level: float = None) -> list:
        """
        Main entry point: scan current SPX brackets for trading opportunities.

        Args:
            current_spx: current SPX price
            predicted_vol: annualized vol (if None, uses vol model)
            hours_to_close: hours until market close (if None, calculates from current time)
            vix_level: optional current VIX level (improves vol estimate)

        Returns:
            list of opportunities sorted by edge
        """
        if hours_to_close is None:
            now = datetime.now()
            close_time = now.replace(hour=16, minute=0, second=0)
            hours_to_close = max(0.01, (close_time - now).total_seconds() / 3600)

        if predicted_vol is None:
            vol_pred = self.vol_model.predict_vol(current_spx, hours_to_close, vix_level=vix_level)
            predicted_vol = vol_pred["annualized_vol"]
            vol_source = vol_pred.get("source", "unknown")
            vol_confidence = vol_pred.get("confidence", 0)
        else:
            vol_source = "manual"
            vol_confidence = 1.0

        brackets = self.get_spx_brackets()
        if not brackets:
            logger.warning("No live brackets from API — falling back to synthetic brackets")
            brackets = self._generate_synthetic_brackets(current_spx, predicted_vol, hours_to_close)
            if not brackets:
                logger.error("Synthetic bracket generation also failed")
                return []
            logger.info(f"Using {len(brackets)} synthetic brackets (no Kalshi API connection)")

        priced = self.pricer.price_all_brackets(brackets, current_spx, predicted_vol, hours_to_close)

        # Filter by minimum edge
        opportunities = [
            p for p in priced
            if p["net_edge_after_fees"] >= self.min_net_edge and p["action"] != "NO_EDGE"
        ]

        logger.info(f"Found {len(opportunities)} opportunities out of {len(priced)} brackets "
                     f"(SPX={current_spx}, vol={predicted_vol:.3f}, hours={hours_to_close:.1f}, "
                     f"source={vol_source}, confidence={vol_confidence:.2f})")

        return opportunities

    def _generate_synthetic_brackets(
        self,
        current_spx: float,
        predicted_vol: float,
        hours_to_close: float,
        bracket_width: int = 25,
        n_brackets_each_side: int = 6,
    ) -> list:
        """
        Generate synthetic SPX bracket contracts when the live Kalshi API is unavailable.

        This replicates what scan_brackets.py does for standalone use.
        Prices are set to represent realistic Kalshi market conditions:
        - ATM brackets get ~30% implied probability (typical for narrow 25pt brackets)
        - Spread is 2 cents around mid (typical for liquid markets)

        The purpose is to allow the vol model's edge vs. market pricing to be
        calculated even when live Kalshi API data is not accessible (e.g., no API
        key, demo API returning 404, or outside market hours).

        Args:
            current_spx: current SPX price
            predicted_vol: annualized vol (used for realistic mock pricing)
            hours_to_close: hours to market close
            bracket_width: width of each bracket in SPX points (Kalshi default = 25)
            n_brackets_each_side: number of brackets on each side of ATM

        Returns:
            list of bracket dicts with ticker, floor, cap, and bid/ask prices
        """
        brackets = []
        base = int(current_spx / bracket_width) * bracket_width

        # Sigma for one bracket period: SPX * vol * sqrt(t)
        time_fraction = hours_to_close / (252 * 6.5)
        sigma = current_spx * predicted_vol * np.sqrt(max(time_fraction, 1e-6))

        for i in range(-n_brackets_each_side, n_brackets_each_side + 1):
            floor = base + i * bracket_width
            cap = floor + bracket_width
            center = (floor + cap) / 2.0
            distance = abs(current_spx - center)

            # Approximate market-implied probability using normal distribution
            # This represents what a typical market maker would price
            # We use a slight premium (fat-tail adjustment) relative to pure normal
            if sigma > 0:
                # Fat-tail market price: slightly wider than normal
                raw_prob = (bracket_width / (sigma * math.sqrt(2 * math.pi))) * math.exp(
                    -0.5 * (distance / sigma) ** 2
                )
                # Add fat-tail premium (market makers charge extra for tails)
                fat_tail_premium = 0.015 if distance > 2 * sigma else 0.005
                market_prob = max(0.02, min(0.85, raw_prob + fat_tail_premium))
            else:
                market_prob = 0.02

            # Spread: 2 cents for liquid markets, wider for tails
            if distance > 3 * sigma:
                spread = 0.04  # 4 cent spread for deep OTM
            elif distance > 1.5 * sigma:
                spread = 0.03
            else:
                spread = 0.02  # 2 cent spread ATM

            yes_mid = round(market_prob, 4)
            no_mid = round(1.0 - market_prob, 4)

            brackets.append({
                "ticker": f"INXD-SIM-B{int(floor)}-{int(cap)}",
                "floor": float(floor),
                "cap": float(cap),
                "yes_ask": round(min(0.99, yes_mid + spread / 2), 4),
                "yes_bid": round(max(0.01, yes_mid - spread / 2), 4),
                "no_ask": round(min(0.99, no_mid + spread / 2), 4),
                "no_bid": round(max(0.01, no_mid - spread / 2), 4),
                "volume": 0,
                "open_interest": 0,
                "status": "active",
                "_synthetic": True,
            })

        return brackets

    def scan_with_mispricing_analysis(self, current_spx: float,
                                      hours_to_close: float = None,
                                      vix_level: float = None) -> dict:
        """Scan brackets with full mispricing analysis from the vol model.

        Unlike scan_for_opportunities which just returns edges, this method
        returns the full model-vs-market comparison including vol source,
        confidence levels, and mispricing magnitudes.

        Args:
            current_spx: current SPX price
            hours_to_close: hours until market close
            vix_level: optional current VIX level

        Returns:
            dict from BracketPricer.find_mispricings()
        """
        if hours_to_close is None:
            now = datetime.now()
            close_time = now.replace(hour=16, minute=0, second=0)
            hours_to_close = max(0.01, (close_time - now).total_seconds() / 3600)

        brackets = self.get_spx_brackets()
        if not brackets:
            logger.warning("No live brackets from API — falling back to synthetic brackets for mispricing analysis")
            # We need a vol estimate to generate synthetic brackets
            vol_est = self.vol_model.get_current_vol_estimate(
                current_spx=current_spx,
                hours_to_close=hours_to_close,
                vix_level=vix_level,
            )
            brackets = self._generate_synthetic_brackets(
                current_spx, vol_est["annualized_vol"], hours_to_close
            )
            if not brackets:
                return {"vol_estimate": {}, "brackets": [], "summary": {}, "top_opportunities": []}
            logger.info(f"Using {len(brackets)} synthetic brackets for mispricing analysis")

        return self.pricer.find_mispricings(
            brackets, current_spx, hours_to_close, vix_level=vix_level
        )

    def place_order(self, ticker: str, side: str, price: float, contracts: int) -> dict:
        """
        Place an order on Kalshi.

        Args:
            ticker: contract ticker (e.g., 'INXD-26MAR03-B5800-5825')
            side: 'yes' or 'no'
            price: limit price (0-1)
            contracts: number of contracts

        Returns:
            order confirmation dict
        """
        # Risk checks — price is in dollars (0-1), cost = contracts * price (dollars)
        cost = contracts * price
        if cost > self.max_single_trade:
            logger.warning(f"Order exceeds max single trade: ${cost:.2f} > ${self.max_single_trade:.2f}")
            contracts = int(self.max_single_trade / price)

        if contracts <= 0:
            return {"error": "Position too small after risk limit"}

        logger.info(f"Placing order: {side.upper()} {contracts}x {ticker} @ {price:.4f} (${contracts * price:.2f})")

        if self.client:
            try:
                # Post March 5, 2026: use _dollars params (FixedPointDollars string).
                # Format price as 4-decimal string for subpenny precision.
                price_str = f"{price:.4f}"
                order = self.client.create_order(
                    ticker=ticker,
                    side=side,
                    type="limit",
                    count=contracts,
                    yes_price_dollars=price_str if side == "yes" else None,
                    no_price_dollars=price_str if side == "no" else None,
                )
                result = {
                    "order_id": order.order_id,
                    "ticker": ticker,
                    "side": side,
                    "price": price,
                    "contracts": contracts,
                    "status": "placed",
                    "timestamp": datetime.now().isoformat(),
                }
                self.trade_log.append(result)
                return result
            except Exception as e:
                logger.error(f"Order placement failed: {e}")
                return {"error": str(e)}

        # Demo mode: simulate
        result = {
            "order_id": f"demo_{int(time.time())}",
            "ticker": ticker,
            "side": side,
            "price": price,
            "contracts": contracts,
            "status": "simulated",
            "timestamp": datetime.now().isoformat(),
        }
        self.trade_log.append(result)
        return result

    def get_portfolio(self) -> dict:
        """Get current portfolio state.

        Post March 5, 2026: balance and average_price use _dollars fields (already in dollars).
        Falls back to legacy cents division for older SDK versions.
        """
        if self.client:
            try:
                balance = self.client.get_balance()
                positions = self.client.get_positions()

                # Balance: prefer balance_dollars (string), fallback to cents integer
                bal_dollars = getattr(balance, 'balance_dollars', None)
                if bal_dollars is not None:
                    bal = float(bal_dollars)
                else:
                    bal = balance.balance / 100.0

                position_list = []
                for p in (positions.market_positions or []):
                    # Average price: prefer _dollars field
                    avg_dollars = getattr(p, 'average_price_dollars', None)
                    if avg_dollars is not None:
                        avg = float(avg_dollars)
                    elif p.average_price:
                        avg = p.average_price / 100.0
                    else:
                        avg = 0.0
                    position_list.append({
                        "ticker": p.ticker,
                        "count": p.total_traded,
                        "avg_price": avg,
                    })

                return {
                    "balance": bal,
                    "positions": position_list,
                }
            except Exception as e:
                logger.error(f"Portfolio fetch failed: {e}")

        return {"balance": 0, "positions": [], "note": "Not connected to API"}

    def save_state(self):
        """Save trading state to disk."""
        state = {
            "positions": self.positions,
            "trade_log": self.trade_log[-100:],  # Last 100 trades
            "pnl_history": self.pnl_history[-365:],  # Last year
            "timestamp": datetime.now().isoformat(),
        }
        state_file = DATA_DIR / "kalshi_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"State saved to {state_file}")

    def load_state(self):
        """Load trading state from disk."""
        state_file = DATA_DIR / "kalshi_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            self.positions = state.get("positions", {})
            self.trade_log = state.get("trade_log", [])
            self.pnl_history = state.get("pnl_history", [])
            logger.info(f"State loaded: {len(self.trade_log)} trades, {len(self.positions)} positions")


def run_backtest(historical_vol_predictions: list, historical_brackets: list) -> dict:
    """
    Backtest the vol-based bracket trading strategy.

    Args:
        historical_vol_predictions: list of {date, predicted_vol, actual_vol, spx_close}
        historical_brackets: list of {date, brackets: [{ticker, floor, cap, yes_ask, no_ask, resolution}]}
                              (prices in dollars 0-1)

    Returns:
        backtest results dict
    """
    pricer = BracketPricer()
    trades = []
    daily_pnl = []

    for day_data in historical_brackets:
        date = day_data["date"]
        brackets = day_data["brackets"]

        # Find matching vol prediction
        vol_pred = next((v for v in historical_vol_predictions if v["date"] == date), None)
        if not vol_pred:
            continue

        spx = vol_pred.get("spx_open", vol_pred.get("spx_close", 5900))
        predicted_vol = vol_pred["predicted_vol"]

        # Price brackets at open (6.5 hours to close)
        priced = pricer.price_all_brackets(brackets, spx, predicted_vol, 6.5)

        day_pnl = 0
        for p in priced:
            if p["net_edge_after_fees"] < 0.005:
                continue

            # Simulate trade
            if p["action"] == "BUY_YES":
                cost = p["entry_price"]
                resolution = next(
                    (b["resolution"] for b in brackets if b["ticker"] == p["ticker"]),
                    None
                )
                if resolution is None:
                    continue
                payout = 1.0 if resolution == "yes" else 0.0
                pnl = payout - cost - p["fee"]
            elif p["action"] == "BUY_NO":
                cost = p["entry_price"]
                resolution = next(
                    (b["resolution"] for b in brackets if b["ticker"] == p["ticker"]),
                    None
                )
                if resolution is None:
                    continue
                payout = 1.0 if resolution == "no" else 0.0
                pnl = payout - cost - p["fee"]
            else:
                continue

            trades.append({
                "date": date,
                "ticker": p["ticker"],
                "action": p["action"],
                "entry_price": cost,
                "edge": p["edge"],
                "pnl": pnl,
            })
            day_pnl += pnl

        daily_pnl.append({"date": date, "pnl": day_pnl, "trades": len([t for t in trades if t["date"] == date])})

    total_pnl = sum(d["pnl"] for d in daily_pnl)
    total_trades = len(trades)
    wins = len([t for t in trades if t["pnl"] > 0])

    return {
        "total_pnl": round(total_pnl, 2),
        "total_trades": total_trades,
        "win_rate": round(wins / total_trades, 3) if total_trades > 0 else 0,
        "avg_pnl_per_trade": round(total_pnl / total_trades, 4) if total_trades > 0 else 0,
        "daily_pnl": daily_pnl,
        "trades": trades,
        "sharpe": round(
            np.mean([d["pnl"] for d in daily_pnl]) / (np.std([d["pnl"] for d in daily_pnl]) + 1e-8)
            * np.sqrt(252), 2
        ) if daily_pnl else 0,
    }


if __name__ == "__main__":
    # Quick test
    print("=== Kalshi Client Test ===")
    print(f"SDK available: {HAS_SDK}")
    print(f"Scipy available: {HAS_SCIPY}")
    print(f"LightGBM available: {HAS_LGB}")

    # Test vol model
    print("\n--- Vol Model Calibration ---")
    vm = VolModel()
    cal = vm.get_calibration_summary()
    print(f"Mode: {cal['mode']}")
    print(f"Calibration loaded from file: {cal['calibration_loaded_from_file']}")
    print(f"Signal file: {cal['signal_file']} (exists: {cal['signal_file_exists']})")
    print("\nCalibration by horizon:")
    for hz, stats in cal["calibration"].items():
        print(f"  {hz:>5s}:  IC={stats['mean_ic']:.3f}  t={stats['t_stat']:.1f}  "
              f"pct+={stats['pct_positive']:.1f}%  n={stats['n_folds']}")

    # Test vol estimate
    print("\n--- Vol Estimate (no model, fallback) ---")
    est = vm.get_current_vol_estimate(current_spx=5900, hours_to_close=4.0)
    print(f"  Vol: {est['annualized_vol']:.3f} ({est['vol_30min_forecast_pct']:.1f}% ann)")
    print(f"  Source: {est['source']}")
    print(f"  Confidence: {est['confidence']:.2f}")
    print(f"  Horizon: {est['horizon']}")
    print(f"  1SD move: {est['expected_move_1sd']:.1f} pts")
    print(f"  1SD range: {est['1sd_range'][0]:.0f} - {est['1sd_range'][1]:.0f}")

    # Test with VIX
    print("\n--- Vol Estimate (VIX=22, adjusted) ---")
    est_vix = vm.get_current_vol_estimate(current_spx=5900, hours_to_close=4.0, vix_level=22.0)
    print(f"  Vol: {est_vix['annualized_vol']:.3f} ({est_vix['vol_30min_forecast_pct']:.1f}% ann)")
    print(f"  Source: {est_vix['source']}")
    print(f"  Confidence: {est_vix['confidence']:.2f}")

    # Test bracket pricing with enhanced model
    print("\n--- Bracket Pricing (Student-t df=4) ---")
    pricer = BracketPricer(vm, df=4.0)
    spx = 5900
    vol = 0.18
    hours = 4.0

    test_brackets = [
        {"ticker": "test-B5800-5825", "floor": 5800, "cap": 5825, "yes_ask": 0.05, "no_ask": 0.96, "yes_bid": 0.04, "no_bid": 0.95},
        {"ticker": "test-B5825-5850", "floor": 5825, "cap": 5850, "yes_ask": 0.08, "no_ask": 0.93, "yes_bid": 0.07, "no_bid": 0.92},
        {"ticker": "test-B5850-5875", "floor": 5850, "cap": 5875, "yes_ask": 0.15, "no_ask": 0.86, "yes_bid": 0.14, "no_bid": 0.85},
        {"ticker": "test-B5875-5900", "floor": 5875, "cap": 5900, "yes_ask": 0.30, "no_ask": 0.71, "yes_bid": 0.29, "no_bid": 0.70},
        {"ticker": "test-B5900-5925", "floor": 5900, "cap": 5925, "yes_ask": 0.30, "no_ask": 0.71, "yes_bid": 0.29, "no_bid": 0.70},
        {"ticker": "test-B5925-5950", "floor": 5925, "cap": 5950, "yes_ask": 0.15, "no_ask": 0.86, "yes_bid": 0.14, "no_bid": 0.85},
        {"ticker": "test-B5950-5975", "floor": 5950, "cap": 5975, "yes_ask": 0.08, "no_ask": 0.93, "yes_bid": 0.07, "no_bid": 0.92},
        {"ticker": "test-B5975-6000", "floor": 5975, "cap": 6000, "yes_ask": 0.05, "no_ask": 0.96, "yes_bid": 0.04, "no_bid": 0.95},
    ]

    results = pricer.price_all_brackets(test_brackets, spx, vol, hours)
    print(f"\nSPX={spx}, Vol={vol}, Hours to close={hours}")
    print(f"{'Ticker':<25} {'Floor-Cap':<12} {'FairVal':>8} {'YesAsk':>8} {'Edge':>8} {'NetEdge':>8} {'Action':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r['ticker']:<25} {r['floor']:.0f}-{r['cap']:.0f}  {r['fair_value']:>8.4f} {r['yes_ask']:>8.2f} {r['edge']:>8.4f} {r['net_edge_after_fees']:>8.4f} {r['action']:<10}")

    # Test mispricing analysis
    print("\n--- Mispricing Analysis ---")
    analysis = pricer.find_mispricings(test_brackets, spx, hours, vix_level=20.0)
    print(f"Summary: {json.dumps(analysis['summary'], indent=2)}")
    if analysis["top_opportunities"]:
        print(f"\nTop opportunity: {analysis['top_opportunities'][0]['ticker']}")
        print(f"  Model prob: {analysis['top_opportunities'][0]['model_prob']:.4f}")
        print(f"  Market prob: {analysis['top_opportunities'][0]['market_prob']:.4f}")
        print(f"  Mispricing: {analysis['top_opportunities'][0]['mispricing']:.4f}")
        print(f"  Adjusted edge: {analysis['top_opportunities'][0]['adjusted_edge']:.4f}")
