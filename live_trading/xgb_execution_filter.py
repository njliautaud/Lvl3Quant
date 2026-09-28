#!/usr/bin/env python3
"""
XGBoost Execution Filter — Inference Module
=============================================
Loads a trained XGBoost model and provides a simple API for the paper trader
to filter signal entries based on predicted profitability.

Usage in paper trader:
    from xgb_execution_filter import XGBExecutionFilter

    # Initialize once at startup
    xgb_filter = XGBExecutionFilter(
        model_path="output/exec_xgb_v1/fold_54_model.json",
        threshold_pct=5.0,  # top 5% = trade
    )

    # On each signal:
    should_trade, prob = xgb_filter.should_trade(
        predictions=np.array([pred_1s, pred_5s, pred_10s]),
        embeddings=embedding_vector,  # 96-dim from CNN-Mamba
    )
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger("xgb_filter")

# Feature engineering must EXACTLY match training (train_exec_lgbm.py / train_exec_xgboost.py)
SIGNAL_FEATURE_NAMES = [
    "pred_1s", "pred_5s", "pred_10s",
    "abs_pred_1s", "abs_pred_5s", "abs_pred_10s",
    "signal_agreement", "signal_trend", "max_signal_magnitude",
    "signal_ratio", "signal_sign",
    "pred_std", "pred_range", "signal_strength", "confidence_tier",
    "pred_1s_x_5s", "pred_1s_x_10s", "pred_5s_x_10s",
    "abs_pred_decay_1s_5s", "abs_pred_decay_5s_10s",
    "signal_skew",
]

EMBEDDING_FEATURE_NAMES = [f"emb_{i:02d}" for i in range(96)]


def build_features_single(
    predictions: np.ndarray,  # (3,) — pred_1s, pred_5s, pred_10s
    embeddings: Optional[np.ndarray] = None,  # (96,) or None
) -> np.ndarray:
    """
    Build feature vector for a single prediction.
    Returns (117,) or (21,) array matching training feature space.
    """
    pred_1s = float(predictions[0])
    pred_5s = float(predictions[1])
    pred_10s = float(predictions[2])

    abs_1s = abs(pred_1s)
    abs_5s = abs(pred_5s)
    abs_10s = abs(pred_10s)

    direction = 1.0 if pred_1s >= 0 else -1.0

    # Signal agreement
    agreement = 1.0 if (pred_1s > 0) == (pred_10s > 0) else 0.0

    # Signal trend
    trend = pred_1s - pred_10s

    # Max signal magnitude
    max_mag = max(abs_1s, abs_5s, abs_10s)

    # Signal ratio (safe)
    safe_10s = pred_10s if abs(pred_10s) > 0.01 else 0.01 * (1.0 if pred_10s >= 0 else -1.0)
    ratio = max(-5.0, min(5.0, pred_1s / safe_10s))

    # Derived features
    preds_arr = np.array([pred_1s, pred_5s, pred_10s])
    pred_std = float(np.std(preds_arr))
    pred_range = float(np.ptp(preds_arr))
    signal_strength = (abs_1s + abs_5s + abs_10s) / 3.0

    # Confidence tier
    if abs_1s >= 0.50:
        conf_tier = 3.0
    elif abs_1s >= 0.25:
        conf_tier = 2.0
    elif abs_1s >= 0.10:
        conf_tier = 1.0
    else:
        conf_tier = 0.0

    # Interaction terms
    p1x5 = pred_1s * pred_5s
    p1x10 = pred_1s * pred_10s
    p5x10 = pred_5s * pred_10s

    # Decay profile
    decay_1s_5s = abs_1s - abs_5s
    decay_5s_10s = abs_5s - abs_10s

    # Signal skew
    mean_pred = np.mean(preds_arr)
    std_safe = max(pred_std, 1e-8)
    skew = float(
        ((pred_1s - mean_pred)**3 + (pred_5s - mean_pred)**3 + (pred_10s - mean_pred)**3)
        / (3.0 * std_safe**3)
    )

    features = np.array([
        pred_1s, pred_5s, pred_10s,
        abs_1s, abs_5s, abs_10s,
        agreement, trend, max_mag,
        ratio, direction,
        pred_std, pred_range, signal_strength, conf_tier,
        p1x5, p1x10, p5x10,
        decay_1s_5s, decay_5s_10s,
        skew,
    ], dtype=np.float32)

    # Add embeddings if available
    if embeddings is not None:
        features = np.concatenate([features, embeddings.astype(np.float32)])

    return features


class XGBExecutionFilter:
    """
    XGBoost-based execution filter for signal quality gating.

    Loads a pre-trained XGBoost model and provides should_trade() API.
    """

    def __init__(
        self,
        model_path: str,
        threshold_pct: float = 5.0,
        calibration_probs: Optional[np.ndarray] = None,
    ):
        """
        Args:
            model_path: Path to XGBoost model JSON file
            threshold_pct: Percentile threshold (e.g., 5.0 = top 5%)
            calibration_probs: Optional array of OOT probabilities for threshold calibration
        """
        try:
            import xgboost as xgb
        except ImportError:
            log.error("xgboost not installed! pip install xgboost")
            raise

        self.model_path = Path(model_path)
        self.threshold_pct = threshold_pct
        self.model = xgb.Booster()
        self.model.load_model(str(self.model_path))
        log.info(f"Loaded XGBoost model from {self.model_path}")

        # If we have calibration probs, compute the threshold
        if calibration_probs is not None:
            self.prob_threshold = float(np.percentile(calibration_probs, 100.0 - threshold_pct))
            log.info(f"Calibrated threshold: prob > {self.prob_threshold:.4f} (top {threshold_pct}%)")
        else:
            # Default threshold — should be calibrated from OOT predictions
            self.prob_threshold = 0.5 + (threshold_pct / 100.0) * 0.1
            log.warning(f"No calibration data — using heuristic threshold: {self.prob_threshold:.4f}")

        self._n_filtered = 0
        self._n_passed = 0

    def should_trade(
        self,
        predictions: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float]:
        """
        Evaluate whether this signal should be traded.

        Args:
            predictions: (3,) array of [pred_1s, pred_5s, pred_10s]
            embeddings: (96,) array of CNN-Mamba embeddings (optional but recommended)

        Returns:
            (should_trade: bool, probability: float)
        """
        import xgboost as xgb

        features = build_features_single(predictions, embeddings)
        dmat = xgb.DMatrix(features.reshape(1, -1))
        prob = float(self.model.predict(dmat)[0])

        should = prob >= self.prob_threshold

        if should:
            self._n_passed += 1
        else:
            self._n_filtered += 1

        return should, prob

    @property
    def filter_rate(self) -> float:
        """Fraction of signals filtered out."""
        total = self._n_filtered + self._n_passed
        return self._n_filtered / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        """Current filter statistics."""
        total = self._n_filtered + self._n_passed
        return {
            "passed": self._n_passed,
            "filtered": self._n_filtered,
            "total": total,
            "pass_rate": self._n_passed / total if total > 0 else 0.0,
            "filter_rate": self.filter_rate,
            "threshold": self.prob_threshold,
        }

    def reset_stats(self):
        """Reset filter statistics."""
        self._n_filtered = 0
        self._n_passed = 0


class LGBMExecutionFilter:
    """
    LightGBM-based execution filter (alternative to XGBoost).
    Same API as XGBExecutionFilter.
    """

    def __init__(
        self,
        model_path: str,
        threshold_pct: float = 5.0,
        calibration_probs: Optional[np.ndarray] = None,
    ):
        try:
            import lightgbm as lgb
        except ImportError:
            log.error("lightgbm not installed! pip install lightgbm")
            raise

        self.model_path = Path(model_path)
        self.threshold_pct = threshold_pct
        self.model = lgb.Booster(model_file=str(self.model_path))
        log.info(f"Loaded LightGBM model from {self.model_path}")

        if calibration_probs is not None:
            self.prob_threshold = float(np.percentile(calibration_probs, 100.0 - threshold_pct))
        else:
            self.prob_threshold = 0.5 + (threshold_pct / 100.0) * 0.1

        self._n_filtered = 0
        self._n_passed = 0

    def should_trade(
        self,
        predictions: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float]:
        features = build_features_single(predictions, embeddings)
        prob = float(self.model.predict(features.reshape(1, -1))[0])

        should = prob >= self.prob_threshold
        if should:
            self._n_passed += 1
        else:
            self._n_filtered += 1

        return should, prob

    @property
    def filter_rate(self) -> float:
        total = self._n_filtered + self._n_passed
        return self._n_filtered / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        total = self._n_filtered + self._n_passed
        return {
            "passed": self._n_passed,
            "filtered": self._n_filtered,
            "total": total,
            "pass_rate": self._n_passed / total if total > 0 else 0.0,
            "filter_rate": self.filter_rate,
            "threshold": self.prob_threshold,
        }

    def reset_stats(self):
        self._n_filtered = 0
        self._n_passed = 0


if __name__ == "__main__":
    # Quick test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to XGBoost model JSON")
    parser.add_argument("--threshold", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    filt = XGBExecutionFilter(args.model, args.threshold)

    # Simulate some predictions
    np.random.seed(42)
    for i in range(20):
        preds = np.random.randn(3) * 0.3
        embs = np.random.randn(96).astype(np.float32) * 0.1
        should, prob = filt.should_trade(preds, embs)
        print(f"  Signal {i}: pred_1s={preds[0]:.3f} → prob={prob:.4f} → {'TRADE' if should else 'SKIP'}")

    print(f"\nFilter stats: {filt.stats}")
