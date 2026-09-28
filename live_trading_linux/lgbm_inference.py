#!/usr/bin/env python3
"""
lgbm_inference.py — Thin wrapper around a joblib-saved LightGBM regressor.

The training pipeline (lgbm_prod_wf_60_5.py) writes one PKL per (fold, horizon).
For live trading we pick a specific fold/horizon PKL (typically the latest
fold's labels_10s model) and load it here.

Confidence tiers replicate the scheme used in training:
    all    — |pred| >= P0   (always true)
    top50  — |pred| >= P50
    top25  — |pred| >= P75
    top10  — |pred| >= P90

Percentile thresholds are learned offline from the training predictions and
stored in a JSON file so the live engine doesn't need to re-compute them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np


log = logging.getLogger("lgbm_inference")


# Tier ordering used by SignalEngine to compare "tier >= X"
TIER_ORDER = {"all": 0, "top50": 1, "top25": 2, "top10": 3}


class LGBMInference:
    """Loads a LightGBM model from disk and produces streaming predictions.

    Args:
        model_path:       path to a joblib .pkl produced by lgbm_prod_wf_60_5.py
        calibration_path: optional path to a JSON with tier thresholds.
                          JSON schema:
                              {
                                "p50": <float>,
                                "p75": <float>,
                                "p90": <float>
                              }
                          Values are thresholds on |pred| — above p90 → top10.

        If calibration is missing we default to zero thresholds, meaning every
        prediction is classified as 'top10' (tier gating effectively disabled).
    """

    def __init__(
        self,
        model_path:       str | Path,
        calibration_path: Optional[str | Path] = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model_path = model_path
        self.model = joblib.load(model_path)
        log.info("Loaded LGBM model: %s", model_path)

        # version string == filename stem (fold{NN}/{horizon}_lgbm)
        self.model_version = f"{model_path.parent.name}/{model_path.stem}"

        # --- Calibration thresholds -----------------------------------------
        self.thresh_p50 = 0.0
        self.thresh_p75 = 0.0
        self.thresh_p90 = 0.0
        self.calibration_path = None
        if calibration_path is not None:
            self._load_calibration(Path(calibration_path))

    # ------------------------------------------------------------- calibration
    def _load_calibration(self, p: Path) -> None:
        if not p.exists():
            log.warning("Calibration file not found: %s — all preds will be 'top10'.", p)
            return
        with open(p, "r") as f:
            data = json.load(f)
        self.thresh_p50 = float(data.get("p50", 0.0))
        self.thresh_p75 = float(data.get("p75", 0.0))
        self.thresh_p90 = float(data.get("p90", 0.0))
        self.calibration_path = p
        log.info("Loaded calibration: p50=%.4f p75=%.4f p90=%.4f",
                 self.thresh_p50, self.thresh_p75, self.thresh_p90)

    @staticmethod
    def build_calibration_from_preds(preds: np.ndarray) -> dict:
        """Helper for building a calibration dict from a vector of historical
        predictions.  Use this once offline and dump the result to JSON."""
        abs_p = np.abs(np.asarray(preds, dtype=np.float32))
        return {
            "p50": float(np.percentile(abs_p, 50)),
            "p75": float(np.percentile(abs_p, 75)),
            "p90": float(np.percentile(abs_p, 90)),
            "n":   int(len(abs_p)),
        }

    # ------------------------------------------------------------- predict
    def predict(self, feat_vec: np.ndarray) -> float:
        """Return a single scalar prediction for a 21-dim feature vector."""
        x = np.asarray(feat_vec, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != 21:
            raise ValueError(f"Expected 21 features, got {x.shape[1]}")
        y = self.model.predict(x)
        return float(y[0])

    def predict_tier(self, feat_vec: np.ndarray) -> tuple[str, float]:
        """Return (tier, prediction).

        Tier is chosen by |pred| vs. the calibration thresholds.
        """
        pred = self.predict(feat_vec)
        tier = self.classify_tier(pred)
        return tier, pred

    def classify_tier(self, pred: float) -> str:
        a = abs(pred)
        if a >= self.thresh_p90:
            return "top10"
        if a >= self.thresh_p75:
            return "top25"
        if a >= self.thresh_p50:
            return "top50"
        return "all"

    # ------------------------------------------------------------- batch
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorised predict — useful for offline calibration sweeps."""
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict(X)
