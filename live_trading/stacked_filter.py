#!/usr/bin/env python3
"""
stacked_filter.py — Stacked Confluence Filter for Paper Trading
================================================================
Integrates three sequential gates into the live/paper trading pipeline:
  Gate 1: Signal threshold (top N% of CNN-Mamba v2 shorts by prediction magnitude)
  Gate 2: Meta-model (MLP ensemble predicting 1s P&L, keep top M%)
  Gate 3: OFI gate (book queue imbalance agrees with short direction)

Config: loads from monday_stacked_v1.json with preset selection.
Meta-model: 15-fold MLP ensemble (256->128->64->32->1), PyTorch.

Usage:
    from live_trading.stacked_filter import StackedFilter

    sf = StackedFilter(preset="balanced")  # or "simpler" / "ultra_selective"

    result = sf.should_trade(
        pred_1s=-0.15, pred_5s=-0.08, pred_10s=-0.04,
        event_features=np.zeros(25),  # 25-dim MBO event vector
        ofi_book_1s=-120.5,
    )
    if result.should_trade:
        # execute entry...

Cost basis: 0.376 ticks RT (passive-only, commission only).
Contract: ESU6.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_LVL3 = _THIS_DIR.parent  # /home/jupiter/Lvl3Quant or equivalent

DEFAULT_CONFIG_PATH = _THIS_DIR / "configs" / "monday_stacked_v1.json"


# ---------------------------------------------------------------------------
# Meta-model architecture (must match train_meta_production_v1.py exactly)
# ---------------------------------------------------------------------------

class ProductionMetaMLP(nn.Module):
    """Dynamic MLP: supports both v6 (256->128->64->32) and v7 (256->128->64) architectures."""
    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]  # v7 default
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Result from the stacked filter evaluation."""
    should_trade: bool
    gate_reasons: Dict[str, dict]  # gate_name -> {passed, value, threshold, ...}
    signal_pct_rank: float = 0.0   # percentile rank of prediction (0=weakest, 1=strongest short)
    meta_score: float = 0.0        # ensemble meta-model prediction
    ofi_agreement: bool = False    # whether OFI agrees with short direction


@dataclass
class GateStats:
    """Running statistics for gate pass rates."""
    total_evaluated: int = 0
    gate1_passed: int = 0
    gate2_passed: int = 0
    gate3_passed: int = 0
    all_passed: int = 0
    last_reset: float = field(default_factory=time.time)

    @property
    def gate1_rate(self) -> float:
        return self.gate1_passed / max(self.total_evaluated, 1)

    @property
    def gate2_rate(self) -> float:
        return self.gate2_passed / max(self.gate1_passed, 1)

    @property
    def gate3_rate(self) -> float:
        return self.gate3_passed / max(self.gate2_passed, 1)

    @property
    def overall_rate(self) -> float:
        return self.all_passed / max(self.total_evaluated, 1)

    def to_dict(self) -> dict:
        return {
            "total_evaluated": self.total_evaluated,
            "gate1_signal_passed": self.gate1_passed,
            "gate2_meta_passed": self.gate2_passed,
            "gate3_ofi_passed": self.gate3_passed,
            "all_passed": self.all_passed,
            "gate1_rate": round(self.gate1_rate, 4),
            "gate2_rate": round(self.gate2_rate, 4),
            "gate3_rate": round(self.gate3_rate, 4),
            "overall_rate": round(self.overall_rate, 4),
        }


# ---------------------------------------------------------------------------
# Main filter class
# ---------------------------------------------------------------------------

class StackedFilter:
    """
    Stacked confluence filter for CNN-Mamba v2 paper/live trading.

    Loads config + meta-model weights on init. Call should_trade() on each
    prediction to get a go/no-go decision with detailed gate information.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        preset: Optional[str] = None,
        meta_weights_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Args:
            config_path: Path to monday_stacked_v1.json. Uses default if None.
            preset: Override the active_preset from config ("balanced"/"simpler"/"ultra_selective").
            meta_weights_dir: Override path to meta-model fold weights directory.
            device: PyTorch device for meta-model inference ("cpu" or "cuda").
        """
        # Load config
        cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        with open(cfg_path, "r") as f:
            self._raw_config = json.load(f)

        # Resolve preset
        active_preset = preset or self._raw_config.get("active_preset", "balanced")
        if active_preset not in self._raw_config["presets"]:
            raise ValueError(f"Unknown preset '{active_preset}'. Available: {list(self._raw_config['presets'].keys())}")
        preset_cfg = self._raw_config["presets"][active_preset]
        self._preset_name = active_preset

        # Gate thresholds from preset
        self._signal_threshold_pct = preset_cfg["signal_threshold_pct"]
        self._meta_filter_pct = preset_cfg["meta_filter_pct"]
        self._ofi_gate_enabled = preset_cfg["ofi_gate"]

        # Meta-model enabled if filter_pct < 1.0
        self._meta_enabled = self._meta_filter_pct < 1.0 and self._raw_config.get("meta_model", {}).get("enabled", True)

        # Signal distribution tracker (rolling window for percentile computation)
        z_window = self._raw_config.get("signal", {}).get("z_score_window", 500)
        self._pred_history = deque(maxlen=max(z_window, 200))

        # Device
        self._device = torch.device(device)

        # Load meta-model ensemble
        self._meta_models: List[Tuple[ProductionMetaMLP, np.ndarray, np.ndarray]] = []
        if self._meta_enabled:
            weights_dir = meta_weights_dir or self._raw_config.get("meta_model", {}).get("weights_dir", "")
            if weights_dir:
                # Resolve relative paths against LVL3
                wdir = Path(weights_dir)
                if not wdir.is_absolute():
                    wdir = _LVL3 / weights_dir
                self._load_meta_ensemble(wdir)

        # Gate statistics
        self.stats = GateStats()

        logger.info(
            "StackedFilter initialized: preset=%s signal_pct=%.0f%% meta_pct=%.0f%% ofi=%s meta_folds=%d",
            self._preset_name,
            self._signal_threshold_pct * 100,
            self._meta_filter_pct * 100,
            self._ofi_gate_enabled,
            len(self._meta_models),
        )

    def _load_meta_ensemble(self, weights_dir: Path) -> None:
        """Load all fold .pt files from the meta-model weights directory."""
        if not weights_dir.exists():
            logger.warning("Meta weights dir not found: %s — meta gate disabled", weights_dir)
            self._meta_enabled = False
            return

        pt_files = sorted(weights_dir.glob("fold_*.pt"))
        if not pt_files:
            logger.warning("No fold_*.pt files in %s — meta gate disabled", weights_dir)
            self._meta_enabled = False
            return

        for pt_file in pt_files:
            try:
                ckpt = torch.load(pt_file, map_location=self._device, weights_only=False)
                input_dim = ckpt["input_dim"]
                # Infer hidden_dims from checkpoint config or state_dict
                cfg = ckpt.get("config", {})
                hidden_dims = cfg.get("hidden_dims", None) if isinstance(cfg, dict) else None
                if hidden_dims is None:
                    # Auto-detect from state_dict: count Linear layers and their sizes
                    state_key_tmp = "model_state_dict" if "model_state_dict" in ckpt else "model_state"
                    sd = ckpt[state_key_tmp]
                    hidden_dims = []
                    for k, v in sd.items():
                        if k.endswith(".weight") and v.dim() == 2 and v.shape[0] != 1:
                            hidden_dims.append(v.shape[0])
                model = ProductionMetaMLP(input_dim, hidden_dims=hidden_dims, dropout=0.0)
                # Support both v6 (model_state/norm_mean/norm_std) and v7 (model_state_dict/feat_mean/feat_std) key names
                state_key = "model_state_dict" if "model_state_dict" in ckpt else "model_state"
                model.load_state_dict(ckpt[state_key])
                model.to(self._device)
                model.eval()

                mean_key = "feat_mean" if "feat_mean" in ckpt else "norm_mean"
                std_key = "feat_std" if "feat_std" in ckpt else "norm_std"
                norm_mean = ckpt[mean_key].astype(np.float32)
                norm_std = ckpt[std_key].astype(np.float32)

                self._meta_models.append((model, norm_mean, norm_std))
            except Exception as e:
                logger.warning("Failed to load meta fold %s: %s", pt_file.name, e)

        if not self._meta_models:
            logger.warning("No meta models loaded successfully — meta gate disabled")
            self._meta_enabled = False
        else:
            logger.info("Loaded %d meta-model folds for ensemble", len(self._meta_models))

    # -----------------------------------------------------------------
    # Gate 1: Signal strength percentile
    # -----------------------------------------------------------------

    def _gate_signal(self, pred_1s: float) -> Tuple[bool, dict]:
        """
        Check if prediction is in the top N% of short signals.

        Uses rolling history of all predictions (not just shorts) to compute
        the percentile rank of the current prediction among shorts.
        """
        # Track all predictions for distribution
        self._pred_history.append(pred_1s)

        # Must be a short signal (negative prediction)
        if pred_1s >= 0:
            return False, {
                "passed": False,
                "reason": "not_short",
                "pred_1s": round(pred_1s, 6),
                "threshold_pct": self._signal_threshold_pct,
            }

        # Need minimum history for percentile calculation
        if len(self._pred_history) < 50:
            return False, {
                "passed": False,
                "reason": "warmup",
                "history_size": len(self._pred_history),
                "required": 50,
            }

        # Compute percentile rank among all predictions
        # For shorts: lower (more negative) = stronger signal
        history = np.array(self._pred_history)
        # What fraction of predictions are MORE negative than this one?
        pct_rank = float(np.mean(history <= pred_1s))
        # pct_rank close to 0 = very strong short (few predictions are more negative)

        # Pass if in the top N% of shorts (pct_rank <= threshold)
        passed = pct_rank <= self._signal_threshold_pct

        return passed, {
            "passed": passed,
            "pred_1s": round(pred_1s, 6),
            "pct_rank": round(pct_rank, 4),
            "threshold_pct": self._signal_threshold_pct,
            "history_size": len(self._pred_history),
        }

    # -----------------------------------------------------------------
    # Gate 2: Meta-model ensemble score
    # -----------------------------------------------------------------

    def _build_meta_features(
        self,
        pred_1s: float,
        pred_5s: float,
        pred_10s: float,
        event_features: np.ndarray,
    ) -> np.ndarray:
        """
        Build feature vector for the meta-model.

        Supports two layouts based on meta model version:
        - v7 (31-dim): [pred_1s, pred_5s, pred_10s, |pred_1s|, |pred_5s|, |pred_10s|, mbo_events(25)]
        - v6/v1 (29-dim): [mbo_events(25), pred_1s, pred_5s, pred_10s, pred_rank]

        Auto-detected from loaded model input_dim.
        """
        # Determine expected dim from first loaded model
        if self._meta_models:
            expected_dim = self._meta_models[0][1].shape[0]  # norm_mean shape
        else:
            expected_dim = 31  # default to v7

        if expected_dim == 31:
            # v7 layout: [preds(3), confidence(3), mbo(25)]
            feat = np.zeros(31, dtype=np.float32)
            feat[0] = pred_1s
            feat[1] = pred_5s
            feat[2] = pred_10s
            feat[3] = abs(pred_1s)
            feat[4] = abs(pred_5s)
            feat[5] = abs(pred_10s)
            n_ev = min(len(event_features), 25)
            feat[6:6+n_ev] = event_features[:n_ev]
        else:
            # v6/v1 layout: [mbo(25), pred_1s, pred_5s, pred_10s, pred_rank]
            feat = np.zeros(max(expected_dim, 29), dtype=np.float32)
            n_ev = min(len(event_features), 25)
            feat[:n_ev] = event_features[:n_ev]
            feat[25] = pred_1s
            feat[26] = pred_5s
            feat[27] = pred_10s
            # Rank: in live, we approximate with percentile from history
            if len(self._pred_history) >= 50:
                history = np.array(self._pred_history)
                feat[28] = float(np.mean(history <= pred_1s))  # same as pct_rank
            else:
                feat[28] = 0.5  # neutral default during warmup
        return feat

    def _gate_meta(
        self,
        pred_1s: float,
        pred_5s: float,
        pred_10s: float,
        event_features: np.ndarray,
    ) -> Tuple[bool, dict]:
        """
        Ensemble meta-model gate: average prediction across all folds.
        Keep signal if meta_score is in the top M% (i.e., highest predicted P&L).

        In live trading we don't have a batch to percentile-rank against, so we
        use the meta_score sign: positive = model expects profit. With meta_filter_pct < 1.0,
        we apply a rolling threshold based on recent meta scores.
        """
        if not self._meta_enabled or not self._meta_models:
            return True, {"passed": True, "reason": "meta_disabled"}

        features = self._build_meta_features(pred_1s, pred_5s, pred_10s, event_features)

        # Ensemble: average across folds
        scores = []
        with torch.no_grad():
            for model, norm_mean, norm_std in self._meta_models:
                feat_n = (features - norm_mean) / (norm_std + 1e-8)
                feat_t = torch.from_numpy(feat_n).unsqueeze(0).to(self._device)
                score = model(feat_t).item()
                scores.append(score)

        meta_score = float(np.mean(scores))
        meta_std = float(np.std(scores)) if len(scores) > 1 else 0.0

        # Track meta scores for rolling percentile
        if not hasattr(self, "_meta_score_history"):
            self._meta_score_history: deque = deque(maxlen=500)
        self._meta_score_history.append(meta_score)

        # Determine threshold from rolling history
        if len(self._meta_score_history) >= 30:
            history = np.array(self._meta_score_history)
            # Keep top M%: threshold at (1 - meta_filter_pct) percentile
            threshold = float(np.percentile(history, (1.0 - self._meta_filter_pct) * 100))
            passed = meta_score >= threshold
        else:
            # During warmup: pass if meta_score > 0 (model expects positive P&L)
            threshold = 0.0
            passed = meta_score > 0.0

        return passed, {
            "passed": passed,
            "meta_score": round(meta_score, 6),
            "meta_std": round(meta_std, 6),
            "threshold": round(threshold, 6),
            "n_folds": len(scores),
            "filter_pct": self._meta_filter_pct,
            "history_size": len(self._meta_score_history),
        }

    # -----------------------------------------------------------------
    # Gate 3: OFI book agreement
    # -----------------------------------------------------------------

    def _gate_ofi(self, ofi_book_1s: float) -> Tuple[bool, dict]:
        """
        OFI confluence gate: for short signals, require negative OFI
        (selling pressure in the order book at 1s horizon).
        """
        if not self._ofi_gate_enabled:
            return True, {"passed": True, "reason": "ofi_disabled"}

        # For shorts: OFI < 0 means selling pressure = agrees with short direction
        agrees = ofi_book_1s < 0

        return agrees, {
            "passed": agrees,
            "ofi_book_1s": round(ofi_book_1s, 4),
            "required_sign": "negative",
            "direction": "short",
        }

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def should_trade(
        self,
        pred_1s: float,
        pred_5s: float = 0.0,
        pred_10s: float = 0.0,
        event_features: Optional[np.ndarray] = None,
        ofi_book_1s: float = 0.0,
    ) -> GateResult:
        """
        Evaluate all three stacked gates sequentially.

        Short-circuits: if an earlier gate fails, later gates are not evaluated.
        This matches the live execution path (no wasted compute on rejected signals).

        Args:
            pred_1s: CNN-Mamba v2 prediction at 1s horizon (negative = short signal)
            pred_5s: CNN-Mamba v2 prediction at 5s horizon
            pred_10s: CNN-Mamba v2 prediction at 10s horizon
            event_features: 25-dim MBO event feature vector (smart_v3 format).
                           Required for meta-model gate. If None, uses zeros.
            ofi_book_1s: Book order flow imbalance at 1s horizon.
                        Negative = selling pressure.

        Returns:
            GateResult with should_trade bool and per-gate details.
        """
        self.stats.total_evaluated += 1

        if event_features is None:
            event_features = np.zeros(25, dtype=np.float32)

        gate_reasons: Dict[str, dict] = {}

        # Gate 1: Signal threshold
        g1_passed, g1_info = self._gate_signal(pred_1s)
        gate_reasons["signal_gate"] = g1_info
        signal_pct_rank = g1_info.get("pct_rank", 1.0)

        if not g1_passed:
            return GateResult(
                should_trade=False,
                gate_reasons=gate_reasons,
                signal_pct_rank=signal_pct_rank,
            )
        self.stats.gate1_passed += 1

        # Gate 2: Meta-model
        g2_passed, g2_info = self._gate_meta(pred_1s, pred_5s, pred_10s, event_features)
        gate_reasons["meta_gate"] = g2_info
        meta_score = g2_info.get("meta_score", 0.0)

        if not g2_passed:
            return GateResult(
                should_trade=False,
                gate_reasons=gate_reasons,
                signal_pct_rank=signal_pct_rank,
                meta_score=meta_score,
            )
        self.stats.gate2_passed += 1

        # Gate 3: OFI agreement
        g3_passed, g3_info = self._gate_ofi(ofi_book_1s)
        gate_reasons["ofi_gate"] = g3_info
        ofi_agreement = g3_info.get("passed", False)

        if not g3_passed:
            return GateResult(
                should_trade=False,
                gate_reasons=gate_reasons,
                signal_pct_rank=signal_pct_rank,
                meta_score=meta_score,
                ofi_agreement=False,
            )
        self.stats.gate3_passed += 1
        self.stats.all_passed += 1

        return GateResult(
            should_trade=True,
            gate_reasons=gate_reasons,
            signal_pct_rank=signal_pct_rank,
            meta_score=meta_score,
            ofi_agreement=True,
        )

    # -----------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return current gate pass-rate statistics."""
        return self.stats.to_dict()

    def reset_stats(self) -> None:
        """Reset gate statistics (e.g., at start of new trading day)."""
        self.stats = GateStats()

    @property
    def config_summary(self) -> dict:
        """Return summary of active configuration."""
        return {
            "preset": self._preset_name,
            "signal_threshold_pct": self._signal_threshold_pct,
            "meta_filter_pct": self._meta_filter_pct,
            "ofi_gate_enabled": self._ofi_gate_enabled,
            "meta_enabled": self._meta_enabled,
            "meta_folds_loaded": len(self._meta_models),
            "contract": self._raw_config.get("contract", {}).get("symbol", "ESU6"),
        }

    def __repr__(self) -> str:
        return (
            f"StackedFilter(preset={self._preset_name!r}, "
            f"signal={self._signal_threshold_pct:.0%}, "
            f"meta={self._meta_filter_pct:.0%}, "
            f"ofi={self._ofi_gate_enabled}, "
            f"folds={len(self._meta_models)})"
        )
