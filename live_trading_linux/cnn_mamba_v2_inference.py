#!/usr/bin/env python3
"""
cnn_mamba_v2_inference.py — CNN Mamba v2 inference engine for live paper trading.

Wraps the CNNMambaV2 model class (from train_cnn_mamba_v2.py) in an API that
mirrors `MambaInferenceEngine` from mamba_inference.py, so paper_trading_mamba.py
and mamba_live_pipeline.py can use it as a drop-in replacement.

Key differences from MambaInferenceEngine:
  * Uses CNNMambaV2 model class (CNN front-end + Mamba backbone, not pure Mamba)
  * Infers dt_rank from x_proj.weight tensor shape (works around a bug where
    train_cnn_mamba_v2.py saves the default dt_rank constant in arch dict
    rather than the actual training value)
  * Uses load_state_dict(strict=False) — older v2 checkpoints lack the
    `time_decay_rate` parameter introduced in newer SSM versions; defaults
    to 0.1 (small modeling diff acceptable for paper trade test)
  * SKIP_NORMALIZE-aware: smart_v3 features are pre-normalized at preprocessing,
    so feature_stats means are zeros and stds are ones (normalize() becomes
    a no-op but we keep the call for API symmetry and potential v3 retraining
    that re-introduces per-fold stats)

Usage:
    engine = CNNMambaV2Inference(
        weights_path="output/cnn_mamba_v2_smart_v3_mar/fold_10_best.pt",
        stats_path="output/cnn_mamba_v2_smart_v3_mar/fold_09_feature_stats.npz",
        window_size=1000,
        stride=500,
        device="cuda",  # RTX 3070 on Razer
    )
    result = engine.predict(feature_window)  # (1000, 25) -> dict
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

# Tell train_cnn_mamba_v2 it's smart_v3 (sets N_TOTAL_FEATURES=25)
os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
os.environ.setdefault("SKIP_NORMALIZE", "1")

# Ensure train_cnn_mamba_v2.py is importable. Caller can also prepend to sys.path.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import train_cnn_mamba_v2 as _T  # provides CNNMambaV2 class + N_TOTAL_FEATURES


def _infer_dt_rank(state_dict: dict, d_state: int) -> int:
    """Infer dt_rank from x_proj.weight tensor shape.

    SelectiveSSM.x_proj outputs (dt_rank + 2*d_state, d_inner) so given d_state
    we can recover dt_rank from the first dim. Use blocks.0.ssm.x_proj.weight.
    Falls back to default 16 if the key is missing.
    """
    key = "blocks.0.ssm.x_proj.weight"
    if key in state_dict:
        out_dim = state_dict[key].shape[0]
        return out_dim - 2 * d_state
    return 16


class CNNMambaV2Inference:
    """Live-trading inference wrapper for CNN Mamba v2 weights.

    API-compatible with MambaInferenceEngine: predict(window) and add_event(features).
    """

    def __init__(
        self,
        weights_path: str,
        stats_path: str,
        window_size: int = 1000,
        stride: int = 500,
        device: str = "cpu",
    ):
        self.window_size = window_size
        self.stride = stride
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        if "model_state" not in ckpt:
            raise ValueError(f"Checkpoint missing 'model_state' key. Keys: {list(ckpt.keys())}")
        state = ckpt["model_state"]
        arch = ckpt.get("arch", {})

        # Build kwargs from arch, then correct dt_rank from tensor shape
        kwargs = {}
        for k in ("d_model", "d_state", "n_layers", "dt_rank", "d_conv",
                  "dropout", "n_targets", "feature_mlp_hidden", "feature_mlp_out",
                  "cnn_channels_per_scale"):
            if k in arch:
                kwargs[k] = arch[k]

        d_state = kwargs.get("d_state", 32)
        kwargs["dt_rank"] = _infer_dt_rank(state, d_state)

        # Override window_size from arch if present
        if "window_size" in arch:
            self.window_size = int(arch["window_size"])
        if "n_features" in arch:
            self.n_features = int(arch["n_features"])
        else:
            self.n_features = _T.N_TOTAL_FEATURES

        print(f"[v2_inference] Building CNNMambaV2 with kwargs={kwargs}")
        print(f"[v2_inference] window_size={self.window_size} n_features={self.n_features}")
        print(f"[v2_inference] fold={ckpt.get('fold')} epoch={ckpt.get('epoch')} "
              f"val_ic_10s={ckpt.get('val_ic_10s')}")

        self.model = _T.CNNMambaV2(**kwargs)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[v2_inference] WARN unexpected keys: {unexpected}")
        if missing:
            print(f"[v2_inference] missing keys defaulted: {missing}")
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[v2_inference] Loaded. params={n_params:,} ({n_params/1e6:.3f}M) device={self.device}")

        self.model.to(self.device)
        self.model.eval()

        # Feature normalization stats (SKIP_NORMALIZE means these are zeros/ones)
        stats = np.load(stats_path)
        self.feat_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=self.device)
        self.feat_std = torch.tensor(stats["std"], dtype=torch.float32, device=self.device)
        self.feat_std = torch.clamp(self.feat_std, min=1e-8)
        # Detect SKIP_NORMALIZE pattern (means~0, stds~1)
        self._skip_normalize = bool(
            torch.allclose(self.feat_mean, torch.zeros_like(self.feat_mean), atol=1e-3)
            and torch.allclose(self.feat_std, torch.ones_like(self.feat_std), atol=1e-3)
        )
        print(f"[v2_inference] feature_stats loaded ({len(self.feat_mean)} features) "
              f"skip_normalize={self._skip_normalize}")

        # Streaming buffer
        self.event_buffer: list = []
        self.prediction_count = 0

        # Confidence thresholds — set via calibrate_thresholds()
        self.confidence_thresholds = {
            "Top5%": None,
            "Top1%": None,
            "Top0.5%": None,
            "Top0.1%": None,
        }

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        if self._skip_normalize:
            return features
        return (features - self.feat_mean) / self.feat_std

    @torch.no_grad()
    def predict(self, feature_window: np.ndarray) -> Dict:
        """Single-window inference. Returns dict matching MambaInferenceEngine API.

        Args:
            feature_window: (window_size, n_features) raw smart_v3 features
        """
        x = torch.tensor(feature_window, dtype=torch.float32, device=self.device)
        x = self.normalize(x)
        x = x.unsqueeze(0)  # (1, L, F)
        out = self.model(x)  # (1, 3) — last-token output
        if isinstance(out, tuple):
            out = out[0]
        preds = out.squeeze(0).cpu().numpy()
        if preds.ndim == 2:
            # Some configs return (L, 3) — take last position
            preds = preds[-1]
        pred_1s, pred_5s, pred_10s = float(preds[0]), float(preds[1]), float(preds[2])
        confidence = abs(pred_1s)
        direction = 1 if pred_1s > 0 else -1
        tier = None
        for tier_name, threshold in self.confidence_thresholds.items():
            if threshold is not None and confidence >= threshold:
                tier = tier_name
        return {
            "predictions": [pred_1s, pred_5s, pred_10s],
            "pred_1s": pred_1s,
            "pred_5s": pred_5s,
            "pred_10s": pred_10s,
            "confidence_1s": confidence,
            "direction": direction,
            "tier": tier,
        }

    def add_event(self, features: np.ndarray) -> Optional[Dict]:
        """Streaming-mode event ingestion. Returns prediction every `stride` events."""
        self.event_buffer.append(features)
        if len(self.event_buffer) >= self.window_size:
            self.prediction_count += 1
            if (self.prediction_count == 1
                    or (len(self.event_buffer) - self.window_size) % self.stride == 0):
                window = np.array(self.event_buffer[-self.window_size:])
                result = self.predict(window)
                result["event_count"] = len(self.event_buffer)
                result["prediction_number"] = self.prediction_count
                if len(self.event_buffer) > self.window_size * 2:
                    self.event_buffer = self.event_buffer[-self.window_size:]
                return result
        return None

    def calibrate_thresholds(self, predictions_path: str):
        """Calibrate Top5/1/0.5/0.1% thresholds from historical OOT predictions."""
        data = np.load(predictions_path, allow_pickle=True)
        # NPZ may store under 'predictions', 'pred_1s', or 'preds_1s'
        if "pred_1s" in data:
            pred_1s = data["pred_1s"]
        elif "preds_1s" in data:
            pred_1s = data["preds_1s"]
        elif "predictions" in data:
            preds = data["predictions"]
            pred_1s = preds[:, 0] if preds.ndim == 2 else preds
        else:
            raise ValueError(f"Cannot find 1s predictions in {predictions_path}. Keys: {list(data.keys())}")
        abs_p = np.abs(pred_1s)
        for tier, pct in [("Top5%", 95), ("Top1%", 99), ("Top0.5%", 99.5), ("Top0.1%", 99.9)]:
            self.confidence_thresholds[tier] = float(np.percentile(abs_p, pct))
        print(f"[v2_inference] thresholds calibrated: {self.confidence_thresholds}")

    def benchmark_speed(self, n_runs: int = 20) -> Dict[str, float]:
        dummy = np.random.randn(self.window_size, self.n_features).astype(np.float32)
        # Warmup
        for _ in range(2):
            self.predict(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            t0 = time.time()
            self.predict(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
        result = {
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "min_ms": float(min(times)),
            "max_ms": float(max(times)),
        }
        print(f"[v2_inference] benchmark: {result}")
        return result


if __name__ == "__main__":
    # Quick sanity: load + benchmark on Razer paths
    LVL3 = Path(os.environ.get("LVL3_ROOT", r"C:\Users\claude\Lvl3Quant"))
    weights = LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_10_best.pt"
    stats = LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_09_feature_stats.npz"
    eng = CNNMambaV2Inference(
        weights_path=str(weights),
        stats_path=str(stats),
        window_size=1000,
        stride=500,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    eng.benchmark_speed(n_runs=10)
    dummy = np.random.randn(1000, 25).astype(np.float32)
    print("Sample prediction:", eng.predict(dummy))
