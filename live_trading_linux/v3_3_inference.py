#!/usr/bin/env python3
"""
v3_3_inference.py — CNN-Mamba v3.3 inference adapter for live paper trading.

DRAFT (Friday harness extension, Step 4). Not yet wired into paper_trading.

Architecture notes
------------------
v3.3 is the "uncertainty-weighted" trainer in
  alpha_discovery/deep_models/train_cnn_mamba_v3_3.py
but its MODEL CLASS is reused unchanged from v3.2: `CNNMambaV32`. v3.3 only
swaps the loss (Kendall log_sigma). So at inference time we instantiate
`CNNMambaV32` and load the v3.3 checkpoint into it.

Input contract (from train_cnn_mamba_v3_2.CNNMambaV32.forward):
  batch = {
    "events_t1": (B, 1500, 39)   # event-paced features (N_T1_FEATURES = 39)
    "events_t2": (B, 1500, 14)   # 100ms order-flow features
    "events_t3": (B, 1500, 25)   # 1Hz session-context features
  }

Output: dict[head_name -> (B,)] over ALL_HEAD_NAMES (~28 heads).

For LIVE use, the caller must provide a `features_window` matching this
shape. The MBO recorder on Razer publishes event-paced T1 features for the
live stream — T2 and T3 are derived per-second from the same feed. If the
live feature stream does NOT yet expose t2/t3 stacks, the caller should pad
zeros and rely on the stats-normalization (mean/std) to dampen the missing
context; that is structurally similar to what training already saw on cold
session starts. We surface a `reason` field when the caller passes a single
(T, F) array we cannot disambiguate.

HC #477 valid horizons
----------------------
The v3.3 model trained heads at 1s / 5s / 10s / 30s / 60s / 5min. v3.3
predates HC #477 — its 60s+ heads are NOT structurally zero-labeled the way
v3.4.2's were, so they ARE technically valid for v3.3. However, the live
harness MUST treat 60s/5min as low-priority because the trade horizon under
HC #428 R2 is <=30s. This adapter therefore EXPOSES `pred_log_ret_60s` and
`pred_log_ret_5min` for diagnostics only (consumers should ignore them for
trade decisions).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

# Mirror v3.3 trainer env defaults BEFORE importing the trainer modules
os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
os.environ.setdefault("EVENT_WINDOW_SIZE", "1500")
os.environ.setdefault("EVENT_STRIDE", "250")

# Bootstrap the repo root onto sys.path so alpha_discovery.* is importable
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent  # /home/.../Lvl3Quant
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import (  # noqa: E402
    CNNMambaV32,
    ALL_HEAD_NAMES,
    WINDOW_SIZE_T1, WINDOW_SIZE_T2, WINDOW_SIZE_T3,
    N_T1_FEATURES, N_T2_FEATURES, N_T3_FEATURES,
)


# Heads we trust per HC #477 / HC #428 R2 (trade horizon <= 30s).
# Longer horizons are returned for diagnostics under a separate suffix.
VALID_TRADE_HEADS = ("log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s")
DIAGNOSTIC_LONG_HEADS = ("log_ret_60s", "log_ret_5min")


class V33Inference:
    """CNN-Mamba v3.3 single-trunk inference wrapper.

    Usage:
        eng = V33Inference(
            weights_path=".../cnn_mamba_v3_3_uncertainty_weighted/fold_00_intra_ckpt.pt",
            stats_path=".../cnn_mamba_v3_3_uncertainty_weighted/fold_00_feature_stats.npz",
            device="cuda",
        )
        out = eng.predict(features_window)
        # out = {pred_log_ret_1s, pred_log_ret_5s, pred_log_ret_10s, pred_log_ret_30s,
        #        pred_log_ret_60s, pred_log_ret_5min, conf, reason}
    """

    def __init__(
        self,
        weights_path: str,
        stats_path: str,
        device: str = "cpu",
    ):
        self.window_t1 = WINDOW_SIZE_T1
        self.window_t2 = WINDOW_SIZE_T2
        self.window_t3 = WINDOW_SIZE_T3
        self.n_t1 = N_T1_FEATURES
        self.n_t2 = N_T2_FEATURES
        self.n_t3 = N_T3_FEATURES

        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )

        # Load ckpt
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))

        # CNNMambaV32 defaults match the v3.3 trainer constants — no kwargs needed.
        self.model = CNNMambaV32()
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[v33_inference] WARN unexpected keys: {len(unexpected)}", flush=True)
        if missing:
            print(f"[v33_inference] missing keys defaulted: {len(missing)}", flush=True)
        self.model.to(self.device).eval()

        # Feature stats — v3.2/v3.3 dataset persists per-tier mean/std under keys
        # like "mean_t1", "std_t1", "mean_t2", "std_t2", "mean_t3", "std_t3". Older
        # ckpts may only have mean_t1/std_t1 (t2/t3 normalized inside the dataset).
        stats = np.load(stats_path, allow_pickle=True)
        self.stats: Dict[str, np.ndarray] = {k: stats[k] for k in stats.files}
        self._stat_keys = list(self.stats.keys())
        print(f"[v33_inference] feature_stats keys={self._stat_keys}", flush=True)

        print(f"[v33_inference] device={self.device} window_t1={self.window_t1} "
              f"n_t1={self.n_t1}  head_count={len(ALL_HEAD_NAMES)}", flush=True)

    # ---------- shape detection ----------

    def _coerce_to_batch(self, features_window: np.ndarray) -> Optional[Dict[str, torch.Tensor]]:
        """Accept a (L, F1) array OR a dict-like with t1/t2/t3 stacks.

        Returns a torch batch dict ready for the model, or None if the shape
        is unrecognized (caller must NaN-out the result).
        """
        # Dict case — caller passed pre-split tensors.
        if isinstance(features_window, dict):
            try:
                e1 = np.asarray(features_window["events_t1"], dtype=np.float32)
                e2 = np.asarray(features_window.get("events_t2",
                                np.zeros((self.window_t2, self.n_t2), dtype=np.float32)),
                                dtype=np.float32)
                e3 = np.asarray(features_window.get("events_t3",
                                np.zeros((self.window_t3, self.n_t3), dtype=np.float32)),
                                dtype=np.float32)
            except Exception:
                return None
        elif isinstance(features_window, np.ndarray):
            if features_window.ndim != 2:
                return None
            L, F = features_window.shape
            if F != self.n_t1:
                # Caller passed an unexpected feature count — refuse.
                return None
            # Pad/truncate L to WINDOW_SIZE_T1
            if L >= self.window_t1:
                e1 = features_window[-self.window_t1:].astype(np.float32)
            else:
                e1 = np.zeros((self.window_t1, self.n_t1), dtype=np.float32)
                e1[-L:] = features_window.astype(np.float32)
            # t2/t3 unavailable from a flat (L,F1) array — zero them.
            e2 = np.zeros((self.window_t2, self.n_t2), dtype=np.float32)
            e3 = np.zeros((self.window_t3, self.n_t3), dtype=np.float32)
        else:
            return None

        # Normalize (z-score) where stats are available. The training-time
        # dataset already z-scores so means~0/stds~1 if SKIP_NORMALIZE was on.
        e1 = self._zscore(e1, "t1")
        e2 = self._zscore(e2, "t2")
        e3 = self._zscore(e3, "t3")

        batch = {
            "events_t1": torch.tensor(e1, dtype=torch.float32, device=self.device).unsqueeze(0),
            "events_t2": torch.tensor(e2, dtype=torch.float32, device=self.device).unsqueeze(0),
            "events_t3": torch.tensor(e3, dtype=torch.float32, device=self.device).unsqueeze(0),
        }
        return batch

    def _zscore(self, x: np.ndarray, tier: str) -> np.ndarray:
        mean_key, std_key = f"mean_{tier}", f"std_{tier}"
        if mean_key in self.stats and std_key in self.stats:
            mean = self.stats[mean_key].astype(np.float32)
            std = np.maximum(self.stats[std_key].astype(np.float32), 1e-8)
            # Broadcast over time axis
            return (x - mean) / std
        return x

    # ---------- inference ----------

    @torch.no_grad()
    def predict(self, features_window) -> Dict[str, float]:
        """Returns dict with the 4 valid trade-horizon log-return heads + conf.

        Long-horizon heads (60s, 5min) are exposed for diagnostics — DO NOT use
        them for trade decisions per HC #428 R2 (trade horizon <=30s).
        """
        batch = self._coerce_to_batch(features_window)
        if batch is None:
            return self._nan_result("input_shape_mismatch")

        try:
            out = self.model(batch)  # dict[head_name -> (1,)]
        except Exception as e:
            return self._nan_result(f"forward_failed: {type(e).__name__}")

        def _h(name: str) -> float:
            t = out.get(name)
            if t is None:
                return float("nan")
            v = t.detach().cpu().numpy()
            return float(v.reshape(-1)[0])

        pred_1s = _h("log_ret_1s")
        pred_5s = _h("log_ret_5s")
        pred_10s = _h("log_ret_10s")
        pred_30s = _h("log_ret_30s")
        # Confidence = abs(1s) — same convention as cnn_mamba_v2_inference.py
        conf = float(abs(pred_1s)) if pred_1s == pred_1s else float("nan")

        result = {
            "pred_log_ret_1s": pred_1s,
            "pred_log_ret_5s": pred_5s,
            "pred_log_ret_10s": pred_10s,
            "pred_log_ret_30s": pred_30s,
            "conf": conf,
            "reason": "ok",
            # Diagnostic-only (HC #428 R2: NOT for trade decisions)
            "diag_pred_log_ret_60s": _h("log_ret_60s"),
            "diag_pred_log_ret_5min": _h("log_ret_5min"),
        }
        return result

    def _nan_result(self, reason: str) -> Dict[str, float]:
        nan = float("nan")
        return {
            "pred_log_ret_1s": nan,
            "pred_log_ret_5s": nan,
            "pred_log_ret_10s": nan,
            "pred_log_ret_30s": nan,
            "conf": nan,
            "reason": reason,
            "diag_pred_log_ret_60s": nan,
            "diag_pred_log_ret_5min": nan,
        }


if __name__ == "__main__":
    # Smoke test (no weights load — just import sanity)
    print("v3_3_inference module loaded.")
    print(f"  WINDOW_SIZE_T1={WINDOW_SIZE_T1}, N_T1_FEATURES={N_T1_FEATURES}")
    print(f"  WINDOW_SIZE_T2={WINDOW_SIZE_T2}, N_T2_FEATURES={N_T2_FEATURES}")
    print(f"  WINDOW_SIZE_T3={WINDOW_SIZE_T3}, N_T3_FEATURES={N_T3_FEATURES}")
    print(f"  ALL_HEAD_NAMES count={len(ALL_HEAD_NAMES)}")
