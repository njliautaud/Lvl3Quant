#!/usr/bin/env python3
"""
v3_4_2_inference.py — CNN-Mamba v3.4.2 inference adapter for live paper trading.

DRAFT (Friday harness extension, Step 4). Not yet wired into paper_trading.

Architecture notes
------------------
v3.4.2 = v3.3 trunk + gated residual book CNN. Model class is
`CNNMambaV341BookResidual`, defined inline in
  scripts/v3_4_research/dispatch_v34_2_fixedmtl.py
(no __init__.py in that dir — we import by file path with importlib, mirroring
v342_run_oot_inference.py lines ~64-73).

Input contract (from CNNMambaV341BookResidual.forward):
  batch = {
    "events_t1": (B, 1500, 39),
    "events_t2": (B, 1500, 14),
    "events_t3": (B, 1500, 25),
    "book_pyramid": (B, N_BOOK_LEVELS, N_BOOK_FEATURES_PER_LEVEL, T_book)   # book CNN input
  }

HC #477 — UNUSABLE HEADS
------------------------
HC #477 (discovered 2026-05-21) established that v3.4.2's 60s+ heads
(`log_ret_60s`, `log_ret_5min`, and all derived long-horizon p_up / quantile /
MFE-MAE heads keyed to those horizons) were trained on structurally-zero
labels — the long-horizon targets in the train set were near-uniformly zero,
so the model learned to predict zero. These heads are NOT VALID and this
adapter DOES NOT expose them. Only the short-horizon heads are returned:
  pred_log_ret_1s, pred_log_ret_5s, pred_log_ret_10s, pred_log_ret_30s.

Step-4 fallback (per HARNESS_EXTENSION_PLAN.md)
-----------------------------------------------
v3.4.2 requires book-pyramid features. If the live stream does NOT publish
book features yet, `predict()` returns all-NaN with `reason="book_features_missing"`.
The live harness must then route signals from v3.3 (or v2) instead.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

# Mirror dispatch env defaults BEFORE importing trainer modules
os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
os.environ.setdefault("EVENT_WINDOW_SIZE", "1500")
os.environ.setdefault("EVENT_STRIDE", "250")

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent  # /home/.../Lvl3Quant
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import (  # noqa: E402
    WINDOW_SIZE_T1, WINDOW_SIZE_T2, WINDOW_SIZE_T3,
    N_T1_FEATURES, N_T2_FEATURES, N_T3_FEATURES,
)
from alpha_discovery.deep_models.train_cnn_mamba_v3_4 import (  # noqa: E402
    N_BOOK_LEVELS,
    N_BOOK_FEATURES_PER_LEVEL,
)


def _import_dispatch_module():
    """Import dispatch_v34_2_fixedmtl by file path (matches v342_run_oot_inference)."""
    dispatch_path = (
        _REPO_ROOT / "scripts" / "v3_4_research" / "dispatch_v34_2_fixedmtl.py"
    )
    if not dispatch_path.exists():
        raise FileNotFoundError(f"dispatch_v34_2_fixedmtl.py not found at {dispatch_path}")
    spec = importlib.util.spec_from_file_location(
        "dispatch_v34_2_fixedmtl", str(dispatch_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dispatch_v34_2_fixedmtl"] = mod
    spec.loader.exec_module(mod)
    return mod


# Heads exposed by this adapter (HC #477: short-horizon only).
VALID_TRADE_HEADS = ("log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s")


class V342Inference:
    """CNN-Mamba v3.4.2 dual-trunk (event + book) inference wrapper.

    Usage:
        eng = V342Inference(
            weights_path=".../cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.pt",
            stats_path=".../cnn_mamba_v3_4_2_fixedmtl/fold_00_feature_stats.npz",
            device="cuda",
        )
        out = eng.predict(features_window_or_dict)
        # If book features absent in input:
        #   out = {pred_log_ret_*: nan, conf: nan, reason: "book_features_missing"}
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
        self.n_book_levels = N_BOOK_LEVELS
        self.n_book_feats_per_level = N_BOOK_FEATURES_PER_LEVEL

        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )

        dispatch_mod = _import_dispatch_module()
        CNNMambaV341BookResidual = dispatch_mod.CNNMambaV341BookResidual

        self.model = CNNMambaV341BookResidual()
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[v342_inference] WARN unexpected keys: {len(unexpected)}", flush=True)
        if missing:
            print(f"[v342_inference] missing keys defaulted: {len(missing)}", flush=True)
        self.model.to(self.device).eval()

        stats = np.load(stats_path, allow_pickle=True)
        self.stats: Dict[str, np.ndarray] = {k: stats[k] for k in stats.files}
        self._stat_keys = list(self.stats.keys())
        print(f"[v342_inference] feature_stats keys={self._stat_keys}", flush=True)

        # book_gate sanity print
        if "book_gate" in state:
            try:
                bg_raw = float(state["book_gate"].reshape(-1)[0])
                print(f"[v342_inference] book_gate raw={bg_raw:.6f} "
                      f"tanh={np.tanh(bg_raw):.6f}", flush=True)
            except Exception:
                pass

        print(f"[v342_inference] device={self.device} valid_heads={VALID_TRADE_HEADS} "
              f"(HC #477: 60s/5min heads NOT exposed)", flush=True)

    # ---------- shape detection ----------

    def _has_book(self, features_window) -> bool:
        if isinstance(features_window, dict):
            bp = features_window.get("book_pyramid")
            if bp is None:
                return False
            try:
                arr = np.asarray(bp)
                # Expected (n_levels, n_feats_per_level, T) or (B, ...) shape
                return arr.ndim >= 3 and arr.size > 0
            except Exception:
                return False
        return False

    def _coerce_to_batch(self, features_window) -> Optional[Dict[str, torch.Tensor]]:
        """Build the model input dict. Returns None if event features are unusable."""
        if isinstance(features_window, dict):
            try:
                e1 = np.asarray(features_window["events_t1"], dtype=np.float32)
                e2 = np.asarray(features_window.get("events_t2",
                                np.zeros((self.window_t2, self.n_t2), dtype=np.float32)),
                                dtype=np.float32)
                e3 = np.asarray(features_window.get("events_t3",
                                np.zeros((self.window_t3, self.n_t3), dtype=np.float32)),
                                dtype=np.float32)
                bp = np.asarray(features_window["book_pyramid"], dtype=np.float32)
            except Exception:
                return None
        else:
            return None  # v3.4.2 requires structured dict input; flat (L,F) is insufficient

        # Normalize event tiers (z-score where stats are present)
        e1 = self._zscore(e1, "t1")
        e2 = self._zscore(e2, "t2")
        e3 = self._zscore(e3, "t3")

        # Pad batch dim if caller passed a single sample (3D for book, 2D for events)
        if e1.ndim == 2:
            e1 = e1[None]
        if e2.ndim == 2:
            e2 = e2[None]
        if e3.ndim == 2:
            e3 = e3[None]
        if bp.ndim == 3:
            bp = bp[None]

        batch = {
            "events_t1": torch.tensor(e1, dtype=torch.float32, device=self.device),
            "events_t2": torch.tensor(e2, dtype=torch.float32, device=self.device),
            "events_t3": torch.tensor(e3, dtype=torch.float32, device=self.device),
            "book_pyramid": torch.tensor(bp, dtype=torch.float32, device=self.device),
        }
        return batch

    def _zscore(self, x: np.ndarray, tier: str) -> np.ndarray:
        mean_key, std_key = f"mean_{tier}", f"std_{tier}"
        if mean_key in self.stats and std_key in self.stats:
            mean = self.stats[mean_key].astype(np.float32)
            std = np.maximum(self.stats[std_key].astype(np.float32), 1e-8)
            return (x - mean) / std
        return x

    # ---------- inference ----------

    @torch.no_grad()
    def predict(self, features_window) -> Dict[str, float]:
        """Returns dict with the 4 valid trade-horizon heads + conf + reason.

        Per HC #477, 60s/5min heads are NOT returned (structurally-zero labels).
        If `features_window` lacks a usable `book_pyramid`, returns all-NaN
        with reason='book_features_missing' per HARNESS_EXTENSION_PLAN Step 4.
        """
        if not self._has_book(features_window):
            return self._nan_result("book_features_missing")

        batch = self._coerce_to_batch(features_window)
        if batch is None:
            return self._nan_result("input_shape_mismatch")

        try:
            out = self.model(batch)  # dict[head_name -> (B,)]
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
        conf = float(abs(pred_1s)) if pred_1s == pred_1s else float("nan")

        return {
            "pred_log_ret_1s": pred_1s,
            "pred_log_ret_5s": pred_5s,
            "pred_log_ret_10s": pred_10s,
            "pred_log_ret_30s": pred_30s,
            "conf": conf,
            "reason": "ok",
            # NOTE: 60s/5min heads intentionally OMITTED per HC #477.
        }

    def _nan_result(self, reason: str) -> Dict[str, float]:
        nan = float("nan")
        return {
            "pred_log_ret_1s": nan,
            "pred_log_ret_5s": nan,
            "pred_log_ret_10s": nan,
            "pred_log_ret_30s": nan,
            "conf": nan,
            "reason": reason,
        }


if __name__ == "__main__":
    # Smoke test (no weights — module import sanity only)
    print("v3_4_2_inference module loaded.")
    print(f"  VALID_TRADE_HEADS={VALID_TRADE_HEADS} (HC #477 exclusion enforced)")
    print(f"  Book pyramid: levels={N_BOOK_LEVELS} feats/level={N_BOOK_FEATURES_PER_LEVEL}")
