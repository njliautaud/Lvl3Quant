#!/usr/bin/env python3
"""
v3_3_inference.py — CNN-Mamba v3.3 inference adapter for live paper trading.

DRAFT — not yet validated on Razer. Test before deploying.

This is the Jupiter STAGING copy of the adapter (HARNESS_EXTENSION_PLAN.md
Step 4). Once unit-tested on Jupiter and signed off, scp it to Razer's
live_trading/ directory and import it from paper_trading_*.py.

Architecture notes
------------------
v3.3 ("uncertainty-weighted" trainer in
alpha_discovery/deep_models/train_cnn_mamba_v3_3.py) reuses the v3.2 model
class `CNNMambaV32` UNCHANGED; only the loss differs (Kendall log_sigma).
So at inference time we instantiate `CNNMambaV32` and load the v3.3 ckpt
into it.

v3.3 is the EVENT-ONLY model — it has NO book-pyramid input, so it is the
correct "unblocked" channel to ship in the v3 harness if the live feed
cannot yet produce v3.4.2's book_pyramid tensor.

Input contract (from train_cnn_mamba_v3_2.CNNMambaV32.forward):
  batch = {
    "events_t1": (B, 1500, 39)   # event-paced features (N_T1_FEATURES = 39)
    "events_t2": (B, 1500, 14)   # 100ms order-flow features (N_T2_FEATURES = 14)
    "events_t3": (B, 1500, 25)   # 1Hz session-context features (N_T3_FEATURES = 25)
  }

LIVE FEATURE GAP (as of 2026-05-22):
The live builder live_trading/streaming_features_smart_v3.py emits only
the 25 raw smart_v3 event features. T1 expects 39 = 25 event + 4 PatchTST
preds + 10 book-history. T2 (14-feat per 100ms) and T3 (25-feat per 1Hz)
are not produced by the live streaming module. Until the live feature
pipeline is extended, the caller MUST either (a) provide the structured
dict with all three tiers populated, or (b) pass a flat (L, 25) array and
accept this adapter's zero-padding fallback for the missing 14 T1 columns
and all of T2/T3. The latter is a soft-degrade — predictions will run but
be calibrated for cold-start-like conditions only.

Output: dict with `pred_log_ret_{1s,5s,10s,30s}`, `conf`, `reason`, plus
diagnostic-only `diag_pred_log_ret_{60s,5min}` (HC #428 R2 forbids using
horizons > 30s for trade decisions).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

# Mirror v3.3 trainer env defaults BEFORE importing the trainer modules.
# Backbone hyperparams INFERRED from fold_00_intra_ckpt.pt tensor shapes
# (see staging/harness_v3/README or the smoke-test log). These override the
# trainer's default 96/32/3/16 — the v3.3 ckpt was trained with the larger
# 128/64/4/8 config.
os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
os.environ.setdefault("EVENT_WINDOW_SIZE", "1500")
os.environ.setdefault("EVENT_STRIDE", "250")
os.environ.setdefault("MAMBA_D_MODEL", "128")    # ckpt-inferred (was default 96)
os.environ.setdefault("MAMBA_D_STATE", "64")     # ckpt-inferred (was default 32)
os.environ.setdefault("MAMBA_N_LAYERS", "4")     # ckpt-inferred (was default 3)
os.environ.setdefault("MAMBA_DT_RANK", "8")      # ckpt-inferred (was default 16)

# Bootstrap the repo root onto sys.path so alpha_discovery.* is importable.
# Staging location: <repo>/staging/harness_v3/v3_3_inference.py  =>  parents[1] = <repo>
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import (  # noqa: E402
    CNNMambaV32,
    ALL_HEAD_NAMES,
    WINDOW_SIZE_T1, WINDOW_SIZE_T2, WINDOW_SIZE_T3,
    N_T1_FEATURES, N_T2_FEATURES, N_T3_FEATURES,
)


# Heads we trust per HC #428 R2 (trade horizon <= 30s).
# Longer horizons are returned for diagnostics under a `diag_` prefix.
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
        #        conf, reason,
        #        diag_pred_log_ret_60s, diag_pred_log_ret_5min}
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

        # The v3.3 ckpt was trained with PER-TIER dt_rank (T1/T2 dt_rank=8 from
        # d_model=128; T3 dt_rank=4 from d_model=64 — i.e., dt_rank = d_model//16).
        # The current trainer uses a single MAMBA_DT_RANK env var for all tiers,
        # so T3 will mismatch. Surgically resize T3's SSM projection layers to
        # match the ckpt before load_state_dict, otherwise load fails.
        self._fixup_per_tier_dt_rank(state)

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[v33_inference] WARN unexpected keys: {len(unexpected)}", flush=True)
        if missing:
            print(f"[v33_inference] missing keys defaulted: {len(missing)}", flush=True)
        self.model.to(self.device).eval()

        # Feature stats — v3.2/v3.3 dataset persists per-tier mean/std under keys
        # like "mean_t1", "std_t1", "mean_t2", "std_t2", "mean_t3", "std_t3".
        stats = np.load(stats_path, allow_pickle=True)
        self.stats: Dict[str, np.ndarray] = {k: stats[k] for k in stats.files}
        self._stat_keys = list(self.stats.keys())
        print(f"[v33_inference] feature_stats keys={self._stat_keys}", flush=True)

        print(f"[v33_inference] device={self.device} window_t1={self.window_t1} "
              f"n_t1={self.n_t1}  head_count={len(ALL_HEAD_NAMES)}", flush=True)

    # ---------- shape coercion ----------

    def _coerce_to_batch(self, features_window) -> Optional[Dict[str, torch.Tensor]]:
        """Accept a (L, F1) array OR a dict with t1/t2/t3 stacks.

        Returns a torch batch dict ready for the model, or None if the shape
        is unrecognized.
        """
        if isinstance(features_window, dict):
            try:
                e1 = np.asarray(features_window["events_t1"], dtype=np.float32)
                e2 = np.asarray(features_window.get(
                    "events_t2",
                    np.zeros((self.window_t2, self.n_t2), dtype=np.float32),
                ), dtype=np.float32)
                e3 = np.asarray(features_window.get(
                    "events_t3",
                    np.zeros((self.window_t3, self.n_t3), dtype=np.float32),
                ), dtype=np.float32)
            except Exception:
                return None
        elif isinstance(features_window, np.ndarray):
            if features_window.ndim != 2:
                return None
            L, F = features_window.shape
            if F == self.n_t1:
                src = features_window.astype(np.float32)
            elif F == 25:
                # Live builder emits only 25 raw event features. Pad the
                # remaining 14 T1 columns (4 PatchTST + 10 book-history) with
                # zeros. This is a soft-degrade fallback documented in the
                # module header; predictions will be calibrated for
                # cold-start-like conditions only.
                src = np.zeros((L, self.n_t1), dtype=np.float32)
                src[:, :25] = features_window.astype(np.float32)
            else:
                return None
            # Pad/truncate L to WINDOW_SIZE_T1
            if L >= self.window_t1:
                e1 = src[-self.window_t1:]
            else:
                e1 = np.zeros((self.window_t1, self.n_t1), dtype=np.float32)
                e1[-L:] = src
            # t2/t3 unavailable from a flat (L,F1) array — zero them.
            e2 = np.zeros((self.window_t2, self.n_t2), dtype=np.float32)
            e3 = np.zeros((self.window_t3, self.n_t3), dtype=np.float32)
        else:
            return None

        # Normalize where stats are available.
        e1 = self._zscore(e1, "t1")
        e2 = self._zscore(e2, "t2")
        e3 = self._zscore(e3, "t3")

        batch = {
            "events_t1": torch.tensor(e1, dtype=torch.float32, device=self.device).unsqueeze(0),
            "events_t2": torch.tensor(e2, dtype=torch.float32, device=self.device).unsqueeze(0),
            "events_t3": torch.tensor(e3, dtype=torch.float32, device=self.device).unsqueeze(0),
        }
        return batch

    def _fixup_per_tier_dt_rank(self, state: Dict[str, torch.Tensor]) -> None:
        """Resize the SSM projection layers in each backbone tier to match the
        ckpt's per-tier dt_rank. The v3.3 ckpt used d_model//16 per tier
        (T1/T2=8, T3=4) but the current trainer code uses a single global value.

        We infer the ckpt's per-tier dt_rank from the `dt_proj.weight` shape
        (= (d_inner, dt_rank)) and rebuild the affected layers in-place. No
        ckpt tensors are modified — only the model's layer dimensions.
        """
        import torch.nn as nn
        for tier in ("t1_backbone", "t2_backbone", "t3_backbone"):
            ckpt_key = f"{tier}.blocks.0.ssm.dt_proj.weight"
            if ckpt_key not in state:
                continue
            ckpt_dt_rank = int(state[ckpt_key].shape[1])
            backbone = getattr(self.model, tier, None)
            if backbone is None:
                continue
            for blk in backbone.blocks:
                ssm = blk.ssm
                if ssm.dt_rank == ckpt_dt_rank:
                    continue
                d_inner = ssm.d_inner
                d_state = ssm.d_state
                # Rebuild x_proj: (d_inner) -> (dt_rank + 2*d_state)
                ssm.x_proj = nn.Linear(d_inner, ckpt_dt_rank + 2 * d_state, bias=False)
                # Rebuild dt_proj: (dt_rank) -> (d_inner)
                ssm.dt_proj = nn.Linear(ckpt_dt_rank, d_inner, bias=True)
                ssm.dt_rank = ckpt_dt_rank
            print(f"[v33_inference] {tier}: rebuilt SSM with dt_rank={ckpt_dt_rank}", flush=True)

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

        return {
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

    # ---------- batched inference (Jupiter shadow processor) ----------

    @torch.no_grad()
    def predict_batch(self, feature_windows):
        """Batched forward over a list of (1500, F) flat ndarrays.

        Builds a (B, 1500, 39) tensor + zero t2/t3 stacks, runs one Mamba
        forward, returns a list of result dicts (same keys as predict()).
        Soft-degrade only — same caveats as predict() for 25-feat input.
        """
        if not feature_windows:
            return []

        B = len(feature_windows)
        e1 = np.zeros((B, self.window_t1, self.n_t1), dtype=np.float32)
        for i, w in enumerate(feature_windows):
            if not isinstance(w, np.ndarray) or w.ndim != 2:
                # skip-bad: leave row zero, mark via NaN later
                continue
            L, F = w.shape
            if F == self.n_t1:
                src = w.astype(np.float32)
            elif F == 25:
                src = np.zeros((L, self.n_t1), dtype=np.float32)
                src[:, :25] = w.astype(np.float32)
            else:
                continue
            if L >= self.window_t1:
                e1[i] = src[-self.window_t1:]
            else:
                e1[i, -L:] = src

        e2 = np.zeros((B, self.window_t2, self.n_t2), dtype=np.float32)
        e3 = np.zeros((B, self.window_t3, self.n_t3), dtype=np.float32)

        # Normalize each tier (broadcasting (F,) stats over (B, L, F))
        if "mean_t1" in self.stats and "std_t1" in self.stats:
            m = self.stats["mean_t1"].astype(np.float32)
            s = np.maximum(self.stats["std_t1"].astype(np.float32), 1e-8)
            e1 = (e1 - m) / s
        # t2/t3 stats may not exist in this ckpt; leave zeros un-normalized
        if "mean_t2" in self.stats and "std_t2" in self.stats:
            m = self.stats["mean_t2"].astype(np.float32)
            s = np.maximum(self.stats["std_t2"].astype(np.float32), 1e-8)
            e2 = (e2 - m) / s
        if "mean_t3" in self.stats and "std_t3" in self.stats:
            m = self.stats["mean_t3"].astype(np.float32)
            s = np.maximum(self.stats["std_t3"].astype(np.float32), 1e-8)
            e3 = (e3 - m) / s

        batch = {
            "events_t1": torch.tensor(e1, dtype=torch.float32, device=self.device),
            "events_t2": torch.tensor(e2, dtype=torch.float32, device=self.device),
            "events_t3": torch.tensor(e3, dtype=torch.float32, device=self.device),
        }

        try:
            out = self.model(batch)
        except Exception as e:
            return [self._nan_result(f"forward_failed: {type(e).__name__}") for _ in range(B)]

        def _head_arr(name: str) -> np.ndarray:
            t = out.get(name)
            if t is None:
                return np.full((B,), np.nan, dtype=np.float32)
            return t.detach().cpu().numpy().reshape(-1)

        a1 = _head_arr("log_ret_1s")
        a5 = _head_arr("log_ret_5s")
        a10 = _head_arr("log_ret_10s")
        a30 = _head_arr("log_ret_30s")
        a60 = _head_arr("log_ret_60s")
        a5m = _head_arr("log_ret_5min")

        results = []
        for i in range(B):
            v1 = float(a1[i])
            results.append({
                "pred_log_ret_1s": v1,
                "pred_log_ret_5s": float(a5[i]),
                "pred_log_ret_10s": float(a10[i]),
                "pred_log_ret_30s": float(a30[i]),
                "conf": float(abs(v1)) if v1 == v1 else float("nan"),
                "reason": "ok",
                "diag_pred_log_ret_60s": float(a60[i]),
                "diag_pred_log_ret_5min": float(a5m[i]),
            })
        return results


if __name__ == "__main__":
    # Smoke test (no weights load — just import sanity)
    print("v3_3_inference module loaded.")
    print(f"  WINDOW_SIZE_T1={WINDOW_SIZE_T1}, N_T1_FEATURES={N_T1_FEATURES}")
    print(f"  WINDOW_SIZE_T2={WINDOW_SIZE_T2}, N_T2_FEATURES={N_T2_FEATURES}")
    print(f"  WINDOW_SIZE_T3={WINDOW_SIZE_T3}, N_T3_FEATURES={N_T3_FEATURES}")
    print(f"  ALL_HEAD_NAMES count={len(ALL_HEAD_NAMES)}")
    print(f"  VALID_TRADE_HEADS={VALID_TRADE_HEADS}")
