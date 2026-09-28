"""
v3.3 RL smart-execution dataset loader (HC #396 #2).

Loads the v3.3 60d champion fold_00 predictions NPZ and exposes a flat per-sample
state matrix consisting of:

    [32 head outputs] + [book context features] + [position-context placeholders]

Position context is filled at env-step time (not here). This module produces the
"static" state matrix (head + book context), the realized-return targets used by
the env as reward signal, the OOT day-boundary array (no cross-day episodes),
and the feature mean/std cache.

Constants binding (HC #396 / HC #392 / CLAUDE.md COST CONSTANTS):
  - target_log_ret_1s in this NPZ is already in TICKS (see full_market_replay.py
    note line ~110). We use it as per-second realized 1-tick-scaled return.
  - 250 ms prediction stride (HC #321).
  - Commission only: 0.376 ticks RT.  No extra spread cost (HC #392).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---- 32 v3.3 head names (canonical order) ---------------------------------
# Source: /home/jupiter/Lvl3Quant/output/v3_3_production_readiness_20260516/head_catalog.csv
HEAD_NAMES = [
    "log_ret_1s",
    "log_ret_5s",
    "log_ret_10s",
    "log_ret_30s",
    "log_ret_60s",
    "log_ret_5min",
    "log_ret_10s_q10",
    "log_ret_10s_q50",
    "log_ret_10s_q90",
    "log_ret_30s_q10",
    "log_ret_30s_q50",
    "log_ret_30s_q90",
    "log_ret_60s_q10",
    "log_ret_60s_q50",
    "log_ret_60s_q90",
    "fifo_tp4sl3_net",
    "fifo_tp8sl5_net",
    "fifo_tp4sl3_hit_tp",
    "fifo_tp8sl5_hit_tp",
    "p_up_5s",
    "p_up_10s",
    "p_up_30s",
    "p_up_60s",
    "p_reversal_15s",
    "p_reversal_30s",
    "p_reversal_60s",
    "mfe_30s_ticks",
    "mae_30s_ticks",
    "mfe_60s_ticks",
    "mae_60s_ticks",
    "realized_vol_30s_ticks",
    "time_to_mfe_secs",
]
assert len(HEAD_NAMES) == 32

# NPZ keys for pred and target use these prefixes; some are "pred_pred_*" due to
# how the trainer named them (mfe/mae/vol/time_to_mfe). Build pred-key map:
PRED_KEY_FOR = {
    "log_ret_1s":               "pred_log_ret_1s",
    "log_ret_5s":               "pred_log_ret_5s",
    "log_ret_10s":              "pred_log_ret_10s",
    "log_ret_30s":              "pred_log_ret_30s",
    "log_ret_60s":              "pred_log_ret_60s",
    "log_ret_5min":             "pred_log_ret_5min",
    "log_ret_10s_q10":          "pred_log_ret_10s_q10",
    "log_ret_10s_q50":          "pred_log_ret_10s_q50",
    "log_ret_10s_q90":          "pred_log_ret_10s_q90",
    "log_ret_30s_q10":          "pred_log_ret_30s_q10",
    "log_ret_30s_q50":          "pred_log_ret_30s_q50",
    "log_ret_30s_q90":          "pred_log_ret_30s_q90",
    "log_ret_60s_q10":          "pred_log_ret_60s_q10",
    "log_ret_60s_q50":          "pred_log_ret_60s_q50",
    "log_ret_60s_q90":          "pred_log_ret_60s_q90",
    "fifo_tp4sl3_net":          "pred_fifo_tp4sl3_net",
    "fifo_tp8sl5_net":          "pred_fifo_tp8sl5_net",
    "fifo_tp4sl3_hit_tp":       "pred_fifo_tp4sl3_hit_tp",
    "fifo_tp8sl5_hit_tp":       "pred_fifo_tp8sl5_hit_tp",
    "p_up_5s":                  "pred_p_up_5s",
    "p_up_10s":                 "pred_p_up_10s",
    "p_up_30s":                 "pred_p_up_30s",
    "p_up_60s":                 "pred_p_up_60s",
    "p_reversal_15s":           "pred_p_reversal_15s",
    "p_reversal_30s":           "pred_p_reversal_30s",
    "p_reversal_60s":           "pred_p_reversal_60s",
    "mfe_30s_ticks":            "pred_pred_mfe_30s_ticks",
    "mae_30s_ticks":            "pred_pred_mae_30s_ticks",
    "mfe_60s_ticks":            "pred_pred_mfe_60s_ticks",
    "mae_60s_ticks":            "pred_pred_mae_60s_ticks",
    "realized_vol_30s_ticks":   "pred_pred_realized_vol_30s_ticks",
    "time_to_mfe_secs":         "pred_pred_time_to_mfe_secs",
}
TARGET_KEY_FOR = {n: PRED_KEY_FOR[n].replace("pred_", "target_", 1) for n in HEAD_NAMES}

# Stride and physical constants
STRIDE_SEC = 0.25
STEPS_PER_SEC = 4
SAMPLES_PER_RTH_DAY = 6 * 3600 * STEPS_PER_SEC + 30 * 60 * STEPS_PER_SEC  # 6.5h = 93600

# Book-context feature names (appended after the 32 head outputs)
BOOK_CONTEXT_NAMES = [
    "log_ret_1s_realized_lag",   # last realized 1s return (proxy for very-short-term book context)
    "spread_ticks",              # =1 in ES RTH (constant placeholder)
    "realized_vol_30s_pred",     # use model's vol prediction as cheap regime tag
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
]

# Position-context placeholders (filled at env step time)
POS_CONTEXT_NAMES = [
    "in_position",
    "position_age_steps",
    "position_pnl_ticks",
    "time_to_cancel_steps",
    "pending_order_side",        # -1=ask, 0=none, +1=bid
    "pending_order_age_steps",
]

N_HEADS = len(HEAD_NAMES)
N_BOOK = len(BOOK_CONTEXT_NAMES)
N_POS  = len(POS_CONTEXT_NAMES)
STATE_DIM = N_HEADS + N_BOOK + N_POS  # 32 + 7 + 6 = 45


@dataclass
class V33RLDataset:
    """Container for the static (non-position) portion of the state plus rewards."""
    head_matrix: np.ndarray         # (N, 32) float32 — clipped to finite, NaN→0
    book_matrix: np.ndarray         # (N, 7)  float32
    realized_log_ret_1s: np.ndarray # (N,) float32 — already in TICKS (see HC note)
    target_fifo_tp4sl3_net: np.ndarray  # (N,) float32 — for ref/diagnostics
    target_fifo_tp8sl5_net: np.ndarray  # (N,) float32 — for ref/diagnostics
    day_idx: np.ndarray             # (N,) int32 — 0..D-1; episodes never cross
    oot_dates: list[str]
    feat_mean: np.ndarray           # (39,) float32 — head+book mean (no pos)
    feat_std: np.ndarray            # (39,) float32 — head+book std  (no pos)

    @property
    def n_samples(self) -> int:
        return int(self.head_matrix.shape[0])


def _sanitize(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with 0, clip extreme tails (>20 std) — robust to bad heads."""
    a = np.asarray(arr, dtype=np.float32).copy()
    bad = ~np.isfinite(a)
    if bad.any():
        a[bad] = 0.0
    # Hard clip to keep PPO advantage estimates stable
    a = np.clip(a, -50.0, 50.0)
    return a


def build_dataset(npz_path: str | Path) -> V33RLDataset:
    """Load fold_00 predictions and assemble the static state matrix + rewards."""
    npz_path = Path(npz_path)
    d = np.load(npz_path, allow_pickle=True)
    n = int(d["n_samples"]) if "n_samples" in d.files else int(d[d.files[0]].shape[0])
    oot_dates = [str(x) for x in d["oot_dates"]] if "oot_dates" in d.files else []
    n_days = max(1, len(oot_dates))

    # ---- 32 head outputs -------------------------------------------------
    heads = np.zeros((n, N_HEADS), dtype=np.float32)
    for i, name in enumerate(HEAD_NAMES):
        key = PRED_KEY_FOR[name]
        if key in d.files:
            heads[:, i] = _sanitize(d[key][:n])
        # else leaves zeros (UNTRAINED head absent — acceptable)

    # ---- realized rewards ------------------------------------------------
    realized_1s = _sanitize(d["target_log_ret_1s"][:n]) if "target_log_ret_1s" in d.files else np.zeros(n, dtype=np.float32)
    fifo43 = _sanitize(d["target_fifo_tp4sl3_net"][:n]) if "target_fifo_tp4sl3_net" in d.files else np.zeros(n, dtype=np.float32)
    fifo85 = _sanitize(d["target_fifo_tp8sl5_net"][:n]) if "target_fifo_tp8sl5_net" in d.files else np.zeros(n, dtype=np.float32)

    # ---- day boundaries --------------------------------------------------
    # Assume samples are concatenated day-by-day in OOT order, equal split.
    # n_per_day is approximate; we split via np.array_split for robustness.
    day_idx = np.zeros(n, dtype=np.int32)
    edges = np.linspace(0, n, n_days + 1, dtype=np.int64)
    for di in range(n_days):
        day_idx[edges[di]:edges[di+1]] = di

    # ---- book-context features -------------------------------------------
    book = np.zeros((n, N_BOOK), dtype=np.float32)
    # Use lag-1 of realized_log_ret_1s as a "last realized 1s return" proxy
    lag1 = np.zeros(n, dtype=np.float32)
    lag1[1:] = realized_1s[:-1]
    book[:, 0] = np.clip(lag1, -10.0, 10.0)
    book[:, 1] = 1.0  # ES RTH spread = 1 tick (HC #392 binding constant)
    # Cheap vol-regime tag: use model's own vol prediction; safe finite-fill
    if "pred_pred_realized_vol_30s_ticks" in d.files:
        book[:, 2] = _sanitize(d["pred_pred_realized_vol_30s_ticks"][:n])
    # Time-of-day: assume each day spans equal samples spanning the RTH window
    for di in range(n_days):
        i0, i1 = edges[di], edges[di+1]
        length = max(1, i1 - i0)
        frac = (np.arange(length, dtype=np.float32) / float(length))
        # tod sin/cos over one RTH period
        book[i0:i1, 3] = np.sin(2 * np.pi * frac)
        book[i0:i1, 4] = np.cos(2 * np.pi * frac)
        # dow sin/cos: use day-of-week from OOT date string YYYYMMDD if parsable
        try:
            from datetime import date
            s = oot_dates[di] if di < len(oot_dates) else ""
            y, m, dd = int(s[:4]), int(s[4:6]), int(s[6:8])
            dow = date(y, m, dd).weekday()  # 0=Mon..6=Sun
        except Exception:
            dow = 0
        book[i0:i1, 5] = np.sin(2 * np.pi * dow / 7.0)
        book[i0:i1, 6] = np.cos(2 * np.pi * dow / 7.0)

    # ---- feature normalization stats (cached for inference) ------------
    full_static = np.concatenate([heads, book], axis=1)  # (N, 39)
    feat_mean = full_static.mean(axis=0).astype(np.float32)
    feat_std  = full_static.std(axis=0).astype(np.float32)
    feat_std[feat_std < 1e-6] = 1.0  # avoid /0

    return V33RLDataset(
        head_matrix=heads,
        book_matrix=book,
        realized_log_ret_1s=realized_1s,
        target_fifo_tp4sl3_net=fifo43,
        target_fifo_tp8sl5_net=fifo85,
        day_idx=day_idx,
        oot_dates=oot_dates,
        feat_mean=feat_mean,
        feat_std=feat_std,
    )


def save_feature_stats(ds: V33RLDataset, path: str | Path) -> None:
    """Cache mean/std + head/book/pos names for inference-time consistency."""
    np.savez(
        Path(path),
        feat_mean=ds.feat_mean,
        feat_std=ds.feat_std,
        head_names=np.array(HEAD_NAMES),
        book_names=np.array(BOOK_CONTEXT_NAMES),
        pos_names=np.array(POS_CONTEXT_NAMES),
        state_dim=STATE_DIM,
    )


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
    )
    ds = build_dataset(p)
    print(f"n_samples={ds.n_samples}, days={len(ds.oot_dates)} dates={ds.oot_dates}")
    print(f"STATE_DIM={STATE_DIM} (heads={N_HEADS} + book={N_BOOK} + pos={N_POS})")
    print(f"head_matrix shape={ds.head_matrix.shape}, book={ds.book_matrix.shape}")
    print(f"feat_mean[:5]={ds.feat_mean[:5]}, feat_std[:5]={ds.feat_std[:5]}")
    print(f"reward(log_ret_1s in ticks) mean={ds.realized_log_ret_1s.mean():.4f} "
          f"std={ds.realized_log_ret_1s.std():.4f} "
          f"min={ds.realized_log_ret_1s.min():.2f} max={ds.realized_log_ret_1s.max():.2f}")
