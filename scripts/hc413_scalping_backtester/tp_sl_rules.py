"""
HC #413 — TP1/TP2/SL exit rule resolver.

Rule (per HC #413 spec, rule 3):
  TP1 = 0.5 * MFE      (scalp out half the expected favorable move)
  TP2 = 1.0 * MFE      (ride to expected MFE)
  SL  = MAE capped at 1.5 * MFE  (stop tighter than worst historical adverse)

Each fill is resolved deterministically against the realized
target_log_ret_{1s,5s,10s,30s} arrays (which are ALREADY in TICKS per
the v3.3 NPZ convention — see full_market_replay.py header note).

For a long entry, in-position move (in ticks) at horizon h is:
    inpos(h) = +target_log_ret_h[idx]
For a short entry:
    inpos(h) = -target_log_ret_h[idx]

We define MFE_realized = max over horizons of inpos(h), MAE_realized =
min over horizons of inpos(h). Exit reason is then resolved by the first
threshold hit chronologically using the horizon ordering 1s -> 5s -> 10s ->
30s as discrete checkpoints (we don't have intra-horizon path here — the
realized 1s/5s/10s/30s end-of-horizon values are our only checkpoints).

EXIT PRIORITY at each checkpoint h (in order):
  1. If inpos(h) <= -SL              -> exit_reason = "sl"  ,  gross = -SL
  2. Elif inpos(h) >= TP2             -> exit_reason = "tp2",  gross = +TP2
  3. Elif inpos(h) >= TP1             -> exit_reason = "tp1",  gross = +TP1
After all horizons checked with no hit:
  4. exit_reason = "time_stop", gross = inpos(30s)  (or last finite horizon)

NOTE: SL is checked FIRST at each checkpoint — this conservatively
assumes that if the end-of-horizon adverse move exceeds SL, the path got
there. This is the realistic worst-case assumption for a backtester. It
will undercount the TP-then-SL "round-trip" case (which biases against
us, i.e. is conservative).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HORIZONS_ORDERED = ("1s", "5s", "10s", "30s")


@dataclass(frozen=True)
class TpSlThresholds:
    """TP1/TP2/SL in TICKS for a (cell, conf_tier, horizon, side)."""
    tp1: float    # +ticks
    tp2: float    # +ticks
    sl: float     # +ticks (magnitude — applied as -sl for adverse)
    mfe_source: float  # MFE from config (informational)
    mae_source: float  # MAE from config (informational)


def thresholds_from_mfe_mae(mfe: float, mae: float) -> TpSlThresholds:
    """Build TP1/TP2/SL per HC #413 rule 3.

    TP1 = 0.5 * MFE
    TP2 = 1.0 * MFE
    SL  = |MAE| capped at 1.5 * MFE

    All inputs/outputs in TICKS. MFE is the favorable move (positive).
    MAE in the config is signed against the trade direction; here we use
    its magnitude.
    """
    mfe_pos = max(float(mfe), 0.0)
    mae_mag = abs(float(mae))
    tp1 = 0.5 * mfe_pos
    tp2 = 1.0 * mfe_pos
    sl_cap = 1.5 * mfe_pos
    sl = min(mae_mag, sl_cap) if mfe_pos > 0 else mae_mag
    return TpSlThresholds(tp1=tp1, tp2=tp2, sl=sl, mfe_source=mfe_pos, mae_source=mae_mag)


def resolve_exits(
    inpos_by_horizon: dict[str, np.ndarray],
    mask_by_horizon: dict[str, np.ndarray],
    thr: TpSlThresholds,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve TP1/TP2/SL/time_stop per fill.

    Parameters
    ----------
    inpos_by_horizon : dict h -> (N,) ticks, signed (+ favorable, - adverse)
    mask_by_horizon  : dict h -> (N,) bool valid (target available + finite)
    thr              : TpSlThresholds

    Returns
    -------
    gross_ticks : (N,) float — pre-cost realized P&L in ticks per HC #413 rules
    exit_code   : (N,) uint8 — 1=tp1, 2=tp2, 3=sl, 4=time_stop, 0=invalid
    """
    n_arr = next(iter(inpos_by_horizon.values()))
    n = n_arr.shape[0]
    gross = np.zeros(n, dtype=np.float64)
    exit_code = np.zeros(n, dtype=np.uint8)
    resolved = np.zeros(n, dtype=bool)

    tp1, tp2, sl = thr.tp1, thr.tp2, thr.sl
    # SL check first per docstring conservative ordering
    for h in HORIZONS_ORDERED:
        if h not in inpos_by_horizon:
            continue
        ip = inpos_by_horizon[h]
        mk = mask_by_horizon[h]
        active = (~resolved) & mk & np.isfinite(ip)
        if not active.any():
            continue
        # SL hit
        sl_hit = active & (ip <= -sl)
        gross[sl_hit] = -sl
        exit_code[sl_hit] = 3
        resolved |= sl_hit
        # Re-evaluate active mask
        active = (~resolved) & mk & np.isfinite(ip)
        # TP2 hit
        tp2_hit = active & (ip >= tp2) & (tp2 > 0)
        gross[tp2_hit] = tp2
        exit_code[tp2_hit] = 2
        resolved |= tp2_hit
        active = (~resolved) & mk & np.isfinite(ip)
        # TP1 hit
        tp1_hit = active & (ip >= tp1) & (tp1 > 0)
        gross[tp1_hit] = tp1
        exit_code[tp1_hit] = 1
        resolved |= tp1_hit

    # Time stop — anyone unresolved, exit at latest available horizon value
    if not resolved.all():
        time_stop = ~resolved
        # find latest finite horizon per row
        latest = np.full(n, np.nan, dtype=np.float64)
        for h in HORIZONS_ORDERED:
            if h not in inpos_by_horizon:
                continue
            ip = inpos_by_horizon[h]
            mk = mask_by_horizon[h] & np.isfinite(ip)
            latest = np.where(mk, ip, latest)
        gross[time_stop] = np.where(np.isfinite(latest[time_stop]),
                                    latest[time_stop], 0.0)
        exit_code[time_stop] = 4
        # rows with no finite horizon at all -> invalid
        no_data = time_stop & ~np.isfinite(latest)
        exit_code[no_data] = 0
        gross[no_data] = 0.0

    return gross, exit_code
