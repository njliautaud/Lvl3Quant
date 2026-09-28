"""
defensive_overlay.py — Defensive scaling overlay for the macro_exposure_v1
Balanced tier, addressing the bear-regime Sortino gate failure (0.39 vs 0.50).

Rule (per task brief):
    defensive = (spy_above_200dma == 0) | (vix_pct > 0.8)
    scaled = target_exposure.copy()
    scaled[ longs & defensive ] *= 0.5
    # shorts and flats untouched

Column mapping in this codebase (no exact 'spy_above_200dma' / 'vix_pct' columns):
  - spy_above_200dma  =  (spy_dma200_dist > 0).astype(int)
                         spy_dma200_dist is (SPY close - 200dma) / 200dma in macro_features
  - vix_pct           =  vix_pct_20d (20-day rolling VIX percentile, 0..1)

This file does ONE thing: produce a scaled allocation series. It is intentionally
small and side-effect free so it can be unit-tested independently of the engine.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def derive_defensive_flags(features: pd.DataFrame) -> pd.DataFrame:
    """
    Given the feature panel used by run_wf.py / build_feature_panel(), return a
    DataFrame indexed by the same dates with two boolean-ish columns:
      spy_above_200dma : 1 when SPY > its 200-day moving average, else 0
      vix_pct          : VIX 20-day percentile in [0, 1]
    The full daily flag "defensive" is also returned for convenience.
    """
    # The build_feature_panel() function maps spy_above_200dma_sign = sign(spy_dma200_dist)
    # so spy_above_200dma_sign > 0 means above 200dma. That column is in `features`.
    above = (features["spy_above_200dma_sign"] > 0).astype(int)

    # vix_pct_20d_centered = (vix_pct_20d - 0.5) * 2.0  -> reverse to get raw vix_pct
    vix_pct = (features["vix_pct_20d_centered"] / 2.0) + 0.5
    vix_pct = vix_pct.clip(0.0, 1.0)

    defensive = ((above == 0) | (vix_pct > 0.8)).astype(int)

    return pd.DataFrame(
        {
            "spy_above_200dma": above,
            "vix_pct": vix_pct,
            "defensive": defensive,
        },
        index=features.index,
    )


def apply_defensive_overlay(
    target_exposure: pd.Series,
    daily_features: pd.DataFrame,
    long_scale: float = 0.5,
) -> pd.Series:
    """
    Scale longs down when the defensive regime is triggered.

    Parameters
    ----------
    target_exposure : pd.Series
        Daily target exposure produced by build_allocation() — values like
        {-1, -0.5, 0, +0.5, +1, +1.5}.
    daily_features : pd.DataFrame
        Must contain columns 'spy_above_200dma' (0/1) and 'vix_pct' (0..1).
        If not present, this function will attempt to derive them from the
        canonical feature-panel columns 'spy_above_200dma_sign' and
        'vix_pct_20d_centered'.
    long_scale : float
        Scale factor applied to long positions when defensive. Default 0.5.

    Returns
    -------
    pd.Series of the scaled allocation, aligned to target_exposure.index.
    """
    if "spy_above_200dma" not in daily_features.columns or "vix_pct" not in daily_features.columns:
        df = derive_defensive_flags(daily_features)
    else:
        df = daily_features[["spy_above_200dma", "vix_pct"]].copy()
        df["defensive"] = ((df["spy_above_200dma"] == 0) | (df["vix_pct"] > 0.8)).astype(int)

    df = df.reindex(target_exposure.index).ffill().fillna(0)

    scaled = target_exposure.copy().astype(float)
    long_mask = scaled > 0
    defensive_mask = df["defensive"].astype(bool)
    scaled.loc[long_mask & defensive_mask] = scaled.loc[long_mask & defensive_mask] * float(long_scale)
    return scaled
