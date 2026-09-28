"""
defensive_overlay_v2.py — Variant of defensive_overlay.py using a LONGER VIX
percentile window (252 trading days = 1 year) instead of 20 days.

Rationale (HC #543 follow-up): overlay-v1 (20d window) cleared the bear-Sortino
gate by only 0.004 and cost ~5pp of pooled CAGR (mostly from over-firing in
early-bull chop when SPY is still below the 200dma). A 252d VIX percentile
should fire less often during normal chop and only kick in during genuine
volatility regimes (2018 Vol-mageddon, 2020 COVID, 2022 bear).

Rule:
    defensive = (spy_dma200_dist <= 0) | (vix_pct_252d > 0.80)
    scaled[longs & defensive] *= 0.5
    # shorts and flats untouched

Column mapping:
  - SPY-below-200dma  =  features['spy_dma200_dist'] <= 0  (raw column in panel)
  - vix_pct_252d      =  raw 'vix' column .rolling(252).rank(pct=True)
                         computed inline here since the panel ships only vix_pct_20d.

Same author/style as overlay-v1. Side-effect free. Authorized under HC #420.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def derive_defensive_flags_v2(features: pd.DataFrame, raw_vix: pd.Series | None = None) -> pd.DataFrame:
    """
    Compute the v2 flag set: SPY-vs-200dma + 252-day VIX percentile.

    Column mapping:
      - SPY-above-200dma : features['spy_above_200dma_sign'] > 0
                           (sign of the dma-distance, post-transform column the
                           GA panel exposes; +1 = above, -1 = below.)
      - vix_pct_252d     : 252-trading-day rolling pct-rank of RAW VIX close.
                           Raw VIX is NOT in the GA panel, so we either accept
                           it as `raw_vix` (caller-supplied) or pull from the
                           cache parquet on disk.

    Returns DataFrame indexed by features.index with columns:
      spy_above_200dma, vix_pct_252d, defensive
    """
    above = (features["spy_above_200dma_sign"] > 0).astype(int)

    if raw_vix is None:
        from pathlib import Path
        cache = Path(__file__).resolve().parents[1] / "data" / "cache" / "macro_features.parquet"
        df = pd.read_parquet(cache)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        raw_vix = df["vix"].astype(float)

    raw_vix = raw_vix.reindex(features.index).ffill()

    # 252-day rolling VIX percentile.
    # min_periods=63 means we start having a sane percentile after ~3 months of
    # history. Earlier days will be NaN; we fill with 0.5 (neutral) so the
    # overlay doesn't over-fire in the burn-in period.
    vix_pct_252d = raw_vix.rolling(252, min_periods=63).rank(pct=True)
    vix_pct_252d = vix_pct_252d.fillna(0.5).clip(0.0, 1.0)

    defensive = ((above == 0) | (vix_pct_252d > 0.80)).astype(int)

    return pd.DataFrame(
        {
            "spy_above_200dma": above,
            "vix_pct_252d": vix_pct_252d,
            "defensive": defensive,
        },
        index=features.index,
    )


def apply_defensive_overlay_v2(
    target_exposure: pd.Series,
    daily_features: pd.DataFrame,
    long_scale: float = 0.5,
) -> pd.Series:
    """
    Scale longs down by `long_scale` when the v2 defensive regime is triggered.

    Parameters
    ----------
    target_exposure : pd.Series
        Daily target exposure produced by build_allocation().
    daily_features : pd.DataFrame
        Must contain the raw `spy_dma200_dist` and `vix` columns (the standard
        macro-features panel does).
    long_scale : float
        Scale factor applied to long positions when defensive. Default 0.5.

    Returns
    -------
    pd.Series of the scaled allocation, aligned to target_exposure.index.
    """
    flags = derive_defensive_flags_v2(daily_features)
    flags = flags.reindex(target_exposure.index).ffill().fillna(0)

    scaled = target_exposure.copy().astype(float)
    long_mask = scaled > 0
    defensive_mask = flags["defensive"].astype(bool)
    scaled.loc[long_mask & defensive_mask] = (
        scaled.loc[long_mask & defensive_mask] * float(long_scale)
    )
    return scaled
