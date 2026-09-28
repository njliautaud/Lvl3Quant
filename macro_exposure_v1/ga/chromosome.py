"""
chromosome.py — GA gene layout for macro-exposure timing strategy.

Genes
-----
Feature weights (each in [-2, +2]):
  w_naaim, w_vix_pct, w_vix_ts_slope, w_aaii_bullbear,
  w_mom_3m, w_mom_6m, w_spy_above_200dma,
  w_yield_2s10s, w_dxy_trend, w_breadth, w_gold_copper_ratio

Threshold genes:
  long_threshold  ∈ [-1.5, +1.5]   (composite score above this → long)
  short_threshold ∈ [-1.5, +1.5]   (composite score below this → short)
                                   we force short_threshold < long_threshold at decode
  flat_band_width ∈ [0, 2]         (extra dead-zone width added around 0)

Basket weights (raw, normalized at decode):
  bw_spy ∈ [0, 1], bw_qqq ∈ [0, 1], bw_iwm ∈ [0, 1]

Discrete-like genes (continuous in [0,1], snapped at decode):
  max_leverage_g ∈ [0, 1]   ->  {1.0, 1.25, 1.5}
  allow_short_g  ∈ [0, 1]   ->  {0, 1}
  cadence_g      ∈ [0, 1]   ->  {"weekly", "biweekly", "monthly"}
  long_strength_g ∈ [0, 1]  ->  {0.5, 1.0, 1.5}   long exposure when triggered
  short_strength_g ∈ [0,1]  ->  {0.5, 1.0}        short exposure when triggered
"""
from __future__ import annotations
from typing import Dict, List, Tuple

FEATURE_WEIGHT_NAMES = [
    "w_naaim",
    "w_vix_pct",
    "w_vix_ts_slope",
    "w_aaii_bullbear",
    "w_mom_3m",
    "w_mom_6m",
    "w_spy_above_200dma",
    "w_yield_2s10s",
    "w_dxy_trend",
    "w_breadth",
    "w_gold_copper_ratio",
]

# The actual feature columns each weight gene multiplies. Some are derived in
# the GA driver (e.g. spy_above_200dma is sign(spy_dma200_dist)).
FEATURE_COLUMN_MAP = {
    "w_naaim":              "naaim_z",
    "w_vix_pct":            "vix_pct_20d_centered",
    "w_vix_ts_slope":       "vix_ts_slope_centered",
    "w_aaii_bullbear":      "aaii_bullbear_z",
    "w_mom_3m":             "spy_mom_3m",
    "w_mom_6m":             "spy_mom_6m",
    "w_spy_above_200dma":   "spy_above_200dma_sign",
    "w_yield_2s10s":        "yield_2s10s",
    "w_dxy_trend":          "dxy_trend_20d",
    "w_breadth":            "breadth_centered",
    "w_gold_copper_ratio":  "gold_copper_z",
}

GENE_NAMES: List[str] = (
    FEATURE_WEIGHT_NAMES
    + ["long_threshold", "short_threshold", "flat_band_width"]
    + ["bw_spy", "bw_qqq", "bw_iwm"]
    + ["max_leverage_g", "allow_short_g", "cadence_g",
       "long_strength_g", "short_strength_g"]
)

GENE_BOUNDS: List[Tuple[float, float]] = (
    [(-2.0, 2.0)] * len(FEATURE_WEIGHT_NAMES)
    + [(-1.5, 1.5), (-1.5, 1.5), (0.0, 2.0)]
    + [(0.0, 1.0)] * 3
    + [(0.0, 1.0)] * 5
)

N_GENES = len(GENE_NAMES)
assert len(GENE_BOUNDS) == N_GENES


def _snap(v: float, choices):
    n = len(choices)
    idx = min(int(v * n), n - 1)
    return choices[idx]


def decode(solution) -> Dict:
    g = {GENE_NAMES[i]: float(solution[i]) for i in range(N_GENES)}

    # Feature weights
    weights = {k: g[k] for k in FEATURE_WEIGHT_NAMES}

    # Thresholds (enforce short < long)
    long_th = g["long_threshold"]
    short_th = g["short_threshold"]
    if short_th > long_th:
        long_th, short_th = short_th, long_th
    flat_band = abs(g["flat_band_width"])

    # Basket weights (normalize, fallback to SPY=1 if all zero)
    bsum = max(g["bw_spy"], 0) + max(g["bw_qqq"], 0) + max(g["bw_iwm"], 0)
    if bsum < 1e-6:
        basket = {"SPY": 1.0, "QQQ": 0.0, "IWM": 0.0}
    else:
        basket = {
            "SPY": max(g["bw_spy"], 0) / bsum,
            "QQQ": max(g["bw_qqq"], 0) / bsum,
            "IWM": max(g["bw_iwm"], 0) / bsum,
        }

    max_leverage = _snap(g["max_leverage_g"], [1.0, 1.25, 1.5])
    allow_short = bool(_snap(g["allow_short_g"], [0, 1]))
    cadence = _snap(g["cadence_g"], ["weekly", "biweekly", "monthly"])
    long_strength = _snap(g["long_strength_g"], [0.5, 1.0, 1.5])
    short_strength = _snap(g["short_strength_g"], [0.5, 1.0])

    # Enforce leverage cap on chosen long strength
    long_strength = min(long_strength, max_leverage)
    short_strength = min(short_strength, max_leverage)

    return dict(
        feature_weights=weights,
        long_threshold=long_th,
        short_threshold=short_th,
        flat_band_width=flat_band,
        basket_weights=basket,
        max_leverage=max_leverage,
        allow_short=allow_short,
        cadence=cadence,
        long_strength=long_strength,
        short_strength=short_strength,
    )
