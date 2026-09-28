"""
hrp.py — Hierarchical Risk Parity allocation for the wheel sleeves.

Per HC #555 R3. Source: extracted/adapted from MacroStrategy
PortfolioOptimizer.get_hrp_weights.

HRP clusters assets by correlation distance, sorts via single-linkage
dendrogram order, then recursively bisects to assign weights inversely
proportional to cluster variance. Produces stable, diversification-aware
weights without inverting a covariance matrix.

Usage:
    from backtest.hrp import hrp_weights, blend_alpha_hrp
    w_hrp = hrp_weights(returns_window)                # pd.Series, sums to 1
    w_blend = blend_alpha_hrp(w_alpha, w_hrp, hrp_frac=0.3)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def _cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
    """Inverse-variance-weighted intra-cluster variance."""
    cov_c = cov.loc[items, items]
    w = 1.0 / (np.diag(cov_c) + 1e-9)
    w /= w.sum()
    return float(np.dot(np.dot(w, cov_c.values), w))


def _recursive_bisection(cov: pd.DataFrame, sorted_items: list[str]) -> pd.Series:
    """Recursively split the sorted ticker list and assign weights by cluster variance."""
    weights = pd.Series(1.0, index=sorted_items)
    items = [list(sorted_items)]
    while len(items) > 0:
        items = [
            sub[j:k]
            for sub in items
            for j, k in ((0, len(sub) // 2), (len(sub) // 2, len(sub)))
            if len(sub) > 1
        ]
        for i in range(0, len(items), 2):
            c_left = items[i]
            c_right = items[i + 1] if i + 1 < len(items) else []
            if not c_right:
                continue
            v_left = _cluster_variance(cov, c_left)
            v_right = _cluster_variance(cov, c_right)
            alpha = 1.0 - v_left / (v_left + v_right + 1e-9)
            weights.loc[c_left] *= alpha
            weights.loc[c_right] *= (1.0 - alpha)
    return weights


def hrp_weights(returns: pd.DataFrame, min_obs: int = 60) -> pd.Series:
    """
    Compute HRP weights from a returns matrix (date x ticker).

    Returns a Series indexed by ticker, summing to 1.0. Equal-weights if
    insufficient data, perfectly degenerate correlation, or linkage fails.
    """
    if returns is None or returns.empty:
        return pd.Series(dtype=float)

    # Drop tickers with too few observations
    keep = returns.columns[returns.notna().sum() >= min_obs]
    r = returns[keep].dropna(how="all")
    if r.shape[1] < 2:
        # Single asset or none — return equal weights on what we have
        if r.shape[1] == 1:
            return pd.Series([1.0], index=r.columns)
        return pd.Series(dtype=float)

    corr = r.corr().fillna(0.0)
    # Distance metric used in classic HRP
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    np.fill_diagonal(dist, 0.0)

    try:
        # linkage expects a condensed distance vector
        cond = squareform(dist, checks=False)
        link = linkage(cond, method="single")
        sort_idx = leaves_list(link)
    except Exception:
        return pd.Series(1.0 / r.shape[1], index=r.columns)

    sorted_tickers = list(r.columns[sort_idx])
    cov = r.cov().fillna(0.0)
    w = _recursive_bisection(cov, sorted_tickers)

    # Normalize — should already be ~1 but guard against drift
    total = w.sum()
    if total > 0:
        w = w / total
    return w.reindex(returns.columns).fillna(0.0)


def blend_alpha_hrp(
    w_alpha: pd.Series,
    w_hrp: pd.Series,
    hrp_frac: float = 0.3,
) -> pd.Series:
    """
    Blend alpha-weighted and HRP weights. Default 70/30 alpha/HRP per HC #555 R3
    (mirrors MacroStrategy's Alpha-Weighted Allocation default).
    """
    if not 0.0 <= hrp_frac <= 1.0:
        raise ValueError(f"hrp_frac must be in [0,1], got {hrp_frac}")
    a = w_alpha.fillna(0.0)
    h = w_hrp.reindex(a.index).fillna(0.0)
    blended = (1.0 - hrp_frac) * a + hrp_frac * h
    total = blended.sum()
    if total > 0:
        blended = blended / total
    return blended


if __name__ == "__main__":
    # Smoke test on wheel prices.parquet
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    px = pd.read_parquet(ROOT / "data" / "cache" / "prices.parquet")
    close_col = "close"
    wide = px.pivot(index="date", columns="ticker", values=close_col).sort_index()
    rets = wide.pct_change().tail(252)  # last year
    w = hrp_weights(rets)
    print("HRP weights — top 10:")
    print(w.sort_values(ascending=False).head(10))
    print(f"sum={w.sum():.6f}  min={w.min():.6f}  max={w.max():.6f}  n={len(w)}")
