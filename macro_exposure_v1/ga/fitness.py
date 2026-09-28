"""
fitness.py — Map BacktestResult.metrics → scalar GA fitness.

Formula (per HC #543 R2):
  fitness = CAGR * Sortino / (1 + max_DD_pct/100)

Penalties:
  -50% multiplicative if max_DD > 25%   (deep-drawdown configs killed)
  -25% multiplicative if turnover > 12 rebalances/year  (kills overtrading)

Guards:
  - negative-CAGR configs get fitness = CAGR * (1 / (1 + dd))   (no Sortino boost)
  - NaN -> -1.0
"""
from __future__ import annotations
import math
from typing import Dict


def compute_fitness(metrics: Dict[str, float]) -> float:
    cagr = float(metrics.get("cagr", 0.0))
    sortino = float(metrics.get("sortino", 0.0))
    dd = float(metrics.get("max_dd_pct", 0.0))  # already in %
    turnover = float(metrics.get("turnover_per_year", 0.0))

    if not all(math.isfinite(x) for x in (cagr, sortino, dd, turnover)):
        return -1.0

    if cagr > 0:
        base = cagr * max(sortino, 0.0) / (1.0 + dd / 100.0)
    else:
        # losing config: don't reward via Sortino magnitude.
        base = cagr / (1.0 + dd / 100.0)

    if dd > 25.0:
        base *= 0.5
    if turnover > 12.0:
        base *= 0.75

    return float(base)
