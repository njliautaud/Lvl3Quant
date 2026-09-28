"""
HC #413 — risk-adjusted metrics (HC #69) + HC #344 day_conc + HC #408 CI gate.

All metrics computed in TICKS-per-trade space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY_CONC_GATE = 0.20         # HC #344
HC408_MIN_FILLS = 50         # HC #408
BOOTSTRAP_REPS = 2000        # CI_low_95 bootstrap reps
RNG_SEED = 42


def sharpe_sqrtN(net: np.ndarray) -> float:
    if net.size < 2:
        return float("nan")
    sd = float(net.std(ddof=1))
    if sd < 1e-12:
        return float("nan")
    return float(net.mean() / sd * np.sqrt(net.size))


def sortino_sqrtN(net: np.ndarray) -> float:
    if net.size < 2:
        return float("nan")
    neg = net[net < 0]
    if neg.size < 2:
        return float("inf") if net.mean() > 0 else float("nan")
    ds = float(neg.std(ddof=1))
    if ds < 1e-12:
        return float("nan")
    return float(net.mean() / ds * np.sqrt(net.size))


def profit_factor(net: np.ndarray) -> float:
    if net.size == 0:
        return float("nan")
    gw = float(net[net > 0].sum())
    gl = -float(net[net < 0].sum())
    if gl < 1e-12:
        return float("inf") if gw > 0 else float("nan")
    return gw / gl


def win_rate(net: np.ndarray) -> float:
    if net.size == 0:
        return float("nan")
    return float((net > 0).mean() * 100.0)


def day_conc(net: np.ndarray, ts_ns: np.ndarray) -> float:
    """Fraction of |P&L| concentrated in the worst single day (HC #344)."""
    if net.size == 0 or ts_ns.size != net.size:
        return float("nan")
    try:
        dts = pd.to_datetime(ts_ns, unit="ns", utc=True).tz_convert("America/Chicago").date
        df = pd.DataFrame({"d": dts, "n": net})
        per_day = df.groupby("d")["n"].sum()
        total_abs = float(per_day.abs().sum())
        if total_abs < 1e-12:
            return float("nan")
        return float(per_day.abs().max() / total_abs)
    except Exception:
        return float("nan")


def bootstrap_ci_low_95(net: np.ndarray, reps: int = BOOTSTRAP_REPS,
                        seed: int = RNG_SEED) -> float:
    """Lower bound of 95% CI for mean ticks/trade via percentile bootstrap."""
    if net.size < 5:
        return float("nan")
    rng = np.random.default_rng(seed)
    n = net.size
    idx = rng.integers(0, n, size=(reps, n))
    means = net[idx].mean(axis=1)
    return float(np.percentile(means, 2.5))


def pass_hc344(day_conc_val: float, n_fills: int) -> bool:
    """HC #344 gate: day_conc <= 0.20 AND n_fills >= 30."""
    return bool(np.isfinite(day_conc_val) and day_conc_val <= DAY_CONC_GATE
                and n_fills >= 30)


def pass_hc408_honesty(n_fills: int, ci_low_95: float, day_conc_val: float) -> bool:
    """HC #408 honesty gate: n_fills>=50, CI_low_95>0, day_conc<=0.20."""
    return bool(n_fills >= HC408_MIN_FILLS
                and np.isfinite(ci_low_95) and ci_low_95 > 0.0
                and np.isfinite(day_conc_val) and day_conc_val <= DAY_CONC_GATE)


def summarize_cell(net: np.ndarray, ts_ns: np.ndarray) -> dict:
    """All HC metrics for one (cell × tier × horizon × side)."""
    n_fills = int(net.size)
    if n_fills == 0:
        return {
            "n_fills": 0,
            "realized_net_per_fill": 0.0,
            "sharpe_sqrtN": float("nan"),
            "sortino_sqrtN": float("nan"),
            "pf": float("nan"),
            "wr": float("nan"),
            "day_conc": float("nan"),
            "ci_low_95_net": float("nan"),
            "pass_hc344": False,
            "pass_hc408_honesty": False,
        }
    dc = day_conc(net, ts_ns) if ts_ns is not None and ts_ns.size == n_fills else float("nan")
    ci = bootstrap_ci_low_95(net)
    return {
        "n_fills": n_fills,
        "realized_net_per_fill": float(net.mean()),
        "sharpe_sqrtN": sharpe_sqrtN(net),
        "sortino_sqrtN": sortino_sqrtN(net),
        "pf": profit_factor(net),
        "wr": win_rate(net),
        "day_conc": dc,
        "ci_low_95_net": ci,
        "pass_hc344": pass_hc344(dc, n_fills),
        "pass_hc408_honesty": pass_hc408_honesty(n_fills, ci, dc),
    }
