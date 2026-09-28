"""
iv_skew.py — HC #556 R3 lightweight equity-vol skew model.

For a given ATM IV (sigma_atm), strike K, spot S, DTE T (years), and option
kind, return a SKEW-ADJUSTED implied vol. Used by the wheel engine when it
prices a CSP or CC at a target Δ — the per-option IV becomes:

    iv(K, T) = sigma_atm * (1 + a * m(T) + b * m(T)^2)

where m(T) = log(K/S) / sqrt(T) is the log-moneyness in vol units.

For US equity index / single-name options the empirical risk-reversal is
NEGATIVE (puts more expensive than calls), so a < 0 in log(K/S) terms. We use
conservative parameters:

    a = -0.10  (slope: 1% strike-down → ~10bp IV-up at 30 DTE)
    b = +0.20  (curvature: convex smile that lifts deep OTM both sides)

These are small effects at typical wheel deltas (15-40Δ) but they DO move the
mark.  Calibration TODO: regress a, b against yfinance current chains as
forward-going anchors per HC #556 R2(d).

We export `iv_at_strike(...)` and `sigma_skew_bump(...)` for unit testing.
"""
from __future__ import annotations
import math


SKEW_SLOPE_A = -0.10
SKEW_CURV_B = +0.20
MAX_BUMP = 0.50   # cap |bump| at 50% relative — protects against numerical blowups.

# Source tag for run_meta reporting: "hardcoded" until load_calibration() runs.
CALIBRATION_SOURCE = "hardcoded"


def load_calibration(path: str | None = None) -> dict:
    """Load DOLT-calibrated skew params (lane A3, HC #556 R3(5)).

    Replaces the hardcoded a/b module defaults with the GLOBAL median fit
    from data/cache/skew_calibration_real.json (built by
    data/calibrate_skew_real.py against real option_chain surfaces).

    Opt-in: existing runs keep hardcoded values unless this is called
    (tier_runner --calibrated-skew). Returns the loaded dict.
    """
    global SKEW_SLOPE_A, SKEW_CURV_B, CALIBRATION_SOURCE
    import json
    from pathlib import Path
    p = Path(path) if path else (
        Path(__file__).resolve().parents[1] / "data" / "cache"
        / "skew_calibration_real.json")
    cal = json.loads(p.read_text())
    SKEW_SLOPE_A = float(cal["global"]["a"])
    SKEW_CURV_B = float(cal["global"]["b"])
    CALIBRATION_SOURCE = f"dolt_real:{p.name}"
    return cal


# Walk-forward schedule: sorted list of (period_start_Timestamp, a, b).
# When set (via load_calibration_walkforward), wheel_engine calls set_asof(dt)
# each simulated day so a/b are always the leak-free, prior-data-only fit.
SCHEDULE = None

# Per-ticker walk-forward schedule (v9): {ticker: sorted [(start_ts, a, b)]}.
# Set by load_calibration_walkforward_perticker(). set_asof() refreshes
# _TICKER_AB = {ticker: (a, b)} for the current simulated day; iv_at_strike
# uses it when called with ticker=..., falling back to the global a/b.
TICKER_SCHEDULE = None
_TICKER_AB = {}


def load_calibration_walkforward(path: str | None = None) -> dict:
    """Load the leak-free walk-forward skew schedule (lane A3 fix).

    Built by data/calibrate_skew_walkforward.py: per backtest year, (a, b)
    are medians of surface fits dated strictly before that year. Sets the
    module SCHEDULE; callers (wheel_engine) advance it via set_asof(date).
    """
    global SCHEDULE, SKEW_SLOPE_A, SKEW_CURV_B, CALIBRATION_SOURCE
    import json
    import pandas as _pd
    from pathlib import Path
    p = Path(path) if path else (
        Path(__file__).resolve().parents[1] / "data" / "cache"
        / "skew_calibration_walkforward.json")
    cal = json.loads(p.read_text())
    sched = sorted(
        (_pd.Timestamp(int(y), 1, 1), float(v["a"]), float(v["b"]))
        for y, v in cal["periods"].items())
    if not sched:
        raise ValueError(f"empty walk-forward schedule in {p}")
    SCHEDULE = sched
    SKEW_SLOPE_A, SKEW_CURV_B = sched[0][1], sched[0][2]
    CALIBRATION_SOURCE = f"dolt_real_walkforward:{p.name}"
    return cal


def load_calibration_walkforward_perticker(path: str | None = None) -> dict:
    """Load PER-TICKER leak-free walk-forward skew schedules (v9).

    Built by data/calibrate_skew_walkforward.py --per-ticker: for each
    backtest year Y, ticker tk gets the median (a, b) of ITS OWN surface
    fits dated strictly before Y-01-01 (expanding prior). Tickers with too
    few prior fits are absent for that year — they fall back to the
    GLOBAL-year (a, b), which this loader also installs as SCHEDULE.
    """
    global SCHEDULE, TICKER_SCHEDULE, SKEW_SLOPE_A, SKEW_CURV_B, \
        CALIBRATION_SOURCE
    import json
    import pandas as _pd
    from pathlib import Path
    p = Path(path) if path else (
        Path(__file__).resolve().parents[1] / "data" / "cache"
        / "skew_calibration_walkforward_perticker.json")
    cal = json.loads(p.read_text())
    sched = sorted(
        (_pd.Timestamp(int(y), 1, 1), float(v["a"]), float(v["b"]))
        for y, v in cal["periods"].items())
    if not sched:
        raise ValueError(f"empty walk-forward schedule in {p}")
    tsched: dict[str, list] = {}
    for y, tks in (cal.get("per_ticker_periods") or {}).items():
        start = _pd.Timestamp(int(y), 1, 1)
        for tk, v in tks.items():
            tsched.setdefault(tk, []).append(
                (start, float(v["a"]), float(v["b"])))
    for tk in tsched:
        tsched[tk].sort()
    SCHEDULE = sched
    TICKER_SCHEDULE = tsched
    SKEW_SLOPE_A, SKEW_CURV_B = sched[0][1], sched[0][2]
    CALIBRATION_SOURCE = f"dolt_real_walkforward_perticker:{p.name}"
    _TICKER_AB.clear()
    return cal


def set_asof(date) -> None:
    """Point a/b at the schedule period covering `date` (no-op without schedule)."""
    global SKEW_SLOPE_A, SKEW_CURV_B
    if not SCHEDULE:
        return
    import pandas as _pd
    d = _pd.Timestamp(date)
    a, b = SCHEDULE[0][1], SCHEDULE[0][2]
    for start, sa, sb in SCHEDULE:
        if start <= d:
            a, b = sa, sb
        else:
            break
    SKEW_SLOPE_A, SKEW_CURV_B = a, b
    # Per-ticker schedules (v9): refresh the current-day per-ticker map.
    # A ticker with no period covering d (e.g. thin prior data that year)
    # is left OUT of the map -> iv_at_strike falls back to global a/b.
    _TICKER_AB.clear()
    if TICKER_SCHEDULE:
        for tk, sched in TICKER_SCHEDULE.items():
            cur = None
            for start, sa, sb in sched:
                if start <= d:
                    cur = (sa, sb)
                else:
                    break
            if cur is not None:
                _TICKER_AB[tk] = cur


def _log_moneyness_vol_units(S: float, K: float, T: float) -> float:
    if S <= 0 or K <= 0 or T <= 1e-6:
        return 0.0
    return math.log(K / S) / math.sqrt(max(T, 1e-6))


def iv_at_strike(sigma_atm: float, S: float, K: float, T: float,
                 a: float | None = None, b: float | None = None,
                 ticker: str | None = None) -> float:
    """Skew-adjusted IV at strike K. Falls back to sigma_atm if inputs invalid.

    a/b default to the CURRENT module globals at call time (so
    load_calibration() takes effect for all callers, incl. wheel_engine).
    If `ticker` is given AND a per-ticker walk-forward schedule is loaded
    (load_calibration_walkforward_perticker + set_asof), that ticker's own
    leak-free (a, b) override the globals; unknown tickers use the globals.
    """
    if a is None and b is None and ticker is not None and _TICKER_AB:
        ab = _TICKER_AB.get(ticker)
        if ab is not None:
            a, b = ab
    if a is None:
        a = SKEW_SLOPE_A
    if b is None:
        b = SKEW_CURV_B
    if sigma_atm is None or sigma_atm <= 0:
        return sigma_atm or 0.0
    m = _log_moneyness_vol_units(S, K, T)
    bump = a * m + b * m * m
    # cap relative bump
    if bump > MAX_BUMP:
        bump = MAX_BUMP
    if bump < -MAX_BUMP:
        bump = -MAX_BUMP
    return sigma_atm * (1.0 + bump)


def sigma_skew_bump(sigma_atm: float, S: float, K: float, T: float) -> float:
    """Return the additive bump (iv_at_strike - sigma_atm). Useful for reporting."""
    return iv_at_strike(sigma_atm, S, K, T) - sigma_atm


if __name__ == "__main__":
    # quick self-test
    S = 100.0
    T = 30 / 365.0
    sigma_atm = 0.25
    print(f"S={S} sigma_atm={sigma_atm:.3f} T={T:.4f} (30 DTE)\n")
    print("Strike | iv      | bump_bp")
    for K in [80, 85, 90, 95, 98, 100, 102, 105, 110, 115, 120]:
        iv = iv_at_strike(sigma_atm, S, K, T)
        bump = (iv - sigma_atm) * 10000
        print(f"  {K:5.0f} | {iv:.4f}  | {bump:+.1f}")
