"""
risk_overlay.py — Drawdown Risk Overlay (v2 — with Offensive Mode)

Provides get_risk_level() for wheel paper engines to scale position sizes
based on macro drawdown risk AND panic-reversal opportunity.

DEFENSIVE (v1 — unchanged):
  When drawdown risk is elevated, scale DOWN positions.
  Based on drawdown_predictor_v1 (AUC 0.554, p=0.000, regime gap=0.035).

OFFENSIVE (v2 — NEW):
  When panic confluence signals fire (VIX spike-then-drop, breadth stress,
  credit stress all REVERSING), scale UP premium selling.
  Based on validated VIX panic edge: Sharpe 1.59, WR 76%, PF 8.10, 66 trades / 16 years.

The two modes are mutually exclusive (can't be high-risk AND panic-reversal
simultaneously — the defensive scoring catches the buildup, the offensive
catches the resolution).

Returns:
    {
        "drawdown_prob": float,          # 0.0–1.0 pseudo-probability
        "risk_level": "normal"|"elevated"|"high"|"offensive",
        "position_scale": float,         # 0.3 / 0.7 / 1.0 / 1.5 / 1.75
        "panic_confluence": int,         # 0-3 panic reversal signals active
        "panic_mode": str,               # "offensive"|"cautious"|"normal"
        "signals": dict,                 # individual signal flags for logging
        "as_of": str,                    # ISO timestamp of last calculation
        "cache_expires": str,            # ISO timestamp of next recalc
    }

Thresholds:
    DEFENSIVE (drawdown risk):
        normal   prob < 0.30  -> position_scale = 1.0
        elevated 0.30-0.50    -> position_scale = 0.7
        high     >= 0.50      -> position_scale = 0.3
    OFFENSIVE (panic reversal):
        confluence >= 2       -> position_scale = 1.5 (override defensive)
        confluence == 3       -> position_scale = 1.75

Caches result for 24 hours. If any data fetch fails, returns normal (don't block trading).

Usage:
    from live_trading_linux.risk_overlay import get_risk_level
    risk = get_risk_level()
    contracts = int(base_contracts * risk["position_scale"])
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Cache config ──────────────────────────────────────────────────────────────
_CACHE_DIR = Path(__file__).parent / "vol_cache"
_CACHE_FILE = _CACHE_DIR / "risk_overlay_cache.json"
_CACHE_TTL_HOURS = 24

# ── Data lookback needed for features ─────────────────────────────────────────
_LOOKBACK_DAYS = 310   # need 252 trading days + buffer for 200d MA

# ── Tickers (subset of predictor — IRX skipped, rarely available on yf) ───────
_TICKERS = {
    "spy":  "SPY",
    "vix":  "^VIX",
    "vix3m": "^VIX3M",
    "hyg":  "HYG",
    "tlt":  "TLT",
    "lqd":  "LQD",
    "ief":  "IEF",
    "tnx":  "^TNX",
}

# ── Fallback result (used when any error occurs) ───────────────────────────────
_FALLBACK = {
    "drawdown_prob": 0.0,
    "risk_level": "normal",
    "position_scale": 1.0,
    "panic_confluence": 0,
    "panic_mode": "normal",
    "signals": {},
    "as_of": None,
    "cache_expires": None,
    "fallback": True,
    "error": None,
}


# ─────────────────────────────────────────────────────────────────────────────
# CACHE I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache() -> Optional[dict]:
    """Return cached result if still valid, else None."""
    if not _CACHE_FILE.exists():
        return None
    try:
        with open(_CACHE_FILE) as f:
            data = json.load(f)
        expires = datetime.fromisoformat(data["cache_expires"])
        if datetime.now(timezone.utc) < expires:
            log.info("[risk_overlay] Using cached result (expires %s)", data["cache_expires"])
            return data
    except Exception as e:
        log.warning("[risk_overlay] Cache read failed: %s", e)
    return None


def _save_cache(result: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        log.warning("[risk_overlay] Cache write failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def _download_data() -> dict:
    """Download price history for all tickers. Returns dict of pd.Series."""
    import yfinance as yf
    from datetime import date, timedelta as td

    start = (date.today() - td(days=_LOOKBACK_DAYS)).isoformat()
    end   = date.today().isoformat()

    raw = {}
    for key, ticker in _TICKERS.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df is not None and len(df) > 50:
                raw[key] = df["Close"].squeeze()
                log.debug("[risk_overlay] %s: %d rows", ticker, len(df))
            else:
                log.warning("[risk_overlay] %s: insufficient data (%s rows)",
                            ticker, len(df) if df is not None else 0)
        except Exception as e:
            log.warning("[risk_overlay] %s download failed: %s", ticker, e)

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COMPUTATION (mirrors drawdown_predictor.py build_features)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_features(raw: dict) -> Optional[dict]:
    """
    Compute the latest-day values of the top features from drawdown_predictor_v1.
    Returns a flat dict of feature values, or None if SPY data is missing.
    """
    spy = raw.get("spy")
    if spy is None or len(spy) < 210:
        return None

    vix   = raw.get("vix")
    vix3m = raw.get("vix3m")
    hyg   = raw.get("hyg")
    tlt   = raw.get("tlt")
    lqd   = raw.get("lqd")
    tnx   = raw.get("tnx")

    idx = spy.index
    log_ret = np.log(spy / spy.shift(1))

    f: dict = {}

    # SPY momentum
    for w in [5, 21, 63]:
        f[f"spy_ret_{w}d"] = float(spy.pct_change(w, fill_method=None).iloc[-1])

    # SPY realized vol (top feature bucket)
    for w in [21, 63]:
        f[f"spy_rvol_{w}d"] = float((log_ret.rolling(w).std() * np.sqrt(252)).iloc[-1])

    # Drawdown from rolling peak
    f["spy_dd_63d"]  = float((spy / spy.rolling(63).max() - 1).iloc[-1])
    f["spy_dd_252d"] = float((spy / spy.rolling(252).max() - 1).iloc[-1])

    # Distance from 200d MA (top feature: spy_vs_ma200)
    ma200 = spy.rolling(200).mean()
    f["spy_vs_ma200"] = float((spy / ma200 - 1).iloc[-1])

    # SPY above MAs (regime flags)
    f["spy_above_ma50"]  = int(spy.iloc[-1] > spy.rolling(50).mean().iloc[-1])
    f["spy_above_ma200"] = int(spy.iloc[-1] > ma200.iloc[-1])

    # Skew / kurtosis (top-2 features: spy_kurt_21d, spy_kurt_63d, spy_skew_63d)
    for w in [21, 63]:
        f[f"spy_skew_{w}d"] = float(log_ret.rolling(w).skew().iloc[-1])
        f[f"spy_kurt_{w}d"] = float(log_ret.rolling(w).kurt().iloc[-1])

    # VIX
    if vix is not None:
        v = vix.reindex(idx, method="ffill")
        f["vix_level"]      = float(v.iloc[-1])
        f["vix_change_5d"]  = float(v.pct_change(5, fill_method=None).iloc[-1])
        f["vix_vs_ma20"]    = float((v / v.rolling(20).mean() - 1).iloc[-1])

        if vix3m is not None:
            v3 = vix3m.reindex(idx, method="ffill")
            f["vix_term_ratio"] = float((v / (v3 + 1e-6)).iloc[-1])   # >1 = backwardation
            f["vix3m_level"]    = float(v3.iloc[-1])

        if "spy_rvol_21d" in f:
            f["iv_rv_gap_21d"] = float(v.iloc[-1] / 100 - f["spy_rvol_21d"])

    # Credit spreads (top features: hyg_tlt_ratio, hyg_tlt_vs_ma63, lqd_tlt_ratio)
    if hyg is not None and tlt is not None:
        h = hyg.reindex(idx, method="ffill")
        t = tlt.reindex(idx, method="ffill")
        ratio = h / t
        f["hyg_tlt_ratio"]     = float(ratio.iloc[-1])
        f["hyg_tlt_change_21d"]= float(ratio.pct_change(21, fill_method=None).iloc[-1])
        f["hyg_tlt_vs_ma63"]   = float((ratio / ratio.rolling(63).mean() - 1).iloc[-1])

    if lqd is not None and tlt is not None:
        l = lqd.reindex(idx, method="ffill")
        t = tlt.reindex(idx, method="ffill")
        ratio_ig = l / t
        f["lqd_tlt_ratio"]      = float(ratio_ig.iloc[-1])
        f["lqd_tlt_change_21d"] = float(ratio_ig.pct_change(21, fill_method=None).iloc[-1])

    # TLT momentum (top feature: tlt_ret_63d, tlt_ret_21d)
    if tlt is not None:
        t = tlt.reindex(idx, method="ffill")
        f["tlt_ret_21d"] = float(t.pct_change(21, fill_method=None).iloc[-1])
        f["tlt_ret_63d"] = float(t.pct_change(63, fill_method=None).iloc[-1])

    # Yield curve
    if tnx is not None:
        t10 = tnx.reindex(idx, method="ffill")
        f["yield_10y"]         = float(t10.iloc[-1])
        f["yield_10y_chg_21d"] = float(t10.diff(21).iloc[-1])
        # No IRX in our ticker set, skip slope

    # Seasonality
    today = idx[-1]
    f["month"]       = int(today.month)
    f["is_sept_oct"] = int(today.month in [9, 10])

    return f


# ─────────────────────────────────────────────────────────────────────────────
# RULES-BASED SCORING (fallback until ML model artifacts are saved)
# ─────────────────────────────────────────────────────────────────────────────

def _rules_based_score(feat: dict) -> tuple[float, dict]:
    """
    Scores top features into a pseudo-probability using normalized weights.
    Derived from LGBM gain importance ranking from drawdown_predictor_v1 results.

    Returns (score 0.0–1.0, signals dict for logging).
    """
    signals: dict = {}
    score = 0.0
    max_score = 0.0

    def add(name: str, triggered: bool, weight: float) -> None:
        nonlocal score, max_score
        signals[name] = triggered
        max_score += weight
        if triggered:
            score += weight

    # ── Tier 1: vol / tail-risk signals (highest LGBM gain) ──
    vix = feat.get("vix_level")
    if vix is not None:
        add("vix_above_25",   vix > 25.0,  weight=3.0)
        add("vix_above_20",   vix > 20.0,  weight=1.5)

    vtr = feat.get("vix_term_ratio")
    if vtr is not None:
        add("vix_backwardation", vtr > 1.0, weight=3.0)   # fear > 30d vol
        add("vix_steep_backw",   vtr > 1.10, weight=1.5)

    kurt_21 = feat.get("spy_kurt_21d")
    if kurt_21 is not None:
        add("fat_tails_21d", kurt_21 > 3.0, weight=2.5)   # excess kurtosis

    rvol_63 = feat.get("spy_rvol_63d")
    if rvol_63 is not None:
        add("high_rvol_63d", rvol_63 > 0.20, weight=2.0)  # >20% annualized

    # ── Tier 2: credit / flight-to-quality ──
    hyg_tlt = feat.get("hyg_tlt_vs_ma63")
    if hyg_tlt is not None:
        add("credit_stress",  hyg_tlt < -0.03, weight=2.5)  # HYG/TLT >3% below 63d avg

    lqd = feat.get("lqd_tlt_change_21d")
    if lqd is not None:
        add("ig_spread_widening", lqd < -0.02, weight=2.0)

    tlt_63 = feat.get("tlt_ret_63d")
    if tlt_63 is not None:
        add("flight_to_quality", tlt_63 > 0.05, weight=1.5)  # bonds rallying = risk-off

    # ── Tier 3: SPY trend / momentum ──
    vs_ma200 = feat.get("spy_vs_ma200")
    if vs_ma200 is not None:
        add("spy_below_ma200", vs_ma200 < 0.0,   weight=2.0)
        add("spy_far_below_200", vs_ma200 < -0.05, weight=1.5)

    spy_ret_63 = feat.get("spy_ret_63d")
    if spy_ret_63 is not None:
        add("spy_3m_negative", spy_ret_63 < 0.0,  weight=1.5)
        add("spy_3m_weak",     spy_ret_63 < -0.05, weight=1.5)

    dd_63 = feat.get("spy_dd_63d")
    if dd_63 is not None:
        add("in_drawdown_3pct", dd_63 < -0.03, weight=1.5)

    # ── Tier 4: skew / seasonality ──
    skew_63 = feat.get("spy_skew_63d")
    if skew_63 is not None:
        add("negative_skew_63d", skew_63 < -0.5, weight=1.0)

    is_sept_oct = feat.get("is_sept_oct", 0)
    add("sept_oct_seasonal", bool(is_sept_oct), weight=0.5)

    # Normalize to [0, 1]
    prob = score / max_score if max_score > 0 else 0.0
    return round(prob, 4), signals


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_risk_level(force_refresh: bool = False) -> dict:
    """
    Returns current drawdown risk level for position sizing.

    Result is cached for 24 hours. If data fetch fails, returns normal
    risk level (fallback=True) so trading is never blocked.

    Args:
        force_refresh: bypass cache and recalculate immediately

    Returns:
        {
            "drawdown_prob": float,
            "risk_level": "normal" | "elevated" | "high",
            "position_scale": float,    # multiply base contracts by this
            "signals": dict,
            "as_of": str,
            "cache_expires": str,
            "fallback": bool,
        }
    """
    if not force_refresh:
        cached = _load_cache()
        if cached is not None:
            return cached

    log.info("[risk_overlay] Recalculating risk level …")

    try:
        raw = _download_data()
    except Exception as e:
        log.error("[risk_overlay] Data download failed: %s — returning fallback", e)
        result = dict(_FALLBACK)
        result["error"] = str(e)
        result["as_of"] = datetime.now(timezone.utc).isoformat()
        return result

    try:
        feat = _compute_features(raw)
    except Exception as e:
        log.error("[risk_overlay] Feature computation failed: %s — returning fallback", e)
        result = dict(_FALLBACK)
        result["error"] = str(e)
        result["as_of"] = datetime.now(timezone.utc).isoformat()
        return result

    if feat is None:
        log.warning("[risk_overlay] Insufficient data for features — returning fallback")
        result = dict(_FALLBACK)
        result["error"] = "insufficient_data"
        result["as_of"] = datetime.now(timezone.utc).isoformat()
        return result

    try:
        prob, signals = _rules_based_score(feat)
    except Exception as e:
        log.error("[risk_overlay] Scoring failed: %s — returning fallback", e)
        result = dict(_FALLBACK)
        result["error"] = str(e)
        result["as_of"] = datetime.now(timezone.utc).isoformat()
        return result

    # ── DEFENSIVE: Map drawdown probability → risk level → position scale ──
    if prob < 0.30:
        risk_level = "normal"
        position_scale = 1.0
    elif prob < 0.50:
        risk_level = "elevated"
        position_scale = 0.7
    else:
        risk_level = "high"
        position_scale = 0.3

    # ── OFFENSIVE: Check panic confluence (v2) ──
    # When panic is REVERSING (VIX dropping from spike, breadth recovering,
    # credit stabilizing), override defensive scaling with offensive sizing.
    # This is the validated edge: Sharpe 1.59, WR 76%, PF 8.10.
    panic_confluence = 0
    panic_mode = "normal"
    try:
        from live_trading_linux.panic_confluence_monitor import get_panic_confluence
        panic = get_panic_confluence()
        panic_confluence = panic.get("confluence_score", 0)
        panic_mode = panic.get("mode", "normal")

        if panic_mode == "offensive" and panic_confluence >= 2:
            # Offensive override: panic is REVERSING, scale UP
            risk_level = "offensive"
            position_scale = panic.get("position_multiplier", 1.5)
            log.info("[risk_overlay] OFFENSIVE OVERRIDE: panic confluence=%d, "
                     "scale=%.2f (overrides defensive prob=%.3f)",
                     panic_confluence, position_scale, prob)
        elif panic_mode == "cautious" and risk_level == "normal":
            # Stress building but defensive score hasn't triggered yet —
            # nudge down slightly as early warning
            position_scale = 0.85
            risk_level = "elevated"
            log.info("[risk_overlay] CAUTIOUS early-warning: panic building, "
                     "scale -> 0.85 (defensive prob was only %.3f)", prob)
    except ImportError:
        log.debug("[risk_overlay] panic_confluence_monitor not available, "
                  "offensive mode disabled")
    except Exception as e:
        log.warning("[risk_overlay] Panic confluence check failed: %s — "
                    "continuing with defensive-only scoring", e)

    now = datetime.now(timezone.utc)
    result = {
        "drawdown_prob":   prob,
        "risk_level":      risk_level,
        "position_scale":  position_scale,
        "panic_confluence": panic_confluence,
        "panic_mode":      panic_mode,
        "signals":         signals,
        "as_of":           now.isoformat(),
        "cache_expires":   (now + timedelta(hours=_CACHE_TTL_HOURS)).isoformat(),
        "fallback":        False,
        "error":           None,
    }

    log.info(
        "[risk_overlay] risk_level=%s  prob=%.3f  scale=%.2f  "
        "panic=%d/%s  triggered=[%s]",
        risk_level, prob, position_scale,
        panic_confluence, panic_mode,
        ", ".join(k for k, v in signals.items() if v)
    )

    _save_cache(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="Risk overlay smoke test")
    parser.add_argument("--force", action="store_true", help="Force cache refresh")
    args = parser.parse_args()

    result = get_risk_level(force_refresh=args.force)
    print("\n=== Risk Overlay Result ===")
    for k, v in result.items():
        if k == "signals":
            triggered = [s for s, on in v.items() if on]
            print(f"  signals_triggered : {triggered}")
        else:
            print(f"  {k:<20}: {v}")

    if result.get("risk_level") == "offensive":
        print(f"\n  >>> OFFENSIVE MODE: Panic reversal detected "
              f"(confluence {result.get('panic_confluence', 0)}/3).")
        print(f"      Position scale: {result['position_scale']:.2f}x "
              f"(sell MORE premium — IV is elevated, mean-reversion tailwind).")
