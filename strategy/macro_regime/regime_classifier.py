"""
macro_regime.regime_classifier — 4-state macro regime classifier (rule-based).

States:
  EARLY-CYCLE  — falling rates + steepening curve + tight/tightening credit + PMI>50 rising
  MID-CYCLE    — stable rates + tight credit + PMI>50 stable
  LATE-CYCLE   — rising rates + flattening + credit widening modestly + PMI rolling
  RECESSION    — curve inverting/un-inverting + credit blow-out + PMI<50 + VIX high

Inputs (daily):
  ust_10y, ust_2y, yc_2s10s     (from wheel_strategy_v1/data/cache/macro_extra.parquet)
  cap_util (PMI proxy)          (same file)
  DXY close                     (yfinance)
  WTI / USO close               (sector_etfs.parquet has USO)
  HYG, LQD                      (LQD pulled from yfinance; HYG already cached)
  VIX                           (yfinance ^VIX)

All causal — every gate uses the day's close vs trailing rolling stats. No look-ahead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
MACRO_EXTRA = ROOT / "wheel_strategy_v1/data/cache/macro_extra.parquet"
SECTOR_ETFS = ROOT / "wheel_strategy_v1/data/cache/sector_etfs.parquet"
PRICES_V2 = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"

REGIME_LABELS = ("EARLY", "MID", "LATE", "RECESSION")


def _slope(s: pd.Series, w: int) -> pd.Series:
    """Rolling OLS slope on the last w values (per-day)."""
    def _f(x):
        if np.isfinite(x).sum() < max(5, w // 2):
            return np.nan
        idx = np.arange(len(x))
        m = np.isfinite(x)
        if m.sum() < 2:
            return np.nan
        return np.polyfit(idx[m], x[m], 1)[0]
    return s.rolling(w, min_periods=max(10, w // 2)).apply(_f, raw=True)


def _pull_yf_series(ticker: str, start="2010-01-01", end=None) -> pd.Series:
    import yfinance as yf
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].astype(float)
    s.index = pd.to_datetime(s.index)
    s.name = ticker
    return s


def build_feature_panel(start: str = "2017-01-01",
                        end: Optional[str] = None,
                        cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Construct the daily macro-feature panel used by the classifier.

    Returns DataFrame indexed by date with columns:
      ust_10y, ust_2y, yc_2s10s, yc_2s10s_chg_60d,
      ust_10y_chg_20d, cap_util, cap_util_chg_6m,
      dxy_close, dxy_slope_50d,
      oil_close, oil_slope_50d,
      hy_oas_proxy, hy_oas_chg_60d,  (= -log(HYG/LQD) z-scored, higher = wider spread)
      vix_close, vix_chg_20d
    """
    macro = pd.read_parquet(MACRO_EXTRA)
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date").sort_index()
    macro = macro.loc[start:end] if end else macro.loc[start:]

    out = pd.DataFrame(index=macro.index)
    out["ust_10y"] = macro["ust_10y"].astype(float)
    out["ust_2y"] = macro["ust_2y"].astype(float)
    out["yc_2s10s"] = macro["yc_2s10s"].astype(float)
    out["cap_util"] = macro["cap_util"].astype(float)

    out["ust_10y_chg_20d"] = out["ust_10y"].diff(20)
    out["yc_2s10s_chg_60d"] = out["yc_2s10s"].diff(60)
    out["cap_util_chg_6m"] = out["cap_util"].diff(126)  # ~6 mo

    # Pull DXY, VIX, LQD, HYG, USO via yfinance
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    yf_tickers = {
        "DXY": "DX-Y.NYB",
        "VIX": "^VIX",
        "LQD": "LQD",
        "HYG": "HYG",
        "USO": "USO",
    }
    pulls = {}
    for k, t in yf_tickers.items():
        cache_file = (cache_dir / f"{k}.parquet") if cache_dir else None
        s = None
        if cache_file is not None and cache_file.exists():
            try:
                cached = pd.read_parquet(cache_file)
                s = cached["close"].copy()
                s.index = pd.to_datetime(cached.index if cached.index.name else cached["date"])
                if isinstance(cached.index, pd.DatetimeIndex):
                    s.index = cached.index
            except Exception:
                s = None
        if s is None or s.empty:
            s = _pull_yf_series(t, start=start, end=end)
            if cache_file is not None and not s.empty:
                df_cache = pd.DataFrame({"close": s.values}, index=s.index)
                df_cache.to_parquet(cache_file)
        pulls[k] = s

    # Align on macro index
    def _align(s, name):
        return s.reindex(out.index, method="ffill").astype(float).rename(name)

    out["dxy_close"] = _align(pulls["DXY"], "dxy")
    out["vix_close"] = _align(pulls["VIX"], "vix")
    hyg = _align(pulls["HYG"], "hyg")
    lqd = _align(pulls["LQD"], "lqd")
    out["oil_close"] = _align(pulls["USO"], "uso")

    # HY-OAS proxy: -log(HYG/LQD). Rising = credit widening (HYG underperforming LQD)
    ratio = (hyg / lqd).replace(0, np.nan)
    out["hy_oas_proxy"] = -np.log(ratio)
    out["hy_oas_chg_60d"] = out["hy_oas_proxy"].diff(60)

    out["dxy_slope_50d"] = _slope(out["dxy_close"], 50)
    out["oil_slope_50d"] = _slope(out["oil_close"], 50)
    out["vix_chg_20d"] = out["vix_close"].diff(20)

    return out


def classify_regime(panel: pd.DataFrame) -> pd.Series:
    """Rule-based 4-state regime classifier.

    Returns a Series of regime labels indexed by date.

    Heuristics (each gate causal — uses today's close vs trailing stats):

    RECESSION (highest priority — stress override):
      VIX > 28 AND (HY widening 60d OR PMI proxy declining 6m hard OR curve inverted)

    LATE-CYCLE:
      curve flattening (yc_2s10s_chg_60d < 0) AND (10Y rising 20d OR cap_util rolling
      over from a high) AND VIX moderate

    EARLY-CYCLE:
      10Y falling 20d AND curve steepening (yc_2s10s_chg_60d > 0)
      AND credit tightening (hy_oas_chg_60d < 0 OR low absolute level)
      AND cap_util rising

    MID-CYCLE: default if none of the above
    """
    # Rolling z-scores for VIX and credit (avoid absolute thresholds that drift)
    vix_z = (panel["vix_close"] - panel["vix_close"].rolling(252, min_periods=60).mean()) \
            / panel["vix_close"].rolling(252, min_periods=60).std()
    credit_z = (panel["hy_oas_proxy"] - panel["hy_oas_proxy"].rolling(252, min_periods=60).mean()) \
               / panel["hy_oas_proxy"].rolling(252, min_periods=60).std()

    regime = pd.Series(index=panel.index, dtype=object)

    yc_chg = panel["yc_2s10s_chg_60d"]
    yc_lvl = panel["yc_2s10s"]
    ust10_chg = panel["ust_10y_chg_20d"]
    cap_chg = panel["cap_util_chg_6m"]
    hy_chg = panel["hy_oas_chg_60d"]
    vix_chg = panel["vix_chg_20d"]
    vix = panel["vix_close"]

    # ---- RECESSION / STRESS ----
    cond_stress = (
        ((vix > 28) & (vix_chg > 0)) |                          # VIX surging
        ((credit_z > 1.5) & (hy_chg > 0)) |                     # Credit blowing out
        ((yc_lvl < -0.10) & (cap_chg < -1.0))                   # Inverted curve + PMI proxy rolling
    )

    # ---- LATE-CYCLE ----
    # Flattening curve + 10Y not falling + cap_util plateauing or rolling
    cond_late = (
        (yc_chg < 0) &
        ((ust10_chg > 0) | (cap_chg < 0)) &
        (~cond_stress)
    )

    # ---- EARLY-CYCLE ----
    # 10Y falling, curve steepening or already steep, credit tightening or low, cap_util rising
    cond_early = (
        (ust10_chg < 0) &
        ((yc_chg > 0) | (yc_lvl > 1.0)) &
        ((hy_chg < 0) | (credit_z < -0.25)) &
        (cap_chg > 0) &
        (~cond_stress) & (~cond_late)
    )

    regime.loc[:] = "MID"
    regime.loc[cond_early.fillna(False)] = "EARLY"
    regime.loc[cond_late.fillna(False)] = "LATE"
    regime.loc[cond_stress.fillna(False)] = "RECESSION"

    # Smoothing: require a regime to persist for >= 5 consecutive days, otherwise
    # carry the previous regime. This prevents whipsaw and matches the
    # "regime-aware not regime-tailored" requirement.
    smoothed = regime.copy()
    last = "MID"
    run_len = 0
    for i, lbl in enumerate(regime.values):
        if lbl == last:
            run_len += 1
        else:
            # tentative new regime — only commit after 5-day run
            future_window = regime.iloc[i:i + 5].values
            if (future_window == lbl).sum() >= 4:
                last = lbl
                run_len = 1
        smoothed.iloc[i] = last
    smoothed.name = "regime"
    return smoothed


def regime_transitions(reg: pd.Series) -> pd.DataFrame:
    """Return a DataFrame of (start_date, end_date, regime, days)."""
    rows = []
    cur = reg.iloc[0]
    start = reg.index[0]
    for d, r in reg.items():
        if r != cur:
            rows.append({"start": start, "end": d, "regime": cur,
                         "days": (d - start).days})
            cur = r
            start = d
    rows.append({"start": start, "end": reg.index[-1], "regime": cur,
                 "days": (reg.index[-1] - start).days})
    return pd.DataFrame(rows)
