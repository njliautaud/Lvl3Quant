#!/usr/bin/env python3
"""
Macro Factor Model v1 — Do Macroeconomic Factors Improve LGBM Sector Ranking?
===============================================================================

HC #0  : Sliding walk-forward only (500d train, 250d test, biweekly rebalance)
HC #428: Regime-agnostic OOT validation (40+ days, all regimes)

Tests whether adding 10 macro factors to the production v4 LGBM (21 features)
improves sector ranking for bull call spreads.

5 Variants:
  A) Baseline      — Production v4 (21 features, no macro)
  B) All macro     — 21 + 10 macro features = 31 total
  C) Best subset   — 21 + top 5 macro by importance (from B's importances)
  D) Macro only    — Only the 10 macro features (standalone value test)
  E) Regime-cond.  — Different macro subsets for VIX>25 vs VIX<25

New macro features:
  1. yield_curve_slope        — TLT/SHY ratio (long-short rate proxy)
  2. yield_curve_momentum     — 21d change in TLT/SHY ratio
  3. credit_spread_level      — HYG/TLT ratio (tighter = risk-on)
  4. credit_spread_momentum   — 21d change in HYG/TLT
  5. real_rates_proxy         — TLT 21d return - GLD 21d return
  6. dollar_strength          — UUP 21d return (inverse GLD momentum fallback)
  7. risk_appetite             — HYG 21d ret - TLT 21d ret (positive = risk-on)
  8. eq_vol_regime            — VIX / VIX3M ratio (contango/backwardation)
  9. sector_rotation_speed    — Rolling 21d stdev of sector rank changes
  10. cross_asset_momentum    — Mean 21d return of TLT, GLD, HYG

$645 starting capital, 15% entry haircut, $2.60 commission, hold to expiry.
Bull call spreads: ATM + 3% OTM, DTE=21. Top 3 sectors per rebalance.
ATR-based BS pricing, iv_multiplier=1.2. VIX > 20 regime filter.
Calendar month Sharpe. 5-gate adversarial validation.
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_builtin_print = print


def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# ── Standardized tools ──
sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

# ── Config ──
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "macro_factor_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUTPUT_DIR / "macro_factor_v1_results.json"

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
MACRO_TICKERS = ["SPY", "TLT", "SHY", "HYG", "GLD", "UUP", "^VIX", "^VIX3M"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
MAX_POS = 200.0

# Walk-forward: 500d train, 250d test, biweekly rebalance
WF_TRAIN_DAYS = 500
WF_TEST_DAYS = 250
WF_TRAIN_PERIODS = 25  # ~500 days of biweekly data
WF_REBAL_FREQ = "2W-FRI"

# 18 legacy momentum/quality features
LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]
# 3 cross-asset features (production v4)
CROSS_ASSET_FEATURES = [
    "sector_spy_beta_63d",
    "sector_relative_vol_21d",
    "cross_sector_dispersion",
]
# 10 new macro features
MACRO_FEATURES = [
    "yield_curve_slope",
    "yield_curve_momentum",
    "credit_spread_level",
    "credit_spread_momentum",
    "real_rates_proxy",
    "dollar_strength",
    "risk_appetite",
    "eq_vol_regime",
    "sector_rotation_speed",
    "cross_asset_momentum",
]

PROD_V4_FEATURES = LEGACY_FEATURES + CROSS_ASSET_FEATURES  # 21 total
ALL_MACRO_FEATURES = PROD_V4_FEATURES + MACRO_FEATURES      # 31 total

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "macro_factor_model_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=2)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable -- results saved to disk only")


# ======================================================================
# DATA DOWNLOAD
# ======================================================================

def download_data():
    """Download sector ETFs + macro tickers from yfinance (2007-2026)."""
    import yfinance as yf

    all_tickers = SECTORS + MACRO_TICKERS
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in all_tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    fprint(f"Downloading {len(unique)} tickers...")
    raw = yf.download(unique, start="2007-01-01", progress=False, auto_adjust=True)
    mi = isinstance(raw.columns, pd.MultiIndex)

    close = raw["Close"] if mi else raw[["Close"]]
    high = raw["High"] if mi else raw[["High"]]
    low = raw["Low"] if mi else raw[["Low"]]

    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)

    close = close.ffill()
    high = high.ffill()
    low = low.ffill()

    rename_map = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)

    fprint(f"Data: {len(close)} days, columns: {list(close.columns)}")
    fprint(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    return close, high, low


# ======================================================================
# FEATURE ENGINEERING
# ======================================================================

def compute_legacy_features(px, spy_slice):
    """Compute the 18 legacy quality-momentum features for a single sector."""
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()
    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f["sharpe_63d"] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0
    pk63 = px.iloc[-63:].cummax()
    f["maxdd_63d"] = float(((px.iloc[-63:] / pk63) - 1).min())
    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = r63[r63 < 0]
    f["sortino_63d"] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f["calmar_1y"] = cagr / (abs(mdd) + 1e-10)
    up_days = rets[rets > 0]
    f["up_capture"] = float(up_days.iloc[-63:].mean() / (up_days.mean() + 1e-10)) if len(up_days) > 10 else 1.0

    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f["trend_r2_63d"] = r_val ** 2
        f["trend_slope_63d"] = slope * 252
    else:
        f["trend_r2_63d"] = 0.0
        f["trend_slope_63d"] = 0.0

    return f


def compute_cross_asset_features(sector_ticker, dt_idx, close_df):
    """Compute the 3 cross-asset features (part of production v4)."""
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {k: 0.0 for k in CROSS_ASSET_FEATURES}

    spy_ret = spy.pct_change().dropna()

    # 1. Sector-SPY beta 63d
    if sector_px is not None and len(sector_px) > 63:
        sec_ret = sector_px.pct_change().dropna()
        common = spy_ret.index.intersection(sec_ret.index)
        if len(common) > 63:
            sr = sec_ret.loc[common].iloc[-63:]
            mr = spy_ret.loc[common].iloc[-63:]
            cov = np.cov(sr.values, mr.values)
            f["sector_spy_beta_63d"] = float(cov[0, 1] / (cov[1, 1] + 1e-10))
        else:
            f["sector_spy_beta_63d"] = 1.0
    else:
        f["sector_spy_beta_63d"] = 1.0

    # 2. Sector relative vol 21d
    if sector_px is not None and len(sector_px) > 21:
        sec_ret = sector_px.pct_change().dropna()
        if len(sec_ret) > 21 and len(spy_ret) > 21:
            f["sector_relative_vol_21d"] = float(sec_ret.iloc[-21:].std() / (spy_ret.iloc[-21:].std() + 1e-10))
        else:
            f["sector_relative_vol_21d"] = 1.0
    else:
        f["sector_relative_vol_21d"] = 1.0

    # 3. Cross-sector dispersion
    sector_cols = [c for c in SECTORS if c in close_df.columns]
    if len(sector_cols) > 3:
        sector_rets = close_df[sector_cols].iloc[:dt_idx + 1].pct_change()
        daily_disp = sector_rets.std(axis=1)
        if len(daily_disp) > 21:
            f["cross_sector_dispersion"] = float(daily_disp.rolling(21).mean().iloc[-1])
        else:
            f["cross_sector_dispersion"] = 0.01
    else:
        f["cross_sector_dispersion"] = 0.01

    return f


def compute_macro_features(dt_idx, close_df):
    """Compute the 10 new macro features. All are cross-sectional (same for all sectors)."""
    f = {}

    def _safe_ratio(a, b):
        """Price ratio a/b at dt_idx."""
        if a in close_df.columns and b in close_df.columns:
            av = close_df[a].iloc[dt_idx]
            bv = close_df[b].iloc[dt_idx]
            if pd.notna(av) and pd.notna(bv) and bv > 0:
                return float(av / bv)
        return np.nan

    def _ret_21d(ticker):
        """21-day return for a ticker at dt_idx."""
        if ticker not in close_df.columns or dt_idx < 21:
            return np.nan
        cur = close_df[ticker].iloc[dt_idx]
        prev = close_df[ticker].iloc[dt_idx - 21]
        if pd.notna(cur) and pd.notna(prev) and prev > 0:
            return float(cur / prev - 1)
        return np.nan

    # 1. Yield curve slope: TLT/SHY ratio (long/short rate proxy)
    f["yield_curve_slope"] = _safe_ratio("TLT", "SHY")

    # 2. Yield curve momentum: 21d change in TLT/SHY ratio
    if dt_idx >= 21 and "TLT" in close_df.columns and "SHY" in close_df.columns:
        tlt = close_df["TLT"]
        shy = close_df["SHY"]
        ratio_now = tlt.iloc[dt_idx] / (shy.iloc[dt_idx] + 1e-10)
        ratio_prev = tlt.iloc[dt_idx - 21] / (shy.iloc[dt_idx - 21] + 1e-10)
        f["yield_curve_momentum"] = float(ratio_now - ratio_prev)
    else:
        f["yield_curve_momentum"] = 0.0

    # 3. Credit spread level: HYG/TLT ratio (tighter credit = risk-on)
    f["credit_spread_level"] = _safe_ratio("HYG", "TLT")

    # 4. Credit spread momentum: 21d change in HYG/TLT
    if dt_idx >= 21 and "HYG" in close_df.columns and "TLT" in close_df.columns:
        hyg = close_df["HYG"]
        tlt = close_df["TLT"]
        r_now = hyg.iloc[dt_idx] / (tlt.iloc[dt_idx] + 1e-10)
        r_prev = hyg.iloc[dt_idx - 21] / (tlt.iloc[dt_idx - 21] + 1e-10)
        f["credit_spread_momentum"] = float(r_now - r_prev)
    else:
        f["credit_spread_momentum"] = 0.0

    # 5. Real rates proxy: TLT 21d return - GLD 21d return
    tlt_ret = _ret_21d("TLT")
    gld_ret = _ret_21d("GLD")
    if not np.isnan(tlt_ret) and not np.isnan(gld_ret):
        f["real_rates_proxy"] = tlt_ret - gld_ret
    else:
        f["real_rates_proxy"] = 0.0

    # 6. Dollar strength: UUP 21d return (fallback: inverse GLD momentum)
    uup_ret = _ret_21d("UUP")
    if not np.isnan(uup_ret):
        f["dollar_strength"] = uup_ret
    elif not np.isnan(gld_ret):
        f["dollar_strength"] = -gld_ret  # inverse GLD as proxy
    else:
        f["dollar_strength"] = 0.0

    # 7. Risk appetite: HYG 21d return - TLT 21d return (positive = risk-on)
    hyg_ret = _ret_21d("HYG")
    if not np.isnan(hyg_ret) and not np.isnan(tlt_ret):
        f["risk_appetite"] = hyg_ret - tlt_ret
    else:
        f["risk_appetite"] = 0.0

    # 8. Equity vol regime: VIX / VIX3M ratio (contango = <1, backwardation = >1)
    if "VIX" in close_df.columns and "VIX3M" in close_df.columns:
        vix_val = close_df["VIX"].iloc[dt_idx]
        vix3m_val = close_df["VIX3M"].iloc[dt_idx]
        if pd.notna(vix_val) and pd.notna(vix3m_val) and vix3m_val > 0:
            f["eq_vol_regime"] = float(vix_val / vix3m_val)
        else:
            f["eq_vol_regime"] = 1.0
    else:
        f["eq_vol_regime"] = 1.0

    # 9. Sector rotation speed: rolling 21d stdev of sector rank changes
    sector_cols = [c for c in SECTORS if c in close_df.columns]
    if len(sector_cols) > 3 and dt_idx >= 42:
        # Compute 5d returns and rank each day, then measure rank volatility
        sector_px = close_df[sector_cols].iloc[max(0, dt_idx - 42):dt_idx + 1]
        sector_ret_5d = sector_px.pct_change(5)
        ranks = sector_ret_5d.rank(axis=1, pct=True)
        rank_changes = ranks.diff().abs()
        # Average rank change per day, then take rolling 21d stdev
        avg_rank_change = rank_changes.mean(axis=1).dropna()
        if len(avg_rank_change) >= 21:
            f["sector_rotation_speed"] = float(avg_rank_change.iloc[-21:].std())
        else:
            f["sector_rotation_speed"] = 0.0
    else:
        f["sector_rotation_speed"] = 0.0

    # 10. Cross-asset momentum: mean 21d return of TLT, GLD, HYG
    returns_21d = []
    for tk in ["TLT", "GLD", "HYG"]:
        r = _ret_21d(tk)
        if not np.isnan(r):
            returns_21d.append(r)
    f["cross_asset_momentum"] = float(np.mean(returns_21d)) if returns_21d else 0.0

    return f


# ======================================================================
# ATR COMPUTATION
# ======================================================================

def compute_atr_series(high, low, close, period=14):
    """Compute ATR series for all sectors."""
    atr_dict = {}
    for tk in SECTORS:
        if tk in high.columns and tk in low.columns and tk in close.columns:
            h = high[tk].dropna()
            l = low[tk].dropna()
            c = close[tk].dropna()
            common = h.index.intersection(l.index).intersection(c.index)
            if len(common) > period:
                tr1 = h.loc[common] - l.loc[common]
                tr2 = (h.loc[common] - c.loc[common].shift(1)).abs()
                tr3 = (l.loc[common] - c.loc[common].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_dict[tk] = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return atr_dict


# ======================================================================
# WALK-FORWARD LGBM RANKING
# ======================================================================

def build_feature_records(close, high, low, rebal_dates, feature_cols, vix_filter=None):
    """Build feature + target records for LGBM walk-forward ranking.

    Args:
        close, high, low: Price DataFrames
        rebal_dates: DatetimeIndex of rebalance dates
        feature_cols: List of feature column names to include
        vix_filter: None = no filter, 'high' = VIX>20 only, 'low' = VIX<=20 only
    """
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        # Optional VIX filter
        if vix_filter is not None and vix is not None:
            cv = float(vix.iloc[idx]) if idx < len(vix) and pd.notna(vix.iloc[idx]) else 20.0
            if vix_filter == "high" and cv < 20:
                continue
            if vix_filter == "low" and cv >= 20:
                continue

        # Compute macro features once per date (shared across sectors)
        needs_macro = any(fc in MACRO_FEATURES for fc in feature_cols)
        macro_feats = compute_macro_features(idx, close) if needs_macro else {}

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            # Cross-asset features
            cross_asset = {}
            if any(fc in CROSS_ASSET_FEATURES for fc in feature_cols):
                cross_asset = compute_cross_asset_features(tk, idx, close)

            # Forward return target (DTE days forward)
            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross_asset, **macro_feats,
                   "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


def walk_forward_lgbm_rank(df, feature_cols):
    """Walk-forward LGBM: sliding 500d train, 250d test window.

    Returns:
        rankings: dict[date] -> dict[ticker] -> score
        imp_df: DataFrame of feature importances (sorted desc)
    """
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    Insufficient data ({len(df)} records)")
        return {}, None

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    rankings = {}
    all_importances = np.zeros(len(feature_cols))
    n_models = 0

    for i in range(WF_TRAIN_PERIODS, len(dates)):
        train_dates = dates[max(0, i - WF_TRAIN_PERIODS):i]
        test_date = dates[i]

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()

        if len(test_df) < 3 or len(train_df) < 50:
            continue

        Xt = np.nan_to_num(train_df[feature_cols].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[feature_cols].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)

            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))

            all_importances += m.feature_importances_
            n_models += 1
        except Exception:
            continue

    if n_models > 0:
        all_importances /= n_models
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": all_importances,
        }).sort_values("importance", ascending=False)
    else:
        imp_df = None

    fprint(f"    {len(rankings)} ranking dates, {n_models} models trained")
    return rankings, imp_df


def walk_forward_lgbm_rank_regime_conditional(df, close):
    """Variant E: Train separate LGBM models for VIX>25 and VIX<25 regimes,
    each with a different macro feature subset.

    High-VIX subset: credit/vol-focused macro features
    Low-VIX subset:  momentum/rotation-focused macro features
    """
    import lightgbm as lgb

    vix = close["VIX"] if "VIX" in close.columns else None

    # Different macro subsets per regime
    high_vix_macro = [
        "credit_spread_level", "credit_spread_momentum",
        "eq_vol_regime", "risk_appetite", "real_rates_proxy",
    ]
    low_vix_macro = [
        "yield_curve_slope", "yield_curve_momentum",
        "dollar_strength", "sector_rotation_speed", "cross_asset_momentum",
    ]

    high_vix_features = PROD_V4_FEATURES + high_vix_macro  # 26
    low_vix_features = PROD_V4_FEATURES + low_vix_macro    # 26

    if len(df) < 100:
        fprint(f"    Insufficient data ({len(df)} records)")
        return {}, None

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    rankings = {}
    # Track combined importances from both regimes
    all_feat_set = list(set(high_vix_features + low_vix_features))
    combined_importances = {f: 0.0 for f in all_feat_set}
    n_models = 0

    for i in range(WF_TRAIN_PERIODS, len(dates)):
        train_dates = dates[max(0, i - WF_TRAIN_PERIODS):i]
        test_date = dates[i]

        # Determine current VIX regime
        if vix is not None and test_date in close.index:
            t_idx = close.index.get_indexer([test_date], method="ffill")[0]
            cv = float(vix.iloc[t_idx]) if t_idx < len(vix) and pd.notna(vix.iloc[t_idx]) else 20.0
        else:
            cv = 20.0

        if cv >= 25:
            feature_cols = high_vix_features
        else:
            feature_cols = low_vix_features

        # Ensure all needed columns exist
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0.0

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()

        if len(test_df) < 3 or len(train_df) < 50:
            continue

        Xt = np.nan_to_num(train_df[feature_cols].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[feature_cols].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)

            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))

            for fi, feat in enumerate(feature_cols):
                combined_importances[feat] += m.feature_importances_[fi]
            n_models += 1
        except Exception:
            continue

    if n_models > 0:
        for k in combined_importances:
            combined_importances[k] /= n_models
        imp_df = pd.DataFrame([
            {"feature": k, "importance": v} for k, v in combined_importances.items()
        ]).sort_values("importance", ascending=False)
    else:
        imp_df = None

    fprint(f"    {len(rankings)} ranking dates, {n_models} models trained (regime-conditional)")
    return rankings, imp_df


# ======================================================================
# TRADE SIMULATION
# ======================================================================

def simulate_bull_call_spreads(rankings, close, atr_dict, top_n=3):
    """Simulate bull call spreads on top-N sectors. VIX > 20 regime filter."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    all_trades = []

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        idx = close.index.get_indexer([dt], method="ffill")[0]
        cv = float(vix.iloc[idx]) if vix is not None and idx < len(vix) and pd.notna(vix.iloc[idx]) else 20.0

        # VIX > 20 regime filter
        if cv < 20:
            continue

        scores = rankings[dt]
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:top_n]]

        max_per_trade = min(MAX_POS, equity / top_n)
        if max_per_trade < 30:
            continue

        for tk in picks:
            if tk not in close.columns or tk not in atr_dict:
                continue

            S = float(close[tk].iloc[idx])
            ei = min(idx + DTE, len(close) - 1)
            if ei <= idx:
                continue

            if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
                av = float(atr_dict[tk].loc[dt])
            else:
                av = S * 0.015

            K1 = round(S, 2)   # ATM
            K2 = round(S * (1 + SPREAD_PCT / 100), 2)  # 3% OTM
            if K2 <= K1:
                K2 = K1 + 1.0

            try:
                entry_cost_ps, max_profit_ps = price_bull_call_spread(
                    S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
                )
            except Exception:
                continue

            total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
            if total_cost <= 0 or total_cost > max_per_trade or total_cost > equity * 0.40:
                continue

            # Hold to expiry -- intrinsic value only
            Se = float(close[tk].iloc[ei])
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
            pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

            equity += pnl
            all_trades.append({
                "pnl": round(pnl, 2),
                "entry_date": str(close.index[idx].date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "vix": round(cv, 1),
                "win": pnl > 0,
            })

    return all_trades, equity


# ======================================================================
# ANALYSIS HELPERS
# ======================================================================

def compute_metrics(trades, final_equity):
    """Compute summary metrics from trade list."""
    if not trades:
        return {"n_trades": 0, "error": "no_trades"}

    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    wr = len(wins) / len(pnls) if len(pnls) > 0 else 0.0
    pf = float(wins.sum() / (abs(losses.sum()) + 1e-10)) if len(losses) > 0 else 999.0
    total_return = (final_equity / CAP - 1) * 100

    # Calendar month Sharpe from equity curve
    eq_series = pd.Series([CAP] + list(CAP + np.cumsum(pnls)),
                          index=range(len(pnls) + 1))
    eq_pct = eq_series.pct_change().dropna()
    sharpe = float(eq_pct.mean() / (eq_pct.std() + 1e-10) * np.sqrt(26))  # ~26 biweekly periods/yr

    # Sortino
    downside = eq_pct[eq_pct < 0]
    sortino = float(eq_pct.mean() / (downside.std() + 1e-10) * np.sqrt(26)) if len(downside) > 3 else sharpe

    # Max drawdown
    cum_pnl = np.cumsum(pnls)
    eq_curve = CAP + cum_pnl
    peak = np.maximum.accumulate(eq_curve)
    dd = (eq_curve - peak) / peak
    max_dd = float(dd.min())

    return {
        "n_trades": len(trades),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 4),
        "max_dd": round(max_dd, 4),
        "total_return_pct": round(total_return, 1),
        "final_equity": round(final_equity, 2),
    }


# ======================================================================
# MAIN
# ======================================================================

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"MACRO FACTOR MODEL v1 -- {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"Walk-forward: {WF_TRAIN_PERIODS} train periods (~{WF_TRAIN_DAYS}d), "
           f"biweekly rebalance, sliding window")
    fprint(f"VIX > 20 regime filter | Bull call spreads | Top 3 sectors | Hold to expiry")
    fprint()

    # ── 1. Download data ──
    close, high, low = download_data()

    # ── 2. ATR ──
    atr_dict = compute_atr_series(high, low, close)

    # ── 3. Rebalance dates ──
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ).last().dropna().values
    )
    fprint(f"Rebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # ══════════════════════════════════════════════════════════════
    # BUILD FEATURE RECORDS (shared base -- all 31 features)
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "=" * 80)
    fprint("BUILDING FEATURE RECORDS (all 31 features)")
    fprint("=" * 80)

    all_records = build_feature_records(close, high, low, rebal_dates, ALL_MACRO_FEATURES)

    if len(all_records) < 100:
        fprint("ERROR: Insufficient data. Exiting.")
        return

    # ══════════════════════════════════════════════════════════════
    # VARIANT A: Baseline (Production v4 -- 21 features)
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "=" * 80)
    fprint("VARIANT A: BASELINE -- Production v4 (21 features)")
    fprint("=" * 80)

    rankings_a, imp_a = walk_forward_lgbm_rank(all_records, PROD_V4_FEATURES)
    trades_a, eq_a = simulate_bull_call_spreads(rankings_a, close, atr_dict)

    # ══════════════════════════════════════════════════════════════
    # VARIANT B: All macro (21 + 10 = 31 features)
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "=" * 80)
    fprint("VARIANT B: ALL MACRO -- 21 + 10 = 31 features")
    fprint("=" * 80)

    rankings_b, imp_b = walk_forward_lgbm_rank(all_records, ALL_MACRO_FEATURES)
    trades_b, eq_b = simulate_bull_call_spreads(rankings_b, close, atr_dict)

    # ══════════════════════════════════════════════════════════════
    # VARIANT C: Best macro subset (21 + top 5 macro from B)
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "=" * 80)
    fprint("VARIANT C: BEST MACRO SUBSET -- 21 + top 5 macro by importance")
    fprint("=" * 80)

    if imp_b is not None:
        # Find top 5 macro features by importance from variant B
        macro_imp = imp_b[imp_b["feature"].isin(MACRO_FEATURES)]
        top5_macro = macro_imp.head(5)["feature"].tolist()
        fprint(f"  Top 5 macro features: {top5_macro}")
    else:
        top5_macro = MACRO_FEATURES[:5]
        fprint(f"  No importance data, using first 5: {top5_macro}")

    subset_features = PROD_V4_FEATURES + top5_macro
    rankings_c, imp_c = walk_forward_lgbm_rank(all_records, subset_features)
    trades_c, eq_c = simulate_bull_call_spreads(rankings_c, close, atr_dict)

    # ══════════════════════════════════════════════════════════════
    # VARIANT D: Macro only (10 features)
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "=" * 80)
    fprint("VARIANT D: MACRO ONLY -- 10 macro features (standalone)")
    fprint("=" * 80)

    rankings_d, imp_d = walk_forward_lgbm_rank(all_records, MACRO_FEATURES)
    trades_d, eq_d = simulate_bull_call_spreads(rankings_d, close, atr_dict)

    # ══════════════════════════════════════════════════════════════
    # VARIANT E: Regime-conditional macro
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "=" * 80)
    fprint("VARIANT E: REGIME-CONDITIONAL MACRO -- VIX>25 vs VIX<25 subsets")
    fprint("=" * 80)

    rankings_e, imp_e = walk_forward_lgbm_rank_regime_conditional(all_records, close)
    trades_e, eq_e = simulate_bull_call_spreads(rankings_e, close, atr_dict)

    # ══════════════════════════════════════════════════════════════
    # COLLECT AND VALIDATE ALL VARIANTS
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "=" * 80)
    fprint("ADVERSARIAL VALIDATION (5-gate)")
    fprint("=" * 80)

    variants = {
        "A_baseline_v4": {
            "desc": "Production v4 (21 features, no macro)",
            "trades": trades_a, "equity": eq_a, "imp": imp_a,
            "n_features": len(PROD_V4_FEATURES),
        },
        "B_all_macro": {
            "desc": "21 + 10 macro = 31 features",
            "trades": trades_b, "equity": eq_b, "imp": imp_b,
            "n_features": len(ALL_MACRO_FEATURES),
        },
        "C_best_macro_subset": {
            "desc": f"21 + top 5 macro: {top5_macro}",
            "trades": trades_c, "equity": eq_c, "imp": imp_c,
            "n_features": len(subset_features),
        },
        "D_macro_only": {
            "desc": "10 macro features only (standalone)",
            "trades": trades_d, "equity": eq_d, "imp": imp_d,
            "n_features": len(MACRO_FEATURES),
        },
        "E_regime_conditional": {
            "desc": "Regime-conditional: credit/vol macro VIX>25, momentum macro VIX<25",
            "trades": trades_e, "equity": eq_e, "imp": imp_e,
            "n_features": 26,  # 21 + 5 regime-specific
        },
    }

    all_results = {}

    for name, vdata in variants.items():
        fprint(f"\n--- {name}: {vdata['desc']} ---")
        trades = vdata["trades"]
        final_eq = vdata["equity"]
        imp_df = vdata["imp"]

        fprint(f"  Trades: {len(trades)}, Final equity: ${final_eq:,.0f}")

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades -- skipping validation")
            all_results[name] = {
                "description": vdata["desc"],
                "n_features": vdata["n_features"],
                "n_trades": len(trades) if trades else 0,
                "final_equity": round(final_eq, 2),
                "error": "insufficient_trades",
            }
            continue

        # 5-gate adversarial validation
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=name,
        )
        result.print_summary()

        rd = result.to_dict()
        rd.update({
            "description": vdata["desc"],
            "n_features": vdata["n_features"],
            "total_return_pct": round((final_eq / CAP - 1) * 100, 1),
        })

        # Feature importance top 10
        if imp_df is not None:
            top10 = imp_df.head(10).to_dict(orient="records")
            for rec in top10:
                rec["importance"] = round(rec["importance"], 2)
            rd["top_10_features"] = top10
            fprint("  Top 10 features by importance:")
            for rec in top10:
                fprint(f"    {rec['feature']}: {rec['importance']:.1f}")

        all_results[name] = rd

    # ══════════════════════════════════════════════════════════════
    # RESULTS SUMMARY TABLE (sorted by Sharpe)
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "=" * 80)
    fprint("RESULTS SUMMARY -- SORTED BY SHARPE")
    fprint("=" * 80)

    sortable = [(k, v) for k, v in all_results.items() if "error" not in v]
    sortable.sort(key=lambda x: x[1].get("sharpe", 0), reverse=True)

    fprint(f"\n{'Variant':<28} {'#Feat':>5} {'Sharpe':>7} {'Sortino':>8} "
           f"{'PF':>6} {'WR%':>6} {'MDD%':>7} {'TotRet%':>8} "
           f"{'#Trades':>8} {'Gates':>6}")
    fprint("-" * 105)

    for name, rd in sortable:
        fprint(f"{name:<28} {rd['n_features']:>5} {rd['sharpe']:>7.2f} "
               f"{rd['sortino']:>8.2f} {rd['profit_factor']:>6.2f} "
               f"{rd['win_rate']*100:>6.1f} {rd['max_dd']*100:>7.1f} "
               f"{rd['total_return_pct']:>8.1f} {rd['n_trades']:>8} "
               f"{rd['gates_passed']}/{rd['gates_total']}")

    errored = [(k, v) for k, v in all_results.items() if "error" in v]
    for name, rd in errored:
        fprint(f"{name:<28} {rd['n_features']:>5} {'--':>7} {'--':>8} "
               f"{'--':>6} {'--':>6} {'--':>7} {'--':>8} "
               f"{rd['n_trades']:>8} {'--':>6}")

    # ── Feature importance comparison ──
    fprint("\n" + "=" * 80)
    fprint("FEATURE IMPORTANCE COMPARISON")
    fprint("=" * 80)

    for name in ["A_baseline_v4", "B_all_macro"]:
        if name in all_results and "top_10_features" in all_results[name]:
            fprint(f"\n  {name}:")
            for rec in all_results[name]["top_10_features"]:
                fprint(f"    {rec['feature']:<30} {rec['importance']:>8.1f}")

    # Macro feature value analysis: compare B vs A
    if "A_baseline_v4" in all_results and "B_all_macro" in all_results:
        a = all_results["A_baseline_v4"]
        b = all_results["B_all_macro"]
        if "error" not in a and "error" not in b:
            sharpe_delta = b["sharpe"] - a["sharpe"]
            fprint(f"\n  MACRO IMPACT: Sharpe delta (B - A) = {sharpe_delta:+.3f}")
            if sharpe_delta > 0.1:
                fprint("  VERDICT: Macro factors provide meaningful improvement")
            elif sharpe_delta > -0.1:
                fprint("  VERDICT: Macro factors are neutral (no significant change)")
            else:
                fprint("  VERDICT: Macro factors HURT performance (possible overfitting)")

    # ── Best variant ──
    if sortable:
        best_name, best_rd = sortable[0]
        fprint(f"\n  BEST VARIANT: {best_name}")
        fprint(f"    Sharpe {best_rd['sharpe']:.2f}, Sortino {best_rd['sortino']:.2f}, "
               f"PF {best_rd['profit_factor']:.2f}, WR {best_rd['win_rate']*100:.1f}%, "
               f"MDD {best_rd['max_dd']*100:.1f}%, Return {best_rd['total_return_pct']:.1f}%, "
               f"Gates {best_rd['gates_passed']}/{best_rd['gates_total']}")

    # ── Save JSON ──
    output = {
        "metadata": {
            "script": "macro_factor_model_v1.py",
            "run_date": t0.strftime("%Y-%m-%d %H:%M:%S"),
            "capital": CAP,
            "dte": DTE,
            "spread_pct": SPREAD_PCT,
            "haircut": DEFAULT_HAIRCUT,
            "commission": COMMISSION_RT_SPREAD,
            "wf_train_periods": WF_TRAIN_PERIODS,
            "rebal_freq": WF_REBAL_FREQ,
            "data_range": f"{close.index[0].date()} to {close.index[-1].date()}",
            "n_sectors": len(SECTORS),
            "prod_v4_features": PROD_V4_FEATURES,
            "macro_features": MACRO_FEATURES,
            "top5_macro_subset": top5_macro,
        },
        "results": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"macro_v1_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "commission": COMMISSION_RT_SPREAD,
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "n_variants": len(variants),
                    "n_prod_features": len(PROD_V4_FEATURES),
                    "n_macro_features": len(MACRO_FEATURES),
                    "top5_macro": str(top5_macro),
                })

                for name, rd in all_results.items():
                    prefix = name.replace(" ", "_")
                    if "error" not in rd:
                        mlflow.log_metrics({
                            f"{prefix}_sharpe": rd.get("sharpe", 0),
                            f"{prefix}_sortino": rd.get("sortino", 0),
                            f"{prefix}_n_trades": rd.get("n_trades", 0),
                            f"{prefix}_maxdd": rd.get("max_dd", 0),
                            f"{prefix}_wr": rd.get("win_rate", 0),
                            f"{prefix}_pf": rd.get("profit_factor", 0),
                            f"{prefix}_total_return": rd.get("total_return_pct", 0),
                            f"{prefix}_gates": rd.get("gates_passed", 0),
                        })

                mlflow.log_artifact(str(RESULTS_PATH))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
