#!/usr/bin/env python3
"""
Correlation Regime V1 — Cross-Sector Correlation as Regime Filter
===================================================================

Research question: Does a correlation-based regime filter improve or complement
the VIX-based filter used in V6?

Background: V6 uses VIX as the sole regime filter (VIX<20 = pair trades,
VIX>=20 = bull only). But sector correlations also change — during risk-off
periods all sectors correlate highly, reducing the value of sector selection.
When correlations are low, sector selection is more powerful.

Correlation measure: Average pairwise Pearson correlation of 21-day returns
across all 11 sectors.

6 Variants (all with V6 structure: weekly, 2% OTM, 17 features, pairs):
  A: Baseline V6 (VIX-only regime filter)
  B: Correlation regime: trade pairs when avg_pairwise_corr < 0.5, bull-only when >= 0.5
  C: VIX + correlation dual filter: pairs only when VIX<20 AND corr<0.5
  D: Correlation-scaled sizing: more capital when sectors are less correlated
  E: Rolling decorrelation: use 3 LEAST correlated sectors for pairs instead of top/bottom LGBM
  F: Regime-adaptive k: trade more sectors when correlations are low, fewer when high

Based on production_v6_candidate_v1.py (canonical base).

Honest pricing rules:
  - HOLD TO EXPIRY ONLY (no early exit)
  - At expiry: INTRINSIC VALUE ONLY (no time value)
  - 15% haircut on ENTRY only (automatic exercise at expiry = no exit haircut)
  - DTE=21, $645 starting capital
  - Walk-forward LGBM, weekly rebalance, regime filter via GRU
  - Commission: $2.60/spread

Full 5-gate adversarial validation + random baseline comparison (5 trials) per variant.
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_builtin_print = print


def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# -- Standardized tools --
sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread,
    price_bear_put_spread,
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

# -- Config --
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "correlation_regime_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4

# V6 defaults
REBAL_FREQ = "W-FRI"
OTM_PCT = 0.02

# Correlation thresholds
CORR_THRESHOLD = 0.5  # avg pairwise correlation threshold
CORR_LOOKBACK = 21    # 21 trading days for rolling correlation

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "correlation_regime_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable -- results saved to disk only")


# ====================================================================
# VARIANT DEFINITIONS
# ====================================================================

VARIANTS = {
    "A_baseline_v6": {
        "desc": "V6 baseline: weekly, 2% OTM, VIX-only regime (VIX<20=pairs, >=20=bull)",
        "corr_filter": False,
        "vix_filter": True,
        "corr_sizing": False,
        "decorr_picks": False,
        "adaptive_k": False,
    },
    "B_corr_regime": {
        "desc": "Correlation regime: pairs when corr<0.5, bull-only when >=0.5",
        "corr_filter": True,
        "vix_filter": False,
        "corr_sizing": False,
        "decorr_picks": False,
        "adaptive_k": False,
    },
    "C_dual_filter": {
        "desc": "Dual filter: pairs only when VIX<20 AND corr<0.5",
        "corr_filter": True,
        "vix_filter": True,
        "corr_sizing": False,
        "decorr_picks": False,
        "adaptive_k": False,
    },
    "D_corr_sizing": {
        "desc": "Corr-scaled sizing: more capital when sectors less correlated (VIX regime)",
        "corr_filter": False,
        "vix_filter": True,
        "corr_sizing": True,
        "decorr_picks": False,
        "adaptive_k": False,
    },
    "E_decorr_picks": {
        "desc": "Rolling decorrelation: 3 LEAST correlated sectors for pairs (VIX regime)",
        "corr_filter": False,
        "vix_filter": True,
        "corr_sizing": False,
        "decorr_picks": True,
        "adaptive_k": False,
    },
    "F_adaptive_k": {
        "desc": "Adaptive k: more sectors when corr low, fewer when high (VIX regime)",
        "corr_filter": False,
        "vix_filter": True,
        "corr_sizing": False,
        "decorr_picks": False,
        "adaptive_k": True,
    },
}


# ====================================================================
# DATA DOWNLOAD
# ====================================================================

def download_data():
    """Download all required tickers via yfinance."""
    import yfinance as yf

    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")

    raw = yf.download(all_tickers, start="2008-01-01", progress=False, auto_adjust=True)
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

    needed = ["SPY", "VIX"]
    for t in needed:
        if t not in close.columns:
            raise ValueError(f"Missing critical ticker: {t}")

    fprint(f"Data: {len(close)} days, columns: {list(close.columns)}")
    fprint(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    return close, high, low


# ====================================================================
# CORRELATION COMPUTATION
# ====================================================================

def compute_rolling_avg_pairwise_corr(close, lookback=21):
    """
    Compute rolling average pairwise Pearson correlation of returns
    across all 11 sectors, using a lookback window.

    Returns a pd.Series indexed by date with the average pairwise correlation.
    """
    sector_cols = [c for c in SECTORS if c in close.columns]
    sector_rets = close[sector_cols].pct_change().dropna()

    fprint(f"  Computing rolling {lookback}d avg pairwise correlation across {len(sector_cols)} sectors...")

    # Compute rolling correlation matrix and average off-diagonal elements
    n_pairs = len(sector_cols) * (len(sector_cols) - 1) // 2
    avg_corrs = []
    dates = []

    for i in range(lookback, len(sector_rets)):
        window = sector_rets.iloc[i - lookback:i]
        corr_mat = window.corr()
        # Average of off-diagonal (upper triangle) elements
        mask = np.triu(np.ones(corr_mat.shape, dtype=bool), k=1)
        avg_corr = corr_mat.values[mask].mean()
        avg_corrs.append(avg_corr)
        dates.append(sector_rets.index[i])

    corr_series = pd.Series(avg_corrs, index=pd.DatetimeIndex(dates), name="avg_pairwise_corr")

    fprint(f"    Computed: {len(corr_series)} days")
    fprint(f"    Mean corr: {corr_series.mean():.3f}, Median: {corr_series.median():.3f}")
    fprint(f"    Min: {corr_series.min():.3f}, Max: {corr_series.max():.3f}")
    fprint(f"    Days < 0.5: {(corr_series < CORR_THRESHOLD).sum()} "
           f"({(corr_series < CORR_THRESHOLD).mean()*100:.1f}%)")
    fprint(f"    Days >= 0.5: {(corr_series >= CORR_THRESHOLD).sum()} "
           f"({(corr_series >= CORR_THRESHOLD).mean()*100:.1f}%)")

    return corr_series


def get_corr_at(corr_series, dt):
    """Get average pairwise correlation at a given date, with nearest-date fallback."""
    if corr_series is None:
        return 0.5
    if dt in corr_series.index:
        return float(corr_series.loc[dt])
    nearest = corr_series.index[corr_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0 and not pd.isna(nearest[0]):
        return float(corr_series.loc[nearest[0]])
    return 0.5


def compute_sector_pairwise_corrs_at(close, dt_idx, lookback=21):
    """
    Compute pairwise correlations between all sectors at a specific date index.
    Returns a dict of {(sector_a, sector_b): correlation} for all pairs.
    """
    sector_cols = [c for c in SECTORS if c in close.columns]
    start = max(0, dt_idx - lookback)
    window = close[sector_cols].iloc[start:dt_idx + 1].pct_change().dropna()

    if len(window) < 10:
        return {}

    corr_mat = window.corr()
    result = {}
    for i, s1 in enumerate(sector_cols):
        for j, s2 in enumerate(sector_cols):
            if i < j:
                result[(s1, s2)] = float(corr_mat.loc[s1, s2])
    return result


def find_least_correlated_sectors(close, dt_idx, k=3, lookback=21):
    """
    Find the k sectors that are LEAST correlated with each other.
    Uses a greedy algorithm: start with the pair with lowest correlation,
    then add sectors that minimize average correlation with selected set.
    """
    sector_cols = [c for c in SECTORS if c in close.columns]
    start = max(0, dt_idx - lookback)
    window = close[sector_cols].iloc[start:dt_idx + 1].pct_change().dropna()

    if len(window) < 10:
        return sector_cols[:k]

    corr_mat = window.corr()

    # Find the pair with the lowest correlation
    min_corr = 2.0
    best_pair = (sector_cols[0], sector_cols[1])
    for i, s1 in enumerate(sector_cols):
        for j, s2 in enumerate(sector_cols):
            if i < j:
                c = abs(corr_mat.loc[s1, s2])
                if c < min_corr:
                    min_corr = c
                    best_pair = (s1, s2)

    selected = list(best_pair)

    # Greedily add sectors that minimize avg correlation with selected
    while len(selected) < k and len(selected) < len(sector_cols):
        best_next = None
        best_avg_corr = 2.0
        for s in sector_cols:
            if s in selected:
                continue
            avg_corr = np.mean([abs(corr_mat.loc[s, sel]) for sel in selected])
            if avg_corr < best_avg_corr:
                best_avg_corr = avg_corr
                best_next = s
        if best_next:
            selected.append(best_next)
        else:
            break

    return selected


# ====================================================================
# REGIME LOADING
# ====================================================================

def load_regime_predictions():
    """Load GRU regime predictions and build a date-indexed Series."""
    if not REGIME_FILE.exists():
        fprint(f"WARNING: Regime file not found at {REGIME_FILE}")
        fprint("  Will use VIX-based regime proxy instead")
        return None

    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
    fprint(f"Regime predictions loaded: {len(regime_series)} days "
           f"({regime_series.index[0].date()} to {regime_series.index[-1].date()})")
    fprint(f"  Mean score: {regime_series.mean():.3f}, "
           f"Days >0.4: {(regime_series > REGIME_BULL_THRESHOLD).sum()}")
    return regime_series


def get_regime_score_at(regime_series, dt):
    """Get regime score at a given date, with nearest-date fallback."""
    if regime_series is None:
        return 0.5
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


# ====================================================================
# FEATURE ENGINEERING (V6: 17 legacy + 3 cross-asset = 20 total, but
# user said "17 features" -- we use the standard V4/V6 feature set)
# ====================================================================

LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]

VALIDATED_CROSS_ASSET = [
    "sector_spy_beta_63d",
    "sector_relative_vol_21d",
    "cross_sector_dispersion",
]

V6_FEATURES = LEGACY_FEATURES + VALIDATED_CROSS_ASSET  # 21 total


def compute_legacy_features(px, spy_slice):
    """Compute the 18 legacy quality-momentum features for a single sector ETF."""
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
    """Compute the 3 validated cross-asset features only."""
    f = {}

    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {k: 0.0 for k in VALIDATED_CROSS_ASSET}

    spy_ret = spy.pct_change().dropna()

    if sector_px is not None and len(sector_px) > 63:
        sec_ret = sector_px.pct_change().dropna()
        common = spy_ret.index.intersection(sec_ret.index)
        if len(common) > 63:
            sr = sec_ret.loc[common].iloc[-63:]
            mr = spy_ret.loc[common].iloc[-63:]
            cov = np.cov(sr.values, mr.values)
            beta = cov[0, 1] / (cov[1, 1] + 1e-10)
            f["sector_spy_beta_63d"] = float(beta)
        else:
            f["sector_spy_beta_63d"] = 1.0
    else:
        f["sector_spy_beta_63d"] = 1.0

    if sector_px is not None and len(sector_px) > 21:
        sec_ret = sector_px.pct_change().dropna()
        if len(sec_ret) > 21 and len(spy_ret) > 21:
            sec_vol = sec_ret.iloc[-21:].std()
            spy_vol = spy_ret.iloc[-21:].std()
            f["sector_relative_vol_21d"] = float(sec_vol / (spy_vol + 1e-10))
        else:
            f["sector_relative_vol_21d"] = 1.0
    else:
        f["sector_relative_vol_21d"] = 1.0

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


# ====================================================================
# WALK-FORWARD LGBM RANKING
# ====================================================================

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """
    Build feature + target records for all sectors on all rebal dates.
    Uses bull_only regime mode (regime>0.4) -- same as V6.
    """
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        rscore = get_regime_score_at(regime_series, dt)
        if rscore <= REGIME_BULL_THRESHOLD:
            continue

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            cross_asset = {}
            for col in feature_cols:
                if col in VALIDATED_CROSS_ASSET:
                    cross_asset = compute_cross_asset_features(tk, idx, close)
                    break

            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


def walk_forward_lgbm_rank(df, feature_cols, variant_name):
    """Walk-forward LGBM ranking: sliding train, predict next period."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    {variant_name}: Insufficient data ({len(df)} records)")
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
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=5,
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

    fprint(f"    {variant_name}: {len(rankings)} ranking dates, {n_models} models trained")
    return rankings, imp_df


# ====================================================================
# ATR COMPUTATION
# ====================================================================

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
                atr_dict[tk] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_dict


# ====================================================================
# STRIKE COMPUTATION (OTM)
# ====================================================================

def compute_strikes(S, direction, otm_pct, spread_pct):
    """Compute strike prices for a spread (same as V6)."""
    if direction == "bull":
        if otm_pct > 0:
            K1 = round(S * (1 + otm_pct), 2)
            K2 = round(K1 * (1 + spread_pct / 100), 2)
        else:
            K1 = round(S, 2)
            K2 = round(S * (1 + spread_pct / 100), 2)
    else:
        if otm_pct > 0:
            K2 = round(S * (1 - otm_pct), 2)
            K1 = round(K2 * (1 - spread_pct / 100), 2)
        else:
            K1 = round(S * (1 - spread_pct / 100), 2)
            K2 = round(S, 2)

    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


# ====================================================================
# REBALANCE DATE GENERATION
# ====================================================================

def generate_rebal_dates(close, freq_str):
    """Generate rebalance dates from close index."""
    if freq_str == "3B":
        bdays = close.index[close.index.dayofweek < 5]
        rebal_dates = pd.DatetimeIndex([bdays[i] for i in range(0, len(bdays), 3)])
    else:
        rebal_dates = pd.DatetimeIndex(
            close.index.to_series().resample(freq_str).last().dropna().values
        )
    return rebal_dates


# ====================================================================
# SINGLE TRADE EXECUTION
# ====================================================================

def _execute_single_trade(tk, dt, direction, max_pos, close, atr_dict, cv, equity):
    """Execute a single spread trade. Returns PnL or None."""
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    K1, K2 = compute_strikes(S, direction, OTM_PCT, SPREAD_PCT)

    try:
        if direction == "bull":
            entry_cost_ps, max_profit_ps = price_bull_call_spread(
                S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
            )
        else:
            entry_cost_ps, max_profit_ps = price_bear_put_spread(
                S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
            )
    except Exception:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    Se = float(close[tk].iloc[ei])

    if direction == "bull":
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    else:
        intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

    pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
    return pnl


# ====================================================================
# TRADE SIMULATION (VARIANT-AWARE)
# ====================================================================

def simulate_trades(name, rankings, close, high, low, atr_dict,
                    corr_series, variant_cfg):
    """
    Simulate trades for a given variant configuration.

    Variant logic:
      A (baseline): VIX<20 = pairs, VIX>=20 = bull-only
      B (corr regime): corr<0.5 = pairs, corr>=0.5 = bull-only
      C (dual filter): VIX<20 AND corr<0.5 = pairs, else bull-only
      D (corr sizing): VIX regime, but position size scaled by (1 - corr)
      E (decorr picks): VIX regime, but bear picks = least correlated sectors
      F (adaptive k): VIX regime, but k = f(corr): low corr -> more sectors
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    use_vix_filter = variant_cfg["vix_filter"]
    use_corr_filter = variant_cfg["corr_filter"]
    use_corr_sizing = variant_cfg["corr_sizing"]
    use_decorr_picks = variant_cfg["decorr_picks"]
    use_adaptive_k = variant_cfg["adaptive_k"]

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        # Get correlation at this date
        corr_val = get_corr_at(corr_series, dt)

        # ---- Determine trade mode ----
        if use_corr_filter and use_vix_filter:
            # Variant C: dual filter
            if cv < 20.0 and corr_val < CORR_THRESHOLD:
                trade_mode = "pairs"
            else:
                trade_mode = "bull_only"
        elif use_corr_filter:
            # Variant B: correlation-only regime
            if corr_val < CORR_THRESHOLD:
                trade_mode = "pairs"
            else:
                trade_mode = "bull_only"
        elif use_vix_filter:
            # Variants A, D, E, F: VIX regime
            if cv < 20.0:
                trade_mode = "pairs"
            else:
                trade_mode = "bull_only"
        else:
            trade_mode = "bull_only"

        # ---- Determine k (number of sectors) ----
        if use_adaptive_k:
            # Variant F: k = f(corr)
            # Low corr (0.0-0.3) -> k=5, medium (0.3-0.6) -> k=3, high (0.6+) -> k=2
            if corr_val < 0.3:
                k = 5
            elif corr_val < 0.6:
                k = 3
            else:
                k = 2
        else:
            k = TOP_K

        # ---- Pick sectors ----
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        bull_picks = [t for t, _ in ranked_desc[:k]]

        if trade_mode == "pairs":
            if use_decorr_picks:
                # Variant E: bear picks = least correlated sectors
                di = close.index.get_loc(dt)
                decorr_sectors = find_least_correlated_sectors(close, di, k=k, lookback=CORR_LOOKBACK)
                # Use least correlated sectors for bear leg (they diverge most in low-corr regimes)
                bear_picks = decorr_sectors[:k]
            else:
                bear_picks = [t for t, _ in ranked_asc[:k]]
        else:
            bear_picks = []

        # ---- Position sizing ----
        if trade_mode == "pairs":
            n_positions = 2 * k  # k bull + k bear
            base_max_pos = min(100, equity / max(n_positions, 1))
        else:
            base_max_pos = min(200, equity / max(k, 1))

        if use_corr_sizing:
            # Variant D: scale position size by (1 - corr)
            # Low correlation -> more capital (up to 1.5x), high -> less (down to 0.5x)
            corr_scale = 0.5 + (1.0 - corr_val)  # range: ~0.5 to ~1.5
            corr_scale = max(0.5, min(1.5, corr_scale))
            max_pos = base_max_pos * corr_scale
        else:
            max_pos = base_max_pos

        if max_pos < 30:
            continue

        # ---- Execute bull leg ----
        for tk in bull_picks:
            pnl = _execute_single_trade(tk, dt, "bull", max_pos, close, atr_dict, cv, equity)
            if pnl is not None:
                equity += pnl
                di = close.index.get_loc(dt)
                ei = min(di + DTE, len(close) - 1)
                sv = float(spy.loc[dt])
                se = float(spy.iloc[ei]) if ei < len(spy) else sv
                spy_regime = "bull" if se >= sv else "bear"
                trades.append({
                    "pnl": round(pnl, 2),
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": spy_regime,
                    "direction": "bull",
                    "vix": round(cv, 1),
                    "corr": round(corr_val, 3),
                    "win": pnl > 0,
                    "trade_mode": trade_mode,
                })

        # ---- Execute bear leg (pairs mode only) ----
        for tk in bear_picks:
            pnl = _execute_single_trade(tk, dt, "bear", max_pos, close, atr_dict, cv, equity)
            if pnl is not None:
                equity += pnl
                di = close.index.get_loc(dt)
                ei = min(di + DTE, len(close) - 1)
                sv = float(spy.loc[dt])
                se = float(spy.iloc[ei]) if ei < len(spy) else sv
                spy_regime = "bull" if se >= sv else "bear"
                trades.append({
                    "pnl": round(pnl, 2),
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": spy_regime,
                    "direction": "bear",
                    "vix": round(cv, 1),
                    "corr": round(corr_val, 3),
                    "win": pnl > 0,
                    "trade_mode": trade_mode,
                })

    return trades, equity


# ====================================================================
# RANDOM BASELINE
# ====================================================================

def random_baseline_test(rankings, close, high, low, atr_dict,
                         corr_series, variant_cfg, n_trials=5):
    """Test if random sector selection also produces similar returns."""
    fprint(f"\n  Random baseline test ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_scores = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = rand_scores

        trades, final_eq = simulate_trades(
            f"Random_{trial}", rand_rankings, close, high, low, atr_dict,
            corr_series, variant_cfg,
        )

        if trades and len(trades) >= 10:
            result = validate_trades(
                trades, initial_capital=CAP,
                spy_prices=close["SPY"],
                strategy_name=f"Random_{trial}",
                n_perms=500,
            )
            random_sharpes.append(result.sharpe)
            fprint(f"    Random trial {trial}: Sharpe {result.sharpe:.2f}, "
                   f"${CAP:.0f}->${final_eq:.0f}")
        else:
            random_sharpes.append(0.0)

    return random_sharpes


# ====================================================================
# CORRELATION REGIME ANALYSIS (extra diagnostics)
# ====================================================================

def correlation_regime_analysis(corr_series, close):
    """Print analysis of the correlation regime over time."""
    fprint("\n" + "=" * 80)
    fprint("CORRELATION REGIME ANALYSIS")
    fprint("=" * 80)

    vix = close["VIX"] if "VIX" in close.columns else None

    # Yearly stats
    yearly = corr_series.resample("YE").agg(["mean", "median", "min", "max"])
    fprint("\n  Yearly avg pairwise correlation:")
    fprint(f"  {'Year':<6} {'Mean':>6} {'Median':>7} {'Min':>6} {'Max':>6}")
    fprint("  " + "-" * 34)
    for idx, row in yearly.iterrows():
        fprint(f"  {idx.year:<6} {row['mean']:>6.3f} {row['median']:>7.3f} "
               f"{row['min']:>6.3f} {row['max']:>6.3f}")

    # VIX vs correlation relationship
    if vix is not None:
        common = corr_series.index.intersection(vix.index)
        if len(common) > 100:
            c = corr_series.loc[common]
            v = vix.loc[common]
            pearson_corr = c.corr(v)
            fprint(f"\n  VIX vs avg sector correlation: Pearson r = {pearson_corr:.3f}")

            # Quadrant analysis
            low_vix_low_corr = ((v < 20) & (c < CORR_THRESHOLD)).sum()
            low_vix_high_corr = ((v < 20) & (c >= CORR_THRESHOLD)).sum()
            high_vix_low_corr = ((v >= 20) & (c < CORR_THRESHOLD)).sum()
            high_vix_high_corr = ((v >= 20) & (c >= CORR_THRESHOLD)).sum()
            total = len(common)

            fprint(f"\n  VIX vs Correlation Quadrants (% of days):")
            fprint(f"                       VIX < 20      VIX >= 20")
            fprint(f"    Corr < 0.5        {low_vix_low_corr:>5} ({low_vix_low_corr/total*100:>5.1f}%)"
                   f"    {high_vix_low_corr:>5} ({high_vix_low_corr/total*100:>5.1f}%)")
            fprint(f"    Corr >= 0.5       {low_vix_high_corr:>5} ({low_vix_high_corr/total*100:>5.1f}%)"
                   f"    {high_vix_high_corr:>5} ({high_vix_high_corr/total*100:>5.1f}%)")
            fprint(f"\n  Key insight: {low_vix_high_corr/total*100:.1f}% of days have low VIX but HIGH "
                   f"correlation — V6 trades pairs here but sector selection may be weak")
            fprint(f"  Key insight: {high_vix_low_corr/total*100:.1f}% of days have high VIX but LOW "
                   f"correlation — V6 goes bull-only but pairs might work here")


# ====================================================================
# MAIN
# ====================================================================

def main():
    t0 = datetime.now()
    fprint("=" * 90)
    fprint(f"CORRELATION REGIME V1 -- Cross-Sector Correlation Filter -- {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 90)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | OTM: {OTM_PCT*100:.0f}%")
    fprint(f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Regime bull threshold: >{REGIME_BULL_THRESHOLD}")
    fprint(f"Correlation threshold: {CORR_THRESHOLD} | Lookback: {CORR_LOOKBACK}d")
    fprint(f"Rebalance: {REBAL_FREQ} | Testing {len(VARIANTS)} variants")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Compute rolling avg pairwise correlation
    corr_series = compute_rolling_avg_pairwise_corr(close, lookback=CORR_LOOKBACK)

    # 4. Correlation regime analysis (diagnostics)
    correlation_regime_analysis(corr_series, close)

    # 5. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 6. Build rebalance dates
    rebal_dates = generate_rebal_dates(close, REBAL_FREQ)
    fprint(f"\nRebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # 7. Build LGBM rankings (shared across all variants -- same features, same freq)
    fprint(f"\n{'=' * 80}")
    fprint("BUILDING LGBM RANKINGS (shared across all variants)")
    fprint(f"{'=' * 80}")

    records = build_feature_records(
        close, high, low, rebal_dates, V6_FEATURES, regime_series
    )
    rankings, imp_df = walk_forward_lgbm_rank(records, V6_FEATURES, "LGBM_shared")

    if not rankings:
        fprint("ERROR: No LGBM rankings produced. Exiting.")
        return

    # 8. Simulate all variants
    fprint(f"\n{'=' * 90}")
    fprint("SIMULATING ALL VARIANTS")
    fprint(f"{'=' * 90}")

    all_results = {}
    for vname, vcfg in VARIANTS.items():
        fprint(f"\n{'~' * 80}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"{'~' * 80}")

        trades, final_eq = simulate_trades(
            vname, rankings, close, high, low, atr_dict,
            corr_series, vcfg,
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            continue

        # 5-gate adversarial validation
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=vname,
        )
        result.print_summary()

        # Direction breakdown
        bull_trades = [t for t in trades if t["direction"] == "bull"]
        bear_trades = [t for t in trades if t["direction"] == "bear"]
        pair_trades = [t for t in trades if t.get("trade_mode") == "pairs"]
        bull_pnl = sum(t["pnl"] for t in bull_trades)
        bear_pnl = sum(t["pnl"] for t in bear_trades)
        bull_wr = sum(1 for t in bull_trades if t["win"]) / max(len(bull_trades), 1) * 100
        bear_wr = sum(1 for t in bear_trades if t["win"]) / max(len(bear_trades), 1) * 100
        fprint(f"  Direction breakdown:")
        fprint(f"    Bull: {len(bull_trades)} trades, WR {bull_wr:.1f}%, PnL ${bull_pnl:.0f}")
        fprint(f"    Bear: {len(bear_trades)} trades, WR {bear_wr:.1f}%, PnL ${bear_pnl:.0f}")
        fprint(f"    Pair-mode trades: {len(pair_trades)}")

        # Correlation regime breakdown
        if trades:
            low_corr_trades = [t for t in trades if t.get("corr", 0.5) < CORR_THRESHOLD]
            high_corr_trades = [t for t in trades if t.get("corr", 0.5) >= CORR_THRESHOLD]
            if low_corr_trades:
                lc_pnl = sum(t["pnl"] for t in low_corr_trades)
                lc_wr = sum(1 for t in low_corr_trades if t["win"]) / len(low_corr_trades) * 100
                fprint(f"    Low-corr (<0.5) trades: {len(low_corr_trades)}, WR {lc_wr:.1f}%, PnL ${lc_pnl:.0f}")
            if high_corr_trades:
                hc_pnl = sum(t["pnl"] for t in high_corr_trades)
                hc_wr = sum(1 for t in high_corr_trades if t["win"]) / len(high_corr_trades) * 100
                fprint(f"    High-corr (>=0.5) trades: {len(high_corr_trades)}, WR {hc_wr:.1f}%, PnL ${hc_pnl:.0f}")

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, atr_dict,
            corr_series, vcfg,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        all_results[vname] = {
            "description": vcfg["desc"],
            **result.to_dict(),
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "n_bull": len(bull_trades),
            "n_bear": len(bear_trades),
            "n_pair_mode": len(pair_trades),
            "bull_pnl": round(bull_pnl, 2),
            "bear_pnl": round(bear_pnl, 2),
            "bull_wr": round(bull_wr, 1),
            "bear_wr": round(bear_wr, 1),
        }

    # ---- SUMMARY COMPARISON ----
    fprint(f"\n{'=' * 120}")
    fprint("SUMMARY COMPARISON -- ALL 6 VARIANTS")
    fprint(f"{'=' * 120}")
    fprint(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'RandSh':>7} {'Bull':>5} {'Bear':>5}")
    fprint("-" * 120)

    for vname in VARIANTS.keys():
        r = all_results.get(vname)
        if not r:
            fprint(f"  {vname:<25} -- NO DATA --")
            continue
        fprint(f"  {vname:<25} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f} {r['random_mean_sharpe']:>7.2f} "
               f"{r.get('n_bull',0):>5} {r.get('n_bear',0):>5}")

    # ---- BASELINE COMPARISON ----
    fprint(f"\n{'=' * 90}")
    fprint("VS BASELINE ANALYSIS")
    fprint(f"{'=' * 90}")

    baseline = all_results.get("A_baseline_v6")
    if baseline:
        baseline_sharpe = baseline["sharpe"]
        fprint(f"\n  BASELINE (A) SHARPE: {baseline_sharpe:.2f}")
        for vname, r in sorted(all_results.items()):
            if vname == "A_baseline_v6":
                continue
            delta = r["sharpe"] - baseline_sharpe
            pct = (delta / max(abs(baseline_sharpe), 0.01)) * 100
            arrow = "+" if delta > 0 else ""
            fprint(f"    {vname:<25} {arrow}{delta:.2f} ({arrow}{pct:.0f}%) Sharpe | "
                   f"Gates: {r['gates_passed']}/{r['gates_total']}")

    # ---- RECOMMENDATION ----
    fprint(f"\n{'=' * 90}")
    fprint("RECOMMENDATION")
    fprint(f"{'=' * 90}")

    valid_results = {k: v for k, v in all_results.items() if v.get("sharpe", 0) > 0}
    if valid_results:
        gated = {k: v for k, v in valid_results.items()
                 if v.get("gates_passed", 0) == v.get("gates_total", 5)}
        if gated:
            best_name = max(gated.keys(), key=lambda k: gated[k]["sharpe"])
            best = gated[best_name]
            baseline_sharpe = all_results.get("A_baseline_v6", {}).get("sharpe", 0)
            delta = best["sharpe"] - baseline_sharpe
            if delta > 0:
                fprint(f"  UPGRADE to {best_name}: +{delta:.2f} Sharpe over V6 baseline")
                fprint(f"  Config: {best.get('description', '')}")
            else:
                fprint(f"  KEEP V6 BASELINE: Best gated variant ({best_name}, Sharpe {best['sharpe']:.2f}) "
                       f"does not beat V6 baseline ({baseline_sharpe:.2f})")
        else:
            fprint("  WARNING: No variant passes all 5 gates. Stick with V6 baseline.")
    else:
        fprint("  ERROR: No valid results produced.")

    # ---- CORRELATION INSIGHT ----
    fprint(f"\n{'=' * 90}")
    fprint("KEY CORRELATION INSIGHTS")
    fprint(f"{'=' * 90}")

    # Compare low-corr vs high-corr performance across variants
    for vname, r in all_results.items():
        fprint(f"\n  {vname}:")
        fprint(f"    Bull WR: {r.get('bull_wr', 0):.1f}% ({r.get('n_bull', 0)} trades)")
        fprint(f"    Bear WR: {r.get('bear_wr', 0):.1f}% ({r.get('n_bear', 0)} trades)")

    # Feature importance
    fprint(f"\n{'=' * 80}")
    fprint("FEATURE IMPORTANCE (Top 10) -- shared LGBM model")
    fprint(f"{'=' * 80}")
    if imp_df is not None:
        for _, row in imp_df.head(10).iterrows():
            bar = "*" * int(row["importance"] / imp_df["importance"].max() * 30)
            fprint(f"    {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # Save results
    results_path = OUTPUT_DIR / "correlation_regime_v1_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save correlation series
    corr_path = OUTPUT_DIR / "avg_pairwise_corr_21d.csv"
    corr_series.to_csv(corr_path, header=True)
    fprint(f"Correlation series saved to {corr_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"corr_regime_v1_{t0.strftime('%Y%m%d_%H%M')}"):
                for vname, r in all_results.items():
                    prefix = vname.split("_")[0]
                    mlflow.log_metric(f"{prefix}_sharpe", r.get("sharpe", 0))
                    mlflow.log_metric(f"{prefix}_sortino", r.get("sortino", 0))
                    mlflow.log_metric(f"{prefix}_win_rate", r.get("win_rate", 0))
                    mlflow.log_metric(f"{prefix}_profit_factor", r.get("profit_factor", 0))
                    mlflow.log_metric(f"{prefix}_max_dd", r.get("max_dd", 0))
                    mlflow.log_metric(f"{prefix}_n_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{prefix}_gates_passed", r.get("gates_passed", 0))
                    mlflow.log_metric(f"{prefix}_final_equity", r.get("final_equity", 0))
                    mlflow.log_metric(f"{prefix}_random_mean_sharpe", r.get("random_mean_sharpe", 0))
                    mlflow.log_metric(f"{prefix}_n_bull", r.get("n_bull", 0))
                    mlflow.log_metric(f"{prefix}_n_bear", r.get("n_bear", 0))

                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "otm_pct": OTM_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "corr_threshold": CORR_THRESHOLD,
                    "corr_lookback": CORR_LOOKBACK,
                    "hold_to_expiry": True,
                    "entry_haircut_only": True,
                    "n_features": len(V6_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "rebal_freq": REBAL_FREQ,
                })

                mlflow.log_artifact(str(results_path))
                mlflow.log_artifact(str(corr_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
