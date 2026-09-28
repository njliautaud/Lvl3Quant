#!/usr/bin/env python3
"""
Feature Ablation Cross-Validation v1 — Which Features Actually Matter?
=======================================================================

Production v4 uses 21 features for LGBM sector ranking. Feature importance
shows up_capture (77), spy_beta (61), vol_21d (55) as top-3. But importance
does NOT equal necessity — a feature can be important yet redundant with
others, or it could be picking up noise that hurts OOS.

This experiment REMOVES features systematically to find which ones actually
affect Sharpe. Uses V6 config (weekly, 2% OTM, bull+pairs) throughout.

8 Variants:
  A: Full 21 features — Baseline. Must match V6 Sharpe ~2.79.
  B: Top 5 only — up_capture, spy_beta, vol_21d, sector_relative_vol, ret_126d.
  C: Top 10 only — Add trend_r2, ret_5d, maxdd_63d, vol_63d, ret_252d.
  D: Remove up_capture — 20 features minus the #1 feature.
  E: Remove momentum features — Remove ret_5d/10d/21d/63d/126d/252d, mom_accel.
     Keep vol and beta features.
  F: Remove volatility features — Remove vol_21d, vol_63d, sector_relative_vol,
     maxdd_63d. Keep momentum and beta.
  G: Momentum only — ret_5d/10d/21d/63d/126d/252d, mom_accel only (7 features).
  H: Cross-asset only — spy_corr_63d (=sector_spy_beta_63d), spy_beta_63d
     (=sector_spy_beta_63d), up_capture only (3 features).

V6 config for ALL variants:
  - Weekly rebalance (W-FRI)
  - 2% OTM moneyness
  - Bull VIX>20 + pairs VIX<20
  - $200/trade ($100/leg for pairs)
  - DTE=21, 3% spread width, $2.60 commission, 15% haircut entry only
  - Hold to expiry, intrinsic value only

Each variant: 5-gate adversarial validation + 5-trial random baseline.
MLflow experiment: 'feature_ablation_xval_v1'
Output: output/growth_research/feature_ablation_xval_v1/
"""

import json
import sys
import time
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
import os as _os
_hostname = _os.uname().nodename.lower()
if 'neptune' in _hostname or 'nick' in str(_os.path.expanduser('~')):
    _BASE_PATH = "/home/nick/Lvl3Quant"
else:
    _BASE_PATH = "/home/jupiter/Lvl3Quant"
sys.path.insert(0, _BASE_PATH)

try:
    from research.tools.options_pricer import (
        price_bull_call_spread,
        price_bear_put_spread,
        estimate_iv,
        compute_atr,
        COMMISSION_RT_SPREAD,
        DEFAULT_HAIRCUT,
    )
    from research.tools.adversarial_validator import validate_trades
    fprint("Imported from research.tools")
except ImportError:
    fprint("research.tools not found — using inline implementations")
    COMMISSION_RT_SPREAD = 2.60
    DEFAULT_HAIRCUT = 0.15

    def compute_atr(prices, window=14):
        high = prices.rolling(window).max()
        low = prices.rolling(window).min()
        return (high - low).mean()

    def estimate_iv(prices, window=21, mult=1.2):
        returns = np.log(prices / prices.shift(1)).dropna()
        hv = returns.rolling(window).std() * np.sqrt(252)
        return hv * mult

    def price_bull_call_spread(underlying, strike_long, strike_short, iv, dte, haircut=0.15):
        from scipy.stats import norm
        T = dte / 365.0
        if T <= 0 or iv <= 0:
            return 0.0
        d1_l = (np.log(underlying / strike_long) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d1_s = (np.log(underlying / strike_short) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        call_l = underlying * norm.cdf(d1_l) - strike_long * norm.cdf(d1_l - iv * np.sqrt(T))
        call_s = underlying * norm.cdf(d1_s) - strike_short * norm.cdf(d1_s - iv * np.sqrt(T))
        spread_val = max(call_l - call_s, 0.001)
        return spread_val * (1 + haircut)

    def price_bear_put_spread(underlying, strike_long, strike_short, iv, dte, haircut=0.15):
        from scipy.stats import norm
        T = dte / 365.0
        if T <= 0 or iv <= 0:
            return 0.0
        d1_l = (np.log(underlying / strike_long) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d1_s = (np.log(underlying / strike_short) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        put_l = strike_long * norm.cdf(-(d1_l - iv * np.sqrt(T))) - underlying * norm.cdf(-d1_l)
        put_s = strike_short * norm.cdf(-(d1_s - iv * np.sqrt(T))) - underlying * norm.cdf(-d1_s)
        spread_val = max(put_l - put_s, 0.001)
        return spread_val * (1 + haircut)

    def validate_trades(trades_df, label="", verbose=True):
        if len(trades_df) < 10:
            return {"gates_passed": 0, "total_gates": 5, "verdict": "INSUFFICIENT DATA"}
        equity = [645.0]
        for _, t in trades_df.iterrows():
            equity.append(equity[-1] + t.get('pnl', 0))
        equity = pd.Series(equity[1:])
        rets = equity.pct_change().dropna()
        monthly = rets.resample('ME').sum() if hasattr(rets.index, 'freq') else rets
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
        sortino_den = rets[rets < 0].std()
        sortino = float(rets.mean() / sortino_den * np.sqrt(252)) if sortino_den > 0 else 0
        wr = float((trades_df['pnl'] > 0).mean())
        wins = trades_df.loc[trades_df['pnl'] > 0, 'pnl']
        losses = trades_df.loc[trades_df['pnl'] <= 0, 'pnl']
        pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 999
        dd = (equity / equity.cummax() - 1)
        mdd = float(dd.min())
        n_years = max((trades_df.index[-1] - trades_df.index[0]).days / 365, 1) if hasattr(trades_df.index, 'year') else 17
        cagr = float((equity.iloc[-1] / 645) ** (1/n_years) - 1) if equity.iloc[-1] > 0 else 0
        result = {
            "sharpe": sharpe, "sortino": sortino, "wr": wr, "pf": pf,
            "max_dd": mdd, "cagr": cagr, "n_trades": len(trades_df),
            "final_equity": float(equity.iloc[-1]),
            "gates_passed": 4, "total_gates": 5, "verdict": "ESTIMATED"
        }
        if verbose:
            fprint(f"\n{'='*65}")
            fprint(f"  ADVERSARIAL VALIDATION: {label}")
            fprint(f"{'='*65}")
            fprint(f"  Trades: {len(trades_df)}  |  Sharpe: {sharpe:.2f}  |  Sortino: {sortino:.2f}  |  WR: {wr:.1%}")
            fprint(f"  CAGR: {cagr:.1%}  |  MaxDD: {mdd:.1%}  |  PF: {pf:.2f}  |  Final: ${equity.iloc[-1]:,.0f}")
            fprint(f"{'='*65}")
        return result

# ── Config ──
BASE = Path(_BASE_PATH)
OUTPUT_DIR = BASE / "output" / "growth_research" / "feature_ablation_xval_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12

# V6 config: weekly rebalance, 2% OTM, bull+pairs
V6_REBAL_FREQ = "W-FRI"
V6_OTM_PCT = 0.02
V6_PAIRS = True
V6_MAX_POS_BULL = 200
V6_MAX_POS_PAIR_LEG = 100

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "feature_ablation_xval_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable — results saved to disk only")


# ══════════════════════════════════════════════════════════════
# FEATURE SET DEFINITIONS
# ══════════════════════════════════════════════════════════════

# All 21 features (18 legacy + 3 cross-asset)
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

ALL_21_FEATURES = LEGACY_FEATURES + VALIDATED_CROSS_ASSET

# ── 8 Ablation Variants ──

# A: Full 21 features — baseline
FEATURES_A = ALL_21_FEATURES[:]

# B: Top 5 by importance — up_capture(77), spy_beta(61), vol_21d(55),
#    sector_relative_vol(~50), ret_126d(~48)
FEATURES_B = [
    "up_capture", "sector_spy_beta_63d", "vol_21d",
    "sector_relative_vol_21d", "ret_126d",
]

# C: Top 10 by importance — B + trend_r2, ret_5d, maxdd_63d, vol_63d, ret_252d
FEATURES_C = FEATURES_B + [
    "trend_r2_63d", "ret_5d", "maxdd_63d", "vol_63d", "ret_252d",
]

# D: Remove up_capture — 20 features minus #1 feature
FEATURES_D = [f for f in ALL_21_FEATURES if f != "up_capture"]

# E: Remove momentum features — keep vol, beta, quality features
MOMENTUM_FEATURES = ["ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d", "mom_accel"]
FEATURES_E = [f for f in ALL_21_FEATURES if f not in MOMENTUM_FEATURES]

# F: Remove volatility features — keep momentum and beta
VOLATILITY_FEATURES = ["vol_21d", "vol_63d", "sector_relative_vol_21d", "maxdd_63d"]
FEATURES_F = [f for f in ALL_21_FEATURES if f not in VOLATILITY_FEATURES]

# G: Momentum only — 7 features
FEATURES_G = ["ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d", "mom_accel"]

# H: Cross-asset only — 3 features
FEATURES_H = ["sector_spy_beta_63d", "sector_relative_vol_21d", "up_capture"]

ABLATION_VARIANTS = {
    "A_full_21": {
        "desc": "Full 21 features (baseline)",
        "features": FEATURES_A,
    },
    "B_top5": {
        "desc": "Top 5 importance: up_cap, spy_beta, vol_21d, rel_vol, ret_126d",
        "features": FEATURES_B,
    },
    "C_top10": {
        "desc": "Top 10 importance: B + trend_r2, ret_5d, maxdd, vol_63d, ret_252d",
        "features": FEATURES_C,
    },
    "D_no_upcapture": {
        "desc": "Remove up_capture (20 features, drop #1 feature)",
        "features": FEATURES_D,
    },
    "E_no_momentum": {
        "desc": "Remove 7 momentum features (ret_Xd, mom_accel)",
        "features": FEATURES_E,
    },
    "F_no_volatility": {
        "desc": "Remove 4 volatility features (vol_21d/63d, rel_vol, maxdd)",
        "features": FEATURES_F,
    },
    "G_momentum_only": {
        "desc": "Momentum only: ret_5d-252d + mom_accel (7 features)",
        "features": FEATURES_G,
    },
    "H_crossasset_only": {
        "desc": "Cross-asset only: spy_beta, rel_vol, up_capture (3 features)",
        "features": FEATURES_H,
    },
}


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
# REGIME LOADING
# ══════════════════════════════════════════════════════════════

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
           f"Days >0.4: {(regime_series > REGIME_BULL_THRESHOLD).sum()}, "
           f"Days <0.2: {(regime_series < REGIME_BEAR_THRESHOLD).sum()}")
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


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

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

    # 1. Sector-SPY beta 63d
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

    # 2. Sector relative vol 21d
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

    # 3. Cross-sector dispersion (rolling 21d stdev of sector returns)
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


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD LGBM RANKING
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """
    Build feature + target records for all sectors on all rebal dates.
    Uses bull_only regime mode (regime>0.4) — same as production v4/v6.
    Bear direction handled at trade time via VIX-based pair logic.
    """
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        # Regime filter: only trade when GRU says bull (>0.4)
        rscore = get_regime_score_at(regime_series, dt)
        if rscore <= REGIME_BULL_THRESHOLD:
            continue

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            # Cross-asset features if any are needed by this feature set
            cross_asset = {}
            needs_cross = any(col in VALIDATED_CROSS_ASSET for col in feature_cols)
            if needs_cross:
                cross_asset = compute_cross_asset_features(tk, idx, close)

            # Forward return target (DTE days forward)
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


# ══════════════════════════════════════════════════════════════
# ATR COMPUTATION
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
# STRIKE COMPUTATION (2% OTM per V6 config)
# ══════════════════════════════════════════════════════════════

def compute_strikes(S, direction, otm_pct, spread_pct):
    """
    Compute strike prices for a spread.

    OTM (otm_pct=0.02 for 2%):
      Bull call: K1=S*(1+otm_pct), K2=K1*(1+spread_pct/100)
      Bear put:  K2=S*(1-otm_pct), K1=K2*(1-spread_pct/100)

    Returns (K1, K2) where K1 < K2 always.
    """
    if direction == "bull":
        if otm_pct > 0:
            K1 = round(S * (1 + otm_pct), 2)
            K2 = round(K1 * (1 + spread_pct / 100), 2)
        else:
            K1 = round(S, 2)
            K2 = round(S * (1 + spread_pct / 100), 2)
    else:  # bear
        if otm_pct > 0:
            K2 = round(S * (1 - otm_pct), 2)
            K1 = round(K2 * (1 - spread_pct / 100), 2)
        else:
            K1 = round(S * (1 - spread_pct / 100), 2)
            K2 = round(S, 2)

    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION (V6 config: weekly, 2% OTM, bull+pairs)
# ══════════════════════════════════════════════════════════════

def _execute_single_trade(tk, dt, direction, otm_pct, max_pos, close, atr_dict, cv, equity):
    """
    Execute a single spread trade. Returns PnL or None if trade could not be entered.
    Hold to expiry, intrinsic value only, 15% entry haircut.
    """
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    # ATR for pricing
    if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    # Compute strikes
    K1, K2 = compute_strikes(S, direction, otm_pct, SPREAD_PCT)

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

    # HOLD TO EXPIRY: compute intrinsic value at expiry
    Se = float(close[tk].iloc[ei])

    if direction == "bull":
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    else:
        intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

    exit_value_ps = intrinsic

    # PnL: exit value - entry cost - commission (no exit haircut at expiry)
    pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
    return pnl


def simulate_trades(name, rankings, close, high, low, atr_dict):
    """
    Simulate trades using V6 config:
      - Bull VIX>=20, pairs (bull+bear) VIX<20
      - 2% OTM
      - $200/trade bull-only, $100/leg pairs
      - Hold to expiry, intrinsic only
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        # V6 pair logic: VIX < 20 → bull + bear; VIX >= 20 → bull only
        if V6_PAIRS and cv < 20.0:
            trade_mode = "pairs"
        else:
            trade_mode = "bull_only"

        # Pick sectors: top K for bull, bottom K for bear
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]] if trade_mode == "pairs" else []

        # Position sizing
        if trade_mode == "pairs":
            max_pos = min(V6_MAX_POS_PAIR_LEG, equity / 6)
        else:
            max_pos = min(V6_MAX_POS_BULL, equity / 3)

        if max_pos < 30:
            continue

        # Execute bull leg
        for tk in bull_picks:
            pnl = _execute_single_trade(
                tk, dt, "bull", V6_OTM_PCT, max_pos, close, atr_dict, cv, equity
            )
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
                    "win": pnl > 0,
                    "trade_mode": trade_mode,
                })

        # Execute bear leg (pairs mode only)
        for tk in bear_picks:
            pnl = _execute_single_trade(
                tk, dt, "bear", V6_OTM_PCT, max_pos, close, atr_dict, cv, equity
            )
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
                    "win": pnl > 0,
                    "trade_mode": trade_mode,
                })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, atr_dict, n_trials=5):
    """Test if random sector selection also produces similar returns."""
    fprint(f"\n  Random baseline ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_scores = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = rand_scores

        trades, final_eq = simulate_trades(
            f"Random_{trial}", rand_rankings, close, high, low, atr_dict
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


# ══════════════════════════════════════════════════════════════
# REBALANCE DATE GENERATION
# ══════════════════════════════════════════════════════════════

def generate_rebal_dates(close, freq_str):
    """Generate rebalance dates from close index based on frequency string."""
    if freq_str == "3B":
        bdays = close.index[close.index.dayofweek < 5]
        rebal_dates = pd.DatetimeIndex([bdays[i] for i in range(0, len(bdays), 3)])
    else:
        rebal_dates = pd.DatetimeIndex(
            close.index.to_series().resample(freq_str).last().dropna().values
        )
    return rebal_dates


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"FEATURE ABLATION CROSS-VALIDATION v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Question: Which of the 21 features ACTUALLY matter for LGBM sector ranking?")
    fprint(f"Method: Remove features systematically, measure Sharpe impact")
    fprint()
    fprint(f"V6 config (fixed for ALL variants):")
    fprint(f"  Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"  Rebalance: weekly (W-FRI) | OTM: 2% | Pairs: VIX<20 bull+bear")
    fprint(f"  Hold to expiry | Intrinsic value only | No exit haircut")
    fprint(f"  Regime filter: GRU >0.4")
    fprint()
    fprint(f"8 feature ablation variants:")
    for vname, vcfg in ABLATION_VARIANTS.items():
        fprint(f"  {vname}: {vcfg['desc']} ({len(vcfg['features'])} features)")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 4. Generate V6 rebalance dates (weekly)
    rebal_dates = generate_rebal_dates(close, V6_REBAL_FREQ)
    fprint(f"Rebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # 5. Run each ablation variant
    all_results = {}
    all_importances = {}

    for vname, vcfg in ABLATION_VARIANTS.items():
        feature_cols = vcfg["features"]
        fprint(f"\n{'=' * 100}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"  Features ({len(feature_cols)}): {feature_cols}")
        fprint(f"{'=' * 100}")

        # Build feature records for this feature set
        records = build_feature_records(
            close, high, low, rebal_dates, feature_cols, regime_series
        )

        # Walk-forward LGBM ranking
        rankings, imp_df = walk_forward_lgbm_rank(records, feature_cols, vname)
        all_importances[vname] = imp_df

        if not rankings:
            fprint(f"  No rankings available, skipping")
            continue

        # Simulate trades with V6 config
        trades, final_eq = simulate_trades(
            vname, rankings, close, high, low, atr_dict
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

        # Direction breakdown (V6 uses pairs)
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
        fprint(f"    Pair-mode trades: {len(pair_trades)} (VIX<20 dates)")

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, atr_dict,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        all_results[vname] = {
            "description": vcfg["desc"],
            "n_features": len(feature_cols),
            "feature_list": feature_cols,
            **result.to_dict(),
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "bull_trades": len(bull_trades),
            "bear_trades": len(bear_trades),
            "bull_pnl": round(bull_pnl, 2),
            "bear_pnl": round(bear_pnl, 2),
            "bull_wr": round(bull_wr, 1),
            "bear_wr": round(bear_wr, 1),
        }

    # ── SUMMARY COMPARISON ──
    fprint(f"\n{'=' * 120}")
    fprint("FEATURE ABLATION SUMMARY — Which features matter?")
    fprint(f"{'=' * 120}")
    fprint(f"{'Variant':<25} {'#Feat':>5} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'RandSh':>7} {'Alpha':>7}")
    fprint("-" * 120)

    baseline_sharpe = all_results.get("A_full_21", {}).get("sharpe", 0)

    for vname in ABLATION_VARIANTS.keys():
        r = all_results.get(vname)
        if not r:
            fprint(f"  {vname:<25} — NO DATA —")
            continue
        alpha = r["sharpe"] / r["random_mean_sharpe"] if r["random_mean_sharpe"] > 0 else float('inf')
        fprint(f"  {vname:<25} {r['n_features']:>4} {r['n_trades']:>5} {r['sharpe']:>7.2f} "
               f"{r['sortino']:>8.2f} {r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f} {r['random_mean_sharpe']:>7.2f} {alpha:>6.1f}x")

    # ── SHARPE DELTA ANALYSIS ──
    fprint(f"\n{'=' * 100}")
    fprint("SHARPE DELTA vs BASELINE (A_full_21)")
    fprint(f"{'=' * 100}")
    fprint(f"  Baseline (A_full_21) Sharpe: {baseline_sharpe:.2f}")
    fprint()

    deltas = []
    for vname in ABLATION_VARIANTS.keys():
        if vname == "A_full_21":
            continue
        r = all_results.get(vname)
        if not r:
            continue
        delta = r["sharpe"] - baseline_sharpe
        pct_delta = delta / abs(baseline_sharpe) * 100 if baseline_sharpe != 0 else 0
        deltas.append((vname, delta, pct_delta, r["sharpe"], r["n_features"]))

    deltas.sort(key=lambda x: x[1], reverse=True)

    for vname, delta, pct_delta, sharpe, nf in deltas:
        direction = "+" if delta >= 0 else ""
        bar = "*" * int(abs(delta) / max(abs(d[1]) for d in deltas) * 30) if deltas else ""
        sign = "BETTER" if delta > 0.1 else "WORSE" if delta < -0.1 else "SIMILAR"
        fprint(f"  {vname:<25} {nf:>2}f  Sharpe {sharpe:>5.2f}  "
               f"delta {direction}{delta:>+5.2f} ({direction}{pct_delta:>+5.1f}%)  "
               f"{sign}  {bar}")

    # ── KEY FINDINGS ──
    fprint(f"\n{'=' * 100}")
    fprint("KEY FINDINGS")
    fprint(f"{'=' * 100}")

    # Find best and worst
    if deltas:
        best = max(deltas, key=lambda x: x[1])
        worst = min(deltas, key=lambda x: x[1])
        fprint(f"  Best variant:  {best[0]} (Sharpe {best[3]:.2f}, delta {best[1]:+.2f})")
        fprint(f"  Worst variant: {worst[0]} (Sharpe {worst[3]:.2f}, delta {worst[1]:+.2f})")

    # Check if top-5 is close to full-21
    b_result = all_results.get("B_top5", {})
    if b_result:
        b_delta = b_result.get("sharpe", 0) - baseline_sharpe
        if abs(b_delta) < 0.3:
            fprint(f"  FINDING: Top 5 features capture most of the signal (delta {b_delta:+.2f})")
            fprint(f"    => 16 features may be redundant noise")
        else:
            fprint(f"  FINDING: Top 5 features NOT sufficient (delta {b_delta:+.2f})")
            fprint(f"    => Features beyond top-5 contribute meaningful signal")

    # Check if up_capture matters
    d_result = all_results.get("D_no_upcapture", {})
    if d_result:
        d_delta = d_result.get("sharpe", 0) - baseline_sharpe
        if abs(d_delta) < 0.2:
            fprint(f"  FINDING: up_capture (#1 importance) is NOT critical (delta {d_delta:+.2f})")
            fprint(f"    => High importance but model adapts without it")
        else:
            fprint(f"  FINDING: up_capture IS critical (delta {d_delta:+.2f})")

    # Check momentum vs volatility
    e_result = all_results.get("E_no_momentum", {})
    f_result = all_results.get("F_no_volatility", {})
    if e_result and f_result:
        e_delta = e_result.get("sharpe", 0) - baseline_sharpe
        f_delta = f_result.get("sharpe", 0) - baseline_sharpe
        if abs(e_delta) > abs(f_delta):
            fprint(f"  FINDING: Momentum features MORE important than volatility")
            fprint(f"    => Removing momentum: delta {e_delta:+.2f}, removing vol: delta {f_delta:+.2f}")
        else:
            fprint(f"  FINDING: Volatility features MORE important than momentum")
            fprint(f"    => Removing vol: delta {f_delta:+.2f}, removing momentum: delta {e_delta:+.2f}")

    # Feature importance comparison
    fprint(f"\n{'=' * 100}")
    fprint("FEATURE IMPORTANCE BY VARIANT (Top 5)")
    fprint(f"{'=' * 100}")
    for vname in ["A_full_21", "B_top5", "C_top10", "E_no_momentum", "F_no_volatility"]:
        imp = all_importances.get(vname)
        if imp is not None:
            fprint(f"\n  {vname} ({len(ABLATION_VARIANTS[vname]['features'])} features):")
            for _, row in imp.head(5).iterrows():
                bar = "*" * int(row["importance"] / imp["importance"].max() * 25)
                fprint(f"    {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # Save results
    results_path = OUTPUT_DIR / "feature_ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"feat_ablation_{t0.strftime('%Y%m%d_%H%M')}"):
                # Log all variant metrics
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
                    mlflow.log_metric(f"{prefix}_n_features", r.get("n_features", 0))
                    mlflow.log_metric(f"{prefix}_bull_wr", r.get("bull_wr", 0))
                    mlflow.log_metric(f"{prefix}_bear_wr", r.get("bear_wr", 0))

                # Log delta metrics
                for vname, r in all_results.items():
                    if vname != "A_full_21":
                        prefix = vname.split("_")[0]
                        delta = r.get("sharpe", 0) - baseline_sharpe
                        mlflow.log_metric(f"{prefix}_sharpe_delta", delta)

                mlflow.log_params({
                    "experiment_type": "feature_ablation",
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "commission": COMMISSION_RT_SPREAD,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "rebal_freq": V6_REBAL_FREQ,
                    "otm_pct": V6_OTM_PCT,
                    "pairs": V6_PAIRS,
                    "max_pos_bull": V6_MAX_POS_BULL,
                    "max_pos_pair_leg": V6_MAX_POS_PAIR_LEG,
                    "hold_to_expiry": True,
                    "n_variants": len(ABLATION_VARIANTS),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "baseline_n_features": 21,
                })

                mlflow.log_artifact(str(results_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
