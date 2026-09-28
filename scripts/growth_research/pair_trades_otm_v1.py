#!/usr/bin/env python3
"""
Pair Trades OTM v1 — Does OTM Moneyness Improve Sector Pair Trades?
=====================================================================

Context: Sector pair trades (VIX<20, long top-3 + short bottom-3) achieved
Sharpe 2.41 with ATM strikes. Bull-only strategy improved from 2.05 to 2.44
when switching ATM to 2% OTM. This script tests whether OTM helps pairs too.

6 VARIANTS:
  1) atm_baseline    — ATM, K_top=3, K_bot=3 (replicates finding #114)
  2) otm_2pct        — 2% OTM on both long and short legs
  3) otm_1pct        — 1% OTM
  4) otm_3pct        — 3% OTM
  5) otm_long_only   — 2% OTM on bull call only, ATM on bear put
  6) weekly_otm2     — 2% OTM + weekly (5d) rebalance instead of biweekly

Strategy:
  - Long bull call spread on top-3 sectors
  - Short bear put spread on bottom-3 sectors
  - VIX<20 only. LGBM ranking with 21 features. Biweekly rebalance (10d).
  - $645 capital, $200 max per trade, hold to expiry.

OTM strike construction:
  Bull call spread: K1_buy = S * (1 + otm_pct/100), K2_sell = K1 * (1 + spread_pct/100)
  Bear put spread:  K1_sell = S * (1 - otm_pct/100), K2_buy = K1 * (1 - spread_pct/100)
    (K1 is the higher-strike sold put, K2 is the lower-strike bought put for the pricer)

Hold to expiry. Intrinsic value at expiry. 15% entry haircut, no exit haircut.
$2.60 commission per spread.

Full 5-gate adversarial validation on all variants.
Log to MLflow experiment "pair_trades_otm_v1".
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


# ── Path auto-detect (Jupiter vs Neptune) ──
import os
_hostname = os.uname().nodename.lower()
if "neptune" in _hostname or "nick" in str(Path.home()):
    _BASE_STR = "/home/nick/Lvl3Quant"
else:
    _BASE_STR = "/home/jupiter/Lvl3Quant"

BASE = Path(_BASE_STR)

# ── Standardized tools with inline fallbacks ──
sys.path.insert(0, _BASE_STR)

try:
    from research.tools.options_pricer import (
        price_bull_call_spread,
        price_bear_put_spread,
        estimate_iv,
        compute_atr,
        COMMISSION_RT_SPREAD,
        DEFAULT_HAIRCUT,
    )
    fprint("Imported options_pricer from research.tools")
except ImportError:
    fprint("WARNING: research.tools.options_pricer not importable — using inline fallbacks")
    from scipy.stats import norm as _norm

    RISK_FREE_RATE = 0.045
    DEFAULT_HAIRCUT = 0.15
    COMMISSION_RT_SPREAD = 2.60

    def _bs_call(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(S * _norm.cdf(d1) - K * np.exp(-r * T) * _norm.cdf(d2))

    def _bs_put(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
        if T <= 0 or sigma <= 0:
            return max(K - S, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(K * np.exp(-r * T) * _norm.cdf(-d2) - S * _norm.cdf(-d1))

    def estimate_iv(atr, spot, vix=20.0, atr_period=14):
        if spot <= 0 or atr <= 0:
            return 0.25
        realized_vol = (atr / spot) * np.sqrt(252 / atr_period)
        iv_mult = 1.2 + 0.01 * max(vix - 20.0, 0.0)
        return max(realized_vol * iv_mult, 0.10)

    def price_bull_call_spread(S, K1, K2, dte, atr, vix=20.0,
                               haircut=DEFAULT_HAIRCUT, r=RISK_FREE_RATE, sigma=None):
        if K2 <= K1:
            raise ValueError(f"K2 ({K2}) must be > K1 ({K1})")
        T = dte / 365.0
        if sigma is None:
            sigma = estimate_iv(atr, S, vix)
        fair_value = _bs_call(S, K1, T, r, sigma) - _bs_call(S, K2, T, r, sigma)
        fair_value = max(fair_value, 0.001)
        entry_cost = fair_value * (1.0 + haircut)
        max_profit = (K2 - K1) - entry_cost
        return float(entry_cost), float(max_profit)

    def price_bear_put_spread(S, K1, K2, dte, atr, vix=20.0,
                              haircut=DEFAULT_HAIRCUT, r=RISK_FREE_RATE, sigma=None):
        if K2 <= K1:
            raise ValueError(f"K2 ({K2}) must be > K1 ({K1})")
        T = dte / 365.0
        if sigma is None:
            sigma = estimate_iv(atr, S, vix)
        fair_value = _bs_put(S, K2, T, r, sigma) - _bs_put(S, K1, T, r, sigma)
        fair_value = max(fair_value, 0.001)
        entry_cost = fair_value * (1.0 + haircut)
        max_profit = (K2 - K1) - entry_cost
        return float(entry_cost), float(max_profit)

    def compute_atr(high, low, close, period=14):
        high = pd.Series(high) if not isinstance(high, pd.Series) else high
        low = pd.Series(low) if not isinstance(low, pd.Series) else low
        close = pd.Series(close) if not isinstance(close, pd.Series) else close
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = tr.ewm(alpha=1 / period, min_periods=period).mean()
        return float(atr_series.iloc[-1])

try:
    from research.tools.adversarial_validator import validate_trades
    fprint("Imported adversarial_validator from research.tools")
except ImportError:
    fprint("FATAL: research.tools.adversarial_validator not importable")
    fprint("  This module is required — cannot proceed without 5-gate validation.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

OUTPUT_DIR = BASE / "output" / "growth_research" / "pair_trades_otm_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
TOP_K = 3
BOT_K = 3

# VIX<20 filter for pair trades
VIX_THRESHOLD = 20.0

# Position sizing: $200 max per trade (user spec)
MAX_POS_PER_TRADE = 200.0

# Regime predictions
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# LGBM walk-forward
WF_TRAIN_PERIODS = 12
WF_REBAL_FREQ_BIWEEKLY = "2W-FRI"   # ~10 business days
WF_REBAL_FREQ_WEEKLY = "W-FRI"       # ~5 business days

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "pair_trades_otm_v1"

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


# ── Features (21 total: 18 legacy + 3 cross-asset) ──

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

ALL_FEATURES = LEGACY_FEATURES + VALIDATED_CROSS_ASSET  # 21 total


# ══════════════════════════════════════════════════════════════
# VARIANT DEFINITIONS
# ══════════════════════════════════════════════════════════════

VARIANTS = {
    "atm_baseline": {
        "description": "ATM baseline pair trades (replicates finding #114)",
        "otm_bull_pct": 0.0,
        "otm_bear_pct": 0.0,
        "rebal_freq": WF_REBAL_FREQ_BIWEEKLY,
    },
    "otm_2pct": {
        "description": "2% OTM on both long and short legs",
        "otm_bull_pct": 2.0,
        "otm_bear_pct": 2.0,
        "rebal_freq": WF_REBAL_FREQ_BIWEEKLY,
    },
    "otm_1pct": {
        "description": "1% OTM on both legs",
        "otm_bull_pct": 1.0,
        "otm_bear_pct": 1.0,
        "rebal_freq": WF_REBAL_FREQ_BIWEEKLY,
    },
    "otm_3pct": {
        "description": "3% OTM on both legs",
        "otm_bull_pct": 3.0,
        "otm_bear_pct": 3.0,
        "rebal_freq": WF_REBAL_FREQ_BIWEEKLY,
    },
    "otm_long_only": {
        "description": "2% OTM on bull call only, ATM on bear put",
        "otm_bull_pct": 2.0,
        "otm_bear_pct": 0.0,
        "rebal_freq": WF_REBAL_FREQ_BIWEEKLY,
    },
    "weekly_otm2": {
        "description": "2% OTM + weekly (5d) rebalance",
        "otm_bull_pct": 2.0,
        "otm_bear_pct": 2.0,
        "rebal_freq": WF_REBAL_FREQ_WEEKLY,
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
# FEATURE ENGINEERING (identical to production v4)
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
    """Compute the 3 validated cross-asset features."""
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


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD LGBM RANKING
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """
    Build feature + target records for all sectors on all rebal dates.
    No regime filtering — we include ALL dates (VIX filtering at trade time).
    """
    import lightgbm as lgb

    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
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


def walk_forward_lgbm_rank(df, feature_cols, label):
    """Walk-forward LGBM ranking: sliding train, predict next period."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    {label}: Insufficient data ({len(df)} records)")
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

    fprint(f"    {label}: {len(rankings)} ranking dates, {n_models} models trained")
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
# TRADE PRICING WITH OTM SUPPORT
# ══════════════════════════════════════════════════════════════

def price_and_enter_bull_otm(tk, S, di, ei, close, atr_dict, cv, dt, max_pos,
                              equity, otm_pct=0.0):
    """
    Price a bull call spread with optional OTM.
    Bull call: K1_buy = S * (1 + otm_pct/100), K2_sell = K1 * (1 + spread_pct/100)
    At expiry: intrinsic = max(Se - K1, 0) - max(Se - K2, 0)
    """
    K1 = round(S * (1 + otm_pct / 100), 2)
    K2 = round(K1 * (1 + SPREAD_PCT / 100), 2)
    if K2 <= K1:
        K2 = K1 + 0.50

    if dt in atr_dict.get(tk, pd.Series()).index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    try:
        entry_cost_ps, max_profit_ps = price_bull_call_spread(
            S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
        )
    except Exception:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    # HOLD TO EXPIRY: intrinsic value
    Se = float(close[tk].iloc[ei])
    intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

    return {
        "pnl": pnl,
        "entry_cost": total_cost,
        "K1": K1,
        "K2": K2,
        "direction": "bull",
        "otm_pct": otm_pct,
    }


def price_and_enter_bear_otm(tk, S, di, ei, close, atr_dict, cv, dt, max_pos,
                              equity, otm_pct=0.0):
    """
    Price a bear put spread with optional OTM.
    Bear put with OTM: K1_sell = S * (1 - otm_pct/100),
                        K2_buy  = K1_sell * (1 - spread_pct/100)
    The pricer expects K1 < K2, so:
      K_lower = K2_buy (the bought put, further OTM)
      K_upper = K1_sell (the sold put, closer to money)

    At expiry: intrinsic = max(K_upper - Se, 0) - max(K_lower - Se, 0)
    """
    K_upper = round(S * (1 - otm_pct / 100), 2)       # sold put (higher strike)
    K_lower = round(K_upper * (1 - SPREAD_PCT / 100), 2)  # bought put (lower strike)

    if K_upper <= K_lower:
        K_upper = K_lower + 0.50

    if dt in atr_dict.get(tk, pd.Series()).index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    try:
        # price_bear_put_spread expects K1 < K2:
        #   K1 = lower strike (short put sold further out), K2 = upper strike (long put)
        # BUT standard bear put = buy high put, sell low put.
        # Our pricer: K1=lower (short), K2=upper (long)
        # Here: K_lower is the bought put (protection), K_upper is the sold put.
        # Wait — re-reading the pricer: bear put = buy put at K2 (higher), sell put at K1 (lower).
        # So K1 = K_lower (sell), K2 = K_upper (buy). Profits when underlying falls below K2.
        # But WE want to short the sector — the bear put spread profits when sector drops.
        # K1 = K_lower (sell this put), K2 = K_upper (buy this put)
        entry_cost_ps, max_profit_ps = price_bear_put_spread(
            S=S, K1=K_lower, K2=K_upper, dte=DTE, atr=av, vix=cv
        )
    except Exception:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    # HOLD TO EXPIRY: bear put spread intrinsic
    Se = float(close[tk].iloc[ei])
    intrinsic = max(K_upper - Se, 0.0) - max(K_lower - Se, 0.0)
    pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

    return {
        "pnl": pnl,
        "entry_cost": total_cost,
        "K1": K_lower,
        "K2": K_upper,
        "direction": "bear",
        "otm_pct": otm_pct,
    }


# ══════════════════════════════════════════════════════════════
# UNIFIED PAIR TRADE SIMULATOR
# ══════════════════════════════════════════════════════════════

def simulate_pair_trades(rankings, close, high, low, atr_dict,
                         otm_bull_pct=0.0, otm_bear_pct=0.0,
                         variant_name="variant"):
    """
    Simulate VIX<20 pair trades: long top-K bull call + short bottom-K bear put.

    Args:
        rankings: dict date -> {ticker: score}
        otm_bull_pct: OTM percentage for bull call spreads
        otm_bear_pct: OTM percentage for bear put spreads
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        # VIX<20 only
        if cv >= VIX_THRESHOLD:
            continue

        scores = rankings[dt]
        if not scores or len(scores) < TOP_K + BOT_K:
            continue

        # Top K for bull, bottom K for bear
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:BOT_K]]

        # Don't overlap
        bear_picks = [t for t in bear_picks if t not in bull_picks]

        max_pos = min(MAX_POS_PER_TRADE, equity / (TOP_K + len(bear_picks)))
        if max_pos < 20:
            continue

        di = close.index.get_loc(dt)
        ei = min(di + DTE, len(close) - 1)
        if ei <= di:
            continue

        sv = float(spy.loc[dt]) if dt in spy.index else 0
        se = float(spy.iloc[ei]) if ei < len(spy) else sv
        spy_regime = "bull" if se >= sv else "bear"

        # Bull leg (long top sectors)
        for tk in bull_picks:
            if tk not in close.columns or tk not in atr_dict:
                continue
            S = float(close[tk].loc[dt])
            result = price_and_enter_bull_otm(
                tk, S, di, ei, close, atr_dict, cv, dt, max_pos, equity,
                otm_pct=otm_bull_pct,
            )
            if result is None:
                continue
            equity += result["pnl"]
            trades.append({
                "pnl": round(result["pnl"], 2),
                "entry_date": str(dt.date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "regime": spy_regime,
                "direction": "bull",
                "vix": round(cv, 1),
                "otm_pct": otm_bull_pct,
                "win": result["pnl"] > 0,
            })

        # Bear leg (short bottom sectors)
        for tk in bear_picks:
            if tk not in close.columns or tk not in atr_dict:
                continue
            S = float(close[tk].loc[dt])
            result = price_and_enter_bear_otm(
                tk, S, di, ei, close, atr_dict, cv, dt, max_pos, equity,
                otm_pct=otm_bear_pct,
            )
            if result is None:
                continue
            equity += result["pnl"]
            trades.append({
                "pnl": round(result["pnl"], 2),
                "entry_date": str(dt.date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "regime": spy_regime,
                "direction": "bear",
                "vix": round(cv, 1),
                "otm_pct": otm_bear_pct,
                "win": result["pnl"] > 0,
            })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, atr_dict,
                         otm_bull_pct, otm_bear_pct, n_trials=5):
    """Test if random sector selection also produces similar returns."""
    fprint(f"  Random baseline test ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_scores = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = rand_scores

        trades, final_eq = simulate_pair_trades(
            rand_rankings, close, high, low, atr_dict,
            otm_bull_pct=otm_bull_pct, otm_bear_pct=otm_bear_pct,
            variant_name=f"random_{trial}",
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
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"PAIR TRADES OTM v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only | VIX<{VIX_THRESHOLD:.0f} only")
    fprint(f"Top-K: {TOP_K} bull + {BOT_K} bear | Max/trade: ${MAX_POS_PER_TRADE:.0f}")
    fprint(f"6 variants: ATM baseline, 1/2/3% OTM, long-only OTM, weekly+OTM")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions (not used for filtering, but available)
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    spy_close = close["SPY"]

    # 4. Build rankings for each rebalance frequency
    # Biweekly rankings (used by variants 1-5)
    rebal_biweekly = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ_BIWEEKLY).last().dropna().values
    )
    fprint(f"Biweekly rebalance dates: {len(rebal_biweekly)} "
           f"({rebal_biweekly[0].date()} to {rebal_biweekly[-1].date()})")

    fprint("\n" + "=" * 80)
    fprint("BUILDING RANKINGS: Biweekly (all dates, no regime filter)")
    fprint("=" * 80)
    records_biweekly = build_feature_records(
        close, high, low, rebal_biweekly, ALL_FEATURES, regime_series,
    )
    rankings_biweekly, imp_biweekly = walk_forward_lgbm_rank(
        records_biweekly, ALL_FEATURES, "biweekly_all"
    )

    # Weekly rankings (used by variant 6 only)
    rebal_weekly = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ_WEEKLY).last().dropna().values
    )
    fprint(f"\nWeekly rebalance dates: {len(rebal_weekly)} "
           f"({rebal_weekly[0].date()} to {rebal_weekly[-1].date()})")

    fprint("\n" + "=" * 80)
    fprint("BUILDING RANKINGS: Weekly (all dates, no regime filter)")
    fprint("=" * 80)
    records_weekly = build_feature_records(
        close, high, low, rebal_weekly, ALL_FEATURES, regime_series,
    )
    rankings_weekly, imp_weekly = walk_forward_lgbm_rank(
        records_weekly, ALL_FEATURES, "weekly_all"
    )

    # ── SIMULATE ALL VARIANTS ──
    fprint("\n" + "=" * 80)
    fprint("SIMULATING TRADES — 6 OTM VARIANTS")
    fprint("=" * 80)

    all_results = {}

    for vname, vcfg in VARIANTS.items():
        fprint(f"\n{'='*65}")
        fprint(f"  {vname}: {vcfg['description']}")
        fprint(f"  OTM bull: {vcfg['otm_bull_pct']}% | OTM bear: {vcfg['otm_bear_pct']}% | "
               f"Rebal: {vcfg['rebal_freq']}")
        fprint(f"{'='*65}")

        # Select rankings based on rebalance frequency
        if vcfg["rebal_freq"] == WF_REBAL_FREQ_WEEKLY:
            rankings = rankings_weekly
        else:
            rankings = rankings_biweekly

        if not rankings:
            fprint(f"  No rankings available, skipping")
            all_results[vname] = {"description": vcfg["description"], "error": "no_rankings"}
            continue

        trades, final_eq = simulate_pair_trades(
            rankings, close, high, low, atr_dict,
            otm_bull_pct=vcfg["otm_bull_pct"],
            otm_bear_pct=vcfg["otm_bear_pct"],
            variant_name=vname,
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[vname] = {
                "description": vcfg["description"],
                "n_trades": len(trades) if trades else 0,
                "error": "insufficient_trades",
            }
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
        bull_pnl = sum(t["pnl"] for t in bull_trades) if bull_trades else 0.0
        bear_pnl = sum(t["pnl"] for t in bear_trades) if bear_trades else 0.0
        bull_wr = sum(1 for t in bull_trades if t["win"]) / max(len(bull_trades), 1) * 100
        bear_wr = sum(1 for t in bear_trades if t["win"]) / max(len(bear_trades), 1) * 100
        fprint(f"  Direction breakdown:")
        fprint(f"    Bull: {len(bull_trades)} trades, WR {bull_wr:.1f}%, PnL ${bull_pnl:.0f}")
        fprint(f"    Bear: {len(bear_trades)} trades, WR {bear_wr:.1f}%, PnL ${bear_pnl:.0f}")

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, atr_dict,
            vcfg["otm_bull_pct"], vcfg["otm_bear_pct"],
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        all_results[vname] = {
            "description": vcfg["description"],
            **result.to_dict(),
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "n_bull_trades": len(bull_trades),
            "n_bear_trades": len(bear_trades),
            "bull_pnl": round(bull_pnl, 2),
            "bear_pnl": round(bear_pnl, 2),
            "bull_wr": round(bull_wr, 1),
            "bear_wr": round(bear_wr, 1),
            "otm_bull_pct": vcfg["otm_bull_pct"],
            "otm_bear_pct": vcfg["otm_bear_pct"],
            "rebal_freq": vcfg["rebal_freq"],
        }

    # ── SUMMARY COMPARISON ──
    fprint("\n" + "=" * 80)
    fprint("SUMMARY COMPARISON — OTM VARIANTS")
    fprint("=" * 80)
    fprint(f"{'Variant':<20} {'OTM%':>5} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} "
           f"{'WR':>6} {'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'RandSh':>7}")
    fprint("-" * 100)

    for vname in VARIANTS.keys():
        r = all_results.get(vname)
        if not r or "error" in r:
            otm = VARIANTS[vname]["otm_bull_pct"]
            fprint(f"  {vname:<20} {otm:>4.0f}% {'NO DATA':>6}")
            continue
        otm_label = f"{r['otm_bull_pct']:.0f}/{r['otm_bear_pct']:.0f}"
        fprint(f"  {vname:<20} {otm_label:>5} {r['n_trades']:>5} {r['sharpe']:>7.2f} "
               f"{r['sortino']:>8.2f} {r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f} {r['random_mean_sharpe']:>7.2f}")

    # ── KEY FINDINGS ──
    fprint("\n" + "=" * 80)
    fprint("KEY FINDINGS")
    fprint("=" * 80)

    valid_variants = {k: v for k, v in all_results.items()
                      if "error" not in v and v.get("sharpe", 0) > 0}

    baseline_sharpe = all_results.get("atm_baseline", {}).get("sharpe", 0)

    if valid_variants:
        best_name = max(valid_variants, key=lambda k: valid_variants[k]["sharpe"])
        best = valid_variants[best_name]
        fprint(f"  Best variant: {best_name} (Sharpe {best['sharpe']:.2f})")

        if baseline_sharpe > 0:
            fprint(f"  ATM baseline Sharpe: {baseline_sharpe:.2f}")
            delta = best["sharpe"] - baseline_sharpe
            fprint(f"  Best vs baseline: {'+'if delta>0 else ''}{delta:.2f} Sharpe")

            if delta > 0.15:
                fprint(f"  CONCLUSION: OTM IMPROVES pair trades by {delta:.2f} Sharpe")
            elif delta > -0.15:
                fprint(f"  CONCLUSION: OTM is NEUTRAL for pair trades (delta within noise)")
            else:
                fprint(f"  CONCLUSION: OTM HURTS pair trades by {abs(delta):.2f} Sharpe")

        # Compare with prior finding #114 (Sharpe 2.41)
        fprint(f"\n  Prior ATM pair trades finding: Sharpe 2.41")
        if baseline_sharpe > 0:
            fprint(f"  Current ATM baseline: Sharpe {baseline_sharpe:.2f} "
                   f"({'confirms' if 2.0 < baseline_sharpe < 3.0 else 'differs from'} prior)")

        # Asymmetric OTM analysis
        long_only_result = all_results.get("otm_long_only", {})
        if long_only_result and "error" not in long_only_result:
            lo_sharpe = long_only_result.get("sharpe", 0)
            otm2_sharpe = all_results.get("otm_2pct", {}).get("sharpe", 0)
            fprint(f"\n  Asymmetric OTM (bull-only OTM vs symmetric OTM):")
            fprint(f"    otm_long_only (2% bull, ATM bear): Sharpe {lo_sharpe:.2f}")
            fprint(f"    otm_2pct (2% both):                Sharpe {otm2_sharpe:.2f}")
            if lo_sharpe > otm2_sharpe + 0.1:
                fprint(f"    --> OTM helps the LONG side more than the SHORT side")
            elif otm2_sharpe > lo_sharpe + 0.1:
                fprint(f"    --> Symmetric OTM is better — both legs benefit")
            else:
                fprint(f"    --> Difference is marginal")

        # Weekly vs biweekly
        weekly_result = all_results.get("weekly_otm2", {})
        otm2_result = all_results.get("otm_2pct", {})
        if weekly_result and "error" not in weekly_result and otm2_result and "error" not in otm2_result:
            w_sharpe = weekly_result.get("sharpe", 0)
            b_sharpe = otm2_result.get("sharpe", 0)
            fprint(f"\n  Rebalance frequency (both at 2% OTM):")
            fprint(f"    Biweekly: Sharpe {b_sharpe:.2f} ({otm2_result.get('n_trades', 0)} trades)")
            fprint(f"    Weekly:   Sharpe {w_sharpe:.2f} ({weekly_result.get('n_trades', 0)} trades)")
    else:
        fprint("  No valid variants produced results.")

    # Feature importance
    fprint("\n" + "=" * 80)
    fprint("FEATURE IMPORTANCE (Top 10)")
    fprint("=" * 80)
    for label, imp_df_var in [("biweekly", imp_biweekly), ("weekly", imp_weekly)]:
        if imp_df_var is not None:
            fprint(f"\n  {label}:")
            for _, row in imp_df_var.head(10).iterrows():
                bar = "*" * int(row["importance"] / imp_df_var["importance"].max() * 30)
                fprint(f"    {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # Save results
    results_path = OUTPUT_DIR / "pair_trades_otm_v1_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"pair_otm_v1_{t0.strftime('%Y%m%d_%H%M')}"):
                for vname, r in all_results.items():
                    if "error" in r:
                        continue
                    mlflow.log_metric(f"{vname}_sharpe", r.get("sharpe", 0))
                    mlflow.log_metric(f"{vname}_sortino", r.get("sortino", 0))
                    mlflow.log_metric(f"{vname}_win_rate", r.get("win_rate", 0))
                    mlflow.log_metric(f"{vname}_profit_factor", r.get("profit_factor", 0))
                    mlflow.log_metric(f"{vname}_max_dd", r.get("max_dd", 0))
                    mlflow.log_metric(f"{vname}_n_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{vname}_final_equity", r.get("final_equity", 0))
                    mlflow.log_metric(f"{vname}_bull_wr", r.get("bull_wr", 0))
                    mlflow.log_metric(f"{vname}_bear_wr", r.get("bear_wr", 0))
                    mlflow.log_metric(f"{vname}_gates_passed", r.get("gates_passed", 0))

                # Log the ATM baseline delta
                if baseline_sharpe > 0 and valid_variants:
                    best_sh = max(v.get("sharpe", 0) for v in valid_variants.values())
                    mlflow.log_metric("best_sharpe", best_sh)
                    mlflow.log_metric("baseline_sharpe", baseline_sharpe)
                    mlflow.log_metric("otm_delta_sharpe", best_sh - baseline_sharpe)

                mlflow.log_artifact(str(results_path))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal elapsed: {elapsed/60:.1f} minutes")
    fprint("DONE")


if __name__ == "__main__":
    main()
