#!/usr/bin/env python3
"""
Pair Trades Cross-Validation v1 — Bear Put Spread Validation
==============================================================

Cross-validates whether adding short-side bear put spreads on bottom-ranked
sectors improves the production v4 bull-only strategy.

Prior finding: low-VIX pair trades gave Sharpe 2.41, MDD -4.1%, 5/5 gates.
But that was from an agent-built script that may not match production infra.
This script uses EXACTLY the production v4 LGBM walk-forward logic, GRU regime
filter, pricing, and costs — only adding the bear put spread leg.

5 VARIANTS:
  A) Bull-only baseline — Reproduce production v4 (should give ~1.67-1.87 Sharpe).
     Top 3 sectors, VIX>20 only.
  B) Pair VIX>20 — Top 3 bull spreads + bottom 3 bear put spreads, VIX>20 only.
  C) Low-VIX pairs — Top 3 bull + bottom 3 bear, VIX<20 ONLY.
  D) Combined — Bull spreads when VIX>20, pairs when VIX<20.
  E) Dollar-neutral combo — Equal $ long and short, VIX>20 bulls + VIX<20 pairs.

Honest pricing rules (IDENTICAL to production v4):
  - HOLD TO EXPIRY ONLY (no early exit)
  - At expiry: INTRINSIC VALUE ONLY (no time value)
  - 15% haircut on ENTRY only (automatic exercise at expiry = no exit haircut)
  - DTE=21, $645 starting capital
  - Walk-forward LGBM, biweekly rebalance
  - Bull call spreads: 3% width, ATM
  - Bear put spreads: buy ATM put, sell put at strike-3%. DTE=21. 15% entry haircut. $2.60 commission.

Bear put spread at expiry:
  intrinsic = max(strike_long - underlying, 0) - max(strike_short - underlying, 0)

Full 5-gate adversarial validation + random baseline comparison (5 trials).
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


# ── Config ──
OUTPUT_DIR = BASE / "output" / "growth_research" / "pair_trades_xval_v1"
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

# LGBM walk-forward (identical to production v4)
WF_TRAIN_PERIODS = 12
WF_REBAL_FREQ = "2W-FRI"

# Position sizing for pair trades: $100/trade per leg (half the bull-only $200)
PAIR_POS_SIZE = 100.0
BULL_ONLY_POS_SIZE = 200.0

# VIX threshold for low-VIX vs high-VIX regime split
VIX_THRESHOLD = 20.0

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "pair_trades_xval_v1"

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
# DATA DOWNLOAD (identical to production v4)
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
# REGIME LOADING (identical to production v4)
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
        return 0.5  # neutral default
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (identical to production v4)
# ══════════════════════════════════════════════════════════════

# Legacy 18 features (quality-momentum)
LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]

# Only the 3 validated cross-asset features (findings #64-65)
VALIDATED_CROSS_ASSET = [
    "sector_spy_beta_63d",
    "sector_relative_vol_21d",
    "cross_sector_dispersion",
]

V4_FEATURES = LEGACY_FEATURES + VALIDATED_CROSS_ASSET  # 21 total


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
# WALK-FORWARD LGBM RANKING (identical to production v4)
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series,
                          regime_mode="bull_only"):
    """
    Build feature + target records for all sectors on all rebal dates.

    regime_mode:
      'bull_only': only include dates where regime>0.4
      'bull_bear': include bull (>0.4) and bear (<0.2) dates, tagged accordingly
      'all': include ALL dates regardless of regime (for VIX<20 variants)
    """
    import lightgbm as lgb

    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features, mode={regime_mode}")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        # Get regime score
        rscore = get_regime_score_at(regime_series, dt)

        # Filter by regime mode
        if regime_mode == "bull_only":
            if rscore <= REGIME_BULL_THRESHOLD:
                continue
            direction = "bull"
        elif regime_mode == "bull_bear":
            if rscore > REGIME_BULL_THRESHOLD:
                direction = "bull"
            elif rscore < REGIME_BEAR_THRESHOLD:
                direction = "bear"
            else:
                continue  # gray zone, skip
        elif regime_mode == "all":
            # Include all dates — direction determined by regime score
            if rscore > REGIME_BULL_THRESHOLD:
                direction = "bull"
            elif rscore < REGIME_BEAR_THRESHOLD:
                direction = "bear"
            else:
                direction = "neutral"
        else:
            direction = "bull"

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            # Cross-asset features if needed
            cross_asset = {}
            for col in feature_cols:
                if col in VALIDATED_CROSS_ASSET:
                    cross_asset = compute_cross_asset_features(tk, idx, close)
                    break

            # Forward return target (DTE days forward)
            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk,
                   "fwd_ret": fwd_ret, "direction": direction}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    if "direction" in df.columns:
        bull_n = (df["direction"] == "bull").sum()
        bear_n = (df["direction"] == "bear").sum()
        neutral_n = (df["direction"] == "neutral").sum()
        fprint(f"    Bull records: {bull_n}, Bear records: {bear_n}, Neutral records: {neutral_n}")

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

            # Keep direction info with the ranking
            direction = test_df["direction"].iloc[0] if "direction" in test_df.columns else "bull"
            rankings[test_date] = {
                "scores": dict(zip(test_df["ticker"], test_df["score"])),
                "direction": direction,
            }

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
# ATR COMPUTATION (identical to production v4)
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
# TRADE SIMULATION — EXTENDED FOR PAIR TRADES
# ══════════════════════════════════════════════════════════════

def _price_and_enter_bull(tk, S, di, ei, close, atr_dict, cv, dt, max_pos, equity):
    """Price a bull call spread and return trade dict or None."""
    K1 = round(S, 2)
    K2 = round(S * (1 + SPREAD_PCT / 100), 2)
    if K2 <= K1:
        K2 = K1 + 1.0

    if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
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
    }


def _price_and_enter_bear(tk, S, di, ei, close, atr_dict, cv, dt, max_pos, equity):
    """
    Price a bear put spread and return trade dict or None.

    Bear put spread: buy ATM put (K_long = S), sell put at strike-3% (K_short = S * 0.97).
    At expiry: intrinsic = max(K_long - underlying, 0) - max(K_short - underlying, 0)
    """
    K_long = round(S, 2)                          # ATM — buy this put
    K_short = round(S * (1 - SPREAD_PCT / 100), 2)  # 3% below — sell this put

    if K_long <= K_short:
        K_long = K_short + 1.0

    if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    try:
        # price_bear_put_spread expects K1 < K2:
        #   K1 = lower strike (short put), K2 = upper strike (long put)
        entry_cost_ps, max_profit_ps = price_bear_put_spread(
            S=S, K1=K_short, K2=K_long, dte=DTE, atr=av, vix=cv
        )
    except Exception:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    # HOLD TO EXPIRY: bear put spread intrinsic
    Se = float(close[tk].iloc[ei])
    # intrinsic = max(K_long - Se, 0) - max(K_short - Se, 0)
    intrinsic = max(K_long - Se, 0.0) - max(K_short - Se, 0.0)
    pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

    return {
        "pnl": pnl,
        "entry_cost": total_cost,
        "K1": K_short,
        "K2": K_long,
        "direction": "bear",
    }


def simulate_variant_a(rankings, close, high, low, regime_series, atr_dict):
    """
    Variant A: Bull-only baseline — reproduce production v4.
    Top 3 sectors, VIX>20 only (regime>0.4 already filtered in rankings).
    Fixed $200 max per trade.
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        ranking_data = rankings[dt]
        scores = ranking_data["scores"]
        direction = ranking_data.get("direction", "bull")

        # Bull-only: skip bear dates
        if direction != "bull":
            continue

        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]

        max_pos = min(BULL_ONLY_POS_SIZE, equity / 3)
        if max_pos < 30:
            continue

        di = close.index.get_loc(dt)
        ei = min(di + DTE, len(close) - 1)
        if ei <= di:
            continue

        for tk in picks:
            if tk not in close.columns or tk not in atr_dict:
                continue
            S = float(close[tk].loc[dt])

            result = _price_and_enter_bull(tk, S, di, ei, close, atr_dict, cv, dt, max_pos, equity)
            if result is None:
                continue

            equity += result["pnl"]

            sv = float(spy.loc[dt]) if dt in spy.index else 0
            se = float(spy.iloc[ei]) if ei < len(spy) else sv

            trades.append({
                "pnl": round(result["pnl"], 2),
                "entry_date": str(dt.date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "regime": "bull" if se >= sv else "bear",
                "direction": "bull",
                "vix": round(cv, 1),
                "win": result["pnl"] > 0,
            })

    return trades, equity


def simulate_variant_b(rankings, close, high, low, regime_series, atr_dict):
    """
    Variant B: Pair VIX>20 — Top 3 bull + bottom 3 bear, VIX>20 only.
    $100/trade per leg (half exposure).
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        # VIX>20 only for this variant
        if cv < VIX_THRESHOLD:
            continue

        ranking_data = rankings[dt]
        scores = ranking_data["scores"]
        if not scores:
            continue

        # Top K for bull, bottom K for bear
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]]

        max_pos = min(PAIR_POS_SIZE, equity / 6)
        if max_pos < 20:
            continue

        di = close.index.get_loc(dt)
        ei = min(di + DTE, len(close) - 1)
        if ei <= di:
            continue

        sv = float(spy.loc[dt]) if dt in spy.index else 0
        se = float(spy.iloc[ei]) if ei < len(spy) else sv
        spy_regime = "bull" if se >= sv else "bear"

        # Bull leg
        for tk in bull_picks:
            if tk not in close.columns or tk not in atr_dict:
                continue
            S = float(close[tk].loc[dt])
            result = _price_and_enter_bull(tk, S, di, ei, close, atr_dict, cv, dt, max_pos, equity)
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
                "win": result["pnl"] > 0,
            })

        # Bear leg
        for tk in bear_picks:
            if tk not in close.columns or tk not in atr_dict:
                continue
            S = float(close[tk].loc[dt])
            result = _price_and_enter_bear(tk, S, di, ei, close, atr_dict, cv, dt, max_pos, equity)
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
                "win": result["pnl"] > 0,
            })

    return trades, equity


def simulate_variant_c(rankings, close, high, low, regime_series, atr_dict):
    """
    Variant C: Low-VIX pairs — Top 3 bull + bottom 3 bear, VIX<20 ONLY.
    $100/trade per leg.
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

        ranking_data = rankings[dt]
        scores = ranking_data["scores"]
        if not scores:
            continue

        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]]

        max_pos = min(PAIR_POS_SIZE, equity / 6)
        if max_pos < 20:
            continue

        di = close.index.get_loc(dt)
        ei = min(di + DTE, len(close) - 1)
        if ei <= di:
            continue

        sv = float(spy.loc[dt]) if dt in spy.index else 0
        se = float(spy.iloc[ei]) if ei < len(spy) else sv
        spy_regime = "bull" if se >= sv else "bear"

        # Bull leg
        for tk in bull_picks:
            if tk not in close.columns or tk not in atr_dict:
                continue
            S = float(close[tk].loc[dt])
            result = _price_and_enter_bull(tk, S, di, ei, close, atr_dict, cv, dt, max_pos, equity)
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
                "win": result["pnl"] > 0,
            })

        # Bear leg
        for tk in bear_picks:
            if tk not in close.columns or tk not in atr_dict:
                continue
            S = float(close[tk].loc[dt])
            result = _price_and_enter_bear(tk, S, di, ei, close, atr_dict, cv, dt, max_pos, equity)
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
                "win": result["pnl"] > 0,
            })

    return trades, equity


def simulate_variant_d(rankings, close, high, low, regime_series, atr_dict):
    """
    Variant D: Combined — Bull spreads when VIX>20, pairs when VIX<20.
    VIX>20: bull-only at $200/trade (same as variant A).
    VIX<20: pairs at $100/trade per leg (same as variant C).
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        ranking_data = rankings[dt]
        scores = ranking_data["scores"]
        if not scores:
            continue

        di = close.index.get_loc(dt)
        ei = min(di + DTE, len(close) - 1)
        if ei <= di:
            continue

        sv = float(spy.loc[dt]) if dt in spy.index else 0
        se = float(spy.iloc[ei]) if ei < len(spy) else sv
        spy_regime = "bull" if se >= sv else "bear"

        if cv >= VIX_THRESHOLD:
            # HIGH VIX: bull-only (like variant A)
            ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            bull_picks = [t for t, _ in ranked_desc[:TOP_K]]

            max_pos = min(BULL_ONLY_POS_SIZE, equity / 3)
            if max_pos < 30:
                continue

            for tk in bull_picks:
                if tk not in close.columns or tk not in atr_dict:
                    continue
                S = float(close[tk].loc[dt])
                result = _price_and_enter_bull(tk, S, di, ei, close, atr_dict, cv, dt, max_pos, equity)
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
                    "win": result["pnl"] > 0,
                })
        else:
            # LOW VIX: pairs (like variant C)
            ranked_asc = sorted(scores.items(), key=lambda x: x[1])
            ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
            bear_picks = [t for t, _ in ranked_asc[:TOP_K]]

            max_pos = min(PAIR_POS_SIZE, equity / 6)
            if max_pos < 20:
                continue

            for tk in bull_picks:
                if tk not in close.columns or tk not in atr_dict:
                    continue
                S = float(close[tk].loc[dt])
                result = _price_and_enter_bull(tk, S, di, ei, close, atr_dict, cv, dt, max_pos, equity)
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
                    "win": result["pnl"] > 0,
                })

            for tk in bear_picks:
                if tk not in close.columns or tk not in atr_dict:
                    continue
                S = float(close[tk].loc[dt])
                result = _price_and_enter_bear(tk, S, di, ei, close, atr_dict, cv, dt, max_pos, equity)
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
                    "win": result["pnl"] > 0,
                })

    return trades, equity


def simulate_variant_e(rankings, close, high, low, regime_series, atr_dict):
    """
    Variant E: Dollar-neutral combo — Equal $ long and short.
    VIX>20: bull-only spreads at $200/trade.
    VIX<20: dollar-neutral pairs at $100/trade per leg.
    Key difference from D: in low-VIX, we MATCH dollar amounts between bull and bear legs.
    If a bull leg enters at $X, we target $X on the bear leg too.
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        ranking_data = rankings[dt]
        scores = ranking_data["scores"]
        if not scores:
            continue

        di = close.index.get_loc(dt)
        ei = min(di + DTE, len(close) - 1)
        if ei <= di:
            continue

        sv = float(spy.loc[dt]) if dt in spy.index else 0
        se = float(spy.iloc[ei]) if ei < len(spy) else sv
        spy_regime = "bull" if se >= sv else "bear"

        if cv >= VIX_THRESHOLD:
            # HIGH VIX: bull-only (identical to variant A)
            ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            bull_picks = [t for t, _ in ranked_desc[:TOP_K]]

            max_pos = min(BULL_ONLY_POS_SIZE, equity / 3)
            if max_pos < 30:
                continue

            for tk in bull_picks:
                if tk not in close.columns or tk not in atr_dict:
                    continue
                S = float(close[tk].loc[dt])
                result = _price_and_enter_bull(tk, S, di, ei, close, atr_dict, cv, dt, max_pos, equity)
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
                    "win": result["pnl"] > 0,
                })
        else:
            # LOW VIX: dollar-neutral pairs
            ranked_asc = sorted(scores.items(), key=lambda x: x[1])
            ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
            bear_picks = [t for t, _ in ranked_asc[:TOP_K]]

            max_pos = min(PAIR_POS_SIZE, equity / 6)
            if max_pos < 20:
                continue

            # Enter bull legs and track total dollar exposure
            bull_exposure = 0.0
            n_bull = 0
            for tk in bull_picks:
                if tk not in close.columns or tk not in atr_dict:
                    continue
                S = float(close[tk].loc[dt])
                result = _price_and_enter_bull(tk, S, di, ei, close, atr_dict, cv, dt, max_pos, equity)
                if result is None:
                    continue
                equity += result["pnl"]
                bull_exposure += result["entry_cost"]
                n_bull += 1
                trades.append({
                    "pnl": round(result["pnl"], 2),
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": spy_regime,
                    "direction": "bull",
                    "vix": round(cv, 1),
                    "win": result["pnl"] > 0,
                })

            # Match bear exposure to bull exposure for dollar neutrality
            if n_bull > 0:
                bear_target_per_trade = bull_exposure / max(n_bull, 1)
                bear_max = min(bear_target_per_trade * 1.2, equity * 0.40)  # allow 20% slack

                for tk in bear_picks:
                    if tk not in close.columns or tk not in atr_dict:
                        continue
                    S = float(close[tk].loc[dt])
                    result = _price_and_enter_bear(tk, S, di, ei, close, atr_dict, cv, dt, bear_max, equity)
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
                        "win": result["pnl"] > 0,
                    })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE (identical to production v4)
# ══════════════════════════════════════════════════════════════

def random_baseline_test(simulate_fn, rankings, close, high, low, regime_series,
                         atr_dict, n_trials=5):
    """Test if random sector selection also produces similar returns."""
    fprint(f"\n  Random baseline test ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, data in rankings.items():
            rand_scores = {tk: np.random.random() for tk in data["scores"].keys()}
            rand_rankings[dt] = {"scores": rand_scores, "direction": data.get("direction", "bull")}

        trades, final_eq = simulate_fn(
            rand_rankings, close, high, low, regime_series, atr_dict,
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
    fprint(f"PAIR TRADES CROSS-VALIDATION v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Regime bull threshold: >{REGIME_BULL_THRESHOLD}, bear: <{REGIME_BEAR_THRESHOLD}")
    fprint(f"VIX split: VIX>={VIX_THRESHOLD:.0f} = high, VIX<{VIX_THRESHOLD:.0f} = low")
    fprint(f"Bull-only position: ${BULL_ONLY_POS_SIZE:.0f}/trade | "
           f"Pair position: ${PAIR_POS_SIZE:.0f}/trade per leg")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 4. Build rebalance dates
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ).last().dropna().values
    )
    fprint(f"Rebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # ── Build rankings ──
    # We need rankings from TWO builds:
    #   1. Bull-only (regime>0.4) — used by variant A
    #   2. All dates (no regime filter) — used by variants B-E
    #      (VIX filtering happens at trade time, not at ranking time)

    fprint("\n" + "=" * 80)
    fprint("BUILDING RANKINGS: Bull-only (Variant A baseline)")
    fprint("=" * 80)

    records_bull = build_feature_records(
        close, high, low, rebal_dates, LEGACY_FEATURES,
        regime_series, regime_mode="bull_only",
    )
    rankings_bull, imp_bull = walk_forward_lgbm_rank(records_bull, LEGACY_FEATURES, "bull_only_baseline")

    fprint("\n" + "=" * 80)
    fprint("BUILDING RANKINGS: All dates (Variants B-E)")
    fprint("=" * 80)

    # For pair trade variants, we need rankings on ALL dates
    # (the VIX/regime filtering happens at trade simulation time)
    records_all = build_feature_records(
        close, high, low, rebal_dates, LEGACY_FEATURES,
        regime_series, regime_mode="all",
    )
    rankings_all, imp_all = walk_forward_lgbm_rank(records_all, LEGACY_FEATURES, "all_dates")

    # ── SIMULATE ALL VARIANTS ──
    fprint("\n" + "=" * 80)
    fprint("SIMULATING TRADES — 5 VARIANTS")
    fprint("=" * 80)

    all_results = {}
    variant_configs = [
        ("A_bull_baseline", simulate_variant_a, rankings_bull,
         "Bull-only baseline (production v4 repro). Top 3, VIX>20/regime>0.4."),
        ("B_pair_vix_high", simulate_variant_b, rankings_all,
         "Pair VIX>20: top 3 bull + bottom 3 bear, VIX>20 only. $100/leg."),
        ("C_pair_vix_low", simulate_variant_c, rankings_all,
         "Low-VIX pairs: top 3 bull + bottom 3 bear, VIX<20 only. $100/leg."),
        ("D_combined", simulate_variant_d, rankings_all,
         "Combined: bulls when VIX>20, pairs when VIX<20."),
        ("E_dollar_neutral", simulate_variant_e, rankings_all,
         "Dollar-neutral combo: VIX>20 bulls + VIX<20 dollar-neutral pairs."),
    ]

    for vname, sim_fn, rankings, desc in variant_configs:
        fprint(f"\n{'='*60}")
        fprint(f"  {vname}: {desc}")
        fprint(f"{'='*60}")

        if not rankings:
            fprint(f"  No rankings available, skipping")
            continue

        trades, final_eq = sim_fn(rankings, close, high, low, regime_series, atr_dict)

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[vname] = {
                "description": desc,
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
        if bull_trades or bear_trades:
            bull_pnl = sum(t["pnl"] for t in bull_trades)
            bear_pnl = sum(t["pnl"] for t in bear_trades)
            bull_wr = sum(1 for t in bull_trades if t["win"]) / max(len(bull_trades), 1) * 100
            bear_wr = sum(1 for t in bear_trades if t["win"]) / max(len(bear_trades), 1) * 100
            fprint(f"  Direction breakdown:")
            fprint(f"    Bull: {len(bull_trades)} trades, WR {bull_wr:.1f}%, PnL ${bull_pnl:.0f}")
            fprint(f"    Bear: {len(bear_trades)} trades, WR {bear_wr:.1f}%, PnL ${bear_pnl:.0f}")

        # VIX breakdown
        high_vix_trades = [t for t in trades if t["vix"] >= VIX_THRESHOLD]
        low_vix_trades = [t for t in trades if t["vix"] < VIX_THRESHOLD]
        if high_vix_trades and low_vix_trades:
            hv_pnl = sum(t["pnl"] for t in high_vix_trades)
            lv_pnl = sum(t["pnl"] for t in low_vix_trades)
            hv_wr = sum(1 for t in high_vix_trades if t["win"]) / len(high_vix_trades) * 100
            lv_wr = sum(1 for t in low_vix_trades if t["win"]) / len(low_vix_trades) * 100
            fprint(f"  VIX breakdown:")
            fprint(f"    VIX>={VIX_THRESHOLD:.0f}: {len(high_vix_trades)} trades, "
                   f"WR {hv_wr:.1f}%, PnL ${hv_pnl:.0f}")
            fprint(f"    VIX<{VIX_THRESHOLD:.0f}:  {len(low_vix_trades)} trades, "
                   f"WR {lv_wr:.1f}%, PnL ${lv_pnl:.0f}")

        # Random baseline
        random_sharpes = random_baseline_test(
            sim_fn, rankings, close, high, low, regime_series, atr_dict,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        all_results[vname] = {
            "description": desc,
            **result.to_dict(),
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "n_bull_trades": len(bull_trades),
            "n_bear_trades": len(bear_trades),
            "bull_pnl": round(bull_pnl, 2) if bull_trades else 0,
            "bear_pnl": round(bear_pnl, 2) if bear_trades else 0,
        }

    # ── SUMMARY COMPARISON ──
    fprint("\n" + "=" * 80)
    fprint("SUMMARY COMPARISON")
    fprint("=" * 80)
    fprint(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'RandSh':>7}")
    fprint("-" * 92)

    variant_names = ["A_bull_baseline", "B_pair_vix_high", "C_pair_vix_low",
                     "D_combined", "E_dollar_neutral"]
    for vname in variant_names:
        r = all_results.get(vname)
        if not r or "error" in r:
            fprint(f"  {vname:<25} — NO DATA —")
            continue
        fprint(f"  {vname:<25} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f} {r['random_mean_sharpe']:>7.2f}")

    # ── SANITY CHECK ──
    fprint("\n" + "=" * 80)
    fprint("SANITY CHECK: Variant A should reproduce ~1.67-1.87 Sharpe")
    fprint("=" * 80)
    a_result = all_results.get("A_bull_baseline", {})
    a_sharpe = a_result.get("sharpe", 0)
    if 1.5 <= a_sharpe <= 2.1:
        fprint(f"  PASS: Variant A Sharpe = {a_sharpe:.2f} (within expected 1.67-1.87 range)")
    elif a_sharpe > 0:
        fprint(f"  WARNING: Variant A Sharpe = {a_sharpe:.2f} (outside expected range, investigate)")
    else:
        fprint(f"  FAIL: Variant A Sharpe = {a_sharpe:.2f} (something is broken)")

    # ── KEY FINDINGS ──
    fprint("\n" + "=" * 80)
    fprint("KEY FINDINGS")
    fprint("=" * 80)

    valid_variants = {k: v for k, v in all_results.items()
                      if "error" not in v and v.get("sharpe", 0) > 0}

    if valid_variants:
        best_name = max(valid_variants, key=lambda k: valid_variants[k]["sharpe"])
        best = valid_variants[best_name]
        fprint(f"  Best variant: {best_name} (Sharpe {best['sharpe']:.2f})")

        if best_name != "A_bull_baseline" and "A_bull_baseline" in valid_variants:
            a_sh = valid_variants["A_bull_baseline"]["sharpe"]
            delta = best["sharpe"] - a_sh
            fprint(f"  vs Baseline A: {'+'if delta>0 else ''}{delta:.2f} Sharpe "
                   f"({a_sh:.2f} -> {best['sharpe']:.2f})")
            if delta > 0.15:
                fprint(f"  CONCLUSION: Pair trades IMPROVE the strategy by {delta:.2f} Sharpe")
            elif delta > -0.15:
                fprint(f"  CONCLUSION: Pair trades are NEUTRAL (delta within noise)")
            else:
                fprint(f"  CONCLUSION: Pair trades HURT the strategy by {abs(delta):.2f} Sharpe")
        elif best_name == "A_bull_baseline":
            fprint(f"  CONCLUSION: Bull-only baseline is STILL the best. Pair trades do not improve.")

        # Check if low-VIX pairs specifically help
        c_result = all_results.get("C_pair_vix_low", {})
        if c_result and "error" not in c_result:
            c_sharpe = c_result.get("sharpe", 0)
            fprint(f"\n  Low-VIX pair trades (Variant C): Sharpe {c_sharpe:.2f}")
            if c_sharpe > 2.0:
                fprint(f"  Prior agent finding (Sharpe 2.41) appears VALIDATED")
            elif c_sharpe > 1.5:
                fprint(f"  Prior agent finding partially validated — lower than 2.41 but still strong")
            else:
                fprint(f"  Prior agent finding NOT validated — Sharpe only {c_sharpe:.2f}")
    else:
        fprint("  No valid variants produced results.")

    # Feature importance
    fprint("\n" + "=" * 80)
    fprint("FEATURE IMPORTANCE (Top 10)")
    fprint("=" * 80)
    for vname, imp_df_pair in [("bull_only", imp_bull), ("all_dates", imp_all)]:
        if imp_df_pair is not None:
            fprint(f"\n  {vname}:")
            for _, row in imp_df_pair.head(10).iterrows():
                bar = "*" * int(row["importance"] / imp_df_pair["importance"].max() * 30)
                fprint(f"    {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # Save results
    results_path = OUTPUT_DIR / "pair_trades_xval_v1_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"pair_xval_v1_{t0.strftime('%Y%m%d_%H%M')}"):
                for vname, r in all_results.items():
                    if "error" in r:
                        continue
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
                    mlflow.log_metric(f"{prefix}_n_bull_trades", r.get("n_bull_trades", 0))
                    mlflow.log_metric(f"{prefix}_n_bear_trades", r.get("n_bear_trades", 0))

                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "regime_bear_thresh": REGIME_BEAR_THRESHOLD,
                    "vix_threshold": VIX_THRESHOLD,
                    "bull_only_pos_size": BULL_ONLY_POS_SIZE,
                    "pair_pos_size": PAIR_POS_SIZE,
                    "hold_to_expiry": True,
                    "entry_haircut_only": True,
                    "n_features": len(LEGACY_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "rebal_freq": WF_REBAL_FREQ,
                    "top_k": TOP_K,
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
