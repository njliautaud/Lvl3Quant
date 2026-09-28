#!/usr/bin/env python3
"""
Entry Timing Cross-Validation v1 — Does Entry Timing Improve Bull-Spread Returns?
==================================================================================

Production v4 enters trades at rebalance-date close. This tests whether timing
the entry differently can improve risk-adjusted returns.

6 Variants:
  A: Same-day close baseline     — Enter at rebalance date close (must reproduce ~1.86 Sharpe)
  B: Next-day open               — Signal at close → trade next morning open
  C: 2-day delay                  — Enter 2 trading days after signal
  D: Best of 3 days              — Enter at LOWEST close in days [0,1,2] (upper bound on timing)
  E: VIX-dip entry               — Enter on first down day (close/open-1 < 0) within 3 days; else day 3
  F: Staggered entry             — Split 3 trades across 3 consecutive days (1/day)

All variants use EXACTLY production v4 LGBM walk-forward, GRU regime, costs, spreads.
5-gate adversarial validation + random baseline comparison (5 trials).

Capital $645, $200/trade, $2.60 commission, 15% haircut, DTE=21, 3% spread,
biweekly rebalance, top-3 sectors, bull-only, hold to expiry.
"""

import json
import os
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


# ── Auto-detect Neptune vs Jupiter ──
_hostname = os.uname().nodename.lower()
if 'neptune' in _hostname or 'nick' in str(os.path.expanduser('~')):
    BASE = Path('/home/nick/Lvl3Quant')
else:
    BASE = Path('/home/jupiter/Lvl3Quant')

# ── Standardized tools ──
sys.path.insert(0, str(BASE))
from research.tools.options_pricer import (
    price_bull_call_spread,
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

# ── Config ──
OUTPUT_DIR = BASE / "output" / "growth_research" / "entry_timing_xval_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12
WF_REBAL_FREQ = "2W-FRI"

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "entry_timing_xval_v1"

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
# DATA DOWNLOAD (includes Open prices for variant E)
# ══════════════════════════════════════════════════════════════

def download_data():
    """Download all required tickers via yfinance, including Open prices."""
    import yfinance as yf

    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")

    raw = yf.download(all_tickers, start="2008-01-01", progress=False, auto_adjust=True)
    mi = isinstance(raw.columns, pd.MultiIndex)

    close = raw["Close"] if mi else raw[["Close"]]
    high = raw["High"] if mi else raw[["High"]]
    low = raw["Low"] if mi else raw[["Low"]]
    open_ = raw["Open"] if mi else raw[["Open"]]

    for df in [close, high, low, open_]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

    # Fix column names for MultiIndex case — need reassignment
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    if isinstance(high.columns, pd.MultiIndex):
        high.columns = high.columns.get_level_values(-1)
    if isinstance(low.columns, pd.MultiIndex):
        low.columns = low.columns.get_level_values(-1)
    if isinstance(open_.columns, pd.MultiIndex):
        open_.columns = open_.columns.get_level_values(-1)

    close = close.ffill()
    high = high.ffill()
    low = low.ffill()
    open_ = open_.ffill()

    rename_map = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)
    open_ = open_.rename(columns=rename_map)

    needed = ["SPY", "VIX"]
    for t in needed:
        if t not in close.columns:
            raise ValueError(f"Missing critical ticker: {t}")

    fprint(f"Data: {len(close)} days, columns: {list(close.columns)}")
    fprint(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    return close, high, low, open_


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


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (identical to production v4)
# ══════════════════════════════════════════════════════════════

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

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """
    Build feature + target records for all sectors on all rebal dates.
    Bull-only mode (regime>0.4 filter).
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
# ENTRY TIMING HELPERS
# ══════════════════════════════════════════════════════════════

def get_future_trading_days(close, dt, n_days):
    """
    Get the next n_days trading day indices after dt (inclusive of dt as day 0).
    Returns list of (date, integer_index) tuples.
    """
    di = close.index.get_loc(dt) if dt in close.index else None
    if di is None:
        di = close.index.get_indexer([dt], method="ffill")[0]
    result = []
    for offset in range(n_days):
        idx = di + offset
        if idx < len(close):
            result.append((close.index[idx], idx))
    return result


def find_vix_dip_day(open_df, close, dt, max_days=3):
    """
    Find the first day within max_days where intra-day move (close/open - 1) < 0.
    Returns (entry_date, entry_index). Falls back to day max_days-1 if no dip found.
    """
    future = get_future_trading_days(close, dt, max_days)
    for entry_dt, entry_idx in future:
        if entry_dt in open_df.index and entry_dt in close.index:
            # Check SPY intra-day move as a proxy for "down day"
            spy_open = float(open_df["SPY"].loc[entry_dt]) if "SPY" in open_df.columns else None
            spy_close = float(close["SPY"].loc[entry_dt]) if "SPY" in close.columns else None
            if spy_open and spy_close and spy_open > 0:
                intraday_ret = spy_close / spy_open - 1
                if intraday_ret < 0:
                    return entry_dt, entry_idx
    # No dip found — enter on last day
    if future:
        return future[-1]
    return dt, close.index.get_loc(dt)


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION — ENTRY TIMING VARIANTS
# ══════════════════════════════════════════════════════════════

def _price_and_trade_single(tk, entry_price, di_entry, close, atr_dict, vix_val,
                            effective_dte, equity, max_pos):
    """
    Price a single bull call spread given an entry price and compute PnL at expiry.
    Returns (pnl, trade_record_dict) or (None, None) if trade cannot be placed.
    """
    S = entry_price
    ei = min(di_entry + effective_dte, len(close) - 1)
    if ei <= di_entry:
        return None, None

    # ATR at entry date
    entry_dt = close.index[di_entry]
    if tk in atr_dict and entry_dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[entry_dt]):
        av = float(atr_dict[tk].loc[entry_dt])
    else:
        av = S * 0.015

    # Strike selection: ATM at entry price
    K1 = round(S, 2)
    K2 = round(S * (1 + SPREAD_PCT / 100), 2)
    if K2 <= K1:
        K2 = K1 + 1.0

    try:
        entry_cost_ps, max_profit_ps = price_bull_call_spread(
            S=S, K1=K1, K2=K2, dte=effective_dte, atr=av, vix=vix_val
        )
    except Exception:
        return None, None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None, None

    # Hold to expiry: intrinsic value
    Se = float(close[tk].iloc[ei])
    intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    exit_value_ps = intrinsic

    pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

    spy = close["SPY"]
    sv = float(spy.iloc[di_entry]) if di_entry < len(spy) else 0
    se = float(spy.iloc[ei]) if ei < len(spy) else sv
    spy_regime = "bull" if se >= sv else "bear"

    trade = {
        "pnl": round(pnl, 2),
        "entry_date": str(entry_dt.date()),
        "exit_date": str(close.index[ei].date()),
        "ticker": tk,
        "regime": spy_regime,
        "vix": round(vix_val, 1),
        "win": pnl > 0,
        "entry_price": round(S, 2),
    }

    return pnl, trade


def simulate_variant_a(rankings, close, atr_dict):
    """Variant A: Same-day close baseline (production v4 exact)."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]
        max_pos = min(200, equity / 3)
        if max_pos < 30:
            continue

        di = close.index.get_loc(dt)
        n_entered = 0
        for tk in picks:
            if tk not in close.columns or tk not in atr_dict or n_entered >= TOP_K:
                continue
            S = float(close[tk].loc[dt])
            pnl, trade = _price_and_trade_single(
                tk, S, di, close, atr_dict, cv, DTE, equity, max_pos
            )
            if pnl is not None:
                equity += pnl
                trades.append(trade)
                n_entered += 1

    return trades, equity


def simulate_variant_b(rankings, close, open_df, atr_dict):
    """Variant B: Next-day open — enter at next trading day's open price."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]
        max_pos = min(200, equity / 3)
        if max_pos < 30:
            continue

        di = close.index.get_loc(dt)
        # Next trading day
        next_di = di + 1
        if next_di >= len(close):
            continue
        next_dt = close.index[next_di]

        n_entered = 0
        for tk in picks:
            if tk not in close.columns or tk not in atr_dict or n_entered >= TOP_K:
                continue
            if tk not in open_df.columns:
                continue
            # Enter at next day's open
            S = float(open_df[tk].iloc[next_di])
            if pd.isna(S) or S <= 0:
                continue
            effective_dte = DTE - 1  # one day less to expiry
            if effective_dte < 5:
                continue
            pnl, trade = _price_and_trade_single(
                tk, S, next_di, close, atr_dict, cv, effective_dte, equity, max_pos
            )
            if pnl is not None:
                trade["entry_type"] = "next_day_open"
                equity += pnl
                trades.append(trade)
                n_entered += 1

    return trades, equity


def simulate_variant_c(rankings, close, atr_dict):
    """Variant C: 2-day delay — enter at close 2 trading days after signal."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]
        max_pos = min(200, equity / 3)
        if max_pos < 30:
            continue

        di = close.index.get_loc(dt)
        # 2-day delay
        delay_di = di + 2
        if delay_di >= len(close):
            continue

        effective_dte = DTE - 2
        if effective_dte < 5:
            continue

        n_entered = 0
        for tk in picks:
            if tk not in close.columns or tk not in atr_dict or n_entered >= TOP_K:
                continue
            S = float(close[tk].iloc[delay_di])
            if pd.isna(S) or S <= 0:
                continue
            pnl, trade = _price_and_trade_single(
                tk, S, delay_di, close, atr_dict, cv, effective_dte, equity, max_pos
            )
            if pnl is not None:
                trade["entry_type"] = "2day_delay"
                equity += pnl
                trades.append(trade)
                n_entered += 1

    return trades, equity


def simulate_variant_d(rankings, close, atr_dict):
    """Variant D: Best of 3 days — enter at LOWEST close in days [0,1,2]."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]
        max_pos = min(200, equity / 3)
        if max_pos < 30:
            continue

        di = close.index.get_loc(dt)
        if di + 2 >= len(close):
            continue

        n_entered = 0
        for tk in picks:
            if tk not in close.columns or tk not in atr_dict or n_entered >= TOP_K:
                continue

            # Find lowest close price in days 0, 1, 2
            best_price = float('inf')
            best_offset = 0
            for offset in range(3):
                check_di = di + offset
                if check_di >= len(close):
                    break
                p = float(close[tk].iloc[check_di])
                if not pd.isna(p) and p > 0 and p < best_price:
                    best_price = p
                    best_offset = offset

            if best_price == float('inf'):
                continue

            entry_di = di + best_offset
            effective_dte = DTE - best_offset
            if effective_dte < 5:
                continue

            pnl, trade = _price_and_trade_single(
                tk, best_price, entry_di, close, atr_dict, cv, effective_dte, equity, max_pos
            )
            if pnl is not None:
                trade["entry_type"] = f"best_of_3_day{best_offset}"
                equity += pnl
                trades.append(trade)
                n_entered += 1

    return trades, equity


def simulate_variant_e(rankings, close, open_df, atr_dict):
    """Variant E: VIX-dip entry — enter on first down day within 3 days, else day 3."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]
        max_pos = min(200, equity / 3)
        if max_pos < 30:
            continue

        di = close.index.get_loc(dt)
        if di + 2 >= len(close):
            continue

        # Find first dip day (intra-day SPY close/open - 1 < 0)
        entry_offset = 2  # default: day 2 (3rd day, 0-indexed)
        for offset in range(3):
            check_di = di + offset
            if check_di >= len(close):
                break
            check_dt = close.index[check_di]
            if "SPY" in open_df.columns and check_dt in open_df.index:
                spy_o = float(open_df["SPY"].iloc[check_di])
                spy_c = float(close["SPY"].iloc[check_di])
                if spy_o > 0 and (spy_c / spy_o - 1) < 0:
                    entry_offset = offset
                    break

        entry_di = di + entry_offset
        if entry_di >= len(close):
            continue
        effective_dte = DTE - entry_offset
        if effective_dte < 5:
            continue

        n_entered = 0
        for tk in picks:
            if tk not in close.columns or tk not in atr_dict or n_entered >= TOP_K:
                continue
            S = float(close[tk].iloc[entry_di])
            if pd.isna(S) or S <= 0:
                continue
            pnl, trade = _price_and_trade_single(
                tk, S, entry_di, close, atr_dict, cv, effective_dte, equity, max_pos
            )
            if pnl is not None:
                trade["entry_type"] = f"vix_dip_day{entry_offset}"
                equity += pnl
                trades.append(trade)
                n_entered += 1

    return trades, equity


def simulate_variant_f(rankings, close, atr_dict):
    """
    Variant F: Staggered entry — split 3 trades across 3 consecutive days.
    Day 0: enter trade for pick #1, Day 1: pick #2, Day 2: pick #3.
    Each gets 1/3 of the normal position sizing.
    """
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]
        max_pos = min(200, equity / 3)
        if max_pos < 30:
            continue

        di = close.index.get_loc(dt)
        if di + 2 >= len(close):
            continue

        # Stagger: pick[i] enters on day i
        for pick_idx, tk in enumerate(picks):
            if tk not in close.columns or tk not in atr_dict:
                continue
            entry_di = di + pick_idx
            if entry_di >= len(close):
                continue
            effective_dte = DTE - pick_idx
            if effective_dte < 5:
                continue
            S = float(close[tk].iloc[entry_di])
            if pd.isna(S) or S <= 0:
                continue
            pnl, trade = _price_and_trade_single(
                tk, S, entry_di, close, atr_dict, cv, effective_dte, equity, max_pos
            )
            if pnl is not None:
                trade["entry_type"] = f"staggered_day{pick_idx}"
                equity += pnl
                trades.append(trade)

    return trades, equity


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, atr_dict, n_trials=5):
    """Test if random sector selection also produces similar returns (variant A baseline)."""
    fprint(f"\n  Random baseline test ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_scores = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = rand_scores

        trades, final_eq = simulate_variant_a(rand_rankings, close, atr_dict)

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
    fprint(f"ENTRY TIMING CROSS-VALIDATION v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Regime bull threshold: >{REGIME_BULL_THRESHOLD} | Bull-only | Top-{TOP_K} sectors")
    fprint(f"Features: 21 (18 legacy + 3 cross-asset) | WF train: {WF_TRAIN_PERIODS} periods")
    fprint()
    fprint("VARIANTS:")
    fprint("  A: Same-day close baseline (must reproduce ~1.86 Sharpe)")
    fprint("  B: Next-day open (realistic execution delay)")
    fprint("  C: 2-day delay (signal persistence test)")
    fprint("  D: Best of 3 days (upper bound on timing)")
    fprint("  E: VIX-dip entry (buy on down days)")
    fprint("  F: Staggered entry (1 trade/day over 3 days)")
    fprint()

    # 1. Download data (including open prices)
    close, high, low, open_df = download_data()

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

    # ── BUILD FEATURES AND LGBM RANKINGS (one pass, shared across all variants) ──
    fprint("\n" + "=" * 80)
    fprint("BUILDING FEATURES + LGBM WALK-FORWARD RANKINGS")
    fprint("=" * 80)

    records = build_feature_records(
        close, high, low, rebal_dates, V4_FEATURES, regime_series,
    )
    rankings, imp_df = walk_forward_lgbm_rank(records, V4_FEATURES, "entry_timing")

    if not rankings:
        fprint("ERROR: No rankings generated. Cannot proceed.")
        return

    fprint(f"  Generated rankings for {len(rankings)} rebalance dates")

    # ── SIMULATE ALL 6 VARIANTS ──
    fprint("\n" + "=" * 80)
    fprint("SIMULATING 6 ENTRY TIMING VARIANTS")
    fprint("=" * 80)

    all_results = {}
    variant_info = {
        "A_sameday_close": "Same-day close (baseline)",
        "B_nextday_open": "Next-day open",
        "C_2day_delay": "2-day delay close",
        "D_best_of_3": "Best (lowest) of 3 days",
        "E_vix_dip": "VIX-dip entry (first down day in 3)",
        "F_staggered": "Staggered (1 trade/day over 3 days)",
    }

    # --- Variant A ---
    fprint(f"\n--- A: Same-day close baseline ---")
    trades_a, eq_a = simulate_variant_a(rankings, close, atr_dict)

    # --- Variant B ---
    fprint(f"\n--- B: Next-day open ---")
    trades_b, eq_b = simulate_variant_b(rankings, close, open_df, atr_dict)

    # --- Variant C ---
    fprint(f"\n--- C: 2-day delay ---")
    trades_c, eq_c = simulate_variant_c(rankings, close, atr_dict)

    # --- Variant D ---
    fprint(f"\n--- D: Best of 3 days ---")
    trades_d, eq_d = simulate_variant_d(rankings, close, atr_dict)

    # --- Variant E ---
    fprint(f"\n--- E: VIX-dip entry ---")
    trades_e, eq_e = simulate_variant_e(rankings, close, open_df, atr_dict)

    # --- Variant F ---
    fprint(f"\n--- F: Staggered entry ---")
    trades_f, eq_f = simulate_variant_f(rankings, close, atr_dict)

    variant_trades = {
        "A_sameday_close": (trades_a, eq_a),
        "B_nextday_open": (trades_b, eq_b),
        "C_2day_delay": (trades_c, eq_c),
        "D_best_of_3": (trades_d, eq_d),
        "E_vix_dip": (trades_e, eq_e),
        "F_staggered": (trades_f, eq_f),
    }

    # ── VALIDATE ALL VARIANTS ──
    fprint("\n" + "=" * 80)
    fprint("5-GATE ADVERSARIAL VALIDATION")
    fprint("=" * 80)

    for vname, (trades, final_eq) in variant_trades.items():
        desc = variant_info[vname]
        fprint(f"\n--- {vname}: {desc} ---")

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[vname] = {
                "description": desc,
                "n_trades": len(trades) if trades else 0,
                "skipped": True,
            }
            continue

        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=vname,
        )
        result.print_summary()

        # Entry type breakdown (for non-A variants)
        if trades and "entry_type" in trades[0]:
            entry_types = {}
            for t in trades:
                et = t.get("entry_type", "unknown")
                if et not in entry_types:
                    entry_types[et] = {"count": 0, "pnl": 0.0, "wins": 0}
                entry_types[et]["count"] += 1
                entry_types[et]["pnl"] += t["pnl"]
                entry_types[et]["wins"] += 1 if t["win"] else 0
            fprint(f"  Entry type breakdown:")
            for et, info in sorted(entry_types.items()):
                wr = info["wins"] / info["count"] * 100 if info["count"] > 0 else 0
                fprint(f"    {et}: {info['count']} trades, WR {wr:.1f}%, PnL ${info['pnl']:.0f}")

        all_results[vname] = {
            "description": desc,
            **result.to_dict(),
            "final_equity": round(final_eq, 2),
        }

    # ── RANDOM BASELINE (against variant A rankings) ──
    fprint("\n" + "=" * 80)
    fprint("RANDOM BASELINE (vs Variant A)")
    fprint("=" * 80)

    random_sharpes = random_baseline_test(rankings, close, atr_dict)
    mean_random = np.mean(random_sharpes) if random_sharpes else 0
    for vname in all_results:
        if not all_results[vname].get("skipped", False):
            all_results[vname]["random_sharpes"] = [round(s, 3) for s in random_sharpes]
            all_results[vname]["random_mean_sharpe"] = round(mean_random, 3)

    # ── SUMMARY COMPARISON ──
    fprint("\n" + "=" * 80)
    fprint("SUMMARY COMPARISON — ENTRY TIMING VARIANTS")
    fprint("=" * 80)
    fprint(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8}")
    fprint("-" * 85)

    for vname in ["A_sameday_close", "B_nextday_open", "C_2day_delay",
                  "D_best_of_3", "E_vix_dip", "F_staggered"]:
        r = all_results.get(vname, {})
        if r.get("skipped", False):
            fprint(f"  {vname:<25} {r.get('n_trades', 0):>5}  — SKIPPED (too few trades) —")
            continue
        fprint(f"  {vname:<25} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f}")

    fprint(f"\n  Random baseline mean Sharpe: {mean_random:.2f}")

    # ── KEY FINDINGS ──
    fprint("\n" + "=" * 80)
    fprint("KEY FINDINGS")
    fprint("=" * 80)

    a_sharpe = all_results.get("A_sameday_close", {}).get("sharpe", 0)
    fprint(f"  Baseline (A) Sharpe: {a_sharpe:.2f}")

    best_name = "A_sameday_close"
    best_sharpe = a_sharpe
    for vname in ["B_nextday_open", "C_2day_delay", "D_best_of_3", "E_vix_dip", "F_staggered"]:
        r = all_results.get(vname, {})
        if not r.get("skipped", False) and r.get("sharpe", 0) > best_sharpe:
            best_sharpe = r["sharpe"]
            best_name = vname

    if best_name != "A_sameday_close":
        delta = best_sharpe - a_sharpe
        fprint(f"  BEST: {best_name} ({variant_info[best_name]}) — Sharpe {best_sharpe:.2f} "
               f"(+{delta:.2f} vs baseline)")
    else:
        fprint(f"  No variant beats baseline. Same-day close is optimal.")

    # Signal persistence check
    for vname, label in [("B_nextday_open", "1-day"), ("C_2day_delay", "2-day")]:
        r = all_results.get(vname, {})
        if not r.get("skipped", False):
            delay_sharpe = r.get("sharpe", 0)
            decay = (a_sharpe - delay_sharpe) / max(a_sharpe, 0.01) * 100
            fprint(f"  Signal decay after {label} delay: {decay:.1f}% (Sharpe {a_sharpe:.2f} -> {delay_sharpe:.2f})")

    # Upper bound check
    d_r = all_results.get("D_best_of_3", {})
    if not d_r.get("skipped", False):
        d_sharpe = d_r.get("sharpe", 0)
        uplift = (d_sharpe - a_sharpe) / max(a_sharpe, 0.01) * 100
        fprint(f"  Upper bound on timing (D): Sharpe {d_sharpe:.2f} ({uplift:+.1f}% vs baseline)")
        if uplift < 10:
            fprint(f"    -> Entry timing has LIMITED potential (<10% uplift ceiling)")
        else:
            fprint(f"    -> Entry timing has MEANINGFUL potential ({uplift:.0f}% uplift ceiling)")

    # Feature importance
    if imp_df is not None:
        fprint("\n" + "=" * 80)
        fprint("FEATURE IMPORTANCE (Top 10)")
        fprint("=" * 80)
        for _, row in imp_df.head(10).iterrows():
            bar = "*" * int(row["importance"] / imp_df["importance"].max() * 30)
            fprint(f"  {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # ── SAVE RESULTS ──
    results_path = OUTPUT_DIR / "entry_timing_xval_v1_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save per-variant trade logs
    for vname, (trades, _) in variant_trades.items():
        if trades:
            trades_path = OUTPUT_DIR / f"{vname}_trades.json"
            with open(trades_path, "w") as f:
                json.dump(trades, f, indent=2, default=str)

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"entry_timing_v1_{t0.strftime('%Y%m%d_%H%M')}"):
                for vname, r in all_results.items():
                    if r.get("skipped", False):
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

                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "hold_to_expiry": True,
                    "entry_haircut_only": True,
                    "n_features": len(V4_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "rebal_freq": WF_REBAL_FREQ,
                    "top_k": TOP_K,
                    "n_variants": 6,
                    "random_mean_sharpe": mean_random,
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
