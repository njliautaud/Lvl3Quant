#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V10 Realistic Pricing V1 -- Honest Performance with Calibrated Option Costs
=============================================================================

CONTEXT (KB #282): Our ATR-based BS IV estimation underprices options by
61-73% vs market mids. A linear calibration found:

    market_mid ~ 1.10 * BS_price + 6.60  (per share, R2=0.873)

This means every V10 backtest Sharpe is inflated because we assume cheaper
entry costs. This script re-runs the V10 sector spread strategy with SIX
pricing variants to bracket the REAL performance:

  A: V10 baseline (original BS pricing with 15% haircut -- control)
  B: Linear calibration (market_mid = 1.10*BS + $6.60/100sh, then 15% haircut)
  C: 2x BS cost (double the BS entry cost as conservative proxy)
  D: 2.5x BS cost (roughly matches median underpricing)
  E: 3x BS cost (worst-case stress test)
  F: Calibrated + 50% profit target (realistic pricing with early exit)

V10 STRATEGY CORE:
  - 11 sectors: XLK XLF XLE XLV XLY XLP XLI XLB XLU XLRE XLC
  - Extra: SPY, ^VIX, ^VIX3M, TLT, SHY, HYG, GLD
  - 21 LGBM features (18 legacy quality-momentum + 3 cross-asset)
  - SLIDING walk-forward: 500d train window, LGBM ranking
  - VIX >= 20: bull spreads on top-2 ranked sectors (high-VIX mode)
  - VIX <  20: bull top-3 + bear bottom-3 (low-VIX pair trades)
  - DTE=28, 3% adaptive width = max($3, K1*0.03), 2% OTM moneyness
  - Biweekly rebalance (10 trading days)
  - $645 starting capital, $200 max per trade
  - Commission: $2.60 per spread RT

4-Gate adversarial validation on each variant:
  1. Permutation test (300 sign-flip trials)
  2. Regime balance (R1 < 0.50)
  3. Sub-period (both halves Sharpe > 0.5)
  4. Outlier removal (trim top 5% months, still profitable)

Output: output/growth_research/v10_realistic_pricing_v1/
"""

import json
import sys
import time
import warnings
from datetime import datetime
from math import exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

_builtin_print = print


def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# -- Detect environment --
_BASES = [
    Path("C:/Users/claude/Lvl3Quant"),   # Razer (Windows)
    Path("/home/nick/Lvl3Quant"),        # Neptune
    Path("/home/jupiter/Lvl3Quant"),     # Jupiter
]
BASE = None
for _b in _BASES:
    if _b.exists():
        BASE = _b
        break
if BASE is None:
    BASE = Path.cwd()
fprint("Running on: %s" % BASE)

OUTPUT_DIR = BASE / "output" / "growth_research" / "v10_realistic_pricing_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Constants --
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]

CAP = 645.0
DTE = 28
MAX_POS_DOLLARS = 200.0
COMMISSION_RT = 2.60
EARLY_EXIT_COMMISSION = 2.60
COST_WIDTH_MAX = 0.50
RISK_FREE_RATE = 0.045

# V10 structural params
OTM_PCT = 0.02           # 2% OTM moneyness
WIDTH_FLOOR_DOLLARS = 3.0
WIDTH_FLOOR_PCT = 0.03   # 3% adaptive width
REBALANCE_DAYS = 10      # biweekly (10 trading days)
WF_TRAIN_DAYS = 500      # 500d sliding window for LGBM

# VIX-based mode switching
VIX_THRESHOLD = 20.0
HIGH_VIX_BULL_K = 2      # VIX >= 20: bull top-2 only
LOW_VIX_BULL_K = 3       # VIX <  20: bull top-3
LOW_VIX_BEAR_K = 3       # VIX <  20: bear bottom-3

# Adversarial config
N_PERMUTATIONS = 300
SUB_PERIOD_SHARPE_MIN = 0.5

# Calibration constants (from KB #282 analysis)
CALIB_SLOPE = 1.10
CALIB_INTERCEPT_PS = 6.60 / 100.0  # $6.60 per 100 shares -> per-share

# -- MLflow (optional) --
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v10_realistic_pricing_v1"
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint("MLflow connected: %s" % MLFLOW_URI)
except Exception:
    fprint("MLflow unavailable -- will skip logging")


# ==============================================================
# PRICING VARIANT DEFINITIONS
# ==============================================================

VARIANTS = [
    # (key, label, config_dict)
    ("A_baseline",
     "A: V10 baseline (BS + 15% haircut)",
     {"mode": "haircut", "haircut": 0.15, "profit_target": None}),

    ("B_calibrated",
     "B: Linear calibration (1.10*BS + $0.066/sh, then 15% haircut)",
     {"mode": "calibrated", "haircut": 0.15, "profit_target": None}),

    ("C_2x_cost",
     "C: 2x BS cost",
     {"mode": "multiplier", "multiplier": 2.0, "profit_target": None}),

    ("D_2p5x_cost",
     "D: 2.5x BS cost (median underpricing proxy)",
     {"mode": "multiplier", "multiplier": 2.5, "profit_target": None}),

    ("E_3x_cost",
     "E: 3x BS cost (worst-case stress)",
     {"mode": "multiplier", "multiplier": 3.0, "profit_target": None}),

    ("F_calib_pt50",
     "F: Calibrated + 50% profit target",
     {"mode": "calibrated", "haircut": 0.15, "profit_target": 0.50}),
]


# ==============================================================
# SELF-CONTAINED BLACK-SCHOLES PRICING
# ==============================================================

def _bs_call(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
    """Black-Scholes European call price per share."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return float(S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2))


def _bs_put(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
    """Black-Scholes European put via put-call parity: P = C - S + K*exp(-rT)."""
    c = _bs_call(S, K, T, r, sigma)
    return max(c - S + K * exp(-r * T), 0.0)


def _estimate_iv(atr, S, vix=20.0):
    """ATR-based IV: max(0.10, atr/S * sqrt(252) * 1.2)."""
    if S <= 0 or atr <= 0:
        return 0.25
    realized = (atr / S) * sqrt(252.0)
    sigma = realized * 1.2
    return max(sigma, 0.10)


def _price_spread_bs(S, K1, K2, dte, atr, vix, direction):
    """Price a vertical spread using BS. Returns (entry_cost_per_share, sigma).
    Bull call: buy C(K1) - sell C(K2).   K1 < K2.
    Bear put:  buy P(K2) - sell P(K1).   K1 < K2.
    """
    T = dte / 365.0
    sigma = _estimate_iv(atr, S, vix)
    if direction == "bull":
        fv = _bs_call(S, K1, T, sigma=sigma) - _bs_call(S, K2, T, sigma=sigma)
    else:
        fv = _bs_put(S, K2, T, sigma=sigma) - _bs_put(S, K1, T, sigma=sigma)
    return max(fv, 0.001), sigma


def _apply_pricing(bs_fair_ps, cfg):
    """Apply variant-specific pricing adjustment to raw BS fair value per share.
    Returns adjusted entry cost per share.
    """
    mode = cfg["mode"]
    if mode == "haircut":
        return bs_fair_ps * (1.0 + cfg["haircut"])
    elif mode == "calibrated":
        # Step 1: calibrate to market mid
        calibrated = CALIB_SLOPE * bs_fair_ps + CALIB_INTERCEPT_PS
        # Step 2: apply haircut on top (you pay more than mid)
        return calibrated * (1.0 + cfg.get("haircut", 0.15))
    elif mode == "multiplier":
        # Simple multiplier on the BS+15% haircut baseline
        baseline = bs_fair_ps * (1.0 + 0.15)
        return baseline * cfg["multiplier"]
    else:
        return bs_fair_ps * 1.15


def _revalue_spread_bs(S, K1, K2, dte_remaining, atr, vix, direction):
    """Revalue a spread at exit. Uses 15% haircut DOWN (conservative exit)."""
    if dte_remaining <= 0:
        if direction == "bull":
            return max(S - K1, 0.0) - max(S - K2, 0.0)
        else:
            return max(K2 - S, 0.0) - max(K1 - S, 0.0)
    fv, _ = _price_spread_bs(S, K1, K2, dte_remaining, atr, vix, direction)
    return fv * (1.0 - 0.15)


# ==============================================================
# FEATURES (21 total: 18 legacy + 3 cross-asset)
# ==============================================================

LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high",
    "mom_accel", "pct_pos_months_12m", "sortino_63d", "calmar_1y",
    "up_capture", "trend_r2_63d", "trend_slope_63d",
]

CROSS_ASSET_FEATURES = [
    "spy_corr_63d",       # 63d rolling correlation with SPY
    "tlt_corr_63d",       # 63d rolling correlation with TLT
    "vix_beta_63d",       # 63d beta to VIX changes
]

ALL_FEATURES = LEGACY_FEATURES + CROSS_ASSET_FEATURES
assert len(ALL_FEATURES) == 21


def compute_features(px, spy_px, tlt_px, vix_px):
    """Compute 21 features for a single sector ETF at a point in time."""
    if len(px) < 260:
        return None
    f = {}
    # Returns
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()
    # Volatility
    f["vol_21d"] = float(rets.iloc[-21:].std() * sqrt(252)) if len(rets) > 21 else 0.2
    f["vol_63d"] = float(rets.iloc[-63:].std() * sqrt(252)) if len(rets) > 63 else 0.2
    # Sharpe 63d
    r63 = rets.iloc[-63:]
    f["sharpe_63d"] = float(r63.mean() / (r63.std() + 1e-10) * sqrt(252)) if len(r63) > 10 else 0.0
    # MaxDD 63d
    pk63 = px.iloc[-63:].cummax()
    f["maxdd_63d"] = float(((px.iloc[-63:] / pk63) - 1).min())
    # Pct of 52w high
    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    # Momentum acceleration
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3
    # Pct positive months (12m)
    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    # Sortino 63d
    dr = r63[r63 < 0]
    f["sortino_63d"] = float(r63.mean() / (dr.std() + 1e-10) * sqrt(252)) if len(dr) > 3 else 0.0
    # Calmar 1y
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f["calmar_1y"] = cagr / (abs(mdd) + 1e-10)
    # Up capture ratio 63d
    up_days = rets[rets > 0]
    f["up_capture"] = float(
        up_days.iloc[-63:].mean() / (up_days.mean() + 1e-10)
    ) if len(up_days) > 10 else 1.0
    # Trend R2 and slope 63d
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f["trend_r2_63d"] = r_val ** 2
        f["trend_slope_63d"] = slope
    else:
        f["trend_r2_63d"] = 0.0
        f["trend_slope_63d"] = 0.0

    # Cross-asset features (3)
    spy_r = spy_px.pct_change().dropna()
    tlt_r = tlt_px.pct_change().dropna()
    vix_r = vix_px.pct_change().dropna()

    # Align to common index
    common = rets.index.intersection(spy_r.index).intersection(tlt_r.index).intersection(vix_r.index)
    if len(common) >= 63:
        sec_c = rets.reindex(common).iloc[-63:]
        spy_c = spy_r.reindex(common).iloc[-63:]
        tlt_c = tlt_r.reindex(common).iloc[-63:]
        vix_c = vix_r.reindex(common).iloc[-63:]
        f["spy_corr_63d"] = float(sec_c.corr(spy_c))
        f["tlt_corr_63d"] = float(sec_c.corr(tlt_c))
        # VIX beta = cov(sector, vix) / var(vix)
        cov_sv = float(sec_c.cov(vix_c))
        var_v = float(vix_c.var())
        f["vix_beta_63d"] = cov_sv / (var_v + 1e-10)
    else:
        f["spy_corr_63d"] = 0.0
        f["tlt_corr_63d"] = 0.0
        f["vix_beta_63d"] = 0.0

    return f


def compute_atr_series(high, low, close, period=14):
    """Compute ATR series for each sector."""
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
                atr_dict[tk] = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
    return atr_dict


# ==============================================================
# DATA DOWNLOAD
# ==============================================================

def download_data():
    """Download all tickers via yfinance."""
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    fprint("Downloading %d tickers..." % len(all_tickers))
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
    fprint("Data: %d days, %s to %s" % (len(close), close.index[0].date(), close.index[-1].date()))
    return close, high, low


# ==============================================================
# LGBM WALK-FORWARD RANKING (500d sliding)
# ==============================================================

def get_biweekly_rebalance_dates(close):
    """Every REBALANCE_DAYS trading days, starting from day 260."""
    dates = close.index
    rebal = []
    for i in range(260, len(dates), REBALANCE_DAYS):
        rebal.append(dates[i])
    return pd.DatetimeIndex(rebal)


def build_feature_records(close, high, low, rebal_dates):
    """Build feature records for all rebalance dates.
    Each record = (date, ticker, 21 features, forward return).
    """
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"] if "SPY" in close.columns else None
    tlt = close["TLT"] if "TLT" in close.columns else None
    vix = close["VIX"] if "VIX" in close.columns else None

    if spy is None or tlt is None or vix is None:
        fprint("WARNING: Missing SPY/TLT/VIX columns")
        return pd.DataFrame()

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_slice = spy.iloc[:idx + 1].dropna()
            tlt_slice = tlt.iloc[:idx + 1].dropna()
            vix_slice = vix.iloc[:idx + 1].dropna()

            feats = compute_features(px, spy_slice, tlt_slice, vix_slice)
            if not feats:
                continue

            # Forward return for label (DTE trading days ahead)
            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)
            rec = {**feats, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(0.0)
    fprint("  Feature records: %d rows, %d dates" % (len(df), len(df["date"].unique()) if len(df) > 0 else 0))
    return df


def walk_forward_lgbm_rank(df):
    """Walk-forward LGBM ranking with 500-day sliding window."""
    import lightgbm as lgb
    if len(df) < 100:
        return {}

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    # Convert 500 calendar days to approximate biweekly periods
    # 500d / 10 trading days ~ 50 periods minimum in training window
    wf_periods = max(30, WF_TRAIN_DAYS // REBALANCE_DAYS)

    rankings = {}
    for i in range(wf_periods, len(dates)):
        train_dates = dates[max(0, i - wf_periods):i]
        test_date = dates[i]

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()

        if len(test_df) < 3 or len(train_df) < 50:
            continue

        Xt = np.nan_to_num(train_df[ALL_FEATURES].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[ALL_FEATURES].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1,
            )
            m.fit(Xt, yt)
            preds = m.predict(Xe)
            test_df["score"] = preds
            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))
        except Exception:
            continue

    fprint("  LGBM ranking dates: %d" % len(rankings))
    return rankings


# ==============================================================
# STRIKES
# ==============================================================

def compute_strikes(S, direction):
    """2% OTM, max($3, K1*3%) adaptive width."""
    if direction == "bull":
        K1 = round(S * (1.0 + OTM_PCT), 2)
        w = max(WIDTH_FLOOR_DOLLARS, K1 * WIDTH_FLOOR_PCT)
        K2 = round(K1 + w, 2)
    else:
        K2 = round(S * (1.0 - OTM_PCT), 2)
        w = max(WIDTH_FLOOR_DOLLARS, K2 * WIDTH_FLOOR_PCT)
        K1 = round(K2 - w, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


# ==============================================================
# TRADE EXECUTION
# ==============================================================

def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity, cfg):
    """Execute a single spread trade with variant-specific pricing.
    cfg = variant config dict with mode, haircut/multiplier, profit_target.
    """
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    # ATR for IV estimation
    if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    K1, K2 = compute_strikes(S, direction)
    spread_width = abs(K2 - K1)

    # BS pricing
    bs_fair_ps, sigma = _price_spread_bs(S, K1, K2, DTE, av, vix_val, direction)

    # Apply variant-specific pricing
    entry_cost_ps = _apply_pricing(bs_fair_ps, cfg)

    if entry_cost_ps is None or entry_cost_ps <= 0:
        return None

    # Cost/width filter: reject if entry_cost / width > 50%
    cwr = entry_cost_ps / max(spread_width, 0.01)
    if cwr > COST_WIDTH_MAX:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT
    max_pos = min(MAX_POS_DOLLARS, equity * 0.40)
    if total_cost <= 0 or total_cost > max_pos:
        return None

    max_profit_ps = spread_width - entry_cost_ps
    if max_profit_ps <= 0:
        return None

    # ---- Profit target exit logic ----
    profit_target = cfg.get("profit_target")
    exited_early = False
    exit_day_idx = ei
    exit_reason = "expiry"
    hold_days = ei - di

    if profit_target is not None and profit_target > 0:
        for check_idx in range(di + 1, ei + 1):
            if check_idx >= len(close):
                break
            check_date = close.index[check_idx]
            dte_remaining = ei - check_idx

            S_now = float(close[tk].iloc[check_idx])
            if check_date in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[check_date]):
                av_now = float(atr_dict[tk].loc[check_date])
            else:
                av_now = S_now * 0.015

            current_value_ps = _revalue_spread_bs(
                S_now, K1, K2, dte_remaining, av_now, vix_val, direction
            )
            unrealized_gain_ps = current_value_ps - entry_cost_ps

            if unrealized_gain_ps >= profit_target * max_profit_ps:
                exited_early = True
                exit_day_idx = check_idx
                exit_reason = "profit_target_%dpct" % int(profit_target * 100)
                hold_days = check_idx - di
                break

    # ---- Compute final P&L ----
    Se = float(close[tk].iloc[exit_day_idx])

    if exited_early:
        dte_at_exit = ei - exit_day_idx
        exit_date = close.index[exit_day_idx]
        if exit_date in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[exit_date]):
            av_exit = float(atr_dict[tk].loc[exit_date])
        else:
            av_exit = Se * 0.015
        exit_value_ps = _revalue_spread_bs(Se, K1, K2, dte_at_exit, av_exit, vix_val, direction)
        pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT - EARLY_EXIT_COMMISSION
    else:
        # Hold to expiry: intrinsic value only
        if direction == "bull":
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
        else:
            intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT
        exit_value_ps = intrinsic

    return {
        "pnl": round(pnl, 2),
        "entry_cost_ps": round(entry_cost_ps, 4),
        "bs_fair_ps": round(bs_fair_ps, 4),
        "total_cost": round(total_cost, 2),
        "K1": K1, "K2": K2,
        "S_entry": round(S, 2), "S_exit": round(Se, 2),
        "exit_value_ps": round(exit_value_ps, 4),
        "spread_width": round(spread_width, 2),
        "cost_width_ratio": round(cwr, 3),
        "max_profit_ps": round(max_profit_ps, 4),
        "exited_early": exited_early,
        "exit_reason": exit_reason,
        "hold_days": hold_days,
        "sigma": round(sigma, 4),
    }


# ==============================================================
# SIMULATION ENGINE (with VIX-based mode switching)
# ==============================================================

def simulate_variant(rankings, close, atr_dict, rebal_dates, cfg, variant_name=""):
    """Run V10 backtest for a pricing variant with VIX-based position selection."""
    vix = close["VIX"] if "VIX" in close.columns else None
    spy = close["SPY"] if "SPY" in close.columns else None

    equity = CAP
    trades = []

    for dt in sorted(rebal_dates):
        if dt not in close.index:
            continue

        # Find most recent ranking
        ranking_date = None
        for rd in sorted(rankings.keys()):
            if rd <= dt:
                ranking_date = rd
        if ranking_date is None:
            continue

        scores = rankings[ranking_date]
        if not scores or len(scores) < 3:
            continue

        # Current VIX
        cv = 20.0
        if vix is not None and dt in vix.index:
            cv = float(vix.loc[dt])

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        # VIX-based mode switching
        if cv >= VIX_THRESHOLD:
            # High VIX: bull spreads on top-2 only (flight to quality)
            bull_picks = [t for t, _ in ranked_desc[:HIGH_VIX_BULL_K]]
            bear_picks = []
        else:
            # Low VIX: bull top-3 + bear bottom-3 (pair trades)
            bull_picks = [t for t, _ in ranked_desc[:LOW_VIX_BULL_K]]
            bear_picks = [t for t, _ in ranked_asc[:LOW_VIX_BEAR_K]]

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                result = execute_trade(
                    tk, dt, direction, close, atr_dict, cv, equity, cfg,
                )
                if result is not None:
                    equity += result["pnl"]
                    # Determine market regime from SPY
                    di = close.index.get_loc(dt)
                    ei = min(di + DTE, len(close) - 1)
                    if spy is not None:
                        sv = float(spy.loc[dt])
                        se = float(spy.iloc[ei]) if ei < len(spy) else sv
                        regime = "bull" if se >= sv else "bear"
                    else:
                        regime = "unknown"

                    exit_idx = di + result["hold_days"]
                    if exit_idx >= len(close):
                        exit_idx = len(close) - 1
                    exit_date_str = str(close.index[exit_idx].date())

                    trades.append({
                        **result,
                        "entry_date": str(dt.date()),
                        "exit_date": exit_date_str,
                        "ticker": tk,
                        "regime": regime,
                        "direction": direction,
                        "vix": round(cv, 1),
                        "win": result["pnl"] > 0,
                        "vix_mode": "high" if cv >= VIX_THRESHOLD else "low",
                    })

    return trades, equity


# ==============================================================
# METRICS
# ==============================================================

def compute_full_stats(trades, initial_capital=CAP):
    """Compute comprehensive risk-adjusted metrics using calendar month Sharpe."""
    if not trades:
        return None

    pnls = np.array([t["pnl"] for t in trades])
    equity = [initial_capital]
    for p in pnls:
        equity.append(equity[-1] + p)
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-10)

    wr = float(np.mean(pnls > 0))
    gross_win = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0
    gross_loss = float(abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-10
    pf = gross_win / max(gross_loss, 1e-10)

    # Calendar month Sharpe (primary)
    trade_df = pd.DataFrame(trades)
    trade_df["entry_month"] = pd.to_datetime(trade_df["entry_date"]).dt.to_period("M")
    monthly_pnl = trade_df.groupby("entry_month")["pnl"].sum()

    if len(monthly_pnl) > 2:
        sharpe = float(monthly_pnl.mean() / (monthly_pnl.std() + 1e-10) * sqrt(12))
    else:
        sharpe = 0.0

    # Sortino (monthly)
    monthly_ret = monthly_pnl / initial_capital
    neg_ret = monthly_ret[monthly_ret < 0]
    if len(neg_ret) > 1:
        sortino = float(monthly_ret.mean() / (neg_ret.std() + 1e-10) * sqrt(12))
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0.0

    # CAGR
    dates_str = [t["entry_date"] for t in trades]
    first_date = pd.Timestamp(min(dates_str))
    last_date = pd.Timestamp(max(dates_str))
    years = max((last_date - first_date).days / 365.25, 0.5)
    total_ret = equity[-1] / max(equity[0], 1e-10)
    if total_ret > 0:
        cagr = total_ret ** (1.0 / years) - 1.0
    else:
        cagr = -1.0

    # MaxDD
    max_dd_pct = float(dd.min()) * 100

    # Calmar
    calmar = cagr / (abs(dd.min()) + 1e-10) if abs(dd.min()) > 1e-10 else 0.0

    early_exits = sum(1 for t in trades if t.get("exited_early", False))
    hold_days_list = [t.get("hold_days", DTE) for t in trades]
    avg_entry_cost = float(np.mean([t.get("entry_cost_ps", 0) for t in trades]))
    avg_total_cost = float(np.mean([t.get("total_cost", 0) for t in trades]))

    return {
        "n_trades": len(trades),
        "total_pnl": round(float(pnls.sum()), 2),
        "avg_pnl": round(float(pnls.mean()), 2),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_dd_pct": round(max_dd_pct, 2),
        "final_equity": round(float(equity[-1]), 2),
        "total_return_pct": round((total_ret - 1) * 100, 1),
        "cagr_pct": round(cagr * 100, 1),
        "early_exit_rate": round(early_exits / max(len(trades), 1), 4),
        "avg_hold_days": round(float(np.mean(hold_days_list)), 1),
        "avg_entry_cost_ps": round(avg_entry_cost, 4),
        "avg_total_cost": round(avg_total_cost, 2),
        "years": round(years, 1),
        "n_months": len(monthly_pnl),
    }


# ==============================================================
# 4-GATE ADVERSARIAL VALIDATION
# ==============================================================

def gate_permutation(trades, n_perms=N_PERMUTATIONS):
    """Gate 1: Sign-flip permutation test (300 trials).
    Real monthly Sharpe must beat >95% of shuffled.
    """
    if len(trades) < 10:
        return False, 1.0, 0.0

    trade_df = pd.DataFrame(trades)
    trade_df["entry_month"] = pd.to_datetime(trade_df["entry_date"]).dt.to_period("M")
    monthly_pnl = trade_df.groupby("entry_month")["pnl"].sum().values

    if len(monthly_pnl) < 3:
        return False, 1.0, 0.0

    real_sharpe = float(monthly_pnl.mean() / (monthly_pnl.std() + 1e-10) * sqrt(12))
    rng = np.random.RandomState(42)
    count_beat = 0
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(monthly_pnl))
        shuffled = monthly_pnl * signs
        null_sharpe = shuffled.mean() / (shuffled.std() + 1e-10) * sqrt(12)
        if real_sharpe > null_sharpe:
            count_beat += 1
    p_value = 1.0 - count_beat / n_perms
    return p_value < 0.05, round(p_value, 4), round(real_sharpe, 3)


def gate_regime_balance(trades):
    """Gate 2: Regime balance. |Sharpe_bull - Sharpe_bear| / max(...) < 0.50."""
    bull_trades = [t for t in trades if t.get("regime") == "bull"]
    bear_trades = [t for t in trades if t.get("regime") == "bear"]

    if len(bull_trades) < 5 or len(bear_trades) < 5:
        return True, 0.0  # insufficient data, pass by default

    def _monthly_sharpe(tlist):
        df = pd.DataFrame(tlist)
        df["month"] = pd.to_datetime(df["entry_date"]).dt.to_period("M")
        mp = df.groupby("month")["pnl"].sum()
        if len(mp) < 2:
            return 0.0
        return float(mp.mean() / (mp.std() + 1e-10) * sqrt(12))

    sh_bull = _monthly_sharpe(bull_trades)
    sh_bear = _monthly_sharpe(bear_trades)
    denom = max(abs(sh_bull), abs(sh_bear), 0.01)
    r1 = abs(sh_bull - sh_bear) / denom
    return r1 < 0.50, round(r1, 3)


def gate_sub_period(trades):
    """Gate 3: Both halves must have Sharpe > 0.5."""
    if len(trades) < 20:
        return False, 0.0, 0.0

    mid = len(trades) // 2
    h1 = trades[:mid]
    h2 = trades[mid:]

    def _half_sharpe(tlist):
        df = pd.DataFrame(tlist)
        df["month"] = pd.to_datetime(df["entry_date"]).dt.to_period("M")
        mp = df.groupby("month")["pnl"].sum()
        if len(mp) < 2:
            return 0.0
        return float(mp.mean() / (mp.std() + 1e-10) * sqrt(12))

    sh1 = _half_sharpe(h1)
    sh2 = _half_sharpe(h2)
    return sh1 > SUB_PERIOD_SHARPE_MIN and sh2 > SUB_PERIOD_SHARPE_MIN, round(sh1, 3), round(sh2, 3)


def gate_outlier_removal(trades):
    """Gate 4: After removing top 5% months by PnL, still profitable."""
    if len(trades) < 20:
        return True, 0.0

    trade_df = pd.DataFrame(trades)
    trade_df["entry_month"] = pd.to_datetime(trade_df["entry_date"]).dt.to_period("M")
    monthly_pnl = trade_df.groupby("entry_month")["pnl"].sum().sort_values()

    # Remove top 5% months (best months)
    n_remove = max(1, int(len(monthly_pnl) * 0.05))
    trimmed = monthly_pnl.iloc[:-n_remove]
    trimmed_total = float(trimmed.sum())
    return trimmed_total > 0, round(trimmed_total, 2)


def run_4_gate_validation(trades, variant_name):
    """Run 4-gate adversarial validation. Returns (n_passed, 4, details)."""
    gates = []

    # Gate 1: Permutation
    passed, p_val, real_sh = gate_permutation(trades)
    gates.append({"name": "permutation_300", "passed": passed,
                  "p_value": p_val, "real_sharpe": real_sh})

    # Gate 2: Regime balance
    passed, r1 = gate_regime_balance(trades)
    gates.append({"name": "regime_balance", "passed": passed, "R1": r1})

    # Gate 3: Sub-period
    passed, sh1, sh2 = gate_sub_period(trades)
    gates.append({"name": "sub_period", "passed": passed,
                  "half1_sharpe": sh1, "half2_sharpe": sh2})

    # Gate 4: Outlier removal
    passed, trimmed = gate_outlier_removal(trades)
    gates.append({"name": "outlier_removal", "passed": passed,
                  "trimmed_pnl": trimmed})

    n_passed = sum(1 for g in gates if g["passed"])
    return n_passed, 4, gates


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = datetime.now()
    fprint("=" * 110)
    fprint("V10 REALISTIC PRICING V1 -- %s" % t0.strftime("%Y-%m-%d %H:%M:%S"))
    fprint("=" * 110)
    fprint("")
    fprint("CONTEXT: BS pricing underprices options by 61-73%% median (KB #282).")
    fprint("Calibration: market_mid = 1.10 * BS + $0.066/sh (R2=0.873)")
    fprint("This tells us the REAL Sharpe of our options strategy.")
    fprint("")
    fprint("V10 config: %d sectors, 21 LGBM features, DTE=%d, %.0f%% OTM" % (
        len(SECTORS), DTE, OTM_PCT * 100))
    fprint("Width: max($%.0f, %.0f%%), Biweekly rebalance (%dd)" % (
        WIDTH_FLOOR_DOLLARS, WIDTH_FLOOR_PCT * 100, REBALANCE_DAYS))
    fprint("VIX mode: >= %.0f -> bull top-%d only | < %.0f -> bull top-%d + bear bottom-%d" % (
        VIX_THRESHOLD, HIGH_VIX_BULL_K, VIX_THRESHOLD, LOW_VIX_BULL_K, LOW_VIX_BEAR_K))
    fprint("Capital: $%.0f | Max per trade: $%.0f | Commission: $%.2f RT" % (
        CAP, MAX_POS_DOLLARS, COMMISSION_RT))
    fprint("")
    fprint("PRICING VARIANTS (%d):" % len(VARIANTS))
    for key, label, _ in VARIANTS:
        fprint("  %s" % label)
    fprint("")

    # ---- Download data ----
    fprint("Downloading price data...")
    close, high, low = download_data()
    atr_dict = compute_atr_series(high, low, close)

    rebal_dates = get_biweekly_rebalance_dates(close)
    fprint("Biweekly rebalance dates: %d (from %s to %s)" % (
        len(rebal_dates), rebal_dates[0].date(), rebal_dates[-1].date()))

    # ---- Build LGBM rankings (shared across all variants) ----
    fprint("")
    fprint("=" * 80)
    fprint("BUILDING LGBM RANKINGS (500d sliding, 21 features, biweekly)")
    fprint("=" * 80)
    records = build_feature_records(close, high, low, rebal_dates)

    if len(records) < 100:
        fprint("ERROR: Only %d feature records -- insufficient data" % len(records))
        return

    rankings = walk_forward_lgbm_rank(records)
    if len(rankings) < 20:
        fprint("ERROR: Only %d ranking dates -- insufficient for backtest" % len(rankings))
        return

    # ---- Run each variant ----
    all_trades = {}
    all_stats = {}
    all_gates = {}

    for var_key, var_label, var_cfg in VARIANTS:
        fprint("")
        fprint("=" * 90)
        fprint(var_label)
        fprint("  config: %s" % json.dumps(var_cfg))
        fprint("=" * 90)
        t_start = time.time()

        trades, final_eq = simulate_variant(
            rankings, close, atr_dict, rebal_dates, var_cfg, var_key
        )
        elapsed = time.time() - t_start
        fprint("  Trades: %d | Final equity: $%s (%.1fs)" % (
            len(trades), "{:,.0f}".format(final_eq), elapsed))

        all_trades[var_key] = trades
        s = compute_full_stats(trades)
        all_stats[var_key] = s

        if s:
            fprint("  Sharpe: %.2f | Sortino: %.2f | Calmar: %.2f" % (
                s["sharpe"], s["sortino"], s["calmar"]))
            fprint("  WR: %.1f%% | PF: %.2f | MDD: %.1f%% | CAGR: %.1f%%" % (
                s["win_rate"] * 100, s["profit_factor"], s["max_dd_pct"], s["cagr_pct"]))
            fprint("  Total return: %.1f%% | Final: $%s | %.1f years" % (
                s["total_return_pct"], "{:,.2f}".format(s["final_equity"]), s["years"]))
            fprint("  Avg entry cost/sh: $%.4f | Avg total cost/trade: $%.2f" % (
                s["avg_entry_cost_ps"], s["avg_total_cost"]))

            # Side breakdown
            for side in ["bull", "bear"]:
                st = [t for t in trades if t["direction"] == side]
                if st:
                    pnls_side = [t["pnl"] for t in st]
                    wr_side = sum(1 for p in pnls_side if p > 0) / len(pnls_side)
                    fprint("    %s: %d trades, WR %.1f%%, PnL $%s" % (
                        side.upper(), len(st), wr_side * 100,
                        "{:,.0f}".format(sum(pnls_side))))

            # VIX mode breakdown
            for mode in ["high", "low"]:
                mt = [t for t in trades if t.get("vix_mode") == mode]
                if mt:
                    mp = [t["pnl"] for t in mt]
                    fprint("    VIX %s: %d trades, WR %.1f%%, PnL $%s" % (
                        mode.upper(), len(mt),
                        sum(1 for p in mp if p > 0) / len(mp) * 100,
                        "{:,.0f}".format(sum(mp))))

        # 4-gate validation
        if len(trades) >= 10:
            n_passed, n_total, gate_details = run_4_gate_validation(trades, var_key)
            all_gates[var_key] = {"passed": n_passed, "total": n_total, "gates": gate_details}
            status = "PASS" if n_passed == n_total else "PARTIAL" if n_passed >= 2 else "FAIL"
            fprint("  4-GATE VALIDATION: [%s] %d/%d" % (status, n_passed, n_total))
            for g in gate_details:
                gs = "PASS" if g["passed"] else "FAIL"
                detail_parts = ["%s=%s" % (k, v) for k, v in g.items()
                                if k not in ("name", "passed")]
                fprint("    [%s] %s: %s" % (gs, g["name"], ", ".join(detail_parts)))

    # ==============================================================
    # COMPARISON TABLE (sorted by Sharpe)
    # ==============================================================
    fprint("")
    fprint("=" * 150)
    fprint("COMPARISON -- ALL PRICING VARIANTS (sorted by Sharpe)")
    fprint("=" * 150)

    hdr = "  %-22s %5s %7s %7s %7s %6s %6s %7s %7s %9s %9s %5s" % (
        "Variant", "N", "Sharpe", "Sort", "Calmar", "WR", "PF",
        "MDD", "CAGR", "AvgCost", "Final$", "Gate")
    fprint(hdr)
    fprint("  " + "-" * 140)

    sorted_variants = sorted(
        [(vk, all_stats.get(vk)) for vk, _, _ in VARIANTS if all_stats.get(vk)],
        key=lambda x: x[1]["sharpe"], reverse=True
    )

    for var_key, s in sorted_variants:
        g = all_gates.get(var_key)
        gates_str = "%d/%d" % (g["passed"], g["total"]) if g else "--"
        fprint("  %-22s %5d %7.2f %7.2f %7.2f %5.1f%% %5.2f %6.1f%% %6.1f%% $%7.4f $%8s %5s" % (
            var_key, s["n_trades"], s["sharpe"], s["sortino"], s["calmar"],
            s["win_rate"] * 100, s["profit_factor"], s["max_dd_pct"],
            s["cagr_pct"], s["avg_entry_cost_ps"],
            "{:,.0f}".format(s["final_equity"]), gates_str))

    # ==============================================================
    # DELTA TABLE (vs baseline A)
    # ==============================================================
    fprint("")
    fprint("=" * 110)
    fprint("IMPACT OF PRICING CORRECTION (delta vs A_baseline)")
    fprint("=" * 110)

    base = all_stats.get("A_baseline")
    if base:
        fprint("")
        fprint("  %-22s %8s %8s %7s %7s %9s %9s" % (
            "Variant", "dSharpe", "dSort", "dWR", "dPF", "dPnL", "dCost/sh"))
        fprint("  " + "-" * 85)
        for var_key, _, _ in VARIANTS[1:]:
            s = all_stats.get(var_key)
            if s is None:
                continue
            fprint("  %-22s %+7.2f %+7.2f %+6.1f%% %+6.2f $%+8s $%+.4f" % (
                var_key,
                s["sharpe"] - base["sharpe"],
                s["sortino"] - base["sortino"],
                (s["win_rate"] - base["win_rate"]) * 100,
                s["profit_factor"] - base["profit_factor"],
                "{:,.0f}".format(s["total_pnl"] - base["total_pnl"]),
                s["avg_entry_cost_ps"] - base["avg_entry_cost_ps"]))

    # ==============================================================
    # HONEST VERDICT
    # ==============================================================
    fprint("")
    fprint("=" * 110)
    fprint("VERDICT: IS THE OPTIONS STRATEGY VIABLE WITH REALISTIC PRICING?")
    fprint("=" * 110)
    fprint("")

    if base:
        fprint("  BASELINE (A, BS + 15%% haircut): Sharpe %.2f, Sortino %.2f, PF %.2f, WR %.1f%%" % (
            base["sharpe"], base["sortino"], base["profit_factor"], base["win_rate"] * 100))
        fprint("")

        # Check each corrected variant
        for var_key, var_label, _ in VARIANTS[1:]:
            s = all_stats.get(var_key)
            if s is None:
                continue
            if base["sharpe"] > 0:
                pct_drop = (1 - s["sharpe"] / base["sharpe"]) * 100
            else:
                pct_drop = 0
            fprint("  %-22s Sharpe %.2f (%.0f%% drop), Sortino %.2f, PF %.2f, WR %.1f%%" % (
                var_key, s["sharpe"], pct_drop, s["sortino"], s["profit_factor"],
                s["win_rate"] * 100))

        fprint("")

        # Key decision metrics
        c_stats = all_stats.get("C_2x_cost")
        d_stats = all_stats.get("D_2p5x_cost")
        b_stats = all_stats.get("B_calibrated")

        fprint("  KEY QUESTION: Does 2x cost give Sharpe > 1.0?")
        if c_stats:
            if c_stats["sharpe"] > 1.0:
                fprint("    YES (Sharpe %.2f) -- options strategy IS VIABLE even at 2x cost." % c_stats["sharpe"])
            elif c_stats["sharpe"] > 0.5:
                fprint("    MARGINAL (Sharpe %.2f) -- barely viable, needs optimization." % c_stats["sharpe"])
            else:
                fprint("    NO (Sharpe %.2f) -- options strategy NOT viable at 2x cost." % c_stats["sharpe"])
                fprint("    RECOMMENDATION: Focus on equity strategies instead.")
        fprint("")

        fprint("  KEY QUESTION: Does calibrated pricing (B) survive?")
        if b_stats:
            if b_stats["sharpe"] > 1.0:
                fprint("    YES (Sharpe %.2f) -- edge is REAL but smaller than reported." % b_stats["sharpe"])
            elif b_stats["sharpe"] > 0.5:
                fprint("    MARGINAL (Sharpe %.2f) -- edge exists but may not justify costs." % b_stats["sharpe"])
            elif b_stats["sharpe"] > 0:
                fprint("    WEAK (Sharpe %.2f) -- significant inflation from BS underpricing." % b_stats["sharpe"])
            else:
                fprint("    DESTROYED (Sharpe %.2f) -- reported edge was BS pricing artifact." % b_stats["sharpe"])
        fprint("")

        # Average across corrected variants (B-E)
        corrected_keys = ["B_calibrated", "C_2x_cost", "D_2p5x_cost", "E_3x_cost"]
        corrected = [all_stats[k] for k in corrected_keys if k in all_stats and all_stats[k] is not None]
        if corrected and base["sharpe"] > 0:
            avg_sh = np.mean([s["sharpe"] for s in corrected])
            avg_drop = (1 - avg_sh / base["sharpe"]) * 100
            fprint("  AVERAGE corrected Sharpe (B-E): %.2f (%.0f%% lower than baseline)" % (
                avg_sh, avg_drop))

        # Cost multiple analysis
        fprint("")
        fprint("  ENTRY COST COMPARISON:")
        fprint("    %-22s  Avg entry/sh  Cost multiple vs A" % "Variant")
        fprint("    " + "-" * 55)
        for var_key, _, _ in VARIANTS:
            s = all_stats.get(var_key)
            if s and base["avg_entry_cost_ps"] > 0:
                mult = s["avg_entry_cost_ps"] / base["avg_entry_cost_ps"]
                fprint("    %-22s  $%.4f       %.2fx" % (
                    var_key, s["avg_entry_cost_ps"], mult))

    # ==============================================================
    # SAVE RESULTS
    # ==============================================================
    save_results = {
        "timestamp": t0.isoformat(),
        "context": "BS pricing underprices options by 61-73%% (KB #282). "
                   "Calibration: market_mid = 1.10*BS + $0.066/sh (R2=0.873). "
                   "6 variants test honest performance.",
        "config": {
            "capital": CAP, "dte": DTE, "otm_pct": OTM_PCT,
            "width_floor_dollars": WIDTH_FLOOR_DOLLARS,
            "width_floor_pct": WIDTH_FLOOR_PCT,
            "commission_rt": COMMISSION_RT,
            "vix_threshold": VIX_THRESHOLD,
            "high_vix_bull_k": HIGH_VIX_BULL_K,
            "low_vix_bull_k": LOW_VIX_BULL_K,
            "low_vix_bear_k": LOW_VIX_BEAR_K,
            "rebalance_days": REBALANCE_DAYS,
            "wf_train_days": WF_TRAIN_DAYS,
            "n_sectors": len(SECTORS),
            "n_features": len(ALL_FEATURES),
            "n_permutations": N_PERMUTATIONS,
            "calib_slope": CALIB_SLOPE,
            "calib_intercept_ps": CALIB_INTERCEPT_PS,
        },
        "variants": {},
    }

    for var_key, var_label, var_cfg in VARIANTS:
        entry = {"label": var_label, "config": var_cfg}
        s = all_stats.get(var_key)
        if s:
            entry["stats"] = s
        g = all_gates.get(var_key)
        if g:
            entry["validation"] = g
        save_results["variants"][var_key] = entry

    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint("")
    fprint("Results saved to %s" % results_file)

    # Save trade details per variant
    for var_key, _, _ in VARIANTS:
        trades = all_trades.get(var_key, [])
        if trades:
            tf = OUTPUT_DIR / ("trades_%s.json" % var_key)
            with open(tf, "w") as f:
                json.dump(trades, f, indent=1, default=str)

    # ---- MLflow logging ----
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="realistic_pricing_%s" % t0.strftime("%Y%m%d_%H%M")):
                mlflow.log_param("capital", CAP)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("otm_pct", OTM_PCT)
                mlflow.log_param("vix_threshold", VIX_THRESHOLD)
                mlflow.log_param("rebalance_days", REBALANCE_DAYS)
                mlflow.log_param("wf_train_days", WF_TRAIN_DAYS)
                mlflow.log_param("n_features", len(ALL_FEATURES))
                mlflow.log_param("n_variants", len(VARIANTS))
                mlflow.log_param("commission_rt", COMMISSION_RT)
                mlflow.log_param("calib_slope", CALIB_SLOPE)
                mlflow.log_param("calib_intercept_ps", CALIB_INTERCEPT_PS)

                for var_key, _, _ in VARIANTS:
                    s = all_stats.get(var_key)
                    if s is None:
                        continue
                    prefix = var_key.split("_")[0]
                    for metric in ["sharpe", "sortino", "calmar", "win_rate",
                                   "profit_factor", "max_dd_pct", "cagr_pct",
                                   "n_trades", "total_pnl", "avg_entry_cost_ps"]:
                        mlflow.log_metric("%s_%s" % (prefix, metric), s[metric])

                    g = all_gates.get(var_key)
                    if g:
                        mlflow.log_metric("%s_gates_passed" % prefix, g["passed"])

                mlflow.log_artifact(str(results_file))
            fprint("MLflow logged to experiment '%s'" % EXPERIMENT_NAME)
        except Exception as e:
            fprint("MLflow error: %s" % e)

    elapsed = (datetime.now() - t0).total_seconds()
    fprint("")
    fprint("Completed in %.1f minutes" % (elapsed / 60))


if __name__ == "__main__":
    main()
