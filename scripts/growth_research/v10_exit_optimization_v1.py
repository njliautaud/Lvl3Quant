#!/usr/bin/env python3
"""
V10 Exit Optimization V1 -- Exit Strategy Variants for Sector Spread Config
============================================================================

Tests 12 exit strategy variations on the V10 sector spread configuration to
find the optimal exit mechanics.

V10 Base Config (KB #267):
  - 11 sector ETFs: XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC
  - LGBM ranker with 21 features (18 legacy quality-momentum + 3 cross-asset)
  - DTE=28, 4% OTM moneyness, spread width = max($3, 3% of strike)
  - 8 positions: top 4 long (bull call spreads), bottom 4 short (bear put spreads)
  - Monthly rebalance (22 trading-day interval)
  - VIX > 20 regime filter
  - $645 starting capital, $200 max per trade, $2.60 commission RT, 15% entry haircut
  - Sliding walk-forward (500d train window), hold-to-expiry with intrinsic value

12 Variants:
  Profit Target Level (no stop-loss):
    A: 20% PT (aggressive)
    B: 30% PT (current V10)
    C: 40% PT
    D: 50% PT (V9.3 setting)
    E: 60% PT
    F: No PT (hold-to-expiry baseline)

  Stop-Loss (with 30% PT):
    G: 30% PT + 50% stop-loss
    H: 30% PT + 75% stop-loss
    I: 30% PT + 100% stop-loss

  Dynamic Exits:
    J: VIX-adaptive PT: 20% VIX>30, 30% 25<VIX<30, 50% 20<VIX<25
    K: Time-decay PT: 30% first 14d, then 20% after day 14
    L: Trailing PT: once gain >= 20% of max profit, trail at 50% of peak gain

5-gate adversarial validation on every variant.

Output: output/growth_research/v10_exit_optimization_v1/
MLflow experiment: v10_exit_optimization_v1
"""

import json
import math
import sys
import time
import warnings
from datetime import datetime
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


# ==============================================================
# ENVIRONMENT DETECTION
# ==============================================================

_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")

if _NEPTUNE_BASE.exists():
    BASE = _NEPTUNE_BASE
    fprint("Running on Neptune: %s" % BASE)
else:
    BASE = _JUPITER_BASE
    fprint("Running on Jupiter: %s" % BASE)

OUTPUT_DIR = BASE / "output" / "growth_research" / "v10_exit_optimization_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================
# CONSTANTS
# ==============================================================

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]

CAP = 645.0
TOP_K = 4
OTM_PCT = 0.04
DTE = 28
HAIRCUT = 0.15
COMMISSION = 2.60
EARLY_EXIT_COMMISSION = 2.60
MAX_PER_TRADE = 200.0

WF_TRAIN_DAYS = 500  # sliding window in trading days
REBAL_INTERVAL = 22  # trading days between rebalances
VIX_REGIME_THRESHOLD = 20.0

COST_WIDTH_MAX = 0.50
N_PERMUTATIONS = 300

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v10_exit_optimization_v1"

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

# 21 features: 18 legacy + 3 cross-asset
FEATURE_NAMES = [
    # 18 legacy quality-momentum
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d",
    "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    # 3 cross-asset
    "sector_spy_beta_63d", "sector_relative_vol_21d", "cross_sector_dispersion",
]
assert len(FEATURE_NAMES) == 21


# ==============================================================
# SELF-CONTAINED BLACK-SCHOLES PRICING
# ==============================================================

def _bs_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def _bs_put(S, K, T, r, sigma):
    """Black-Scholes put price via put-call parity."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _estimate_iv(atr, S, vix_val):
    """Estimate implied volatility from ATR and VIX."""
    atr_iv = (atr / S) * math.sqrt(252) if S > 0 else 0.20
    vix_iv = vix_val / 100.0 if vix_val is not None and vix_val > 0 else 0.20
    # Blend: 60% ATR-derived, 40% VIX-derived
    iv = 0.6 * atr_iv + 0.4 * vix_iv
    return max(0.05, min(iv, 2.0))


def _price_bull_call_spread(S, K1, K2, dte, atr, vix_val, r=0.045):
    """Price a bull call spread (buy K1 call, sell K2 call, K1 < K2)."""
    T = max(dte / 365.0, 1e-6)
    sigma = _estimate_iv(atr, S, vix_val)
    c1 = _bs_call(S, K1, T, r, sigma)
    c2 = _bs_call(S, K2, T, r, sigma)
    cost = c1 - c2
    return max(cost, 0.001), sigma


def _price_bear_put_spread(S, K1, K2, dte, atr, vix_val, r=0.045):
    """Price a bear put spread (buy K2 put, sell K1 put, K1 < K2)."""
    T = max(dte / 365.0, 1e-6)
    sigma = _estimate_iv(atr, S, vix_val)
    p2 = _bs_put(S, K2, T, r, sigma)
    p1 = _bs_put(S, K1, T, r, sigma)
    cost = p2 - p1
    return max(cost, 0.001), sigma


def _revalue_spread_bs(S, K1, K2, dte_remaining, atr, vix_val, direction, r=0.045):
    """Revalue a spread using BS at a given point in time."""
    if dte_remaining <= 0:
        if direction == "bull":
            return max(S - K1, 0.0) - max(S - K2, 0.0)
        else:
            return max(K2 - S, 0.0) - max(K1 - S, 0.0)
    T = max(dte_remaining / 365.0, 1e-6)
    sigma = _estimate_iv(atr, S, vix_val)
    if direction == "bull":
        return _bs_call(S, K1, T, r, sigma) - _bs_call(S, K2, T, r, sigma)
    else:
        return _bs_put(S, K2, T, r, sigma) - _bs_put(S, K1, T, r, sigma)


# ==============================================================
# DATA DOWNLOAD
# ==============================================================

def download_data():
    """Download price data via yfinance."""
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    fprint("Downloading %d tickers..." % len(all_tickers))
    raw = yf.download(all_tickers, start="2007-01-01", progress=False, auto_adjust=True)
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


def compute_atr_series(high, low, close, period=14):
    """Compute ATR for each sector ETF."""
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


# ==============================================================
# FEATURES (21 total: 18 legacy + 3 cross-asset)
# ==============================================================

def compute_legacy_features(px, spy_slice):
    """Compute 18 legacy quality-momentum features for a single sector at a point in time."""
    if len(px) < 260:
        return None
    f = {}
    # Return features
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()

    # Volatility features
    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.20
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.20

    # Sharpe
    f["sharpe_63d"] = float(rets.iloc[-63:].mean() / (rets.iloc[-63:].std() + 1e-10) * np.sqrt(252)) if len(rets) > 63 else 0.0

    # Max drawdown 63d
    if len(px) >= 63:
        pk63 = px.iloc[-63:].cummax()
        f["maxdd_63d"] = float(((px.iloc[-63:] / pk63) - 1).min())
    else:
        f["maxdd_63d"] = 0.0

    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5

    dr = rets.iloc[-63:][rets.iloc[-63:] < 0]
    f["sortino_63d"] = float(rets.iloc[-63:].mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0

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
    """Compute 3 cross-asset features."""
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {"sector_spy_beta_63d": 1.0, "sector_relative_vol_21d": 1.0, "cross_sector_dispersion": 0.01}

    spy_ret = spy.pct_change().dropna()

    # Beta
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

    # Relative vol
    if sector_px is not None and len(sector_px) > 21:
        sec_ret = sector_px.pct_change().dropna()
        sec_vol = float(sec_ret.iloc[-21:].std()) if len(sec_ret) >= 21 else 0.01
        spy_vol = float(spy_ret.iloc[-21:].std()) if len(spy_ret) >= 21 else 0.01
        f["sector_relative_vol_21d"] = sec_vol / (spy_vol + 1e-10)
    else:
        f["sector_relative_vol_21d"] = 1.0

    # Cross-sector dispersion
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


# ==============================================================
# REBALANCE DATES + FEATURE RECORDS
# ==============================================================

def get_rebalance_dates(close, interval=REBAL_INTERVAL):
    """Generate rebalance dates every `interval` trading days."""
    dates = close.index
    rebal = []
    i = 0
    while i < len(dates):
        rebal.append(dates[i])
        i += interval
    return pd.DatetimeIndex(rebal)


def build_feature_records(close, high, low, rebal_dates):
    """Build feature records for all rebalance dates with VIX > 20 filter."""
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    for dt in rebal_dates:
        if dt not in close.index:
            continue
        idx = close.index.get_loc(dt)
        if idx < 500:  # need enough history for WF_TRAIN_DAYS
            continue

        # VIX regime filter
        if vix is not None and dt in vix.index:
            vix_val = float(vix.loc[dt])
            if vix_val <= VIX_REGIME_THRESHOLD:
                continue
        else:
            continue  # skip if no VIX data

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            legacy = compute_legacy_features(px, spy.iloc[:idx + 1])
            if not legacy:
                continue
            cross_asset = compute_cross_asset_features(tk, idx, close)

            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)
            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        fprint("  WARNING: No feature records generated!")
        return df
    for c in FEATURE_NAMES:
        if c not in df.columns:
            df[c] = 0.0
    df[FEATURE_NAMES] = df[FEATURE_NAMES].fillna(0.0)
    fprint("    %d records, %d dates" % (len(df), len(df["date"].unique())))
    return df


# ==============================================================
# LGBM WALK-FORWARD RANKING (500d sliding)
# ==============================================================

def walk_forward_lgbm_rank(df):
    """Walk-forward LGBM ranking with 500-day sliding window."""
    import lightgbm as lgb
    if len(df) < 100:
        return {}
    df = df.copy()
    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    # Convert 500 trading days to approximate rebalance periods
    # With 22-day rebalance interval, 500 trading days ~ 23 periods
    train_periods = max(10, WF_TRAIN_DAYS // REBAL_INTERVAL)

    rankings = {}
    for i in range(train_periods, len(dates)):
        train_dates = dates[max(0, i - train_periods):i]
        test_date = dates[i]
        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()
        if len(test_df) < 3 or len(train_df) < 50:
            continue
        Xt = np.nan_to_num(train_df[FEATURE_NAMES].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[FEATURE_NAMES].values.astype(np.float32))
        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
            )
            m.fit(Xt, yt)
            preds = m.predict(Xe)
            test_df["score"] = preds
            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))
        except Exception:
            continue
    fprint("    %d ranking dates" % len(rankings))
    return rankings


# ==============================================================
# STRIKES
# ==============================================================

def compute_strikes(S, direction, otm_pct=OTM_PCT):
    """Compute strike prices: otm_pct OTM, adaptive max($3, 3%) width."""
    if direction == "bull":
        K1 = round(S * (1.0 + otm_pct), 2)
        pct_w = K1 * 0.03
        w = max(3.0, pct_w)
        K2 = round(K1 + w, 2)
    else:
        K2 = round(S * (1.0 - otm_pct), 2)
        pct_w = K2 * 0.03
        w = max(3.0, pct_w)
        K1 = round(K2 - w, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


# ==============================================================
# VARIANT DEFINITIONS
# ==============================================================

# Each variant is: (name, description, config_dict)
# config_dict keys: profit_target, stop_loss, dynamic_type

VARIANTS = [
    ("A_pt20", "A: 20% Profit Target", {"profit_target": 0.20, "stop_loss": None, "dynamic_type": None}),
    ("B_pt30", "B: 30% Profit Target (V10)", {"profit_target": 0.30, "stop_loss": None, "dynamic_type": None}),
    ("C_pt40", "C: 40% Profit Target", {"profit_target": 0.40, "stop_loss": None, "dynamic_type": None}),
    ("D_pt50", "D: 50% Profit Target (V9.3)", {"profit_target": 0.50, "stop_loss": None, "dynamic_type": None}),
    ("E_pt60", "E: 60% Profit Target", {"profit_target": 0.60, "stop_loss": None, "dynamic_type": None}),
    ("F_noExit", "F: No PT (hold-to-expiry)", {"profit_target": None, "stop_loss": None, "dynamic_type": None}),
    ("G_pt30_sl50", "G: 30% PT + 50% Stop-Loss", {"profit_target": 0.30, "stop_loss": 0.50, "dynamic_type": None}),
    ("H_pt30_sl75", "H: 30% PT + 75% Stop-Loss", {"profit_target": 0.30, "stop_loss": 0.75, "dynamic_type": None}),
    ("I_pt30_sl100", "I: 30% PT + 100% Stop-Loss", {"profit_target": 0.30, "stop_loss": 1.00, "dynamic_type": None}),
    ("J_vixAdaptive", "J: VIX-Adaptive PT", {"profit_target": None, "stop_loss": None, "dynamic_type": "vix_adaptive"}),
    ("K_timeDecay", "K: Time-Decay PT", {"profit_target": None, "stop_loss": None, "dynamic_type": "time_decay"}),
    ("L_trailing", "L: Trailing PT", {"profit_target": None, "stop_loss": None, "dynamic_type": "trailing"}),
]


# ==============================================================
# TRADE EXECUTION WITH FLEXIBLE EXIT
# ==============================================================

def execute_trade(tk, dt, direction, close, atr_dict, vix_series, equity, config):
    """Execute a spread trade with configurable exit strategy.

    config keys: profit_target, stop_loss, dynamic_type
    """
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    av = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015
    vix_val = float(vix_series.loc[dt]) if dt in vix_series.index else 20.0
    K1, K2 = compute_strikes(S, direction)

    # Price the spread
    try:
        if direction == "bull":
            entry_cost_ps, _ = _price_bull_call_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix_val=vix_val)
        else:
            entry_cost_ps, _ = _price_bear_put_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix_val=vix_val)
    except Exception:
        return None

    if entry_cost_ps is None or entry_cost_ps <= 0:
        return None

    # Apply haircut
    entry_cost_ps = entry_cost_ps * (1.0 + HAIRCUT)

    spread_width = abs(K2 - K1)
    cwr = entry_cost_ps / max(spread_width, 0.01)
    if cwr > COST_WIDTH_MAX:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION
    if total_cost <= 0 or total_cost > MAX_PER_TRADE or total_cost > equity * 0.40:
        return None

    max_profit_ps = spread_width - entry_cost_ps

    # Unpack config
    pt = config.get("profit_target")
    sl = config.get("stop_loss")
    dynamic_type = config.get("dynamic_type")

    # -----------------------------------------------------------------
    # EXIT LOGIC: walk through each day from entry+1 to expiry
    # -----------------------------------------------------------------
    exited_early = False
    exit_day_idx = ei
    exit_reason = "expiry"
    hold_days = DTE
    peak_gain_ps = 0.0  # for trailing stop

    for check_idx in range(di + 1, ei + 1):
        if check_idx >= len(close):
            break
        check_date = close.index[check_idx]
        dte_remaining = ei - check_idx
        days_held = check_idx - di

        S_now = float(close[tk].iloc[check_idx])
        av_now = float(atr_dict[tk].loc[check_date]) if check_date in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[check_date]) else S_now * 0.015
        vix_now = float(vix_series.loc[check_date]) if check_date in vix_series.index else vix_val

        current_value_ps = _revalue_spread_bs(S_now, K1, K2, dte_remaining, av_now, vix_now, direction)
        unrealized_gain_ps = current_value_ps - entry_cost_ps
        unrealized_loss_pct = -unrealized_gain_ps / (entry_cost_ps + 1e-10) if unrealized_gain_ps < 0 else 0.0

        # Track peak gain for trailing stop
        if unrealized_gain_ps > peak_gain_ps:
            peak_gain_ps = unrealized_gain_ps

        # Determine effective PT for this day
        effective_pt = None
        if dynamic_type == "vix_adaptive":
            if vix_now > 30:
                effective_pt = 0.20
            elif vix_now > 25:
                effective_pt = 0.30
            else:
                effective_pt = 0.50
        elif dynamic_type == "time_decay":
            if days_held <= 14:
                effective_pt = 0.30
            else:
                effective_pt = 0.20
        elif dynamic_type == "trailing":
            # Trailing: once gain reaches 20% of max_profit, set trailing stop
            if max_profit_ps > 0 and peak_gain_ps >= 0.20 * max_profit_ps:
                # Trailing stop: exit if gain drops to 50% of peak gain
                trailing_floor = 0.50 * peak_gain_ps
                if unrealized_gain_ps <= trailing_floor and peak_gain_ps > 0:
                    exited_early = True
                    exit_day_idx = check_idx
                    exit_reason = "trailing_stop"
                    hold_days = days_held
                    break
            # No fixed PT for trailing variant, only the trailing stop logic above
            effective_pt = None
        elif pt is not None:
            effective_pt = pt

        # Check profit target
        if effective_pt is not None and max_profit_ps > 0:
            if unrealized_gain_ps >= effective_pt * max_profit_ps:
                exited_early = True
                exit_day_idx = check_idx
                exit_reason = "profit_target"
                hold_days = days_held
                break

        # Check stop-loss
        if sl is not None:
            if unrealized_loss_pct >= sl:
                exited_early = True
                exit_day_idx = check_idx
                exit_reason = "stop_loss"
                hold_days = days_held
                break

    # Compute final P&L
    Se = float(close[tk].iloc[exit_day_idx])

    if exited_early:
        dte_at_exit = ei - exit_day_idx
        av_exit = float(atr_dict[tk].loc[close.index[exit_day_idx]]) if close.index[exit_day_idx] in atr_dict[tk].index else Se * 0.015
        vix_exit = float(vix_series.loc[close.index[exit_day_idx]]) if close.index[exit_day_idx] in vix_series.index else vix_val
        exit_value_ps = _revalue_spread_bs(Se, K1, K2, dte_at_exit, av_exit, vix_exit, direction)
        pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION - EARLY_EXIT_COMMISSION
    else:
        # At expiry: intrinsic value
        if direction == "bull":
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
        else:
            intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION
        exit_value_ps = intrinsic

    return {
        "pnl": round(pnl, 2),
        "entry_cost_ps": round(entry_cost_ps, 4),
        "total_cost": round(total_cost, 2),
        "K1": K1, "K2": K2,
        "S_entry": round(S, 2), "S_exit": round(Se, 2),
        "exit_value_ps": round(exit_value_ps, 4) if exit_value_ps is not None else 0.0,
        "spread_width": round(spread_width, 2),
        "cost_width_ratio": round(cwr, 3),
        "max_profit_ps": round(max_profit_ps, 4),
        "exited_early": exited_early,
        "exit_reason": exit_reason,
        "hold_days": hold_days,
        "vix_at_entry": round(vix_val, 1),
    }


# ==============================================================
# SIMULATION ENGINE
# ==============================================================

def simulate_variant(rankings, close, atr_dict, rebal_dates, variant_name, config):
    """Run backtest simulation for a variant."""
    vix = close["VIX"] if "VIX" in close.columns else pd.Series(dtype=float)
    spy = close["SPY"]

    equity = CAP
    trades = []
    early_exits = 0

    for dt in sorted(rebal_dates):
        if dt not in close.index:
            continue

        # VIX > 20 filter at trade entry
        if dt in vix.index:
            vix_val = float(vix.loc[dt])
            if vix_val <= VIX_REGIME_THRESHOLD:
                continue
        else:
            continue

        # Find closest ranking date
        ranking_date = None
        for rd in sorted(rankings.keys()):
            if rd <= dt:
                ranking_date = rd
        if ranking_date is None:
            continue

        scores = rankings[ranking_date]
        if not scores or len(scores) < 6:
            continue

        k = TOP_K
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:k]]
        bear_picks = [t for t, _ in ranked_asc[:k]]

        n_positions = len(bull_picks) + len(bear_picks)

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                result = execute_trade(
                    tk, dt, direction, close, atr_dict, vix, equity, config,
                )
                if result is not None:
                    equity += result["pnl"]
                    if result["exited_early"]:
                        early_exits += 1
                    di = close.index.get_loc(dt)
                    ei = min(di + DTE, len(close) - 1)
                    sv = float(spy.loc[dt])
                    se = float(spy.iloc[ei]) if ei < len(spy) else sv
                    exit_idx = min(di + result["hold_days"], len(close) - 1)
                    trades.append({
                        **result,
                        "entry_date": str(dt.date()),
                        "exit_date": str(close.index[exit_idx].date()),
                        "ticker": tk,
                        "regime": "bull" if se >= sv else "bear",
                        "direction": direction,
                        "win": result["pnl"] > 0,
                        "n_positions": n_positions,
                    })

    return trades, equity, early_exits


# ==============================================================
# STATISTICS
# ==============================================================

def compute_full_stats(trades, initial_capital=CAP):
    """Compute comprehensive stats for a variant."""
    if not trades:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    equity = [initial_capital]
    for p in pnls:
        equity.append(equity[-1] + p)
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak

    wr = float(np.mean(pnls > 0))
    gross_win = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0
    gross_loss = float(abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-10
    pf = gross_win / max(gross_loss, 1e-10)

    # Monthly Sharpe
    trade_df = pd.DataFrame(trades)
    trade_df["entry_month"] = pd.to_datetime(trade_df["entry_date"]).dt.to_period("M")
    monthly_pnl = trade_df.groupby("entry_month")["pnl"].sum()
    if len(monthly_pnl) > 1:
        sharpe = float(monthly_pnl.mean() / (monthly_pnl.std() + 1e-10) * np.sqrt(12))
    else:
        sharpe = 0.0

    # Sortino
    monthly_ret = monthly_pnl / initial_capital
    neg_ret = monthly_ret[monthly_ret < 0]
    downside_std = float(neg_ret.std()) if len(neg_ret) > 1 else 1e-10
    sortino = float(monthly_ret.mean() / (downside_std + 1e-10) * np.sqrt(12))

    max_dd_pct = float(dd.min()) * 100
    total_return = (equity[-1] / equity[0] - 1) * 100

    # Early exit stats
    early_ex = sum(1 for t in trades if t.get("exited_early", False))
    early_exit_rate = early_ex / len(trades) if trades else 0.0
    hold_days_list = [t.get("hold_days", DTE) for t in trades]
    avg_hold = float(np.mean(hold_days_list))

    return {
        "n_trades": len(trades),
        "total_pnl": round(float(pnls.sum()), 2),
        "total_return_pct": round(total_return, 2),
        "avg_pnl": round(float(pnls.mean()), 2),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd_pct, 2),
        "final_equity": round(equity[-1], 2),
        "early_exit_rate": round(early_exit_rate, 4),
        "avg_hold_days": round(avg_hold, 1),
        "early_exits": early_ex,
        "n_months": len(monthly_pnl),
    }


# ==============================================================
# 5-GATE ADVERSARIAL VALIDATION
# ==============================================================

def _permutation_test(trades, n_trials=N_PERMUTATIONS, seed=42):
    """Gate 1: Permutation test -- p < 0.05 required."""
    if len(trades) < 10:
        return False, 1.0
    pnls = np.array([t["pnl"] for t in trades])
    actual_mean = pnls.mean()
    rng = np.random.RandomState(seed)
    count_gte = 0
    for _ in range(n_trials):
        shuffled = pnls.copy()
        rng.shuffle(shuffled)
        # Randomly flip signs
        signs = rng.choice([-1, 1], size=len(shuffled))
        if (shuffled * signs).mean() >= actual_mean:
            count_gte += 1
    p_val = (count_gte + 1) / (n_trials + 1)
    return p_val < 0.05, round(p_val, 4)


def _regime_stability(trades):
    """Gate 2: Regime stability -- |Sharpe_bull - Sharpe_bear| / max < 0.50."""
    if len(trades) < 10:
        return False, {}
    bull_pnls = [t["pnl"] for t in trades if t.get("regime") == "bull"]
    bear_pnls = [t["pnl"] for t in trades if t.get("regime") == "bear"]

    def _sharpe_from_pnls(pnl_list):
        if len(pnl_list) < 3:
            return 0.0
        arr = np.array(pnl_list)
        return float(arr.mean() / (arr.std() + 1e-10) * np.sqrt(12))

    s_bull = _sharpe_from_pnls(bull_pnls)
    s_bear = _sharpe_from_pnls(bear_pnls)
    max_abs = max(abs(s_bull), abs(s_bear), 1e-10)
    ratio = abs(s_bull - s_bear) / max_abs
    passed = ratio < 0.50
    return passed, {"sharpe_bull": round(s_bull, 3), "sharpe_bear": round(s_bear, 3), "ratio": round(ratio, 3)}


def _sub_period(trades):
    """Gate 3: Sub-period -- both halves Sharpe > 0.5."""
    if len(trades) < 20:
        return False, {}
    mid = len(trades) // 2
    s1 = compute_full_stats(trades[:mid])
    s2 = compute_full_stats(trades[mid:])
    if s1 is None or s2 is None:
        return False, {}
    passed = s1["sharpe"] > 0.5 and s2["sharpe"] > 0.5
    return passed, {"sharpe_h1": s1["sharpe"], "sharpe_h2": s2["sharpe"]}


def _outlier_removal(trades):
    """Gate 4: Outlier removal -- trim 5% extreme trades, Sharpe > 0.5."""
    if len(trades) < 20:
        return False, {}
    pnls = sorted([t["pnl"] for t in trades])
    n_trim = max(1, int(len(pnls) * 0.05))
    trimmed_pnls = pnls[n_trim:-n_trim] if n_trim > 0 else pnls
    # Reconstruct fake trades for stats
    trimmed_trades = []
    pnl_set = list(trimmed_pnls)
    for t in trades:
        if t["pnl"] in pnl_set:
            trimmed_trades.append(t)
            pnl_set.remove(t["pnl"])
        if len(trimmed_trades) >= len(trimmed_pnls):
            break
    s = compute_full_stats(trimmed_trades)
    if s is None:
        return False, {}
    passed = s["sharpe"] > 0.5
    return passed, {"trimmed_sharpe": s["sharpe"], "n_trimmed": len(trimmed_trades)}


def _yearly_consistency(trades):
    """Gate 5: Yearly consistency -- >= 60% years profitable."""
    if len(trades) < 10:
        return False, {}
    trade_df = pd.DataFrame(trades)
    trade_df["year"] = pd.to_datetime(trade_df["entry_date"]).dt.year
    yearly_pnl = trade_df.groupby("year")["pnl"].sum()
    if len(yearly_pnl) < 2:
        return False, {}
    pct_profitable = float((yearly_pnl > 0).mean())
    passed = pct_profitable >= 0.60
    return passed, {"pct_years_profitable": round(pct_profitable, 3),
                    "n_years": len(yearly_pnl),
                    "yearly_pnl": {str(k): round(v, 2) for k, v in yearly_pnl.items()}}


def run_adversarial_validation(trades, variant_name):
    """Run all 5 adversarial gates."""
    fprint("    Running 5-gate adversarial validation...")
    results = {}

    g1_pass, g1_pval = _permutation_test(trades)
    results["gate1_permutation"] = {"passed": g1_pass, "p_value": g1_pval}
    fprint("      Gate 1 (Permutation p<0.05): %s (p=%.4f)" % ("PASS" if g1_pass else "FAIL", g1_pval))

    g2_pass, g2_detail = _regime_stability(trades)
    results["gate2_regime"] = {"passed": g2_pass, **g2_detail}
    fprint("      Gate 2 (Regime stability):   %s %s" % ("PASS" if g2_pass else "FAIL", g2_detail))

    g3_pass, g3_detail = _sub_period(trades)
    results["gate3_subperiod"] = {"passed": g3_pass, **g3_detail}
    fprint("      Gate 3 (Sub-period halves):  %s %s" % ("PASS" if g3_pass else "FAIL", g3_detail))

    g4_pass, g4_detail = _outlier_removal(trades)
    results["gate4_outlier"] = {"passed": g4_pass, **g4_detail}
    fprint("      Gate 4 (Outlier removal):    %s %s" % ("PASS" if g4_pass else "FAIL", g4_detail))

    g5_pass, g5_detail = _yearly_consistency(trades)
    results["gate5_yearly"] = {"passed": g5_pass, **g5_detail}
    fprint("      Gate 5 (Yearly consistency): %s %s" % ("PASS" if g5_pass else "FAIL", g5_detail))

    gates_passed = sum(1 for g in [g1_pass, g2_pass, g3_pass, g4_pass, g5_pass] if g)
    results["gates_passed"] = gates_passed
    results["all_passed"] = gates_passed == 5
    fprint("      TOTAL: %d/5 gates passed %s" % (gates_passed, "*** ALL PASSED ***" if gates_passed == 5 else ""))

    return results


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint("V10 EXIT OPTIMIZATION V1 -- %s" % t0.strftime("%Y-%m-%d %H:%M:%S"))
    fprint("=" * 100)
    fprint("V10 Base Config:")
    fprint("  Sectors: %s" % ", ".join(SECTORS))
    fprint("  Positions: %d bull + %d bear = %d total" % (TOP_K, TOP_K, TOP_K * 2))
    fprint("  OTM: %.0f%% | DTE: %d | Width: max($3, 3%%)" % (OTM_PCT * 100, DTE))
    fprint("  Capital: $%.0f | Commission: $%.2f | Haircut: %.0f%% | Max/trade: $%.0f" % (CAP, COMMISSION, HAIRCUT * 100, MAX_PER_TRADE))
    fprint("  VIX regime filter: > %.0f" % VIX_REGIME_THRESHOLD)
    fprint("  LGBM 21 features | 500d sliding walk-forward | %d-day rebalance" % REBAL_INTERVAL)
    fprint("  Adversarial: 5 gates, %d permutation trials" % N_PERMUTATIONS)
    fprint()
    fprint("12 EXIT VARIANTS:")
    for name, desc, _ in VARIANTS:
        fprint("  %s" % desc)
    fprint()

    # Download data
    fprint("Downloading price data...")
    close, high, low = download_data()
    atr_dict = compute_atr_series(high, low, close)

    # Rebalance dates (every 22 trading days)
    rebal_dates = get_rebalance_dates(close, interval=REBAL_INTERVAL)
    fprint("Rebalance schedule: %d dates (every %d trading days)" % (len(rebal_dates), REBAL_INTERVAL))

    # Build features and LGBM rankings
    fprint("\n" + "=" * 80)
    fprint("BUILDING LGBM RANKINGS (500d sliding, 21 features, VIX>20 filter)")
    fprint("=" * 80)
    records = build_feature_records(close, high, low, rebal_dates)
    if records.empty:
        fprint("FATAL: No feature records -- cannot proceed.")
        return

    rankings = walk_forward_lgbm_rank(records)
    if len(rankings) < 10:
        fprint("FATAL: Only %d ranking dates -- insufficient data" % len(rankings))
        return

    # Run all 12 variants
    all_results = {}
    all_trades = {}

    for var_name, var_desc, var_config in VARIANTS:
        fprint("\n" + "=" * 90)
        fprint("%s" % var_desc)
        fprint("=" * 90)

        trades, final_eq, early_ex = simulate_variant(
            rankings, close, atr_dict, rebal_dates, var_name, var_config,
        )
        fprint("  Trades: %d | Early exits: %d | Final equity: $%.0f" % (len(trades), early_ex, final_eq))

        stats = compute_full_stats(trades)
        if stats:
            fprint("  Sharpe: %.2f | Sortino: %.2f | PF: %.2f | WR: %.1f%%" % (
                stats["sharpe"], stats["sortino"], stats["profit_factor"], stats["win_rate"] * 100))
            fprint("  Total Return: %.1f%% | MDD: %.1f%% | Avg hold: %.1f days | Early exit: %.1f%%" % (
                stats["total_return_pct"], stats["max_dd_pct"], stats["avg_hold_days"], stats["early_exit_rate"] * 100))

        # Adversarial validation
        adv_results = None
        if len(trades) >= 10:
            adv_results = run_adversarial_validation(trades, var_name)

        all_results[var_name] = {
            "variant": var_name,
            "description": var_desc,
            "config": {k: v for k, v in var_config.items()},
            "stats": stats,
            "adversarial": adv_results,
        }
        all_trades[var_name] = trades

    # ==============================================================
    # SUMMARY TABLE
    # ==============================================================
    fprint("\n" + "=" * 120)
    fprint("SUMMARY TABLE -- SORTED BY SHARPE (DESCENDING)")
    fprint("=" * 120)

    sorted_variants = sorted(
        [(k, v) for k, v in all_results.items() if v["stats"] is not None],
        key=lambda x: x[1]["stats"]["sharpe"],
        reverse=True,
    )

    header = "%-16s %7s %7s %7s %7s %7s %7s %7s %7s %6s %6s" % (
        "Variant", "Sharpe", "Sortino", "PF", "WR%", "Return%", "MDD%", "AvgHold", "Early%", "Gates", "Trades")
    fprint(header)
    fprint("-" * len(header))

    for var_name, var_data in sorted_variants:
        s = var_data["stats"]
        gates = var_data["adversarial"]["gates_passed"] if var_data["adversarial"] else 0
        fprint("%-16s %7.2f %7.2f %7.2f %6.1f%% %6.1f%% %6.1f%% %7.1f %6.1f%% %5d/5 %6d" % (
            var_name, s["sharpe"], s["sortino"], s["profit_factor"],
            s["win_rate"] * 100, s["total_return_pct"], s["max_dd_pct"],
            s["avg_hold_days"], s["early_exit_rate"] * 100, gates, s["n_trades"]))

    # Best variant
    if sorted_variants:
        best_name, best_data = sorted_variants[0]
        fprint("\nBEST VARIANT: %s (Sharpe %.2f)" % (best_data["description"], best_data["stats"]["sharpe"]))
        if best_data["adversarial"]:
            fprint("  Gates: %d/5 passed" % best_data["adversarial"]["gates_passed"])

    # ==============================================================
    # SAVE RESULTS
    # ==============================================================

    # Save JSON
    output_json = {}
    for var_name, var_data in all_results.items():
        output_json[var_name] = {
            "description": var_data["description"],
            "config": var_data["config"],
            "stats": var_data["stats"],
            "adversarial": var_data["adversarial"],
        }

    json_path = OUTPUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2, default=str)
    fprint("\nResults saved to %s" % json_path)

    # Save trade-level detail
    for var_name, trades in all_trades.items():
        if trades:
            trade_path = OUTPUT_DIR / ("trades_%s.json" % var_name)
            with open(trade_path, "w") as f:
                json.dump(trades, f, indent=2, default=str)

    # ==============================================================
    # MLFLOW LOGGING
    # ==============================================================

    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="v10_exit_optimization_v1_%s" % t0.strftime("%Y%m%d_%H%M")):
                mlflow.log_param("sectors", ",".join(SECTORS))
                mlflow.log_param("n_features", len(FEATURE_NAMES))
                mlflow.log_param("dte", DTE)
                mlflow.log_param("otm_pct", OTM_PCT)
                mlflow.log_param("capital", CAP)
                mlflow.log_param("commission", COMMISSION)
                mlflow.log_param("haircut", HAIRCUT)
                mlflow.log_param("vix_threshold", VIX_REGIME_THRESHOLD)
                mlflow.log_param("wf_train_days", WF_TRAIN_DAYS)
                mlflow.log_param("rebal_interval", REBAL_INTERVAL)
                mlflow.log_param("n_variants", len(VARIANTS))
                mlflow.log_param("n_permutations", N_PERMUTATIONS)

                for var_name, var_data in all_results.items():
                    if var_data["stats"]:
                        s = var_data["stats"]
                        mlflow.log_metric("%s_sharpe" % var_name, s["sharpe"])
                        mlflow.log_metric("%s_sortino" % var_name, s["sortino"])
                        mlflow.log_metric("%s_pf" % var_name, s["profit_factor"])
                        mlflow.log_metric("%s_wr" % var_name, s["win_rate"])
                        mlflow.log_metric("%s_mdd" % var_name, s["max_dd_pct"])
                        mlflow.log_metric("%s_return" % var_name, s["total_return_pct"])
                        mlflow.log_metric("%s_avg_hold" % var_name, s["avg_hold_days"])
                        mlflow.log_metric("%s_early_exit" % var_name, s["early_exit_rate"])
                    if var_data["adversarial"]:
                        mlflow.log_metric("%s_gates" % var_name, var_data["adversarial"]["gates_passed"])

                if sorted_variants:
                    best_name, best_data = sorted_variants[0]
                    mlflow.log_metric("best_sharpe", best_data["stats"]["sharpe"])
                    mlflow.log_param("best_variant", best_name)

                mlflow.log_artifact(str(json_path))
            fprint("MLflow run logged to experiment '%s'" % EXPERIMENT_NAME)
        except Exception as e:
            fprint("MLflow logging error: %s" % e)

    elapsed = (datetime.now() - t0).total_seconds()
    fprint("\nCompleted in %.1f minutes." % (elapsed / 60))


if __name__ == "__main__":
    main()
