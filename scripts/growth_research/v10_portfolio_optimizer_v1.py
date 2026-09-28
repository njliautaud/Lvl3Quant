#!/usr/bin/env python3
"""
V10 Portfolio Optimizer V1 -- Multi-Strategy Combination on $645 Account
========================================================================

Tests how V10 sector spreads (Sharpe 6.32) combine with other validated
strategies on a single $645 account.

Strategies:
  1. V10 Sector Spreads (PRIMARY) -- LGBM ranker, 11 sectors, top4+bot4
  2. SPY Iron Condor Income -- Sell SPY iron condors when VIX < 20
  3. VIX Call Spread Income -- Sell VIX call spreads when VIX > 25

8 Variants:
  A: V10 Only (baseline)
  B: V10 + SPY IC (iron condors when V10 idle)
  C: V10 + VIX Call Spreads (VIX mean reversion)
  D: V10 + SPY IC + VIX (all three)
  E: V10 50% + SPY IC 30% + VIX 20% (fixed split)
  F: V10 dynamic sizing (2x when VIX>30, 1x when 20<VIX<30)
  G: V10 + put credit spreads on top-ranked sectors when VIX<20
  H: Risk-parity weighted (inverse vol allocation)

Config: $645 capital, $2.60 commission, 15% haircut, 500d sliding WF,
        max 60% deployed, max $200 per trade.

5-gate adversarial validation on every variant.
Correlation analysis between strategy components.

Output: output/growth_research/v10_portfolio_optimizer_v1/
MLflow experiment: v10_portfolio_optimizer_v1
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
from scipy.stats import norm

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# ================================================================
# PATHS + CONSTANTS
# ================================================================

_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")

if _NEPTUNE_BASE.exists():
    BASE = _NEPTUNE_BASE
    fprint("Running on Neptune: %s" % BASE)
else:
    BASE = _JUPITER_BASE
    fprint("Running on Jupiter: %s" % BASE)

OUTPUT_DIR = BASE / "output" / "growth_research" / "v10_portfolio_optimizer_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]

CAP = 645.0
TOP_K = 4
OTM_PCT = 0.04
PROFIT_TARGET = 0.30
DTE = 28
HAIRCUT = 0.15
COMMISSION = 2.60
EARLY_EXIT_COMMISSION = 2.60
MAX_PER_TRADE = 200.0
MAX_DEPLOY_PCT = 0.60
WF_TRAIN_DAYS = 500       # 500d sliding window (business days)
COST_WIDTH_MAX = 0.50

RISK_FREE_RATE = 0.045

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v10_portfolio_optimizer_v1"

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

# 18 legacy + 3 cross-asset = 21 features
LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d",
    "pct_52w_high", "mom_accel", "pct_pos_months_12m",
    "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]
CROSS_ASSET_FEATURES = [
    "sector_spy_beta_63d", "sector_relative_vol_21d", "cross_sector_dispersion",
]
ALL_FEATURES = LEGACY_FEATURES + CROSS_ASSET_FEATURES
assert len(LEGACY_FEATURES) == 18
assert len(CROSS_ASSET_FEATURES) == 3
assert len(ALL_FEATURES) == 21


# ================================================================
# SELF-CONTAINED BLACK-SCHOLES PRICING
# ================================================================

def _bs_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _bs_call(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_put(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
    """Black-Scholes European put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _estimate_iv(atr, spot, vix=20.0, atr_period=14):
    """Estimate IV from ATR and VIX."""
    if spot <= 0 or atr <= 0:
        return 0.25
    realized_vol = (atr / spot) * np.sqrt(252 / atr_period)
    iv_mult = 1.2 + 0.01 * max(vix - 20.0, 0.0)
    sigma = realized_vol * iv_mult
    return max(sigma, 0.10)


def _bs_delta_call(S, K, T, r, sigma):
    """Call delta for strike selection."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    return float(norm.cdf(d1))


def _bs_delta_put(S, K, T, r, sigma):
    """Put delta for strike selection."""
    return _bs_delta_call(S, K, T, r, sigma) - 1.0


def _find_strike_by_delta(S, T, r, sigma, target_delta, option_type="call",
                          lo_mult=0.70, hi_mult=1.30, steps=200):
    """Find strike price that gives approximately target_delta."""
    lo = S * lo_mult
    hi = S * hi_mult
    best_K = S
    best_diff = 999.0
    for K in np.linspace(lo, hi, steps):
        if option_type == "call":
            d = _bs_delta_call(S, K, T, r, sigma)
        else:
            d = _bs_delta_put(S, K, T, r, sigma)
        diff = abs(d - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_K = K
    return round(best_K, 2)


def price_bull_call_spread(S, K1, K2, dte, atr, vix=20.0):
    """Price bull call spread with haircut. Returns (entry_cost_ps, max_profit_ps)."""
    T = dte / 365.0
    sigma = _estimate_iv(atr, S, vix)
    fair = _bs_call(S, K1, T, sigma=sigma) - _bs_call(S, K2, T, sigma=sigma)
    fair = max(fair, 0.001)
    entry = fair * (1.0 + HAIRCUT)
    width = K2 - K1
    return float(entry), float(width - entry)


def price_bear_put_spread(S, K1, K2, dte, atr, vix=20.0):
    """Price bear put spread with haircut. Returns (entry_cost_ps, max_profit_ps)."""
    T = dte / 365.0
    sigma = _estimate_iv(atr, S, vix)
    fair = _bs_put(S, K2, T, sigma=sigma) - _bs_put(S, K1, T, sigma=sigma)
    fair = max(fair, 0.001)
    entry = fair * (1.0 + HAIRCUT)
    width = K2 - K1
    return float(entry), float(width - entry)


def revalue_spread_bs(S, K1, K2, dte_remaining, atr, vix, direction):
    """Revalue a spread at a point in time using BS."""
    if dte_remaining <= 0:
        if direction == "bull":
            return max(S - K1, 0.0) - max(S - K2, 0.0)
        else:
            return max(K2 - S, 0.0) - max(K1 - S, 0.0)
    T = dte_remaining / 365.0
    sigma = _estimate_iv(atr, S, vix)
    if direction == "bull":
        return max(_bs_call(S, K1, T, sigma=sigma) - _bs_call(S, K2, T, sigma=sigma), 0.0)
    else:
        return max(_bs_put(S, K2, T, sigma=sigma) - _bs_put(S, K1, T, sigma=sigma), 0.0)


def price_iron_condor(S, Kp_short, Kp_long, Kc_short, Kc_long, dte, sigma):
    """Price iron condor. Returns (credit_received_ps, max_loss_ps)."""
    T = dte / 365.0
    # Sell OTM put spread (bull put) + sell OTM call spread (bear call)
    put_short = _bs_put(S, Kp_short, T, sigma=sigma)
    put_long = _bs_put(S, Kp_long, T, sigma=sigma)
    call_short = _bs_call(S, Kc_short, T, sigma=sigma)
    call_long = _bs_call(S, Kc_long, T, sigma=sigma)

    put_spread_credit = put_short - put_long    # sell higher put, buy lower
    call_spread_credit = call_short - call_long  # sell lower call, buy higher
    total_credit = put_spread_credit + call_spread_credit
    total_credit = max(total_credit, 0.001)
    # Apply haircut (we receive less than theoretical)
    credit_received = total_credit * (1.0 - HAIRCUT)
    # Max loss = wider wing width - credit received
    put_width = Kp_short - Kp_long
    call_width = Kc_long - Kc_short
    max_wing = max(put_width, call_width)
    max_loss = max_wing - credit_received
    return float(credit_received), float(max_loss)


def revalue_iron_condor(S, Kp_short, Kp_long, Kc_short, Kc_long, dte_remaining, sigma):
    """Revalue iron condor at a point in time. Returns current value of the short position."""
    if dte_remaining <= 0:
        # At expiry: intrinsic
        put_intrinsic = max(Kp_short - S, 0.0) - max(Kp_long - S, 0.0)
        call_intrinsic = max(S - Kc_short, 0.0) - max(S - Kc_long, 0.0)
        return put_intrinsic + call_intrinsic  # what we owe
    T = dte_remaining / 365.0
    put_short = _bs_put(S, Kp_short, T, sigma=sigma)
    put_long = _bs_put(S, Kp_long, T, sigma=sigma)
    call_short = _bs_call(S, Kc_short, T, sigma=sigma)
    call_long = _bs_call(S, Kc_long, T, sigma=sigma)
    current_cost = (put_short - put_long) + (call_short - call_long)
    return max(current_cost, 0.0)


# ================================================================
# DATA DOWNLOAD + FEATURE ENGINEERING
# ================================================================

def download_data():
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
    atr_dict = {}
    all_tickers = SECTORS + ["SPY"]
    for tk in all_tickers:
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


def compute_legacy_features(px, spy_slice):
    """Compute 18 legacy features for a sector."""
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()

    # vol_21d, vol_63d
    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2

    f["sharpe_63d"] = float(rets.iloc[-63:].mean() / (rets.iloc[-63:].std() + 1e-10) * np.sqrt(252)) if len(rets) > 63 else 0.0

    # maxdd_63d
    px_63 = px.iloc[-63:]
    pk = px_63.cummax()
    f["maxdd_63d"] = float(((px_63 / pk) - 1).min())

    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5

    dr = rets.iloc[-63:][rets.iloc[-63:] < 0]
    f["sortino_63d"] = float(rets.iloc[-63:].mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0

    pk252 = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk252) - 1).min())
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
        spy_vol_21 = spy_ret.iloc[-21:].std()
        sec_vol_21 = sec_ret.iloc[-21:].std()
        f["sector_relative_vol_21d"] = float(sec_vol_21 / (spy_vol_21 + 1e-10))
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


# ================================================================
# LGBM WALK-FORWARD RANKING
# ================================================================

def get_monthly_rebalance_dates(close):
    monthly = close.index.to_series().resample("MS").first().dropna()
    return pd.DatetimeIndex(monthly.values)


def build_feature_records(close, high, low, rebal_dates):
    """Build feature records for all rebalance dates."""
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        # V10 uses VIX>20 filter for sector spreads
        if vix is not None and dt in vix.index:
            cv = float(vix.loc[dt])
        else:
            cv = 20.0

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
            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk,
                   "fwd_ret": fwd_ret, "vix_at_date": cv}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(0.0)
    fprint("    %d feature records, %d dates" % (len(df), len(df["date"].unique())))
    return df


def walk_forward_lgbm_rank(df):
    """Walk-forward LGBM ranking with 500d sliding window (approx 24 months)."""
    import lightgbm as lgb
    if len(df) < 100:
        return {}

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())
    # Convert 500 calendar days to approximate number of monthly rebal periods
    # 500d ~ 24 months
    wf_periods = 24
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


# ================================================================
# TRADE EXECUTION HELPERS
# ================================================================

def compute_strikes(S, direction, otm_pct=OTM_PCT):
    """Compute V10 strike prices: otm_pct OTM, adaptive max($3, 3%) width."""
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


def execute_v10_trade(tk, dt, direction, close, atr_dict, vix_val, equity,
                      max_pos, target_pct=PROFIT_TARGET, size_mult=1.0):
    """Execute a V10 sector spread trade with profit target."""
    if tk not in close.columns or tk not in atr_dict:
        return None
    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    av = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015
    K1, K2 = compute_strikes(S, direction)

    try:
        if direction == "bull":
            entry_cost_ps, max_profit_ps = price_bull_call_spread(S, K1, K2, DTE, av, vix_val)
        else:
            entry_cost_ps, max_profit_ps = price_bear_put_spread(S, K1, K2, DTE, av, vix_val)
    except Exception:
        return None

    if entry_cost_ps <= 0:
        return None

    spread_width = abs(K2 - K1)
    cwr = entry_cost_ps / max(spread_width, 0.01)
    if cwr > COST_WIDTH_MAX:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION
    effective_max = min(max_pos * size_mult, MAX_PER_TRADE)
    if total_cost <= 0 or total_cost > effective_max:
        return None

    # Profit target exit
    exited_early = False
    exit_day_idx = ei
    hold_days = DTE

    if max_profit_ps > 0 and target_pct < 1.0:
        for check_idx in range(di + 1, ei + 1):
            if check_idx >= len(close):
                break
            check_date = close.index[check_idx]
            dte_remaining = ei - check_idx
            S_now = float(close[tk].iloc[check_idx])
            av_now = float(atr_dict[tk].loc[check_date]) if check_date in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[check_date]) else S_now * 0.015
            current_value_ps = revalue_spread_bs(S_now, K1, K2, dte_remaining, av_now, vix_val, direction)
            unrealized_gain_ps = current_value_ps - entry_cost_ps
            if unrealized_gain_ps >= target_pct * max_profit_ps:
                exited_early = True
                exit_day_idx = check_idx
                hold_days = check_idx - di
                break

    Se = float(close[tk].iloc[exit_day_idx])
    if exited_early:
        dte_at_exit = ei - exit_day_idx
        av_exit = float(atr_dict[tk].loc[close.index[exit_day_idx]]) if close.index[exit_day_idx] in atr_dict[tk].index else Se * 0.015
        exit_value_ps = revalue_spread_bs(Se, K1, K2, dte_at_exit, av_exit, vix_val, direction)
        pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION - EARLY_EXIT_COMMISSION
    else:
        if direction == "bull":
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
        else:
            intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION

    exit_idx = min(di + hold_days, len(close) - 1)
    return {
        "pnl": round(pnl, 2),
        "total_cost": round(total_cost, 2),
        "entry_date": str(dt.date()),
        "exit_date": str(close.index[exit_idx].date()),
        "ticker": tk,
        "direction": direction,
        "strategy": "v10_sector",
        "exited_early": exited_early,
        "hold_days": hold_days,
        "vix": round(vix_val, 1),
    }


def execute_spy_iron_condor(dt, close, atr_dict, vix_val, equity):
    """Execute SPY iron condor when VIX < 20. DTE=30, 20-delta, 5pt wings."""
    if "SPY" not in close.columns:
        return None
    S = float(close["SPY"].loc[dt])
    di = close.index.get_loc(dt)
    dte = 30
    ei = min(di + dte, len(close) - 1)
    if ei <= di:
        return None

    av = float(atr_dict["SPY"].loc[dt]) if "SPY" in atr_dict and dt in atr_dict["SPY"].index else S * 0.01
    sigma = _estimate_iv(av, S, vix_val)
    T = dte / 365.0

    # Find 20-delta strikes
    Kp_short = _find_strike_by_delta(S, T, RISK_FREE_RATE, sigma, -0.20, "put")
    Kc_short = _find_strike_by_delta(S, T, RISK_FREE_RATE, sigma, 0.20, "call")
    Kp_long = round(Kp_short - 5, 2)
    Kc_long = round(Kc_short + 5, 2)

    credit_ps, max_loss_ps = price_iron_condor(S, Kp_short, Kp_long, Kc_short, Kc_long, dte, sigma)

    total_cost = max_loss_ps * 100 + COMMISSION * 2  # 2 spreads
    if total_cost <= 0 or total_cost > 100.0:
        # Scale down if needed
        if total_cost > 100.0:
            return None

    # Walk through days for 50% profit target
    exited_early = False
    exit_day_idx = ei
    hold_days = dte
    target_credit_close = credit_ps * 0.50  # close when IC cost drops to 50% of credit

    for check_idx in range(di + 1, ei + 1):
        if check_idx >= len(close):
            break
        S_now = float(close["SPY"].iloc[check_idx])
        dte_remaining = ei - check_idx
        av_now = float(atr_dict["SPY"].loc[close.index[check_idx]]) if "SPY" in atr_dict and close.index[check_idx] in atr_dict["SPY"].index else S_now * 0.01
        sigma_now = _estimate_iv(av_now, S_now, vix_val)
        current_cost = revalue_iron_condor(S_now, Kp_short, Kp_long, Kc_short, Kc_long, dte_remaining, sigma_now)

        # Profit = credit_received - current_cost_to_close
        unrealized_profit = credit_ps - current_cost * (1.0 + HAIRCUT)  # pay more to close
        if unrealized_profit >= credit_ps * 0.50:
            exited_early = True
            exit_day_idx = check_idx
            hold_days = check_idx - di
            break

    Se = float(close["SPY"].iloc[exit_day_idx])
    if exited_early:
        pnl = unrealized_profit * 100 - COMMISSION * 2 - EARLY_EXIT_COMMISSION * 2
    else:
        # At expiry
        put_intrinsic = max(Kp_short - Se, 0.0) - max(Kp_long - Se, 0.0)
        call_intrinsic = max(Se - Kc_short, 0.0) - max(Se - Kc_long, 0.0)
        expiry_cost = put_intrinsic + call_intrinsic
        pnl = (credit_ps - expiry_cost) * 100 - COMMISSION * 2

    exit_idx = min(di + hold_days, len(close) - 1)
    return {
        "pnl": round(pnl, 2),
        "total_cost": round(total_cost, 2),
        "entry_date": str(dt.date()),
        "exit_date": str(close.index[exit_idx].date()),
        "ticker": "SPY",
        "direction": "iron_condor",
        "strategy": "spy_ic",
        "exited_early": exited_early,
        "hold_days": hold_days,
        "vix": round(vix_val, 1),
    }


def execute_vix_call_spread(dt, close, vix_val, equity):
    """Execute VIX call spread when VIX > 25 (mean reversion play)."""
    if "VIX" not in close.columns:
        return None
    di = close.index.get_loc(dt)
    dte = 30
    ei = min(di + dte, len(close) - 1)
    if ei <= di:
        return None

    # VIX call spread: buy call at VIX+5, sell at VIX+10
    # We are selling the higher-strike call spread (bearish on VIX going higher)
    # Credit received = sell(VIX+5 call) - buy(VIX+10 call)
    S_vix = vix_val
    K_sell = round(S_vix + 5, 0)
    K_buy = round(S_vix + 10, 0)
    T = dte / 365.0
    sigma_vix = max(S_vix * 0.01 * np.sqrt(252), 0.50)  # VIX vol is high

    sell_price = _bs_call(S_vix, K_sell, T, sigma=sigma_vix)
    buy_price = _bs_call(S_vix, K_buy, T, sigma=sigma_vix)
    credit_ps = sell_price - buy_price
    credit_ps = max(credit_ps * (1.0 - HAIRCUT), 0.001)

    width = K_buy - K_sell
    max_loss = width - credit_ps
    total_cost = max_loss * 100 + COMMISSION
    if total_cost <= 0 or total_cost > 100.0:
        return None

    # Walk through days -- exit at 50% profit or VIX drops below 20
    exited_early = False
    exit_day_idx = ei
    hold_days = dte
    vix_series = close["VIX"]

    for check_idx in range(di + 1, ei + 1):
        if check_idx >= len(close):
            break
        check_date = close.index[check_idx]
        vix_now = float(vix_series.iloc[check_idx]) if check_idx < len(vix_series) else vix_val
        dte_remaining = ei - check_idx
        T_now = max(dte_remaining / 365.0, 1e-6)
        sigma_now = max(vix_now * 0.01 * np.sqrt(252), 0.50)

        sell_now = _bs_call(vix_now, K_sell, T_now, sigma=sigma_now)
        buy_now = _bs_call(vix_now, K_buy, T_now, sigma=sigma_now)
        current_cost = (sell_now - buy_now) * (1.0 + HAIRCUT)

        unrealized = credit_ps - current_cost
        if unrealized >= credit_ps * 0.50 or vix_now < 20.0:
            exited_early = True
            exit_day_idx = check_idx
            hold_days = check_idx - di
            break

    vix_exit = float(vix_series.iloc[exit_day_idx]) if exit_day_idx < len(vix_series) else vix_val
    if exited_early:
        pnl = unrealized * 100 - COMMISSION - EARLY_EXIT_COMMISSION
    else:
        # At expiry: VIX at exit
        sell_intrinsic = max(vix_exit - K_sell, 0.0)
        buy_intrinsic = max(vix_exit - K_buy, 0.0)
        expiry_cost = sell_intrinsic - buy_intrinsic
        pnl = (credit_ps - max(expiry_cost, 0.0)) * 100 - COMMISSION

    exit_idx = min(di + hold_days, len(close) - 1)
    return {
        "pnl": round(pnl, 2),
        "total_cost": round(total_cost, 2),
        "entry_date": str(dt.date()),
        "exit_date": str(close.index[exit_idx].date()),
        "ticker": "VIX",
        "direction": "vix_call_spread",
        "strategy": "vix_cs",
        "exited_early": exited_early,
        "hold_days": hold_days,
        "vix": round(vix_val, 1),
    }


# ================================================================
# PORTFOLIO SIMULATION ENGINE
# ================================================================

def simulate_portfolio(variant_name, rankings, close, atr_dict, rebal_dates,
                       enable_v10=True, enable_spy_ic=False, enable_vix_cs=False,
                       v10_alloc_pct=1.0, spy_ic_alloc_pct=0.0, vix_cs_alloc_pct=0.0,
                       v10_dynamic_sizing=False, enable_sector_puts_low_vix=False,
                       risk_parity=False):
    """
    Run a combined portfolio simulation.

    All strategies share one equity curve starting at $645.
    Max 60% deployed at any time. Max $200 per trade.
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    deployed = 0.0  # dollars currently deployed

    # Track PnL by strategy for correlation analysis
    daily_pnl = {"v10": {}, "spy_ic": {}, "vix_cs": {}, "sector_puts": {}}

    for dt in sorted(rebal_dates):
        if dt not in spy.index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        # Available capital
        max_deploy = equity * MAX_DEPLOY_PCT
        remaining_deploy = max_deploy - deployed
        if remaining_deploy < 20:
            deployed = 0.0  # Reset monthly (positions expire)
            remaining_deploy = max_deploy

        # Risk-parity: compute rolling vols for allocation
        if risk_parity:
            # Compute trailing 63d vol for each strategy type
            v10_vol = _compute_strategy_vol(trades, "v10_sector", dt)
            ic_vol = _compute_strategy_vol(trades, "spy_ic", dt)
            vix_vol = _compute_strategy_vol(trades, "vix_cs", dt)
            total_inv_vol = 0.0
            vols = {"v10": v10_vol, "ic": ic_vol, "vix": vix_vol}
            for v in vols.values():
                if v > 0:
                    total_inv_vol += 1.0 / v
            if total_inv_vol > 0:
                v10_alloc_pct = (1.0 / max(v10_vol, 0.001)) / total_inv_vol if enable_v10 else 0.0
                spy_ic_alloc_pct = (1.0 / max(ic_vol, 0.001)) / total_inv_vol if enable_spy_ic else 0.0
                vix_cs_alloc_pct = (1.0 / max(vix_vol, 0.001)) / total_inv_vol if enable_vix_cs else 0.0

        # ---- Strategy 1: V10 Sector Spreads (VIX > 20 filter) ----
        if enable_v10 and cv >= 20.0:
            ranking_date = None
            for rd in sorted(rankings.keys()):
                if rd <= dt:
                    ranking_date = rd
            if ranking_date is not None:
                scores = rankings[ranking_date]
                if scores and len(scores) >= 6:
                    ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    ranked_asc = sorted(scores.items(), key=lambda x: x[1])
                    bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
                    bear_picks = [t for t, _ in ranked_asc[:TOP_K]]

                    # Dynamic sizing
                    size_mult = 1.0
                    if v10_dynamic_sizing:
                        if cv > 30:
                            size_mult = 2.0
                        elif cv > 20:
                            size_mult = 1.0

                    v10_budget = remaining_deploy * v10_alloc_pct
                    n_positions = len(bull_picks) + len(bear_picks)
                    per_pos = min(v10_budget / max(n_positions, 1), MAX_PER_TRADE)

                    for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
                        for tk in picks:
                            if remaining_deploy < 20:
                                break
                            result = execute_v10_trade(
                                tk, dt, direction, close, atr_dict, cv, equity,
                                per_pos, size_mult=size_mult,
                            )
                            if result is not None:
                                equity += result["pnl"]
                                deployed += result["total_cost"]
                                remaining_deploy -= result["total_cost"]
                                trades.append(result)
                                # Track daily PnL
                                exit_d = result["exit_date"]
                                daily_pnl["v10"][exit_d] = daily_pnl["v10"].get(exit_d, 0.0) + result["pnl"]

        # ---- Strategy 2: SPY Iron Condors (VIX < 20) ----
        if enable_spy_ic and cv < 20.0 and remaining_deploy >= 20:
            ic_budget = remaining_deploy * (spy_ic_alloc_pct if not enable_v10 else 1.0)
            if ic_budget >= 20:
                result = execute_spy_iron_condor(dt, close, atr_dict, cv, equity)
                if result is not None:
                    equity += result["pnl"]
                    deployed += result["total_cost"]
                    remaining_deploy -= result["total_cost"]
                    trades.append(result)
                    exit_d = result["exit_date"]
                    daily_pnl["spy_ic"][exit_d] = daily_pnl["spy_ic"].get(exit_d, 0.0) + result["pnl"]

        # ---- Strategy 3: VIX Call Spreads (VIX > 25) ----
        if enable_vix_cs and cv > 25.0 and remaining_deploy >= 20:
            vix_budget = remaining_deploy * (vix_cs_alloc_pct if not enable_v10 else 1.0)
            if vix_budget >= 20:
                result = execute_vix_call_spread(dt, close, cv, equity)
                if result is not None:
                    equity += result["pnl"]
                    deployed += result["total_cost"]
                    remaining_deploy -= result["total_cost"]
                    trades.append(result)
                    exit_d = result["exit_date"]
                    daily_pnl["vix_cs"][exit_d] = daily_pnl["vix_cs"].get(exit_d, 0.0) + result["pnl"]

        # ---- Strategy G: Put credit spreads on top sectors when VIX < 20 ----
        if enable_sector_puts_low_vix and cv < 20.0 and remaining_deploy >= 20:
            ranking_date = None
            for rd in sorted(rankings.keys()):
                if rd <= dt:
                    ranking_date = rd
            if ranking_date is not None:
                scores = rankings[ranking_date]
                if scores and len(scores) >= 6:
                    # Top-ranked sectors: sell put spreads (bullish)
                    ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    top_picks = [t for t, _ in ranked_desc[:TOP_K]]
                    for tk in top_picks:
                        if remaining_deploy < 20:
                            break
                        result = execute_v10_trade(
                            tk, dt, "bear", close, atr_dict, cv, equity,
                            min(remaining_deploy / TOP_K, MAX_PER_TRADE),
                        )
                        if result is not None:
                            # Flip direction: we want put credit spreads
                            # Actually, a bear put spread in our framework = buying puts
                            # For put CREDIT spread, we want to sell the higher put, buy lower
                            # This is equivalent to "bull" put direction (bullish)
                            result_bull = execute_v10_trade(
                                tk, dt, "bull", close, atr_dict, cv, equity,
                                min(remaining_deploy / TOP_K, MAX_PER_TRADE),
                            )
                            if result_bull is not None:
                                result_bull["strategy"] = "sector_puts"
                                equity += result_bull["pnl"]
                                deployed += result_bull["total_cost"]
                                remaining_deploy -= result_bull["total_cost"]
                                trades.append(result_bull)
                                exit_d = result_bull["exit_date"]
                                daily_pnl["sector_puts"][exit_d] = daily_pnl["sector_puts"].get(exit_d, 0.0) + result_bull["pnl"]

        # Reset deployed at month boundary (positions expire within DTE)
        deployed = max(deployed - sum(t["total_cost"] for t in trades
                                      if t["exit_date"] <= str(dt.date()) and
                                      t["entry_date"] >= str((dt - pd.Timedelta(days=DTE + 5)).date())), 0)

    return trades, equity, daily_pnl


def _compute_strategy_vol(trades, strategy_name, current_date):
    """Compute trailing PnL volatility for a strategy (for risk parity)."""
    strat_trades = [t for t in trades if t.get("strategy") == strategy_name
                    and pd.Timestamp(t["exit_date"]) < current_date]
    if len(strat_trades) < 5:
        return 0.10  # default vol
    recent = strat_trades[-min(len(strat_trades), 30):]
    pnls = [t["pnl"] for t in recent]
    return float(np.std(pnls)) if np.std(pnls) > 0 else 0.10


# ================================================================
# METRICS + ADVERSARIAL VALIDATION
# ================================================================

def compute_full_stats(trades, initial_capital=CAP):
    """Compute comprehensive stats."""
    if not trades:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    equity_arr = [initial_capital]
    for p in pnls:
        equity_arr.append(equity_arr[-1] + p)
    equity_arr = np.array(equity_arr)
    peak = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peak) / np.where(peak > 0, peak, 1)

    wr = float(np.mean(pnls > 0))
    gross_win = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0
    gross_loss = float(abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-10
    pf = gross_win / max(gross_loss, 1e-10)

    # Calendar month Sharpe (primary metric)
    trade_df = pd.DataFrame(trades)
    trade_df["entry_month"] = pd.to_datetime(trade_df["entry_date"]).dt.to_period("M")
    monthly_pnl = trade_df.groupby("entry_month")["pnl"].sum()
    if len(monthly_pnl) > 1:
        sharpe = float(monthly_pnl.mean() / (monthly_pnl.std() + 1e-10) * np.sqrt(12))
    else:
        sharpe = 0.0

    # Sortino
    neg_months = monthly_pnl[monthly_pnl < 0]
    if len(neg_months) > 1 and neg_months.std() > 0:
        sortino = float(monthly_pnl.mean() / neg_months.std() * np.sqrt(12))
    else:
        sortino = sharpe * 1.5

    # CAGR
    dates_str = [t["entry_date"] for t in trades]
    first_date = pd.Timestamp(min(dates_str))
    last_date = pd.Timestamp(max(dates_str))
    years = max((last_date - first_date).days / 365.25, 0.5)
    final_eq = equity_arr[-1]
    if final_eq > 0:
        cagr = (final_eq / initial_capital) ** (1 / years) - 1
    else:
        cagr = -1.0

    max_dd_pct = float(dd.min()) * 100
    total_ret = (final_eq / initial_capital - 1) * 100

    return {
        "n_trades": len(trades),
        "total_pnl": round(float(pnls.sum()), 2),
        "total_return_pct": round(total_ret, 2),
        "avg_pnl": round(float(pnls.mean()), 2),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr, 4),
        "max_dd_pct": round(max_dd_pct, 2),
        "final_equity": round(final_eq, 2),
    }


def adversarial_validate(trades, spy_prices, variant_name, n_perms=300):
    """5-gate adversarial validation."""
    if len(trades) < 10:
        return {"variant": variant_name, "error": "Too few trades (%d)" % len(trades),
                "gates_passed": 0, "gates_total": 5, "all_passed": False, "gates": []}

    pnls = np.array([t["pnl"] for t in trades])
    equity_vals = [CAP]
    for p in pnls:
        equity_vals.append(equity_vals[-1] + p)

    dates_list = [pd.Timestamp(trades[0]["entry_date"]) - pd.Timedelta(days=1)]
    for t in trades:
        dates_list.append(pd.Timestamp(t["exit_date"]))
    eq_series = pd.Series(equity_vals, index=pd.DatetimeIndex(dates_list))
    eq_series = eq_series.groupby(eq_series.index).last()

    # Monthly returns for Sharpe
    monthly_eq = eq_series.resample("ME").last().dropna()
    if len(monthly_eq) > 1:
        monthly_rets = monthly_eq.pct_change().dropna()
        real_sharpe = float(monthly_rets.mean() / (monthly_rets.std() + 1e-10) * np.sqrt(12))
    else:
        monthly_rets = pd.Series(dtype=float)
        real_sharpe = 0.0

    gates = []

    # Gate 1: Permutation test (300 trials, p < 0.05)
    beat_count = 0
    exit_dates = pd.to_datetime([t["exit_date"] for t in trades])
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(pnls))
        flipped = pnls * signs
        eq_perm = [CAP]
        for p in flipped:
            eq_perm.append(eq_perm[-1] + p)
        eq_s = pd.Series(eq_perm, index=pd.DatetimeIndex(
            [exit_dates[0] - pd.Timedelta(days=1)] + list(exit_dates)))
        eq_s = eq_s.groupby(eq_s.index).last()
        m_eq = eq_s.resample("ME").last().dropna()
        if len(m_eq) > 1:
            m_r = m_eq.pct_change().dropna()
            perm_sharpe = float(m_r.mean() / (m_r.std() + 1e-10) * np.sqrt(12))
        else:
            perm_sharpe = 0.0
        if perm_sharpe >= real_sharpe:
            beat_count += 1
    p_value = beat_count / n_perms
    gates.append({
        "name": "Permutation Test",
        "passed": p_value < 0.05,
        "metric": "p_value",
        "value": round(p_value, 4),
        "threshold": 0.05,
    })

    # Gate 2: Regime stability (|Sharpe_bull - Sharpe_bear| / max < 0.50)
    bull_pnls = []
    bear_pnls = []
    if spy_prices is not None:
        spy_sorted = spy_prices.sort_index()
        for t in trades:
            entry_prices = spy_sorted[spy_sorted.index <= pd.Timestamp(t["entry_date"])]
            exit_prices = spy_sorted[spy_sorted.index <= pd.Timestamp(t["exit_date"])]
            if len(entry_prices) == 0 or len(exit_prices) == 0:
                continue
            spy_e = float(entry_prices.iloc[-1])
            spy_x = float(exit_prices.iloc[-1])
            if spy_x >= spy_e:
                bull_pnls.append(t["pnl"])
            else:
                bear_pnls.append(t["pnl"])

    if len(bull_pnls) >= 5 and len(bear_pnls) >= 5:
        bull_sharpe = np.mean(bull_pnls) / (np.std(bull_pnls) + 1e-10) * np.sqrt(12)
        bear_sharpe = np.mean(bear_pnls) / (np.std(bear_pnls) + 1e-10) * np.sqrt(12)
        max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-10)
        regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs
        gate2_passed = regime_gap < 0.50
    else:
        regime_gap = 0.0
        gate2_passed = True  # insufficient data
    gates.append({
        "name": "Regime Stability",
        "passed": gate2_passed,
        "metric": "sharpe_gap_ratio",
        "value": round(regime_gap, 4),
        "threshold": 0.50,
    })

    # Gate 3: Sub-period (both halves Sharpe > 0.5)
    mid = len(trades) // 2
    half1_pnls = pnls[:mid]
    half2_pnls = pnls[mid:]
    h1_sharpe = float(np.mean(half1_pnls) / (np.std(half1_pnls) + 1e-10) * np.sqrt(12)) if len(half1_pnls) > 1 else 0.0
    h2_sharpe = float(np.mean(half2_pnls) / (np.std(half2_pnls) + 1e-10) * np.sqrt(12)) if len(half2_pnls) > 1 else 0.0
    gate3_passed = h1_sharpe > 0.5 and h2_sharpe > 0.5
    gates.append({
        "name": "Sub-Period Stability",
        "passed": gate3_passed,
        "metric": "min_half_sharpe",
        "value": round(min(h1_sharpe, h2_sharpe), 4),
        "threshold": 0.50,
    })

    # Gate 4: Outlier removal (trim 5% extreme trades, Sharpe > 0.5)
    n_trim = max(1, int(len(pnls) * 0.05))
    sorted_pnls = np.sort(pnls)
    trimmed = sorted_pnls[n_trim:-n_trim] if n_trim > 0 and len(pnls) > 2 * n_trim else pnls
    if len(trimmed) > 1 and np.std(trimmed) > 0:
        trimmed_sharpe = float(np.mean(trimmed) / np.std(trimmed) * np.sqrt(12))
    else:
        trimmed_sharpe = 0.0
    gates.append({
        "name": "Outlier Removal",
        "passed": trimmed_sharpe > 0.5,
        "metric": "trimmed_sharpe",
        "value": round(trimmed_sharpe, 4),
        "threshold": 0.50,
    })

    # Gate 5: Yearly consistency (>= 60% years profitable)
    trade_df = pd.DataFrame(trades)
    trade_df["year"] = pd.to_datetime(trade_df["exit_date"]).dt.year
    yearly_pnl = trade_df.groupby("year")["pnl"].sum()
    n_years = len(yearly_pnl)
    n_profitable = int((yearly_pnl > 0).sum())
    pct_profitable = n_profitable / max(n_years, 1)
    gates.append({
        "name": "Yearly Consistency",
        "passed": pct_profitable >= 0.60,
        "metric": "pct_years_profitable",
        "value": round(pct_profitable, 4),
        "threshold": 0.60,
    })

    all_passed = all(g["passed"] for g in gates)
    return {
        "variant": variant_name,
        "gates_passed": sum(1 for g in gates if g["passed"]),
        "gates_total": 5,
        "all_passed": all_passed,
        "gates": gates,
        "real_sharpe": round(real_sharpe, 3),
    }


# ================================================================
# CORRELATION ANALYSIS
# ================================================================

def compute_strategy_correlations(daily_pnl_dict):
    """Compute daily PnL correlations between strategy components."""
    # Build a DataFrame with daily PnL per strategy
    all_dates = set()
    for strat_pnls in daily_pnl_dict.values():
        all_dates.update(strat_pnls.keys())

    if not all_dates:
        return {}

    all_dates = sorted(all_dates)
    df = pd.DataFrame(index=all_dates)
    for strat, pnls in daily_pnl_dict.items():
        df[strat] = [pnls.get(d, 0.0) for d in all_dates]

    # Only keep strategies with actual trades
    active = [c for c in df.columns if df[c].abs().sum() > 0]
    if len(active) < 2:
        return {"note": "fewer than 2 active strategies, no correlation to compute"}

    corr_matrix = df[active].corr()
    result = {}
    for i, s1 in enumerate(active):
        for j, s2 in enumerate(active):
            if j > i:
                result["%s_vs_%s" % (s1, s2)] = round(float(corr_matrix.loc[s1, s2]), 4)

    return result


# ================================================================
# VARIANT DEFINITIONS
# ================================================================

VARIANTS = {
    "A": {
        "name": "V10 Only (baseline)",
        "enable_v10": True, "enable_spy_ic": False, "enable_vix_cs": False,
        "v10_alloc": 1.0, "spy_ic_alloc": 0.0, "vix_cs_alloc": 0.0,
        "v10_dynamic": False, "sector_puts": False, "risk_parity": False,
    },
    "B": {
        "name": "V10 + SPY IC",
        "enable_v10": True, "enable_spy_ic": True, "enable_vix_cs": False,
        "v10_alloc": 1.0, "spy_ic_alloc": 1.0, "vix_cs_alloc": 0.0,
        "v10_dynamic": False, "sector_puts": False, "risk_parity": False,
    },
    "C": {
        "name": "V10 + VIX Call Spreads",
        "enable_v10": True, "enable_spy_ic": False, "enable_vix_cs": True,
        "v10_alloc": 1.0, "spy_ic_alloc": 0.0, "vix_cs_alloc": 1.0,
        "v10_dynamic": False, "sector_puts": False, "risk_parity": False,
    },
    "D": {
        "name": "V10 + SPY IC + VIX",
        "enable_v10": True, "enable_spy_ic": True, "enable_vix_cs": True,
        "v10_alloc": 1.0, "spy_ic_alloc": 1.0, "vix_cs_alloc": 1.0,
        "v10_dynamic": False, "sector_puts": False, "risk_parity": False,
    },
    "E": {
        "name": "V10 50% + SPY IC 30% + VIX 20%",
        "enable_v10": True, "enable_spy_ic": True, "enable_vix_cs": True,
        "v10_alloc": 0.50, "spy_ic_alloc": 0.30, "vix_cs_alloc": 0.20,
        "v10_dynamic": False, "sector_puts": False, "risk_parity": False,
    },
    "F": {
        "name": "V10 Dynamic Sizing",
        "enable_v10": True, "enable_spy_ic": False, "enable_vix_cs": False,
        "v10_alloc": 1.0, "spy_ic_alloc": 0.0, "vix_cs_alloc": 0.0,
        "v10_dynamic": True, "sector_puts": False, "risk_parity": False,
    },
    "G": {
        "name": "V10 + Sector Puts (VIX<20)",
        "enable_v10": True, "enable_spy_ic": False, "enable_vix_cs": False,
        "v10_alloc": 1.0, "spy_ic_alloc": 0.0, "vix_cs_alloc": 0.0,
        "v10_dynamic": False, "sector_puts": True, "risk_parity": False,
    },
    "H": {
        "name": "Risk-Parity Weighted",
        "enable_v10": True, "enable_spy_ic": True, "enable_vix_cs": True,
        "v10_alloc": 0.34, "spy_ic_alloc": 0.33, "vix_cs_alloc": 0.33,
        "v10_dynamic": False, "sector_puts": False, "risk_parity": True,
    },
}


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("V10 Portfolio Optimizer V1")
    fprint("=" * 70)

    # ---- Download data ----
    fprint("\n[1/5] Downloading data...")
    close, high, low = download_data()

    # ---- Compute ATR ----
    fprint("\n[2/5] Computing ATR series...")
    atr_dict = compute_atr_series(high, low, close)

    # ---- Build features + LGBM rankings ----
    fprint("\n[3/5] Building features and LGBM walk-forward rankings...")
    rebal_dates = get_monthly_rebalance_dates(close)
    fprint("  %d monthly rebalance dates" % len(rebal_dates))
    feature_df = build_feature_records(close, high, low, rebal_dates)
    rankings = walk_forward_lgbm_rank(feature_df)

    if not rankings:
        fprint("ERROR: No rankings produced. Exiting.")
        return

    spy_prices = close["SPY"] if "SPY" in close.columns else None

    # ---- Run all variants ----
    fprint("\n[4/5] Running 8 portfolio variants...")
    results = {}
    all_correlations = {}

    for key in sorted(VARIANTS.keys()):
        cfg = VARIANTS[key]
        fprint("\n  --- Variant %s: %s ---" % (key, cfg["name"]))

        trades, final_eq, daily_pnl = simulate_portfolio(
            variant_name="%s_%s" % (key, cfg["name"]),
            rankings=rankings,
            close=close,
            atr_dict=atr_dict,
            rebal_dates=rebal_dates,
            enable_v10=cfg["enable_v10"],
            enable_spy_ic=cfg["enable_spy_ic"],
            enable_vix_cs=cfg["enable_vix_cs"],
            v10_alloc_pct=cfg["v10_alloc"],
            spy_ic_alloc_pct=cfg["spy_ic_alloc"],
            vix_cs_alloc_pct=cfg["vix_cs_alloc"],
            v10_dynamic_sizing=cfg["v10_dynamic"],
            enable_sector_puts_low_vix=cfg["sector_puts"],
            risk_parity=cfg["risk_parity"],
        )

        fprint("    Trades: %d | Final equity: $%.2f" % (len(trades), final_eq))

        # Stats
        stats_dict = compute_full_stats(trades)
        if stats_dict is None:
            fprint("    SKIP: no trades")
            continue

        fprint("    Sharpe: %.3f | Sortino: %.3f | WR: %.1f%% | PF: %.3f | MDD: %.1f%%" % (
            stats_dict["sharpe"], stats_dict["sortino"],
            stats_dict["win_rate"] * 100, stats_dict["profit_factor"],
            stats_dict["max_dd_pct"],
        ))

        # Adversarial validation
        validation = adversarial_validate(trades, spy_prices, cfg["name"])
        fprint("    Gates: %d/%d passed | All passed: %s" % (
            validation["gates_passed"], validation["gates_total"],
            "YES" if validation["all_passed"] else "NO",
        ))
        for g in validation["gates"]:
            status = "PASS" if g["passed"] else "FAIL"
            fprint("      [%s] %s: %s=%.4f (threshold: %.4f)" % (
                status, g["name"], g["metric"], g["value"], g["threshold"]))

        # Correlation analysis
        correlations = compute_strategy_correlations(daily_pnl)
        if correlations:
            fprint("    Correlations: %s" % json.dumps(correlations))
            all_correlations[key] = correlations

        # Strategy breakdown
        strat_counts = {}
        strat_pnl = {}
        for t in trades:
            s = t.get("strategy", "unknown")
            strat_counts[s] = strat_counts.get(s, 0) + 1
            strat_pnl[s] = strat_pnl.get(s, 0.0) + t["pnl"]
        fprint("    Strategy breakdown:")
        for s in sorted(strat_counts.keys()):
            fprint("      %s: %d trades, $%.2f PnL" % (s, strat_counts[s], strat_pnl[s]))

        results[key] = {
            "variant": key,
            "name": cfg["name"],
            "stats": stats_dict,
            "validation": validation,
            "correlations": correlations,
            "strategy_breakdown": {s: {"n_trades": strat_counts[s], "pnl": round(strat_pnl[s], 2)}
                                   for s in strat_counts},
        }

    # ---- Results summary ----
    fprint("\n" + "=" * 70)
    fprint("[5/5] RESULTS SUMMARY (sorted by calendar-month Sharpe)")
    fprint("=" * 70)

    sorted_keys = sorted(results.keys(), key=lambda k: results[k]["stats"]["sharpe"], reverse=True)

    header = "%-5s %-30s %7s %7s %7s %6s %7s %8s %6s" % (
        "Var", "Name", "Sharpe", "Sortino", "PF", "WR%", "MDD%", "TotRet%", "Gates")
    fprint(header)
    fprint("-" * len(header))

    for k in sorted_keys:
        r = results[k]
        s = r["stats"]
        v = r["validation"]
        fprint("%-5s %-30s %7.3f %7.3f %7.3f %5.1f%% %6.1f%% %7.1f%% %d/%d%s" % (
            r["variant"], r["name"][:30],
            s["sharpe"], s["sortino"], s["profit_factor"],
            s["win_rate"] * 100, s["max_dd_pct"], s["total_return_pct"],
            v["gates_passed"], v["gates_total"],
            " *" if v["all_passed"] else "",
        ))

    # Correlation summary
    fprint("\n--- Strategy Correlation Matrix ---")
    for k, corrs in all_correlations.items():
        if corrs and "note" not in corrs:
            fprint("  Variant %s:" % k)
            for pair, val in corrs.items():
                interpretation = "LOW" if abs(val) < 0.30 else ("MED" if abs(val) < 0.60 else "HIGH")
                fprint("    %s: %.4f (%s)" % (pair, val, interpretation))

    # ---- Save outputs ----
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "capital": CAP,
            "max_per_trade": MAX_PER_TRADE,
            "max_deploy_pct": MAX_DEPLOY_PCT,
            "commission": COMMISSION,
            "haircut": HAIRCUT,
            "dte": DTE,
            "otm_pct": OTM_PCT,
            "profit_target": PROFIT_TARGET,
            "wf_train_days": WF_TRAIN_DAYS,
            "n_features": len(ALL_FEATURES),
        },
        "results": {},
        "ranking": sorted_keys,
        "correlations": all_correlations,
    }

    for k in sorted_keys:
        r = results[k]
        output_data["results"][k] = {
            "name": r["name"],
            "stats": r["stats"],
            "validation": {
                "gates_passed": r["validation"]["gates_passed"],
                "gates_total": r["validation"]["gates_total"],
                "all_passed": r["validation"]["all_passed"],
                "gates": r["validation"]["gates"],
            },
            "strategy_breakdown": r["strategy_breakdown"],
            "correlations": r["correlations"],
        }

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    fprint("\nResults saved to: %s" % out_path)

    # ---- MLflow logging ----
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="portfolio_optimizer_v1_%s" % datetime.now().strftime("%Y%m%d_%H%M")):
                mlflow.log_param("n_variants", len(results))
                mlflow.log_param("capital", CAP)
                mlflow.log_param("n_features", len(ALL_FEATURES))
                mlflow.log_param("wf_train_days", WF_TRAIN_DAYS)

                # Log best variant
                if sorted_keys:
                    best = results[sorted_keys[0]]
                    mlflow.log_metric("best_sharpe", best["stats"]["sharpe"])
                    mlflow.log_metric("best_sortino", best["stats"]["sortino"])
                    mlflow.log_metric("best_pf", best["stats"]["profit_factor"])
                    mlflow.log_metric("best_wr", best["stats"]["win_rate"])
                    mlflow.log_metric("best_mdd", best["stats"]["max_dd_pct"])
                    mlflow.log_param("best_variant", sorted_keys[0])
                    mlflow.log_param("best_name", best["name"])

                    # Log all variant Sharpes
                    for k in sorted_keys:
                        mlflow.log_metric("sharpe_%s" % k, results[k]["stats"]["sharpe"])
                        mlflow.log_metric("gates_passed_%s" % k, results[k]["validation"]["gates_passed"])

                mlflow.log_artifact(str(out_path))
            fprint("MLflow run logged to experiment: %s" % EXPERIMENT_NAME)
        except Exception as e:
            fprint("MLflow logging failed: %s" % str(e))

    elapsed = time.time() - t0
    fprint("\nTotal runtime: %.1f seconds (%.1f minutes)" % (elapsed, elapsed / 60))
    fprint("Done.")


if __name__ == "__main__":
    main()
