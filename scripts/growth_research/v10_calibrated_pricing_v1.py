#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
V10 Calibrated Pricing V1 -- BS Pricing Correction from Real Market Data
=========================================================================

CONTEXT (KB #282): Our ATR-based IV estimation underprices options by ~73%
median. A linear calibration model found: market_mid ~ 1.10 * bs_price + 6.60.
This means our backtest Sharpes are inflated because we assume cheaper entry.

This script re-runs the V10 strategy (our best sector rotation: LGBM ranking,
bull call + bear put spreads, monthly rebalance, DTE=28, 4% OTM, 8 positions,
50% profit target) with 4 pricing variants:

  A: Original 15% haircut (current BS baseline -- what we reported)
  B: 90% haircut (calibrated from real data -- concurrent entry 1187)
  C: 73% haircut (median BS-to-real gap from KB #282)
  D: Linear correction: real_cost_ps = 1.10 * bs_cost_ps + 0.066

All variants use identical LGBM rankings, same trade signals, same exits.
Only the ENTRY COST changes. This isolates the pricing impact on performance.

5-Gate adversarial validation on each variant:
  1. Sign-flip permutation (1000 trials)
  2. Regime balance (bull vs bear WR gap < 0.50)
  3. Sub-period (both halves profitable)
  4. Outlier removal (trim 5% extremes)
  5. Yearly consistency (>=60% years profitable)

Output: output/growth_research/v10_calibrated_pricing_v1/
MLflow: v10_calibrated_pricing_v1
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

# Detect environment
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")

if _NEPTUNE_BASE.exists():
    BASE = _NEPTUNE_BASE
    fprint("Running on Neptune: %s" % BASE)
else:
    BASE = _JUPITER_BASE
    fprint("Running on Jupiter: %s" % BASE)

OUTPUT_DIR = BASE / "output" / "growth_research" / "v10_calibrated_pricing_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHAINS_DIR = BASE / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 28
MAX_POS_DOLLARS = 200.0
COMMISSION_RT_SPREAD = 2.60
EARLY_EXIT_COMMISSION = 2.60
PROFIT_TARGET_PCT = 0.50
COST_WIDTH_MAX = 0.50
DTE_TOLERANCE = 5
STRIKE_TOLERANCE = 0.03
RISK_FREE_RATE = 0.045
REGIME_BULL_THRESHOLD = 0.4
WF_TRAIN_PERIODS = 500  # 500d sliding window for LGBM (daily)

# V10 structural params
TOP_K = 4
OTM_PCT = 0.04
WIDTH_FLOOR_DOLLARS = 3.0
WIDTH_FLOOR_PCT = 0.03

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v10_calibrated_pricing_v1"
N_BOOTSTRAP = 1000

V6_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high",
    "mom_accel", "pct_pos_months_12m", "sortino_63d", "calmar_1y",
    "up_capture", "trend_r2_63d",
]
assert len(V6_FEATURES) == 17

# MLflow connection
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
    # (name, description, haircut, use_linear_correction)
    ("A_original_15pct", "A: Original 15% haircut (current BS baseline)", 0.15, False),
    ("B_calibrated_90pct", "B: 90% haircut (calibrated from real data)", 0.90, False),
    ("C_median_73pct", "C: 73% haircut (median BS-to-real gap)", 0.73, False),
    ("D_linear_correction", "D: Linear correction (1.10*BS + $0.066/sh)", 0.00, True),
]


# ==============================================================
# SELF-CONTAINED BLACK-SCHOLES PRICING
# ==============================================================

def _bs_call(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
    """Black-Scholes European call price per share."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_put(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
    """Black-Scholes European put price per share."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _estimate_iv(atr, spot, vix=20.0, atr_period=14):
    """ATR-based IV estimation: realized_vol * iv_multiplier."""
    if spot <= 0 or atr <= 0:
        return 0.25
    realized_vol = (atr / spot) * np.sqrt(252.0 / atr_period)
    iv_mult = 1.2 + 0.01 * max(vix - 20.0, 0.0)
    sigma = realized_vol * iv_mult
    return max(sigma, 0.10)


def _price_spread_bs(S, K1, K2, dte, atr, vix, direction):
    """Price a spread using BS. Returns (fair_value_ps, sigma)."""
    T = dte / 365.0
    sigma = _estimate_iv(atr, S, vix)
    if direction == "bull":
        fv = _bs_call(S, K1, T, sigma=sigma) - _bs_call(S, K2, T, sigma=sigma)
    else:
        fv = _bs_put(S, K2, T, sigma=sigma) - _bs_put(S, K1, T, sigma=sigma)
    return max(fv, 0.001), sigma


def apply_pricing_variant(bs_fair_value_ps, haircut, use_linear_correction):
    """Apply pricing variant to get entry cost per share.

    For variants A/B/C: entry_cost_ps = bs_fair_value_ps * (1 + haircut)
    For variant D: entry_cost_ps = 1.10 * bs_fair_value_ps + 0.066
    """
    if use_linear_correction:
        return 1.10 * bs_fair_value_ps + 0.066
    else:
        return bs_fair_value_ps * (1.0 + haircut)


def revalue_spread_bs(S, K1, K2, dte_remaining, atr, vix, direction):
    """Revalue a spread at a given point. Used for profit target checks.
    EXIT revaluation always uses the same 15% haircut (conservative)."""
    if dte_remaining <= 0:
        if direction == "bull":
            return max(S - K1, 0.0) - max(S - K2, 0.0)
        else:
            return max(K2 - S, 0.0) - max(K1 - S, 0.0)
    fv, _ = _price_spread_bs(S, K1, K2, dte_remaining, atr, vix, direction)
    # Exit value: haircut DOWN (you sell the spread, so you get less)
    return fv * (1.0 - 0.15)


# ==============================================================
# CHAIN DATA
# ==============================================================

def load_all_chains():
    """Load real options chain data if available."""
    chains = {}
    if not CHAINS_DIR.exists():
        fprint("  No chains directory found -- will use BS pricing only")
        return chains
    for tk in SECTORS:
        path = CHAINS_DIR / ("%s.parquet" % tk)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df["expiration"] = pd.to_datetime(df["expiration"])
        for c in ["strike", "bid", "ask", "mid", "vol", "delta"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["dte"] = (df["expiration"] - df["date"]).dt.days
        chains[tk] = df
        fprint("  %s: %s rows" % (tk, "{:,}".format(len(df))))
    return chains


def find_chain_spread_price(chain_df, trade_date, direction, K1, K2, dte_target):
    """Find real market spread price from chain data."""
    if chain_df is None:
        return None
    day_mask = chain_df["date"] == pd.Timestamp(trade_date)
    chain_day = chain_df[day_mask]
    if chain_day.empty:
        nearby = chain_df[
            (chain_df["date"] >= pd.Timestamp(trade_date) - pd.Timedelta(days=2)) &
            (chain_df["date"] <= pd.Timestamp(trade_date) + pd.Timedelta(days=2))
        ]
        if nearby.empty:
            return None
        nearest_date = min(
            nearby["date"].unique(),
            key=lambda x: abs((x - pd.Timestamp(trade_date)).days)
        )
        chain_day = chain_df[chain_df["date"] == nearest_date]

    exps = chain_day[["expiration", "dte"]].drop_duplicates()
    exps["dte_dist"] = (exps["dte"] - dte_target).abs()
    valid_exps = exps[exps["dte_dist"] <= DTE_TOLERANCE]
    if valid_exps.empty:
        return None
    best_exp_row = valid_exps.loc[valid_exps["dte_dist"].idxmin()]
    chain_exp = chain_day[chain_day["expiration"] == best_exp_row["expiration"]]

    opt_type = "c" if direction == "bull" else "p"
    near_target = K1 if direction == "bull" else K2
    far_target = K2 if direction == "bull" else K1

    near_opts = chain_exp[chain_exp["type"] == opt_type].copy()
    if near_opts.empty:
        return None
    near_opts["dist"] = (near_opts["strike"] - near_target).abs()
    near_leg = near_opts.sort_values("dist").iloc[0]
    if near_leg["dist"] / max(near_target, 1) > STRIKE_TOLERANCE:
        return None

    far_opts = chain_exp[chain_exp["type"] == opt_type].copy()
    far_opts["dist"] = (far_opts["strike"] - far_target).abs()
    far_leg = far_opts.sort_values("dist").iloc[0]
    if far_leg["dist"] / max(far_target, 1) > STRIKE_TOLERANCE:
        return None

    near_mid = float(near_leg["mid"]) if not pd.isna(near_leg["mid"]) else \
        (float(near_leg["bid"]) + float(near_leg["ask"])) / 2
    far_mid = float(far_leg["mid"]) if not pd.isna(far_leg["mid"]) else \
        (float(far_leg["bid"]) + float(far_leg["ask"])) / 2

    spread_cost_mid = abs(near_mid - far_mid)
    return {
        "found": True,
        "spread_cost_mid": spread_cost_mid,
        "near_strike": float(near_leg["strike"]),
        "far_strike": float(far_leg["strike"]),
    }


# ==============================================================
# DATA + FEATURES
# ==============================================================

def download_data():
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


def load_regime_predictions():
    if not REGIME_FILE.exists():
        return None
    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
    return regime_series


def get_regime_score_at(regime_series, dt):
    if regime_series is None:
        return 0.5
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


def compute_features(px, spy_slice):
    """Compute 17 LGBM features for a single sector ETF."""
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
    f["up_capture"] = float(
        up_days.iloc[-63:].mean() / (up_days.mean() + 1e-10)
    ) if len(up_days) > 10 else 1.0
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f["trend_r2_63d"] = r_val ** 2
    else:
        f["trend_r2_63d"] = 0.0
    return f


def compute_cross_asset_features(sector_ticker, dt_idx, close_df):
    """Not used in V6_FEATURES list but kept for compatibility."""
    return {}


def compute_atr_series(high, low, close, period=14):
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


# ==============================================================
# LGBM WALK-FORWARD RANKING (500d sliding window)
# ==============================================================

def get_monthly_rebalance_dates(close):
    """Monthly rebalance -- first trading day of each month."""
    monthly = close.index.to_series().resample("MS").first().dropna()
    return pd.DatetimeIndex(monthly.values)


def get_weekly_fridays(close):
    """Every Friday -- for building training features."""
    return pd.DatetimeIndex(
        close.index.to_series().resample("W-FRI").last().dropna().values
    )


def build_feature_records(close, high, low, rebal_dates, regime_series):
    """Build feature records for all rebalance dates."""
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
            feats = compute_features(px, spy.iloc[:idx + 1])
            if not feats:
                continue
            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)
            rec = {**feats, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)
    df = pd.DataFrame(records)
    for c in V6_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[V6_FEATURES] = df[V6_FEATURES].fillna(0.0)
    fprint("    %d records, %d dates" % (len(df), len(df["date"].unique())))
    return df


def walk_forward_lgbm_rank(df):
    """Walk-forward LGBM ranking with 500-day sliding window."""
    import lightgbm as lgb
    if len(df) < 100:
        return {}
    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())
    # Convert 500 days to approximate weekly count (500/7 ~ 71 weeks)
    wf_periods = max(50, WF_TRAIN_PERIODS // 7)
    rankings = {}
    for i in range(wf_periods, len(dates)):
        train_dates = dates[max(0, i - wf_periods):i]
        test_date = dates[i]
        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()
        if len(test_df) < 3 or len(train_df) < 50:
            continue
        Xt = np.nan_to_num(train_df[V6_FEATURES].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[V6_FEATURES].values.astype(np.float32))
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
# PARAMETERIZED STRIKES
# ==============================================================

def compute_strikes(S, direction):
    """Compute V10 strikes: 4% OTM, max($3,3%) width."""
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

def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity,
                  chains, haircut, use_linear_correction):
    """Execute a single spread trade with variant-specific pricing."""
    if tk not in close.columns or tk not in atr_dict:
        return None
    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    av = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015
    K1, K2 = compute_strikes(S, direction)

    # Try real chain data first
    used_real = False
    entry_cost_ps = None
    chain_df = chains.get(tk)
    if chain_df is not None:
        result = find_chain_spread_price(chain_df, dt, direction, K1, K2, DTE)
        if result and result["found"]:
            entry_cost_ps = result["spread_cost_mid"]
            used_real = True
            if direction == "bull":
                K1 = result["near_strike"]
                K2 = result["far_strike"]
            else:
                K1 = result["far_strike"]
                K2 = result["near_strike"]

    # Fallback: BS pricing with variant-specific correction
    if entry_cost_ps is None:
        bs_fair, sigma = _price_spread_bs(S, K1, K2, DTE, av, vix_val, direction)
        entry_cost_ps = apply_pricing_variant(bs_fair, haircut, use_linear_correction)

    if entry_cost_ps is None or entry_cost_ps <= 0:
        return None

    spread_width = abs(K2 - K1)
    cwr = entry_cost_ps / max(spread_width, 0.01)
    if cwr > COST_WIDTH_MAX:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
    max_pos = min(MAX_POS_DOLLARS, equity * 0.40)
    if total_cost <= 0 or total_cost > max_pos:
        return None

    max_profit_ps = spread_width - entry_cost_ps
    if max_profit_ps <= 0:
        return None

    # ---- 50% Profit Target Exit Logic ----
    exited_early = False
    exit_day_idx = ei
    exit_reason = "expiry"
    hold_days = DTE

    for check_idx in range(di + 1, ei + 1):
        if check_idx >= len(close):
            break
        check_date = close.index[check_idx]
        dte_remaining = ei - check_idx
        days_held = check_idx - di

        S_now = float(close[tk].iloc[check_idx])
        av_now = float(atr_dict[tk].loc[check_date]) if check_date in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[check_date]) else S_now * 0.015

        current_value_ps = revalue_spread_bs(S_now, K1, K2, dte_remaining, av_now, vix_val, direction)
        unrealized_gain_ps = current_value_ps - entry_cost_ps

        if unrealized_gain_ps >= PROFIT_TARGET_PCT * max_profit_ps:
            exited_early = True
            exit_day_idx = check_idx
            exit_reason = "profit_target_50pct"
            hold_days = days_held
            break

    # Compute final P&L
    Se = float(close[tk].iloc[exit_day_idx])

    if exited_early:
        dte_at_exit = ei - exit_day_idx
        av_exit = float(atr_dict[tk].loc[close.index[exit_day_idx]]) if close.index[exit_day_idx] in atr_dict[tk].index else Se * 0.015
        exit_value_ps = revalue_spread_bs(Se, K1, K2, dte_at_exit, av_exit, vix_val, direction)
        pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD - EARLY_EXIT_COMMISSION
    else:
        if direction == "bull":
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
        else:
            intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
        exit_value_ps = intrinsic

    return {
        "pnl": round(pnl, 2),
        "entry_cost_ps": round(entry_cost_ps, 4),
        "total_cost": round(total_cost, 2),
        "used_real_pricing": used_real,
        "K1": K1, "K2": K2,
        "S_entry": round(S, 2), "S_exit": round(Se, 2),
        "exit_value_ps": round(exit_value_ps, 4) if exit_value_ps is not None else 0.0,
        "spread_width": round(spread_width, 2),
        "cost_width_ratio": round(cwr, 3),
        "max_profit_ps": round(max_profit_ps, 4),
        "exited_early": exited_early,
        "exit_reason": exit_reason,
        "hold_days": hold_days,
    }


# ==============================================================
# SIMULATION ENGINE
# ==============================================================

def simulate_variant(rankings, close, atr_dict, chains, rebal_dates,
                     haircut, use_linear_correction, variant_name=""):
    """Run V10 backtest for a pricing variant."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    real_count = 0
    bs_count = 0

    for dt in sorted(rebal_dates):
        if dt not in spy.index:
            continue

        ranking_date = None
        for rd in sorted(rankings.keys()):
            if rd <= dt:
                ranking_date = rd
        if ranking_date is None:
            continue

        scores = rankings[ranking_date]
        if not scores or len(scores) < (TOP_K * 2):
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]]

        n_positions = len(bull_picks) + len(bear_picks)

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                result = execute_trade(
                    tk, dt, direction, close, atr_dict, cv, equity,
                    chains, haircut, use_linear_correction,
                )
                if result is not None:
                    equity += result["pnl"]
                    real_count += 1 if result["used_real_pricing"] else 0
                    bs_count += 0 if result["used_real_pricing"] else 1
                    di = close.index.get_loc(dt)
                    ei = min(di + DTE, len(close) - 1)
                    sv = float(spy.loc[dt])
                    se = float(spy.iloc[ei]) if ei < len(spy) else sv
                    trades.append({
                        **result,
                        "entry_date": str(dt.date()),
                        "exit_date": str(close.index[di + result["hold_days"]].date()) if di + result["hold_days"] < len(close) else str(close.index[ei].date()),
                        "ticker": tk,
                        "regime": "bull" if se >= sv else "bear",
                        "direction": direction,
                        "vix": round(cv, 1),
                        "win": result["pnl"] > 0,
                        "n_positions": n_positions,
                    })

    return trades, equity, real_count, bs_count


# ==============================================================
# METRICS
# ==============================================================

def compute_full_stats(trades, initial_capital=CAP):
    """Compute comprehensive risk-adjusted metrics."""
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

    # Calendar month Sharpe (primary metric)
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
    if len(neg_ret) > 1:
        sortino = float(monthly_ret.mean() / (neg_ret.std() + 1e-10) * np.sqrt(12))
    else:
        sortino = 0.0

    # CAGR
    dates_str = [t["entry_date"] for t in trades]
    first_date = pd.Timestamp(min(dates_str))
    last_date = pd.Timestamp(max(dates_str))
    years = max((last_date - first_date).days / 365.25, 0.5)
    total_ret = equity[-1] / equity[0]
    cagr = total_ret ** (1 / years) - 1

    # MDD
    max_dd_pct = float(dd.min()) * 100

    # Calmar
    calmar = cagr / (abs(dd.min()) + 1e-10) if abs(dd.min()) > 1e-10 else 0.0

    early_exits = sum(1 for t in trades if t.get("exited_early", False))
    hold_days_list = [t.get("hold_days", DTE) for t in trades]
    avg_entry_cost = float(np.mean([t.get("entry_cost_ps", 0) for t in trades]))

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
        "years": round(years, 1),
    }


# ==============================================================
# 5-GATE ADVERSARIAL VALIDATION
# ==============================================================

def gate_sign_flip(trades, n_perms=1000):
    """Gate 1: Sign-flip permutation test. Real Sharpe must beat >95% of shuffled."""
    if len(trades) < 10:
        return False, 0.0, 0.0
    pnls = np.array([t["pnl"] for t in trades])
    real_sharpe = pnls.mean() / (pnls.std() + 1e-10) * np.sqrt(len(pnls))
    rng = np.random.RandomState(42)
    count_beat = 0
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(pnls))
        shuffled = pnls * signs
        null_sharpe = shuffled.mean() / (shuffled.std() + 1e-10) * np.sqrt(len(shuffled))
        if real_sharpe > null_sharpe:
            count_beat += 1
    p_value = 1.0 - count_beat / n_perms
    return p_value < 0.05, p_value, real_sharpe


def gate_regime_balance(trades):
    """Gate 2: Bull vs bear regime WR gap must be < 0.50."""
    bull_trades = [t for t in trades if t.get("regime") == "bull"]
    bear_trades = [t for t in trades if t.get("regime") == "bear"]
    if len(bull_trades) < 5 or len(bear_trades) < 5:
        return True, 0.0  # insufficient data, pass by default
    bull_wr = np.mean([t["pnl"] > 0 for t in bull_trades])
    bear_wr = np.mean([t["pnl"] > 0 for t in bear_trades])
    gap = abs(bull_wr - bear_wr)
    return gap < 0.50, gap


def gate_sub_period(trades):
    """Gate 3: Both halves must be profitable."""
    if len(trades) < 10:
        return False, 0.0, 0.0
    mid = len(trades) // 2
    first_half = sum(t["pnl"] for t in trades[:mid])
    second_half = sum(t["pnl"] for t in trades[mid:])
    return first_half > 0 and second_half > 0, first_half, second_half


def gate_outlier_removal(trades):
    """Gate 4: Strategy still profitable after trimming 5% extreme trades."""
    if len(trades) < 20:
        return True, 0.0
    pnls = sorted([t["pnl"] for t in trades])
    trim = max(1, int(len(pnls) * 0.05))
    trimmed = pnls[trim:-trim]
    trimmed_pnl = sum(trimmed)
    return trimmed_pnl > 0, trimmed_pnl


def gate_yearly_consistency(trades):
    """Gate 5: At least 60% of years must be profitable."""
    if len(trades) < 10:
        return False, 0.0
    trade_df = pd.DataFrame(trades)
    trade_df["year"] = pd.to_datetime(trade_df["entry_date"]).dt.year
    yearly_pnl = trade_df.groupby("year")["pnl"].sum()
    if len(yearly_pnl) < 2:
        return False, 0.0
    pct_profitable = (yearly_pnl > 0).mean()
    return pct_profitable >= 0.60, float(pct_profitable)


def run_5_gate_validation(trades, variant_name):
    """Run all 5 adversarial gates. Returns (gates_passed, gates_total, details)."""
    gates = []

    # Gate 1: Sign-flip
    passed, p_val, real_sh = gate_sign_flip(trades)
    gates.append({"name": "sign_flip", "passed": passed, "p_value": round(p_val, 4), "real_sharpe": round(real_sh, 3)})

    # Gate 2: Regime balance
    passed, gap = gate_regime_balance(trades)
    gates.append({"name": "regime_balance", "passed": passed, "wr_gap": round(gap, 3)})

    # Gate 3: Sub-period
    passed, h1, h2 = gate_sub_period(trades)
    gates.append({"name": "sub_period", "passed": passed, "half1_pnl": round(h1, 2), "half2_pnl": round(h2, 2)})

    # Gate 4: Outlier removal
    passed, trimmed = gate_outlier_removal(trades)
    gates.append({"name": "outlier_removal", "passed": passed, "trimmed_pnl": round(trimmed, 2)})

    # Gate 5: Yearly consistency
    passed, pct = gate_yearly_consistency(trades)
    gates.append({"name": "yearly_consistency", "passed": passed, "pct_profitable_years": round(pct, 3)})

    n_passed = sum(1 for g in gates if g["passed"])
    return n_passed, len(gates), gates


# ==============================================================
# MONTE CARLO BOOTSTRAP
# ==============================================================

def monte_carlo_bootstrap(trades, n_resamples=N_BOOTSTRAP, initial_capital=CAP):
    """Monte Carlo bootstrap resampling of trade-level P&L."""
    if not trades or len(trades) < 10:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    n_trades = len(pnls)
    rng = np.random.RandomState(42)

    boot_sharpe = np.zeros(n_resamples)
    boot_pf = np.zeros(n_resamples)
    boot_wr = np.zeros(n_resamples)
    boot_total = np.zeros(n_resamples)

    for b in range(n_resamples):
        idx = rng.choice(n_trades, size=n_trades, replace=True)
        sp = pnls[idx]
        boot_wr[b] = float(np.mean(sp > 0))
        gw = float(sp[sp > 0].sum()) if (sp > 0).any() else 0
        gl = float(abs(sp[sp < 0].sum())) if (sp < 0).any() else 1e-10
        boot_pf[b] = gw / max(gl, 1e-10)
        boot_total[b] = float(sp.sum())
        # Approximate monthly Sharpe
        chunk = max(1, n_trades // max(3, n_trades // 6))
        monthly = [sp[j:j+chunk].sum() for j in range(0, n_trades, chunk)]
        monthly = np.array(monthly)
        if len(monthly) > 1 and monthly.std() > 1e-10:
            boot_sharpe[b] = monthly.mean() / monthly.std() * np.sqrt(12)

    return {
        "sharpe_mean": round(float(boot_sharpe.mean()), 3),
        "sharpe_ci5": round(float(np.percentile(boot_sharpe, 5)), 3),
        "sharpe_ci95": round(float(np.percentile(boot_sharpe, 95)), 3),
        "sharpe_pct_pos": round(float(np.mean(boot_sharpe > 0) * 100), 1),
        "pf_mean": round(float(boot_pf.mean()), 3),
        "pf_ci5": round(float(np.percentile(boot_pf, 5)), 3),
        "pf_ci95": round(float(np.percentile(boot_pf, 95)), 3),
        "wr_mean": round(float(boot_wr.mean()), 4),
        "total_pnl_mean": round(float(boot_total.mean()), 2),
        "total_pnl_ci5": round(float(np.percentile(boot_total, 5)), 2),
        "total_pnl_ci95": round(float(np.percentile(boot_total, 95)), 2),
        "pct_profitable": round(float(np.mean(boot_total > 0) * 100), 1),
    }


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint("V10 CALIBRATED PRICING V1 -- %s" % t0.strftime("%Y-%m-%d %H:%M:%S"))
    fprint("=" * 100)
    fprint("")
    fprint("CONTEXT: BS pricing underprices options by ~73%% median (KB #282).")
    fprint("This re-runs V10 with corrected pricing to see how much Sharpe inflation exists.")
    fprint("")
    fprint("V10 config: %d sectors, LGBM 17 features, DTE=%d, %.0f%% OTM, 8 positions" % (
        len(SECTORS), DTE, OTM_PCT * 100))
    fprint("Capital: $%.0f | Max per trade: $%.0f | Commission: $%.2f RT" % (
        CAP, MAX_POS_DOLLARS, COMMISSION_RT_SPREAD))
    fprint("Profit target: %.0f%% | Walk-forward: %dd sliding" % (
        PROFIT_TARGET_PCT * 100, WF_TRAIN_PERIODS))
    fprint("")
    fprint("PRICING VARIANTS:")
    for name, desc, hc, lc in VARIANTS:
        fprint("  %s" % desc)
    fprint("")

    # Load data
    fprint("Loading chain data...")
    chains = load_all_chains()

    fprint("")
    fprint("Downloading price data...")
    close, high, low = download_data()
    regime_series = load_regime_predictions()
    atr_dict = compute_atr_series(high, low, close)

    monthly_dates = get_monthly_rebalance_dates(close)
    weekly_dates = get_weekly_fridays(close)

    fprint("Rebalance: %d monthly dates | Feature dates: %d weekly" % (
        len(monthly_dates), len(weekly_dates)))

    # Build LGBM rankings (shared across all variants)
    fprint("")
    fprint("=" * 80)
    fprint("BUILDING LGBM RANKINGS (500d sliding, 17 features)")
    fprint("=" * 80)
    records = build_feature_records(close, high, low, weekly_dates, regime_series)
    rankings = walk_forward_lgbm_rank(records)

    if len(rankings) < 20:
        fprint("ERROR: Only %d ranking dates -- insufficient data" % len(rankings))
        return

    spy_close = close["SPY"]

    # Run each variant
    all_trades = {}
    all_stats = {}
    all_gates = {}
    all_mc = {}

    for var_name, description, haircut, use_lc in VARIANTS:
        fprint("")
        fprint("=" * 90)
        fprint(description)
        fprint("  haircut=%.0f%%, linear_correction=%s" % (haircut * 100, use_lc))
        fprint("=" * 90)
        t_start = time.time()

        trades, eq, real, bs = simulate_variant(
            rankings, close, atr_dict, chains, monthly_dates,
            haircut=haircut, use_linear_correction=use_lc,
            variant_name=var_name,
        )
        elapsed = time.time() - t_start
        fprint("  Trades: %d | Real: %d | BS: %d | Final: $%s (%.1fs)" % (
            len(trades), real, bs, "{:,.0f}".format(eq), elapsed))

        all_trades[var_name] = trades
        s = compute_full_stats(trades)
        all_stats[var_name] = s

        if s:
            fprint("  Sharpe: %.2f | Sortino: %.2f | Calmar: %.2f" % (
                s["sharpe"], s["sortino"], s["calmar"]))
            fprint("  WR: %.1f%% | PF: %.2f | MDD: %.1f%% | CAGR: %.1f%%" % (
                s["win_rate"] * 100, s["profit_factor"], s["max_dd_pct"], s["cagr_pct"]))
            fprint("  Total return: %.1f%% | Final equity: $%s" % (
                s["total_return_pct"], "{:,.2f}".format(s["final_equity"])))
            fprint("  Avg entry cost/sh: $%.4f | Avg hold: %.1f days" % (
                s["avg_entry_cost_ps"], s["avg_hold_days"]))

            # Side breakdown
            for side in ["bull", "bear"]:
                st = [t for t in trades if t["direction"] == side]
                if st:
                    pnls = [t["pnl"] for t in st]
                    wr = sum(1 for p in pnls if p > 0) / len(pnls)
                    fprint("    %s: %d trades, WR %.1f%%, PnL $%s" % (
                        side.upper(), len(st), wr * 100, "{:,.0f}".format(sum(pnls))))

        # 5-gate validation
        if len(trades) >= 10:
            n_passed, n_total, gate_details = run_5_gate_validation(trades, var_name)
            all_gates[var_name] = {"passed": n_passed, "total": n_total, "gates": gate_details}
            status = "PASS" if n_passed == n_total else "PARTIAL" if n_passed >= 3 else "FAIL"
            fprint("  5-GATE VALIDATION: [%s] %d/%d" % (status, n_passed, n_total))
            for g in gate_details:
                gs = "PASS" if g["passed"] else "FAIL"
                detail_parts = ["%s=%s" % (k, v) for k, v in g.items() if k not in ("name", "passed")]
                fprint("    [%s] %s: %s" % (gs, g["name"], ", ".join(detail_parts)))

        # Monte Carlo
        fprint("  Monte Carlo bootstrap (%d resamples)..." % N_BOOTSTRAP)
        mc = monte_carlo_bootstrap(trades)
        if mc:
            all_mc[var_name] = mc
            fprint("    Sharpe: %.2f [%.2f, %.2f] (%.0f%% positive)" % (
                mc["sharpe_mean"], mc["sharpe_ci5"], mc["sharpe_ci95"], mc["sharpe_pct_pos"]))
            fprint("    PF: %.2f [%.2f, %.2f] | WR: %.1f%%" % (
                mc["pf_mean"], mc["pf_ci5"], mc["pf_ci95"], mc["wr_mean"] * 100))
            fprint("    Total PnL: $%s [$%s, $%s] (%.0f%% profitable)" % (
                "{:,.0f}".format(mc["total_pnl_mean"]),
                "{:,.0f}".format(mc["total_pnl_ci5"]),
                "{:,.0f}".format(mc["total_pnl_ci95"]),
                mc["pct_profitable"]))

    # ==============================================================
    # COMPARISON TABLE
    # ==============================================================
    fprint("")
    fprint("=" * 140)
    fprint("COMPARISON -- ALL PRICING VARIANTS (sorted by Sharpe)")
    fprint("=" * 140)

    hdr = "  %-25s %5s %7s %7s %7s %6s %6s %7s %7s %7s %9s %5s" % (
        "Variant", "N", "Sharpe", "Sort", "Calmar", "WR", "PF", "MDD", "CAGR", "AvgCost", "Final$", "Gate")
    fprint(hdr)
    fprint("  " + "-" * 130)

    sorted_variants = sorted(
        [(vn, all_stats.get(vn)) for vn, _, _, _ in VARIANTS if all_stats.get(vn)],
        key=lambda x: x[1]["sharpe"], reverse=True
    )

    for var_name, s in sorted_variants:
        g = all_gates.get(var_name)
        gates_str = "%d/%d" % (g["passed"], g["total"]) if g else "--"
        fprint("  %-25s %5d %7.2f %7.2f %7.2f %5.1f%% %5.2f %6.1f%% %6.1f%% $%5.4f $%8s %5s" % (
            var_name, s["n_trades"], s["sharpe"], s["sortino"], s["calmar"],
            s["win_rate"] * 100, s["profit_factor"], s["max_dd_pct"],
            s["cagr_pct"], s["avg_entry_cost_ps"],
            "{:,.0f}".format(s["final_equity"]), gates_str))

    # ==============================================================
    # DELTA TABLE (vs original 15% haircut)
    # ==============================================================
    fprint("")
    fprint("=" * 100)
    fprint("IMPACT OF PRICING CORRECTION (delta vs A_original_15pct)")
    fprint("=" * 100)

    base = all_stats.get("A_original_15pct")
    if base:
        fprint("")
        fprint("  %-25s %8s %8s %7s %7s %9s %9s" % (
            "Variant", "dSharpe", "dSort", "dWR", "dPF", "dPnL", "dCost/sh"))
        fprint("  " + "-" * 80)
        for var_name, _, _, _ in VARIANTS[1:]:
            s = all_stats.get(var_name)
            if s is None:
                continue
            fprint("  %-25s %+7.2f %+7.2f %+6.1f%% %+6.2f $%+8s $%+.4f" % (
                var_name,
                s["sharpe"] - base["sharpe"],
                s["sortino"] - base["sortino"],
                (s["win_rate"] - base["win_rate"]) * 100,
                s["profit_factor"] - base["profit_factor"],
                "{:,.0f}".format(s["total_pnl"] - base["total_pnl"]),
                s["avg_entry_cost_ps"] - base["avg_entry_cost_ps"]))

    # ==============================================================
    # HONEST ASSESSMENT
    # ==============================================================
    fprint("")
    fprint("=" * 100)
    fprint("HONEST ASSESSMENT: HOW MUCH DOES BS INFLATION AFFECT REPORTED NUMBERS?")
    fprint("=" * 100)
    fprint("")

    if base:
        b_stats = all_stats.get("B_calibrated_90pct")
        c_stats = all_stats.get("C_median_73pct")
        d_stats = all_stats.get("D_linear_correction")

        fprint("  ORIGINAL (15%% haircut):   Sharpe %.2f, Sortino %.2f, PF %.2f, WR %.1f%%" % (
            base["sharpe"], base["sortino"], base["profit_factor"], base["win_rate"] * 100))

        if b_stats:
            sharpe_drop = (1 - b_stats["sharpe"] / max(base["sharpe"], 0.01)) * 100 if base["sharpe"] > 0 else 0
            fprint("  CALIBRATED (90%% haircut): Sharpe %.2f, Sortino %.2f, PF %.2f, WR %.1f%%  (Sharpe drop: %.0f%%)" % (
                b_stats["sharpe"], b_stats["sortino"], b_stats["profit_factor"],
                b_stats["win_rate"] * 100, sharpe_drop))

        if c_stats:
            sharpe_drop = (1 - c_stats["sharpe"] / max(base["sharpe"], 0.01)) * 100 if base["sharpe"] > 0 else 0
            fprint("  MEDIAN GAP (73%% haircut): Sharpe %.2f, Sortino %.2f, PF %.2f, WR %.1f%%  (Sharpe drop: %.0f%%)" % (
                c_stats["sharpe"], c_stats["sortino"], c_stats["profit_factor"],
                c_stats["win_rate"] * 100, sharpe_drop))

        if d_stats:
            sharpe_drop = (1 - d_stats["sharpe"] / max(base["sharpe"], 0.01)) * 100 if base["sharpe"] > 0 else 0
            fprint("  LINEAR CORR (1.10x+$0.066):Sharpe %.2f, Sortino %.2f, PF %.2f, WR %.1f%%  (Sharpe drop: %.0f%%)" % (
                d_stats["sharpe"], d_stats["sortino"], d_stats["profit_factor"],
                d_stats["win_rate"] * 100, sharpe_drop))

        fprint("")
        # Average Sharpe drop across corrected variants
        corrected = [all_stats.get(vn) for vn, _, _, _ in VARIANTS[1:] if all_stats.get(vn)]
        if corrected and base["sharpe"] > 0:
            avg_corrected_sharpe = np.mean([s["sharpe"] for s in corrected])
            avg_drop = (1 - avg_corrected_sharpe / base["sharpe"]) * 100
            fprint("  AVERAGE Sharpe across corrected variants: %.2f (%.0f%% lower than reported)" % (
                avg_corrected_sharpe, avg_drop))

            if avg_corrected_sharpe > 1.0:
                fprint("  VERDICT: Strategy SURVIVES pricing correction. Sharpe drops but remains above 1.0.")
                fprint("           The edge is real but SMALLER than reported. Adjust expectations accordingly.")
            elif avg_corrected_sharpe > 0.5:
                fprint("  VERDICT: Strategy MARGINAL after pricing correction. Sharpe between 0.5-1.0.")
                fprint("           Edge exists but may not justify deployment costs. Needs optimization.")
            elif avg_corrected_sharpe > 0:
                fprint("  VERDICT: Strategy WEAK after pricing correction. Sharpe below 0.5.")
                fprint("           Reported numbers were significantly inflated by BS underpricing.")
            else:
                fprint("  VERDICT: Strategy DESTROYED by pricing correction. Negative Sharpe.")
                fprint("           The entire reported edge was BS pricing artifact.")
        fprint("")

        # Cost comparison
        fprint("  ENTRY COST IMPACT:")
        fprint("    Original avg entry: $%.4f/sh" % base["avg_entry_cost_ps"])
        for vn, _, _, _ in VARIANTS[1:]:
            s = all_stats.get(vn)
            if s:
                pct_higher = (s["avg_entry_cost_ps"] / max(base["avg_entry_cost_ps"], 0.001) - 1) * 100
                fprint("    %s avg entry: $%.4f/sh (%.0f%% higher)" % (vn, s["avg_entry_cost_ps"], pct_higher))

    # ==============================================================
    # SAVE RESULTS
    # ==============================================================
    save_results = {
        "timestamp": t0.isoformat(),
        "context": "BS pricing underprices options by ~73% median (KB #282). "
                   "This tests V10 with corrected pricing to quantify Sharpe inflation.",
        "config": {
            "capital": CAP, "dte": DTE, "otm_pct": OTM_PCT,
            "top_k": TOP_K, "width_floor_dollars": WIDTH_FLOOR_DOLLARS,
            "width_floor_pct": WIDTH_FLOOR_PCT,
            "commission": COMMISSION_RT_SPREAD,
            "profit_target_pct": PROFIT_TARGET_PCT,
            "rebalance": "monthly",
            "wf_days": WF_TRAIN_PERIODS,
            "n_sectors": len(SECTORS),
            "features": len(V6_FEATURES),
            "n_bootstrap": N_BOOTSTRAP,
        },
    }

    for var_name, _, _, _ in VARIANTS:
        s = all_stats.get(var_name)
        if s:
            save_results[var_name] = s
        g = all_gates.get(var_name)
        if g:
            save_results["%s_validation" % var_name] = g
        mc = all_mc.get(var_name)
        if mc:
            save_results["%s_monte_carlo" % var_name] = mc

    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint("")
    fprint("Results saved to %s" % results_file)

    # Save trade details per variant
    for var_name, _, _, _ in VARIANTS:
        trades = all_trades.get(var_name, [])
        if trades:
            tf = OUTPUT_DIR / ("trades_%s.json" % var_name)
            with open(tf, "w") as f:
                json.dump(trades, f, indent=1, default=str)

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="calibrated_pricing_%s" % t0.strftime("%Y%m%d_%H%M")):
                mlflow.log_param("capital", CAP)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("otm_pct", OTM_PCT)
                mlflow.log_param("top_k", TOP_K)
                mlflow.log_param("commission", COMMISSION_RT_SPREAD)
                mlflow.log_param("profit_target_pct", PROFIT_TARGET_PCT)
                mlflow.log_param("rebalance", "monthly")
                mlflow.log_param("wf_days", WF_TRAIN_PERIODS)
                mlflow.log_param("n_features", len(V6_FEATURES))
                mlflow.log_param("n_bootstrap", N_BOOTSTRAP)
                mlflow.log_param("n_variants", len(VARIANTS))

                for var_name, _, haircut, use_lc in VARIANTS:
                    s = all_stats.get(var_name)
                    if s is None:
                        continue
                    prefix = var_name.split("_")[0]
                    mlflow.log_metric("%s_sharpe" % prefix, s["sharpe"])
                    mlflow.log_metric("%s_sortino" % prefix, s["sortino"])
                    mlflow.log_metric("%s_calmar" % prefix, s["calmar"])
                    mlflow.log_metric("%s_wr" % prefix, s["win_rate"])
                    mlflow.log_metric("%s_pf" % prefix, s["profit_factor"])
                    mlflow.log_metric("%s_mdd" % prefix, s["max_dd_pct"])
                    mlflow.log_metric("%s_n_trades" % prefix, s["n_trades"])
                    mlflow.log_metric("%s_total_pnl" % prefix, s["total_pnl"])
                    mlflow.log_metric("%s_cagr" % prefix, s["cagr_pct"])
                    mlflow.log_metric("%s_avg_entry_cost" % prefix, s["avg_entry_cost_ps"])

                    mc = all_mc.get(var_name)
                    if mc:
                        mlflow.log_metric("%s_mc_sharpe" % prefix, mc["sharpe_mean"])
                        mlflow.log_metric("%s_mc_sharpe_ci5" % prefix, mc["sharpe_ci5"])
                        mlflow.log_metric("%s_mc_sharpe_ci95" % prefix, mc["sharpe_ci95"])

                    g = all_gates.get(var_name)
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
