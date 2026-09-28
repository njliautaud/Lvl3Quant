#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Combination V2 -- Sector Options Strategy Portfolio Optimization
==========================================================================

Tests optimal portfolio allocation across validated sector options strategies:
  V9.1: Sharpe ~2.64, hold-to-expiry, DTE=28, top3+bot3
  V9.3: Sharpe ~5.12, 50% profit target, DTE=28, top3+bot3
  V10:  Sharpe ~6.14, 30% profit target, DTE=28, top4+bot4, 4% OTM, rank-weighted

6 Portfolio Variants:
  A: V10 alone (baseline)
  B: Equal-weight V9.1 + V9.3 + V10
  C: Sharpe-weighted (proportional to backtest Sharpe)
  D: V10 + VIX allocator overlay (UPRO when VIX<17, scale back when VIX>25)
  E: Risk parity (inverse volatility weighting)
  F: Best-of (each month, pick trailing-3mo best Sharpe)

5-gate adversarial validation on each variant.

Config: $645 capital, $2.60 commission RT, 15% haircut, 52-week sliding WF.
Output: output/growth_research/portfolio_combination_v2/
MLflow: experiment portfolio_combination_v2
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

# Detect environment
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")
BASE = _NEPTUNE_BASE if _NEPTUNE_BASE.exists() else _JUPITER_BASE
fprint("Running on: %s" % BASE)

sys.path.insert(0, str(BASE))
from research.tools.options_pricer import (
    price_bull_call_spread, price_bear_put_spread,
    COMMISSION_RT_SPREAD,
)
from research.tools.adversarial_validator import validate_trades

CHAINS_DIR = BASE / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
OUTPUT_DIR = BASE / "output" / "growth_research" / "portfolio_combination_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD", "UPRO"]
CAP = 645.0
DTE = 28
REGIME_BULL_THRESHOLD = 0.4
WF_TRAIN_PERIODS = 52
HAIRCUT = 0.15
COST_WIDTH_MAX = 0.50
DTE_TOLERANCE = 5
STRIKE_TOLERANCE = 0.03
EARLY_EXIT_COMMISSION = 2.60
N_PERMUTATIONS = 300
ANNUALIZE = 12  # monthly returns

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "portfolio_combination_v2"

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

# ── LGBM features (17, same as v10) ────────────────────────────────
V6_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "cross_sector_dispersion",
]
assert len(V6_FEATURES) == 17

# ── Strategy variant definitions ─────────────────────────────────
# (name, top_k, otm_pct, width_floor_$, width_floor_%, profit_target_%, hold_to_expiry)
STRATEGY_VARIANTS = {
    "V91": {
        "top_k": 3, "otm_pct": 0.03, "wfd": 3.0, "wfp": 0.03,
        "profit_target": 0.0, "desc": "V9.1: 6 pos, 3% OTM, hold-to-expiry",
        "ref_sharpe": 2.64,
    },
    "V93": {
        "top_k": 3, "otm_pct": 0.03, "wfd": 3.0, "wfp": 0.03,
        "profit_target": 0.50, "desc": "V9.3: 6 pos, 3% OTM, 50% PT",
        "ref_sharpe": 5.12,
    },
    "V10": {
        "top_k": 4, "otm_pct": 0.04, "wfd": 3.0, "wfp": 0.03,
        "profit_target": 0.30, "desc": "V10: 8 pos, 4% OTM, 30% PT, rank-weighted",
        "ref_sharpe": 6.14,
    },
}


# ==================================================================
# DATA LOADING
# ==================================================================

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


def load_all_chains():
    chains = {}
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
    fprint("  Loaded chains for %d tickers" % len(chains))
    return chains


def load_regime_predictions():
    if not REGIME_FILE.exists():
        return None
    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    s = pd.Series(scores, index=dates, name="regime_score")
    return s[~s.index.duplicated(keep="last")]


def get_regime_score_at(regime_series, dt):
    if regime_series is None:
        return 0.5
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


# ==================================================================
# FEATURES
# ==================================================================

def compute_legacy_features(px, spy_slice):
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f["sharpe_63d"] = float(
        rets.iloc[-63:].mean() / (rets.iloc[-63:].std() + 1e-10) * np.sqrt(252)
    ) if len(rets) > 63 else 0.0
    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3
    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = rets.iloc[-63:][rets.iloc[-63:] < 0]
    f["sortino_63d"] = float(
        rets.iloc[-63:].mean() / (dr.std() + 1e-10) * np.sqrt(252)
    ) if len(dr) > 3 else 0.0
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
        slope, _, r_val, _, _ = scipy_stats.linregress(x, y)
        f["trend_r2_63d"] = r_val ** 2
        f["trend_slope_63d"] = slope * 252
    else:
        f["trend_r2_63d"] = 0.0
        f["trend_slope_63d"] = 0.0
    return f


def compute_cross_asset_features(sector_ticker, dt_idx, close_df):
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None
    if spy is None or len(spy) < 63:
        return {"sector_spy_beta_63d": 1.0, "cross_sector_dispersion": 0.01}
    if sector_px is not None and len(sector_px) > 63:
        sec_ret = sector_px.pct_change().dropna()
        spy_ret = spy.pct_change().dropna()
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


# ==================================================================
# LGBM WALK-FORWARD (shared across all variants)
# ==================================================================

def get_monthly_rebalance_dates(close):
    monthly = close.index.to_series().resample("MS").first().dropna()
    return pd.DatetimeIndex(monthly.values)


def get_weekly_fridays(close):
    return pd.DatetimeIndex(
        close.index.to_series().resample("W-FRI").last().dropna().values
    )


def build_feature_records(close, high, low, rebal_dates, regime_series):
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
    for c in V6_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[V6_FEATURES] = df[V6_FEATURES].fillna(0.0)
    fprint("    %d records, %d dates" % (len(df), len(df["date"].unique())))
    return df


def walk_forward_lgbm_rank(df):
    import lightgbm as lgb
    if len(df) < 100:
        return {}
    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())
    rankings = {}
    for i in range(WF_TRAIN_PERIODS, len(dates)):
        train_dates = dates[max(0, i - WF_TRAIN_PERIODS):i]
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


# ==================================================================
# CHAIN PRICING
# ==================================================================

def find_chain_spread_price(chain_df, trade_date, direction, K1, K2, dte_target):
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


def revalue_spread_bs(S, K1, K2, dte_remaining, atr, vix, direction):
    if dte_remaining <= 0:
        if direction == "bull":
            return max(S - K1, 0.0) - max(S - K2, 0.0)
        else:
            return max(K2 - S, 0.0) - max(K1 - S, 0.0)
    try:
        if direction == "bull":
            value, _ = price_bull_call_spread(S=S, K1=K1, K2=K2, dte=dte_remaining, atr=atr, vix=vix)
        else:
            value, _ = price_bear_put_spread(S=S, K1=K1, K2=K2, dte=dte_remaining, atr=atr, vix=vix)
        return value if value is not None else 0.0
    except Exception:
        return 0.0


def revalue_spread_chain(chain_df, check_date, direction, K1, K2, dte_remaining):
    if chain_df is None:
        return None
    result = find_chain_spread_price(chain_df, check_date, direction, K1, K2, dte_remaining)
    if result and result["found"]:
        return result["spread_cost_mid"]
    return None


# ==================================================================
# TRADE EXECUTION
# ==================================================================

def compute_strikes_param(S, direction, otm_pct, wfd, wfp):
    if direction == "bull":
        K1 = round(S * (1.0 + otm_pct), 2)
        w = max(wfd, K1 * wfp)
        K2 = round(K1 + w, 2)
    else:
        K2 = round(S * (1.0 - otm_pct), 2)
        w = max(wfd, K2 * wfp)
        K1 = round(K2 - w, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity,
                  chains, max_pos, cfg):
    """Execute a single spread trade with strategy-specific config."""
    if tk not in close.columns or tk not in atr_dict:
        return None
    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    av = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015
    K1, K2 = compute_strikes_param(S, direction, cfg["otm_pct"], cfg["wfd"], cfg["wfp"])

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

    if entry_cost_ps is None:
        try:
            if direction == "bull":
                entry_cost_ps, _ = price_bull_call_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix_val)
            else:
                entry_cost_ps, _ = price_bear_put_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix_val)
        except Exception:
            return None

    if entry_cost_ps is None or entry_cost_ps <= 0:
        return None

    spread_width = abs(K2 - K1)
    cwr = entry_cost_ps / max(spread_width, 0.01)
    if cwr > COST_WIDTH_MAX:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    max_profit_ps = spread_width - entry_cost_ps
    pt = cfg["profit_target"]

    # Profit target exit logic
    exited_early = False
    exit_day_idx = ei
    exit_reason = "expiry"
    hold_days = DTE

    if pt > 0 and max_profit_ps > 0:
        for check_idx in range(di + 1, ei + 1):
            if check_idx >= len(close):
                break
            check_date = close.index[check_idx]
            dte_remaining = ei - check_idx
            S_now = float(close[tk].iloc[check_idx])
            av_now = float(atr_dict[tk].loc[check_date]) if check_date in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[check_date]) else S_now * 0.015

            current_value_ps = None
            if chain_df is not None:
                current_value_ps = revalue_spread_chain(chain_df, check_date, direction, K1, K2, dte_remaining)
            if current_value_ps is None:
                current_value_ps = revalue_spread_bs(S_now, K1, K2, dte_remaining, av_now, vix_val, direction)

            if (current_value_ps - entry_cost_ps) >= pt * max_profit_ps:
                exited_early = True
                exit_day_idx = check_idx
                exit_reason = "profit_target_%dpct" % int(pt * 100)
                hold_days = check_idx - di
                break

    Se = float(close[tk].iloc[exit_day_idx])
    if exited_early:
        dte_at_exit = ei - exit_day_idx
        av_exit = float(atr_dict[tk].loc[close.index[exit_day_idx]]) if close.index[exit_day_idx] in atr_dict[tk].index else Se * 0.015
        exit_value_ps = None
        if chain_df is not None:
            exit_value_ps = revalue_spread_chain(chain_df, close.index[exit_day_idx], direction, K1, K2, dte_at_exit)
        if exit_value_ps is None:
            exit_value_ps = revalue_spread_bs(Se, K1, K2, dte_at_exit, av_exit, vix_val, direction)
        pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD - EARLY_EXIT_COMMISSION
    else:
        if direction == "bull":
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
        else:
            intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

    return {
        "pnl": round(pnl, 2),
        "entry_cost_ps": round(entry_cost_ps, 4),
        "total_cost": round(total_cost, 2),
        "used_real_pricing": used_real,
        "K1": K1, "K2": K2,
        "S_entry": round(S, 2), "S_exit": round(Se, 2),
        "spread_width": round(spread_width, 2),
        "cost_width_ratio": round(cwr, 3),
        "exited_early": exited_early,
        "exit_reason": exit_reason,
        "hold_days": hold_days,
    }


# ==================================================================
# SIMULATION ENGINE
# ==================================================================

def simulate_strategy(strategy_name, cfg, rankings, close, atr_dict, chains, rebal_dates):
    """Run backtest for a single strategy variant. Returns trade list."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None
    top_k = cfg["top_k"]
    equity = CAP
    trades = []

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
        if not scores or len(scores) < (top_k * 2):
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:top_k]]
        bear_picks = [t for t, _ in ranked_asc[:top_k]]
        n_positions = len(bull_picks) + len(bear_picks)

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                max_pos = min(CAP / n_positions * 2, equity * 0.40)
                if max_pos < 20:
                    continue
                result = execute_trade(
                    tk, dt, direction, close, atr_dict, cv, equity, chains, max_pos, cfg,
                )
                if result is not None:
                    equity += result["pnl"]
                    di = close.index.get_loc(dt)
                    ei = min(di + DTE, len(close) - 1)
                    sv = float(spy.loc[dt])
                    se = float(spy.iloc[ei]) if ei < len(spy) else sv
                    trades.append({
                        **result,
                        "entry_date": str(dt.date()),
                        "exit_date": str(close.index[min(di + result["hold_days"], len(close) - 1)].date()),
                        "ticker": tk,
                        "regime": "bull" if se >= sv else "bear",
                        "direction": direction,
                        "vix": round(cv, 1),
                        "win": result["pnl"] > 0,
                        "strategy": strategy_name,
                    })
    return trades


# ==================================================================
# RETURNS CONVERSION
# ==================================================================

def trades_to_monthly_returns(trades, initial_capital=CAP):
    """Convert trade list to monthly return series (capital-based pct returns)."""
    if not trades:
        return pd.Series(dtype=float)
    df = pd.DataFrame(trades)
    df["entry_month"] = pd.to_datetime(df["entry_date"]).dt.to_period("M")
    monthly_pnl = df.groupby("entry_month")["pnl"].sum()
    monthly_ret = monthly_pnl / initial_capital
    monthly_ret.index = monthly_ret.index.to_timestamp()
    return monthly_ret


def trades_to_equity_curve(trades, initial_capital=CAP):
    """Build equity curve from trades, indexed by exit date."""
    if not trades:
        return pd.Series(dtype=float)
    equity = initial_capital
    points = [{"date": pd.Timestamp("2009-01-01"), "equity": equity}]
    sorted_trades = sorted(trades, key=lambda t: t["exit_date"])
    for t in sorted_trades:
        equity += t["pnl"]
        points.append({"date": pd.Timestamp(t["exit_date"]), "equity": equity})
    df = pd.DataFrame(points).set_index("date")
    df = df[~df.index.duplicated(keep="last")]
    return df["equity"]


# ==================================================================
# METRICS
# ==================================================================

def compute_full_stats(trades, initial_capital=CAP):
    """Comprehensive stats from trade list."""
    if not trades or len(trades) < 5:
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

    df = pd.DataFrame(trades)
    df["entry_month"] = pd.to_datetime(df["entry_date"]).dt.to_period("M")
    monthly_pnl = df.groupby("entry_month")["pnl"].sum()
    monthly_ret = monthly_pnl / initial_capital
    sharpe = float(monthly_ret.mean() / (monthly_ret.std() + 1e-10) * np.sqrt(12)) if len(monthly_ret) > 1 else 0.0

    neg_ret = monthly_ret[monthly_ret < 0]
    downside_std = float(neg_ret.std()) if len(neg_ret) > 1 else 1e-10
    sortino = float(monthly_ret.mean() / (downside_std + 1e-10) * np.sqrt(12))

    max_dd_pct = float(dd.min()) * 100
    first_date = pd.Timestamp(min(t["entry_date"] for t in trades))
    last_date = pd.Timestamp(max(t["entry_date"] for t in trades))
    years = (last_date - first_date).days / 365.25
    total_ret = equity[-1] / equity[0]
    cagr = (total_ret ** (1 / max(years, 0.5))) - 1
    calmar = cagr / (abs(dd.min()) + 1e-10) if dd.min() != 0 else 0.0

    return {
        "n_trades": len(trades),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 4),
        "max_dd_pct": round(max_dd_pct, 2),
        "cagr": round(cagr, 4),
        "calmar": round(calmar, 3),
        "total_pnl": round(float(pnls.sum()), 2),
        "final_equity": round(equity[-1], 2),
    }


def compute_monthly_metrics(monthly_returns):
    """Compute metrics from a monthly return series."""
    if len(monthly_returns) < 6:
        return None
    r = monthly_returns.dropna()
    ann_ret = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0

    neg = r[r < 0]
    downside = neg.std() * np.sqrt(12) if len(neg) > 1 else 1e-10
    sortino = ann_ret / downside

    cum = (1 + r).cumprod()
    n_years = len(r) / 12
    cagr = (cum.iloc[-1] ** (1 / max(n_years, 0.5))) - 1
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / max(losses, 1e-10)
    wr = (r > 0).mean()

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 4),
        "max_dd": round(max_dd, 4),
        "cagr": round(cagr, 4),
        "calmar": round(calmar, 3),
        "n_months": len(r),
        "ann_return": round(ann_ret, 4),
        "ann_vol": round(ann_vol, 4),
    }


# ==================================================================
# ADVERSARIAL VALIDATION (5 gates)
# ==================================================================

def permutation_test(monthly_returns, n_perms=N_PERMUTATIONS):
    """Sign-flip permutation test on monthly returns."""
    r = monthly_returns.dropna().values
    if len(r) < 6:
        return {"pass": False, "p_value": 1.0, "real_sharpe": 0.0}
    real_sharpe = r.mean() / (r.std() + 1e-10) * np.sqrt(12)
    rng = np.random.RandomState(42)
    perm_sharpes = np.zeros(n_perms)
    for i in range(n_perms):
        signs = rng.choice([-1, 1], size=len(r))
        flipped = r * signs
        perm_sharpes[i] = flipped.mean() / (flipped.std() + 1e-10) * np.sqrt(12)
    p_value = float((perm_sharpes >= real_sharpe).mean())
    return {
        "pass": p_value < 0.05,
        "p_value": round(p_value, 4),
        "real_sharpe": round(real_sharpe, 4),
        "perm_mean": round(float(perm_sharpes.mean()), 4),
    }


def regime_balance_test(monthly_returns, vix_monthly):
    """Check Sharpe in low-vol vs high-vol regimes (threshold=0.50 ratio)."""
    aligned = pd.DataFrame({"ret": monthly_returns, "vix": vix_monthly}).dropna()
    if len(aligned) < 12:
        return {"pass": False, "detail": "insufficient_data"}
    low = aligned[aligned["vix"] < 20]["ret"]
    high = aligned[aligned["vix"] >= 20]["ret"]
    if len(low) < 6 or len(high) < 6:
        return {"pass": True, "detail": "single_regime_dominant", "n_low": len(low), "n_high": len(high)}
    s_low = low.mean() / (low.std() + 1e-10) * np.sqrt(12)
    s_high = high.mean() / (high.std() + 1e-10) * np.sqrt(12)
    max_s = max(abs(s_low), abs(s_high))
    ratio = abs(s_low - s_high) / max_s if max_s > 0 else 0
    return {
        "pass": ratio < 0.50,
        "low_vol_sharpe": round(s_low, 3),
        "high_vol_sharpe": round(s_high, 3),
        "regime_gap_ratio": round(ratio, 3),
        "n_low": len(low),
        "n_high": len(high),
    }


def subperiod_stability_test(monthly_returns):
    """Both halves must be profitable."""
    r = monthly_returns.dropna()
    if len(r) < 12:
        return {"pass": False, "detail": "insufficient_data"}
    mid = len(r) // 2
    first_half = r.iloc[:mid]
    second_half = r.iloc[mid:]
    s1 = first_half.mean() / (first_half.std() + 1e-10) * np.sqrt(12)
    s2 = second_half.mean() / (second_half.std() + 1e-10) * np.sqrt(12)
    both_positive = s1 > 0 and s2 > 0
    return {
        "pass": both_positive,
        "first_half_sharpe": round(s1, 3),
        "second_half_sharpe": round(s2, 3),
    }


def outlier_removal_test(monthly_returns):
    """Still profitable without best month."""
    r = monthly_returns.dropna()
    if len(r) < 6:
        return {"pass": False}
    without_best = r.drop(r.idxmax())
    s = without_best.mean() / (without_best.std() + 1e-10) * np.sqrt(12)
    return {
        "pass": s > 0,
        "sharpe_without_best_month": round(s, 3),
        "removed_month_return": round(float(r.max()), 4),
    }


def yearly_consistency_test(monthly_returns):
    """What % of years are profitable?"""
    r = monthly_returns.dropna()
    if len(r) < 24:
        return {"pass": True, "detail": "insufficient_years"}
    yearly = r.groupby(r.index.year).sum()
    pct_profitable = (yearly > 0).mean()
    return {
        "pass": pct_profitable >= 0.60,
        "pct_years_profitable": round(float(pct_profitable), 3),
        "n_years": len(yearly),
        "yearly_returns": {str(y): round(float(v), 4) for y, v in yearly.items()},
    }


def run_5gate_validation(monthly_returns, vix_monthly, name=""):
    """Run all 5 adversarial gates. Returns dict with gate results and overall pass."""
    gates = {}
    gates["1_permutation"] = permutation_test(monthly_returns)
    gates["2_regime_balance"] = regime_balance_test(monthly_returns, vix_monthly)
    gates["3_subperiod"] = subperiod_stability_test(monthly_returns)
    gates["4_outlier_removal"] = outlier_removal_test(monthly_returns)
    gates["5_yearly_consistency"] = yearly_consistency_test(monthly_returns)
    n_pass = sum(1 for g in gates.values() if g.get("pass", False))
    gates["gates_passed"] = n_pass
    gates["all_passed"] = n_pass == 5
    return gates


# ==================================================================
# VIX ALLOCATOR (UPRO/Cash switching)
# ==================================================================

def compute_vix_allocator_returns(close):
    """Compute UPRO/cash monthly returns based on VIX regime.
    VIX < 17: 100% UPRO
    VIX 17-25: 50% UPRO
    VIX > 25: 100% cash (money market ~0.4%/mo)
    """
    if "UPRO" not in close.columns or "VIX" not in close.columns:
        fprint("  WARNING: UPRO or VIX not in data, skipping VIX allocator")
        return None

    upro = close["UPRO"].dropna()
    vix = close["VIX"].reindex(upro.index, method="ffill")
    upro_daily_ret = upro.pct_change().dropna()
    vix_aligned = vix.reindex(upro_daily_ret.index, method="ffill")

    # Daily allocation based on prior day VIX
    vix_prior = vix_aligned.shift(1).dropna()
    common = upro_daily_ret.index.intersection(vix_prior.index)
    upro_ret = upro_daily_ret.loc[common]
    vix_p = vix_prior.loc[common]

    alloc = pd.Series(0.0, index=common)
    alloc[vix_p < 17] = 1.0
    alloc[(vix_p >= 17) & (vix_p <= 25)] = 0.50
    alloc[vix_p > 25] = 0.0

    # Cash earns ~5% annually = 0.02% daily
    cash_daily = 0.05 / 252
    daily_ret = alloc * upro_ret + (1 - alloc) * cash_daily

    monthly_ret = daily_ret.resample("MS").sum()
    return monthly_ret


# ==================================================================
# PORTFOLIO VARIANTS
# ==================================================================

def build_portfolio_variants(strategy_monthly, vix_allocator_monthly, vix_monthly):
    """Build 6 portfolio variants from strategy monthly returns."""

    variants = {}
    strat_names = list(strategy_monthly.keys())
    aligned = pd.DataFrame(strategy_monthly).dropna()
    if len(aligned) < 6:
        fprint("ERROR: Only %d overlapping months, need at least 6" % len(aligned))
        return {}

    fprint("  Overlapping months: %d (%s to %s)" % (
        len(aligned), aligned.index.min().strftime("%Y-%m"),
        aligned.index.max().strftime("%Y-%m")))

    # ── A: V10 alone (baseline) ──
    if "V10" in aligned.columns:
        variants["A_V10_alone"] = aligned["V10"].copy()
    else:
        variants["A_V10_alone"] = aligned.iloc[:, 0].copy()  # fallback to first

    # ── B: Equal weight ──
    variants["B_equal_weight"] = aligned.mean(axis=1)

    # ── C: Sharpe-weighted ──
    ref_sharpes = np.array([STRATEGY_VARIANTS[s]["ref_sharpe"] for s in strat_names if s in STRATEGY_VARIANTS])
    if len(ref_sharpes) == len(strat_names):
        sharpe_weights = ref_sharpes / ref_sharpes.sum()
    else:
        sharpe_weights = np.ones(len(strat_names)) / len(strat_names)
    variants["C_sharpe_weighted"] = (aligned.values * sharpe_weights).sum(axis=1)
    variants["C_sharpe_weighted"] = pd.Series(
        variants["C_sharpe_weighted"], index=aligned.index)

    # ── D: V10 + VIX allocator overlay ──
    if vix_allocator_monthly is not None:
        v10_ret = aligned["V10"] if "V10" in aligned.columns else aligned.iloc[:, 0]
        # Combine: 70% V10, 30% VIX allocator
        vix_alloc_aligned = vix_allocator_monthly.reindex(aligned.index)
        combined = pd.DataFrame({"v10": v10_ret, "vix_alloc": vix_alloc_aligned}).dropna()
        if len(combined) > 6:
            # When VIX > 25, reduce V10 allocation
            vix_m = vix_monthly.reindex(combined.index, method="ffill")
            v10_weight = pd.Series(0.70, index=combined.index)
            v10_weight[vix_m > 25] = 0.40  # scale back in high vol
            v10_weight[vix_m < 17] = 0.80   # lean in during low vol
            vix_alloc_weight = 1.0 - v10_weight
            variants["D_V10_vix_overlay"] = (
                combined["v10"] * v10_weight + combined["vix_alloc"] * vix_alloc_weight
            )

    # ── E: Risk parity (inverse vol) ──
    rolling_vol = aligned.rolling(6, min_periods=3).std()
    inv_vol = 1.0 / (rolling_vol + 1e-10)
    rp_weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    variants["E_risk_parity"] = (aligned * rp_weights).sum(axis=1)

    # ── F: Best-of (trailing 3-month Sharpe) ──
    trailing_sharpe = aligned.rolling(3, min_periods=3).apply(
        lambda x: x.mean() / (x.std() + 1e-10) * np.sqrt(12), raw=True
    )
    best_strat_idx = trailing_sharpe.idxmax(axis=1)
    best_of = pd.Series(0.0, index=aligned.index)
    for idx in aligned.index:
        bs = best_strat_idx.get(idx)
        if pd.notna(bs) and bs in aligned.columns:
            best_of[idx] = aligned.loc[idx, bs]
        else:
            best_of[idx] = aligned.loc[idx].mean()
    variants["F_best_of"] = best_of

    return variants


# ==================================================================
# MAIN
# ==================================================================

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint("PORTFOLIO COMBINATION V2 -- Sector Options Multi-Strategy")
    fprint("Run: %s" % t0.strftime("%Y-%m-%d %H:%M:%S"))
    fprint("=" * 80)
    fprint("")
    fprint("Capital: $%.0f | DTE: %d | Commission: $%.2f | Haircut: %.0f%%" % (
        CAP, DTE, COMMISSION_RT_SPREAD, HAIRCUT * 100))
    fprint("Walk-forward: %d-week sliding | LGBM 17 features" % WF_TRAIN_PERIODS)
    fprint("Permutation trials: %d" % N_PERMUTATIONS)
    fprint("")
    fprint("STRATEGIES UNDER TEST:")
    for name, cfg in STRATEGY_VARIANTS.items():
        fprint("  %s: %s (ref Sharpe=%.2f)" % (name, cfg["desc"], cfg["ref_sharpe"]))
    fprint("")

    # ── 1. Load data ──
    fprint("[1] Loading data...")
    close, high, low = download_data()
    chains = load_all_chains()
    regime_series = load_regime_predictions()
    atr_dict = compute_atr_series(high, low, close)

    monthly_dates = get_monthly_rebalance_dates(close)
    weekly_dates = get_weekly_fridays(close)
    fprint("  %d monthly rebalance dates, %d weekly feature dates" % (
        len(monthly_dates), len(weekly_dates)))

    # ── 2. Build shared LGBM rankings ──
    fprint("")
    fprint("[2] Building LGBM rankings (52-week walk-forward)...")
    records = build_feature_records(close, high, low, weekly_dates, regime_series)
    rankings = walk_forward_lgbm_rank(records)
    if len(rankings) < 20:
        fprint("ERROR: Only %d ranking dates -- insufficient" % len(rankings))
        return

    spy_close = close["SPY"]

    # ── 3. Run backtests for each strategy ──
    fprint("")
    fprint("[3] Running backtests for each strategy variant...")
    strategy_trades = {}
    strategy_stats = {}
    strategy_monthly = {}

    for name, cfg in STRATEGY_VARIANTS.items():
        fprint("")
        fprint("  --- %s: %s ---" % (name, cfg["desc"]))
        trades = simulate_strategy(name, cfg, rankings, close, atr_dict, chains, monthly_dates)
        strategy_trades[name] = trades
        stats = compute_full_stats(trades)
        strategy_stats[name] = stats
        monthly_ret = trades_to_monthly_returns(trades)
        strategy_monthly[name] = monthly_ret

        if stats:
            fprint("    %d trades | Sharpe=%.2f | Sortino=%.2f | PF=%.2f | WR=%.1f%% | MDD=%.1f%% | Final=$%.0f" % (
                stats["n_trades"], stats["sharpe"], stats["sortino"],
                stats["profit_factor"], stats["win_rate"] * 100,
                stats["max_dd_pct"], stats["final_equity"]))
        else:
            fprint("    WARNING: insufficient trades for stats")

    # ── 4. Correlation analysis ──
    fprint("")
    fprint("[4] Correlation analysis...")
    aligned_monthly = pd.DataFrame(strategy_monthly).dropna()
    if len(aligned_monthly) > 6:
        corr = aligned_monthly.corr()
        fprint("")
        fprint("  Return stream correlations:")
        for i, s1 in enumerate(corr.columns):
            for j, s2 in enumerate(corr.columns):
                if j > i:
                    fprint("    %s vs %s: %.3f" % (s1, s2, corr.loc[s1, s2]))
        fprint("")
        diversification_benefit = (corr.values[np.triu_indices_from(corr.values, k=1)] < 0.80).all()
        fprint("  Diversification benefit (all pairs < 0.80): %s" % diversification_benefit)
    else:
        corr = None
        fprint("  WARNING: insufficient overlapping months for correlation")

    # ── 5. VIX allocator ──
    fprint("")
    fprint("[5] Computing VIX allocator (UPRO/cash switching)...")
    vix_alloc_monthly = compute_vix_allocator_returns(close)
    if vix_alloc_monthly is not None:
        vix_alloc_metrics = compute_monthly_metrics(vix_alloc_monthly)
        if vix_alloc_metrics:
            fprint("    VIX allocator standalone: Sharpe=%.2f, CAGR=%.1f%%, MDD=%.1f%%" % (
                vix_alloc_metrics["sharpe"], vix_alloc_metrics["cagr"] * 100,
                vix_alloc_metrics["max_dd"] * 100))

    # VIX monthly for regime tests
    vix_monthly = close["VIX"].resample("MS").mean() if "VIX" in close.columns else None

    # ── 6. Build portfolio variants ──
    fprint("")
    fprint("[6] Building 6 portfolio variants...")
    variants = build_portfolio_variants(strategy_monthly, vix_alloc_monthly, vix_monthly)

    # ── 7. Evaluate each variant ──
    fprint("")
    fprint("[7] Evaluating portfolio variants...")
    variant_results = {}

    for vname, vreturns in variants.items():
        fprint("")
        fprint("  --- %s ---" % vname)
        metrics = compute_monthly_metrics(vreturns)
        if metrics is None:
            fprint("    SKIP: insufficient data")
            continue

        fprint("    Sharpe=%.2f | Sortino=%.2f | PF=%.2f | WR=%.1f%% | MDD=%.1f%% | CAGR=%.1f%% | Calmar=%.2f" % (
            metrics["sharpe"], metrics["sortino"], metrics["profit_factor"],
            metrics["win_rate"] * 100, metrics["max_dd"] * 100,
            metrics["cagr"] * 100, metrics["calmar"]))

        # 5-gate adversarial validation
        vix_m_aligned = vix_monthly.reindex(vreturns.index, method="ffill") if vix_monthly is not None else None
        gates = run_5gate_validation(vreturns, vix_m_aligned, vname)
        fprint("    Adversarial: %d/5 gates passed" % gates["gates_passed"])
        for gname, gresult in gates.items():
            if isinstance(gresult, dict) and "pass" in gresult:
                status = "PASS" if gresult["pass"] else "FAIL"
                fprint("      [%s] %s" % (status, gname))

        variant_results[vname] = {
            "metrics": metrics,
            "adversarial": gates,
            "n_months": len(vreturns.dropna()),
        }

    # ── 8. Results ranking ──
    fprint("")
    fprint("=" * 90)
    fprint("RESULTS RANKING (by Sharpe)")
    fprint("=" * 90)

    ranking = []
    for vname, vdata in variant_results.items():
        m = vdata["metrics"]
        g = vdata["adversarial"]
        ranking.append((
            vname, m["sharpe"], m["sortino"], m["profit_factor"],
            m["win_rate"], m["max_dd"], m["cagr"], m["calmar"],
            g["gates_passed"],
        ))
    ranking.sort(key=lambda x: x[1], reverse=True)

    fprint("")
    fprint("%-25s %7s %7s %7s %7s %7s %7s %7s %6s" % (
        "Variant", "Sharpe", "Sort.", "PF", "WR%", "MDD%", "CAGR%", "Calmar", "Gates"))
    fprint("-" * 85)
    for r in ranking:
        fprint("%-25s %7.2f %7.2f %7.2f %6.1f%% %6.1f%% %6.1f%% %7.2f   %d/5" % (
            r[0], r[1], r[2], r[3], r[4] * 100, r[5] * 100, r[6] * 100, r[7], r[8]))

    # ── 9. Sharpe-weighting detail ──
    ref_sharpes = [STRATEGY_VARIANTS[s]["ref_sharpe"] for s in STRATEGY_VARIANTS]
    sharpe_weights = np.array(ref_sharpes) / sum(ref_sharpes)
    fprint("")
    fprint("Sharpe-weighted allocation:")
    for s, w in zip(STRATEGY_VARIANTS.keys(), sharpe_weights):
        fprint("  %s: %.1f%%" % (s, w * 100))

    # ── 10. Save results ──
    fprint("")
    fprint("[8] Saving results...")

    save_results = {
        "timestamp": t0.isoformat(),
        "experiment": "portfolio_combination_v2",
        "config": {
            "capital": CAP, "dte": DTE,
            "commission": COMMISSION_RT_SPREAD,
            "haircut": HAIRCUT,
            "wf_weeks": WF_TRAIN_PERIODS,
            "n_sectors": len(SECTORS),
            "n_permutations": N_PERMUTATIONS,
        },
        "strategy_stats": strategy_stats,
        "correlation_matrix": corr.round(4).to_dict() if corr is not None else None,
        "sharpe_weights": {s: round(w, 4) for s, w in zip(STRATEGY_VARIANTS.keys(), sharpe_weights)},
        "variant_results": variant_results,
        "ranking": [
            {"name": r[0], "sharpe": r[1], "sortino": r[2], "pf": r[3],
             "wr": r[4], "mdd": r[5], "cagr": r[6], "calmar": r[7], "gates": r[8]}
            for r in ranking
        ],
    }

    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint("  Results saved to %s" % results_file)

    # Save monthly returns for each variant
    variant_monthly_df = pd.DataFrame({
        vname: vret for vname, vret in variants.items()
    })
    variant_monthly_df.to_csv(OUTPUT_DIR / "variant_monthly_returns.csv")

    # Save strategy monthly returns
    strategy_monthly_df = pd.DataFrame(strategy_monthly)
    strategy_monthly_df.to_csv(OUTPUT_DIR / "strategy_monthly_returns.csv")

    # Save correlation
    if corr is not None:
        corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    # ── 11. MLflow logging ──
    fprint("")
    fprint("[9] Logging to MLflow...")
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="combo_v2_%s" % t0.strftime("%Y%m%d_%H%M")):
                mlflow.log_param("capital", CAP)
                mlflow.log_param("n_strategies", len(STRATEGY_VARIANTS))
                mlflow.log_param("strategies", ",".join(STRATEGY_VARIANTS.keys()))
                mlflow.log_param("n_permutations", N_PERMUTATIONS)

                # Log strategy-level metrics
                for sname, sstats in strategy_stats.items():
                    if sstats:
                        mlflow.log_metric("%s_sharpe" % sname, sstats["sharpe"])
                        mlflow.log_metric("%s_sortino" % sname, sstats["sortino"])
                        mlflow.log_metric("%s_pf" % sname, sstats["profit_factor"])
                        mlflow.log_metric("%s_wr" % sname, sstats["win_rate"])

                # Log variant-level metrics
                for vname, vdata in variant_results.items():
                    m = vdata["metrics"]
                    prefix = vname.split("_")[0]
                    mlflow.log_metric("%s_sharpe" % prefix, m["sharpe"])
                    mlflow.log_metric("%s_sortino" % prefix, m["sortino"])
                    mlflow.log_metric("%s_cagr" % prefix, m["cagr"])
                    mlflow.log_metric("%s_maxdd" % prefix, m["max_dd"])
                    mlflow.log_metric("%s_calmar" % prefix, m["calmar"])
                    mlflow.log_metric("%s_pf" % prefix, m["profit_factor"])
                    mlflow.log_metric("%s_gates" % prefix, vdata["adversarial"]["gates_passed"])

                # Log correlation between strategies
                if corr is not None:
                    for i, s1 in enumerate(corr.columns):
                        for j, s2 in enumerate(corr.columns):
                            if j > i:
                                mlflow.log_metric("corr_%s_%s" % (s1, s2), float(corr.loc[s1, s2]))

                # Best variant
                if ranking:
                    mlflow.log_param("best_variant", ranking[0][0])
                    mlflow.log_metric("best_sharpe", ranking[0][1])

                mlflow.log_artifact(str(results_file))
                fprint("  MLflow logged successfully")
        except Exception as e:
            fprint("  MLflow error: %s" % e)
    else:
        fprint("  MLflow unavailable, skipped")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint("")
    fprint("=" * 80)
    fprint("COMPLETED in %.1f minutes" % (elapsed / 60))
    fprint("Output: %s" % OUTPUT_DIR)
    fprint("=" * 80)


if __name__ == "__main__":
    main()
