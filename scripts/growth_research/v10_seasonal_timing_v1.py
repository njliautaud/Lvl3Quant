#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
V10 Seasonal Timing V1 -- Can Calendar/Seasonal Timing Improve Sector Rotation?
================================================================================

Tests whether calendar-based timing filters can improve the V10 sector rotation
strategy (LGBM-ranked bull call / bear put spreads on 11 sector ETFs).

6 Variants:
  A: V10 baseline       -- no timing filter (control)
  B: OPEX avoidance     -- skip rebalance if within 3 days of monthly OPEX (3rd Friday)
  C: Month-start entry  -- only enter in first 5 trading days of calendar month
  D: Sell-in-May        -- 50% position size May-Oct, full size Nov-Apr
  E: Quarter-end mom    -- double position count (top_k=8) in last 5 days of quarter
  F: VIX seasonality    -- skip bear trades if VIX < 63d 25th pctile AND month in Jun/Jul/Aug/Dec

Config: $645 capital, $2.60 commission (+$2.60 early exits), 15% haircut,
        ~500d sliding WF, LGBM 17 features, DTE=28, monthly rebalance,
        8 positions (top_k=4 bull + bottom_k=4 bear), 2% OTM, max($3,3%), 50% PT.

Self-contained: embeds BS pricing and 5-gate adversarial validator.

Output: output/growth_research/v10_seasonal_timing_v1/
MLflow: experiment v10_seasonal_timing_v1, server http://jupiter:5000
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
# PATHS
# ================================================================
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")
BASE = _NEPTUNE_BASE if _NEPTUNE_BASE.exists() else _JUPITER_BASE
fprint("Running on %s: %s" % ("Neptune" if BASE == _NEPTUNE_BASE else "Jupiter", BASE))

OUTPUT_DIR = BASE / "output" / "growth_research" / "v10_seasonal_timing_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# CONSTANTS
# ================================================================
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 28
MAX_PER_TRADE = 200.0
HAIRCUT = 0.15
COMMISSION_RT = 2.60
EARLY_EXIT_COMMISSION = 2.60
RISK_FREE_RATE = 0.045
COST_WIDTH_MAX = 0.50

# V10 base structural params
BASE_TOP_K = 4
BASE_OTM_PCT = 0.02
BASE_WIDTH_FLOOR_DOLLARS = 3.0
BASE_WIDTH_FLOOR_PCT = 0.03
BASE_PROFIT_TARGET_PCT = 0.50

WF_TRAIN_DAYS = 500  # ~500d sliding window (~2 years)

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v10_seasonal_timing_v1"

N_BOOTSTRAP = 1000
N_PERMUTATIONS = 300
VIX_HIGH_THRESHOLD = 20.0

V6_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d",
    "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]
assert len(V6_FEATURES) == 18, "Expected 18 features but got %d" % len(V6_FEATURES)
# Note: user spec says 17 features but lists vol_21d, vol_63d, maxdd_63d which
# brings it to 18. We include all listed features. The model handles it fine.

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

# ================================================================
# SELF-CONTAINED BLACK-SCHOLES PRICING
# ================================================================

def _bs_call(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_put(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
    """Black-Scholes European put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _estimate_iv(atr, spot, vix=20.0, atr_period=14):
    """Estimate IV from ATR + VIX."""
    if spot <= 0 or atr <= 0:
        return 0.25
    realized_vol = (atr / spot) * np.sqrt(252 / atr_period)
    iv_mult = 1.2 + 0.01 * max(vix - 20.0, 0.0)
    return max(realized_vol * iv_mult, 0.10)


def price_bull_call_spread(S, K1, K2, dte, atr, vix=20.0):
    """Price a bull call spread with 15% haircut. Returns (entry_cost_ps, max_profit_ps)."""
    if K2 <= K1:
        return None, None
    T = dte / 365.0
    sigma = _estimate_iv(atr, S, vix)
    fair = _bs_call(S, K1, T, sigma=sigma) - _bs_call(S, K2, T, sigma=sigma)
    fair = max(fair, 0.001)
    entry_cost = fair * (1.0 + HAIRCUT)
    max_profit = (K2 - K1) - entry_cost
    return float(entry_cost), float(max_profit)


def price_bear_put_spread(S, K1, K2, dte, atr, vix=20.0):
    """Price a bear put spread with 15% haircut. Returns (entry_cost_ps, max_profit_ps)."""
    if K2 <= K1:
        return None, None
    T = dte / 365.0
    sigma = _estimate_iv(atr, S, vix)
    fair = _bs_put(S, K2, T, sigma=sigma) - _bs_put(S, K1, T, sigma=sigma)
    fair = max(fair, 0.001)
    entry_cost = fair * (1.0 + HAIRCUT)
    max_profit = (K2 - K1) - entry_cost
    return float(entry_cost), float(max_profit)


def _revalue_spread_bs(S, K1, K2, dte_remaining, atr, vix, direction):
    """Revalue a spread using BS at a given point in time."""
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


# ================================================================
# SELF-CONTAINED 5-GATE ADVERSARIAL VALIDATOR
# ================================================================

def _honest_sharpe_sortino(equity_series):
    """Compute honest Sharpe/Sortino from equity series using pct_change."""
    if len(equity_series) < 2:
        return 0.0, 0.0, pd.Series(dtype=float)
    monthly_eq = equity_series.resample("ME").last().dropna()
    if len(monthly_eq) < 2:
        return 0.0, 0.0, pd.Series(dtype=float)
    monthly_ret = monthly_eq.pct_change().dropna()
    if len(monthly_ret) < 2 or monthly_ret.std() == 0:
        return 0.0, 0.0, monthly_ret
    sharpe = float(monthly_ret.mean() / monthly_ret.std() * np.sqrt(12))
    neg = monthly_ret[monthly_ret < 0]
    if len(neg) > 0 and neg.std() > 0:
        sortino = float(monthly_ret.mean() / neg.std() * np.sqrt(12))
    else:
        sortino = sharpe * 1.5
    return sharpe, sortino, monthly_ret


def _build_equity_series(trades, initial_capital):
    """Build equity time series from trade list."""
    if not trades:
        return pd.Series(dtype=float), [], []
    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("exit_date")
    pnls = df["pnl"].values
    equity_values = [initial_capital]
    for p in pnls:
        equity_values.append(equity_values[-1] + p)
    dates = [df["entry_date"].iloc[0] - pd.Timedelta(days=1)]
    dates.extend(df["exit_date"].tolist())
    eq_series = pd.Series(equity_values, index=pd.DatetimeIndex(dates))
    eq_series = eq_series.groupby(eq_series.index).last()
    return eq_series, pnls, df


def _gate1_permutation(trades, initial_capital, real_sharpe, n_perms=300):
    """Sign-flip permutation test."""
    pnls = np.array([t["pnl"] for t in trades])
    dates = pd.to_datetime([t["exit_date"] for t in trades])
    beat = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(pnls))
        flipped = pnls * signs
        eq_vals = [initial_capital]
        for p in flipped:
            eq_vals.append(eq_vals[-1] + p)
        eq_s = pd.Series(eq_vals, index=pd.DatetimeIndex(
            [dates[0] - pd.Timedelta(days=1)] + list(dates)))
        eq_s = eq_s.groupby(eq_s.index).last()
        ps, _, _ = _honest_sharpe_sortino(eq_s)
        if ps >= real_sharpe:
            beat += 1
    p_val = beat / n_perms
    return {"name": "Permutation", "passed": p_val < 0.05,
            "metric_name": "p_value", "metric_value": round(p_val, 4),
            "detail": "%d/%d beat real Sharpe %.3f" % (beat, n_perms, real_sharpe)}


def _gate2_regime_balance(trades, spy_prices):
    """Regime balance: WR gap between bull and bear markets < 0.50."""
    bull_pnls, bear_pnls = [], []
    if spy_prices is not None and len(spy_prices) > 0:
        spy_prices = spy_prices.sort_index()
        for t in trades:
            ed = pd.Timestamp(t["entry_date"])
            xd = pd.Timestamp(t["exit_date"])
            ep = spy_prices[spy_prices.index <= ed]
            xp = spy_prices[spy_prices.index <= xd]
            if len(ep) == 0 or len(xp) == 0:
                continue
            if float(xp.iloc[-1]) >= float(ep.iloc[-1]):
                bull_pnls.append(t["pnl"])
            else:
                bear_pnls.append(t["pnl"])
    else:
        for t in trades:
            if t.get("regime") in ("bull", "green"):
                bull_pnls.append(t["pnl"])
            else:
                bear_pnls.append(t["pnl"])
    if len(bull_pnls) < 5 or len(bear_pnls) < 5:
        return {"name": "Regime Balance", "passed": True,
                "metric_name": "wr_gap", "metric_value": 0.0,
                "detail": "Insufficient split: %d bull, %d bear" % (len(bull_pnls), len(bear_pnls))}
    bwr = np.mean([1 if p > 0 else 0 for p in bull_pnls])
    bea = np.mean([1 if p > 0 else 0 for p in bear_pnls])
    gap = abs(bwr - bea)
    return {"name": "Regime Balance", "passed": gap < 0.50,
            "metric_name": "wr_gap", "metric_value": round(gap, 4),
            "detail": "Bull WR=%.1f%% (%d), Bear WR=%.1f%% (%d)" % (bwr*100, len(bull_pnls), bea*100, len(bear_pnls))}


def _gate3_sub_period(trades):
    """Both halves must be independently profitable."""
    df = pd.DataFrame(trades).sort_values("exit_date", key=pd.to_datetime)
    mid = len(df) // 2
    h1 = df.iloc[:mid]["pnl"].sum()
    h2 = df.iloc[mid:]["pnl"].sum()
    ok = h1 > 0 and h2 > 0
    return {"name": "Sub-Period", "passed": ok,
            "metric_name": "min_half_pnl", "metric_value": round(min(h1, h2), 2),
            "detail": "H1=$%.0f, H2=$%.0f" % (h1, h2)}


def _gate4_outlier_removal(monthly_returns):
    """Remove best month, check still profitable."""
    if len(monthly_returns) < 3:
        return {"name": "Outlier Removal", "passed": False,
                "metric_name": "sharpe_ex_best", "metric_value": 0.0, "detail": "Too few months"}
    best_idx = monthly_returns.idxmax()
    trimmed = monthly_returns.drop(best_idx)
    if len(trimmed) < 2 or trimmed.std() == 0:
        return {"name": "Outlier Removal", "passed": False,
                "metric_name": "sharpe_ex_best", "metric_value": 0.0, "detail": "Not enough data after trim"}
    ts = float(trimmed.mean() / trimmed.std() * np.sqrt(12))
    ok = trimmed.sum() > 0 and ts > 0
    return {"name": "Outlier Removal", "passed": ok,
            "metric_name": "sharpe_ex_best", "metric_value": round(ts, 3),
            "detail": "Trimmed Sharpe=%.2f" % ts}


def _gate5_yearly_consistency(trades, threshold=0.60):
    """At least 60% of calendar years must be net positive."""
    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["year"] = df["exit_date"].dt.year
    yearly = df.groupby("year")["pnl"].sum()
    n_y = len(yearly)
    n_pos = int((yearly > 0).sum())
    pct = n_pos / n_y if n_y > 0 else 0
    return {"name": "Yearly Consistency", "passed": pct >= threshold,
            "metric_name": "pct_years_pos", "metric_value": round(pct, 4),
            "detail": "%d/%d years profitable" % (n_pos, n_y)}


def validate_trades_5gate(trades, initial_capital, spy_prices=None, name="Strategy"):
    """Run 5-gate adversarial validation. Returns dict with gates and metrics."""
    if len(trades) < 10:
        return {"name": name, "error": "Too few trades (%d)" % len(trades),
                "gates_passed": 0, "gates_total": 5, "all_passed": False, "gates": []}
    eq_s, pnls, _ = _build_equity_series(trades, initial_capital)
    sharpe, sortino, monthly_ret = _honest_sharpe_sortino(eq_s)
    gates = [
        _gate1_permutation(trades, initial_capital, sharpe, N_PERMUTATIONS),
        _gate2_regime_balance(trades, spy_prices),
        _gate3_sub_period(trades),
        _gate4_outlier_removal(monthly_ret),
        _gate5_yearly_consistency(trades),
    ]
    passed = sum(1 for g in gates if g["passed"])
    return {"name": name, "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
            "gates_passed": passed, "gates_total": 5,
            "all_passed": passed == 5, "gates": gates}


# ================================================================
# DATA + FEATURES
# ================================================================

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


def compute_features(px, spy_slice):
    """Compute 18 momentum/quality features for a single sector at a point in time."""
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2
    f["sharpe_63d"] = float(
        rets.iloc[-63:].mean() / (rets.iloc[-63:].std() + 1e-10) * np.sqrt(252)
    ) if len(rets) > 63 else 0.0
    # maxdd_63d
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
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f["trend_r2_63d"] = r_val ** 2
        f["trend_slope_63d"] = slope * 252
    else:
        f["trend_r2_63d"] = 0.0
        f["trend_slope_63d"] = 0.0
    return f


# ================================================================
# CALENDAR HELPERS
# ================================================================

def get_monthly_opex(year, month):
    """Return the 3rd Friday of the given month (monthly OPEX)."""
    # 1st day of month
    first = pd.Timestamp(year, month, 1)
    # weekday: 0=Mon, 4=Fri
    dow = first.weekday()
    # days until first Friday
    first_friday = first + pd.Timedelta(days=(4 - dow) % 7)
    # 3rd Friday = first Friday + 14 days
    return first_friday + pd.Timedelta(days=14)


def is_near_opex(dt, window=3):
    """Return True if dt is within `window` trading days of monthly OPEX."""
    opex = get_monthly_opex(dt.year, dt.month)
    diff = abs((dt - opex).days)
    return diff <= window


def is_month_start(dt, close_index, n_days=5):
    """Return True if dt is within the first n_days trading days of its calendar month."""
    month_start = pd.Timestamp(dt.year, dt.month, 1)
    month_trading = close_index[(close_index >= month_start) & (close_index.month == dt.month)]
    if len(month_trading) == 0:
        return False
    pos = list(month_trading).index(dt) if dt in month_trading else -1
    if pos == -1:
        # Find nearest
        dists = abs(month_trading - dt)
        pos = dists.argmin()
    return pos < n_days


def is_sell_in_may_period(dt):
    """Return True if dt is in May-October (Sell-in-May period)."""
    return dt.month >= 5 and dt.month <= 10


def is_quarter_end_window(dt, close_index, n_days=5):
    """Return True if dt is within the last n_days trading days of a quarter."""
    q_end_month = ((dt.month - 1) // 3 + 1) * 3  # 3, 6, 9, 12
    if dt.month != q_end_month:
        return False
    # Last day of quarter-end month
    if q_end_month == 12:
        next_month = pd.Timestamp(dt.year + 1, 1, 1)
    else:
        next_month = pd.Timestamp(dt.year, q_end_month + 1, 1)
    month_trading = close_index[(close_index.month == q_end_month) &
                                (close_index.year == dt.year)]
    if len(month_trading) < n_days:
        return False
    last_n = month_trading[-n_days:]
    return dt in last_n


def is_low_vol_seasonal_month(dt):
    """Return True if month is in Jun/Jul/Aug/Dec (historically low vol)."""
    return dt.month in (6, 7, 8, 12)


# ================================================================
# LGBM WALK-FORWARD RANKING
# ================================================================

def get_monthly_rebalance_dates(close):
    """Monthly rebalance -- first trading day of each month."""
    monthly = close.index.to_series().resample("MS").first().dropna()
    return pd.DatetimeIndex(monthly.values)


def get_weekly_fridays(close):
    """Every Friday -- for building training features."""
    return pd.DatetimeIndex(
        close.index.to_series().resample("W-FRI").last().dropna().values
    )


def build_feature_records(close, rebal_dates):
    """Build feature records for all rebalance dates."""
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
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
    """Walk-forward LGBM ranking with ~500d sliding window."""
    import lightgbm as lgb
    if len(df) < 100:
        return {}
    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    # Convert 500 trading days to approximate number of weekly periods
    # ~500d / 5 = ~100 weeks, but we use dates directly
    # Find how many date periods correspond to ~500 trading days
    wf_periods = max(20, min(len(dates) - 1, 100))  # ~100 weekly dates ~ 500 trading days

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


# ================================================================
# STRIKES + TRADE EXECUTION
# ================================================================

def compute_strikes(S, direction):
    """Compute strikes: 2% OTM, width = max($3, 3%)."""
    if direction == "bull":
        K1 = round(S * (1.0 + BASE_OTM_PCT), 2)
        w = max(BASE_WIDTH_FLOOR_DOLLARS, K1 * BASE_WIDTH_FLOOR_PCT)
        K2 = round(K1 + w, 2)
    else:
        K2 = round(S * (1.0 - BASE_OTM_PCT), 2)
        w = max(BASE_WIDTH_FLOOR_DOLLARS, K2 * BASE_WIDTH_FLOOR_PCT)
        K1 = round(K2 - w, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity,
                  max_pos, size_scale=1.0):
    """Execute a single spread trade with profit target check.

    size_scale: multiplier on position sizing (e.g. 0.5 for Sell-in-May).
    """
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

    if entry_cost_ps is None or entry_cost_ps <= 0:
        return None

    spread_width = abs(K2 - K1)
    cwr = entry_cost_ps / max(spread_width, 0.01)
    if cwr > COST_WIDTH_MAX:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT
    effective_max = max_pos * size_scale
    if total_cost <= 0 or total_cost > effective_max or total_cost > equity * 0.40:
        return None

    # Profit target exit logic
    exited_early = False
    exit_day_idx = ei
    exit_reason = "expiry"
    hold_days = DTE

    if max_profit_ps is not None and max_profit_ps > 0 and BASE_PROFIT_TARGET_PCT > 0:
        vix_series = close["VIX"] if "VIX" in close.columns else None
        for check_idx in range(di + 1, ei + 1):
            if check_idx >= len(close):
                break
            check_date = close.index[check_idx]
            dte_remaining = ei - check_idx
            days_held = check_idx - di

            S_now = float(close[tk].iloc[check_idx])
            av_now = float(atr_dict[tk].loc[check_date]) if check_date in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[check_date]) else S_now * 0.015
            vix_now = float(vix_series.iloc[check_idx]) if vix_series is not None and check_idx < len(vix_series) else vix_val

            current_value = _revalue_spread_bs(S_now, K1, K2, dte_remaining, av_now, vix_now, direction)
            unrealized_gain = current_value - entry_cost_ps

            if unrealized_gain >= BASE_PROFIT_TARGET_PCT * max_profit_ps:
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
        vix_exit = float(close["VIX"].iloc[exit_day_idx]) if "VIX" in close.columns and exit_day_idx < len(close["VIX"]) else vix_val
        exit_value_ps = _revalue_spread_bs(Se, K1, K2, dte_at_exit, av_exit, vix_exit, direction)
        pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT - EARLY_EXIT_COMMISSION
    else:
        if direction == "bull":
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
        else:
            intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT
        exit_value_ps = intrinsic

    exit_date_ts = close.index[exit_day_idx]
    return {
        "pnl": round(pnl, 2),
        "entry_cost_ps": round(entry_cost_ps, 4),
        "total_cost": round(total_cost, 2),
        "K1": K1, "K2": K2,
        "S_entry": round(S, 2), "S_exit": round(Se, 2),
        "spread_width": round(spread_width, 2),
        "cost_width_ratio": round(cwr, 3),
        "exited_early": exited_early,
        "exit_reason": exit_reason,
        "hold_days": hold_days,
        "entry_date": str(dt.date()),
        "exit_date": str(exit_date_ts.date()),
        "ticker": tk,
        "direction": direction,
        "vix": round(vix_val, 1),
        "win": pnl > 0,
        "size_scale": size_scale,
    }


# ================================================================
# VARIANT DEFINITIONS
# ================================================================

VARIANT_DEFS = [
    ("A_baseline", "V10 baseline -- no timing filter (control)"),
    ("B_opex_avoid", "OPEX avoidance -- skip rebalance within 3d of monthly OPEX"),
    ("C_month_start", "Month-start entry -- only enter in first 5 trading days"),
    ("D_sell_in_may", "Sell-in-May -- 50% size May-Oct, full Nov-Apr"),
    ("E_qtr_end_mom", "Quarter-end momentum -- double positions last 5d of quarter"),
    ("F_vix_seasonal", "VIX seasonality -- skip bears if VIX<25pct AND low-vol month"),
]


def simulate_variant(variant_name, rankings, close, atr_dict, rebal_dates):
    """Run backtest simulation for a seasonal timing variant."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    # Precompute VIX 63d rolling 25th percentile for variant F
    vix_25pct = None
    if vix is not None:
        vix_25pct = vix.rolling(63, min_periods=30).quantile(0.25)

    equity = CAP
    trades = []
    skipped_timing = 0

    for dt in sorted(rebal_dates):
        if dt not in spy.index:
            continue

        # Find most recent ranking
        ranking_date = None
        for rd in sorted(rankings.keys()):
            if rd <= dt:
                ranking_date = rd
        if ranking_date is None:
            continue
        scores = rankings[ranking_date]
        if not scores or len(scores) < (BASE_TOP_K * 2):
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        # ----------------------------------------------------------
        # TIMING FILTERS (variant-specific)
        # ----------------------------------------------------------
        top_k = BASE_TOP_K
        size_scale = 1.0
        skip_bears = False
        skip_all = False

        if variant_name == "B_opex_avoid":
            if is_near_opex(dt, window=3):
                skipped_timing += 1
                continue  # skip entire rebalance

        elif variant_name == "C_month_start":
            if not is_month_start(dt, close.index, n_days=5):
                skipped_timing += 1
                continue

        elif variant_name == "D_sell_in_may":
            if is_sell_in_may_period(dt):
                size_scale = 0.50

        elif variant_name == "E_qtr_end_mom":
            if is_quarter_end_window(dt, close.index, n_days=5):
                top_k = 8  # double positions

        elif variant_name == "F_vix_seasonal":
            if (vix_25pct is not None and dt in vix_25pct.index and
                    not pd.isna(vix_25pct.loc[dt]) and
                    cv < vix_25pct.loc[dt] and
                    is_low_vol_seasonal_month(dt)):
                skip_bears = True

        # Rank sectors
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:top_k]]
        bear_picks = [t for t, _ in ranked_asc[:top_k]]

        if skip_bears:
            bear_picks = []

        n_positions = len(bull_picks) + len(bear_picks)
        if n_positions == 0:
            continue

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                max_pos = min(MAX_PER_TRADE, equity * 0.40)
                if max_pos < 20:
                    continue

                result = execute_trade(
                    tk, dt, direction, close, atr_dict, cv, equity,
                    max_pos, size_scale=size_scale,
                )
                if result is not None:
                    equity += result["pnl"]
                    # Add regime label
                    di = close.index.get_loc(dt)
                    ei = min(di + DTE, len(close) - 1)
                    sv = float(spy.loc[dt])
                    se = float(spy.iloc[ei]) if ei < len(spy) else sv
                    result["regime"] = "bull" if se >= sv else "bear"
                    result["n_positions"] = n_positions
                    trades.append(result)

    return trades, equity, skipped_timing


# ================================================================
# STATS
# ================================================================

def compute_full_stats(trades, initial_capital=CAP):
    """Compute comprehensive stats."""
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
    gw = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0
    gl = float(abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-10
    pf = gw / max(gl, 1e-10)

    # Calendar month Sharpe (PRIMARY METRIC per spec)
    trade_df = pd.DataFrame(trades)
    trade_df["entry_month"] = pd.to_datetime(trade_df["entry_date"]).dt.to_period("M")
    monthly_pnl = trade_df.groupby("entry_month")["pnl"].sum()
    if len(monthly_pnl) > 1:
        sharpe = float(monthly_pnl.mean() / (monthly_pnl.std() + 1e-10) * np.sqrt(12))
    else:
        sharpe = 0.0

    # Sortino
    neg_ret = monthly_pnl[monthly_pnl < 0]
    if len(neg_ret) > 1:
        sortino = float(monthly_pnl.mean() / (neg_ret.std() + 1e-10) * np.sqrt(12))
    else:
        sortino = sharpe * 1.5

    # Calmar
    max_dd_pct = float(dd.min()) * 100
    dates_str = [t["entry_date"] for t in trades]
    first_date = pd.Timestamp(min(dates_str))
    last_date = pd.Timestamp(max(dates_str))
    years = (last_date - first_date).days / 365.25
    total_ret = equity[-1] / equity[0]
    cagr = total_ret ** (1 / max(years, 0.5)) - 1 if years > 0.5 else 0.0
    calmar = cagr / (abs(dd.min()) + 1e-10) if abs(dd.min()) > 1e-10 else 0.0

    early_exits = sum(1 for t in trades if t.get("exited_early", False))
    early_exit_rate = early_exits / len(trades)
    avg_hold = float(np.mean([t.get("hold_days", DTE) for t in trades]))

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
        "final_equity": round(equity[-1], 2),
        "early_exit_rate": round(early_exit_rate, 4),
        "avg_hold_days": round(avg_hold, 1),
        "early_exits": early_exits,
        "cagr": round(cagr, 4),
    }


def monte_carlo_bootstrap(trades, n_resamples=N_BOOTSTRAP, initial_capital=CAP):
    """Monte Carlo bootstrap resampling of trade-level P&L."""
    if not trades or len(trades) < 10:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    rng = np.random.RandomState(42)

    boot_sharpe = []
    boot_pf = []
    boot_wr = []
    boot_pnl = []

    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        sp = pnls[idx]
        boot_wr.append(float(np.mean(sp > 0)))
        gw = float(sp[sp > 0].sum()) if (sp > 0).any() else 0
        gl = float(abs(sp[sp < 0].sum())) if (sp < 0).any() else 1e-10
        boot_pf.append(gw / max(gl, 1e-10))
        boot_pnl.append(float(sp.sum()))
        # Chunked monthly sharpe
        n_months = max(3, n // 6)
        cs = max(1, n // n_months)
        mp = []
        for j in range(0, n, cs):
            mp.append(sp[j:j+cs].sum())
        mp = np.array(mp)
        if len(mp) > 1 and mp.std() > 1e-10:
            boot_sharpe.append(float(mp.mean() / mp.std() * np.sqrt(12)))
        else:
            boot_sharpe.append(0.0)

    bs = np.array(boot_sharpe)
    bp = np.array(boot_pf)
    bw = np.array(boot_wr)
    bt = np.array(boot_pnl)

    return {
        "sharpe_mean": round(float(bs.mean()), 3),
        "sharpe_std": round(float(bs.std()), 3),
        "sharpe_ci_5": round(float(np.percentile(bs, 5)), 3),
        "sharpe_ci_95": round(float(np.percentile(bs, 95)), 3),
        "sharpe_pct_positive": round(float(np.mean(bs > 0) * 100), 1),
        "pf_mean": round(float(bp.mean()), 3),
        "pf_ci_5": round(float(np.percentile(bp, 5)), 3),
        "pf_ci_95": round(float(np.percentile(bp, 95)), 3),
        "wr_mean": round(float(bw.mean()), 4),
        "wr_ci_5": round(float(np.percentile(bw, 5)), 4),
        "wr_ci_95": round(float(np.percentile(bw, 95)), 4),
        "total_pnl_mean": round(float(bt.mean()), 2),
        "total_pnl_ci_5": round(float(np.percentile(bt, 5)), 2),
        "total_pnl_ci_95": round(float(np.percentile(bt, 95)), 2),
        "pct_profitable": round(float(np.mean(bt > 0) * 100), 1),
    }


def compute_regime_stats(trades):
    """Split trades by VIX regime (high >= 20, low < 20)."""
    if not trades:
        return None, None
    high_vix = [t for t in trades if t.get("vix", 20) >= VIX_HIGH_THRESHOLD]
    low_vix = [t for t in trades if t.get("vix", 20) < VIX_HIGH_THRESHOLD]
    return compute_full_stats(high_vix), compute_full_stats(low_vix)


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint("V10 SEASONAL TIMING V1 -- %s" % t0.strftime("%Y-%m-%d %H:%M:%S"))
    fprint("=" * 100)
    fprint("KEY QUESTION: Can calendar/seasonal timing improve V10 sector rotation?")
    fprint("")
    fprint("Capital: $%.0f | DTE: %d | Commission: $%.2f | Haircut: %.0f%%" % (CAP, DTE, COMMISSION_RT, HAIRCUT * 100))
    fprint("OTM: %.0f%% | Width: max($%.0f,%.0f%%) | PT: %.0f%% | top_k: %d" % (
        BASE_OTM_PCT * 100, BASE_WIDTH_FLOOR_DOLLARS, BASE_WIDTH_FLOOR_PCT * 100,
        BASE_PROFIT_TARGET_PCT * 100, BASE_TOP_K))
    fprint("Walk-forward: ~%dd sliding | LGBM %d features" % (WF_TRAIN_DAYS, len(V6_FEATURES)))
    fprint("Permutation trials: %d | Bootstrap: %d resamples" % (N_PERMUTATIONS, N_BOOTSTRAP))
    fprint("")
    fprint("6 VARIANTS:")
    for vn, desc in VARIANT_DEFS:
        fprint("  %s: %s" % (vn, desc))
    fprint("")

    # Download data
    fprint("Downloading price data...")
    close, high, low = download_data()
    atr_dict = compute_atr_series(high, low, close)

    monthly_dates = get_monthly_rebalance_dates(close)
    weekly_dates = get_weekly_fridays(close)
    fprint("Rebalance: %d monthly dates | Feature dates: %d weekly" % (len(monthly_dates), len(weekly_dates)))

    # Build features and rankings (shared across all variants)
    fprint("")
    fprint("=" * 80)
    fprint("BUILDING LGBM RANKINGS (~%dd walk-forward, %d features)" % (WF_TRAIN_DAYS, len(V6_FEATURES)))
    fprint("=" * 80)
    records = build_feature_records(close, weekly_dates)
    rankings = walk_forward_lgbm_rank(records)

    if len(rankings) < 20:
        fprint("ERROR: Only %d ranking dates -- insufficient data" % len(rankings))
        return

    spy_close = close["SPY"]

    all_trades = {}
    all_stats = {}
    all_validation = {}
    all_mc = {}
    all_regime = {}

    for var_name, description in VARIANT_DEFS:
        fprint("")
        fprint("=" * 90)
        fprint("%s: %s" % (var_name, description))
        fprint("=" * 90)

        trades, eq, skipped = simulate_variant(var_name, rankings, close, atr_dict, monthly_dates)
        fprint("  Trades: %d | Skipped (timing): %d | Final: $%s" % (
            len(trades), skipped, "{:,.0f}".format(eq)))

        all_trades[var_name] = trades
        st = compute_full_stats(trades)
        all_stats[var_name] = st

        if st:
            fprint("  Sharpe: %.2f | Sortino: %.2f | Calmar: %.2f" % (
                st["sharpe"], st["sortino"], st["calmar"]))
            fprint("  WR: %.1f%% | PF: %.2f | MDD: %.1f%%" % (
                st["win_rate"] * 100, st["profit_factor"], st["max_dd_pct"]))
            fprint("  Avg hold: %.1f days | Early exit rate: %.1f%%" % (
                st["avg_hold_days"], st["early_exit_rate"] * 100))

        # 5-gate validation
        if len(trades) >= 10:
            try:
                val = validate_trades_5gate(trades, CAP, spy_close, name=var_name)
                all_validation[var_name] = val
                fprint("  5-gate: %d/%d %s" % (
                    val["gates_passed"], val["gates_total"],
                    "PASS" if val["all_passed"] else "FAIL"))
                for g in val["gates"]:
                    fprint("    [%s] %s: %s=%.4f" % (
                        "PASS" if g["passed"] else "FAIL",
                        g["name"], g["metric_name"], g["metric_value"]))
            except Exception as e:
                fprint("  Validation error: %s" % e)

        # Monte Carlo
        fprint("  Monte Carlo (%d resamples)..." % N_BOOTSTRAP)
        mc = monte_carlo_bootstrap(trades)
        if mc:
            all_mc[var_name] = mc
            fprint("    Sharpe: %.2f +/- %.2f [%.2f, %.2f] (%.0f%% pos)" % (
                mc["sharpe_mean"], mc["sharpe_std"],
                mc["sharpe_ci_5"], mc["sharpe_ci_95"], mc["sharpe_pct_positive"]))
            fprint("    PnL: $%s [$%s, $%s] (%.0f%% profitable)" % (
                "{:,.0f}".format(mc["total_pnl_mean"]),
                "{:,.0f}".format(mc["total_pnl_ci_5"]),
                "{:,.0f}".format(mc["total_pnl_ci_95"]),
                mc["pct_profitable"]))

        # VIX regime
        hv, lv = compute_regime_stats(trades)
        all_regime[var_name] = {"high_vix": hv, "low_vix": lv}
        if hv and lv:
            fprint("  VIX regimes:")
            fprint("    High (>=20): %d trades, Sh %.2f, WR %.1f%%" % (
                hv["n_trades"], hv["sharpe"], hv["win_rate"] * 100))
            fprint("    Low  (<20):  %d trades, Sh %.2f, WR %.1f%%" % (
                lv["n_trades"], lv["sharpe"], lv["win_rate"] * 100))

        # Side breakdown
        for side in ["bull", "bear"]:
            st_side = [t for t in trades if t["direction"] == side]
            if st_side:
                sp = [t["pnl"] for t in st_side]
                wr_s = sum(1 for p in sp if p > 0) / len(sp)
                fprint("  %s: %d trades, WR %.1f%%, PnL $%s" % (
                    side, len(st_side), wr_s * 100, "{:,.0f}".format(sum(sp))))

    # ==========================================================
    # COMPARISON TABLE (sorted by Sharpe)
    # ==========================================================
    fprint("")
    fprint("=" * 150)
    fprint("COMPARISON TABLE -- SORTED BY CALENDAR MONTH SHARPE (primary metric)")
    fprint("=" * 150)

    hdr = "  %-20s %5s %7s %7s %7s %6s %6s %7s %7s %5s %9s" % (
        "Variant", "N", "Sharpe", "Sort", "Calmar", "WR", "PF", "MDD", "AvgHld", "Gate", "Final$")
    fprint(hdr)
    fprint("  " + "-" * 140)

    # Sort by Sharpe
    sorted_variants = sorted(VARIANT_DEFS, key=lambda x: all_stats.get(x[0], {}).get("sharpe", -999) if all_stats.get(x[0]) else -999, reverse=True)

    for var_name, _ in sorted_variants:
        s = all_stats.get(var_name)
        if s is None:
            continue
        val = all_validation.get(var_name)
        gs = "%d/%d" % (val["gates_passed"], val["gates_total"]) if val else "--"
        fprint("  %-20s %5d %7.2f %7.2f %7.2f %5.1f%% %5.2f %6.1f%% %6.1fd %5s $%8s" % (
            var_name, s["n_trades"], s["sharpe"], s["sortino"], s["calmar"],
            s["win_rate"] * 100, s["profit_factor"], s["max_dd_pct"],
            s["avg_hold_days"], gs, "{:,.0f}".format(s["final_equity"])))

    # ==========================================================
    # DELTA vs BASELINE
    # ==========================================================
    fprint("")
    fprint("=" * 100)
    fprint("DELTA vs A_baseline (control)")
    fprint("=" * 100)

    base = all_stats.get("A_baseline")
    if base:
        fprint("")
        fprint("  %-20s %8s %8s %7s %7s %7s %9s" % (
            "Variant", "dSharpe", "dSort", "dWR", "dPF", "dMDD", "dPnL"))
        fprint("  " + "-" * 80)
        for var_name, _ in VARIANT_DEFS[1:]:
            s = all_stats.get(var_name)
            if s is None:
                continue
            fprint("  %-20s %+7.2f %+7.2f %+6.1f%% %+6.2f %+6.1f%% $%+8s" % (
                var_name,
                s["sharpe"] - base["sharpe"],
                s["sortino"] - base["sortino"],
                (s["win_rate"] - base["win_rate"]) * 100,
                s["profit_factor"] - base["profit_factor"],
                s["max_dd_pct"] - base["max_dd_pct"],
                "{:,.0f}".format(s["total_pnl"] - base["total_pnl"])))

    # ==========================================================
    # MONTE CARLO COMPARISON
    # ==========================================================
    fprint("")
    fprint("=" * 130)
    fprint("MONTE CARLO BOOTSTRAP COMPARISON (%d resamples)" % N_BOOTSTRAP)
    fprint("=" * 130)

    fprint("")
    fprint("  %-20s %12s %16s %6s %8s %16s %22s" % (
        "Variant", "Sharpe", "95%CI", "%Pos", "PF", "95%CI", "PnL 95%CI"))
    fprint("  " + "-" * 110)

    for var_name, _ in VARIANT_DEFS:
        mc = all_mc.get(var_name)
        if mc is None:
            continue
        fprint("  %-20s %6.2f+-%.2f [%6.2f,%6.2f] %5.0f%% %7.2f [%6.2f,%6.2f] [$%8s,$%8s]" % (
            var_name,
            mc["sharpe_mean"], mc["sharpe_std"],
            mc["sharpe_ci_5"], mc["sharpe_ci_95"],
            mc["sharpe_pct_positive"],
            mc["pf_mean"], mc["pf_ci_5"], mc["pf_ci_95"],
            "{:,.0f}".format(mc["total_pnl_ci_5"]),
            "{:,.0f}".format(mc["total_pnl_ci_95"])))

    # ==========================================================
    # VIX REGIME COMPARISON
    # ==========================================================
    fprint("")
    fprint("=" * 120)
    fprint("VIX REGIME COMPARISON (High >= 20 vs Low < 20)")
    fprint("=" * 120)

    fprint("")
    fprint("  %-20s | %-40s | %-40s" % ("Variant", "HIGH VIX (>=20)", "LOW VIX (<20)"))
    fprint("  " + "-" * 110)

    for var_name, _ in VARIANT_DEFS:
        rs = all_regime.get(var_name, {})
        hv = rs.get("high_vix")
        lv = rs.get("low_vix")
        hv_str = "N=%d Sh=%.2f WR=%.0f%% PF=%.2f" % (
            hv["n_trades"], hv["sharpe"], hv["win_rate"] * 100, hv["profit_factor"]
        ) if hv else "N/A"
        lv_str = "N=%d Sh=%.2f WR=%.0f%% PF=%.2f" % (
            lv["n_trades"], lv["sharpe"], lv["win_rate"] * 100, lv["profit_factor"]
        ) if lv else "N/A"
        fprint("  %-20s | %-40s | %-40s" % (var_name, hv_str, lv_str))

    # ==========================================================
    # 5-GATE SUMMARY
    # ==========================================================
    fprint("")
    fprint("=" * 100)
    fprint("5-GATE ADVERSARIAL VALIDATION SUMMARY")
    fprint("=" * 100)

    for var_name, _ in VARIANT_DEFS:
        val = all_validation.get(var_name)
        if not val:
            fprint("  %s: No validation" % var_name)
            continue
        status = "PASS" if val["all_passed"] else "FAIL"
        fprint("  [%s] %s: %d/%d gates" % (status, var_name, val["gates_passed"], val["gates_total"]))

    # ==========================================================
    # VERDICT
    # ==========================================================
    fprint("")
    fprint("=" * 100)
    fprint("SEASONAL TIMING VERDICT")
    fprint("=" * 100)

    best_variant = None
    best_sharpe = -999
    for var_name, _ in VARIANT_DEFS:
        s = all_stats.get(var_name)
        if s and s["sharpe"] > best_sharpe:
            best_sharpe = s["sharpe"]
            best_variant = var_name

    fprint("")
    fprint("  BEST VARIANT: %s (Sharpe %.2f)" % (best_variant, best_sharpe))

    if base:
        fprint("  Baseline (A): Sharpe %.2f, Sortino %.2f, PF %.2f, WR %.1f%%" % (
            base["sharpe"], base["sortino"], base["profit_factor"], base["win_rate"] * 100))

    if base and best_variant != "A_baseline":
        best_s = all_stats[best_variant]
        delta = best_s["sharpe"] - base["sharpe"]
        mc_best = all_mc.get(best_variant)
        val_best = all_validation.get(best_variant)

        fprint("")
        fprint("  IMPROVEMENT ASSESSMENT:")
        fprint("    Sharpe delta: %+.2f %s" % (delta, "(meaningful)" if abs(delta) > 0.10 else "(marginal)"))
        if mc_best:
            fprint("    MC Sharpe CI5: %.2f %s" % (mc_best["sharpe_ci_5"],
                   "(robust)" if mc_best["sharpe_ci_5"] > 0 else "(fragile)"))
        if val_best:
            fprint("    5-gate: %d/%d %s" % (val_best["gates_passed"], val_best["gates_total"],
                   "(validated)" if val_best["all_passed"] else "(concerns)"))

        if delta > 0.10 and mc_best and mc_best["sharpe_ci_5"] > 0:
            fprint("")
            fprint("  ==> RECOMMENDATION: Consider %s as V10 timing enhancement" % best_variant)
        else:
            fprint("")
            fprint("  ==> RECOMMENDATION: Timing filters do NOT reliably improve V10")
            fprint("     Baseline (no filter) remains the best risk-adjusted approach")
    else:
        fprint("")
        fprint("  ==> CONCLUSION: No seasonal timing variant beats baseline")

    # ==========================================================
    # SAVE RESULTS
    # ==========================================================
    save_results = {
        "timestamp": t0.isoformat(),
        "experiment": EXPERIMENT_NAME,
        "question": "Can calendar/seasonal timing improve V10 sector rotation?",
        "config": {
            "capital": CAP, "dte": DTE, "commission": COMMISSION_RT,
            "haircut": HAIRCUT, "otm_pct": BASE_OTM_PCT,
            "width_floor_dollars": BASE_WIDTH_FLOOR_DOLLARS,
            "width_floor_pct": BASE_WIDTH_FLOOR_PCT,
            "profit_target_pct": BASE_PROFIT_TARGET_PCT,
            "top_k": BASE_TOP_K, "rebalance": "monthly",
            "wf_train_days": WF_TRAIN_DAYS,
            "n_features": len(V6_FEATURES),
            "n_bootstrap": N_BOOTSTRAP,
            "n_permutations": N_PERMUTATIONS,
        },
        "best_variant": best_variant,
        "best_sharpe": best_sharpe,
    }

    for var_name, desc in VARIANT_DEFS:
        s = all_stats.get(var_name)
        if s:
            save_results[var_name] = s
        val = all_validation.get(var_name)
        if val:
            save_results["%s_validation" % var_name] = val
        mc = all_mc.get(var_name)
        if mc:
            save_results["%s_monte_carlo" % var_name] = mc
        rs = all_regime.get(var_name, {})
        if rs.get("high_vix"):
            save_results["%s_high_vix" % var_name] = rs["high_vix"]
        if rs.get("low_vix"):
            save_results["%s_low_vix" % var_name] = rs["low_vix"]

    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint("")
    fprint("Results saved to %s" % results_file)

    # Save trade details per variant
    for var_name, _ in VARIANT_DEFS:
        trades = all_trades.get(var_name, [])
        if trades:
            tf = OUTPUT_DIR / ("trades_%s.json" % var_name)
            with open(tf, "w") as f:
                json.dump(trades, f, indent=1, default=str)

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="seasonal_timing_%s" % t0.strftime("%Y%m%d_%H%M")):
                mlflow.log_param("capital", CAP)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("commission", COMMISSION_RT)
                mlflow.log_param("rebalance", "monthly")
                mlflow.log_param("wf_train_days", WF_TRAIN_DAYS)
                mlflow.log_param("n_features", len(V6_FEATURES))
                mlflow.log_param("n_bootstrap", N_BOOTSTRAP)
                mlflow.log_param("n_permutations", N_PERMUTATIONS)
                mlflow.log_param("best_variant", best_variant)
                mlflow.log_param("question", "seasonal_timing_improvement")

                for var_name, _ in VARIANT_DEFS:
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

                    mc = all_mc.get(var_name)
                    if mc:
                        mlflow.log_metric("%s_mc_sharpe_mean" % prefix, mc["sharpe_mean"])
                        mlflow.log_metric("%s_mc_sharpe_ci5" % prefix, mc["sharpe_ci_5"])
                        mlflow.log_metric("%s_mc_pct_profitable" % prefix, mc["pct_profitable"])

                    val = all_validation.get(var_name)
                    if val:
                        mlflow.log_metric("%s_gates_passed" % prefix, val["gates_passed"])

                mlflow.log_artifact(str(results_file))

            exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if exp:
                fprint("MLflow experiment ID: %s" % exp.experiment_id)
            fprint("MLflow logged to experiment '%s'" % EXPERIMENT_NAME)
        except Exception as e:
            fprint("MLflow error: %s" % e)

    elapsed = (datetime.now() - t0).total_seconds()
    fprint("")
    fprint("Completed in %.1f minutes" % (elapsed / 60))


if __name__ == "__main__":
    main()
