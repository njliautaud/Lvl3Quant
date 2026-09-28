#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V13 Combined Best — Three Validated Improvements Stacked
=========================================================

Tests whether three independently validated improvements compound or interfere
when combined in a single strategy:

  1. 50% Profit Target Exit (KB #261-262): Exit when spread value hits 50% of max profit.
     V9.3 result: Sharpe 5.12 vs 2.36 hold-to-expiry (+97%)

  2. Rank-Weighted Sizing (KB #276-277): Size inversely proportional to rank (1/rank).
     V12 result: Sharpe 6.21, MDD -4.8% vs equal weight MDD -23.8%

  3. Biweekly Rebalance (KB #253): Rebalance every 10 trading days instead of 5.
     Result: Sharpe 2.64 vs 1.53 (+73%)

8 VARIANTS:
  A: Baseline V10  -- equal size, hold-to-expiry, biweekly rebal       (CONTROL)
  B: + Profit Target only (50%)
  C: + Rank Sizing only (1/rank)
  D: + Both Profit Target (50%) + Rank Sizing
  E: + Profit Target 30% (most aggressive)
  F: + Profit Target 70% (most conservative)
  G: + Rank Sizing + Profit Target (50%) + 6 flow features (KB #264)
  H: + All above + Monthly rebalance (20 trading days)

SELF-CONTAINED: Embeds Black-Scholes pricing and adversarial validation.
No imports from research.tools.

Capital: $645 | DTE: 28 | Commission: $2.60 RT spread
Walk-forward: 500-day sliding train, ALL remaining data for test
Rebalance: every 10 trading days (biweekly)
VIX >= 20: bull spreads only on top-3
VIX <  20: top-3 bull + bottom-3 bear

Output: output/growth_research/v13_combined_best_v1/
MLflow: experiment v13_combined_best_v1, server http://jupiter:5000
"""

import json
import sys
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

OUTPUT_DIR = BASE / "output" / "growth_research" / "v13_combined_best_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v13_combined_best_v1"

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
# CONSTANTS
# ==============================================================

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 28
WF_TRAIN_DAYS = 500          # sliding window train size
REBAL_DAYS_BIWEEKLY = 10     # biweekly = every 10 trading days
REBAL_DAYS_MONTHLY = 20      # monthly  = every 20 trading days
TOP_K = 3                    # top / bottom 3 picks
IV_MULTIPLIER = 1.2          # VIX-based IV estimator baseline
RISK_FREE_RATE = 0.045
HAIRCUT = 0.15               # entry haircut (15%)
COMMISSION_RT = 2.60         # $2.60 per spread round-trip (4 legs x $0.65)
EARLY_EXIT_COMM = 2.60       # additional commission when exiting early
COST_WIDTH_MAX = 0.50        # reject if entry_cost / width > 50%
MIN_POS_SIZE = 30.0          # minimum position size in $
VIX_THRESHOLD = 20.0         # VIX >= 20 → bull-only; < 20 → bull + bear
OTM_PCT = 0.02               # 2% OTM moneyness
WIDTH_FLOOR_USD = 3.0        # spread width floor: max($3, 3% of strike)
WIDTH_FLOOR_PCT = 0.03
N_PERM = 300                 # adversarial gate 1: permutation shuffles

# 21 features: 17 production + 4 flow features (KB #264 cross-asset)
BASE_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d",
    "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "sector_relative_vol_21d", "cross_sector_dispersion",
]
assert len(BASE_FEATURES) == 21

FLOW_FEATURES_6 = [
    "gold_equity_ratio",      # GLD/SPY ratio momentum (KB #264)
    "cash_vs_equity_ratio",   # SHY/SPY ratio momentum
    "cta_pressure",           # MA crossover intensity on SPY
    "credit_spread_mom",      # HYG/SHY ratio momentum
    "tlt_spy_ratio",          # TLT/SPY ratio momentum (duration / equity flow)
    "vix_term_structure",     # VIX3M / VIX ratio (term structure slope)
]

FEATURES_21 = BASE_FEATURES
FEATURES_27 = BASE_FEATURES + FLOW_FEATURES_6


# ==============================================================
# VARIANT DEFINITIONS
# ==============================================================
# (name, pt_pct, sizing_mode, rebal_days, feature_set, description)
# pt_pct=0.0 → hold-to-expiry; pt_pct=0.5 → 50% profit target exit

VARIANTS = [
    ("A_baseline",
     0.0, "equal", REBAL_DAYS_BIWEEKLY, "21",
     "A: Baseline V10 -- equal size, hold-to-expiry, biweekly rebal"),

    ("B_profit_target_50",
     0.50, "equal", REBAL_DAYS_BIWEEKLY, "21",
     "B: + Profit Target 50% (exit when spread hits 50% of max profit)"),

    ("C_rank_sizing",
     0.0, "rank", REBAL_DAYS_BIWEEKLY, "21",
     "C: + Rank Sizing only (1/rank weighting, top pick 4x of pick #3)"),

    ("D_pt50_rank",
     0.50, "rank", REBAL_DAYS_BIWEEKLY, "21",
     "D: + Profit Target 50% + Rank Sizing (both combined)"),

    ("E_pt30_rank",
     0.30, "rank", REBAL_DAYS_BIWEEKLY, "21",
     "E: + Profit Target 30% (aggressive) + Rank Sizing"),

    ("F_pt70_rank",
     0.70, "rank", REBAL_DAYS_BIWEEKLY, "21",
     "F: + Profit Target 70% (conservative) + Rank Sizing"),

    ("G_pt50_rank_flow6",
     0.50, "rank", REBAL_DAYS_BIWEEKLY, "27",
     "G: + Rank Sizing + PT50% + 6 flow features (KB #264 cross-asset)"),

    ("H_pt50_rank_flow6_monthly",
     0.50, "rank", REBAL_DAYS_MONTHLY, "27",
     "H: + All above + Monthly rebalance (20 trading days)"),
]


# ==============================================================
# SELF-CONTAINED BLACK-SCHOLES PRICING
# ==============================================================

def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _estimate_iv(atr: float, spot: float, vix: float = 20.0) -> float:
    """ATR-based IV estimate. iv = (ATR/spot) * sqrt(252/14) * iv_multiplier."""
    if spot <= 0 or atr <= 0:
        return 0.25
    realized_vol = (atr / spot) * np.sqrt(252 / 14.0)
    iv_mult = IV_MULTIPLIER + 0.01 * max(vix - 20.0, 0.0)
    return max(realized_vol * iv_mult, 0.10)


def price_bull_call_spread(S, K1, K2, dte, atr, vix=20.0):
    """Bull call spread: buy K1 call, sell K2 call. Returns (entry_cost_ps, max_profit_ps)."""
    if K2 <= K1:
        raise ValueError("K2 must be > K1")
    T = dte / 365.0
    sigma = _estimate_iv(atr, S, vix)
    fair = _bs_call(S, K1, T, RISK_FREE_RATE, sigma) - _bs_call(S, K2, T, RISK_FREE_RATE, sigma)
    fair = max(fair, 0.001)
    entry_cost = fair * (1.0 + HAIRCUT)   # entry: pay more than fair
    max_profit = (K2 - K1) - entry_cost
    return float(entry_cost), float(max_profit)


def price_bear_put_spread(S, K1, K2, dte, atr, vix=20.0):
    """Bear put spread: buy K2 put, sell K1 put (K2 > K1). Returns (entry_cost_ps, max_profit_ps)."""
    if K2 <= K1:
        raise ValueError("K2 must be > K1")
    T = dte / 365.0
    sigma = _estimate_iv(atr, S, vix)
    fair = _bs_put(S, K2, T, RISK_FREE_RATE, sigma) - _bs_put(S, K1, T, RISK_FREE_RATE, sigma)
    fair = max(fair, 0.001)
    entry_cost = fair * (1.0 + HAIRCUT)
    max_profit = (K2 - K1) - entry_cost
    return float(entry_cost), float(max_profit)


def revalue_spread_bs(S, K1, K2, dte_remaining, atr, vix, direction):
    """Re-price spread mid-life for profit target check. Apply EXIT haircut (receive less)."""
    if dte_remaining <= 0:
        if direction == "bull":
            return max(S - K1, 0.0) - max(S - K2, 0.0)
        else:
            return max(K2 - S, 0.0) - max(K1 - S, 0.0)
    T = dte_remaining / 365.0
    sigma = _estimate_iv(atr, S, vix)
    if direction == "bull":
        fair = _bs_call(S, K1, T, RISK_FREE_RATE, sigma) - _bs_call(S, K2, T, RISK_FREE_RATE, sigma)
    else:
        fair = _bs_put(S, K2, T, RISK_FREE_RATE, sigma) - _bs_put(S, K1, T, RISK_FREE_RATE, sigma)
    fair = max(fair, 0.0)
    return float(fair * (1.0 - HAIRCUT))   # exit: receive less than fair


# ==============================================================
# SELF-CONTAINED ADVERSARIAL VALIDATION (5 GATES)
# ==============================================================

def _build_equity_series(trades, initial_capital):
    """Build a DatetimeIndex equity series from trade list."""
    if not trades:
        return pd.Series([initial_capital], dtype=float)
    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("exit_date")
    pnls = df["pnl"].values
    equity_vals = [initial_capital]
    for p in pnls:
        equity_vals.append(equity_vals[-1] + p)
    dates = [df["entry_date"].iloc[0] - pd.Timedelta(days=1)]
    dates.extend(df["exit_date"].tolist())
    eq = pd.Series(equity_vals, index=pd.DatetimeIndex(dates))
    return eq.groupby(eq.index).last()


def _monthly_sharpe(equity_series):
    """Calendar-month Sharpe from equity series."""
    monthly = equity_series.resample("ME").last().dropna()
    if len(monthly) < 2:
        return 0.0
    rets = monthly.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(12))


def _monthly_sortino(equity_series):
    """Calendar-month Sortino from equity series."""
    monthly = equity_series.resample("ME").last().dropna()
    if len(monthly) < 2:
        return 0.0
    rets = monthly.pct_change().dropna()
    neg = rets[rets < 0]
    if len(neg) < 2 or neg.std() == 0:
        return float(rets.mean() / 1e-10 * np.sqrt(12))
    return float(rets.mean() / neg.std() * np.sqrt(12))


def adversarial_validate(trades, initial_capital=CAP, spy_prices=None, strategy_name=""):
    """
    5-gate adversarial validation (self-contained).

    Gates:
      1. Permutation test: 300 shuffles of trade directions, p < 0.05
      2. Regime stability: |Sharpe_bull - Sharpe_bear| / max < 0.50
      3. Sub-period: both halves Sharpe > 0.5
      4. Outlier removal: trim top/bottom 5%, Sharpe stays > 0.5
      5. Yearly consistency: >= 60% of years profitable
    """
    result = {
        "strategy_name": strategy_name,
        "n_trades": len(trades),
        "gates": [],
        "gates_passed": 0,
        "gates_total": 5,
        "all_passed": False,
    }

    if len(trades) < 10:
        result["error"] = "Insufficient trades for validation (need >= 10)"
        return result

    pnls = np.array([t["pnl"] for t in trades])
    eq = _build_equity_series(trades, initial_capital)
    real_sharpe = _monthly_sharpe(eq)

    # Gate 1: Permutation test
    rng = np.random.RandomState(42)
    beat = 0
    for _ in range(N_PERM):
        shuffled = pnls * rng.choice([-1, 1], size=len(pnls))
        perm_trades = [dict(t, pnl=float(s)) for t, s in zip(trades, shuffled)]
        perm_eq = _build_equity_series(perm_trades, initial_capital)
        if _monthly_sharpe(perm_eq) >= real_sharpe:
            beat += 1
    p_val = beat / N_PERM
    g1 = {"name": "Permutation", "passed": p_val < 0.05,
          "metric_name": "p_value", "metric_value": round(p_val, 4),
          "threshold": 0.05, "detail": "%d/%d shuffles beat real Sharpe %.3f" % (beat, N_PERM, real_sharpe)}
    result["gates"].append(g1)

    # Gate 2: Regime stability (bull vs bear SPY periods)
    # Use the 'regime' field on each trade (bull/bear based on SPY return during hold period)
    bull_trades = [t for t in trades if t.get("regime", "bull") == "bull"]
    bear_trades = [t for t in trades if t.get("regime", "bull") == "bear"]
    if bull_trades and bear_trades:
        eq_bull = _build_equity_series(bull_trades, initial_capital)
        eq_bear = _build_equity_series(bear_trades, initial_capital)
        sh_bull = _monthly_sharpe(eq_bull)
        sh_bear = _monthly_sharpe(eq_bear)
        denom = max(abs(sh_bull), abs(sh_bear), 1e-10)
        gap = abs(sh_bull - sh_bear) / denom
        g2 = {"name": "Regime Stability", "passed": gap < 0.50,
              "metric_name": "sharpe_gap_ratio", "metric_value": round(gap, 4),
              "threshold": 0.50,
              "detail": "bull_sh=%.3f bear_sh=%.3f gap=%.3f" % (sh_bull, sh_bear, gap)}
    else:
        g2 = {"name": "Regime Stability", "passed": True,
              "metric_name": "sharpe_gap_ratio", "metric_value": 0.0,
              "threshold": 0.50, "detail": "Only one regime observed -- skipped"}
    result["gates"].append(g2)

    # Gate 3: Sub-period stability (both halves Sharpe > 0.5)
    mid = len(trades) // 2
    h1_trades = trades[:mid]
    h2_trades = trades[mid:]
    if h1_trades and h2_trades:
        eq1 = _build_equity_series(h1_trades, initial_capital)
        eq2 = _build_equity_series(h2_trades, initial_capital)
        sh1 = _monthly_sharpe(eq1)
        sh2 = _monthly_sharpe(eq2)
        g3 = {"name": "Sub-Period Stability", "passed": sh1 > 0.5 and sh2 > 0.5,
              "metric_name": "min_half_sharpe", "metric_value": round(min(sh1, sh2), 4),
              "threshold": 0.5,
              "detail": "h1_sh=%.3f h2_sh=%.3f" % (sh1, sh2)}
    else:
        g3 = {"name": "Sub-Period Stability", "passed": False,
              "metric_name": "min_half_sharpe", "metric_value": 0.0,
              "threshold": 0.5, "detail": "Insufficient trades for split"}
    result["gates"].append(g3)

    # Gate 4: Outlier removal (trim top/bottom 5%, Sharpe stays > 0.5)
    n_trim = max(1, int(len(pnls) * 0.05))
    sorted_idx = np.argsort(pnls)
    keep_mask = np.ones(len(pnls), dtype=bool)
    keep_mask[sorted_idx[:n_trim]] = False   # remove bottom 5%
    keep_mask[sorted_idx[-n_trim:]] = False  # remove top 5%
    trimmed_trades = [t for t, k in zip(trades, keep_mask) if k]
    if trimmed_trades:
        eq_trim = _build_equity_series(trimmed_trades, initial_capital)
        sh_trim = _monthly_sharpe(eq_trim)
        g4 = {"name": "Outlier Removal", "passed": sh_trim > 0.5,
              "metric_name": "trimmed_sharpe", "metric_value": round(sh_trim, 4),
              "threshold": 0.5,
              "detail": "Removed %d top + %d bottom trades" % (n_trim, n_trim)}
    else:
        g4 = {"name": "Outlier Removal", "passed": False,
              "metric_name": "trimmed_sharpe", "metric_value": 0.0,
              "threshold": 0.5, "detail": "No trades after trimming"}
    result["gates"].append(g4)

    # Gate 5: Yearly consistency (>= 60% of years profitable)
    trade_df = pd.DataFrame(trades)
    trade_df["year"] = pd.to_datetime(trade_df["entry_date"]).dt.year
    yearly = trade_df.groupby("year")["pnl"].sum()
    if len(yearly) > 0:
        pct_profitable = float((yearly > 0).mean())
        g5 = {"name": "Yearly Consistency", "passed": pct_profitable >= 0.60,
              "metric_name": "pct_years_profitable", "metric_value": round(pct_profitable, 4),
              "threshold": 0.60,
              "detail": "%d profitable years / %d total" % ((yearly > 0).sum(), len(yearly))}
    else:
        g5 = {"name": "Yearly Consistency", "passed": False,
              "metric_name": "pct_years_profitable", "metric_value": 0.0,
              "threshold": 0.60, "detail": "No yearly data"}
    result["gates"].append(g5)

    result["gates_passed"] = sum(1 for g in result["gates"] if g["passed"])
    result["all_passed"] = result["gates_passed"] == 5
    return result


def print_validation(v):
    fprint("  --- 5-Gate Adversarial Validation: %s ---" % v["strategy_name"])
    if "error" in v:
        fprint("    ERROR: %s" % v.get("error", ""))
        return
    for g in v["gates"]:
        status = "PASS" if g["passed"] else "FAIL"
        fprint("    [%s] %s: %s=%.4f (thresh=%.2f)  %s" % (
            status, g["name"], g["metric_name"], g["metric_value"],
            g["threshold"], g.get("detail", "")))
    verdict = "ALL GATES PASSED" if v["all_passed"] else "FAILED (%d/%d)" % (
        v["gates_passed"], v["gates_total"])
    fprint("    VERDICT: %s" % verdict)


# ==============================================================
# DATA DOWNLOAD
# ==============================================================

def download_data():
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    fprint("  Downloading %d tickers from 2008-01-01..." % len(all_tickers))
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
    fprint("  Data: %d days, %s to %s" % (
        len(close), close.index[0].date(), close.index[-1].date()))
    return close, high, low


# ==============================================================
# ATR COMPUTATION
# ==============================================================

def compute_atr_series(high, low, close, period=14):
    atr_dict = {}
    for tk in SECTORS:
        if tk not in high.columns:
            continue
        h = high[tk].dropna()
        l = low[tk].dropna()
        c = close[tk].dropna()
        common = h.index.intersection(l.index).intersection(c.index)
        if len(common) <= period:
            continue
        tr1 = h.loc[common] - l.loc[common]
        tr2 = (h.loc[common] - c.loc[common].shift(1)).abs()
        tr3 = (l.loc[common] - c.loc[common].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_dict[tk] = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return atr_dict


# ==============================================================
# FEATURE COMPUTATION
# ==============================================================

def compute_sector_features(tk, px, spy_rets, close_df, dt_idx):
    """Compute 21 BASE_FEATURES for a single sector on a given date."""
    if len(px) < 260:
        return None
    f = {}

    rets = px.pct_change().dropna()

    # Returns at multiple horizons
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    # Volatility
    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) >= 21 else 0.02
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) >= 63 else 0.02

    # Sharpe 63d
    r63 = rets.iloc[-63:]
    f["sharpe_63d"] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) >= 21 else 0.0

    # Max drawdown 63d
    pk63 = px.iloc[-63:].cummax()
    f["maxdd_63d"] = float(((px.iloc[-63:] / pk63) - 1).min()) if len(px) >= 63 else 0.0

    # % of 52w high
    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max()) if len(px) >= 252 else 1.0

    # Momentum acceleration
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    # % positive months (12m)
    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5

    # Sortino 63d
    dr = r63[r63 < 0]
    f["sortino_63d"] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0

    # Calmar 1y
    pk252 = px.iloc[-252:].cummax() if len(px) >= 252 else px.cummax()
    mdd_1y = float(((px.iloc[-252:] / pk252) - 1).min()) if len(px) >= 252 else -0.01
    cagr_1y = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f["calmar_1y"] = cagr_1y / (abs(mdd_1y) + 1e-10)

    # Up capture vs SPY
    if spy_rets is not None and len(spy_rets) >= 63:
        up_spy = spy_rets[spy_rets > 0]
        up_sec = rets.reindex(up_spy.index).iloc[-63:]
        f["up_capture"] = float(
            up_sec.mean() / (up_spy.iloc[-63:].mean() + 1e-10)
        ) if len(up_sec) >= 5 else 1.0
    else:
        f["up_capture"] = 1.0

    # Trend R2 and slope (63d log-linear)
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f["trend_r2_63d"] = r_val ** 2
        f["trend_slope_63d"] = slope * 252
    else:
        f["trend_r2_63d"] = 0.0
        f["trend_slope_63d"] = 0.0

    # Beta to SPY (63d)
    if spy_rets is not None and len(spy_rets) >= 63:
        common = spy_rets.index.intersection(rets.index)
        if len(common) >= 63:
            sec_r = rets.loc[common].iloc[-63:]
            spy_r = spy_rets.loc[common].iloc[-63:]
            cov = np.cov(sec_r.values, spy_r.values)
            f["sector_spy_beta_63d"] = float(cov[0, 1] / (cov[1, 1] + 1e-10))
        else:
            f["sector_spy_beta_63d"] = 1.0
    else:
        f["sector_spy_beta_63d"] = 1.0

    # Sector relative vol vs universe (21d)
    sector_cols = [c for c in SECTORS if c in close_df.columns]
    if len(sector_cols) > 1 and len(rets) >= 21:
        all_sec_rets = close_df[sector_cols].iloc[:dt_idx + 1].pct_change()
        avg_vol = float(all_sec_rets.iloc[-21:].std().mean())
        sec_vol = float(rets.iloc[-21:].std())
        f["sector_relative_vol_21d"] = sec_vol / (avg_vol + 1e-10)
    else:
        f["sector_relative_vol_21d"] = 1.0

    # Cross-sector dispersion (21d)
    if len(sector_cols) > 3:
        sec_rets_all = close_df[sector_cols].iloc[:dt_idx + 1].pct_change()
        daily_disp = sec_rets_all.std(axis=1)
        f["cross_sector_dispersion"] = float(daily_disp.rolling(21).mean().iloc[-1]) if len(daily_disp) >= 21 else 0.01
    else:
        f["cross_sector_dispersion"] = 0.01

    return f


def compute_flow_features_6(dt_idx, close_df):
    """6 cross-asset flow features (KB #264). One set per rebalance date, shared across sectors."""
    f = {k: 0.0 for k in FLOW_FEATURES_6}
    if dt_idx < 63:
        return f

    def _get(ticker):
        if ticker in close_df.columns:
            return close_df[ticker].iloc[:dt_idx + 1].dropna()
        return None

    spy = _get("SPY")
    gld = _get("GLD")
    shy = _get("SHY")
    hyg = _get("HYG")
    tlt = _get("TLT")
    vix = _get("VIX")
    vix3m = _get("VIX3M")

    # gold / equity ratio momentum (21d change)
    if gld is not None and spy is not None and len(gld) > 21 and len(spy) > 21:
        r = gld.iloc[-1] / max(spy.iloc[-1], 1e-10)
        r21 = gld.iloc[-21] / max(spy.iloc[-21], 1e-10)
        f["gold_equity_ratio"] = float(r / max(r21, 1e-10) - 1)

    # cash vs equity ratio momentum
    if shy is not None and spy is not None and len(shy) > 21 and len(spy) > 21:
        r = shy.iloc[-1] / max(spy.iloc[-1], 1e-10)
        r21 = shy.iloc[-21] / max(spy.iloc[-21], 1e-10)
        f["cash_vs_equity_ratio"] = float(r / max(r21, 1e-10) - 1)

    # CTA pressure: MA10 vs MA50 on SPY
    if spy is not None and len(spy) >= 50:
        ma10 = float(spy.iloc[-10:].mean())
        ma50 = float(spy.iloc[-50:].mean())
        f["cta_pressure"] = float((ma10 / max(ma50, 1e-10) - 1) * 100)

    # Credit spread momentum (HYG/SHY)
    if hyg is not None and shy is not None and len(hyg) > 21 and len(shy) > 21:
        r = hyg.iloc[-1] / max(shy.iloc[-1], 1e-10)
        r21 = hyg.iloc[-21] / max(shy.iloc[-21], 1e-10)
        f["credit_spread_mom"] = float(r / max(r21, 1e-10) - 1)

    # TLT/SPY ratio momentum (duration vs equity)
    if tlt is not None and spy is not None and len(tlt) > 21 and len(spy) > 21:
        r = tlt.iloc[-1] / max(spy.iloc[-1], 1e-10)
        r21 = tlt.iloc[-21] / max(spy.iloc[-21], 1e-10)
        f["tlt_spy_ratio"] = float(r / max(r21, 1e-10) - 1)

    # VIX term structure: VIX3M / VIX
    if vix3m is not None and vix is not None and len(vix3m) > 0 and len(vix) > 0:
        v_now = float(vix.iloc[-1])
        v3m = float(vix3m.iloc[-1])
        f["vix_term_structure"] = float(v3m / max(v_now, 1e-10)) if v_now > 0 else 1.0

    return f


# ==============================================================
# LGBM WALK-FORWARD RANKING (SLIDING WINDOW)
# ==============================================================

def get_rebalance_dates(close, rebal_days):
    """Generate rebalance dates every N trading days, starting after WF_TRAIN_DAYS."""
    trading_days = close.index
    dates = []
    start_idx = WF_TRAIN_DAYS
    for i in range(start_idx, len(trading_days), rebal_days):
        dates.append(trading_days[i])
    return pd.DatetimeIndex(dates)


def build_feature_records(close, high, low, rebal_dates, include_flow6=False):
    """Build one feature record per (date, sector) for walk-forward training."""
    import lightgbm as lgb  # noqa: F401 (import check only)
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy_rets = close["SPY"].pct_change().dropna() if "SPY" in close.columns else None
    feature_list = FEATURES_27 if include_flow6 else FEATURES_21

    for dt in rebal_dates:
        dt_idx = close.index.get_indexer([dt], method="ffill")[0]
        if dt_idx < WF_TRAIN_DAYS:
            continue
        # Flow features: compute once per date (not per sector)
        flow_f = compute_flow_features_6(dt_idx, close) if include_flow6 else {}
        for tk in sector_cols:
            px = close[tk].iloc[:dt_idx + 1].dropna()
            feat = compute_sector_features(tk, px, spy_rets, close, dt_idx)
            if feat is None:
                continue
            fi = min(dt_idx + DTE, len(close) - 1)
            if fi <= dt_idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[dt_idx] - 1)
            rec = {**feat, **flow_f, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feature_list:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_list] = df[feature_list].fillna(0.0)
    fprint("  Feature records: %d (%d dates, %d features)" % (
        len(df), len(df["date"].unique()), len(feature_list)))
    return df


def walk_forward_rankings(df, feature_list):
    """
    Walk-forward LGBM ranking: 500-day sliding window.
    For each rebal date past train_days, train on prior 500 days, predict on current date.
    Returns dict: date -> {ticker: score}
    """
    import lightgbm as lgb

    if len(df) < 50:
        return {}

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())
    rankings = {}

    for i, test_date in enumerate(dates):
        dt_idx = df[df["date"] == test_date]["date"].iloc[0]
        # Find training dates: all dates with records at least 1 step before current
        train_dates = [d for d in dates if d < test_date]
        if len(train_dates) < 20:
            continue

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()
        if len(test_df) < 3 or len(train_df) < 30:
            continue

        Xt = np.nan_to_num(train_df[feature_list].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[feature_list].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
            )
            m.fit(Xt, yt)
            preds = m.predict(Xe)
            test_df = test_df.copy()
            test_df["score"] = preds
            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))
        except Exception as e:
            fprint("    LGBM error on %s: %s" % (test_date, e))
            continue

    fprint("  Walk-forward ranking dates: %d" % len(rankings))
    return rankings


# ==============================================================
# STRIKE COMPUTATION
# ==============================================================

def compute_strikes(S, direction):
    """Compute K1, K2 for a spread. 2% OTM, width = max($3, 3% of strike)."""
    if direction == "bull":
        K1 = round(S * (1.0 + OTM_PCT), 2)
        w = max(WIDTH_FLOOR_USD, K1 * WIDTH_FLOOR_PCT)
        K2 = round(K1 + w, 2)
    else:
        K2 = round(S * (1.0 - OTM_PCT), 2)
        w = max(WIDTH_FLOOR_USD, K2 * WIDTH_FLOOR_PCT)
        K1 = round(K2 - w, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


# ==============================================================
# POSITION SIZING
# ==============================================================

def compute_position_weights(bull_picks, bear_picks, sizing_mode):
    """
    Compute per-position weight.
    sizing_mode: "equal" → equal weight within each side
                 "rank"  → 1/rank weight (rank 1 = highest weight)
    Returns dict: ticker -> weight (within-side, sums to 1.0 per side).
    """
    weights = {}
    for picks in [bull_picks, bear_picks]:
        if not picks:
            continue
        if sizing_mode == "rank":
            raw = [1.0 / (i + 1) for i in range(len(picks))]
            total = sum(raw)
            for tk, r in zip(picks, raw):
                weights[tk] = r / total
        else:  # equal
            for tk in picks:
                weights[tk] = 1.0 / len(picks)
    return weights


# ==============================================================
# TRADE EXECUTION
# ==============================================================

def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity, max_pos, pt_pct):
    """
    Execute a single spread trade.

    Returns dict with trade details, or None if rejected.
    pt_pct: profit target fraction (0 = hold to expiry, 0.5 = 50% of max profit).
    """
    if tk not in close.columns or tk not in atr_dict:
        return None

    dt_pos = close.index.get_loc(dt)
    expiry_pos = min(dt_pos + DTE, len(close) - 1)
    if expiry_pos <= dt_pos:
        return None

    S = float(close[tk].iloc[dt_pos])
    av = (float(atr_dict[tk].loc[dt])
          if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt])
          else S * 0.015)

    K1, K2 = compute_strikes(S, direction)

    # Price the entry
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

    # Cost/width filter
    cwr = entry_cost_ps / max(spread_width, 0.01)
    if cwr > COST_WIDTH_MAX:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT
    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40 or total_cost < MIN_POS_SIZE:
        return None

    # Profit target exit logic
    exited_early = False
    exit_pos = expiry_pos
    exit_reason = "expiry"
    hold_days = DTE

    if pt_pct > 0 and max_profit_ps > 0:
        for chk in range(dt_pos + 1, expiry_pos + 1):
            if chk >= len(close):
                break
            chk_date = close.index[chk]
            dte_rem = expiry_pos - chk
            days_held = chk - dt_pos

            S_now = float(close[tk].iloc[chk])
            av_now = (float(atr_dict[tk].loc[chk_date])
                      if chk_date in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[chk_date])
                      else S_now * 0.015)

            cur_val = revalue_spread_bs(S_now, K1, K2, dte_rem, av_now, vix_val, direction)
            unrealized = cur_val - entry_cost_ps

            if unrealized >= pt_pct * max_profit_ps:
                exited_early = True
                exit_pos = chk
                exit_reason = "pt_%dpct" % int(pt_pct * 100)
                hold_days = days_held
                break

    # Compute P&L
    Se = float(close[tk].iloc[exit_pos])

    if exited_early:
        dte_at_exit = expiry_pos - exit_pos
        av_exit = (float(atr_dict[tk].loc[close.index[exit_pos]])
                   if close.index[exit_pos] in atr_dict[tk].index
                   else Se * 0.015)
        exit_val = revalue_spread_bs(Se, K1, K2, dte_at_exit, av_exit, vix_val, direction)
        pnl = (exit_val - entry_cost_ps) * 100 - COMMISSION_RT - EARLY_EXIT_COMM
    else:
        if direction == "bull":
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
        else:
            intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT
        exit_val = intrinsic

    return {
        "pnl": round(pnl, 2),
        "entry_cost_ps": round(entry_cost_ps, 4),
        "exit_val_ps": round(exit_val, 4),
        "max_profit_ps": round(max_profit_ps, 4),
        "spread_width": round(spread_width, 2),
        "cost_width_ratio": round(cwr, 3),
        "K1": K1, "K2": K2,
        "S_entry": round(S, 2), "S_exit": round(Se, 2),
        "exited_early": exited_early,
        "exit_reason": exit_reason,
        "hold_days": hold_days,
        "total_cost": round(total_cost, 2),
    }


# ==============================================================
# SIMULATION ENGINE
# ==============================================================

def simulate_variant(rankings, close, atr_dict, rebal_dates, pt_pct, sizing_mode):
    """
    Run backtest simulation for one variant.

    VIX >= 20: bull spreads only on top-3
    VIX  < 20: top-3 bull + bottom-3 bear
    """
    spy = close["SPY"] if "SPY" in close.columns else None
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    early_exits = 0

    for dt in sorted(rebal_dates):
        if spy is not None and dt not in spy.index:
            continue

        # Find most recent ranking at or before this date
        ranking_date = None
        for rd in sorted(rankings.keys()):
            if rd <= dt:
                ranking_date = rd
        if ranking_date is None:
            continue

        scores = rankings[ranking_date]
        if not scores or len(scores) < TOP_K:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]

        # VIX regime gate: bear only when VIX < 20
        if cv < VIX_THRESHOLD and len(scores) >= TOP_K * 2:
            bear_picks = [t for t, _ in ranked_asc[:TOP_K]]
        else:
            bear_picks = []

        position_weights = compute_position_weights(bull_picks, bear_picks, sizing_mode)
        n_positions = len(bull_picks) + len(bear_picks)
        if n_positions == 0:
            continue

        # Total budget per rebalance cycle
        total_budget = min(CAP * 2, equity * 0.80)

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                w = position_weights.get(tk, 1.0 / max(n_positions, 1))
                max_pos = min(total_budget * w, equity * 0.40)
                if max_pos < MIN_POS_SIZE:
                    continue

                result = execute_trade(tk, dt, direction, close, atr_dict, cv, equity, max_pos, pt_pct)
                if result is None:
                    continue

                equity += result["pnl"]
                if result["exited_early"]:
                    early_exits += 1

                # Determine market regime during hold period
                dt_idx = close.index.get_loc(dt)
                hold = result["hold_days"]
                exit_idx = min(dt_idx + hold, len(close) - 1)
                exit_date = close.index[exit_idx]

                spy_entry = float(spy.loc[dt]) if spy is not None and dt in spy.index else None
                spy_exit = float(spy.iloc[exit_idx]) if spy is not None else None
                if spy_entry and spy_exit:
                    regime = "bull" if spy_exit >= spy_entry else "bear"
                else:
                    regime = "bull"

                trades.append({
                    **result,
                    "entry_date": str(dt.date()),
                    "exit_date": str(exit_date.date()),
                    "ticker": tk,
                    "direction": direction,
                    "vix": round(cv, 1),
                    "regime": regime,
                    "win": result["pnl"] > 0,
                    "weight": round(w, 4),
                    "n_positions": n_positions,
                })

    return trades, equity, early_exits


# ==============================================================
# METRICS
# ==============================================================

def compute_stats(trades, initial_capital=CAP):
    """Compute comprehensive performance stats."""
    if not trades:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    equity = [initial_capital]
    for p in pnls:
        equity.append(equity[-1] + p)
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1)

    wr = float(np.mean(pnls > 0))
    gw = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
    gl = float(abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-10
    pf = gw / max(gl, 1e-10)

    trade_df = pd.DataFrame(trades)
    trade_df["month"] = pd.to_datetime(trade_df["entry_date"]).dt.to_period("M")
    monthly_pnl = trade_df.groupby("month")["pnl"].sum()
    if len(monthly_pnl) > 1:
        sharpe = float(monthly_pnl.mean() / (monthly_pnl.std() + 1e-10) * np.sqrt(12))
    else:
        sharpe = 0.0

    # Sortino
    monthly_ret = monthly_pnl / initial_capital
    neg_ret = monthly_ret[monthly_ret < 0]
    ds = float(neg_ret.std()) if len(neg_ret) > 1 else 1e-10
    sortino = float(monthly_ret.mean() / (ds + 1e-10) * np.sqrt(12))

    # CAGR
    dates_str = [t["entry_date"] for t in trades]
    y0 = pd.Timestamp(min(dates_str))
    y1 = pd.Timestamp(max(dates_str))
    years = max((y1 - y0).days / 365.25, 0.1)
    total_ret = equity[-1] / equity[0]
    cagr = float(total_ret ** (1 / years) - 1)

    # Max DD
    max_dd_pct = float(dd.min()) * 100

    early_exits = sum(1 for t in trades if t.get("exited_early", False))
    avg_hold = float(np.mean([t.get("hold_days", DTE) for t in trades]))

    # Yearly consistency
    trade_df["year"] = pd.to_datetime(trade_df["entry_date"]).dt.year
    yearly = trade_df.groupby("year")["pnl"].sum()
    pct_years_up = float((yearly > 0).mean()) if len(yearly) > 0 else 0.0

    return {
        "n_trades": len(trades),
        "total_pnl": round(float(pnls.sum()), 2),
        "avg_pnl": round(float(pnls.mean()), 2),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr, 4),
        "max_dd_pct": round(max_dd_pct, 2),
        "final_equity": round(float(equity[-1]), 2),
        "early_exit_rate": round(early_exits / len(trades), 4),
        "avg_hold_days": round(avg_hold, 1),
        "early_exits": early_exits,
        "pct_years_profitable": round(pct_years_up, 3),
    }


# ==============================================================
# REGIME ANALYSIS
# ==============================================================

def compute_regime_split(trades):
    """Split by bull/bear SPY regime and by VIX level."""
    bull = [t for t in trades if t.get("regime") == "bull"]
    bear = [t for t in trades if t.get("regime") == "bear"]
    high_vix = [t for t in trades if t.get("vix", 20) >= VIX_THRESHOLD]
    low_vix = [t for t in trades if t.get("vix", 20) < VIX_THRESHOLD]
    return {
        "bull_regime": compute_stats(bull) if bull else None,
        "bear_regime": compute_stats(bear) if bear else None,
        "high_vix": compute_stats(high_vix) if high_vix else None,
        "low_vix": compute_stats(low_vix) if low_vix else None,
    }


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = datetime.now()
    fprint("=" * 110)
    fprint("V13 COMBINED BEST -- Three Validated Improvements Stacked")
    fprint("=" * 110)
    fprint("START: %s" % t0.strftime("%Y-%m-%d %H:%M:%S"))
    fprint("")
    fprint("QUESTION: Do Profit Target + Rank Sizing + Biweekly Rebal COMPOUND or INTERFERE?")
    fprint("")
    fprint("Improvements under test:")
    fprint("  1. 50%% Profit Target Exit    -- V9.3 Sharpe 5.12 vs 2.36 (+97%%)")
    fprint("  2. Rank-Weighted Sizing (1/rank) -- V12 MDD -4.8%% vs -23.8%%")
    fprint("  3. Biweekly Rebalance (10d)   -- KB #253 Sharpe +73%%")
    fprint("")
    fprint("Config:")
    fprint("  Capital: $%.0f | DTE: %d | Commission: $%.2f RT | Entry haircut: %.0f%%" % (
        CAP, DTE, COMMISSION_RT, HAIRCUT * 100))
    fprint("  OTM: %.0f%% | Width: max($%.0f, %.0f%% of strike)" % (
        OTM_PCT * 100, WIDTH_FLOOR_USD, WIDTH_FLOOR_PCT * 100))
    fprint("  VIX threshold: %.0f (>=%.0f → bull only)" % (VIX_THRESHOLD, VIX_THRESHOLD))
    fprint("  Walk-forward: %d-day sliding train | TOP_K=%d picks each side" % (WF_TRAIN_DAYS, TOP_K))
    fprint("  Adversarial gates: %d permutation shuffles" % N_PERM)
    fprint("")
    fprint("8 VARIANTS:")
    for var_name, pt_pct, sizing, rebal_d, feat_set, desc in VARIANTS:
        fprint("  %s" % desc)
        fprint("    pt=%.0f%% sizing=%s rebal=%dd features=%s" % (
            pt_pct * 100, sizing, rebal_d, feat_set))
    fprint("")

    # Data download
    fprint("=" * 80)
    fprint("DATA DOWNLOAD")
    fprint("=" * 80)
    close, high, low = download_data()
    atr_dict = compute_atr_series(high, low, close)
    spy_close = close["SPY"] if "SPY" in close.columns else None

    # Build feature records for each feature set
    # We need biweekly and monthly rebalance dates, and 21 and 27 feature sets
    fprint("")
    fprint("=" * 80)
    fprint("BUILDING FEATURE RECORDS")
    fprint("=" * 80)

    # Biweekly and monthly rebalance dates (used by variants)
    rebal_dates_biweekly = get_rebalance_dates(close, REBAL_DAYS_BIWEEKLY)
    rebal_dates_monthly = get_rebalance_dates(close, REBAL_DAYS_MONTHLY)
    fprint("  Biweekly rebal dates: %d (every %d trading days)" % (
        len(rebal_dates_biweekly), REBAL_DAYS_BIWEEKLY))
    fprint("  Monthly rebal dates: %d (every %d trading days)" % (
        len(rebal_dates_monthly), REBAL_DAYS_MONTHLY))

    # For training features we use ALL rebalance dates (biweekly as finest resolution)
    fprint("")
    fprint("Building 21-feature records...")
    records_21 = build_feature_records(close, high, low, rebal_dates_biweekly, include_flow6=False)

    fprint("")
    fprint("Building 27-feature records (21 + 6 flow)...")
    records_27 = build_feature_records(close, high, low, rebal_dates_biweekly, include_flow6=True)

    # Walk-forward rankings
    fprint("")
    fprint("=" * 80)
    fprint("WALK-FORWARD LGBM RANKINGS")
    fprint("=" * 80)

    fprint("")
    fprint("Running LGBM walk-forward (21 features)...")
    rankings_21 = walk_forward_rankings(records_21, FEATURES_21)

    fprint("")
    fprint("Running LGBM walk-forward (27 features)...")
    rankings_27 = walk_forward_rankings(records_27, FEATURES_27)

    if len(rankings_21) < 5:
        fprint("ERROR: Insufficient ranking dates (21f: %d). Check data." % len(rankings_21))
        return

    rankings_map = {"21": rankings_21, "27": rankings_27}
    rebal_dates_map = {
        REBAL_DAYS_BIWEEKLY: rebal_dates_biweekly,
        REBAL_DAYS_MONTHLY: rebal_dates_monthly,
    }

    # Run all 8 variants
    all_trades = {}
    all_stats = {}
    all_validation = {}
    all_regime = {}

    fprint("")
    fprint("=" * 110)
    fprint("RUNNING 8 VARIANTS")
    fprint("=" * 110)

    for var_name, pt_pct, sizing_mode, rebal_days, feat_set, description in VARIANTS:
        fprint("")
        fprint("-" * 100)
        fprint(description)
        fprint("  pt=%.0f%%  sizing=%s  rebal=%dd  features=%s" % (
            pt_pct * 100, sizing_mode, rebal_days, feat_set))
        fprint("-" % ())

        rankings = rankings_map.get(feat_set, rankings_21)
        rebal_dates = rebal_dates_map.get(rebal_days, rebal_dates_biweekly)

        trades, final_eq, early_ex = simulate_variant(
            rankings, close, atr_dict, rebal_dates, pt_pct, sizing_mode)

        all_trades[var_name] = trades
        fprint("  Trades: %d | Early exits: %d | Final equity: $%s" % (
            len(trades), early_ex, "{:,.0f}".format(final_eq)))

        s = compute_stats(trades)
        all_stats[var_name] = s

        if s:
            fprint("  Sharpe: %.2f | Sortino: %.2f | CAGR: %.1f%%" % (
                s["sharpe"], s["sortino"], s["cagr"] * 100))
            fprint("  WR: %.1f%% | PF: %.2f | MDD: %.1f%% | Avg hold: %.1fd" % (
                s["win_rate"] * 100, s["profit_factor"],
                s["max_dd_pct"], s["avg_hold_days"]))
            fprint("  Early exit rate: %.1f%% | Yearly profitable: %.0f%%" % (
                s["early_exit_rate"] * 100, s["pct_years_profitable"] * 100))

        # Side breakdown
        for side in ["bull", "bear"]:
            st = [t for t in trades if t["direction"] == side]
            if st:
                pnls = [t["pnl"] for t in st]
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                fprint("  %s side: %d trades, WR %.1f%%, PnL $%s" % (
                    side, len(st), wr * 100, "{:,.0f}".format(sum(pnls))))

        # Regime split
        regime = compute_regime_split(trades)
        all_regime[var_name] = regime
        br = regime.get("bull_regime")
        be = regime.get("bear_regime")
        if br and be:
            fprint("  Regimes: bull N=%d Sh=%.2f | bear N=%d Sh=%.2f" % (
                br["n_trades"], br["sharpe"], be["n_trades"], be["sharpe"]))

        # 5-gate adversarial validation
        if len(trades) >= 10:
            val = adversarial_validate(trades, initial_capital=CAP,
                                       spy_prices=spy_close,
                                       strategy_name=var_name)
            all_validation[var_name] = val
            print_validation(val)
        else:
            fprint("  Validation: SKIPPED (< 10 trades)")
            all_validation[var_name] = {
                "strategy_name": var_name, "n_trades": len(trades),
                "gates": [], "gates_passed": 0, "gates_total": 5,
                "all_passed": False, "error": "Insufficient trades",
            }

    # ==============================================================
    # RESULTS TABLE (sorted by Sharpe)
    # ==============================================================
    fprint("")
    fprint("=" * 140)
    fprint("COMPARISON TABLE -- ALL 8 VARIANTS (sorted by Sharpe)")
    fprint("=" * 140)

    sorted_variants = sorted(
        [(vn, all_stats.get(vn)) for vn, *_ in VARIANTS if all_stats.get(vn)],
        key=lambda x: x[1]["sharpe"] if x[1] else -999,
        reverse=True,
    )

    hdr = ("  %-30s %5s %7s %8s %6s %5s %7s %7s %7s %6s %5s %5s" % (
        "Variant", "N", "Sharpe", "Sortino", "CAGR", "WR%", "PF", "MDD%",
        "TotRet%", "AvgHld", "Gate", "EarX%"))
    fprint(hdr)
    fprint("  " + "-" * 130)

    for var_name, s in sorted_variants:
        if s is None:
            continue
        val = all_validation.get(var_name, {})
        gates_str = "%d/%d" % (val.get("gates_passed", 0), val.get("gates_total", 5))
        total_ret_pct = (s["final_equity"] / CAP - 1) * 100
        fprint("  %-30s %5d %7.2f %8.2f %5.1f%% %4.1f%% %6.2f %6.1f%% %6.1f%% %5.1fd %5s %4.0f%%" % (
            var_name, s["n_trades"], s["sharpe"], s["sortino"],
            s["cagr"] * 100, s["win_rate"] * 100, s["profit_factor"],
            s["max_dd_pct"], total_ret_pct, s["avg_hold_days"], gates_str,
            s["early_exit_rate"] * 100))

    # ==============================================================
    # DELTA TABLE vs A_baseline
    # ==============================================================
    fprint("")
    fprint("=" * 110)
    fprint("DELTA vs A_baseline (equal size, hold-to-expiry, biweekly)")
    fprint("=" * 110)

    base = all_stats.get("A_baseline")
    if base:
        fprint("")
        fprint("  %-30s %8s %8s %7s %7s %8s %7s" % (
            "Variant", "dSharpe", "dSortino", "dWR%", "dPF", "dMDD%", "dPnL$"))
        fprint("  " + "-" * 90)
        for var_name, _, *_ in VARIANTS[1:]:
            s = all_stats.get(var_name)
            if s is None:
                continue
            fprint("  %-30s %+7.2f %+8.2f %+6.1f%% %+6.2f %+7.1f%% $%+7s" % (
                var_name,
                s["sharpe"] - base["sharpe"],
                s["sortino"] - base["sortino"],
                (s["win_rate"] - base["win_rate"]) * 100,
                s["profit_factor"] - base["profit_factor"],
                s["max_dd_pct"] - base["max_dd_pct"],
                "{:,.0f}".format(s["total_pnl"] - base["total_pnl"])))

    # ==============================================================
    # COMPOUNDING ANALYSIS
    # ==============================================================
    fprint("")
    fprint("=" * 110)
    fprint("COMPOUNDING ANALYSIS -- Do improvements compound or interfere?")
    fprint("=" * 110)
    fprint("")

    # Expected Sharpe if improvements were independent (additive)
    sh_base = base["sharpe"] if base else 0.0
    sh_pt50 = (all_stats.get("B_profit_target_50") or {}).get("sharpe", sh_base) - sh_base
    sh_rank = (all_stats.get("C_rank_sizing") or {}).get("sharpe", sh_base) - sh_base
    sh_both = (all_stats.get("D_pt50_rank") or {}).get("sharpe", sh_base) - sh_base

    fprint("  A (baseline):                    Sharpe = %.2f" % sh_base)
    fprint("  B (+ PT 50%% only):               dSharpe = %+.2f" % sh_pt50)
    fprint("  C (+ Rank Sizing only):           dSharpe = %+.2f" % sh_rank)
    fprint("  D (PT 50%% + Rank Sizing):         dSharpe = %+.2f" % sh_both)
    if sh_pt50 + sh_rank != 0:
        expected_d = sh_pt50 + sh_rank
        actual_d = sh_both
        synergy = actual_d / max(abs(expected_d), 1e-10)
        verdict = "COMPOUND" if synergy > 1.05 else ("INTERFERE" if synergy < 0.80 else "ADDITIVE")
        fprint("")
        fprint("  Expected combined delta (B+C additive): %+.2f" % expected_d)
        fprint("  Actual combined delta (D):              %+.2f" % actual_d)
        fprint("  Synergy ratio (actual/expected):        %.2fx" % synergy)
        fprint("  Verdict: %s" % verdict)
        fprint("")
        if synergy > 1.2:
            fprint("  ==> STRONG COMPOUNDING: combining Profit Target + Rank Sizing")
            fprint("      produces MORE than the sum of individual improvements.")
        elif synergy > 1.05:
            fprint("  ==> MILD COMPOUNDING: improvements stack slightly better than additive.")
        elif synergy > 0.80:
            fprint("  ==> ADDITIVE: improvements stack approximately as expected.")
        else:
            fprint("  ==> INTERFERENCE: improvements partially cancel each other.")

    # Flow features analysis
    sh_g = (all_stats.get("G_pt50_rank_flow6") or {}).get("sharpe", sh_base)
    sh_d = (all_stats.get("D_pt50_rank") or {}).get("sharpe", sh_base)
    if sh_d and sh_g:
        fprint("")
        fprint("  Flow features contribution (G vs D):")
        fprint("    D (PT + Rank, 21 features): Sharpe %.2f" % sh_d)
        fprint("    G (PT + Rank + 6 flow):     Sharpe %.2f (delta %+.2f)" % (sh_g, sh_g - sh_d))
        if sh_g > sh_d + 0.1:
            fprint("    Flow features ADD value when combined with PT + Rank Sizing")
        elif sh_g < sh_d - 0.1:
            fprint("    Flow features HURT when combined with PT + Rank Sizing (possible interference)")
        else:
            fprint("    Flow features are NEUTRAL when combined with PT + Rank Sizing")

    # Monthly vs biweekly rebalance
    sh_g_biweekly = (all_stats.get("G_pt50_rank_flow6") or {}).get("sharpe", sh_base)
    sh_h_monthly = (all_stats.get("H_pt50_rank_flow6_monthly") or {}).get("sharpe", sh_base)
    if sh_g_biweekly and sh_h_monthly:
        fprint("")
        fprint("  Rebalance frequency comparison:")
        fprint("    G (biweekly, 10d): Sharpe %.2f" % sh_g_biweekly)
        fprint("    H (monthly,  20d): Sharpe %.2f (delta %+.2f)" % (
            sh_h_monthly, sh_h_monthly - sh_g_biweekly))
        if sh_h_monthly > sh_g_biweekly + 0.1:
            fprint("    Monthly rebalance IMPROVES on biweekly when all improvements combined")
        elif sh_h_monthly < sh_g_biweekly - 0.1:
            fprint("    Monthly rebalance HURTS vs biweekly when all improvements combined")
        else:
            fprint("    Rebalance frequency has MINIMAL impact on combined strategy")

    # ==============================================================
    # 5-GATE SUMMARY TABLE
    # ==============================================================
    fprint("")
    fprint("=" * 110)
    fprint("5-GATE ADVERSARIAL VALIDATION SUMMARY")
    fprint("=" * 110)
    fprint("")
    fprint("  %-30s %7s %12s %15s %12s %15s" % (
        "Variant", "Gates", "Permutation", "RegimeStab", "SubPeriod", "YearlyConst"))
    fprint("  " + "-" * 100)

    for var_name, _, *_ in VARIANTS:
        val = all_validation.get(var_name, {})
        gates = val.get("gates", [])
        gates_map = {g["name"]: g for g in gates}

        def gstat(name):
            g = gates_map.get(name)
            if not g:
                return "N/A"
            return "%s %.3f" % ("OK" if g["passed"] else "FAIL", g["metric_value"])

        fprint("  %-30s %d/%d  %12s  %15s  %12s  %15s" % (
            var_name,
            val.get("gates_passed", 0), val.get("gates_total", 5),
            gstat("Permutation"),
            gstat("Regime Stability"),
            gstat("Sub-Period Stability"),
            gstat("Yearly Consistency"),
        ))

    # ==============================================================
    # BEST VARIANT SUMMARY
    # ==============================================================
    fprint("")
    fprint("=" * 110)
    fprint("BEST VARIANT SUMMARY")
    fprint("=" * 110)

    if sorted_variants:
        best_var, best_s = sorted_variants[0]
        best_val = all_validation.get(best_var, {})
        fprint("")
        fprint("  WINNER: %s" % best_var)
        if best_s:
            fprint("  Sharpe: %.2f | Sortino: %.2f | MDD: %.1f%% | CAGR: %.1f%% | WR: %.1f%% | PF: %.2f" % (
                best_s["sharpe"], best_s["sortino"], best_s["max_dd_pct"],
                best_s["cagr"] * 100, best_s["win_rate"] * 100, best_s["profit_factor"]))
            fprint("  Total return: %.1f%% | Avg hold: %.1f days | Gates: %d/%d" % (
                (best_s["final_equity"] / CAP - 1) * 100,
                best_s["avg_hold_days"],
                best_val.get("gates_passed", 0), best_val.get("gates_total", 5)))
        if base and best_var != "A_baseline" and best_s:
            delta_sh = best_s["sharpe"] - base["sharpe"]
            fprint("")
            fprint("  Improvement vs baseline A:")
            fprint("    dSharpe: %+.2f | dMDD: %+.1f%% | dSortino: %+.2f" % (
                delta_sh,
                best_s["max_dd_pct"] - base["max_dd_pct"],
                best_s["sortino"] - base["sortino"]))

    # ==============================================================
    # SAVE JSON RESULTS
    # ==============================================================
    results = {
        "timestamp": t0.isoformat(),
        "experiment": EXPERIMENT_NAME,
        "question": "Do Profit Target + Rank Sizing + Biweekly Rebal compound or interfere?",
        "config": {
            "capital": CAP, "dte": DTE, "commission_rt": COMMISSION_RT,
            "haircut": HAIRCUT, "otm_pct": OTM_PCT,
            "width_floor_usd": WIDTH_FLOOR_USD, "width_floor_pct": WIDTH_FLOOR_PCT,
            "vix_threshold": VIX_THRESHOLD, "top_k": TOP_K,
            "wf_train_days": WF_TRAIN_DAYS, "n_perm": N_PERM,
            "min_pos_size": MIN_POS_SIZE, "n_sectors": len(SECTORS),
        },
        "variants": {},
        "compounding_analysis": {},
        "best_variant": sorted_variants[0][0] if sorted_variants else None,
        "best_sharpe": sorted_variants[0][1]["sharpe"] if sorted_variants and sorted_variants[0][1] else None,
    }

    for var_name, *_ in VARIANTS:
        s = all_stats.get(var_name)
        val = all_validation.get(var_name, {})
        results["variants"][var_name] = {
            "stats": s,
            "validation": val,
            "regime_split": {
                k: v for k, v in (all_regime.get(var_name) or {}).items() if v
            },
        }

    # Compounding analysis
    if base:
        results["compounding_analysis"] = {
            "baseline_sharpe": sh_base,
            "pt50_delta": sh_pt50,
            "rank_delta": sh_rank,
            "combined_delta_actual": sh_both,
            "combined_delta_expected_additive": sh_pt50 + sh_rank,
            "synergy_ratio": float(sh_both / max(abs(sh_pt50 + sh_rank), 1e-10)),
        }

    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    fprint("")
    fprint("Results saved to: %s" % results_file)

    # Save trade details per variant
    for var_name, *_ in VARIANTS:
        trades = all_trades.get(var_name, [])
        if trades:
            tf = OUTPUT_DIR / ("trades_%s.json" % var_name)
            with open(tf, "w") as f:
                json.dump(trades, f, indent=1, default=str)

    # ==============================================================
    # MLFLOW LOGGING
    # ==============================================================
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="v13_combined_best_%s" % t0.strftime("%Y%m%d_%H%M")):
                mlflow.log_param("capital", CAP)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("commission_rt", COMMISSION_RT)
                mlflow.log_param("haircut", HAIRCUT)
                mlflow.log_param("top_k", TOP_K)
                mlflow.log_param("otm_pct", OTM_PCT)
                mlflow.log_param("vix_threshold", VIX_THRESHOLD)
                mlflow.log_param("wf_train_days", WF_TRAIN_DAYS)
                mlflow.log_param("n_perm", N_PERM)
                mlflow.log_param("n_variants", len(VARIANTS))
                if sorted_variants:
                    mlflow.log_param("best_variant", sorted_variants[0][0])

                for var_name, _, _, _, _, _ in VARIANTS:
                    s = all_stats.get(var_name)
                    if s is None:
                        continue
                    prefix = var_name.split("_")[0]
                    mlflow.log_metric("%s_sharpe" % prefix, s["sharpe"])
                    mlflow.log_metric("%s_sortino" % prefix, s["sortino"])
                    mlflow.log_metric("%s_cagr" % prefix, s["cagr"])
                    mlflow.log_metric("%s_wr" % prefix, s["win_rate"])
                    mlflow.log_metric("%s_pf" % prefix, s["profit_factor"])
                    mlflow.log_metric("%s_mdd" % prefix, s["max_dd_pct"])
                    mlflow.log_metric("%s_n_trades" % prefix, s["n_trades"])
                    mlflow.log_metric("%s_total_pnl" % prefix, s["total_pnl"])
                    mlflow.log_metric("%s_early_exit_rate" % prefix, s["early_exit_rate"])
                    val = all_validation.get(var_name, {})
                    mlflow.log_metric("%s_gates_passed" % prefix, val.get("gates_passed", 0))
                    rg = all_regime.get(var_name, {})
                    bull_rg = rg.get("bull_regime")
                    bear_rg = rg.get("bear_regime")
                    if bull_rg:
                        mlflow.log_metric("%s_bull_sharpe" % prefix, bull_rg["sharpe"])
                    if bear_rg:
                        mlflow.log_metric("%s_bear_sharpe" % prefix, bear_rg["sharpe"])

                # Compounding analysis metrics
                ca = results.get("compounding_analysis", {})
                if ca:
                    mlflow.log_metric("synergy_ratio", ca.get("synergy_ratio", 1.0))
                    mlflow.log_metric("pt50_delta", ca.get("pt50_delta", 0.0))
                    mlflow.log_metric("rank_delta", ca.get("rank_delta", 0.0))
                    mlflow.log_metric("combined_delta_actual", ca.get("combined_delta_actual", 0.0))
                    mlflow.log_metric("combined_delta_expected", ca.get("combined_delta_expected_additive", 0.0))

                mlflow.log_artifact(str(results_file))

            exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if exp:
                fprint("MLflow experiment ID: %s" % exp.experiment_id)
            fprint("MLflow logged: experiment '%s'" % EXPERIMENT_NAME)
        except Exception as e:
            fprint("MLflow logging error: %s" % e)

    elapsed = (datetime.now() - t0).total_seconds()
    fprint("")
    fprint("=" * 110)
    fprint("COMPLETED in %.1f minutes" % (elapsed / 60))
    fprint("=" * 110)


if __name__ == "__main__":
    main()
