#!/usr/bin/env python3
"""
V10 Single-Leg Options Backtest V1
====================================

Tests V10 sector rotation rankings (proven Sharpe 6.32 with spreads)
using SINGLE-LEG options (long calls / long puts) for Level 2 accounts.

Six variants:
  A. Baseline ATM: Long ATM calls on top-2 + long ATM puts on bottom-2 (DTE=28, hold to expiry)
  B. Slightly OTM: Long 2% OTM calls on top-2 + long 2% OTM puts on bottom-2
  C. Deep OTM: Long 4% OTM calls on top-2 + long 4% OTM puts on bottom-2
  D. Top-4 calls only (bull-only, no puts)
  E. Bottom-4 puts only (bear-only, no calls)
  F. ATM with 50% profit-target exit (exit when option doubles from entry)

Pricing: Black-Scholes with IV = max(VIX/100, realized_vol_21d * 1.2)
Commission: $0.65 per leg ($1.30 round-trip)
Account: $645, max position = min($200, 30% of equity)
Option premium = 100 * BS_price per contract

Walk-forward: 500-day sliding window, monthly rebalance (every 20 trading days).
Target: 14-day forward return for LGBM ranking.

Output: output/growth_research/v10_single_leg/
MLflow experiment: v10_single_leg_backtest
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

# ---- Environment detection ----
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")

if _NEPTUNE_BASE.exists():
    BASE = _NEPTUNE_BASE
    fprint(f"Running on Neptune: {BASE}")
else:
    BASE = _JUPITER_BASE
    fprint(f"Running on Jupiter: {BASE}")

OUTPUT_DIR = BASE / "output" / "growth_research" / "v10_single_leg"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Constants ----
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX"]

CAP = 645.0
COMMISSION_PER_LEG = 0.65  # single-leg commission
COMMISSION_RT = 1.30       # round-trip (open + close)
DTE = 28
RISK_FREE_RATE = 0.05      # annualized
MAX_POS_DOLLAR = 200.0     # absolute cap per position
MAX_POS_PCT = 0.30         # 30% of equity cap

WF_TRAIN_DAYS = 500        # sliding window
REBAL_FREQ = 20            # every 20 trading days
FWD_HORIZON = 14           # 14-day forward return target

N_PERMUTATIONS = 100       # permutation test

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v10_single_leg_backtest"

FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high",
    "mom_accel", "pct_pos_months_12m", "sortino_63d", "calmar_1y",
    "trend_r2_63d", "trend_slope_63d",
]
assert len(FEATURES) == 17

# MLflow setup
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable -- will skip logging")


# ==============================================================
# BLACK-SCHOLES PRICING (INLINE)
# ==============================================================

def bs_d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))


def bs_d2(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    return bs_d1(S, K, T, r, sigma) - sigma * sqrt(T)


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 1e-10:
        return max(S - K * exp(-r * T), 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 1e-10:
        return max(K * exp(-r * T) - S, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def compute_iv(vix_val, realized_vol_21d):
    """IV = max(VIX/100, realized_vol_21d * 1.2) -- conservative."""
    vix_iv = vix_val / 100.0 if vix_val is not None and vix_val > 0 else 0.20
    rv_iv = realized_vol_21d * 1.2 if realized_vol_21d is not None and realized_vol_21d > 0 else 0.20
    return max(vix_iv, rv_iv)


# ==============================================================
# DATA DOWNLOAD
# ==============================================================

def download_data():
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start="2020-01-01", end="2026-07-25",
                      progress=False, auto_adjust=True)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw[["Close"]]
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.ffill()
    rename_map = {"^VIX": "VIX"}
    close = close.rename(columns=rename_map)
    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close


# ==============================================================
# FEATURE ENGINEERING (17 features)
# ==============================================================

def compute_features(px, spy_px):
    """Compute 17 momentum/quality features for a single sector at a point in time."""
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
        pk = px.iloc[-63:].cummax()
        f["maxdd_63d"] = float(((px.iloc[-63:] / pk) - 1).min())
    else:
        f["maxdd_63d"] = 0.0

    # Percent of 52w high
    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max()) if len(px) >= 252 else 1.0

    # Momentum acceleration
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    # Percent positive months (12m)
    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5

    # Sortino
    dr = rets.iloc[-63:][rets.iloc[-63:] < 0]
    f["sortino_63d"] = float(rets.iloc[-63:].mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0

    # Calmar
    if len(px) >= 252:
        pk252 = px.iloc[-252:].cummax()
        mdd252 = float(((px.iloc[-252:] / pk252) - 1).min())
        cagr = float(px.iloc[-1] / px.iloc[-252] - 1)
        f["calmar_1y"] = cagr / (abs(mdd252) + 1e-10)
    else:
        f["calmar_1y"] = 0.0

    # Trend R2 and slope (63d)
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


# ==============================================================
# LGBM WALK-FORWARD RANKING (SLIDING WINDOW)
# ==============================================================

def build_feature_matrix(close, rebal_indices):
    """Build feature matrix for all rebalance dates and all sectors."""
    records = []
    spy = close["SPY"]
    sector_cols = [c for c in SECTORS if c in close.columns]

    for di in rebal_indices:
        dt = close.index[di]
        for tk in sector_cols:
            px = close[tk].iloc[:di + 1].dropna()
            spy_px = spy.iloc[:di + 1].dropna()
            feats = compute_features(px, spy_px)
            if feats is None:
                continue
            # Forward return (14d)
            fi = min(di + FWD_HORIZON, len(close) - 1)
            if fi <= di:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[di] - 1)
            rec = {**feats, "date_idx": di, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[FEATURES] = df[FEATURES].fillna(0.0)
    fprint(f"  Feature matrix: {len(df)} records, {len(df['date_idx'].unique())} rebalance dates")
    return df


def walk_forward_lgbm_rank(df):
    """Walk-forward LGBM ranking with 500-day sliding window (in rebalance steps).

    Train window = last WF_TRAIN_DAYS / REBAL_FREQ rebalance periods.
    """
    import lightgbm as lgb

    if len(df) < 50:
        fprint("  WARNING: Too few records for walk-forward")
        return {}

    # Rank label per date
    df["rank_label"] = df.groupby("date_idx")["fwd_ret"].rank(pct=True)
    date_indices = sorted(df["date_idx"].unique())

    # Number of rebalance periods that fit in the training window
    train_periods = WF_TRAIN_DAYS // REBAL_FREQ  # 500/20 = 25

    rankings = {}
    for i in range(train_periods, len(date_indices)):
        train_di_list = date_indices[max(0, i - train_periods):i]
        test_di = date_indices[i]

        train_df = df[df["date_idx"].isin(train_di_list)]
        test_df = df[df["date_idx"] == test_di].copy()

        if len(test_df) < 3 or len(train_df) < 30:
            continue

        Xt = np.nan_to_num(train_df[FEATURES].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[FEATURES].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1,
            )
            m.fit(Xt, yt)
            preds = m.predict(Xe)
            test_df["score"] = preds
            rankings[test_di] = dict(zip(test_df["ticker"], test_df["score"]))
        except Exception as e:
            fprint(f"  LGBM error at idx {test_di}: {e}")
            continue

    fprint(f"  Walk-forward produced {len(rankings)} ranking dates")
    return rankings


# ==============================================================
# SINGLE-LEG OPTION TRADE EXECUTION
# ==============================================================

def get_max_position_size(equity):
    """Max position size = min($200, 30% of equity)."""
    return min(MAX_POS_DOLLAR, MAX_POS_PCT * equity)


def price_single_leg(S, otm_pct, direction, dte, iv):
    """Price a single-leg option.

    Args:
        S: underlying price
        otm_pct: how far OTM (0.0 = ATM, 0.02 = 2% OTM, 0.04 = 4% OTM)
        direction: "call" or "put"
        dte: days to expiration
        iv: implied volatility (annualized)

    Returns:
        (premium_per_share, strike) -- premium is per-share BS price
    """
    T = dte / 365.0
    if direction == "call":
        K = S * (1.0 + otm_pct)
    else:
        K = S * (1.0 - otm_pct)
    K = round(K, 2)

    if direction == "call":
        premium = bs_call_price(S, K, T, RISK_FREE_RATE, iv)
    else:
        premium = bs_put_price(S, K, T, RISK_FREE_RATE, iv)

    return premium, K


def execute_single_leg_trade(tk, di, direction, close, vix_series, rv_series,
                              equity, otm_pct=0.0, profit_target=None):
    """Execute a single-leg option trade.

    Args:
        tk: ticker
        di: date index into close
        direction: "call" or "put"
        close: close price DataFrame
        vix_series: VIX close series
        rv_series: dict of {ticker: realized_vol_21d_series}
        equity: current account equity
        otm_pct: OTM percentage (0=ATM, 0.02=2%, 0.04=4%)
        profit_target: if set, exit when option value >= entry * (1 + profit_target)
                       e.g., profit_target=1.0 means exit when option doubles

    Returns:
        dict with trade result or None
    """
    if tk not in close.columns:
        return None

    S_entry = float(close[tk].iloc[di])
    dt_entry = close.index[di]

    # Get IV
    vix_val = float(vix_series.iloc[di]) if di < len(vix_series) else 20.0
    rv_21d = float(rv_series[tk].iloc[di]) if tk in rv_series and di < len(rv_series[tk]) else 0.20
    iv = compute_iv(vix_val, rv_21d)

    # Price the option at entry
    premium_per_share, K = price_single_leg(S_entry, otm_pct, direction, DTE, iv)

    # Cost per contract = 100 * premium_per_share
    contract_cost = 100.0 * premium_per_share

    if contract_cost < 0.50:
        return None  # option too cheap, likely deep OTM garbage

    # Position sizing: how many contracts can we buy?
    max_pos = get_max_position_size(equity)
    cost_with_commission = contract_cost + COMMISSION_PER_LEG
    if cost_with_commission <= 0:
        return None

    n_contracts = max(1, int(max_pos / cost_with_commission))
    # But also ensure we can actually afford it
    if n_contracts * cost_with_commission > equity * 0.95:
        n_contracts = max(1, int(equity * 0.95 / cost_with_commission))

    total_entry_cost = n_contracts * contract_cost + n_contracts * COMMISSION_PER_LEG

    if total_entry_cost > equity:
        return None

    # Simulate hold period -- check daily for profit target or hold to expiry
    exit_di = min(di + DTE, len(close) - 1)
    if exit_di <= di:
        return None

    exit_value_per_share = 0.0
    actual_exit_di = exit_di
    exit_reason = "expiry"

    for check_di in range(di + 1, exit_di + 1):
        S_now = float(close[tk].iloc[check_di])
        days_remaining = DTE - (check_di - di)
        T_rem = max(days_remaining, 0) / 365.0

        # Get current IV for revaluation
        vix_now = float(vix_series.iloc[check_di]) if check_di < len(vix_series) else vix_val
        rv_now = float(rv_series[tk].iloc[check_di]) if tk in rv_series and check_di < len(rv_series[tk]) else rv_21d
        iv_now = compute_iv(vix_now, rv_now)

        if days_remaining <= 0:
            # At expiry -- intrinsic value only
            if direction == "call":
                exit_value_per_share = max(S_now - K, 0.0)
            else:
                exit_value_per_share = max(K - S_now, 0.0)
            actual_exit_di = check_di
            exit_reason = "expiry"
            break

        # Mid-life revaluation via BS
        if direction == "call":
            current_val = bs_call_price(S_now, K, T_rem, RISK_FREE_RATE, iv_now)
        else:
            current_val = bs_put_price(S_now, K, T_rem, RISK_FREE_RATE, iv_now)

        # Check profit target
        if profit_target is not None and premium_per_share > 0:
            if current_val >= premium_per_share * (1.0 + profit_target):
                exit_value_per_share = current_val
                actual_exit_di = check_di
                exit_reason = "profit_target"
                break

        exit_value_per_share = current_val

    # If we reached expiry without breaking, use intrinsic
    if exit_reason == "expiry":
        S_exit = float(close[tk].iloc[actual_exit_di])
        if direction == "call":
            exit_value_per_share = max(S_exit - K, 0.0)
        else:
            exit_value_per_share = max(K - S_exit, 0.0)

    # P&L calculation
    exit_proceeds = n_contracts * 100.0 * exit_value_per_share
    exit_commission = n_contracts * COMMISSION_PER_LEG if exit_value_per_share > 0 else 0.0
    # No commission on worthless expiry

    total_pnl = exit_proceeds - total_entry_cost - exit_commission
    pnl_pct = total_pnl / total_entry_cost if total_entry_cost > 0 else 0.0

    return {
        "ticker": tk,
        "direction": direction,
        "entry_date": str(close.index[di].date()),
        "exit_date": str(close.index[actual_exit_di].date()),
        "entry_idx": di,
        "exit_idx": actual_exit_di,
        "S_entry": S_entry,
        "S_exit": float(close[tk].iloc[actual_exit_di]),
        "strike": K,
        "otm_pct": otm_pct,
        "iv": iv,
        "premium_per_share": premium_per_share,
        "contract_cost": contract_cost,
        "n_contracts": n_contracts,
        "total_entry_cost": total_entry_cost,
        "exit_value_per_share": exit_value_per_share,
        "exit_proceeds": exit_proceeds,
        "exit_commission": exit_commission,
        "pnl": total_pnl,
        "pnl_pct": pnl_pct,
        "exit_reason": exit_reason,
        "hold_days": actual_exit_di - di,
    }


# ==============================================================
# VARIANT DEFINITIONS
# ==============================================================

VARIANT_CONFIGS = {
    "A_atm_baseline": {
        "description": "ATM calls top-2 + ATM puts bottom-2",
        "otm_pct": 0.00,
        "top_k": 2,
        "bottom_k": 2,
        "calls": True,
        "puts": True,
        "profit_target": None,
    },
    "B_otm_2pct": {
        "description": "2% OTM calls top-2 + 2% OTM puts bottom-2",
        "otm_pct": 0.02,
        "top_k": 2,
        "bottom_k": 2,
        "calls": True,
        "puts": True,
        "profit_target": None,
    },
    "C_otm_4pct": {
        "description": "4% OTM calls top-2 + 4% OTM puts bottom-2",
        "otm_pct": 0.04,
        "top_k": 2,
        "bottom_k": 2,
        "calls": True,
        "puts": True,
        "profit_target": None,
    },
    "D_calls_only": {
        "description": "Top-4 calls only (bull-only, no puts)",
        "otm_pct": 0.00,
        "top_k": 4,
        "bottom_k": 0,
        "calls": True,
        "puts": False,
        "profit_target": None,
    },
    "E_puts_only": {
        "description": "Bottom-4 puts only (bear-only, no calls)",
        "otm_pct": 0.00,
        "top_k": 0,
        "bottom_k": 4,
        "calls": False,
        "puts": True,
        "profit_target": None,
    },
    "F_atm_profit_target": {
        "description": "ATM top-2/bottom-2 with 50% profit target (option doubles)",
        "otm_pct": 0.00,
        "top_k": 2,
        "bottom_k": 2,
        "calls": True,
        "puts": True,
        "profit_target": 1.0,  # exit when value >= 2x entry (100% gain = doubled)
    },
}


# ==============================================================
# BACKTEST ENGINE
# ==============================================================

def run_variant(variant_name, config, close, vix_series, rv_series, rankings):
    """Run a single variant backtest."""
    fprint(f"\n{'='*60}")
    fprint(f"VARIANT {variant_name}: {config['description']}")
    fprint(f"{'='*60}")

    otm_pct = config["otm_pct"]
    top_k = config["top_k"]
    bottom_k = config["bottom_k"]
    do_calls = config["calls"]
    do_puts = config["puts"]
    profit_target = config["profit_target"]

    trades = []
    equity = CAP
    equity_curve = []
    ranked_dates = sorted(rankings.keys())

    for di in ranked_dates:
        dt = close.index[di]
        scores = rankings[di]
        if len(scores) < max(top_k, bottom_k, 1):
            continue

        # Sort sectors by predicted score
        sorted_sectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_sectors = [s[0] for s in sorted_sectors[:top_k]] if top_k > 0 else []
        bottom_sectors = [s[0] for s in sorted_sectors[-bottom_k:]] if bottom_k > 0 else []

        # Total positions this period
        n_positions = (len(top_sectors) if do_calls else 0) + (len(bottom_sectors) if do_puts else 0)
        if n_positions == 0:
            continue

        # Allocate equity equally across positions
        alloc_per_position = equity / n_positions

        period_trades = []

        # Long calls on top sectors
        if do_calls:
            for tk in top_sectors:
                result = execute_single_leg_trade(
                    tk, di, "call", close, vix_series, rv_series,
                    alloc_per_position, otm_pct=otm_pct, profit_target=profit_target
                )
                if result is not None:
                    period_trades.append(result)

        # Long puts on bottom sectors
        if do_puts:
            for tk in bottom_sectors:
                result = execute_single_leg_trade(
                    tk, di, "put", close, vix_series, rv_series,
                    alloc_per_position, otm_pct=otm_pct, profit_target=profit_target
                )
                if result is not None:
                    period_trades.append(result)

        # Update equity
        period_pnl = sum(t["pnl"] for t in period_trades)
        equity += period_pnl
        equity = max(equity, 10.0)  # floor to avoid negative equity

        for t in period_trades:
            t["equity_after"] = equity
        trades.extend(period_trades)

        equity_curve.append({"date": str(dt.date()), "equity": equity, "pnl": period_pnl})

    fprint(f"  Total trades: {len(trades)}")
    fprint(f"  Final equity: ${equity:.2f} (started ${CAP:.2f})")
    return trades, equity_curve


# ==============================================================
# METRICS COMPUTATION
# ==============================================================

def compute_metrics(trades, equity_curve, close):
    """Compute validation metrics for a variant."""
    if not trades:
        return {"sharpe": 0, "win_rate": 0, "profit_factor": 0, "max_dd": 0,
                "total_return": 0, "n_trades": 0, "avg_pnl": 0}

    pnls = [t["pnl"] for t in trades]
    pnl_pcts = [t["pnl_pct"] for t in trades]
    n = len(trades)

    # Basic stats
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / n if n > 0 else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe on trade returns
    if len(pnl_pcts) > 1:
        mean_ret = np.mean(pnl_pcts)
        std_ret = np.std(pnl_pcts, ddof=1)
        # Annualize: ~12 rebalances/year (monthly), trades per rebalance varies
        periods_per_year = 252 / REBAL_FREQ  # ~12.6
        sharpe = mean_ret / (std_ret + 1e-10) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    # Max drawdown from equity curve
    if equity_curve:
        eq_vals = [CAP] + [e["equity"] for e in equity_curve]
        eq_arr = np.array(eq_vals)
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak
        max_dd = float(dd.min())
    else:
        max_dd = 0.0

    total_return = (sum(pnls)) / CAP

    # Sortino
    neg_rets = [r for r in pnl_pcts if r < 0]
    if len(neg_rets) > 1:
        downside_std = np.std(neg_rets, ddof=1)
        periods_per_year = 252 / REBAL_FREQ
        sortino = np.mean(pnl_pcts) / (downside_std + 1e-10) * np.sqrt(periods_per_year)
    else:
        sortino = sharpe  # no downside, use sharpe as proxy

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(min(profit_factor, 99.99), 3),
        "max_dd": round(max_dd, 4),
        "total_return": round(total_return, 4),
        "avg_pnl": round(np.mean(pnls), 2),
        "median_pnl": round(np.median(pnls), 2),
        "total_pnl": round(sum(pnls), 2),
        "avg_hold_days": round(np.mean([t["hold_days"] for t in trades]), 1),
    }


def permutation_test(trades, n_perms=N_PERMUTATIONS):
    """Permutation test: shuffle trade direction assignments, compute Sharpe distribution."""
    if len(trades) < 10:
        return {"p_value": 1.0, "z_score": 0.0}

    pnl_pcts = np.array([t["pnl_pct"] for t in trades])
    actual_sharpe = np.mean(pnl_pcts) / (np.std(pnl_pcts, ddof=1) + 1e-10)

    rng = np.random.RandomState(42)
    null_sharpes = []
    for _ in range(n_perms):
        # Shuffle the sign of PnLs (random direction assignment)
        signs = rng.choice([-1, 1], size=len(pnl_pcts))
        shuffled = pnl_pcts * signs
        s = np.mean(shuffled) / (np.std(shuffled, ddof=1) + 1e-10)
        null_sharpes.append(s)

    null_sharpes = np.array(null_sharpes)
    p_value = float(np.mean(null_sharpes >= actual_sharpe))
    z_score = float((actual_sharpe - np.mean(null_sharpes)) / (np.std(null_sharpes) + 1e-10))

    return {
        "p_value": round(p_value, 4),
        "z_score": round(z_score, 3),
        "actual_sharpe_raw": round(actual_sharpe, 4),
        "null_mean": round(float(np.mean(null_sharpes)), 4),
        "null_std": round(float(np.std(null_sharpes)), 4),
    }


def regime_stratification(trades, close):
    """Stratify trades by regime: green (SPY up) vs red (SPY down) days at entry."""
    if len(trades) < 5 or "SPY" not in close.columns:
        return {"green": {}, "red": {}}

    spy = close["SPY"]
    spy_daily_ret = spy.pct_change()

    green_trades = []
    red_trades = []

    for t in trades:
        di = t["entry_idx"]
        if di >= len(spy_daily_ret):
            continue
        spy_ret = float(spy_daily_ret.iloc[di])
        if spy_ret >= 0:
            green_trades.append(t)
        else:
            red_trades.append(t)

    def summarize(tlist):
        if not tlist:
            return {"n": 0, "sharpe": 0, "wr": 0, "avg_pnl": 0}
        pnl_pcts = [t["pnl_pct"] for t in tlist]
        wins = [p for p in pnl_pcts if p > 0]
        return {
            "n": len(tlist),
            "sharpe": round(np.mean(pnl_pcts) / (np.std(pnl_pcts, ddof=1) + 1e-10), 3) if len(pnl_pcts) > 1 else 0,
            "wr": round(len(wins) / len(tlist), 3),
            "avg_pnl": round(np.mean([t["pnl"] for t in tlist]), 2),
        }

    result = {
        "green": summarize(green_trades),
        "red": summarize(red_trades),
    }

    # Regime divergence check
    gs = result["green"].get("sharpe", 0)
    rs = result["red"].get("sharpe", 0)
    max_s = max(abs(gs), abs(rs), 1e-10)
    result["regime_divergence"] = round(abs(gs - rs) / max_s, 3)
    result["regime_agnostic"] = result["regime_divergence"] <= 0.50

    return result


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("V10 SINGLE-LEG OPTIONS BACKTEST V1")
    fprint(f"Started: {datetime.now().isoformat()}")
    fprint("=" * 70)

    # ---- Download data ----
    close = download_data()

    # ---- VIX and realized vol ----
    if "VIX" in close.columns:
        vix_series = close["VIX"]
    else:
        fprint("WARNING: VIX not found, using default IV=0.20")
        vix_series = pd.Series(20.0, index=close.index)

    # Realized vol (21d annualized) per sector
    rv_series = {}
    for tk in SECTORS:
        if tk in close.columns:
            rets = close[tk].pct_change()
            rv_series[tk] = rets.rolling(21).std() * np.sqrt(252)
            rv_series[tk] = rv_series[tk].fillna(0.20)

    # ---- Build rebalance schedule ----
    # Every 20 trading days, starting after enough data for features
    min_start_idx = 260  # need 252 days for features + buffer
    rebal_indices = list(range(min_start_idx, len(close) - DTE, REBAL_FREQ))
    fprint(f"Rebalance dates: {len(rebal_indices)} (every {REBAL_FREQ} trading days)")
    fprint(f"  First: {close.index[rebal_indices[0]].date()}, Last: {close.index[rebal_indices[-1]].date()}")

    # ---- Build features and rank ----
    fprint("\nBuilding feature matrix...")
    feat_df = build_feature_matrix(close, rebal_indices)

    fprint("\nRunning LGBM walk-forward ranking...")
    rankings = walk_forward_lgbm_rank(feat_df)

    if not rankings:
        fprint("ERROR: No rankings produced. Exiting.")
        return

    # ---- Run all variants ----
    all_results = {}
    for vname, vconfig in VARIANT_CONFIGS.items():
        trades, eq_curve = run_variant(vname, vconfig, close, vix_series, rv_series, rankings)
        metrics = compute_metrics(trades, eq_curve, close)
        perm = permutation_test(trades)
        regime = regime_stratification(trades, close)

        all_results[vname] = {
            "config": vconfig,
            "metrics": metrics,
            "permutation_test": perm,
            "regime_stratification": regime,
            "equity_curve": eq_curve,
            "n_trades": len(trades),
            "trades_sample": trades[:5] if trades else [],  # save first 5 for inspection
        }

        # Print summary
        m = metrics
        fprint(f"\n  --- {vname} Results ---")
        fprint(f"  Trades: {m['n_trades']}, Sharpe: {m['sharpe']}, Sortino: {m['sortino']}")
        fprint(f"  Win Rate: {m['win_rate']:.1%}, PF: {m['profit_factor']}, MaxDD: {m['max_dd']:.1%}")
        fprint(f"  Total Return: {m['total_return']:.1%}, Total PnL: ${m['total_pnl']:.2f}")
        fprint(f"  Avg Hold: {m['avg_hold_days']}d, Avg PnL: ${m['avg_pnl']:.2f}")
        fprint(f"  Permutation p={perm['p_value']}, z={perm['z_score']}")
        fprint(f"  Regime: green_sharpe={regime['green'].get('sharpe',0)}, "
               f"red_sharpe={regime['red'].get('sharpe',0)}, "
               f"divergence={regime.get('regime_divergence',0)}, "
               f"agnostic={regime.get('regime_agnostic','N/A')}")

    # ---- Print comparison table ----
    fprint("\n" + "=" * 100)
    fprint("VARIANT COMPARISON TABLE")
    fprint("=" * 100)
    header = f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'MaxDD':>7} {'TotRet':>8} {'Perm_p':>7} {'Regime':>7}"
    fprint(header)
    fprint("-" * 100)

    for vname, vdata in all_results.items():
        m = vdata["metrics"]
        p = vdata["permutation_test"]
        r = vdata["regime_stratification"]
        regime_ok = "PASS" if r.get("regime_agnostic", False) else "FAIL"
        perm_ok = "***" if p["p_value"] < 0.05 else ""
        fprint(f"{vname:<25} {m['n_trades']:>6} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
               f"{m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} {m['max_dd']:>6.1%} "
               f"{m['total_return']:>7.1%} {p['p_value']:>6.3f}{perm_ok} {regime_ok:>7}")

    fprint("-" * 100)

    # ---- Validation gates ----
    fprint("\n" + "=" * 70)
    fprint("VALIDATION GATE SUMMARY")
    fprint("=" * 70)
    for vname, vdata in all_results.items():
        m = vdata["metrics"]
        p = vdata["permutation_test"]
        r = vdata["regime_stratification"]

        gates = []
        gates.append(("Sharpe > 0.5", m["sharpe"] > 0.5))
        gates.append(("WR > 40%", m["win_rate"] > 0.40))
        gates.append(("PF > 1.0", m["profit_factor"] > 1.0))
        gates.append(("MaxDD < 50%", m["max_dd"] > -0.50))
        gates.append(("Perm p < 0.05", p["p_value"] < 0.05))
        gates.append(("Regime agnostic", r.get("regime_agnostic", False)))

        passed = sum(1 for _, v in gates if v)
        total = len(gates)
        status = "PASS" if passed == total else f"PARTIAL ({passed}/{total})"
        fprint(f"\n  {vname}: {status}")
        for gname, gval in gates:
            fprint(f"    {'[x]' if gval else '[ ]'} {gname}")

    # ---- Save results ----
    elapsed = time.time() - t0

    output_data = {
        "metadata": {
            "script": "v10_single_leg_backtest_v1.py",
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "account_size": CAP,
            "commission_per_leg": COMMISSION_PER_LEG,
            "dte": DTE,
            "wf_train_days": WF_TRAIN_DAYS,
            "rebal_freq": REBAL_FREQ,
            "n_permutations": N_PERMUTATIONS,
            "data_range": f"{close.index[0].date()} to {close.index[-1].date()}",
            "n_sectors": len(SECTORS),
        },
        "variants": {},
    }

    for vname, vdata in all_results.items():
        output_data["variants"][vname] = {
            "config": vdata["config"],
            "metrics": vdata["metrics"],
            "permutation_test": vdata["permutation_test"],
            "regime_stratification": vdata["regime_stratification"],
            "n_trades": vdata["n_trades"],
        }

    results_path = OUTPUT_DIR / "results_v1.json"
    with open(results_path, "w") as fp:
        json.dump(output_data, fp, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save equity curves separately
    eq_path = OUTPUT_DIR / "equity_curves_v1.json"
    eq_data = {vname: vdata["equity_curve"] for vname, vdata in all_results.items()}
    with open(eq_path, "w") as fp:
        json.dump(eq_data, fp, indent=2, default=str)
    fprint(f"Equity curves saved to {eq_path}")

    # ---- MLflow logging ----
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"single_leg_v1_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_param("account_size", CAP)
                mlflow.log_param("commission_per_leg", COMMISSION_PER_LEG)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("wf_train_days", WF_TRAIN_DAYS)
                mlflow.log_param("n_variants", len(VARIANT_CONFIGS))

                for vname, vdata in all_results.items():
                    m = vdata["metrics"]
                    for metric_name, metric_val in m.items():
                        if isinstance(metric_val, (int, float)):
                            mlflow.log_metric(f"{vname}_{metric_name}", metric_val)
                    p = vdata["permutation_test"]
                    mlflow.log_metric(f"{vname}_perm_pvalue", p["p_value"])

                mlflow.log_artifact(str(results_path))
                mlflow.log_artifact(str(eq_path))
            fprint("MLflow logging complete")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    fprint(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    fprint("DONE.")


if __name__ == "__main__":
    main()
