#!/usr/bin/env python3
"""
V9.3 Honest Pricing Audit v1
==============================

KB #282 discovered that ATR-based IV estimation underprices options by ~72.7%
vs market mid. This script re-runs V9.3 (DTE=28, biweekly rebal, 50% profit
target, LGBM 21-feature ranking) with THREE pricing variants:

  A: Current pricing -- ATR-based IV, 15% haircut (baseline, known-buggy)
  B: Linear correction -- BS_price * 1.10 + $0.066/share, 15% haircut
     (calibrated from KB #282: market_mid ~ 1.10 * BS_price + 6.60 per contract)
  C: High haircut -- same BS pricing, 90% haircut instead of 15%
     (KB #282: for DTE=28 4% OTM, the empirical haircut is ~90%)

Key question: Does V9.3's edge SURVIVE honest pricing?

SELF-CONTAINED: embeds BS pricing, does NOT import from research.tools.options_pricer.

Config (V9.3):
  - 21 LGBM features (17 production + vol_21d/vol_63d/maxdd_63d/sector_relative_vol_21d)
  - 500d sliding walk-forward window
  - Biweekly rebalance (every 10 trading days)
  - DTE=28, 2% OTM, adaptive max($3, 3%) spread width
  - 50% profit target exit (daily check)
  - VIX >= 20: bull call spreads only (top 2, confluence). VIX < 20: pair trades
  - Commission: $2.60 per spread RT ($0.65 per leg)
  - Starting capital: $645, max $200 per trade
  - Cost/width filter: reject if entry_cost / spread_width > 50%

Output: output/growth_research/v93_honest_pricing_audit_v1/
MLflow experiment: v93_honest_pricing_audit
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


# ==============================================================
# ENVIRONMENT DETECTION
# ==============================================================

_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")

if _NEPTUNE_BASE.exists():
    BASE = _NEPTUNE_BASE
    fprint(f"Running on Neptune: {BASE}")
else:
    BASE = _JUPITER_BASE
    fprint(f"Running on Jupiter: {BASE}")

OUTPUT_DIR = BASE / "output" / "growth_research" / "v93_honest_pricing_audit_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================
# CONSTANTS
# ==============================================================

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]

CAP = 645.0
DTE = 28
WF_TRAIN_DAYS = 500             # sliding window train size
REBAL_DAYS = 10                 # biweekly = every 10 trading days
TOP_K_BULL_VIX_HIGH = 2         # VIX >= 20: top 2 bull + confluence
TOP_K_PAIRS = 3                 # VIX < 20: top 3 bull + bottom 3 bear
PROFIT_TARGET_PCT = 0.50        # 50% of max profit
VIX_THRESHOLD = 20.0
OTM_PCT = 0.02                  # 2% OTM
WIDTH_FLOOR_USD = 3.0           # max($3, 3% of strike)
WIDTH_FLOOR_PCT = 0.03
COST_WIDTH_MAX = 0.50           # reject if entry_cost / width > 50%
MIN_POS_SIZE = 30.0             # minimum position size in $
MAX_POS_SIZE = 200.0            # max $200 per trade

RISK_FREE_RATE = 0.045
COMMISSION_RT = 2.60            # $0.65 per leg x 4 legs
EARLY_EXIT_COMM = 2.60          # additional commission for early exit

N_PERM = 2000                   # sign-flip permutation trials

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v93_honest_pricing_audit"


# ==============================================================
# 21 FEATURES (KB #264 production set)
# ==============================================================

FEATURES_21 = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d",
    "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "sector_relative_vol_21d", "cross_sector_dispersion",
]
assert len(FEATURES_21) == 21


# ==============================================================
# PRICING VARIANT DEFINITIONS
# ==============================================================

VARIANTS = {
    "A_current_pricing": {
        "haircut": 0.15,
        "linear_correction": False,
        "desc": "Current ATR-IV + 15% haircut (BASELINE, known underpriced)",
    },
    "B_linear_correction": {
        "haircut": 0.15,
        "linear_correction": True,
        "desc": "Linear correction: BS*1.10 + $0.066/sh (KB #282 calibration)",
    },
    "C_high_haircut_90pct": {
        "haircut": 0.90,
        "linear_correction": False,
        "desc": "Current BS + 90% haircut (KB #282 empirical DTE=28 4% OTM)",
    },
}


# ==============================================================
# MLFLOW SETUP
# ==============================================================

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
    """
    ATR-based IV estimate (Variant A, the BROKEN method from options_pricer).
    IV = (ATR/spot) * sqrt(252/14) * iv_multiplier
    iv_multiplier = 1.2 + 0.01 * max(VIX - 20, 0)
    """
    if spot <= 0 or atr <= 0:
        return 0.25
    realized_vol = (atr / spot) * np.sqrt(252.0 / 14.0)
    iv_mult = 1.2 + 0.01 * max(vix - 20.0, 0.0)
    return max(realized_vol * iv_mult, 0.10)


def price_spread(S, K1, K2, dte, atr, vix, direction, haircut, linear_correction):
    """
    Price a vertical spread with configurable pricing variant.

    Returns (entry_cost_ps, max_profit_ps) -- per-share values.
    """
    if K2 <= K1:
        return None, None

    T = dte / 365.0
    sigma = _estimate_iv(atr, S, vix)

    if direction == "bull":
        fair = _bs_call(S, K1, T, RISK_FREE_RATE, sigma) - _bs_call(S, K2, T, RISK_FREE_RATE, sigma)
    else:
        fair = _bs_put(S, K2, T, RISK_FREE_RATE, sigma) - _bs_put(S, K1, T, RISK_FREE_RATE, sigma)
    fair = max(fair, 0.001)

    if linear_correction:
        # KB #282 calibration: market_mid ~ 1.10 * BS_price + 6.60 per contract
        # Per share: market_mid_ps ~ 1.10 * BS_price + 0.066
        entry_cost = max(fair * 1.10 + 0.066, 0.01) * (1.0 + haircut)
    else:
        entry_cost = fair * (1.0 + haircut)

    spread_width = K2 - K1
    max_profit = spread_width - entry_cost

    return float(entry_cost), float(max_profit)


def revalue_spread_bs(S, K1, K2, dte_remaining, atr, vix, direction, haircut, linear_correction):
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

    if linear_correction:
        fair = max(fair * 1.10 + 0.066, 0.0)

    # Exit: receive LESS than fair (haircut down)
    return float(fair * (1.0 - haircut))


# ==============================================================
# DATA DOWNLOAD
# ==============================================================

def download_data():
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start="2018-01-01", progress=False, auto_adjust=True)
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
    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
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
                atr_dict[tk] = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return atr_dict


# ==============================================================
# FEATURE COMPUTATION (21 features)
# ==============================================================

def compute_sector_features(tk, px, spy_rets, close_df, dt_idx):
    """Compute 21 features for a single sector on a given date."""
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


# ==============================================================
# LGBM WALK-FORWARD RANKING (500d sliding)
# ==============================================================

def get_rebalance_dates(close):
    """Generate biweekly rebalance dates every 10 trading days."""
    trading_days = close.index
    dates = []
    start_idx = WF_TRAIN_DAYS
    for i in range(start_idx, len(trading_days), REBAL_DAYS):
        dates.append(trading_days[i])
    return pd.DatetimeIndex(dates)


def build_feature_records(close, high, low, rebal_dates):
    """Build one feature record per (date, sector) for walk-forward training."""
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy_rets = close["SPY"].pct_change().dropna() if "SPY" in close.columns else None

    for dt in rebal_dates:
        dt_idx = close.index.get_indexer([dt], method="ffill")[0]
        if dt_idx < WF_TRAIN_DAYS:
            continue
        for tk in sector_cols:
            px = close[tk].iloc[:dt_idx + 1].dropna()
            feat = compute_sector_features(tk, px, spy_rets, close, dt_idx)
            if feat is None:
                continue
            fi = min(dt_idx + DTE, len(close) - 1)
            if fi <= dt_idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[dt_idx] - 1)
            rec = {**feat, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in FEATURES_21:
        if c not in df.columns:
            df[c] = 0.0
    df[FEATURES_21] = df[FEATURES_21].fillna(0.0)
    fprint(f"  Feature records: {len(df)} ({len(df['date'].unique())} dates, 21 features)")
    return df


def walk_forward_lgbm_rank(df):
    """Walk-forward LGBM ranking: 500-day sliding window."""
    import lightgbm as lgb

    if len(df) < 50:
        return {}

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())
    rankings = {}

    for i, test_date in enumerate(dates):
        train_dates = [d for d in dates if d < test_date]
        if len(train_dates) < 20:
            continue

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()
        if len(test_df) < 3 or len(train_df) < 30:
            continue

        Xt = np.nan_to_num(train_df[FEATURES_21].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[FEATURES_21].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
            )
            m.fit(Xt, yt)
            preds = m.predict(Xe)
            test_df["score"] = preds
            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))
        except Exception as e:
            continue

    fprint(f"  Walk-forward ranking dates: {len(rankings)}")
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
# TRADE EXECUTION
# ==============================================================

def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity,
                  haircut, linear_correction):
    """
    Execute a single spread trade with V9.3 config + configurable pricing.

    Returns dict with trade details, or None if rejected.
    Includes 50% profit target daily check.
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

    # Price the entry with the specified pricing variant
    entry_cost_ps, max_profit_ps = price_spread(
        S, K1, K2, DTE, av, vix_val, direction, haircut, linear_correction
    )
    if entry_cost_ps is None or entry_cost_ps <= 0:
        return None

    spread_width = abs(K2 - K1)

    # Cost/width filter
    cwr = entry_cost_ps / max(spread_width, 0.01)
    if cwr > COST_WIDTH_MAX:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT
    if total_cost <= 0 or total_cost > MAX_POS_SIZE or total_cost > equity * 0.40:
        return None
    if total_cost < MIN_POS_SIZE:
        return None

    # Profit target exit logic (50% of max profit, daily check)
    exited_early = False
    exit_pos = expiry_pos
    exit_reason = "expiry"
    hold_days = DTE

    if PROFIT_TARGET_PCT > 0 and max_profit_ps > 0:
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

            cur_val = revalue_spread_bs(
                S_now, K1, K2, dte_rem, av_now, vix_val, direction,
                haircut, linear_correction
            )
            unrealized = cur_val - entry_cost_ps

            if unrealized >= PROFIT_TARGET_PCT * max_profit_ps:
                exited_early = True
                exit_pos = chk
                exit_reason = "pt_50pct"
                hold_days = days_held
                break

    # Compute P&L
    Se = float(close[tk].iloc[exit_pos])

    if exited_early:
        dte_at_exit = expiry_pos - exit_pos
        av_exit = (float(atr_dict[tk].loc[close.index[exit_pos]])
                   if close.index[exit_pos] in atr_dict[tk].index
                   else Se * 0.015)
        exit_val = revalue_spread_bs(
            Se, K1, K2, dte_at_exit, av_exit, vix_val, direction,
            haircut, linear_correction
        )
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

def simulate_variant(rankings, close, atr_dict, rebal_dates, haircut, linear_correction):
    """
    Run V9.3 backtest with specified pricing variant.

    VIX >= 20: bull spreads only (top 2 LGBM-ranked, confluence check)
    VIX  < 20: pair trades (top 3 bull + bottom 3 bear)
    """
    spy = close["SPY"] if "SPY" in close.columns else None
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    early_exits = 0

    for dt in sorted(rebal_dates):
        if spy is not None and dt not in spy.index:
            continue
        if dt not in rankings:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        if cv >= VIX_THRESHOLD:
            # Bull-only mode: top 2 with confluence check
            # Confluence: top pick must have score > median
            median_score = np.median([s for _, s in ranked_desc]) if len(ranked_desc) > 0 else 0.5
            bull_picks = [t for t, s in ranked_desc[:TOP_K_BULL_VIX_HIGH] if s > median_score]
            bear_picks = []
            n_positions = len(bull_picks)
        else:
            # Pairs mode: top 3 bull + bottom 3 bear
            bull_picks = [t for t, _ in ranked_desc[:TOP_K_PAIRS]]
            bear_picks = [t for t, _ in ranked_asc[:TOP_K_PAIRS]]
            n_positions = len(bull_picks) + len(bear_picks)

        if n_positions == 0:
            continue

        max_pos = min(MAX_POS_SIZE, equity / max(n_positions, 1))
        if max_pos < MIN_POS_SIZE:
            continue

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                result = execute_trade(
                    tk, dt, direction, close, atr_dict, cv, equity,
                    haircut, linear_correction
                )
                if result is None:
                    continue

                equity += result["pnl"]
                if result["exited_early"]:
                    early_exits += 1

                # Determine market regime during hold period
                dt_idx = close.index.get_loc(dt)
                exit_idx = min(dt_idx + result["hold_days"], len(close) - 1)
                spy_entry = float(spy.iloc[dt_idx]) if spy is not None else 100.0
                spy_exit = float(spy.iloc[exit_idx]) if spy is not None else spy_entry
                regime = "bull" if spy_exit >= spy_entry else "bear"

                trades.append({
                    **result,
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[exit_idx].date()),
                    "ticker": tk,
                    "direction": direction,
                    "regime": regime,
                    "vix": round(cv, 1),
                    "win": result["pnl"] > 0,
                    "n_positions": n_positions,
                })

    return trades, equity, early_exits


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


def _compute_sharpe_sortino(equity_series):
    """Honest monthly Sharpe + Sortino."""
    if len(equity_series) < 2:
        return 0.0, 0.0
    monthly_eq = equity_series.resample("ME").last().dropna()
    if len(monthly_eq) < 2:
        return 0.0, 0.0
    monthly_rets = monthly_eq.pct_change().dropna()
    if len(monthly_rets) < 2 or monthly_rets.std() == 0:
        return 0.0, 0.0
    sharpe = float(monthly_rets.mean() / monthly_rets.std() * np.sqrt(12))
    downside = monthly_rets[monthly_rets < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = float(monthly_rets.mean() / downside.std() * np.sqrt(12))
    else:
        sortino = sharpe * 1.5
    return sharpe, sortino


def compute_full_metrics(trades, initial_capital=CAP):
    """Compute honest metrics from a trade list."""
    if not trades:
        return {"sharpe": 0.0, "sortino": 0.0, "cagr": 0.0, "max_dd": -1.0,
                "win_rate": 0.0, "profit_factor": 0.0, "n_trades": 0,
                "final_equity": initial_capital, "total_pnl": 0.0, "total_return_pct": 0.0}

    pnls = np.array([t["pnl"] for t in trades])
    equity_vals = [initial_capital]
    for p in pnls:
        equity_vals.append(equity_vals[-1] + p)
    eq_arr = np.array(equity_vals)

    eq_series = _build_equity_series(trades, initial_capital)
    sharpe, sortino = _compute_sharpe_sortino(eq_series)

    # CAGR
    final_eq = eq_arr[-1]
    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    total_days = (df["exit_date"].max() - df["entry_date"].min()).days
    years = max(total_days / 365.25, 0.1)
    cagr = (final_eq / initial_capital) ** (1 / years) - 1 if final_eq > 0 else -1.0

    # Max drawdown
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(dd.min())

    # Win rate, profit factor
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / len(pnls)
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")

    total_pnl = float(pnls.sum())
    total_return_pct = (final_eq / initial_capital - 1) * 100

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr, 4),
        "max_dd": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "n_trades": len(pnls),
        "final_equity": round(final_eq, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 1),
    }


def run_5_gate_validation(trades, initial_capital=CAP, spy_prices=None, label=""):
    """5-gate adversarial validation. Returns dict with gate results."""
    if not trades or len(trades) < 10:
        fprint(f"  [{label}] Too few trades ({len(trades) if trades else 0}) for validation")
        return {"gates_passed": 0, "gates_total": 5, "gates": []}

    pnls = np.array([t["pnl"] for t in trades])
    eq_series = _build_equity_series(trades, initial_capital)
    real_sharpe, _ = _compute_sharpe_sortino(eq_series)

    gates = []

    # GATE 1: Sign-flip permutation (2000 trials)
    beat_count = 0
    dates = pd.to_datetime([t["exit_date"] for t in trades])
    for _ in range(N_PERM):
        signs = np.random.choice([-1, 1], size=len(pnls))
        flipped = pnls * signs
        eq_vals = [initial_capital]
        for p in flipped:
            eq_vals.append(eq_vals[-1] + p)
        eq_s = pd.Series(eq_vals, index=[dates[0] - pd.Timedelta(days=1)] + list(dates))
        eq_s = eq_s.groupby(eq_s.index).last()
        sh, _ = _compute_sharpe_sortino(eq_s)
        if real_sharpe > sh:
            beat_count += 1
    p_value = 1.0 - beat_count / N_PERM
    g1_pass = p_value < 0.05
    gates.append({"name": "sign_flip", "passed": g1_pass, "p_value": round(p_value, 4)})

    # GATE 2: Regime balance
    bull_trades = [t for t in trades if t.get("regime") == "bull"]
    bear_trades = [t for t in trades if t.get("regime") == "bear"]
    bull_pnl = sum(t["pnl"] for t in bull_trades) if bull_trades else 0
    bear_pnl = sum(t["pnl"] for t in bear_trades) if bear_trades else 0
    g2_pass = bull_pnl > 0 and bear_pnl > 0
    gates.append({"name": "regime_balance", "passed": g2_pass,
                  "bull_pnl": round(bull_pnl, 2), "bear_pnl": round(bear_pnl, 2)})

    # GATE 3: Sub-period stability (both halves profitable)
    mid = len(trades) // 2
    half1_pnl = sum(t["pnl"] for t in trades[:mid])
    half2_pnl = sum(t["pnl"] for t in trades[mid:])
    g3_pass = half1_pnl > 0 and half2_pnl > 0
    gates.append({"name": "subperiod_stability", "passed": g3_pass,
                  "half1_pnl": round(half1_pnl, 2), "half2_pnl": round(half2_pnl, 2)})

    # GATE 4: Outlier removal (still profitable without best month)
    df_trades = pd.DataFrame(trades)
    df_trades["exit_date"] = pd.to_datetime(df_trades["exit_date"])
    df_trades["month"] = df_trades["exit_date"].dt.to_period("M")
    monthly_pnl = df_trades.groupby("month")["pnl"].sum()
    best_month = monthly_pnl.idxmax()
    pnl_without_best = float(monthly_pnl.drop(best_month).sum())
    g4_pass = pnl_without_best > 0
    gates.append({"name": "outlier_removal", "passed": g4_pass,
                  "pnl_without_best_month": round(pnl_without_best, 2)})

    # GATE 5: Yearly consistency (>= 50% of years profitable)
    df_trades["year"] = df_trades["exit_date"].dt.year
    yearly_pnl = df_trades.groupby("year")["pnl"].sum()
    pct_profitable_years = float((yearly_pnl > 0).mean())
    g5_pass = pct_profitable_years >= 0.50
    gates.append({"name": "yearly_consistency", "passed": g5_pass,
                  "pct_profitable_years": round(pct_profitable_years, 3)})

    passed = sum(1 for g in gates if g["passed"])
    return {"gates_passed": passed, "gates_total": 5, "gates": gates}


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = datetime.now()
    fprint("=" * 110)
    fprint(f"V9.3 HONEST PRICING AUDIT v1 -- {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 110)
    fprint("KB #282: ATR-based IV underprices by ~72.7%. Testing if V9.3 edge survives honest pricing.")
    fprint(f"Config: DTE={DTE} | Biweekly rebal | 50% profit target | 21 LGBM features | $645 capital")
    fprint(f"Commission: ${COMMISSION_RT:.2f}/spread RT | Cost/width filter: {COST_WIDTH_MAX*100:.0f}%")
    fprint(f"\n{len(VARIANTS)} pricing variants:")
    for vn, vc in VARIANTS.items():
        fprint(f"  {vn}: {vc['desc']}")
    fprint()

    # ── DATA ──
    close, high, low = download_data()
    atr_dict = compute_atr_series(high, low, close)

    # ── LGBM RANKINGS (shared across all pricing variants) ──
    fprint(f"\n{'=' * 90}")
    fprint("BUILDING LGBM RANKINGS (500d sliding WF, 21 features, biweekly)")
    fprint(f"{'=' * 90}")
    rebal_dates = get_rebalance_dates(close)
    fprint(f"  Rebalance dates: {len(rebal_dates)} ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")
    records = build_feature_records(close, high, low, rebal_dates)
    rankings = walk_forward_lgbm_rank(records)

    if not rankings:
        fprint("ERROR: No rankings generated. Aborting.")
        return

    # ── SIMULATE ALL VARIANTS ──
    all_results = {}
    all_trades = {}
    spy_close = close["SPY"] if "SPY" in close.columns else None

    for vname, vcfg in VARIANTS.items():
        fprint(f"\n{'~' * 100}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"  Haircut: {vcfg['haircut']*100:.0f}% | Linear correction: {vcfg['linear_correction']}")
        fprint(f"{'~' * 100}")

        trades, final_eq, early_exits = simulate_variant(
            rankings, close, atr_dict, rebal_dates,
            haircut=vcfg["haircut"],
            linear_correction=vcfg["linear_correction"],
        )

        all_trades[vname] = trades

        if not trades or len(trades) < 5:
            fprint(f"  Only {len(trades) if trades else 0} trades. Skipping.")
            all_results[vname] = {"n_trades": len(trades) if trades else 0, "sharpe": 0.0}
            continue

        metrics = compute_full_metrics(trades)
        fprint(f"\n  Trades: {metrics['n_trades']} | Early exits: {early_exits} "
               f"({early_exits/max(len(trades),1)*100:.0f}%)")
        fprint(f"  Sharpe: {metrics['sharpe']:.2f} | Sortino: {metrics['sortino']:.2f} | "
               f"PF: {metrics['profit_factor']:.2f} | WR: {metrics['win_rate']*100:.1f}%")
        fprint(f"  CAGR: {metrics['cagr']*100:.1f}% | MDD: {metrics['max_dd']*100:.1f}% | "
               f"Total return: {metrics['total_return_pct']:.1f}%")
        fprint(f"  Final equity: ${metrics['final_equity']:,.0f} | Total PnL: ${metrics['total_pnl']:,.0f}")

        # Side breakdown
        for side in ["bull", "bear"]:
            st = [t for t in trades if t["direction"] == side]
            if st:
                sp = [t["pnl"] for t in st]
                wr = sum(1 for p in sp if p > 0) / len(sp)
                fprint(f"    {side}: {len(st)} trades, WR {wr:.1%}, PnL ${sum(sp):,.0f}")

        # Average entry cost comparison
        avg_entry = np.mean([t["entry_cost_ps"] for t in trades])
        avg_width = np.mean([t["spread_width"] for t in trades])
        avg_cwr = np.mean([t["cost_width_ratio"] for t in trades])
        fprint(f"    Avg entry cost: ${avg_entry:.3f}/sh | Avg width: ${avg_width:.2f} | Avg CWR: {avg_cwr:.3f}")

        # Early exit stats
        early_t = [t for t in trades if t["exited_early"]]
        if early_t:
            avg_hold_early = np.mean([t["hold_days"] for t in early_t])
            fprint(f"    Early exits: avg hold {avg_hold_early:.1f}d")

        # 5-gate adversarial validation
        fprint(f"\n  5-GATE ADVERSARIAL VALIDATION:")
        val_result = run_5_gate_validation(trades, CAP, spy_close, vname)
        for g in val_result["gates"]:
            status = "PASS" if g["passed"] else "FAIL"
            detail = {k: v for k, v in g.items() if k not in ("name", "passed")}
            fprint(f"    [{status}] {g['name']}: {detail}")
        fprint(f"    VERDICT: {val_result['gates_passed']}/{val_result['gates_total']} gates passed")

        all_results[vname] = {
            **metrics,
            "early_exits": early_exits,
            "early_exit_rate": round(early_exits / max(len(trades), 1), 4),
            "avg_entry_cost_ps": round(avg_entry, 4),
            "avg_cost_width_ratio": round(avg_cwr, 4),
            "gates_passed": val_result["gates_passed"],
            "gates_total": val_result["gates_total"],
            "gates": val_result["gates"],
            "description": vcfg["desc"],
        }

    # ══════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════

    fprint(f"\n{'=' * 140}")
    fprint("COMPARISON -- V9.3 ACROSS PRICING VARIANTS")
    fprint(f"{'=' * 140}")
    fprint(f"  {'Variant':<28} {'N':>5} {'Sharpe':>7} {'Sort':>7} {'WR':>6} "
           f"{'PF':>6} {'MDD':>7} {'Return':>8} {'Gate':>5} {'Avg$entry':>10} {'AvgCWR':>7}")
    fprint(f"  {'-' * 110}")
    for vn in VARIANTS:
        r = all_results.get(vn)
        if not r or r["n_trades"] < 5:
            fprint(f"  {vn:<28} {'-- insufficient trades --':>82}")
            continue
        fprint(f"  {vn:<28} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>7.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['total_return_pct']:>7.1f}% "
               f"{r['gates_passed']}/{r['gates_total']} "
               f"${r['avg_entry_cost_ps']:>8.3f} {r['avg_cost_width_ratio']:>6.3f}")

    # ══════════════════════════════════════════════════════════════
    # PRICING IMPACT ANALYSIS
    # ══════════════════════════════════════════════════════════════

    fprint(f"\n{'=' * 110}")
    fprint("PRICING IMPACT ANALYSIS")
    fprint(f"{'=' * 110}")

    baseline = all_results.get("A_current_pricing")
    if baseline and baseline["n_trades"] >= 5:
        for vn in ["B_linear_correction", "C_high_haircut_90pct"]:
            r = all_results.get(vn)
            if not r or r["n_trades"] < 5:
                continue
            sh_delta = r["sharpe"] - baseline["sharpe"]
            pnl_delta = r["total_pnl"] - baseline["total_pnl"]
            wr_delta = (r["win_rate"] - baseline["win_rate"]) * 100
            trade_delta = r["n_trades"] - baseline["n_trades"]
            entry_ratio = r["avg_entry_cost_ps"] / max(baseline["avg_entry_cost_ps"], 1e-6)

            fprint(f"\n  {vn} vs A_current_pricing:")
            fprint(f"    Sharpe delta: {sh_delta:+.2f} ({baseline['sharpe']:.2f} -> {r['sharpe']:.2f})")
            fprint(f"    PnL delta: ${pnl_delta:+,.0f} (${baseline['total_pnl']:,.0f} -> ${r['total_pnl']:,.0f})")
            fprint(f"    WR delta: {wr_delta:+.1f}pp ({baseline['win_rate']*100:.1f}% -> {r['win_rate']*100:.1f}%)")
            fprint(f"    Trade count delta: {trade_delta:+d} ({baseline['n_trades']} -> {r['n_trades']})")
            fprint(f"    Entry cost multiplier: {entry_ratio:.2f}x")

    # ══════════════════════════════════════════════════════════════
    # VERDICT
    # ══════════════════════════════════════════════════════════════

    fprint(f"\n{'=' * 110}")
    fprint("VERDICT: DOES V9.3 EDGE SURVIVE HONEST PRICING?")
    fprint(f"{'=' * 110}")

    surviving = []
    for vn in VARIANTS:
        r = all_results.get(vn)
        if not r or r["n_trades"] < 5:
            continue
        if r["sharpe"] > 0.5 and r["gates_passed"] >= 3:
            surviving.append((vn, r["sharpe"], r["gates_passed"]))

    if surviving:
        fprint(f"\n  SURVIVING variants (Sharpe > 0.5 AND >= 3/5 gates):")
        for vn, sh, gp in surviving:
            fprint(f"    {vn}: Sharpe {sh:.2f}, {gp}/5 gates")
    else:
        fprint(f"\n  NO variant survives the honest pricing test.")

    # Check if edge is real or pricing illusion
    a_res = all_results.get("A_current_pricing", {})
    b_res = all_results.get("B_linear_correction", {})
    c_res = all_results.get("C_high_haircut_90pct", {})

    if a_res.get("sharpe", 0) > 0.5 and b_res.get("sharpe", 0) <= 0:
        fprint(f"\n  CONCLUSION: V9.3 edge is a PRICING ILLUSION. Linear correction kills it.")
        fprint(f"  The ATR-based IV underpricing was inflating apparent edge.")
    elif a_res.get("sharpe", 0) > 0.5 and c_res.get("sharpe", 0) <= 0:
        fprint(f"\n  CONCLUSION: V9.3 edge SENSITIVE to pricing assumptions.")
        fprint(f"  90% haircut destroys it, but linear correction may preserve some edge.")
    elif b_res.get("sharpe", 0) > 0.5 and c_res.get("sharpe", 0) > 0.5:
        fprint(f"\n  CONCLUSION: V9.3 edge SURVIVES honest pricing! Robust to pricing corrections.")
    elif b_res.get("sharpe", 0) > 0.5:
        fprint(f"\n  CONCLUSION: V9.3 edge survives linear correction but not 90% haircut.")
        fprint(f"  Edge is real but margin-sensitive -- needs accurate pricing for production.")
    else:
        fprint(f"\n  CONCLUSION: Mixed results. Manual inspection recommended.")
        fprint(f"  A Sharpe: {a_res.get('sharpe', 'N/A')}, B: {b_res.get('sharpe', 'N/A')}, C: {c_res.get('sharpe', 'N/A')}")

    # ══════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════

    results_file = OUTPUT_DIR / "results.json"

    # Make JSON-serializable
    serializable = {}
    for k, v in all_results.items():
        sv = {}
        for k2, v2 in v.items():
            if isinstance(v2, (np.floating, np.integer)):
                sv[k2] = float(v2)
            elif isinstance(v2, list):
                sv[k2] = [
                    {k3: (float(v3) if isinstance(v3, (np.floating, np.integer)) else v3)
                     for k3, v3 in item.items()} if isinstance(item, dict) else item
                    for item in v2
                ]
            elif isinstance(v2, dict):
                sv[k2] = {k3: float(v3) if isinstance(v3, (np.floating, np.integer)) else v3
                          for k3, v3 in v2.items()}
            else:
                sv[k2] = v2
        serializable[k] = sv

    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_file}")

    # ── MLFLOW LOGGING ──
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"honest_pricing_{t0.strftime('%Y%m%d_%H%M')}"):
                for vn, r in all_results.items():
                    if r.get("n_trades", 0) < 5:
                        continue
                    prefix = vn.replace(" ", "_")
                    mlflow.log_metric(f"{prefix}_sharpe", r.get("sharpe", 0))
                    mlflow.log_metric(f"{prefix}_sortino", r.get("sortino", 0))
                    mlflow.log_metric(f"{prefix}_wr", r.get("win_rate", 0))
                    mlflow.log_metric(f"{prefix}_pf", r.get("profit_factor", 0))
                    mlflow.log_metric(f"{prefix}_n_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{prefix}_mdd", r.get("max_dd", 0))
                    mlflow.log_metric(f"{prefix}_total_pnl", r.get("total_pnl", 0))
                    mlflow.log_metric(f"{prefix}_gates_passed", r.get("gates_passed", 0))
                    mlflow.log_metric(f"{prefix}_avg_entry_cost", r.get("avg_entry_cost_ps", 0))
                    mlflow.log_metric(f"{prefix}_early_exit_rate", r.get("early_exit_rate", 0))
                mlflow.log_artifact(str(results_file))
            fprint(f"MLflow logged to '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow error: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nCompleted in {elapsed / 60:.1f} minutes")
    fprint(f"\n{'=' * 110}")
    fprint("DONE -- V9.3 Honest Pricing Audit v1")
    fprint(f"{'=' * 110}")


if __name__ == "__main__":
    main()
