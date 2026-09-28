#!/usr/bin/env python3
"""
Profit Target Stress Test V1 -- Adversarial Validation of 50% PT Finding
=========================================================================

The exit_strategy_optimizer_v1 found that a 50% profit target increases Sharpe
from 2.36 to 4.66 (+97%). This is a HUGE improvement that needs adversarial
stress testing before we believe it.

KEY QUESTION: Is 50% uniquely good, or does ANY profit target beat hold-to-expiry?
If ALL variants A-E beat the baseline, then "taking profits early" is generally
good (robust). If only 50% works, it might be overfit.

6 Variants:
  A: Profit target 30% -- close when gain >= 30% of max profit
  B: Profit target 40% -- close when gain >= 40% of max profit
  C: Profit target 50% -- the candidate (should match ~4.66 Sharpe)
  D: Profit target 60% -- close when gain >= 60% of max profit
  E: Profit target 70% -- close when gain >= 70% of max profit
  F: Profit target 50% + 3x commission -- cost sensitivity stress test
     Uses $7.80 commission ($2.60 * 3) instead of $2.60

Config: V9.2 -- DTE=28, monthly rebalance, 3% OTM, adaptive max($3,3%),
        LGBM 17 features, $645 capital, $2.60 commission (+$2.60 early exit),
        15% haircut, real Dolt chain data.

Output: output/growth_research/profit_target_stress_test_v1/
MLflow experiment: profit_target_stress_test_v1
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

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

# Detect environment -- Jupiter vs Neptune
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")

if _NEPTUNE_BASE.exists():
    BASE = _NEPTUNE_BASE
    fprint(f"Running on Neptune: {BASE}")
else:
    BASE = _JUPITER_BASE
    fprint(f"Running on Jupiter: {BASE}")

sys.path.insert(0, str(BASE))
from research.tools.options_pricer import (
    price_bull_call_spread, price_bear_put_spread, COMMISSION_RT_SPREAD,
)
from research.tools.adversarial_validator import validate_trades

CHAINS_DIR = BASE / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
OUTPUT_DIR = BASE / "output" / "growth_research" / "profit_target_stress_test_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
TOP_K = 3
DTE = 28
REGIME_BULL_THRESHOLD = 0.4
WF_TRAIN_PERIODS = 52  # 52-week sliding window
HAIRCUT = 0.15

COST_WIDTH_MAX = 0.50
DTE_TOLERANCE = 5
STRIKE_TOLERANCE = 0.03

# Early exit adds a second commission to close the position
EARLY_EXIT_COMMISSION = 2.60

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "profit_target_stress_test_v1"

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

V6_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "cross_sector_dispersion",
]
assert len(V6_FEATURES) == 17

# Monte Carlo settings
N_BOOTSTRAP = 1000


# ==============================================================
# CHAIN DATA
# ==============================================================

def load_all_chains():
    chains = {}
    for tk in SECTORS:
        path = CHAINS_DIR / f"{tk}.parquet"
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
        fprint(f"  {tk}: {len(df):,} rows")
    return chains


def find_chain_spread_price(chain_df, trade_date, direction, K1, K2, dte_target):
    if chain_df is None:
        return None
    day_mask = chain_df["date"] == pd.Timestamp(trade_date)
    chain_day = chain_df[day_mask]
    if chain_day.empty:
        nearby = chain_df[(chain_df["date"] >= pd.Timestamp(trade_date) - pd.Timedelta(days=2)) &
                          (chain_df["date"] <= pd.Timestamp(trade_date) + pd.Timedelta(days=2))]
        if nearby.empty:
            return None
        nearest_date = min(nearby["date"].unique(),
                          key=lambda x: abs((x - pd.Timestamp(trade_date)).days))
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
    """Revalue a spread using Black-Scholes at a given point in time.

    Returns the current spread value (what you'd receive if you closed it).
    For bull call spread: value = long_call(K1) - short_call(K2)
    For bear put spread: value = long_put(K2) - short_put(K1)
    """
    if dte_remaining <= 0:
        # At expiry, intrinsic value
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
    """Revalue a spread using chain data at a given date."""
    if chain_df is None:
        return None
    result = find_chain_spread_price(chain_df, check_date, direction, K1, K2, dte_remaining)
    if result and result["found"]:
        return result["spread_cost_mid"]
    return None


# ==============================================================
# DATA + FEATURES
# ==============================================================

def download_data():
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start="2008-01-01", progress=False, auto_adjust=True)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw[["Close"]]
    high = raw["High"] if mi else raw[["High"]]
    low = raw["Low"] if mi else raw[["Low"]]
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)
    close = close.ffill(); high = high.ffill(); low = low.ffill()
    rename_map = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)
    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
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


def compute_legacy_features(px, spy_slice):
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f["sharpe_63d"] = float(rets.iloc[-63:].mean() / (rets.iloc[-63:].std() + 1e-10) * np.sqrt(252)) if len(rets) > 63 else 0.0
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
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None
    if spy is None or len(spy) < 63:
        return {"sector_spy_beta_63d": 1.0, "cross_sector_dispersion": 0.01}
    spy_ret = spy.pct_change().dropna()
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
            h = high[tk].dropna(); l = low[tk].dropna(); c = close[tk].dropna()
            common = h.index.intersection(l.index).intersection(c.index)
            if len(common) > period:
                tr1 = h.loc[common] - l.loc[common]
                tr2 = (h.loc[common] - c.loc[common].shift(1)).abs()
                tr3 = (l.loc[common] - c.loc[common].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_dict[tk] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_dict


# ==============================================================
# LGBM WALK-FORWARD RANKING
# ==============================================================

def get_monthly_rebalance_dates(close):
    """Monthly rebalance -- first trading day of each month."""
    monthly = close.index.to_series().resample("MS").first().dropna()
    return pd.DatetimeIndex(monthly.values)


def get_weekly_fridays(close):
    """Every Friday -- used for building training features."""
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
    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


def walk_forward_lgbm_rank(df):
    """Walk-forward LGBM ranking with 52-week sliding window."""
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
    fprint(f"    {len(rankings)} ranking dates")
    return rankings


# ==============================================================
# STRIKES + EXECUTION
# ==============================================================

def compute_strikes(S, direction):
    if direction == "bull":
        K1 = round(S * 1.03, 2)  # 3% OTM
        pct_w = K1 * 0.03
        w = max(3.0, pct_w)
        K2 = round(K1 + w, 2)
    else:
        K2 = round(S * 0.97, 2)  # 3% OTM
        pct_w = K2 * 0.03
        w = max(3.0, pct_w)
        K1 = round(K2 - w, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


def execute_trade_with_profit_target(tk, dt, direction, close, atr_dict, vix_val, equity,
                                      chains, max_pos, target_pct=0.50,
                                      commission_override=None,
                                      early_exit_commission_override=None):
    """Execute a single spread trade with profit target early exit.

    Profit target logic (for both bull call spread and bear put spread):
      max_profit = (K2 - K1) - cost_paid  (per share)
      unrealized_gain = current_spread_value - cost_paid
      Close when: unrealized_gain >= target_pct * max_profit

    Each early exit incurs an additional commission to close.

    Args:
        target_pct: profit target as fraction of max profit (0.30, 0.40, 0.50, etc.)
        commission_override: override entry commission
        early_exit_commission_override: override early exit commission (for 3x cost stress test)
    """
    commission = commission_override if commission_override is not None else COMMISSION_RT_SPREAD
    exit_comm = early_exit_commission_override if early_exit_commission_override is not None else EARLY_EXIT_COMMISSION

    if tk not in close.columns or tk not in atr_dict:
        return None
    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    av = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015
    K1, K2 = compute_strikes(S, direction)

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

    total_cost = entry_cost_ps * 100 + commission
    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    # Max profit for this spread (per share)
    # For both bull call spread and bear put spread:
    # max_profit = spread_width - cost_paid
    max_profit_ps = spread_width - entry_cost_ps

    # -----------------------------------------------------------------
    # PROFIT TARGET EXIT LOGIC
    # Walk through each trading day from entry+1 to expiry
    # Close when: unrealized_gain >= target_pct * max_profit
    # -----------------------------------------------------------------
    exited_early = False
    exit_day_idx = ei  # default: expiry
    exit_reason = "expiry"
    hold_days = DTE

    if max_profit_ps > 0:
        for check_idx in range(di + 1, ei + 1):
            if check_idx >= len(close):
                break
            check_date = close.index[check_idx]
            dte_remaining = ei - check_idx  # trading days remaining
            days_held = check_idx - di

            S_now = float(close[tk].iloc[check_idx])
            av_now = float(atr_dict[tk].loc[check_date]) if check_date in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[check_date]) else S_now * 0.015
            vix_now = vix_val  # approximate

            # Try chain revaluation first, fall back to BS
            current_value_ps = None
            if chain_df is not None:
                current_value_ps = revalue_spread_chain(chain_df, check_date, direction, K1, K2, dte_remaining)
            if current_value_ps is None:
                current_value_ps = revalue_spread_bs(S_now, K1, K2, dte_remaining, av_now, vix_now, direction)

            # Unrealized gain per share = current_value - entry_cost
            unrealized_gain_ps = current_value_ps - entry_cost_ps

            # Check profit target
            if unrealized_gain_ps >= target_pct * max_profit_ps:
                exited_early = True
                exit_day_idx = check_idx
                exit_reason = f"profit_target_{int(target_pct*100)}pct"
                hold_days = days_held
                break

    # Compute final P&L
    Se = float(close[tk].iloc[exit_day_idx])

    if exited_early:
        # Early exit: use the spread value at exit time
        dte_at_exit = ei - exit_day_idx
        av_exit = float(atr_dict[tk].loc[close.index[exit_day_idx]]) if close.index[exit_day_idx] in atr_dict[tk].index else Se * 0.015

        exit_value_ps = None
        if chain_df is not None:
            exit_value_ps = revalue_spread_chain(chain_df, close.index[exit_day_idx], direction, K1, K2, dte_at_exit)
        if exit_value_ps is None:
            exit_value_ps = revalue_spread_bs(Se, K1, K2, dte_at_exit, av_exit, vix_val, direction)

        # P&L = (exit_value - entry_cost) * 100 - entry_commission - exit_commission
        pnl = (exit_value_ps - entry_cost_ps) * 100 - commission - exit_comm
    else:
        # Hold to expiry: intrinsic value
        if direction == "bull":
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
        else:
            intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - commission
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
        "target_pct": target_pct,
    }


# ==============================================================
# SIMULATION ENGINE
# ==============================================================

def simulate_variant(rankings, close, atr_dict, chains,
                     rebal_dates, variant_name="",
                     target_pct=0.50,
                     commission_override=None,
                     early_exit_commission_override=None):
    """Run backtest simulation for a profit target variant.

    Args:
        rankings: dict {date: {ticker: score}}
        close: price DataFrame
        atr_dict: ATR series per ticker
        chains: chain data dict
        rebal_dates: DatetimeIndex of rebalance dates
        variant_name: for logging
        target_pct: profit target as fraction of max profit
        commission_override: override commission per trade
        early_exit_commission_override: override early exit commission
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    real_count = 0
    bs_count = 0
    turnover_events = 0
    early_exits = 0

    for dt in sorted(rebal_dates):
        if dt not in spy.index:
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

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        k = TOP_K

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:k]]
        bear_picks = [t for t, _ in ranked_asc[:k]]

        turnover_events += 1
        n_positions = len(bull_picks) + len(bear_picks)

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                max_pos = min(CAP / n_positions * 2, equity * 0.40)
                if max_pos < 20:
                    continue

                result = execute_trade_with_profit_target(
                    tk, dt, direction, close, atr_dict, cv, equity,
                    chains, max_pos,
                    target_pct=target_pct,
                    commission_override=commission_override,
                    early_exit_commission_override=early_exit_commission_override,
                )
                if result is not None:
                    equity += result["pnl"]
                    real_count += 1 if result["used_real_pricing"] else 0
                    bs_count += 0 if result["used_real_pricing"] else 1
                    if result["exited_early"]:
                        early_exits += 1
                    di = close.index.get_loc(dt)
                    ei = min(di + DTE, len(close) - 1)
                    sv = float(spy.loc[dt])
                    se = float(spy.iloc[ei]) if ei < len(spy) else sv
                    trades.append({
                        **result, "entry_date": str(dt.date()),
                        "exit_date": str(close.index[di + result["hold_days"]].date()) if di + result["hold_days"] < len(close) else str(close.index[ei].date()),
                        "ticker": tk, "regime": "bull" if se >= sv else "bear",
                        "direction": direction, "vix": round(cv, 1),
                        "win": result["pnl"] > 0,
                        "n_positions": n_positions,
                    })

    return trades, equity, real_count, bs_count, turnover_events, early_exits


def compute_calmar_ratio(trades, initial_capital=CAP):
    if not trades:
        return 0.0
    equity = [initial_capital]
    for t in trades:
        equity.append(equity[-1] + t["pnl"])
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = abs(dd.min())
    if max_dd < 1e-10:
        return 0.0
    dates_str = [t["entry_date"] for t in trades]
    first_date = pd.Timestamp(min(dates_str))
    last_date = pd.Timestamp(max(dates_str))
    years = (last_date - first_date).days / 365.25
    if years < 0.5:
        return 0.0
    total_ret = equity[-1] / equity[0]
    cagr = total_ret ** (1 / years) - 1
    return cagr / max_dd


def compute_sortino(trades, initial_capital=CAP):
    if not trades or len(trades) < 5:
        return 0.0
    trade_df = pd.DataFrame(trades)
    trade_df["entry_month"] = pd.to_datetime(trade_df["entry_date"]).dt.to_period("M")
    monthly_pnl = trade_df.groupby("entry_month")["pnl"].sum()
    if len(monthly_pnl) < 3:
        return 0.0
    monthly_ret = monthly_pnl / initial_capital
    neg_ret = monthly_ret[monthly_ret < 0]
    downside_std = float(neg_ret.std()) if len(neg_ret) > 1 else 1e-10
    return float(monthly_ret.mean() / (downside_std + 1e-10) * np.sqrt(12))


def compute_max_dd(trades, initial_capital=CAP):
    if not trades:
        return 0.0
    equity = [initial_capital]
    for t in trades:
        equity.append(equity[-1] + t["pnl"])
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


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

    sortino = compute_sortino(trades, initial_capital)
    calmar = compute_calmar_ratio(trades, initial_capital)
    max_dd_pct = float(dd.min()) * 100

    # Early exit stats
    early_exits = sum(1 for t in trades if t.get("exited_early", False))
    early_exit_rate = early_exits / len(trades) if trades else 0.0
    hold_days_list = [t.get("hold_days", DTE) for t in trades]
    avg_hold = float(np.mean(hold_days_list))

    return {
        "n_trades": len(trades),
        "total_pnl": float(pnls.sum()),
        "avg_pnl": float(pnls.mean()),
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
    }


# ==============================================================
# MONTE CARLO BOOTSTRAP
# ==============================================================

def monte_carlo_bootstrap(trades, n_resamples=N_BOOTSTRAP, initial_capital=CAP):
    """Monte Carlo bootstrap resampling of trade-level P&L.

    Resamples trades with replacement, computes Sharpe/Sortino/PF/WR for each
    resample. Returns distribution statistics and confidence intervals.
    """
    if not trades or len(trades) < 10:
        return None

    pnls = np.array([t["pnl"] for t in trades])
    n_trades = len(pnls)

    # Build monthly P&L mapping for Sharpe computation
    trade_df = pd.DataFrame(trades)
    trade_df["entry_month"] = pd.to_datetime(trade_df["entry_date"]).dt.to_period("M")

    boot_sharpe = []
    boot_sortino = []
    boot_pf = []
    boot_wr = []
    boot_total_pnl = []

    rng = np.random.RandomState(42)

    for _ in range(n_resamples):
        # Resample trades with replacement
        idx = rng.choice(n_trades, size=n_trades, replace=True)
        sample_pnls = pnls[idx]

        # Win rate
        wr = float(np.mean(sample_pnls > 0))
        boot_wr.append(wr)

        # Profit factor
        gw = float(sample_pnls[sample_pnls > 0].sum()) if (sample_pnls > 0).any() else 0
        gl = float(abs(sample_pnls[sample_pnls < 0].sum())) if (sample_pnls < 0).any() else 1e-10
        boot_pf.append(gw / max(gl, 1e-10))

        # Total P&L
        boot_total_pnl.append(float(sample_pnls.sum()))

        # Approximate monthly Sharpe: group resampled trades into ~equal monthly buckets
        n_months = max(3, n_trades // 6)  # approximate monthly grouping
        chunk_size = max(1, n_trades // n_months)
        monthly_pnls = []
        for j in range(0, n_trades, chunk_size):
            monthly_pnls.append(sample_pnls[j:j+chunk_size].sum())
        monthly_pnls = np.array(monthly_pnls)
        if len(monthly_pnls) > 1 and monthly_pnls.std() > 1e-10:
            sharpe = float(monthly_pnls.mean() / monthly_pnls.std() * np.sqrt(12))
        else:
            sharpe = 0.0
        boot_sharpe.append(sharpe)

        # Sortino
        neg = monthly_pnls[monthly_pnls < 0]
        ds = float(neg.std()) if len(neg) > 1 else 1e-10
        sortino = float(monthly_pnls.mean() / (ds + 1e-10) * np.sqrt(12))
        boot_sortino.append(sortino)

    boot_sharpe = np.array(boot_sharpe)
    boot_sortino = np.array(boot_sortino)
    boot_pf = np.array(boot_pf)
    boot_wr = np.array(boot_wr)
    boot_total_pnl = np.array(boot_total_pnl)

    return {
        "n_resamples": n_resamples,
        "sharpe_mean": round(float(boot_sharpe.mean()), 3),
        "sharpe_std": round(float(boot_sharpe.std()), 3),
        "sharpe_ci_5": round(float(np.percentile(boot_sharpe, 5)), 3),
        "sharpe_ci_95": round(float(np.percentile(boot_sharpe, 95)), 3),
        "sharpe_pct_positive": round(float(np.mean(boot_sharpe > 0) * 100), 1),
        "sortino_mean": round(float(boot_sortino.mean()), 3),
        "sortino_ci_5": round(float(np.percentile(boot_sortino, 5)), 3),
        "sortino_ci_95": round(float(np.percentile(boot_sortino, 95)), 3),
        "pf_mean": round(float(boot_pf.mean()), 3),
        "pf_ci_5": round(float(np.percentile(boot_pf, 5)), 3),
        "pf_ci_95": round(float(np.percentile(boot_pf, 95)), 3),
        "wr_mean": round(float(boot_wr.mean()), 4),
        "wr_ci_5": round(float(np.percentile(boot_wr, 5)), 4),
        "wr_ci_95": round(float(np.percentile(boot_wr, 95)), 4),
        "total_pnl_mean": round(float(boot_total_pnl.mean()), 2),
        "total_pnl_ci_5": round(float(np.percentile(boot_total_pnl, 5)), 2),
        "total_pnl_ci_95": round(float(np.percentile(boot_total_pnl, 95)), 2),
        "pct_profitable": round(float(np.mean(boot_total_pnl > 0) * 100), 1),
    }


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"PROFIT TARGET STRESS TEST V1 -- {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Commission: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"Early exit adds: ${EARLY_EXIT_COMMISSION:.2f} extra commission to close")
    fprint(f"Universe: {len(SECTORS)} sector ETFs | Top/Bottom K: {TOP_K}")
    fprint(f"Rebalance: MONTHLY | Walk-forward: {WF_TRAIN_PERIODS}-week sliding")
    fprint(f"OTM: 3% | Width: adaptive max($3, 3%) | Haircut: {HAIRCUT*100:.0f}%")
    fprint(f"Monte Carlo: {N_BOOTSTRAP} bootstrap resamples per variant")
    fprint()
    fprint("KEY QUESTION: Is 50% profit target uniquely good, or does ANY profit target")
    fprint("beat hold-to-expiry? If ALL variants beat baseline, 'taking profits early'")
    fprint("is generally robust. If only 50% works, it might be overfit.")
    fprint()
    fprint("VARIANTS:")
    fprint("  A: Profit target 30% -- close when gain >= 30% of max profit")
    fprint("  B: Profit target 40% -- close when gain >= 40% of max profit")
    fprint("  C: Profit target 50% -- the candidate (~4.66 Sharpe expected)")
    fprint("  D: Profit target 60% -- close when gain >= 60% of max profit")
    fprint("  E: Profit target 70% -- close when gain >= 70% of max profit")
    fprint("  F: Profit target 50% + 3x commission ($7.80) -- cost sensitivity")
    fprint()

    fprint("Loading chains...")
    chains = load_all_chains()

    fprint("\nDownloading price data...")
    close, high, low = download_data()
    regime_series = load_regime_predictions()
    atr_dict = compute_atr_series(high, low, close)

    # Monthly rebalance dates (V9.2 optimal)
    monthly_dates = get_monthly_rebalance_dates(close)
    # Weekly dates for feature building
    weekly_dates = get_weekly_fridays(close)

    fprint(f"\nRebalance schedule: {len(monthly_dates)} monthly dates")
    fprint(f"Feature dates: {len(weekly_dates)} weekly dates (for LGBM training)")

    # Build features and LGBM rankings
    fprint(f"\n{'=' * 80}")
    fprint("BUILDING LGBM RANKINGS (52-week walk-forward, 17 momentum features)")
    fprint(f"{'=' * 80}")
    records = build_feature_records(close, high, low, weekly_dates, regime_series)
    rankings = walk_forward_lgbm_rank(records)

    if len(rankings) < 20:
        fprint(f"ERROR: Only {len(rankings)} ranking dates -- insufficient data")
        return

    spy_close = close["SPY"]

    # Also run hold-to-expiry baseline for comparison
    # We simulate it as a variant with target_pct=999 (never triggers)
    variants = [
        ("baseline_hold_to_expiry", 999.0, None, None, "BASELINE: Hold-to-expiry (no profit target)"),
        ("A_pt30", 0.30, None, None, "A: Profit target 30%"),
        ("B_pt40", 0.40, None, None, "B: Profit target 40%"),
        ("C_pt50", 0.50, None, None, "C: Profit target 50% (candidate)"),
        ("D_pt60", 0.60, None, None, "D: Profit target 60%"),
        ("E_pt70", 0.70, None, None, "E: Profit target 70%"),
        ("F_pt50_3x_cost", 0.50, None, 7.80, "F: Profit target 50% + 3x commission ($7.80)"),
    ]

    all_trades = {}
    all_stats = {}
    all_results = {}  # 5-gate validation results
    all_mc = {}  # Monte Carlo results

    for var_name, target_pct, comm_override, exit_comm_override, description in variants:
        fprint(f"\n{'=' * 90}")
        fprint(f"{description}")
        fprint(f"{'=' * 90}")

        trades, eq, real, bs, turns, early_ex = simulate_variant(
            rankings, close, atr_dict, chains, monthly_dates, var_name,
            target_pct=target_pct,
            commission_override=comm_override,
            early_exit_commission_override=exit_comm_override,
        )
        fprint(f"  Trades: {len(trades)} | Real: {real} | BS: {bs} | "
               f"Early exits: {early_ex} | Final: ${eq:,.0f}")

        all_trades[var_name] = trades
        stats_dict = compute_full_stats(trades)
        all_stats[var_name] = stats_dict

        if stats_dict:
            fprint(f"  Sharpe: {stats_dict['sharpe']:.2f} | Sortino: {stats_dict['sortino']:.2f} | "
                   f"Calmar: {stats_dict['calmar']:.2f}")
            fprint(f"  WR: {stats_dict['win_rate']*100:.1f}% | PF: {stats_dict['profit_factor']:.2f} | "
                   f"MDD: {stats_dict['max_dd_pct']:.1f}%")
            fprint(f"  Avg hold: {stats_dict['avg_hold_days']:.1f} days | "
                   f"Early exit rate: {stats_dict['early_exit_rate']*100:.1f}%")

        # 5-gate validation
        if len(trades) >= 5:
            try:
                result_val = validate_trades(trades, initial_capital=CAP, spy_prices=spy_close,
                                             strategy_name=var_name)
                result_val.print_summary()
                all_results[var_name] = result_val
            except Exception as e:
                fprint(f"  Validation error: {e}")

        # Monte Carlo bootstrap
        fprint(f"  Running Monte Carlo bootstrap ({N_BOOTSTRAP} resamples)...")
        mc = monte_carlo_bootstrap(trades, n_resamples=N_BOOTSTRAP)
        if mc:
            all_mc[var_name] = mc
            fprint(f"    Sharpe: {mc['sharpe_mean']:.2f} +/- {mc['sharpe_std']:.2f} "
                   f"[{mc['sharpe_ci_5']:.2f}, {mc['sharpe_ci_95']:.2f}] "
                   f"({mc['sharpe_pct_positive']:.0f}% positive)")
            fprint(f"    PF: {mc['pf_mean']:.2f} [{mc['pf_ci_5']:.2f}, {mc['pf_ci_95']:.2f}]")
            fprint(f"    WR: {mc['wr_mean']*100:.1f}% [{mc['wr_ci_5']*100:.1f}%, {mc['wr_ci_95']*100:.1f}%]")
            fprint(f"    Total PnL: ${mc['total_pnl_mean']:,.0f} "
                   f"[${mc['total_pnl_ci_5']:,.0f}, ${mc['total_pnl_ci_95']:,.0f}] "
                   f"({mc['pct_profitable']:.0f}% profitable)")

        # Side breakdown
        for side in ["bull", "bear"]:
            st = [t for t in trades if t["direction"] == side]
            if st:
                pnls = [t["pnl"] for t in st]
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                fprint(f"  {side}: {len(st)} trades, WR {wr:.1%}, PnL ${sum(pnls):,.0f}")

    # ==========================================================
    # COMPARISON TABLE
    # ==========================================================
    fprint(f"\n{'=' * 160}")
    fprint("COMPARISON -- ALL PROFIT TARGET VARIANTS")
    fprint(f"{'=' * 160}")

    fprint(f"\n  {'Variant':<25} {'N':>5} {'Sharpe':>7} {'Sort':>7} {'Calmar':>7} {'WR':>6} "
           f"{'PF':>6} {'MDD':>7} {'AvgHold':>8} {'EarlyX%':>8} {'Gate':>5} {'Final$':>9}")
    fprint(f"  {'-' * 150}")

    for var_name, target_pct, _, _, description in variants:
        stats_dict = all_stats.get(var_name)
        if stats_dict is None:
            continue
        result_val = all_results.get(var_name)
        gates_str = f"{result_val.to_dict()['gates_passed']}/{result_val.to_dict()['gates_total']}" if result_val else "--"

        fprint(f"  {var_name:<25} {stats_dict['n_trades']:>5} {stats_dict['sharpe']:>7.2f} "
               f"{stats_dict['sortino']:>7.2f} {stats_dict['calmar']:>7.2f} "
               f"{stats_dict['win_rate']*100:>5.1f}% {stats_dict['profit_factor']:>5.2f} "
               f"{stats_dict['max_dd_pct']:>6.1f}% {stats_dict['avg_hold_days']:>7.1f}d "
               f"{stats_dict['early_exit_rate']*100:>7.1f}% {gates_str:>5} "
               f"${stats_dict['final_equity']:>8,.0f}")

    # ==========================================================
    # DELTA ANALYSIS (vs baseline)
    # ==========================================================
    fprint(f"\n{'=' * 100}")
    fprint("DELTA vs BASELINE (Hold-to-expiry)")
    fprint(f"{'=' * 100}")

    base = all_stats.get("baseline_hold_to_expiry")
    if base:
        fprint(f"\n  {'Variant':<25} {'dSharpe':>8} {'dSort':>8} {'dWR':>7} {'dPF':>7} {'dMDD':>7} {'dPnL':>9}")
        fprint(f"  {'-' * 80}")
        for var_name, _, _, _, _ in variants[1:]:  # skip baseline
            s = all_stats.get(var_name)
            if s is None:
                continue
            fprint(f"  {var_name:<25} {s['sharpe']-base['sharpe']:>+7.2f} "
                   f"{s['sortino']-base['sortino']:>+7.2f} "
                   f"{(s['win_rate']-base['win_rate'])*100:>+6.1f}% "
                   f"{s['profit_factor']-base['profit_factor']:>+6.2f} "
                   f"{s['max_dd_pct']-base['max_dd_pct']:>+6.1f}% "
                   f"${s['total_pnl']-base['total_pnl']:>+8,.0f}")

    # ==========================================================
    # MONTE CARLO COMPARISON
    # ==========================================================
    fprint(f"\n{'=' * 130}")
    fprint("MONTE CARLO BOOTSTRAP COMPARISON (1000 resamples)")
    fprint(f"{'=' * 130}")

    fprint(f"\n  {'Variant':<25} {'Sharpe':>12} {'95%CI':>16} {'%Pos':>6} "
           f"{'PF':>8} {'95%CI':>16} {'PnL':>10} {'95%CI':>22}")
    fprint(f"  {'-' * 120}")

    for var_name, _, _, _, _ in variants:
        mc = all_mc.get(var_name)
        if mc is None:
            continue
        fprint(f"  {var_name:<25} {mc['sharpe_mean']:>7.2f}+-{mc['sharpe_std']:.2f} "
               f"[{mc['sharpe_ci_5']:>6.2f},{mc['sharpe_ci_95']:>6.2f}] "
               f"{mc['sharpe_pct_positive']:>5.0f}% "
               f"{mc['pf_mean']:>7.2f} [{mc['pf_ci_5']:>6.2f},{mc['pf_ci_95']:>6.2f}] "
               f"${mc['total_pnl_mean']:>9,.0f} "
               f"[${mc['total_pnl_ci_5']:>8,.0f},${mc['total_pnl_ci_95']:>8,.0f}]")

    # ==========================================================
    # EARLY EXIT ANALYSIS
    # ==========================================================
    fprint(f"\n{'=' * 100}")
    fprint("EARLY EXIT DETAIL")
    fprint(f"{'=' * 100}")

    for var_name, _, _, _, _ in variants[1:]:  # skip baseline
        trades = all_trades.get(var_name, [])
        early = [t for t in trades if t.get("exited_early", False)]
        held = [t for t in trades if not t.get("exited_early", False)]

        if not early:
            fprint(f"\n  {var_name}: No early exits triggered")
            continue

        early_pnls = [t["pnl"] for t in early]
        held_pnls = [t["pnl"] for t in held]

        fprint(f"\n  {var_name}:")
        fprint(f"    Early exits: {len(early)} ({len(early)/len(trades)*100:.1f}%)")
        fprint(f"      Avg PnL: ${np.mean(early_pnls):+.2f} | WR: {sum(1 for p in early_pnls if p>0)/len(early_pnls)*100:.1f}%")
        fprint(f"      Avg hold: {np.mean([t['hold_days'] for t in early]):.1f} days")
        if held:
            fprint(f"    Held to expiry: {len(held)} ({len(held)/len(trades)*100:.1f}%)")
            fprint(f"      Avg PnL: ${np.mean(held_pnls):+.2f} | WR: {sum(1 for p in held_pnls if p>0)/len(held_pnls)*100:.1f}%")

        # Exit reason breakdown
        reasons = {}
        for t in early:
            r = t.get("exit_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        fprint(f"    Exit reasons: {reasons}")

    # ==========================================================
    # REGIME ANALYSIS (all variants)
    # ==========================================================
    fprint(f"\n{'=' * 100}")
    fprint("REGIME ANALYSIS -- ALL VARIANTS")
    fprint(f"{'=' * 100}")

    for var_name, _, _, _, _ in variants:
        trades = all_trades.get(var_name, [])
        if not trades:
            continue
        fprint(f"\n  {var_name}:")
        for regime in ["bull", "bear"]:
            rt = [t for t in trades if t["regime"] == regime]
            if rt:
                pnls = [t["pnl"] for t in rt]
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                fprint(f"    {regime} regime: {len(rt)} trades, WR {wr:.1%}, PnL ${sum(pnls):,.0f}")

    # ==========================================================
    # 5-GATE VALIDATION SUMMARY
    # ==========================================================
    fprint(f"\n{'=' * 100}")
    fprint("5-GATE VALIDATION SUMMARY")
    fprint(f"{'=' * 100}")

    for var_name, _, _, _, _ in variants:
        result_val = all_results.get(var_name)
        if result_val is None:
            fprint(f"\n  {var_name}: No validation (insufficient trades)")
            continue
        r = result_val.to_dict()
        status = "PASS" if r["all_passed"] else "FAIL"
        fprint(f"\n  [{status}] {var_name}: {r['gates_passed']}/{r['gates_total']} gates")
        if r.get("gates"):
            for g in r["gates"]:
                gs = "PASS" if g["passed"] else "FAIL"
                fprint(f"         [{gs}] {g['name']}: {g['metric_name']}={g['metric_value']:.4f}")

    # ==========================================================
    # MONOTONICITY CHECK
    # ==========================================================
    fprint(f"\n{'=' * 100}")
    fprint("MONOTONICITY CHECK -- Does Sharpe vary smoothly with target %?")
    fprint(f"{'=' * 100}")

    pt_variants = ["A_pt30", "B_pt40", "C_pt50", "D_pt60", "E_pt70"]
    pt_pcts = [30, 40, 50, 60, 70]
    pt_sharpes = []
    for vn in pt_variants:
        s = all_stats.get(vn)
        pt_sharpes.append(s["sharpe"] if s else 0.0)

    fprint(f"\n  Target%  Sharpe")
    fprint(f"  {'-' * 20}")
    for pct, sh in zip(pt_pcts, pt_sharpes):
        marker = " <-- candidate" if pct == 50 else ""
        fprint(f"    {pct}%    {sh:.2f}{marker}")

    # Check if monotonically increasing then decreasing (inverted U)
    peak_idx = np.argmax(pt_sharpes)
    fprint(f"\n  Peak at: {pt_pcts[peak_idx]}% (Sharpe {pt_sharpes[peak_idx]:.2f})")

    # Check if ALL beat baseline
    base_sharpe = base["sharpe"] if base else 0.0
    all_beat_baseline = all(sh > base_sharpe for sh in pt_sharpes)
    fprint(f"  Baseline Sharpe: {base_sharpe:.2f}")
    fprint(f"  ALL variants beat baseline: {'YES' if all_beat_baseline else 'NO'}")

    beats = [pct for pct, sh in zip(pt_pcts, pt_sharpes) if sh > base_sharpe]
    fprint(f"  Variants that beat baseline: {beats}")

    # Smoothness check
    diffs = [pt_sharpes[i+1] - pt_sharpes[i] for i in range(len(pt_sharpes)-1)]
    fprint(f"  Sharpe diffs (30->40->50->60->70): {[f'{d:+.2f}' for d in diffs]}")

    if all_beat_baseline:
        fprint(f"\n  --> FINDING: ALL profit targets (30-70%) beat hold-to-expiry.")
        fprint(f"      This suggests 'taking profits early' is GENERALLY robust,")
        fprint(f"      not just a lucky 50% finding. Reduces overfit concern.")
    else:
        only_beats = beats
        fprint(f"\n  --> WARNING: Only {only_beats} beat baseline.")
        fprint(f"      The 50% target may be overfit if nearby targets fail.")

    # ==========================================================
    # COST SENSITIVITY
    # ==========================================================
    fprint(f"\n{'=' * 100}")
    fprint("COST SENSITIVITY -- Does 50% survive 3x commission?")
    fprint(f"{'=' * 100}")

    c_stats = all_stats.get("C_pt50")
    f_stats = all_stats.get("F_pt50_3x_cost")
    if c_stats and f_stats:
        fprint(f"\n  C (50%, $2.60 exit comm): Sharpe {c_stats['sharpe']:.2f}, "
               f"PF {c_stats['profit_factor']:.2f}, WR {c_stats['win_rate']*100:.1f}%")
        fprint(f"  F (50%, $7.80 exit comm): Sharpe {f_stats['sharpe']:.2f}, "
               f"PF {f_stats['profit_factor']:.2f}, WR {f_stats['win_rate']*100:.1f}%")
        sharpe_drop = c_stats['sharpe'] - f_stats['sharpe']
        fprint(f"  Sharpe drop from 3x cost: {sharpe_drop:+.2f}")
        if f_stats['sharpe'] > base_sharpe:
            fprint(f"  --> ROBUST: Even with 3x commission, 50% PT still beats baseline ({base_sharpe:.2f})")
        else:
            fprint(f"  --> WARNING: 3x commission kills the edge. Sensitive to costs.")

    # ==========================================================
    # VERDICT
    # ==========================================================
    fprint(f"\n{'=' * 100}")
    fprint("PROFIT TARGET STRESS TEST VERDICT")
    fprint(f"{'=' * 100}")

    best_variant = None
    best_sharpe = -999
    verdict_notes = []

    for var_name, _, _, _, _ in variants:
        s = all_stats.get(var_name)
        if s is None:
            continue
        if s["sharpe"] > best_sharpe:
            best_sharpe = s["sharpe"]
            best_variant = var_name

    if base:
        fprint(f"\n  Baseline (hold-to-expiry): Sharpe {base['sharpe']:.2f}, "
               f"Sortino {base['sortino']:.2f}, PF {base['profit_factor']:.2f}, "
               f"WR {base['win_rate']*100:.1f}%, MDD {base['max_dd_pct']:.1f}%")

    for var_name, target_pct, _, _, desc in variants[1:]:
        s = all_stats.get(var_name)
        if s is None:
            continue
        delta_sharpe = s["sharpe"] - base["sharpe"] if base else 0
        mc = all_mc.get(var_name)
        mc_str = f" (MC 90%CI: [{mc['sharpe_ci_5']:.2f},{mc['sharpe_ci_95']:.2f}])" if mc else ""
        fprint(f"  {var_name}: Sharpe {s['sharpe']:.2f} ({delta_sharpe:+.2f}){mc_str}")

        if delta_sharpe > 0.50:
            verdict_notes.append(f"{var_name}: STRONG improvement ({delta_sharpe:+.2f})")
        elif delta_sharpe > 0.10:
            verdict_notes.append(f"{var_name}: Moderate improvement ({delta_sharpe:+.2f})")
        elif delta_sharpe > -0.10:
            verdict_notes.append(f"{var_name}: Neutral ({delta_sharpe:+.2f})")
        else:
            verdict_notes.append(f"{var_name}: Hurts performance ({delta_sharpe:+.2f})")

    fprint(f"\n  BEST VARIANT: {best_variant} (Sharpe {best_sharpe:.2f})")
    fprint(f"\n  KEY FINDINGS:")
    for note in verdict_notes:
        fprint(f"    {note}")

    fprint(f"\n  INTERPRETATION:")
    if all_beat_baseline:
        fprint(f"    The profit target finding is ROBUST -- all targets 30-70% improve")
        fprint(f"    over hold-to-expiry. This is 'take profits early' as a general")
        fprint(f"    principle, not 50%-specific overfitting.")
    else:
        fprint(f"    Mixed results -- not all targets beat baseline. The optimal target")
        fprint(f"    may be data-dependent. Use with caution.")

    if f_stats and f_stats['sharpe'] > base_sharpe:
        fprint(f"    Cost sensitivity: PASSES -- even 3x commission doesn't kill the edge.")
    elif f_stats:
        fprint(f"    Cost sensitivity: FAILS -- edge is cost-sensitive. Careful with execution.")

    # -- Save --
    save_results = {
        "timestamp": t0.isoformat(),
        "config": {
            "capital": CAP, "dte": DTE, "otm_pct": 0.03,
            "commission": COMMISSION_RT_SPREAD,
            "early_exit_commission": EARLY_EXIT_COMMISSION,
            "rebalance": "monthly",
            "top_k": TOP_K, "wf_weeks": WF_TRAIN_PERIODS,
            "n_sectors": len(SECTORS), "features": len(V6_FEATURES),
            "haircut": HAIRCUT,
            "n_bootstrap": N_BOOTSTRAP,
        },
        "best_variant": best_variant,
        "best_sharpe": best_sharpe,
        "all_beat_baseline": all_beat_baseline,
        "monotonicity": {
            "targets": pt_pcts,
            "sharpes": pt_sharpes,
            "peak_target": pt_pcts[peak_idx],
        },
        "verdict_notes": verdict_notes,
    }

    for var_name, _, _, _, _ in variants:
        s = all_stats.get(var_name)
        if s:
            save_results[var_name] = s
        rv = all_results.get(var_name)
        if rv:
            save_results[f"{var_name}_validation"] = rv.to_dict()
        mc = all_mc.get(var_name)
        if mc:
            save_results[f"{var_name}_monte_carlo"] = mc

    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_file}")

    # Save trade details
    for var_name, _, _, _, _ in variants:
        trades = all_trades.get(var_name, [])
        if trades:
            tf = OUTPUT_DIR / f"trades_{var_name}.json"
            with open(tf, "w") as f:
                json.dump(trades, f, indent=1, default=str)

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"pt_stress_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_param("capital", CAP)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("otm_pct", 0.03)
                mlflow.log_param("commission", COMMISSION_RT_SPREAD)
                mlflow.log_param("early_exit_commission", EARLY_EXIT_COMMISSION)
                mlflow.log_param("rebalance", "monthly")
                mlflow.log_param("top_k", TOP_K)
                mlflow.log_param("wf_train_weeks", WF_TRAIN_PERIODS)
                mlflow.log_param("n_sectors", len(SECTORS))
                mlflow.log_param("n_features", len(V6_FEATURES))
                mlflow.log_param("n_bootstrap", N_BOOTSTRAP)
                mlflow.log_param("best_variant", best_variant)
                mlflow.log_param("all_beat_baseline", all_beat_baseline)

                for var_name, _, _, _, _ in variants:
                    s = all_stats.get(var_name)
                    if s is None:
                        continue
                    prefix = var_name.replace("baseline_hold_to_expiry", "base")
                    if prefix.startswith("base"):
                        prefix = "base"
                    else:
                        prefix = prefix.split("_")[0]  # A, B, C, D, E, F
                    mlflow.log_metric(f"{prefix}_sharpe", s["sharpe"])
                    mlflow.log_metric(f"{prefix}_sortino", s["sortino"])
                    mlflow.log_metric(f"{prefix}_calmar", s["calmar"])
                    mlflow.log_metric(f"{prefix}_wr", s["win_rate"])
                    mlflow.log_metric(f"{prefix}_pf", s["profit_factor"])
                    mlflow.log_metric(f"{prefix}_mdd", s["max_dd_pct"])
                    mlflow.log_metric(f"{prefix}_n_trades", s["n_trades"])
                    mlflow.log_metric(f"{prefix}_early_exit_rate", s["early_exit_rate"])
                    mlflow.log_metric(f"{prefix}_avg_hold", s["avg_hold_days"])
                    mlflow.log_metric(f"{prefix}_total_pnl", s["total_pnl"])

                    mc = all_mc.get(var_name)
                    if mc:
                        mlflow.log_metric(f"{prefix}_mc_sharpe_mean", mc["sharpe_mean"])
                        mlflow.log_metric(f"{prefix}_mc_sharpe_ci5", mc["sharpe_ci_5"])
                        mlflow.log_metric(f"{prefix}_mc_sharpe_ci95", mc["sharpe_ci_95"])

                    rv = all_results.get(var_name)
                    if rv:
                        rd = rv.to_dict()
                        mlflow.log_metric(f"{prefix}_gates_passed", rd["gates_passed"])

                mlflow.log_artifact(str(results_file))
            fprint(f"MLflow logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow error: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nCompleted in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
