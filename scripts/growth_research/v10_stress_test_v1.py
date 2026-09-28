#!/usr/bin/env python3
"""
V10 Stress Test V1 -- Pre-Deployment Adversarial Validation
============================================================

V10 achieved Sharpe 6.32 -- our best ever. This stress test validates the
config before deployment.

V10 Config:
  - 8 positions (top 4 bull + bottom 4 bear)
  - 4% OTM
  - 30% profit target (close when gain >= 30% of max profit)
  - Monthly rebalance
  - DTE=28
  - LGBM 17 momentum features
  - $645 capital, $2.60 commission, 15% haircut
  - Adaptive width: max($3, 3%)
  - Real Dolt chain data

6 Stress Test Variants:
  A: V10 Baseline -- The V10 D config as-is
  B: 3x Commission -- Commission $7.80 instead of $2.60 (cost sensitivity)
  C: 25% Haircut -- Entry haircut 25% instead of 15% (pricing assumption)
  D: Remove Top 3 Tickers -- Remove 3 highest-PnL tickers (breadth test)
  E: COVID Window -- Only trades 2020-01-01 to 2021-12-31 (extreme regime)
  F: Bear 2022 -- Only trades 2022-01-01 to 2022-12-31 (bear market)

5-gate adversarial validation + Monte Carlo 1000 bootstrap on each variant.

Output: output/growth_research/v10_stress_test_v1/
MLflow experiment: v10_stress_test_v1
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
OUTPUT_DIR = BASE / "output" / "growth_research" / "v10_stress_test_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]

# ---- V10 Config ----
CAP = 645.0
TOP_K = 4              # V10: top 4 + bottom 4 = 8 positions
OTM_PCT = 0.04         # V10: 4% OTM (was 3% in V9.2)
PROFIT_TARGET = 0.30   # V10: 30% profit target
DTE = 28
HAIRCUT = 0.15
COMMISSION = 2.60      # per trade
EARLY_EXIT_COMMISSION = 2.60  # extra commission to close early

REGIME_BULL_THRESHOLD = 0.4
WF_TRAIN_PERIODS = 52  # 52-week sliding window

COST_WIDTH_MAX = 0.50
DTE_TOLERANCE = 5
STRIKE_TOLERANCE = 0.03

N_BOOTSTRAP = 1000

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v10_stress_test_v1"

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
    """Revalue a spread using Black-Scholes at a given point in time."""
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

def compute_strikes(S, direction, otm_pct=OTM_PCT):
    """Compute strike prices for V10 config: otm_pct OTM, adaptive max($3,3%) width."""
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


def execute_trade_with_profit_target(tk, dt, direction, close, atr_dict, vix_val, equity,
                                      chains, max_pos, target_pct=PROFIT_TARGET,
                                      commission_override=None,
                                      haircut_override=None):
    """Execute a spread trade with profit target early exit.

    Profit target logic:
      max_profit = (K2 - K1) - cost_paid  (per share)
      unrealized_gain = current_spread_value - cost_paid
      Close when: unrealized_gain >= target_pct * max_profit
    """
    comm = commission_override if commission_override is not None else COMMISSION
    haircut = haircut_override if haircut_override is not None else HAIRCUT

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

    # Apply haircut to entry price (we pay more than mid)
    entry_cost_ps = entry_cost_ps * (1.0 + haircut)

    spread_width = abs(K2 - K1)
    cwr = entry_cost_ps / max(spread_width, 0.01)
    if cwr > COST_WIDTH_MAX:
        return None

    total_cost = entry_cost_ps * 100 + comm
    n_pos = TOP_K * 2  # total positions
    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    # Max profit for this spread (per share)
    max_profit_ps = spread_width - entry_cost_ps

    # -----------------------------------------------------------------
    # PROFIT TARGET EXIT LOGIC
    # Walk through each trading day from entry+1 to expiry
    # Close when: unrealized_gain >= target_pct * max_profit
    # -----------------------------------------------------------------
    exited_early = False
    exit_day_idx = ei
    exit_reason = "expiry"
    hold_days = DTE

    if max_profit_ps > 0 and target_pct < 1.0:
        for check_idx in range(di + 1, ei + 1):
            if check_idx >= len(close):
                break
            check_date = close.index[check_idx]
            dte_remaining = ei - check_idx
            days_held = check_idx - di

            S_now = float(close[tk].iloc[check_idx])
            av_now = float(atr_dict[tk].loc[check_date]) if check_date in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[check_date]) else S_now * 0.015
            vix_now = vix_val

            # Try chain revaluation first, fall back to BS
            current_value_ps = None
            if chain_df is not None:
                current_value_ps = revalue_spread_chain(chain_df, check_date, direction, K1, K2, dte_remaining)
            if current_value_ps is None:
                current_value_ps = revalue_spread_bs(S_now, K1, K2, dte_remaining, av_now, vix_now, direction)

            unrealized_gain_ps = current_value_ps - entry_cost_ps

            if unrealized_gain_ps >= target_pct * max_profit_ps:
                exited_early = True
                exit_day_idx = check_idx
                exit_reason = f"profit_target_{int(target_pct*100)}pct"
                hold_days = days_held
                break

    # Compute final P&L
    Se = float(close[tk].iloc[exit_day_idx])

    if exited_early:
        dte_at_exit = ei - exit_day_idx
        av_exit = float(atr_dict[tk].loc[close.index[exit_day_idx]]) if close.index[exit_day_idx] in atr_dict[tk].index else Se * 0.015

        exit_value_ps = None
        if chain_df is not None:
            exit_value_ps = revalue_spread_chain(chain_df, close.index[exit_day_idx], direction, K1, K2, dte_at_exit)
        if exit_value_ps is None:
            exit_value_ps = revalue_spread_bs(Se, K1, K2, dte_at_exit, av_exit, vix_val, direction)

        # P&L = (exit_value - entry_cost) * 100 - entry_commission - exit_commission
        pnl = (exit_value_ps - entry_cost_ps) * 100 - comm - EARLY_EXIT_COMMISSION
    else:
        if direction == "bull":
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
        else:
            intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - comm
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
                     target_pct=PROFIT_TARGET,
                     commission_override=None,
                     haircut_override=None,
                     exclude_tickers=None,
                     date_filter=None):
    """Run backtest simulation for a variant."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None
    exclude_set = set(exclude_tickers) if exclude_tickers else set()

    equity = CAP
    trades = []
    real_count = 0
    bs_count = 0
    turnover_events = 0
    early_exits = 0

    for dt in sorted(rebal_dates):
        if dt not in spy.index:
            continue
        # Apply date filter
        if date_filter is not None:
            if dt < pd.Timestamp(date_filter[0]) or dt > pd.Timestamp(date_filter[1]):
                continue

        # Find closest ranking date
        ranking_date = None
        for rd in sorted(rankings.keys()):
            if rd <= dt:
                ranking_date = rd
        if ranking_date is None:
            continue

        scores = rankings[ranking_date]
        # Remove excluded tickers
        if exclude_set:
            scores = {k: v for k, v in scores.items() if k not in exclude_set}
        if not scores or len(scores) < 6:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        k = TOP_K  # V10: 4

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
                    haircut_override=haircut_override,
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
                    exit_idx = di + result["hold_days"]
                    if exit_idx >= len(close):
                        exit_idx = ei
                    trades.append({
                        **result, "entry_date": str(dt.date()),
                        "exit_date": str(close.index[exit_idx].date()),
                        "ticker": tk, "regime": "bull" if se >= sv else "bear",
                        "direction": direction, "vix": round(cv, 1),
                        "win": result["pnl"] > 0,
                        "n_positions": n_positions,
                    })

    return trades, equity, real_count, bs_count, turnover_events, early_exits


# ==============================================================
# STATS + MONTE CARLO
# ==============================================================

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
    early_ex = sum(1 for t in trades if t.get("exited_early", False))
    early_exit_rate = early_ex / len(trades) if trades else 0.0
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
        "early_exits": early_ex,
    }


def monte_carlo_bootstrap(trades, n_resamples=N_BOOTSTRAP, initial_capital=CAP):
    """Monte Carlo bootstrap resampling of trade-level P&L."""
    if not trades or len(trades) < 10:
        return None

    pnls = np.array([t["pnl"] for t in trades])
    n_trades = len(pnls)

    boot_sharpe = []
    boot_sortino = []
    boot_pf = []
    boot_wr = []
    boot_total_pnl = []

    rng = np.random.RandomState(42)

    for _ in range(n_resamples):
        idx = rng.choice(n_trades, size=n_trades, replace=True)
        sample_pnls = pnls[idx]

        wr = float(np.mean(sample_pnls > 0))
        boot_wr.append(wr)

        gw = float(sample_pnls[sample_pnls > 0].sum()) if (sample_pnls > 0).any() else 0
        gl = float(abs(sample_pnls[sample_pnls < 0].sum())) if (sample_pnls < 0).any() else 1e-10
        boot_pf.append(gw / max(gl, 1e-10))

        boot_total_pnl.append(float(sample_pnls.sum()))

        # Approximate monthly Sharpe
        n_months = max(3, n_trades // 6)
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


def find_top_pnl_tickers(trades, top_n=3):
    """Find the top N tickers by total PnL contribution."""
    ticker_pnl = {}
    for t in trades:
        tk = t["ticker"]
        ticker_pnl[tk] = ticker_pnl.get(tk, 0) + t["pnl"]
    sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
    top_tickers = [tk for tk, _ in sorted_tickers[:top_n]]
    return top_tickers, sorted_tickers


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"V10 STRESS TEST V1 -- {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"V10 Config: Sharpe 6.32 candidate")
    fprint(f"  Positions: {TOP_K} bull + {TOP_K} bear = {TOP_K*2} total")
    fprint(f"  OTM: {OTM_PCT*100:.0f}% | DTE: {DTE} | Profit target: {PROFIT_TARGET*100:.0f}%")
    fprint(f"  Capital: ${CAP:.0f} | Commission: ${COMMISSION:.2f} | Haircut: {HAIRCUT*100:.0f}%")
    fprint(f"  Width: adaptive max($3, 3%) | Rebalance: MONTHLY")
    fprint(f"  LGBM 17 features | 52-week sliding walk-forward")
    fprint(f"  Monte Carlo: {N_BOOTSTRAP} bootstrap resamples per variant")
    fprint()
    fprint("STRESS TEST VARIANTS:")
    fprint("  A: V10 Baseline -- the V10 D config as-is")
    fprint("  B: 3x Commission -- $7.80 instead of $2.60 (cost sensitivity)")
    fprint("  C: 25% Haircut -- 25% instead of 15% (pricing assumption)")
    fprint("  D: Remove Top 3 Tickers -- breadth of edge test")
    fprint("  E: COVID Window -- 2020-01-01 to 2021-12-31 (extreme regime)")
    fprint("  F: Bear 2022 -- 2022-01-01 to 2022-12-31 (bear market)")
    fprint()

    fprint("Loading chains...")
    chains = load_all_chains()

    fprint("\nDownloading price data...")
    close, high, low = download_data()
    regime_series = load_regime_predictions()
    atr_dict = compute_atr_series(high, low, close)

    # Monthly rebalance dates
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

    # ==========================================================
    # VARIANT A: V10 Baseline
    # ==========================================================
    fprint(f"\n{'=' * 90}")
    fprint("VARIANT A: V10 BASELINE -- 8 pos, 4% OTM, 30% PT, DTE=28, monthly")
    fprint(f"{'=' * 90}")

    trades_a, eq_a, real_a, bs_a, turns_a, early_a = simulate_variant(
        rankings, close, atr_dict, chains, monthly_dates, "A_v10_baseline",
        target_pct=PROFIT_TARGET,
    )
    fprint(f"  Trades: {len(trades_a)} | Real: {real_a} | BS: {bs_a} | "
           f"Early exits: {early_a} | Final: ${eq_a:,.0f}")

    stats_a = compute_full_stats(trades_a)
    if stats_a:
        fprint(f"  Sharpe: {stats_a['sharpe']:.2f} | Sortino: {stats_a['sortino']:.2f} | "
               f"Calmar: {stats_a['calmar']:.2f}")
        fprint(f"  WR: {stats_a['win_rate']*100:.1f}% | PF: {stats_a['profit_factor']:.2f} | "
               f"MDD: {stats_a['max_dd_pct']:.1f}%")
        fprint(f"  Avg hold: {stats_a['avg_hold_days']:.1f} days | "
               f"Early exit rate: {stats_a['early_exit_rate']*100:.1f}%")

    result_a = None
    if len(trades_a) >= 5:
        try:
            result_a = validate_trades(trades_a, initial_capital=CAP, spy_prices=spy_close,
                                       strategy_name="A_v10_baseline")
            result_a.print_summary()
        except Exception as e:
            fprint(f"  Validation error: {e}")

    # Monte Carlo for A
    fprint(f"  Running Monte Carlo bootstrap ({N_BOOTSTRAP} resamples)...")
    mc_a = monte_carlo_bootstrap(trades_a)
    if mc_a:
        fprint(f"    Sharpe: {mc_a['sharpe_mean']:.2f} +/- {mc_a['sharpe_std']:.2f} "
               f"[{mc_a['sharpe_ci_5']:.2f}, {mc_a['sharpe_ci_95']:.2f}] "
               f"({mc_a['sharpe_pct_positive']:.0f}% positive)")

    # Find top 3 PnL tickers for Variant D
    top3_tickers, ticker_pnl_sorted = find_top_pnl_tickers(trades_a, top_n=3)
    fprint(f"\n  Top PnL tickers (for Variant D removal):")
    for tk, pnl in ticker_pnl_sorted[:5]:
        fprint(f"    {tk}: ${pnl:,.0f}")
    fprint(f"  -> Removing: {top3_tickers}")

    # ==========================================================
    # VARIANT B: 3x Commission ($7.80)
    # ==========================================================
    fprint(f"\n{'=' * 90}")
    fprint("VARIANT B: 3x COMMISSION -- $7.80 per trade (cost sensitivity)")
    fprint(f"{'=' * 90}")

    COMMISSION_3X = 7.80
    trades_b, eq_b, real_b, bs_b, turns_b, early_b = simulate_variant(
        rankings, close, atr_dict, chains, monthly_dates, "B_3x_commission",
        target_pct=PROFIT_TARGET,
        commission_override=COMMISSION_3X,
    )
    fprint(f"  Trades: {len(trades_b)} | Final: ${eq_b:,.0f} | Early exits: {early_b}")

    stats_b = compute_full_stats(trades_b)
    result_b = None
    if stats_b:
        fprint(f"  Sharpe: {stats_b['sharpe']:.2f} | WR: {stats_b['win_rate']*100:.1f}% | "
               f"PF: {stats_b['profit_factor']:.2f}")
    if len(trades_b) >= 5:
        try:
            result_b = validate_trades(trades_b, initial_capital=CAP, spy_prices=spy_close,
                                       strategy_name="B_3x_commission")
            result_b.print_summary()
        except Exception as e:
            fprint(f"  Validation error: {e}")

    mc_b = monte_carlo_bootstrap(trades_b)
    if mc_b:
        fprint(f"  MC Sharpe: {mc_b['sharpe_mean']:.2f} [{mc_b['sharpe_ci_5']:.2f}, {mc_b['sharpe_ci_95']:.2f}]")

    # ==========================================================
    # VARIANT C: 25% Haircut
    # ==========================================================
    fprint(f"\n{'=' * 90}")
    fprint("VARIANT C: 25% HAIRCUT -- tests pricing assumption (was 15%)")
    fprint(f"{'=' * 90}")

    trades_c, eq_c, real_c, bs_c, turns_c, early_c = simulate_variant(
        rankings, close, atr_dict, chains, monthly_dates, "C_25pct_haircut",
        target_pct=PROFIT_TARGET,
        haircut_override=0.25,
    )
    fprint(f"  Trades: {len(trades_c)} | Final: ${eq_c:,.0f} | Early exits: {early_c}")

    stats_c = compute_full_stats(trades_c)
    result_c = None
    if stats_c:
        fprint(f"  Sharpe: {stats_c['sharpe']:.2f} | WR: {stats_c['win_rate']*100:.1f}% | "
               f"PF: {stats_c['profit_factor']:.2f}")
    if len(trades_c) >= 5:
        try:
            result_c = validate_trades(trades_c, initial_capital=CAP, spy_prices=spy_close,
                                       strategy_name="C_25pct_haircut")
            result_c.print_summary()
        except Exception as e:
            fprint(f"  Validation error: {e}")

    mc_c = monte_carlo_bootstrap(trades_c)
    if mc_c:
        fprint(f"  MC Sharpe: {mc_c['sharpe_mean']:.2f} [{mc_c['sharpe_ci_5']:.2f}, {mc_c['sharpe_ci_95']:.2f}]")

    # ==========================================================
    # VARIANT D: Remove Top 3 Tickers
    # ==========================================================
    fprint(f"\n{'=' * 90}")
    fprint(f"VARIANT D: REMOVE TOP 3 TICKERS -- {top3_tickers}")
    fprint(f"{'=' * 90}")

    trades_d, eq_d, real_d, bs_d, turns_d, early_d = simulate_variant(
        rankings, close, atr_dict, chains, monthly_dates, "D_remove_top3",
        target_pct=PROFIT_TARGET,
        exclude_tickers=top3_tickers,
    )
    fprint(f"  Trades: {len(trades_d)} | Final: ${eq_d:,.0f} | Early exits: {early_d}")

    stats_d = compute_full_stats(trades_d)
    result_d = None
    if stats_d:
        fprint(f"  Sharpe: {stats_d['sharpe']:.2f} | WR: {stats_d['win_rate']*100:.1f}% | "
               f"PF: {stats_d['profit_factor']:.2f}")
    if len(trades_d) >= 5:
        try:
            result_d = validate_trades(trades_d, initial_capital=CAP, spy_prices=spy_close,
                                       strategy_name="D_remove_top3")
            result_d.print_summary()
        except Exception as e:
            fprint(f"  Validation error: {e}")

    mc_d = monte_carlo_bootstrap(trades_d)
    if mc_d:
        fprint(f"  MC Sharpe: {mc_d['sharpe_mean']:.2f} [{mc_d['sharpe_ci_5']:.2f}, {mc_d['sharpe_ci_95']:.2f}]")

    # ==========================================================
    # VARIANT E: COVID Window (2020-01-01 to 2021-12-31)
    # ==========================================================
    fprint(f"\n{'=' * 90}")
    fprint("VARIANT E: COVID WINDOW -- 2020-01-01 to 2021-12-31")
    fprint(f"{'=' * 90}")

    trades_e, eq_e, real_e, bs_e, turns_e, early_e = simulate_variant(
        rankings, close, atr_dict, chains, monthly_dates, "E_covid_window",
        target_pct=PROFIT_TARGET,
        date_filter=("2020-01-01", "2021-12-31"),
    )
    fprint(f"  Trades: {len(trades_e)} | Final: ${eq_e:,.0f} | Early exits: {early_e}")

    stats_e = compute_full_stats(trades_e)
    if stats_e:
        fprint(f"  Sharpe: {stats_e['sharpe']:.2f} | WR: {stats_e['win_rate']*100:.1f}% | "
               f"PF: {stats_e['profit_factor']:.2f} | MDD: {stats_e['max_dd_pct']:.1f}%")

    for side in ["bull", "bear"]:
        st = [t for t in trades_e if t["direction"] == side]
        if st:
            pnls = [t["pnl"] for t in st]
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            fprint(f"  {side}: {len(st)} trades, WR {wr:.1%}, PnL ${sum(pnls):,.0f}")

    mc_e = monte_carlo_bootstrap(trades_e)

    # ==========================================================
    # VARIANT F: Bear 2022 (2022-01-01 to 2022-12-31)
    # ==========================================================
    fprint(f"\n{'=' * 90}")
    fprint("VARIANT F: BEAR 2022 -- 2022-01-01 to 2022-12-31")
    fprint(f"{'=' * 90}")

    trades_f, eq_f, real_f, bs_f, turns_f, early_f = simulate_variant(
        rankings, close, atr_dict, chains, monthly_dates, "F_bear_2022",
        target_pct=PROFIT_TARGET,
        date_filter=("2022-01-01", "2022-12-31"),
    )
    fprint(f"  Trades: {len(trades_f)} | Final: ${eq_f:,.0f} | Early exits: {early_f}")

    stats_f = compute_full_stats(trades_f)
    if stats_f:
        fprint(f"  Sharpe: {stats_f['sharpe']:.2f} | WR: {stats_f['win_rate']*100:.1f}% | "
               f"PF: {stats_f['profit_factor']:.2f} | MDD: {stats_f['max_dd_pct']:.1f}%")

    for side in ["bull", "bear"]:
        st = [t for t in trades_f if t["direction"] == side]
        if st:
            pnls = [t["pnl"] for t in st]
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            fprint(f"  {side}: {len(st)} trades, WR {wr:.1%}, PnL ${sum(pnls):,.0f}")

    mc_f = monte_carlo_bootstrap(trades_f)

    # ==========================================================
    # COMPARISON TABLE
    # ==========================================================
    fprint(f"\n{'=' * 160}")
    fprint("COMPARISON -- ALL V10 STRESS TEST VARIANTS")
    fprint(f"{'=' * 160}")

    all_variants = [
        ("A_v10_baseline", stats_a, result_a, mc_a, trades_a, real_a, bs_a),
        ("B_3x_commission", stats_b, result_b, mc_b, trades_b, real_b, bs_b),
        ("C_25pct_haircut", stats_c, result_c, mc_c, trades_c, real_c, bs_c),
        ("D_remove_top3", stats_d, result_d, mc_d, trades_d, real_d, bs_d),
        ("E_covid_window", stats_e, None, mc_e, trades_e, real_e, bs_e),
        ("F_bear_2022", stats_f, None, mc_f, trades_f, real_f, bs_f),
    ]

    fprint(f"\n  {'Variant':<25} {'N':>5} {'Sharpe':>7} {'Sort':>7} {'Calmar':>7} {'WR':>6} "
           f"{'PF':>6} {'MDD':>7} {'AvgHold':>8} {'EarlyX%':>8} {'Gate':>5} {'Final$':>9}")
    fprint(f"  {'-' * 120}")

    for label, st, rv, mc, trades, real, bs in all_variants:
        if st is None:
            continue
        gates_str = "--"
        if rv is not None:
            try:
                rd = rv.to_dict()
                gates_str = f"{rd['gates_passed']}/{rd['gates_total']}"
            except Exception:
                pass
        fprint(f"  {label:<25} {st['n_trades']:>5} {st['sharpe']:>7.2f} "
               f"{st['sortino']:>7.2f} {st['calmar']:>7.2f} "
               f"{st['win_rate']*100:>5.1f}% {st['profit_factor']:>5.2f} "
               f"{st['max_dd_pct']:>6.1f}% {st['avg_hold_days']:>7.1f}d "
               f"{st['early_exit_rate']*100:>7.1f}% {gates_str:>5} "
               f"${st['final_equity']:>8,.0f}")

    # ==========================================================
    # DELTA ANALYSIS (vs baseline A)
    # ==========================================================
    fprint(f"\n{'=' * 100}")
    fprint("DELTA vs V10 BASELINE (A)")
    fprint(f"{'=' * 100}")

    if stats_a:
        fprint(f"\n  {'Variant':<25} {'dSharpe':>8} {'dSort':>8} {'dWR':>7} {'dPF':>7} {'dMDD':>7} {'dPnL':>9}")
        fprint(f"  {'-' * 80}")
        for label, st, rv, mc, trades, real, bs in all_variants[1:]:
            if st is None:
                continue
            fprint(f"  {label:<25} {st['sharpe']-stats_a['sharpe']:>+7.2f} "
                   f"{st['sortino']-stats_a['sortino']:>+7.2f} "
                   f"{(st['win_rate']-stats_a['win_rate'])*100:>+6.1f}% "
                   f"{st['profit_factor']-stats_a['profit_factor']:>+6.2f} "
                   f"{st['max_dd_pct']-stats_a['max_dd_pct']:>+6.1f}% "
                   f"${st['total_pnl']-stats_a['total_pnl']:>+8,.0f}")

    # ==========================================================
    # MONTE CARLO COMPARISON
    # ==========================================================
    fprint(f"\n{'=' * 130}")
    fprint(f"MONTE CARLO BOOTSTRAP COMPARISON ({N_BOOTSTRAP} resamples)")
    fprint(f"{'=' * 130}")

    fprint(f"\n  {'Variant':<25} {'Sharpe':>12} {'90%CI':>16} {'%Pos':>6} "
           f"{'PF':>8} {'90%CI':>16} {'TotalPnL':>10}")
    fprint(f"  {'-' * 100}")

    for label, st, rv, mc, trades, real, bs in all_variants:
        if mc is None:
            continue
        fprint(f"  {label:<25} {mc['sharpe_mean']:>7.2f}+-{mc['sharpe_std']:.2f} "
               f"[{mc['sharpe_ci_5']:>6.2f},{mc['sharpe_ci_95']:>6.2f}] "
               f"{mc['sharpe_pct_positive']:>5.0f}% "
               f"{mc['pf_mean']:>7.2f} [{mc['pf_ci_5']:>6.2f},{mc['pf_ci_95']:>6.2f}] "
               f"${mc['total_pnl_mean']:>9,.0f}")

    # ==========================================================
    # SIDE ANALYSIS (A only)
    # ==========================================================
    fprint(f"\n{'=' * 80}")
    fprint("SIDE ANALYSIS -- VARIANT A (V10 Baseline)")
    fprint(f"{'=' * 80}")
    for side in ["bull", "bear"]:
        st = [t for t in trades_a if t["direction"] == side]
        if st:
            pnls = [t["pnl"] for t in st]
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            avg = np.mean(pnls)
            fprint(f"  {side}: {len(st)} trades, WR {wr:.1%}, avg ${avg:.2f}, total ${sum(pnls):,.0f}")

    # Regime analysis
    fprint(f"\n{'=' * 80}")
    fprint("REGIME ANALYSIS -- VARIANT A")
    fprint(f"{'=' * 80}")
    for regime in ["bull", "bear"]:
        rt = [t for t in trades_a if t["regime"] == regime]
        if rt:
            pnls = [t["pnl"] for t in rt]
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            fprint(f"  {regime} regime: {len(rt)} trades, WR {wr:.1%}, PnL ${sum(pnls):,.0f}")

    # Real-only analysis
    fprint(f"\n{'=' * 80}")
    fprint("REAL-PRICED TRADES ONLY -- VARIANT A")
    fprint(f"{'=' * 80}")
    real_trades = [t for t in trades_a if t.get("used_real_pricing")]
    if len(real_trades) >= 10:
        rp = [t["pnl"] for t in real_trades]
        wr = sum(1 for p in rp if p > 0) / len(rp)
        sh = float(np.mean(rp) / (np.std(rp) + 1e-10) * np.sqrt(52))
        fprint(f"  {len(real_trades)} real-priced trades")
        fprint(f"  Sharpe: {sh:.2f}, WR: {wr:.1%}, total PnL: ${sum(rp):,.0f}")

    # Ticker contribution
    fprint(f"\n{'=' * 80}")
    fprint("TICKER PnL CONTRIBUTION -- VARIANT A")
    fprint(f"{'=' * 80}")
    for tk, pnl in ticker_pnl_sorted:
        n_tk = sum(1 for t in trades_a if t["ticker"] == tk)
        fprint(f"  {tk}: ${pnl:>8,.0f} ({n_tk} trades)")

    # Early exit analysis
    fprint(f"\n{'=' * 80}")
    fprint("EARLY EXIT DETAIL -- VARIANT A")
    fprint(f"{'=' * 80}")
    early = [t for t in trades_a if t.get("exited_early", False)]
    held = [t for t in trades_a if not t.get("exited_early", False)]
    if early:
        early_pnls = [t["pnl"] for t in early]
        fprint(f"  Early exits: {len(early)} ({len(early)/len(trades_a)*100:.1f}%)")
        fprint(f"    Avg PnL: ${np.mean(early_pnls):+.2f} | WR: {sum(1 for p in early_pnls if p>0)/len(early_pnls)*100:.1f}%")
        fprint(f"    Avg hold: {np.mean([t['hold_days'] for t in early]):.1f} days")
    if held:
        held_pnls = [t["pnl"] for t in held]
        fprint(f"  Held to expiry: {len(held)} ({len(held)/len(trades_a)*100:.1f}%)")
        fprint(f"    Avg PnL: ${np.mean(held_pnls):+.2f} | WR: {sum(1 for p in held_pnls if p>0)/len(held_pnls)*100:.1f}%")

    # ==========================================================
    # 5-GATE VALIDATION SUMMARY
    # ==========================================================
    fprint(f"\n{'=' * 100}")
    fprint("5-GATE VALIDATION SUMMARY")
    fprint(f"{'=' * 100}")

    for label, st, rv, mc, trades, real, bs in all_variants:
        if rv is None:
            fprint(f"\n  {label}: No validation (insufficient trades or date-filtered)")
            continue
        try:
            r = rv.to_dict()
            status = "PASS" if r["all_passed"] else "FAIL"
            fprint(f"\n  [{status}] {label}: {r['gates_passed']}/{r['gates_total']} gates")
            if r.get("gates"):
                for g in r["gates"]:
                    gs = "PASS" if g["passed"] else "FAIL"
                    fprint(f"         [{gs}] {g['name']}: {g['metric_name']}={g['metric_value']:.4f}")
        except Exception as e:
            fprint(f"\n  {label}: Validation display error: {e}")

    # ==========================================================
    # VERDICT
    # ==========================================================
    fprint(f"\n{'=' * 100}")
    fprint("V10 STRESS TEST VERDICT")
    fprint(f"{'=' * 100}")

    verdict_pass = True
    verdict_notes = []

    # Check A passes all gates
    if result_a:
        try:
            ra = result_a.to_dict()
            fprint(f"\n  A (V10 Baseline): Sharpe {stats_a['sharpe']:.2f}, "
                   f"Sortino {stats_a['sortino']:.2f}, "
                   f"PF {stats_a['profit_factor']:.2f}, "
                   f"WR {stats_a['win_rate']*100:.1f}%, MDD {stats_a['max_dd_pct']:.1f}%")
            if not ra["all_passed"]:
                verdict_pass = False
                verdict_notes.append("A: Not all 5 gates passed")
            else:
                verdict_notes.append(f"A: All {ra['gates_total']} gates PASSED")
        except Exception:
            pass

    # Check B survives 3x commission
    if stats_b:
        fprint(f"  B (3x cost):     Sharpe {stats_b['sharpe']:.2f}, PF {stats_b['profit_factor']:.2f}")
        if stats_b["sharpe"] < 0.5:
            verdict_pass = False
            verdict_notes.append(f"B: Sharpe {stats_b['sharpe']:.2f} < 0.5 under 3x commission -- FRAGILE")
        else:
            verdict_notes.append(f"B: Sharpe {stats_b['sharpe']:.2f} survives 3x commission")

    # Check C (25% haircut)
    if stats_c and stats_a:
        fprint(f"  C (25% haircut): Sharpe {stats_c['sharpe']:.2f}")
        sharpe_drop = stats_a["sharpe"] - stats_c["sharpe"]
        if stats_c["sharpe"] < 0.5:
            verdict_pass = False
            verdict_notes.append(f"C: Sharpe {stats_c['sharpe']:.2f} < 0.5 under 25% haircut -- pricing sensitive")
        else:
            verdict_notes.append(f"C: Sharpe {stats_c['sharpe']:.2f} survives 25% haircut (drop {sharpe_drop:+.2f})")

    # Check D edge is broad-based
    if stats_d and stats_a:
        sharpe_drop = stats_a["sharpe"] - stats_d["sharpe"]
        fprint(f"  D (no top 3):    Sharpe {stats_d['sharpe']:.2f} (drop {sharpe_drop:+.2f})")
        if stats_d["sharpe"] < 0.3:
            verdict_pass = False
            verdict_notes.append(f"D: Sharpe {stats_d['sharpe']:.2f} -- edge concentrated, NOT broad-based")
        elif sharpe_drop > 0.8:
            verdict_notes.append(f"D: Large drop ({sharpe_drop:.2f}) -- edge partially concentrated")
        else:
            verdict_notes.append(f"D: Sharpe {stats_d['sharpe']:.2f} -- edge is broad-based")

    # Check E (COVID)
    if stats_e:
        fprint(f"  E (COVID):       Sharpe {stats_e['sharpe']:.2f}, WR {stats_e['win_rate']*100:.1f}%, "
               f"MDD {stats_e['max_dd_pct']:.1f}%")
        if stats_e["total_pnl"] < -100:
            verdict_notes.append(f"E: Lost ${abs(stats_e['total_pnl']):,.0f} during COVID -- watch drawdown")
        else:
            verdict_notes.append(f"E: PnL ${stats_e['total_pnl']:,.0f} during COVID -- acceptable")

    # Check F (2022 bear)
    if stats_f:
        fprint(f"  F (2022 bear):   Sharpe {stats_f['sharpe']:.2f}, WR {stats_f['win_rate']*100:.1f}%, "
               f"MDD {stats_f['max_dd_pct']:.1f}%")
        if stats_f["total_pnl"] < -100:
            verdict_notes.append(f"F: Lost ${abs(stats_f['total_pnl']):,.0f} in 2022 bear -- investigate")
        else:
            verdict_notes.append(f"F: PnL ${stats_f['total_pnl']:,.0f} in 2022 bear -- acceptable")

    # Check Monte Carlo for A
    if mc_a:
        p5 = mc_a["sharpe_ci_5"]
        fprint(f"  MC (A p5):       Worst realistic Sharpe {p5:.3f}")
        if p5 < 0:
            verdict_pass = False
            verdict_notes.append(f"MC: 5th pct Sharpe {p5:.3f} < 0 -- strategy can go negative")
        elif p5 < 0.5:
            verdict_notes.append(f"MC: 5th pct Sharpe {p5:.3f} -- marginal under worst-case")
        else:
            verdict_notes.append(f"MC: 5th pct Sharpe {p5:.3f} -- robust even worst-case")

    fprint(f"\n  NOTES:")
    for note in verdict_notes:
        fprint(f"    {note}")

    if verdict_pass:
        fprint(f"\n  >>> VERDICT: PASS -- V10 config cleared for deployment <<<")
    else:
        fprint(f"\n  >>> VERDICT: FAIL -- V10 config needs investigation <<<")

    # ==========================================================
    # SAVE RESULTS
    # ==========================================================
    save_results = {
        "timestamp": t0.isoformat(),
        "config": {
            "version": "V10",
            "capital": CAP, "dte": DTE, "otm_pct": OTM_PCT,
            "profit_target": PROFIT_TARGET,
            "commission": COMMISSION,
            "early_exit_commission": EARLY_EXIT_COMMISSION,
            "haircut": HAIRCUT,
            "rebalance": "monthly",
            "top_k": TOP_K, "n_positions": TOP_K * 2,
            "wf_weeks": WF_TRAIN_PERIODS,
            "n_sectors": len(SECTORS), "n_features": len(V6_FEATURES),
            "n_bootstrap": N_BOOTSTRAP,
            "width": "adaptive max($3, 3%)",
        },
        "verdict": "PASS" if verdict_pass else "FAIL",
        "verdict_notes": verdict_notes,
    }

    # Save stats per variant
    for label, st, rv, mc, trades, real, bs in all_variants:
        if st:
            save_results[label] = st
        if rv:
            try:
                save_results[f"{label}_validation"] = rv.to_dict()
            except Exception:
                pass
        if mc:
            save_results[f"{label}_monte_carlo"] = mc

    # Ticker contribution
    save_results["ticker_pnl"] = dict(ticker_pnl_sorted)
    if top3_tickers:
        save_results["removed_top3"] = top3_tickers

    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_file}")

    # Save trade details per variant
    for label, st, rv, mc, trades, real, bs in all_variants:
        if trades:
            tf = OUTPUT_DIR / f"trades_{label}.json"
            with open(tf, "w") as f:
                json.dump(trades, f, indent=1, default=str)

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"v10_stress_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_param("version", "V10")
                mlflow.log_param("capital", CAP)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("otm_pct", OTM_PCT)
                mlflow.log_param("profit_target", PROFIT_TARGET)
                mlflow.log_param("commission", COMMISSION)
                mlflow.log_param("haircut", HAIRCUT)
                mlflow.log_param("rebalance", "monthly")
                mlflow.log_param("top_k", TOP_K)
                mlflow.log_param("n_positions", TOP_K * 2)
                mlflow.log_param("wf_train_weeks", WF_TRAIN_PERIODS)
                mlflow.log_param("n_sectors", len(SECTORS))
                mlflow.log_param("n_features", len(V6_FEATURES))
                mlflow.log_param("n_bootstrap", N_BOOTSTRAP)
                mlflow.log_param("verdict", "PASS" if verdict_pass else "FAIL")
                if top3_tickers:
                    mlflow.log_param("removed_top3", ",".join(top3_tickers))

                # Log metrics for each variant
                prefix_map = {
                    "A_v10_baseline": "A",
                    "B_3x_commission": "B",
                    "C_25pct_haircut": "C",
                    "D_remove_top3": "D",
                    "E_covid_window": "E",
                    "F_bear_2022": "F",
                }
                for label, st, rv, mc, trades, real, bs in all_variants:
                    if st is None:
                        continue
                    prefix = prefix_map.get(label, label[0])
                    mlflow.log_metric(f"{prefix}_sharpe", st["sharpe"])
                    mlflow.log_metric(f"{prefix}_sortino", st["sortino"])
                    mlflow.log_metric(f"{prefix}_calmar", st["calmar"])
                    mlflow.log_metric(f"{prefix}_wr", st["win_rate"])
                    mlflow.log_metric(f"{prefix}_pf", st["profit_factor"])
                    mlflow.log_metric(f"{prefix}_mdd", st["max_dd_pct"])
                    mlflow.log_metric(f"{prefix}_n_trades", st["n_trades"])
                    mlflow.log_metric(f"{prefix}_total_pnl", st["total_pnl"])
                    mlflow.log_metric(f"{prefix}_early_exit_rate", st["early_exit_rate"])
                    mlflow.log_metric(f"{prefix}_avg_hold", st["avg_hold_days"])

                    if rv:
                        try:
                            rd = rv.to_dict()
                            mlflow.log_metric(f"{prefix}_gates_passed", rd["gates_passed"])
                        except Exception:
                            pass
                    if mc:
                        mlflow.log_metric(f"{prefix}_mc_sharpe_mean", mc["sharpe_mean"])
                        mlflow.log_metric(f"{prefix}_mc_sharpe_ci5", mc["sharpe_ci_5"])
                        mlflow.log_metric(f"{prefix}_mc_sharpe_ci95", mc["sharpe_ci_95"])

                mlflow.log_artifact(str(results_file))
            fprint(f"MLflow logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow error: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nCompleted in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
