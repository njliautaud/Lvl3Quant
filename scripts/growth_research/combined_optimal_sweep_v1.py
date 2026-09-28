#!/usr/bin/env python3
"""
Combined Optimal Sweep V1 -- Test ALL proven improvements together
==================================================================

Individual findings we are combining:
  - DTE=28 beats DTE=14 (KB #245)
  - Monthly rebalance beats weekly (KB #252)
  - 50% profit target doubles Sharpe (KB #261-262)
  - Equal sizing is optimal (KB #258)
  - Correlation gate / dispersion / VIX term structure = useless
  - LGBM 17 features beats GRU (KB #248)

5 Combined Variants:
  A: V9.2 baseline (hold-to-expiry) -- Monthly + DTE=28 + hold-to-expiry
  B: V9.3 (50% PT) -- Monthly + DTE=28 + 50% profit target
  C: V9.3 aggressive (30% PT) -- Monthly + DTE=28 + 30% profit target (highest WR)
  D: V9.3 conservative (60% PT) -- Monthly + DTE=28 + 60% profit target (lowest MDD)
  E: V9.3 + biweekly -- Biweekly rebalance + DTE=28 + 50% profit target

Config: 3% OTM, adaptive max($3,3%), LGBM 17 features, $645 capital,
        $2.60 commission (+$2.60 for early exits), 15% haircut,
        real Dolt chain data, 52-week walk-forward

5-gate validation + Monte Carlo bootstrap 1000 on all variants.

Output: output/growth_research/combined_optimal_sweep_v1/
MLflow experiment: combined_optimal_sweep_v1
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

# Detect environment
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")

if _NEPTUNE_BASE.exists():
    BASE = _NEPTUNE_BASE
    fprint("Running on Neptune: %s" % BASE)
else:
    BASE = _JUPITER_BASE
    fprint("Running on Jupiter: %s" % BASE)

sys.path.insert(0, str(BASE))
from research.tools.options_pricer import (
    price_bull_call_spread, price_bear_put_spread, COMMISSION_RT_SPREAD,
)
from research.tools.adversarial_validator import validate_trades

CHAINS_DIR = BASE / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
OUTPUT_DIR = BASE / "output" / "growth_research" / "combined_optimal_sweep_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
TOP_K = 3
DTE = 28
REGIME_BULL_THRESHOLD = 0.4
WF_TRAIN_PERIODS = 52
HAIRCUT = 0.15

COST_WIDTH_MAX = 0.50
DTE_TOLERANCE = 5
STRIKE_TOLERANCE = 0.03

EARLY_EXIT_COMMISSION = 2.60

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "combined_optimal_sweep_v1"

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

V6_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "cross_sector_dispersion",
]
assert len(V6_FEATURES) == 17

N_BOOTSTRAP = 1000


# ==============================================================
# CHAIN DATA
# ==============================================================

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
        fprint("  %s: %d rows" % (tk, len(df)))
    return chains


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
# LGBM WALK-FORWARD RANKING
# ==============================================================

def get_monthly_rebalance_dates(close):
    """Monthly rebalance -- first trading day of each month."""
    monthly = close.index.to_series().resample("MS").first().dropna()
    return pd.DatetimeIndex(monthly.values)


def get_biweekly_rebalance_dates(close):
    """Biweekly rebalance -- every other Friday."""
    fridays = close.index.to_series().resample("W-FRI").last().dropna()
    # Take every other Friday
    return pd.DatetimeIndex(fridays.values[::2])


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
    fprint("    %d records, %d dates" % (len(df), len(df["date"].unique())))
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
    fprint("    %d ranking dates" % len(rankings))
    return rankings


# ==============================================================
# STRIKES + EXECUTION
# ==============================================================

def compute_strikes(S, direction):
    if direction == "bull":
        K1 = round(S * 1.03, 2)
        pct_w = K1 * 0.03
        w = max(3.0, pct_w)
        K2 = round(K1 + w, 2)
    else:
        K2 = round(S * 0.97, 2)
        pct_w = K2 * 0.03
        w = max(3.0, pct_w)
        K1 = round(K2 - w, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


def execute_trade_with_profit_target(tk, dt, direction, close, atr_dict, vix_val, equity,
                                      chains, max_pos, target_pct=None):
    """Execute a single spread trade, optionally with profit target early exit.

    Args:
        target_pct: profit target as fraction of max profit. None = hold to expiry.
    """
    commission = COMMISSION_RT_SPREAD
    exit_comm = EARLY_EXIT_COMMISSION

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

    max_profit_ps = spread_width - entry_cost_ps

    # -----------------------------------------------------------------
    # PROFIT TARGET EXIT LOGIC (only if target_pct is set)
    # -----------------------------------------------------------------
    exited_early = False
    exit_day_idx = ei
    exit_reason = "expiry"
    hold_days = DTE

    if target_pct is not None and max_profit_ps > 0:
        for check_idx in range(di + 1, ei + 1):
            if check_idx >= len(close):
                break
            check_date = close.index[check_idx]
            dte_remaining = ei - check_idx
            days_held = check_idx - di

            S_now = float(close[tk].iloc[check_idx])
            av_now = float(atr_dict[tk].loc[check_date]) if check_date in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[check_date]) else S_now * 0.015
            vix_now = vix_val

            current_value_ps = None
            if chain_df is not None:
                current_value_ps = revalue_spread_chain(chain_df, check_date, direction, K1, K2, dte_remaining)
            if current_value_ps is None:
                current_value_ps = revalue_spread_bs(S_now, K1, K2, dte_remaining, av_now, vix_now, direction)

            unrealized_gain_ps = current_value_ps - entry_cost_ps

            if unrealized_gain_ps >= target_pct * max_profit_ps:
                exited_early = True
                exit_day_idx = check_idx
                exit_reason = "profit_target_%dpct" % int(target_pct * 100)
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

        pnl = (exit_value_ps - entry_cost_ps) * 100 - commission - exit_comm
    else:
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
        "target_pct": target_pct if target_pct is not None else "hold_to_expiry",
    }


# ==============================================================
# SIMULATION ENGINE
# ==============================================================

def simulate_variant(rankings, close, atr_dict, chains, rebal_dates, variant_name="",
                     target_pct=None):
    """Run backtest simulation for a variant.

    Args:
        target_pct: profit target fraction, or None for hold-to-expiry.
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
                    chains, max_pos, target_pct=target_pct,
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
                    exit_date_str = str(close.index[exit_idx].date()) if exit_idx < len(close) else str(close.index[ei].date())
                    trades.append({
                        **result,
                        "entry_date": str(dt.date()),
                        "exit_date": exit_date_str,
                        "ticker": tk,
                        "regime": "bull" if se >= sv else "bear",
                        "direction": direction,
                        "vix": round(cv, 1),
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

    early_exits = sum(1 for t in trades if t.get("exited_early", False))
    early_exit_rate = early_exits / len(trades) if trades else 0.0
    hold_days_list = [t.get("hold_days", DTE) for t in trades]
    avg_hold = float(np.mean(hold_days_list))

    # CAGR
    dates_str = [t["entry_date"] for t in trades]
    first_date = pd.Timestamp(min(dates_str))
    last_date = pd.Timestamp(max(dates_str))
    years = (last_date - first_date).days / 365.25
    cagr = (equity[-1] / equity[0]) ** (1 / max(years, 0.5)) - 1 if years > 0.25 else 0.0

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
        "cagr": round(cagr, 4),
        "early_exit_rate": round(early_exit_rate, 4),
        "avg_hold_days": round(avg_hold, 1),
        "early_exits": early_exits,
        "years": round(years, 2) if years > 0 else 0,
    }


# ==============================================================
# MONTE CARLO BOOTSTRAP
# ==============================================================

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
    boot_max_dd = []

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

        # Max drawdown
        eq = np.cumsum(np.concatenate([[initial_capital], sample_pnls]))
        pk = np.maximum.accumulate(eq)
        dd = (eq - pk) / pk
        boot_max_dd.append(float(dd.min()) * 100)

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
    boot_max_dd = np.array(boot_max_dd)

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
        "max_dd_mean": round(float(boot_max_dd.mean()), 2),
        "max_dd_ci_5": round(float(np.percentile(boot_max_dd, 5)), 2),
        "max_dd_ci_95": round(float(np.percentile(boot_max_dd, 95)), 2),
    }


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint("COMBINED OPTIMAL SWEEP V1 -- %s" % t0.strftime("%Y-%m-%d %H:%M:%S"))
    fprint("=" * 100)
    fprint("Capital: $%d | DTE: %d | Commission: $%.2f" % (CAP, DTE, COMMISSION_RT_SPREAD))
    fprint("Early exit adds: $%.2f extra commission to close" % EARLY_EXIT_COMMISSION)
    fprint("Universe: %d sector ETFs | Top/Bottom K: %d" % (len(SECTORS), TOP_K))
    fprint("OTM: 3%% | Width: adaptive max($3, 3%%) | Haircut: %d%%" % (HAIRCUT * 100))
    fprint("LGBM: 17 features | Walk-forward: %d-week sliding" % WF_TRAIN_PERIODS)
    fprint("Monte Carlo: %d bootstrap resamples per variant" % N_BOOTSTRAP)
    fprint()
    fprint("COMBINING ALL PROVEN IMPROVEMENTS:")
    fprint("  - DTE=28 (KB #245)")
    fprint("  - Monthly rebalance (KB #252)")
    fprint("  - Profit targets 30/50/60%% (KB #261-262)")
    fprint("  - Equal sizing (KB #258)")
    fprint("  - LGBM 17 features (KB #248)")
    fprint("  - No correlation gate, dispersion, or VIX term structure (useless)")
    fprint()
    fprint("5 VARIANTS:")
    fprint("  A: V9.2 baseline -- Monthly + DTE=28 + hold-to-expiry")
    fprint("  B: V9.3 (50%% PT) -- Monthly + DTE=28 + 50%% profit target")
    fprint("  C: V9.3 aggressive (30%% PT) -- Monthly + DTE=28 + 30%% profit target")
    fprint("  D: V9.3 conservative (60%% PT) -- Monthly + DTE=28 + 60%% profit target")
    fprint("  E: V9.3 biweekly -- Biweekly + DTE=28 + 50%% profit target")
    fprint()

    fprint("Loading chains...")
    chains = load_all_chains()

    fprint("\nDownloading price data...")
    close, high, low = download_data()
    regime_series = load_regime_predictions()
    atr_dict = compute_atr_series(high, low, close)

    # Rebalance schedules
    monthly_dates = get_monthly_rebalance_dates(close)
    biweekly_dates = get_biweekly_rebalance_dates(close)
    weekly_dates = get_weekly_fridays(close)

    fprint("\nRebalance schedules:")
    fprint("  Monthly: %d dates" % len(monthly_dates))
    fprint("  Biweekly: %d dates" % len(biweekly_dates))
    fprint("  Weekly (for LGBM training): %d dates" % len(weekly_dates))

    # Build features and LGBM rankings (shared across variants)
    fprint("\n%s" % ("=" * 80))
    fprint("BUILDING LGBM RANKINGS (52-week walk-forward, 17 momentum features)")
    fprint("=" * 80)
    records = build_feature_records(close, high, low, weekly_dates, regime_series)
    rankings = walk_forward_lgbm_rank(records)

    if len(rankings) < 20:
        fprint("ERROR: Only %d ranking dates -- insufficient data" % len(rankings))
        return

    spy_close = close["SPY"]

    # Define variants: (name, target_pct, rebal_dates, description)
    # target_pct=None means hold-to-expiry
    variants = [
        ("A_v92_baseline", None, monthly_dates, "A: V9.2 baseline -- Monthly + DTE=28 + hold-to-expiry"),
        ("B_v93_pt50", 0.50, monthly_dates, "B: V9.3 (50% PT) -- Monthly + DTE=28 + 50% profit target"),
        ("C_v93_pt30", 0.30, monthly_dates, "C: V9.3 aggressive (30% PT) -- Monthly + DTE=28 + 30% PT"),
        ("D_v93_pt60", 0.60, monthly_dates, "D: V9.3 conservative (60% PT) -- Monthly + DTE=28 + 60% PT"),
        ("E_v93_biweekly", 0.50, biweekly_dates, "E: V9.3 biweekly -- Biweekly + DTE=28 + 50% PT"),
    ]

    all_trades = {}
    all_stats = {}
    all_results = {}
    all_mc = {}

    for var_name, target_pct, rebal_dates, description in variants:
        fprint("\n%s" % ("=" * 90))
        fprint(description)
        fprint("=" * 90)

        trades, eq, real, bs, turns, early_ex = simulate_variant(
            rankings, close, atr_dict, chains, rebal_dates, var_name,
            target_pct=target_pct,
        )
        fprint("  Trades: %d | Real pricing: %d | BS fallback: %d | Early exits: %d | Final: $%s" % (
            len(trades), real, bs, early_ex, "{:,.0f}".format(eq)))

        all_trades[var_name] = trades
        stats_dict = compute_full_stats(trades)
        all_stats[var_name] = stats_dict

        if stats_dict:
            fprint("  Sharpe: %.2f | Sortino: %.2f | Calmar: %.2f | CAGR: %.1f%%" % (
                stats_dict["sharpe"], stats_dict["sortino"], stats_dict["calmar"], stats_dict["cagr"] * 100))
            fprint("  WR: %.1f%% | PF: %.2f | MDD: %.1f%%" % (
                stats_dict["win_rate"] * 100, stats_dict["profit_factor"], stats_dict["max_dd_pct"]))
            fprint("  Avg hold: %.1f days | Early exit rate: %.1f%%" % (
                stats_dict["avg_hold_days"], stats_dict["early_exit_rate"] * 100))

        # 5-gate validation
        if len(trades) >= 5:
            try:
                result_val = validate_trades(
                    trades, initial_capital=CAP, spy_prices=spy_close,
                    strategy_name=var_name
                )
                result_val.print_summary()
                all_results[var_name] = result_val
            except Exception as e:
                fprint("  Validation error: %s" % str(e))

        # Monte Carlo bootstrap
        fprint("  Running Monte Carlo bootstrap (%d resamples)..." % N_BOOTSTRAP)
        mc = monte_carlo_bootstrap(trades, n_resamples=N_BOOTSTRAP)
        if mc:
            all_mc[var_name] = mc
            fprint("    Sharpe: %.2f +/- %.2f [%.2f, %.2f] (%d%% positive)" % (
                mc["sharpe_mean"], mc["sharpe_std"], mc["sharpe_ci_5"], mc["sharpe_ci_95"],
                mc["sharpe_pct_positive"]))
            fprint("    PF: %.2f [%.2f, %.2f]" % (mc["pf_mean"], mc["pf_ci_5"], mc["pf_ci_95"]))
            fprint("    WR: %.1f%% [%.1f%%, %.1f%%]" % (
                mc["wr_mean"] * 100, mc["wr_ci_5"] * 100, mc["wr_ci_95"] * 100))
            fprint("    Total PnL: $%s [$%s, $%s] (%d%% profitable)" % (
                "{:,.0f}".format(mc["total_pnl_mean"]),
                "{:,.0f}".format(mc["total_pnl_ci_5"]),
                "{:,.0f}".format(mc["total_pnl_ci_95"]),
                mc["pct_profitable"]))
            fprint("    Max DD: %.1f%% [%.1f%%, %.1f%%]" % (
                mc["max_dd_mean"], mc["max_dd_ci_5"], mc["max_dd_ci_95"]))

        # Side breakdown
        for side in ["bull", "bear"]:
            st = [t for t in trades if t["direction"] == side]
            if st:
                pnls = [t["pnl"] for t in st]
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                fprint("  %s: %d trades, WR %.1f%%, PnL $%s" % (
                    side, len(st), wr * 100, "{:,.0f}".format(sum(pnls))))

    # ==========================================================
    # COMPARISON TABLE
    # ==========================================================
    fprint("\n%s" % ("=" * 160))
    fprint("COMPARISON -- ALL COMBINED VARIANTS")
    fprint("=" * 160)

    fprint("\n  %-25s %5s %7s %7s %7s %6s %6s %7s %8s %8s %5s %9s %7s" % (
        "Variant", "N", "Sharpe", "Sort", "Calmar", "WR", "PF", "MDD", "AvgHold", "EarlyX%", "Gate", "Final$", "CAGR"))
    fprint("  %s" % ("-" * 155))

    for var_name, target_pct, rebal_dates, description in variants:
        stats_dict = all_stats.get(var_name)
        if stats_dict is None:
            continue
        result_val = all_results.get(var_name)
        gates_str = "%d/%d" % (result_val.to_dict()["gates_passed"], result_val.to_dict()["gates_total"]) if result_val else "--"

        fprint("  %-25s %5d %7.2f %7.2f %7.2f %5.1f%% %5.2f %6.1f%% %7.1fd %7.1f%% %5s $%8s %6.1f%%" % (
            var_name, stats_dict["n_trades"], stats_dict["sharpe"],
            stats_dict["sortino"], stats_dict["calmar"],
            stats_dict["win_rate"] * 100, stats_dict["profit_factor"],
            stats_dict["max_dd_pct"], stats_dict["avg_hold_days"],
            stats_dict["early_exit_rate"] * 100, gates_str,
            "{:,.0f}".format(stats_dict["final_equity"]),
            stats_dict["cagr"] * 100))

    # ==========================================================
    # DELTA ANALYSIS (vs baseline A)
    # ==========================================================
    fprint("\n%s" % ("=" * 100))
    fprint("DELTA vs BASELINE (A: V9.2 hold-to-expiry)")
    fprint("=" * 100)

    base = all_stats.get("A_v92_baseline")
    if base:
        fprint("\n  %-25s %8s %8s %7s %7s %7s %9s" % (
            "Variant", "dSharpe", "dSort", "dWR", "dPF", "dMDD", "dPnL"))
        fprint("  %s" % ("-" * 80))
        for var_name, _, _, _ in variants[1:]:
            s = all_stats.get(var_name)
            if s is None:
                continue
            fprint("  %-25s %+7.2f %+7.2f %+6.1f%% %+6.2f %+6.1f%% $%+8s" % (
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
    fprint("\n%s" % ("=" * 140))
    fprint("MONTE CARLO BOOTSTRAP COMPARISON (%d resamples)" % N_BOOTSTRAP)
    fprint("=" * 140)

    fprint("\n  %-25s %12s %16s %6s %8s %16s %10s %22s" % (
        "Variant", "Sharpe", "90%CI", "%Pos", "PF", "90%CI", "PnL", "90%CI"))
    fprint("  %s" % ("-" * 130))

    for var_name, _, _, _ in variants:
        mc = all_mc.get(var_name)
        if mc is None:
            continue
        fprint("  %-25s %7.2f+-%.2f [%6.2f,%6.2f] %5.0f%% %7.2f [%6.2f,%6.2f] $%9s [$%8s,$%8s]" % (
            var_name,
            mc["sharpe_mean"], mc["sharpe_std"],
            mc["sharpe_ci_5"], mc["sharpe_ci_95"],
            mc["sharpe_pct_positive"],
            mc["pf_mean"], mc["pf_ci_5"], mc["pf_ci_95"],
            "{:,.0f}".format(mc["total_pnl_mean"]),
            "{:,.0f}".format(mc["total_pnl_ci_5"]),
            "{:,.0f}".format(mc["total_pnl_ci_95"])))

    # ==========================================================
    # EARLY EXIT ANALYSIS
    # ==========================================================
    fprint("\n%s" % ("=" * 100))
    fprint("EARLY EXIT DETAIL")
    fprint("=" * 100)

    for var_name, target_pct, _, _ in variants[1:]:
        trades = all_trades.get(var_name, [])
        early = [t for t in trades if t.get("exited_early", False)]
        held = [t for t in trades if not t.get("exited_early", False)]

        if not early:
            fprint("\n  %s: No early exits triggered" % var_name)
            continue

        early_pnls = [t["pnl"] for t in early]
        held_pnls = [t["pnl"] for t in held]

        fprint("\n  %s:" % var_name)
        fprint("    Early exits: %d (%.1f%%)" % (len(early), len(early) / len(trades) * 100))
        fprint("      Avg PnL: $%+.2f | WR: %.1f%%" % (
            np.mean(early_pnls),
            sum(1 for p in early_pnls if p > 0) / len(early_pnls) * 100))
        fprint("      Avg hold: %.1f days" % np.mean([t["hold_days"] for t in early]))
        if held:
            fprint("    Held to expiry: %d (%.1f%%)" % (len(held), len(held) / len(trades) * 100))
            fprint("      Avg PnL: $%+.2f | WR: %.1f%%" % (
                np.mean(held_pnls),
                sum(1 for p in held_pnls if p > 0) / len(held_pnls) * 100))

        reasons = {}
        for t in early:
            r = t.get("exit_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        fprint("    Exit reasons: %s" % str(reasons))

    # ==========================================================
    # REGIME ANALYSIS
    # ==========================================================
    fprint("\n%s" % ("=" * 100))
    fprint("REGIME ANALYSIS -- ALL VARIANTS")
    fprint("=" * 100)

    for var_name, _, _, _ in variants:
        trades = all_trades.get(var_name, [])
        if not trades:
            continue
        fprint("\n  %s:" % var_name)
        for regime in ["bull", "bear"]:
            rt = [t for t in trades if t["regime"] == regime]
            if rt:
                pnls = [t["pnl"] for t in rt]
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                fprint("    %s regime: %d trades, WR %.1f%%, PnL $%s" % (
                    regime, len(rt), wr * 100, "{:,.0f}".format(sum(pnls))))

    # ==========================================================
    # 5-GATE VALIDATION SUMMARY
    # ==========================================================
    fprint("\n%s" % ("=" * 100))
    fprint("5-GATE VALIDATION SUMMARY")
    fprint("=" * 100)

    for var_name, _, _, _ in variants:
        result_val = all_results.get(var_name)
        if result_val is None:
            fprint("\n  %s: No validation (insufficient trades)" % var_name)
            continue
        r = result_val.to_dict()
        status = "PASS" if r["all_passed"] else "FAIL"
        fprint("\n  [%s] %s: %d/%d gates" % (status, var_name, r["gates_passed"], r["gates_total"]))
        if r.get("gates"):
            for g in r["gates"]:
                gs = "PASS" if g["passed"] else "FAIL"
                fprint("         [%s] %s: %s=%.4f" % (gs, g["name"], g["metric_name"], g["metric_value"]))

    # ==========================================================
    # BIWEEKLY vs MONTHLY (Variant E vs B)
    # ==========================================================
    fprint("\n%s" % ("=" * 100))
    fprint("BIWEEKLY vs MONTHLY REBALANCE (E vs B, both 50%% PT)")
    fprint("=" * 100)

    b_stats = all_stats.get("B_v93_pt50")
    e_stats = all_stats.get("E_v93_biweekly")
    if b_stats and e_stats:
        fprint("\n  B (Monthly + 50%% PT):  Sharpe %.2f, Sortino %.2f, PF %.2f, WR %.1f%%, MDD %.1f%%, N=%d" % (
            b_stats["sharpe"], b_stats["sortino"], b_stats["profit_factor"],
            b_stats["win_rate"] * 100, b_stats["max_dd_pct"], b_stats["n_trades"]))
        fprint("  E (Biweekly + 50%% PT): Sharpe %.2f, Sortino %.2f, PF %.2f, WR %.1f%%, MDD %.1f%%, N=%d" % (
            e_stats["sharpe"], e_stats["sortino"], e_stats["profit_factor"],
            e_stats["win_rate"] * 100, e_stats["max_dd_pct"], e_stats["n_trades"]))
        fprint("  Delta (E-B):           dSharpe %+.2f, dSortino %+.2f, dPF %+.2f, dWR %+.1f%%, dMDD %+.1f%%" % (
            e_stats["sharpe"] - b_stats["sharpe"],
            e_stats["sortino"] - b_stats["sortino"],
            e_stats["profit_factor"] - b_stats["profit_factor"],
            (e_stats["win_rate"] - b_stats["win_rate"]) * 100,
            e_stats["max_dd_pct"] - b_stats["max_dd_pct"]))
        if e_stats["sharpe"] > b_stats["sharpe"]:
            fprint("  --> Biweekly rebalance + profit taking IMPROVES over monthly")
        else:
            fprint("  --> Monthly rebalance remains optimal even with profit targets")

    # ==========================================================
    # PROFIT TARGET SENSITIVITY (C vs B vs D)
    # ==========================================================
    fprint("\n%s" % ("=" * 100))
    fprint("PROFIT TARGET SENSITIVITY (30%% vs 50%% vs 60%%)")
    fprint("=" * 100)

    c_stats = all_stats.get("C_v93_pt30")
    d_stats = all_stats.get("D_v93_pt60")
    if b_stats and c_stats and d_stats:
        fprint("\n  Target  Sharpe  Sortino  PF    WR     MDD    EarlyX%%  CAGR")
        fprint("  %s" % ("-" * 65))
        for label, st in [("30%%", c_stats), ("50%%", b_stats), ("60%%", d_stats)]:
            fprint("  %-6s  %.2f    %.2f     %.2f  %.1f%%  %.1f%%  %.1f%%     %.1f%%" % (
                label, st["sharpe"], st["sortino"], st["profit_factor"],
                st["win_rate"] * 100, st["max_dd_pct"],
                st["early_exit_rate"] * 100, st["cagr"] * 100))
        # Find best
        best_pt = max(
            [("30%%", c_stats), ("50%%", b_stats), ("60%%", d_stats)],
            key=lambda x: x[1]["sharpe"]
        )
        fprint("\n  Best profit target by Sharpe: %s (%.2f)" % (best_pt[0], best_pt[1]["sharpe"]))

    # ==========================================================
    # VERDICT
    # ==========================================================
    fprint("\n%s" % ("=" * 100))
    fprint("COMBINED OPTIMAL SWEEP -- FINAL VERDICT")
    fprint("=" * 100)

    best_variant = None
    best_sharpe = -999
    for var_name, _, _, _ in variants:
        s = all_stats.get(var_name)
        if s is None:
            continue
        if s["sharpe"] > best_sharpe:
            best_sharpe = s["sharpe"]
            best_variant = var_name

    if base:
        fprint("\n  Baseline (A: hold-to-expiry): Sharpe %.2f, Sortino %.2f, PF %.2f, WR %.1f%%, MDD %.1f%%" % (
            base["sharpe"], base["sortino"], base["profit_factor"],
            base["win_rate"] * 100, base["max_dd_pct"]))

    fprint("\n  VARIANT RANKING (by Sharpe):")
    ranked = sorted(
        [(vn, all_stats[vn]) for vn, _, _, _ in variants if all_stats.get(vn)],
        key=lambda x: x[1]["sharpe"], reverse=True
    )
    for i, (vn, st) in enumerate(ranked, 1):
        mc = all_mc.get(vn)
        mc_str = " (MC 90%%CI: [%.2f, %.2f])" % (mc["sharpe_ci_5"], mc["sharpe_ci_95"]) if mc else ""
        delta = st["sharpe"] - base["sharpe"] if base else 0
        fprint("    %d. %s: Sharpe %.2f (%+.2f vs baseline)%s" % (i, vn, st["sharpe"], delta, mc_str))

    fprint("\n  BEST OVERALL: %s (Sharpe %.2f)" % (best_variant, best_sharpe))

    # Determine recommendation
    all_pt_beat = True
    if base:
        for vn in ["B_v93_pt50", "C_v93_pt30", "D_v93_pt60"]:
            s = all_stats.get(vn)
            if s and s["sharpe"] <= base["sharpe"]:
                all_pt_beat = False

    fprint("\n  KEY FINDINGS:")
    if all_pt_beat:
        fprint("    - ALL profit targets beat hold-to-expiry (robust finding)")
    else:
        fprint("    - NOT all profit targets beat baseline (mixed results)")

    if e_stats and b_stats:
        if e_stats["sharpe"] > b_stats["sharpe"] + 0.1:
            fprint("    - Biweekly + PT outperforms Monthly + PT")
        elif e_stats["sharpe"] < b_stats["sharpe"] - 0.1:
            fprint("    - Monthly rebalance remains superior even with profit targets")
        else:
            fprint("    - Biweekly and Monthly perform similarly with profit targets")

    # Production recommendation
    fprint("\n  PRODUCTION RECOMMENDATION:")
    best_s = all_stats.get(best_variant)
    if best_s:
        fprint("    Config: %s" % best_variant)
        fprint("    Sharpe: %.2f | Sortino: %.2f | PF: %.2f | WR: %.1f%% | MDD: %.1f%%" % (
            best_s["sharpe"], best_s["sortino"], best_s["profit_factor"],
            best_s["win_rate"] * 100, best_s["max_dd_pct"]))
        fprint("    CAGR: %.1f%% | Avg hold: %.1f days" % (best_s["cagr"] * 100, best_s["avg_hold_days"]))

    # -- Save results --
    save_results = {
        "timestamp": t0.isoformat(),
        "config": {
            "capital": CAP, "dte": DTE, "otm_pct": 0.03,
            "commission": COMMISSION_RT_SPREAD,
            "early_exit_commission": EARLY_EXIT_COMMISSION,
            "top_k": TOP_K, "wf_weeks": WF_TRAIN_PERIODS,
            "n_sectors": len(SECTORS), "features": len(V6_FEATURES),
            "haircut": HAIRCUT, "n_bootstrap": N_BOOTSTRAP,
        },
        "best_variant": best_variant,
        "best_sharpe": best_sharpe,
        "all_pt_beat_baseline": all_pt_beat,
    }

    for var_name, _, _, _ in variants:
        s = all_stats.get(var_name)
        if s:
            save_results[var_name] = s
        rv = all_results.get(var_name)
        if rv:
            save_results["%s_validation" % var_name] = rv.to_dict()
        mc = all_mc.get(var_name)
        if mc:
            save_results["%s_monte_carlo" % var_name] = mc

    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint("\nResults saved to %s" % results_file)

    # Save trade details
    for var_name, _, _, _ in variants:
        trades = all_trades.get(var_name, [])
        if trades:
            tf = OUTPUT_DIR / ("trades_%s.json" % var_name)
            with open(tf, "w") as f:
                json.dump(trades, f, indent=1, default=str)

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="combined_sweep_%s" % t0.strftime("%Y%m%d_%H%M")):
                mlflow.log_param("capital", CAP)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("otm_pct", 0.03)
                mlflow.log_param("commission", COMMISSION_RT_SPREAD)
                mlflow.log_param("early_exit_commission", EARLY_EXIT_COMMISSION)
                mlflow.log_param("top_k", TOP_K)
                mlflow.log_param("wf_train_weeks", WF_TRAIN_PERIODS)
                mlflow.log_param("n_sectors", len(SECTORS))
                mlflow.log_param("n_features", len(V6_FEATURES))
                mlflow.log_param("n_bootstrap", N_BOOTSTRAP)
                mlflow.log_param("best_variant", best_variant)
                mlflow.log_param("all_pt_beat_baseline", all_pt_beat)

                for var_name, _, _, _ in variants:
                    s = all_stats.get(var_name)
                    if s is None:
                        continue
                    # Use first letter as prefix
                    prefix = var_name.split("_")[0]
                    mlflow.log_metric("%s_sharpe" % prefix, s["sharpe"])
                    mlflow.log_metric("%s_sortino" % prefix, s["sortino"])
                    mlflow.log_metric("%s_calmar" % prefix, s["calmar"])
                    mlflow.log_metric("%s_wr" % prefix, s["win_rate"])
                    mlflow.log_metric("%s_pf" % prefix, s["profit_factor"])
                    mlflow.log_metric("%s_mdd" % prefix, s["max_dd_pct"])
                    mlflow.log_metric("%s_n_trades" % prefix, s["n_trades"])
                    mlflow.log_metric("%s_early_exit_rate" % prefix, s["early_exit_rate"])
                    mlflow.log_metric("%s_avg_hold" % prefix, s["avg_hold_days"])
                    mlflow.log_metric("%s_total_pnl" % prefix, s["total_pnl"])
                    mlflow.log_metric("%s_cagr" % prefix, s["cagr"])

                    mc = all_mc.get(var_name)
                    if mc:
                        mlflow.log_metric("%s_mc_sharpe_mean" % prefix, mc["sharpe_mean"])
                        mlflow.log_metric("%s_mc_sharpe_ci5" % prefix, mc["sharpe_ci_5"])
                        mlflow.log_metric("%s_mc_sharpe_ci95" % prefix, mc["sharpe_ci_95"])
                        mlflow.log_metric("%s_mc_max_dd_mean" % prefix, mc["max_dd_mean"])

                    rv = all_results.get(var_name)
                    if rv:
                        rd = rv.to_dict()
                        mlflow.log_metric("%s_gates_passed" % prefix, rd["gates_passed"])

                mlflow.log_artifact(str(results_file))
            fprint("MLflow logged to experiment '%s'" % EXPERIMENT_NAME)
        except Exception as e:
            fprint("MLflow error: %s" % str(e))

    elapsed = (datetime.now() - t0).total_seconds()
    fprint("\nCompleted in %.1f minutes" % (elapsed / 60))


if __name__ == "__main__":
    main()
