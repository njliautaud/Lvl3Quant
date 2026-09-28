#!/usr/bin/env python3
"""
V9.2 Optimal Config Stress Test -- Pre-Deployment Validation
=============================================================

Stress-tests the BEST COMBINED V9.2 config before paper engine deployment.

V9.2 Optimal Config:
  - DTE=28, 3% OTM, adaptive max($3,3%) width
  - LGBM 17 momentum features (not GRU)
  - Monthly rebalance (Sharpe 1.88, +16% vs weekly -- KB #252)
  - 3 bull + 3 bear positions (standard)
  - Hold-to-expiry
  - $645 capital, $2.60 commission, 15% haircut
  - Real Dolt chain data

6 Stress Test Variants:
  A: V9.2 Config -- Monthly rebalance + DTE=28 + 3% OTM (the candidate)
  B: 3x Commission -- Same as A but $7.80 commission per trade
  C: Remove Top 3 Tickers -- Same as A but remove 3 highest-PnL tickers
  D: 2020 COVID Crash Only -- Same as A, Feb-Dec 2020 only
  E: 2022 Bear Market Only -- Same as A, Jan-Dec 2022 only
  F: Monte Carlo Shuffle -- 1000 bootstrap resamples, report 5th pct Sharpe

5-gate validation on A, B, C. D/E report stats only. F reports bootstrap CI.

Output: output/growth_research/v92_optimal_stress_test_v1/
MLflow experiment: v92_optimal_stress_test_v1
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
OUTPUT_DIR = BASE / "output" / "growth_research" / "v92_optimal_stress_test_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
TOP_K = 3
DTE = 28
REGIME_BULL_THRESHOLD = 0.4
WF_TRAIN_PERIODS = 52  # 52-week sliding window

COST_WIDTH_MAX = 0.50
DTE_TOLERANCE = 5
STRIKE_TOLERANCE = 0.03

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v92_optimal_stress_test_v1"

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


def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity,
                  chains, max_pos, commission_override=None):
    """Execute a single spread trade, return result dict or None."""
    commission = commission_override if commission_override is not None else COMMISSION_RT_SPREAD

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

    Se = float(close[tk].iloc[ei])
    if direction == "bull":
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    else:
        intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

    pnl = (intrinsic - entry_cost_ps) * 100 - commission

    return {
        "pnl": round(pnl, 2), "entry_cost_ps": round(entry_cost_ps, 4),
        "total_cost": round(total_cost, 2), "used_real_pricing": used_real,
        "K1": K1, "K2": K2, "S_entry": round(S, 2), "S_exit": round(Se, 2),
        "intrinsic": round(intrinsic, 4), "spread_width": round(spread_width, 2),
        "cost_width_ratio": round(cwr, 3),
    }


# ==============================================================
# SIMULATION ENGINE
# ==============================================================

def simulate_variant(rankings, close, atr_dict, chains,
                     rebal_dates, variant_name="",
                     commission_override=None, exclude_tickers=None,
                     date_filter=None):
    """Run backtest simulation for a variant.

    Args:
        rankings: dict {date: {ticker: score}}
        close: price DataFrame
        atr_dict: ATR series per ticker
        chains: chain data dict
        rebal_dates: DatetimeIndex of rebalance dates
        variant_name: for logging
        commission_override: override commission per trade (Variant B)
        exclude_tickers: set of tickers to exclude (Variant C)
        date_filter: tuple (start, end) to restrict trading window (Variants D, E)
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None
    exclude_set = set(exclude_tickers) if exclude_tickers else set()

    equity = CAP
    trades = []
    real_count = 0
    bs_count = 0
    turnover_events = 0

    for dt in sorted(rebal_dates):
        if dt not in spy.index:
            continue
        # Apply date filter for Variants D, E
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

                result = execute_trade(
                    tk, dt, direction, close, atr_dict, cv, equity,
                    chains, max_pos,
                    commission_override=commission_override,
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
                        **result, "entry_date": str(dt.date()),
                        "exit_date": str(close.index[ei].date()),
                        "ticker": tk, "regime": "bull" if se >= sv else "bear",
                        "direction": direction, "vix": round(cv, 1),
                        "win": result["pnl"] > 0,
                        "n_positions": n_positions,
                    })

    return trades, equity, real_count, bs_count, turnover_events


def monte_carlo_bootstrap(trades, n_bootstrap=1000, seed=42):
    """Monte Carlo bootstrap of monthly returns. Returns distribution of Sharpes."""
    if len(trades) < 10:
        return None

    # Group trades by month to get monthly returns
    trade_df = pd.DataFrame(trades)
    trade_df["entry_month"] = pd.to_datetime(trade_df["entry_date"]).dt.to_period("M")
    monthly_pnl = trade_df.groupby("entry_month")["pnl"].sum()

    if len(monthly_pnl) < 6:
        # Fall back to individual trade resampling
        pnls = np.array([t["pnl"] for t in trades])
        rng = np.random.RandomState(seed)
        sharpes = []
        for _ in range(n_bootstrap):
            sample = rng.choice(pnls, size=len(pnls), replace=True)
            sh = float(np.mean(sample) / (np.std(sample) + 1e-10) * np.sqrt(52))
            sharpes.append(sh)
        sharpes = np.array(sharpes)
        return {
            "method": "trade_resample",
            "n_bootstrap": n_bootstrap,
            "mean_sharpe": float(np.mean(sharpes)),
            "median_sharpe": float(np.median(sharpes)),
            "p5_sharpe": float(np.percentile(sharpes, 5)),
            "p10_sharpe": float(np.percentile(sharpes, 10)),
            "p25_sharpe": float(np.percentile(sharpes, 25)),
            "p75_sharpe": float(np.percentile(sharpes, 75)),
            "p95_sharpe": float(np.percentile(sharpes, 95)),
            "ci_95_low": float(np.percentile(sharpes, 2.5)),
            "ci_95_high": float(np.percentile(sharpes, 97.5)),
            "pct_positive_sharpe": float((sharpes > 0).mean() * 100),
            "pct_above_1": float((sharpes > 1.0).mean() * 100),
        }

    # Bootstrap resample monthly returns
    monthly_vals = monthly_pnl.values.astype(float)
    rng = np.random.RandomState(seed)
    sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(monthly_vals, size=len(monthly_vals), replace=True)
        sh = float(np.mean(sample) / (np.std(sample) + 1e-10) * np.sqrt(12))
        sharpes.append(sh)
    sharpes = np.array(sharpes)

    return {
        "method": "monthly_resample",
        "n_months": len(monthly_vals),
        "n_bootstrap": n_bootstrap,
        "mean_sharpe": float(np.mean(sharpes)),
        "median_sharpe": float(np.median(sharpes)),
        "p5_sharpe": float(np.percentile(sharpes, 5)),
        "p10_sharpe": float(np.percentile(sharpes, 10)),
        "p25_sharpe": float(np.percentile(sharpes, 25)),
        "p75_sharpe": float(np.percentile(sharpes, 75)),
        "p95_sharpe": float(np.percentile(sharpes, 95)),
        "ci_95_low": float(np.percentile(sharpes, 2.5)),
        "ci_95_high": float(np.percentile(sharpes, 97.5)),
        "pct_positive_sharpe": float((sharpes > 0).mean() * 100),
        "pct_above_1": float((sharpes > 1.0).mean() * 100),
    }


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


def compute_stats_only(trades, initial_capital=CAP):
    """Compute summary stats for variants with too few trades for 5-gate validation."""
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

    return {
        "n_trades": len(trades),
        "total_pnl": float(pnls.sum()),
        "avg_pnl": float(pnls.mean()),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 3),
        "sharpe_monthly": round(sharpe, 3),
        "max_dd_pct": round(float(dd.min()) * 100, 2),
        "final_equity": round(equity[-1], 2),
        "calmar": round(compute_calmar_ratio(trades, initial_capital), 3),
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
    fprint(f"V9.2 OPTIMAL CONFIG STRESS TEST -- {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Commission: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"Universe: {len(SECTORS)} sector ETFs | Top/Bottom K: {TOP_K}")
    fprint(f"Rebalance: MONTHLY | Walk-forward: {WF_TRAIN_PERIODS}-week sliding")
    fprint(f"OTM: 3% | Width: adaptive max($3, 3%) | Hold: to expiry")
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

    # ==========================================================
    # VARIANT A: V9.2 Candidate Config
    # ==========================================================
    fprint(f"\n{'=' * 90}")
    fprint("VARIANT A: V9.2 CONFIG -- Monthly + DTE=28 + 3% OTM (THE CANDIDATE)")
    fprint(f"{'=' * 90}")

    trades_a, eq_a, real_a, bs_a, turns_a = simulate_variant(
        rankings, close, atr_dict, chains, monthly_dates, "A_v92_candidate"
    )
    fprint(f"  Trades: {len(trades_a)} | Real: {real_a} | BS: {bs_a} | Final: ${eq_a:,.0f}")

    result_a = None
    if len(trades_a) >= 5:
        result_a = validate_trades(trades_a, initial_capital=CAP, spy_prices=spy_close,
                                   strategy_name="A_v92_candidate")
        result_a.print_summary()

    # Find top 3 PnL tickers for Variant C
    top3_tickers, ticker_pnl_sorted = find_top_pnl_tickers(trades_a, top_n=3)
    fprint(f"\n  Top 3 PnL tickers (to remove for Variant C):")
    for tk, pnl in ticker_pnl_sorted[:5]:
        fprint(f"    {tk}: ${pnl:,.0f}")
    fprint(f"  -> Removing: {top3_tickers}")

    # ==========================================================
    # VARIANT B: 3x Commission ($7.80)
    # ==========================================================
    fprint(f"\n{'=' * 90}")
    fprint("VARIANT B: 3x COMMISSION -- $7.80 per trade (stress test cost sensitivity)")
    fprint(f"{'=' * 90}")

    COMMISSION_3X = 7.80
    trades_b, eq_b, real_b, bs_b, turns_b = simulate_variant(
        rankings, close, atr_dict, chains, monthly_dates, "B_3x_commission",
        commission_override=COMMISSION_3X,
    )
    fprint(f"  Trades: {len(trades_b)} | Final: ${eq_b:,.0f}")

    result_b = None
    if len(trades_b) >= 5:
        result_b = validate_trades(trades_b, initial_capital=CAP, spy_prices=spy_close,
                                   strategy_name="B_3x_commission")
        result_b.print_summary()

    # ==========================================================
    # VARIANT C: Remove Top 3 Tickers
    # ==========================================================
    fprint(f"\n{'=' * 90}")
    fprint(f"VARIANT C: REMOVE TOP 3 TICKERS -- {top3_tickers}")
    fprint(f"{'=' * 90}")

    trades_c, eq_c, real_c, bs_c, turns_c = simulate_variant(
        rankings, close, atr_dict, chains, monthly_dates, "C_remove_top3",
        exclude_tickers=top3_tickers,
    )
    fprint(f"  Trades: {len(trades_c)} | Final: ${eq_c:,.0f}")

    result_c = None
    if len(trades_c) >= 5:
        result_c = validate_trades(trades_c, initial_capital=CAP, spy_prices=spy_close,
                                   strategy_name="C_remove_top3")
        result_c.print_summary()

    # ==========================================================
    # VARIANT D: 2020 COVID Crash Only
    # ==========================================================
    fprint(f"\n{'=' * 90}")
    fprint("VARIANT D: 2020 COVID CRASH -- Feb-Dec 2020 only")
    fprint(f"{'=' * 90}")

    trades_d, eq_d, real_d, bs_d, turns_d = simulate_variant(
        rankings, close, atr_dict, chains, monthly_dates, "D_covid_2020",
        date_filter=("2020-02-01", "2020-12-31"),
    )
    fprint(f"  Trades: {len(trades_d)} | Final: ${eq_d:,.0f}")
    stats_d = compute_stats_only(trades_d)
    if stats_d:
        fprint(f"  Stats: Sharpe={stats_d['sharpe_monthly']:.2f}, WR={stats_d['win_rate']*100:.1f}%, "
               f"PF={stats_d['profit_factor']:.2f}, MDD={stats_d['max_dd_pct']:.1f}%, "
               f"Total PnL=${stats_d['total_pnl']:,.0f}")

    # Side breakdown
    for side in ["bull", "bear"]:
        st = [t for t in trades_d if t["direction"] == side]
        if st:
            pnls = [t["pnl"] for t in st]
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            fprint(f"  {side}: {len(st)} trades, WR {wr:.1%}, PnL ${sum(pnls):,.0f}")

    # ==========================================================
    # VARIANT E: 2022 Bear Market Only
    # ==========================================================
    fprint(f"\n{'=' * 90}")
    fprint("VARIANT E: 2022 BEAR MARKET -- Jan-Dec 2022 only")
    fprint(f"{'=' * 90}")

    trades_e, eq_e, real_e, bs_e, turns_e = simulate_variant(
        rankings, close, atr_dict, chains, monthly_dates, "E_bear_2022",
        date_filter=("2022-01-01", "2022-12-31"),
    )
    fprint(f"  Trades: {len(trades_e)} | Final: ${eq_e:,.0f}")
    stats_e = compute_stats_only(trades_e)
    if stats_e:
        fprint(f"  Stats: Sharpe={stats_e['sharpe_monthly']:.2f}, WR={stats_e['win_rate']*100:.1f}%, "
               f"PF={stats_e['profit_factor']:.2f}, MDD={stats_e['max_dd_pct']:.1f}%, "
               f"Total PnL=${stats_e['total_pnl']:,.0f}")

    for side in ["bull", "bear"]:
        st = [t for t in trades_e if t["direction"] == side]
        if st:
            pnls = [t["pnl"] for t in st]
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            fprint(f"  {side}: {len(st)} trades, WR {wr:.1%}, PnL ${sum(pnls):,.0f}")

    # ==========================================================
    # VARIANT F: Monte Carlo Shuffle (1000 bootstrap)
    # ==========================================================
    fprint(f"\n{'=' * 90}")
    fprint("VARIANT F: MONTE CARLO -- 1000 bootstrap resamples of monthly returns")
    fprint(f"{'=' * 90}")

    mc_result = monte_carlo_bootstrap(trades_a, n_bootstrap=1000, seed=42)
    if mc_result:
        fprint(f"  Method: {mc_result['method']}")
        if 'n_months' in mc_result:
            fprint(f"  Monthly returns: {mc_result['n_months']} months")
        fprint(f"  Bootstrap Sharpe distribution (n={mc_result['n_bootstrap']}):")
        fprint(f"    Mean:    {mc_result['mean_sharpe']:.3f}")
        fprint(f"    Median:  {mc_result['median_sharpe']:.3f}")
        fprint(f"    5th pct: {mc_result['p5_sharpe']:.3f}  <-- WORST REALISTIC CASE")
        fprint(f"    10th:    {mc_result['p10_sharpe']:.3f}")
        fprint(f"    25th:    {mc_result['p25_sharpe']:.3f}")
        fprint(f"    75th:    {mc_result['p75_sharpe']:.3f}")
        fprint(f"    95th:    {mc_result['p95_sharpe']:.3f}")
        fprint(f"    95% CI:  [{mc_result['ci_95_low']:.3f}, {mc_result['ci_95_high']:.3f}]")
        fprint(f"    % Sharpe > 0: {mc_result['pct_positive_sharpe']:.1f}%")
        fprint(f"    % Sharpe > 1: {mc_result['pct_above_1']:.1f}%")
    else:
        fprint("  Not enough trades for Monte Carlo")

    # ==========================================================
    # COMPARISON TABLE
    # ==========================================================
    fprint(f"\n{'=' * 140}")
    fprint("COMPARISON -- ALL STRESS TEST VARIANTS")
    fprint(f"{'=' * 140}")

    fprint(f"\n  {'Variant':<30} {'N':>5} {'Sharpe':>7} {'Sort':>7} {'Calmar':>7} {'WR':>6} "
           f"{'PF':>6} {'MDD':>7} {'Gate':>5} {'Final$':>9} {'%Real':>6}")
    fprint(f"  {'-' * 120}")

    # A, B, C have full validation
    for label, result, trades, real, bs in [
        ("A_v92_candidate", result_a, trades_a, real_a, bs_a),
        ("B_3x_commission", result_b, trades_b, real_b, bs_b),
        ("C_remove_top3", result_c, trades_c, real_c, bs_c),
    ]:
        if result is None:
            continue
        r = result.to_dict()
        total = real + bs
        real_pct = real / max(total, 1) * 100
        calmar = compute_calmar_ratio(trades)
        fprint(f"  {label:<30} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>7.2f} "
               f"{calmar:>7.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>8,.0f} {real_pct:>5.1f}%")

    # D, E stats only
    for label, stats_dict in [
        ("D_covid_2020", stats_d),
        ("E_bear_2022", stats_e),
    ]:
        if stats_dict is None:
            continue
        fprint(f"  {label:<30} {stats_dict['n_trades']:>5} {stats_dict['sharpe_monthly']:>7.2f} "
               f"{'--':>7} {stats_dict['calmar']:>7.2f} "
               f"{stats_dict['win_rate']*100:>5.1f}% {stats_dict['profit_factor']:>5.2f} "
               f"{stats_dict['max_dd_pct']:>6.1f}% {'--':>5} "
               f"${stats_dict['final_equity']:>8,.0f} {'--':>6}")

    # F summary row
    if mc_result:
        p5_str = f"p5={mc_result['p5_sharpe']:.2f}"
        fprint(f"  {'F_monte_carlo':<30} {'--':>5} "
               f"{p5_str:>7} "
               f"{'--':>7} {'--':>7} {'--':>6} {'--':>6} {'--':>7} {'--':>5} "
               f"{'--':>9} {'--':>6}")

    # ==========================================================
    # SIDE ANALYSIS (A only)
    # ==========================================================
    fprint(f"\n{'=' * 80}")
    fprint("SIDE ANALYSIS -- VARIANT A")
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

    # Ticker contribution analysis
    fprint(f"\n{'=' * 80}")
    fprint("TICKER PnL CONTRIBUTION -- VARIANT A")
    fprint(f"{'=' * 80}")
    for tk, pnl in ticker_pnl_sorted:
        n_tk = sum(1 for t in trades_a if t["ticker"] == tk)
        fprint(f"  {tk}: ${pnl:>8,.0f} ({n_tk} trades)")

    # ==========================================================
    # 5-GATE VALIDATION SUMMARY
    # ==========================================================
    fprint(f"\n{'=' * 100}")
    fprint("5-GATE VALIDATION SUMMARY (A, B, C only -- D/E too few trades)")
    fprint(f"{'=' * 100}")

    for label, result in [
        ("A_v92_candidate", result_a),
        ("B_3x_commission", result_b),
        ("C_remove_top3", result_c),
    ]:
        if result is None:
            continue
        r = result.to_dict()
        status = "PASS" if r["all_passed"] else "FAIL"
        fprint(f"\n  [{status}] {label}: {r['gates_passed']}/{r['gates_total']} gates")
        if r.get("gates"):
            for g in r["gates"]:
                gs = "PASS" if g["passed"] else "FAIL"
                fprint(f"         [{gs}] {g['name']}: {g['metric_name']}={g['metric_value']:.4f}")

    # ==========================================================
    # VERDICT
    # ==========================================================
    fprint(f"\n{'=' * 100}")
    fprint("STRESS TEST VERDICT")
    fprint(f"{'=' * 100}")

    verdict_pass = True
    verdict_notes = []

    # Check A passes all gates
    if result_a:
        ra = result_a.to_dict()
        fprint(f"\n  A (Candidate): Sharpe {ra['sharpe']:.2f}, PF {ra['profit_factor']:.2f}, "
               f"WR {ra['win_rate']*100:.1f}%, MDD {ra['max_dd']*100:.1f}%")
        if not ra["all_passed"]:
            verdict_pass = False
            verdict_notes.append("A: Not all 5 gates passed")
        else:
            verdict_notes.append(f"A: All {ra['gates_total']} gates PASSED")

    # Check B survives 3x commission
    if result_b:
        rb = result_b.to_dict()
        fprint(f"  B (3x cost):   Sharpe {rb['sharpe']:.2f}, PF {rb['profit_factor']:.2f}")
        if rb["sharpe"] < 0.5:
            verdict_pass = False
            verdict_notes.append(f"B: Sharpe {rb['sharpe']:.2f} < 0.5 under 3x commission -- FRAGILE")
        else:
            verdict_notes.append(f"B: Sharpe {rb['sharpe']:.2f} survives 3x commission")

    # Check C edge is broad-based
    if result_c and result_a:
        rc = result_c.to_dict()
        ra = result_a.to_dict()
        sharpe_drop = ra["sharpe"] - rc["sharpe"]
        fprint(f"  C (no top 3):  Sharpe {rc['sharpe']:.2f} (drop {sharpe_drop:+.2f})")
        if rc["sharpe"] < 0.3:
            verdict_pass = False
            verdict_notes.append(f"C: Sharpe {rc['sharpe']:.2f} -- edge concentrated in top 3 tickers, NOT broad-based")
        elif sharpe_drop > 0.8:
            verdict_notes.append(f"C: Large drop ({sharpe_drop:.2f}) -- edge partially concentrated")
        else:
            verdict_notes.append(f"C: Sharpe {rc['sharpe']:.2f} -- edge is broad-based")

    # Check D (COVID)
    if stats_d:
        fprint(f"  D (COVID):     Sharpe {stats_d['sharpe_monthly']:.2f}, WR {stats_d['win_rate']*100:.1f}%, "
               f"MDD {stats_d['max_dd_pct']:.1f}%")
        if stats_d["total_pnl"] < -100:
            verdict_notes.append(f"D: Lost ${abs(stats_d['total_pnl']):,.0f} during COVID -- watch drawdown")
        else:
            verdict_notes.append(f"D: PnL ${stats_d['total_pnl']:,.0f} during COVID -- acceptable")

    # Check E (2022 bear)
    if stats_e:
        fprint(f"  E (2022 bear): Sharpe {stats_e['sharpe_monthly']:.2f}, WR {stats_e['win_rate']*100:.1f}%, "
               f"MDD {stats_e['max_dd_pct']:.1f}%")
        if stats_e["total_pnl"] < -100:
            verdict_notes.append(f"E: Lost ${abs(stats_e['total_pnl']):,.0f} in 2022 bear -- investigate")
        else:
            verdict_notes.append(f"E: PnL ${stats_e['total_pnl']:,.0f} in 2022 bear -- acceptable")

    # Check F (Monte Carlo)
    if mc_result:
        p5 = mc_result["p5_sharpe"]
        fprint(f"  F (MC p5):     Worst realistic Sharpe {p5:.3f}")
        if p5 < 0:
            verdict_pass = False
            verdict_notes.append(f"F: 5th pct Sharpe {p5:.3f} < 0 -- strategy can go negative under bad luck")
        elif p5 < 0.5:
            verdict_notes.append(f"F: 5th pct Sharpe {p5:.3f} -- marginal under worst-case")
        else:
            verdict_notes.append(f"F: 5th pct Sharpe {p5:.3f} -- robust even worst-case")

    fprint(f"\n  NOTES:")
    for note in verdict_notes:
        fprint(f"    {note}")

    if verdict_pass:
        fprint(f"\n  >>> VERDICT: PASS -- V9.2 config cleared for paper engine deployment <<<")
    else:
        fprint(f"\n  >>> VERDICT: FAIL -- V9.2 config needs investigation before deployment <<<")

    # -- Save --
    all_results = {
        "timestamp": t0.isoformat(),
        "config": {
            "capital": CAP, "dte": DTE, "otm_pct": 0.03,
            "commission": COMMISSION_RT_SPREAD, "rebalance": "monthly",
            "top_k": TOP_K, "wf_weeks": WF_TRAIN_PERIODS,
            "n_sectors": len(SECTORS), "features": len(V6_FEATURES),
        },
        "verdict": "PASS" if verdict_pass else "FAIL",
        "verdict_notes": verdict_notes,
    }

    if result_a:
        all_results["A_v92_candidate"] = result_a.to_dict()
        all_results["A_v92_candidate"]["calmar"] = round(compute_calmar_ratio(trades_a), 3)
    if result_b:
        all_results["B_3x_commission"] = result_b.to_dict()
    if result_c:
        all_results["C_remove_top3"] = result_c.to_dict()
        all_results["C_remove_top3"]["excluded_tickers"] = top3_tickers
    if stats_d:
        all_results["D_covid_2020"] = stats_d
    if stats_e:
        all_results["E_bear_2022"] = stats_e
    if mc_result:
        all_results["F_monte_carlo"] = mc_result

    # Ticker contribution
    all_results["ticker_pnl"] = dict(ticker_pnl_sorted)

    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_file}")

    # Save trade details
    for label, trades in [
        ("A_v92_candidate", trades_a),
        ("B_3x_commission", trades_b),
        ("C_remove_top3", trades_c),
        ("D_covid_2020", trades_d),
        ("E_bear_2022", trades_e),
    ]:
        if trades:
            tf = OUTPUT_DIR / f"trades_{label}.json"
            with open(tf, "w") as f:
                json.dump(trades, f, indent=1, default=str)

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"v92_stress_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_param("capital", CAP)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("otm_pct", 0.03)
                mlflow.log_param("commission", COMMISSION_RT_SPREAD)
                mlflow.log_param("rebalance", "monthly")
                mlflow.log_param("top_k", TOP_K)
                mlflow.log_param("wf_train_weeks", WF_TRAIN_PERIODS)
                mlflow.log_param("n_sectors", len(SECTORS))
                mlflow.log_param("n_features", len(V6_FEATURES))
                mlflow.log_param("verdict", "PASS" if verdict_pass else "FAIL")
                if top3_tickers:
                    mlflow.log_param("removed_top3", ",".join(top3_tickers))

                # Variant A metrics
                if result_a:
                    ra = result_a.to_dict()
                    mlflow.log_metric("A_sharpe", ra["sharpe"])
                    mlflow.log_metric("A_sortino", ra["sortino"])
                    mlflow.log_metric("A_wr", ra["win_rate"])
                    mlflow.log_metric("A_pf", ra["profit_factor"])
                    mlflow.log_metric("A_mdd", ra["max_dd"])
                    mlflow.log_metric("A_n_trades", ra["n_trades"])
                    mlflow.log_metric("A_gates_passed", ra["gates_passed"])
                    mlflow.log_metric("A_calmar", compute_calmar_ratio(trades_a))

                # Variant B metrics
                if result_b:
                    rb = result_b.to_dict()
                    mlflow.log_metric("B_sharpe", rb["sharpe"])
                    mlflow.log_metric("B_pf", rb["profit_factor"])
                    mlflow.log_metric("B_gates_passed", rb["gates_passed"])

                # Variant C metrics
                if result_c:
                    rc = result_c.to_dict()
                    mlflow.log_metric("C_sharpe", rc["sharpe"])
                    mlflow.log_metric("C_pf", rc["profit_factor"])
                    mlflow.log_metric("C_gates_passed", rc["gates_passed"])

                # Variant D/E stats
                if stats_d:
                    mlflow.log_metric("D_sharpe", stats_d["sharpe_monthly"])
                    mlflow.log_metric("D_pnl", stats_d["total_pnl"])
                    mlflow.log_metric("D_wr", stats_d["win_rate"])
                if stats_e:
                    mlflow.log_metric("E_sharpe", stats_e["sharpe_monthly"])
                    mlflow.log_metric("E_pnl", stats_e["total_pnl"])
                    mlflow.log_metric("E_wr", stats_e["win_rate"])

                # Variant F Monte Carlo
                if mc_result:
                    mlflow.log_metric("F_mc_p5_sharpe", mc_result["p5_sharpe"])
                    mlflow.log_metric("F_mc_mean_sharpe", mc_result["mean_sharpe"])
                    mlflow.log_metric("F_mc_pct_positive", mc_result["pct_positive_sharpe"])

                mlflow.log_artifact(str(results_file))
            fprint(f"MLflow logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow error: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nCompleted in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
